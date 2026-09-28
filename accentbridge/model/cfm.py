from random import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.cfm import CFM
from f5_tts.model.utils import (exists, get_epss_timesteps, lens_to_mask, list_str_to_idx, list_str_to_tensor,
                                mask_from_frac_lengths)

from .injection import attach_accent


class AccentCFM(CFM):
    # F5-TTS flow-matching model with an accent vector injected into time and text embeddings.
    def __init__(self, *args, accent_dim=128, accent_drop_prob=0.15, **kw):
        super().__init__(*args, **kw)
        attach_accent(self.transformer, accent_dim)
        self.accent_drop_prob = accent_drop_prob

    @property
    def accent(self):
        return self.transformer.time_embed

    def _tokens(self, text, device):
        if isinstance(text, list):
            text = list_str_to_idx(text, self.vocab_char_map) if exists(self.vocab_char_map) else list_str_to_tensor(text)
        return text.to(device)

    # Flow-matching loss on the masked span; accent (B, D) is replaced by the null vector with prob accent_drop_prob.
    def forward(self, inp, text, *, lens, span_mask, accent=None):
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        text = self._tokens(text, device)
        mask = lens_to_mask(lens, length=seq_len)
        span_mask = span_mask & mask
        x1 = inp
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=device)
        t = time[:, None, None]
        xt = (1 - t) * x0 + t * x1
        cond = torch.where(span_mask[..., None], torch.zeros_like(x1), x1)
        drop_audio_cond = random() < self.audio_drop_prob
        drop_text = False
        if random() < self.cond_drop_prob:
            drop_audio_cond, drop_text = True, True
        if accent is not None and self.accent_drop_prob > 0:
            kept = torch.rand(batch, device=device) >= self.accent_drop_prob
            accent = torch.where(kept[:, None], accent, self.accent.null.unsqueeze(0).to(accent.dtype))
        self.accent.accent = accent
        try:
            pred = self.transformer(x=xt, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond,
                                    drop_text=drop_text, mask=mask)
        finally:
            self.accent.accent = None
        return F.mse_loss(pred, x1 - x0, reduction="none")[span_mask].mean()

    # Sample mel (B, T, 100) with v = v_null + cfg*(v_null - v_uncond) + w*(v_accent - v_null).
    @torch.no_grad()
    def sample_accent(self, cond, text, duration, *, accent=None, accent_weight=0.0, steps=32, cfg_strength=2.0,
                      sway_sampling_coef=-1.0, seed=None, max_duration=65536):
        self.eval()
        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        text = self._tokens(text, device)
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        duration = torch.maximum(torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration).clamp(max=max_duration)
        max_dur = duration.amax()
        cond = F.pad(cond, (0, 0, 0, max_dur - cond_seq_len), value=0.0)
        cond_mask = F.pad(lens_to_mask(lens), (0, max_dur - cond_seq_len), value=False).unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        mask = lens_to_mask(duration) if batch > 1 else None
        a = None if accent is None else accent.to(step_cond.dtype).expand(batch, -1)
        use_accent = a is not None and accent_weight != 0.0

        def fn(t, x):
            self.accent.accent = None
            packed = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, cfg_infer=True, cache=True)
            pred_null, uncond = torch.chunk(packed, 2, dim=0)
            out = pred_null + (pred_null - uncond) * cfg_strength
            if use_accent:
                self.accent.accent = a
                pred_a = self.transformer(x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False,
                                          drop_text=False, cache=True)
                self.accent.accent = None
                out = out + accent_weight * (pred_a - pred_null)
            return out

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        t = get_epss_timesteps(steps, device=device, dtype=step_cond.dtype)
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        try:
            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        finally:
            self.accent.accent = None
            self.transformer.clear_cache()
        return torch.where(cond_mask, cond, trajectory[-1])
