import numpy as np
import torch

from .audio import SR, load_audio, resample, speech_crops
from .model.encoder import pool_exemplars

MAX_FRAMES = 8192


# Output length in mel frames, scaled from the prompt's frames-per-byte rate.
def gen_duration(ref_frames, prompt_text, gen_text, speed=1.0, max_frames=MAX_FRAMES):
    dur = ref_frames + int(ref_frames / max(len(prompt_text.encode()), 1) * len(gen_text.encode()) / speed)
    return min(dur, max_frames)


# Exemplar wav paths -> pooled unit accent vector (1, D); k crops of win_s seconds per clip.
def embed_exemplars(paths, feat_fn, enc, device, k=2, win_s=4.0, seed=0):
    rng = np.random.default_rng(seed)
    crops = np.concatenate([speech_crops(resample(load_audio(p, SR), SR, 16000), 16000, win_s=win_s, k=k, rng=rng)
                            for p in paths])
    feats, lens = feat_fn(torch.from_numpy(crops).to(device))
    with torch.inference_mode():
        a = enc(feats, lens)
    return pool_exemplars(a, a.shape[0])


# Accent label -> unit prototype vector (1, D) from the learned table.
def lookup_accent(table, classes, accent):
    if accent not in classes:
        raise ValueError(f"unknown accent '{accent}'; choose from {classes}")
    return table[classes.index(accent)][None]


# Prompt audio (24 kHz) + texts + accent vector -> generated waveform (24 kHz numpy).
def synthesize(cfm, mel_fn, vocoder, prompt_wav, prompt_text, text, accent, w, device, nfe=32, cfg=2.0, speed=1.0,
               seed=0):
    cond = mel_fn(torch.from_numpy(prompt_wav)[None].to(device)).permute(0, 2, 1)
    ref_frames = cond.shape[1]
    dur = gen_duration(ref_frames, prompt_text, text, speed)
    with torch.inference_mode():
        out = cfm.sample_accent(cond, [f"{prompt_text} {text}"], dur, accent=accent, accent_weight=w, steps=nfe,
                                cfg_strength=cfg, sway_sampling_coef=-1.0, seed=seed)
        wav = vocoder.decode(out[:, ref_frames:].permute(0, 2, 1))
    return wav.squeeze().float().cpu().numpy()
