import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .encoder import pool_exemplars
from .lora import adapter_state_dict


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


class AccentTrainer:
    # mode: "lookup" (accent table), "proto" (table + exemplar encoder anchored to it) or "encoder" (exemplars only).
    def __init__(self, cfm, encoder, feat_fn, param_groups, mode="proto", accent_table=None, classes=None, lr=1e-4,
                 weight_decay=0.01, warmup=1000, total_steps=30000, grad_clip=1.0, lambda_spk=0.05, lambda_con=0.3,
                 proto_p_enc=0.5, lambda_proto=1.0, lambda_proto_ce=1.0, proto_tau=0.1, device="cuda"):
        self.cfm, self.encoder, self.feat_fn = cfm, encoder, feat_fn
        self.mode, self.accent_table, self.classes = mode, accent_table, classes
        self.opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
        self.params = [p for g in self.opt.param_groups for p in g["params"]]
        self.lr, self.warmup, self.total, self.grad_clip = lr, warmup, total_steps, grad_clip
        self.lambda_spk, self.lambda_con = lambda_spk, lambda_con
        self.proto_p_enc, self.lambda_proto, self.lambda_proto_ce, self.proto_tau = \
            proto_p_enc, lambda_proto, lambda_proto_ce, proto_tau
        self.device = device
        self.step_n = 0

    # Exemplar crops (B, K, S) -> pooled accent (B, D) and per-crop embeddings (B*K, D).
    def embed_accent(self, exemplar):
        k = exemplar.shape[1]
        feats, frame_lens = self.feat_fn(exemplar.flatten(0, 1))
        a = self.encoder(feats, frame_lens)
        return pool_exemplars(a, k), a

    def losses(self, batch):
        acc_id = batch["accent_id"].to(self.device)
        out, aux, proto = {}, {}, None
        if self.mode in ("proto", "encoder"):
            k = batch["exemplar"].shape[1]
            a_enc, a_all = self.embed_accent(batch["exemplar"].to(self.device))
            aux = self.encoder.losses(a_all, acc_id.repeat_interleave(k),
                                      batch["speaker_id"].to(self.device).repeat_interleave(k))
        if self.mode == "encoder":
            a = a_enc.detach()
        else:
            a_tab = F.normalize(self.accent_table(acc_id), dim=-1)
            a = a_tab
            if self.mode == "proto":
                proto = (1 - F.cosine_similarity(a_enc, a_tab.detach(), dim=-1)).mean()
                P = F.normalize(self.accent_table.weight, dim=-1)
                proto = proto + self.lambda_proto_ce * F.cross_entropy(a_enc @ P.T / self.proto_tau, acc_id)
                use_enc = torch.rand(a_tab.shape[0], device=self.device) < self.proto_p_enc
                a = torch.where(use_enc[:, None], a_enc.detach(), a_tab)
        fm = self.cfm(batch["mel"].to(self.device), batch["text"], lens=batch["lens"].to(self.device),
                      span_mask=batch["span_mask"].to(self.device), accent=a)
        total = fm + self.lambda_con * aux.get("supcon", 0.0) + self.lambda_spk * aux.get("grl_spk", 0.0)
        out.update(fm=fm.detach(), **{k_: v.detach() for k_, v in aux.items()})
        if proto is not None:
            total = total + self.lambda_proto * proto
            out["proto"] = proto.detach()
        out["loss"] = total
        return out

    def step(self, batch):
        self.cfm.train()
        self.encoder.train()
        for g in self.opt.param_groups:
            g["lr"] = cosine_lr(self.step_n, self.total, self.warmup, self.lr) * g.get("lr_mult", 1.0)
        out = self.losses(batch)
        self.opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.opt.step()
        self.step_n += 1
        return {k: float(v.detach()) for k, v in out.items()}

    def save(self, out_dir, tag="last", with_opt=True):
        ck = {"adapter": adapter_state_dict(self.cfm.transformer), "encoder": self.encoder.state_dict(),
              "step": self.step_n}
        if with_opt:
            ck["opt"] = self.opt.state_dict()
        if self.accent_table is not None:
            ck["accent_table"] = self.accent_table.state_dict()
            ck["classes"] = list(self.classes)
        path = Path(out_dir) / f"accent_{tag}.pt"
        torch.save(ck, path)
        return path

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cfm.transformer.load_state_dict(ck["adapter"], strict=False)
        self.encoder.load_state_dict(ck["encoder"])
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        if "accent_table" in ck and self.accent_table is not None:
            self.accent_table.load_state_dict(ck["accent_table"])
        self.step_n = ck.get("step", 0)
