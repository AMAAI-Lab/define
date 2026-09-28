import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from accentbridge.config import Config
from accentbridge.data import FrameBatchSampler, PairDataset, build_rows, collate, frame_lens
from accentbridge.model.build import build_f5, build_feat_fn
from accentbridge.model.encoder import AccentEncoder
from accentbridge.model.injection import injection_parameters
from accentbridge.model.trainer import AccentTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", default="proto", choices=["lookup", "proto", "encoder"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default=None, help="checkpoint to initialise LoRA, injection, encoder and table")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-frames", type=int, default=20000)
    ap.add_argument("--accent-dim", type=int, default=128)
    ap.add_argument("--ssl-layer", type=int, default=15)
    ap.add_argument("--exemplar-k", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--accent-drop", type=float, default=0.15)
    ap.add_argument("--audio-drop", type=float, default=0.5)
    ap.add_argument("--accent-lr-mult", type=float, default=5.0)
    ap.add_argument("--lambda-spk", type=float, default=0.05)
    ap.add_argument("--lambda-con", type=float, default=0.3)
    ap.add_argument("--lambda-proto", type=float, default=1.0)
    ap.add_argument("--lambda-proto-ce", type=float, default=1.0)
    ap.add_argument("--proto-p-enc", type=float, default=0.5)
    ap.add_argument("--proto-tau", type=float, default=0.1)
    ap.add_argument("--aug-repeat", type=int, default=1)
    ap.add_argument("--no-bank", action="store_true", help="train on Common Voice (+ aug if present) only")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    C = Config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfm, mel_fn = build_f5(C.f5_ckpt(), a.accent_dim, a.lora_r, a.lora_alpha, a.accent_drop)
    cfm.audio_drop_prob = a.audio_drop
    cfm.to(a.device)
    cv = pd.read_parquet(C.manifest / "cv.parquet")
    bank_path, aug_path = C.manifest / "bank.parquet", C.manifest / "aug.parquet"
    bank = pd.read_parquet(bank_path) if bank_path.exists() and not a.no_bank else None
    aug = pd.read_parquet(aug_path) if aug_path.exists() else None
    rows = build_rows(cv, bank, aug, split="train", aug_repeat=a.aug_repeat)
    print("rows by source:", rows["source"].value_counts().to_dict(), "accents:", rows["exemplar_group"].nunique())
    ds = PairDataset(rows, mel_fn, exemplar_k=a.exemplar_k)
    sampler = FrameBatchSampler(frame_lens(rows), a.max_frames)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=a.workers,
                        persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)

    enc = AccentEncoder(out_dim=a.accent_dim, n_speakers=int(rows["speaker_id"].max()) + 1).to(a.device)
    feat_fn = build_feat_fn(C.ssl_model(), a.device, a.ssl_layer, C.hf_cache) if a.mode != "lookup" else None
    inj = injection_parameters(cfm.transformer)
    lora = [p for p in cfm.transformer.parameters() if p.requires_grad and all(p is not q for q in inj)]
    groups = [dict(params=lora, lr_mult=1.0), dict(params=inj, lr_mult=a.accent_lr_mult, weight_decay=0.0)]
    table, classes = None, None
    if a.mode != "encoder":
        classes = rows.drop_duplicates("accent_id").sort_values("accent_id")["exemplar_group"].tolist()
        table = torch.nn.Embedding(len(classes), a.accent_dim).to(a.device)
        groups.append(dict(params=list(table.parameters()), lr_mult=a.accent_lr_mult, weight_decay=0.0))
        print(f"accent table: {len(classes)} classes {classes}")
    if a.mode != "lookup":
        groups.append(dict(params=list(enc.parameters()), lr_mult=1.0))
    n_tts = sum(p.numel() for p in cfm.transformer.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in cfm.transformer.parameters())
    print(f"rows {len(rows)}  batches/epoch {len(sampler)}  trainable {n_tts / 1e6:.2f}M of {n_all / 1e6:.1f}M")

    tr = AccentTrainer(cfm, enc, feat_fn, groups, mode=a.mode, accent_table=table, classes=classes, lr=a.lr,
                       warmup=a.warmup, total_steps=a.steps, lambda_spk=a.lambda_spk, lambda_con=a.lambda_con,
                       proto_p_enc=a.proto_p_enc, lambda_proto=a.lambda_proto, lambda_proto_ce=a.lambda_proto_ce,
                       proto_tau=a.proto_tau, device=a.device)
    if a.resume and (out / "accent_last.pt").exists():
        tr.load(out / "accent_last.pt")
        print(f"resumed from step {tr.step_n}")
    elif a.init_from:
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        cfm.transformer.load_state_dict({k: v for k, v in ck["adapter"].items() if k != "time_embed.gain"}, strict=False)
        enc.load_state_dict({k: v for k, v in ck["encoder"].items() if not k.startswith("spk_head")}, strict=False)
        if table is not None and ck.get("classes") == classes:
            table.load_state_dict(ck["accent_table"])
        print(f"initialised from {a.init_from} (step {ck.get('step')})")
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    while tr.step_n < a.steps:
        for batch in loader:
            m = tr.step(batch)
            if tr.step_n % a.log_every == 0:
                m.update(step=tr.step_n, elapsed=round(time.time() - t0, 1), lr=tr.opt.param_groups[0]["lr"])
                print(m, flush=True)
                log.write(json.dumps(m) + "\n")
                log.flush()
            if tr.step_n % a.save_every == 0:
                tr.save(out, tag=f"step{tr.step_n}")
                tr.save(out, tag="last")
            if tr.step_n >= a.steps:
                break
    tr.save(out, tag="last")
    print("done", out)


if __name__ == "__main__":
    main()
