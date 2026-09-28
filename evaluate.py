import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from accentbridge.audio import SR, load_audio, resample, slice_seconds, write_wav
from accentbridge.config import Config
from accentbridge.model.build import build_feat_fn, load_accentbridge, load_vocoder
from accentbridge.synth import embed_exemplars, lookup_accent, synthesize


def synth_grid(a, C, grid, weights, out_dir):
    cfm, enc, mel_fn, table, classes = load_accentbridge(a.ckpt, C.f5_ckpt(), a.device, no_adapter=a.no_adapter)
    voc = load_vocoder(a.device, C.vocoder_dir(), C.hf_cache)
    feat_fn = build_feat_fn(C.ssl_model(), a.device, a.ssl_layer, C.hf_cache) if a.mode == "exemplar" else None
    t0, n = time.time(), 0
    for r in grid.itertuples():
        todo = [w for w in weights if not (out_dir / f"{r.item}_w{w:g}.wav").exists()]
        if not todo:
            continue
        if a.mode == "lookup":
            accent = lookup_accent(table, classes, r.target_accent)
        else:
            wavs = json.loads(r.exemplar_wavs)
            accent = embed_exemplars(wavs[:a.n_exemplars] if a.n_exemplars else wavs, feat_fn, enc, a.device,
                                     k=a.exemplar_k, seed=a.seed)
        prompt = slice_seconds(load_audio(r.ref_wav, SR), SR, 0, float(r.split_t))
        for w in todo:
            wav = synthesize(cfm, mel_fn, voc, prompt, r.prompt_text, r.text, accent, w, a.device, nfe=a.nfe,
                             cfg=a.cfg, seed=a.seed)
            write_wav(out_dir / f"{r.item}_w{w:g}.wav", wav, SR)
            n += 1
        if n and n % 50 == 0:
            print(f"{n} synthesized, {(time.time() - t0) / n:.2f} s/utt", flush=True)
    del cfm, enc, voc
    torch.cuda.empty_cache()


def build_probe(a, C, grid, embed):
    from accentbridge.metrics import AccentProbe
    cv = pd.read_parquet(C.manifest / "cv.parquet")
    used = {p for ex in grid.exemplar_wavs for p in json.loads(ex)}
    pool = cv[cv["wav_ok"] & cv["split"].isin(["test", "heldout", "ood"]) & ~cv["wav_path"].isin(used)]
    pool = pool.groupby("accent").head(a.probe_per_accent)[["wav_path", "accent"]]
    bank_path = C.manifest / "bank.parquet"
    if bank_path.exists():
        bank = pd.read_parquet(bank_path)
        bank = bank[~bank["voice_id"].isin(set(grid.ref_voice)) & bank["accent"].isin(set(pool["accent"]))]
        pool = pd.concat([pool, bank.groupby("accent").head(a.probe_per_accent // 2)[["wav_path", "accent"]]])
    E = np.stack([embed(resample(load_audio(p, SR), SR, 16000)) for p in pool["wav_path"]])
    print(f"accent probe on {len(pool)} clips, {pool['accent'].nunique()} accents", flush=True)
    return AccentProbe(E, pool["accent"].to_numpy())


def score_grid(a, C, grid, weights, out_dir):
    from transformers import AutoModel
    from accentbridge.metrics import UTMOS, SpeakerSim, WhisperWER, ssl_embed_fn
    ssl = AutoModel.from_pretrained(C.ssl_model(), cache_dir=C.hf_cache).eval().to(a.device)
    embed = ssl_embed_fn(ssl, a.ssl_layer, a.device)
    probe = build_probe(a, C, grid, embed)
    sim, mos, asr = SpeakerSim(a.device, savedir=str(C.ckpt / "ecapa")), UTMOS(a.device), WhisperWER(device=a.device)
    rows = []
    for r in grid.itertuples():
        ref16 = resample(slice_seconds(load_audio(r.ref_wav, SR), SR, 0, float(r.split_t)), SR, 16000)
        cands = [(w, out_dir / f"{r.item}_w{w:g}.wav") for w in weights]
        if a.real:
            cands.append((-1.0, Path(json.loads(r.exemplar_wavs)[1])))
        for w, p in cands:
            if not p.exists():
                continue
            g16 = resample(load_audio(p, SR), SR, 16000)
            rec = dict(item=r.item, w=w, target_accent=r.target_accent, condition=r.condition,
                       spk_sim=sim.cosine(g16, ref16), wer=asr(r.text, g16), utmos=mos.score(g16))
            rec.update(probe.score(embed(g16), r.target_accent))
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="exemplar", choices=["exemplar", "lookup"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", default="0,9,15")
    ap.add_argument("--conditions", default=None, help="subset of seen,heldout,ood")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exemplar-k", type=int, default=2)
    ap.add_argument("--n-exemplars", type=int, default=0, help="use only the first n exemplar clips (0 = all)")
    ap.add_argument("--no-adapter", action="store_true", help="stock F5-TTS (adapter not loaded)")
    ap.add_argument("--real", action="store_true", help="also score a real clip of the target accent (w=-1)")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--ssl-layer", type=int, default=15)
    ap.add_argument("--probe-per-accent", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    C = Config(a.config)
    weights = [float(x) for x in a.weights.split(",")]
    out_dir = C.data / "eval" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(C.manifest / "eval_grid.parquet")
    if a.conditions:
        grid = grid[grid["condition"].isin(a.conditions.split(","))].reset_index(drop=True)
    if a.limit:
        grid = grid.sample(a.limit, random_state=a.seed).reset_index(drop=True)
    grid["item"] = [f"{r.ref_voice}__{r.target_accent}__{i}" for i, r in enumerate(grid.itertuples())]
    if a.mode == "lookup":
        classes = torch.load(a.ckpt, map_location="cpu", weights_only=False)["classes"]
        grid = grid[grid["target_accent"].isin(classes)].reset_index(drop=True)
    if not a.skip_synth:
        synth_grid(a, C, grid, weights, out_dir)
    df = score_grid(a, C, grid, weights, out_dir)
    df.to_parquet(out_dir / "scores.parquet", index=False)
    g = df.groupby(["condition", "w"])
    summ = pd.DataFrame({"n": g.size(), "acc": g["probe_hit"].mean(), "p_target": g["p_target"].mean(),
                         "spk": g["spk_sim"].mean(), "wer": g["wer"].mean(), "utmos": g["utmos"].mean()})
    summ.round(4).to_csv(out_dir / "summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
