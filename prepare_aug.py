import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from accentbridge.config import Config
from accentbridge.labels import UNSEEN


# Pair each sampled CV training clip with a prompt-bank voice of a different accent.
def build_pairs(cv, bank, out_dir, clips_per_accent=2000, seed=0):
    rng = np.random.default_rng(seed)
    bank = bank[(bank["split"] == "train") & ~bank["accent"].isin(UNSEEN) & (bank["duration_s"] >= 5.0)]
    src = cv[cv["wav_ok"] & (cv["split"] == "train")]
    src = pd.concat([g.sample(min(clips_per_accent, len(g)), random_state=seed) for _, g in src.groupby("accent")])
    rows = []
    for r in src.itertuples():
        cands = bank[bank["accent"] != r.accent]
        t = cands.iloc[int(rng.integers(len(cands)))]
        rows.append(dict(utt_id=f"{r.utt_id}__{t.voice_id}", accent=r.accent, text=r.text, source_wav=r.wav_path,
                         target_voice=t.voice_id, target_wav=t.wav_path, prompt_split_t=float(t.split_t),
                         prompt_text=t.prompt_text, out_wav=str(out_dir / f"{r.utt_id}__{t.voice_id}.wav")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--clips-per-accent", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--skip-vc", action="store_true", help="only assemble the manifest from converted files")
    a = ap.parse_args()
    C = Config(a.config)
    out_dir = C.data / "aug"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = C.manifest / "aug_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path, dtype={"target_voice": str})
    else:
        pairs = build_pairs(pd.read_parquet(C.manifest / "cv.parquet"), pd.read_parquet(C.manifest / "bank.parquet"),
                            out_dir, a.clips_per_accent)
        pairs.to_csv(pairs_path, index=False)
    print(f"{len(pairs)} pairs over {pairs.accent.nunique()} accents")
    if not a.skip_vc:
        vc_dir = str(C.path("seedvc_dir"))
        env = dict(os.environ, SEEDVC_DIR=vc_dir)
        script = str(Path(__file__).resolve().parent / "seedvc_convert.py")
        subprocess.run([C.cfg.get("seedvc_python") or sys.executable, script, "--pairs", str(pairs_path),
                        "--steps", str(a.steps), "--shard", a.shard], env=env, check=True)
    ok = pairs["out_wav"].map(os.path.exists).to_numpy()
    p = pairs[ok]
    aug = pd.DataFrame(dict(source="aug", voice_id=p["target_voice"], wav_path=p["out_wav"],
                            duration_s=[sf.info(f).duration for f in p["out_wav"]], split_t=np.nan,
                            prompt_wav_path=p["target_wav"], prompt_split_t=p["prompt_split_t"],
                            prompt_text=p["prompt_text"], body_text=p["text"], exemplar_group=p["accent"],
                            max_prompt_s=np.nan))
    aug.to_parquet(C.manifest / "aug.parquet", index=False)
    print(f"converted {int(ok.sum())} / {len(pairs)}  hours {aug.duration_s.sum() / 3600:.1f}")


if __name__ == "__main__":
    main()
