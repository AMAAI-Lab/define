import argparse
import json

import numpy as np
import pandas as pd

from accentbridge.config import Config
from accentbridge.labels import condition


# Test-split prompt-bank voices, n_per_cell per (accent, gender).
def pick_refs(bank, n_per_cell=4, seed=0):
    df = bank[bank["split"] == "test"]
    return (df.sample(frac=1.0, random_state=seed).groupby(["accent", "gender"]).head(n_per_cell)
            .reset_index(drop=True))


# CV exemplar pool: unseen test speakers of seen accents plus all held-out and OOD speakers.
def exemplar_pool(cv):
    return cv[cv["wav_ok"] & cv["split"].isin(["test", "heldout", "ood"])]


def pick_texts(pool, n, min_words=6, max_words=14, seed=0):
    t = pool["text"].drop_duplicates()
    nw = t.str.split().str.len()
    t = t[(nw >= min_words) & (nw <= max_words)]
    return t.sample(min(n, len(t)), random_state=seed).tolist()


# Every reference voice x every target accent x n_text sentences, with k exemplar clips from k distinct speakers.
def build_grid(refs, pool, texts, k=3, n_text=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in refs.itertuples():
        for acc, g in pool.groupby("accent"):
            spk = g["speaker"].drop_duplicates().to_numpy()
            if len(spk) < k:
                continue
            ex = g[g["speaker"].isin(rng.choice(spk, k, replace=False))].groupby("speaker").head(1)
            for t in rng.choice(texts, n_text, replace=False):
                rows.append(dict(ref_voice=r.voice_id, ref_wav=r.wav_path, ref_accent=r.accent, ref_gender=r.gender,
                                 split_t=r.split_t, prompt_text=r.prompt_text, target_accent=acc, text=t,
                                 exemplar_wavs=json.dumps(ex["wav_path"].tolist()), condition=condition(acc)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-per-cell", type=int, default=4)
    ap.add_argument("--n-text", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-texts-pool", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    C = Config(a.config)
    pool = exemplar_pool(pd.read_parquet(C.manifest / "cv.parquet"))
    refs = pick_refs(pd.read_parquet(C.manifest / "bank.parquet"), a.n_per_cell, a.seed)
    grid = build_grid(refs, pool, pick_texts(pool, a.n_texts_pool, seed=a.seed), a.k, a.n_text, a.seed)
    grid.to_parquet(C.manifest / "eval_grid.parquet", index=False)
    print(f"{len(grid)} items, {grid.ref_voice.nunique()} reference voices, {grid.target_accent.nunique()} target accents")
    print(grid.groupby("condition").size().to_string())


if __name__ == "__main__":
    main()
