import argparse
import glob
import hashlib
import os
import tarfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

from accentbridge.audio import SR, load_audio, loudness_normalize, trim_silence, write_wav
from accentbridge.config import Config
from accentbridge.labels import EXCLUDE, HELDOUT, OOD, cv_accent


def load_tsv(en_dir):
    v = pd.read_csv(os.path.join(en_dir, "validated.tsv"), sep="\t", quoting=3, low_memory=False, dtype=str)
    d = pd.read_csv(os.path.join(en_dir, "clip_durations.tsv"), sep="\t", quoting=3, dtype={"clip": str})
    d.columns = ["path", "duration_ms"]
    v = v.merge(d, on="path", how="left")
    v["duration_s"] = pd.to_numeric(v["duration_ms"], errors="coerce") / 1000
    return v


# Speaker-capped, hour-capped subset per accent.
def select(v, max_hours=25.0, max_min_per_speaker=15.0, min_clips=3, min_s=2.0, max_s=10.0, min_speakers=50, seed=0):
    rng = np.random.default_rng(seed)
    df = v.copy()
    df["accent"] = df["accents"].fillna("").map(cv_accent)
    df = df[df["accent"].notna() & ~df["accent"].isin(EXCLUDE)]
    df = df[(df["duration_s"] >= min_s) & (df["duration_s"] <= max_s)]
    n = df.groupby("client_id").size()
    df = df[df["client_id"].isin(n[n >= min_clips].index)].sample(frac=1.0, random_state=seed)
    df = df[df.groupby("client_id")["duration_s"].cumsum() <= max_min_per_speaker * 60]
    out = []
    for _, g in df.groupby("accent"):
        spk_h = g.groupby("client_id")["duration_s"].sum() / 3600
        if len(spk_h) < min_speakers:
            continue
        order = spk_h.index.to_numpy()
        rng.shuffle(order)
        cum = spk_h.loc[order].cumsum()
        keep = cum.index[cum <= max_hours]
        if len(keep) < len(order) and len(keep) < min_speakers:
            keep = order[:min_speakers]
        out.append(g[g["client_id"].isin(set(keep))])
    sel = pd.concat(out).reset_index(drop=True)
    sel["speaker"] = sel["client_id"]
    sel["utt_id"] = "cv_" + sel["path"].str.replace(".mp3", "", regex=False).str.replace("common_voice_en_", "", regex=False)
    return sel


# Speaker-disjoint train/val/test (5%/5%) for seen accents; held-out and OOD accents are test-only.
def assign_split(sel, val_frac=0.05, test_frac=0.05):
    spk = sel.drop_duplicates("speaker")[["speaker", "accent"]]
    spk["h"] = spk["speaker"].map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    spk["q"] = spk.groupby("accent")["h"].rank(pct=True, method="first")
    q = sel["speaker"].map(spk.set_index("speaker")["q"]).to_numpy()
    split = np.where(q <= test_frac, "test", np.where(q <= test_frac + val_frac, "val", "train")).astype(object)
    split[sel["accent"].isin(HELDOUT).to_numpy()] = "heldout"
    split[sel["accent"].isin(OOD).to_numpy()] = "ood"
    sel["split"] = split
    return sel


def extract(archive, members, dest):
    todo = {m for m in members if not os.path.exists(os.path.join(dest, m))}
    print(f"{len(members)} clips, {len(todo)} to extract", flush=True)
    if not todo:
        return
    with tarfile.open(archive, "r|gz", bufsize=16 * 1024 * 1024) as tar:
        for m in tar:
            if m.name in todo:
                out = os.path.join(dest, m.name)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as w:
                    w.write(tar.extractfile(m).read())
                todo.discard(m.name)
                if not todo:
                    break
    print(f"missing after extraction: {len(todo)}")


def convert(task):
    src, out = task
    if os.path.exists(out):
        return "ok"
    try:
        y, _ = trim_silence(loudness_normalize(load_audio(src, SR), SR), SR)
        if len(y) < SR:
            return "short"
        write_wav(out, y, SR)
        return "ok"
    except Exception:
        return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--archive", default=None, help="Common Voice English .tar.gz; clips are extracted if given")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    C = Config(a.config)
    root = C.path("cv_dir")
    en = sorted(glob.glob(os.path.join(root, "cv-corpus-*", "en")))[-1]
    corpus = os.path.basename(os.path.dirname(en))
    sel = assign_split(select(load_tsv(en), seed=a.seed))
    sel["mp3"] = f"{corpus}/en/clips/" + sel["path"]
    wav_dir = C.data / "cv24"
    wav_dir.mkdir(parents=True, exist_ok=True)
    sel["wav_path"] = [str(wav_dir / f"{u}.wav") for u in sel["utt_id"]]
    if a.archive:
        extract(a.archive, set(sel["mp3"]), root)
    with Pool(a.workers) as pool:
        status = pool.map(convert, [(os.path.join(root, m), w) for m, w in zip(sel["mp3"], sel["wav_path"])], chunksize=64)
    sel["wav_ok"] = np.array(status) == "ok"
    sel = sel[["utt_id", "speaker", "accent", "gender", "sentence", "duration_s", "wav_path", "wav_ok", "split"]]
    C.manifest.mkdir(parents=True, exist_ok=True)
    sel.rename(columns={"sentence": "text"}).to_parquet(C.manifest / "cv.parquet", index=False)
    print(f"{int(sel['wav_ok'].sum())} usable clips, {sel['speaker'].nunique()} speakers, {sel['accent'].nunique()} accents")
    print(sel.groupby(["accent", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
