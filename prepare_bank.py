import argparse
import re
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd

from accentbridge.audio import SR, load_audio, loudness_normalize, split_point, trim_silence, write_wav
from accentbridge.config import Config
from accentbridge.labels import bank_accent

MIN_PROMPT, MAX_PROMPT, MIN_BODY = 2.5, 6.0, 4.0


# Normalise, trim and find a prompt/body split at a pause; returns a row dict.
def prep_clip(task):
    vid, src, out = task
    try:
        y, _ = trim_silence(loudness_normalize(load_audio(src, SR), SR), SR)
        dur = len(y) / SR
        t, _, _ = split_point(y, SR, MIN_PROMPT, MAX_PROMPT, MIN_BODY)
        if t is None or dur - t < MIN_BODY - 1e-6 or t < MIN_PROMPT - 1e-6:
            return dict(voice_id=vid, ok=False)
        write_wav(out, y, SR)
        return dict(voice_id=vid, ok=True, wav_path=str(out), duration_s=dur, split_t=t)
    except Exception:
        return dict(voice_id=vid, ok=False)


def join_words(words):
    return re.sub(r"\s+([,.!?;:])", r"\1", " ".join(w["word"].strip() for w in words)).strip()


def non_latin(text):
    return any(ch.isalpha() and "LATIN" not in unicodedata.name(ch, "") for ch in text)


# Move the split to the closest word gap that keeps both parts long enough; returns (prompt, body, split_s).
def assign_words(words, split_t, dur, gap_cap=0.15):
    words = sorted(words, key=lambda w: (w["start"], w["end"]))
    cands = []
    for a, b in zip(words[:-1], words[1:]):
        gap = b["start"] - a["end"]
        cands.append(a["end"] + min(gap / 2, gap_cap) if gap > 0 else (a["end"] + b["start"]) / 2)
    if cands:
        ok = [t for t in cands if t >= MIN_PROMPT - 1e-9 and dur - t >= MIN_BODY - 1e-9]
        split_t = min(ok, key=lambda t: abs(t - split_t)) if ok else \
            max(cands, key=lambda t: min(t - MIN_PROMPT, dur - t - MIN_BODY))
    return [w for w in words if w["end"] <= split_t], [w for w in words if w["start"] >= split_t], float(split_t)


# Whisper transcript with word timestamps -> prompt/body texts; None if the clip fails the quality filters.
def transcribe(model, r):
    segs, _ = model.transcribe(r.wav_path, language="en", word_timestamps=True, beam_size=5)
    segs = list(segs)
    if not segs:
        return None
    words = [dict(start=float(w.start), end=float(w.end), word=w.word.strip()) for s in segs for w in (s.words or [])]
    durs = [max(1e-3, s.end - s.start) for s in segs]
    logprob = float(np.average([s.avg_logprob for s in segs], weights=durs))
    no_speech = float(max(s.no_speech_prob for s in segs))
    prompt, body, t = assign_words(words, r.split_t, r.duration_s)
    pt, bt = join_words(prompt), join_words(body)
    if not prompt or len(body) < 8 or logprob < -0.6 or no_speech > 0.4 or non_latin(pt + bt) \
            or t < 1.5 or r.duration_s - t < 2.5:
        return None
    return dict(split_t=t, prompt_text=pt, body_text=bt)


# Voice-level 80/10/10 split stratified by accent, gender and age.
def make_splits(df, seed=0, fracs=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    key = df["accent"].fillna("?") + "|" + df["gender"].fillna("?").astype(str) + "|" + df["age"].fillna("?").astype(str)
    split = {}
    for _, g in df.groupby(key):
        ids = np.array(sorted(g["voice_id"]))
        rng.shuffle(ids)
        n_val, n_test = int(round(len(ids) * fracs[1])), int(round(len(ids) * fracs[2]))
        split.update({v: "test" for v in ids[:n_test]})
        split.update({v: "val" for v in ids[n_test:n_test + n_val]})
        split.update({v: "train" for v in ids[n_test + n_val:]})
    return df["voice_id"].map(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    C = Config(a.config)
    meta = pd.read_csv(C.path("bank_csv"))
    for c in ("gender", "age"):
        if c not in meta:
            meta[c] = None
    meta = meta.rename(columns={"id": "voice_id"})
    meta["voice_id"] = meta["voice_id"].astype(str)
    meta["accent"] = meta["accent"].map(bank_accent)
    meta = meta[meta["accent"].notna()]
    meta["split"] = make_splits(meta)
    base = C.path("bank_csv").parent
    out_dir = C.data / "bank24"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(v, p if str(p).startswith("/") else str(base / p), out_dir / f"{v}.wav")
             for v, p in zip(meta["voice_id"], meta["path"])]
    with Pool(a.workers) as pool:
        prep = pd.DataFrame(pool.map(prep_clip, tasks, chunksize=4))
    prep = prep[prep["ok"]].drop(columns="ok")
    print(f"audio prep: {len(prep)} / {len(meta)} clips kept")
    from faster_whisper import WhisperModel
    model = WhisperModel(a.whisper, device=a.device, compute_type="float16" if a.device == "cuda" else "int8")
    rows = []
    for r in prep.itertuples():
        t = transcribe(model, r)
        if t is not None:
            rows.append(dict(voice_id=r.voice_id, wav_path=r.wav_path, duration_s=r.duration_s, **t))
    bank = pd.DataFrame(rows).merge(meta[["voice_id", "accent", "gender", "age", "split"]], on="voice_id")
    bank["gender"] = bank["gender"].fillna("unknown").astype(str).str.lower()
    C.manifest.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(C.manifest / "bank.parquet", index=False)
    print(f"{len(bank)} clips kept after transcription")
    print(bank.groupby(["accent", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
