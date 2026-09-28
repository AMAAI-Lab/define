import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

SR = 24000


# Load mono float32 audio at sample rate sr.
def load_audio(path, sr=SR):
    try:
        y, s = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        y, s = librosa.load(str(path), sr=None, mono=True)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr, res_type="soxr_hq")
    return np.ascontiguousarray(y, dtype=np.float32)


def resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)


# Normalise to target LUFS with a peak limit.
def loudness_normalize(y, sr, target_lufs=-23.0, peak_dbfs=-1.0):
    lufs = pyln.Meter(sr).integrated_loudness(y.astype(np.float64))
    if np.isfinite(lufs):
        y = y * 10 ** ((target_lufs - lufs) / 20)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    lim = 10 ** (peak_dbfs / 20)
    if peak > lim:
        y = y * (lim / peak)
    return y.astype(np.float32)


def frame_db(y, sr, frame_ms=25, hop_ms=10):
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((y[idx] ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms), hop, frame


def _runs(mask):
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


# Energy VAD: list of (start_s, end_s) speech segments.
def speech_segments(y, sr, thresh_db=-40.0, floor_db=-60.0, min_speech_ms=100, min_sil_ms=150, frame_ms=25, hop_ms=10):
    db, hop, frame = frame_db(y, sr, frame_ms, hop_ms)
    speech = db > max(np.percentile(db, 95) + thresh_db, floor_db)
    merged = []
    for s, e in _runs(speech):
        if merged and (s - merged[-1][1]) * hop_ms < min_sil_ms:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    merged = [(s, e) for s, e in merged if (e - s) * hop_ms >= min_speech_ms]
    return [(s * hop / sr, min(len(y), (e - 1) * hop + frame) / sr) for s, e in merged]


# Trim leading/trailing silence; returns (audio, offset_s).
def trim_silence(y, sr, pad_ms=50):
    segs = speech_segments(y, sr)
    if not segs:
        return y, 0.0
    a = max(0, int((segs[0][0] - pad_ms / 1000) * sr))
    b = min(len(y), int((segs[-1][1] + pad_ms / 1000) * sr))
    return y[a:b], a / sr


def choose_split(segs, dur, min_prompt=2.5, max_prompt=6.0, min_body=4.0, snap_margin=0.35):
    hi = min(max_prompt, dur - min_body)
    if hi < min_prompt:
        return None, False
    cands = [(a_end + b_start) / 2 for (_, a_end), (b_start, _) in zip(segs[:-1], segs[1:])]
    cands = [t for t in cands if min_prompt <= t <= hi]
    if cands:
        return float(max(cands)), True
    return float(max(min_prompt, hi - snap_margin)), False


def nearest_zero_crossing(y, sr, t, window_ms=50):
    c = int(t * sr)
    a = max(1, c - int(window_ms / 1000 * sr))
    b = min(c, len(y) - 1)
    if b <= a:
        return float(t)
    seg = y[a:b + 1]
    zc = np.where(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0] + a + 1
    if len(zc) == 0:
        return float(t)
    return float(zc[np.argmin(np.abs(zc - c))] / sr)


# Pick a prompt/body split time at a pause; returns (split_s or None, clean, segments).
def split_point(y, sr, min_prompt=2.5, max_prompt=6.0, min_body=4.0, snap_margin=0.35):
    segs = speech_segments(y, sr)
    t, clean = choose_split(segs, len(y) / sr, min_prompt, max_prompt, min_body, snap_margin)
    if t is None:
        return None, False, segs
    if not clean:
        t = max(nearest_zero_crossing(y, sr, t), min_prompt)
    return float(t), clean, segs


def slice_seconds(y, sr, start, end=None):
    a = int(round(start * sr))
    b = len(y) if end is None else int(round(end * sr))
    return y[a:b]


def write_wav(path, y, sr):
    sf.write(str(path), y, sr, subtype="PCM_16")


def pad_to(y, n):
    if len(y) >= n:
        return y[:n]
    return np.tile(y, int(np.ceil(n / max(len(y), 1))))[:n]


# Fixed-length crops over speech regions: (k, win_s*sr) array.
def speech_crops(y, sr, win_s=4.0, k=1, rng=None, min_coverage=0.8, stride_s=0.05):
    y = np.asarray(y, dtype=np.float32)
    win = int(win_s * sr)
    rng = rng or np.random.default_rng()
    segs = speech_segments(y, sr)
    if not segs:
        if len(y) <= win:
            starts = [0] * k
        elif k == 1:
            starts = [int(rng.integers(0, len(y) - win + 1))]
        else:
            starts = [int(round(x)) for x in np.linspace(0, len(y) - win, k)]
        return np.stack([pad_to(y[s:s + win], win) for s in starts])
    parts = [y[int(t0 * sr):int(t1 * sr)] for t0, t1 in segs]
    speech = np.concatenate([p for p in parts if len(p)] or [y])
    if len(y) < win or len(speech) < win * min_coverage:
        return np.stack([pad_to(speech, win)] * k)
    mask = np.zeros(len(y), dtype=bool)
    for t0, t1 in segs:
        mask[int(t0 * sr):int(t1 * sr)] = True
    csum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    starts = np.arange(0, len(y) - win + 1, max(1, int(stride_s * sr)))
    if len(starts) == 0:
        return np.stack([pad_to(speech, win)] * k)
    cov = (csum[starts + win] - csum[starts]) / win
    valid = starts[cov >= min_coverage]
    if len(valid) == 0:
        chosen = np.sort(starts[np.argsort(-cov)[:max(k, 1)]])
    elif k == 1:
        chosen = np.array([int(rng.choice(valid))])
    else:
        chosen = valid[np.unique(np.round(np.linspace(0, len(valid) - 1, k)).astype(int))]
    out = [pad_to(y[s:s + win], win) for s in chosen]
    while len(out) < k:
        out.append(out[len(out) % len(chosen)])
    return np.stack(out[:k])
