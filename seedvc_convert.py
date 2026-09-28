import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# Resample to sr_out and normalise RMS to target_dbfs with a peak limit.
def postprocess(audio, sr_in, sr_out, target_dbfs=-20.0):
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr_in != sr_out:
        n = int(round(len(y) * sr_out / sr_in))
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    return y * min(10 ** (target_dbfs / 20) / rms, 0.99 / (np.abs(y).max() + 1e-8))


def run_convert(wrapper, src, tgt, steps, cfg):
    gen = wrapper.convert_voice(str(src), str(tgt), diffusion_steps=steps, inference_cfg_rate=cfg,
                                f0_condition=False, stream_output=False)
    if not hasattr(gen, "__next__"):
        return gen
    last = None
    try:
        while True:
            _, last = next(gen)
    except StopIteration as e:
        return e.value if e.value is not None else last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    pairs = pd.read_csv(os.path.abspath(a.pairs)).iloc[k::n]
    os.chdir(os.environ["SEEDVC_DIR"])
    sys.path.insert(0, os.environ["SEEDVC_DIR"])
    from seed_vc_wrapper import SeedVCWrapper
    import torch
    todo = pairs[~pairs["out_wav"].map(lambda p: Path(p).exists())]
    print(f"{len(pairs)} pairs, {len(todo)} to convert")
    if not len(todo):
        return
    w = SeedVCWrapper(device=torch.device(a.device))
    for r in todo.itertuples():
        try:
            audio = run_convert(w, r.source_wav, r.target_wav, a.steps, a.cfg)
            sf.write(r.out_wav, postprocess(audio, 22050, a.sr), a.sr, subtype="PCM_16")
        except Exception as e:
            print(f"[fail] {r.source_wav}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
