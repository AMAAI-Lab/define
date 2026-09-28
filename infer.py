import argparse
from pathlib import Path

from accentbridge.audio import SR, load_audio, write_wav
from accentbridge.config import Config
from accentbridge.model.build import build_feat_fn, load_accentbridge, load_vocoder
from accentbridge.synth import embed_exemplars, lookup_accent, synthesize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref-wav", required=True, help="voice prompt (a few seconds of speech)")
    ap.add_argument("--ref-text", required=True, help="transcript of the voice prompt")
    ap.add_argument("--text", required=True, help="text to synthesise")
    ap.add_argument("--accent", default=None, help="accent label (lookup mode)")
    ap.add_argument("--exemplars", nargs="+", default=None, help="accent exemplar clips (exemplar mode)")
    ap.add_argument("--w", type=float, default=9.0, help="accent guidance weight (0 = accent off)")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if (a.accent is None) == (a.exemplars is None):
        ap.error("give exactly one of --accent or --exemplars")
    C = Config(a.config)
    cfm, enc, mel_fn, table, classes = load_accentbridge(a.ckpt, C.f5_ckpt(), a.device)
    voc = load_vocoder(a.device, C.vocoder_dir(), C.hf_cache)
    if a.accent is not None:
        accent = lookup_accent(table, classes, a.accent)
    else:
        feat_fn = build_feat_fn(C.ssl_model(), a.device, cache_dir=C.hf_cache)
        accent = embed_exemplars(a.exemplars, feat_fn, enc, a.device, seed=a.seed)
    wav = synthesize(cfm, mel_fn, voc, load_audio(a.ref_wav, SR), a.ref_text, a.text, accent, a.w, a.device,
                     nfe=a.nfe, cfg=a.cfg, speed=a.speed, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_wav(a.out, wav, SR)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
