import numpy as np
import torch

from .audio import speech_crops


class SpeakerSim:
    # ECAPA-TDNN cosine similarity between two 16 kHz waveforms.
    def __init__(self, device="cuda", savedir=None):
        from speechbrain.inference.speaker import EncoderClassifier
        self.enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir,
                                                  run_opts={"device": device})
        self.device = device

    def embed(self, wav16):
        with torch.inference_mode():
            e = self.enc.encode_batch(torch.from_numpy(np.asarray(wav16, dtype=np.float32))[None].to(self.device))
        return torch.nn.functional.normalize(e.squeeze(), dim=-1).cpu().numpy()

    def cosine(self, a, b):
        return float(np.dot(self.embed(a), self.embed(b)))


class UTMOS:
    # UTMOS22 strong naturalness score for a 16 kHz waveform.
    def __init__(self, device="cuda"):
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True,
                                    skip_validation=True).to(device)
        self.device = device

    def score(self, wav16):
        with torch.inference_mode():
            return float(self.model(torch.from_numpy(np.asarray(wav16, dtype=np.float32))[None].to(self.device),
                                    sr=16000).item())


class WhisperWER:
    # Whisper large-v3 transcription and normalised word error rate.
    def __init__(self, model="large-v3", device="cuda"):
        import jiwer
        from faster_whisper import WhisperModel
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        self.model = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")
        self.norm = EnglishTextNormalizer({})
        self.jiwer = jiwer

    def __call__(self, ref, wav16):
        segs, _ = self.model.transcribe(wav16, language="en", beam_size=5)
        hyp = " ".join(s.text.strip() for s in segs).strip()
        r, h = self.norm(ref), self.norm(hyp)
        if not r.strip():
            return float("nan")
        return float(self.jiwer.wer(r, h if h.strip() else "<empty>"))


# Mean XLS-R hidden state at `layer` over k speech crops: 16 kHz wav -> (1024,).
def ssl_embed_fn(model, layer, device, win_s=4.0, k=2):
    def f(wav16):
        x = torch.from_numpy(speech_crops(wav16, 16000, win_s=win_s, k=k)).to(device)
        with torch.inference_mode():
            hs = model(x, output_hidden_states=True).hidden_states[layer]
        return hs.mean(dim=(0, 1)).float().cpu().numpy()
    return f


class AccentProbe:
    # Logistic-regression accent classifier on SSL embeddings; score() returns hit and target probability.
    def __init__(self, emb, labels, C=0.5):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C)).fit(emb, labels)
        self.classes = list(self.clf.classes_)

    def score(self, emb, target):
        p = self.clf.predict_proba(emb[None])[0]
        pred = self.classes[int(np.argmax(p))]
        return dict(probe_pred=pred, probe_hit=pred == target,
                    p_target=float(p[self.classes.index(target)]) if target in self.classes else np.nan)
