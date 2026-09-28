from pathlib import Path

import yaml

F5_REPO, F5_FILE = "SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors"
SSL_REPO = "facebook/wav2vec2-xls-r-300m"


class Config:
    # Paths from config.yaml; relative entries resolve against the config file's folder.
    def __init__(self, path="config.yaml"):
        self.file = Path(path).resolve()
        self.base = self.file.parent
        self.cfg = yaml.safe_load(self.file.read_text()) or {}

    def path(self, key):
        v = self.cfg.get(key)
        if v is None:
            return None
        p = Path(v).expanduser()
        return p if p.is_absolute() else self.base / p

    @property
    def data(self):
        return self.path("data_root")

    @property
    def manifest(self):
        return self.data / "manifest"

    @property
    def ckpt(self):
        return self.path("ckpt_root")

    @property
    def hf_cache(self):
        return str(self.path("hf_cache"))

    def f5_ckpt(self):
        p = self.path("f5_ckpt")
        if p is not None and p.exists():
            return str(p)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(F5_REPO, F5_FILE, cache_dir=self.hf_cache)

    def ssl_model(self):
        p = self.path("ssl_model")
        return str(p) if p is not None and p.exists() else SSL_REPO

    def vocoder_dir(self):
        p = self.path("vocoder")
        return str(p) if p is not None and p.exists() else None
