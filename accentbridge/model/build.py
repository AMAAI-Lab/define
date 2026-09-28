import torch
import torch.nn.functional as F

from .cfm import AccentCFM
from .encoder import AccentEncoder
from .injection import injection_parameters
from .lora import apply_lora, load_f5_state


# Frozen F5-TTS v1 Base + accent injection + LoRA; returns (AccentCFM, MelSpec).
def build_f5(f5_ckpt, accent_dim=128, lora_r=16, lora_alpha=32, accent_drop=0.15):
    from importlib.resources import files

    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.model.modules import MelSpec
    from f5_tts.model.utils import get_tokenizer
    cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    vocab_map, vocab_size = get_tokenizer(str(files("f5_tts").joinpath("infer/examples/vocab.txt")), "custom")
    mel_kw = dict(cfg.model.mel_spec)
    mel_kw.pop("mel_spec_type", None)
    dit = get_class(f"f5_tts.model.{cfg.model.backbone}")(**cfg.model.arch, text_num_embeds=vocab_size,
                                                          mel_dim=mel_kw["n_mel_channels"])
    cfm = AccentCFM(transformer=dit, mel_spec_kwargs=mel_kw, vocab_char_map=vocab_map, accent_dim=accent_dim,
                    accent_drop_prob=accent_drop)
    if f5_ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(f5_ckpt)
    else:
        state = torch.load(f5_ckpt, map_location="cpu")
    load_f5_state(cfm, state)
    apply_lora(cfm.transformer, r=lora_r, alpha=lora_alpha)
    for p in injection_parameters(cfm.transformer):
        p.requires_grad_(True)
    return cfm, MelSpec(**mel_kw)


# Frozen XLS-R feature extractor: 16 kHz crops (N, S) -> (hidden states at `layer`, frame lengths).
def build_feat_fn(ssl_model, device, layer=15, cache_dir=None):
    from transformers import AutoModel
    ssl = AutoModel.from_pretrained(ssl_model, cache_dir=cache_dir).eval().to(device)
    for p in ssl.parameters():
        p.requires_grad_(False)

    def fn(crops):
        with torch.no_grad():
            out = ssl(crops.to(device), output_hidden_states=True)
        n = torch.full((crops.shape[0],), crops.shape[1], device=device)
        return out.hidden_states[layer], ssl._get_feat_extract_output_lengths(n)
    return fn


def load_vocoder(device, local_dir=None, hf_cache=None):
    from f5_tts.infer.utils_infer import load_vocoder as _load
    return _load("vocos", is_local=local_dir is not None, local_path=local_dir or "", device=device,
                 hf_cache_dir=hf_cache)


# Load a trained AccentBridge checkpoint for inference; returns (cfm, encoder, mel_fn, table, classes).
def load_accentbridge(ckpt, f5_ckpt, device, accent_dim=128, no_adapter=False):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfm, mel_fn = build_f5(f5_ckpt, accent_dim, accent_drop=0.0)
    if not no_adapter:
        cfm.transformer.load_state_dict(ck["adapter"], strict=False)
    enc = AccentEncoder(out_dim=accent_dim)
    enc.load_state_dict({k: v for k, v in ck["encoder"].items() if not k.startswith("spk_head")})
    table, classes = None, None
    if "accent_table" in ck and not no_adapter:
        table = F.normalize(ck["accent_table"]["weight"].float(), dim=-1).to(device)
        classes = list(ck["classes"])
    return cfm.to(device).eval(), enc.to(device).eval(), mel_fn.to(device), table, classes
