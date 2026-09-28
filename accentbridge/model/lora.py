TARGETS = r".*transformer_blocks\.\d+\.(attn\.to_q|attn\.to_v|ff\.ff\.0\.0|ff\.ff\.2|attn_norm\.linear)$"


# Inject LoRA into attention q/v, feed-forward and AdaLN projections; only LoRA weights stay trainable.
def apply_lora(dit, r=16, alpha=32):
    from peft import LoraConfig, inject_adapter_in_model
    inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS, bias="none"), dit)
    for n, p in dit.named_parameters():
        p.requires_grad_("lora_" in n)
    return dit


# Trainable state saved in checkpoints: LoRA, accent injection and its wrapped embeddings.
def adapter_state_dict(dit):
    return {n: p.detach().cpu() for n, p in dit.state_dict().items()
            if "lora_" in n or n.startswith("time_embed.") or n.startswith("text_embed.proj") or n == "text_embed.gain"}


# Load the released F5-TTS weights into the wrapped model (renames keys for LoRA and injection wrappers).
def load_f5_state(cfm, state):
    state = state.get("ema_model_state_dict", state)
    wrapped = {n for n, m in cfm.named_modules() if hasattr(m, "base_layer")}
    out = {}
    for k, v in state.items():
        if k in ("initted", "step") or k.endswith(".initted") or k.endswith(".step"):
            continue
        k = k[len("ema_model."):] if k.startswith("ema_model.") else k
        for name in ("time_embed", "text_embed"):
            if k.startswith(f"transformer.{name}."):
                k = k.replace(f"transformer.{name}.", f"transformer.{name}.base.", 1)
        mod, _, leaf = k.rpartition(".")
        out[f"{mod}.base_layer.{leaf}" if mod in wrapped else k] = v
    missing, unexpected = cfm.load_state_dict(out, strict=False)
    hard = [k for k in missing if not (".lora_" in k or k.startswith("mel_spec.") or "time_embed.proj" in k
                                       or "text_embed.proj" in k or k.endswith("time_embed.null")
                                       or k.endswith("_embed.gain"))]
    if hard or unexpected:
        raise RuntimeError(f"missing={hard[:5]} unexpected={list(unexpected)[:5]}")
