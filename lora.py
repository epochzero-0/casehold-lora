"""LoRA, written out: a frozen linear layer plus a trained low-rank correction."""
import math

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * B A x

    base is the pretrained layer, frozen. A is (r, in), B is (out, r), and only they train.
    B starts at zero, so at step 0 the correction is zero and the model is the pretrained one.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        w = base.weight
        # A and B stay float32 whatever the base is stored in: AdamW's small updates vanish in bf16
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=w.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=w.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        delta = (x.to(self.A.dtype) @ self.A.T @ self.B.T) * self.scale
        return self.base(x) + delta.to(x.dtype)


def inject(model, r=16, alpha=32, targets=TARGETS):
    """Freeze every weight, then wrap the attention projections of every layer.
    Returns {"layers.3.q_proj": LoRALinear, ...}."""
    for p in model.parameters():
        p.requires_grad = False
    wrapped = {}
    for i, layer in enumerate(model.model.layers):
        for name in targets:
            lora = LoRALinear(getattr(layer.self_attn, name), r, alpha)
            setattr(layer.self_attn, name, lora)
            wrapped[f"layers.{i}.{name}"] = lora
    return wrapped


def lora_state(model):
    """Only the trained tensors: this is the whole adapter."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}


def load_adapter(model, path, r=16, alpha=32):
    """Wrap the model and load trained A and B into it."""
    inject(model, r, alpha)
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and not [k for k in missing if k.endswith((".A", ".B"))]
    return model


def count(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
