from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            if base.bias is not None
            else None
        )
        self.r = int(rank)
        self.scaling = float(alpha) / max(1, self.r)
        self.A = nn.Parameter(torch.zeros((self.r, self.in_features)))
        self.B = nn.Parameter(torch.zeros((self.out_features, self.r)))
        if self.r > 0:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_scale = 1.0

    def forward(self, value):
        result = torch.nn.functional.linear(value, self.weight, self.bias)
        if self.r > 0:
            result = result + self.lora_scale * self.scaling * (
                self.dropout(value) @ self.A.t() @ self.B.t()
            )
        return result


def _matches(name: str, target: str) -> bool:
    patterns = {
        "attn_qkv": r"(q_proj|k_proj|v_proj|to_q|to_k|to_v|W[qkv])$",
        "attn_o": r"(out_proj|to_out|Wo|o_proj)$",
        "ffn_in": r"(fc1|linear1|proj_in|mlp.*(in|fc1))$",
        "ffn_out": r"(fc2|linear2|proj_out|mlp.*(out|fc2))$",
    }
    return re.search(patterns[target], name) is not None


def inject_lora(
    model: nn.Module,
    targets: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(_matches(name, target) for target in targets):
            continue
        parent = model
        parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        for part in parent_name.split(".") if parent_name else ():
            parent = getattr(parent, part)
        setattr(parent, leaf, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    return replaced


def inject_named_lora(
    model: nn.Module,
    names: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    available = dict(model.named_modules())
    missing = [name for name in names if name not in available]
    invalid = [name for name in names if name in available and not isinstance(available[name], nn.Linear)]
    if missing or invalid:
        raise RuntimeError(
            "Cannot inject LoRA: missing={}, non_linear={}".format(missing[:20], invalid[:20])
        )
    replaced = []
    for name in names:
        module = available[name]
        parent = model
        parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        for part in parent_name.split(".") if parent_name else ():
            parent = getattr(parent, part)
        setattr(parent, leaf, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    return replaced


def inject_main_table_lora(model: nn.Module) -> List[str]:
    return inject_lora(
        model,
        ("attn_qkv", "attn_o", "ffn_in", "ffn_out"),
        rank=16,
        alpha=16,
        dropout=0.05,
    )


def freeze_for_lora(model: nn.Module) -> List[str]:
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.endswith(".A") or name.endswith(".B")
        if parameter.requires_grad:
            trainable.append(name)
    return trainable


def set_lora_scale(model: nn.Module, scale: float) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_scale = float(scale)


class FixedLoraNegation(nn.Module):
    def __init__(self, model: nn.Module, alpha: float):
        super().__init__()
        self.model = model
        self.alpha = float(alpha)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, x, timesteps, y=None, **kwargs):
        set_lora_scale(self.model, -self.alpha)
        return self.model(x, timesteps, y=y, **kwargs)


def main_table_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset="humanml",
        unconstrained=False,
        arch="trans_dec",
        text_encoder_type="bert",
        latent_dim=512,
        layers=8,
        cond_mask_prob=0.1,
        emb_trans_dec=False,
        emb_before_mask=False,
        emb_policy="concat",
        pos_embed_max_len=5000,
        mask_frames=True,
        keyframe_cond_type="",
        pred_len=40,
        context_len=20,
        multi_target_cond=False,
        multi_encoder_type="single",
        target_enc_layers=1,
        noise_schedule="cosine",
        diffusion_steps=10,
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_target_loc=0.0,
        init_model_path=None,
        use_ema=True,
        enable_lora=False,
        enable_sku=False,
    )


def _checkpoint_state(path: Path, use_ema: bool):
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and use_ema and "model_avg" in payload:
        return payload["model_avg"], "model_avg"
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], "model"
    return payload, "raw"


def checkpoint_lora_spec(path: Path, use_ema: bool = True) -> Dict[str, object]:
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and use_ema and "model_avg" in payload:
        state, component = payload["model_avg"], "model_avg"
    elif isinstance(payload, dict) and "model" in payload:
        state, component = payload["model"], "model"
    else:
        state, component = payload, "raw"
    a_keys = sorted(key for key in state if key.endswith(".A"))
    b_keys = {key for key in state if key.endswith(".B")}
    if not a_keys:
        raise RuntimeError("Checkpoint contains no LoRA tensors")
    missing_b = [key[:-2] + ".B" for key in a_keys if key[:-2] + ".B" not in b_keys]
    if missing_b:
        raise RuntimeError("Checkpoint is missing paired LoRA B tensors: {}".format(missing_b[:20]))
    ranks = {int(state[key].shape[0]) for key in a_keys}
    if len(ranks) != 1:
        raise RuntimeError("Mixed LoRA ranks are not supported: {}".format(sorted(ranks)))
    rank = ranks.pop()
    metadata = payload.get("safemo_mmu", {}) if isinstance(payload, dict) else {}
    lora_metadata = metadata.get("lora", {}) if isinstance(metadata, dict) else {}
    modules = [key[:-2] for key in a_keys]
    declared = lora_metadata.get("modules") if isinstance(lora_metadata, dict) else None
    if declared is not None and sorted(declared) != modules:
        raise RuntimeError("Checkpoint LoRA metadata does not match its tensors")
    alpha = float(lora_metadata.get("alpha", rank))
    return {
        "component": component,
        "modules": modules,
        "rank": rank,
        "alpha": alpha,
    }


def load_backbone_checkpoint(
    model: nn.Module, path: Path, component: str = "model_avg"
) -> Dict[str, object]:
    payload = torch.load(str(path), map_location="cpu")
    if component == "raw":
        state = payload
    elif not isinstance(payload, dict) or component not in payload:
        raise RuntimeError("Checkpoint has no '{}' component".format(component))
    else:
        state = payload[component]
    state = dict(state)
    state.pop("sequence_pos_encoder.pe", None)
    state.pop("embed_timestep.sequence_pos_encoder.pe", None)
    state = {
        key: value for key, value in state.items()
        if not key.endswith(".A") and not key.endswith(".B")
    }
    result = model.load_state_dict(state, strict=False)
    missing = [
        key for key in result.missing_keys
        if not key.startswith("clip_model.") and "sequence_pos_encoder.pe" not in key
    ]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "Backbone checkpoint mismatch: missing={}, unexpected={}".format(
                missing[:20], list(result.unexpected_keys)[:20]
            )
        )
    return {"component": component}


def load_main_table_checkpoint(
    model: nn.Module, path: Path, use_ema: bool = True
) -> Dict[str, object]:
    state, component = _checkpoint_state(path, use_ema)
    expected = {
        key for key in model.state_dict() if key.endswith(".A") or key.endswith(".B")
    }
    present = {key for key in state if key.endswith(".A") or key.endswith(".B")}
    if expected - present:
        raise RuntimeError("Checkpoint is missing {} LoRA tensors".format(len(expected - present)))
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("clip_model.")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: missing={}, unexpected={}".format(
                missing[:20], list(result.unexpected_keys)[:20]
            )
        )
    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    if not modules:
        raise RuntimeError("No LoRA modules were injected")
    if any(
        not torch.isfinite(module.A).all() or not torch.isfinite(module.B).all()
        for module in modules
    ):
        raise RuntimeError("LoRA checkpoint contains NaN or Inf")
    b_norm = sum(float(module.B.detach().norm().item()) for module in modules)
    if b_norm == 0.0:
        raise RuntimeError("All LoRA B matrices are zero")
    return {
        "component": component,
        "lora_modules": len(modules),
        "lora_B_norm_sum": b_norm,
    }
