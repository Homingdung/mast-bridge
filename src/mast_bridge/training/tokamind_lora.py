"""LoRA utilities for parameter-efficient TokaMind fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize


SCALER_EPS = 1e-6
LORA_TARGETS = "transformer_qkv_out_ffn"


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration shared by all LoRA fine-tuning experiments."""

    rank: int = 8
    alpha: float = 16.0
    targets: str = LORA_TARGETS

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if self.targets != LORA_TARGETS:
            raise ValueError(f"Unsupported LoRA targets: {self.targets!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"method": "lora", **asdict(self)}


class LoRAWeight(nn.Module):
    """Add a trainable low-rank update to one frozen two-dimensional weight."""

    def __init__(self, weight: torch.Tensor, config: LoRAConfig) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"LoRA requires a 2-D weight, got {tuple(weight.shape)}")
        out_features, in_features = weight.shape
        self.scale = float(config.alpha) / float(config.rank)
        self.lora_A = nn.Parameter(
            torch.empty(
                config.rank,
                in_features,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                config.rank,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5.0))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.lora_B @ self.lora_A) * self.scale


class LoRABackbone(nn.Module):
    """Delegate to TokaMind's backbone while exposing only LoRA optimizer params."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        del recurse
        for module in self.modules():
            if isinstance(module, LoRAWeight):
                yield from module.parameters(recurse=False)


def _safe_scale(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), SCALER_EPS)


def rebase_input_linear(
    layer: nn.Linear,
    *,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> None:
    """Preserve a linear projection when changing input standardization."""
    source_mean64 = np.asarray(source_mean, dtype=np.float64)
    target_mean64 = np.asarray(target_mean, dtype=np.float64)
    source_scale = _safe_scale(source_std)
    target_scale = _safe_scale(target_std)
    if source_mean64.shape != target_mean64.shape or source_scale.shape != target_scale.shape:
        raise ValueError("Input scaler shapes do not match")
    if layer.in_features != source_mean64.size:
        raise ValueError(
            f"Input projection expects {layer.in_features} features, "
            f"but scaler contains {source_mean64.size}"
        )
    with torch.no_grad():
        weight = layer.weight.detach().to(dtype=torch.float64)
        offset = torch.as_tensor(
            (target_mean64 - source_mean64) / source_scale,
            device=weight.device,
            dtype=weight.dtype,
        )
        ratio = torch.as_tensor(
            target_scale / source_scale,
            device=weight.device,
            dtype=weight.dtype,
        )
        bias = layer.bias.detach().to(dtype=torch.float64) if layer.bias is not None else None
        if bias is None:
            raise ValueError("Input projection must have a bias for scaler rebasing")
        rebased_bias = bias + weight @ offset
        layer.weight.copy_((weight * ratio.unsqueeze(0)).to(dtype=layer.weight.dtype))
        layer.bias.copy_(rebased_bias.to(dtype=layer.bias.dtype))


def rebase_output_linear(
    layer: nn.Linear,
    *,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> None:
    """Preserve raw predictions when changing output standardization."""
    source_mean64 = np.asarray(source_mean, dtype=np.float64)
    target_mean64 = np.asarray(target_mean, dtype=np.float64)
    source_scale = _safe_scale(source_std)
    target_scale = _safe_scale(target_std)
    if source_mean64.shape != target_mean64.shape or source_scale.shape != target_scale.shape:
        raise ValueError("Output scaler shapes do not match")
    if layer.out_features != source_mean64.size:
        raise ValueError(
            f"Output projection predicts {layer.out_features} values, "
            f"but scaler contains {source_mean64.size}"
        )
    with torch.no_grad():
        ratio = torch.as_tensor(
            source_scale / target_scale,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        offset = torch.as_tensor(
            (source_mean64 - target_mean64) / target_scale,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        if layer.bias is None:
            raise ValueError("Output projection must have a bias for scaler rebasing")
        layer.weight.mul_(ratio.unsqueeze(1))
        layer.bias.mul_(ratio).add_(offset)


def rebase_model_scalers(
    model: nn.Module,
    *,
    source_scalers: dict[str, Any],
    target_scalers: dict[str, Any],
) -> None:
    """Rebase the TokaMind input and output projections to downstream scalers."""
    source_features = [str(value) for value in source_scalers["feature_names"]]
    target_features = [str(value) for value in target_scalers["feature_names"]]
    if source_features != target_features:
        raise ValueError("Pretraining and fine-tuning feature schemas differ")

    input_layers = list(model.tokens.proj_layers.values())
    if len(input_layers) != 1 or not isinstance(input_layers[0], nn.Linear):
        raise ValueError("Expected exactly one TokaMind input projection")
    output_adapters = list(model.output_adapters.values())
    if len(output_adapters) != 1:
        raise ValueError("Expected exactly one TokaMind output adapter")
    output_linears = [
        module
        for module in output_adapters[0].modules()
        if isinstance(module, nn.Linear)
    ]
    if not output_linears:
        raise ValueError("TokaMind output adapter has no linear output layer")

    rebase_input_linear(
        input_layers[0],
        source_mean=source_scalers["input_mean"],
        source_std=source_scalers["input_std"],
        target_mean=target_scalers["input_mean"],
        target_std=target_scalers["input_std"],
    )
    rebase_output_linear(
        output_linears[-1],
        source_mean=source_scalers["output_mean"],
        source_std=source_scalers["output_std"],
        target_mean=target_scalers["output_mean"],
        target_std=target_scalers["output_std"],
    )


def _register_weight(module: nn.Module, name: str, config: LoRAConfig) -> None:
    weight = getattr(module, name)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{type(module).__name__}.{name} is not a tensor")
    parametrize.register_parametrization(
        module,
        name,
        LoRAWeight(weight, config),
    )


def inject_lora_backbone(model: nn.Module, config: LoRAConfig) -> dict[str, int]:
    """Freeze a TokaMind model and add LoRA to attention and FFN weights."""
    if isinstance(model.backbone, LoRABackbone):
        raise ValueError("LoRA has already been injected")
    for parameter in model.parameters():
        parameter.requires_grad = False

    layers = getattr(getattr(model.backbone, "encoder", None), "layers", None)
    if layers is None:
        raise ValueError("TokaMind backbone does not expose encoder.layers")
    target_count = 0
    for layer in layers:
        targets = (
            (layer.self_attn, "in_proj_weight"),
            (layer.self_attn.out_proj, "weight"),
            (layer.linear1, "weight"),
            (layer.linear2, "weight"),
        )
        for module, name in targets:
            _register_weight(module, name, config)
            target_count += 1

    model.backbone = LoRABackbone(model.backbone)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "target_count": target_count,
        "trainable_parameters": trainable,
        "total_parameters_with_lora": total,
    }
