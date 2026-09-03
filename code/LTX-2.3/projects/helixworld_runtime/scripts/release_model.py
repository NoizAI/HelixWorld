"""Load the complete HelixWorld inference model from one checkpoint."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

logger = logging.getLogger(__name__)
MODEL_PREFIX = "model.diffusion_model."


def _configure_attention(model: nn.Module, backend: str) -> None:
    if backend == "automatic":
        return
    if backend != "sdpa_flash":
        raise ValueError(f"unsupported attention backend: {backend}")

    from ltx_core.model.transformer.attention import (
        Attention,
        AttentionFunction,
        MaskedAttentionFunction,
    )

    fast_path = AttentionFunction.SDPA_FLASH.to_callable()
    masked_path = MaskedAttentionFunction.SDPA_CUDNN.to_callable()
    count = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.attention_function = fast_path
            module.masked_attention_function = masked_path
            count += 1
    if count == 0:
        raise RuntimeError("no attention modules were found")
    logger.info("Configured %d attention modules", count)


def _meta_tensors(model: nn.Module) -> list[str]:
    values = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    values.extend(
        name for name, buffer in model.named_buffers() if buffer.device.type == "meta"
    )
    return values


def load_release_model(
    *,
    model_path: Path,
    device: torch.device,
    attention_backend: str,
) -> nn.Module:
    """Materialize and return the complete transformer through one checkpoint read."""

    from ltx_core.loader.helpers import create_meta_model
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator

    loader = SafetensorsModelStateDictLoader()
    config = loader.metadata(str(model_path))
    if not config:
        raise RuntimeError(f"model architecture metadata is missing: {model_path}")

    model = create_meta_model(LTXModelConfigurator, config)
    checkpoint = load_file(str(model_path), device="cpu")
    normalized = {
        name.removeprefix(MODEL_PREFIX): tensor
        for name, tensor in checkpoint.items()
        if name.startswith(MODEL_PREFIX)
    }
    del checkpoint

    standard_names = set(model.state_dict())
    standard = {name: value for name, value in normalized.items() if name in standard_names}
    result = model.load_state_dict(standard, strict=False, assign=True)
    del standard
    if result.missing_keys:
        raise RuntimeError(f"model weights are incomplete: {result.missing_keys[:8]}")
    remaining = _meta_tensors(model)
    if remaining:
        raise RuntimeError(f"model has uninitialized tensors: {remaining[:8]}")

    model.enable_video_controls()
    expected = set(model.state_dict())
    complete = {name: value for name, value in normalized.items() if name in expected}
    del normalized
    if set(complete) != expected:
        missing = sorted(expected - set(complete))
        unexpected = sorted(set(complete) - expected)
        raise RuntimeError(
            f"model tensor inventory mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    result = model.load_state_dict(complete, strict=True, assign=True)
    del complete
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"model checkpoint mismatch: {result}")

    _configure_attention(model, attention_backend)
    model.requires_grad_(False)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    gc.collect()
    return model


__all__ = ["load_release_model"]
