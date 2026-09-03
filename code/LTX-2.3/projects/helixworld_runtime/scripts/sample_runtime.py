"""Numerical helpers used by the HelixWorld release runtime."""

from __future__ import annotations

import torch
from torch import Tensor

RELEASE_SAMPLER = "release_v1"
SAMPLING_SEED_OFFSET = 1_000_003


def derive_sampling_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("sampling seed must be an integer")
    return (seed + SAMPLING_SEED_OFFSET) % (2**63 - 1)


def _sigma_view(sigma: Tensor, value: Tensor) -> Tensor:
    if sigma.ndim == 0:
        sigma = sigma.expand(value.shape[0])
    if sigma.ndim != 1 or sigma.shape[0] != value.shape[0]:
        raise ValueError(
            f"sigma/value batch mismatch: {tuple(sigma.shape)} / {tuple(value.shape)}"
        )
    return sigma.to(device=value.device, dtype=torch.float32).view(
        -1, *([1] * (value.ndim - 1))
    )


def _mask_view(mask: Tensor, value: Tensor) -> Tensor:
    if mask.shape == value.shape[:-1]:
        mask = mask.unsqueeze(-1)
    if mask.ndim != value.ndim or mask.shape[:-1] != value.shape[:-1]:
        raise ValueError(
            f"mask/value shape mismatch: {tuple(mask.shape)} / {tuple(value.shape)}"
        )
    if mask.shape[-1] not in {1, value.shape[-1]}:
        raise ValueError(
            f"mask must have one or {value.shape[-1]} channels, got {mask.shape[-1]}"
        )
    resolved = mask.to(device=value.device, dtype=torch.float32)
    if not bool(((resolved >= 0.0) & (resolved <= 1.0)).all()):
        raise ValueError("mask values must lie in [0, 1]")
    return resolved


def sample_next_state(
    clean_prediction: Tensor,
    sigma: Tensor,
    *,
    generation_mask: Tensor,
    condition_latent: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    sigma_view = _sigma_view(sigma, clean_prediction)
    mask = _mask_view(generation_mask, clean_prediction)
    if bool((sigma_view == 0).all()):
        random_state = torch.zeros_like(clean_prediction)
    else:
        random_state = torch.randn(
            clean_prediction.shape,
            device=clean_prediction.device,
            dtype=clean_prediction.dtype,
            generator=generator,
        )
    blended = clean_prediction.float() + sigma_view * (
        random_state.float() - clean_prediction.float()
    )
    conditioned = condition_latent.float() + mask * (
        blended - condition_latent.float()
    )
    return conditioned.to(clean_prediction.dtype)


def predict_clean_state(noisy: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    if noisy.shape != velocity.shape:
        raise ValueError(
            f"state/velocity shape mismatch: {tuple(noisy.shape)} / {tuple(velocity.shape)}"
        )
    return (noisy.float() - _sigma_view(sigma, noisy) * velocity.float()).to(
        noisy.dtype
    )


__all__ = [
    "RELEASE_SAMPLER",
    "derive_sampling_seed",
    "predict_clean_state",
    "sample_next_state",
]
