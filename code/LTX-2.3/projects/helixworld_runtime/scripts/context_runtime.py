"""Bounded context handling with local AV temporal coordinates.

The temporal K/V cache stores keys *after* RoPE (and camera keys after the
projective transform).  Evicting an old block and merely relabelling the
remaining cache is therefore incorrect.  This module keeps the source clean or
generated modality blocks and, before every target, selects the bounded
history, maps it onto a compact local timeline, and commits it into a fresh
cache.  The transformer consequently recomputes both RoPE K/V and camera
projective K/V at the positions used by the target read.

No module wrapping or parameter replacement is performed here, so this
contract does not affect model state-dict names or FSDP wrapping boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from context_cache import CleanContextKVCache, KVCachePolicy, use_kv_cache
from ltx_core.model.transformer.modality import Modality
from torch import Tensor


@dataclass(frozen=True)
class AVModalityBlock:
    """One time-aligned video/audio block saved in its original coordinates."""

    video: Modality
    audio: Modality


AVModalityMapper = Callable[[AVModalityBlock], AVModalityBlock]


class AVTransformer(Protocol):
    """Minimal LTX transformer call surface needed while rebuilding memory."""

    def __call__(
        self,
        *,
        video: Modality,
        audio: Modality,
        perturbations: None,
    ) -> object: ...


@dataclass(frozen=True)
class PreparedContext:
    """Fresh local cache and the target that must read it at ``local_target_index``."""

    cache: CleanContextKVCache
    target: AVModalityBlock
    selected_history_indices: tuple[int, ...]
    local_history: tuple[AVModalityBlock, ...]
    local_target_index: int


def select_history_indices(
    history_length: int,
    *,
    max_history_blocks: int,
    sink_blocks: int,
) -> tuple[int, ...]:
    """Select a deterministic sink-prefix plus most-recent history window.

    When the history already fits, all blocks are retained.  Once it exceeds
    the budget, the first ``sink_blocks`` and latest
    ``max_history_blocks - sink_blocks`` blocks are retained, in source order.
    """

    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    if max_history_blocks <= 0:
        raise ValueError("max_history_blocks must be positive")
    if sink_blocks < 0 or sink_blocks >= max_history_blocks:
        raise ValueError("sink_blocks must satisfy 0 <= sink_blocks < max_history_blocks")
    if history_length <= max_history_blocks:
        return tuple(range(history_length))
    recent_blocks = max_history_blocks - sink_blocks
    return (*range(sink_blocks), *range(history_length - recent_blocks, history_length))


def _validate_positions(positions: Tensor, *, axes: int, label: str) -> None:
    if positions.ndim != 4 or positions.shape[1] != axes or positions.shape[-1] != 2:
        raise ValueError(f"{label} positions must have shape [B, {axes}, T, 2], got {tuple(positions.shape)}")
    if positions.shape[0] <= 0 or positions.shape[2] <= 0:
        raise ValueError(f"{label} positions cannot be empty")
    if not torch.isfinite(positions).all().item():
        raise ValueError(f"{label} positions contain non-finite values")
    if not torch.all(positions[..., 1] >= positions[..., 0]).item():
        raise ValueError(f"{label} position bounds are reversed")


def _video_temporal_bounds(video: Modality) -> tuple[Tensor, Tensor]:
    _validate_positions(video.positions, axes=3, label="video")
    temporal = video.positions[:, 0]
    return temporal[..., 0].amin(dim=1), temporal[..., 1].amax(dim=1)


def _resolve_local_start(source_start: Tensor, local_start: float | Tensor) -> Tensor:
    resolved = torch.as_tensor(local_start, device=source_start.device, dtype=source_start.dtype)
    if resolved.ndim == 0:
        return resolved.expand_as(source_start)
    if resolved.shape != source_start.shape:
        raise ValueError(
            f"local_start must be scalar or have one value per batch item, got {tuple(resolved.shape)}"
        )
    return resolved


def _shift_positions(positions: Tensor, shift: Tensor) -> Tensor:
    shifted = positions.clone()
    shifted[:, 0] = shifted[:, 0] + shift[:, None, None]
    return shifted


def reframe_av_block(
    block: AVModalityBlock,
    *,
    local_start: float | Tensor,
) -> AVModalityBlock:
    """Move an AV block onto a local timeline using one shared time shift.

    Only temporal position axis 0 is changed.  Video height/width coordinates,
    every latent/context tensor, and the complete ``VideoControlCondition``
    (camera, precomputed projective matrices, and action IDs) are retained
    without modification.  Inputs are never changed in place.
    """

    _validate_positions(block.audio.positions, axes=1, label="audio")
    if block.video.positions.shape[0] != block.audio.positions.shape[0]:
        raise ValueError("Video/audio position batches differ")
    source_start, _ = _video_temporal_bounds(block.video)
    resolved_start = _resolve_local_start(source_start, local_start)
    shared_shift = resolved_start - source_start
    return AVModalityBlock(
        video=replace(block.video, positions=_shift_positions(block.video.positions, shared_shift)),
        audio=replace(block.audio, positions=_shift_positions(block.audio.positions, shared_shift)),
    )


def _map_block(block: AVModalityBlock, mapper: AVModalityMapper | None) -> AVModalityBlock:
    mapped = block if mapper is None else mapper(block)
    if not isinstance(mapped, AVModalityBlock):
        raise TypeError(f"modality_mapper must return AVModalityBlock, got {type(mapped)}")
    return mapped


def prepare_context(
    *,
    transformer: AVTransformer,
    history: Sequence[AVModalityBlock],
    target: AVModalityBlock,
    policy: KVCachePolicy,
    modality_mapper: AVModalityMapper | None = None,
    static_context_id: Hashable | None = None,
    commit_no_grad: bool = True,
) -> PreparedContext:
    """Rebuild one target's bounded clean/generated context from source blocks.

    ``modality_mapper`` is applied independently to every selected history
    block and to the target before reframing.  It is the intended hook for
    positive/negative text contexts: call this function once with the positive
    mapper and once with the negative mapper to obtain independent caches.

    The returned cache is always a new object.  Selected history blocks are
    placed consecutively from local time zero and committed with local indices
    ``0..M-1``.  The returned target follows the final history block and must be
    read with ``block_index=M``.
    """

    selected_indices = select_history_indices(
        len(history),
        max_history_blocks=policy.max_history_blocks,
        sink_blocks=policy.sink_blocks,
    )
    cache = CleanContextKVCache(policy)
    local_history: list[AVModalityBlock] = []
    local_cursor: Tensor | None = None

    for local_index, source_index in enumerate(selected_indices):
        mapped = _map_block(history[source_index], modality_mapper)
        source_start, source_end = _video_temporal_bounds(mapped.video)
        duration = source_end - source_start
        if not torch.all(duration > 0).item():
            raise ValueError(f"History block {source_index} has a non-positive video duration")
        if local_cursor is None:
            local_cursor = torch.zeros_like(source_start)
        localized = reframe_av_block(mapped, local_start=local_cursor)
        local_history.append(localized)
        grad_context = torch.no_grad() if commit_no_grad else nullcontext()
        with grad_context, use_kv_cache(
            cache,
            phase="commit",
            block_index=local_index,
            static_context_id=static_context_id,
        ):
            transformer(video=localized.video, audio=localized.audio, perturbations=None)
        local_cursor = local_cursor + duration

    mapped_target = _map_block(target, modality_mapper)
    target_start, _ = _video_temporal_bounds(mapped_target.video)
    if local_cursor is None:
        local_cursor = torch.zeros_like(target_start)
    local_target = reframe_av_block(mapped_target, local_start=local_cursor)
    return PreparedContext(
        cache=cache,
        target=local_target,
        selected_history_indices=selected_indices,
        local_history=tuple(local_history),
        local_target_index=len(local_history),
    )


__all__ = [
    "AVModalityBlock",
    "AVModalityMapper",
    "AVTransformer",
    "PreparedContext",
    "prepare_context",
    "reframe_av_block",
    "select_history_indices",
]
