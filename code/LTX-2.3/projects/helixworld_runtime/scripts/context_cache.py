"""Project-local context runtime for the LTX-2.3 AV transformer."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MethodType
from typing import Literal

import torch
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.video_control import (
    VideoControlCondition,
    apply_tiled_projective_matrix,
    build_projective_matrices,
)
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

CachePhase = Literal["read", "commit"]
CacheRole = Literal[
    "video_self",
    "audio_self",
    "video_text",
    "audio_text",
    "audio_to_video",
    "video_to_audio",
    "video_camera",
]
CacheKey = tuple[int, CacheRole]
StaticCacheKey = tuple[int, CacheRole, Hashable]

_TEMPORAL_ROLES = frozenset({"video_self", "audio_self", "audio_to_video", "video_to_audio"})
_TEXT_ROLES = frozenset({"video_text", "audio_text"})


@dataclass(frozen=True)
class KVCachePolicy:
    """Bounded clean-context cache policy."""

    max_history_blocks: int
    sink_blocks: int = 0
    cache_text_context: bool = False

    def __post_init__(self) -> None:
        if self.max_history_blocks <= 0:
            raise ValueError("max_history_blocks must be positive")
        if self.sink_blocks < 0 or self.sink_blocks >= self.max_history_blocks:
            raise ValueError("sink_blocks must satisfy 0 <= sink_blocks < max_history_blocks")


@dataclass
class _TemporalEntry:
    key: Tensor
    value: Tensor
    key_mask: Tensor | None = None
    block_indices: list[int] = field(default_factory=list)
    block_lengths: list[int] = field(default_factory=list)


@dataclass
class _StaticEntry:
    key: Tensor
    value: Tensor
    batch_size: int
    token_count: int
    device: torch.device
    dtype: torch.dtype


class CleanContextKVCache:
    """Per-rollout KV cache containing only committed clean context blocks."""

    def __init__(self, policy: KVCachePolicy) -> None:
        self.policy = policy
        self._temporal: dict[CacheKey, _TemporalEntry] = {}
        self._static: dict[StaticCacheKey, _StaticEntry] = {}
        self._committed_blocks: set[int] = set()

    @property
    def committed_blocks(self) -> tuple[int, ...]:
        return tuple(sorted(self._committed_blocks))

    def temporal(self, key: CacheKey) -> tuple[Tensor, Tensor, Tensor | None] | None:
        entry = self._temporal.get(key)
        return None if entry is None else (entry.key, entry.value, entry.key_mask)

    def snapshot(self) -> "CleanContextKVCache":
        """Freeze the current detached history for checkpoint recomputation."""

        snapshot = CleanContextKVCache(self.policy)
        snapshot._temporal = {
            key: _TemporalEntry(
                key=entry.key,
                value=entry.value,
                key_mask=entry.key_mask,
                block_indices=list(entry.block_indices),
                block_lengths=list(entry.block_lengths),
            )
            for key, entry in self._temporal.items()
        }
        snapshot._static = {
            key: _StaticEntry(
                key=entry.key,
                value=entry.value,
                batch_size=entry.batch_size,
                token_count=entry.token_count,
                device=entry.device,
                dtype=entry.dtype,
            )
            for key, entry in self._static.items()
        }
        snapshot._committed_blocks = set(self._committed_blocks)
        return snapshot

    def static(self, key: StaticCacheKey, context: Tensor) -> tuple[Tensor, Tensor] | None:
        entry = self._static.get(key)
        if entry is None:
            return None
        signature = (context.shape[0], context.shape[1], context.device, context.dtype)
        expected = (entry.batch_size, entry.token_count, entry.device, entry.dtype)
        if signature != expected:
            raise RuntimeError(f"Static KV context changed during one rollout: {signature} != {expected}")
        return entry.key, entry.value

    def set_static(
        self,
        key: StaticCacheKey,
        context: Tensor,
        projected_key: Tensor,
        projected_value: Tensor,
    ) -> None:
        if key in self._static:
            raise RuntimeError(f"Static KV cache was initialized twice for {key}")
        self._static[key] = _StaticEntry(
            key=projected_key.detach().contiguous(),
            value=projected_value.detach().contiguous(),
            batch_size=context.shape[0],
            token_count=context.shape[1],
            device=context.device,
            dtype=context.dtype,
        )

    def append(
        self,
        key: CacheKey,
        block_index: int,
        projected_key: Tensor,
        projected_value: Tensor,
        *,
        key_mask: Tensor | None = None,
    ) -> None:
        if projected_key.shape != projected_value.shape or projected_key.ndim != 3:
            raise ValueError("Committed K/V tensors must have matching [B, T, D] shapes")
        if key_mask is not None and key_mask.shape != projected_key.shape[:2]:
            raise ValueError("Committed camera key mask must have [B, T] shape")
        cached_key = projected_key.detach().contiguous()
        cached_value = projected_value.detach().contiguous()
        cached_mask = None if key_mask is None else key_mask.detach().to(dtype=torch.bool).contiguous()
        entry = self._temporal.get(key)
        if entry is None:
            self._temporal[key] = _TemporalEntry(
                key=cached_key,
                value=cached_value,
                key_mask=cached_mask,
                block_indices=[block_index],
                block_lengths=[cached_key.shape[1]],
            )
            return
        if block_index in entry.block_indices:
            raise RuntimeError(f"Block {block_index} was committed twice for {key}")
        if entry.block_indices and block_index <= entry.block_indices[-1]:
            raise RuntimeError(f"KV blocks must be committed in increasing order for {key}")
        if entry.key.shape[0] != cached_key.shape[0] or entry.key.shape[2] != cached_key.shape[2]:
            raise RuntimeError(f"Committed KV shape changed for {key}")
        if (entry.key_mask is None) != (cached_mask is None):
            raise RuntimeError(f"Committed KV mask presence changed for {key}")
        entry.key = torch.cat((entry.key, cached_key), dim=1)
        entry.value = torch.cat((entry.value, cached_value), dim=1)
        if entry.key_mask is not None and cached_mask is not None:
            entry.key_mask = torch.cat((entry.key_mask, cached_mask), dim=1)
        entry.block_indices.append(block_index)
        entry.block_lengths.append(cached_key.shape[1])
        self._evict_middle_blocks(entry)

    def finish_commit(self, block_index: int) -> None:
        if block_index in self._committed_blocks:
            raise RuntimeError(f"Block {block_index} was globally committed twice")
        expected = 0 if not self._committed_blocks else max(self._committed_blocks) + 1
        if block_index != expected:
            raise RuntimeError(f"Expected block {expected} commit, got {block_index}")
        self._committed_blocks.add(block_index)

    def _evict_middle_blocks(self, entry: _TemporalEntry) -> None:
        while len(entry.block_lengths) > self.policy.max_history_blocks:
            removal_index = self.policy.sink_blocks
            start = sum(entry.block_lengths[:removal_index])
            stop = start + entry.block_lengths[removal_index]
            entry.key = torch.cat((entry.key[:, :start], entry.key[:, stop:]), dim=1)
            entry.value = torch.cat((entry.value[:, :start], entry.value[:, stop:]), dim=1)
            if entry.key_mask is not None:
                entry.key_mask = torch.cat((entry.key_mask[:, :start], entry.key_mask[:, stop:]), dim=1)
            del entry.block_indices[removal_index]
            del entry.block_lengths[removal_index]

    def stats(self) -> dict[str, int]:
        temporal_tokens = sum(entry.key.shape[1] for entry in self._temporal.values())
        static_tokens = sum(entry.key.shape[1] for entry in self._static.values())
        bytes_used = sum(
            entry.key.numel() * entry.key.element_size() + entry.value.numel() * entry.value.element_size()
            for entry in (*self._temporal.values(), *self._static.values())
        )
        bytes_used += sum(
            entry.key_mask.numel() * entry.key_mask.element_size()
            for entry in self._temporal.values()
            if entry.key_mask is not None
        )
        return {
            "committed_blocks": len(self._committed_blocks),
            "temporal_entries": len(self._temporal),
            "temporal_tokens": temporal_tokens,
            "static_entries": len(self._static),
            "static_tokens": static_tokens,
            "bytes": bytes_used,
        }


@dataclass(frozen=True)
class _Runtime:
    cache: CleanContextKVCache
    phase: CachePhase
    block_index: int
    static_context_id: Hashable | None


_ACTIVE_RUNTIME: ContextVar[_Runtime | None] = ContextVar("ltx_runtime_kv_runtime", default=None)


@contextmanager
def _restore_runtime(runtime: _Runtime) -> Iterator[None]:
    token = _ACTIVE_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _ACTIVE_RUNTIME.reset(token)


@contextmanager
def use_kv_cache(
    cache: CleanContextKVCache,
    *,
    phase: CachePhase,
    block_index: int,
    static_context_id: Hashable | None = None,
) -> Iterator[None]:
    """Activate one read or clean-context commit pass."""

    if block_index < 0:
        raise ValueError("block_index must be non-negative")
    if phase == "read" and block_index != len(cache.committed_blocks):
        raise RuntimeError(
            f"Read block {block_index} requires exactly {block_index} committed predecessors, "
            f"got {cache.committed_blocks}"
        )
    if cache.policy.cache_text_context and static_context_id is None:
        raise ValueError("Text KV caching requires a sigma/denoising-step-specific static_context_id")
    token = _ACTIVE_RUNTIME.set(
        _Runtime(
            cache=cache,
            phase=phase,
            block_index=block_index,
            static_context_id=static_context_id,
        )
    )
    try:
        yield
    finally:
        _ACTIVE_RUNTIME.reset(token)
    if phase == "commit":
        cache.finish_commit(block_index)


def _prepend_history_mask(mask: Tensor | None, history_tokens: int) -> Tensor | None:
    if mask is None or history_tokens == 0:
        return mask
    prefix_shape = [*mask.shape]
    prefix_shape[-1] = history_tokens
    prefix = torch.zeros(prefix_shape, device=mask.device, dtype=mask.dtype)
    return torch.cat((prefix, mask), dim=-1)


class RuntimeAttention(Attention):
    """Parameter-identical ``Attention`` with an opt-in runtime KV path."""

    _runtime_layer: int
    _runtime_role: CacheRole

    def _finish_attention(
        self,
        *,
        x: Tensor,
        out: Tensor,
        current_value: Tensor,
        perturbation_mask: Tensor | None,
        camera_out: Tensor | None,
    ) -> Tensor:
        if perturbation_mask is not None:
            if current_value.shape != out.shape:
                raise RuntimeError("Perturbation blending is only valid when query and current-value shapes match")
            out = out * perturbation_mask + current_value * (1 - perturbation_mask)
        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)
        out = self.to_out(out)
        return out if camera_out is None else out + camera_out

    def _forward_text_cached(
        self,
        runtime: _Runtime,
        x: Tensor,
        context: Tensor,
        mask: Tensor | None,
        perturbation_mask: Tensor | None,
        all_perturbed: bool,
    ) -> Tensor:
        if all_perturbed:
            return super().forward(
                x,
                context=context,
                mask=mask,
                perturbation_mask=perturbation_mask,
                all_perturbed=True,
            )
        if runtime.cache.policy.cache_text_context and runtime.static_context_id is None:
            raise RuntimeError("Text KV cache has no sigma-specific static context identity")
        cache_key = (
            self._runtime_layer,
            self._runtime_role,
            runtime.static_context_id,
        )
        cached = runtime.cache.static(cache_key, context) if runtime.cache.policy.cache_text_context else None
        if cached is None:
            projected_key = self.k_norm(self.to_k(context))
            projected_value = self.to_v(context)
            if runtime.cache.policy.cache_text_context:
                runtime.cache.set_static(cache_key, context, projected_key, projected_value)
        else:
            projected_key, projected_value = cached
        query = self.q_norm(self.to_q(x))
        if mask is None:
            out = self.attention_function(query, projected_key, projected_value, self.heads)
        else:
            out = self.masked_attention_function(query, projected_key, projected_value, self.heads, mask)
        return self._finish_attention(
            x=x,
            out=out,
            current_value=projected_value,
            perturbation_mask=perturbation_mask,
            camera_out=None,
        )

    def _forward_camera_cached(
        self,
        *,
        runtime: _Runtime,
        x: Tensor,
        normalized_query: Tensor,
        normalized_key: Tensor,
        current_value: Tensor,
        mask: Tensor | None,
        perturbation_mask: Tensor | None,
        camera_control: VideoControlCondition,
    ) -> Tensor:
        if self.camera_to_out is None:
            raise RuntimeError("Camera cache requires an enabled camera projection")
        projection, projection_inverse = build_projective_matrices(
            camera_control.camera_intrinsics,
            camera_control.camera_w2c,
            camera_control.camera_valid_mask,
        )
        camera_query = apply_tiled_projective_matrix(
            normalized_query,
            projection.transpose(-1, -2),
            self.heads,
        )
        current_key = apply_tiled_projective_matrix(normalized_key, projection_inverse, self.heads)
        current_value = apply_tiled_projective_matrix(current_value, projection_inverse, self.heads)
        current_key_mask = (
            camera_control.camera_key_mask
            if camera_control.camera_key_mask is not None
            else camera_control.camera_valid_mask
        ).to(dtype=torch.bool)

        cache_key: CacheKey = (self._runtime_layer, "video_camera")
        history = runtime.cache.temporal(cache_key)
        if history is None:
            key, value, key_mask = current_key, current_value, current_key_mask
            history_tokens = 0
        else:
            history_key, history_value, history_key_mask = history
            if history_key_mask is None:
                raise RuntimeError("Historical camera KV cache has no key-validity mask")
            history_tokens = history_key.shape[1]
            key = torch.cat((history_key, current_key), dim=1)
            value = torch.cat((history_value, current_value), dim=1)
            key_mask = torch.cat((history_key_mask, current_key_mask), dim=1)

        expanded_mask = _prepend_history_mask(mask, history_tokens)
        key_bias = torch.zeros(
            key_mask.shape[0],
            1,
            1,
            key_mask.shape[1],
            device=key_mask.device,
            dtype=camera_query.dtype,
        )
        key_bias.masked_fill_(
            ~key_mask.view(key_mask.shape[0], 1, 1, -1),
            torch.finfo(camera_query.dtype).min,
        )
        camera_mask = key_bias if expanded_mask is None else expanded_mask + key_bias
        camera_out = self.masked_attention_function(
            camera_query,
            key,
            value,
            self.heads,
            camera_mask,
        )
        camera_out = apply_tiled_projective_matrix(camera_out, projection, self.heads)
        if self.to_gate_logits is not None:
            camera_out = self.gated_attention_function(x, camera_out, self)
        camera_out = self.camera_to_out(camera_out)
        camera_out = camera_out * camera_control.camera_valid_mask.to(camera_out.dtype).unsqueeze(-1)
        if perturbation_mask is not None:
            camera_out = camera_out * perturbation_mask
        if runtime.phase == "commit":
            runtime.cache.append(
                cache_key,
                runtime.block_index,
                current_key,
                current_value,
                key_mask=current_key_mask,
            )
        return camera_out

    def _forward_temporal_cached(
        self,
        runtime: _Runtime,
        x: Tensor,
        context: Tensor,
        mask: Tensor | None,
        pe: Tensor | None,
        k_pe: Tensor | None,
        perturbation_mask: Tensor | None,
        all_perturbed: bool,
        camera_control: VideoControlCondition | None,
    ) -> Tensor:
        if all_perturbed:
            raise RuntimeError("release runtime cached rollout does not support fully perturbed temporal attention")
        current_value = self.to_v(context)
        query = self.to_q(x)
        current_key = self.to_k(context)
        camera_out = None
        if (
            camera_control is not None
            and self._runtime_role == "video_self"
            and camera_control.has_camera
            and self.camera_to_out is not None
        ):
            camera_query, camera_key = self.preattention_function(query, current_key, self, mask, None, None)
            camera_out = self._forward_camera_cached(
                runtime=runtime,
                x=x,
                normalized_query=camera_query,
                normalized_key=camera_key,
                current_value=current_value,
                mask=mask,
                perturbation_mask=perturbation_mask,
                camera_control=camera_control,
            )
        query, current_key = self.preattention_function(query, current_key, self, mask, pe, k_pe)
        cache_key = (self._runtime_layer, self._runtime_role)
        history = runtime.cache.temporal(cache_key)
        if history is None:
            key, value = current_key, current_value
            history_tokens = 0
        else:
            history_key, history_value, history_key_mask = history
            if history_key_mask is not None:
                raise RuntimeError(f"Non-camera KV cache unexpectedly has a key mask for {cache_key}")
            history_tokens = history_key.shape[1]
            key = torch.cat((history_key, current_key), dim=1)
            value = torch.cat((history_value, current_value), dim=1)
        expanded_mask = _prepend_history_mask(mask, history_tokens)
        if expanded_mask is None:
            out = self.attention_function(query, key, value, self.heads)
        else:
            out = self.masked_attention_function(query, key, value, self.heads, expanded_mask)
        result = self._finish_attention(
            x=x,
            out=out,
            current_value=current_value,
            perturbation_mask=perturbation_mask,
            camera_out=camera_out,
        )
        if runtime.phase == "commit":
            runtime.cache.append(cache_key, runtime.block_index, current_key, current_value)
        return result

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        mask: Tensor | None = None,
        pe: Tensor | None = None,
        k_pe: Tensor | None = None,
        perturbation_mask: Tensor | None = None,
        all_perturbed: bool = False,
        camera_control: VideoControlCondition | None = None,
    ) -> Tensor:
        runtime = _ACTIVE_RUNTIME.get()
        role = getattr(self, "_runtime_role", None)
        if runtime is None or role is None:
            return super().forward(
                x,
                context=context,
                mask=mask,
                pe=pe,
                k_pe=k_pe,
                perturbation_mask=perturbation_mask,
                all_perturbed=all_perturbed,
                camera_control=camera_control,
            )
        resolved_context = x if context is None else context
        if role in _TEXT_ROLES:
            if pe is not None or k_pe is not None or camera_control is not None:
                raise RuntimeError("Static text KV cache received unsupported positional or camera inputs")
            return self._forward_text_cached(
                runtime,
                x,
                resolved_context,
                mask,
                perturbation_mask,
                all_perturbed,
            )
        if role not in _TEMPORAL_ROLES:
            raise RuntimeError(f"Unknown release runtime attention role: {role}")
        return self._forward_temporal_cached(
            runtime,
            x,
            resolved_context,
            mask,
            pe,
            k_pe,
            perturbation_mask,
            all_perturbed,
            camera_control,
        )


def install_context_attention(model: nn.Module) -> dict[str, int]:
    """Install the parameter-schema-preserving runtime on one LTX model."""

    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("Expected an LTX model with a non-empty transformer_blocks ModuleList")
    role_attributes: tuple[tuple[str, CacheRole], ...] = (
        ("attn1", "video_self"),
        ("audio_attn1", "audio_self"),
        ("attn2", "video_text"),
        ("audio_attn2", "audio_text"),
        ("audio_to_video_attn", "audio_to_video"),
        ("video_to_audio_attn", "video_to_audio"),
    )
    before_keys = tuple(model.state_dict().keys())
    counts = {role: 0 for _, role in role_attributes}
    for layer_index, block in enumerate(blocks):
        for attribute, role in role_attributes:
            attention = getattr(block, attribute, None)
            if not isinstance(attention, Attention):
                raise TypeError(f"Transformer block {layer_index} has no Attention at {attribute}")
            if not isinstance(attention, RuntimeAttention):
                attention.__class__ = RuntimeAttention
            attention._runtime_layer = layer_index
            attention._runtime_role = role
            counts[role] += 1
    after_keys = tuple(model.state_dict().keys())
    if after_keys != before_keys:
        raise RuntimeError("Installing context attention changed the model state-dict schema")
    return counts


def install_runtime_checkpointing(model: nn.Module) -> int:
    """Checkpoint each block while restoring its immutable context runtime."""

    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("Expected an LTX model with a non-empty transformer_blocks ModuleList")
    installed = 0
    for block in blocks:
        if getattr(block, "_runtime_checkpoint_installed", False):
            installed += 1
            continue
        original_forward = block.forward

        def checkpointed_forward(
            self: nn.Module,
            video: object,
            audio: object,
            *,
            _original_forward: Callable[..., tuple[object, object]] = original_forward,
        ) -> tuple[object, object]:
            runtime = _ACTIVE_RUNTIME.get()
            should_checkpoint = (
                runtime is not None
                and runtime.phase == "read"
                and self.training
                and torch.is_grad_enabled()
            )
            if not should_checkpoint:
                return _original_forward(video=video, audio=audio)

            def run(checkpoint_video: object, checkpoint_audio: object) -> tuple[object, object]:
                with _restore_runtime(runtime):
                    return _original_forward(video=checkpoint_video, audio=checkpoint_audio)

            return checkpoint(
                run,
                video,
                audio,
                use_reentrant=False,
                preserve_rng_state=True,
            )

        block.forward = MethodType(checkpointed_forward, block)
        block._runtime_checkpoint_installed = True
        installed += 1
    return installed


__all__ = [
    "RuntimeAttention",
    "CleanContextKVCache",
    "KVCachePolicy",
    "install_runtime_checkpointing",
    "install_context_attention",
    "use_kv_cache",
]
