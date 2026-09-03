"""Per-token camera and action conditioning for the LTX video transformer.

The data-side camera normalization intentionally lives outside this module.  In
particular, ``camera_intrinsics`` is expected to already describe the actual
resize/crop applied to the video and to be normalized by that sample's target
width and height.  Keeping the model input resolution-free avoids tying the
control layers to a particular training bucket.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from ltx_core.model.transformer.timestep_embedding import TimestepEmbedding, Timesteps


@dataclass(frozen=True)
class VideoControlCondition:
    """Optional per-token controls for a video modality.

    All local tensors use the video-token sequence length ``N``.  During
    sequence parallelism, ``camera_key_mask`` may retain the padded global
    sequence length while the other tensors are sliced to the local rank.

    Attributes:
        camera_intrinsics: Normalized intrinsics ``[B, N, 3, 3]``.  ``fx`` and
            ``cx`` are divided by the sample's *actual target width*; ``fy``
            and ``cy`` by its *actual target height*.
        camera_w2c: First-frame-relative world-to-camera matrices
            ``[B, N, 4, 4]``.
        camera_valid_mask: Boolean-like query mask ``[B, N]``.  Invalid and
            reference-prefix tokens receive exactly zero PRoPE residual.
        camera_key_mask: Optional boolean-like key mask ``[B, N_keys]``.  It is
            normally the same as ``camera_valid_mask``; sequence-parallel
            callers keep the padded global mask here so valid camera queries
            cannot attend invalid/reference camera keys.
        camera_projection: Optional precomputed PRoPE matrix ``[B, N, 4, 4]``.
            Static block compilation materializes it once before the block loop.
        camera_projection_inverse: Optional precomputed inverse PRoPE matrix
            ``[B, N, 4, 4]``. It must be present together with
            ``camera_projection``.
        action_ids: Discrete action IDs ``[B, N]`` (0..80).
        action_valid_mask: Boolean-like action validity mask ``[B, N]``.
    """

    camera_intrinsics: torch.Tensor | None = None
    camera_w2c: torch.Tensor | None = None
    camera_valid_mask: torch.Tensor | None = None
    camera_key_mask: torch.Tensor | None = None
    camera_projection: torch.Tensor | None = None
    camera_projection_inverse: torch.Tensor | None = None
    action_ids: torch.Tensor | None = None
    action_valid_mask: torch.Tensor | None = None

    @property
    def has_camera(self) -> bool:
        return self.camera_intrinsics is not None

    @property
    def has_action(self) -> bool:
        return self.action_ids is not None

    def validate(self, *, batch_size: int, num_tokens: int) -> None:
        """Validate tensor presence and local batch/token shapes.

        This deliberately does not synchronize devices to check numeric camera
        values.  Data preprocessing owns SE(3), focal-length and ID-range
        validation, while this boundary catches wiring mistakes cheaply.
        """

        camera_fields = (self.camera_intrinsics, self.camera_w2c, self.camera_valid_mask)
        if any(value is not None for value in camera_fields) and not all(value is not None for value in camera_fields):
            raise ValueError(
                "camera_intrinsics, camera_w2c and camera_valid_mask must either all be provided or all be None"
            )
        if self.camera_intrinsics is not None:
            if self.camera_intrinsics.shape != (batch_size, num_tokens, 3, 3):
                raise ValueError(
                    "camera_intrinsics must have shape "
                    f"{(batch_size, num_tokens, 3, 3)}, got {tuple(self.camera_intrinsics.shape)}"
                )
            if self.camera_w2c.shape != (batch_size, num_tokens, 4, 4):
                raise ValueError(
                    f"camera_w2c must have shape {(batch_size, num_tokens, 4, 4)}, got {tuple(self.camera_w2c.shape)}"
                )
            if self.camera_valid_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"camera_valid_mask must have shape {(batch_size, num_tokens)}, "
                    f"got {tuple(self.camera_valid_mask.shape)}"
                )
            if self.camera_key_mask is not None and (
                self.camera_key_mask.ndim != 2 or self.camera_key_mask.shape[0] != batch_size
            ):
                raise ValueError(
                    f"camera_key_mask must have shape [B, N_keys], got {tuple(self.camera_key_mask.shape)}"
                )
        elif self.camera_key_mask is not None:
            raise ValueError("camera_key_mask cannot be provided without camera tensors")

        projection_fields = (self.camera_projection, self.camera_projection_inverse)
        if any(value is not None for value in projection_fields) and not all(
            value is not None for value in projection_fields
        ):
            raise ValueError(
                "camera_projection and camera_projection_inverse must either both be provided or both be None"
            )
        if self.camera_projection is not None:
            if self.camera_intrinsics is None:
                raise ValueError("precomputed camera projection requires camera tensors")
            expected_projection_shape = (batch_size, num_tokens, 4, 4)
            if self.camera_projection.shape != expected_projection_shape:
                raise ValueError(
                    f"camera_projection must have shape {expected_projection_shape}, "
                    f"got {tuple(self.camera_projection.shape)}"
                )
            if self.camera_projection_inverse.shape != expected_projection_shape:
                raise ValueError(
                    f"camera_projection_inverse must have shape {expected_projection_shape}, "
                    f"got {tuple(self.camera_projection_inverse.shape)}"
                )

        action_fields = (self.action_ids, self.action_valid_mask)
        if any(value is not None for value in action_fields) and not all(value is not None for value in action_fields):
            raise ValueError("action_ids and action_valid_mask must either both be provided or both be None")
        if self.action_ids is not None:
            if self.action_ids.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"action_ids must have shape {(batch_size, num_tokens)}, got {tuple(self.action_ids.shape)}"
                )
            if self.action_valid_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"action_valid_mask must have shape {(batch_size, num_tokens)}, "
                    f"got {tuple(self.action_valid_mask.shape)}"
                )

    def split(self, sizes: list[int]) -> list[VideoControlCondition]:
        """Split every control tensor along the batch dimension."""

        split_fields: dict[str, list[torch.Tensor | None]] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            split_fields[field.name] = list(value.split(sizes, dim=0)) if value is not None else [None] * len(sizes)
        return [
            VideoControlCondition(**{name: parts[index] for name, parts in split_fields.items()})
            for index in range(len(sizes))
        ]

    def token_slice(self, start: int, end: int, *, keep_global_key_mask: bool = False) -> VideoControlCondition:
        """Return a local token slice, useful for sequence-parallel wrappers."""

        def slice_tokens(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value[:, start:end]

        return VideoControlCondition(
            camera_intrinsics=slice_tokens(self.camera_intrinsics),
            camera_w2c=slice_tokens(self.camera_w2c),
            camera_valid_mask=slice_tokens(self.camera_valid_mask),
            camera_key_mask=(
                self.camera_key_mask
                if keep_global_key_mask and self.camera_key_mask is not None
                else slice_tokens(self.camera_key_mask)
            ),
            camera_projection=slice_tokens(self.camera_projection),
            camera_projection_inverse=slice_tokens(self.camera_projection_inverse),
            action_ids=slice_tokens(self.action_ids),
            action_valid_mask=slice_tokens(self.action_valid_mask),
        )


class ActionTimestepEmbedder(torch.nn.Module):
    """Discrete action embedder.

    Action IDs are projected with a 256-dimensional cos/sin embedding followed
    by ``Linear -> SiLU -> Linear``.  The final linear is zero-initialized so
    enabling controls preserves the pretrained model exactly at step zero.
    """

    def __init__(self, hidden_dim: int, embedding_dim: int = 256) -> None:
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=embedding_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.mlp = TimestepEmbedding(
            in_channels=embedding_dim,
            time_embed_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        torch.nn.init.zeros_(self.mlp.linear_2.weight)
        torch.nn.init.zeros_(self.mlp.linear_2.bias)

    def forward(
        self,
        action_ids: torch.Tensor,
        action_valid_mask: torch.Tensor,
        *,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        action_proj = self.time_proj(action_ids.flatten())
        action_emb = self.mlp(action_proj.to(dtype=hidden_dtype))
        action_emb = action_emb.view(*action_ids.shape, -1)
        return action_emb * action_valid_mask.to(dtype=action_emb.dtype).unsqueeze(-1)


def build_projective_matrices(
    intrinsics: torch.Tensor,
    w2c: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-token PRoPE projection matrices ``P`` and ``P^-1``.

    Principal points are deliberately removed. Invalid/reference entries are
    replaced by identity transforms before inversion; their final residual is
    additionally query-masked after the learned output projection.
    """

    valid = valid_mask.to(dtype=torch.bool)
    calc_dtype = torch.float32 if intrinsics.dtype in (torch.float16, torch.bfloat16) else intrinsics.dtype
    intrinsics = intrinsics.to(dtype=calc_dtype)
    w2c = w2c.to(dtype=calc_dtype)

    eye3 = torch.eye(3, dtype=calc_dtype, device=intrinsics.device).view(1, 1, 3, 3)
    focal_k = torch.zeros_like(intrinsics)
    focal_k[..., 0, 0] = intrinsics[..., 0, 0]
    focal_k[..., 1, 1] = intrinsics[..., 1, 1]
    focal_k[..., 2, 2] = 1.0
    focal_k = torch.where(valid[..., None, None], focal_k, eye3)

    eye4 = torch.eye(4, dtype=calc_dtype, device=w2c.device).view(1, 1, 4, 4)
    w2c = torch.where(valid[..., None, None], w2c, eye4)

    lifted_k = torch.zeros_like(w2c)
    lifted_k[..., :3, :3] = focal_k
    lifted_k[..., 3, 3] = 1.0

    inv_k = torch.zeros_like(focal_k)
    inv_k[..., 0, 0] = focal_k[..., 0, 0].reciprocal()
    inv_k[..., 1, 1] = focal_k[..., 1, 1].reciprocal()
    inv_k[..., 2, 2] = 1.0
    lifted_inv_k = torch.zeros_like(w2c)
    lifted_inv_k[..., :3, :3] = inv_k
    lifted_inv_k[..., 3, 3] = 1.0

    rotation_inv = w2c[..., :3, :3].transpose(-1, -2)
    w2c_inv = torch.zeros_like(w2c)
    w2c_inv[..., :3, :3] = rotation_inv
    w2c_inv[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inv, w2c[..., :3, 3])
    w2c_inv[..., 3, 3] = 1.0

    projection = lifted_k @ w2c
    projection_inverse = w2c_inv @ lifted_inv_k
    return projection, projection_inverse


def apply_tiled_projective_matrix(features: torch.Tensor, matrix: torch.Tensor, heads: int) -> torch.Tensor:
    """Apply a per-token 4x4 matrix repeatedly over every attention head.

    ``features`` uses LTX attention's flattened layout ``[B, N, H*D]`` and
    ``matrix`` is ``[B, N, 4, 4]``.
    """

    batch, tokens, inner_dim = features.shape
    if inner_dim % heads != 0:
        raise ValueError(f"attention inner dim {inner_dim} is not divisible by heads {heads}")
    head_dim = inner_dim // heads
    if head_dim % 4 != 0:
        raise ValueError(f"PRoPE requires head_dim divisible by 4, got {head_dim}")
    if matrix.shape != (batch, tokens, 4, 4):
        raise ValueError(f"projective matrix must have shape {(batch, tokens, 4, 4)}, got {tuple(matrix.shape)}")

    chunks = features.view(batch, tokens, heads, head_dim // 4, 4)
    transformed = torch.einsum("btij,bthkj->bthki", matrix.to(dtype=features.dtype), chunks)
    return transformed.reshape_as(features)


__all__ = [
    "ActionTimestepEmbedder",
    "VideoControlCondition",
    "apply_tiled_projective_matrix",
    "build_projective_matrices",
]
