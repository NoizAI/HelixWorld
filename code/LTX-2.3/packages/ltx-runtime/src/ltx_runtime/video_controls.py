"""Resolution-independent camera and action conditioning utilities.

The functions in this module deliberately operate on pixel-space camera data.
They keep the resize/crop geometry explicit, so changing a resolution bucket
does not require changing the model or rewriting camera calibration files.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import torch


class ActionSource(IntEnum):
    """Origin of a resolved per-frame action."""

    INVALID = 0
    EXPLICIT = 1
    CAMERA_DERIVED = 2


@dataclass(frozen=True)
class RobustActionV2Config:
    """Scale-adaptive pseudo-action defaults for camera trajectories."""

    position_smoothing_window: int = 3
    translation_scale_quantile: float = 0.75
    translation_noise_quantile: float = 0.75
    translation_noise_multiplier: float = 2.5
    translation_deadzone_min_scale_fraction: float = 0.05
    translation_deadzone_max_scale_fraction: float = 0.50
    rotation_deadzone_degrees: float = 0.5


DEFAULT_ROBUST_ACTION_V2_CONFIG = RobustActionV2Config()


@dataclass(frozen=True)
class CameraTranslationNormalizationConfig:
    """Robust per-sample scale normalization for camera translation."""

    quantile: float = 0.75
    target_step: float = 0.03
    only_shrink: bool = True
    max_radius: float = 1.5
    epsilon: float = 1e-8


DEFAULT_CAMERA_TRANSLATION_NORMALIZATION_CONFIG = CameraTranslationNormalizationConfig()


@dataclass(frozen=True)
class CameraTranslationNormalizationResult:
    """Normalized relative poses and per-sample scale diagnostics."""

    relative_w2c: torch.Tensor
    scale: torch.Tensor
    step_quantile: torch.Tensor
    max_radius_before: torch.Tensor
    max_radius_after: torch.Tensor


@dataclass(frozen=True)
class DiscreteCameraConfig:
    """Action-to-camera integration constants per latent step."""

    translation_step: float = 0.08
    yaw_step_degrees: float = 3.0
    pitch_step_degrees: float = 3.0
    require_neutral_first: bool = True


DEFAULT_DISCRETE_CAMERA_CONFIG = DiscreteCameraConfig()


@dataclass(frozen=True)
class DiscreteCameraTrajectory:
    """Absolute camera-to-world and world-to-camera trajectories."""

    c2w: torch.Tensor
    w2c: torch.Tensor


@dataclass(frozen=True)
class SpatialTransformMetadata:
    """Exact integer resize and crop applied to a decoded video."""

    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    target_height: int
    target_width: int

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def crop_bottom(self) -> int:
        return self.resized_height - self.target_height - self.crop_top

    @property
    def crop_right(self) -> int:
        return self.resized_width - self.target_width - self.crop_left

    def as_dict(self) -> dict[str, int | float]:
        """Return serialization-friendly metadata without losing integer geometry."""
        return {
            "source_height": self.source_height,
            "source_width": self.source_width,
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "crop_top": self.crop_top,
            "crop_left": self.crop_left,
            "crop_bottom": self.crop_bottom,
            "crop_right": self.crop_right,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "scale_y": self.scale_y,
            "scale_x": self.scale_x,
        }

    @classmethod
    def from_dict(cls, metadata: Mapping[str, Any]) -> SpatialTransformMetadata:
        """Restore the exact transform from serialized video metadata."""
        required = (
            "source_height",
            "source_width",
            "resized_height",
            "resized_width",
            "crop_top",
            "crop_left",
            "target_height",
            "target_width",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"spatial transform metadata is missing keys: {missing}")
        return cls(**{key: int(metadata[key]) for key in required})


@dataclass(frozen=True)
class TransformedIntrinsics:
    """Intrinsics at each stage of the calibration-to-training transform."""

    calibration_intrinsics: torch.Tensor
    decoded_intrinsics: torch.Tensor
    target_intrinsics: torch.Tensor
    normalized_intrinsics: torch.Tensor
    calibration_to_decoded: torch.Tensor
    decoded_to_target: torch.Tensor
    calibration_height: int
    calibration_width: int
    spatial_transform: SpatialTransformMetadata

    def metadata_dict(self) -> dict[str, Any]:
        """Return transform metadata suitable for storing next to latent tensors."""
        return {
            "calibration_height": self.calibration_height,
            "calibration_width": self.calibration_width,
            **self.spatial_transform.as_dict(),
        }


@dataclass(frozen=True)
class ResolvedActions:
    """Per-frame action IDs, validity and provenance."""

    action_ids: torch.Tensor
    action_valid_mask: torch.Tensor
    action_source: torch.Tensor


@dataclass(frozen=True)
class VideoControlBatch:
    """Camera/action controls aligned to the latent-frame rate."""

    normalized_intrinsics: torch.Tensor
    relative_w2c: torch.Tensor
    camera_valid_mask: torch.Tensor
    action_ids: torch.Tensor
    action_valid_mask: torch.Tensor
    action_source: torch.Tensor
    frame_indices: torch.Tensor


@dataclass(frozen=True)
class TokenVideoControls(VideoControlBatch):
    """Camera/action controls expanded to the LTX video-token sequence."""


def compute_spatial_transform_metadata(
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    crop_top: int | None = None,
    crop_left: int | None = None,
) -> SpatialTransformMetadata:
    """Reproduce the official aspect-preserving resize and crop geometry.

    ``int`` conversion intentionally matches ``MediaDataset._resize_and_crop``.
    When no crop offsets are supplied, the crop is centered. Explicit offsets
    allow callers to describe a random crop while retaining exact metadata.
    """
    dimensions = (source_height, source_width, target_height, target_width)
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"source and target dimensions must be positive, got {dimensions}")

    current_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if current_aspect > target_aspect:
        resized_height = target_height
        resized_width = int(source_width * target_height / source_height)
    else:
        resized_height = int(source_height * target_width / source_width)
        resized_width = target_width

    max_top = resized_height - target_height
    max_left = resized_width - target_width
    if (crop_top is None) != (crop_left is None):
        raise ValueError("crop_top and crop_left must either both be supplied or both be omitted")
    if crop_top is None:
        crop_top = max_top // 2
        crop_left = max_left // 2
    if not 0 <= crop_top <= max_top or not 0 <= crop_left <= max_left:
        raise ValueError(
            "crop offsets must fit inside resized media: "
            f"top={crop_top} in [0, {max_top}], left={crop_left} in [0, {max_left}]"
        )

    return SpatialTransformMetadata(
        source_height=source_height,
        source_width=source_width,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=crop_top,
        crop_left=crop_left,
        target_height=target_height,
        target_width=target_width,
    )


def transform_camera_intrinsics(
    intrinsics: torch.Tensor,
    *,
    calibration_size: tuple[int, int],
    spatial_transform: SpatialTransformMetadata,
) -> TransformedIntrinsics:
    """Transform pixel intrinsics through calibration, decode, resize and crop.

    ``calibration_size`` and all size tuples use ``(height, width)`` order. The
    returned normalized matrix divides its first row by the actual target width
    and its second row by the actual target height; no principal-point inference
    or fixed resolution is used.
    """
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must end in [3, 3], got {tuple(intrinsics.shape)}")
    calibration_height, calibration_width = calibration_size
    if calibration_height <= 0 or calibration_width <= 0:
        raise ValueError(f"calibration dimensions must be positive, got {calibration_size}")

    if not intrinsics.is_floating_point():
        intrinsics = intrinsics.to(torch.float32)
    calibration_intrinsics = intrinsics.clone()
    dtype = calibration_intrinsics.dtype
    device = calibration_intrinsics.device

    calibration_to_decoded = torch.eye(3, dtype=dtype, device=device)
    calibration_to_decoded[0, 0] = spatial_transform.source_width / calibration_width
    calibration_to_decoded[1, 1] = spatial_transform.source_height / calibration_height

    decoded_to_target = torch.eye(3, dtype=dtype, device=device)
    decoded_to_target[0, 0] = spatial_transform.scale_x
    decoded_to_target[1, 1] = spatial_transform.scale_y
    decoded_to_target[0, 2] = -spatial_transform.crop_left
    decoded_to_target[1, 2] = -spatial_transform.crop_top

    decoded_intrinsics = calibration_to_decoded @ calibration_intrinsics
    target_intrinsics = decoded_to_target @ decoded_intrinsics
    normalized_intrinsics = target_intrinsics.clone()
    normalized_intrinsics[..., 0, :] /= spatial_transform.target_width
    normalized_intrinsics[..., 1, :] /= spatial_transform.target_height

    return TransformedIntrinsics(
        calibration_intrinsics=calibration_intrinsics,
        decoded_intrinsics=decoded_intrinsics,
        target_intrinsics=target_intrinsics,
        normalized_intrinsics=normalized_intrinsics,
        calibration_to_decoded=calibration_to_decoded,
        decoded_to_target=decoded_to_target,
        calibration_height=calibration_height,
        calibration_width=calibration_width,
        spatial_transform=spatial_transform,
    )


def relative_w2c_to_first(w2c: torch.Tensor) -> torch.Tensor:
    """Express world-to-camera matrices relative to the first frame.

    For absolute matrices ``E_t``, this returns ``E_t @ inverse(E_0)``. Both
    ``[F, 4, 4]`` and ``[B, F, 4, 4]`` inputs are supported.
    """
    if w2c.ndim not in (3, 4) or w2c.shape[-2:] != (4, 4):
        raise ValueError(f"w2c must have shape [F,4,4] or [B,F,4,4], got {tuple(w2c.shape)}")
    if w2c.shape[-3] == 0:
        return w2c.clone()
    first_inverse = torch.linalg.inv(w2c[..., 0, :, :])
    relative_w2c = w2c @ first_inverse.unsqueeze(-3)
    # E_0 @ inverse(E_0) is mathematically the identity. With float32
    # absolute poses whose world translations are large, the matrix product
    # can nevertheless leave translation residuals above the strict identity
    # tolerance used by normalize_relative_w2c_translation. Pinning only the
    # reference frame to its exact analytical value keeps the remaining
    # trajectory unchanged and makes this invariant independent of world
    # coordinate magnitude.
    relative_w2c[..., 0, :, :] = torch.eye(4, dtype=w2c.dtype, device=w2c.device)
    return relative_w2c


def normalize_relative_w2c_translation(
    relative_w2c: torch.Tensor,
    camera_valid_mask: torch.Tensor | None = None,
    *,
    mode: Literal["none", "robust_latent_step"] = "robust_latent_step",
    config: CameraTranslationNormalizationConfig = DEFAULT_CAMERA_TRANSLATION_NORMALIZATION_CONFIG,
) -> CameraTranslationNormalizationResult:
    """Normalize relative translation per sample without changing rotation.

    The robust mode uses valid adjacent camera pairs only. It targets the
    configured latent-step quantile, optionally only shrinks, and also caps
    the valid camera-center radius. Inputs must be relative to frame zero.
    """

    if mode not in ("none", "robust_latent_step"):
        raise ValueError(f"unsupported camera translation normalization mode: {mode}")
    values = (config.quantile, config.target_step, config.max_radius, config.epsilon)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera translation normalization config values must be finite")
    if not 0.0 <= config.quantile <= 1.0:
        raise ValueError("camera translation normalization quantile must be in [0, 1]")
    if config.target_step <= 0.0 or config.max_radius <= 0.0 or config.epsilon <= 0.0:
        raise ValueError("target_step, max_radius and epsilon must be positive")

    relative_w2c = torch.as_tensor(relative_w2c)
    unbatched = relative_w2c.ndim == 3
    if unbatched:
        relative_w2c = relative_w2c.unsqueeze(0)
    if relative_w2c.ndim != 4 or relative_w2c.shape[-2:] != (4, 4):
        raise ValueError(
            "relative_w2c must have shape [F,4,4] or [B,F,4,4], "
            f"got {tuple(relative_w2c.shape)}"
        )
    if relative_w2c.shape[1] == 0:
        raise ValueError("at least one relative camera frame is required")
    if not relative_w2c.is_floating_point():
        raise ValueError("relative_w2c must use a floating-point dtype")
    if not bool(torch.isfinite(relative_w2c).all()):
        raise ValueError("relative_w2c contains non-finite values")

    batch_size, num_frames = relative_w2c.shape[:2]
    if camera_valid_mask is None:
        valid_mask = torch.ones((batch_size, num_frames), dtype=torch.bool, device=relative_w2c.device)
    else:
        valid_mask = torch.as_tensor(camera_valid_mask, dtype=torch.bool, device=relative_w2c.device)
        if unbatched and valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != (batch_size, num_frames):
            raise ValueError(
                f"camera_valid_mask must have shape {(batch_size, num_frames)}, got {tuple(valid_mask.shape)}"
            )

    identity = torch.eye(4, dtype=relative_w2c.dtype, device=relative_w2c.device)
    if not torch.allclose(
        relative_w2c[:, 0], identity.expand(batch_size, -1, -1), rtol=1e-4, atol=1e-5
    ):
        raise ValueError("relative_w2c first frame must be identity")

    statistics_dtype = torch.float64 if relative_w2c.dtype == torch.float64 else torch.float32
    rotation = relative_w2c[..., :3, :3].to(dtype=statistics_dtype)
    translation = relative_w2c[..., :3, 3].to(dtype=statistics_dtype)
    try:
        camera_centers = torch.linalg.solve(rotation, -translation.unsqueeze(-1)).squeeze(-1)
    except RuntimeError as error:
        raise ValueError("relative_w2c rotation blocks must be invertible") from error
    if not bool(torch.isfinite(camera_centers).all()):
        raise ValueError("relative camera centers contain non-finite values")

    radii = torch.linalg.vector_norm(camera_centers, dim=-1)
    if num_frames > 1:
        step_norms = torch.linalg.vector_norm(camera_centers[:, 1:] - camera_centers[:, :-1], dim=-1)
        valid_pairs = valid_mask[:, 1:] & valid_mask[:, :-1]
    else:
        step_norms = torch.empty((batch_size, 0), dtype=statistics_dtype, device=relative_w2c.device)
        valid_pairs = torch.empty((batch_size, 0), dtype=torch.bool, device=relative_w2c.device)

    zero = torch.zeros((), dtype=statistics_dtype, device=relative_w2c.device)
    one = torch.ones((), dtype=statistics_dtype, device=relative_w2c.device)
    step_quantiles: list[torch.Tensor] = []
    max_radii: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        valid_steps = step_norms[batch_index][valid_pairs[batch_index]]
        step_quantile = torch.quantile(valid_steps, config.quantile) if valid_steps.numel() else zero.clone()
        valid_radii = radii[batch_index][valid_mask[batch_index]]
        max_radius_before = valid_radii.max() if valid_radii.numel() else zero.clone()
        scale = one.clone()
        if mode == "robust_latent_step":
            if step_quantile > config.epsilon:
                scale = config.target_step / step_quantile
            if config.only_shrink:
                scale = torch.minimum(scale, one)
            if max_radius_before > config.epsilon:
                scale = torch.minimum(scale, config.max_radius / max_radius_before)
        step_quantiles.append(step_quantile)
        max_radii.append(max_radius_before)
        scales.append(scale)

    scale = torch.stack(scales)
    step_quantile = torch.stack(step_quantiles)
    max_radius_before = torch.stack(max_radii)
    normalized = relative_w2c.clone()
    normalized[..., :3, 3] *= scale.to(dtype=normalized.dtype)[:, None, None]
    normalized[:, 0] = identity
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("normalized relative_w2c contains non-finite values")
    max_radius_after = max_radius_before * scale
    if unbatched:
        normalized = normalized.squeeze(0)
        scale = scale.squeeze(0)
        step_quantile = step_quantile.squeeze(0)
        max_radius_before = max_radius_before.squeeze(0)
        max_radius_after = max_radius_after.squeeze(0)
    return CameraTranslationNormalizationResult(
        relative_w2c=normalized,
        scale=scale,
        step_quantile=step_quantile,
        max_radius_before=max_radius_before,
        max_radius_after=max_radius_after,
    )


def sample_frame_controls(frame_controls: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
    """Sample dense frame controls at explicit decoded-video frame indices.

    Unbatched ``frame_controls`` are paired with ``frame_indices [F]``. Batched
    controls require ``frame_indices [B,F]`` (or ``[1,F]`` for broadcasting).
    This explicit mapping prevents assuming pixel and latent frame counts match.
    """
    indices = torch.as_tensor(frame_indices, dtype=torch.long, device=frame_controls.device)
    if indices.ndim == 1:
        if frame_controls.ndim < 1:
            raise ValueError("frame_controls must have a frame dimension")
        if indices.numel() and (indices.min() < 0 or indices.max() >= frame_controls.shape[0]):
            raise IndexError("frame_indices are outside the available frame_controls range")
        return frame_controls.index_select(0, indices)
    if indices.ndim != 2 or frame_controls.ndim < 2:
        raise ValueError("batched controls require frame_indices with shape [B,F]")
    if indices.shape[0] == 1 and frame_controls.shape[0] != 1:
        indices = indices.expand(frame_controls.shape[0], -1)
    if indices.shape[0] != frame_controls.shape[0]:
        raise ValueError(
            f"batch mismatch between controls ({frame_controls.shape[0]}) and indices ({indices.shape[0]})"
        )
    if indices.numel() and (indices.min() < 0 or indices.max() >= frame_controls.shape[1]):
        raise IndexError("frame_indices are outside the available frame_controls range")
    feature_shape = frame_controls.shape[2:]
    gather_indices = indices.reshape(*indices.shape, *([1] * len(feature_shape))).expand(
        *indices.shape, *feature_shape
    )
    return torch.gather(frame_controls, dim=1, index=gather_indices)


def map_pixel_frames_to_latent_frames(
    pixel_frame_indices: torch.Tensor,
    *,
    latent_num_frames: int,
    temporal_compression: int,
) -> torch.Tensor:
    """Map actual processed pixel-frame indices to causal LTX latent frames.

    LTX's causal layout has a standalone first frame followed by compression
    groups. Latent frame ``i`` therefore samples processed pixel position
    ``i * temporal_compression``. Passing the actual processed frame-index array
    preserves any earlier temporal subsampling.
    """
    if latent_num_frames <= 0 or temporal_compression <= 0:
        raise ValueError("latent_num_frames and temporal_compression must be positive")
    pixel_frame_indices = torch.as_tensor(pixel_frame_indices, dtype=torch.long)
    if pixel_frame_indices.ndim not in (1, 2):
        raise ValueError("pixel_frame_indices must have shape [F] or [B,F]")
    positions = torch.arange(latent_num_frames, device=pixel_frame_indices.device) * temporal_compression
    if positions[-1] >= pixel_frame_indices.shape[-1]:
        raise ValueError(
            f"{pixel_frame_indices.shape[-1]} pixel frames cannot map to {latent_num_frames} latent frames "
            f"with temporal_compression={temporal_compression}"
        )
    if pixel_frame_indices.ndim == 1:
        return pixel_frame_indices.index_select(0, positions)
    return pixel_frame_indices.index_select(1, positions)


def _rotation_matrix_to_xyz_degrees(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SciPy-compatible lowercase ``xyz`` X/Y angles in degrees."""
    sy = torch.sqrt(rotation[..., 0, 0].square() + rotation[..., 1, 0].square())
    singular = sy < 1e-6
    x_regular = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    x_singular = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    x = torch.where(singular, x_singular, x_regular)
    y = torch.atan2(-rotation[..., 2, 0], sy)
    degrees = 180.0 / math.pi
    return x * degrees, y * degrees


def _one_hot_directions_to_label(one_hot: torch.Tensor) -> torch.Tensor:
    """Map a four-bit direction one-hot to a nine-class label."""
    bit_weights = torch.tensor([1, 2, 4, 8], dtype=torch.long, device=one_hot.device)
    bitmask = (one_hot.to(torch.long) * bit_weights).sum(dim=-1)
    lookup = torch.full((16,), -1, dtype=torch.long, device=one_hot.device)
    lookup[torch.tensor([0, 1, 2, 4, 8, 5, 9, 6, 10], device=one_hot.device)] = torch.arange(
        9, device=one_hot.device
    )
    labels = lookup[bitmask]
    if (labels < 0).any():
        raise ValueError("opposing directions cannot be represented by the 9-class mapping")
    return labels


def _camera_rotation_x(theta: torch.Tensor) -> torch.Tensor:
    """Build batched local-X rotation matrices."""
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    result = torch.zeros((*theta.shape, 3, 3), dtype=theta.dtype, device=theta.device)
    result[..., 0, 0] = 1
    result[..., 1, 1] = cosine
    result[..., 1, 2] = -sine
    result[..., 2, 1] = sine
    result[..., 2, 2] = cosine
    return result


def _camera_rotation_y(theta: torch.Tensor) -> torch.Tensor:
    """Build batched local-Y rotation matrices."""
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    result = torch.zeros((*theta.shape, 3, 3), dtype=theta.dtype, device=theta.device)
    result[..., 0, 0] = cosine
    result[..., 0, 2] = sine
    result[..., 1, 1] = 1
    result[..., 2, 0] = -sine
    result[..., 2, 2] = cosine
    return result


def actions_to_discrete_camera(
    action_ids: torch.Tensor | Sequence[int],
    *,
    initial_c2w: torch.Tensor | None = None,
    config: DiscreteCameraConfig = DEFAULT_DISCRETE_CAMERA_CONFIG,
) -> DiscreteCameraTrajectory:
    """Integrate 81-class actions into a camera trajectory.

    The action layout is ``translation_label * 9 + rotation_label``. Both
    nine-class labels use ``none, primary+, primary-, secondary+,
    secondary-, ++, +-, -+, --``. Translation is camera-local ``+Z``
    forward and ``+X`` right; rotation is ``+Y`` yaw-right and ``+X``
    pitch-up. Rotation is applied before local translation.

    ``action_ids[..., 0]`` describes the initial pose and is therefore
    required to be neutral by default. Returned tensors preserve a leading
    batch dimension when the input is batched.
    """

    action_ids = torch.as_tensor(action_ids, dtype=torch.long)
    unbatched = action_ids.ndim == 1
    if unbatched:
        action_ids = action_ids.unsqueeze(0)
    if action_ids.ndim != 2 or action_ids.shape[1] == 0:
        raise ValueError("action_ids must have shape [F] or [B,F] with at least one frame")
    if ((action_ids < 0) | (action_ids > 80)).any():
        raise ValueError("action IDs must be in [0, 80]")
    if config.require_neutral_first and (action_ids[:, 0] != 0).any():
        raise ValueError("the first action must be neutral (ID 0) because it represents the initial pose")
    if min(config.translation_step, config.yaw_step_degrees, config.pitch_step_degrees) < 0:
        raise ValueError("camera step magnitudes must be non-negative")

    batch_size, num_frames = action_ids.shape
    device = action_ids.device
    if initial_c2w is None:
        current = torch.eye(4, dtype=torch.float32, device=device).expand(batch_size, -1, -1).clone()
    else:
        current = torch.as_tensor(initial_c2w, device=device)
        if not current.is_floating_point():
            current = current.to(torch.float32)
        if current.ndim == 2:
            current = current.unsqueeze(0).expand(batch_size, -1, -1).clone()
        elif current.ndim == 3 and current.shape[0] == 1 and batch_size != 1:
            current = current.expand(batch_size, -1, -1).clone()
        else:
            current = current.clone()
        if current.shape != (batch_size, 4, 4):
            raise ValueError(f"initial_c2w must broadcast to {(batch_size, 4, 4)}, got {tuple(current.shape)}")

    # Rows are label IDs 0..8. Translation columns are (right, forward),
    # rotation columns are (yaw-right, pitch-up).
    directions = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
            [1, 1],
            [-1, 1],
            [1, -1],
            [-1, -1],
        ],
        dtype=current.dtype,
        device=device,
    )
    rotations = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ],
        dtype=current.dtype,
        device=device,
    )
    degrees_to_radians = math.pi / 180.0
    poses = [current]
    for frame_index in range(1, num_frames):
        frame_actions = action_ids[:, frame_index]
        translation = directions[torch.div(frame_actions, 9, rounding_mode="floor")]
        rotation = rotations[torch.remainder(frame_actions, 9)]
        yaw = rotation[:, 0] * (config.yaw_step_degrees * degrees_to_radians)
        pitch = rotation[:, 1] * (config.pitch_step_degrees * degrees_to_radians)

        previous = current
        current = previous.clone()
        current_rotation = (
            previous[:, :3, :3] @ _camera_rotation_y(yaw) @ _camera_rotation_x(pitch)
        )
        local_translation = torch.stack(
            (
                translation[:, 0] * config.translation_step,
                torch.zeros(batch_size, dtype=current.dtype, device=device),
                translation[:, 1] * config.translation_step,
            ),
            dim=-1,
        )
        current[:, :3, :3] = current_rotation
        current[:, :3, 3] = previous[:, :3, 3] + (
            current_rotation @ local_translation.unsqueeze(-1)
        ).squeeze(-1)
        poses.append(current)

    c2w = torch.stack(poses, dim=1)
    w2c = torch.linalg.inv(c2w)
    if unbatched:
        c2w = c2w.squeeze(0)
        w2c = w2c.squeeze(0)
    return DiscreteCameraTrajectory(c2w=c2w, w2c=w2c)


def _median_smooth_positions(positions: torch.Tensor, window: int) -> torch.Tensor:
    """Apply a centered component-wise median without changing camera tensors."""
    if window == 1 or positions.shape[1] == 1:
        return positions.clone()
    if window <= 0 or window % 2 == 0:
        raise ValueError("position_smoothing_window must be a positive odd integer")
    radius = window // 2
    padded = torch.cat(
        (
            positions[:, :1].expand(-1, radius, -1),
            positions,
            positions[:, -1:].expand(-1, radius, -1),
        ),
        dim=1,
    )
    return padded.unfold(1, window, 1).median(dim=-1).values


def _eight_sector_labels(primary: torch.Tensor, secondary: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Quantize a 2D direction into disjoint 45-degree labels."""
    angle_degrees = torch.rad2deg(torch.atan2(secondary, primary))
    sector = torch.remainder(torch.floor((angle_degrees + 22.5) / 45.0).to(torch.long), 8)
    lookup = torch.tensor([1, 5, 3, 7, 2, 8, 4, 6], dtype=torch.long, device=primary.device)
    labels = lookup[sector]
    return torch.where(active, labels, torch.zeros_like(labels))


def _derive_actions_from_camera_robust_v2(
    w2c: torch.Tensor,
    camera_valid_mask: torch.Tensor | None,
    config: RobustActionV2Config = DEFAULT_ROBUST_ACTION_V2_CONFIG,
) -> ResolvedActions:
    """Derive scale-adaptive, jitter-resistant actions for each camera pair."""
    unbatched = w2c.ndim == 3
    if unbatched:
        w2c = w2c.unsqueeze(0)
    if w2c.ndim != 4 or w2c.shape[-2:] != (4, 4):
        raise ValueError(f"w2c must have shape [F,4,4] or [B,F,4,4], got {tuple(w2c.shape)}")
    batch_size, num_frames = w2c.shape[:2]
    if num_frames == 0:
        raise ValueError("at least one camera frame is required")

    if camera_valid_mask is None:
        camera_valid_mask = torch.ones((batch_size, num_frames), dtype=torch.bool, device=w2c.device)
    else:
        camera_valid_mask = torch.as_tensor(camera_valid_mask, dtype=torch.bool, device=w2c.device)
        if unbatched and camera_valid_mask.ndim == 1:
            camera_valid_mask = camera_valid_mask.unsqueeze(0)
        if camera_valid_mask.shape != (batch_size, num_frames):
            raise ValueError(
                f"camera_valid_mask must have shape {(batch_size, num_frames)}, got {tuple(camera_valid_mask.shape)}"
            )

    c2w = torch.linalg.inv(w2c)
    raw_delta = torch.eye(4, dtype=w2c.dtype, device=w2c.device).expand(batch_size, num_frames, 4, 4).clone()
    if num_frames > 1:
        raw_delta[:, 1:] = torch.linalg.inv(c2w[:, :-1]) @ c2w[:, 1:]

    smoothed_positions = _median_smooth_positions(c2w[..., :3, 3], config.position_smoothing_window)
    smoothed_translation = torch.zeros((batch_size, num_frames, 3), dtype=w2c.dtype, device=w2c.device)
    deadzone = torch.zeros((batch_size,), dtype=w2c.dtype, device=w2c.device)
    if num_frames > 1:
        world_steps = smoothed_positions[:, 1:] - smoothed_positions[:, :-1]
        previous_world_to_camera_rotation = c2w[:, :-1, :3, :3].transpose(-1, -2)
        smoothed_translation[:, 1:] = (
            previous_world_to_camera_rotation @ world_steps.unsqueeze(-1)
        ).squeeze(-1)
        smoothed_planar = smoothed_translation[:, 1:, (0, 2)]
        raw_planar = raw_delta[:, 1:, (0, 2), 3]
        step_norm = torch.linalg.vector_norm(smoothed_planar, dim=-1)
        noise_norm = torch.linalg.vector_norm(raw_planar - smoothed_planar, dim=-1)
        clip_scale = torch.quantile(step_norm, config.translation_scale_quantile, dim=1)
        estimated_noise = torch.quantile(noise_norm, config.translation_noise_quantile, dim=1)
        minimum = clip_scale * config.translation_deadzone_min_scale_fraction
        maximum = clip_scale * config.translation_deadzone_max_scale_fraction
        deadzone = torch.maximum(minimum, torch.minimum(estimated_noise * config.translation_noise_multiplier, maximum))

    planar_norm = torch.linalg.vector_norm(smoothed_translation[..., (0, 2)], dim=-1)
    translation_labels = _eight_sector_labels(
        smoothed_translation[..., 2],
        smoothed_translation[..., 0],
        planar_norm > deadzone.unsqueeze(1),
    )
    rotation_x, rotation_y = _rotation_matrix_to_xyz_degrees(raw_delta[..., :3, :3])
    rotation_norm = torch.linalg.vector_norm(torch.stack((rotation_y, rotation_x), dim=-1), dim=-1)
    rotation_labels = _eight_sector_labels(
        rotation_y,
        rotation_x,
        rotation_norm > config.rotation_deadzone_degrees,
    )
    action_ids = translation_labels * 9 + rotation_labels
    action_ids[:, 0] = 0

    valid_mask = camera_valid_mask.clone()
    if num_frames > 1:
        valid_mask[:, 1:] &= camera_valid_mask[:, :-1]
    action_ids = torch.where(valid_mask, action_ids, torch.zeros_like(action_ids))
    source = torch.where(
        valid_mask,
        torch.full_like(action_ids, int(ActionSource.CAMERA_DERIVED), dtype=torch.int8),
        torch.full_like(action_ids, int(ActionSource.INVALID), dtype=torch.int8),
    )
    result = ResolvedActions(action_ids=action_ids, action_valid_mask=valid_mask, action_source=source)
    if not unbatched:
        return result
    return ResolvedActions(
        action_ids=result.action_ids.squeeze(0),
        action_valid_mask=result.action_valid_mask.squeeze(0),
        action_source=result.action_source.squeeze(0),
    )


def derive_actions_from_camera(
    w2c: torch.Tensor,
    camera_valid_mask: torch.Tensor | None = None,
    action_algorithm: Literal["action_v1", "robust_v2"] = "action_v1",
) -> ResolvedActions:
    """Derive 9x9 action IDs from adjacent camera poses.

    Translation uses the original ``0.01`` norm, 60/120 degree thresholds;
    rotation uses lowercase-xyz Euler angles and the original ``0.05`` degree
    threshold. The first frame is the valid neutral action (ID 0).
    """
    if action_algorithm == "robust_v2":
        return _derive_actions_from_camera_robust_v2(w2c, camera_valid_mask)
    if action_algorithm != "action_v1":
        raise ValueError(f"unsupported action algorithm: {action_algorithm}")

    unbatched = w2c.ndim == 3
    if unbatched:
        w2c = w2c.unsqueeze(0)
    if w2c.ndim != 4 or w2c.shape[-2:] != (4, 4):
        raise ValueError(f"w2c must have shape [F,4,4] or [B,F,4,4], got {tuple(w2c.shape)}")
    batch_size, num_frames = w2c.shape[:2]
    if num_frames == 0:
        raise ValueError("at least one camera frame is required")

    if camera_valid_mask is None:
        camera_valid_mask = torch.ones((batch_size, num_frames), dtype=torch.bool, device=w2c.device)
    else:
        camera_valid_mask = torch.as_tensor(camera_valid_mask, dtype=torch.bool, device=w2c.device)
        if unbatched and camera_valid_mask.ndim == 1:
            camera_valid_mask = camera_valid_mask.unsqueeze(0)
        if camera_valid_mask.shape != (batch_size, num_frames):
            raise ValueError(
                f"camera_valid_mask must have shape {(batch_size, num_frames)}, got {tuple(camera_valid_mask.shape)}"
            )

    c2w = torch.linalg.inv(w2c)
    relative_c2w = torch.eye(4, dtype=w2c.dtype, device=w2c.device).expand(batch_size, num_frames, 4, 4).clone()
    if num_frames > 1:
        relative_c2w[:, 1:] = torch.linalg.inv(c2w[:, :-1]) @ c2w[:, 1:]

    translation_one_hot = torch.zeros((batch_size, num_frames, 4), dtype=torch.bool, device=w2c.device)
    rotation_one_hot = torch.zeros_like(translation_one_hot)
    if num_frames > 1:
        movement = relative_c2w[:, 1:, :3, 3]
        movement_norm = torch.linalg.vector_norm(movement, dim=-1)
        movement_is_valid = movement_norm > 0.01
        safe_norm = movement_norm.clamp_min(torch.finfo(w2c.dtype).eps).unsqueeze(-1)
        movement_direction = movement / safe_norm
        translation_angles = torch.acos(movement_direction.clamp(-1.0, 1.0)) * (180.0 / math.pi)
        translation_one_hot[:, 1:, 0] = movement_is_valid & (translation_angles[..., 2] < 60.0)
        translation_one_hot[:, 1:, 1] = movement_is_valid & (translation_angles[..., 2] > 120.0)
        translation_one_hot[:, 1:, 2] = movement_is_valid & (translation_angles[..., 0] < 60.0)
        translation_one_hot[:, 1:, 3] = movement_is_valid & (translation_angles[..., 0] > 120.0)

        rotation_x, rotation_y = _rotation_matrix_to_xyz_degrees(relative_c2w[:, 1:, :3, :3])
        rotation_one_hot[:, 1:, 0] = rotation_y > 5e-2
        rotation_one_hot[:, 1:, 1] = rotation_y < -5e-2
        rotation_one_hot[:, 1:, 2] = rotation_x > 5e-2
        rotation_one_hot[:, 1:, 3] = rotation_x < -5e-2

    translation_labels = _one_hot_directions_to_label(translation_one_hot)
    rotation_labels = _one_hot_directions_to_label(rotation_one_hot)
    action_ids = translation_labels * 9 + rotation_labels
    valid_mask = camera_valid_mask.clone()
    if num_frames > 1:
        valid_mask[:, 1:] &= camera_valid_mask[:, :-1]
    action_ids = torch.where(valid_mask, action_ids, torch.zeros_like(action_ids))
    source = torch.where(
        valid_mask,
        torch.full_like(action_ids, int(ActionSource.CAMERA_DERIVED), dtype=torch.int8),
        torch.full_like(action_ids, int(ActionSource.INVALID), dtype=torch.int8),
    )
    result = ResolvedActions(action_ids=action_ids, action_valid_mask=valid_mask, action_source=source)
    if not unbatched:
        return result
    return ResolvedActions(
        action_ids=result.action_ids.squeeze(0),
        action_valid_mask=result.action_valid_mask.squeeze(0),
        action_source=result.action_source.squeeze(0),
    )


def resolve_actions(
    w2c: torch.Tensor,
    *,
    camera_valid_mask: torch.Tensor | None = None,
    explicit_action_ids: torch.Tensor | None = None,
    explicit_action_valid_mask: torch.Tensor | None = None,
    action_algorithm: Literal["action_v1", "robust_v2"] = "action_v1",
) -> ResolvedActions:
    """Resolve actions per frame, preferring valid explicit labels.

    Explicit labels are used only where their valid mask is true. Every missing
    frame independently falls back to a camera-derived action when the adjacent
    camera pair is valid.
    """
    derived = derive_actions_from_camera(w2c, camera_valid_mask, action_algorithm)
    return _overlay_explicit_actions(
        derived,
        explicit_action_ids=explicit_action_ids,
        explicit_action_valid_mask=explicit_action_valid_mask,
    )


def _overlay_explicit_actions(
    derived: ResolvedActions,
    *,
    explicit_action_ids: torch.Tensor | None,
    explicit_action_valid_mask: torch.Tensor | None,
) -> ResolvedActions:
    """Overlay valid explicit labels on already-derived camera actions."""
    if explicit_action_ids is None:
        return derived

    explicit_action_ids = torch.as_tensor(
        explicit_action_ids, dtype=torch.long, device=derived.action_ids.device
    )
    if explicit_action_ids.shape != derived.action_ids.shape:
        raise ValueError(
            f"explicit_action_ids must have shape {tuple(derived.action_ids.shape)}, "
            f"got {tuple(explicit_action_ids.shape)}"
        )
    if explicit_action_valid_mask is None:
        explicit_action_valid_mask = (explicit_action_ids >= 0) & (explicit_action_ids <= 80)
    else:
        explicit_action_valid_mask = torch.as_tensor(
            explicit_action_valid_mask, dtype=torch.bool, device=explicit_action_ids.device
        )
        if explicit_action_valid_mask.shape != explicit_action_ids.shape:
            raise ValueError("explicit_action_valid_mask must match explicit_action_ids")
    invalid_explicit = explicit_action_valid_mask & ((explicit_action_ids < 0) | (explicit_action_ids > 80))
    if invalid_explicit.any():
        raise ValueError("valid explicit action IDs must be in [0, 80]")

    action_ids = torch.where(explicit_action_valid_mask, explicit_action_ids, derived.action_ids)
    valid_mask = explicit_action_valid_mask | derived.action_valid_mask
    source = torch.where(
        explicit_action_valid_mask,
        torch.full_like(derived.action_source, int(ActionSource.EXPLICIT)),
        derived.action_source,
    )
    action_ids = torch.where(valid_mask, action_ids, torch.zeros_like(action_ids))
    return ResolvedActions(action_ids=action_ids, action_valid_mask=valid_mask, action_source=source)


def _as_batched_camera(intrinsics: torch.Tensor, w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if intrinsics.ndim == 3:
        intrinsics = intrinsics.unsqueeze(0)
    if w2c.ndim == 3:
        w2c = w2c.unsqueeze(0)
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [F,3,3] or [B,F,3,3]")
    if w2c.ndim != 4 or w2c.shape[-2:] != (4, 4):
        raise ValueError("w2c must have shape [F,4,4] or [B,F,4,4]")
    if intrinsics.shape[:2] != w2c.shape[:2]:
        raise ValueError("intrinsics and w2c batch/frame dimensions must match")
    return intrinsics, w2c


def _batched_optional_frames(
    values: torch.Tensor | None,
    *,
    batch_size: int,
    source_frames: int,
    sampled_indices: torch.Tensor,
) -> torch.Tensor | None:
    if values is None:
        return None
    values = torch.as_tensor(values, device=sampled_indices.device)
    if values.ndim == 1:
        values = values.unsqueeze(0).expand(batch_size, -1)
    if values.shape[0] == 1 and batch_size != 1:
        values = values.expand(batch_size, *values.shape[1:])
    if values.shape[0] != batch_size:
        raise ValueError(f"expected optional frame data batch {batch_size}, got {values.shape[0]}")
    if values.shape[1] == source_frames:
        return sample_frame_controls(values, sampled_indices)
    if values.shape[1] == sampled_indices.shape[1]:
        return values
    raise ValueError(
        f"optional frame data must contain {source_frames} source or {sampled_indices.shape[1]} sampled frames"
    )


def _batched_frame_indices(
    frame_indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    values = torch.as_tensor(frame_indices, dtype=torch.long, device=device)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[0] not in (1, batch_size):
        raise ValueError(f"{name} must have shape [F] or [B,F]")
    if values.shape[0] == 1 and batch_size != 1:
        values = values.expand(batch_size, -1)
    return values


def _selected_positions_in_full_action_timeline(
    sampled_indices: torch.Tensor,
    full_action_indices: torch.Tensor,
) -> torch.Tensor:
    if full_action_indices.shape[1] == 0:
        raise ValueError("full_action_frame_indices must contain at least one frame")
    if full_action_indices.shape[1] > 1 and bool(
        (full_action_indices[:, 1:] <= full_action_indices[:, :-1]).any()
    ):
        raise ValueError("full_action_frame_indices must be strictly increasing")

    matches = sampled_indices.unsqueeze(-1) == full_action_indices.unsqueeze(1)
    match_count = matches.sum(dim=-1)
    if bool((match_count != 1).any()):
        raise ValueError("frame_indices must be a subset of full_action_frame_indices")
    return matches.to(dtype=torch.long).argmax(dim=-1)


def _full_action_and_sampled_camera_valid_masks(
    camera_valid_mask: torch.Tensor | None,
    *,
    batch_size: int,
    source_frames: int,
    full_action_indices: torch.Tensor,
    sampled_indices: torch.Tensor,
    sampled_action_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve camera validity on the full latent and packed timelines."""
    if camera_valid_mask is None:
        full_action_valid = torch.ones(
            full_action_indices.shape,
            dtype=torch.bool,
            device=sampled_indices.device,
        )
        return full_action_valid, sample_frame_controls(
            full_action_valid, sampled_action_positions
        )

    values = torch.as_tensor(camera_valid_mask, dtype=torch.bool, device=sampled_indices.device)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[0] not in (1, batch_size):
        raise ValueError("camera_valid_mask must have shape [F] or [B,F]")
    if values.shape[0] == 1 and batch_size != 1:
        values = values.expand(batch_size, -1)

    if values.shape[1] == source_frames:
        return (
            sample_frame_controls(values, full_action_indices),
            sample_frame_controls(values, sampled_indices),
        )
    if values.shape[1] == sampled_indices.shape[1]:
        sampled_valid = values
        full_action_valid = torch.ones(
            full_action_indices.shape,
            dtype=torch.bool,
            device=sampled_indices.device,
        )
        full_action_valid.scatter_(1, sampled_action_positions, sampled_valid)
        return full_action_valid, sampled_valid
    if values.shape[1] == full_action_indices.shape[1]:
        return values, sample_frame_controls(values, sampled_action_positions)
    raise ValueError(
        "camera_valid_mask must contain "
        f"{source_frames} source, {full_action_indices.shape[1]} full-action, "
        f"or {sampled_indices.shape[1]} sampled frames"
    )


def _batch_calibration_sizes(
    calibration_size: tuple[int, int] | Sequence[tuple[int, int]] | torch.Tensor,
    batch_size: int,
) -> list[tuple[int, int]]:
    values = calibration_size.tolist() if isinstance(calibration_size, torch.Tensor) else calibration_size
    if len(values) == 2 and all(isinstance(value, int) for value in values):
        return [(int(values[0]), int(values[1]))] * batch_size
    result = [(int(size[0]), int(size[1])) for size in values]  # type: ignore[index]
    if len(result) != batch_size:
        raise ValueError(f"expected {batch_size} calibration sizes, got {len(result)}")
    return result


def prepare_video_controls(
    intrinsics: torch.Tensor,
    w2c: torch.Tensor,
    *,
    spatial_transform: (
        SpatialTransformMetadata
        | Mapping[str, Any]
        | Sequence[SpatialTransformMetadata | Mapping[str, Any]]
    ),
    calibration_size: tuple[int, int] | Sequence[tuple[int, int]] | torch.Tensor,
    frame_indices: torch.Tensor | None = None,
    full_action_frame_indices: torch.Tensor | None = None,
    camera_valid_mask: torch.Tensor | None = None,
    explicit_action_ids: torch.Tensor | None = None,
    explicit_action_valid_mask: torch.Tensor | None = None,
    action_algorithm: Literal["action_v1", "robust_v2"] = "action_v1",
    translation_normalization: Literal["none", "robust_latent_step"] = "none",
    translation_normalization_config: (
        CameraTranslationNormalizationConfig
    ) = DEFAULT_CAMERA_TRANSLATION_NORMALIZATION_CONFIG,
) -> VideoControlBatch:
    """Normalize and align a camera/action sample for LTX training.

    Raw camera arrays may be unbatched or batched; output is always batched.
    ``frame_indices`` contains decoded-video indices for the packed latent
    frames. ``full_action_frame_indices``, when supplied, contains every
    latent endpoint on the original continuous clip. Camera-derived actions are
    computed between adjacent poses on that full latent timeline, then gathered
    by packed-frame membership. If it is omitted, ``frame_indices`` is treated
    as the complete latent timeline for ordinary non-packed samples. Camera
    PRoPE keeps its existing sampled view and is normalized to its first pose.
    """
    intrinsics, w2c = _as_batched_camera(intrinsics, w2c)
    batch_size, source_frames = intrinsics.shape[:2]
    if frame_indices is None:
        sampled_indices = torch.arange(source_frames, device=w2c.device).unsqueeze(0).expand(batch_size, -1)
    else:
        sampled_indices = _batched_frame_indices(
            frame_indices,
            batch_size=batch_size,
            device=w2c.device,
            name="frame_indices",
        )

    if full_action_frame_indices is None:
        full_action_indices = sampled_indices
    else:
        full_action_indices = _batched_frame_indices(
            full_action_frame_indices,
            batch_size=batch_size,
            device=w2c.device,
            name="full_action_frame_indices",
        )
    sampled_action_positions = _selected_positions_in_full_action_timeline(
        sampled_indices,
        full_action_indices,
    )

    sampled_intrinsics = sample_frame_controls(intrinsics, sampled_indices)
    sampled_w2c = sample_frame_controls(w2c, sampled_indices)
    full_action_camera_valid, sampled_camera_valid = (
        _full_action_and_sampled_camera_valid_masks(
            camera_valid_mask,
            batch_size=batch_size,
            source_frames=source_frames,
            full_action_indices=full_action_indices,
            sampled_indices=sampled_indices,
            sampled_action_positions=sampled_action_positions,
        )
    )

    if isinstance(spatial_transform, SpatialTransformMetadata):
        transforms = [spatial_transform] * batch_size
    elif isinstance(spatial_transform, Mapping):
        transforms = [SpatialTransformMetadata.from_dict(spatial_transform)] * batch_size
    else:
        transforms = [
            item if isinstance(item, SpatialTransformMetadata) else SpatialTransformMetadata.from_dict(item)
            for item in spatial_transform
        ]
    if len(transforms) != batch_size:
        raise ValueError(f"expected {batch_size} spatial transforms, got {len(transforms)}")
    calibration_sizes = _batch_calibration_sizes(calibration_size, batch_size)
    normalized_intrinsics = torch.stack(
        [
            transform_camera_intrinsics(
                sampled_intrinsics[index],
                calibration_size=calibration_sizes[index],
                spatial_transform=transforms[index],
            ).normalized_intrinsics
            for index in range(batch_size)
        ]
    )

    identity = torch.eye(4, dtype=sampled_w2c.dtype, device=sampled_w2c.device)
    safe_w2c = torch.where(sampled_camera_valid[..., None, None], sampled_w2c, identity)
    raw_relative_w2c = relative_w2c_to_first(safe_w2c)

    full_action_w2c = sample_frame_controls(w2c, full_action_indices)
    action_identity = torch.eye(4, dtype=w2c.dtype, device=w2c.device)
    safe_full_action_w2c = torch.where(
        full_action_camera_valid[..., None, None],
        full_action_w2c,
        action_identity,
    )
    full_action_relative_w2c = relative_w2c_to_first(safe_full_action_w2c)
    full_actions = derive_actions_from_camera(
        full_action_relative_w2c,
        camera_valid_mask=full_action_camera_valid,
        action_algorithm=action_algorithm,
    )
    sampled_derived_actions = ResolvedActions(
        action_ids=sample_frame_controls(full_actions.action_ids, sampled_action_positions),
        action_valid_mask=sample_frame_controls(
            full_actions.action_valid_mask, sampled_action_positions
        ),
        action_source=sample_frame_controls(
            full_actions.action_source, sampled_action_positions
        ),
    )

    sampled_explicit_actions = _batched_optional_frames(
        explicit_action_ids,
        batch_size=batch_size,
        source_frames=source_frames,
        sampled_indices=sampled_indices,
    )
    sampled_explicit_valid = _batched_optional_frames(
        explicit_action_valid_mask,
        batch_size=batch_size,
        source_frames=source_frames,
        sampled_indices=sampled_indices,
    )
    actions = _overlay_explicit_actions(
        sampled_derived_actions,
        explicit_action_ids=sampled_explicit_actions,
        explicit_action_valid_mask=sampled_explicit_valid,
    )
    # Pseudo-actions describe the raw observed trajectory. Only the pose passed
    # to camera PRoPE is scale-normalized, so normalization cannot alter labels.
    if translation_normalization == "none":
        prope_relative_w2c = raw_relative_w2c
    elif full_action_frame_indices is not None:
        full_normalized_w2c = normalize_relative_w2c_translation(
            full_action_relative_w2c,
            full_action_camera_valid,
            mode=translation_normalization,
            config=translation_normalization_config,
        ).relative_w2c
        sampled_normalized_w2c = sample_frame_controls(
            full_normalized_w2c,
            sampled_action_positions,
        )
        prope_relative_w2c = relative_w2c_to_first(sampled_normalized_w2c)
    else:
        prope_relative_w2c = normalize_relative_w2c_translation(
            raw_relative_w2c,
            sampled_camera_valid,
            mode=translation_normalization,
            config=translation_normalization_config,
        ).relative_w2c
    return VideoControlBatch(
        normalized_intrinsics=normalized_intrinsics,
        relative_w2c=prope_relative_w2c,
        camera_valid_mask=sampled_camera_valid,
        action_ids=actions.action_ids,
        action_valid_mask=actions.action_valid_mask,
        action_source=actions.action_source,
        frame_indices=sampled_indices,
    )


def expand_frame_controls_to_tokens(
    frame_controls: torch.Tensor,
    *,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    prefix_tokens: int = 0,
    neutral_value: int | float | bool | torch.Tensor = 0,
) -> torch.Tensor:
    """Expand ``[B,F,...]`` controls in LTX frame-major video-token order."""
    if frame_controls.ndim < 2 or frame_controls.shape[1] != latent_num_frames:
        raise ValueError(
            f"frame_controls must have shape [B,{latent_num_frames},...], got {tuple(frame_controls.shape)}"
        )
    if min(latent_num_frames, latent_height, latent_width) <= 0 or prefix_tokens < 0:
        raise ValueError("latent dimensions must be positive and prefix_tokens must be non-negative")
    batch_size = frame_controls.shape[0]
    feature_shape = frame_controls.shape[2:]
    tokens_per_frame = latent_height * latent_width
    expanded = frame_controls.unsqueeze(2).expand(batch_size, latent_num_frames, tokens_per_frame, *feature_shape)
    expanded = expanded.reshape(batch_size, latent_num_frames * tokens_per_frame, *feature_shape)
    if prefix_tokens == 0:
        return expanded
    neutral = torch.as_tensor(neutral_value, dtype=frame_controls.dtype, device=frame_controls.device)
    neutral = torch.broadcast_to(neutral, feature_shape)
    prefix = neutral.reshape(*([1, 1]), *feature_shape).expand(batch_size, prefix_tokens, *feature_shape)
    return torch.cat((prefix, expanded), dim=1)


def expand_video_controls_to_tokens(
    controls: VideoControlBatch,
    *,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    prefix_tokens: int = 0,
) -> TokenVideoControls:
    """Expand camera/action controls and prepend masked neutral reference tokens."""
    common = {
        "latent_num_frames": latent_num_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "prefix_tokens": prefix_tokens,
    }
    identity_3 = torch.eye(
        3, dtype=controls.normalized_intrinsics.dtype, device=controls.normalized_intrinsics.device
    )
    identity_4 = torch.eye(4, dtype=controls.relative_w2c.dtype, device=controls.relative_w2c.device)
    return TokenVideoControls(
        normalized_intrinsics=expand_frame_controls_to_tokens(
            controls.normalized_intrinsics, **common, neutral_value=identity_3
        ),
        relative_w2c=expand_frame_controls_to_tokens(controls.relative_w2c, **common, neutral_value=identity_4),
        camera_valid_mask=expand_frame_controls_to_tokens(
            controls.camera_valid_mask, **common, neutral_value=False
        ),
        action_ids=expand_frame_controls_to_tokens(controls.action_ids, **common, neutral_value=0),
        action_valid_mask=expand_frame_controls_to_tokens(
            controls.action_valid_mask, **common, neutral_value=False
        ),
        action_source=expand_frame_controls_to_tokens(
            controls.action_source, **common, neutral_value=int(ActionSource.INVALID)
        ),
        frame_indices=expand_frame_controls_to_tokens(controls.frame_indices, **common, neutral_value=-1),
    )
