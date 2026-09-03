"""Camera/action input and optional output overlay helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.video_control import VideoControlCondition
from ltx_core.types import LatentState
from ltx_runtime.config import InferenceConfig, InferenceSample
from ltx_runtime.inference_runner import (
    CachedPromptEmbeddings,
    CachedSampleMedia,
    InferenceRunner,
)
from ltx_runtime.video_utils import save_video
from torch import Tensor

if TYPE_CHECKING:
    from ltx_core.model.transformer import LTXModel
    from ltx_runtime.progress import SamplingContext

TEMPORAL_COMPRESSION = 8


def load_video_control(
    controls_path: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
) -> tuple[VideoControlCondition, list[int]]:
    payload: dict[str, Tensor] = torch.load(
        controls_path,
        map_location="cpu",
        weights_only=True,
    )
    required = {
        "camera_intrinsics",
        "camera_w2c",
        "camera_valid_mask",
        "action_valid_mask",
        "action_ids",
    }
    if set(payload) != required:
        raise ValueError(
            f"control fields drifted: missing={sorted(required - set(payload))}, "
            f"unexpected={sorted(set(payload) - required)}"
        )

    explicit_action_ids = payload["action_ids"].to(dtype=torch.long).flatten()
    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    if explicit_action_ids.numel() != latent_frames:
        raise ValueError(
            f"control has {explicit_action_ids.numel()} latent actions, "
            f"expected {latent_frames}"
        )
    if bool(((explicit_action_ids < 0) | (explicit_action_ids > 80)).any()):
        raise ValueError("action IDs must be in [0, 80]")
    if explicit_action_ids[0].item() != 0:
        raise ValueError("the first action must be the neutral initial pose (ID 0)")

    tokens_per_frame = (height // 32) * (width // 32)
    expected_tokens = latent_frames * tokens_per_frame
    camera_intrinsics = payload["camera_intrinsics"].to(dtype=torch.bfloat16)
    camera_w2c = payload["camera_w2c"].to(dtype=torch.bfloat16)
    camera_valid_mask = payload["camera_valid_mask"].to(dtype=torch.bool)
    action_valid_mask = payload["action_valid_mask"].to(dtype=torch.bool)
    if camera_intrinsics.shape[:2] != (1, expected_tokens):
        raise ValueError(
            f"camera control has shape {tuple(camera_intrinsics.shape)}, "
            f"expected token count {expected_tokens}"
        )
    action_ids_per_token = explicit_action_ids.view(1, latent_frames).repeat_interleave(
        tokens_per_frame,
        dim=1,
    )
    control = VideoControlCondition(
        camera_intrinsics=camera_intrinsics,
        camera_w2c=camera_w2c,
        camera_valid_mask=camera_valid_mask,
        camera_key_mask=camera_valid_mask,
        action_ids=action_ids_per_token,
        action_valid_mask=action_valid_mask,
    )
    control.validate(batch_size=1, num_tokens=expected_tokens)
    return control, explicit_action_ids.tolist()


def _pixel_frame_to_action(frame_index: int, action_count: int) -> int:
    if frame_index == 0:
        return 0
    return min(
        (frame_index + TEMPORAL_COMPRESSION - 1) // TEMPORAL_COMPRESSION,
        action_count - 1,
    )


def overlay_action_hud(video: Tensor, action_ids: list[int]) -> Tensor:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(
            f"expected decoded video with shape [3,F,H,W], got {tuple(video.shape)}"
        )
    if not action_ids:
        raise ValueError("at least one action is required for the HUD")
    frames = (
        video.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    for frame_index, frame in enumerate(frames):
        action_index = _pixel_frame_to_action(frame_index, len(action_ids))
        action_id = action_ids[action_index]
        height, width = frame.shape[:2]
        scale = max(0.65, min(width / 768.0, height / 512.0))
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (14, height - 65),
            (190, height - 14),
            (4, 7, 11),
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0, dst=frame)
        cv2.putText(
            frame,
            f"CAM ACTION {action_id:02d}",
            (25, height - 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62 * scale,
            (245, 245, 245),
            max(1, round(2 * scale)),
            cv2.LINE_AA,
        )
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float().div(255.0)


def _move_control(
    control: VideoControlCondition,
    device: torch.device,
) -> VideoControlCondition:
    def move(value: Tensor | None) -> Tensor | None:
        return (
            value.to(device=device, non_blocking=True) if value is not None else None
        )

    return VideoControlCondition(
        camera_intrinsics=move(control.camera_intrinsics),
        camera_w2c=move(control.camera_w2c),
        camera_valid_mask=move(control.camera_valid_mask),
        camera_key_mask=move(control.camera_key_mask),
        action_ids=move(control.action_ids),
        action_valid_mask=move(control.action_valid_mask),
    )


class ControlledInferenceRunner(InferenceRunner):
    """Inference runner with one camera/action control sequence."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        model_path: str | Path,
        text_encoder_path: str | Path,
        video_control: VideoControlCondition,
        action_ids: list[int],
        overlay_actions: bool,
        preprocess_device: torch.device,
    ) -> None:
        self._video_control_cpu = video_control
        self._video_control_device: VideoControlCondition | None = None
        self._validated_control_shape: tuple[int, int] | None = None
        self._action_ids = action_ids
        self._overlay_actions = overlay_actions
        self._generated_output_cache: list[tuple[Tensor | None, Tensor | None]] = []
        super().__init__(
            config=config,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            preprocess_device=preprocess_device,
        )

    def bind_video_control(self, device: torch.device) -> None:
        self._video_control_device = _move_control(self._video_control_cpu, device)
        self._validated_control_shape = None

    def unbind_video_control(self) -> None:
        self._video_control_device = None
        self._validated_control_shape = None

    def clear_generated_output_cache(self) -> None:
        self._generated_output_cache.clear()

    def save_clean_copy(
        self,
        *,
        output_dir: Path,
        output_name: str,
    ) -> tuple[Path, bool]:
        if len(self._generated_output_cache) != 1:
            raise RuntimeError(
                "expected exactly one generated output, "
                f"found {len(self._generated_output_cache)}"
            )
        video, audio = self._generated_output_cache.pop()
        if video is None or audio is None:
            raise RuntimeError("inference must return both video and audio")
        if not self._overlay_actions:
            return (output_dir / output_name).resolve(), True
        clean_dir = output_dir / "clean_samples"
        clean_dir.mkdir(exist_ok=True, parents=True)
        clean_path = clean_dir / output_name
        save_video(
            video_tensor=video,
            output_path=clean_path,
            fps=self._config.frame_rate,
            audio=audio,
            audio_sample_rate=self._vocoder.output_sampling_rate,
            video_format="CFHW",
        )
        return clean_path.resolve(), True

    def _modality_from_latent_state(
        self,
        state: LatentState,
        context: Tensor,
        sigma: Tensor,
    ) -> Modality:
        modality = InferenceRunner._modality_from_latent_state(
            state,
            context,
            sigma,
        )
        if modality.positions.shape[1] != 3:
            return modality
        if self._video_control_device is None:
            raise RuntimeError("video control must be bound before generation")
        shape = (modality.latent.shape[0], modality.latent.shape[1])
        if shape != self._validated_control_shape:
            self._video_control_device.validate(
                batch_size=shape[0],
                num_tokens=shape[1],
            )
            self._validated_control_shape = shape
        return replace(modality, video_control=self._video_control_device)

    def _generate_sample(
        self,
        *,
        sample: InferenceSample,
        cached_embeddings: CachedPromptEmbeddings,
        cached_media: CachedSampleMedia,
        transformer: "LTXModel",
        device: torch.device,
        sampling_ctx: "SamplingContext",
    ) -> tuple[Tensor | None, Tensor | None]:
        video, audio = super()._generate_sample(
            sample=sample,
            cached_embeddings=cached_embeddings,
            cached_media=cached_media,
            transformer=transformer,
            device=device,
            sampling_ctx=sampling_ctx,
        )
        self._generated_output_cache.append((video, audio))
        if video is not None and self._overlay_actions:
            video = overlay_action_hud(video, self._action_ids)
        return video, audio


__all__ = [
    "ControlledInferenceRunner",
    "load_video_control",
    "overlay_action_hud",
]
