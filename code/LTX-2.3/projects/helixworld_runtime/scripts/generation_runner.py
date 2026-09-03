"""HelixWorld Preview v1 generation runner."""

from __future__ import annotations

from dataclasses import asdict, replace
from statistics import mean, median
from time import perf_counter
from typing import Any

import torch
from control_io import ControlledInferenceRunner
from generation_core import generate_sequence
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.video_control import VideoControlCondition
from ltx_core.types import LatentState
from ltx_runtime.config import InferenceConfig, InferenceSample
from ltx_runtime.inference_runner import CachedPromptEmbeddings, CachedSampleMedia
from ltx_runtime.progress import SamplingContext
from runtime_policy import RuntimePolicy
from sample_runtime import derive_sampling_seed
from torch import Tensor

TEMPORAL_COMPRESSION = 8
LATENT_FRAMES_PER_SEGMENT = 4
TIMING_RECORDS: list[dict[str, Any]] = []


class InteractiveInferenceRunner(ControlledInferenceRunner):
    """Generate one camera/action-conditioned joint audio-video sample."""

    _active_timing_record: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        config: InferenceConfig,
        model_path: str,
        text_encoder_path: str,
        video_control: VideoControlCondition,
        action_ids: list[int],
        overlay_actions: bool,
        release_nodes: tuple[float, ...],
        runtime_policy: RuntimePolicy,
        preprocess_device: torch.device,
    ) -> None:
        super().__init__(
            config=config,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            video_control=video_control,
            action_ids=action_ids,
            overlay_actions=overlay_actions,
            preprocess_device=preprocess_device,
        )
        self._release_nodes = release_nodes
        self._runtime_policy = runtime_policy
        self._state_generator: torch.Generator | None = None

    def _run_denoising(  # noqa: PLR0913
        self,
        *,
        transformer: LTXModel,
        video_state: LatentState,
        audio_state: LatentState,
        video_clean: LatentState,
        audio_clean: LatentState,
        video_context: Tensor,
        audio_context: Tensor,
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[LatentState, LatentState]:
        if self._state_generator is None:
            raise RuntimeError("the release sampling generator is not initialized")
        if self._config.inference_steps != 4:
            raise RuntimeError("the release generation settings have drifted")

        nodes = torch.tensor(
            self._release_nodes,
            device=device,
            dtype=torch.float32,
        )
        initial_node = nodes[0].expand(video_state.latent.shape[0])
        video_template = self._modality_from_latent_state(
            video_state, video_context, initial_node
        )
        audio_template = self._modality_from_latent_state(
            audio_state, audio_context, initial_node
        )
        result = generate_sequence(
            transformer,
            video_template=video_template,
            audio_template=audio_template,
            initial_video=video_state.latent,
            initial_audio=audio_state.latent,
            clean_video=video_clean.clean_latent,
            clean_audio=audio_clean.clean_latent,
            video_denoise_mask=video_state.denoise_mask,
            audio_denoise_mask=audio_state.denoise_mask,
            sigmas=nodes,
            generator=self._state_generator,
            latent_frames_per_block=LATENT_FRAMES_PER_SEGMENT,
            max_history_blocks=self._runtime_policy.history_limit,
            sink_blocks=self._runtime_policy.fixed_prefix,
        )
        expected_segments = (
            video_state.latent.shape[1] // 384 // LATENT_FRAMES_PER_SEGMENT
        )
        if len(result.blocks) != expected_segments:
            raise RuntimeError(
                f"generation produced {len(result.blocks)} segments, "
                f"expected {expected_segments}"
            )
        self._active_timing_record = {
            "generation": asdict(result.timing),
            "context_stats": result.cache_stats,
        }
        TIMING_RECORDS.append(self._active_timing_record)
        for _ in range(4):
            sampling_ctx.advance_step()
        return (
            replace(video_state, latent=result.video),
            replace(audio_state, latent=result.audio),
        )

    def _generate_sample(
        self,
        *,
        sample: InferenceSample,
        cached_embeddings: CachedPromptEmbeddings,
        cached_media: CachedSampleMedia,
        transformer: LTXModel,
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[Tensor | None, Tensor | None]:
        seed = sample.seed
        self._state_generator = torch.Generator(device=device).manual_seed(
            derive_sampling_seed(seed)
        )
        try:
            return super()._generate_sample(
                sample=sample,
                cached_embeddings=cached_embeddings,
                cached_media=cached_media,
                transformer=transformer,
                device=device,
                sampling_ctx=sampling_ctx,
            )
        finally:
            self._state_generator = None

    def _decode_video(self, video_state: LatentState, device: torch.device) -> Tensor:
        if self._active_timing_record is None:
            raise RuntimeError("video decode started without a timing record")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        output = super()._decode_video(video_state, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._active_timing_record["video_decode_seconds"] = (
            perf_counter() - started
        )
        return output

    def _decode_audio(self, audio_state: LatentState, device: torch.device) -> Tensor:
        if self._active_timing_record is None:
            raise RuntimeError("audio decode started without a timing record")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        output = super()._decode_audio(audio_state, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._active_timing_record["audio_decode_seconds"] = (
            perf_counter() - started
        )
        return output


def inference_geometry(num_frames: int) -> dict[str, int | bool]:
    if num_frames <= 0 or (num_frames - 1) % TEMPORAL_COMPRESSION:
        raise ValueError(
            f"num_frames must satisfy (num_frames - 1) % 8 == 0; got {num_frames}"
        )
    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    if latent_frames % LATENT_FRAMES_PER_SEGMENT:
        raise ValueError(
            "latent frames must contain complete segments; "
            f"got latent_frames={latent_frames}, "
            f"segment_size={LATENT_FRAMES_PER_SEGMENT}"
        )
    return {
        "pixel_frames": num_frames,
        "latent_frames": latent_frames,
        "segments": latent_frames // LATENT_FRAMES_PER_SEGMENT,
        "extended_length": num_frames > 121,
    }


def _timing_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": mean(values),
        "median_seconds": median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def build_timing_report(
    records: list[dict[str, Any]],
    *,
    num_frames: int,
    frame_rate: float,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("generation produced no timing records")
    model_calls = [
        call for record in records for call in record["generation"]["model_calls"]
    ]
    segment_times = [
        float(seconds)
        for record in records
        for seconds in record["generation"]["block_seconds"]
    ]
    generation_times = [
        float(record["generation"]["total_seconds"]) for record in records
    ]
    video_times = [float(record["video_decode_seconds"]) for record in records]
    audio_times = [float(record["audio_decode_seconds"]) for record in records]
    combined = [
        generation + video + audio
        for generation, video, audio in zip(
            generation_times, video_times, audio_times, strict=True
        )
    ]
    generated_seconds = num_frames / frame_rate
    return {
        "schema_version": 1,
        "generated_video_seconds_per_sample": generated_seconds,
        "sample_count": len(records),
        "summary": {
            "model_calls": _timing_stats(
                [float(call["seconds"]) for call in model_calls]
            ),
            "segments": _timing_stats(segment_times),
            "generation_per_sample": _timing_stats(generation_times),
            "video_decode_per_sample": _timing_stats(video_times),
            "audio_decode_per_sample": _timing_stats(audio_times),
            "generation_plus_decode_per_sample": _timing_stats(combined),
            "realtime_factor": mean(combined) / generated_seconds,
        },
        "samples": records,
    }


__all__ = [
    "InteractiveInferenceRunner",
    "TIMING_RECORDS",
    "build_timing_report",
    "inference_geometry",
]
