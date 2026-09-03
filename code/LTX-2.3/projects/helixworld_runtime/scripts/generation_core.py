"""Four-step ordered AV inference with rebuilt generated-context memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import torch
from context_cache import (
    KVCachePolicy,
    install_context_attention,
    use_kv_cache,
)
from context_runtime import AVModalityBlock, prepare_context
from ltx_core.model.transformer.modality import Modality
from sample_runtime import predict_clean_state, sample_next_state
from sequence_layout import AVBlock, plan_av_blocks, slice_modality, stitch_token_chunks
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelCallTiming:
    block_index: int
    phase: str
    denoising_index: int | None
    sigma: float
    history_blocks_before: tuple[int, ...]
    seconds: float


@dataclass(frozen=True)
class GenerationTiming:
    total_seconds: float
    block_seconds: tuple[float, ...]
    model_calls: tuple[ModelCallTiming, ...]


@dataclass(frozen=True)
class GenerationResult:
    video: Tensor
    audio: Tensor
    blocks: tuple[AVBlock, ...]
    cache_stats: dict[str, int]
    timing: GenerationTiming


@dataclass
class _PendingModelCall:
    block_index: int
    phase: str
    denoising_index: int | None
    sigma: float
    history_blocks_before: tuple[int, ...]
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None
    cpu_seconds: float | None

    def finish(self) -> ModelCallTiming:
        if self.cpu_seconds is not None:
            seconds = self.cpu_seconds
        elif self.start_event is not None and self.end_event is not None:
            seconds = self.start_event.elapsed_time(self.end_event) / 1000.0
        else:
            raise RuntimeError("Incomplete model-call timer")
        return ModelCallTiming(
            block_index=self.block_index,
            phase=self.phase,
            denoising_index=self.denoising_index,
            sigma=self.sigma,
            history_blocks_before=self.history_blocks_before,
            seconds=seconds,
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_transformer_call(
    transformer: nn.Module,
    *,
    video: Modality,
    audio: Modality,
    block_index: int,
    phase: str,
    denoising_index: int | None,
    sigma: float,
    history_blocks_before: tuple[int, ...],
) -> tuple[tuple[Tensor | None, Tensor | None], _PendingModelCall]:
    device = video.latent.device
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        outputs = transformer(video=video, audio=audio, perturbations=None)
        end_event.record(stream)
        pending = _PendingModelCall(
            block_index=block_index,
            phase=phase,
            denoising_index=denoising_index,
            sigma=sigma,
            history_blocks_before=history_blocks_before,
            start_event=start_event,
            end_event=end_event,
            cpu_seconds=None,
        )
    else:
        started = perf_counter()
        outputs = transformer(video=video, audio=audio, perturbations=None)
        pending = _PendingModelCall(
            block_index=block_index,
            phase=phase,
            denoising_index=denoising_index,
            sigma=sigma,
            history_blocks_before=history_blocks_before,
            start_event=None,
            end_event=None,
            cpu_seconds=perf_counter() - started,
        )
    return outputs, pending


@dataclass
class _TimedReplayTransformer:
    transformer: nn.Module
    global_block_index: int
    pending_calls: list[_PendingModelCall]
    replay_index: int = 0

    def __call__(
        self,
        *,
        video: Modality,
        audio: Modality,
        perturbations: None,
    ) -> tuple[Tensor | None, Tensor | None]:
        del perturbations
        outputs, pending_call = _timed_transformer_call(
            self.transformer,
            video=video,
            audio=audio,
            block_index=self.global_block_index,
            phase="rebuild",
            denoising_index=None,
            sigma=0.0,
            history_blocks_before=tuple(range(self.replay_index)),
        )
        self.pending_calls.append(pending_call)
        self.replay_index += 1
        return outputs


def _token_mask(mask: Tensor, latent: Tensor) -> Tensor:
    if mask.shape == latent.shape[:2]:
        return mask
    if mask.shape == (*latent.shape[:2], 1):
        return mask.squeeze(-1)
    raise ValueError(f"Denoise mask shape {tuple(mask.shape)} does not match {tuple(latent.shape)}")


def _masked_clean(denoised: Tensor, clean: Tensor, mask: Tensor) -> Tensor:
    if denoised.shape != clean.shape:
        raise ValueError("Denoised and clean latent shapes differ")
    expanded = mask.unsqueeze(-1) if mask.shape == denoised.shape[:2] else mask
    return (clean.float() + expanded.float() * (denoised.float() - clean.float())).to(denoised.dtype)


def _at_sigma(template: Modality, *, latent: Tensor, sigma: Tensor, denoise_mask: Tensor) -> Modality:
    token_mask = _token_mask(denoise_mask, latent)
    timesteps = token_mask.to(template.timesteps) * sigma.to(template.timesteps).unsqueeze(1)
    return replace(template, latent=latent, sigma=sigma, timesteps=timesteps)


@torch.inference_mode()
def generate_sequence(  # noqa: PLR0913, PLR0915
    transformer: nn.Module,
    *,
    video_template: Modality,
    audio_template: Modality,
    initial_video: Tensor,
    initial_audio: Tensor,
    clean_video: Tensor,
    clean_audio: Tensor,
    video_denoise_mask: Tensor,
    audio_denoise_mask: Tensor,
    sigmas: Tensor,
    generator: torch.Generator,
    latent_frames_per_block: int = 4,
    video_tokens_per_latent_frame: int = 384,
    max_history_blocks: int = 3,
    sink_blocks: int = 1,
) -> GenerationResult:
    """Generate AV trunks using fresh, locally reframed HelixWorld memory.

    ``cache_stats`` reports the final target's rebuilt cache.  Replayed history
    calls are exposed as ``phase='rebuild'`` entries in ``timing.model_calls``.
    """

    if sigmas.ndim != 1 or sigmas.numel() != 5:
        raise ValueError(f"HelixWorld release inference requires five sigma nodes, got {tuple(sigmas.shape)}")
    if (
        not torch.isfinite(sigmas).all()
        or not torch.all(sigmas[:-1] > sigmas[1:])
        or abs(float(sigmas[0]) - 1.0) > 5.0e-7
        or float(sigmas[-1]) != 0.0
    ):
        raise ValueError(f"Invalid HelixWorld release sigma schedule: {sigmas.tolist()}")
    if initial_video.shape != clean_video.shape or initial_audio.shape != clean_audio.shape:
        raise ValueError("Initial and clean latent shapes differ")
    if initial_video.shape != video_template.latent.shape or initial_audio.shape != audio_template.latent.shape:
        raise ValueError("Inference templates do not match latent tensors")
    install_context_attention(transformer)
    blocks = plan_av_blocks(
        video_template.positions,
        audio_template.positions,
        latent_frames_per_block=latent_frames_per_block,
        video_tokens_per_latent_frame=video_tokens_per_latent_frame,
    )
    if not blocks:
        raise RuntimeError("HelixWorld release planner returned no temporal trunks")
    policy = KVCachePolicy(
        max_history_blocks=max_history_blocks,
        sink_blocks=sink_blocks,
        cache_text_context=True,
    )
    video_chunks: list[Tensor] = []
    audio_chunks: list[Tensor] = []
    generated_history: list[AVModalityBlock] = []
    block_seconds: list[float] = []
    model_call_timings: list[ModelCallTiming] = []
    batch_size = initial_video.shape[0]
    device = initial_video.device
    _synchronize(device)
    inference_started = perf_counter()

    for block in blocks:
        _synchronize(device)
        block_started = perf_counter()
        pending_calls: list[_PendingModelCall] = []
        global_video_template = slice_modality(video_template, block.video_start, block.video_end)
        global_audio_template = slice_modality(audio_template, block.audio_start, block.audio_end)
        timed_replay_transformer = _TimedReplayTransformer(
            transformer=transformer,
            global_block_index=block.index,
            pending_calls=pending_calls,
        )

        rebuilt = prepare_context(
            transformer=timed_replay_transformer,
            history=generated_history,
            target=AVModalityBlock(video=global_video_template, audio=global_audio_template),
            policy=policy,
            static_context_id=("rebuild", block.index),
        )
        cache = rebuilt.cache
        block_video_template = rebuilt.target.video
        block_audio_template = rebuilt.target.audio
        local_target_index = rebuilt.local_target_index
        block_clean_video = clean_video[:, block.video_start : block.video_end]
        block_clean_audio = clean_audio[:, block.audio_start : block.audio_end]
        block_video_mask = video_denoise_mask[:, block.video_start : block.video_end]
        block_audio_mask = audio_denoise_mask[:, block.audio_start : block.audio_end]
        video_latent = initial_video[:, block.video_start : block.video_end]
        audio_latent = initial_audio[:, block.audio_start : block.audio_end]

        for denoising_index, sigma_scalar in enumerate(sigmas[:-1]):
            sigma = sigma_scalar.expand(batch_size)
            video = _at_sigma(
                block_video_template,
                latent=video_latent,
                sigma=sigma,
                denoise_mask=block_video_mask,
            )
            audio = _at_sigma(
                block_audio_template,
                latent=audio_latent,
                sigma=sigma,
                denoise_mask=block_audio_mask,
            )
            with use_kv_cache(
                cache,
                phase="read",
                block_index=local_target_index,
                static_context_id=("denoise", denoising_index),
            ):
                outputs, pending_call = _timed_transformer_call(
                    transformer,
                    video=video,
                    audio=audio,
                    block_index=block.index,
                    phase="denoise",
                    denoising_index=denoising_index,
                    sigma=float(sigma_scalar),
                    history_blocks_before=cache.committed_blocks,
                )
            video_velocity, audio_velocity = outputs
            pending_calls.append(pending_call)
            if video_velocity is None or audio_velocity is None:
                raise RuntimeError("HelixWorld release inference requires joint AV outputs")
            generated_video = _masked_clean(
                predict_clean_state(video_latent, video_velocity, sigma),
                block_clean_video,
                block_video_mask,
            )
            generated_audio = _masked_clean(
                predict_clean_state(audio_latent, audio_velocity, sigma),
                block_clean_audio,
                block_audio_mask,
            )
            sigma_next = sigmas[denoising_index + 1].expand(batch_size)
            video_latent = sample_next_state(
                generated_video,
                sigma_next,
                generation_mask=block_video_mask,
                condition_latent=block_clean_video,
                generator=generator,
            )
            audio_latent = sample_next_state(
                generated_audio,
                sigma_next,
                generation_mask=block_audio_mask,
                condition_latent=block_clean_audio,
                generator=generator,
            )

        video_latent = _masked_clean(video_latent, block_clean_video, block_video_mask)
        audio_latent = _masked_clean(audio_latent, block_clean_audio, block_audio_mask)
        video_chunks.append(video_latent)
        audio_chunks.append(audio_latent)
        clean_sigma = torch.zeros(batch_size, device=initial_video.device, dtype=torch.float32)
        history_video = _at_sigma(
            global_video_template,
            latent=video_latent,
            sigma=clean_sigma,
            denoise_mask=block_video_mask,
        )
        history_audio = _at_sigma(
            global_audio_template,
            latent=audio_latent,
            sigma=clean_sigma,
            denoise_mask=block_audio_mask,
        )
        generated_history.append(AVModalityBlock(video=history_video, audio=history_audio))
        _synchronize(device)
        block_seconds.append(perf_counter() - block_started)
        model_call_timings.extend(call.finish() for call in pending_calls)

    _synchronize(device)
    total_seconds = perf_counter() - inference_started

    return GenerationResult(
        video=stitch_token_chunks(
            video_chunks,
            expected_tokens=initial_video.shape[1],
            label="video",
        ),
        audio=stitch_token_chunks(
            audio_chunks,
            expected_tokens=initial_audio.shape[1],
            label="audio",
        ),
        blocks=blocks,
        cache_stats=cache.stats(),
        timing=GenerationTiming(
            total_seconds=total_seconds,
            block_seconds=tuple(block_seconds),
            model_calls=tuple(model_call_timings),
        ),
    )


__all__ = [
    "GenerationResult",
    "GenerationTiming",
    "ModelCallTiming",
    "generate_sequence",
]
