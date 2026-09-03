"""Time-aligned configurable-latent block planning for joint LTX video and audio."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import LatentState
from torch import Tensor


@dataclass(frozen=True)
class AVBlock:
    index: int
    video_start: int
    video_end: int
    audio_start: int
    audio_end: int
    time_start_seconds: float
    time_end_seconds: float

    @property
    def video_tokens(self) -> int:
        return self.video_end - self.video_start

    @property
    def audio_tokens(self) -> int:
        return self.audio_end - self.audio_start


def _validate_positions(positions: Tensor, *, axes: int, label: str) -> None:
    if positions.ndim != 4 or positions.shape[1] != axes or positions.shape[-1] != 2:
        raise ValueError(f"{label} positions must have shape [B, {axes}, T, 2], got {tuple(positions.shape)}")
    if positions.shape[0] <= 0 or positions.shape[2] <= 0:
        raise ValueError(f"{label} positions cannot be empty")
    if not torch.isfinite(positions).all().item():
        raise ValueError(f"{label} positions contain non-finite values")
    if not torch.all(positions[..., 1] >= positions[..., 0]).item():
        raise ValueError(f"{label} position bounds are reversed")


def plan_av_blocks(  # noqa: PLR0912, PLR0915
    video_positions: Tensor,
    audio_positions: Tensor,
    *,
    latent_frames_per_block: int = 4,
    video_tokens_per_latent_frame: int = 384,
) -> tuple[AVBlock, ...]:
    """Partition video into fixed latent blocks and audio by matching timestamps."""

    _validate_positions(video_positions, axes=3, label="video")
    _validate_positions(audio_positions, axes=1, label="audio")
    if video_positions.shape[0] != audio_positions.shape[0]:
        raise ValueError("Video/audio position batches differ")
    if latent_frames_per_block <= 0 or video_tokens_per_latent_frame <= 0:
        raise ValueError("Block and per-frame token counts must be positive")
    video_tokens = video_positions.shape[2]
    if video_tokens % video_tokens_per_latent_frame:
        raise ValueError(f"Video token count {video_tokens} is not divisible by {video_tokens_per_latent_frame}")
    latent_frames = video_tokens // video_tokens_per_latent_frame
    if latent_frames % latent_frames_per_block:
        raise ValueError(
            f"Video latent frames {latent_frames} are not divisible by block size {latent_frames_per_block}"
        )

    temporal = video_positions[:, 0]
    reference_temporal = temporal[0]
    if not torch.allclose(temporal, reference_temporal.unsqueeze(0).expand_as(temporal)):
        raise ValueError("All samples in a batch must share the same video temporal grid")
    frame_starts = reference_temporal[::video_tokens_per_latent_frame, 0]
    frame_ends = reference_temporal[::video_tokens_per_latent_frame, 1]
    expanded_starts = frame_starts.repeat_interleave(video_tokens_per_latent_frame)
    expanded_ends = frame_ends.repeat_interleave(video_tokens_per_latent_frame)
    if not torch.equal(reference_temporal[:, 0], expanded_starts) or not torch.equal(
        reference_temporal[:, 1], expanded_ends
    ):
        raise ValueError("Video tokens are not contiguous frame-major groups")
    if not torch.all(frame_starts[1:] >= frame_ends[:-1]).item():
        raise ValueError("Video latent-frame time bounds overlap or regress")

    audio_temporal = audio_positions[:, 0]
    reference_audio = audio_temporal[0]
    if not torch.allclose(audio_temporal, reference_audio.unsqueeze(0).expand_as(audio_temporal)):
        raise ValueError("All samples in a batch must share the same audio temporal grid")
    audio_midpoints = reference_audio.mean(dim=-1)
    if not torch.all(audio_midpoints[1:] >= audio_midpoints[:-1]).item():
        raise ValueError("Audio timestamps are not monotonic")

    num_blocks = latent_frames // latent_frames_per_block
    boundary_frame_indices = torch.arange(
        latent_frames_per_block - 1,
        latent_frames,
        latent_frames_per_block,
        device=frame_ends.device,
    )
    block_ends = frame_ends[boundary_frame_indices]
    audio_assignments = torch.bucketize(audio_midpoints, block_ends, right=False)
    if (audio_assignments >= num_blocks).any().item():
        raise ValueError(
            f"Audio timeline extends past video timeline: audio={float(audio_midpoints[-1])}, "
            f"video={float(block_ends[-1])}"
        )

    blocks: list[AVBlock] = []
    audio_cursor = 0
    for block_index in range(num_blocks):
        video_frame_start = block_index * latent_frames_per_block
        video_frame_end = video_frame_start + latent_frames_per_block
        selected_audio = torch.nonzero(audio_assignments == block_index, as_tuple=False).flatten()
        if selected_audio.numel() == 0:
            raise ValueError(f"AV block {block_index} has no aligned audio tokens")
        audio_start = int(selected_audio[0].item())
        audio_end = int(selected_audio[-1].item()) + 1
        if audio_start != audio_cursor or selected_audio.numel() != audio_end - audio_start:
            raise ValueError("Audio block assignment is not contiguous")
        blocks.append(
            AVBlock(
                index=block_index,
                video_start=video_frame_start * video_tokens_per_latent_frame,
                video_end=video_frame_end * video_tokens_per_latent_frame,
                audio_start=audio_start,
                audio_end=audio_end,
                time_start_seconds=float(frame_starts[video_frame_start].item()),
                time_end_seconds=float(frame_ends[video_frame_end - 1].item()),
            )
        )
        audio_cursor = audio_end
    if audio_cursor != audio_positions.shape[2]:
        raise ValueError(f"AV blocks consumed {audio_cursor}/{audio_positions.shape[2]} audio tokens")
    return tuple(blocks)


def _slice_attention_mask(mask: Tensor | None, start: int, end: int, total_tokens: int) -> Tensor | None:
    if mask is None:
        return None
    if mask.shape[-1] != total_tokens:
        raise ValueError(f"Attention key dimension {mask.shape[-1]} != token count {total_tokens}")
    sliced = mask[..., start:end]
    if mask.ndim >= 2 and mask.shape[-2] == total_tokens:
        sliced = sliced[..., start:end, :]
    elif mask.ndim >= 2 and mask.shape[-2] not in (1,):
        raise ValueError(f"Unsupported attention query dimension: {tuple(mask.shape)}")
    return sliced


def slice_modality(modality: Modality, start: int, end: int) -> Modality:
    """Slice one contiguous token block while retaining global text context."""

    total_tokens = modality.latent.shape[1]
    if not 0 <= start < end <= total_tokens:
        raise ValueError(f"Invalid modality token slice [{start}, {end})/{total_tokens}")
    video_control = modality.video_control.token_slice(start, end) if modality.video_control is not None else None
    return replace(
        modality,
        latent=modality.latent[:, start:end],
        timesteps=modality.timesteps[:, start:end],
        positions=modality.positions[:, :, start:end],
        attention_mask=_slice_attention_mask(modality.attention_mask, start, end, total_tokens),
        video_control=video_control,
    )


def slice_latent_state(state: LatentState, start: int, end: int) -> LatentState:
    """Slice an inference latent state along its token dimension."""

    total_tokens = state.latent.shape[1]
    if not 0 <= start < end <= total_tokens:
        raise ValueError(f"Invalid latent-state token slice [{start}, {end})/{total_tokens}")
    return LatentState(
        latent=state.latent[:, start:end],
        denoise_mask=state.denoise_mask[:, start:end],
        positions=state.positions[:, :, start:end],
        clean_latent=state.clean_latent[:, start:end],
        attention_mask=_slice_attention_mask(state.attention_mask, start, end, total_tokens),
    )


def stitch_token_chunks(chunks: list[Tensor], *, expected_tokens: int, label: str) -> Tensor:
    if not chunks:
        raise ValueError(f"No {label} chunks to stitch")
    result = torch.cat(chunks, dim=1)
    if result.shape[1] != expected_tokens:
        raise RuntimeError(f"Stitched {label} has {result.shape[1]} tokens, expected {expected_tokens}")
    return result


__all__ = [
    "AVBlock",
    "plan_av_blocks",
    "slice_latent_state",
    "slice_modality",
    "stitch_token_chunks",
]
