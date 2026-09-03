"""Minimal first-frame-conditioned audio-video inference runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import (
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF  # noqa: N812

from ltx_runtime.config import InferenceConfig, InferenceSample
from ltx_runtime.gpu_utils import free_gpu_memory_context
from ltx_runtime.model_loader import (
    load_audio_decoder,
    load_embeddings_processor,
    load_text_encoder,
    load_video_decoder,
    load_video_encoder,
    load_vocoder,
)
from ltx_runtime.progress import InferenceProgress, SamplingContext
from ltx_runtime.utils import open_image_as_srgb
from ltx_runtime.video_utils import read_video, save_video

if TYPE_CHECKING:
    from ltx_core.model.transformer import LTXModel

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
DECODE_TILING = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=192, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=48, tile_overlap_in_frames=24),
)


@dataclass
class CachedPromptEmbeddings:
    video_context: Tensor
    audio_context: Tensor


@dataclass
class CachedSampleMedia:
    first_frame_latent: Tensor


class InferenceRunner:
    """Load inference auxiliaries once and generate the configured clip."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        model_path: str | Path,
        text_encoder_path: str | Path,
        preprocess_device: torch.device,
    ) -> None:
        self._config = config
        self._model_path = Path(model_path)
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)
        self._cached_embeddings = self._cache_prompt_embeddings(text_encoder_path, preprocess_device)
        self._cached_media = self._encode_first_frame(preprocess_device)
        self._load_decoders()

    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def run(
        self,
        *,
        transformer: "LTXModel",
        output_dir: Path,
        device: torch.device,
        progress: InferenceProgress,
    ) -> Path:
        sampling = progress.start_sampling(self._config.inference_steps)
        sampling.start()
        try:
            video, audio = self._generate_sample(
                sample=self._config.sample,
                cached_embeddings=self._cached_embeddings,
                cached_media=self._cached_media,
                transformer=transformer,
                device=device,
                sampling_ctx=sampling,
            )
        finally:
            sampling.cleanup()
        if video is None or audio is None:
            raise RuntimeError("inference must return both video and audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "generated.mp4"
        save_video(
            video_tensor=video,
            output_path=output_path,
            fps=self._config.frame_rate,
            audio=audio,
            audio_sample_rate=self._vocoder.output_sampling_rate,
            video_format="CFHW",
        )
        return output_path

    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _cache_prompt_embeddings(
        self,
        text_encoder_path: str | Path,
        device: torch.device,
    ) -> CachedPromptEmbeddings:
        text_encoder = load_text_encoder(text_encoder_path, device=device, dtype=torch.bfloat16)
        processor = load_embeddings_processor(self._model_path, device=device, dtype=torch.bfloat16)
        hidden_states, attention_mask = text_encoder.encode([self._config.sample.prompt])[0]
        encoded = processor.process_hidden_states(hidden_states, attention_mask)
        if encoded.audio_encoding is None:
            raise RuntimeError("the text encoder did not return an audio context")
        cached = CachedPromptEmbeddings(
            video_context=encoded.video_encoding.cpu(),
            audio_context=encoded.audio_encoding.cpu(),
        )
        del text_encoder, processor
        return cached

    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _encode_first_frame(self, device: torch.device) -> CachedSampleMedia:
        width, height, _ = self._config.video_dims
        encoder = load_video_encoder(self._model_path, device="cpu", dtype=torch.bfloat16)
        image = self._load_first_frame(self._config.sample.first_frame)
        latent = self._encode_image(image, height, width, encoder, device)
        del encoder
        return CachedSampleMedia(first_frame_latent=latent)

    def _load_decoders(self) -> None:
        self._video_decoder = load_video_decoder(self._model_path, device="cpu", dtype=torch.bfloat16)
        self._audio_decoder = load_audio_decoder(self._model_path, device="cpu", dtype=torch.bfloat16)
        self._vocoder = load_vocoder(self._model_path, device="cpu", dtype=torch.bfloat16)
        self._video_decoder.requires_grad_(False)
        self._audio_decoder.requires_grad_(False)
        self._vocoder.requires_grad_(False)

    def _generate_sample(
        self,
        *,
        sample: InferenceSample,
        cached_embeddings: CachedPromptEmbeddings,
        cached_media: CachedSampleMedia,
        transformer: "LTXModel",
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[Tensor | None, Tensor | None]:
        width, height, num_frames = self._config.video_dims
        video_context = cached_embeddings.video_context.to(device)
        audio_context = cached_embeddings.audio_context.to(device)
        generator = torch.Generator(device=device).manual_seed(sample.seed)
        noiser = GaussianNoiser(generator=generator)

        video_tools = self._create_video_tools(width, height, num_frames)
        video_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        first_frame = cached_media.first_frame_latent.to(device=device, dtype=torch.bfloat16)
        video_state = VideoConditionByLatentIndex(
            latent=first_frame,
            strength=1.0,
            latent_idx=0,
        ).apply_to(video_state, video_tools)
        video_clean = video_state
        video_state = noiser(video_state, noise_scale=1.0)

        audio_tools = self._create_audio_tools(num_frames)
        audio_state = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        audio_clean = audio_state
        audio_state = noiser(audio_state, noise_scale=1.0)

        video_state, audio_state = self._run_denoising(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            video_clean=video_clean,
            audio_clean=audio_clean,
            video_context=video_context,
            audio_context=audio_context,
            device=device,
            sampling_ctx=sampling_ctx,
        )
        video = self._finalize_modality(video_state, video_tools, self._decode_video, device)
        audio = self._finalize_modality(audio_state, audio_tools, self._decode_audio, device)
        return video, audio

    def _run_denoising(  # noqa: PLR0913
        self,
        *,
        transformer: "LTXModel",
        video_state: LatentState,
        audio_state: LatentState,
        video_clean: LatentState,
        audio_clean: LatentState,
        video_context: Tensor,
        audio_context: Tensor,
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[LatentState, LatentState]:
        raise NotImplementedError

    @staticmethod
    def _finalize_modality(
        state: LatentState,
        tools: VideoLatentTools | AudioLatentTools,
        decode_fn: Callable[[LatentState, torch.device], Tensor],
        device: torch.device,
    ) -> Tensor:
        state = tools.clear_conditioning(state)
        state = tools.unpatchify(state)
        return decode_fn(state, device)

    def _decode_video(self, state: LatentState, device: torch.device) -> Tensor:
        self._video_decoder.to(device)
        latent = state.latent.to(dtype=torch.bfloat16)
        chunks = list(self._video_decoder.tiled_decode(latent, tiling_config=DECODE_TILING))
        decoded = torch.cat(chunks, dim=2)
        self._video_decoder.to("cpu")
        return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)[0].float().cpu()

    def _decode_audio(self, state: LatentState, device: torch.device) -> Tensor:
        self._audio_decoder.to(device)
        parameter = next(self._audio_decoder.parameters(), None)
        dtype = parameter.dtype if parameter is not None else state.latent.dtype
        decoded = self._audio_decoder(state.latent.to(device=device, dtype=dtype))
        self._audio_decoder.to("cpu")
        self._vocoder.to(device)
        waveform = self._vocoder(decoded)
        self._vocoder.to("cpu")
        return waveform.squeeze(0).float().cpu()

    def _create_video_tools(self, width: int, height: int, frames: int) -> VideoLatentTools:
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=frames,
            height=height,
            width=width,
            fps=self._config.frame_rate,
        )
        return VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(pixel_shape),
            fps=self._config.frame_rate,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )

    def _create_audio_tools(self, frames: int) -> AudioLatentTools:
        return AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(
                batch=1,
                duration=frames / self._config.frame_rate,
            ),
        )

    @staticmethod
    def _modality_from_latent_state(state: LatentState, context: Tensor, sigma: Tensor) -> Modality:
        return Modality(
            enabled=True,
            latent=state.latent,
            sigma=sigma,
            timesteps=state.denoise_mask * sigma,
            positions=state.positions,
            context=context,
            context_mask=None,
        )

    @staticmethod
    def _load_first_frame(path: Path) -> Tensor:
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            frames, _ = read_video(str(path), max_frames=1)
            return frames[0]
        return TF.to_tensor(open_image_as_srgb(path))

    @staticmethod
    def _resize_and_center_crop(tensor: Tensor, height: int, width: int) -> Tensor:
        current_height, current_width = tensor.shape[2:]
        if (current_height, current_width) == (height, width):
            return tensor
        aspect = current_width / current_height
        target_aspect = width / height
        if aspect > target_aspect:
            resize_height, resize_width = height, int(height * aspect)
        else:
            resize_height, resize_width = int(width / aspect), width
        tensor = TF.resize(
            tensor,
            size=[resize_height, resize_width],
            interpolation=InterpolationMode.BICUBIC,
        ).clamp(0, 1)
        y = (resize_height - height) // 2
        x = (resize_width - width) // 2
        return tensor[:, :, y : y + height, x : x + width]

    @classmethod
    def _encode_image(
        cls,
        image: Tensor,
        height: int,
        width: int,
        encoder: torch.nn.Module,
        device: torch.device,
    ) -> Tensor:
        image = cls._resize_and_center_crop(image.unsqueeze(0), height, width)
        image = (image.unsqueeze(2) * 2.0 - 1.0).to(device=device, dtype=torch.bfloat16)
        encoder.to(device)
        encoded = encoder(image)
        encoder.to("cpu")
        return encoded.cpu()


__all__ = ["CachedPromptEmbeddings", "CachedSampleMedia", "InferenceRunner"]
