"""Strict configuration objects for the packaged inference path."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InferenceSample(RuntimeConfig):
    prompt: str = Field(min_length=1)
    first_frame: Path
    seed: int = 42


class InferenceConfig(RuntimeConfig):
    """The only supported release task: one first-frame-conditioned AV clip."""

    sample: InferenceSample
    video_dims: tuple[int, int, int] = (768, 512, 121)
    frame_rate: float = Field(default=24.0, gt=0)
    inference_steps: Literal[4] = 4

    @field_validator("video_dims")
    @classmethod
    def validate_video_dims(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        width, height, frames = value
        if width % 32:
            raise ValueError(f"width must be divisible by 32; got {width}")
        if height % 32:
            raise ValueError(f"height must be divisible by 32; got {height}")
        if frames < 121 or frames % 8 != 1:
            raise ValueError(f"frames must be >= 121 and satisfy frames % 8 == 1; got {frames}")
        latent_frames = (frames - 1) // 8 + 1
        if latent_frames % 4:
            raise ValueError(f"latent frame count must be divisible by 4; got {latent_frames}")
        return value


__all__ = ["InferenceConfig", "InferenceSample"]
