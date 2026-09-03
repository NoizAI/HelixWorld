#!/usr/bin/env python3
"""Build HelixWorld camera/action tensors from a compact action plan."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import install_source_paths
from caption_pipeline import render_triplet_caption
from PIL import Image

install_source_paths()

from ltx_runtime.video_controls import (  # noqa: E402
    CameraTranslationNormalizationConfig,
    compute_spatial_transform_metadata,
    expand_video_controls_to_tokens,
    prepare_video_controls,
)

WIDTH = 768
HEIGHT = 512
FPS = 24
TEMPORAL_COMPRESSION = 8
LATENT_HEIGHT = HEIGHT // 32
LATENT_WIDTH = WIDTH // 32

FORWARD_SPEED = 0.08
YAW_SPEED = np.deg2rad(3.0)
PITCH_SPEED = np.deg2rad(3.0)
ORBIT_RADIUS = FORWARD_SPEED / YAW_SPEED
ORBIT_HEIGHT = 0.3
DEFAULT_INTRINSIC = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]

SINGLE_KEY_NAV: dict[str, dict[str, Any]] = {
    "W": {"move": [1, 0], "yaw": 0, "pitch": 0},
    "S": {"move": [-1, 0], "yaw": 0, "pitch": 0},
    "A": {"move": [0, -1], "yaw": 0, "pitch": 0},
    "D": {"move": [0, 1], "yaw": 0, "pitch": 0},
    "right": {"move": [0, 0], "yaw": 1, "pitch": 0},
    "left": {"move": [0, 0], "yaw": -1, "pitch": 0},
    "up": {"move": [0, 0], "yaw": 0, "pitch": 1},
    "down": {"move": [0, 0], "yaw": 0, "pitch": -1},
    "stop": {"move": [0, 0], "yaw": 0, "pitch": 0},
}
ALIASES = {"→": "right", "←": "left", "↑": "up", "↓": "down"}
NORMALIZATION = CameraTranslationNormalizationConfig(
    quantile=0.75,
    target_step=0.03,
    only_shrink=True,
    max_radius=1.5,
    epsilon=1.0e-8,
)


def latent_frames_for(num_frames: int) -> int:
    if num_frames < 121 or num_frames % TEMPORAL_COMPRESSION != 1:
        raise ValueError("num_frames must be >= 121 and satisfy num_frames % 8 == 1")
    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    if latent_frames % 4:
        raise ValueError(
            "the video latent-frame count must be divisible by 4; "
            "supported values are 121, 153, 185, ..."
        )
    return latent_frames


def _canonical_key(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty action key")
    aliased = ALIASES.get(stripped, stripped)
    upper = aliased.upper()
    if upper in {"W", "A", "S", "D"}:
        return upper
    lowered = aliased.lower()
    if lowered in {"left", "right", "up", "down", "stop"}:
        return lowered
    raise ValueError(
        f"unsupported action {value!r}; use W/A/S/D, left/right/up/down, stop, or '+' combinations"
    )


def action_to_navigation(action: str) -> dict[str, Any]:
    parts = [_canonical_key(part) for part in action.split("+")]
    forward = right = yaw = pitch = 0
    for part in parts:
        navigation = SINGLE_KEY_NAV[part]
        forward += int(navigation["move"][0])
        right += int(navigation["move"][1])
        yaw += int(navigation["yaw"])
        pitch += int(navigation["pitch"])
    return {"move": [forward, right], "yaw": yaw, "pitch": pitch}


def _split_segment(segment: str) -> tuple[str, int | None]:
    value = segment.strip()
    match = re.fullmatch(r"(.+?)(?::(\d+)|-(\d+))?", value)
    if match is None:
        raise ValueError(f"invalid action segment: {segment!r}")
    action = match.group(1).strip()
    duration_text = match.group(2) or match.group(3)
    action_to_navigation(action)
    if duration_text is None:
        return action, None
    duration = int(duration_text)
    if duration <= 0:
        raise ValueError(f"action duration must be positive: {segment!r}")
    return action, duration


def expand_action_plan(spec: str, *, transition_count: int) -> list[str]:
    """Expand ``W:5,right:5,stop:5`` into one action per latent transition."""

    if transition_count <= 0:
        raise ValueError("transition_count must be positive")
    segments = [_split_segment(value) for value in spec.split(",") if value.strip()]
    if not segments:
        raise ValueError("the action plan is empty")
    implicit = [
        index for index, (_, duration) in enumerate(segments) if duration is None
    ]
    if implicit and implicit != [len(segments) - 1]:
        raise ValueError("only the final action may omit its duration")
    explicit_steps = sum(duration or 0 for _, duration in segments)
    if implicit:
        remaining = transition_count - explicit_steps
        if remaining <= 0:
            raise ValueError("the final implicit action has no remaining latent steps")
        action, _ = segments[-1]
        segments[-1] = (action, remaining)
    elif explicit_steps != transition_count:
        raise ValueError(
            f"action durations total {explicit_steps}, but this clip needs {transition_count}; "
            "omit the final duration to fill the remainder"
        )
    actions = [action for action, duration in segments for _ in range(int(duration))]
    if len(actions) != transition_count:
        raise RuntimeError("expanded action plan length drift")
    return actions


def _direction_label(primary: int, secondary: int) -> int:
    labels = {
        (0, 0): 0,
        (1, 0): 1,
        (-1, 0): 2,
        (0, 1): 3,
        (0, -1): 4,
        (1, 1): 5,
        (1, -1): 6,
        (-1, 1): 7,
        (-1, -1): 8,
    }
    return labels[(int(np.sign(primary)), int(np.sign(secondary)))]


def action_id(action: str) -> int:
    navigation = action_to_navigation(action)
    translation = _direction_label(navigation["move"][0], navigation["move"][1])
    rotation = _direction_label(navigation["yaw"], navigation["pitch"])
    return translation * 9 + rotation


def _motion(action: str) -> dict[str, float]:
    navigation = action_to_navigation(action)
    motion: dict[str, float] = {}
    forward, right = navigation["move"]
    if forward:
        motion["forward"] = FORWARD_SPEED * float(np.sign(forward))
    if right:
        motion["right"] = FORWARD_SPEED * float(np.sign(right))
    if navigation["yaw"]:
        motion["yaw"] = YAW_SPEED * float(np.sign(navigation["yaw"]))
    if navigation["pitch"]:
        motion["pitch"] = PITCH_SPEED * float(np.sign(navigation["pitch"]))
    return motion


def _rot_x(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])


def _rot_y(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])


def _first_person_poses(motions: list[dict[str, float]]) -> list[np.ndarray]:
    transform = np.eye(4)
    poses = [transform.copy()]
    for motion in motions:
        if "yaw" in motion:
            transform[:3, :3] = transform[:3, :3] @ _rot_y(motion["yaw"])
        if "pitch" in motion:
            transform[:3, :3] = transform[:3, :3] @ _rot_x(motion["pitch"])
        if "forward" in motion:
            transform[:3, 3] += transform[:3, :3] @ np.array([0, 0, motion["forward"]])
        if "right" in motion:
            transform[:3, 3] += transform[:3, :3] @ np.array([motion["right"], 0, 0])
        poses.append(transform.copy())
    return poses


def _third_person_poses(motions: list[dict[str, float]]) -> list[np.ndarray]:
    azimuth = np.pi
    elevation = 0.0
    character_position = np.zeros(3)

    def camera_pose() -> np.ndarray:
        camera_position = character_position + np.array(
            [
                ORBIT_RADIUS * np.cos(elevation) * np.sin(azimuth),
                ORBIT_HEIGHT + ORBIT_RADIUS * np.sin(elevation),
                ORBIT_RADIUS * np.cos(elevation) * np.cos(azimuth),
            ]
        )
        forward = (
            character_position + np.array([0, ORBIT_HEIGHT * 0.5, 0]) - camera_position
        )
        forward /= np.linalg.norm(forward) + 1.0e-8
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right) + 1.0e-8
        up = np.cross(right, forward)
        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = forward
        transform[:3, 3] = camera_position
        return transform

    poses = [camera_pose()]
    for raw_motion in motions:
        # Character-right is opposite camera-local +X in the initial rear-view
        # orbit. Flip only this scalar so A/D match the action IDs.
        motion = dict(raw_motion)
        if "right" in motion:
            motion["right"] = -motion["right"]
        if "yaw" in motion:
            azimuth -= motion["yaw"]
        if "pitch" in motion:
            elevation = np.clip(
                elevation - motion["pitch"], np.deg2rad(-60), np.deg2rad(60)
            )
        character_position += np.array(
            [motion.get("right", 0.0), 0.0, motion.get("forward", 0.0)]
        )
        poses.append(camera_pose())
    return poses


def camera_poses(actions: list[str], *, perspective: str) -> list[np.ndarray]:
    motions = [_motion(action) for action in actions]
    if perspective == "first_person":
        return _first_person_poses(motions)
    if perspective == "third_person":
        return _third_person_poses(motions)
    raise ValueError("perspective must be 'first_person' or 'third_person'")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_control(
    path: Path,
    *,
    image_size: tuple[int, int],
    poses: list[np.ndarray],
    action_ids: list[int],
) -> None:
    latent_frames = len(action_ids)
    if len(poses) != latent_frames or action_ids[0] != 0:
        raise ValueError(
            "control must have one pose per action and start with neutral action 0"
        )
    source_width, source_height = image_size
    spatial_transform = compute_spatial_transform_metadata(
        source_height=source_height,
        source_width=source_width,
        target_height=HEIGHT,
        target_width=WIDTH,
    )
    camera_to_world = torch.from_numpy(np.stack(poses)).to(torch.float32)
    world_to_camera = torch.linalg.inv(camera_to_world)
    intrinsics = torch.tensor(DEFAULT_INTRINSIC, dtype=torch.float32).repeat(
        latent_frames, 1, 1
    )
    valid = torch.ones(latent_frames, dtype=torch.bool)
    explicit = torch.tensor(action_ids, dtype=torch.long)
    controls = prepare_video_controls(
        intrinsics=intrinsics,
        w2c=world_to_camera,
        spatial_transform=spatial_transform,
        calibration_size=(1080, 1920),
        frame_indices=torch.arange(latent_frames),
        camera_valid_mask=valid,
        explicit_action_ids=explicit,
        explicit_action_valid_mask=valid,
        action_algorithm="robust_v2",
        translation_normalization="robust_latent_step",
        translation_normalization_config=NORMALIZATION,
    )
    if not torch.equal(controls.action_ids.squeeze(0), explicit):
        raise RuntimeError("explicit action overlay drift")
    token_controls = expand_video_controls_to_tokens(
        controls,
        latent_num_frames=latent_frames,
        latent_height=LATENT_HEIGHT,
        latent_width=LATENT_WIDTH,
    )
    expected_tokens = latent_frames * LATENT_HEIGHT * LATENT_WIDTH
    if token_controls.relative_w2c.shape[:2] != (1, expected_tokens):
        raise RuntimeError("camera token shape drift")
    payload = {
        "camera_intrinsics": token_controls.normalized_intrinsics.to(
            torch.bfloat16
        ).cpu(),
        "camera_w2c": token_controls.relative_w2c.to(torch.bfloat16).cpu(),
        "camera_valid_mask": token_controls.camera_valid_mask.to(torch.bool).cpu(),
        "action_valid_mask": token_controls.action_valid_mask.to(torch.bool).cpu(),
        "action_ids": explicit.cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def prepare_manifest(
    *,
    image_path: Path,
    output_dir: Path,
    video_prompt: str,
    audio_prompt: str,
    av_prompt: str,
    actions_spec: str,
    perspective: str,
    num_frames: int,
) -> Path:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file() or image_path.is_symlink():
        raise FileNotFoundError(
            f"conditioning image is missing or symlinked: {image_path}"
        )
    with Image.open(image_path) as image:
        image.verify()
        image_size = image.size
    latent_frames = latent_frames_for(num_frames)
    actions = expand_action_plan(actions_spec, transition_count=latent_frames - 1)
    native_ids = [0] + [action_id(action) for action in actions]
    native_poses = camera_poses(actions, perspective=perspective)
    neutral_actions = ["stop"] * (latent_frames - 1)
    neutral_ids = [0] * latent_frames
    neutral_poses = camera_poses(neutral_actions, perspective=perspective)

    input_dir = output_dir.expanduser().resolve() / "_prepared_input"
    native_path = input_dir / "controls/native.pt"
    neutral_path = input_dir / "controls/neutral.pt"
    _write_control(
        native_path,
        image_size=image_size,
        poses=native_poses,
        action_ids=native_ids,
    )
    _write_control(
        neutral_path,
        image_size=image_size,
        poses=neutral_poses,
        action_ids=neutral_ids,
    )
    caption = render_triplet_caption(video_prompt, audio_prompt, av_prompt)
    manifest = {
        "schema_version": 1,
        "deployment": "helixworld_preview_v1",
        "frame_rate": FPS,
        "generation": {"width": WIDTH, "height": HEIGHT, "num_frames": num_frames},
        "samples": [
            {
                "ordinal": 0,
                "sample_id": "helixworld_preview_sample_000",
                "source_video": str(image_path),
                "caption": caption,
                "caption_components": {
                    "video": video_prompt.strip(),
                    "audio": audio_prompt.strip(),
                    "av": av_prompt.strip(),
                },
                "perspective": perspective,
                "action_plan": actions_spec,
                "expanded_actions": actions,
                "controls": {
                    "native": "controls/native.pt",
                    "native_action_ids": native_ids,
                    "neutral": "controls/neutral.pt",
                    "neutral_action_ids": neutral_ids,
                },
            }
        ],
    }
    manifest_path = input_dir / "selection_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


__all__ = [
    "action_id",
    "action_to_navigation",
    "camera_poses",
    "expand_action_plan",
    "latent_frames_for",
    "prepare_manifest",
]
