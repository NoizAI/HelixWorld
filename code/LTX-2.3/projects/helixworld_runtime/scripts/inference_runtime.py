#!/usr/bin/env python3
"""Single-sample runtime for HelixWorld Preview v1."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from control_io import load_video_control
from generation_runner import (
    TIMING_RECORDS,
    InteractiveInferenceRunner,
    build_timing_report,
    inference_geometry,
)
from ltx_core.model.transformer.video_control import VideoControlCondition
from ltx_runtime.config import InferenceConfig
from ltx_runtime.progress import InferenceProgress
from runtime_policy import RuntimePolicy
from schedule_runtime import nodes_from_spec

DELIVERY_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COMPONENTS = DELIVERY_ROOT / "models/weights/model.safetensors"
DEFAULT_TEXT_ENCODER = DELIVERY_ROOT / "models/text_encoder/gemma-3-12b"
RUNTIME_POLICY: RuntimePolicy | None = None


def validate_release_assets() -> tuple[str, dict[str, Any], dict[str, Any]]:
    raise RuntimeError("the release asset hook is not installed")


def load_inference_model(
    *,
    device: torch.device,
    attention_backend: str,
) -> torch.nn.Module:
    del device, attention_backend
    raise RuntimeError("the release model loader hook is not installed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--components-path", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--text-encoder-path", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-mode", choices=("native", "neutral"), default="native")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-backend",
        choices=("automatic", "sdpa_flash"),
        default="sdpa_flash",
    )
    parser.add_argument("--no-action-overlay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported input manifest: {path}")
    if payload.get("deployment") != "helixworld_preview_v1":
        raise RuntimeError(f"input manifest deployment drift: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise RuntimeError("inference requires exactly one manifest sample")
    if not isinstance(samples[0], dict):
        raise RuntimeError("manifest sample must be an object")
    return payload


def _resolve_file(value: object, root: Path, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing or symlinked {label}: {path}")
    return path


def _prepare_sample(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    control_mode: str,
    num_frames: int,
) -> tuple[dict[str, Any], VideoControlCondition, list[int]]:
    sample = payload["samples"][0]
    caption = sample.get("caption")
    sample_id = sample.get("sample_id")
    if not isinstance(caption, str) or not caption.strip():
        raise RuntimeError("manifest sample has no caption")
    if not isinstance(sample_id, str) or not sample_id:
        raise RuntimeError("manifest sample has no sample_id")
    source = _resolve_file(sample.get("source_video"), manifest_path.parent, "source image")
    controls = sample.get("controls")
    if not isinstance(controls, dict):
        raise RuntimeError("manifest sample has no controls")
    control_path = _resolve_file(
        controls.get(control_mode),
        manifest_path.parent,
        f"{control_mode} control",
    )
    control, action_ids = load_video_control(
        control_path,
        width=768,
        height=512,
        num_frames=num_frames,
    )
    prepared = {
        "sample_id": sample_id,
        "caption": caption.strip(),
        "source": source,
    }
    return prepared, control, action_ids


def _build_config(
    sample: dict[str, Any],
    *,
    num_frames: int,
    seed: int,
) -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "sample": {
                "prompt": sample["caption"],
                "first_frame": str(sample["source"]),
                "seed": seed,
            },
            "video_dims": [768, 512, num_frames],
            "frame_rate": 24.0,
            "inference_steps": 4,
        }
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    geometry = inference_geometry(args.num_frames)
    release_id, spec, asset_check = validate_release_assets()
    if RUNTIME_POLICY is None:
        raise RuntimeError("the release runtime policy is not installed")

    manifest_path = args.manifest.expanduser().resolve()
    payload = _read_manifest(manifest_path)
    sample, control, action_ids = _prepare_sample(
        manifest_path,
        payload,
        control_mode=args.control_mode,
        num_frames=args.num_frames,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "deployment": "helixworld_preview_v1",
                    "release": release_id,
                    "sample_id": sample["sample_id"],
                    "geometry": geometry,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise RuntimeError(f"an explicit CUDA device is required, got {device}")
    torch.cuda.set_device(device)

    config = _build_config(sample, num_frames=args.num_frames, seed=args.seed)
    runner = InteractiveInferenceRunner(
        config=config,
        model_path=str(args.components_path.expanduser().resolve()),
        text_encoder_path=str(args.text_encoder_path.expanduser().resolve()),
        video_control=control,
        action_ids=action_ids,
        overlay_actions=not args.no_action_overlay,
        release_nodes=nodes_from_spec(spec["schedule"]),
        runtime_policy=RUNTIME_POLICY,
        preprocess_device=device,
    )
    transformer = load_inference_model(
        device=device,
        attention_backend=args.attention_backend,
    )
    runner.bind_video_control(device)

    target_dir = args.output_dir.expanduser().resolve() / "release" / args.control_mode
    if (target_dir / "INFERENCE_COMPLETE.json").exists() and not args.overwrite:
        raise FileExistsError(f"inference output already exists: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    runner.clear_generated_output_cache()
    TIMING_RECORDS.clear()
    with InferenceProgress(enabled=False) as progress:
        generated = runner.run(
            transformer=transformer,
            output_dir=target_dir,
            device=device,
            progress=progress,
        )
    annotated = generated.resolve()
    clean, audio_muxed = runner.save_clean_copy(
        output_dir=target_dir,
        output_name=annotated.name,
    )
    clean = clean.resolve()
    timing = build_timing_report(
        TIMING_RECORDS,
        num_frames=args.num_frames,
        frame_rate=24.0,
    )

    record = {
        "schema_version": 1,
        "deployment": "helixworld_preview_v1",
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "release": release_id,
        "asset_check": asset_check,
        "sample": {
            "sample_id": sample["sample_id"],
            "caption": sample["caption"],
            "source": str(sample["source"]),
        },
        "geometry": geometry,
        "settings": {
            "seed": config.sample.seed,
            "control_mode": args.control_mode,
            "attention_backend": args.attention_backend,
        },
        "outputs": {
            "annotated_av_mp4": str(annotated),
            "clean_av_mp4": str(clean),
            "audio_muxed": audio_muxed,
        },
        "timing_summary": timing["summary"],
    }
    _atomic_json(target_dir / "INFERENCE_COMPLETE.json", record)
    _atomic_json(args.output_dir.expanduser().resolve() / "INFERENCE_TIMING.json", timing)

    runner.unbind_video_control()
    del transformer, runner
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {"status": "complete", "outputs": record["outputs"]},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
