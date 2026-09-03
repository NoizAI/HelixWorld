#!/usr/bin/env python3
"""Run HelixWorld Preview v1 joint audio-video inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from _common import (
    MODEL_WEIGHTS,
    PACKAGE_ROOT,
    TEXT_ENCODER,
    install_source_paths,
    materialize_runtime_manifest,
    validate_model_file,
)
from caption_pipeline import DEFAULT_PROMPT_PATH, REWRITE_MODES

install_source_paths()

import inference_runtime  # noqa: E402
from release_model import load_release_model  # noqa: E402
from runtime_policy import RuntimePolicy  # noqa: E402
from schedule_runtime import release_schedule_spec  # noqa: E402

DEPLOYMENT = "helixworld_preview_v1"
RUNTIME_POLICY = RuntimePolicy(
    contract=DEPLOYMENT,
    history_limit=3,
    fixed_prefix=1,
    refresh_mode="per_segment",
    selection="fixed_prefix_plus_recent",
)
RUNTIME_SPEC: dict[str, Any] = {
    "schema_version": 1,
    "deployment": DEPLOYMENT,
    "schedule": release_schedule_spec(),
}


def add_caption_rewrite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--caption-rewrite-mode",
        choices=REWRITE_MODES,
        default="auto",
        help="auto rewrites only noncanonical captions with packaged local Gemma.",
    )
    parser.add_argument(
        "--caption-rewrite-device",
        default=None,
        help="Gemma rewrite device; defaults to the inference --device.",
    )
    parser.add_argument(
        "--caption-rewrite-prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
    )
    parser.add_argument("--caption-rewrite-max-new-tokens", type=int, default=640)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-model-hashes", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single", help="Generate one controlled clip.")
    single.add_argument("--manifest", type=Path, required=True)
    single.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "outputs/interactive_single",
    )
    single.add_argument("--num-frames", type=int, default=121)
    single.add_argument("--control-mode", choices=("native", "neutral"), default="native")
    single.add_argument("--device", default="cuda:0")
    single.add_argument("--seed", type=int, default=42)
    single.add_argument(
        "--attention-backend",
        choices=("automatic", "sdpa_flash"),
        default="sdpa_flash",
    )
    add_caption_rewrite_args(single)
    single.add_argument("--hud", action="store_true")
    single.add_argument("--preflight-only", action="store_true")
    single.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_release_assets() -> tuple[str, dict[str, Any], dict[str, Any]]:
    model = validate_model_file("model", verify_hash=False)
    return (
        "v1",
        RUNTIME_SPEC,
        {
            "deployment": DEPLOYMENT,
            "model": model,
        },
    )


def load_model_once(
    *,
    device: torch.device,
    attention_backend: str,
) -> torch.nn.Module:
    return load_release_model(
        model_path=MODEL_WEIGHTS,
        device=device,
        attention_backend=attention_backend,
    )


def materialize_manifest(args: argparse.Namespace, destination: Path) -> dict[str, Any]:
    return materialize_runtime_manifest(
        args.manifest,
        destination,
        caption_rewrite_mode=args.caption_rewrite_mode,
        caption_rewrite_device=args.caption_rewrite_device or args.device,
        caption_rewrite_prompt=args.caption_rewrite_prompt,
        caption_rewrite_max_new_tokens=args.caption_rewrite_max_new_tokens,
    )


def install_runtime_hooks() -> None:
    inference_runtime.validate_release_assets = validate_release_assets
    inference_runtime.load_inference_model = load_model_once
    inference_runtime.RUNTIME_POLICY = RUNTIME_POLICY


def run_single(args: argparse.Namespace) -> None:
    if args.num_frames < 121 or args.num_frames % 8 != 1:
        raise ValueError("--num-frames must be >=121 and satisfy frames % 8 == 1")
    latent_frames = (args.num_frames - 1) // 8 + 1
    if latent_frames % 4:
        raise ValueError("the resulting latent-frame count must be divisible by 4")
    output_dir = args.output_dir.expanduser().resolve()
    runtime_manifest = output_dir / "_runtime_input/selection_manifest.json"
    runtime = materialize_manifest(args, runtime_manifest)
    if len(runtime["samples"]) != 1:
        raise RuntimeError("single mode requires exactly one manifest sample")

    forwarded = [
        sys.argv[0],
        "--manifest",
        str(runtime_manifest),
        "--output-dir",
        str(output_dir),
        "--components-path",
        str(MODEL_WEIGHTS),
        "--text-encoder-path",
        str(TEXT_ENCODER),
        "--device",
        args.device,
        "--control-mode",
        args.control_mode,
        "--num-frames",
        str(args.num_frames),
        "--seed",
        str(args.seed),
        "--attention-backend",
        args.attention_backend,
    ]
    if not args.hud:
        forwarded.append("--no-action-overlay")
    if args.preflight_only:
        forwarded.append("--preflight-only")
    if args.overwrite:
        forwarded.append("--overwrite")
    sys.argv = forwarded
    inference_runtime.main()


def main() -> None:
    args = parse_args()
    validate_model_file("model", verify_hash=args.verify_model_hashes)
    if not TEXT_ENCODER.is_dir() or TEXT_ENCODER.is_symlink():
        raise FileNotFoundError(TEXT_ENCODER)
    install_runtime_hooks()
    run_single(args)
    print(json.dumps({"status": "helixworld_preview_v1_ok", "mode": args.command}))


if __name__ == "__main__":
    main()
