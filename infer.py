#!/usr/bin/env python3
"""One-command inference for HelixWorld Preview v1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from controls import prepare_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, required=True, help="Conditioning image (JPG/PNG/WebP)."
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="JSON object containing video, audio, and av strings.",
    )
    parser.add_argument("--video-prompt", help="Visual scene and motion description.")
    parser.add_argument("--audio-prompt", help="Soundscape description.")
    parser.add_argument("--av-prompt", help="Joint audio-visual description.")
    parser.add_argument(
        "--actions",
        default="W",
        help="Comma-separated latent-step plan, e.g. 'W:5,right:5,stop:5'.",
    )
    parser.add_argument(
        "--perspective",
        choices=("first_person", "third_person"),
        default="first_person",
    )
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-backend",
        choices=("automatic", "sdpa_flash"),
        default="sdpa_flash",
    )
    parser.add_argument(
        "--hud", action="store_true", help="Overlay the action HUD on an extra MP4."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-model-hashes", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all inputs and model paths without loading the diffusion model.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only materialize the manifest/control tensors; model files are not required.",
    )
    return parser.parse_args()


def read_prompts(args: argparse.Namespace) -> tuple[str, str, str]:
    inline = (args.video_prompt, args.audio_prompt, args.av_prompt)
    if args.prompt_file is not None and any(value is not None for value in inline):
        raise ValueError(
            "use either --prompt-file or the three inline prompt arguments, not both"
        )
    if args.prompt_file is not None:
        prompt_path = args.prompt_file.expanduser().resolve()
        payload: Any = json.loads(prompt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"prompt file must contain a JSON object: {prompt_path}")
        try:
            values = tuple(payload[key] for key in ("video", "audio", "av"))
        except KeyError as exc:
            raise ValueError(
                f"prompt file is missing key {exc.args[0]!r}: {prompt_path}"
            ) from exc
    else:
        if any(value is None for value in inline):
            raise ValueError(
                "provide --prompt-file or all of --video-prompt, --audio-prompt, and --av-prompt"
            )
        values = inline
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("video, audio, and av prompts must all be non-empty strings")
    return values[0].strip(), values[1].strip(), values[2].strip()


def main() -> None:
    args = parse_args()
    video_prompt, audio_prompt, av_prompt = read_prompts(args)
    output_dir = args.output_dir.expanduser().resolve()
    manifest = prepare_manifest(
        image_path=args.image,
        output_dir=output_dir,
        video_prompt=video_prompt,
        audio_prompt=audio_prompt,
        av_prompt=av_prompt,
        actions_spec=args.actions,
        perspective=args.perspective,
        num_frames=args.num_frames,
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "manifest": str(manifest),
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
            )
        )
        return

    command = [sys.executable, str(SCRIPTS / "infer_model.py")]
    if args.verify_model_hashes:
        command.append("--verify-model-hashes")
    command.extend(
        [
            "single",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--num-frames",
            str(args.num_frames),
            "--control-mode",
            "native",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--attention-backend",
            args.attention_backend,
            "--caption-rewrite-mode",
            "never",
        ]
    )
    if args.hud:
        command.append("--hud")
    if args.preflight_only:
        command.append("--preflight-only")
    if args.overwrite:
        command.append("--overwrite")
    environment = dict(os.environ)
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
