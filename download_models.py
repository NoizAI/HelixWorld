#!/usr/bin/env python3
"""Download and verify the pinned HelixWorld Preview v1 inference assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "configs/model_manifest.json"
MODELS_DIR = ROOT / "models"
MODEL_REPO = "NoizAI/HelixWorld-preview"
MODEL_REVISION = "0f6aa9f329a118fcf0e16e0d315da026455ad8da"


def read_manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise RuntimeError("unsupported model manifest schema")
    if value.get("repository") != MODEL_REPO:
        raise RuntimeError("model repository drift")
    if value.get("revision") != MODEL_REVISION:
        raise RuntimeError("model revision drift")
    if value.get("deployment") != "helixworld_preview_v1":
        raise RuntimeError("unexpected deployment in model manifest")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != {"model"}:
        raise RuntimeError("model manifest must contain exactly one model checkpoint")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_file(
    role: str,
    path: Path,
    record: dict[str, Any],
    *,
    verify_hash: bool,
) -> None:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{role}: missing regular file {path}")
    actual_size = path.stat().st_size
    expected_size = int(record["size_bytes"])
    if actual_size != expected_size:
        raise RuntimeError(
            f"{role}: size mismatch {actual_size} != {expected_size}: {path}"
        )
    if verify_hash:
        actual_hash = sha256_file(path)
        expected_hash = str(record["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{role}: SHA256 mismatch {actual_hash} != {expected_hash}: {path}"
            )
    print(f"OK {role}: {path} ({actual_size:,} bytes)", flush=True)


def verify_text_encoder(record: dict[str, Any]) -> None:
    directory = MODELS_DIR / "text_encoder/gemma-3-12b"
    required = {
        "config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    shards = (
        sorted(directory.glob("model-*-of-*.safetensors"))
        if directory.is_dir()
        else []
    )
    if missing or not shards or directory.is_symlink():
        raise FileNotFoundError(
            f"incomplete text encoder at {directory}; "
            f"missing={missing}, shard_count={len(shards)}"
        )
    print(
        f"OK text_encoder: {directory.resolve()} "
        f"({len(shards)} shards, revision {record['revision']})",
        flush=True,
    )


def download(manifest: dict[str, Any]) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "install requirements first: python -m pip install -r requirements.txt"
        ) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    record = manifest["files"]["model"]
    remote_path = str(record["path"])
    print(f"Downloading model from {MODEL_REPO}@{MODEL_REVISION}...", flush=True)
    hf_hub_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        filename=remote_path,
        local_dir=MODELS_DIR,
    )

    text_encoder = manifest["text_encoder"]
    print(
        "Downloading text encoder from "
        f"{text_encoder['repo_id']}@{text_encoder['revision']}...",
        flush=True,
    )
    snapshot_download(
        repo_id=str(text_encoder["repo_id"]),
        revision=str(text_encoder["revision"]),
        local_dir=MODELS_DIR / "text_encoder/gemma-3-12b",
        allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "*.jinja"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not access the network; only validate existing local assets.",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Validate sizes but skip SHA256 reads of the large model files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest()
    if not args.verify_only:
        if not os.environ.get("HF_TOKEN"):
            print(
                "HF_TOKEN is not set; using the credential saved by `hf auth login`.",
                flush=True,
            )
        download(manifest)

    record = manifest["files"]["model"]
    verify_file(
        "model",
        MODELS_DIR / str(record["path"]),
        record,
        verify_hash=not args.skip_hash,
    )
    verify_text_encoder(manifest["text_encoder"])
    print(
        json.dumps(
            {
                "status": "models_ready",
                "deployment": "helixworld_preview_v1",
                "revision": MODEL_REVISION,
                "hashes_verified": not args.skip_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
