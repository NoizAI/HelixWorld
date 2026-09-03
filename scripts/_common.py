#!/usr/bin/env python3
"""Shared path, manifest, and integrity helpers for HelixWorld inference."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from caption_pipeline import (
    DEFAULT_PROMPT_PATH,
    normalize_manifest_captions,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKAGE_ROOT / "code/LTX-2.3"
RUNTIME_SCRIPTS = CODE_ROOT / "projects/helixworld_runtime/scripts"
MODEL_MANIFEST_PATH = PACKAGE_ROOT / "configs/model_manifest.json"
MODELS_DIR = PACKAGE_ROOT / "models"

MODEL_WEIGHTS = MODELS_DIR / "weights/model.safetensors"
TEXT_ENCODER = MODELS_DIR / "text_encoder/gemma-3-12b"


def install_source_paths() -> None:
    """Prefer the bundled, patched source tree over globally installed copies."""

    candidates = (
        RUNTIME_SCRIPTS,
        CODE_ROOT / "packages/ltx-runtime/src",
        CODE_ROOT / "packages/ltx-core/src",
    )
    for candidate in reversed(candidates):
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        candidate_text = str(candidate)
        if candidate_text in sys.path:
            sys.path.remove(candidate_text)
        sys.path.insert(0, candidate_text)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest() -> dict[str, Any]:
    return read_json(MODEL_MANIFEST_PATH)


def validate_model_file(role: str, *, verify_hash: bool = False) -> dict[str, Any]:
    manifest = model_manifest()
    records = manifest.get("files")
    if not isinstance(records, dict) or role not in records:
        raise KeyError(f"unknown model role: {role}")
    record = records[role]
    path = MODELS_DIR / str(record["path"])
    if not isinstance(record, dict):
        raise TypeError(f"invalid model record: {role}")
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing or symlinked model asset: {path}")
    expected_size = int(record["size_bytes"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"model size mismatch for {role}: {path.stat().st_size} != {expected_size}"
        )
    expected_hash = str(record["sha256"])
    if verify_hash and sha256_file(path) != expected_hash:
        raise RuntimeError(f"model SHA256 mismatch for {role}: {path}")
    return {
        "role": role,
        "path": str(path),
        "size_bytes": expected_size,
        "sha256": expected_hash,
        "hash_verified": verify_hash,
    }


def _resolve_input_path(value: object, root: Path, label: str) -> Path:
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"missing or symlinked {label}: {candidate}")
    return candidate


def materialize_runtime_manifest(
    source: Path,
    destination: Path,
    *,
    caption_rewrite_mode: str = "auto",
    caption_rewrite_device: str = "cuda:0",
    caption_rewrite_prompt: Path = DEFAULT_PROMPT_PATH,
    caption_rewrite_max_new_tokens: int = 640,
) -> dict[str, Any]:
    """Resolve paths and enforce the HelixWorld triplet-caption contract.

    Canonical captions pass through without loading Gemma. In ``auto`` mode,
    any other non-empty caption is rewritten by the packaged local Gemma model
    and cached beside the runtime manifest before diffusion inference starts.
    """

    source = source.expanduser().resolve()
    payload = read_json(source)
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported manifest schema: {source}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"manifest has no samples: {source}")

    canonical: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(samples):
        if not isinstance(raw, dict):
            raise TypeError(f"sample {ordinal} is not an object")
        sample = dict(raw)
        recorded_ordinal = sample.get("ordinal", ordinal)
        if recorded_ordinal != ordinal:
            raise RuntimeError(
                f"sample order is noncanonical: position={ordinal} ordinal={recorded_ordinal}"
            )
        caption = sample.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise RuntimeError(f"sample {ordinal} has no caption")
        source_video = _resolve_input_path(
            sample.get("source_video"), source.parent, f"sample {ordinal} source video"
        )
        controls = sample.get("controls")
        if not isinstance(controls, dict):
            raise RuntimeError(f"sample {ordinal} has no controls")
        canonical_controls = dict(controls)
        for mode in ("native", "neutral"):
            canonical_controls[mode] = str(
                _resolve_input_path(
                    controls.get(mode),
                    source.parent,
                    f"sample {ordinal} {mode} control",
                )
            )
            action_ids = canonical_controls.get(f"{mode}_action_ids")
            if not isinstance(action_ids, list) or not action_ids:
                raise RuntimeError(f"sample {ordinal} has invalid {mode} action IDs")
        sample.update(
            {
                "ordinal": ordinal,
                "source_video": str(source_video),
                "controls": canonical_controls,
            }
        )
        canonical.append(sample)

    canonical, caption_pipeline = normalize_manifest_captions(
        canonical,
        mode=caption_rewrite_mode,
        model_path=TEXT_ENCODER,
        prompt_path=caption_rewrite_prompt,
        device=caption_rewrite_device,
        max_new_tokens=caption_rewrite_max_new_tokens,
        cache_path=destination.parent / "caption_rewrite_cache.json",
    )

    runtime = {
        **payload,
        "schema_version": 1,
        "source_manifest": str(source),
        "caption_pipeline": caption_pipeline,
        "samples": canonical,
    }
    atomic_json(destination, runtime)
    return runtime
