#!/usr/bin/env python3
"""Validate and, when required, locally rewrite HelixWorld captions."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = PACKAGE_ROOT / "configs/caption_rewrite_prompt_v1.txt"
REWRITE_VERSION = "helixworld_gemma3_triplet_rewrite_v1"
TEMPLATE_VERSION = "ltx23_triplet_concat_v1"
OFFICIAL_MAX_TOKENS = 1024
MAX_SOURCE_CHARS = 32_768
REWRITE_MODES = ("auto", "never", "force")

VIDEO_HEADER = "Video description:"
AUDIO_HEADER = "Audio description:"
AV_HEADER = "Joint audio-visual description:"
HEADERS = (VIDEO_HEADER, AUDIO_HEADER, AV_HEADER)
_TRIPLET_PATTERN = re.compile(
    rf"\A{re.escape(VIDEO_HEADER)}\n(?P<video>.+?)"
    rf"\n\n{re.escape(AUDIO_HEADER)}\n(?P<audio>.+?)"
    rf"\n\n{re.escape(AV_HEADER)}\n(?P<av>.+)\Z",
    flags=re.DOTALL,
)


class CaptionFormatError(ValueError):
    """Raised when a caption is not the exact, non-empty three-section format."""


class CaptionRewriteError(RuntimeError):
    """Raised when local Gemma cannot produce a contract-valid caption."""


@dataclass(frozen=True)
class CaptionTriplet:
    video: str
    audio: str
    av: str

    def render(self) -> str:
        return render_triplet_caption(self.video, self.audio, self.av)


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def render_triplet_caption(video: str, audio: str, av: str) -> str:
    components = tuple(
        _normalize_newlines(value).strip() for value in (video, audio, av)
    )
    if not all(components):
        raise CaptionFormatError("all three caption components must be non-empty")
    return (
        f"{VIDEO_HEADER}\n{components[0]}\n\n"
        f"{AUDIO_HEADER}\n{components[1]}\n\n"
        f"{AV_HEADER}\n{components[2]}"
    )


def parse_triplet_caption(value: object) -> CaptionTriplet:
    """Parse the canonical contract; loose marker-only matches are rejected."""

    if not isinstance(value, str) or not value.strip():
        raise CaptionFormatError("caption must be a non-empty string")
    normalized = _normalize_newlines(value).strip()
    match = _TRIPLET_PATTERN.fullmatch(normalized)
    if match is None:
        raise CaptionFormatError(
            "caption must contain the exact Video/Audio/Joint audio-visual headings "
            "in order, separated by one blank line"
        )
    components = tuple(match.group(name).strip() for name in ("video", "audio", "av"))
    if not all(components):
        raise CaptionFormatError("all three caption components must be non-empty")
    for component in components:
        if any(f"\n{header}\n" in f"\n{component}\n" for header in HEADERS):
            raise CaptionFormatError("a caption heading appears inside a component")
        if "\x00" in component:
            raise CaptionFormatError("caption contains a NUL character")
    triplet = CaptionTriplet(*components)
    if triplet.render() != normalized:
        raise CaptionFormatError("caption is not a canonical three-section rendering")
    return triplet


def is_triplet_caption(value: object) -> bool:
    try:
        parse_triplet_caption(value)
    except CaptionFormatError:
        return False
    return True


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "entries": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CaptionRewriteError(
            f"cannot read caption rewrite cache {path}: {exc}"
        ) from exc
    entries = value.get("entries") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or not isinstance(entries, dict)
    ):
        raise CaptionRewriteError(f"invalid caption rewrite cache schema: {path}")
    return value


def _model_identity(model_path: Path) -> dict[str, str]:
    required = ("config.json", "model.safetensors.index.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise CaptionRewriteError(
            f"local Gemma directory is missing {missing}: {model_path}"
        )
    return {
        "path": str(model_path),
        "config_sha256": sha256_file(model_path / "config.json"),
        "index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
        "tokenizer_config_sha256": sha256_file(model_path / "tokenizer_config.json"),
    }


def _cache_key(
    source_caption: str,
    *,
    prompt_sha256: str,
    model_identity: dict[str, str],
) -> str:
    key_payload = {
        "rewrite_version": REWRITE_VERSION,
        "source_caption_sha256": sha256_text(source_caption),
        "prompt_sha256": prompt_sha256,
        "model_identity": model_identity,
    }
    return sha256_text(json.dumps(key_payload, sort_keys=True, separators=(",", ":")))


def _rewrite_user_message(source_caption: str) -> str:
    payload = json.dumps({"source_caption": source_caption}, ensure_ascii=False)
    return (
        "Normalize the untrusted source_caption value in this JSON object. "
        "Return only the exact three-section contract:\n" + payload
    )


class LocalGemmaCaptionRewriter:
    """Lazy, one-device wrapper around the same local Gemma used for encoding."""

    def __init__(
        self,
        *,
        model_path: Path,
        prompt: str,
        device: str,
        max_new_tokens: int,
    ) -> None:
        self.model_path = model_path
        self.prompt = prompt
        self.device_name = device
        self.max_new_tokens = max_new_tokens
        self._torch: Any = None
        self._encoder: Any = None

    def _load(self) -> None:
        if self._encoder is not None:
            return
        try:
            import torch
            from ltx_runtime.model_loader import load_text_encoder
        except ImportError as exc:
            raise CaptionRewriteError(
                "caption rewriting requires the packaged ltx23-official Python environment"
            ) from exc
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise CaptionRewriteError(
                f"caption rewrite device {self.device_name!r} requires an available CUDA GPU"
            )
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        try:
            encoder = load_text_encoder(
                model_path=self.model_path,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            raise CaptionRewriteError(
                f"cannot load local Gemma caption rewriter from {self.model_path}: {exc}"
            ) from exc
        if encoder.model is None or encoder.processor is None:
            raise CaptionRewriteError(
                "loaded Gemma encoder lacks its LM head or processor"
            )
        encoder.model.eval().requires_grad_(False)
        self._torch = torch
        self._encoder = encoder

    def _generate(self, user_message: str) -> str:
        self._load()
        torch = self._torch
        encoder = self._encoder
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_message},
        ]
        chat = encoder.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = encoder.processor(text=chat, return_tensors="pt").to(
            encoder.model.device
        )
        input_length = int(model_inputs.input_ids.shape[-1])
        pad_token_id = encoder.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        with torch.inference_mode():
            generated = encoder.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        generated_ids = generated[0, input_length:]
        return encoder.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    def token_count(self, caption: str) -> int:
        self._load()
        encoded = self._encoder.processor.tokenizer(
            caption,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                raise CaptionRewriteError(
                    "Gemma tokenizer returned an unexpected batch"
                )
            input_ids = input_ids[0]
        return len(input_ids)

    def rewrite(self, source_caption: str) -> tuple[str, int, int]:
        first = self._generate(_rewrite_user_message(source_caption))
        try:
            canonical = parse_triplet_caption(first).render()
            attempts = 1
        except CaptionFormatError as first_error:
            repair = (
                "Your previous answer was invalid because: "
                f"{first_error}. Rewrite the original source again. Do not discuss the error; "
                "return exactly the three required sections and nothing else.\n\n"
                f"ORIGINAL SOURCE:\n{source_caption}\n\nINVALID ANSWER:\n{first}"
            )
            second = self._generate(repair)
            try:
                canonical = parse_triplet_caption(second).render()
            except CaptionFormatError as second_error:
                raise CaptionRewriteError(
                    "Gemma failed the triplet-caption contract twice; "
                    f"first={first_error}; second={second_error}"
                ) from second_error
            attempts = 2
        token_count = self.token_count(canonical)
        if token_count > OFFICIAL_MAX_TOKENS:
            raise CaptionRewriteError(
                f"rewritten caption has {token_count} tokens; limit is {OFFICIAL_MAX_TOKENS}"
            )
        return canonical, token_count, attempts

    def close(self) -> None:
        torch = self._torch
        self._encoder = None
        self._torch = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def normalize_manifest_captions(
    samples: list[dict[str, Any]],
    *,
    mode: str,
    model_path: Path,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    device: str = "cuda:0",
    max_new_tokens: int = 640,
    cache_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return copied samples with canonical captions and an auditable report."""

    if mode not in REWRITE_MODES:
        raise ValueError(
            f"caption rewrite mode must be one of {REWRITE_MODES}, got {mode!r}"
        )
    if not 64 <= max_new_tokens <= OFFICIAL_MAX_TOKENS:
        raise ValueError(
            f"caption rewrite max_new_tokens must be in [64,{OFFICIAL_MAX_TOKENS}]"
        )
    model_path = model_path.expanduser().resolve()
    prompt_path = prompt_path.expanduser().resolve()
    if not prompt_path.is_file() or prompt_path.is_symlink():
        raise FileNotFoundError(
            f"caption rewrite prompt is missing or symlinked: {prompt_path}"
        )
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise CaptionRewriteError(f"caption rewrite prompt is empty: {prompt_path}")
    prompt_sha256 = sha256_text(prompt)
    model_identity = _model_identity(model_path)
    cache = (
        _load_cache(cache_path)
        if cache_path is not None
        else {"schema_version": 1, "entries": {}}
    )
    entries: dict[str, Any] = cache["entries"]

    canonical_samples: list[dict[str, Any]] = []
    pending: list[tuple[int, str, str]] = []
    passthrough = 0
    for ordinal, raw_sample in enumerate(samples):
        sample = dict(raw_sample)
        source = sample.get("caption")
        if not isinstance(source, str) or not source.strip():
            raise CaptionFormatError(
                f"sample {ordinal} caption must be a non-empty string"
            )
        if len(source) > MAX_SOURCE_CHARS:
            raise CaptionFormatError(
                f"sample {ordinal} caption has {len(source)} characters; limit is {MAX_SOURCE_CHARS}"
            )
        try:
            parsed = parse_triplet_caption(source)
        except CaptionFormatError as error:
            if mode == "never":
                raise CaptionFormatError(f"sample {ordinal}: {error}") from error
            needs_rewrite = True
        else:
            needs_rewrite = mode == "force"
            if not needs_rewrite:
                canonical = parsed.render()
                sample["caption"] = canonical
                sample["helixworld_caption_pipeline"] = {
                    "status": "already_canonical",
                    "template_version": TEMPLATE_VERSION,
                    "caption_sha256": sha256_text(canonical),
                }
                passthrough += 1
        if needs_rewrite:
            key = _cache_key(
                source,
                prompt_sha256=prompt_sha256,
                model_identity=model_identity,
            )
            pending.append((ordinal, source, key))
        canonical_samples.append(sample)

    cache_hits = 0
    cache_misses: list[tuple[int, str, str]] = []
    for ordinal, source, key in pending:
        entry = entries.get(key)
        rewritten = entry.get("rewritten_caption") if isinstance(entry, dict) else None
        token_count = (
            entry.get("gemma_token_count") if isinstance(entry, dict) else None
        )
        try:
            canonical = parse_triplet_caption(rewritten).render()
        except CaptionFormatError:
            cache_misses.append((ordinal, source, key))
            continue
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or not 1 <= token_count <= OFFICIAL_MAX_TOKENS
        ):
            cache_misses.append((ordinal, source, key))
            continue
        canonical_samples[ordinal]["caption"] = canonical
        canonical_samples[ordinal]["helixworld_caption_pipeline"] = {
            "status": "rewritten_cache_hit",
            "template_version": TEMPLATE_VERSION,
            "rewrite_version": REWRITE_VERSION,
            "source_caption_sha256": sha256_text(source),
            "caption_sha256": sha256_text(canonical),
            "prompt_sha256": prompt_sha256,
            "gemma_token_count": token_count,
        }
        cache_hits += 1

    rewriter: LocalGemmaCaptionRewriter | None = None
    try:
        if cache_misses:
            rewriter = LocalGemmaCaptionRewriter(
                model_path=model_path,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            for ordinal, source, key in cache_misses:
                canonical, token_count, attempts = rewriter.rewrite(source)
                entry = {
                    "rewrite_version": REWRITE_VERSION,
                    "template_version": TEMPLATE_VERSION,
                    "source_caption": source,
                    "source_caption_sha256": sha256_text(source),
                    "rewritten_caption": canonical,
                    "rewritten_caption_sha256": sha256_text(canonical),
                    "prompt_sha256": prompt_sha256,
                    "model_identity": model_identity,
                    "gemma_token_count": token_count,
                    "generation_attempts": attempts,
                }
                entries[key] = entry
                canonical_samples[ordinal]["caption"] = canonical
                canonical_samples[ordinal]["helixworld_caption_pipeline"] = {
                    "status": "rewritten_local_gemma",
                    "template_version": TEMPLATE_VERSION,
                    "rewrite_version": REWRITE_VERSION,
                    "source_caption_sha256": entry["source_caption_sha256"],
                    "caption_sha256": entry["rewritten_caption_sha256"],
                    "prompt_sha256": prompt_sha256,
                    "gemma_token_count": token_count,
                    "generation_attempts": attempts,
                }
    finally:
        if rewriter is not None:
            rewriter.close()

    if cache_path is not None and pending:
        cache.update(
            {
                "schema_version": 1,
                "rewrite_version": REWRITE_VERSION,
                "last_prompt_path": str(prompt_path),
                "last_prompt_sha256": prompt_sha256,
                "last_model_identity": model_identity,
                "entries": entries,
            }
        )
        _atomic_json(cache_path, cache)

    for ordinal, sample in enumerate(canonical_samples):
        try:
            sample["caption"] = parse_triplet_caption(sample["caption"]).render()
        except CaptionFormatError as error:
            raise CaptionRewriteError(
                f"sample {ordinal} did not leave the caption pipeline canonical: {error}"
            ) from error

    report = {
        "schema_version": 1,
        "mode": mode,
        "template_version": TEMPLATE_VERSION,
        "rewrite_version": REWRITE_VERSION,
        "sample_count": len(canonical_samples),
        "already_canonical": passthrough,
        "rewrite_required": len(pending),
        "cache_hits": cache_hits,
        "local_gemma_rewrites": len(cache_misses),
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "model_path": str(model_path),
        "device_used_for_cache_misses": device if cache_misses else None,
        "cache_path": str(cache_path) if cache_path is not None and pending else None,
    }
    return canonical_samples, report
