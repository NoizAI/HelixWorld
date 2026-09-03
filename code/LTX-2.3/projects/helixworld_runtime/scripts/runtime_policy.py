"""Bounded context policy used by the release runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    contract: str
    history_limit: int
    fixed_prefix: int
    refresh_mode: str
    selection: str

    def metadata(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["RuntimePolicy"]
