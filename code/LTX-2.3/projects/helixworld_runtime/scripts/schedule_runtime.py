"""Validated schedule helpers for the HelixWorld release runtime."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import pairwise

RELEASE_NODES = (1.0, 0.9, 0.7, 0.4, 0.0)
SCHEDULE_TYPE = "release_schedule_v1"


def validate_nodes(values: Iterable[float], *, steps: int = 4) -> tuple[float, ...]:
    nodes = tuple(values)
    if len(nodes) != steps + 1:
        raise ValueError(f"expected {steps + 1} schedule nodes, got {len(nodes)}")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in nodes
    ):
        raise ValueError("schedule nodes must be numeric")
    resolved = tuple(float(value) for value in nodes)
    if not all(math.isfinite(value) for value in resolved):
        raise ValueError("schedule nodes must be finite")
    if abs(resolved[0] - 1.0) > 5.0e-7 or resolved[-1] != 0.0:
        raise ValueError("schedule endpoints are invalid")
    if any(left <= right for left, right in pairwise(resolved)):
        raise ValueError("schedule nodes must be strictly descending")
    return resolved


def release_schedule_spec() -> dict[str, object]:
    return {
        "type": SCHEDULE_TYPE,
        "steps": 4,
        "nodes": list(validate_nodes(RELEASE_NODES)),
    }


def nodes_from_spec(schedule: object) -> tuple[float, ...]:
    if not isinstance(schedule, dict) or schedule.get("type") != SCHEDULE_TYPE:
        raise ValueError("unsupported schedule contract")
    steps = schedule.get("steps")
    nodes = schedule.get("nodes")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("schedule steps must be a positive integer")
    if not isinstance(nodes, list):
        raise ValueError("schedule nodes must be a list")
    return validate_nodes(nodes, steps=steps)


__all__ = [
    "RELEASE_NODES",
    "nodes_from_spec",
    "release_schedule_spec",
    "validate_nodes",
]
