"""GPU memory management utilities used by inference."""

import functools
import gc
from typing import Callable, TypeVar

import torch

from ltx_runtime import logger

F = TypeVar("F", bound=Callable)


def free_gpu_memory(log: bool = False) -> None:
    """Free GPU memory by running garbage collection and emptying CUDA cache.
    Args:
        log: If True, log memory stats after clearing
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if log:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"GPU memory freed. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


class free_gpu_memory_context:  # noqa: N801
    """Context manager and decorator to free GPU memory before and/or after execution.
    Can be used as a decorator:
        @free_gpu_memory_context(after=True)
        def my_function():
            ...
    Or as a context manager:
        with free_gpu_memory_context():
            heavy_operation()
    Args:
        before: Free memory before execution (default: False)
        after: Free memory after execution (default: True)
        log: Log memory stats when freeing (default: False)
    """

    def __init__(self, *, before: bool = False, after: bool = True, log: bool = False) -> None:
        self.before = before
        self.after = after
        self.log = log

    def __enter__(self) -> "free_gpu_memory_context":
        if self.before:
            free_gpu_memory(log=self.log)
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        if self.after:
            free_gpu_memory(log=self.log)

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> object:
            with self:
                return func(*args, **kwargs)

        return wrapper  # type: ignore
