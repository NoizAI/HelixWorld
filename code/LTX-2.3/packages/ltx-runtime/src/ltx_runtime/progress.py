"""Progress display used while generating a clip."""

from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn


class SamplingContext:
    def __init__(self, progress: Progress | None, task: TaskID | None, num_steps: int):
        self._progress = progress
        self._task = task
        self._num_steps = num_steps

    def start(self) -> None:
        if self._progress is None or self._task is None:
            return
        self._progress.reset(self._task, total=self._num_steps)
        self._progress.update(self._task, completed=0, info=f"step 0/{self._num_steps}")

    def advance_step(self) -> None:
        if self._progress is None or self._task is None:
            return
        self._progress.advance(self._task)
        completed = int(self._progress.tasks[self._task].completed)
        self._progress.update(self._task, info=f"step {completed}/{self._num_steps}")

    def cleanup(self) -> None:
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, visible=False)


class InferenceProgress:
    def __init__(self, *, enabled: bool):
        self._progress = None
        if enabled:
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40, style="blue"),
                TextColumn("{task.fields[info]}", style="cyan"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
            )

    def __enter__(self) -> "InferenceProgress":
        if self._progress is not None:
            self._progress.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._progress is not None:
            self._progress.__exit__(*args)

    def start_sampling(self, num_steps: int) -> SamplingContext:
        if self._progress is None:
            return SamplingContext(None, None, num_steps)
        task = self._progress.add_task(
            "Generating",
            total=num_steps,
            completed=0,
            info=f"step 0/{num_steps}",
        )
        return SamplingContext(self._progress, task, num_steps)


__all__ = ["InferenceProgress", "SamplingContext"]
