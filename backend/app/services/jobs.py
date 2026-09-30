"""Execucao em segundo plano. ``JobRunner`` e a porta; trocar por fila externa = novo adapter."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol


class JobRunner(Protocol):
    def submit(self, fn: Callable[[], None]) -> None: ...

    def shutdown(self) -> None: ...


class ThreadPoolJobRunner:
    def __init__(self, max_workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run")

    def submit(self, fn: Callable[[], None]) -> None:
        self._pool.submit(fn)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class InlineJobRunner:
    """Executa na hora (testes deterministas)."""

    def submit(self, fn: Callable[[], None]) -> None:
        fn()

    def shutdown(self) -> None:
        return None
