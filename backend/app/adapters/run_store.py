"""Repositorio de execucoes em arquivos JSON (um arquivo por run)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from ..domain import NotFoundError, Run, RunStatus, is_safe_id, utc_now


class FileRunRepository:
    def __init__(self, runs_dir: Path) -> None:
        self._dir = runs_dir
        self._lock = threading.RLock()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fail_interrupted()

    def _path(self, run_id: str) -> Path:
        if not is_safe_id(run_id):
            raise NotFoundError(f"Execucao '{run_id}' nao existe.")
        return self._dir / f"{run_id}.json"

    def save(self, run: Run) -> None:
        path = self._path(run.id)
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(json.dumps(run.to_dict(), ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)  # escrita atomica: nunca deixa JSON pela metade

    def get(self, run_id: str) -> Run:
        path = self._path(run_id)
        with self._lock:
            if not path.is_file():
                raise NotFoundError(f"Execucao '{run_id}' nao existe.")
            return Run.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[Run]:
        with self._lock:
            runs = [Run.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in self._dir.glob("*.json")]
        return sorted(runs, key=lambda run: run.created_at, reverse=True)

    def delete(self, run_id: str) -> None:
        path = self._path(run_id)
        with self._lock:
            if not path.is_file():
                raise NotFoundError(f"Execucao '{run_id}' nao existe.")
            path.unlink()

    def _fail_interrupted(self) -> None:
        """Runs que estavam em andamento quando o processo caiu nunca terminarao."""
        for run in self.list():
            if run.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
                run.status = RunStatus.FAILED
                run.error = "Execucao interrompida por reinicio do servidor."
                run.finished_at = utc_now()
                self.save(run)
