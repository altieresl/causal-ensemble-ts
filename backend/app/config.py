"""Configuracao por variaveis de ambiente (Twelve-Factor, item III)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    cors_origins: tuple[str, ...]
    max_upload_bytes: int
    max_workers: int
    frontend_dist: Path | None
    ollama_url: str = "http://localhost:11434"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(os.environ.get("CAUSAL_REPO_ROOT", REPO_ROOT)).resolve()
        origins = os.environ.get("CAUSAL_CORS_ORIGINS", "http://localhost:5173")
        dist = Path(os.environ.get("CAUSAL_FRONTEND_DIST", repo_root / "frontend" / "dist"))
        return cls(
            repo_root=repo_root,
            data_dir=Path(os.environ.get("CAUSAL_DATA_DIR", repo_root / "backend" / "var")).resolve(),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            max_upload_bytes=int(os.environ.get("CAUSAL_MAX_UPLOAD_MB", "20")) * 1024 * 1024,
            max_workers=max(1, int(os.environ.get("CAUSAL_MAX_WORKERS", "2"))),
            frontend_dist=dist if dist.is_dir() else None,
            ollama_url=os.environ.get("CAUSAL_OLLAMA_URL", "http://localhost:11434"),
        )
