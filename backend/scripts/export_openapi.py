"""Exporta o OpenAPI da API para `frontend/openapi.json` (fonte dos tipos TS gerados).

Uso (raiz do repo): python -m backend.scripts.export_openapi [--check]
``--check`` falha se o arquivo versionado divergir do contrato atual.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.jobs import InlineJobRunner

TARGET = Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"


def build_spec() -> str:
    base = Settings.from_env()
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            repo_root=base.repo_root,
            data_dir=Path(tmp),
            cors_origins=(),
            max_upload_bytes=base.max_upload_bytes,
            max_workers=1,
            frontend_dist=None,
        )
        app = create_app(settings, jobs=InlineJobRunner(), method_weights={})
        return json.dumps(app.openapi(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    spec = build_spec()
    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current.replace("\r\n", "\n") != spec:
            print("frontend/openapi.json desatualizado: rode `npm run generate:api` em frontend/.")
            return 1
        return 0
    TARGET.write_text(spec, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
