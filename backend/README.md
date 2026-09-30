# Backend (FastAPI)

API HTTP sobre o núcleo científico (`causal_discovery/`, `causal_algorithms_atlas/`), que
permanece inalterado. Arquitetura completa: `.local/ARQUITETURA_WEB.md`.

```powershell
pip install -r requirements.txt -r backend/requirements.txt
uvicorn backend.app.main:create_app --factory --reload --port 8000   # docs: /docs
python -m pytest backend/tests -q
```

Camadas: `api/` (HTTP + schemas) → `services/` (casos de uso, pipeline, jobs) → `adapters/`
(catálogo de datasets, persistência JSON, serialização). `services/pipeline.py` é o port do
`pipeline_runner` do notebook; a seleção não consulta ground truth (só a validação pós-hoc).

Execuções são assíncronas: `POST /api/v1/runs` (pipeline), `/runs/benchmark`,
`/runs/replicated-validation`, `/runs/atlas-experiment` ou `/runs/atlas-chat` → `202` + `Location`;
consulte `GET /api/v1/runs/{id}` (`status`, `progress`) até `succeeded` e leia
`GET /api/v1/runs/{id}/result`. O atlas tem `GET /api/v1/atlas/algorithms` e `POST /api/v1/atlas/ask`.
Um novo tipo de execução se registra em `services/kinds.py` (`validate` + `execute`).

Variáveis de ambiente: `CAUSAL_DATA_DIR`, `CAUSAL_CORS_ORIGINS`, `CAUSAL_MAX_UPLOAD_MB`,
`CAUSAL_MAX_WORKERS`, `CAUSAL_FRONTEND_DIST`.
