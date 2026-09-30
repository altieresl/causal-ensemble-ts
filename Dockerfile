# Imagem unica: o FastAPI serve a API e o build do React (frontend/dist).
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependencias primeiro (camada em cache). torch CPU: a imagem nao precisa de CUDA.
COPY requirements.txt core-requirements.txt
COPY backend/requirements.txt api-requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r core-requirements.txt -r api-requirements.txt

COPY causal_discovery/ causal_discovery/
COPY causal_algorithms_atlas/ causal_algorithms_atlas/
COPY backend/ backend/
COPY datasets/ datasets/
COPY DailyDelhiClimateTrain.csv ./
COPY --from=frontend /app/frontend/dist frontend/dist

ENV CAUSAL_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
