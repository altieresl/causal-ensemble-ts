"""Composicao das dependencias: o container vive em ``app.state`` (criado em ``create_app``)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from ..services.assistant import AssistantService
from ..services.atlas import AtlasService
from ..services.datasets import DatasetService
from ..services.runs import RunService


@dataclass
class Container:
    datasets: DatasetService
    runs: RunService
    atlas: AtlasService
    assistant: AssistantService
    method_weights: dict[str, float]


def get_container(request: Request) -> Container:
    return request.app.state.container
