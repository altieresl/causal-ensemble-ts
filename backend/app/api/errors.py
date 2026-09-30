"""Erros HTTP no formato ``application/problem+json`` (RFC 9457)."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..domain import DomainError

PROBLEM = "application/problem+json"


def _problem(status: int, title: str, detail: str, **extra: object) -> JSONResponse:
    body = {"type": "about:blank", "title": title, "status": status, "detail": detail, **extra}
    return JSONResponse(body, status_code=status, media_type=PROBLEM)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error(_: Request, error: DomainError) -> JSONResponse:
        return _problem(error.status, error.title, str(error))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": [str(part) for part in item["loc"]], "msg": item["msg"]}
            for item in error.errors()
        ]
        return _problem(422, "Corpo da requisicao invalido", "Falha de validacao.", errors=errors)
