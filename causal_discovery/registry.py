from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import importlib
import inspect
import math
import pkgutil
import re
from types import MappingProxyType
from typing import Any

import pandas as pd


MethodCallable = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class CausalMethodSpec:
    """Metadata declared next to a causal-discovery method."""

    name: str
    function: MethodCallable
    signed_score: bool = False
    weight: float = 1.0
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_kwargs", MappingProxyType(dict(self.default_kwargs)))


_METHOD_REGISTRY: dict[str, CausalMethodSpec] = {}
_DISCOVERY_COMPLETE = False


def _method_name_from_function(function: MethodCallable) -> str:
    raw_name = function.__name__
    if raw_name.startswith("run_"):
        raw_name = raw_name[4:]
    parts = [part for part in re.split(r"_+", raw_name) if part]
    if not parts:
        raise ValueError("Nao foi possivel derivar o nome do metodo a partir da funcao.")
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _validate_method_signature(function: MethodCallable) -> None:
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        raise TypeError(
            f"O metodo {function.__name__!r} deve receber o DataFrame como primeiro argumento."
        )
    if "max_lag" not in signature.parameters:
        raise TypeError(
            f"O metodo {function.__name__!r} deve declarar o parametro 'max_lag'."
        )
    if signature.parameters["max_lag"].kind is inspect.Parameter.POSITIONAL_ONLY:
        raise TypeError(
            f"O parametro 'max_lag' de {function.__name__!r} deve aceitar chamada por nome."
        )


def causal_method(
    *,
    name: str | None = None,
    signed_score: bool = False,
    weight: float = 1.0,
    default_kwargs: Mapping[str, Any] | None = None,
) -> Callable[[MethodCallable], MethodCallable]:
    """Register a method when its module is discovered.

    With ``@causal_method()`` the public name is derived from ``run_<name>`` and
    all optional metadata receives safe defaults.
    """

    if (
        not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) < 0.0
    ):
        raise ValueError("O peso padrao do metodo deve ser um numero finito nao negativo.")

    def decorator(function: MethodCallable) -> MethodCallable:
        _validate_method_signature(function)
        signature = inspect.signature(function)
        configured_kwargs = dict(default_kwargs or {})
        reserved = sorted({"data", "max_lag"}.intersection(configured_kwargs))
        if reserved:
            raise ValueError(
                f"default_kwargs de {function.__name__!r} nao pode redefinir {reserved}."
            )
        accepts_extra_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        unknown_kwargs = sorted(set(configured_kwargs) - set(signature.parameters))
        if unknown_kwargs and not accepts_extra_kwargs:
            raise ValueError(
                f"default_kwargs desconhecidos em {function.__name__!r}: {unknown_kwargs}."
            )
        method_name = str(name or _method_name_from_function(function)).strip()
        if not method_name:
            raise ValueError("O nome registrado do metodo nao pode ser vazio.")
        if method_name in _METHOD_REGISTRY:
            previous = _METHOD_REGISTRY[method_name].function
            if previous is not function:
                raise ValueError(
                    f"Ja existe um metodo causal registrado com o nome {method_name!r}."
                )
            return function

        spec = CausalMethodSpec(
            name=method_name,
            function=function,
            signed_score=bool(signed_score),
            weight=float(weight),
            default_kwargs=configured_kwargs,
        )
        _METHOD_REGISTRY[method_name] = spec
        setattr(function, "__causal_method_spec__", spec)
        return function

    return decorator


def discover_causal_methods() -> dict[str, CausalMethodSpec]:
    """Import method modules once and return all decorated algorithms."""

    global _DISCOVERY_COMPLETE
    if not _DISCOVERY_COMPLETE:
        package = importlib.import_module("causal_discovery.methods")
        for module in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
            if not module.name.rsplit(".", 1)[-1].startswith("_"):
                importlib.import_module(module.name)
        _DISCOVERY_COMPLETE = True
    return dict(_METHOD_REGISTRY)


def get_registered_methods() -> dict[str, MethodCallable]:
    return {
        name: spec.function
        for name, spec in discover_causal_methods().items()
    }


def get_registered_method_kwargs(max_lag: int) -> dict[str, dict[str, Any]]:
    return {
        name: {**dict(spec.default_kwargs), "max_lag": max_lag}
        for name, spec in discover_causal_methods().items()
    }


def get_registered_method_weights() -> dict[str, float]:
    return {
        name: spec.weight
        for name, spec in discover_causal_methods().items()
    }


def get_signed_score_methods(*, discover: bool = True) -> frozenset[str]:
    specs = discover_causal_methods() if discover else dict(_METHOD_REGISTRY)
    return frozenset(
        name
        for name, spec in specs.items()
        if spec.signed_score
    )
