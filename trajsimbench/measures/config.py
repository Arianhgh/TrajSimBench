"""Strict, serializable configuration models for the seven classical methods.

Pydantic is intentionally not a hard dependency of the CPU measure layer.  A
small BaseModel-compatible validator is provided here so the package can be
used in the foundation bootstrap environment, while still exposing the
``model_validate``/``model_dump`` vocabulary used by the later YAML config
loader.  Unknown keys are rejected instead of silently ignored.
"""

# The conditional Pydantic base classes intentionally share a name; mypy sees
# the runtime-selected branch as three definitions.
# mypy: disable-error-code="misc,no-redef"

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

try:  # Pydantic is part of the full foundation environment, not the CPU shim.
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import ConfigDict as _PydanticConfigDict

    _PYDANTIC_V2 = hasattr(_PydanticBaseModel, "model_validate")
except ImportError:  # pragma: no cover - exercised by the minimal bootstrap runtime
    _PydanticBaseModel = None  # type: ignore[assignment]
    _PydanticConfigDict = None  # type: ignore[assignment]
    _PYDANTIC_V2 = False


class ConfigValidationError(ValueError):
    """Raised for unknown or invalid method configuration fields."""


def _all_annotations(cls: type) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for parent in reversed(cls.__mro__):
        annotations.update(getattr(parent, "__annotations__", {}))
    return {
        name: value
        for name, value in annotations.items()
        if not name.startswith("_") and name not in {"model_config"}
    }


class _ConfigBehavior:
    """Shared strict validation behavior for Pydantic and bootstrap runtimes."""

    _aliases: ClassVar[dict[str, str]] = {}

    def __init__(self, **values: Any) -> None:
        fields = _all_annotations(type(self))
        normalized: dict[str, Any] = {}
        for key, value in values.items():
            canonical = type(self)._aliases.get(key, key)
            if canonical in normalized:
                raise ConfigValidationError(f"{type(self).__name__}: duplicate field {canonical!r}")
            if canonical not in fields:
                allowed = ", ".join(sorted(fields)) or "(none)"
                raise ConfigValidationError(
                    f"{type(self).__name__}: unknown field {key!r}; allowed fields: {allowed}"
                )
            normalized[canonical] = value

        normalized_values: dict[str, Any] = {}
        for name in fields:
            if name in normalized:
                value = normalized[name]
            elif hasattr(type(self), name):
                value = getattr(type(self), name)
            else:
                field = getattr(type(self), "model_fields", {}).get(name)
                if field is None:
                    field = getattr(type(self), "__fields__", {}).get(name)
                is_required = getattr(field, "is_required", None)
                if callable(is_required):
                    is_required = is_required()
                else:
                    is_required = bool(getattr(field, "required", True))
                if field is None or is_required:
                    raise ConfigValidationError(
                        f"{type(self).__name__}: missing required field {name!r}"
                    )
                value = field.default
            normalized_values[name] = self._validate_field(name, value)
            # Make earlier validated fields visible to dependent validators
            # (LCSS delta validation depends on delta_mode). Pydantic's own
            # initializer runs immediately afterwards and revalidates fields.
            object.__setattr__(self, name, normalized_values[name])
        if _PydanticBaseModel is not None:
            super().__init__(**normalized_values)
        else:
            for name, value in normalized_values.items():
                object.__setattr__(self, name, value)

    @classmethod
    def model_validate(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        if hasattr(value, "model_dump"):
            return cls(**value.model_dump())
        if hasattr(value, "dict"):
            return cls(**value.dict())
        raise TypeError(f"{cls.__name__}.model_validate expects a mapping or config instance")

    @classmethod
    def fields(cls) -> tuple[str, ...]:
        return tuple(_all_annotations(cls))

    def model_dump(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _all_annotations(type(self))}

    def dict(self, **_: Any) -> dict[str, Any]:
        return self.model_dump()

    def model_copy(self, *, update: Mapping[str, Any] | None = None, **_: Any) -> Any:
        values = self.model_dump()
        values.update(dict(update or {}))
        return type(self)(**values)

    def _validate_field(self, name: str, value: Any) -> Any:
        del name
        return value

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
        return f"{type(self).__name__}({args})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _ConfigBehavior)
            and type(self) is type(other)
            and self.model_dump() == other.model_dump()
        )


if _PydanticBaseModel is not None and _PYDANTIC_V2:

    class BaseMethodConfig(_ConfigBehavior, _PydanticBaseModel):
        """Strict Pydantic v2 model used when the foundation dependencies exist."""

        model_config = _PydanticConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

elif _PydanticBaseModel is not None:

    class BaseMethodConfig(_ConfigBehavior, _PydanticBaseModel):
        """Strict Pydantic v1 model used when available."""

        class Config:
            extra = "forbid"
            validate_assignment = True
            arbitrary_types_allowed = True

else:

    class BaseMethodConfig(_ConfigBehavior):
        """Dependency-free strict model for the minimal CPU bootstrap runtime."""


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigValidationError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ConfigValidationError(f"{name} must be >= {minimum}")
    return result


def _strict_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfigValidationError(f"{name} must be a real number")
    result = float(value)
    if math.isnan(result):
        raise ConfigValidationError(f"{name} must not be NaN")
    if minimum is not None and result < minimum:
        raise ConfigValidationError(f"{name} must be >= {minimum}")
    return result


def _strict_choice(value: Any, *, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigValidationError(f"{name} must be one of {choices!r}")
    return value


class EuclideanConfig(BaseMethodConfig):
    """Common arc-length resampling count for pointwise Euclidean distance."""

    n_samples: int = 100
    _aliases: ClassVar[dict[str, str]] = {
        "sampling_count": "n_samples",
        "resample_count": "n_samples",
        "num_samples": "n_samples",
    }

    def _validate_field(self, name: str, value: Any) -> Any:
        if name == "n_samples":
            return _strict_int(value, name=name, minimum=1)
        return super()._validate_field(name, value)

    @property
    def sampling_count(self) -> int:
        return self.n_samples

    @property
    def resample_count(self) -> int:
        return self.n_samples


class DTWConfig(BaseMethodConfig):
    """DTW normalization and optional inclusive Sakoe--Chiba window."""

    normalization: str = "none"
    window: int | None = None
    _aliases: ClassVar[dict[str, str]] = {
        "global_normalization": "normalization",
        "window_size": "window",
        "sakoe_chiba_window": "window",
    }

    def _validate_field(self, name: str, value: Any) -> Any:
        if name == "normalization":
            return _strict_choice(
                value,
                name=name,
                choices=("none", "path_length", "max_input_length"),
            )
        if name == "window":
            if value is None:
                return None
            return _strict_int(value, name=name, minimum=0)
        return super()._validate_field(name, value)

    @property
    def global_normalization(self) -> str:
        return self.normalization

    @property
    def window_size(self) -> int | None:
        return self.window


class HausdorffConfig(BaseMethodConfig):
    """Hausdorff has no tunable parameters in the v1 contract."""


class DiscreteFrechetConfig(BaseMethodConfig):
    """Discrete Fréchet has no tunable parameters in the v1 contract."""


class LCSSConfig(BaseMethodConfig):
    """LCSS spatial threshold with optional index and timestamp constraints."""

    epsilon: float = 1.0
    delta_mode: str = "index"
    delta: int | float | None = None
    time_delta_s: float | None = None
    _aliases: ClassVar[dict[str, str]] = {"time_delta": "time_delta_s", "delta_s": "time_delta_s"}

    def _validate_field(self, name: str, value: Any) -> Any:
        if name == "epsilon":
            return _strict_float(value, name=name, minimum=0.0)
        if name == "delta":
            if value is None:
                return None
            if getattr(self, "delta_mode", "index") == "time":
                return _strict_float(value, name=name, minimum=0.0)
            return _strict_int(value, name=name, minimum=0)
        if name == "delta_mode":
            return _strict_choice(value, name=name, choices=("index", "time"))
        if name == "time_delta_s":
            if value is None:
                return None
            return _strict_float(value, name=name, minimum=0.0)
        return super()._validate_field(name, value)


class EDRConfig(BaseMethodConfig):
    """EDR substitution threshold."""

    epsilon: float = 1.0

    def _validate_field(self, name: str, value: Any) -> Any:
        if name == "epsilon":
            return _strict_float(value, name=name, minimum=0.0)
        return super()._validate_field(name, value)


class ERPConfig(BaseMethodConfig):
    """ERP gap vector and explicit optional normalization policy."""

    gap_point: tuple[float, float] = (0.0, 0.0)
    normalization: str = "none"
    _aliases: ClassVar[dict[str, str]] = {"gap": "gap_point"}

    def __init__(self, **values: Any) -> None:
        if "normalize" in values:
            normalize = values.pop("normalize")
            if not isinstance(normalize, bool):
                raise ConfigValidationError("normalize must be a boolean")
            if "normalization" in values:
                raise ConfigValidationError("provide either normalize or normalization, not both")
            values["normalization"] = "max_input_length" if normalize else "none"
        super().__init__(**values)

    def _validate_field(self, name: str, value: Any) -> Any:
        if name == "gap_point":
            if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
                raise ConfigValidationError("gap_point must contain exactly two coordinates")
            result = tuple(_strict_float(item, name="gap_point coordinate") for item in value)
            if not all(math.isfinite(item) for item in result):
                raise ConfigValidationError("gap_point coordinates must be finite")
            return result
        if name == "normalization":
            return _strict_choice(value, name=name, choices=("none", "max_input_length"))
        return super()._validate_field(name, value)

    @property
    def normalize(self) -> bool:
        return self.normalization != "none"


CONFIG_MODELS: dict[str, type[BaseMethodConfig]] = {
    "euclidean": EuclideanConfig,
    "dtw": DTWConfig,
    "hausdorff": HausdorffConfig,
    "discrete_frechet": DiscreteFrechetConfig,
    "lcss": LCSSConfig,
    "edr": EDRConfig,
    "erp": ERPConfig,
}

# Explicit aliases keep config naming ergonomic for callers while the registry
# continues to expose one canonical model per stable method name.
MethodConfig = BaseMethodConfig
EuclideanMethodConfig = EuclideanConfig
DTWMethodConfig = DTWConfig
HausdorffMethodConfig = HausdorffConfig
DiscreteFrechetMethodConfig = DiscreteFrechetConfig
LCSSMethodConfig = LCSSConfig
EDRMethodConfig = EDRConfig
ERPMethodConfig = ERPConfig


__all__ = [
    "BaseMethodConfig",
    "ConfigValidationError",
    "CONFIG_MODELS",
    "DiscreteFrechetConfig",
    "DTWConfig",
    "EDRConfig",
    "ERPConfig",
    "EuclideanConfig",
    "HausdorffConfig",
    "LCSSConfig",
    "MethodConfig",
    "EuclideanMethodConfig",
    "DTWMethodConfig",
    "HausdorffMethodConfig",
    "DiscreteFrechetMethodConfig",
    "LCSSMethodConfig",
    "EDRMethodConfig",
    "ERPMethodConfig",
]
