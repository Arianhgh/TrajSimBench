"""Importable page contracts for the eight dashboard views."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Keep the dashboard surface import-safe: page modules expose pure ``render``
# functions and Streamlit is only imported by the app entry point.
PAGE_REGISTRY: dict[str, str] = {
    "dataset": "trajsimbench.dashboard.pages.dataset",
    "pair": "trajsimbench.dashboard.pages.pair",
    "counterfactual": "trajsimbench.dashboard.pages.counterfactual",
    "disagreement": "trajsimbench.dashboard.pages.disagreement",
    "fingerprints": "trajsimbench.dashboard.pages.fingerprints",
    "robustness": "trajsimbench.dashboard.pages.robustness",
    "efficiency": "trajsimbench.dashboard.pages.efficiency",
    "builder": "trajsimbench.dashboard.pages.builder",
}


def available_pages() -> tuple[str, ...]:
    """Return the stable page order used by the dashboard and smoke tests."""

    return tuple(PAGE_REGISTRY)


def load_page(name: str) -> Callable[..., dict[str, Any]]:
    """Load one pure page renderer on demand, without requiring Streamlit."""

    if name not in PAGE_REGISTRY:
        raise KeyError(f"unknown dashboard page {name!r}")
    from importlib import import_module

    module = import_module(PAGE_REGISTRY[name])
    render = getattr(module, "render", None)
    if not callable(render):
        raise TypeError(f"dashboard page {name!r} does not expose a callable render")
    return render
