from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def select_value(label: str, values: Sequence[Any], *, default: Any = None) -> Any:
    choices = list(values)
    if not choices:
        return default
    try:
        import streamlit as st
    except ImportError:
        return default if default in choices else choices[0]
    index = choices.index(default) if default in choices else 0
    return st.selectbox(label, choices, index=index)
