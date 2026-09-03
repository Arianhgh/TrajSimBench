from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any


def export_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = sorted({key for row in rows for key in row}) or ["value"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return output.getvalue()


def render_table(
    rows: Sequence[Mapping[str, Any]], *, title: str | None = None
) -> Sequence[Mapping[str, Any]]:
    try:
        import streamlit as st
    except ImportError:
        return rows
    if title:
        st.subheader(title)
    st.dataframe(list(rows), use_container_width=True)
    return rows
