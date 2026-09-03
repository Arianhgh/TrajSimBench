"""Dashboard entry point with a dependency-safe import contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pages import available_pages
from .services.results import list_runs, read_result_table


def main(results_root: str | Path = "results") -> dict[str, Any] | None:
    root = Path(results_root)
    try:
        import streamlit as st
    except ImportError:
        return {
            "results_root": str(root),
            "runs": list_runs(root),
            "pages": list(available_pages()),
            "streamlit": False,
        }
    st.set_page_config(page_title="TrajSimBench", layout="wide")
    st.title("TrajSimBench")
    st.caption("CPU-first reproducible trajectory similarity benchmark")
    runs = list_runs(root)
    if not runs:
        st.info("No result runs found. Run the Tiny benchmark first.")
        return None
    labels = [run["run_id"] for run in runs]
    selected = st.sidebar.selectbox("Run", labels)
    run = next(item for item in runs if item["run_id"] == selected)
    st.caption(f"Status: {run.get('status', 'unknown')} · experiment: {run.get('experiment_id')}")
    aggregate = read_result_table(Path(run["run_dir"]), "aggregate_metrics")
    if aggregate:
        st.dataframe(aggregate, use_container_width=True)
    return None


app = main


if __name__ == "__main__":
    main()
