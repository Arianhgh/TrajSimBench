"""Reproducible result summaries and portable figure specifications."""

from .figures import generate_figures
from .tables import generate_tables, load_result_rows

__all__ = ["generate_figures", "generate_tables", "load_result_rows"]
