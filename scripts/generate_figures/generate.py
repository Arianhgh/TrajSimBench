"""Regenerate saved analysis artifacts without benchmark recomputation."""

from __future__ import annotations

import argparse

from trajsimbench.analysis.figures import generate_figures
from trajsimbench.analysis.tables import generate_tables


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="analysis")
    args = parser.parse_args()
    tables = generate_tables(args.results_root, f"{args.output}/tables")
    figures = generate_figures(args.results_root, f"{args.output}/figures")
    print(
        {
            "tables": {key: str(value) for key, value in tables.items()},
            "figures": {key: str(value) for key, value in figures.items()},
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
