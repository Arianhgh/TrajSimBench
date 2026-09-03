"""DuckDB query layer over authoritative Parquet artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .parquet import read_parquet


class ResultQuery:
    def __init__(self, result_root: Path) -> None:
        self.result_root = Path(result_root)
        self.connection = None
        try:
            import duckdb
        except ImportError:
            self.backend = "python"
        else:
            self.backend = "duckdb"
            self.connection = duckdb.connect(database=":memory:")
            self._create_duckdb_views()

    def _create_duckdb_views(self) -> None:
        assert self.connection is not None
        for table_path in sorted(self.result_root.rglob("*.parquet")):
            name = table_path.stem
            if name in {"queries", "triplets", "variants"} and table_path.parent.name == "tasks":
                view_name = name
            else:
                view_name = name
            escaped = str(table_path).replace("'", "''")
            try:
                self.connection.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped}')"
                )
            except Exception:
                continue

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        if self.backend == "duckdb":
            assert self.connection is not None
            result = self.connection.execute(sql, parameters)
            columns = [description[0] for description in result.description]
            return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
        raise RuntimeError("DuckDB is not installed; use table() for the portable reader")

    def table(self, name: str) -> list[dict[str, Any]]:
        candidates = list(self.result_root.rglob(f"{name}.parquet"))
        rows: list[dict[str, Any]] = []
        for path in candidates:
            rows.extend(read_parquet(path))
        return rows


def connect_result_root(result_root: Path) -> ResultQuery:
    return ResultQuery(result_root)


def create_duckdb_views(result_root: Path) -> ResultQuery:
    return connect_result_root(result_root)


query_results = connect_result_root
