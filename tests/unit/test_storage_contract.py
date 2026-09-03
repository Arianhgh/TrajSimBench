from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trajsimbench.storage.parquet import read_parquet, write_parquet
from trajsimbench.storage.schemas import validate_rows


class StorageTests(unittest.TestCase):
    def test_roundtrip_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(
                Path(directory) / "rankings.parquet",
                [
                    {
                        "query_id": "q",
                        "candidate_id": "c",
                        "rank": 1,
                        "distance": 0.0,
                        "raw_score": 0.0,
                    }
                ],
                table="rankings",
            )
            self.assertEqual(read_parquet(path)[0]["schema_version"], "1.0")
            self.assertTrue(validate_rows(read_parquet(path), table="rankings")["valid"])

    def test_unknown_schema_column_is_actionable(self) -> None:
        report = validate_rows([{"schema_version": "9.0", "query_id": "q"}], table="queries")
        self.assertFalse(report["valid"])
        self.assertTrue(any("schema_version" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
