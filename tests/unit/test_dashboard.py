from pathlib import Path

from trajsimbench.dashboard.app import main
from trajsimbench.dashboard.pages import available_pages, load_page


def test_dashboard_has_eight_import_safe_pages() -> None:
    assert len(available_pages()) == 8
    assert all(callable(load_page(name)) for name in available_pages())


def test_dashboard_without_streamlit_returns_run_index(tmp_path: Path) -> None:
    result = main(tmp_path)
    assert result is not None
    assert result["streamlit"] is False
    assert result["pages"] == list(available_pages())
