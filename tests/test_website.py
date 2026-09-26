"""Every website section renders without an exception (Streamlit AppTest, no browser)."""
from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["Team", "Problem and Approach", "EDA of sample videos", "Results on sample videos",
            "Live Demo", "Report", "Links"]


@pytest.mark.parametrize("section", SECTIONS)
def test_section_renders(section, monkeypatch):
    monkeypatch.chdir(ROOT)
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.session_state["selected_section"] = section
    at.run()
    assert not at.exception, [e.value for e in at.exception]
