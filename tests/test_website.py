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


def test_live_demo_runs_on_the_bundled_clip(monkeypatch):
    """Click through the demo exactly like a visitor: bundled clip -> Run -> results."""
    if not (ROOT / "samples/demo/C3896_60-95s_720p.mp4").exists():
        pytest.skip("bundled demo clip missing")
    monkeypatch.chdir(ROOT)
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    at.session_state["selected_section"] = "Live Demo"
    at.run()
    at.radio(key="demo_source").set_value(at.radio(key="demo_source").options[1]).run()
    at.button(key="demo_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    video_key, result, _, _ = at.session_state["demo_result"]
    assert video_key.startswith("sample:")
    labels = {e[2] for e in result.events}
    assert {"stop_line", "red_light"} <= labels, result.events     # same events as the full pipeline on this window
    assert len(result.risk) > 100 and all(0.0 <= r <= 1.0 for _, r in result.risk)
    assert result.clips and all(Path(c).exists() for c, _ in result.clips)
