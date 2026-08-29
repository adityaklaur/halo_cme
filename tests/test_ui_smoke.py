from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_scientific_dataset_tab_loads_without_exception():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert any("TopoCross-SWIS" in title.value for title in app.title)
    assert any(tab.label == "Scientific Dataset" for tab in app.tabs)
    assert any(tab.label == "Ground Truth" for tab in app.tabs)
    assert any(tab.label == "Feature Engineering" for tab in app.tabs)
    assert any(tab.label == "OPDI Ablation" for tab in app.tabs)
