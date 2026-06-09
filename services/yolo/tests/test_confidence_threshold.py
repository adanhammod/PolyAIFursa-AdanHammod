import importlib
import app


def test_default_confidence_threshold(monkeypatch):
    # Remove environment variable if it exists
    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)

    # Reload module so initialization code runs again
    importlib.reload(app)

    assert app.CONFIDENCE_THRESHOLD == 0.5