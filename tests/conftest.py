"""Test isolation: never read or write the real Disconnect-Gemini switch state (runs/_server_state/hosted_gate.json)."""
import os
import pytest

os.environ.setdefault("LIBRARY_POST_SWEEP_BUILD", "0")   # tests never start real post-sweep 2D/3D builds (tests/test_post_sweep.py enables a fake one)
# Legacy synthetic scenes place lone boxes that are not shelf rows; the non-book scene gate is tested explicitly in tests/test_non_book_gate.py.
os.environ.setdefault("LIBRARY_NON_BOOK_GATE", "0")


@pytest.fixture(autouse=True)
def _isolated_hosted_gate(tmp_path, monkeypatch):
    try:
        from providers import hosted_gate
    except Exception:  # module absent in partial checkouts
        yield
        return
    monkeypatch.setattr(hosted_gate, "STATE_PATH", tmp_path / "hosted_gate.json", raising=False)
    yield
