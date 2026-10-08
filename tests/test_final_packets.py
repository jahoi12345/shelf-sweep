"""Structural invariants of the final packets (skipped when runs/ is absent, e.g. a clean clone). Gate verdicts are NOT asserted here.
Reads runs/ only."""
import os
from pathlib import Path

import pytest

from tests.packet_audit import audit

ROOT = Path(os.environ.get("FINAL_PACKETS_RUNS_ROOT") or Path(__file__).resolve().parents[1] / "runs")
RUNS = [n for n in ("final_yt1_v1", "final_yt2_v1", "final_yt2_gemini_v1", "final_yt1_v1b", "final_yt2_v1b", "final_yt2_gemini_v1b")
        if (ROOT / n / "claim_packet.json").is_file()]
KNOWN = {"offline_catalogue_not_disclosed_per_line"}


@pytest.mark.skipif(not RUNS, reason="final runs absent (clean checkout)")
@pytest.mark.parametrize("name", RUNS or ["none"])
def test_final_packet_is_structurally_sound(name):
    violations = {k: v for k, v in audit(ROOT / name)["violations"].items() if k not in KNOWN}
    assert not violations, violations


@pytest.mark.skipif(not RUNS, reason="final runs absent (clean checkout)")
@pytest.mark.parametrize("name", RUNS or ["none"])
@pytest.mark.xfail(strict=True, reason="C1 open: lines whose resolution hit catalogue_offline_cache_miss are reported as 'unreadable/unresolved' and the resolve stage says ok; the offline catalogue is not disclosed")
def test_offline_catalogue_is_disclosed_on_affected_lines(name):
    assert "offline_catalogue_not_disclosed_per_line" not in audit(ROOT / name)["violations"]
