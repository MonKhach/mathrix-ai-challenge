"""segment_timing in app/main.py: what --debug shows about where the time went (M5)."""
from app.main import segment_timing
from app.pipeline import Segment


def make_segment():
    return Segment("system", "Speaker 1", "բարև", "hello", start_time=10.0, end_time=14.0,
                   finished_at=17.0, similarity=0.9, confidence=0.9, unclear=False,
                   step_times={"asr": 0.3, "translate": 1.2, "speaker": 0.1}, started_at=15.5)


def test_live_timing_has_phrase_wait_and_steps_in_order():
    timing = segment_timing(make_segment(), latency=True)
    assert list(timing) == ["phrase", "wait", "asr", "translate", "speaker"]
    assert timing["phrase"] == 4.0 and timing["wait"] == 1.5


def test_replay_timing_has_no_wait():
    assert "wait" not in segment_timing(make_segment(), latency=False)
