"""Tests for the phrase logic in app/vad.py, with a fake speech detector (no model needed):
frames full of 1.0 count as speech, frames full of 0.0 as silence."""
import numpy as np

from app.vad import FRAME, PhraseDetector

FRAME_S = FRAME / 16_000  # 0.032 s


def fake_prob(frame):
    return 1.0 if frame.mean() > 0.5 else 0.0


def feed(detector, pattern, t0=0.0):
    """pattern: list of (value, seconds). Feeds frame by frame; returns phrases and end time."""
    phrases, t = [], t0
    for value, seconds in pattern:
        for _ in range(round(seconds / FRAME_S)):
            t += FRAME_S
            phrases += detector.feed(np.full(FRAME, value, dtype=np.float32), t)
    return phrases, t


def test_phrase_ends_after_0_6_s_of_silence():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(0, 0.5), (1, 1.0), (0, 0.7)])
    assert len(phrases) == 1
    duration = len(phrases[0].audio) / 16_000
    assert 1.0 <= duration <= 1.0 + 0.2 + 0.15  # speech + pre-roll + kept tail


def test_short_pause_does_not_end_the_phrase():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(1, 1.0), (0, 0.4), (1, 1.0), (0, 0.4)])
    assert phrases == []


def test_click_shorter_than_0_3_s_is_ignored():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(1, 0.1), (0, 1.0)])
    assert phrases == []


def test_long_speech_without_pauses_is_cut_at_10_s_and_continues():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(1, 15.0), (0, 1.0)])
    assert len(phrases) == 2
    assert abs(len(phrases[0].audio) / 16_000 - 10.0) < 0.1


def test_pre_roll_keeps_audio_before_speech_start():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(0, 1.0), (1, 1.0), (0, 1.0)])
    audio = phrases[0].audio
    first_speech = np.argmax(audio > 0.5)
    assert 0.1 <= first_speech / 16_000 <= 0.25  # ~0.2 s of silence kept before speech


def test_end_time_is_the_last_speech_audio_not_the_detection_time():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(1, 1.0), (0, 1.0)])
    assert abs(phrases[0].end_time - 1.0) < 2 * FRAME_S


def test_idle_source_ends_the_phrase():
    detector = PhraseDetector("system", fake_prob)
    phrases, t = feed(detector, [(1, 1.0)])  # loopback stops sending audio right after speech
    assert phrases == []
    assert detector.on_idle(t + 0.3) == []
    assert len(detector.on_idle(t + 0.7)) == 1


def test_soft_end_short_pause_ends_a_long_phrase():
    detector = PhraseDetector("mic", fake_prob, soft_after_s=6.0, soft_silence_s=0.25)
    phrases, _ = feed(detector, [(1, 7.0), (0, 0.3)])
    assert len(phrases) == 1
    assert 7.0 <= len(phrases[0].audio) / 16_000 <= 7.0 + 0.2 + 0.15


def test_soft_end_short_pause_does_not_end_a_short_phrase():
    detector = PhraseDetector("mic", fake_prob, soft_after_s=6.0, soft_silence_s=0.25)
    phrases, _ = feed(detector, [(1, 3.0), (0, 0.3), (1, 1.0)])
    assert phrases == []


def test_default_a_0_15_s_pause_ends_a_phrase_longer_than_6_s():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(1, 7.0), (0, 0.2), (1, 1.0)])
    assert len(phrases) == 1


def test_default_a_0_15_s_pause_does_not_end_a_short_phrase():
    phrases, _ = feed(PhraseDetector("mic", fake_prob), [(1, 3.0), (0, 0.2), (1, 1.0)])
    assert phrases == []


def test_soft_end_can_be_turned_off():
    detector = PhraseDetector("mic", fake_prob, max_phrase_s=15.0, soft_after_s=None)
    phrases, _ = feed(detector, [(1, 7.0), (0, 0.3), (1, 1.0)])
    assert phrases == []


def test_max_phrase_can_be_set():
    phrases, _ = feed(PhraseDetector("mic", fake_prob, max_phrase_s=15.0), [(1, 17.0), (0, 1.0)])
    assert abs(len(phrases[0].audio) / 16_000 - 15.0) < 0.1


def test_idle_source_uses_the_shorter_pause_for_a_long_phrase():
    detector = PhraseDetector("system", fake_prob, soft_after_s=6.0, soft_silence_s=0.25)
    phrases, t = feed(detector, [(1, 7.0)])
    assert detector.on_idle(t + 0.1) == []
    assert len(detector.on_idle(t + 0.3)) == 1


def test_invalid_soft_settings_fail_loudly():
    import pytest
    with pytest.raises(ValueError):
        PhraseDetector("mic", fake_prob, soft_after_s=6.0, soft_silence_s=0.8)
    with pytest.raises(ValueError):
        PhraseDetector("mic", fake_prob, max_phrase_s=5.0, soft_after_s=6.0)
