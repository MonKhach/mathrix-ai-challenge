"""Tests for the clustering logic in app/speakers.py with hand-made vectors (no model needed)."""
import numpy as np
import pytest

from app.speakers import SpeakerTracker, cosine_similarity

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
A_SIMILAR = np.array([0.9, 0.3, 0.0], dtype=np.float32)   # cos(A, A_SIMILAR) ~ 0.95
B = np.array([0.0, 1.0, 0.0], dtype=np.float32)           # cos(A, B) = 0
C = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def test_cosine_similarity_values():
    assert cosine_similarity(A, A) == pytest.approx(1.0)
    assert cosine_similarity(A, B) == pytest.approx(0.0)
    assert cosine_similarity(A, A * 5) == pytest.approx(1.0)  # length does not matter


def test_first_phrase_is_speaker_1():
    assert SpeakerTracker(0.6).assign(A) == ("Speaker 1", None)


def test_similar_voice_keeps_the_label():
    tracker = SpeakerTracker(0.6)
    tracker.assign(A)
    label, similarity = tracker.assign(A_SIMILAR)
    assert label == "Speaker 1"
    assert similarity > 0.9


def test_different_voice_gets_a_new_label():
    tracker = SpeakerTracker(0.6)
    tracker.assign(A)
    assert tracker.assign(B)[0] == "Speaker 2"
    assert tracker.assign(C)[0] == "Speaker 3"


def test_returning_speaker_gets_the_old_label():
    tracker = SpeakerTracker(0.6)
    for vector in (A, B, A_SIMILAR, B, A):
        tracker.assign(vector)
    assert tracker.speaker_count == 2
    assert tracker.assign(A)[0] == "Speaker 1"
    assert tracker.assign(B)[0] == "Speaker 2"


def test_threshold_decides_merge_or_split():
    strict = SpeakerTracker(0.99)
    strict.assign(A)
    assert strict.assign(A_SIMILAR)[0] == "Speaker 2"  # 0.95 < 0.99 -> split
    loose = SpeakerTracker(0.5)
    loose.assign(A)
    assert loose.assign(A_SIMILAR)[0] == "Speaker 1"   # 0.95 >= 0.5 -> same


def test_zero_vector_fails_loudly():
    with pytest.raises(ValueError):
        SpeakerTracker(0.6).assign(np.zeros(3, dtype=np.float32))


def test_closest_does_not_create_or_update_speakers():
    tracker = SpeakerTracker(0.6)
    tracker.assign(A)
    tracker.assign(B)
    assert tracker.closest(C * 0.1 + A)[0] == "Speaker 1"
    assert tracker.speaker_count == 2


def test_closest_without_known_speakers_fails_loudly():
    with pytest.raises(ValueError):
        SpeakerTracker(0.6).closest(A)
