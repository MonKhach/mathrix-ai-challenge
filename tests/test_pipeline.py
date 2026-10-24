"""Tests for app/pipeline.py with fake models (no downloads needed)."""
import numpy as np
import pytest

from app.pipeline import Pipeline, is_abbreviation_only
from app.vad import Phrase

VOICE_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
VOICE_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class FakeASR:
    def transcribe_with_confidence(self, audio):
        if audio[0] == 9:
            return "", None                # first sample 9 -> nothing recognised
        if audio[0] == 7:
            return "Բարև", 0.2             # first sample 7 -> low-confidence phrase
        if audio[0] == 5:
            return "Հովհ.", 0.95           # first sample 5 -> confident abbreviation artifact
        return "բարև", 0.9


class FakeTranslator:
    def translate(self, text):
        return "hello"


class FakeEmbedder:
    def embed(self, audio):
        return VOICE_A if audio[0] == 1 else VOICE_B   # first sample selects the voice


def phrase(source, voice, seconds):
    audio = np.zeros(int(seconds * 16_000), dtype=np.float32)
    audio[0] = voice
    return Phrase(source, audio, start_time=0.0, end_time=1.0)


def pipeline(rule="assign"):
    return Pipeline(FakeASR(), FakeTranslator(), FakeEmbedder(), 0.4, 1.5, rule)


def test_mic_is_always_me_and_never_touches_the_tracker():
    p = pipeline()
    segment = p.process(phrase("mic", 1, 3.0))
    assert (segment.label, segment.armenian, segment.english) == ("Me", "բարև", "hello")
    assert p.tracker.speaker_count == 0


def test_two_voices_on_the_call_side_get_two_labels():
    p = pipeline()
    assert p.process(phrase("system", 1, 3.0)).label == "Speaker 1"
    assert p.process(phrase("system", 2, 3.0)).label == "Speaker 2"
    assert p.process(phrase("system", 1, 3.0)).label == "Speaker 1"


def test_rule_assign_short_phrase_gets_closest_label_without_new_speaker():
    p = pipeline("assign")
    p.process(phrase("system", 1, 3.0))
    assert p.process(phrase("system", 2, 1.0)).label == "Speaker 1"   # short: no new speaker
    assert p.tracker.speaker_count == 1


def test_rule_assign_first_ever_phrase_may_be_short():
    p = pipeline("assign")
    assert p.process(phrase("system", 1, 1.0)).label == "Speaker 1"


def test_rule_unknown_short_phrase_is_speaker_question_mark():
    p = pipeline("unknown")
    p.process(phrase("system", 1, 3.0))
    assert p.process(phrase("system", 1, 1.0)).label == "Speaker ?"


def test_rule_none_short_phrase_can_create_a_speaker():
    p = pipeline("none")
    p.process(phrase("system", 1, 3.0))
    assert p.process(phrase("system", 2, 1.0)).label == "Speaker 2"


def test_no_text_recognised_does_not_create_a_speaker():
    p = pipeline()
    segment = p.process(phrase("system", 9, 3.0))
    assert segment.label == "(unknown)" and segment.armenian == "" and segment.unclear
    assert p.tracker.speaker_count == 0


def test_confidence_filter_is_off_by_default():
    segment = pipeline().process(phrase("system", 7, 3.0))
    assert not segment.unclear and segment.english == "hello" and segment.confidence == 0.2


def test_low_confidence_phrase_is_unclear_not_translated_and_no_speaker():
    p = Pipeline(FakeASR(), FakeTranslator(), FakeEmbedder(), 0.4, 1.5, "assign",
                 min_confidence=0.5)
    segment = p.process(phrase("system", 7, 3.0))
    assert segment.unclear and segment.english == "" and segment.armenian == "Բարև"
    assert p.tracker.speaker_count == 0
    assert not p.process(phrase("system", 1, 3.0)).unclear   # confidence 0.9 passes


def test_unknown_rule_name_fails_loudly():
    with pytest.raises(ValueError):
        Pipeline(FakeASR(), FakeTranslator(), FakeEmbedder(), 0.4, 1.5, "maybe")


def test_abbreviation_detection():
    assert is_abbreviation_only("Հովհ.")
    assert is_abbreviation_only("Հ.")
    assert is_abbreviation_only(" Հովհ. Հ. ")
    assert not is_abbreviation_only("Այո։")                 # Armenian full stop: real speech
    assert not is_abbreviation_only("Ես չեմ սիրում։")
    assert not is_abbreviation_only("Հովհ. Թումանյանը")      # abbreviation + real word
    assert not is_abbreviation_only("Այո")


def test_abbreviation_artifact_is_unclear_even_with_high_confidence():
    p = pipeline()
    segment = p.process(phrase("system", 5, 1.0))
    assert segment.unclear and segment.english == "" and segment.armenian == "Հովհ."
    assert p.tracker.speaker_count == 0


def test_abbreviation_artifact_on_mic_stays_me_but_unclear():
    segment = pipeline().process(phrase("mic", 5, 1.0))
    assert segment.unclear and segment.label == "Me"
