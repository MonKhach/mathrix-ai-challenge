"""Per-phrase processing, shared by the live program (app/main.py) and the replay tool
(evaluation/replay.py), so the replay tests exactly the live code.

phrase -> Armenian text (ASR) -> English (translation) -> speaker label -> Segment
"""
import time
from dataclasses import dataclass

from app.audio_utils import TARGET_RATE
from app.speakers import SpeakerTracker

SHORT_RULES = ("assign", "unknown", "none")


@dataclass
class Segment:
    source: str
    label: str
    armenian: str
    english: str
    start_time: float
    end_time: float              # when the last speech audio arrived
    finished_at: float           # when processing finished -> latency = finished_at - end_time
    similarity: float | None     # to the chosen/closest speaker (call side only)
    confidence: float | None     # speech-recognition confidence (0-1), None if no text
    unclear: bool                # True: no text, or confidence below the cut-off -> not translated


class Pipeline:
    def __init__(self, asr, translator, embedder, speaker_threshold: float,
                 short_phrase_s: float, short_rule: str, min_confidence: float | None = None):
        if short_rule not in SHORT_RULES:
            raise ValueError(f"short_rule must be one of {SHORT_RULES}, got {short_rule!r}")
        self._asr = asr
        self._translator = translator
        self._embedder = embedder
        self._short_phrase_s = short_phrase_s
        self._short_rule = short_rule
        self._min_confidence = min_confidence  # None = no filter (DECISIONS.md D29: measure first)
        self.tracker = SpeakerTracker(speaker_threshold)

    def label_from_embedding(self, source: str, embedding, duration_s: float):
        """Speaker label for one phrase. Mic is always 'Me' (the tracker is not used)."""
        if source == "mic":
            return "Me", None
        is_short = duration_s < self._short_phrase_s
        if is_short and self._short_rule == "unknown":
            return "Speaker ?", None
        if is_short and self._short_rule == "assign" and self.tracker.speaker_count > 0:
            return self.tracker.closest(embedding)
        return self.tracker.assign(embedding)

    def process(self, phrase, clock=time.monotonic) -> Segment:
        armenian, confidence = self._asr.transcribe_with_confidence(phrase.audio)
        armenian = armenian.strip()
        too_unsure = (self._min_confidence is not None and confidence is not None
                      and confidence < self._min_confidence)
        if not armenian or too_unsure:
            # Noise, breath, "hm"...: not translated, and it may not create or change a speaker.
            label = "Me" if phrase.source == "mic" else "(unknown)"
            return Segment(phrase.source, label, armenian, "", phrase.start_time, phrase.end_time,
                           clock(), None, confidence, unclear=True)
        english = self._translator.translate(armenian)
        if phrase.source == "mic":
            label, similarity = "Me", None
        else:
            embedding = self._embedder.embed(phrase.audio)
            label, similarity = self.label_from_embedding(
                phrase.source, embedding, len(phrase.audio) / TARGET_RATE)
        return Segment(phrase.source, label, armenian, english, phrase.start_time,
                       phrase.end_time, clock(), similarity, confidence, unclear=False)
