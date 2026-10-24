"""Per-phrase processing, shared by the live program (app/main.py) and the replay tool
(evaluation/replay.py), so the replay tests exactly the live code.

phrase -> Armenian text (ASR) -> English (translation) -> speaker label -> Segment
"""
import re
import time
from dataclasses import dataclass, field

from app.audio_utils import TARGET_RATE
from app.speakers import SpeakerTracker

SHORT_RULES = ("assign", "unknown", "none")

# The ASR model outputs abbreviations such as "Հովհ." or "Հ." (letters + plain ".") for breaths,
# "hm" and similar non-speech sounds - with HIGH confidence (D29, D31). Real sentences end with the
# Armenian full stop "։". A text made ONLY of such abbreviations is treated as an unclear sound.
_ABBREVIATIONS_ONLY = re.compile(r"^(?:[\u0531-\u0587]+\.\s*)+$")


def is_abbreviation_only(text: str) -> bool:
    return bool(_ABBREVIATIONS_ONLY.match(text.strip()))


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
    # Measurement (M5): seconds spent per step ("asr", "translate", "speaker") and processing start
    # (live: wait = started_at - end_time = end-of-phrase silence + time queued behind other phrases).
    step_times: dict = field(default_factory=dict)
    started_at: float | None = None


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
        started_at = clock()
        armenian, confidence = self._asr.transcribe_with_confidence(phrase.audio)
        armenian = armenian.strip()
        steps = {"asr": clock() - started_at}
        too_unsure = (self._min_confidence is not None and confidence is not None
                      and confidence < self._min_confidence)
        if not armenian or too_unsure or is_abbreviation_only(armenian):
            # Noise, breath, "hm"...: shown as unclear, not translated, no speaker change.
            label = "Me" if phrase.source == "mic" else "(unknown)"
            return Segment(phrase.source, label, armenian, "", phrase.start_time, phrase.end_time,
                           clock(), None, confidence, unclear=True, step_times=steps,
                           started_at=started_at)
        step_start = clock()
        english = self._translator.translate(armenian)
        steps["translate"] = clock() - step_start
        if phrase.source == "mic":
            label, similarity = "Me", None
        else:
            step_start = clock()
            embedding = self._embedder.embed(phrase.audio)
            label, similarity = self.label_from_embedding(
                phrase.source, embedding, len(phrase.audio) / TARGET_RATE)
            steps["speaker"] = clock() - step_start
        return Segment(phrase.source, label, armenian, english, phrase.start_time,
                       phrase.end_time, clock(), similarity, confidence, unclear=False,
                       step_times=steps, started_at=started_at)
