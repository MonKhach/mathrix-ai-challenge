"""Phrase detection on live audio with Silero VAD (DECISIONS.md D23).

SileroModel     : 32 ms of 16 kHz audio -> probability that it contains speech.
PhraseDetector  : feeds audio frame by frame and returns finished phrases.

A phrase ends after END_SILENCE_S of silence, is cut at MAX_PHRASE_S, and is dropped if it
contains less than MIN_PHRASE_S of speech. PRE_ROLL_S of audio before the speech start is kept
so the first syllable is not lost.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.audio_utils import TARGET_RATE

FRAME = 512          # samples per VAD frame (32 ms at 16 kHz), required by Silero
CONTEXT = 64         # samples of the previous frame the model also sees, required by Silero
THRESHOLD = 0.5      # probability >= 0.5 -> speech starts
NEG_THRESHOLD = 0.35 # probability < 0.35 -> counts as silence (hysteresis, as in onnx-asr)
END_SILENCE_S = 0.6
MAX_PHRASE_S = 15.0
MIN_PHRASE_S = 0.3
PRE_ROLL_S = 0.2
TAIL_KEEP_S = 0.1    # silence kept after the last speech frame

VAD_REPO_ID = "istupakov/silero-vad-onnx"
VAD_FILE = "silero_vad.onnx"


def _frames(seconds: float) -> int:
    return max(1, round(seconds * TARGET_RATE / FRAME))


@dataclass
class Phrase:
    source: str
    audio: np.ndarray     # 16 kHz mono float32
    start_time: float     # time.monotonic() when speech started
    end_time: float       # time.monotonic() when the last speech audio arrived


class SileroModel:
    """Silero VAD ONNX model with its own memory (state); one instance per audio source."""

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(VAD_REPO_ID, VAD_FILE)  # ~2 MB, cached after the first run
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._sr = np.array([TARGET_RATE], dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame])[None, :].astype(np.float32)
        output, self._state = self._session.run(
            ["output", "stateN"], {"input": x, "state": self._state, "sr": self._sr})
        self._context = frame[-CONTEXT:]
        return float(output[0, 0])


class PhraseDetector:
    def __init__(self, source: str, speech_prob):
        """speech_prob: function(512-sample frame) -> probability (SileroModel, or a fake in tests)."""
        self.source = source
        self._speech_prob = speech_prob
        self._buffer = np.zeros(0, dtype=np.float32)
        self._pre_roll = deque(maxlen=_frames(PRE_ROLL_S))
        self._in_phrase = False
        self._last_audio_time = None

    def feed(self, samples: np.ndarray, captured_at: float) -> list[Phrase]:
        """Add new audio (captured_at = time.monotonic() when it arrived). Returns finished phrases."""
        self._last_audio_time = captured_at
        self._buffer = np.concatenate([self._buffer, samples])
        phrases = []
        while len(self._buffer) >= FRAME:
            frame, self._buffer = self._buffer[:FRAME], self._buffer[FRAME:]
            phrase = self._process(frame, captured_at)
            if phrase is not None:
                phrases.append(phrase)
        return phrases

    def on_idle(self, now: float) -> list[Phrase]:
        """Called when no audio arrives. WASAPI loopback sends nothing while nothing plays,
        so a phrase would never 'see' its ending silence; we end it after END_SILENCE_S of no audio."""
        if self._in_phrase and now - self._last_audio_time >= END_SILENCE_S:
            phrase = self._finish(trailing_silence=self._silent_frames)
            return [phrase] if phrase is not None else []
        return []

    def _process(self, frame: np.ndarray, t: float):
        prob = self._speech_prob(frame)
        if not self._in_phrase:
            self._pre_roll.append(frame)
            if prob >= THRESHOLD:
                self._start(t, list(self._pre_roll))
            return None

        self._frames.append(frame)
        if prob >= NEG_THRESHOLD:
            self._silent_frames = 0
            self._speech_frames += 1
            self._end_time = t
        else:
            self._silent_frames += 1

        if self._silent_frames >= _frames(END_SILENCE_S):
            return self._finish(trailing_silence=self._silent_frames)
        if len(self._frames) >= _frames(MAX_PHRASE_S):
            phrase = self._finish(trailing_silence=0)
            self._start(t, [])  # speaker is still talking: continue in a new phrase
            return phrase
        return None

    def _start(self, t: float, frames: list):
        self._in_phrase = True
        self._frames = frames
        self._speech_frames = 1
        self._silent_frames = 0
        self._start_time = self._end_time = t
        self._pre_roll.clear()

    def _finish(self, trailing_silence: int):
        self._in_phrase = False
        drop = max(0, trailing_silence - _frames(TAIL_KEEP_S))
        frames = self._frames[:len(self._frames) - drop]
        if self._speech_frames < _frames(MIN_PHRASE_S):
            return None  # too short: a click or cough
        return Phrase(self.source, np.concatenate(frames), self._start_time, self._end_time)
