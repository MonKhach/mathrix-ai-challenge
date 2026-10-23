"""Audio format helpers: everything the models see is 16 kHz, mono, float32 in [-1, 1].

Devices on this laptop deliver 48 kHz stereo 16-bit audio (DECISIONS.md D9),
so we convert once, right after capture (or after reading a .wav file).
"""
import wave
from math import gcd

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

TARGET_RATE = 16_000


def _to_float_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """int16/float32 interleaved samples -> float32 mono in [-1, 1] (channels averaged, D14)."""
    if samples.dtype == np.int16:
        audio = samples.astype(np.float32) / 32768.0  # int16 range -> [-1, 1)
    elif samples.dtype == np.float32:
        audio = samples
    else:
        raise TypeError(f"Expected int16 or float32 samples, got {samples.dtype}")

    if samples.ndim != 1 or len(audio) % channels != 0:
        raise ValueError(f"Expected 1-D interleaved audio divisible by {channels} channels, "
                         f"got shape {samples.shape}")

    # Stereo -> mono: average the channels (L, R, L, R... -> rows of [L, R] -> mean per row)
    return audio.reshape(-1, channels).mean(axis=1)


def to_mono_16k(samples: np.ndarray, rate: int, channels: int) -> np.ndarray:
    """Convert interleaved audio to 16 kHz mono float32.

    samples:  1-D array as delivered by the device / wav file (int16 or float32),
              channels interleaved: L, R, L, R, ...
    rate:     sample rate of the input, e.g. 48000
    channels: number of interleaved channels, e.g. 2
    """
    mono = _to_float_mono(samples, channels)

    if rate == TARGET_RATE:
        return mono.astype(np.float32)

    # Resample by the ratio TARGET_RATE / rate, reduced to small integers (48000 -> 16000 = 1/3).
    # resample_poly applies an anti-aliasing low-pass filter, so frequencies above 8 kHz
    # (which 16 kHz audio cannot represent) are removed instead of turning into noise.
    divisor = gcd(TARGET_RATE, rate)
    resampled = resample_poly(mono, TARGET_RATE // divisor, rate // divisor)
    return resampled.astype(np.float32)


def load_wav_16k(path) -> np.ndarray:
    """Read a 16-bit PCM .wav file and return 16 kHz mono float32 audio."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {8 * f.getsampwidth()}-bit")
        rate = f.getframerate()
        channels = f.getnchannels()
        raw = f.readframes(f.getnframes())
    return to_mono_16k(np.frombuffer(raw, dtype=np.int16), rate, channels)


class StreamResampler:
    """Live version of to_mono_16k for audio that arrives in small chunks (DECISIONS.md D23).

    Resampling each 10 ms chunk separately would restart the filter every chunk and create
    clicks at the chunk borders. This class keeps the filter memory between chunks instead.
    It uses the same filter as scipy's resample_poly (Kaiser window, 61 taps for 48 -> 16 kHz),
    so the output equals to_mono_16k's output, delayed by 10 samples (0.6 ms).
    Only integer ratios are supported (48 kHz -> 16 kHz = 3); anything else fails loudly.
    """

    def __init__(self, rate: int, channels: int):
        if rate % TARGET_RATE != 0:
            raise ValueError(f"Device rate {rate} Hz is not an integer multiple of {TARGET_RATE} Hz")
        self.channels = channels
        self.factor = rate // TARGET_RATE
        if self.factor > 1:
            # identical to resample_poly's default filter for up=1, down=factor
            self._taps = firwin(2 * 10 * self.factor + 1, 1.0 / self.factor, window=("kaiser", 5.0))
            self._memory = np.zeros(len(self._taps) - 1)  # last samples of the previous chunk
            self._offset = 0  # position of the next sample to keep inside the next chunk

    def process(self, samples: np.ndarray) -> np.ndarray:
        mono = _to_float_mono(samples, self.channels)
        if self.factor == 1:
            return mono.astype(np.float32)
        filtered, self._memory = lfilter(self._taps, 1.0, mono, zi=self._memory)
        out = filtered[self._offset::self.factor]  # keep every 3rd sample, continuing across chunks
        self._offset = (self._offset - len(mono)) % self.factor
        return out.astype(np.float32)
