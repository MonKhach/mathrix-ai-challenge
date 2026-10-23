"""Audio format helpers: everything the models see is 16 kHz, mono, float32 in [-1, 1].

Devices on this laptop deliver 48 kHz stereo 16-bit audio (DECISIONS.md D9),
so we convert once, right after capture (or after reading a .wav file).
"""
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16_000


def to_mono_16k(samples: np.ndarray, rate: int, channels: int) -> np.ndarray:
    """Convert interleaved audio to 16 kHz mono float32.

    samples:  1-D array as delivered by the device / wav file (int16 or float32),
              channels interleaved: L, R, L, R, ...
    rate:     sample rate of the input, e.g. 48000
    channels: number of interleaved channels, e.g. 2
    """
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
    mono = audio.reshape(-1, channels).mean(axis=1)

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
