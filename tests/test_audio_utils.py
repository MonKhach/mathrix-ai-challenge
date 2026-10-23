"""Tests for app/audio_utils.py (format conversion to 16 kHz mono)."""
import wave

import numpy as np
import pytest

from app.audio_utils import TARGET_RATE, StreamResampler, load_wav_16k, to_mono_16k


def sine(freq, seconds, rate, amplitude=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def interleave(left, right):
    return np.stack([left, right], axis=1).reshape(-1)


def dominant_frequency(audio, rate):
    spectrum = np.abs(np.fft.rfft(audio))
    return np.fft.rfftfreq(len(audio), 1 / rate)[np.argmax(spectrum)]


def test_one_second_48k_stereo_becomes_16000_mono_samples():
    tone = sine(440, 1.0, 48_000)
    out = to_mono_16k(interleave(tone, tone), 48_000, 2)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert len(out) == 16_000


def test_speech_band_frequency_is_preserved():
    tone = sine(440, 1.0, 48_000)
    out = to_mono_16k(interleave(tone, tone), 48_000, 2)
    assert dominant_frequency(out, TARGET_RATE) == pytest.approx(440, abs=2)


def test_frequency_above_8khz_is_filtered_not_aliased():
    # 10 kHz cannot exist in 16 kHz audio; without a filter it would fold back to 6 kHz noise
    tone = sine(10_000, 1.0, 48_000)
    out = to_mono_16k(interleave(tone, tone), 48_000, 2)
    assert np.sqrt(np.mean(out ** 2)) < 0.01  # input RMS is ~0.35


def test_stereo_channels_are_averaged():
    left = np.full(4800, 0.4, dtype=np.float32)
    right = np.full(4800, 0.2, dtype=np.float32)
    out = to_mono_16k(interleave(left, right), 48_000, 2)
    assert out[200:-200].mean() == pytest.approx(0.3, abs=1e-3)  # edges affected by the filter


def test_int16_is_scaled_to_minus_one_plus_one():
    samples = np.array([16384, -16384, 0, 32767], dtype=np.int16)
    out = to_mono_16k(samples, TARGET_RATE, 1)
    assert out.tolist() == pytest.approx([0.5, -0.5, 0.0, 32767 / 32768])


def test_16k_mono_input_is_unchanged():
    tone = sine(440, 0.5, TARGET_RATE)
    assert np.array_equal(to_mono_16k(tone, TARGET_RATE, 1), tone)


def test_wrong_sample_type_fails_loudly():
    with pytest.raises(TypeError):
        to_mono_16k(np.zeros(100, dtype=np.float64), 48_000, 2)


def test_odd_number_of_stereo_samples_fails_loudly():
    with pytest.raises(ValueError):
        to_mono_16k(np.zeros(101, dtype=np.float32), 48_000, 2)


def test_load_wav_like_our_recordings(tmp_path):
    tone = sine(440, 1.0, 48_000)
    pcm = (interleave(tone, tone) * 32767).astype(np.int16)
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(48_000)
        f.writeframes(pcm.tobytes())
    out = load_wav_16k(path)
    assert len(out) == 16_000
    assert dominant_frequency(out, TARGET_RATE) == pytest.approx(440, abs=2)


# --- StreamResampler (live audio arrives in ~10 ms chunks) ------------------------------

def _stream(samples, chunk_sizes, rate=48_000, channels=2):
    resampler = StreamResampler(rate, channels)
    out, i, k = [], 0, 0
    while i < len(samples):
        size = chunk_sizes[k % len(chunk_sizes)] * channels
        out.append(resampler.process(samples[i:i + size]))
        i, k = i + size, k + 1
    return np.concatenate(out)


def test_stream_equals_whole_file_conversion_with_10_sample_delay():
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(48_000 * 2) * 3000).astype(np.int16)  # 1 s stereo
    whole = to_mono_16k(noise, 48_000, 2)
    streamed = _stream(noise, [480, 512, 333, 1024])  # uneven chunk sizes on purpose
    n = len(whole) - 20
    np.testing.assert_allclose(streamed[10:10 + n], whole[:n], atol=1e-5)


def test_stream_result_does_not_depend_on_chunk_size():
    tone = (sine(440, 0.5, 48_000) * 20000).astype(np.int16)
    stereo = interleave(tone, tone)
    np.testing.assert_allclose(_stream(stereo, [512]), _stream(stereo, [97, 1000]), atol=1e-6)


def test_stream_removes_frequencies_above_8khz():
    tone = (sine(10_000, 1.0, 48_000) * 32767).astype(np.int16)
    out = _stream(interleave(tone, tone), [512])
    assert np.sqrt(np.mean(out[100:] ** 2)) < 0.01


def test_stream_rejects_44100_hz_loudly():
    with pytest.raises(ValueError):
        StreamResampler(44_100, 2)
