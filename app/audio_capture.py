"""Live audio capture (microphone or system-audio loopback) with PyAudioWPatch / WASAPI (D8, D9).

Each source runs in PyAudio's own thread and puts 16 kHz mono chunks into a shared queue.
"""
import queue
import time
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio

from app.audio_utils import StreamResampler

CHUNK_SIZE = 512  # frames per callback at the device rate (~10.7 ms at 48 kHz)


@dataclass
class AudioChunk:
    source: str          # "mic" or "system"
    samples: np.ndarray  # 16 kHz mono float32
    captured_at: float   # time.monotonic() when the chunk arrived


def recording_devices(pa):
    """All WASAPI devices that can record: microphones and loopbacks."""
    wasapi_index = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
    return [info for i in range(pa.get_device_count())
            if (info := pa.get_device_info_by_index(i))["hostApi"] == wasapi_index
            and info["maxInputChannels"] > 0]


def print_devices(pa):
    print(f"{'index':>5}  {'type':<9} {'ch':>3} {'rate':>6}  name")
    for info in recording_devices(pa):
        kind = "loopback" if info["isLoopbackDevice"] else "mic"
        print(f"{info['index']:>5}  {kind:<9} {info['maxInputChannels']:>3} "
              f"{int(info['defaultSampleRate']):>6}  {info['name']}")


class AudioCapture:
    def __init__(self, pa, device_index: int, source: str, out_queue: queue.Queue):
        info = pa.get_device_info_by_index(device_index)
        want_loopback = source == "system"
        if info["maxInputChannels"] == 0 or info["isLoopbackDevice"] != want_loopback:
            expected = "a loopback device" if want_loopback else "a microphone"
            raise ValueError(f"Device {device_index} ({info['name']}) is not {expected}. "
                             f"Run with --list-devices.")
        self.source = source
        self.name = info["name"]
        self.overflows = 0
        self._queue = out_queue
        self._error = None
        channels = info["maxInputChannels"]
        rate = int(info["defaultSampleRate"])
        self._resampler = StreamResampler(rate, channels)
        self._stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                               frames_per_buffer=CHUNK_SIZE, input=True,
                               input_device_index=device_index, stream_callback=self._callback)

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            if status:
                self.overflows += 1
            samples = self._resampler.process(np.frombuffer(in_data, dtype=np.int16))
            self._queue.put(AudioChunk(self.source, samples, time.monotonic()))
            return (in_data, pyaudio.paContinue)
        except Exception as error:  # an exception here would otherwise stop the stream silently
            self._error = error
            return (in_data, pyaudio.paAbort)

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"Audio capture from {self.name} failed") from self._error

    def close(self):
        self._stream.stop_stream()
        self._stream.close()
