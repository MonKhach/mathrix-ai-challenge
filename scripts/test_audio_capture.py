"""M1 throwaway test: record the microphone and system audio (WASAPI loopback) at the same time.

Usage (from the project folder, with .venv activated):
    python scripts/test_audio_capture.py --list                  # show devices
    python scripts/test_audio_capture.py                         # default mic + default loopback
    python scripts/test_audio_capture.py --mic 3 --system 12     # pick devices by index
"""
import argparse
import time
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

OUT_DIR = Path(__file__).resolve().parent.parent / "recordings"
CHUNK_SIZE = 512  # audio frames delivered per callback


def wasapi_devices(p):
    """Return all WASAPI devices that can record (microphones and loopbacks)."""
    wasapi_index = p.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["hostApi"] == wasapi_index and info["maxInputChannels"] > 0:
            devices.append(info)
    return devices


def print_devices(p):
    print(f"{'index':>5}  {'type':<9} {'ch':>3} {'rate':>6}  name")
    for info in wasapi_devices(p):
        kind = "loopback" if info["isLoopbackDevice"] else "mic"
        print(f"{info['index']:>5}  {kind:<9} {info['maxInputChannels']:>3} "
              f"{int(info['defaultSampleRate']):>6}  {info['name']}")


def pick_device(p, index, want_loopback):
    """Return device info for the given index, or the Windows default. Fails loudly if wrong type."""
    if index is None:
        if want_loopback:
            return p.get_default_wasapi_loopback()
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        return p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
    info = p.get_device_info_by_index(index)
    if info["maxInputChannels"] == 0 or info["isLoopbackDevice"] != want_loopback:
        expected = "a loopback device" if want_loopback else "a microphone"
        raise ValueError(f"Device {index} ({info['name']}) is not {expected}. Run with --list.")
    return info


class Recorder:
    """Collects raw audio from one device via a callback (called by PyAudio in its own thread)."""

    def __init__(self, label, info):
        self.label = label
        self.name = info["name"]
        self.index = info["index"]
        self.channels = info["maxInputChannels"]
        self.rate = int(info["defaultSampleRate"])
        self.chunks = []
        self.problem_callbacks = 0  # callbacks where PyAudio reported overflow/underflow

    def callback(self, in_data, frame_count, time_info, status):
        self.chunks.append(in_data)
        if status:
            self.problem_callbacks += 1
        return (in_data, pyaudio.paContinue)

    def open_stream(self, p):
        return p.open(format=pyaudio.paInt16, channels=self.channels, rate=self.rate,
                      frames_per_buffer=CHUNK_SIZE, input=True,
                      input_device_index=self.index, stream_callback=self.callback)

    def save_and_report(self, seconds):
        raw = b"".join(self.chunks)
        path = OUT_DIR / f"test_{self.label}.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(self.channels)
            f.setsampwidth(2)  # 16-bit
            f.setframerate(self.rate)
            f.writeframes(raw)

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        captured = len(samples) / self.channels / self.rate
        peak_db = 20 * np.log10(np.max(np.abs(samples)) + 1e-9)
        rms_db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-9)

        print(f"\n[{self.label}] {self.name} (index {self.index})")
        print(f"  format:   {self.rate} Hz, {self.channels} channel(s)")
        print(f"  captured: {captured:.2f} s of {seconds:.2f} s requested")
        print(f"  level:    peak {peak_db:.1f} dBFS, average {rms_db:.1f} dBFS")
        print(f"  problems: {self.problem_callbacks} callbacks with overflow/underflow")
        print(f"  saved:    {path}")


def main():
    parser = argparse.ArgumentParser(description="Record mic + system audio at the same time.")
    parser.add_argument("--list", action="store_true", help="list recording devices and exit")
    parser.add_argument("--mic", type=int, help="microphone device index (default: Windows default)")
    parser.add_argument("--system", type=int, help="loopback device index (default: default speakers)")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    with pyaudio.PyAudio() as p:
        if args.list:
            print_devices(p)
            return

        mic = Recorder("mic", pick_device(p, args.mic, want_loopback=False))
        system = Recorder("system", pick_device(p, args.system, want_loopback=True))
        OUT_DIR.mkdir(exist_ok=True)

        print(f"Recording {args.seconds} s — speak now, and keep audio playing...")
        with mic.open_stream(p), system.open_stream(p):
            time.sleep(args.seconds)
        print("Done.")

        mic.save_and_report(args.seconds)
        system.save_and_report(args.seconds)


if __name__ == "__main__":
    main()
