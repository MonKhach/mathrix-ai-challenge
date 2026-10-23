"""Record one evaluation clip from the microphone or system audio (WASAPI loopback).

Recording starts immediately and stops when you press Enter.
The clip is saved unchanged (device's native format) to eval/audio/<name>.wav.

Usage (from the project folder, with .venv activated):
    python scripts/record_clip.py --list
    python scripts/record_clip.py --device 12 --name me_01        # microphone
    python scripts/record_clip.py --device 13 --name video_01     # system audio (loopback)
"""
import argparse
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

AUDIO_DIR = Path(__file__).resolve().parent.parent / "eval" / "audio"
CHUNK_SIZE = 512


def print_devices(p):
    wasapi_index = p.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
    print(f"{'index':>5}  {'type':<9} {'ch':>3} {'rate':>6}  name")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["hostApi"] == wasapi_index and info["maxInputChannels"] > 0:
            kind = "loopback" if info["isLoopbackDevice"] else "mic"
            print(f"{info['index']:>5}  {kind:<9} {info['maxInputChannels']:>3} "
                  f"{int(info['defaultSampleRate']):>6}  {info['name']}")


def record(p, device_index, path):
    info = p.get_device_info_by_index(device_index)
    if info["maxInputChannels"] == 0:
        raise ValueError(f"Device {device_index} ({info['name']}) cannot record. Run with --list.")
    channels = info["maxInputChannels"]
    rate = int(info["defaultSampleRate"])
    chunks = []
    problems = 0

    def callback(in_data, frame_count, time_info, status):
        nonlocal problems
        chunks.append(in_data)
        if status:
            problems += 1
        return (in_data, pyaudio.paContinue)

    with p.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                frames_per_buffer=CHUNK_SIZE, input=True,
                input_device_index=device_index, stream_callback=callback):
        input(f"Recording from '{info['name']}'... press Enter to stop.")

    raw = b"".join(chunks)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)  # 16-bit
        f.setframerate(rate)
        f.writeframes(raw)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(samples) / channels / rate
    peak_db = 20 * np.log10(np.max(np.abs(samples)) + 1e-9)
    rms_db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-9)
    print(f"Saved {path}")
    print(f"  {duration:.2f} s, {rate} Hz, {channels} ch, peak {peak_db:.1f} dBFS, "
          f"average {rms_db:.1f} dBFS, {problems} problem callbacks")


def main():
    parser = argparse.ArgumentParser(description="Record one evaluation clip.")
    parser.add_argument("--list", action="store_true", help="list recording devices and exit")
    parser.add_argument("--device", type=int, help="device index (mic or loopback) from --list")
    parser.add_argument("--name", help="clip name without extension, e.g. me_01")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing clip")
    args = parser.parse_args()

    with pyaudio.PyAudio() as p:
        if args.list:
            print_devices(p)
            return
        if args.device is None or args.name is None:
            parser.error("--device and --name are required (use --list to see devices)")

        path = AUDIO_DIR / f"{args.name}.wav"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists. Use another --name or add --overwrite.")
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        record(p, args.device, path)
        print(f"Now add a line to eval/references.tsv:  {args.name}.wav<TAB>exact Armenian text")


if __name__ == "__main__":
    main()
