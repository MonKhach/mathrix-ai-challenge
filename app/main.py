"""Live Armenian transcription -> English translation in the terminal.

Walking skeleton (PLAN.md M3): ONE audio source.
    python -m app.main --download            # once, with internet: fetch all models
    python -m app.main --list-devices
    python -m app.main --source mic --device 12
    python -m app.main --source system --device 13
Stop with Ctrl+C.

Runs OFFLINE by default (DECISIONS.md D24): models are only read from the local cache.
"""
import argparse
import faulthandler
import os
import queue
import time

import pyaudiowpatch as pyaudio

from app.audio_capture import AudioCapture, print_devices
from app.output import format_segment
from app.vad import PhraseDetector, SileroModel

LABELS = {"mic": "Me", "system": "Call"}  # "Call" until speaker identification (M4)


def load_models():
    """Load every model once. In offline mode a missing model fails with a clear hint."""
    from app.asr import NvidiaOnnxASR
    from app.translator import NllbTranslator
    try:
        print("[1/3] Loading speech recognition (NVIDIA FastConformer)...")
        asr = NvidiaOnnxASR()
        print("[2/3] Loading translation (NLLB-600M)...")
        translator = NllbTranslator()
        print("[3/3] Loading voice activity detection (Silero)...")
        vad_model = SileroModel()
    except Exception as error:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError("A model could not be loaded in offline mode. If this is the first "
                               "run, download the models once with: python -m app.main --download"
                               ) from error
        raise
    return asr, translator, vad_model


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Live Armenian -> English transcription.")
    parser.add_argument("--download", action="store_true",
                        help="download all models once (needs internet), then exit")
    parser.add_argument("--list-devices", action="store_true", help="list recording devices and exit")
    parser.add_argument("--source", choices=["mic", "system"], help="which kind of device --device is")
    parser.add_argument("--device", type=int, help="device index from --list-devices")
    args = parser.parse_args()

    # Must be set before any Hugging Face library is imported (they read it at import time).
    os.environ["HF_HUB_OFFLINE"] = "0" if args.download else "1"
    if args.download:
        load_models()
        print("All models are downloaded. From now on the program runs offline.")
        return

    with pyaudio.PyAudio() as pa:
        if args.list_devices:
            print_devices(pa)
            return
        if args.source is None or args.device is None:
            parser.error("--source and --device are required (see --list-devices)")

        asr, translator, vad_model = load_models()
        detector = PhraseDetector(args.source, vad_model)

        chunks = queue.Queue()
        capture = AudioCapture(pa, args.device, args.source, chunks)
        session_start = time.monotonic()
        overflows_reported = 0
        print(f"\nListening to {capture.name} as '{LABELS[args.source]}'. Press Ctrl+C to stop.\n")
        try:
            while True:
                capture.raise_if_failed()
                if capture.overflows > overflows_reported:
                    print(f"WARNING: {capture.overflows - overflows_reported} audio overflow(s) "
                          f"- some audio was lost")
                    overflows_reported = capture.overflows
                try:
                    chunk = chunks.get(timeout=0.1)
                    phrases = detector.feed(chunk.samples, chunk.captured_at)
                except queue.Empty:
                    phrases = detector.on_idle(time.monotonic())

                for phrase in phrases:
                    armenian = asr.transcribe(phrase.audio)
                    if not armenian.strip():
                        print(f"[{LABELS[phrase.source]}] (speech detected, but no text recognised)\n")
                        continue
                    english = translator.translate(armenian)
                    latency = time.monotonic() - phrase.end_time
                    print(format_segment(phrase.start_time - session_start, LABELS[phrase.source],
                                         armenian, english, latency), flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            capture.close()


if __name__ == "__main__":
    main()
