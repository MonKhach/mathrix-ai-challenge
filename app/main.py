"""Live Armenian transcription -> English translation in the terminal, labelled by speaker.

    python -m app.main --download                 # once, with internet: fetch all models
    python -m app.main --list-devices
    python -m app.main --mic 12 --system 13       # microphone = "Me", system audio = Speaker 1, 2, ...
Stop with Ctrl+C.

Runs OFFLINE by default (DECISIONS.md D24): models are only read from the local cache.
Both sources are processed in one loop, phrases in the order they finish (D25).
"""
import argparse
import faulthandler
import os
import queue
import time

from app.output import format_segment, format_timestamp
from app.pipeline import SHORT_RULES, Pipeline
from app.speakers import SHORT_PHRASE_S, SHORT_RULE, SIMILARITY_THRESHOLD


def add_pipeline_arguments(parser):
    """Pipeline settings, shared with evaluation/replay.py."""
    parser.add_argument("--speaker-threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help=f"similarity needed to reuse a speaker label (default {SIMILARITY_THRESHOLD})")
    parser.add_argument("--short-seconds", type=float, default=SHORT_PHRASE_S,
                        help=f"phrases shorter than this are 'short' (default {SHORT_PHRASE_S})")
    parser.add_argument("--short-rule", choices=SHORT_RULES, default=SHORT_RULE,
                        help=f"how short call-side phrases are labelled (default {SHORT_RULE})")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="phrases with lower recognition confidence are shown as unclear "
                             "and not translated (default: off, D29)")
    parser.add_argument("--debug", action="store_true",
                        help="also show speaker similarity and recognition confidence")


def load_models():
    """Load every model once. In offline mode a missing model fails with a clear hint."""
    from app.asr import NvidiaOnnxASR
    from app.speakers import SpeakerEmbedder
    from app.translator import NllbTranslator
    from app.vad import SileroModel
    try:
        print("[1/4] Loading speech recognition (NVIDIA FastConformer)...")
        asr = NvidiaOnnxASR()
        print("[2/4] Loading translation (NLLB-600M)...")
        translator = NllbTranslator()
        print("[3/4] Loading speaker identification (WeSpeaker)...")
        embedder = SpeakerEmbedder()
        print("[4/4] Loading voice activity detection (Silero)...")
        SileroModel()  # one instance per source is created later; this checks it is available
    except Exception as error:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError("A model could not be loaded in offline mode. If this is the first "
                               "run, download the models once with: python -m app.main --download"
                               ) from error
        raise
    return asr, translator, embedder


def print_segment(segment, offset, latency=True, debug=False):
    """offset: seconds since the start of the session (live) or of the recording (replay)."""
    if segment.unclear:
        heard = f": '{segment.armenian}'" if segment.armenian else ""
        conf = f", confidence {segment.confidence:.2f}" if segment.confidence is not None else ""
        print(f"[{format_timestamp(offset)}] {segment.label}   (unclear sound{heard}{conf})\n",
              flush=True)
        return
    print(format_segment(offset, segment.label, segment.armenian, segment.english,
                         latency_s=segment.finished_at - segment.end_time if latency else None,
                         similarity=segment.similarity if debug else None,
                         confidence=segment.confidence if debug else None), flush=True)


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Live Armenian -> English transcription.")
    parser.add_argument("--download", action="store_true",
                        help="download all models once (needs internet), then exit")
    parser.add_argument("--list-devices", action="store_true", help="list recording devices and exit")
    parser.add_argument("--mic", type=int, help="microphone device index (labelled 'Me')")
    parser.add_argument("--system", type=int, help="system-audio loopback device index (the call)")
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    # Must be set before any Hugging Face library is imported (they read it at import time).
    os.environ["HF_HUB_OFFLINE"] = "0" if args.download else "1"
    if args.download:
        load_models()
        print("All models are downloaded. From now on the program runs offline.")
        return

    import pyaudiowpatch as pyaudio

    from app.audio_capture import AudioCapture, print_devices
    from app.vad import PhraseDetector, SileroModel

    with pyaudio.PyAudio() as pa:
        if args.list_devices:
            print_devices(pa)
            return
        sources = {name: index for name, index in (("mic", args.mic), ("system", args.system))
                   if index is not None}
        if not sources:
            parser.error("give --mic and/or --system (see --list-devices)")

        asr, translator, embedder = load_models()
        pipeline = Pipeline(asr, translator, embedder, args.speaker_threshold,
                            args.short_seconds, args.short_rule, args.min_confidence)
        detectors = {name: PhraseDetector(name, SileroModel()) for name in sources}
        chunks = queue.Queue()
        captures = [AudioCapture(pa, index, name, chunks) for name, index in sources.items()]
        overflows_reported = {c.source: 0 for c in captures}
        session_start = time.monotonic()

        print()
        for c in captures:
            print(f"Listening to {c.name} ({'Me' if c.source == 'mic' else 'call side'})")
        print(f"Speaker settings: threshold {args.speaker_threshold}, short phrases "
              f"< {args.short_seconds} s -> rule '{args.short_rule}', min confidence "
              f"{args.min_confidence}. Press Ctrl+C to stop.\n")
        try:
            while True:
                for c in captures:
                    c.raise_if_failed()
                    if c.overflows > overflows_reported[c.source]:
                        print(f"WARNING: {c.overflows - overflows_reported[c.source]} audio "
                              f"overflow(s) on {c.name} - some audio was lost")
                        overflows_reported[c.source] = c.overflows

                phrases = []
                try:
                    chunk = chunks.get(timeout=0.1)
                    phrases += detectors[chunk.source].feed(chunk.samples, chunk.captured_at)
                except queue.Empty:
                    pass
                # Checked for every source on every loop: the mic sends audio all the time, so the
                # queue is rarely empty, but a silent loopback must still end its open phrase.
                now = time.monotonic()
                for detector in detectors.values():
                    phrases += detector.on_idle(now)

                for phrase in sorted(phrases, key=lambda p: p.end_time):
                    print_segment(pipeline.process(phrase), phrase.start_time - session_start,
                                  debug=args.debug)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            for c in captures:
                c.close()


if __name__ == "__main__":
    main()
