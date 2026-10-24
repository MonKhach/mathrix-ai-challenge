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
import sys
import time
from datetime import datetime

from app.memory import memory_line
from app.output import format_segment, format_timestamp
from app.pipeline import SHORT_RULES, Pipeline
from app.speakers import SHORT_PHRASE_S, SHORT_RULE, SIMILARITY_THRESHOLD
from app.vad import MAX_PHRASE_S, SOFT_AFTER_S, SOFT_SILENCE_S

MEMORY_REPORT_EVERY_S = 60  # with --debug (TESTS.md B3)


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
    parser.add_argument("--max-phrase", type=float, default=MAX_PHRASE_S,
                        help=f"longest phrase in seconds before it is cut (default {MAX_PHRASE_S})")
    parser.add_argument("--soft-after", type=float, default=SOFT_AFTER_S,
                        help="phrases longer than this many seconds end at a shorter pause "
                             f"(default {SOFT_AFTER_S}; 0 = off)")
    parser.add_argument("--soft-silence", type=float, default=SOFT_SILENCE_S,
                        help=f"that shorter pause in seconds (default {SOFT_SILENCE_S})")
    parser.add_argument("--debug", action="store_true",
                        help="also show speaker similarity and recognition confidence "
                             "(live: also memory use every 60 s)")


def phrase_settings(args) -> dict:
    """Phrase-length settings for PhraseDetector, from the command line."""
    return dict(max_phrase_s=args.max_phrase, soft_after_s=args.soft_after or None,
                soft_silence_s=args.soft_silence)


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


def use_utf8_output():
    """Armenian text must never crash printing. On Windows, output redirected to a pipe or file
    otherwise uses the old cp1252 encoding, which has no Armenian letters (D30)."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def segment_timing(segment, latency=True) -> dict:
    """Where the time went (M5): phrase length, waiting (live only), then each processing step."""
    timing = {"phrase": segment.end_time - segment.start_time}
    if latency:
        timing["wait"] = segment.started_at - segment.end_time
    return timing | segment.step_times


def segment_text(segment, offset, latency=True, debug=False) -> str:
    """offset: seconds since the start of the session (live) or of the recording (replay)."""
    if segment.unclear:
        heard = f": '{segment.armenian}'" if segment.armenian else ""
        conf = f", confidence {segment.confidence:.2f}" if segment.confidence is not None else ""
        return f"[{format_timestamp(offset)}] {segment.label}   (unclear sound{heard}{conf})\n"
    return format_segment(offset, segment.label, segment.armenian, segment.english,
                          latency_s=segment.finished_at - segment.end_time if latency else None,
                          similarity=segment.similarity if debug else None,
                          confidence=segment.confidence if debug else None,
                          timing=segment_timing(segment, latency) if debug else None)


def main():
    faulthandler.enable()
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Live Armenian -> English transcription.")
    parser.add_argument("--download", action="store_true",
                        help="download all models once (needs internet), then exit")
    parser.add_argument("--list-devices", action="store_true", help="list recording devices and exit")
    parser.add_argument("--mic", type=int, help="microphone device index (labelled 'Me')")
    parser.add_argument("--system", type=int, help="system-audio loopback device index (the call)")
    parser.add_argument("--log", type=str,
                        help="also append the transcript to this UTF-8 file, e.g. "
                             "private/session.txt (private/ is git-ignored)")
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    # Must be set before any Hugging Face library is imported (they read it at import time).
    os.environ["HF_HUB_OFFLINE"] = "0" if args.download else "1"
    if args.download:
        load_models()
        print("All models are downloaded. From now on the program runs offline.")
        return

    import pyaudiowpatch as pyaudio

    from app.audio_capture import AudioCapture, AudioDeviceLost, print_devices
    from app.vad import PhraseDetector, SileroModel

    with pyaudio.PyAudio() as pa:
        if args.list_devices:
            print_devices(pa)
            return
        sources = {name: index for name, index in (("mic", args.mic), ("system", args.system))
                   if index is not None}
        if not sources:
            parser.error("give --mic and/or --system (see --list-devices)")

        log = open(args.log, "a", encoding="utf-8") if args.log else None

        def emit(text):
            print(text, flush=True)
            if log:
                log.write(text + "\n")
                log.flush()

        asr, translator, embedder = load_models()
        pipeline = Pipeline(asr, translator, embedder, args.speaker_threshold,
                            args.short_seconds, args.short_rule, args.min_confidence)
        detectors = {name: PhraseDetector(name, SileroModel(), **phrase_settings(args))
                     for name in sources}
        chunks = queue.Queue()
        captures = [AudioCapture(pa, index, name, chunks) for name, index in sources.items()]
        overflows_reported = {c.source: 0 for c in captures}
        device_lost = False
        session_start = time.monotonic()
        last_memory_report = session_start

        emit(f"\n=== Session {datetime.now():%Y-%m-%d %H:%M:%S} ===")
        for c in captures:
            emit(f"Listening to {c.name} ({'Me' if c.source == 'mic' else 'call side'})")
        emit(f"Speaker settings: threshold {args.speaker_threshold}, short phrases "
             f"< {args.short_seconds} s -> rule '{args.short_rule}', min confidence "
             f"{args.min_confidence}.")
        emit(f"Phrase settings: max {args.max_phrase} s, shorter pause {args.soft_silence} s after "
             f"{args.soft_after} s (0 = off). Press Ctrl+C to stop.")
        emit(memory_line("models loaded") + "\n")
        try:
            while True:
                for c in captures:
                    c.raise_if_failed()
                    if c.overflows > overflows_reported[c.source]:
                        emit(f"WARNING: {c.overflows - overflows_reported[c.source]} audio "
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
                    emit(segment_text(pipeline.process(phrase), phrase.start_time - session_start,
                                      debug=args.debug))

                if args.debug and now - last_memory_report >= MEMORY_REPORT_EVERY_S:
                    emit(memory_line(format_timestamp(now - session_start)))
                    last_memory_report = now
        except KeyboardInterrupt:
            emit("Stopped.")
        except AudioDeviceLost as error:
            emit(f"\nERROR: {error}")
            device_lost = True
        finally:
            emit(memory_line("end of session"))
            for c in captures:
                c.close()
            if log:
                log.close()
    if device_lost:
        sys.exit(1)


if __name__ == "__main__":
    main()
