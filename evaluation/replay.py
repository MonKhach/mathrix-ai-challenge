"""Replay recorded audio through the SAME pipeline as the live program (no audio devices).

    python -m evaluation.replay --system eval/audio/call_system_mixed.wav --mic eval/audio/call_mic.wav
    python -m evaluation.replay --system eval/audio/call_system_mixed.wav --sweep

The audio goes through the same steps as live: StreamResampler in 512-frame chunks -> Silero
phrase detector -> app.pipeline.Pipeline. Times shown are positions in the recording.
Latency is NOT measured here (the file is processed as fast as possible) - measure it live.
Transcripts are only printed, never saved (the recordings are private conversations).

--sweep: no ASR/translation; embeds the call-side phrases once and counts the speaker labels
for several thresholds x short-phrase rules -> results/replay_sweep_<file>.md
"""
import argparse
import faulthandler
import os
import wave
from collections import Counter
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"  # models must already be downloaded (python -m app.main --download)

from app.audio_capture import CHUNK_SIZE  # noqa: E402
from app.audio_utils import TARGET_RATE, StreamResampler  # noqa: E402
from app.main import add_pipeline_arguments, load_models, print_segment  # noqa: E402
from app.pipeline import SHORT_RULES, Pipeline  # noqa: E402
from app.vad import PhraseDetector, SileroModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        rate, channels = f.getframerate(), f.getnchannels()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return samples, rate, channels


def detect_phrases(files):
    """files: {"mic": path, "system": path}. Returns phrases of all sources, ordered by end time."""
    streams = {}
    for source, path in files.items():
        samples, rate, channels = read_wav(path)
        streams[source] = dict(samples=samples, rate=rate, channels=channels, pos=0,
                               resampler=StreamResampler(rate, channels),
                               detector=PhraseDetector(source, SileroModel()))
    phrases, end_time = [], 0.0
    while any(s["pos"] < len(s["samples"]) for s in streams.values()):
        for s in streams.values():
            if s["pos"] >= len(s["samples"]):
                continue
            step = CHUNK_SIZE * s["channels"]
            chunk = s["samples"][s["pos"]:s["pos"] + step]
            s["pos"] += step
            t = s["pos"] / s["channels"] / s["rate"]  # recording time at the end of this chunk
            end_time = max(end_time, t)
            phrases += s["detector"].feed(s["resampler"].process(chunk), t)
    for s in streams.values():  # end phrases still open when the file ends
        phrases += s["detector"].on_idle(end_time + 10)
    return sorted(phrases, key=lambda p: p.end_time)


def sweep(phrases, embedder, short_seconds, out_name):
    calls = [p for p in phrases if p.source == "system"]
    embeddings = [embedder.embed(p.audio) for p in calls]
    durations = [len(p.audio) / TARGET_RATE for p in calls]
    n_short = sum(d < short_seconds for d in durations)
    lines = [f"# Replay sweep: {out_name}", "",
             f"- call-side phrases: {len(calls)} ({n_short} shorter than {short_seconds} s)",
             "- labels counted per setting; with ONE person on the call side the ideal is "
             "'Speaker 1' only", "",
             "| rule | threshold | labels used | phrases per label |", "|---|---|---|---|"]
    for rule in SHORT_RULES:
        for threshold in SWEEP_THRESHOLDS:
            p = Pipeline(None, None, None, threshold, short_seconds, rule)
            labels = [p.label_from_embedding("system", e, d)[0] for e, d in zip(embeddings, durations)]
            counts = Counter(labels)
            lines.append(f"| {rule} | {threshold:.2f} | {len(counts)} | "
                         + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + " |")
    out = ROOT / "results" / f"replay_sweep_{out_name}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten {out}")


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Replay recordings through the live pipeline.")
    parser.add_argument("--system", type=Path, help="recording of the call side (system audio)")
    parser.add_argument("--mic", type=Path, help="recording of the microphone")
    parser.add_argument("--sweep", action="store_true",
                        help="only count speaker labels for several settings (no ASR/translation)")
    add_pipeline_arguments(parser)
    args = parser.parse_args()
    files = {name: path for name, path in (("mic", args.mic), ("system", args.system)) if path}
    if not files:
        parser.error("give --system and/or --mic")
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)

    asr, translator, embedder = load_models()
    print("Detecting phrases...")
    phrases = detect_phrases(files)
    print(f"{len(phrases)} phrases found\n")

    if args.sweep:
        sweep(phrases, embedder, args.short_seconds, (args.system or args.mic).stem)
        return

    pipeline = Pipeline(asr, translator, embedder, args.speaker_threshold,
                        args.short_seconds, args.short_rule, args.min_confidence)
    labels, ranking = Counter(), []
    for phrase in phrases:
        segment = pipeline.process(phrase)
        labels[segment.label] += 1
        print_segment(segment, phrase.start_time, latency=False, debug=True)
        if segment.confidence is not None:
            ranking.append((segment.confidence, len(phrase.audio) / 16_000, segment.label,
                            segment.armenian))
    print("Labels: " + ", ".join(f"{k}: {v}" for k, v in sorted(labels.items())))
    print("\nRecognition confidence, lowest first (to choose --min-confidence, D29):")
    for confidence, seconds, label, text in sorted(ranking):
        print(f"  {confidence:.2f}  {seconds:4.1f} s  {label:10}  {text[:60]}")


if __name__ == "__main__":
    main()
