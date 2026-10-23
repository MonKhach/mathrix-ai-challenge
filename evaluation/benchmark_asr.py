"""Benchmark one speech-to-text model on the evaluation set (PLAN.md M2).

Usage (from the project folder, with .venv activated):
    python -m evaluation.benchmark_asr --model nvidia
    python -m evaluation.benchmark_asr --model nvidia-int8
    python -m evaluation.benchmark_asr --model whisper-turbo-hy   # needs requirements-benchmark.txt

Writes results/asr_<model>.tsv (one row per clip) and results/asr_<model>.md (summary).
Run one model per command, so the RAM measurement belongs to that model only.
"""
import argparse
import csv
import faulthandler
import time
from pathlib import Path

import jiwer
import psutil

from app.audio_utils import TARGET_RATE, load_wav_16k
from evaluation.metrics import collapse_spaces, normalize, score

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "eval" / "audio"
REFERENCES = ROOT / "eval" / "references.tsv"
RESULTS_DIR = ROOT / "results"


def load_model(name):
    from app.asr import NvidiaOnnxASR, WhisperASR
    if name == "whisper-turbo-hy":
        return WhisperASR()
    if name == "nvidia":
        return NvidiaOnnxASR()
    if name == "nvidia-int8":
        return NvidiaOnnxASR(quantization="int8")
    raise ValueError(f"Unknown model {name!r}")


def read_references():
    with open(REFERENCES, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"No references in {REFERENCES}")
    for row in rows:
        if not (AUDIO_DIR / row["file"]).exists():
            raise FileNotFoundError(f"Audio for reference {row['file']} not found in {AUDIO_DIR}")
    return rows


def corpus_scores(rows):
    """Scores over a group of clips: total errors / total reference words (not an average of %)."""
    refs = [r["reference"] for r in rows]
    hyps = [r["hypothesis"] for r in rows]
    return {
        "clips": len(rows),
        "raw_wer": jiwer.wer([collapse_spaces(t) for t in refs], [collapse_spaces(t) for t in hyps]),
        "wer": jiwer.wer([normalize(t) for t in refs], [normalize(t) for t in hyps]),
        "cer": jiwer.cer([normalize(t) for t in refs], [normalize(t) for t in hyps]),
        "rtf": sum(r["seconds"] for r in rows) / sum(r["duration"] for r in rows),
        "max_seconds": max(r["seconds"] for r in rows),
    }


def main():
    faulthandler.enable()  # a crash below Python (e.g. in a DLL) still prints where it happened
    parser = argparse.ArgumentParser(description="Benchmark one ASR model on eval/.")
    parser.add_argument("--model", required=True, choices=["nvidia", "nvidia-int8", "whisper-turbo-hy"])
    args = parser.parse_args()

    references = read_references()
    process = psutil.Process()

    print(f"[1/4] Loading {args.model} (imports the runtime; first run downloads the model)...")
    t0 = time.perf_counter()
    model = load_model(args.model)
    load_seconds = time.perf_counter() - t0
    print(f"Loaded in {load_seconds:.1f} s")

    # Warm-up: the first call initialises things and is slower; it is not counted.
    print("[2/4] Warm-up on the first clip (not counted)...")
    model.transcribe(load_wav_16k(AUDIO_DIR / references[0]["file"]))

    print(f"[3/4] Transcribing {len(references)} clips...")
    rows = []
    for ref in references:
        audio = load_wav_16k(AUDIO_DIR / ref["file"])
        t0 = time.perf_counter()
        hypothesis = model.transcribe(audio)
        seconds = time.perf_counter() - t0

        s = score(ref["text"], hypothesis)
        duration = len(audio) / TARGET_RATE
        rows.append({
            "file": ref["file"],
            "group": "me" if ref["file"].startswith("me_") else "video",
            "has_digits": any(ch.isdigit() for ch in ref["text"]),
            "duration": duration,
            "seconds": seconds,
            "rtf": seconds / duration,
            **{k: s[k] for k in ("raw_wer", "wer", "cer", "substitutions", "deletions", "insertions")},
            "reference": ref["text"],
            "hypothesis": hypothesis,
        })
        print(f"{ref['file']:14} {seconds:5.2f} s  WER {s['wer']:6.1%}  CER {s['cer']:6.1%}  | {hypothesis}")

    peak_ram_mb = process.memory_info().peak_wset / 2**20  # Windows: peak working set of this process

    print("[4/4] Writing results...")
    RESULTS_DIR.mkdir(exist_ok=True)
    tsv_path = RESULTS_DIR / f"asr_{args.model}.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})

    groups = {"all": rows,
              "me (mic)": [r for r in rows if r["group"] == "me"],
              "video (system)": [r for r in rows if r["group"] == "video"],
              "clips with digits": [r for r in rows if r["has_digits"]]}
    lines = [f"# ASR benchmark: {args.model}", "",
             f"- model load time: {load_seconds:.1f} s",
             f"- peak RAM (whole process): {peak_ram_mb:.0f} MB", "",
             "| group | clips | raw WER | WER | CER | RTF | slowest clip (s) |",
             "|---|---|---|---|---|---|---|"]
    for label, group_rows in groups.items():
        if group_rows:
            c = corpus_scores(group_rows)
            lines.append(f"| {label} | {c['clips']} | {c['raw_wer']:.1%} | {c['wer']:.1%} | "
                         f"{c['cer']:.1%} | {c['rtf']:.2f} | {c['max_seconds']:.2f} |")
    summary = "\n".join(lines) + "\n"
    (RESULTS_DIR / f"asr_{args.model}.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Per-clip details: {tsv_path}")


if __name__ == "__main__":
    main()
