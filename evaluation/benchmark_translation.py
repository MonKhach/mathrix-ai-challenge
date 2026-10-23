"""Benchmark one translation model (DECISIONS.md D20).

Usage (from the project folder, with .venv activated):
    python -m evaluation.benchmark_translation --model opus-mt
    python -m evaluation.benchmark_translation --model nllb-600m
    python -m evaluation.compare_translations        # side-by-side file for reading

Translates (a) the Armenian references -> pure translation quality, and
           (b) NVIDIA's transcriptions (results/asr_nvidia.tsv) -> what the live program shows.
chrF is computed on the clips that have an English reference (eval/references_en.tsv).
"""
import argparse
import csv
import faulthandler
import time
from pathlib import Path

import psutil
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parent.parent
REFERENCES_HY = ROOT / "eval" / "references.tsv"
REFERENCES_EN = ROOT / "eval" / "references_en.tsv"
ASR_RESULTS = ROOT / "results" / "asr_nvidia.tsv"
RESULTS_DIR = ROOT / "results"


def read_tsv(path, key="file", value="text"):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row[key]: row[value] for row in csv.DictReader(f, delimiter="\t")}


def load_model(name):
    from app.translator import NllbTranslator, OpusMTTranslator
    return {"opus-mt": OpusMTTranslator, "nllb-600m": NllbTranslator}[name]()


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Benchmark one hy->en translation model.")
    parser.add_argument("--model", required=True, choices=["opus-mt", "nllb-600m"])
    args = parser.parse_args()

    armenian_refs = read_tsv(REFERENCES_HY)
    asr_outputs = read_tsv(ASR_RESULTS, value="hypothesis")
    english_refs = read_tsv(REFERENCES_EN)
    missing = (set(english_refs) | set(asr_outputs)) - set(armenian_refs)
    if missing:
        raise ValueError(f"Files without an Armenian reference: {sorted(missing)}")

    print(f"[1/3] Loading {args.model}...")
    t0 = time.perf_counter()
    model = load_model(args.model)
    load_seconds = time.perf_counter() - t0
    model.translate(next(iter(armenian_refs.values())))  # warm-up, not counted

    print(f"[2/3] Translating {len(armenian_refs)} references + {len(asr_outputs)} transcriptions...")
    rows, times = [], []
    for file, hy_ref in armenian_refs.items():
        t0 = time.perf_counter()
        en_from_ref = model.translate(hy_ref)
        times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        en_from_asr = model.translate(asr_outputs[file])
        times.append(time.perf_counter() - t0)
        rows.append({"file": file, "hy_reference": hy_ref, "en_from_reference": en_from_ref,
                     "hy_asr": asr_outputs[file], "en_from_asr": en_from_asr,
                     "en_reference": english_refs.get(file, "")})
        print(f"{file:13} {en_from_ref}")

    peak_ram_mb = psutil.Process().memory_info().peak_wset / 2**20

    print("[3/3] Scoring and writing results...")
    chrf = CHRF()
    scored = [r for r in rows if r["en_reference"]]
    refs = [[r["en_reference"] for r in scored]]
    chrf_ref = chrf.corpus_score([r["en_from_reference"] for r in scored], refs).score
    chrf_asr = chrf.corpus_score([r["en_from_asr"] for r in scored], refs).score

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"mt_{args.model}.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = "\n".join([
        f"# Translation benchmark: {args.model}", "",
        f"- model load time: {load_seconds:.1f} s",
        f"- peak RAM (whole process): {peak_ram_mb:.0f} MB",
        f"- time per sentence: average {sum(times) / len(times):.2f} s, slowest {max(times):.2f} s",
        f"- chrF on {len(scored)} clips with English references "
        f"(0-100, higher is better):",
        f"  - from Armenian references (pure translation): {chrf_ref:.1f}",
        f"  - from NVIDIA transcriptions (live pipeline): {chrf_asr:.1f}", ""])
    (RESULTS_DIR / f"mt_{args.model}.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
