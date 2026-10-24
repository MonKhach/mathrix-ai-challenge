"""Compare translation decoding settings against loops (M5, translation loop at 04:06 in the call).

Usage (from the project folder, with .venv activated):
    python -m evaluation.translation_settings --logs private/m5_live_friends.txt private/live_call2.txt ...

For every setting, the same NLLB model translates:
  - the 5 clips with English references -> chrF (Armenian reference and NVIDIA transcription, as D22)
  - eval/translation_repetitions.tsv -> real repetitions that must survive ("Ի դեպ, ի դեպ")
  - every Armenian line of the given live/replay logs -> loops found, lines changed vs baseline
and times every sentence. Writes a report (default private/mt_settings.md: it contains log text,
so it stays private).
"""
import argparse
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from sacrebleu.metrics import CHRF

from app.translator import NllbTranslator
from evaluation.benchmark_translation import ASR_RESULTS, REFERENCES_EN, REFERENCES_HY, read_tsv

ROOT = Path(__file__).resolve().parent.parent
REPETITIONS = ROOT / "eval" / "translation_repetitions.tsv"

# "baseline" = the library's behaviour (all off), so new settings are compared with the original.
SETTINGS = {
    "baseline": dict(max_length_ratio=None, no_repeat_ngram_size=0, repetition_penalty=1.0),
    "cap2": dict(max_length_ratio=2.0),
    "cap2+ngram3": dict(max_length_ratio=2.0, no_repeat_ngram_size=3),
    "cap2+ngram4": dict(max_length_ratio=2.0, no_repeat_ngram_size=4),
    "cap2+pen1.1": dict(max_length_ratio=2.0, repetition_penalty=1.1),
    "cap2+pen1.2": dict(max_length_ratio=2.0, repetition_penalty=1.2),
}


def read_log_lines(path) -> list[str]:
    """Armenian lines ("HY: ...") of a transcript; logs from Tee-Object are UTF-16."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    return [line[4:].strip() for line in text.splitlines() if line.startswith("HY: ") and line[4:].strip()]


def looks_like_loop(english: str) -> bool:
    """Same three words three or more times (e.g. 'you want to go' x50)."""
    words = re.findall(r"[\w']+", english.lower())
    counts = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    return bool(counts) and max(counts.values()) >= 3


def repetition_kept(english: str, keep: str) -> bool:
    """keep: 'a|b' alternatives; kept if any alternative appears at least twice."""
    words = re.findall(r"[\w']+", english.lower())
    for alt in keep.split("|"):
        target = alt.split()
        hits = sum(words[i:i + len(target)] == target for i in range(len(words)))
        if hits >= 2:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Compare translation settings against loops.")
    parser.add_argument("--logs", nargs="*", default=[], help="transcript files with HY: lines")
    parser.add_argument("--out", default="private/mt_settings.md")
    args = parser.parse_args()

    hy_refs, english_refs = read_tsv(REFERENCES_HY), read_tsv(REFERENCES_EN)
    asr = read_tsv(ASR_RESULTS, value="hypothesis")
    scored = sorted(english_refs)
    repetitions = list(read_tsv(REPETITIONS, key="text", value="keep").items())
    log_lines = list(dict.fromkeys(line for p in args.logs for line in read_log_lines(p)))
    print(f"{len(scored)} reference clips, {len(repetitions)} repetition sentences, "
          f"{len(log_lines)} distinct log lines")

    model = NllbTranslator()
    model.translate("Բարեւ։")  # warm-up, not timed
    tokens = lambda text: len(model._tokenizer.encode(text))  # noqa: E731

    results = {}
    for name, setting in SETTINGS.items():
        model.decoding = setting
        times = []

        def translate(text):
            t0 = time.perf_counter()
            english = model.translate(text)
            times.append(time.perf_counter() - t0)
            return english

        r = {"from_ref": [translate(hy_refs[f]) for f in scored],
             "from_asr": [translate(asr[f]) for f in scored],
             "repetitions": [translate(t) for t, _ in repetitions],
             "logs": [translate(t) for t in log_lines]}
        refs = [[english_refs[f] for f in scored]]
        r["chrf_ref"] = CHRF().corpus_score(r["from_ref"], refs).score
        r["chrf_asr"] = CHRF().corpus_score(r["from_asr"], refs).score
        r["kept"] = sum(repetition_kept(e, k) for e, (_, k) in zip(r["repetitions"], repetitions))
        r["loops"] = sum(looks_like_loop(e) for e in r["logs"])
        r["median_s"], r["max_s"] = statistics.median(times), max(times)
        normal = [(tokens(e), tokens(h)) for e, h in zip(r["logs"], log_lines) if not looks_like_loop(e)]
        r["max_ratio"] = max((o / i for o, i in normal), default=0.0)
        results[name] = r
        print(f"{name:12} chrF {r['chrf_ref']:.1f}/{r['chrf_asr']:.1f}  repetitions kept "
              f"{r['kept']}/{len(repetitions)}  loops {r['loops']}  time median {r['median_s']:.2f} s "
              f"max {r['max_s']:.2f} s")

    base = results["baseline"]
    out = [f"# Translation settings against loops ({len(log_lines)} log lines from: "
           f"{', '.join(Path(p).name for p in args.logs) or 'none'})", "",
           "| Setting | chrF (ref / ASR) | Real repetitions kept | Loops in logs | Log lines changed "
           "vs baseline | Time median / max | Max out/in tokens (non-loop) |", "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        changed = sum(a != b for a, b in zip(r["logs"], base["logs"]))
        out.append(f"| {name} | {r['chrf_ref']:.1f} / {r['chrf_asr']:.1f} | {r['kept']}/{len(repetitions)} "
                   f"| {r['loops']} | {changed} | {r['median_s']:.2f} / {r['max_s']:.2f} s "
                   f"| {r['max_ratio']:.2f} |")
    out += ["", "## Real repetitions", ""]
    for i, (text, keep) in enumerate(repetitions):
        out.append(f"**HY:** {text}  (keep: {keep})")
        out += [f"- {name}: {r['repetitions'][i]}" for name, r in results.items()]
        out.append("")
    out += ["## Log lines where any setting differs from baseline (loops marked ⟳)", ""]
    for i, text in enumerate(log_lines):
        if all(r["logs"][i] == base["logs"][i] for r in results.values()):
            continue
        out.append(f"**HY:** {text}")
        for name, r in results.items():
            mark = " ⟳" if looks_like_loop(r["logs"][i]) else ""
            out.append(f"- {name}{mark}: {r['logs'][i]}")
        out.append("")
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
