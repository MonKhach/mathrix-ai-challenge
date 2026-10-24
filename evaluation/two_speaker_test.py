"""A6 test with ground truth: two recordings, each containing ONE known person (DECISIONS.md D27).

    python -m evaluation.two_speaker_test eval/audio/call2_system.wav eval/audio/call2_system_2.wav

Phrases are detected exactly as live (same resampler, Silero detector), embedded once, then fed
to the speaker labelling in 20 random orders (a real conversation mixes turns) for every
short-phrase rule x threshold. Counts per order, averaged:
  splits = one person received extra labels   merges = one label covered both people
  "?"    = phrases labelled "Speaker ?" (rule 'unknown')
Writes results/two_speaker_test.md (numbers only, no transcripts).
"""
import faulthandler
import itertools
import random
import sys
from pathlib import Path

import numpy as np

from evaluation.replay import detect_phrases  # sets offline mode, same detection as live
from app.audio_utils import TARGET_RATE
from app.main import load_models, use_utf8_output
from app.output import format_timestamp
from app.pipeline import SHORT_RULES, Pipeline
from app.speakers import SHORT_PHRASE_S, cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
ORDERS = 20


def stats(values):
    if not values:
        return "-"
    return f"{min(values):.2f} / {np.mean(values):.2f} / {max(values):.2f} (n={len(values)})"


def main():
    faulthandler.enable()
    use_utf8_output()
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) != 2 or not all(p.exists() for p in paths):
        raise SystemExit("Give exactly two existing recordings, one person each.")

    _, _, embedder = load_models()
    items = []  # (person, embedding, duration)
    starts = []  # position of each phrase in its recording
    for person, path in zip("AB", paths):
        phrases = detect_phrases({"system": path})
        print(f"Person {person} ({path.name}): {len(phrases)} phrases")
        for p in phrases:
            items.append((person, embedder.embed(p.audio), len(p.audio) / TARGET_RATE))
            starts.append(p.start_time)

    long_items = [i for i in items if i[2] >= SHORT_PHRASE_S]
    groups = {"same A": [], "same B": [], "A vs B": []}
    for (pa, ea, _), (pb, eb, _) in itertools.combinations(long_items, 2):
        key = f"same {pa}" if pa == pb else "A vs B"
        groups[key].append(cosine_similarity(ea, eb))

    n_short = sum(d < SHORT_PHRASE_S for _, _, d in items)
    lines = ["# Two-speaker test (A6): two known people through the real call app", "",
             f"- files: A = {paths[0].name}, B = {paths[1].name}",
             f"- phrases: {len(items)} ({sum(i[0] == 'A' for i in items)} A, "
             f"{sum(i[0] == 'B' for i in items)} B), {n_short} shorter than {SHORT_PHRASE_S} s", "",
             f"## Similarity of phrases >= {SHORT_PHRASE_S} s: min / mean / max", "",
             "| same person A | same person B | A vs B (different people) |", "|---|---|---|",
             f"| {stats(groups['same A'])} | {stats(groups['same B'])} | {stats(groups['A vs B'])} |",
             "", f"## Labelling in {ORDERS} random orders (ideal: 0 splits, 0 merges, 2 speakers)", "",
             "| rule | threshold | splits | merges | speakers found | \"?\" lines |",
             "|---|---|---|---|---|---|"]
    for rule in SHORT_RULES:
        for threshold in THRESHOLDS:
            totals = np.zeros(4)
            for seed in range(ORDERS):
                order = items[:]
                random.Random(seed).shuffle(order)
                pipeline = Pipeline(None, None, None, threshold, SHORT_PHRASE_S, rule)
                labelled = [(person, pipeline.label_from_embedding("system", e, d)[0])
                            for person, e, d in order]
                known = [(p, l) for p, l in labelled if l != "Speaker ?"]
                labels_of = {p: {l for q, l in known if q == p} for p in "AB"}
                people_of = {l: {p for p, m in known if m == l} for _, l in known}
                totals += [sum(max(0, len(v) - 1) for v in labels_of.values()),
                           sum(len(v) - 1 for v in people_of.values()),
                           pipeline.tracker.speaker_count,
                           len(labelled) - len(known)]
            s, m, n, q = totals / ORDERS
            lines.append(f"| {rule} | {threshold:.2f} | {s:.1f} | {m:.1f} | {n:.1f} | {q:.1f} |")

    lines += ["", "## Per-phrase check (diagnostic)", "",
              "Average similarity to the same person's other phrases, and the highest similarity "
              "to the other person. Below 0.30 to its own person = suspicious (listen at that time).",
              "", "| person | time in file | length (s) | avg vs own | max vs other | flag |",
              "|---|---|---|---|---|---|"]
    for i, (person, e, d) in enumerate(items):
        own = [cosine_similarity(e, items[j][1]) for j in range(len(items))
               if j != i and items[j][0] == person]
        other = [cosine_similarity(e, items[j][1]) for j in range(len(items)) if items[j][0] != person]
        avg_own = np.mean(own) if own else float("nan")
        flag = "SUSPICIOUS" if own and avg_own < 0.30 else ""
        lines.append(f"| {person} | {format_timestamp(starts[i])} | {d:.1f} | {avg_own:.2f} | "
                     f"{max(other) if other else float('nan'):.2f} | {flag} |")

    out = ROOT / "results" / "two_speaker_test.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
