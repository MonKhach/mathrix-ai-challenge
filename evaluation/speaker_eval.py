"""Offline test of speaker identification (DECISIONS.md D26) on eval/speakers.tsv.

Usage:  python -m evaluation.speaker_eval
Writes: results/speakers.md

1. Embeds every clip at three lengths: full clip, 2 s and 1 s excerpts (taken from the middle;
   short phrases are the expected weak spot).
2. Similarity of same-speaker vs different-speaker pairs (mic and system separately).
3. Threshold sweep: simulates live labelling of the call-side (video) clips in 20 random orders
   and counts splits (one person got several labels) and merges (one label covers several people).
"""
import csv
import faulthandler
import itertools
import os
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "eval" / "audio"
SPEAKERS = ROOT / "eval" / "speakers.tsv"
RESULTS = ROOT / "results" / "speakers.md"
LENGTHS = {"full": None, "2s": 2.0, "1s": 1.0}
THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(13)]  # 0.20 ... 0.80
ORDERS = 20


def middle_excerpt(audio, seconds, rate=16_000):
    if seconds is None:
        return audio
    n = int(seconds * rate)
    if len(audio) < n:
        raise ValueError(f"Clip shorter than {seconds} s")
    start = (len(audio) - n) // 2
    return audio[start:start + n]


def stats(values):
    return f"{min(values):.2f} / {np.mean(values):.2f} / {max(values):.2f} (n={len(values)})"


def simulate(clips, truth, embeddings, threshold, seed):
    from app.speakers import SpeakerTracker
    order = clips[:]
    random.Random(seed).shuffle(order)
    tracker = SpeakerTracker(threshold)
    assigned = {c: tracker.assign(embeddings[c])[0] for c in order}
    true = {c: truth[c] for c in assigned}
    labels_per_person = {p: {assigned[c] for c in assigned if true[c] == p} for p in set(true.values())}
    people_per_label = {l: {true[c] for c in assigned if assigned[c] == l} for l in set(assigned.values())}
    splits = sum(len(v) - 1 for v in labels_per_person.values())
    merges = sum(len(v) - 1 for v in people_per_label.values())
    return splits, merges, tracker.speaker_count


def main():
    faulthandler.enable()
    os.environ.setdefault("HF_HUB_OFFLINE", "0")  # may need to download the model on first run
    from app.audio_utils import load_wav_16k
    from app.speakers import SpeakerEmbedder, cosine_similarity

    with open(SPEAKERS, encoding="utf-8-sig", newline="") as f:
        clips_speaker = {r["file"]: r["speaker"] for r in csv.DictReader(f, delimiter="\t")}
    print(f"[1/3] Loading WeSpeaker model; {len(clips_speaker)} labelled clips...")
    embedder = SpeakerEmbedder()

    print("[2/3] Computing embeddings (full, 2 s, 1 s)...")
    emb = {length: {} for length in LENGTHS}
    times = []
    for clip in clips_speaker:
        audio = load_wav_16k(AUDIO_DIR / clip)
        for length, seconds in LENGTHS.items():
            t0 = time.perf_counter()
            emb[length][clip] = embedder.embed(middle_excerpt(audio, seconds))
            times.append(time.perf_counter() - t0)

    print("[3/3] Comparing pairs and simulating live labelling...")
    is_mic = lambda c: clips_speaker[c] == "me"
    lines = ["# Speaker identification test (WeSpeaker ResNet34-LM)", "",
             f"- clips: {len(clips_speaker)} ({sum(map(is_mic, clips_speaker))} mic = me, "
             f"{sum(not is_mic(c) for c in clips_speaker)} system/video, "
             f"{len(set(clips_speaker.values())) - 1} different video speakers)",
             f"- time per embedding: average {np.mean(times):.3f} s, slowest {max(times):.3f} s",
             "- excerpts: 2 s / 1 s cut from the middle of each clip", "",
             "## Cosine similarity of pairs: min / mean / max", "",
             "| length | same person, mic (me) | same person, system | different people, system |",
             "|---|---|---|---|"]
    for length in LENGTHS:
        groups = {"mic_same": [], "sys_same": [], "sys_diff": []}
        for a, b in itertools.combinations(clips_speaker, 2):
            s = cosine_similarity(emb[length][a], emb[length][b])
            if is_mic(a) and is_mic(b):
                groups["mic_same"].append(s)
            elif not is_mic(a) and not is_mic(b):
                groups["sys_same" if clips_speaker[a] == clips_speaker[b] else "sys_diff"].append(s)
        lines.append(f"| {length} | {stats(groups['mic_same'])} | {stats(groups['sys_same'])} | "
                     f"{stats(groups['sys_diff'])} |")

    system_clips = [c for c in clips_speaker if not is_mic(c)]
    true_count = len({clips_speaker[c] for c in system_clips})
    lines += ["", f"## Simulated live labelling of the {len(system_clips)} system clips "
              f"({true_count} people), {ORDERS} random orders", "",
              "Splits = one person received extra labels; merges = one label covered extra people "
              "(totals per order, averaged). Ideal: 0 / 0 and speakers found = "
              f"{true_count}.", "",
              "| threshold | full: splits / merges / speakers | 2 s: splits / merges / speakers | "
              "1 s: splits / merges / speakers |", "|---|---|---|---|"]
    for threshold in THRESHOLDS:
        cells = []
        for length in LENGTHS:
            runs = [simulate(system_clips, clips_speaker, emb[length], threshold, seed) for seed in range(ORDERS)]
            s, m, n = (np.mean([r[i] for r in runs]) for i in range(3))
            cells.append(f"{s:.1f} / {m:.1f} / {n:.1f}")
        lines.append(f"| {threshold:.2f} | " + " | ".join(cells) + " |")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
