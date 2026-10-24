"""Does the live audio path make recognition worse than the benchmark path? (M5 step 4)

Live/replay lines often contain doubled syllables (`Շատատ`, `գործործը`); the M2 benchmark shows a
few too (`դուրր`, `ծանրրացնում`). This compares, on the evaluation clips, three paths into the
same NVIDIA model (all clips in eval/references.tsv):
  whole  - load_wav_16k: whole file converted at once (= M2 benchmark)
  stream - StreamResampler in live-sized chunks, then the whole clip at once
  live   - StreamResampler + PhraseDetector (Silero VAD, D32 settings), phrase texts joined
           (= exactly what live and replay do)
Reports WER/CER per path and "doubled" words: words with an adjacent repeated 1-3 letters
(e.g. դուրր, Շատատ) that do not occur in the reference (so real double letters are not counted).

Edges (M5): --edges 0.2/0.1,0.3/0.2 adds one "live" path per pre-roll/tail setting (seconds of
audio kept before speech starts / after it ends; default 0.2/0.1).

Usage:  python -m evaluation.asr_paths [--edges 0.2/0.1,0.3/0.2,0.4/0.3]   -> results/asr_paths.md
"""
import argparse
import os
import re
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import jiwer  # noqa: E402
import numpy as np  # noqa: E402

from app.audio_capture import CHUNK_SIZE  # noqa: E402
from app.audio_utils import StreamResampler, load_wav_16k  # noqa: E402
from evaluation.benchmark_translation import REFERENCES_HY, read_tsv  # noqa: E402
from evaluation.metrics import normalize  # noqa: E402
from evaluation.replay import detect_phrases, read_wav  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "eval" / "audio"
DOUBLED = re.compile(r"(\w{1,3})\1")


def doubled_words(hypothesis: str, reference: str) -> list[str]:
    """Words with an adjacent repeated 1-3 letter sequence that the reference does not contain."""
    ref_words = set(normalize(reference).split())
    return [w for w in normalize(hypothesis).split() if DOUBLED.search(w) and w not in ref_words]


def streamed(path) -> np.ndarray:
    samples, rate, channels = read_wav(path)
    resampler, step = StreamResampler(rate, channels), CHUNK_SIZE * channels
    return np.concatenate([resampler.process(samples[i:i + step])
                           for i in range(0, len(samples), step)])


def main():
    parser = argparse.ArgumentParser(description="Live audio path vs benchmark path.")
    parser.add_argument("--edges", default="0.2/0.1",
                        help="comma-separated pre-roll/tail settings in seconds, e.g. 0.2/0.1,0.3/0.2")
    args = parser.parse_args()
    edges = {f"live {e}": dict(zip(("pre_roll_s", "tail_keep_s"), map(float, e.split("/"))))
             for e in args.edges.split(",")}

    from app.asr import NvidiaOnnxASR
    refs = read_tsv(REFERENCES_HY)
    asr = NvidiaOnnxASR()
    paths = ("whole", "stream", *edges)
    rows = []
    for file, ref in refs.items():
        whole, stream = load_wav_16k(AUDIO / file), streamed(AUDIO / file)
        n = min(len(whole), len(stream)) - 20
        audio_diff = float(np.max(np.abs(whole[:n] - stream[10:n + 10])))  # 10-sample delay (D23)
        hyps = {"whole": asr.transcribe(whole), "stream": asr.transcribe(stream)}
        n_phrases = []
        for name, setting in edges.items():
            phrases = detect_phrases({"mic": AUDIO / file}, setting)
            hyps[name] = " ".join(asr.transcribe(p.audio) for p in phrases)
            n_phrases.append(str(len(phrases)))
        rows.append(dict(file=file, ref=ref, hyps=hyps, phrases="/".join(n_phrases),
                         audio_diff=audio_diff))
        print(f"{file:14} phrases {'/'.join(n_phrases)}  audio diff {audio_diff:.1e}  " + "  ".join(
            f"{p}: {len(doubled_words(hyps[p], ref))} doubled" for p in paths))

    out = [f"# Live audio path vs benchmark path ({len(rows)} evaluation clips)", "",
           "live a/b = pre-roll a s / tail b s; phrase counts per clip are listed in that order.", "",
           "| Path | WER all | WER mic | WER video | CER | Doubled words |", "|---|---|---|---|---|---|"]
    for p in paths:
        def wer(group):
            sel = [r for r in rows if r["file"].startswith(group)]
            return jiwer.wer([normalize(r["ref"]) for r in sel], [normalize(r["hyps"][p]) for r in sel])
        cer = jiwer.cer([normalize(r["ref"]) for r in rows], [normalize(r["hyps"][p]) for r in rows])
        doubled = sum(len(doubled_words(r["hyps"][p], r["ref"])) for r in rows)
        out.append(f"| {p} | {100 * wer(''):.1f} % | {100 * wer('me_'):.1f} % | "
                   f"{100 * wer('video_'):.1f} % | {100 * cer:.1f} % | {doubled} |")
    out += ["", f"Largest whole vs stream audio difference (after the 10-sample delay): "
                f"{max(r['audio_diff'] for r in rows):.1e}", "", "## Per clip", ""]
    for r in rows:
        out.append(f"**{r['file']}** ({r['phrases']} phrase(s))  \nREF: {r['ref']}  ")
        for p in paths:
            d = doubled_words(r["hyps"][p], r["ref"])
            out.append(f"{p}: {r['hyps'][p]}" + (f"  — doubled: {', '.join(d)}" if d else "") + "  ")
        out.append("")
    target = ROOT / "results" / "asr_paths.md"
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"Report: {target}")


if __name__ == "__main__":
    main()
