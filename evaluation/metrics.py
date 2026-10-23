"""Text normalization and error rates for scoring Armenian transcriptions.

Rules (DECISIONS.md D11), applied identically to the reference and the model output:
  1. remove punctuation
  2. lowercase
  3. 'և' -> 'եւ' (same word, two spellings)
  4. collapse whitespace
NOT normalized on purpose: colloquial vs literary forms, digits vs number words.
"""
import unicodedata

import jiwer

# Armenian stress/exclamation/question marks are written *inside* a word (e.g. եկա՞վ),
# and a hyphen (- or Armenian ֊) joins one word (e.g. 30-ին). These are deleted without leaving a space.
IN_WORD_MARKS = {"\u055B", "\u055C", "\u055E", "-", "\u058A"}  # ՛ ՜ ՞, hyphen -, Armenian hyphen ֊


def normalize(text: str) -> str:
    """Apply the D11 rules. Every other punctuation character becomes a space."""
    text = text.lower().replace("և", "եւ")
    chars = []
    for ch in text:
        if ch in IN_WORD_MARKS:
            continue
        if unicodedata.category(ch).startswith("P"):  # Unicode category "P*" = punctuation
            chars.append(" ")
        else:
            chars.append(ch)
    return " ".join("".join(chars).split())


def collapse_spaces(text: str) -> str:
    """The only cleaning used for the *raw* WER."""
    return " ".join(text.split())


def score(reference: str, hypothesis: str) -> dict:
    """Compare one model output with its reference. Fails loudly on an empty reference."""
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)
    if not ref_norm:
        raise ValueError(f"Empty reference after normalization: {reference!r}")

    words = jiwer.process_words(ref_norm, hyp_norm)
    return {
        "raw_wer": jiwer.wer(collapse_spaces(reference), collapse_spaces(hypothesis)),
        "wer": words.wer,
        "cer": jiwer.cer(ref_norm, hyp_norm),
        "ref_words": words.hits + words.substitutions + words.deletions,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
    }
