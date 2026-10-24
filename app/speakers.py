"""Speaker identification for the call side (DECISIONS.md D26).

SpeakerEmbedder : phrase audio -> "voice fingerprint" (embedding vector, length 1)
SpeakerTracker  : online clustering -> "Speaker 1", "Speaker 2", ... kept for the whole session

The microphone side never goes through this: it is always "Me".
"""
import numpy as np

from app.audio_utils import TARGET_RATE

SPEAKER_REPO_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"

# Chosen from measurements (DECISIONS.md D27): offline YouTube test (results/speakers.md) and
# replay of a real call (results/replay_sweep_*.md). Can be changed from the command line.
SIMILARITY_THRESHOLD = 0.35
SHORT_PHRASE_S = 1.5      # phrases shorter than this give unreliable fingerprints
SHORT_RULE = "assign"     # "assign": closest known speaker, no new speaker, no update
                          # "unknown": label "Speaker ?"   "none": treat like any phrase


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Embedding has zero or invalid length")
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = same direction (same voice), ~0 = unrelated."""
    return float(np.dot(_unit(a), _unit(b)))


class SpeakerEmbedder:
    """WeSpeaker ResNet34-LM (ONNX) through onnx-asr's speaker-embedding support (pinned 0.12.0)."""

    def __init__(self):
        from onnx_asr.loader import Manager  # internal API, hence the version pin

        self._model = Manager().create_se(SPEAKER_REPO_ID)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32 or audio.ndim != 1:
            raise ValueError(f"Expected 1-D float32 audio at {TARGET_RATE} Hz, "
                             f"got dtype={audio.dtype}, shape={audio.shape}")
        embedding = np.asarray(self._model.embedding(audio, sample_rate=TARGET_RATE), dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError(f"Expected one embedding vector, got shape {embedding.shape}")
        return _unit(embedding)


class SpeakerTracker:
    """Online clustering. Each known speaker is represented by the average of their embeddings.

    A new phrase gets the label of the most similar known speaker if the similarity is at least
    `threshold`; otherwise it becomes a new speaker.
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._sums: list[np.ndarray] = []  # sum of unit embeddings per speaker (direction = average)

    @property
    def speaker_count(self) -> int:
        return len(self._sums)

    def _best_match(self, e: np.ndarray) -> tuple[int | None, float | None]:
        best_index, best_similarity = None, None
        for index, total in enumerate(self._sums):
            similarity = cosine_similarity(e, total)
            if best_similarity is None or similarity > best_similarity:
                best_index, best_similarity = index, similarity
        return best_index, best_similarity

    def closest(self, embedding: np.ndarray) -> tuple[str, float]:
        """Label of the most similar known speaker, WITHOUT creating or updating anyone
        (used for short phrases, whose fingerprints are unreliable)."""
        if not self._sums:
            raise ValueError("No known speakers yet")
        best_index, best_similarity = self._best_match(_unit(embedding))
        return f"Speaker {best_index + 1}", best_similarity

    def assign(self, embedding: np.ndarray) -> tuple[str, float | None]:
        """Returns (label, similarity to the best known speaker or None for the very first)."""
        e = _unit(embedding)
        best_index, best_similarity = self._best_match(e)

        if best_similarity is not None and best_similarity >= self.threshold:
            self._sums[best_index] += e  # update this speaker's average voice
            return f"Speaker {best_index + 1}", best_similarity

        self._sums.append(e.copy())
        return f"Speaker {len(self._sums)}", best_similarity
