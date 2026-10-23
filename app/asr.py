"""Speech-to-text models. Every model class has the same interface:

    model = SomeASR(...)               # loads the model (slow, once)
    text = model.transcribe(audio)     # audio: 16 kHz mono float32 numpy array -> Armenian text
"""
import numpy as np

from app.audio_utils import TARGET_RATE


def _check_audio(audio: np.ndarray) -> None:
    if audio.dtype != np.float32 or audio.ndim != 1:
        raise ValueError(f"Expected 1-D float32 audio at {TARGET_RATE} Hz, "
                         f"got dtype={audio.dtype}, shape={audio.shape}")


class NvidiaOnnxASR:
    """NVIDIA FastConformer Armenian (CTC head), ONNX export, run with onnx-asr (DECISIONS.md D12).

    quantization=None   -> model.onnx       (458 MB, full precision)
    quantization="int8" -> model.int8.onnx  (132 MB, smaller/faster, possibly less accurate)
    """

    REPO_ID = "OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx"

    def __init__(self, quantization: str | None = None):
        import onnx_asr  # imported here so other models don't require this library

        self.name = "nvidia-onnx" + (f"-{quantization}" if quantization else "")
        self._model = onnx_asr.load_model(self.REPO_ID, quantization=quantization)

    def transcribe(self, audio: np.ndarray) -> str:
        _check_audio(audio)
        text = self._model.recognize(audio, sample_rate=TARGET_RATE)
        if not isinstance(text, str):
            raise TypeError(f"onnx-asr returned {type(text).__name__}, expected str")
        return text


class WhisperASR:
    """Whisper large-v3-turbo fine-tuned on Armenian (chillarmo), run with Hugging Face Transformers
    on CPU (DECISIONS.md D12). Benchmark comparison only; needs requirements-benchmark.txt.
    Language is fixed to Armenian so the model never has to guess it.
    """

    REPO_ID = "chillarmo/whisper-large-v3-turbo-armenian"

    def __init__(self):
        import torch  # imported here so the main program does not require PyTorch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.name = "whisper-turbo-hy"
        self._torch = torch
        self._processor = WhisperProcessor.from_pretrained(self.REPO_ID)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.REPO_ID)
        self._model.eval()  # inference mode: no training behaviour (e.g. dropout)

    def transcribe(self, audio: np.ndarray) -> str:
        _check_audio(audio)
        # 1. audio -> log-mel spectrogram features (what Whisper actually reads)
        features = self._processor(audio, sampling_rate=TARGET_RATE, return_tensors="pt").input_features
        # 2. generate text tokens one by one, Armenian, transcription (not translation)
        with self._torch.inference_mode():
            token_ids = self._model.generate(features, language="hy", task="transcribe")
        # 3. tokens -> text
        return self._processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
