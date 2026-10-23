"""Armenian -> English translation with CTranslate2 (DECISIONS.md D21).

Every translator has the same interface:
    translator = SomeTranslator(...)        # loads the model (slow, once)
    english = translator.translate(text)    # Armenian text -> English text

Models must be converted once with:  python scripts/convert_translation_models.py
"""
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class _CT2Translator:
    """Shared logic: text -> tokens -> CTranslate2 -> tokens -> text."""

    hf_name: str        # original Hugging Face model (used for its tokenizer)
    local_dir: str      # converted CTranslate2 model inside models/
    tokenizer_kwargs: dict = {}
    target_prefix: list | None = None  # NLLB needs the target language as the first token

    def __init__(self):
        model_path = MODELS_DIR / self.local_dir
        if not (model_path / "model.bin").exists():
            raise FileNotFoundError(f"Converted model not found in {model_path}. "
                                    f"Run: python scripts/convert_translation_models.py")
        self._translator = ctranslate2.Translator(str(model_path), device="cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_name, **self.tokenizer_kwargs)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        tokens = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))
        prefix = [self.target_prefix] if self.target_prefix else None
        result = self._translator.translate_batch([tokens], target_prefix=prefix)
        output_tokens = result[0].hypotheses[0]
        if self.target_prefix:
            output_tokens = output_tokens[len(self.target_prefix):]  # drop the language token
        return self._tokenizer.decode(self._tokenizer.convert_tokens_to_ids(output_tokens),
                                      skip_special_tokens=True).strip()


class OpusMTTranslator(_CT2Translator):
    """Helsinki-NLP OPUS-MT hy->en (Marian, Apache-2.0)."""
    name = "opus-mt"
    hf_name = "Helsinki-NLP/opus-mt-hy-en"
    local_dir = "opus-mt-hy-en-ct2-int8"


class NllbTranslator(_CT2Translator):
    """Meta NLLB-200 distilled 600M (CC-BY-NC-4.0, non-commercial). Armenian = hye_Armn."""
    name = "nllb-600m"
    hf_name = "facebook/nllb-200-distilled-600M"
    local_dir = "nllb-200-distilled-600M-ct2-int8"
    tokenizer_kwargs = {"src_lang": "hye_Armn"}
    target_prefix = ["eng_Latn"]
