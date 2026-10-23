"""One-time setup: download the translation models and convert them to CTranslate2 int8.

Needs transformers + torch for the conversion only:
    pip install -r requirements-benchmark.txt
    python scripts/convert_translation_models.py
The converted models are written to models/ (git-ignored). After this, translation runs without torch.
"""
import faulthandler
import sys
from pathlib import Path

import ctranslate2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.translator import MODELS_DIR, NllbTranslator, OpusMTTranslator  # noqa: E402

faulthandler.enable()

for model in (OpusMTTranslator, NllbTranslator):
    output_dir = MODELS_DIR / model.local_dir
    if (output_dir / "model.bin").exists():
        print(f"{model.name}: already converted in {output_dir}")
        continue
    print(f"{model.name}: downloading {model.hf_name} and converting to int8 (may take minutes)...")
    ctranslate2.converters.TransformersConverter(model.hf_name).convert(
        str(output_dir), quantization="int8")
    print(f"{model.name}: done -> {output_dir}")
