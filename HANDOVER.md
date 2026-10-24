# HANDOVER — Mathrix AI challenge (live Armenian → English, speaker labels)

Written 2026-10-24 13:05 at the end of chat session 1, before M5/M6.
**New chat: read this file, then `CLAUDE.md`, `PLAN.md`, `DECISIONS.md`, `TESTS.md`.** Follow the
working rules in CLAUDE.md (user decides; explain options first; step by step; fail loudly).

## Goal
Python terminal program, Windows 11 laptop (i5-1135G7, 8 GB RAM, no GPU, Python 3.11): captures
microphone ("Me") + system audio (call side), transcribes **Eastern Armenian**, translates to
**English**, labels call-side speakers **Speaker 1, 2…**, ≤ 5 s latency, **all local, offline,
no paid tools**. Deadline **2026-11-01 23:59 (Yerevan)**. Interview: live call with ≥ 2 people on
the call side, code walk-through, original chat history.

## Current state: M1–M4 done
Pipeline (one loop, D25): WASAPI capture (PyAudioWPatch) → StreamResampler 48 kHz stereo → 16 kHz
mono → Silero VAD phrase detector (0.6 s end, 15 s max, 0.3 s min) → NVIDIA FastConformer Armenian
(ONNX, onnx-asr 0.12.0 pinned) → abbreviation-artifact rule → NLLB-200 distilled 600M
(CTranslate2 int8) → speaker label (mic = Me; call side: WeSpeaker ResNet34 embedding + online
clustering, threshold 0.35, short phrases < 1.5 s get the closest known speaker) → terminal (+ log).

| Module | Role |
|---|---|
| `app/audio_capture.py` | device list, WASAPI capture → queue of 16 kHz chunks, loud errors |
| `app/audio_utils.py` | `to_mono_16k`, `load_wav_16k`, `StreamResampler` (= resample_poly, delayed 10 samples) |
| `app/vad.py` | `SileroModel` (ONNX), `PhraseDetector` (incl. idle end for silent loopback) |
| `app/asr.py` | `NvidiaOnnxASR` (+ `transcribe_with_confidence`), `WhisperASR` (benchmark only) |
| `app/translator.py` | `NllbTranslator`, `OpusMTTranslator` (CTranslate2) |
| `app/speakers.py` | `SpeakerEmbedder`, `SpeakerTracker`, settings (0.35 / 1.5 s / assign) |
| `app/pipeline.py` | per-phrase processing shared by live + replay; fragment rule (D31) |
| `app/output.py` | line formatting |
| `app/main.py` | CLI, model loading, offline mode, main loop, `--log`, `--debug` |
| `evaluation/` | metrics (WER/CER, D11), benchmark_asr, benchmark_translation, compare_translations, speaker_eval, replay (+ `--sweep`), two_speaker_test |
| `scripts/` | test_audio_capture (M1), record_clip, convert_translation_models |
| `tests/` | ~65 pytest tests (metrics, audio, VAD, output, speakers, pipeline) — all passing |

## Commands
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                      # + requirements-benchmark.txt for Whisper/model conversion
python scripts/convert_translation_models.py         # once (needs torch from requirements-benchmark.txt)
python -m app.main --download                        # once, with internet
python -m app.main --list-devices                    # indices can change! mic 12, headphones loopback 13 (Oct 24)
python -m app.main --mic 12 --system 13 [--debug] [--log private/x.txt]
python -m pytest -v
python -m evaluation.replay --system eval/audio/call2_system_mixed_2.wav --mic eval/audio/call2_mic.wav
```
Prerequisites found the hard way: **latest Microsoft Visual C++ Redistributable** (D15);
**headphones** (D3); program writes UTF-8 itself — do not pipe through Tee-Object (D30).

## Key results (details in DECISIONS.md / results/)
- ASR (D19): NVIDIA WER 23.8 % (mic 4.8 %, video 32.4 %), CER 12.1 %, ≤ 0.24 s/clip, 715 MB.
  Whisper-turbo-hy: WER 48.5 %, 25–85 s/clip, 3.65 GB, repetition hallucination. NVIDIA int8: same
  accuracy, 445 MB, slower → **fp32 vs int8 to decide at M5 from total RAM (D17)**.
- Translation (D22): NLLB chrF 62.5 / 62.6 (from ASR output) vs OPUS-MT 52.5 / 43.6; 1.24 s avg,
  2.15 s worst; risk: fluent but wrong.
- Speakers (D26, D27): YouTube 8 people + real Meet calls: different people ≤ 0.28 similarity,
  0 merges at 0.35 with two real friends; short phrases unreliable → `assign` rule.
- Latency (B1): mic 1.0–1.6 s; call side 2.2–3.3 s; max 3.3 s with both sides (small sample).
- Confidence filter (D29) measured and NOT used; fragment rule (D31) instead.

## Open items / next steps
**M5 — measure & improve weakest link**
1. B3: RAM of the full program during a call (Task Manager) → decide fp32 vs int8 (D17).
2. Longer live latency run with both sides (≥ 5 min), median and worst case.
3. B2 again (Wi-Fi off) now that the speaker model is included; E2/E3 (silence on one source,
   unplugging headphones); stability over a longer session.
4. Then improve only the weakest link (currently: recognition of natural call-side speech and
   short-phrase translation are the biggest quality limits; speakers and latency look good).

**M6 — deliver**
5. README: setup incl. VC++ redistributable, model download/conversion, audio device setup,
   models + alternatives with the measured numbers, architecture, known limitations (TESTS.md G),
   timestamp order (D28), NLLB licence (CC-BY-NC), what to improve with more time.
6. Run all of TESTS.md; live call with 2 friends (A6 live); full interview rehearsal.
7. Push to GitHub (check no audio/models/private files); **export EVERY Cowork chat in full as its
   own .md into `chats/`** (this first chat included); keep originals available.

## Known limitations so far (TESTS.md section G)
Proper names / rare words (Sherlock Holmes, Ամուլսար, COP 17); Russian words in Armenian speech;
short colloquial words merged (`մամ`); very short phrases translate poorly; fluent-but-wrong
translations; noise could create an extra speaker; Western Armenian not targeted (D4); evaluation
set is small (15 clips + 2 recorded calls).

## Never commit
`eval/audio/` recordings, `models/`, `.venv/`, `private/` transcripts, keys/tokens.
