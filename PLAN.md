# PLAN — Live Armenian Transcription → English Translation

## 1. Goal
A Python terminal program (Windows) that captures the **microphone** and **system audio** at the same time, transcribes spoken **Eastern Armenian**, translates it into **English**, and labels each segment **Me / Speaker 1 / Speaker 2…** — each segment shown **≤ 5 seconds** after the end of a phrase. Everything runs **locally** on a Dell Vostro 3510 (i5-1135G7, 8 GB RAM, no GPU).

## 2. Scope
**In scope:** the 5 required features, device selection from the command line, README.
**Out of scope:** GUI, installer, echo handling, speaker names, memory across sessions, saving transcripts, other input languages, optimization for Western Armenian.

**Hard constraints:** all models run locally (no cloud APIs); no paid tools, services or APIs.

## 3. Modules (inputs → outputs)

| Module | Input | Output |
|---|---|---|
| `audio_capture` | Selected devices (mic, system) | Continuous stream of small audio chunks, tagged with **source** (`mic`/`system`) and **timestamp** |
| `segmenter` (VAD) | Audio stream of one source | Complete **phrases**: audio + source + start/end time |
| `asr` (speech-to-text) | Audio of one phrase | Armenian text |
| `translator` | Armenian text | English text |
| `speaker_id` | Audio of one phrase + source | Label: `Me` (mic) or `Speaker N` (system) |
| `output` | Time, label, Armenian, English | Printed lines in the terminal |
| `main` | CLI flags (`--list-devices`, `--mic`, `--system`) | Wires the modules together and runs them |
| `benchmark` (separate tool) | Test recordings + correct text | Accuracy and speed table per model |

Rule: each module only knows its own input and output, so components can be swapped independently.

## 4. Audio format
All models expect **16 kHz mono**. System audio usually arrives as 48 kHz stereo, so it is converted once, right after capture.

## 5. Latency budget (targets — to be verified by measurement)

| Step | Target |
|---|---|
| VAD detects end of phrase | ≤ 0.8 s |
| Speech-to-text | ≤ 2.5 s |
| Translation | ≤ 1.0 s |
| Speaker identification | ≤ 0.3 s |
| **Total** | **≤ 5 s (with a small safety margin)** |

## 6. Main risks

| Risk | Plan |
|---|---|
| CPU too slow for a good Armenian model | Benchmark early; accept a smaller/faster model and document the trade-off |
| 8 GB RAM not enough for all models + call app | Measure memory during the benchmark; prefer compact/optimized versions |
| Poor Armenian accuracy | Compare candidates by error rate; be honest in the README |
| WASAPI loopback misbehaves on this laptop | Test on day 1; switch to VB-Cable as a documented, deliberate decision (not a hidden automatic fallback) |
| Both sources speaking at once slows everything | Measure and handle processing order explicitly |
| Speaker labels mixed up (short phrases, similar voices) | Tune similarity threshold on real test calls; skip phrases that are too short |
| Very long phrases with no pause | Maximum phrase length |
| Running out of time | Walking skeleton first, then improve only the weakest link |

## 7. Testing
- **Evaluation set:** ~10–20 short Armenian clips (own voice + at least 2 other people / online videos), each with its correct text written out. Audio is **not** committed to Git.
- **Metrics:** WER (word error rate), CER (character error rate), latency (end of phrase → printed line), RAM usage, speaker-label accuracy on a test call.
- **TESTS.md:** checklist derived directly from the challenge requirements.

## 8. Milestones

| # | Milestone | When |
|---|---|---|
| M0 | Planning docs, project structure, Git, Python environment | Fri morning |
| M1 | WASAPI loopback + mic capture tested (save 5 s from each, listen) | Fri morning |
| M2 | Evaluation set recorded, speech-to-text benchmark done → model chosen | Fri morning/midday |
| M3 | **Walking skeleton:** one source → Armenian → English → terminal | Fri afternoon |
| M4 | Both sources + speaker identification | Sat morning |
| M5 | Latency measured, weakest link improved | Sat afternoon |
| M6 | README, run TESTS.md, real call test, export chats | Sat / before Nov 1 (Sunday as buffer) |

## 9. Working rules
- One module per chat when complex; short handover when a chat gets long.
- Fail **loudly and clearly** — no silent fallbacks.
- Commit after every working step; change one thing at a time.
- Never commit model weights, audio recordings or keys.
- Verify every model/library exists before relying on it.
