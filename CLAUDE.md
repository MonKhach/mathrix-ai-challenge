# Project rules (read first in every new chat)

## Context
- Mathrix AI coding challenge: live Armenian speech → Armenian text → English translation, labelled by speaker, in the terminal, ≤ 5 s latency.
- Read `PLAN.md` and `DECISIONS.md` before doing anything.
- Machine: Windows 11, Intel i5-1135G7, 8 GB RAM, no GPU (CPU only), Python 3.11.

## How we work
- The user makes the decisions. Before implementing, explain options, pros, cons and risks in simple language, and explain technical terms.
- Work step by step; wait for the user's confirmation before moving to the next stage.
- Change one thing at a time; suggest a Git commit after each working step.

## Hard rules
- All speech recognition, translation and speaker identification run **locally**. No cloud APIs.
- **No paid tools, services or APIs.** If something paid seems essential, stop and ask.
- Eastern Armenian only (no Western Armenian optimization in this task).
- Never commit model weights, audio recordings, API keys or tokens.
- Verify that models and libraries actually exist (and cite the source) before relying on them.
- Measure speed and memory — do not assume them.

## Code style
- Simple, readable code. One module per responsibility: audio capture, segmenter (VAD), asr, translator, speaker_id, output, main.
- Fail loudly and clearly: no silent fallbacks, hidden retries or swallowed exceptions.
- No extra features beyond the requirements.
