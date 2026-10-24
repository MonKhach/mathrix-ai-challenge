# HANDOVER — final state

Updated 2026-10-24 19:57 at the end of chat session 2 (M5 + M6). Previous version (end of session 1,
before M5) is in Git history.
**New chat: read this file, then `CLAUDE.md`, `README.md`, `DECISIONS.md`, `TESTS.md`.** Follow the
working rules in CLAUDE.md (user decides; explain options first; step by step; fail loudly).

## Goal
Python terminal program, Windows 11 laptop (i5-1135G7, 8 GB RAM, no GPU, Python 3.11): captures
microphone ("Me") + system audio (call side), transcribes **Eastern Armenian**, translates to
**English**, labels call-side speakers **Speaker 1, 2…**, ≤ 5 s latency, **all local, offline,
no paid tools**. Deadline **2026-11-01 23:59 (Yerevan)**. Interview: live call with ≥ 2 people on
the call side, code walk-through, original chat history.

## State: M1–M5 done, M6 almost done
Code, README and all tests are on GitHub (`MonKhach/mathrix-ai-challenge`, branch `master`).
Every TESTS.md row is ✅ (B3 RAM: fits but tight). **Only open item: export the chats (below).**

Pipeline (one loop, D25): WASAPI capture (PyAudioWPatch) → StreamResampler 48 kHz stereo → 16 kHz
mono → Silero VAD phrases (end after 0.6 s silence; after 6 s a 0.15 s pause ends the phrase;
max 10 s — D32) → NVIDIA FastConformer Armenian (ONNX, fp32) → abbreviation-artifact rule (D31) →
NLLB-200 600M (CTranslate2 int8; output ≤ 2 × input + 10 tokens, no repeated 4-grams — D33) →
speaker label (mic = Me; call side: WeSpeaker embedding + online clustering, 0.35, short phrases
→ closest speaker — D27) → terminal (+ `--log`). Lost audio device → clear `ERROR`, exit (D34).

## What session 2 did (M5 + M6)
| Step | Result | Decision |
|---|---|---|
| RAM (B3) | program 1.6–1.9 GB; PC 88–96 % used with Meet + camera, stable; closing apps did not help | fp32 kept (D17) |
| Latency | long phrases (11–15 s) of fast speakers gave 4.8–7.1 s → no backlog, translation time grows with length | D32 shorter phrases |
| Translation loop | "you want to go" ×50, 27.8 s for one line; 6 settings compared (chrF, real repetitions, 132 lines, time) | D33 cap + no-repeat 4 (penalty rejected: dropped half-sentences) |
| Offline / silence (B2, E2) | pass with all 4 models | — |
| Unplugged headphones (E3) | call side went silent with no message → now clear error | D34 (no auto-reconnect) |
| Doubled syllables | same in whole-file, streamed and live paths → model property; phrase cutting costs ~2 WER points | D35 (documented) |
| README, TESTS.md, GitHub | written, all rows recorded, pushed; no audio/models/private files in history | — |

Live results: friends' call (5 min) median 2.1–2.4 s, max 4.7 s; rehearsal (6 min, 65 lines)
median 2.3–2.8 s, 2 long phrases 5.3 / 5.8 s with a busy CPU (documented, settings unchanged);
speaker labels correct with 2 friends; no translation loops. A 30-min run was not possible.

## Commands
```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main --list-devices                    # indices can change! mic 12, loopback 13 (Oct 24)
python -m app.main --mic 12 --system 13 [--debug] [--log private/x.txt]
python -m pytest -v                                  # all tests must pass
python -m evaluation.replay --system eval/audio/call2_system_mixed_2.wav --mic eval/audio/call2_mic.wav
python -m evaluation.translation_settings --logs private/*.txt   # translation settings comparison (D33)
python -m evaluation.asr_paths --edges 0.2/0.1,0.4/0.3            # audio path comparison (D35)
```
Git from PowerShell: one command per line (`git commit …` and `git push` on one line commits nothing).

## Before the interview
- Wired headphones; call app output on them; check `--list-devices`; close heavy apps.
- If the headphones are unplugged: plug back in, switch the call app's output back, restart.
- Be ready to explain: model choices with numbers (README, D19/D22/D26), D32–D35 (measure first,
  then decide), known limitations (README).

## Open
1. **Export every Cowork chat in full** (session 1 and session 2), each as its own `.md` in
   `chats/`; check for private content, commit and push. Keep the originals available.

## Never commit
`eval/audio/` recordings, `models/`, `.venv/`, `private/` transcripts and logs, keys/tokens.
