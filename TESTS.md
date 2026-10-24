# TESTS — checks derived from the challenge requirements

Status: ⬜ not run · ✅ pass · ❌ fail. Update the status and date when a check is run.

## A. Required features (from the challenge brief)
| # | Check | How | Status |
|---|---|---|---|
| A1 | Mic and system audio are captured **at the same time** | Run program, speak while a video/call plays → both sides appear | ⬜ (M1 capture test ✅ 2026-10-23) |
| A2 | User can **choose input devices** | `--list` prints devices; `--mic` / `--system` select them; wrong device type fails with a clear error | ⬜ |
| A3 | Spoken Armenian → **Armenian text** | Live test + benchmark results | ✅ 2026-10-23 benchmark WER 23.8 % (mic 4.8 %, video 32.4 %); live M3 test OK |
| A4 | Each segment → **English translation** | Live test: English line present and understandable | ✅ 2026-10-23 M3 live test (NLLB chrF 62.5) |
| A5 | Mic is always labelled **Me** | Speak alone → only "Me" | ✅ 2026-10-23 M3 (single source) — recheck with both sources in M4 |
| A6 | Call side: different people → **Speaker 1, Speaker 2…** | Test call with ≥ 2 friends | ✅ 2026-10-24 two-speaker test (friend A, friend B through Meet): A vs B similarity ≤ 0.26, **0 merges** at all thresholds; conversation replay: A = Speaker 1, B = Speaker 2, confirmed by user. ⬜ live with 2 friends |
| A7 | Same person keeps the **same label** all session | ≥ 5 min call, each person speaks several times | ✅ replays 2026-10-24 (call 1: 28 phrases one label; call 2: labels consistent). ⚠️ one outlier phrase of friend A (similarity −0.02) under investigation |
| A8 | Output shows **time, speaker, Armenian, English** | Visual check of terminal | ✅ 2026-10-23 M3 |

## B. Constraints
| # | Check | How | Status |
|---|---|---|---|
| B1 | Latency **≤ 5 s** from end of phrase to English line | Measured and logged per segment (median and worst case) | ✅ single source 2026-10-23: mic 1.1–1.3 s, YouTube 2.2–2.7 s. ✅ both sources 2026-10-24 (mic + 2-person podcast): Me 1.6 s, call side 3.2–3.3 s, max 3.3 s (5 phrases — repeat longer in rehearsal) |
| B2 | Runs **fully offline** (no cloud APIs) | Disconnect from the internet after models are downloaded → program still works | ✅ 2026-10-23 Wi-Fi off, mic source: models loaded, no HF warning, transcription + translation OK. ⬜ recheck after speaker model is added |
| B3 | Fits in **8 GB RAM** while a call app is running | Task Manager during a live test | ⬜ |
| B4 | No paid tools/services used | Review dependencies list | ⬜ |
| B5 | Runs on **this laptop** (Windows 11, CPU only) | Full rehearsal before the interview | ⬜ |

## C. Text normalization (D11) — the scoring itself must be correct
| # | Input | Expected output | Status |
|---|---|---|---|
| C1 | `քեզ դուր եկա՞վ։` | `քեզ դուր եկավ` | ⬜ |
| C2 | `Ողջույն, ինչ ուրախ եմ։` | `ողջույն ինչ ուրախ եմ` | ⬜ |
| C3 | `սիրելի ռադիունկնդիրներ և դիտողներ` | `սիրելի ռադիունկնդիրներ եւ դիտողներ` | ⬜ |
| C4 | `ենք   ու` (extra spaces) | `ենք ու` | ⬜ |
| C5 | `թանկացել ա` (colloquial) | unchanged: `թանկացել ա` | ⬜ |
| C6 | `30 տարեկան` (digits) | unchanged: `30 տարեկան` | ⬜ |
| C7 | WER of identical texts = 0; WER of `քեզ դուր էր եկել` vs ref `քեզ դուր եկա՞վ։` > 0 | | ⬜ |

## D. Evaluation set
| # | Check | Status |
|---|---|---|
| D1 | 15 clips in `eval/audio`, each has exactly one reference in `eval/references.tsv` (Tab-separated, UTF-8) | ✅ 2026-10-23 |
| D2 | Clip levels: no silent or clipped clips | ✅ 2026-10-23 (mic avg −31…−39 dBFS, videos −19…−27 dBFS, max peak −2.5 dBFS) |

## E. Robustness (to be extended as issues are found)
| # | Check | Status |
|---|---|---|
| E1 | Very long phrase without a pause is cut at the maximum length, not lost | ✅ unit test (tests/test_vad.py); ⬜ live |
| E2 | Silence on one source does not block the other source | ⬜ |
| E3 | Unplugging/replugging headphones: clear error, documented in README | ⬜ |

## F. Before every commit
- `git status` → no `.wav`, model files, `.venv` or secrets listed.

## G. Observations and known limitations (collected for the README)
| Date | Observation |
|---|---|
| 2026-10-23 | M3 live: each sentence appeared as one block with the 0.6 s end-of-phrase setting |
| 2026-10-23 | Foreign proper names are not recognised (e.g. "Sherlock Holmes" in Armenian speech) — the ASR model is Armenian-only |
| 2026-10-23 | Russian words mixed into Armenian speech (common in everyday Armenian) are not recognised correctly — Armenian-only model; non-Armenian input is out of scope in the brief |
| 2026-10-23 | Translation can be fluent but wrong (D22) |
| 2026-10-23 | Program contacts Hugging Face at start-up ("unauthenticated requests" warning) → must work offline (B2) |
| 2026-10-24 | Replay of real call: proper names / rare words mis-recognised (Ամուլսար → "ամյուսարի" → "Autumn mine"; COP 17 → "17 cup") |
| 2026-10-24 | Very short phrases translate poorly without context (`Այո` → "Yes, I did."; `Եվ` → "And then.") |
| 2026-10-24 | Breaths/"hm" recognised as fragments (`Հ.`, `Հովհ.`) and translated into invented sentences → D29 confidence filter |
| 2026-10-24 | No duplicated lines between Me and the call side with headphones (D3 works) |
