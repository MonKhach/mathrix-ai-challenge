# TESTS — checks derived from the challenge requirements

Status: ⬜ not run · ✅ pass · ❌ fail. Update the status and date when a check is run.

## A. Required features (from the challenge brief)
| # | Check | How | Status |
|---|---|---|---|
| A1 | Mic and system audio are captured **at the same time** | Run program, speak while a video/call plays → both sides appear | ⬜ (M1 capture test ✅ 2026-10-23) |
| A2 | User can **choose input devices** | `--list` prints devices; `--mic` / `--system` select them; wrong device type fails with a clear error | ⬜ |
| A3 | Spoken Armenian → **Armenian text** | Live test + benchmark results | ⬜ |
| A4 | Each segment → **English translation** | Live test: English line present and understandable | ⬜ |
| A5 | Mic is always labelled **Me** | Speak alone → only "Me" | ⬜ |
| A6 | Call side: different people → **Speaker 1, Speaker 2…** | Test call with ≥ 2 friends | ⬜ |
| A7 | Same person keeps the **same label** all session | ≥ 5 min call, each person speaks several times | ⬜ |
| A8 | Output shows **time, speaker, Armenian, English** | Visual check of terminal | ⬜ |

## B. Constraints
| # | Check | How | Status |
|---|---|---|---|
| B1 | Latency **≤ 5 s** from end of phrase to English line | Measured and logged per segment (median and worst case) | ⬜ |
| B2 | Runs **fully offline** (no cloud APIs) | Disconnect from the internet after models are downloaded → program still works | ⬜ |
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
| E1 | Very long phrase without a pause is cut at the maximum length, not lost | ⬜ |
| E2 | Silence on one source does not block the other source | ⬜ |
| E3 | Unplugging/replugging headphones: clear error, documented in README | ⬜ |

## F. Before every commit
- `git status` → no `.wav`, model files, `.venv` or secrets listed.
