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
| A6 | Call side: different people → **Speaker 1, Speaker 2…** | Test call with ≥ 2 friends | ✅ 2026-10-24 two-speaker test (friend A, friend B through Meet): A vs B similarity ≤ 0.26, **0 merges** at all thresholds; conversation replay: A = Speaker 1, B = Speaker 2, confirmed by user. ✅ live 2026-10-24 (Meet, 2 friends, 5 min): Speaker 1 = 18 lines, Speaker 2 = 16, labels correct (confirmed by user); one stray Speaker 3 on a short phrase (similarity 0.23) |
| A7 | Same person keeps the **same label** all session | ≥ 5 min call, each person speaks several times | ✅ replays 2026-10-24 (call 1: 28 phrases one label; call 2: labels consistent). ⚠️ one outlier phrase of friend A (similarity −0.02) under investigation |
| A8 | Output shows **time, speaker, Armenian, English** | Visual check of terminal | ✅ 2026-10-23 M3 |

## B. Constraints
| # | Check | How | Status |
|---|---|---|---|
| B1 | Latency **≤ 5 s** from end of phrase to English line | Measured and logged per segment (median and worst case) | ✅ single source 2026-10-23: mic 1.1–1.3 s, YouTube 2.2–2.7 s. ✅ both sources 2026-10-24 (mic + 2-person podcast): Me 1.6 s, call side 3.2–3.3 s, max 3.3 s (5 phrases — repeat longer in rehearsal). ❌ 2026-10-24 live with Meet + fast podcast: phrases of 11–15 s → 4.8–7.1 s (no backlog, translation of long phrases) → D32 (0.15 s pause after 6 s, max 10 s). ✅ D32 confirmed live 2026-10-24 (Meet, 2 friends, 5 min, 55 lines): Me median 2.4 s / max 4.7 s; call side median 2.1 s, max 3.6 s except **one 28.9 s translation loop** → D33 (now max 3.1 s offline for that line; ⬜ recheck live) |
| B2 | Runs **fully offline** (no cloud APIs) | Disconnect from the internet after models are downloaded → program still works | ✅ 2026-10-23 Wi-Fi off, mic source: models loaded, no HF warning, transcription + translation OK. ✅ 2026-10-24 recheck with all 4 models (Wi-Fi off, mic + local recordings on the call side): no network warning, transcription, translation and speaker labels OK |
| B3 | Fits in **8 GB RAM** while a call app is running | Task Manager during a live test + program's own memory report (`[memory, …]` lines, `app/memory.py`) | ⚠️ fits but tight, 2026-10-24 (fp32, Meet with camera + YouTube): program 1.7–1.9 GB (peak 1.93 GB), PC 0.3–0.8 GB available (90–96 % used), commit charge 13.1 / 14.3 GB, no crash. Closing other apps did not help (Chrome/Meet grew to 1.1 GB). 5-min friends' call: program 1.73–1.77 GB (peak 1.82 GB), stable, PC 0.3–0.7 GB available. **fp32 kept** (D17); ⬜ recheck in a ~30 min rehearsal |
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
| E1 | Very long phrase without a pause is cut at the maximum length, not lost | ✅ unit test (tests/test_vad.py); ✅ replay of podcast_long_02 (D32): cut at 10 s, nothing lost, but a word can be split at the cut |
| E4 | Translation never loops / runs away ("you want to go" ×50) | ✅ 2026-10-24 D33: 0 loops in 132 lines, worst 3.1 s; real repetitions kept (`evaluation/translation_settings.py`) |
| E2 | Silence on one source does not block the other source | ✅ 2026-10-24: call side silent → Me lines 1.5–1.6 s; mic silent → call-side lines 2.2–2.6 s |
| E3 | Unplugging/replugging headphones: clear error, documented in README | ❌ 2026-10-24 before fix: call side went **silent with no message**, stayed dead after replugging (mic kept working). ✅ after D34: `ERROR: Audio device stopped: Headphones … restart the program` right after unplugging, program exits (4 unit tests). ⬜ README |

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
| 2026-10-24 | Short colloquial words/interjections (e.g. `մամ`) are merged into a neighbouring word — rare in the ASR training data |
| 2026-10-24 | Noise before speech (friend's mic problem) gives a fingerprint unlike any voice (similarity −0.02); live pipeline only lets phrases with real text create speakers, but noise transcribed as a word could still create an extra speaker |
| 2026-10-24 | NLLB can loop on garbled input ("you want to go" ×50, 27.8 s) → D33 length cap + no repeated 4-token sequence |
| 2026-10-24 | Hard cut at 10 s (D32) can split a word for speakers who never pause (podcast host); not seen in a normal conversation |
| 2026-10-24 | Headphones unplugged → Windows plays the call through the laptop speakers, so the mic can pick up the other side (would appear as "Me"); after replugging, the call app may keep using the speakers → set its output back to the headphones (README) |
| 2026-10-24 | Speech model sometimes doubles a syllable or letter (`Շատատ`, `դուրր`, `գործործը`), more often on fast/call audio; not caused by our audio path (D35, `results/asr_paths.md`) |
| 2026-10-24 | Splitting speech into phrases costs ~2 WER points: words at a phrase boundary are recognised slightly worse (`հանդիպել` → `հանդիպեք`) — the price of showing text phrase by phrase (D35) |
