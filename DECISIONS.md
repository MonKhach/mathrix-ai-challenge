# DECISIONS

Each entry: decision, alternatives considered, reason. Newest at the bottom.

| # | Date | Decision | Alternatives | Reason |
|---|---|---|---|---|
| D1 | 2026-10-22 | Benchmark speech-to-text candidates on this laptop before choosing a model | Pick a model by reputation | Armenian is low-resource and the CPU is limited; only measurements on this machine are reliable |
| D2 | 2026-10-22 | Try WASAPI loopback first for system audio capture | VB-Audio Virtual Cable | Built into Windows, nothing extra to install, user still hears the call normally |
| D3 | 2026-10-22 | Use headphones during testing and the interview | Speakers | Echo handling is out of scope; with speakers the mic would pick up the call side and label it "Me" |
| D4 | 2026-10-23 | Target **Eastern Armenian only**; no optimization for Western Armenian in this task | Also optimize for Western Armenian | Main use case and limited time; Western Armenian is listed as a known limitation / future work |
| D5 | 2026-10-23 | **No paid tools, services or APIs.** If something paid turns out to be essential, stop and decide together | Paid APIs or services | Project rule set by the user |
| D6 | 2026-10-23 | Speech recognition, translation and speaker identification run **only on the local PC** — no external cloud APIs | Cloud speech/translation APIs | Challenge constraint and user requirement. (Downloading free model files once is allowed; inference is local.) |
| D7 | 2026-10-23 | Python 3.11 (3.11.3 already installed) | Newer Python (3.12+) | Already installed; 3.11 has the widest support among ML/audio libraries |
| D8 | 2026-10-23 | Use **PyAudioWPatch** (WASAPI) for both microphone and system-audio loopback capture | SoundCard; VB-Audio Virtual Cable | Built for Windows WASAPI loopback, Python 3.11 wheel, prints device list. **M1 test passed:** mic (idx 12) + headphones loopback (idx 13) recorded simultaneously, 5.00 s / 4.99 s captured, 0 overflows, both recordings clean |
| D9 | 2026-10-23 | Open devices at their native format (48 kHz stereo) and convert to 16 kHz mono in our code | Ask Windows for 16 kHz directly | WASAPI shared mode does not convert formats; all devices on this laptop report 48 kHz / 2 ch. Device indices can change → always choose devices explicitly (`--list` first) |
