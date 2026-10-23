# ASR benchmark: nvidia-int8

- model load time: 12.8 s
- peak RAM (whole process): 445 MB

| group | clips | raw WER | WER | CER | RTF | slowest clip (s) |
|---|---|---|---|---|---|---|
| all | 15 | 31.2% | 23.3% | 11.9% | 0.04 | 0.43 |
| me (mic) | 7 | 15.9% | 4.8% | 1.4% | 0.04 | 0.36 |
| video (system) | 8 | 38.1% | 31.7% | 16.3% | 0.04 | 0.43 |
| clips with digits | 3 | 30.6% | 25.8% | 21.9% | 0.04 | 0.43 |
