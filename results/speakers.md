# Speaker identification test (WeSpeaker ResNet34-LM)

- clips: 21 (7 mic = me, 14 system/video, 8 different video speakers)
- time per embedding: average 0.066 s, slowest 0.268 s
- excerpts: 2 s / 1 s cut from the middle of each clip

## Cosine similarity of pairs: min / mean / max

| length | same person, mic (me) | same person, system | different people, system |
|---|---|---|---|
| full | 0.58 / 0.67 / 0.76 (n=21) | 0.62 / 0.76 / 0.85 (n=12) | -0.10 / 0.09 / 0.28 (n=79) |
| 2s | 0.19 / 0.45 / 0.60 (n=21) | 0.45 / 0.53 / 0.73 (n=12) | -0.09 / 0.08 / 0.24 (n=79) |
| 1s | 0.02 / 0.27 / 0.47 (n=21) | 0.25 / 0.37 / 0.54 (n=12) | -0.15 / 0.05 / 0.31 (n=79) |

## Simulated live labelling of the 14 system clips (8 people), 20 random orders

Splits = one person received extra labels; merges = one label covered extra people (totals per order, averaged). Ideal: 0 / 0 and speakers found = 8.

| threshold | full: splits / merges / speakers | 2 s: splits / merges / speakers | 1 s: splits / merges / speakers |
|---|---|---|---|
| 0.20 | 0.0 / 2.6 / 5.3 | 0.0 / 2.6 / 5.4 | 0.1 / 2.2 / 5.9 |
| 0.25 | 0.0 / 1.4 / 6.5 | 0.0 / 0.3 / 7.7 | 0.1 / 1.7 / 6.4 |
| 0.30 | 0.0 / 0.0 / 8.0 | 0.0 / 0.0 / 8.0 | 0.6 / 0.1 / 8.4 |
| 0.35 | 0.0 / 0.0 / 8.0 | 0.0 / 0.0 / 8.0 | 1.2 / 0.0 / 9.2 |
| 0.40 | 0.0 / 0.0 / 8.0 | 0.0 / 0.0 / 8.0 | 1.9 / 0.0 / 9.8 |
| 0.45 | 0.0 / 0.0 / 8.0 | 0.0 / 0.0 / 8.0 | 3.9 / 0.0 / 11.8 |
| 0.50 | 0.0 / 0.0 / 8.0 | 0.6 / 0.0 / 8.6 | 5.0 / 0.0 / 13.0 |
| 0.55 | 0.0 / 0.0 / 8.0 | 2.8 / 0.0 / 10.8 | 6.0 / 0.0 / 14.0 |
| 0.60 | 0.0 / 0.0 / 8.0 | 5.0 / 0.0 / 13.0 | 6.0 / 0.0 / 14.0 |
| 0.65 | 0.3 / 0.0 / 8.3 | 5.0 / 0.0 / 13.0 | 6.0 / 0.0 / 14.0 |
| 0.70 | 0.8 / 0.0 / 8.8 | 5.0 / 0.0 / 13.0 | 6.0 / 0.0 / 14.0 |
| 0.75 | 1.5 / 0.0 / 9.5 | 6.0 / 0.0 / 14.0 | 6.0 / 0.0 / 14.0 |
| 0.80 | 3.4 / 0.0 / 11.3 | 6.0 / 0.0 / 14.0 | 6.0 / 0.0 / 14.0 |
