# Two-speaker test (A6): two known people through the real call app

- files: A = call2_system.wav, B = call2_system_2.wav
- phrases: 12 (7 A, 5 B), 0 shorter than 1.5 s

## Similarity of phrases >= 1.5 s: min / mean / max

| same person A | same person B | A vs B (different people) |
|---|---|---|
| -0.02 / 0.60 / 0.92 (n=21) | 0.62 / 0.77 / 0.89 (n=10) | -0.11 / 0.13 / 0.26 (n=35) |

## Labelling in 20 random orders (ideal: 0 splits, 0 merges, 2 speakers)

| rule | threshold | splits | merges | speakers found | "?" lines |
|---|---|---|---|---|---|
| assign | 0.30 | 1.0 | 0.0 | 3.0 | 0.0 |
| assign | 0.35 | 1.0 | 0.0 | 3.0 | 0.0 |
| assign | 0.40 | 1.0 | 0.0 | 3.0 | 0.0 |
| assign | 0.45 | 1.0 | 0.0 | 3.0 | 0.0 |
| assign | 0.50 | 1.0 | 0.0 | 3.0 | 0.0 |
| unknown | 0.30 | 1.0 | 0.0 | 3.0 | 0.0 |
| unknown | 0.35 | 1.0 | 0.0 | 3.0 | 0.0 |
| unknown | 0.40 | 1.0 | 0.0 | 3.0 | 0.0 |
| unknown | 0.45 | 1.0 | 0.0 | 3.0 | 0.0 |
| unknown | 0.50 | 1.0 | 0.0 | 3.0 | 0.0 |
| none | 0.30 | 1.0 | 0.0 | 3.0 | 0.0 |
| none | 0.35 | 1.0 | 0.0 | 3.0 | 0.0 |
| none | 0.40 | 1.0 | 0.0 | 3.0 | 0.0 |
| none | 0.45 | 1.0 | 0.0 | 3.0 | 0.0 |
| none | 0.50 | 1.0 | 0.0 | 3.0 | 0.0 |

## Per-phrase check (diagnostic)

Average similarity to the same person's other phrases, and the highest similarity to the other person. Below 0.30 to its own person = suspicious (listen at that time).

| person | time in file | length (s) | avg vs own | max vs other | flag |
|---|---|---|---|---|---|
| A | 00:00:48 | 3.7 | 0.02 | -0.02 | SUSPICIOUS |
| A | 00:00:58 | 15.0 | 0.68 | 0.18 |  |
| A | 00:01:13 | 14.3 | 0.72 | 0.20 |  |
| A | 00:01:29 | 3.5 | 0.62 | 0.26 |  |
| A | 00:01:32 | 15.0 | 0.73 | 0.20 |  |
| A | 00:01:47 | 15.0 | 0.71 | 0.17 |  |
| A | 00:02:02 | 9.6 | 0.70 | 0.17 |  |
| B | 00:00:03 | 15.0 | 0.73 | 0.18 |  |
| B | 00:00:18 | 15.0 | 0.76 | 0.17 |  |
| B | 00:00:33 | 15.0 | 0.81 | 0.16 |  |
| B | 00:00:48 | 15.0 | 0.81 | 0.26 |  |
| B | 00:01:03 | 2.1 | 0.71 | 0.23 |  |
