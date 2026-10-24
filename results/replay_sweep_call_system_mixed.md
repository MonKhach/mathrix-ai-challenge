# Replay sweep: call_system_mixed

- call-side phrases: 20 (9 shorter than 1.5 s)
- labels counted per setting; with ONE person on the call side the ideal is 'Speaker 1' only

| rule | threshold | labels used | phrases per label |
|---|---|---|---|
| assign | 0.30 | 1 | Speaker 1: 20 |
| assign | 0.35 | 1 | Speaker 1: 20 |
| assign | 0.40 | 2 | Speaker 1: 19, Speaker 2: 1 |
| assign | 0.45 | 2 | Speaker 1: 19, Speaker 2: 1 |
| assign | 0.50 | 4 | Speaker 1: 5, Speaker 2: 13, Speaker 3: 1, Speaker 4: 1 |
| unknown | 0.30 | 2 | Speaker 1: 11, Speaker ?: 9 |
| unknown | 0.35 | 2 | Speaker 1: 11, Speaker ?: 9 |
| unknown | 0.40 | 3 | Speaker 1: 10, Speaker 2: 1, Speaker ?: 9 |
| unknown | 0.45 | 3 | Speaker 1: 10, Speaker 2: 1, Speaker ?: 9 |
| unknown | 0.50 | 4 | Speaker 1: 9, Speaker 2: 1, Speaker 3: 1, Speaker ?: 9 |
| none | 0.30 | 6 | Speaker 1: 14, Speaker 2: 1, Speaker 3: 1, Speaker 4: 1, Speaker 5: 2, Speaker 6: 1 |
| none | 0.35 | 8 | Speaker 1: 12, Speaker 2: 1, Speaker 3: 1, Speaker 4: 1, Speaker 5: 1, Speaker 6: 1, Speaker 7: 2, Speaker 8: 1 |
| none | 0.40 | 9 | Speaker 1: 11, Speaker 2: 1, Speaker 3: 1, Speaker 4: 1, Speaker 5: 1, Speaker 6: 1, Speaker 7: 2, Speaker 8: 1, Speaker 9: 1 |
| none | 0.45 | 10 | Speaker 1: 11, Speaker 10: 1, Speaker 2: 1, Speaker 3: 1, Speaker 4: 1, Speaker 5: 1, Speaker 6: 1, Speaker 7: 1, Speaker 8: 1, Speaker 9: 1 |
| none | 0.50 | 12 | Speaker 1: 1, Speaker 10: 1, Speaker 11: 1, Speaker 12: 1, Speaker 2: 9, Speaker 3: 1, Speaker 4: 1, Speaker 5: 1, Speaker 6: 1, Speaker 7: 1, Speaker 8: 1, Speaker 9: 1 |
