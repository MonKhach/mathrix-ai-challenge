# Translation benchmark: nllb-600m

- model load time: 9.6 s
- peak RAM (whole process): 1277 MB
- time per sentence: average 1.24 s, slowest 2.15 s
- chrF on 5 clips with English references (0-100, higher is better):
  - from Armenian references (pure translation): 62.5
  - from NVIDIA transcriptions (live pipeline): 62.6
