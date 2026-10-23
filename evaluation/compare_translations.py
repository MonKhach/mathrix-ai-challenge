"""Write results/mt_side_by_side.md: both models' translations next to each other, for reading."""
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS = ["opus-mt", "nllb-600m"]


def main():
    tables = {}
    for m in MODELS:
        with open(RESULTS_DIR / f"mt_{m}.tsv", encoding="utf-8", newline="") as f:
            tables[m] = {r["file"]: r for r in csv.DictReader(f, delimiter="\t")}
    lines = ["# Translation side by side", ""]
    for file, row in tables[MODELS[0]].items():
        lines += [f"## {file}", "", f"- **HY reference:** {row['hy_reference']}"]
        if row["en_reference"]:
            lines.append(f"- **EN reference (human):** {row['en_reference']}")
        for m in MODELS:
            lines.append(f"- **{m}** (from reference): {tables[m][file]['en_from_reference']}")
        lines.append(f"- **HY from NVIDIA:** {row['hy_asr']}")
        for m in MODELS:
            lines.append(f"- **{m}** (from NVIDIA): {tables[m][file]['en_from_asr']}")
        lines.append("")
    out = RESULTS_DIR / "mt_side_by_side.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {out}")


if __name__ == "__main__":
    main()
