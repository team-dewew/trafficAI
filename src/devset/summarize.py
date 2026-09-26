"""Turn evaluate.py's --json report into a Markdown table (devset/REPORT.md).

    python evaluate.py --pred predictions_samples.json --gt devset/labels.json --json devset/report.json
    python -m src.devset.summarize [devset/report.json] [devset/REPORT.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TAUS = ("0.3", "0.5", "0.7")


def render(rep: dict) -> str:
    a, b = rep["part_a"], rep.get("part_b") or {}
    lines = [
        "# Dev-set evaluation",
        "",
        f"- Model score: **{rep['model_score']:.4f}**",
        f"- Score A: **{a['score_a']:.4f}** over {len(a['classes'])} classes",
    ]
    if b:
        lines.append(
            f"- Score B: **{b.get('score_b', 0):.4f}** (AP {b.get('ap', 0):.3f}, "
            f"F1_alarm {b.get('f1_alarm', 0):.3f}, mTTA {b.get('mtta_sec', 0):.2f} s)"
        )
    lines += ["", "| class | TP | FP | FN | P | R | F1@0.3 | F1@0.5 | F1@0.7 | mean |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in a["classes"]:
        pc = a["per_class"][c]
        mid = pc["0.5"]
        f1s = " | ".join(f"{pc[t]['f1']:.3f}" for t in TAUS)
        lines.append(f"| {c} | {mid['tp']} | {mid['fp']} | {mid['fn']} | {mid['precision']:.3f} | "
                     f"{mid['recall']:.3f} | {f1s} | {pc['f1_mean']:.3f} |")
    lines.append("")
    lines.append("TP/FP/FN, precision and recall are at tIoU 0.5.")
    return "\n".join(lines) + "\n"


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "devset" / "report.json"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("REPORT.md")
    if not src.exists():
        sys.exit(f"no report at {src} (run evaluate.py with --json first)")
    dst.write_text(render(json.loads(src.read_text())), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
