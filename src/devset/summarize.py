import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    report_json = repo_root / "devset" / "report.json"
    if not report_json.exists():
        print(f"No report found at {report_json}")
        return
        
    with open(report_json, "r") as f:
        data = json.load(f)
        
    md_path = repo_root / "devset" / "REPORT.md"
    
    part_a = data.get("part_a", {})
    classes = part_a.get("classes", {})
    
    md = [
        "# Dev Set Evaluation Report\n",
        f"**Model Score**: {data.get('model_score', 0):.4f}\n",
        f"**Score A (Part A)**: {part_a.get('score_a', 0):.4f}\n",
        "## Part A - Class-wise Metrics\n",
        "| Class | TP | FP | FN | Precision | Recall | F1@0.3 | F1@0.5 | F1@0.7 |",
        "|-------|---|---|---|-----------|--------|---------|---------|---------|"
    ]
    
    for cls_name, mets in classes.items():
        tp = mets.get("tp", 0)
        fp = mets.get("fp", 0)
        fn = mets.get("fn", 0)
        p = mets.get("precision", 0.0)
        r = mets.get("recall", 0.0)
        f1_3 = mets.get("f1_at_03", 0.0)
        f1_5 = mets.get("f1_at_05", 0.0)
        f1_7 = mets.get("f1_at_07", 0.0)
        md.append(f"| {cls_name} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f1_3:.3f} | {f1_5:.3f} | {f1_7:.3f} |")
        
    md.append("\n## Part B - Anomaly Metrics\n")
    part_b = data.get("part_b")
    if part_b:
        md.append(f"- **Score B**: {part_b.get('score_b', 0):.4f}")
        md.append(f"- **AP**: {part_b.get('ap', 0):.3f}")
        md.append(f"- **mTTA**: {part_b.get('mtta_sec', 0):.2f}s")
        md.append(f"- **Frames Scored**: {part_b.get('n_frames_scored', 0)}")
    else:
        md.append("No accidents in ground truth, Part B not scored.")
        
    with open(md_path, "w") as f:
        f.write("\n".join(md))
        
    print(f"Summary written to {md_path}")

if __name__ == "__main__":
    main()
