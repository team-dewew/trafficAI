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
        
    # Simply pretty-print to Markdown
    md_path = repo_root / "devset" / "REPORT.md"
    with open(md_path, "w") as f:
        f.write("# Dev Set Evaluation Report\n\n")
        f.write("```json\n")
        f.write(json.dumps(data, indent=2))
        f.write("\n```\n")
        
    print(f"Summary written to {md_path}")

if __name__ == "__main__":
    main()
