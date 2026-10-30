import json
from src.validation import create_validation_report

if __name__ == "__main__":
    report = create_validation_report(
        "data/processed/products_labeled.csv",
        "reports/validation_report.json",
    )
    print(json.dumps(report, indent=2))
