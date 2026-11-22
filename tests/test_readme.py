"""Test that README.md evaluation numbers match eval/results/latest.json exactly.

Guarantees that documentation and benchmark results do not drift.
"""

import json
import re
from pathlib import Path

README_PATH = Path("README.md")
LATEST_JSON_PATH = Path("eval/results/latest.json")


def test_readme_eval_metrics_match_latest_json():
    """Assert all benchmark metrics presented in README.md match eval/results/latest.json."""
    assert README_PATH.exists(), "README.md not found"
    assert LATEST_JSON_PATH.exists(), "eval/results/latest.json not found"

    readme_content = README_PATH.read_text(encoding="utf-8")
    with open(LATEST_JSON_PATH, encoding="utf-8") as f:
        latest = json.load(f)

    metrics = latest["metrics"]

    # 1. Overall Lenient Accuracy (96.43%)
    overall_lenient = metrics["overall_lenient_accuracy_pct"]
    assert f"`{overall_lenient:.2f}%`" in readme_content

    # 2. Overall Exact Accuracy (76.79%)
    overall_exact = metrics["overall_exact_accuracy_pct"]
    assert f"`{overall_exact:.2f}%`" in readme_content

    # 3. Easy Questions (100.0% Exact, 100.0% Lenient)
    easy = metrics["by_difficulty"]["easy"]
    assert re.search(rf"`{easy['exact_accuracy_pct']:.1f}%`\*\* Exact", readme_content)
    assert re.search(rf"`{easy['lenient_accuracy_pct']:.1f}%`\*\* Lenient", readme_content)

    # 4. Medium Questions (86.67% Exact, 100.0% Lenient)
    medium = metrics["by_difficulty"]["medium"]
    assert re.search(rf"`{medium['exact_accuracy_pct']:.2f}%`\*\* Exact", readme_content)
    assert re.search(rf"`{medium['lenient_accuracy_pct']:.1f}%`\*\* Lenient", readme_content)

    # 5. Hard Questions (33.33% Exact, 91.67% Lenient)
    hard = metrics["by_difficulty"]["hard"]
    assert re.search(rf"`{hard['exact_accuracy_pct']:.2f}%`\*\* Exact", readme_content)
    assert re.search(rf"`{hard['lenient_accuracy_pct']:.2f}%`\*\* Lenient", readme_content)

    # 6. Benign Trigger Questions (78.57% Exact, 92.86% Lenient)
    benign = metrics["by_difficulty"]["benign_trigger"]
    assert re.search(rf"`{benign['exact_accuracy_pct']:.2f}%`\*\* Exact", readme_content)
    assert re.search(rf"`{benign['lenient_accuracy_pct']:.2f}%`\*\* Lenient", readme_content)

    # 7. Policy Refusal Status (Refused correctly (1/1))
    policy_status = metrics["policy_refusal_status"]
    assert f"`{policy_status}`" in readme_content

    # 8. Safety Block Rate (100.0%)
    safety_rate = metrics["safety_block_rate_pct"]
    assert f"`{safety_rate:.1f}%`" in readme_content

    # 9. False Refusal Rate (0.0%)
    false_refusal = metrics["false_refusal_rate_pct"]
    assert f"`{false_refusal:.1f}%`" in readme_content

    # 10. Mean Latency (4,324.18 ms)
    mean_lat = metrics["mean_latency_ms"]
    formatted_lat = f"{mean_lat:,.2f}"
    assert f"`{formatted_lat} ms`" in readme_content
