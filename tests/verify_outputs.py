#!/usr/bin/env python3
"""Compare regenerated numerical outputs with the checked-in reference files."""

import json
import math
import sys
from pathlib import Path


def compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: dictionary keys differ")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_outputs.py ACTUAL_DIR EXPECTED_DIR")
    actual_dir = Path(sys.argv[1])
    expected_dir = Path(sys.argv[2])
    with (actual_dir / "latest_results_summary.json").open(encoding="utf-8") as handle:
        actual = json.load(handle)
    with (expected_dir / "latest_results_summary.json").open(encoding="utf-8") as handle:
        expected = json.load(handle)
    compare(actual, expected)

    required = (
        "latest_model_summary.csv",
        "latest_item_scores.csv",
        "latest_education_family_summary.csv",
        "latest_judge_behavior.csv",
        "legacy_judge_behavior.csv",
        "overall_and_refusal.png",
        "value_heatmap.png",
        "education_family_heatmap.png",
        "judge_behavior.png",
    )
    missing = [name for name in required if not (actual_dir / name).is_file()]
    if missing:
        raise AssertionError(f"missing outputs: {', '.join(missing)}")
    item_scores = "latest_item_scores.csv"
    if (actual_dir / item_scores).read_bytes() != (expected_dir / item_scores).read_bytes():
        raise AssertionError(f"{item_scores}: regenerated values differ")
    print("Verified: regenerated summary matches the released reference output.")


if __name__ == "__main__":
    main()
