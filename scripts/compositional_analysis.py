#!/usr/bin/env python3
"""
Compositional Generalisation Analysis

This script runs the evaluation pipeline for multi-rule datasets with varying numbers of rules
(2, 3, and 4) and records key metrics such as:
- Symbolic equivalence rate (accuracy)
- Mean L2 embedding distance
- Invalid equation rate

The results are saved to `reports/compositional_analysis.md` for easy review.
"""
import os
import json
import logging
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation(num_rules: int, problems: int = 200) -> dict:
    """Run eval_algebra for a given number of rules and return parsed JSON results.

    Args:
        num_rules: Number of rules to chain (2, 3, or 4).
        problems: Number of problems per dataset (default 200 for a quick run).
    """
    cmd = [
        "python",
        "eval_algebra.py",
        "--eval_type",
        "multi_rule",
        "--num_rules",
        str(num_rules),
        "--multi_rule_problems",
        str(problems),
        "--verbose",
        "--output_dir",
        "./tmp_eval_results",
    ]
    logger.info(f"Running evaluation for {num_rules}-rule dataset (problems={problems})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Evaluation failed for {num_rules}-rule: {result.stderr}")
        raise RuntimeError(f"Evaluation failed for {num_rules}-rule")
    # The script writes a JSON file named evaluation_results_multi_rule.json in the output dir
    json_path = Path("./tmp_eval_results/evaluation_results_multi_rule.json")
    if not json_path.is_file():
        logger.error("Result JSON not found after evaluation")
        raise FileNotFoundError("Result JSON missing")
    with open(json_path) as f:
        data = json.load(f)
    # Extract the specific dataset entry (e.g., multi_rule_2, multi_rule_3, ...)
    key = f"multi_rule_{num_rules}"
    if key not in data:
        logger.error(f"Key {key} not present in results")
        raise KeyError(key)
    return data[key]

def generate_report(results_by_rules: dict) -> str:
    """Create a markdown report summarising the metrics for each rule depth."""
    lines = []
    lines.append("# Compositional Generalisation Analysis")
    lines.append("")
    lines.append("This report summarises the evaluation of multi‑rule datasets with varying numbers of rules.")
    lines.append("")
    lines.append("| Rules | Accuracy | Invalid Rate | Mean L2 Distance | Total Samples |
|---|---|---|---|---|")
    for rules, result in sorted(results_by_rules.items()):
        summary = result.get("summary", {})
        accuracy = summary.get("accuracy", 0.0)
        invalid = summary.get("invalid_rate", 0.0)
        mean_l2 = summary.get("mean_l2_distance", 0.0)
        total = summary.get("total_evaluated", 0)
        lines.append(f"| {rules} | {accuracy:.3f} | {invalid:.3f} | {mean_l2:.3f} | {total} |")
    lines.append("")
    lines.append("*Interpretation:* A decreasing accuracy and increasing L2 distance as the number of rules grows indicates compositional difficulty.")
    return "\n".join(lines)

def main():
    os.makedirs("reports", exist_ok=True)
    results = {}
    for n in [2, 3, 4]:
        try:
            res = run_evaluation(num_rules=n, problems=200)
            results[n] = res
        except Exception as e:
            logger.error(f"Skipping {n}-rule evaluation due to error: {e}")
    report_md = generate_report(results)
    report_path = Path("reports/compositional_analysis.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Compositional analysis report written to {report_path}")

if __name__ == "__main__":
    main()
