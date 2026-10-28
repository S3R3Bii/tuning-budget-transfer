"""Refit every search/audit in a fresh directory and compare numerical evidence."""

from pathlib import Path
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
start = time.perf_counter()
commands = []


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-7), (a, b)


with tempfile.TemporaryDirectory(prefix="tuning-budget-replay-") as directory:
    replay = Path(directory)
    for name in ["src", "sql", "data"]:
        shutil.copytree(
            ROOT / name,
            replay / name,
            ignore=shutil.ignore_patterns("raw", "cache", "__pycache__"),
        )
    (replay / "results").mkdir()
    tasks = [("verify.py", ["source"])]
    for role in ["development", "held_dataset"]:
        tasks.extend(
            [
                ("experiment.py", ["search", role]),
                ("verify.py", ["search", role]),
                ("experiment.py", ["test", role]),
                ("verify.py", ["test", role]),
            ]
        )
    tasks.extend([("verify.py", ["summary"]), ("figures.py", [])])
    for script, args in tasks:
        phase_start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(replay / "src" / script), *args],
            cwd=replay,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        commands.append(
            {
                "script": script,
                "arguments": args,
                "elapsed_seconds": time.perf_counter() - phase_start,
                "exit_code": result.returncode,
            }
        )
        print("PASS replay step:", script, *args, flush=True)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    trial_count = 0
    for record in manifest:
        for seed in [110, 211, 312, 413, 514, 615]:
            relative = Path(f'results/runs/{record["id"]}/{seed}')
            a = json.loads((ROOT / relative / "selection.json").read_text())
            b = json.loads((replay / relative / "selection.json").read_text())
            assert (
                a["pool"] == b["pool"]
                and a["partition_counts"] == b["partition_counts"]
            )
            for strategy in ["random", "halving"]:
                assert (
                    a["selections"][strategy]["candidate"]
                    == b["selections"][strategy]["candidate"]
                )
                close(
                    a["selections"][strategy]["validation_mse"],
                    b["selections"][strategy]["validation_mse"],
                )

            def ledger(path):
                with open(path, newline="") as handle:
                    return {
                        (r["strategy"], r["stage"], r["candidate"]): r
                        for r in csv.DictReader(handle)
                    }

            old = ledger(ROOT / relative / "trials.csv")
            new = ledger(replay / relative / "trials.csv")
            assert old.keys() == new.keys()
            for key, row in old.items():
                for metric in [
                    "mse",
                    "rmse",
                    "mae",
                    "normalized_mse",
                    "mean_error",
                    "n",
                    "requested_iterations",
                    "actual_iterations",
                ]:
                    close(row[metric], new[key][metric])
                trial_count += 1
            a = json.loads((ROOT / relative / "test_metrics.json").read_text())[
                "metrics"
            ]
            b = json.loads((replay / relative / "test_metrics.json").read_text())[
                "metrics"
            ]
            for old, new in zip(a, b):
                assert (
                    old["candidate"] == new["candidate"]
                    and old["strategy"] == new["strategy"]
                )
                for metric in [
                    "mse",
                    "rmse",
                    "mae",
                    "mean_error",
                    "normalized_mse",
                    "n",
                    "outer_train_n",
                    "outer_train_variance",
                ]:
                    close(old[metric], new[metric])
    a = json.loads((ROOT / "results/summary.json").read_text())
    b = json.loads((replay / "results/summary.json").read_text())
    for old, new in zip(a["datasets"], b["datasets"]):
        for key, value in old.items():
            if isinstance(value, (int, float)):
                close(value, new[key])
            else:
                assert value == new[key]
elapsed = time.perf_counter() - start
receipt = {
    "status": "PASS",
    "search_audit_trial_rows_refitted": trial_count,
    "dataset_seed_runs": 24,
    "final_metric_rows_reproduced": 72,
    "absolute_tolerance": 1e-7,
    "relative_tolerance": 1e-9,
    "end_to_end_replay_seconds": elapsed,
    "timing_scope": "Fresh directory source-panel checks, every cold search/audit/refit, independent checks, comparison and figures; excludes raw retrieval, dependency install and private paper. Per-fit timings are not equality targets.",
    "commands": commands,
}
(ROOT / "results/reproduction.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({k: v for k, v in receipt.items() if k != "commands"}), flush=True)
