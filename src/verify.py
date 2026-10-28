"""Independent source, split, search-ledger and prediction arithmetic checks."""

from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import sqlite3
import statistics
import struct
import sys
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (110, 211, 312, 413, 514, 615)


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-7, rel_tol=1e-9), (a, b)


def save(name, data):
    (ROOT / "results" / name).write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(data), flush=True)


def packed(values):
    values = [
        float("nan") if math.isnan(float(v)) else 0.0 if float(v) == 0 else float(v)
        for v in values
    ]
    return struct.pack("<" + "d" * len(values), *values)


def group_id(values):
    return hashlib.sha256(packed(values)).hexdigest()


def source_check():
    checks = []
    for record in json.loads((ROOT / "data/manifest.json").read_text()):
        number = record["id"]
        meta = ROOT / f"data/metadata/{number}.json"
        assert (
            hashlib.sha256(meta.read_bytes()).hexdigest() == record["metadata_sha256"]
        )
        description = json.loads(meta.read_bytes())["data_set_description"]
        assert description["id"] == str(number) and description["status"] == "active"
        panel_path = ROOT / record["panel_file"]
        assert (
            hashlib.sha256(panel_path.read_bytes()).hexdigest()
            == record["panel_sha256"]
        )
        panel = pd.read_csv(panel_path, float_precision="round_trip")
        features = record["features"]
        target = record["target"]
        assert target not in features and panel.source_row.is_unique
        assert all(
            group_id(row) == group
            for row, group in zip(panel[features].to_numpy(), panel.group_id)
        )
        evidence = {
            "dataset": number,
            "panel_rows": len(panel),
            "feature_groups": panel.group_id.nunique(),
            "panel_and_metadata_hashes": "PASS",
            "raw_independent_reconciliation": "not available in offline checkout",
        }
        raw_path = ROOT / f"data/raw/{number}.arff"
        if raw_path.exists():
            digest = hashlib.sha256()
            with raw_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == record["raw_sha256"]
            names = []
            unique = {}
            groups = Counter()
            source_count = 0
            finite_count = 0
            selected_lookup = {int(r.source_row): r for _, r in panel.iterrows()}
            with raw_path.open(encoding="utf-8", newline="") as handle:
                for line in handle:
                    if line.strip().casefold() == "@data":
                        break
                    match = re.match(
                        r'\s*@attribute\s+(?:\x27([^\x27]+)\x27|"([^"]+)"|(\S+))\s+',
                        line,
                        re.I,
                    )
                    if match:
                        names.append(next(x for x in match.groups() if x is not None))
                indexes = [names.index(f) for f in features]
                yi = names.index(target)
                for fields in csv.reader(handle):
                    if not fields or fields[0].lstrip().startswith("%"):
                        continue
                    row_index = source_count
                    source_count += 1
                    y = float(fields[yi]) if fields[yi] != "?" else float("nan")
                    if not math.isfinite(y):
                        continue
                    finite_count += 1
                    x = [
                        float(fields[i]) if fields[i] != "?" else float("nan")
                        for i in indexes
                    ]
                    assert all(not math.isinf(v) for v in x)
                    key = packed(x + [y])
                    if key not in unique:
                        unique[key] = row_index
                        groups[group_id(x)] += 1
                    if row_index in selected_lookup:
                        selected = selected_lookup[row_index]
                        assert (
                            packed([selected[f] for f in features] + [selected[target]])
                            == key
                        )
            total = 0
            chosen = set()
            for group, size in sorted(groups.items()):
                if total + size > 6000:
                    break
                chosen.add(group)
                total += size
            expected = {
                index
                for key, index in unique.items()
                if hashlib.sha256(key[:-8]).hexdigest() in chosen
            }
            assert expected == set(panel.source_row.astype(int))
            quality = record["quality"]
            assert (
                source_count == quality["source_rows"]
                and finite_count == quality["finite_target_rows"]
                and finite_count - len(unique)
                == quality["exact_full_row_duplicates_removed"]
                and len(unique) == quality["eligible_distinct_rows"]
                and len(expected) == quality["selected_rows"]
                and len(chosen) == quality["selected_feature_groups"]
            )
            evidence.update(
                raw_independent_reconciliation="PASS: independent streaming CSV/struct parser, every source row, deduplication, hash-cap group selection and included source values",
                source_rows_checked=source_count,
            )
        checks.append(evidence)
    save(
        "verification_source.json",
        {
            "status": "PASS",
            "datasets": checks,
            "scope": "Raw reconciliation runs when original files are present; offline checkouts verify frozen panel/metadata bytes and feature signatures.",
        },
    )


def partition(group, seed):
    remainder = (
        int(
            hashlib.sha256((group + ":" + str(seed)).encode("utf-8")).hexdigest()[0:8],
            16,
        )
        % 10
    )
    return (
        "fit"
        if remainder in (0, 1, 2, 3, 4, 5)
        else "validation" if remainder in (6, 7) else "test"
    )


def configurations(dataset, seed):
    configs = []
    for values in itertools.product([0.03, 0.1, 0.3], [7, 15, 31], [0, 1], [10, 30]):
        a, b, c, d = values
        configs.append(
            {
                "id": f"lr{a:g}_leaf{b}_l2{c}_min{d}",
                "learning_rate": a,
                "max_leaf_nodes": b,
                "l2_regularization": c,
                "min_samples_leaf": d,
            }
        )
    configs.sort(key=lambda r: r["id"])
    order = np.random.default_rng(seed + dataset).permutation(36)[:18]
    return [configs[int(i)] for i in order]


def arithmetic(actual, prediction, variance):
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    mse = float(np.sum(residual * residual, dtype=np.float64) / len(residual))
    return {
        "n": len(residual),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.sum(np.abs(residual)) / len(residual)),
        "mean_error": float(np.sum(residual) / len(residual)),
        "normalized_mse": mse / variance,
    }


def verify_runs(phase, role):
    checked = []
    for record in json.loads((ROOT / "data/manifest.json").read_text()):
        if record["role"] != role:
            continue
        panel = pd.read_csv(
            ROOT / record["panel_file"], float_precision="round_trip"
        ).set_index("source_row")
        target = record["target"]
        for seed in SEEDS:
            folder = ROOT / f'results/runs/{record["id"]}/{seed}'
            selection = json.loads((folder / "selection.json").read_text())
            pool = configurations(record["id"], seed)
            assert (
                selection["pool"] == pool
                and selection["role"] == role
                and not selection["test_evaluated"]
                and not selection["audit_used_for_selection"]
            )
            splits = pd.read_csv(folder / "splits.csv").set_index("source_row")
            assert set(splits.index) == set(panel.index) and splits.index.is_unique
            for i, row in splits.iterrows():
                assert row["group_id"] == panel.loc[i, "group_id"] and row[
                    "split"
                ] == partition(row["group_id"], seed)
            db = sqlite3.connect(":memory:")
            db.execute(
                "CREATE TABLE assignments(source_row INTEGER PRIMARY KEY,group_id TEXT,split TEXT)"
            )
            db.executemany(
                "INSERT INTO assignments VALUES(?,?,?)",
                [(int(i), r.group_id, r.split) for i, r in splits.iterrows()],
            )
            assert not db.execute(
                (ROOT / "sql/split_contract.sql").read_text()
            ).fetchall()
            assert db.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == len(
                panel
            )
            db.close()
            counts = splits["split"].value_counts().to_dict()
            assert (
                counts == selection["partition_counts"] and min(counts.values()) >= 200
            )
            train_ids = splits.index[splits["split"] == "fit"]
            val_ids = set(splits.index[splits["split"] == "validation"])
            test_ids = set(splits.index[splits["split"] == "test"])
            fit_y = panel.loc[train_ids, target].to_numpy()
            variance = statistics.pvariance(float(v) for v in fit_y)
            center = statistics.mean(float(v) for v in fit_y)
            close(variance, selection["fit_target_variance"])
            trials = pd.read_csv(folder / "trials.csv", float_precision="round_trip")
            assert len(trials) == 50 and set(trials.status) == {"success"}
            predictions = pd.read_csv(
                folder / "validation_predictions.csv.gz", float_precision="round_trip"
            )
            for _, trial in trials.iterrows():
                pred = predictions.loc[
                    (predictions.strategy == trial.strategy)
                    & (predictions.stage == trial.stage)
                    & (predictions.candidate == trial.candidate)
                ]
                assert (
                    len(pred) == len(val_ids)
                    and pred.source_row.is_unique
                    and set(pred.source_row) == val_ids
                )
                assert np.array_equal(
                    pred.actual.to_numpy(),
                    panel.loc[pred.source_row, target].to_numpy(),
                )
                for key, value in arithmetic(
                    pred.actual, pred.prediction, variance
                ).items():
                    close(value, trial[key])
                close(trial.target_center, center)
                close(trial.target_scale, math.sqrt(variance))
                assert 0 < trial.actual_iterations <= trial.requested_iterations
                config = next(c for c in pool if c["id"] == trial.candidate)
                for key, value in config.items():
                    if key != "id":
                        close(trial[key], value)

            def candidates(strategy, stage):
                return trials.loc[
                    (trials.strategy == strategy) & (trials.stage == stage)
                ].sort_values(["mse", "candidate"])

            random = candidates("random", 0)
            first = candidates("halving", 0)
            second = candidates("halving", 1)
            last = candidates("halving", 2)
            audit = candidates("audit", 0)
            assert set(random.candidate) == {c["id"] for c in pool[:6]} and set(
                random.requested_iterations
            ) == {90}
            assert set(first.candidate) == {c["id"] for c in pool} and set(
                first.requested_iterations
            ) == {10}
            assert set(second.candidate) == set(first.head(6).candidate) and set(
                second.requested_iterations
            ) == {30}
            assert set(last.candidate) == set(second.head(2).candidate) and set(
                last.requested_iterations
            ) == {90}
            assert set(audit.candidate) == {c["id"] for c in pool} and set(
                audit.requested_iterations
            ) == {90}
            assert trials.groupby("strategy").requested_iterations.sum().to_dict() == {
                "audit": 1620,
                "halving": 540,
                "random": 540,
            }
            for strategy, final in [("random", random), ("halving", last)]:
                chosen = selection["selections"][strategy]
                winner = final.iloc[0]
                assert chosen["candidate"] == winner.candidate and chosen[
                    "config"
                ] == next(c for c in pool if c["id"] == winner.candidate)
                close(chosen["validation_mse"], winner.mse)
                own = trials.loc[trials.strategy == strategy]
                assert chosen["search_requested_iterations"] == 540 and chosen[
                    "search_actual_iterations"
                ] == int(own.actual_iterations.sum())
                close(chosen["search_seconds"], own.elapsed_seconds.sum())
            audit_record = json.loads((folder / "audit.json").read_text())
            oracle = audit.iloc[0]
            assert oracle.candidate == audit_record["oracle_candidate"]
            close(oracle.mse, audit_record["oracle_validation_mse"])
            assert audit_record["requested_iterations"] == 1620 and audit_record[
                "actual_iterations"
            ] == int(audit.actual_iterations.sum())
            close(audit_record["elapsed_seconds"], audit.elapsed_seconds.sum())
            assert audit_record["oracle_survives_10"] == (
                oracle.candidate in set(second.candidate)
            ) and audit_record["oracle_survives_30"] == (
                oracle.candidate in set(last.candidate)
            )
            for strategy in ["random", "halving"]:
                close(
                    audit_record["regret"][strategy],
                    selection["selections"][strategy]["validation_mse"] - oracle.mse,
                )
            evidence = {
                "dataset": record["id"],
                "seed": seed,
                "validation_predictions": len(predictions),
                "trials": len(trials),
                "search_iterations_each": 540,
                "extra_audit_iterations": 1620,
                "splits_sql": "PASS",
                "promotions_and_selection": "PASS",
            }
            if phase == "test":
                result = json.loads((folder / "test_metrics.json").read_text())
                assert datetime.fromisoformat(
                    result["evaluated_utc"]
                ) > datetime.fromisoformat(selection["frozen_utc"])
                observed = pd.read_csv(
                    folder / "test_predictions.csv.gz", float_precision="round_trip"
                )
                outer_ids = splits.index[splits["split"] != "test"]
                outer_y = panel.loc[outer_ids, target].to_numpy()
                outer_center = statistics.mean(float(v) for v in outer_y)
                outer_var = statistics.pvariance(float(v) for v in outer_y)
                assert {m["strategy"] for m in result["metrics"]} == {
                    "random",
                    "halving",
                    "mean",
                }
                for metric in result["metrics"]:
                    pred = observed.loc[observed.strategy == metric["strategy"]]
                    assert (
                        set(pred.source_row) == test_ids and pred.source_row.is_unique
                    )
                    assert np.array_equal(
                        pred.actual.to_numpy(),
                        panel.loc[pred.source_row, target].to_numpy(),
                    )
                    assert list(pred.group_id) == list(
                        panel.loc[pred.source_row, "group_id"]
                    )
                    for key, value in arithmetic(
                        pred.actual, pred.prediction, outer_var
                    ).items():
                        close(metric[key], value)
                    close(metric["outer_train_variance"], outer_var)
                    close(metric["target_center"], outer_center)
                    close(metric["target_scale"], math.sqrt(outer_var))
                    assert metric["outer_train_n"] == len(outer_ids)
                    if metric["strategy"] == "mean":
                        assert np.allclose(
                            pred.prediction, outer_center, atol=1e-7, rtol=1e-9
                        )
                    else:
                        assert (
                            metric["candidate"]
                            == selection["selections"][metric["strategy"]]["candidate"]
                            and metric["actual_iterations"] <= 90
                        )
                evidence["test_predictions"] = len(observed)
                evidence["independent_test_metrics"] = 3
            checked.append(evidence)
    save(
        f"verification_{phase}_{role}.json",
        {
            "status": "PASS",
            "runs": checked,
            "scope": "All exported arithmetic, source/group/split identities, resource ledgers and frozen selectors independently checked; no training metric helpers imported.",
        },
    )


if __name__ == "__main__":
    if sys.argv[1] == "source":
        source_check()
    else:
        verify_runs(sys.argv[1], sys.argv[2])
