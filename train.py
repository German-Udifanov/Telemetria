import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from charts import save_timeline
from telemetry import (
    GAP_SECONDS, SENSORS, aggregate, alerts_from_flags, assess, detect, forest_features,
    load_recording, reference_events,
)

SEED = 42
GROUPS = ("valve1", "valve2", "other")
TAILS = (0.001, 0.005, 0.01, 0.025, 0.05)
CONFIRMATIONS = (1, 3, 5)


def split_recordings(root):
    rng = random.Random(SEED)
    validation, test = [], []
    for group in GROUPS:
        paths = sorted(path.relative_to(root).as_posix() for path in (root / group).glob("*.csv"))
        if len(paths) < 2:
            raise ValueError(f"Expected at least two CSVs in {group}")
        rng.shuffle(paths)
        count = max(1, round(0.30 * len(paths)))
        test.extend(paths[:count])
        validation.extend(paths[count:])
    return sorted(validation), sorted(test)


def ranking(metrics):
    return metrics["event_f1"], -metrics["extra_alerts"]


def tune(normal, recordings):
    train_values = normal[SENSORS].to_numpy(dtype=float)
    validation_values = [data[SENSORS].to_numpy(dtype=float) for data in recordings]
    forests = []
    for window in (1, 10):
        fit_features = forest_features(normal, window)
        forest = IsolationForest(n_estimators=150, random_state=SEED, n_jobs=1).fit(fit_features)
        train_scores = -forest.score_samples(fit_features)
        validation_scores = [-forest.score_samples(forest_features(data, window)) for data in recordings]
        forests.append((window, forest, train_scores, validation_scores))
    best = {}
    for tail in TAILS:
        lower, upper = np.quantile(train_values, [tail, 1 - tail], axis=0)
        candidates = [
            ({"method": "quantile_rule", "lower": lower, "upper": upper, "tail": tail},
             [((values < lower) | (values > upper)).any(axis=1) for values in validation_values]),
        ]
        for window, forest, train_scores, validation_scores in forests:
            threshold = float(np.quantile(train_scores, 1 - tail))
            method = "isolation_forest" if window == 1 else "isolation_forest_rolling"
            candidates.append(
                ({"method": method, "estimator": forest, "threshold": threshold,
                  "tail": tail, "window": window}, [scores > threshold for scores in validation_scores])
            )
        for base, flags in candidates:
            for consecutive in CONFIRMATIONS:
                metrics = aggregate([
                    assess(data, alerts_from_flags(data["datetime"], mask, consecutive))
                    for data, mask in zip(recordings, flags, strict=True)
                ])
                method = base["method"]
                if method not in best or ranking(metrics) > ranking(best[method][1]):
                    best[method] = ({**base, "consecutive": consecutive}, metrics)
    return best


def plot_examples(experiments, detector, output):
    examples = {}
    for name, data in experiments.items():
        alerts = detect(detector, data)
        result = assess(data, alerts)
        choices = {"detected": result["matched_times"], "missed": result["missed_starts"],
                   "extra": result["extra_times"]}
        for kind, marks in choices.items():
            if marks and kind not in examples:
                save_timeline(data, alerts, output / f"{kind}.png",
                              title=f"{kind.title()} | {name}", highlight=marks[0])
                examples[kind] = name
    return examples


def train(root, output):
    normal_name = "anomaly-free/anomaly-free.csv"
    normal = load_recording(root / normal_name)
    validation, test = split_recordings(root)
    val_data = {name: load_recording(root / name, labeled=True) for name in validation}
    best = tune(normal, list(val_data.values()))
    selected = max(best, key=lambda method: ranking(best[method][1]))
    test_data = {name: load_recording(root / name, labeled=True) for name in test}
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": SEED, "gap_seconds": GAP_SECONDS, "selected_detector": selected,
        "split": {"train": [normal_name], "validation": validation, "test": test},
        "detectors": {},
    }
    audit = []
    for partition, recordings in (
        ("train", {normal_name: normal}), ("validation", val_data), ("test", test_data),
    ):
        for name, data in recordings.items():
            times = data["datetime"]
            events = len(reference_events(data)) if "anomaly" in data else 0
            audit.append({"split": partition, "file": name, "rows": len(data),
                          "elapsed_seconds": (times.iloc[-1] - times.iloc[0]).total_seconds(),
                          "gaps_over_5s": int(times.diff().dt.total_seconds().gt(GAP_SECONDS).sum()),
                          "events": events})
    per_file = []
    for method, (detector, validation_metrics) in best.items():
        outcomes = []
        for name, data in test_data.items():
            result = assess(data, detect(detector, data))
            outcomes.append(result)
            per_file.append({"detector": method, "file": name, **aggregate([result])})
        report["detectors"][method] = {
            "tail": detector["tail"], "consecutive": detector["consecutive"],
            "window": detector.get("window", 1),
            "validation": validation_metrics, "test": aggregate(outcomes),
        }
    report["examples"] = plot_examples(test_data, best[selected][0], output)
    save_timeline(normal, detect(best[selected][0], normal), output / "normal.png",
                  title="Normal reference recording | training data")
    pd.DataFrame(audit).to_csv(output / "recordings.csv", index=False)
    pd.DataFrame(per_file).to_csv(output / "test_by_recording.csv", index=False)
    joblib.dump(best[selected][0], output / "detector.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for method, result in report["detectors"].items():
        print(f"{method}: validation F1={result['validation']['event_f1']:.3f}, "
              f"test F1={result['test']['event_f1']:.3f}")
    print(f"Selected on validation: {selected}")


def main():
    parser = argparse.ArgumentParser(description="Train and compare SKAB telemetry detectors.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        train(args.data_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
