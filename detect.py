import argparse
from pathlib import Path

import joblib

from charts import save_timeline
from telemetry import detect, load_recording


def main():
    parser = argparse.ArgumentParser(description="Detect anomalous intervals in one telemetry CSV.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        sources = {args.input.resolve(), args.model.resolve()}
        if any((args.output_dir / name).resolve() in sources for name in ("alerts.csv", "timeline.png")):
            raise ValueError("Output files must be distinct from the input and the model")
        data = load_recording(args.input)
        detector = joblib.load(args.model)
        alerts = detect(detector, data)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        alerts.to_csv(args.output_dir / "alerts.csv", index=False)
        save_timeline(data, alerts, args.output_dir / "timeline.png", title=f"Telemetry | {args.input.name}")
        print(f"{len(alerts)} alert(s) in {len(data)} observations.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Detection failed: {exc}\n")


if __name__ == "__main__":
    main()
