import numpy as np
import pandas as pd

SENSORS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]
GAP_SECONDS = 5
SAMPLE_SECONDS = 1
ALERT_COLUMNS = ["start", "confirmed_at", "end", "observations"]


def load_recording(path, *, labeled=False):
    data = pd.read_csv(path, sep=";")
    required = ["datetime", *SENSORS]
    if labeled:
        required.append("anomaly")
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {', '.join(missing)}")
    if data.empty:
        raise ValueError(f"No telemetry in {path.name}")
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    if (data["datetime"].isna().any() or not data["datetime"].is_monotonic_increasing
            or data["datetime"].duplicated().any()):
        raise ValueError(f"Timestamps must be present, unique and ordered in {path.name}")
    data[SENSORS] = data[SENSORS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[SENSORS].to_numpy()).all():
        raise ValueError(f"Sensor readings must be finite in {path.name}")
    for label in ("anomaly", "changepoint"):
        if label in data:
            values = pd.to_numeric(data[label], errors="raise")
            if not values.isin([0, 1]).all():
                raise ValueError(f"{label} must contain only 0 or 1 in {path.name}")
            data[label] = values.astype(bool)
    return data


def runs(mask, breaks=None):
    mask = np.asarray(mask, dtype=bool)
    previous = np.r_[False, mask[:-1]]
    following = np.r_[mask[1:], False]
    starts = mask & ~previous
    ends = mask & ~following
    if breaks is not None:
        starts |= mask & breaks
        ends |= mask & np.r_[breaks[1:], False]
    return zip(np.flatnonzero(starts), np.flatnonzero(ends), strict=True)


def alerts_from_flags(times, flags, consecutive):
    breaks = times.diff().dt.total_seconds().gt(GAP_SECONDS).to_numpy()
    rows = [
        (times.iloc[start], times.iloc[start + consecutive - 1], times.iloc[end], end - start + 1)
        for start, end in runs(flags, breaks)
        if end - start + 1 >= consecutive
    ]
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def forest_features(data, window):
    readings = data[SENSORS]
    if window == 1:
        return readings.to_numpy(dtype=float)
    segments = data["datetime"].diff().dt.total_seconds().gt(GAP_SECONDS).cumsum()
    rolling = readings.groupby(segments, sort=False).rolling(window, min_periods=1)
    return np.column_stack([readings, rolling.mean(), rolling.std(ddof=0)])


def model_flags(detector, data):
    readings = data[SENSORS].to_numpy(dtype=float)
    if detector["method"] == "quantile_rule":
        return ((readings < detector["lower"]) | (readings > detector["upper"])).any(axis=1)
    features = forest_features(data, detector["window"])
    return -detector["estimator"].score_samples(features) > detector["threshold"]


def detect(detector, data):
    return alerts_from_flags(data["datetime"], model_flags(detector, data), detector["consecutive"])


def reference_events(data):
    times = data["datetime"]
    return [(times.iloc[start], times.iloc[end]) for start, end in runs(data["anomaly"])]


def assess(data, alerts):
    events = reference_events(data)
    delays = {}
    confirmations = []
    extras = []
    for confirmed in alerts["confirmed_at"]:
        for index, (start, end) in enumerate(events):
            if index not in delays and start <= confirmed <= end:
                delays[index] = (confirmed - start).total_seconds()
                confirmations.append(confirmed)
                break
        else:
            extras.append(confirmed)
    return {
        "events": len(events),
        "matched": len(delays),
        "extra": len(extras),
        "delays": list(delays.values()),
        "observed_hours": len(data) * SAMPLE_SECONDS / 3600,
        "missed_starts": [start for index, (start, _) in enumerate(events) if index not in delays],
        "extra_times": extras,
        "matched_times": confirmations,
    }


def aggregate(results):
    events = sum(result["events"] for result in results)
    matched = sum(result["matched"] for result in results)
    extra = sum(result["extra"] for result in results)
    hours = sum(result["observed_hours"] for result in results)
    delays = [delay for result in results for delay in result["delays"]]
    return {
        "experiments": len(results), "events": events, "matched": matched,
        "missed": events - matched, "extra_alerts": extra, "observed_hours": hours,
        "event_recall": matched / events if events else 0.0,
        "alert_precision": matched / (matched + extra) if matched + extra else 0.0,
        "event_f1": 2 * matched / (events + matched + extra) if events + extra else 0.0,
        "extra_alerts_per_hour": extra / hours,
        "median_delay_seconds": float(np.median(delays)) if delays else None,
    }
