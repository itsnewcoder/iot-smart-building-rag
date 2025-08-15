from pathlib import Path
from typing import Optional, List
import threading
import time

import pandas as pd
import numpy as np

_stream_thread: Optional[threading.Thread] = None
_stream_stop = threading.Event()
_latest_frame: Optional[pd.DataFrame] = None


def load_sensor_csvs(folder: str) -> pd.DataFrame:
	base = Path(folder)
	frames = []
	for p in base.glob("*.csv"):
		df = pd.read_csv(p)
		if "timestamp" in df.columns:
			df["timestamp"] = pd.to_datetime(df["timestamp"])
		df["sensor_id"] = p.stem
		frames.append(df)
	if not frames:
		return pd.DataFrame(columns=["timestamp", "sensor_id"])
	return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def _simulate(folder: str, interval_seconds: float) -> None:
	global _latest_frame
	df = load_sensor_csvs(folder)
	if df.empty:
		_latest_frame = df
		return
	# Stream by timestamp order
	for _, row in df.iterrows():
		if _stream_stop.is_set():
			break
		if _latest_frame is None or _latest_frame.empty:
			_latest_frame = pd.DataFrame([row])
		else:
			_latest_frame = pd.concat([_latest_frame, pd.DataFrame([row])], ignore_index=True)
		time.sleep(interval_seconds)


def start_stream_simulation(folder: str, interval_seconds: float = 1.0) -> None:
	global _stream_thread
	stop_stream_simulation()
	_stream_stop.clear()
	_stream_thread = threading.Thread(target=_simulate, args=(folder, interval_seconds), daemon=True)
	_stream_thread.start()


def stop_stream_simulation() -> None:
	_stream_stop.set()
	global _stream_thread
	if _stream_thread and _stream_thread.is_alive():
		_stream_thread.join(timeout=0.1)
	_stream_thread = None


def get_latest_stream_frame() -> Optional[pd.DataFrame]:
	return _latest_frame


def detect_anomalies(frame: pd.DataFrame, rolling: int = 50, z_thresh: float = 3.0) -> pd.DataFrame:
	if frame is None or frame.empty:
		return pd.DataFrame()
	numeric_cols = [c for c in frame.columns if c not in ["timestamp", "sensor_id"]]
	if not numeric_cols:
		return pd.DataFrame()
	f = frame.copy()
	for c in numeric_cols:
		f[f"{c}_mean"] = f[c].rolling(rolling, min_periods=max(2, rolling//2)).mean()
		f[f"{c}_std"] = f[c].rolling(rolling, min_periods=max(2, rolling//2)).std().fillna(0.0)
		f[f"{c}_z"] = (f[c] - f[f"{c}_mean"]) / (f[f"{c}_std"] + 1e-6)
		f[f"{c}_anomaly"] = f[f"{c}_z"].abs() > z_thresh
	anom_rows = []
	for c in numeric_cols:
		mask = f[f"{c}_anomaly"]
		if mask.any():
			anom_rows.append(pd.DataFrame({
				"timestamp": f.loc[mask, "timestamp"],
				"metric": c,
				"z": f.loc[mask, f"{c}_z"],
			}))
	if not anom_rows:
		return pd.DataFrame(columns=["timestamp", "metric", "z"]) 
	return pd.concat(anom_rows).sort_values("timestamp")


def maintenance_recommendations(frame: pd.DataFrame, anomalies: pd.DataFrame) -> List[str]:
	recs: List[str] = []
	if anomalies is None or anomalies.empty:
		recs.append("No anomalies detected. Continue routine maintenance schedule.")
		return recs
	count_by_metric = anomalies["metric"].value_counts().to_dict()
	for metric, cnt in count_by_metric.items():
		if "temp" in metric.lower():
			recs.append(f"High variance in {metric}: Inspect cooling loops, check refrigerant levels, and verify sensor calibration.")
		elif "vibration" in metric.lower():
			recs.append(f"Vibration spikes in {metric}: Schedule bearing inspection and balance rotating components.")
		elif "power" in metric.lower() or "kw" in metric.lower():
			recs.append(f"Power anomalies in {metric}: Check motor load, drive settings, and potential phase imbalance.")
		else:
			recs.append(f"Anomalies in {metric}: Perform equipment inspection and review recent maintenance logs.")
	return recs


def efficiency_tips(frame: pd.DataFrame) -> List[str]:
	tips: List[str] = []
	if frame is None or frame.empty:
		return tips
	# Simple heuristics
	numeric_cols = [c for c in frame.columns if c not in ["timestamp", "sensor_id"]]
	if any("kw" in c.lower() or "power" in c.lower() for c in numeric_cols):
		tips.append("Consider optimizing setpoints during low occupancy to reduce kW draw.")
	if any("temp" in c.lower() for c in numeric_cols):
		tips.append("Tighten temperature deadbands to avoid short cycling.")
	tips.append("Review AHU schedules to align with occupancy and reduce after-hours runtime.")
	return tips
