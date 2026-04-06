from __future__ import annotations

"""Generate a synthetic 1D bogie-vibration signal with recurring defect events.

The script writes three artifacts next to the output path:
- `<name>.npy`: full synthetic vibration signal
- `<name>_event_times.npy`: defect event timestamps in seconds
- `<name>_meta.json`: generation parameters and summary statistics

The generated signal combines:
- low-frequency baseline vibration components,
- impulse-like defect events with damped ringing,
- additive white noise,
- slow drift.

Generation design note:
This synthetic generator is configured using assumptions inspired by the data
description in the Kaggle dataset:
https://www.kaggle.com/datasets/tamaryovell/predictive-maintanace-train-bogie-vibrations
"""

import argparse
import json
from pathlib import Path

import numpy as np


def wheel_period_seconds(speed_kmh: float, wheel_diameter_m: float) -> float:
    """Return wheel rotation period in seconds.

    Args:
        speed_kmh: Train speed in km/h.
        wheel_diameter_m: Wheel diameter in meters.

    Returns:
        Rotation period of one wheel revolution in seconds.
    """
    speed_mps = speed_kmh / 3.6
    f_wheel_hz = speed_mps / (np.pi * wheel_diameter_m)
    return 1.0 / f_wheel_hz


def build_synthetic_signal(
    fs_hz: int = 575,
    duration_s: int = 60,
    speed_kmh: float = 80.0,
    wheel_diameter_m: float = 0.920,
    seed: int = 26,
    noise_std: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a synthetic vibration signal and companion metadata.

    Args:
        fs_hz: Sampling frequency in Hz.
        duration_s: Signal duration in seconds.
        speed_kmh: Train speed in km/h used to derive wheel periodicity.
        wheel_diameter_m: Wheel diameter in meters.
        seed: Random seed for reproducibility.
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        A tuple with:
        - signal: 1D float64 array of vibration samples
        - event_times: 1D array of injected defect-event timestamps in seconds
        - meta: dict with generation parameters and descriptive statistics
    """
    rng = np.random.default_rng(seed)

    n_samples = fs_hz * duration_s
    time_s = np.arange(n_samples) / fs_hz
    t_wheel_s = wheel_period_seconds(speed_kmh, wheel_diameter_m)

    x_base = (
        0.12 * np.sin(2 * np.pi * 7.5 * time_s)
        + 0.06 * np.sin(2 * np.pi * 15.0 * time_s + 0.7)
        + 0.03 * np.sin(2 * np.pi * 42.0 * time_s + 1.1)
    )

    event_times = np.arange(0.0, duration_s, t_wheel_s)
    event_times += rng.normal(0.0, 0.0012, size=event_times.size)
    event_times = event_times[(event_times > 0.0) & (event_times < duration_s)]

    x_defect = np.zeros_like(time_s)
    sigma_s = 0.0022
    ring_freq_hz = 95.0
    ring_tau_s = 0.018

    for t0 in event_times:
        dt = time_s - t0
        dt_pos = np.clip(dt, 0.0, None)
        impact = 0.42 * np.exp(-(dt**2) / (2 * sigma_s**2))
        ring = 0.09 * np.exp(-dt_pos / ring_tau_s) * np.sin(2 * np.pi * ring_freq_hz * dt_pos)
        x_defect += impact + ring

    noise = rng.normal(0.0, noise_std, size=n_samples)
    drift = 0.015 * np.sin(2 * np.pi * 0.25 * time_s)

    signal = (x_base + x_defect + noise + drift).astype(np.float64)

    meta = {
        "source_description": "Synthetic generator configured from Kaggle dataset description",
        "source_url": "https://www.kaggle.com/datasets/tamaryovell/predictive-maintanace-train-bogie-vibrations",
        "fs_hz": fs_hz,
        "duration_s": duration_s,
        "samples": int(n_samples),
        "speed_kmh": speed_kmh,
        "wheel_diameter_m": wheel_diameter_m,
        "wheel_period_s": float(t_wheel_s),
        "injected_event_count": int(event_times.size),
        "noise_std": noise_std,
        "seed": seed,
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "min": float(signal.min()),
        "max": float(signal.max()),
    }

    return signal, event_times, meta


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for output path and stochastic settings."""
    root = Path(__file__).resolve().parents[1]
    default_out = root / "data" / "synthetic_defect_signal.npy"

    parser = argparse.ArgumentParser(description="Generate a 60 s synthetic bogie vibration signal with defects.")
    parser.add_argument("--out", type=Path, default=default_out, help="Output .npy file path for the synthetic signal.")
    parser.add_argument("--seed", type=int, default=26, help="Random seed for reproducibility.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Standard deviation of additive noise.")
    return parser.parse_args()


def main() -> None:
    """Generate signal files and print a concise generation summary."""
    args = parse_args()
    signal, event_times, meta = build_synthetic_signal(seed=args.seed, noise_std=args.noise_std)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, signal)

    events_path = args.out.with_name(args.out.stem + "_event_times.npy")
    np.save(events_path, event_times)

    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved signal to: {args.out}")
    print(f"Saved event times to: {events_path}")
    print(f"Saved metadata to: {meta_path}")
    print(
        "Signal stats -> "
        f"mean={meta['mean']:.4f}, std={meta['std']:.4f}, "
        f"min={meta['min']:.4f}, max={meta['max']:.4f}"
    )


if __name__ == "__main__":
    main()
