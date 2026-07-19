# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Generate a marginally-interesting synthetic hourly signal for poking at.

Writes two CSVs to scratch/ (both gitignored):
  synthetic_hourly.csv        timestamp, y, z   (what a "user" sees; y has a gap)
  synthetic_hourly_truth.csv  timestamp, trend, seasonal, exog, spikes, noise

Reproducible (fixed seed). Run: uv run python scratch/make_synthetic_hourly.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20240718)
T = 24 * 90
t = np.arange(T)
idx = pd.date_range("2024-01-01 00:00", periods=T, freq="h")

u = (t - T / 2) / T                 # vertex at mid-series
trend = 40.0 * u**2 - 5.0            # ~5-unit valley, clearly non-monotone

def harm(period, amps):
    s = np.zeros(T)
    for k, a in enumerate(amps, start=1):
        s += a * np.sin(2 * np.pi * k * t / period + 0.5 * k)
    return s
daily = harm(24, [3.0, 1.0, 0.4])
weekly = harm(24 * 7, [1.8, 0.6])
seasonal = daily + weekly

white = rng.standard_normal(T)
kernel = np.ones(49) / 49.0
z = np.convolve(white, kernel, mode="same")
z = (z - z.mean()) / z.std()
sigmoid = 1.0 / (1.0 + np.exp(-z))
exog = 4.0 * (sigmoid - 0.5)

spikes = np.zeros(T)
eligible = np.r_[np.arange(0, 24 * 40), np.arange(24 * 42, T)]  # exclude the gap
spike_idx = rng.choice(eligible, size=5, replace=False)
spikes[spike_idx] = rng.uniform(8.0, 14.0, 5) * rng.choice([-1, 1], 5)

noise = 0.5 * rng.standard_normal(T)

y_full = trend + seasonal + exog + spikes + noise
y = y_full.copy()
gap = slice(24 * 40, 24 * 42)
y[gap] = np.nan

pd.DataFrame({"timestamp": idx, "y": y, "z": z}).to_csv(
    "scratch/synthetic_hourly.csv", index=False)
pd.DataFrame({
    "timestamp": idx, "trend": trend, "seasonal": seasonal,
    "exog": exog, "spikes": spikes, "noise": noise,
}).to_csv("scratch/synthetic_hourly_truth.csv", index=False)

obs = ~np.isnan(y)
recon_err = np.max(np.abs(y[obs] - y_full[obs]))
print(f"T={T}  gap={gap.start}..{gap.stop} ({gap.stop-gap.start}h)  spikes at {sorted(spike_idx.tolist())}")
print(f"y range [{np.nanmin(y):.2f}, {np.nanmax(y):.2f}]  z range [{z.min():.2f}, {z.max():.2f}]")
print(f"trend range [{trend.min():.2f}, {trend.max():.2f}]  exog range [{exog.min():.2f}, {exog.max():.2f}]")
print(f"consistency (off-gap |y - sum|) = {recon_err:.2e}")
print("wrote scratch/synthetic_hourly.csv and scratch/synthetic_hourly_truth.csv")
