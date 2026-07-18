# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "rdtools @ git+https://github.com/NREL/rdtools.git",
#     "pandas",
# ]
# ///
"""One-time prep: produce the cached daily normalized energy `y` (PV DOMAIN layer).

THIS SCRIPT IS THE DOMAIN LAYER, NOT part of signaldecomp. It runs the
photovoltaic-specific pipeline (temperature correction, PVWatts normalization,
daily aggregation) via rdtools to turn the raw SUB-DAILY PVDAQ system 4 record
into DAILY NORMALIZED ENERGY -- the 1-D series `y` that the domain-agnostic
signaldecomp substrate then decomposes.

rdtools is declared ONLY in this script's inline (PEP 723) header, so it is
isolated to this script's ephemeral environment: neither the signaldecomp
library nor the example notebook depends on rdtools.

Run (uv builds the ephemeral env from the header):

    uv run examples/prep_pvdaq4_daily_y.py

Output (gitignored; regenerate locally):

    examples/pvdaq4_daily_y.pickle   # pandas Series: daily normalized energy

Data: PVDAQ system 4, public release asset used in rdtools' TrendAnalysis
examples.
"""

from pathlib import Path

import pandas as pd
import rdtools

FILE_URL = (
    "https://github.com/NatLabRockies/rdtools/releases/download/3.0.0/"
    "pvdaq_system_4_2010-2016_subset_soil_signal.csv"
)
OUT = Path(__file__).with_name("pvdaq4_daily_y.pickle")

META = {
    "timezone": "Etc/GMT+7",
    "gamma_pdc": -0.0034,
    "power_dc_rated": 1000.0,
    "temp_model_params": "open_rack_glass_polymer",
}


def main():
    print(f"downloading {FILE_URL} ...")
    df = pd.read_csv(FILE_URL, index_col=0, parse_dates=True)
    df = df.rename(
        columns={
            "ac_power": "power_ac",
            "ambient_temp": "Tamb",
            "poa_irradiance": "poa",
        }
    )
    df.index = df.index.tz_localize(META["timezone"])
    freq = pd.infer_freq(df.index[:10])
    print(f"loaded {len(df):,} rows, {df.index.min()} -> {df.index.max()}, freq={freq}")

    # --- PV DOMAIN pipeline (rdtools) -> daily normalized energy ---
    ta = rdtools.TrendAnalysis(
        df["power_ac"],
        df["poa"],
        temperature_ambient=df["Tamb"],
        gamma_pdc=META["gamma_pdc"],
        interp_freq=freq,
        windspeed=df["wind_speed"],
        power_dc_rated=META["power_dc_rated"],
        temperature_model=META["temp_model_params"],
    )
    ta.sensor_analysis(analyses=["yoy_degradation"])

    # Daily normalized energy: this is our `y`.
    y = ta.sensor_aggregated_performance.copy()
    y.name = "daily_normalized_energy"
    print(
        f"daily normalized energy: {len(y)} points, "
        f"{y.index.min().date()} -> {y.index.max().date()}, "
        f"{int(y.isna().sum())} missing"
    )

    y.to_pickle(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
