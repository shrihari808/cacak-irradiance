"""Build the monthly NASA POWER series for the Cacak grid cell.

Fetches daily values for 43.89 N, 20.35 E and averages them to calendar months,
giving 507 months from January 1984 to March 2026. The clearness index and
surface albedo are only produced from 2001 onward, so 303 of those months are
complete and the earlier 204 are left empty for the imputation step.

Two sources are available. The daily API is the one the analysis uses. A wide
"Monthly and Annual" POWER export can be reshaped instead for offline work, but
that product rounds the clearness index and albedo to two decimals, which
removes most of their distinct values, and it lags the daily archive.

    python build_cacak_monthly.py --source daily
    python build_cacak_monthly.py --source csv
"""

import argparse
import calendar
import glob
import os

import numpy as np
import pandas as pd

PARAMS = ["ALLSKY_KT", "ALLSKY_SRF_ALB", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"]
MONTHS = [m.upper() for m in calendar.month_abbr[1:]]  # JAN .. DEC

# Paths are anchored on this file so the script runs from any directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "cacak-monthly.csv")

LAT, LON = 43.89, 20.35
START, END = "19840101", "20260331"
POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"


def find_power_csv():
    """Locate a POWER export in data/ by its header rather than its filename.

    Exports get renamed, but every one carries the same NASA/POWER preamble.
    """
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        with open(path) as fh:
            head = fh.read(2048)
        if "NASA/POWER" in head and "-END HEADER-" in head:
            return path
    raise SystemExit(
        f"No NASA POWER export found in {DATA_DIR}. Pass one with --input, "
        "or use --source daily to fetch from the API instead."
    )


def from_csv(path):
    """Reshape the wide POWER monthly export into a Date-indexed frame."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    # The preamble varies in length and ends with a -END HEADER- line.
    header = next(i for i, ln in enumerate(lines) if ln.startswith("-END HEADER-"))
    wide = pd.read_csv(path, skiprows=header + 1)

    # ANN is an annual roll-up and is not used.
    wide = wide.drop(columns=["ANN"], errors="ignore")

    long = wide.melt(
        id_vars=["PARAMETER", "YEAR"],
        value_vars=MONTHS,
        var_name="MONTH",
        value_name="value",
    )
    long["Date"] = pd.to_datetime(
        long["YEAR"].astype(str) + long["MONTH"].map(lambda m: MONTHS.index(m) + 1).astype(str).str.zfill(2),
        format="%Y%m",
    )
    df = long.pivot(index="Date", columns="PARAMETER", values="value")
    df.columns.name = None
    return df.replace(-999.0, np.nan)


def from_daily():
    """Fetch the POWER daily series and average it to monthly means."""
    import requests

    r = requests.get(
        POWER_DAILY,
        params={
            "parameters": ",".join(PARAMS),
            "community": "RE",
            "latitude": LAT,
            "longitude": LON,
            "start": START,
            "end": END,
            "format": "JSON",
        },
        timeout=120,
    )
    r.raise_for_status()
    daily = pd.DataFrame(r.json()["properties"]["parameter"])
    daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
    daily = daily.replace(-999.0, np.nan)
    return daily.resample("MS").mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["csv", "daily"], default="csv")
    ap.add_argument("--input", help="POWER monthly CSV (default: autodetected in data/)")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if args.source == "daily":
        df = from_daily()
        origin = f"POWER daily API, {START}-{END}, averaged to months"
    else:
        path = args.input or find_power_csv()
        df = from_csv(path)
        origin = path

    df = df.reindex(columns=PARAMS)
    df.index.name = "Date"

    # Drop the leading years POWER has nothing for. Months missing only the
    # clearness index and albedo are kept as NaN: the model scripts drop them,
    # and the imputation step needs them present.
    df = df.dropna(how="all")

    df.to_csv(args.output, float_format="%.6f")

    complete = df.dropna()
    print(f"source            : {origin}")
    print(f"wrote             : {args.output}")
    print(f"rows              : {len(df)}  ({df.index.min():%Y-%m} to {df.index.max():%Y-%m})")
    print(f"complete rows     : {len(complete)}  ({complete.index.min():%Y-%m} to {complete.index.max():%Y-%m})")
    print(f"columns           : {', '.join(df.columns)}")
    if len(complete) != 303:
        print(f"note              : manuscript reports 303 complete months; this file has {len(complete)}.")


if __name__ == "__main__":
    main()
