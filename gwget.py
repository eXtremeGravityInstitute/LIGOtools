#!/usr/bin/env python3
"""Fetch, downsample, crop, and save frame data from the LIGO data grid.

Examples
--------
Fetch H1 calibrated strain from GPS 1126259460 to 1126259464, downsample to
2048 Hz, and write two-column GPS/strain data to ``data.txt``:

    ./gwget.py H1:GDS-CALIB_STRAIN 1126259460 1126259464

Use a different target sample rate, padding interval, and output file:

    ./gwget.py L1:GDS-CALIB_STRAIN 1126259460 1126259464 --rate 1024 --padding 4 --output L1_downsampled.dat

The script fetches an extra ``--padding`` seconds on each side before
resampling, then crops back to the requested GPS start/end times.  This keeps
the FIR resampling transients out of the saved analysis segment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch LIGO frame data, downsample it, crop to the requested GPS interval, and save GPS/strain columns."
    )
    parser.add_argument("channel", help="Frame channel name, e.g. H1:GDS-CALIB_STRAIN.")
    parser.add_argument("start", type=float, help="Analysis start time in GPS seconds.")
    parser.add_argument("end", type=float, help="Analysis end time in GPS seconds.")
    parser.add_argument(
        "--rate",
        type=float,
        default=2048.0,
        help="Target sample rate in Hz after downsampling. Default: 2048.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=2.0,
        help="Seconds fetched on each side before resampling, then cropped away. Default: 2.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data.txt",
        help="Output filename for the final two-column GPS/strain data. Default: data.txt.",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Skip resampling and only crop the fetched frame data.",
    )
    return parser.parse_args()


def save_two_column_timeseries(series, output_file: str) -> None:
    times = np.asarray(series.times.value, dtype=np.float64)
    strain = np.asarray(series.value, dtype=np.float64)
    if times.size != strain.size:
        raise RuntimeError("GWPy returned mismatched time and strain arrays")
    np.savetxt(output_file, np.column_stack((times, strain)), fmt="%.18e")


def main() -> None:
    args = parse_args()

    if args.end <= args.start:
        raise ValueError("end time must be greater than start time")
    if args.rate <= 0.0:
        raise ValueError("target sample rate must be positive")
    if args.padding < 0.0:
        raise ValueError("padding must be non-negative")

    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise SystemExit("gwget.py requires gwpy: pip install gwpy") from exc

    fetch_start = args.start - args.padding
    fetch_end = args.end + args.padding

    print(f"Fetching {args.channel} from GPS {fetch_start:.6f} to {fetch_end:.6f}")
    raw_data = TimeSeries.get(args.channel, fetch_start, fetch_end)
    print(f"Original sample rate: {raw_data.sample_rate}")

    if args.no_resample:
        prepared_data = raw_data
        print("Skipping resampling")
    else:
        print(f"Resampling to {args.rate:g} Hz")
        prepared_data = raw_data.resample(args.rate)
        print(f"Downsampled sample rate: {prepared_data.sample_rate}")

    clean_data = prepared_data.crop(args.start, args.end)
    print(f"Cropped span: {clean_data.span}")
    print(f"Output samples: {clean_data.size}")

    output_path = Path(args.output)
    save_two_column_timeseries(clean_data, str(output_path))
    print(f"Wrote data to {output_path}")


if __name__ == "__main__":
    main()
