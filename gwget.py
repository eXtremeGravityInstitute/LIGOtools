#!/usr/bin/env python3
"""Fetch, downsample, crop, and save frame data from the LIGO data grid.

Examples
--------
Fetch H1 calibrated strain from GPS 1126259460 to 1126259464, downsample to
2048 Hz using LALSuite's resampler, and write two-column GPS/strain data to
``data.txt``:

    ./gwget.py H1:GDS-CALIB_STRAIN 1126259460 1126259464

Use a different target sample rate, padding interval, and output file:

    ./gwget.py L1:GDS-CALIB_STRAIN 1126259460 1126259464 --rate 1024 --padding 4 --output L1_downsampled.dat

Use GWPy's resampler instead of the default LALSuite resampler:

    ./gwget.py H1:GDS-CALIB_STRAIN 1126259460 1126259464 --resampler gwpy

The script fetches an extra ``--padding`` seconds on each side before
resampling, then crops back to the requested GPS start/end times.  This keeps
the FIR resampling transients out of the saved analysis segment.
"""

from __future__ import annotations

import argparse
import math
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
        "--resampler",
        choices=("lal", "gwpy"),
        default="lal",
        help="resampler used after fetching padded data; default lal uses LALSuite's Kaiser-windowed sinc routine",
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
    save_two_column_arrays(times, strain, output_file)


def save_two_column_arrays(times: np.ndarray, strain: np.ndarray, output_file: str) -> None:
    if times.size != strain.size:
        raise RuntimeError("mismatched time and strain arrays")
    np.savetxt(output_file, np.column_stack((times, strain)), fmt="%.18e")


def gwpy_sample_rate_hz(series) -> float:
    """Return a GWPy sample rate as a plain floating-point Hz value."""

    sample_rate = series.sample_rate
    if hasattr(sample_rate, "to_value"):
        return float(sample_rate.to_value("Hz"))
    if hasattr(sample_rate, "value"):
        return float(sample_rate.value)
    return float(sample_rate)


def lal_epoch_to_float(epoch) -> float:
    """Convert a LAL LIGOTimeGPS-like object into floating GPS seconds."""

    if hasattr(epoch, "gpsSeconds") and hasattr(epoch, "gpsNanoSeconds"):
        return float(epoch.gpsSeconds) + 1.0e-9 * float(epoch.gpsNanoSeconds)
    return float(epoch)


def crop_regular_arrays(times: np.ndarray, strain: np.ndarray, start: float, end: float,
                        sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Crop regularly sampled arrays by nearest sample index."""

    dt = 1.0 / sample_rate
    start_index = int(round((start - times[0]) / dt))
    sample_count = int(round((end - start) * sample_rate))
    end_index = start_index + sample_count
    if sample_count <= 0:
        raise RuntimeError("requested crop has no samples")
    if start_index < 0 or end_index > strain.size:
        raise RuntimeError("resampled data do not cover the requested cropped interval")
    return times[start_index:end_index].copy(), strain[start_index:end_index].copy()


def lal_resample_and_crop(series, target_rate: float, start: float,
                          end: float) -> tuple[np.ndarray, np.ndarray]:
    """Use LALSuite XLALResampleREAL8TimeSeries, then crop padded samples."""

    try:
        import lal
    except ImportError as exc:
        raise RuntimeError("LAL resampling requires LALSuite Python bindings: import lal failed") from exc

    resample = getattr(lal, "ResampleREAL8TimeSeries", None)
    if resample is None:
        raise RuntimeError("this LALSuite Python module does not expose ResampleREAL8TimeSeries")

    times = np.asarray(series.times.value, dtype=np.float64)
    strain = np.asarray(series.value, dtype=np.float64)
    if times.size != strain.size:
        raise RuntimeError("GWPy returned mismatched time and strain arrays")
    if times.size < 2:
        raise RuntimeError("not enough samples to resample")

    input_rate = gwpy_sample_rate_hz(series)
    input_dt = 1.0 / input_rate
    target_dt = 1.0 / target_rate
    unit = getattr(lal, "StrainUnit", getattr(lal, "DimensionlessUnit", None))
    if unit is None:
        raise RuntimeError("LALSuite Python module does not expose a usable strain unit")

    lal_series = lal.CreateREAL8TimeSeries(
        "strain", lal.LIGOTimeGPS(float(times[0])), 0.0, input_dt, unit, strain.size
    )
    lal_series.data.data[:] = strain

    maybe_series = resample(lal_series, target_dt)
    if hasattr(maybe_series, "data"):
        lal_series = maybe_series

    output = np.asarray(lal_series.data.data, dtype=np.float64).copy()
    output_dt = float(lal_series.deltaT)
    output_rate = 1.0 / output_dt
    output_start = lal_epoch_to_float(lal_series.epoch)
    output_times = output_start + output_dt * np.arange(output.size, dtype=np.float64)
    return crop_regular_arrays(output_times, output, start, end, output_rate)


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

    original_rate = gwpy_sample_rate_hz(raw_data)
    sample_rate_matches_target = math.isclose(original_rate, args.rate, rel_tol=1.0e-6, abs_tol=1.0e-6)

    if args.no_resample or sample_rate_matches_target:
        prepared_data = raw_data
        if args.no_resample:
            print("Skipping resampling")
        else:
            print("Sample rate already matches target; skipping resampling")
        clean_data = prepared_data.crop(args.start, args.end)
        output_path = Path(args.output)
        print(f"Cropped span: {clean_data.span}")
        print(f"Output samples: {clean_data.size}")
        save_two_column_timeseries(clean_data, str(output_path))
    else:
        output_path = Path(args.output)
        if args.resampler == "lal":
            print(f"Resampling to {args.rate:g} Hz with LALSuite")
            times, strain = lal_resample_and_crop(raw_data, args.rate, args.start, args.end)
            print(f"Output samples: {strain.size}")
            save_two_column_arrays(times, strain, str(output_path))
        else:
            print(f"Resampling to {args.rate:g} Hz with GWPy")
            prepared_data = raw_data.resample(args.rate)
            print(f"Downsampled sample rate: {prepared_data.sample_rate}")
            clean_data = prepared_data.crop(args.start, args.end)
            print(f"Cropped span: {clean_data.span}")
            print(f"Output samples: {clean_data.size}")
            save_two_column_timeseries(clean_data, str(output_path))
    print(f"Wrote data to {output_path}")


if __name__ == "__main__":
    main()
