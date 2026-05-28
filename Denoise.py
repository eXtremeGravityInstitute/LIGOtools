#!/usr/bin/env python
"""Numba-ready Python port of Denoise.c.

The numerical kernels keep the same half-complex Fourier layout and
time-frequency reconstruction logic used by the C/GSL implementation.

Cluster output notes:
Lines of the form

    SNR of cluster k = S (t, f)

report the reconstructed time-domain SNR of cluster k, followed by the
time and frequency of the highest-power Q-transform pixel in that labeled
cluster. The coordinate is a label for where the cluster is loudest in the
time-frequency map; it is not a statement that all of the cluster power is
localized at that point.

The SNR is computed only from pixels inside the Tukey-safe time interval,
excluding the first and last T_RISE seconds. The peak-pixel coordinate is
computed from the full labeled cluster, including those excluded edge
regions. Therefore an edge-only cluster can be reported with SNR = 0, while
a cluster whose loudest pixel lies in the excluded region can still have
nonzero SNR if the same connected cluster extends into the interior window.

python -m py_compile Denoise.py

examples. On a LIGO machine
./Denoise.py 1126259462.4 8 --channel H1:GDS-CALIB_STRAIN --nyquist 1024

Using GWOSC open data
./Denoise.py 1126259462.4 8 --source open --ifo L1 --nyquist 1024 --snr-thresh 2

Using an existing two-column strain file
./Denoise.py --file H1.txt --nyquist 1024

"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def configure_matplotlib_cache_dir() -> None:
    def usable(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.write_text("", encoding="utf-8")
            test_file.unlink()
        except OSError:
            return False
        return True

    existing = os.environ.get("MPLCONFIGDIR")
    if existing and usable(Path(existing).expanduser()):
        return

    candidates: list[Path] = []
    for env_name in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser() / "matplotlib")
    candidates.append(Path(tempfile.gettempdir()) / "matplotlib")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(Path(xdg_cache).expanduser() / "matplotlib")
    home = os.environ.get("HOME")
    if home:
        candidates.append(Path(home).expanduser() / ".cache" / "matplotlib")
    candidates.append(Path.cwd() / ".matplotlib")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if usable(resolved):
            os.environ["MPLCONFIGDIR"] = str(resolved)
            return


configure_matplotlib_cache_dir()

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numba is absent
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def wrap(func):
            return func

        return wrap


PI = 3.1415926535897931159979634685442
TPI = 6.2831853071795862319959269370884
RTPI = 1.772453850905516
LN2 = 0.6931471805599453

SNRthresh = 4.0
STHRESH = 9.0
WARM = 5.0
QS = 8.0
T_RISE = 1.0
VERBOSE = 0
PRINT_QSCAN = 0
LINEMUL = 9.0
QPRINT = 8.0
QSCAN_SUBSCALE = 40
DEFAULT_ANALYSIS_FMAX = 1024.0
TRIGGER_OFFSET_FROM_END = 4.0


@dataclass
class PSDProducts:
    dec: int
    n: int
    nf: int
    tobs: float
    times: np.ndarray
    downsampled_data: np.ndarray
    freqs: np.ndarray
    dfreq: np.ndarray
    asd: np.ndarray
    smasd: np.ndarray
    clean_filename: str


def _require_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("data provided does not lead to 2^n samples")


def valid_power_of_two_frequency(value: float) -> tuple[bool, int]:
    nearest = int(round(value))
    if not math.isclose(value, float(nearest), rel_tol=0.0, abs_tol=1.0e-9):
        return False, nearest
    if nearest <= 0 or (nearest & (nearest - 1)) != 0:
        return False, nearest
    return True, nearest


def set_clean_filename(input_file: str) -> str:
    path = Path(input_file)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_clean{path.suffix}"))
    return str(path.with_name(f"{path.name}_clean"))


def segment_bounds(trigger_time: float, duration: float) -> tuple[float, float]:
    end = trigger_time + TRIGGER_OFFSET_FROM_END
    start = end - duration
    return start, end


def fetch_ligo_data(trigger_time: float, duration: float, ifo: str, source: str, channel: str | None, output_file: str | None) -> str:
    start, end = segment_bounds(trigger_time, duration)
    outfile = output_file if output_file is not None else f"{ifo}.txt"

    print(f"Fetching {ifo} data from {start:.6f} to {end:.6f}")
    print(f"Trigger time {trigger_time:.6f} is {TRIGGER_OFFSET_FROM_END:.1f} seconds before segment end")

    try:
        if source == "open":
            from gwosc.timeline import get_segments
            from gwpy.timeseries import TimeSeries

            segments = get_segments(f"{ifo}_DATA", int(math.floor(start)), int(math.ceil(end)))
            if len(segments) == 0:
                raise RuntimeError(f"no {ifo} open-data segment covers the requested time")
            series = TimeSeries.fetch_open_data(ifo, start, end)
        else:
            from gwpy.timeseries import TimeSeries

            channel_name = channel if channel is not None else f"{ifo}:GDS-CALIB_STRAIN"
            print(f"Using frame channel {channel_name}")
            series = TimeSeries.get(channel_name, start, end)
    except ImportError as exc:
        raise RuntimeError("gwpy/gwosc are required for trigger-time fetching") from exc

    times = np.asarray(series.times.value, dtype=np.float64)
    strain = np.asarray(series.value, dtype=np.float64)
    np.savetxt(outfile, np.column_stack((times, strain)), fmt="%.18e")
    print(f"Fetched data written to {outfile}")
    return outfile


def real_fft_to_halfcomplex(data: np.ndarray) -> np.ndarray:
    """Match gsl_fft_real_radix2_transform half-complex packing."""
    n = data.shape[0]
    nh = n // 2
    z = np.fft.rfft(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = z[0].real
    out[nh] = z[nh].real
    idx = np.arange(1, nh)
    out[idx] = z[idx].real
    out[n - idx] = z[idx].imag
    return out


def halfcomplex_inverse(data: np.ndarray) -> np.ndarray:
    """Match gsl_fft_halfcomplex_radix2_inverse scaling and packing."""
    n = data.shape[0]
    nh = n // 2
    z = np.empty(nh + 1, dtype=np.complex128)
    z[0] = data[0] + 0.0j
    z[nh] = data[nh] + 0.0j
    idx = np.arange(1, nh)
    z[idx] = data[idx] + 1j * data[n - idx]
    return np.fft.irfft(z, n).astype(np.float64)


@njit(cache=True)
def tukey_inplace(data: np.ndarray, alpha: float) -> None:
    n = data.shape[0]
    imin = int(alpha * float(n - 1) / 2.0)
    imax = int(float(n - 1) * (1.0 - alpha / 2.0))
    nwin = n - imax
    for i in range(n):
        filt = 1.0
        if i < imin:
            filt = 0.5 * (1.0 + math.cos(PI * (float(i) / float(imin) - 1.0)))
        if i > imax:
            filt = 0.5 * (1.0 + math.cos(PI * (float(i - imax) / float(nwin))))
        data[i] *= filt


@njit(cache=True)
def bwbpf_inplace(data: np.ndarray, fwrv: int, order: int, sample_rate: float, f1: float, f2: float) -> None:
    if order % 4 != 0:
        return
    m = data.shape[0]
    n = order // 4
    a = math.cos(PI * (f1 + f2) / sample_rate) / math.cos(PI * (f1 - f2) / sample_rate)
    a2 = a * a
    b = math.tan(PI * (f1 - f2) / sample_rate)
    b2 = b * b
    A = np.empty(n, dtype=np.float64)
    d1 = np.empty(n, dtype=np.float64)
    d2 = np.empty(n, dtype=np.float64)
    d3 = np.empty(n, dtype=np.float64)
    d4 = np.empty(n, dtype=np.float64)
    w0 = np.zeros(n, dtype=np.float64)
    w1 = np.zeros(n, dtype=np.float64)
    w2 = np.zeros(n, dtype=np.float64)
    w3 = np.zeros(n, dtype=np.float64)
    w4 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = math.sin(PI * (2.0 * float(i) + 1.0) / (4.0 * float(n)))
        s = b2 + 2.0 * b * r + 1.0
        A[i] = b2 / s
        d1[i] = 4.0 * a * (1.0 + b * r) / s
        d2[i] = 2.0 * (b2 - 2.0 * a2 - 1.0) / s
        d3[i] = 4.0 * a * (1.0 - b * r) / s
        d4[i] = -(b2 - 2.0 * b * r + 1.0) / s

    out = np.empty(m, dtype=np.float64)
    for j in range(m):
        if fwrv == 1:
            x = data[j]
        else:
            x = data[m - j - 1]
        for i in range(n):
            w0[i] = d1[i] * w1[i] + d2[i] * w2[i] + d3[i] * w3[i] + d4[i] * w4[i] + x
            x = A[i] * (w0[i] - 2.0 * w2[i] + w4[i])
            w4[i] = w3[i]
            w3[i] = w2[i]
            w2[i] = w1[i]
            w1[i] = w0[i]
        if fwrv == 1:
            out[j] = x
        else:
            out[m - j - 1] = x
    for j in range(m):
        data[j] = out[j]


@njit(cache=True)
def f_nwip(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    arg = 0.0
    for i in range(1, n // 2):
        j = i
        k = n - j
        arg += a[j] * b[j] + a[k] * b[k]
    return arg


@njit(cache=True)
def sine_gaussian_c_inplace(hs: np.ndarray, f0: float, qval: float, tobs: float) -> None:
    n = hs.shape[0]
    tau = qval / (TPI * f0)
    fmx = f0 + 3.0 / tau
    fmn = f0 - 3.0 / tau
    imid = int(f0 * tobs)
    imin = int(fmn * tobs)
    imax = int(fmx * tobs)
    if imax - imin < 10:
        imin = imid - 5
        imax = imid + 5
    if imin < 0:
        imin = 1
    if imax > n // 2:
        imax = n // 2

    hs[0] = 0.0
    hs[n // 2] = 0.0
    for i in range(1, n // 2):
        hs[i] = 0.0
        hs[n - i] = 0.0

    dt = tobs / float(n)
    fac = math.sqrt(math.sqrt(2.0) * PI * tau / dt)
    p = PI * PI * tau * tau / (tobs * tobs)
    bm = math.exp(-p * ((float(imid) - f0 * tobs) * (float(imid) - f0 * tobs)))
    cm = 1.0
    bstep = math.exp(-p * (1.0 + 2.0 * (float(imid) - f0 * tobs)))
    cstep = math.exp(-2.0 * p)

    B = bm
    C = cm
    for i in range(imid, imax):
        sf = fac * B
        hs[i] = sf
        hs[n - i] = 0.0
        B *= C * bstep
        C *= cstep

    B = bm
    C = cm
    bstep = math.exp(p * (-1.0 + 2.0 * (float(imid) - f0 * tobs)))
    for i in range(imid, imin, -1):
        sf = fac * B
        hs[i] = sf
        hs[n - i] = 0.0
        B *= C * bstep
        C *= cstep


@njit(cache=True)
def recursive_phase_evolution(dre: float, dim: float, cos_phase: float, sin_phase: float) -> tuple[float, float]:
    x = cos_phase * dre + sin_phase * dim
    y = sin_phase * dre - cos_phase * dim
    return cos_phase - x, sin_phase - y


@njit(cache=True)
def sine_gaussian_f_inplace(hs: np.ndarray, t0: float, f0: float, qval: float, amp: float, phi: float, tobs: float) -> None:
    n = hs.shape[0]
    tau = qval / (TPI * f0)
    fmx = f0 + 3.0 / tau
    fmn = f0 - 3.0 / tau
    imid = int(f0 * tobs)
    imin = int(fmn * tobs)
    imax = int(fmx * tobs)
    if imax - imin < 10:
        imin = imid - 5
        imax = imid + 5
    if imin < 0:
        imin = 1
    if imax > n // 2:
        imax = n // 2

    hs[0] = 0.0
    hs[n // 2] = 0.0
    for i in range(1, n // 2):
        hs[i] = 0.0
        hs[n - i] = 0.0

    dim = math.sin(TPI * t0 / tobs)
    dre = math.sin(0.5 * (TPI * t0 / tobs))
    dre = 2.0 * dre * dre
    amplitude = 0.5 * amp * RTPI * tau
    q = qval * qval / (f0 * tobs)
    p = PI * PI * tau * tau / (tobs * tobs)

    Am = math.exp(-q * float(imid))
    Bm = math.exp(-p * ((float(imid) - f0 * tobs) * (float(imid) - f0 * tobs)))
    Cm = 1.0
    astep = math.exp(-q)
    bstep = math.exp(-p * (1.0 + 2.0 * (float(imid) - f0 * tobs)))
    cstep = math.exp(-2.0 * p)

    f = float(imid) / tobs
    phase = TPI * f * t0
    cos_m0 = math.cos(phase - phi)
    sin_m0 = math.sin(phase - phi)
    cos_p0 = math.cos(phase + phi)
    sin_p0 = math.sin(phase + phi)

    A = Am
    B = Bm
    C = Cm
    cos_m = cos_m0
    sin_m = sin_m0
    cos_p = cos_p0
    sin_p = sin_p0
    for i in range(imid, imax):
        sf = amplitude * B
        sx = A
        hs[i] = sf * (cos_m + sx * cos_p)
        hs[n - i] = -sf * (sin_m + sx * sin_p)
        A *= astep
        B *= C * bstep
        C *= cstep
        cos_m, sin_m = recursive_phase_evolution(dre, dim, cos_m, sin_m)
        cos_p, sin_p = recursive_phase_evolution(dre, dim, cos_p, sin_p)

    A = Am
    B = Bm
    C = Cm
    cos_m = cos_m0
    sin_m = sin_m0
    cos_p = cos_p0
    sin_p = sin_p0
    astep = 1.0 / astep
    bstep = math.exp(p * (-1.0 + 2.0 * (float(imid) - f0 * tobs)))
    dim *= -1.0
    for i in range(imid, imin, -1):
        sf = amplitude * B
        sx = A
        hs[i] = sf * (cos_m + sx * cos_p)
        hs[n - i] = -sf * (sin_m + sx * sin_p)
        A *= astep
        B *= C * bstep
        C *= cstep
        cos_m, sin_m = recursive_phase_evolution(dre, dim, cos_m, sin_m)
        cos_p, sin_p = recursive_phase_evolution(dre, dim, cos_p, sin_p)


@njit(cache=True)
def phase_products_inplace(corr: np.ndarray, corrf: np.ndarray, data1: np.ndarray, data2: np.ndarray) -> None:
    n = data1.shape[0]
    nb2 = n // 2
    corr[0] = 0.0
    corrf[0] = 0.0
    corr[nb2] = 0.0
    corrf[nb2] = 0.0
    for i in range(1, nb2):
        l = i
        k = n - i
        corr[l] = data1[l] * data2[l] + data1[k] * data2[k]
        corr[k] = data1[k] * data2[l] - data1[l] * data2[k]
        corrf[l] = corr[k]
        corrf[k] = -corr[l]


@njit(cache=True)
def layer_finish_inplace(tf: np.ndarray, tfR: np.ndarray, tfI: np.ndarray, ac: np.ndarray, af: np.ndarray, bmag: float) -> None:
    n = tf.shape[0]
    for i in range(n):
        tfR[i] = ac[i] / bmag
        tfI[i] = af[i] / bmag
        tf[i] = tfR[i] * tfR[i] + tfI[i] * tfI[i]


def layer_c(data: np.ndarray, freq: float, qval: float, tobs: float, fix: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = data.shape[0]
    b = np.empty(n, dtype=np.float64)
    sine_gaussian_c_inplace(b, freq, qval, tobs)
    bmag = math.sqrt(f_nwip(b, b) / float(n))
    bmag /= fix

    corr = np.zeros(n, dtype=np.float64)
    corrf = np.zeros(n, dtype=np.float64)
    phase_products_inplace(corr, corrf, data, b)
    ac = halfcomplex_inverse(corr)
    af = halfcomplex_inverse(corrf)

    tf = np.empty(n, dtype=np.float64)
    tfR = np.empty(n, dtype=np.float64)
    tfI = np.empty(n, dtype=np.float64)
    layer_finish_inplace(tf, tfR, tfI, ac, af, bmag)
    return tf, tfR, tfI


def transform_c(data: np.ndarray, freqs: np.ndarray, qval: float, tobs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = freqs.shape[0]
    n = data.shape[0]
    fix = math.sqrt(float(n // 2))
    tf = np.empty((m, n), dtype=np.float64)
    tfR = np.empty((m, n), dtype=np.float64)
    tfI = np.empty((m, n), dtype=np.float64)
    for j in range(m):
        tf[j], tfR[j], tfI[j] = layer_c(data, freqs[j], qval, tobs, fix)
    return tf, tfR, tfI


@njit(cache=True)
def akima_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    slopes = np.empty(n + 3, dtype=np.float64)
    for i in range(n - 1):
        slopes[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    slopes[1] = 2.0 * slopes[2] - slopes[3]
    slopes[0] = 3.0 * slopes[2] - 2.0 * slopes[3]
    slopes[n + 1] = 2.0 * slopes[n] - slopes[n - 1]
    slopes[n + 2] = 3.0 * slopes[n] - 2.0 * slopes[n - 1]

    deriv = np.empty(n, dtype=np.float64)
    for i in range(n):
        w1 = abs(slopes[i + 3] - slopes[i + 2])
        w2 = abs(slopes[i + 1] - slopes[i])
        if w1 + w2 > 0.0:
            deriv[i] = (w1 * slopes[i + 1] + w2 * slopes[i + 2]) / (w1 + w2)
        else:
            deriv[i] = 0.5 * (slopes[i + 1] + slopes[i + 2])
    return deriv


@njit(cache=True)
def akima_eval(x: np.ndarray, y: np.ndarray, deriv: np.ndarray, xe: float) -> float:
    n = x.shape[0]
    if xe <= x[0]:
        return y[0]
    if xe >= x[n - 1]:
        return y[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xe:
            lo = mid
        else:
            hi = mid
    h = x[lo + 1] - x[lo]
    t = (xe - x[lo]) / h
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[lo]
        + (t3 - 2.0 * t2 + t) * h * deriv[lo]
        + (-2.0 * t3 + 3.0 * t2) * y[lo + 1]
        + (t3 - t2) * h * deriv[lo + 1]
    )


@njit(cache=True)
def median_sorted(values: np.ndarray) -> float:
    n = values.shape[0]
    if n % 2 == 1:
        return values[n // 2]
    return 0.5 * (values[n // 2 - 1] + values[n // 2])


@njit(cache=True)
def medspecspline(data: np.ndarray, df: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = data.shape[0]
    half = n // 2
    S = np.zeros(half, dtype=np.float64)
    SN = np.zeros(half, dtype=np.float64)
    SM = np.zeros(half, dtype=np.float64)
    for i in range(1, half):
        S[i] = 2.0 * (data[i] * data[i] + data[n - i] * data[n - i])
    S[0] = S[1]

    fspace = 2.0
    k = int(float(half) * df / fspace) + 3
    sc = np.zeros(k, dtype=np.float64)
    fc = np.zeros(k, dtype=np.float64)
    fw = fspace
    fc[0] = df
    fend = float(half - 1) * df

    i = 0
    while True:
        i += 1
        fc[i] = fspace * float(i)
        if not (fc[i] < fend):
            break
    nc = i + 1

    mm = int(fspace / df)
    mw = int(fw / df)
    for j in range(1, nc - 1):
        if fc[j] >= 30.0 - 0.5 * fspace and fc[j] < 30.0 + 0.5 * fspace:
            fw *= 2.0
            mw = int(fw / df)
        if fc[j] >= 100.0 - 0.5 * fspace and fc[j] < 100.0 + 0.5 * fspace:
            fw *= 2.0
            mw = int(fw / df)
        if fc[j] >= 200.0 - 0.5 * fspace and fc[j] < 200.0 + 0.5 * fspace:
            fw *= 2.0
            mw = int(fw / df)
        if fc[j] + fw > fend:
            fw = fend - fc[j]
            mw = int(fw / df)
        chunk = np.empty(2 * mw, dtype=np.float64)
        start = j * mm - mw
        for ii in range(2 * mw):
            chunk[ii] = S[start + ii]
        chunk.sort()
        sc[j] = math.log(median_sorted(chunk) / LN2)

    sc[0] = sc[1]
    sc[nc - 1] = sc[nc - 2]
    xnodes = fc[:nc]
    ynodes = sc[:nc]
    deriv = akima_derivatives(xnodes, ynodes)

    SM[1] = math.exp(sc[0])
    SM[half - 1] = math.exp(sc[nc - 1])
    for i in range(2, half - 1):
        SM[i] = math.exp(akima_eval(xnodes, ynodes, deriv, float(i) * df))
    SM[0] = SM[1]

    for i in range(1, half):
        SN[i] = SM[i]
        x = S[i] / SM[i]
        if x > LINEMUL:
            SN[i] = S[i]
    SN[0] = SN[1]
    return S, SN, SM


@njit(cache=True)
def whiten_inplace(data: np.ndarray, sn: np.ndarray) -> None:
    n = data.shape[0]
    data[0] = 0.0
    data[n // 2] = 0.0
    for i in range(1, n // 2):
        x = 1.0 / math.sqrt(sn[i])
        data[i] *= x
        data[n - i] *= x


@njit(cache=True)
def isums(x: float) -> tuple[float, float, float]:
    x2 = x * x
    x3 = x2 * x
    return x, (x2 + x) / 2.0, (2.0 * x3 + 3.0 * x2 + x) / 6.0


@njit(cache=True)
def solve_2x2(m00: float, m01: float, m10: float, m11: float, y0: float, y1: float) -> tuple[float, float]:
    det = m00 * m11 - m01 * m10
    return (y0 * m11 - m01 * y1) / det, (m00 * y1 - y0 * m10) / det


def makespec(SM: np.ndarray, freqs: np.ndarray, istart: int, iend: int, splinef: np.ndarray, splineA: np.ndarray) -> None:
    deriv = akima_derivatives(splinef, splineA)
    SM[istart] = math.exp(splineA[0])
    SM[iend] = math.exp(splineA[-1])
    for i in range(istart + 1, iend):
        SM[i] = math.exp(akima_eval(splinef, splineA, deriv, freqs[i]))


def splinespace(ns: int, istart: int, iend: int, SM: np.ndarray, freqs: np.ndarray, tobs: float) -> tuple[np.ndarray, np.ndarray]:
    lsm = np.zeros(ns, dtype=np.float64)
    sfit = np.zeros(ns, dtype=np.float64)
    for i in range(istart, iend + 1):
        lsm[i] = math.log(SM[i])

    splinef = [freqs[istart]]
    splineA = [lsm[istart]]
    inc = int(tobs * 8.0)
    chunks = (iend - istart) // inc
    scale = np.zeros(chunks, dtype=np.float64)
    for i in range(chunks):
        y2 = 0.0
        ym0 = 0.0
        ym1 = 0.0
        cnt = 0.0
        for j in range(inc):
            cnt += 1.0
            ii = istart + i * inc + j
            y2 += lsm[ii] * lsm[ii]
            ym0 += lsm[ii]
            ym1 += cnt * lsm[ii]
        im0, im1, im2 = isums(cnt)
        a0, a1 = solve_2x2(im0, im1, im1, im2, ym0, ym1)
        chisqlin = y2 - 2.0 * a0 * ym0 - 2.0 * a1 * ym1
        chisqlin += a0 * a0 * im0 + 2.0 * a0 * a1 * im1 + a1 * a1 * im2
        scale[i] = chisqlin / cnt

    xtol = np.sort(scale)[int(0.95 * float(chunks))]
    inc = int(tobs)
    chunks = (iend - istart) // inc
    y2 = 0.0
    ym0 = 0.0
    ym1 = 0.0
    cnt = 0.0
    kk = istart
    kkold = istart
    a0 = 0.0
    a1 = 0.0
    for i in range(chunks):
        for j in range(inc):
            cnt += 1.0
            ii = istart + i * inc + j
            y2 += lsm[ii] * lsm[ii]
            ym0 += lsm[ii]
            ym1 += cnt * lsm[ii]
        im0, im1, im2 = isums(cnt)
        a0, a1 = solve_2x2(im0, im1, im1, im2, ym0, ym1)
        chisqlin = y2 - 2.0 * a0 * ym0 - 2.0 * a1 * ym1
        chisqlin += a0 * a0 * im0 + 2.0 * a0 * a1 * im1 + a1 * a1 * im2
        x = cnt / tobs
        z = chisqlin / cnt
        if x >= 8.0 and (z > xtol or x >= 256.0):
            kk = istart + (i + 1) * inc - 1
            splinef.append(freqs[kk])
            splineA.append(lsm[kk])
            cnt_fit = 0.0
            for ii in range(kkold, kk):
                cnt_fit += 1.0
                sfit[ii] = a0 + a1 * cnt_fit
            kkold = kk
            y2 = 0.0
            ym0 = 0.0
            ym1 = 0.0
            cnt = 0.0

    if kk < iend:
        splinef.append(freqs[iend])
        splineA.append(lsm[iend])
    while len(splinef) < 7:
        spacings = np.diff(np.asarray(splinef))
        j = int(np.argmax(spacings))
        xmid = 0.5 * (splinef[j + 1] + splinef[j])
        idx = int(xmid * tobs)
        splinef.append(freqs[idx])
        splineA.append(lsm[idx])
        order = np.argsort(np.asarray(splinef))
        splinef = list(np.asarray(splinef)[order])
        splineA = list(np.asarray(splineA)[order])
    return np.asarray(splinef, dtype=np.float64), np.asarray(splineA, dtype=np.float64)


@njit(cache=True)
def clean_threshold_model(tfD: np.ndarray, tfR: np.ndarray, sqf: np.ndarray, scale: float, imin: int, imax: int) -> tuple[np.ndarray, float]:
    nf, n = tfD.shape
    live = np.empty((nf, n), dtype=np.int8)
    live2 = np.empty((nf, n), dtype=np.int8)
    for j in range(nf):
        for i in range(n):
            hot = 1 if tfD[j, i] > STHRESH else -1
            live[j, i] = hot
            live2[j, i] = hot

    for j in range(1, nf - 1):
        for i in range(1, n - 1):
            flag = 0
            for jj in range(-1, 2):
                for ii in range(-1, 2):
                    if live[j + jj, i + ii] > 0:
                        flag = 1
            if flag == 1 and tfD[j, i] > WARM:
                live2[j, i] = 1

    dtemp = np.zeros(n, dtype=np.float64)
    for j in range(nf):
        for i in range(imin, imax):
            if live2[j, i] > 0:
                dtemp[i] += scale * sqf[j] * tfR[j, i]

    snr = 0.0
    for i in range(imin, imax):
        snr += dtemp[i] * dtemp[i]
    return dtemp, math.sqrt(snr)


def clean(D: np.ndarray, Draw: np.ndarray, sqf: np.ndarray, freqs: np.ndarray, df: float, qval: float, tobs: float, scale: float, imin: int, imax: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    n = D.shape[0]
    dtemp = Draw.copy()
    Df = real_fft_to_halfcomplex(D)
    Dtempf = real_fft_to_halfcomplex(dtemp)
    Sn, specD, sspecD = medspecspline(Df, df)
    whiten_inplace(Dtempf, specD)
    tfD, tfR, _ = transform_c(Dtempf, freqs, qval, tobs)
    model_white, snr = clean_threshold_model(tfD, tfR, sqf, scale, imin, imax)
    print(f"Excess SNR at Q {qval:f} = {snr:f}")

    model_f = real_fft_to_halfcomplex(model_white)
    model_f[0] = 0.0
    for i in range(1, n // 2):
        x = math.sqrt(sspecD[i])
        model_f[i] *= x
        model_f[n - i] *= x
    model_f[n // 2] = 0.0
    glitch_t = halfcomplex_inverse(model_f) / math.sqrt(float(2 * n))
    cleaned = Draw - glitch_t
    return cleaned, snr, Sn, specD, sspecD


def getscale(freqs: np.ndarray, qval: float, tobs: float, fmx: float, n: int) -> float:
    data = np.empty(n, dtype=np.float64)
    ref = np.empty(n, dtype=np.float64)
    f = fmx / 4.0
    t0 = tobs / 2.0
    delt = tobs / 8.0
    dt = tobs / float(n)
    for i in range(n):
        t = float(i) * dt
        x = (t - t0) / delt
        x = x * x / 2.0
        data[i] = math.cos(TPI * t * f) * math.exp(-x)
        ref[i] = data[i]

    data_f = real_fft_to_halfcomplex(data)
    _, tfR, _ = transform_c(data_f, freqs, qval, tobs)
    intime = np.zeros(n, dtype=np.float64)
    for j in range(freqs.shape[0]):
        sqf = math.sqrt(freqs[j])
        intime += sqf * tfR[j]

    xsum = 0.0
    count = 0
    for i in range(n):
        if abs(ref[i]) > 0.01:
            count += 1
            xsum += intime[i] / ref[i]
    xsum /= math.sqrt(float(2 * n))
    return float(count) / xsum


def specest(data: np.ndarray, dt: float, fmin: float, fmx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = data.shape[0]
    tobs = float(n) * dt
    df = 1.0 / tobs
    fny = 1.0 / (2.0 * dt)
    if fmx > fny:
        fmx = fny

    D = data.copy()
    Draw = data.copy()
    imin = int(2.0 * T_RISE / dt)
    imax = n - imin

    qval = QS
    dx = math.log(2.0) / float(QSCAN_SUBSCALE)
    nf = int(math.floor(math.log(fmx / fmin) / dx)) + 1
    freqs = np.empty(nf, dtype=np.float64)
    sqf = np.empty(nf, dtype=np.float64)
    x = math.log(fmin)
    for i in range(nf):
        freqs[i] = math.exp(x)
        sqf[i] = math.sqrt(freqs[i])
        x += dx

    scale = getscale(freqs, qval, tobs, fmx, n)
    snr_old = 0.0
    D, snr, Sn, specD, sspecD = clean(D, Draw, sqf, freqs, df, qval, tobs, scale, imin, imax)

    i = 0
    while i < 10 and (snr - snr_old) > 10.0:
        snr_old = snr
        D, snr, Sn, specD, sspecD = clean(D, Draw, sqf, freqs, df, qval, tobs, scale, imin, imax)
        i += 1

    np.savetxt("glitchSNR.dat", np.array([snr]), fmt="%f")
    Df = real_fft_to_halfcomplex(D)
    Sn, specD, sspecD = medspecspline(Df, df)
    fac = tobs / float(n * n)
    SM = sspecD * fac
    SN = specD * fac
    PS = Sn * fac
    data[:] = D
    return SN, SM, PS


@njit(cache=True)
def flood_fill(img: np.ndarray, x: int, y: int, new_clr: int) -> None:
    m, n = img.shape
    prev = img[x, y]
    if prev == new_clr:
        return
    size = m * n
    qx = np.empty(size, dtype=np.int64)
    qy = np.empty(size, dtype=np.int64)
    front = 0
    rear = 0
    qx[rear] = x
    qy[rear] = y
    rear += 1
    img[x, y] = new_clr
    while front < rear:
        x = qx[front]
        y = qy[front]
        front += 1
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                xx = x + dx
                yy = y + dy
                if xx >= 0 and xx < m and yy >= 0 and yy < n and img[xx, yy] == prev:
                    img[xx, yy] = new_clr
                    qx[rear] = xx
                    qy[rear] = yy
                    rear += 1


@njit(cache=True)
def nongaussian_cluster_core(freqs: np.ndarray, tfD: np.ndarray, tfR: np.ndarray, scale: float, tobs: float, snr_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, int, float, float, float, float, float]:
    nf, nt = tfD.shape
    live = np.empty((nf, nt), dtype=np.int64)
    live2 = np.empty((nf, nt), dtype=np.int64)
    hot0 = 0
    for j in range(nf):
        for i in range(nt):
            live[j, i] = -1
            if tfD[j, i] > STHRESH:
                live[j, i] = 1
                hot0 += 1
            live2[j, i] = live[j, i]

    for j in range(1, nf - 1):
        for i in range(1, nt - 1):
            flag = 0
            for jj in range(-1, 2):
                for ii in range(-1, 2):
                    if live[j + jj, i + ii] == 1:
                        flag = 1
            if flag == 1 and tfD[j, i] > WARM:
                live2[j, i] = 1

    hot = 0
    for j in range(nf):
        for i in range(nt):
            if live2[j, i] == 1:
                hot += 1

    new_clr = 1
    size = nf * nt
    while True:
        flag = 0
        cnt = 0
        xpix = 0
        ypix = 0
        while flag == 0 and cnt < size:
            xpix = cnt % nf
            ypix = cnt // nf
            if live2[xpix, ypix] == 1:
                flag = 1
            cnt += 1
        if flag == 0:
            break
        new_clr += 1
        flood_fill(live2, xpix, ypix, new_clr)

    cf = new_clr - 1
    total_all = np.zeros(nt, dtype=np.float64)
    total_sig = np.zeros(nt, dtype=np.float64)
    if cf <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return total_all, total_sig, empty, empty, empty, empty, hot0, float(hot), -1, 0.0, 0.0, 0.0, 0.0, 0.0

    DT = np.zeros((cf, nt), dtype=np.float64)
    num = np.zeros(cf, dtype=np.int64)
    dt = tobs / float(nt)
    imin = int(T_RISE / tobs * float(nt))
    imax = int((tobs - T_RISE) / tobs * float(nt))
    for j in range(nf):
        sqf = scale * math.sqrt(freqs[j])
        for i in range(imin, imax):
            if live2[j, i] > 0:
                jj = live2[j, i] - 2
                DT[jj, i] += sqf * tfR[j, i]
                num[jj] += 1

    csnr = np.zeros(cf, dtype=np.float64)
    smax = 0.0
    jmax = 0
    for j in range(cf):
        s = 0.0
        for i in range(imin, imax):
            s += DT[j, i] * DT[j, i]
        csnr[j] = math.sqrt(s)
        if csnr[j] > smax:
            smax = csnr[j]
            jmax = j

    itmin = nt
    itmax = 0
    ifmin = nf
    ifmax = 0
    fmin = 0.0
    fmax = 0.0
    tmin = 0.0
    tmax = 0.0
    peak_t = np.zeros(cf, dtype=np.float64)
    peak_f = np.zeros(cf, dtype=np.float64)
    peak_amp = np.empty(cf, dtype=np.float64)
    for j in range(cf):
        peak_amp[j] = -1.0e300
    for j in range(nf):
        for i in range(nt):
            if live2[j, i] > 1:
                jj = live2[j, i] - 2
                amp = tfD[j, i]
                if amp > peak_amp[jj]:
                    peak_amp[jj] = amp
                    peak_t[jj] = dt * float(i)
                    peak_f[jj] = freqs[j]
    for j in range(nf):
        for i in range(imin, imax):
            if live2[j, i] > 1:
                jj = live2[j, i] - 2
                if jj == jmax:
                    if j < ifmin:
                        ifmin = j
                        fmin = freqs[j]
                    if j > ifmax:
                        ifmax = j
                        fmax = freqs[j]
                    if i < itmin:
                        itmin = i
                        tmin = dt * float(i)
                    if i > itmax:
                        itmax = i
                        tmax = dt * float(i)
    for j in range(cf):
        if peak_amp[j] < -1.0e200:
            peak_amp[j] = 0.0

    for j in range(cf):
        for i in range(nt):
            total_all[i] += DT[j, i]
            if csnr[j] > snr_thresh:
                total_sig[i] += DT[j, i]
    return total_all, total_sig, csnr, peak_t, peak_f, peak_amp, hot0, float(hot), jmax, smax, tmin, tmax, fmin if ifmin < nf else 0.0, fmax if ifmax > 0 else 0.0


def qscan_stats(tfD: np.ndarray, tobs: float) -> tuple[float, float]:
    nt = tfD.shape[1]
    dt = tobs / float(nt)
    total = 0.0
    count = 0
    above = 0
    for j in range(tfD.shape[0]):
        for i in range(nt):
            t = float(i) * dt
            if t > 1.0 and t < tobs - 1.0:
                total += tfD[j, i]
                count += 1
                if tfD[j, i] > 9.0:
                    above += 1
    if count == 0:
        return 0.0, 0.0
    return total / float(count), float(above) / float(count)


def save_qscan_plot(filename: str, tfD: np.ndarray, freqs: np.ndarray, tobs: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    nt = tfD.shape[1]
    dt = tobs / float(nt)
    times = np.arange(nt, dtype=np.float64) * dt
    mask_t = (times > 1.0) & (times < tobs - 1.0)
    mask_f = (freqs >= 16.0) & (freqs <= 1024.0)

    colors = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"]
    cmap = LinearSegmentedColormap.from_list("qscan_orred", colors, N=256)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    image = ax.pcolormesh(
        times[mask_t],
        freqs[mask_f],
        tfD[np.ix_(mask_f, mask_t)],
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=12.0,
    )
    ax.set_yscale("log")
    ax.set_ylim(16.0, 1024.0)
    ax.set_yticks([16, 32, 64, 128, 248, 512, 1024])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("t (s)", fontsize=26)
    ax.set_ylabel("f (Hz)", fontsize=26)
    ax.tick_params(axis="both", labelsize=22)
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)
    print(f"Q-scan plot written to {filename}")


def save_blink_gif(raw_png: str = "Qscan.png", clean_png: str = "Qscan_clean.png", gif_name: str = "Qscan_blink.gif") -> None:
    import imageio.v2 as imageio
    from PIL import Image

    raw = np.asarray(Image.open(raw_png).convert("RGB"))
    clean = np.asarray(Image.open(clean_png).convert("RGB"))
    imageio.mimsave(gif_name, [raw, clean], duration=700, loop=0)
    print(f"Blink map written to {gif_name}")


def nongaussian(data: np.ndarray, freqs: np.ndarray, smasd: np.ndarray, tfD: np.ndarray, tfR: np.ndarray, fmax: float, qval: float, tobs: float, dec: int, snr_thresh: float) -> np.ndarray:
    nt = data.shape[0]
    dt = tobs / float(nt)
    scale = getscale(freqs, qval, tobs, fmax, nt)
    print(f"{nt:d} {freqs.shape[0]:d}")
    total_all, total_sig, csnr, peak_t, peak_f, _peak_amp, hot0, hot, jmax, smax, tmin, tmax, fmin, fmax_found = nongaussian_cluster_core(freqs, tfD, tfR, scale, tobs, snr_thresh)
    print(f"{hot0:d} initial hot pixels")
    print(f"{int(hot):d} hot pixels")
    print(f"{csnr.shape[0]:d} clusters")
    for j in range(csnr.shape[0]):
        # The coordinate labels the highest-power pixel in the full cluster.
        # The SNR is computed only from the Tukey-safe interior pixels.
        print(f"SNR of cluster {j:d} = {csnr[j]:f} ({peak_t[j]:.2f}, {peak_f[j]:.2f})")
    print(f"Loudest cluster is #{jmax:d} SNR = {smax:f}")
    print(f"tmin {tmin:f} tmax {tmax:f} fmin {fmin:f} fmax {fmax_found:f}")

    dec_scale = math.sqrt(float(dec))
    np.savetxt("excess.dat", np.column_stack((np.arange(nt) * dt, total_all / dec_scale)), fmt="%e")
    significant = int(np.sum(csnr > snr_thresh))
    print(f"{significant:d} significant cluster(s)")
    np.savetxt("features.dat", np.column_stack((np.arange(nt) * dt, total_sig / dec_scale)), fmt="%e")

    S = math.sqrt(float(np.dot(total_sig, total_sig)))
    print(f"Excess SNR = {S:f}")
    model_f = real_fft_to_halfcomplex(total_sig)
    norm = math.sqrt(float(2 * nt))
    data -= model_f / norm
    sf = 0.0
    for i in range(1, nt // 2):
        sf += 2.0 * (model_f[i] * model_f[i] + model_f[nt - i] * model_f[nt - i])
    print(f"Excess computed in frequency domain SNR = {math.sqrt(sf / float(nt)):f}")

    fcut = 20.0
    model_f[0] = 0.0
    for i in range(1, nt // 2):
        x = smasd[i]
        f = float(i) / tobs
        if f < fcut:
            x *= math.exp(f - fcut)
        model_f[i] *= x
        model_f[nt - i] *= x
    model_f[nt // 2] = 0.0
    freqs_out = np.arange(1, nt // 2, dtype=np.float64) / tobs
    np.savetxt(
        "colored_template.dat",
        np.column_stack((freqs_out, model_f[1 : nt // 2], model_f[nt - 1 : nt // 2 : -1])),
        fmt="%e",
    )

    colored = halfcomplex_inverse(model_f) / norm
    np.savetxt("features_colored.dat", np.column_stack((np.arange(nt) * dt, colored)), fmt="%e")
    return colored


def qscanf(data: np.ndarray, smasd: np.ndarray, qval: float, tobs: float, fmin: float, fmax: float, dec: int, snr_thresh: float) -> np.ndarray:
    octaves = int(np.rint(math.log(fmax / fmin) / math.log(2.0)))
    nf = QSCAN_SUBSCALE * octaves + 1
    freqs = np.empty(nf, dtype=np.float64)
    dx = math.log(2.0) / float(QSCAN_SUBSCALE)
    x = math.log(fmin)
    for i in range(nf):
        freqs[i] = math.exp(x)
        x += dx
    print(f"frequency layers = {nf:d} fmin {fmin:e} fmax {fmax:e}")

    tfD, tfR, _ = transform_c(data, freqs, qval, tobs)
    feature = nongaussian(data, freqs, smasd, tfD, tfR, fmax, qval, tobs, dec, snr_thresh)
    mean, frac = qscan_stats(tfD, tobs)
    save_qscan_plot("Qscan.png", tfD, freqs, tobs)
    print(f"{mean:f} {frac:f}")

    tfD_clean, _, _ = transform_c(data, freqs, qval, tobs)
    save_qscan_plot("Qscan_clean.png", tfD_clean, freqs, tobs)
    save_blink_gif()
    return feature


def estimate_psd_products(input_file: str, fmax: float) -> PSDProducts:
    print("starting PSD estimation")
    raw = np.loadtxt(input_file, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("expected two-column input: time, hoft")
    time_f = raw[:, 0].copy()
    data_f = raw[:, 1].copy()
    nd = data_f.shape[0]
    _require_power_of_two(nd)

    dt = (time_f[nd - 1] - time_f[0]) / float(nd)
    tobs = float(np.rint(float(nd) * dt))
    if tobs < 8.0:
        raise ValueError(f"input segment duration is {tobs:g} seconds; Denoise requires at least 8 seconds")
    dt = tobs / float(nd)
    fny = 1.0 / (2.0 * dt)
    print(f"{tobs:f} {dt:e} {fny:f} {fmax:f}")

    dec = int(fny / fmax)
    if dec > 8:
        dec = 8
    if dec < 1:
        dec = 1
    n = nd // dec

    if dec > 1:
        fmn = 1.0 / tobs
        bwbpf_inplace(data_f, 1, 8, 1.0 / dt, fmax, fmn)
        bwbpf_inplace(data_f, -1, 8, 1.0 / dt, fmax, fmn)

    times = time_f[::dec][:n].copy()
    data = data_f[::dec][:n].copy()
    downsampled = data.copy()

    dt = tobs / float(n)
    fmin = 1.0 / tobs
    alpha = 2.0 * T_RISE / tobs
    tukey_inplace(data, alpha)
    dfreq = real_fft_to_halfcomplex(data)

    freqs = np.arange(n // 2, dtype=np.float64) / tobs
    start = time.process_time()
    SN, SM, _ = specest(data, dt, fmin, fmax)
    print(f"spectrum took {time.process_time() - start:f} seconds")

    return PSDProducts(
        dec=dec,
        n=n,
        nf=n // 2,
        tobs=tobs,
        times=times,
        downsampled_data=downsampled,
        freqs=freqs,
        dfreq=dfreq,
        asd=SN,
        smasd=SM,
        clean_filename=set_clean_filename(input_file),
    )


def extract_features_from_products(products: PSDProducts, snr_thresh: float) -> None:
    nf = products.nf
    nt = products.n
    tobs = products.tobs
    print(f"Observation time {tobs:f}")
    print(f"Frequency bins = {nf:d}")

    data = products.dfreq.copy()
    dataw = np.zeros(nt, dtype=np.float64)
    asd = np.zeros(nf, dtype=np.float64)
    smasd = np.zeros(nf, dtype=np.float64)
    for i in range(1, nf):
        asd[i] = math.sqrt(products.asd[i])
        smasd[i] = math.sqrt(products.smasd[i])

    fac = float(nt) / math.sqrt(tobs)
    for i in range(1, nf):
        asd[i] *= fac
        smasd[i] *= fac

    x = 0.0
    count = 0
    for i in range(1, nf):
        data[i] /= asd[i]
        data[nt - i] /= asd[i]
        x += 2.0 * (data[i] * data[i] + data[nt - i] * data[nt - i])
        count += 1
    print(f"white var {x / float(count):e}")

    dt = tobs / float(nt)
    dataw[1:] = data[1:]
    dataw_t = halfcomplex_inverse(dataw)
    np.savetxt("wdata.dat", np.column_stack((np.arange(nt) * dt, dataw_t * math.sqrt(float(2 * nt)))), fmt="%e")

    feature = qscanf(data, smasd, 8.0, tobs, products.freqs[1], products.freqs[nf - 1], products.dec, snr_thresh)
    cleaned = products.downsampled_data - feature
    np.savetxt(products.clean_filename, np.column_stack((products.times, cleaned)), fmt="%.17e")
    print(f"Cleaned data written to {products.clean_filename}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=Path(argv[0]).name,
        description="Fetch a LIGO strain segment by trigger time, or run Denoise on an existing two-column strain file.",
    )
    parser.add_argument("trigger_time", type=float, nargs="?", help="GPS trigger time; omit when using --file")
    parser.add_argument("duration", type=float, nargs="?", help="segment duration in seconds; omit when using --file")
    parser.add_argument("--ifo", default="H1", choices=("H1", "L1"), help="interferometer to fetch, default H1")
    parser.add_argument(
        "--source",
        default="cluster",
        choices=("cluster", "open"),
        help="fetch from a LIGO cluster frame channel or GWOSC open data, default %(default)s",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="frame channel for --source cluster, default <IFO>:GDS-CALIB_STRAIN",
    )
    parser.add_argument(
        "--file",
        dest="input_file",
        default=None,
        help="existing two-column strain text file; skips fetching and infers duration from the file",
    )
    parser.add_argument(
        "--nyquist",
        "--fmax",
        dest="nyquist",
        type=float,
        default=DEFAULT_ANALYSIS_FMAX,
        help="analysis Nyquist frequency in Hz; must be a power of two, default %(default)g",
    )
    parser.add_argument(
        "--snr-thresh",
        "--SNRthresh",
        dest="snr_thresh",
        type=float,
        default=SNRthresh,
        help="cluster SNR threshold for feature removal, default %(default)g",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional fetched strain text file name, default <IFO>.txt",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ok, nyquist = valid_power_of_two_frequency(args.nyquist)
    if not ok:
        print(f"warning: requested Nyquist frequency {args.nyquist:g} Hz is not a power of two")
        return 1
    if not math.isfinite(args.snr_thresh):
        print(f"warning: requested SNR threshold {args.snr_thresh:g} is not finite")
        return 1
    if not NUMBA_AVAILABLE:
        print("warning: numba is not installed; running the same Python kernels without JIT")

    if args.input_file is not None:
        input_file = args.input_file
        if args.trigger_time is not None or args.duration is not None:
            print("warning: --file was supplied, so trigger_time and duration are ignored")
    else:
        if args.trigger_time is None or args.duration is None:
            print("warning: provide trigger_time and duration, or use --file INPUT")
            return 1
        if args.duration < 8.0:
            print(f"warning: requested segment duration is {args.duration:g} seconds; Denoise requires at least 8 seconds")
            return 1
        input_file = fetch_ligo_data(args.trigger_time, args.duration, args.ifo, args.source, args.channel, args.output)

    try:
        products = estimate_psd_products(input_file, float(nyquist))
    except (OSError, ValueError) as exc:
        print(f"warning: {exc}")
        return 1
    extract_features_from_products(products, args.snr_thresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
