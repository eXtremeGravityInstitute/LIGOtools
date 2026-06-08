#!/opt/anaconda3/bin/python
"""Anderson-Darling normality checks for frequency-domain data.

This script reads a PSD estimate and complex frequency-domain data, whitens the
real and imaginary parts, and applies variance and Anderson-Darling (AD)
normality checks in frequency bands.

Input files:

    psd.dat
        Two columns: frequency, one-sided PSD.

    datafreq.dat
        Three columns: frequency, Re(data), Im(data), on the same frequency
        grid as psd.dat.

The whitening convention is

    sample = Re(data) / sqrt(2 * PSD)
    sample = Im(data) / sqrt(2 * PSD)

so each frequency bin contributes two real samples. For each test band, the
code computes the sample variance and an AD statistic for the whitened samples.

Typical fitted-normal run:

    python ADtest.py psd.dat datafreq.dat 8

Known mean/variance run, appropriate when the null distribution is fully
specified as N(0, 1):

    python ADtest.py psd_welch_for_adtest.dat datafreq_for_adtest.dat 8 \
        --known-parameters --normal-mean 0 --normal-variance 1

Main outputs:

    ADtest.dat
        Three columns: band central frequency, AD p-value, sample variance.

    ADsummary.dat
        Text summary of frequency bands with variance more than 3 sigma above
        or below the expected value, and unusually small p-values. Low-p-value
        bins already flagged as strong spectral-line bins by the PSD variance
        test are excluded from that portion of the summary.

    ADplot.pdf
        Main diagnostic plot. Top panel shows band variance with 1, 2, and 3
        sigma bands; variance outliers outside the plot range are shown as
        stars at +/-4 sigma. Bottom panel shows AD p-values. Frequency bands
        dominated by loud spectral lines are shaded gray.

    Spectra.pdf
        Spectrum view with PSD-line bands shaded gray.

P-value modes:

    Default fitted-normal mode estimates the mean and variance in each band
    before applying the AD test, and uses the fitted-normal calibration table
    inherited from the C code.

    --known-parameters mode uses the supplied mean and variance directly. The
    p-value is mapped with the asymptotic simple-null AD distribution, or by
    finite-sample Monte Carlo if --pvalue-method monte-carlo is selected.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adtest_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import CubicSpline
    from scipy.special import ndtr
except ModuleNotFoundError as exc:
    raise SystemExit("ADtest.py requires scipy: install it with `python3 -m pip install scipy`.") from exc


TPI = 6.2831853071795862319959269370884
SQPI = 2.5066282746310002

# AD statistic to log(p-value) calibration table from the C code.
AV = np.array([
    0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
    0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38,
    0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58,
    0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
    0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
    1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18,
    1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38,
    1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58,
    1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78,
    1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,
    1.999800,
], dtype=float)

ADC = np.array([
    0.000000e+00, 0.000000e+00, 0.000000e+00, -6.000002e-07,
    -6.590217e-05, -9.718721e-04, -5.347673e-03, -1.702806e-02,
    -3.928110e-02, -7.370628e-02, -1.203567e-01, -1.783287e-01,
    -2.462505e-01, -3.227490e-01, -4.063929e-01, -4.960332e-01,
    -5.905115e-01, -6.890052e-01, -7.908381e-01, -8.952793e-01,
    -1.001889e+00, -1.110290e+00, -1.220061e+00, -1.331033e+00,
    -1.442868e+00, -1.555449e+00, -1.668667e+00, -1.782282e+00,
    -1.896382e+00, -2.010684e+00, -2.125117e+00, -2.239706e+00,
    -2.354675e+00, -2.469649e+00, -2.584469e+00, -2.699301e+00,
    -2.814132e+00, -2.929191e+00, -3.043736e+00, -3.158495e+00,
    -3.273014e+00, -3.387418e+00, -3.501725e+00, -3.616137e+00,
    -3.729707e+00, -3.843611e+00, -3.957667e+00, -4.072023e+00,
    -4.185272e+00, -4.298510e+00, -4.412142e+00, -4.525421e+00,
    -4.639168e+00, -4.752583e+00, -4.866879e+00, -4.979522e+00,
    -5.091913e+00, -5.205259e+00, -5.318487e+00, -5.430978e+00,
    -5.543193e+00, -5.655833e+00, -5.767883e+00, -5.880059e+00,
    -5.992341e+00, -6.104338e+00, -6.216144e+00, -6.327318e+00,
    -6.440462e+00, -6.552048e+00, -6.663304e+00, -6.776420e+00,
    -6.888600e+00, -7.001418e+00, -7.113071e+00, -7.226144e+00,
    -7.336909e+00, -7.445952e+00, -7.556292e+00, -7.668668e+00,
    -7.783096e+00, -7.896697e+00, -8.007098e+00, -8.114231e+00,
    -8.227214e+00, -8.335705e+00, -8.450189e+00, -8.558953e+00,
    -8.666854e+00, -8.777520e+00, -8.883918e+00, -8.994020e+00,
    -9.104360e+00, -9.212643e+00, -9.318371e+00, -9.425640e+00,
    -9.532062e+00, -9.645749e+00, -9.756448e+00, -9.868507e+00,
    -9.987087e+00,
], dtype=float)

AD_TO_LOGP = CubicSpline(AV, ADC, bc_type="natural")


def c_style_mean_variance(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    mean = float(np.sum(values) / n)
    variance = float(np.sum(values * values) / (n - 1) - mean * mean)
    return mean, variance


def upper_tail_series(u: float) -> float:
    u2 = u * u
    return 1.0 - 1.0 / u2 + 3.0 / (u2 * u2) - 15.0 / (u2**3) + 105.0 / (u2**4)


def gaussian_log_terms(u: float) -> tuple[float, float]:
    cdf = 0.5 * math.erfc(-u / math.sqrt(2.0))
    if 0.0 < cdf < 0.999:
        return math.log(cdf), math.log1p(-cdf)

    if cdf <= 0.0:
        z = upper_tail_series(-u)
        log_cdf = -u * u / 2.0 + math.log(z / (-u * SQPI))
        return log_cdf, -math.exp(log_cdf)

    z = upper_tail_series(u)
    tail = z * math.exp(-u * u / 2.0) / (u * SQPI)
    return -tail, -u * u / 2.0 + math.log(z / (u * SQPI))


def anderson_darling_statistic(standardized_samples: np.ndarray, fitted_normal: bool) -> float:
    """Compute the AD statistic from samples already mapped to N(0, 1).

    When mean and variance are fitted from the same data, Stephens' finite-n
    correction is applied. For the known-parameter test this correction is not
    used because the null CDF is fully specified before seeing the data.
    """
    samples = np.sort(standardized_samples)
    ntest = len(samples)
    total = 0.0

    for idx, sample in enumerate(samples, start=1):
        lx, ly = gaussian_log_terms(float(sample))
        total += ((2.0 * idx - 1.0) * lx + (2.0 * ntest - 2.0 * idx + 1.0) * ly) / ntest

    statistic = -float(ntest) - total
    if fitted_normal:
        statistic *= 1.0 + 0.75 / ntest + 2.25 / (ntest * ntest)

    return statistic


def fitted_normal_ad_pvalue(standardized_samples: np.ndarray, default_p: float) -> tuple[float, float]:
    statistic = anderson_darling_statistic(standardized_samples, fitted_normal=True)

    pvalue = default_p
    if statistic < AV[-1]:
        pvalue = float(math.exp(AD_TO_LOGP(statistic)))
    return statistic, pvalue


def known_normal_ad_asymptotic_pvalue(statistic: float) -> float:
    """Upper-tail p-value for the simple-null AD statistic.

    This is the common asymptotic AD CDF approximation for fully specified
    continuous distributions after transforming samples through the null CDF.
    """
    if statistic <= 0.0:
        return 1.0
    if statistic < 2.0:
        z = statistic
        cdf = (
            math.exp(-1.2337141 / z)
            / math.sqrt(z)
            * (
                2.00012
                + 0.247105 * z
                - 0.0649821 * z**2
                + 0.0347962 * z**3
                - 0.0116720 * z**4
                + 0.00168691 * z**5
            )
        )
    else:
        z = statistic
        cdf = math.exp(
            -math.exp(
                1.0776
                - 2.30695 * z
                + 0.43424 * z**2
                - 0.082433 * z**3
                + 0.008056 * z**4
                - 0.0003146 * z**5
            )
        )
    return min(1.0, max(0.0, 1.0 - cdf))


def vectorized_known_normal_ad(samples: np.ndarray) -> np.ndarray:
    samples = np.sort(samples, axis=1)
    ntest = samples.shape[1]
    weights = np.arange(1, ntest + 1, dtype=float)
    lower_weights = 2.0 * weights - 1.0
    upper_weights = 2.0 * ntest - 2.0 * weights + 1.0
    cdf = np.clip(ndtr(samples), np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    total = (np.log(cdf) * lower_weights + np.log1p(-cdf) * upper_weights).sum(axis=1) / ntest
    return -float(ntest) - total


def monte_carlo_known_normal_ad_pvalues(
    statistics: np.ndarray,
    ntest: int,
    draws: int,
    seed: int | None,
    chunk_size: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(len(statistics), dtype=int)
    remaining = draws
    if chunk_size is None:
        chunk_size = max(1, min(256, 2_000_000 // ntest))

    while remaining > 0:
        chunk = min(chunk_size, remaining)
        simulated = rng.normal(size=(chunk, ntest))
        null_statistics = vectorized_known_normal_ad(simulated)
        exceedances += np.count_nonzero(null_statistics[:, np.newaxis] >= statistics[np.newaxis, :], axis=0)
        remaining -= chunk

    return (exceedances + 1.0) / (draws + 1.0)


def load_inputs(psd_file: Path, data_file: Path) -> tuple[np.ndarray, np.ndarray]:
    psd = np.loadtxt(psd_file)
    data = np.loadtxt(data_file)
    if psd.ndim == 1:
        psd = psd.reshape(1, -1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if psd.shape[1] < 2:
        raise ValueError(f"{psd_file} must contain at least two columns")
    if data.shape[1] < 3:
        raise ValueError(f"{data_file} must contain at least three columns")

    rows = min(len(psd), len(data))
    if len(psd) != len(data):
        print(f"Warning: using first {rows} rows because input lengths differ")
    return psd[:rows, :2], data[:rows, :3]


def build_samples(psd: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    freqs = psd[:, 0].astype(float)
    scale = np.sqrt(2*psd[:, 1].astype(float))
    samples = np.empty(2 * len(freqs), dtype=float)
    samples[0::2] = data[:, 1].astype(float) / scale
    samples[1::2] = data[:, 2].astype(float) / scale
    df = float(freqs[1] - freqs[0])
    t_obs = 1.0 / df
    return freqs, samples, df, t_obs


def standardize_samples(values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    if variance <= 0.0:
        raise ValueError("normal variance must be positive")
    return (values - mean) / math.sqrt(variance)


def run_tests(
    psd: np.ndarray,
    data: np.ndarray,
    bandwidth: float,
    fmin: float = 20.0,
    known_parameters: bool = False,
    normal_mean: float = 0.0,
    normal_variance: float = 1.0,
    pvalue_method: str = "asymptotic",
    mc_draws: int = 20000,
    mc_seed: int | None = None,
) -> dict[str, object]:
    """Run the variance and AD tests over the full band and sub-bands.

    The first and last 20 Hz are skipped to avoid edge effects. The data are
    then split into bands of width ``bandwidth`` Hz. In each band the code
    computes a sample variance, an AD statistic, and a p-value. Separately, it
    checks the variation of log(PSD) in each band; large variation flags strong
    spectral-line regions that should be treated cautiously in the p-value
    summary.
    """
    freqs, samples, df, t_obs = build_samples(psd, data)
    nsamp = len(samples)
    nmin = 2 * int(t_obs * fmin)
    fmax = float(freqs[-1])

    full_ntest = nsamp - 2 * nmin
    if full_ntest <= 2:
        raise ValueError("not enough samples remain after trimming 20 Hz at each end")

    trimmed = samples[nmin : nsamp - nmin]
    full_mean, full_var = c_style_mean_variance(trimmed)
    if known_parameters:
        full_standardized = standardize_samples(trimmed, normal_mean, normal_variance)
        full_statistic = anderson_darling_statistic(full_standardized, fitted_normal=False)
        full_pvalue = known_normal_ad_asymptotic_pvalue(full_statistic)
        if pvalue_method == "monte-carlo":
            full_pvalue = float(
                monte_carlo_known_normal_ad_pvalues(
                    np.array([full_statistic], dtype=float),
                    full_ntest,
                    mc_draws,
                    mc_seed,
                )[0]
            )
    else:
        full_standardized = (trimmed - full_mean) / math.sqrt(full_var)
        full_statistic, full_pvalue = fitted_normal_ad_pvalue(full_standardized, 1.0e-5)

    band_ntest = int(2.0 * bandwidth / df)
    bands = int((nsamp - 2 * nmin) / band_ntest)
    if band_ntest <= 2 or bands < 1:
        raise ValueError("test bandwidth is too small or leaves no complete frequency bands")

    uv = math.sqrt(2.0 / band_ntest)
    vlow = 1.0 - 5.0 * uv
    vhigh = 1.0 + 5.0 * uv

    ad_rows = []
    ad_statistics = []
    line_rows = []
    outlier_freqs = []
    outlier_bands = []
    for j in range(bands):
        freq = fmin + (j + 0.5) * bandwidth
        band_start = fmin + j * bandwidth
        band_stop = band_start + bandwidth
        psd_start = (nmin + j * band_ntest) // 2
        psd_stop = psd_start + band_ntest // 2
        log_psd = np.log(psd[psd_start:psd_stop, 1])
        pav = float(np.sum(log_psd) / (band_ntest // 2))
        pvar = float(np.sum(log_psd * log_psd) / (band_ntest // 2))
        # Here the variance of the ASD is used to catch spectral lines.
        # The threshold 0.3 will flag moderate and strong lines. Setting
        # the threshold much less than this will trigger on areas where
        # the spectrum is steep, even if lines are not present (e.g. at
        # low frequencies). Setting the threshold at 0.5 only catches
        # segments with strong lines.
        if pvar - pav * pav > 0.3:
            outlier_freqs.append(freq)
            outlier_bands.append((band_start, band_stop))
            line_rows.append((freq, vlow, 0.00005, 1.0e-50))
            line_rows.append((freq, vhigh, 1.0, 1.0e-40))
            line_rows.append((freq, vlow, 0.00005, 1.0e-50))

        start = nmin + j * band_ntest
        stop = start + band_ntest
        band = samples[start:stop].copy()
        mean, var = c_style_mean_variance(band)
        if known_parameters:
            standardized = standardize_samples(band, normal_mean, normal_variance)
            statistic = anderson_darling_statistic(standardized, fitted_normal=False)
            pvalue = known_normal_ad_asymptotic_pvalue(statistic)
        else:
            standardized = (band - mean) / math.sqrt(var)
            statistic, pvalue = fitted_normal_ad_pvalue(standardized, 1.0e-4)
        ad_statistics.append(statistic)
        if not known_parameters:
            pvalue = max(pvalue, 1.0e-4)
        ad_rows.append((freq, pvalue, var))

    if known_parameters and pvalue_method == "monte-carlo":
        band_pvalues = monte_carlo_known_normal_ad_pvalues(
            np.array(ad_statistics, dtype=float),
            band_ntest,
            mc_draws,
            None if mc_seed is None else mc_seed + 1,
        )
        ad_rows = [
            (freq, float(pvalue), variance)
            for (freq, _, variance), pvalue in zip(ad_rows, band_pvalues)
        ]

    return {
        "adtest": np.array(ad_rows, dtype=float),
        "lines": np.array(line_rows, dtype=float),
        "outlier_freqs": np.array(outlier_freqs, dtype=float),
        "outlier_bands": np.array(outlier_bands, dtype=float).reshape(-1, 2),
        "df": df,
        "t_obs": t_obs,
        "fmin": fmin,
        "fmax": fmax,
        "bandwidth": bandwidth,
        "band_ntest": band_ntest,
        "bands": bands,
        "uv": uv,
        "vlow": vlow,
        "vhigh": vhigh,
        "full_mean": full_mean,
        "full_var": full_var,
        "full_statistic": full_statistic,
        "full_pvalue": full_pvalue,
        "known_parameters": known_parameters,
        "normal_mean": normal_mean,
        "normal_variance": normal_variance,
        "pvalue_method": pvalue_method,
    }


def write_adtest(adtest: np.ndarray) -> None:
    with Path("ADtest.dat").open("w") as out:
        for freq, pvalue, variance in adtest:
            out.write(f"{freq:f} {pvalue:e} {variance:f}\n")


def write_lines(lines: np.ndarray) -> None:
    with Path("lines.dat").open("w") as out:
        for row in lines:
            out.write(f"{row[0]:f} {row[1]:e} {row[2]:e} {row[3]:e}\n")


def build_summary(
    adtest: np.ndarray,
    outlier_freqs: np.ndarray,
    uv: float,
    bands: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Build the screen/file summary for the most important outliers."""
    low_variance_threshold = 1.0 - 3.0 * uv
    high_variance_threshold = 1.0 + 3.0 * uv
    pvalue_threshold = 1.0 / (10.0 * bands)

    low_variance = adtest[adtest[:, 2] < low_variance_threshold]
    high_variance = adtest[adtest[:, 2] > high_variance_threshold]
    if len(outlier_freqs):
        line_mask = np.isin(adtest[:, 0], outlier_freqs)
    else:
        line_mask = np.zeros(len(adtest), dtype=bool)
    low_pvalue = adtest[(adtest[:, 1] < pvalue_threshold) & ~line_mask]

    lines = [
        "AD test summary",
        f"Number of frequency bands: {bands}",
        f"Variance low threshold (mean - 3 sigma): {low_variance_threshold:.9e}",
        f"Variance high threshold (mean + 3 sigma): {high_variance_threshold:.9e}",
        f"P-value threshold (1 / (10 * bands)): {pvalue_threshold:.9e}",
        f"PSD-line bins excluded from low-p summary: {len(outlier_freqs)}",
        "",
        "Low-variance bands (central_frequency variance):",
    ]
    if len(low_variance):
        lines.extend(f"{freq:.9e} {variance:.9e}" for freq, _, variance in low_variance)
    else:
        lines.append("none")

    lines.extend([
        "",
        "High-variance bands (central_frequency variance):",
    ])
    if len(high_variance):
        lines.extend(f"{freq:.9e} {variance:.9e}" for freq, _, variance in high_variance)
    else:
        lines.append("none")

    lines.extend(["", "Low-p bands excluding PSD-line bins (central_frequency p_value):"])
    if len(low_pvalue):
        lines.extend(f"{freq:.9e} {pvalue:.9e}" for freq, pvalue, _ in low_pvalue)
    else:
        lines.append("none")

    return (
        "\n".join(lines),
        low_variance,
        high_variance,
        low_pvalue,
        low_variance_threshold,
        high_variance_threshold,
        pvalue_threshold,
    )


def write_summary(path: Path, summary: str) -> None:
    path.write_text(summary + "\n")


def save_adplot_pdf(
    adtest: np.ndarray,
    outlier_bands: np.ndarray,
    fmin: float,
    bandwidth: float,
    bands: int,
    band_ntest: int,
    uv: float,
    vlow: float,
    vhigh: float,
) -> None:
    xmax = fmin + bands * bandwidth

    fig = plt.figure(figsize=(7.0, 7.0))
    top = fig.add_axes((0.12, 0.53, 0.84, 0.40))
    bottom = fig.add_axes((0.12, 0.10, 0.84, 0.43))

    top.fill_between(adtest[:, 0], 1.0 - 3.0 * uv, 1.0 + 3.0 * uv, color="red", alpha=0.2)
    top.fill_between(adtest[:, 0], 1.0 - 2.0 * uv, 1.0 + 2.0 * uv, color="red", alpha=0.4)
    top.fill_between(adtest[:, 0], 1.0 - uv, 1.0 + uv, color="red", alpha=0.6)
    for start, stop in outlier_bands:
        top.axvspan(start, stop, color="0.75", alpha=0.45, linewidth=0)
    variance = adtest[:, 2]
    in_range = (variance >= vlow) & (variance <= vhigh)
    high_outliers = variance > vhigh
    low_outliers = variance < vlow
    top.scatter(adtest[in_range, 0], variance[in_range], s=3, color="blue")
    top.scatter(
        adtest[high_outliers, 0],
        np.full(np.count_nonzero(high_outliers), 1.0 + 4.0 * uv),
        s=24,
        marker="*",
        color="blue",
    )
    top.scatter(
        adtest[low_outliers, 0],
        np.full(np.count_nonzero(low_outliers), 1.0 - 4.0 * uv),
        s=24,
        marker="*",
        color="blue",
    )
    top.set_xlim(fmin, xmax)
    top.set_ylim(vlow, vhigh)
    top.set_ylabel("Variance")
    top.tick_params(labelbottom=False)

    for start, stop in outlier_bands:
        bottom.axvspan(start, stop, color="0.75", alpha=0.45, linewidth=0)
    bottom.scatter(adtest[:, 0], adtest[:, 1], s=3, color="blue")
    bottom.axhline(1.0 / bands, color="red", linewidth=1.0)
    bottom.set_xlim(fmin, xmax)
    bottom.set_ylim(0.00005, 1.0)
    bottom.set_yscale("log")
    bottom.set_xlabel("f (Hz)")
    bottom.set_ylabel("p")

    fig.savefig("ADplot.pdf")
    plt.close(fig)


def save_spectra_pdf(
    psd: np.ndarray,
    data: np.ndarray,
    outlier_bands: np.ndarray,
    t_obs: float,
    xlim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for start, stop in outlier_bands:
        ax.axvspan(start, stop, color="0.75", alpha=0.45, linewidth=0)
    # Use the same normalization as the AD whitening convention:
    # Re/sqrt(2*PSD), Im/sqrt(2*PSD). This makes periodogram/PSD a chi^2_2
    # variate with mean 2 when the PSD normalization is correct.
    periodogram = 0.5 * (data[:, 1] ** 2 + data[:, 2] ** 2)
    ax.plot(data[:, 0], periodogram, color="0.55", linewidth=0.8)
    ax.plot(psd[:, 0], psd[:, 1], color="blue", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-50, 1.0e-40)
    ax.set_xlim(*xlim)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("Power")
    fig.tight_layout()
    fig.savefig("Spectra.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    epilog = """
Examples:
  /opt/anaconda3/bin/python ADtest.py psd.dat datafreq.dat 8

  /opt/anaconda3/bin/python ADtest.py psd_welch_for_adtest.dat datafreq_for_adtest.dat 8 \\
      --known-parameters --normal-mean 0 --normal-variance 1

  /opt/anaconda3/bin/python ADtest.py psd_welch_for_adtest.dat datafreq_for_adtest.dat 8 \\
      --known-parameters --normal-mean 0 --normal-variance 1 \\
      --pvalue-method monte-carlo --mc-draws 20000 --mc-seed 7

Input format:
  PSD file:      frequency  PSD
  Data file:     frequency  Re(data)  Im(data)

Output files:
  ADtest.dat     central_frequency  p_value  variance
  ADsummary.dat  low/high-variance and low-p-value summary
  ADplot.pdf     main variance/p-value diagnostic plot
  Spectra.pdf    spectrum with flagged line regions

Notes:
  Each frequency bin supplies two samples, Re/sqrt(2*PSD) and Im/sqrt(2*PSD).
  Spectra.pdf plots 0.5*(Re^2 + Im^2), so periodogram/PSD has expected mean 2
  when the same whitening convention is satisfied.
  Use --known-parameters when the whitened samples should be tested directly
  against a fully specified normal distribution. Without it, the code fits
  mean and variance in each band and uses the fitted-normal AD calibration.
"""
    parser = argparse.ArgumentParser(
        description="Variance and Anderson-Darling tests for frequency-domain data.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("psd_file", type=Path, help="two-column PSD file: frequency, PSD")
    parser.add_argument("data_file", type=Path, help="three-column frequency-domain data file: frequency, Re, Im")
    parser.add_argument("bandwidth", nargs="?", type=float, default=8.0, help="test bandwidth in Hz")
    parser.add_argument(
        "--known-parameters",
        action="store_true",
        help="test against supplied normal mean/variance instead of fitting them in each band",
    )
    parser.add_argument("--normal-mean", type=float, default=0.0, help="known null normal mean")
    parser.add_argument("--normal-variance", type=float, default=1.0, help="known null normal variance")
    parser.add_argument(
        "--pvalue-method",
        choices=("asymptotic", "monte-carlo"),
        default="asymptotic",
        help="p-value mapping for --known-parameters; asymptotic is fast, monte-carlo is finite-sample calibrated",
    )
    parser.add_argument(
        "--mc-draws",
        type=int,
        default=20000,
        help="Monte Carlo null draws for --pvalue-method monte-carlo",
    )
    parser.add_argument("--mc-seed", type=int, default=None, help="random seed for Monte Carlo calibration")
    parser.add_argument("--summary-file", type=Path, default=Path("ADsummary.dat"), help="summary output file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    psd, data = load_inputs(args.psd_file, args.data_file)
    results = run_tests(
        psd,
        data,
        args.bandwidth,
        known_parameters=args.known_parameters,
        normal_mean=args.normal_mean,
        normal_variance=args.normal_variance,
        pvalue_method=args.pvalue_method,
        mc_draws=args.mc_draws,
        mc_seed=args.mc_seed,
    )
    adtest = results["adtest"]
    lines = results["lines"]
    outlier_freqs = results["outlier_freqs"]
    outlier_bands = results["outlier_bands"]
    (
        summary,
        low_variance,
        high_variance,
        low_pvalue,
        low_variance_threshold,
        high_variance_threshold,
        pvalue_threshold,
    ) = build_summary(
        adtest,
        outlier_freqs,
        float(results["uv"]),
        int(results["bands"]),
    )

    write_adtest(adtest)
    write_lines(lines)
    write_summary(args.summary_file, summary)
    save_adplot_pdf(
        adtest,
        outlier_bands,
        float(results["fmin"]),
        float(results["bandwidth"]),
        int(results["bands"]),
        int(results["band_ntest"]),
        float(results["uv"]),
        float(results["vlow"]),
        float(results["vhigh"]),
    )
    save_spectra_pdf(
        psd,
        data,
        outlier_bands,
        float(results["t_obs"]),
        (float(results["fmin"]), float(results["fmax"])),
    )

    mode = "known-parameter normal" if bool(results["known_parameters"]) else "fitted normal"
    print(f"AD test mode {mode}")
    if bool(results["known_parameters"]):
        print(
            f"Known Mean {float(results['normal_mean']):f}  "
            f"Known Variance {float(results['normal_variance']):f} "
            f"P-value method {results['pvalue_method']}"
        )
    print(
        f"Overall Sample Mean {float(results['full_mean']):f}  "
        f"Sample Variance {float(results['full_var']):f} "
        f"AD Statistic {float(results['full_statistic']):f} "
        f"P-value {float(results['full_pvalue']):f}"
    )
    print(f"Samples per band {int(results['band_ntest'])} Number of Bands {int(results['bands'])}")
    print(
        "frequency range covered by test "
        f"{float(results['fmin']):f} "
        f"{float(results['fmin']) + int(results['bands']) * float(results['bandwidth']):f}"
    )
    print(f"frequency range covered by spectrum plot {float(results['fmin']):f} {float(results['fmax']):f}")
    print(
        f"Low-variance bands below 3 sigma ({low_variance_threshold:.6f}): "
        f"{len(low_variance)}"
    )
    for freq, _, variance in low_variance:
        print(f"  f={freq:.6f} variance={variance:.6f}")
    print(
        f"High-variance bands above 3 sigma ({high_variance_threshold:.6f}): "
        f"{len(high_variance)}"
    )
    for freq, _, variance in high_variance:
        print(f"  f={freq:.6f} variance={variance:.6f}")
    print(
        f"Low-p bands below 1/(10*bands) ({pvalue_threshold:.6e}), "
        f"excluding PSD-line bins: {len(low_pvalue)}"
    )
    for freq, pvalue, _ in low_pvalue:
        print(f"  f={freq:.6f} p={pvalue:.6e}")
    print(f"Wrote summary {args.summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
