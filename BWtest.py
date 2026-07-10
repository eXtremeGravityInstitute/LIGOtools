#!/usr/bin/env python3
"""
Self-contained Python port of the cleaned BWtest/BayesLine workflow.

This file mirrors the functional algorithm in BWtest.c and BayesLine.c:

1. Read a two-column frame file: time, strain.
2. Match the C preprocessing: optional decimation filter, Tukey window, FFT.
3. Run BayesLine burn-in:
   - wavelet clean non-Gaussian excess power,
   - build the initial smooth PSD with Akima splines,
   - find spectral lines with a four-pass windowed-Lorentzian startup,
   - optionally write BL_start.dat with frequency, cleaned periodogram, smooth PSD, line PSD.
4. Run the Lorentzian+spline RJMCMC refinement, optionally adding broad
   Gaussian PSD bumps when --gaussian-bumps is requested.
5. Write output files using the same normalization convention expected by
   ADtest.py.

Usage examples:


    File provided
    python BWtest.py frame.dat
    python BWtest.py --file frame.dat
    python BWtest.py frame.dat --psd-samples 500
    
    Running on a LVK machine
    python BWtest.py 1126259462.4 8 --channel H1:GDS-CALIB_STRAIN --nyquist 1024

    LVK frame read with a symmetric context prior
    python BWtest.py 1126259462.4 8 --channel H1:GDS-CALIB_STRAIN \
        --nyquist 1024 --expand 4

        Here the requested analysis duration is 8 seconds. The target analysis
        segment is still the usual [1126259458.4, 1126259466.4], ending 4 seconds
        after the trigger.  With --expand 4, BWtest takes two 8-second chunks
        before the target and two 8-second chunks after it, wavelet-cleans each
        chunk independently, averages their powers on the 8-second frequency
        grid, and runs a context RJMCMC with the proper four-segment Welch
        likelihood. Those same-grid context PSD samples become an inflated
        component prior plus a line-birth proposal for a second RJMCMC run on
        the actual 8-second target segment.
    
    Running anywhere and getting the data from GWOSC
    python BWtest.py 1126259462.4 8 --source open --ifo L1 --nyquist 1024

    Re-processing a BWtest time-domain diagnostic that is already Tukey-tapered
    python BWtest.py line_subtracted_time.dat --no-tukey

    Frame/GWOSC inputs are fetched with padding, resampled to 2*Nyquist, and
    cropped to the requested segment before the BayesLine preprocessing starts.
    By default, LVK cluster frame inputs use LALSuite's Kaiser-windowed sinc
    resampler via lal.ResampleREAL8TimeSeries, while --source open/GWOSC inputs
    use GWPy's resampling path.  Use --resampler lal or --resampler gwpy to
    override this source-dependent default.  The old Butterworth decimation
    path is reserved for existing two-column files whose sample rate is above
    the requested Nyquist.
    
    Use python BWtest.py --help to see the many additional options. Some of these are
    --expand N Uses N target-duration context chunks split evenly before and
        after the target segment for a two-stage PSD run. N must be a power of
        two. The default is 1, which keeps the original one-segment path.

        When N > 1, BWtest uses two segments:
        * the requested analysis segment still ends 4 seconds after the trigger
          and is used for frequency_data.dat, whitening, glitch/feature output,
          and line-subtracted diagnostics;
        * the context has N independent Tobs chunks split evenly before and
          after the target. Initialization uses the nearest preceding chunk.
          The context RJMCMC then uses the average cleaned power from all chunks
          with likelihood multiplicity N, equivalent to the correct Welch
          likelihood for N independent periodograms on the same grid.
          Context PSD samples are summarized as component priors: the smooth
          spline component uses a Gaussian prior in log PSD, while line and
          Gaussian-bump components use Gaussian priors in
          log1p(component/smooth). The standard deviations are inflated by
          --context-prior-inflate, default sqrt(N), and floored by
          --context-prior-sigma-floor, default 0.5.

        The target run skips startup line/spline discovery and starts from the
        final same-grid context state. It uses the context components
        as priors and a context-derived line proposal, but the likelihood is
        computed only from the target analysis segment.  Expanded runs write
        context_psd_prior.dat, context_component_prior.dat, fprop_context.dat,
        and line_proposal.dat so the context prior and proposal can be
        inspected. Since context and target use the same duration, the final
        context state is copied directly onto the target line/spline grids.

        For user-supplied files, providing a duration, e.g.
        ``python BWtest.py frame_40.dat 8 --expand 4``, means BWtest assumes the
        target 8-second analysis segment is centered in the file, with two
        8-second context chunks before and two after. With --expand 1 the
        historical file behavior is unchanged: the file center is treated as the
        trigger and the trigger+4 analysis segment is cropped.
    --linetrim Turns on the prior that exponentially supresses the number of lines.
        The C BayesLine sampler currently keeps this prior on, so use this flag
        when comparing Python BWtest output directly against BWtest.c.
    --linecut X Sets the cut_lorentz delta-log-likelihood threshold used to
        retain refined Lorentzian lines. The earlier lineget/Lpeak admission
        passes use staged periodogram thresholds 100, 30, linemul, linemul;
        the late Lpeak likelihood admission threshold is LINETHRESH.
    --prior-recovery Runs the RJMCMC with a constant likelihood and writes
        prior_recovery_counts.dat with iteration, number of lines, and number
        of spline knots for reversible-jump balance checks. With
        --gaussian-bumps, the Gaussian bump count is included too.
        It also writes prior_recovery_histograms.pdf plus histogram data files
        using the second half of the trace by default; with --gaussian-bumps,
        this includes prior_recovery_gaussian_histogram.dat.
    --rjmcmc-steps N Sets the number of RJMCMC iterations, default 100000.
    --seed N Sets the random seed used by both the startup fit and RJMCMC,
        default 1234.
    --gaussian-bumps Turns on an additional reversible-jump model component
        for broad Gaussian PSD excesses. The model starts with zero bumps; the
        sampler can add/delete/update bumps only in the RJMCMC stage.

    After the startup line fit and spline trimming, the RJMCMC dimension caps are
    reset adaptively to twice the initialized number of lines and spline knots.
    The larger duration-based line allocation is only used during startup, where
    the candidate list can be much larger than the final initialized model.

    WARNING for --no-tukey / --input-already-tukeyed:
        This option only skips multiplying the input data by a new Tukey window.
        The line model still uses BWtest's standard Tukey-windowed Lorentzian
        lookup table, and the output PSD/frequency-data scaling still applies
        the usual one-window Tukey power correction. It is intended for files
        that already contain the same BWtest Tukey roll-off, such as
        line_subtracted_time.dat. If the input is untapered or was made with a
        different window, the line model and normalization may be inconsistent.


Important output files:

    BWpsd.dat
        Median duration-independent one-sided PSD from RJMCMC samples, scaled
        consistently with frequency_data.dat for ADtest.py.
        By default the median uses 200 PSD states sampled evenly from the second
        half of the RJMCMC. Change this with --psd-samples N.

        For trigger-time data fetched from the LVK grid or GWOSC, this main
        PSD file is tagged with detector and rounded trigger GPS, e.g.
        BWpsd_H1_1126259462.dat. Existing input-file runs keep BWpsd.dat.

    BWfairdrawpsd.dat
        Final fair-draw one-sided PSD from the last RJMCMC state.

    frequency_data.dat
        Final cleaned complex frequency-domain data, scaled for ADtest.py.
        For fetched trigger-time data this is tagged in the same way as BWpsd,
        e.g. frequency_data_H1_1126259462.dat.

    BL_start.dat
        Optional startup diagnostic: frequency, periodogram, smooth PSD, line PSD.
        The PSD columns are scaled to the same physical one-sided convention as
        BWpsd.dat.

    start_psd_bw.dat and spline.dat
        Startup total PSD and Akima-spline knot diagnostics. These are also
        scaled to the BWpsd.dat one-sided PSD convention.

    BWpsd_components.dat
        Median total, smooth, and line PSD components. With --gaussian-bumps,
        a fifth column gives the broad-Gaussian component.

    context_psd_prior.dat and context_component_prior.dat
        Written for --expand > 1. context_psd_prior.dat is a plotting diagnostic
        for the same-grid context PSD prior: frequency, total PSD center, and
        total log-PSD sigma. context_component_prior.dat records the active
        component priors used by the target run: smooth PSD center and log sigma,
        plus log1p(component/smooth) centers and sigmas for the line component
        and, when enabled, Gaussian bumps.

    line_parameters.dat
        Optional with --write-line-params: final fair-draw Lorentzian line
        parameters, including the fitted windowed peak height and the inferred
        raw Lorentzian amplitude. For fetched data this file is tagged in the
        same way as BWpsd.dat.

    psd_range.dat
        Optional with --psd-range: frequency, MCMC-sample mean PSD, and
        MCMC-sample standard deviation at each frequency. The scaling matches
        BWpsd.dat, while BWpsd.dat itself stays two columns for compatibility.

    BW_likelihood_summand.dat
        Frequency and Whittle log-likelihood summand for the arithmetic mean
        PSD model. Summing the second column gives the printed average-PSD
        model log likelihood.

    BWfairdrawpsd_components.dat
        Final fair-draw total, smooth, and line PSD components. With
        --gaussian-bumps, a fifth column gives the broad-Gaussian component.

    whitelsf.dat and whitelsf_noglitch.dat
        Optional with --writewhite: line-subtracted, broadband-background-PSD
        whitened Fourier data before and after wavelet glitch cleaning. The
        broadband background is the Akima spline component plus any optional
        Gaussian bumps, but excludes the coherent Lorentzian lines.

    freq_nolines.dat, line_subtracted_time.dat, and line_subtracted_noglitch_time.dat
        Optional with --write-line-subtracted-time: unwhitened line-subtracted
        data. freq_nolines.dat and line_subtracted_time.dat are built from the
        original Tukey-windowed FFT with the final fair-draw Lorentzian line
        contribution subtracted; line_subtracted_noglitch_time.dat starts from
        the wavelet-cleaned FFT used for PSD estimation. The time-domain files
        retain the Tukey roll-off used before the FFT.

        WARNING: the Tukey window roll-off is not undone in these files. The
        endpoints are still tapered. This is intentional because dividing by
        the small window values in the roll-off would amplify noise and edge
        transients. Treat these files as tapered, analysis-segment time-domain
        diagnostics rather than a reconstruction of the untapered frame data.

    glitch_time.dat and whitened_glitch_time.dat
        Optional with --write-glitch: time-domain wavelet excess subtracted
        during BayesLine's initial cleaning stage, and the same excess whitened
        using the final median BWpsd. The glitch is defined as the tapered raw
        analysis data minus the wavelet-cleaned data used for PSD estimation,
        so it inherits the same Tukey/window status as the analyzed data.

    whitened_feature_time.dat
        Optional with --feature: the whitened part of the glitch that survives
        Denoise.py-style time-frequency clustering and the cluster SNR cut.
        The feature SNR is computed from Denoise-normalized Fourier
        coefficients, not by FFTing whitened_glitch_time.dat again.

    sinefit.dat
        Optional with --sinefit: for each final Lorentzian line, compare the
        reconstructed windowed-Lorentzian power profile to a Tukey-windowed
        pure sinusoid power profile. The sinusoid frequency is scanned over
        f_line +/- 0.5*nu, where nu is the line width in Hz. Inner products
        are weighted by the final broadband non-line PSD.
        The columns are line frequency, line width nu in Hz, best-fit sinusoid
        frequency, line SNR sqrt((a|a)), predicted residual SNR, and mismatch
        1-match.

    freq_nosine.dat
        Optional with --sinefit: Tukey-windowed original frequency-domain data
        with a best-fit pure sinusoid subtracted for each final line. The
        sinusoid frequency, amplitude, and phase are fitted against the local
        Fourier data with the final broadband non-line PSD as the noise
        weighting. Columns
        match freq_nolines.dat: frequency, real part, imaginary part, in
        BayesWave's internal scaled Fourier convention.

    freq_nolines.dat
        Optional with --write-line-subtracted-time: Tukey-windowed original
        frequency-domain data with a final-state
        fair-draw Lorentzian line contribution subtracted. This uses the
        original analyzed data, not the wavelet-cleaned data used for PSD
        estimation, and is written in BayesWave's internal scaled Fourier
        convention.

To run the functional whitening check used during development:

    /opt/anaconda3/bin/python ADtest.py BWpsd.dat frequency_data.dat 8 \
        --known-parameters --normal-mean 0 --normal-variance 1

The original C code uses GSL, OpenMP, and a large RJMCMC implementation. This
Python version uses NumPy plus optional numba acceleration for hot loops. SciPy
is used, when available, to speed up windowed-Lorentzian lookup generation with
FFT convolution. When numba or SciPy is unavailable the same code keeps running
through slower compatibility paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.signal import fftconvolve
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on systems without scipy
    fftconvolve = None
    SCIPY_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on systems without numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorate(func):
            return func

        return decorate


# ---------------------------------------------------------------------------
# Constants and state containers
# ---------------------------------------------------------------------------
#
# The names and default values here intentionally track BayesLine.h/BayesLine.c.
# Keeping the same concepts visible makes it easier to compare the Python port
# against the C reference when diagnosing stochastic differences.
TPI = 2.0 * math.pi
LN2 = math.log(2.0)
LINEMUL = 9.0
LINECUT = 6.0
LINETHRESH = 8.0
# medspecspline uses LINEMUL for the PSD construction.  The line proposal can
# use a lower threshold to help the RJMCMC visit weaker excesses without
# changing the posterior target or the startup PSD model.
LINE_PROPOSAL_THRESH = 6.0
LINE_PROPOSAL_BOOST = 10.0
T_RISE = 1.0
FSTEP = 4.0
DFMIN = 0.5
Q_S = 8.0
STHRESH = 9.0
WARM = 6.0
FEATURE_WARM = 5.0
FEATURE_SNR_THRESH = 4.0
FEATURE_QSCAN_SUBSCALE = 40
SINEFIT_SCAN_POINTS = 41
LINE_COUNT_PRIOR_ZETA = 0.5
LINE_NU_MIN = 1.0e-5
LINE_NU_MAX = 1.0e-1
GAUSSIAN_BUMP_ARRAY_SCALE = 10.0
GAUSSIAN_BUMP_SIGMA_MIN = 0.2
GAUSSIAN_BUMP_SIGMA_MAX = 5.0
DEFAULT_ANALYSIS_FMIN = 20.0
DEFAULT_ANALYSIS_FMAX = 1024.0
TRIGGER_OFFSET_FROM_END = 4.0
BL_START_FILENAME = "BL_start.dat"
LINE_ARRAY_SCALE = 500.0
CONTEXT_PRIOR_LOG_SIGMA_FLOOR = 0.5


@dataclass
class LorentzianParams:
    """Line-model state, including the windowed-Lorentzian lookup table."""

    size: int
    n: int = 0
    f: np.ndarray = field(init=False)
    q: np.ndarray = field(init=False)
    a: np.ndarray = field(init=False)
    nu: np.ndarray = field(init=False)
    nf: int = 0
    nnu: int = 0
    wdth: int = 0
    imin: int = 0
    numin: float = LINE_NU_MIN
    numax: float = LINE_NU_MAX
    lnumin: float = math.log(LINE_NU_MIN)
    lnumax: float = math.log(LINE_NU_MAX)
    ltemplate: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.f = np.zeros(self.size, dtype=np.float64)
        self.q = np.zeros(self.size, dtype=np.float64)
        self.a = np.zeros(self.size, dtype=np.float64)
        self.nu = np.zeros(self.size, dtype=np.float64)


@dataclass
class GaussianBumpParams:
    """Broad Gaussian PSD bumps added during the RJMCMC."""

    size: int
    n: int = 0
    f: np.ndarray = field(init=False)
    a: np.ndarray = field(init=False)
    sigma: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.f = np.zeros(self.size, dtype=np.float64)
        self.a = np.zeros(self.size, dtype=np.float64)
        self.sigma = np.zeros(self.size, dtype=np.float64)


@dataclass
class InputSpec:
    """Resolved input file plus whether it came from direct frame fetching."""

    filename: str
    fetched_from_frame: bool = False
    output_tag: Optional[str] = None
    trigger_time: Optional[float] = None
    analysis_duration: Optional[float] = None


@dataclass
class DataParams:
    """Frequency grid and active analysis band used by BayesLine."""

    n: int
    t_obs: float
    fmin: float
    fmax: float
    df: float
    imin: int
    imax: int
    ncut: int
    flow: float
    fhigh: float
    fgrid: float = FSTEP


@dataclass
class SplineParams:
    """Akima spline knot locations and log-PSD values."""

    points: np.ndarray
    data: np.ndarray

    @property
    def n(self) -> int:
        return int(self.points.size)


@dataclass
class BayesLinePriors:
    """Priors and soft bounds used by the RJMCMC updates."""

    SAmin: float = 0.0
    SAmax: float = 0.0
    LQmin: float = 0.0
    LQmax: float = 0.0
    LAmin: float = 1.0e-60
    LAmax: float = 1.0
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    use_context_psd_prior: bool = False
    use_context_component_prior: bool = False
    smooth_mean: Optional[np.ndarray] = None
    smooth_sigma: Optional[np.ndarray] = None
    component_reference: Optional[np.ndarray] = None
    line_log1p_mean: Optional[np.ndarray] = None
    line_log1p_sigma: Optional[np.ndarray] = None
    gaussian_log1p_mean: Optional[np.ndarray] = None
    gaussian_log1p_sigma: Optional[np.ndarray] = None


@dataclass
class BayesLineParams:
    """Top-level BayesLine state shared by burn-in and RJMCMC."""

    maxBLLines: int = 0
    maxBLBumps: int = 0
    maxSplineKnots: int = 0
    data: Optional[DataParams] = None
    spline: Optional[SplineParams] = None
    spline_x: Optional[SplineParams] = None
    lines_x: Optional[LorentzianParams] = None
    bumps_x: Optional[GaussianBumpParams] = None
    priors: BayesLinePriors = field(default_factory=BayesLinePriors)
    Snf: Optional[np.ndarray] = None
    Sna: Optional[np.ndarray] = None
    fa: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None
    power: Optional[np.ndarray] = None
    spow: Optional[np.ndarray] = None
    sfreq: Optional[np.ndarray] = None
    Sbase: Optional[np.ndarray] = None
    Sline: Optional[np.ndarray] = None
    Sgauss: Optional[np.ndarray] = None
    gaussianSigmaMin: float = GAUSSIAN_BUMP_SIGMA_MIN
    gaussianSigmaMax: float = GAUSSIAN_BUMP_SIGMA_MAX
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1234))
    constantLogLFlag: int = 0
    flatPriorFlag: int = 0
    power_multiplicity: float = 1.0


def line_array_size_for_duration(t_obs: float) -> int:
    """Capacity for the Lorentzian line arrays, scaling with segment duration."""

    if not math.isfinite(t_obs) or t_obs <= 0.0:
        raise ValueError("Tobs must be positive and finite")
    return max(1, int(math.ceil(LINE_ARRAY_SCALE * math.sqrt(t_obs))))


def gaussian_bump_array_size_for_duration(t_obs: float) -> int:
    """Capacity for broad Gaussian PSD bumps."""

    if not math.isfinite(t_obs) or t_obs <= 0.0:
        raise ValueError("Tobs must be positive and finite")
    return max(4, int(math.ceil(GAUSSIAN_BUMP_ARRAY_SCALE * math.sqrt(t_obs))))


@njit(cache=True)
def tukey_inplace(data: np.ndarray, alpha: float) -> None:
    """Apply the same symmetric Tukey taper used by the C driver."""

    n = data.size
    imin = int(alpha * (n - 1.0) / 2.0)
    imax = int((n - 1.0) * (1.0 - alpha / 2.0))
    nwin = n - imax
    for i in range(n):
        filt = 1.0
        if imin > 0 and i < imin:
            filt = 0.5 * (1.0 + math.cos(math.pi * (i / imin - 1.0)))
        if nwin > 0 and i > imax:
            filt = 0.5 * (1.0 + math.cos(math.pi * ((i - imax) / nwin)))
        data[i] *= filt


def tukey_power_correction(n: int, alpha: float) -> float:
    """Return the power correction N/sum(w^2) for the C-style Tukey taper."""

    imin = int(alpha * (n - 1.0) / 2.0)
    imax = int((n - 1.0) * (1.0 - alpha / 2.0))
    nwin = n - imax
    sum_w2 = 0.0
    for i in range(n):
        filt = 1.0
        if imin > 0 and i < imin:
            filt = 0.5 * (1.0 + math.cos(math.pi * (i / imin - 1.0)))
        if nwin > 0 and i > imax:
            filt = 0.5 * (1.0 + math.cos(math.pi * ((i - imax) / nwin)))
        sum_w2 += filt * filt
    if sum_w2 <= 0.0:
        raise ValueError("Tukey window has zero power")
    return float(n) / sum_w2


@njit(cache=True)
def bwbpf_numba(inp: np.ndarray, fwrv: int, order: int, sample_rate: float, f1: float, f2: float) -> np.ndarray:
    """Butterworth bandpass filter ported from BWtest.c's bwbpf routine."""

    if order % 4 != 0:
        raise ValueError("Order must be 4,8,12,16,...")

    m = inp.size
    n = order // 4
    out = np.empty_like(inp)
    a = math.cos(math.pi * (f1 + f2) / sample_rate) / math.cos(math.pi * (f1 - f2) / sample_rate)
    a2 = a * a
    b = math.tan(math.pi * (f1 - f2) / sample_rate)
    b2 = b * b

    A = np.empty(n)
    d1 = np.empty(n)
    d2 = np.empty(n)
    d3 = np.empty(n)
    d4 = np.empty(n)
    w0 = np.zeros(n)
    w1 = np.zeros(n)
    w2 = np.zeros(n)
    w3 = np.zeros(n)
    w4 = np.zeros(n)

    for i in range(n):
        r = math.sin(math.pi * (2.0 * i + 1.0) / (4.0 * n))
        s = b2 + 2.0 * b * r + 1.0
        A[i] = b2 / s
        d1[i] = 4.0 * a * (1.0 + b * r) / s
        d2[i] = 2.0 * (b2 - 2.0 * a2 - 1.0) / s
        d3[i] = 4.0 * a * (1.0 - b * r) / s
        d4[i] = -(b2 - 2.0 * b * r + 1.0) / s

    for j in range(m):
        if fwrv == 1:
            x = inp[j]
        else:
            x = inp[m - j - 1]
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

    return out


# ---------------------------------------------------------------------------
# Low-level numerical kernels
# ---------------------------------------------------------------------------
#
# These routines are deliberately small and numba-friendly. They cover the
# repeated operations that dominate runtime: likelihood sums, simple smoothers,
# and interpolation of the precomputed windowed-Lorentzian lookup table.
@njit(cache=True)
def moving_average_numba(x: np.ndarray, half_width: int) -> np.ndarray:
    """Uniform moving average used only for pre-burn-in placeholder smoothing."""

    n = x.size
    y = np.empty(n)
    for i in range(n):
        lo = max(0, i - half_width)
        hi = min(n, i + half_width + 1)
        acc = 0.0
        for j in range(lo, hi):
            acc += x[j]
        y[i] = acc / (hi - lo)
    return y


@njit(cache=True)
def loglike_numba(power: np.ndarray, sn: np.ndarray, multiplicity: float) -> float:
    """Whittle likelihood for averaged periodogram samples given a PSD model."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    for i in range(power.size):
        s = sn[i]
        if s < tiny:
            s = tiny
        val -= multiplicity * (power[i] / s + math.log(s))
    return val


@njit(cache=True)
def logprior_bounds_numba(lower: np.ndarray, upper: np.ndarray, sn: np.ndarray) -> float:
    """Soft log-prior penalty for PSD samples outside configured bounds."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    for i in range(sn.size):
        s = max(sn[i], tiny)
        if s > upper[i]:
            ds = math.log(s) - math.log(max(upper[i], tiny))
            val -= 0.5 * ds * ds
        if s < lower[i]:
            ds = math.log(s) - math.log(max(lower[i], tiny))
            val -= 0.5 * ds * ds
    return val


@njit(cache=True)
def logprior_logpsd_gaussian_numba(mean: np.ndarray, sigma: np.ndarray, sn: np.ndarray) -> float:
    """Gaussian log-PSD prior used by the two-stage --expand context run."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    for i in range(sn.size):
        s = max(sn[i], tiny)
        mu = math.log(max(mean[i], tiny))
        sig = max(sigma[i], 1.0e-12)
        z = (math.log(s) - mu) / sig
        val -= 0.5 * z * z
    return val


@njit(cache=True)
def logprior_smooth_component_numba(mean: np.ndarray, sigma: np.ndarray,
                                    sbase: np.ndarray) -> float:
    """Gaussian log-prior for the smooth Akima component."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    for i in range(sbase.size):
        s = max(sbase[i], tiny)
        mu = math.log(max(mean[i], tiny))
        sig = max(sigma[i], 1.0e-12)
        z = (math.log(s) - mu) / sig
        val -= 0.5 * z * z
    return val


@njit(cache=True)
def logprior_ratio_component_numba(reference: np.ndarray, mean: np.ndarray,
                                   sigma: np.ndarray, component: np.ndarray) -> float:
    """Gaussian prior for non-negative PSD components in log1p(component/reference)."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    for i in range(component.size):
        ref = max(reference[i], tiny)
        x = math.log1p(max(component[i], 0.0) / ref)
        sig = max(sigma[i], 1.0e-12)
        z = (x - mean[i]) / sig
        val -= 0.5 * z * z
    return val


@njit(cache=True)
def positive_range_numba(sn: np.ndarray, ilow: int, ihigh: int) -> bool:
    """Check positivity only over bins touched by a local proposal."""

    lo = max(0, ilow)
    hi = min(sn.size, ihigh)
    for i in range(lo, hi):
        if sn[i] <= 0.0:
            return False
    return True


@njit(cache=True)
def delta_logprior_bounds_range_numba(lower: np.ndarray, upper: np.ndarray,
                                      sn_new: np.ndarray, sn_old: np.ndarray,
                                      ilow: int, ihigh: int) -> float:
    """Soft-bound prior difference over a local proposal window."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    lo = max(0, ilow)
    hi = min(sn_new.size, ihigh)
    for i in range(lo, hi):
        snew = max(sn_new[i], tiny)
        sold = max(sn_old[i], tiny)

        old_penalty = 0.0
        if sold > upper[i]:
            ds = math.log(sold) - math.log(max(upper[i], tiny))
            old_penalty -= 0.5 * ds * ds
        if sold < lower[i]:
            ds = math.log(sold) - math.log(max(lower[i], tiny))
            old_penalty -= 0.5 * ds * ds

        new_penalty = 0.0
        if snew > upper[i]:
            ds = math.log(snew) - math.log(max(upper[i], tiny))
            new_penalty -= 0.5 * ds * ds
        if snew < lower[i]:
            ds = math.log(snew) - math.log(max(lower[i], tiny))
            new_penalty -= 0.5 * ds * ds

        val += new_penalty - old_penalty
    return val


@njit(cache=True)
def delta_logprior_logpsd_gaussian_range_numba(mean: np.ndarray, sigma: np.ndarray,
                                               sn_new: np.ndarray, sn_old: np.ndarray,
                                               ilow: int, ihigh: int) -> float:
    """Local difference of the context Gaussian log-PSD prior."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    lo = max(0, ilow)
    hi = min(sn_new.size, ihigh)
    for i in range(lo, hi):
        mu = math.log(max(mean[i], tiny))
        sig = max(sigma[i], 1.0e-12)
        sold = max(sn_old[i], tiny)
        snew = max(sn_new[i], tiny)
        zold = (math.log(sold) - mu) / sig
        znew = (math.log(snew) - mu) / sig
        val += -0.5 * znew * znew + 0.5 * zold * zold
    return val


@njit(cache=True)
def delta_logprior_smooth_component_range_numba(mean: np.ndarray, sigma: np.ndarray,
                                                sbase_new: np.ndarray, sbase_old: np.ndarray,
                                                ilow: int, ihigh: int) -> float:
    """Local difference of the smooth-component context prior."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    lo = max(0, ilow)
    hi = min(sbase_new.size, ihigh)
    for i in range(lo, hi):
        mu = math.log(max(mean[i], tiny))
        sig = max(sigma[i], 1.0e-12)
        sold = max(sbase_old[i], tiny)
        snew = max(sbase_new[i], tiny)
        zold = (math.log(sold) - mu) / sig
        znew = (math.log(snew) - mu) / sig
        val += -0.5 * znew * znew + 0.5 * zold * zold
    return val


@njit(cache=True)
def delta_logprior_ratio_component_range_numba(reference: np.ndarray, mean: np.ndarray,
                                               sigma: np.ndarray, comp_new: np.ndarray,
                                               comp_old: np.ndarray, ilow: int,
                                               ihigh: int) -> float:
    """Local difference of a log1p(component/reference) context prior."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    lo = max(0, ilow)
    hi = min(comp_new.size, ihigh)
    for i in range(lo, hi):
        ref = max(reference[i], tiny)
        sig = max(sigma[i], 1.0e-12)
        xold = math.log1p(max(comp_old[i], 0.0) / ref)
        xnew = math.log1p(max(comp_new[i], 0.0) / ref)
        zold = (xold - mean[i]) / sig
        znew = (xnew - mean[i]) / sig
        val += -0.5 * znew * znew + 0.5 * zold * zold
    return val


@njit(cache=True)
def lookup_line_delta(sn_old: np.ndarray, data_imin: int, t_obs: float,
                      nf: int, nnu: int, wdth: int, lnumin: float, lnumax: float,
                      ltemplate: np.ndarray, f0: float, amp: float, nu: float,
                      sign: float) -> np.ndarray:
    """Add or subtract one interpolated windowed Lorentzian from a PSD array.

    This is the Python equivalent of BayesLine.c's llook plus the surrounding
    add/subtract loops. The lookup table encodes the Tukey-windowed line shape,
    indexed by sub-bin frequency offset and log line width.
    """

    out = sn_old.copy()
    k = int(round(f0 * t_obs))
    xoff = f0 - k / t_obs
    if xoff * t_obs <= -0.5:
        k -= 1
        xoff = f0 - k / t_obs
    if xoff * t_obs >= 0.5:
        k += 1
        xoff = f0 - k / t_obs

    dfx = 1.0 / (nf * t_obs)
    dnux = (lnumax - lnumin) / nnu
    ii = (nf + int(math.floor(2.0 * xoff * t_obs * nf))) // 2
    if ii == nf:
        ii -= 1
    fgrid = (2.0 * ii - nf) / nf * 0.5 / t_obs
    y = (xoff - fgrid) / dfx

    lnu = math.log(nu)
    if lnu < lnumin:
        lnu = lnumin
    if lnu > lnumax:
        lnu = lnumax
    xn = (math.log(nu) - lnumin) / dnux
    jj = int(xn)
    z = xn - jj
    if jj == nnu:
        jj -= 1
        z = xn - jj

    idelt = wdth // 2
    imid = k - data_imin
    istart = imid - idelt
    for it in range(wdth):
        idx = istart + it
        if 0 <= idx < out.size:
            logv = (1.0 - z) * ((1.0 - y) * ltemplate[ii, jj, it] + y * ltemplate[ii + 1, jj, it])
            logv += z * ((1.0 - y) * ltemplate[ii, jj + 1, it] + y * ltemplate[ii + 1, jj + 1, it])
            out[idx] += sign * amp * math.exp(logv)
    return out


@njit(cache=True)
def lorentzian_lookup_params(t_obs: float, nf: int, nnu: int, lnumin: float,
                             lnumax: float, f0: float, nu: float) -> Tuple[int, int, int, float, float]:
    """Return the llook grid indices and interpolation weights."""

    k = int(round(f0 * t_obs))
    xoff = f0 - k / t_obs
    if xoff * t_obs <= -0.5:
        k -= 1
        xoff = f0 - k / t_obs
    if xoff * t_obs >= 0.5:
        k += 1
        xoff = f0 - k / t_obs

    dfx = 1.0 / (nf * t_obs)
    dnux = (lnumax - lnumin) / nnu
    ii = (nf + int(math.floor(2.0 * xoff * t_obs * nf))) // 2
    if ii < 0:
        ii = 0
    if ii >= nf:
        ii = nf - 1

    fgrid = (2.0 * ii - nf) / nf * 0.5 / t_obs
    y = (xoff - fgrid) / dfx

    lognu = math.log(nu)
    if lognu < lnumin:
        lognu = lnumin
    if lognu > lnumax:
        lognu = lnumax
    xn = (lognu - lnumin) / dnux
    jj = int(xn)
    z = xn - jj
    if jj < 0:
        jj = 0
        z = 0.0
    if jj >= nnu:
        jj = nnu - 1
        z = 1.0
    return k, ii, jj, y, z


@njit(cache=True)
def lorentzian_lookup_value(ltemplate: np.ndarray, ii: int, jj: int,
                            y: float, z: float, amp: float, i: int) -> float:
    """Evaluate one bin of the interpolated windowed-Lorentzian template."""

    logv = (1.0 - z) * ((1.0 - y) * ltemplate[ii, jj, i] + y * ltemplate[ii + 1, jj, i])
    logv += z * ((1.0 - y) * ltemplate[ii, jj + 1, i] + y * ltemplate[ii + 1, jj + 1, i])
    return amp * math.exp(logv)


@njit(cache=True)
def local_lorentzian_template(t_obs: float, nf: int, nnu: int, wdth: int,
                              lnumin: float, lnumax: float, ltemplate: np.ndarray,
                              f0: float, amp: float, nu: float) -> Tuple[int, np.ndarray]:
    """C llook equivalent: return center bin and only the local wdth template."""

    lt = np.empty(wdth, dtype=np.float64)
    k, ii, jj, y, z = lorentzian_lookup_params(t_obs, nf, nnu, lnumin, lnumax, f0, nu)

    for i in range(wdth):
        lt[i] = lorentzian_lookup_value(ltemplate, ii, jj, y, z, amp, i)
    return k, lt


@njit(cache=True)
def add_lookup_line_inplace(line_model: np.ndarray, t_obs: float, nf: int, nnu: int,
                            wdth: int, lnumin: float, lnumax: float,
                            ltemplate: np.ndarray, f0: float, amp: float,
                            nu: float, sign: float) -> None:
    """Add/subtract one local lookup line into an existing full line model."""

    k, ii, jj, y, z = lorentzian_lookup_params(t_obs, nf, nnu, lnumin, lnumax, f0, nu)
    istart = k - wdth // 2
    for i in range(wdth):
        idx = istart + i
        if 0 < idx < line_model.size:
            line_model[idx] += sign * lorentzian_lookup_value(ltemplate, ii, jj, y, z, amp, i)


@njit(cache=True)
def add_lookup_line_band_inplace(line_model: np.ndarray, data_imin: int, t_obs: float,
                                 nf: int, nnu: int, wdth: int, lnumin: float,
                                 lnumax: float, ltemplate: np.ndarray, f0: float,
                                 amp: float, nu: float, sign: float) -> Tuple[int, int]:
    """Add/subtract one lookup line into an active-band model and return touched bins."""

    k, ii, jj, y, z = lorentzian_lookup_params(t_obs, nf, nnu, lnumin, lnumax, f0, nu)
    imid = k - data_imin
    idelt = wdth // 2
    istart = imid - idelt
    istop = imid + idelt
    ilow = line_model.size
    ihigh = 0
    for i in range(wdth):
        idx = istart + i
        if 0 <= idx < line_model.size:
            line_model[idx] += sign * lorentzian_lookup_value(ltemplate, ii, jj, y, z, amp, i)
            if idx < ilow:
                ilow = idx
            if idx + 1 > ihigh:
                ihigh = idx + 1
    if ihigh <= ilow:
        ilow = 0
        ihigh = 0
    return ilow, ihigh


@njit(cache=True)
def delta_loglike_range(power: np.ndarray, sn_new: np.ndarray, sn_old: np.ndarray,
                        ilow: int, ihigh: int, multiplicity: float) -> float:
    """Likelihood difference over a local proposal window."""

    val = 0.0
    tiny = np.finfo(np.float64).tiny
    lo = max(0, ilow)
    hi = min(power.size, ihigh)
    for i in range(lo, hi):
        snew = max(sn_new[i], tiny)
        sold = max(sn_old[i], tiny)
        val += multiplicity * (power[i] / sold - power[i] / snew + math.log(sold / snew))
    return val


@njit(cache=True)
def sline_from_lookup(ncut: int, data_imin: int, t_obs: float,
                      nf: int, nnu: int, wdth: int, lnumin: float, lnumax: float,
                      ltemplate: np.ndarray, lf: np.ndarray, la: np.ndarray,
                      lnu: np.ndarray, nlines: int) -> np.ndarray:
    """Assemble the full line PSD from all active windowed Lorentzians."""

    out = np.zeros(ncut, dtype=np.float64)
    for k in range(nlines):
        add_lookup_line_band_inplace(out, data_imin, t_obs, nf, nnu, wdth, lnumin, lnumax,
                                     ltemplate, lf[k], la[k], lnu[k], 1.0)
    return out


def add_gaussian_bump_band_inplace(model: np.ndarray, freq: np.ndarray, f0: float,
                                   amp: float, sigma: float, sign: float,
                                   nsigma: float = 5.0) -> Tuple[int, int]:
    """Add/subtract one broad Gaussian bump over its local frequency support."""

    if sigma <= 0.0 or amp <= 0.0 or freq.size == 0:
        return 0, 0
    lo = int(np.searchsorted(freq, f0 - nsigma * sigma, side="left"))
    hi = int(np.searchsorted(freq, f0 + nsigma * sigma, side="right"))
    lo = max(0, min(lo, model.size))
    hi = max(lo, min(hi, model.size))
    if hi <= lo:
        return 0, 0
    x = (freq[lo:hi] - f0) / sigma
    model[lo:hi] += sign * amp * np.exp(-0.5 * x * x)
    return lo, hi


def gaussian_bumps_from_params(freq: np.ndarray, bumps: GaussianBumpParams) -> np.ndarray:
    """Assemble the full broad-Gaussian PSD component over the active band."""

    out = np.zeros(freq.size, dtype=np.float64)
    for k in range(bumps.n):
        add_gaussian_bump_band_inplace(out, freq, bumps.f[k], bumps.a[k], bumps.sigma[k], 1.0)
    return out


# ---------------------------------------------------------------------------
# Akima spline and median-spectrum construction
# ---------------------------------------------------------------------------
#
# BayesLine uses Akima interpolation in medspecspline, splinestart, and
# trim_spline. These helpers keep that behavior in pure Python/numba rather than
# falling back to piecewise-linear interpolation.
@njit(cache=True)
def akima_derivatives(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Compute Akima slopes for a one-dimensional knot sequence."""

    d = np.empty(n, dtype=np.float64)
    if n == 1:
        d[0] = 0.0
        return d
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        d[0] = slope
        d[1] = slope
        return d

    m = np.empty(n + 3, dtype=np.float64)
    for i in range(n - 1):
        dx = x[i + 1] - x[i]
        if dx == 0.0:
            m[i + 2] = 0.0
        else:
            m[i + 2] = (y[i + 1] - y[i]) / dx

    m[1] = 2.0 * m[2] - m[3]
    m[0] = 2.0 * m[1] - m[2]
    m[n + 1] = 2.0 * m[n] - m[n - 1]
    m[n + 2] = 2.0 * m[n + 1] - m[n]

    for i in range(n):
        w1 = abs(m[i + 3] - m[i + 2])
        w2 = abs(m[i + 1] - m[i])
        if w1 + w2 > 0.0:
            d[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2)
        else:
            d[i] = 0.5 * (m[i + 1] + m[i + 2])
    return d


@njit(cache=True)
def akima_eval_one(x: np.ndarray, y: np.ndarray, n: int, xq: float) -> float:
    """Evaluate one Akima-interpolated value with constant extrapolation."""

    if xq <= x[0]:
        return y[0]
    if xq >= x[n - 1]:
        return y[n - 1]
    d = akima_derivatives(x, y, n)
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xq:
            lo = mid
        else:
            hi = mid
    h = x[lo + 1] - x[lo]
    t = (xq - x[lo]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y[lo] + h10 * h * d[lo] + h01 * y[lo + 1] + h11 * h * d[lo + 1]


@njit(cache=True)
def akima_eval_array(x: np.ndarray, y: np.ndarray, n: int, xq: np.ndarray) -> np.ndarray:
    """Vectorized Akima evaluation used for PSD spline models."""

    out = np.empty(xq.size, dtype=np.float64)
    if n == 1:
        out.fill(y[0])
        return out
    d = akima_derivatives(x, y, n)
    for j in range(xq.size):
        q = xq[j]
        if q <= x[0]:
            out[j] = y[0]
        elif q >= x[n - 1]:
            out[j] = y[n - 1]
        else:
            lo = 0
            hi = n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if x[mid] <= q:
                    lo = mid
                else:
                    hi = mid
            h = x[lo + 1] - x[lo]
            t = (q - x[lo]) / h
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            out[j] = h00 * y[lo] + h10 * h * d[lo] + h01 * y[lo + 1] + h11 * h * d[lo + 1]
    return out


@njit(cache=True)
def getrangeakima_numba(iu: int, nsy: int, x: np.ndarray, t_obs: float, nend: int) -> Tuple[int, int]:
    """C getrangeakima: bins affected by changing Akima control point iu."""

    nst = 3
    if nsy <= 0 or nend <= 0:
        return 0, 0
    if iu < 0:
        iu = 0
    if iu >= nsy:
        iu = nsy - 1

    if iu > nst - 1:
        imin = int(math.floor((x[iu - nst] - x[0]) * t_obs))
    else:
        imin = 0

    if iu < nsy - nst:
        imax = int(math.ceil((x[iu + nst] - x[0]) * t_obs + 1.0))
    else:
        imax = nend

    if imin < 0:
        imin = 0
    if imax > nend:
        imax = nend
    if imax < imin:
        imax = imin
    return imin, imax


def robust_smooth(power: np.ndarray, width: int = 24) -> np.ndarray:
    """Cheap robust PSD floor estimate: log-average followed by clipping."""
    floor = np.maximum(power, np.finfo(float).tiny)
    logp = np.log(floor)
    smooth = moving_average_numba(logp, width)
    model = np.exp(smooth)
    ratio = np.clip(power / np.maximum(model, np.finfo(float).tiny), 0.0, LINEMUL)
    clipped = np.minimum(power, model * np.maximum(1.0, ratio))
    return np.exp(moving_average_numba(np.log(np.maximum(clipped, np.finfo(float).tiny)), width))


def medspecspline_power(power: np.ndarray, df: float, N: int,
                        return_line_proposal: bool = False):
    """Python equivalent of C medspecspline, with Akima interpolation.

    The input is a periodogram-like power spectrum. The output triplet mirrors
    the C arrays: raw power S, line-preserving spectrum SN, and smooth median
    spectrum SM. Loud line bins are copied into SN but not into SM.
    """
    n2 = N // 2
    S = np.zeros(n2, dtype=np.float64)
    S[:min(n2, power.size)] = power[:min(n2, power.size)]
    if n2 > 1:
        S[0] = S[1]

    fspace = 2.0
    fw = fspace
    fend = (n2 - 1) * df
    fc = [df]
    i = 0
    while fc[-1] < fend:
        i += 1
        fc.append(fspace * i)
    fc_arr = np.array(fc, dtype=np.float64)
    nc = fc_arr.size
    sc = np.zeros(nc, dtype=np.float64)
    mm = max(1, int(fspace / df))
    mw = max(1, int(fw / df))

    for j in range(1, nc - 1):
        if fc_arr[j] >= 30.0 - 0.5 * fspace and fc_arr[j] < 30.0 + 0.5 * fspace:
            fw *= 2.0
            mw = max(1, int(fw / df))
        if fc_arr[j] >= 100.0 - 0.5 * fspace and fc_arr[j] < 100.0 + 0.5 * fspace:
            fw *= 2.0
            mw = max(1, int(fw / df))
        if fc_arr[j] >= 200.0 - 0.5 * fspace and fc_arr[j] < 200.0 + 0.5 * fspace:
            fw *= 2.0
            mw = max(1, int(fw / df))
        if fc_arr[j] + fw > fend:
            fw = max(df, fend - fc_arr[j])
            mw = max(1, int(fw / df))

        center = j * mm
        lo = max(1, center - mw)
        hi = min(n2, center + mw)
        if hi <= lo:
            lo = max(1, min(n2 - 1, center))
            hi = min(n2, lo + 1)
        med = float(np.median(S[lo:hi]))
        sc[j] = math.log(max(med / LN2, np.finfo(float).tiny))

    if nc > 1:
        sc[0] = sc[1]
        sc[-1] = sc[-2]

    freqs = np.arange(n2, dtype=np.float64) * df
    SM = np.zeros(n2, dtype=np.float64)
    if n2 > 1:
        SM[1:] = np.exp(akima_eval_array(fc_arr, sc, nc, freqs[1:]))
        SM[0] = SM[1]
    SN = SM.copy()
    line_proposal = np.ones(n2, dtype=np.float64)
    ratio = S / np.maximum(SM, np.finfo(float).tiny)
    SN[ratio > LINEMUL] = S[ratio > LINEMUL]
    line_proposal[ratio > LINE_PROPOSAL_THRESH] = LINE_PROPOSAL_BOOST
    if return_line_proposal:
        return S, SN, SM, line_proposal
    return S, SN, SM


# ---------------------------------------------------------------------------
# C-style wavelet glitch cleaning used by blstart
# ---------------------------------------------------------------------------
#
# This block ports SineGaussianC, TransformC, Getscale, and clean. The clean step
# identifies high-Q-transform excess power in whitened data, recolors the excess
# with the smooth PSD, and subtracts it from the time series before the startup
# PSD and line model are built.
def sine_gaussian_templates(freqs: np.ndarray, Q: float, t_obs: float, n: int) -> np.ndarray:
    """Build the frequency-domain sine-Gaussian templates used by TransformC."""

    n2 = n // 2
    dt = t_obs / n
    templates = np.zeros((freqs.size, n2 + 1), dtype=np.float64)
    for j, f0 in enumerate(freqs):
        tau = Q / (TPI * f0)
        fmax = f0 + 3.0 / tau
        fmin = f0 - 3.0 / tau
        fac = math.sqrt(math.sqrt(2.0) * math.pi * tau / dt)
        ic = int(f0 * t_obs)
        imin = int(fmin * t_obs)
        imax = int(fmax * t_obs)
        if imax - imin < 10:
            imin = ic - 5
            imax = ic + 5
        imin = max(0, imin)
        imax = min(n2, imax)
        for i in range(max(1, imin + 1), imax):
            f = i / t_obs
            templates[j, i] = fac * math.exp(-math.pi * math.pi * tau * tau * (f - f0) * (f - f0))
    return templates


def transform_c(fft_data: np.ndarray, freqs: np.ndarray, Q: float,
                t_obs: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Phase-blind Q-transform used to find clustered excess power."""

    n2 = n // 2
    templates = sine_gaussian_templates(freqs, Q, t_obs, n)
    fix = math.sqrt(n / 2.0)
    bmag = np.sqrt(np.sum(templates[:, 1:n2] * templates[:, 1:n2], axis=1) / n) / fix
    bmag = np.maximum(bmag, np.finfo(float).tiny)
    product = templates * fft_data[np.newaxis, :]
    tfR = np.fft.irfft(product, n=n, axis=1) / bmag[:, np.newaxis]
    tfI = np.fft.irfft(-1j * product, n=n, axis=1) / bmag[:, np.newaxis]
    return tfR * tfR + tfI * tfI, tfR


def getscale(freqs: np.ndarray, Q: float, t_obs: float, fmax: float, n: int) -> float:
    """Calibrate inverse Q-transform amplitude, matching C Getscale."""

    dt = t_obs / n
    f = fmax / 4.0
    t0 = t_obs / 2.0
    delt = t_obs / 8.0
    times = np.arange(n, dtype=np.float64) * dt
    ref = np.cos(TPI * times * f) * np.exp(-0.5 * ((times - t0) / delt) ** 2)
    packet_fft = np.fft.rfft(ref)
    _, tfR = transform_c(packet_fft, freqs, Q, t_obs, n)
    intime = np.sum(np.sqrt(freqs)[:, np.newaxis] * tfR, axis=0)
    live = np.abs(ref) > 0.01
    x = float(np.sum(intime[live] / ref[live]))
    x /= math.sqrt(2.0 * n)
    return float(np.count_nonzero(live)) / x


def clean_once(D: np.ndarray, Draw: np.ndarray, sqf: np.ndarray, freqs: np.ndarray,
               df: float, Q: float, t_obs: float, scale: float,
               imin: int, imax: int) -> Tuple[np.ndarray, float]:
    """Run one wavelet-clean pass.

    D supplies the current cleaned estimate used to build the whitening PSD.
    Draw is always the original windowed data. The returned array is Draw minus
    the recolored excess-power model, exactly as in BayesLine.c's clean.
    """

    n = D.size
    n2 = n // 2
    fft_D = np.fft.rfft(D)
    fft_raw = np.fft.rfft(Draw)

    power = np.zeros(n2, dtype=np.float64)
    power[1:] = 2.0 * np.abs(fft_D[1:n2]) ** 2
    _, specD, sspecD = medspecspline_power(power, df, n)

    whitened = fft_raw.copy()
    whitened[0] = 0.0
    whitened[n2] = 0.0
    whitened[1:n2] /= np.sqrt(np.maximum(specD[1:n2], np.finfo(float).tiny))

    tfD, tfDR = transform_c(whitened, freqs, Q, t_obs, n)
    live = tfD > STHRESH
    live2 = live.copy()
    if live.shape[0] > 2 and live.shape[1] > 2:
        neighbors = np.zeros_like(live[1:-1, 1:-1], dtype=bool)
        for jj in range(-1, 2):
            for ii in range(-1, 2):
                neighbors |= live[1 + jj:live.shape[0] - 1 + jj,
                                  1 + ii:live.shape[1] - 1 + ii]
        live2[1:-1, 1:-1] |= neighbors & (tfD[1:-1, 1:-1] > WARM)

    glitch = np.zeros(n, dtype=np.float64)
    if imax > imin:
        weights = scale * sqf
        glitch[imin:imax] = np.sum(weights[:, np.newaxis] * tfDR[:, imin:imax] * live2[:, imin:imax], axis=0)
    snr = math.sqrt(float(np.sum(glitch[imin:imax] * glitch[imin:imax])))
    print(f"   Excess SNR = {snr:f}")

    glitch_fft = np.fft.rfft(glitch)
    glitch_fft[0] = 0.0
    glitch_fft[n2] = 0.0
    glitch_fft[1:n2] *= np.sqrt(np.maximum(sspecD[1:n2], np.finfo(float).tiny))
    recolored = np.fft.irfft(glitch_fft, n=n)
    return Draw - recolored / math.sqrt(2.0 * n), snr


def bayesline_clean_time(data: np.ndarray, dt: float) -> np.ndarray:
    """Repeat wavelet cleaning until the C stopping criterion is satisfied."""

    n = data.size
    t_obs = n * dt
    fny = 1.0 / (2.0 * dt)
    imin = int(2.0 * T_RISE / dt)
    imax = n - imin
    subscale = 40
    # Clean below the BayesLine analysis band so low-frequency excess power
    # cannot leak upward into the PSD region. The PSD/line model still starts
    # from the analysis fmin in blstart.
    clean_fmin = 1.0 / t_obs
    octaves = int(np.rint(math.log(fny / clean_fmin) / math.log(2.0)))
    Nf = subscale * octaves + 1
    freqs = np.exp(math.log(clean_fmin) + np.arange(Nf, dtype=np.float64) * math.log(2.0) / subscale)
    sqf = np.sqrt(freqs)
    scale = getscale(freqs, Q_S, t_obs, fny, n)

    Draw = data.copy()
    D = data.copy()
    D, snr = clean_once(D, Draw, sqf, freqs, 1.0 / t_obs, Q_S, t_obs, scale, imin, imax)
    snr_old = 0.0
    i = 0
    while i < 5 and (snr - snr_old) > 10.0:
        snr_old = snr
        D, snr = clean_once(D, Draw, sqf, freqs, 1.0 / t_obs, Q_S, t_obs, scale, imin, imax)
        i += 1
    return D


def gakima_range(k: int, nknot: int, splinef: np.ndarray, t_obs: float) -> Tuple[int, int]:
    """Return the local Akima support range affected by removing one knot."""

    flag = 0
    if k > 3:
        imin = int(splinef[k - 3] * t_obs)
    else:
        imin = int(splinef[0] * t_obs)
        flag = 1
    if k < nknot - 4:
        imax = int(splinef[k + 3] * t_obs)
    else:
        imax = int(splinef[nknot - 1] * t_obs)
        flag = 2
    if flag == 1 and nknot > 6:
        imax = int(splinef[6] * t_obs)
    if flag == 2 and nknot > 6:
        imin = int(splinef[nknot - 7] * t_obs)
    return imin, imax


def trim_spline_akima(splinef: np.ndarray, splineA: np.ndarray, freqs: np.ndarray,
                      lSM: np.ndarray, lSMx: np.ndarray, istart: int, iend: int,
                      t_obs: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastically remove redundant spline knots using the C trim criterion."""

    nknot = splinef.size
    fkeep = 16.0
    kx = 0
    while kx < nknot - 1 and splinef[kx] < fkeep:
        kx += 1
    dLsq = 0.2 * 0.2
    active = np.ones(nknot, dtype=bool)
    NK = nknot
    trials = 2 * nknot

    ii = 0
    while ii < trials and NK > 50:
        ii += 1
        candidates = np.where(active)[0]
        candidates = candidates[(candidates >= kx) & (candidates < nknot - 1)]
        if candidates.size == 0:
            break
        j = int(rng.choice(candidates))
        keep = active.copy()
        keep[j] = False
        sf = splinef[keep]
        sA = splineA[keep]
        if sf.size < 7:
            continue
        ist, ied = gakima_range(j, nknot, splinef, t_obs)
        ist = max(istart, ist)
        ied = min(iend, ied)
        if ied <= ist:
            continue
        lSMy = akima_eval_array(sf, sA, sf.size, freqs[ist:ied + 1])
        dx = lSM[ist:ied + 1] - lSMx[ist:ied + 1]
        dy = lSM[ist:ied + 1] - lSMy
        logLx = float(np.sum((dx * dx / dLsq) ** 2))
        logLy = float(np.sum((dy * dy / dLsq) ** 2))
        z = float(np.max(np.abs(dy))) if dy.size else 0.0
        if logLx - logLy > -1.0 and z < 0.5:
            active[j] = False
            NK = int(np.sum(active))

    print(f"check {int(np.sum(active))} {NK}")
    return splinef[active], splineA[active]


def splinestart_akima(N: int, istart: int, iend: int, fstep: float, SM: np.ndarray,
                      freqs: np.ndarray, t_obs: float,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Create and trim the initial smooth-PSD Akima spline knots."""

    n2 = N // 2
    lSM = np.zeros(n2, dtype=np.float64)
    lSM[istart:n2] = np.log(np.maximum(SM[istart:n2], np.finfo(float).tiny))
    ii = max(1, int(t_obs * fstep))
    Np = max(1, iend - istart)
    kk = Np // ii
    if Np - kk * ii != 0:
        kk += 1

    splinef = []
    splineA = []
    for i in range(kk):
        k = min(iend, istart + i * ii)
        splinef.append(freqs[k])
        splineA.append(lSM[k])
    splinef.append(freqs[iend])
    splineA.append(lSM[iend])
    splinef_arr = np.array(splinef, dtype=np.float64)
    splineA_arr = np.array(splineA, dtype=np.float64)
    order = np.argsort(splinef_arr)
    splinef_arr = splinef_arr[order]
    splineA_arr = splineA_arr[order]

    Sfit = np.zeros(n2, dtype=np.float64)
    Sfit[istart:iend + 1] = akima_eval_array(splinef_arr, splineA_arr, splinef_arr.size, freqs[istart:iend + 1])
    splinef_arr, splineA_arr = trim_spline_akima(splinef_arr, splineA_arr, freqs, lSM, Sfit,
                                                 istart, iend, t_obs, rng)
    return splinef_arr, splineA_arr


# ---------------------------------------------------------------------------
# Windowed-Lorentzian lookup table generation and loading
# ---------------------------------------------------------------------------
#
# The lookup table is central to parity with the C code. It depends on segment
# duration and Tukey rolloff, so BayesLineSetup loads lookup_<Tobs>_<rise>.dat
# when present or generates the same table otherwise.
def lorentzraw_grid(f0: float, nu: float, amp: float, freqs: np.ndarray) -> np.ndarray:
    """Evaluate the unwindowed raw Lorentzian power profile.

    ``nu`` is the line width in Hz.  The oscillator denominator is written in
    angular frequency, so the width is converted to radians/sec here.
    """

    om0 = TPI * f0
    om = TPI * freqs
    nu_ang = TPI * nu
    x = om0 * om0 - om * om
    return amp / (x * x + nu_ang * nu_ang * om * om)


def windowed_lorentzian_template(f0: float, nu: float, amp: float, jwidth: int, n: int,
                                 t_obs: float,
                                 oversample: int, twl: np.ndarray) -> np.ndarray:
    """Convolve a raw Lorentzian with the squared Tukey-window response."""

    tlong = oversample * t_obs
    m = int(f0 * tlong)
    j1 = m - 2 * jwidth
    j2 = m + 2 * jwidth
    freqs = np.arange(j1, j2, dtype=np.float64) / tlong
    pf = lorentzraw_grid(f0, nu, amp, freqs)
    kernel = np.concatenate((twl[:0:-1], twl))
    if fftconvolve is not None:
        conv = fftconvolve(pf, kernel, mode="full")
    else:
        conv = np.convolve(pf, kernel, mode="full")
    line = np.zeros(n, dtype=np.float64)
    for i in range(m - jwidth, m + jwidth):
        if i % oversample == 0 and 0 < i < n * oversample // 2:
            idx = i - j1 + jwidth - 1
            if 0 <= idx < conv.size:
                line[i // oversample] = conv[idx]
    return line


def generate_lorentzian_lookup(t_obs: float, tukey_rise: float = T_RISE, nf: int = 40,
                               nnu: int = 20, wdth: Optional[int] = None,
                               numin: float = LINE_NU_MIN, numax: float = LINE_NU_MAX,
                               n_samples: Optional[int] = None,
                               oversample: int = 128) -> Tuple[int, int, int, float, float, np.ndarray]:
    """Generate the duration/rise-dependent lookup table used by llook."""

    if n_samples is None:
        n_samples = int(round(2.0 * t_obs * 1024.0))
    if wdth is None:
        wdth = int(16.0 * t_obs)
    if wdth < 4:
        wdth = 4

    alpha = 2.0 * tukey_rise / t_obs
    nx = oversample * n_samples
    jwidth = wdth * oversample // 2

    tuk = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
    tukey_inplace(tuk, alpha)
    tuk_long = np.zeros(nx, dtype=np.float64)
    start = nx // 2
    tuk_long[start:start + n_samples] = tuk
    tuk_fft = np.fft.rfft(tuk_long)
    twl = np.abs(tuk_fft[:jwidth]) ** 2

    ltemplate = np.zeros((nf + 1, nnu + 1, wdth), dtype=np.float64)
    lnumin = math.log(numin)
    lnumax = math.log(numax)
    m = int(40.0 * t_obs)

    for ii in range(nf + 1):
        f0 = 40.0 + (2.0 * ii - nf) / nf * 0.5 / t_obs
        for jj in range(nnu + 1):
            nu = math.exp(lnumin + (lnumax - lnumin) * jj / nnu)
            line = windowed_lorentzian_template(f0, nu, 1.0, jwidth, n_samples, t_obs, oversample, twl)
            lo = max(0, m - 4)
            hi = min(line.size, m + 5)
            peak = float(np.max(line[lo:hi]))
            if peak <= 0.0:
                peak = np.finfo(float).tiny
            for kk in range(wdth):
                idx = m + kk - wdth // 2
                val = line[idx] if 0 <= idx < line.size else np.finfo(float).tiny
                ltemplate[ii, jj, kk] = math.log(max(val / peak, np.finfo(float).tiny))

    return nf, nnu, wdth, numin, numax, ltemplate


def write_lorentzian_lookup(filename: Path, nf: int, nnu: int, wdth: int,
                            numin: float, numax: float, ltemplate: np.ndarray) -> None:
    """Write a lookup table in the same flat text format as BayesLine.c."""

    with filename.open("w", encoding="utf-8") as handle:
        handle.write(f"{nf} {nnu} {wdth} {numin:e} {numax:e}\n")
        flat = ltemplate.reshape(-1)
        for value in flat:
            handle.write(f"{value:.16e}\n")


def load_lorentzian_lookup(t_obs: float, tukey_rise: float = T_RISE,
                           n_samples: Optional[int] = None) -> Tuple[int, int, int, float, float, np.ndarray]:
    """Load an existing lookup table, or compute and save it if missing."""

    filename = Path(f"lookup_{int(round(t_obs))}_{tukey_rise:.2f}.dat")
    expected_nf = 40
    expected_nnu = 20
    expected_wdth = max(4, int(16.0 * t_obs))
    expected_numin = LINE_NU_MIN
    expected_numax = LINE_NU_MAX

    def generate_and_write(reason: str) -> Tuple[int, int, int, float, float, np.ndarray]:
        print(f"{filename} {reason}; generating windowed Lorentzian lookup table")
        if not SCIPY_AVAILABLE:
            print("warning: scipy is not available; falling back to slow np.convolve lookup generation")
        nf, nnu, wdth, numin, numax, ltemplate = generate_lorentzian_lookup(
            t_obs, tukey_rise=tukey_rise, nf=expected_nf, nnu=expected_nnu,
            wdth=expected_wdth, numin=expected_numin, numax=expected_numax,
            n_samples=n_samples
        )
        write_lorentzian_lookup(filename, nf, nnu, wdth, numin, numax, ltemplate)
        return nf, nnu, wdth, numin, numax, ltemplate

    if not filename.exists():
        return generate_and_write("not found")

    try:
        with filename.open("r", encoding="utf-8") as handle:
            header = handle.readline().split()
        if len(header) != 5:
            return generate_and_write("header is malformed")
        nf = int(header[0])
        nnu = int(header[1])
        wdth = int(header[2])
        numin = float(header[3])
        numax = float(header[4])
    except (OSError, ValueError):
        return generate_and_write("header could not be read")

    header_matches = (
        nf == expected_nf
        and nnu == expected_nnu
        and wdth == expected_wdth
        and math.isclose(numin, expected_numin, rel_tol=1.0e-12, abs_tol=0.0)
        and math.isclose(numax, expected_numax, rel_tol=1.0e-12, abs_tol=0.0)
    )
    if not header_matches:
        print(
            f"{filename} header {nf} {nnu} {wdth} {numin:e} {numax:e} "
            f"does not match expected {expected_nf} {expected_nnu} {expected_wdth} "
            f"{expected_numin:e} {expected_numax:e}"
        )
        return generate_and_write("has stale lookup settings")

    nf = int(header[0])
    nnu = int(header[1])
    wdth = int(header[2])
    numin = float(header[3])
    numax = float(header[4])
    try:
        values = np.loadtxt(filename, dtype=np.float64, skiprows=1)
    except (OSError, ValueError):
        return generate_and_write("body could not be read")
    expected = (nf + 1) * (nnu + 1) * wdth
    if values.size != expected:
        print(f"{filename} has {values.size} template values; expected {expected}")
        return generate_and_write("has incomplete lookup data")
    return nf, nnu, wdth, numin, numax, values.reshape((nf + 1, nnu + 1, wdth))


def attach_lorentzian_lookup(lines: LorentzianParams, data: DataParams) -> None:
    """Attach the lookup metadata and table to the active line model."""

    nf, nnu, wdth, numin, numax, ltemplate = load_lorentzian_lookup(data.t_obs, n_samples=data.n)
    lines.nf = nf
    lines.nnu = nnu
    lines.wdth = wdth
    lines.imin = data.imin
    lines.numin = numin
    lines.numax = numax
    lines.lnumin = math.log(numin)
    lines.lnumax = math.log(numax)
    lines.ltemplate = ltemplate


# ---------------------------------------------------------------------------
# Fast initial line fit: lineget, Lpeak, cut_lorentz, lmcmc
# ---------------------------------------------------------------------------
#
# The C startup line model is not a simple maximum-likelihood subtraction. It
# finds candidates in four threshold passes, refines them with the windowed
# lookup table, prunes overlapping/weak lines, and uses lmcmc to avoid
# over-subtracting the Fourier coefficients used to build the smooth spline.
def lmcmc_sample(M: int, SM: float, SL: float, dr: float, di: float,
                 xr: float, xi: float, rng: np.random.Generator) -> Tuple[float, float]:
    """Small MCMC line-subtraction step used inside C lorentzfit."""

    hrx = xr
    hix = xi
    sigma2 = 0.25 * SM * SL / max(SM + SL, np.finfo(float).tiny)
    sigma = math.sqrt(max(sigma2, 0.0))
    x = dr - hrx
    y = di - hix
    logLx = -2.0 * (x * x + y * y) / max(SM, np.finfo(float).tiny)

    for _ in range(1, M):
        hry = hrx + rng.normal(0.0, sigma)
        hiy = hix + rng.normal(0.0, sigma)
        x = dr - hry
        y = di - hiy
        logLy = -2.0 * (x * x + y * y) / max(SM, np.finfo(float).tiny)
        if math.log(max(rng.random(), np.finfo(float).tiny)) < logLy - logLx:
            logLx = logLy
            hrx = hry
            hix = hiy
    return hrx, hix


def subtract_lines_lmcmc(data_fft: np.ndarray, smooth: np.ndarray, line_power: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Subtract line contributions conservatively using lmcmc per frequency bin."""

    seed = int(rng.integers(1, 2**31 - 1))
    return subtract_lines_lmcmc_numba(data_fft, smooth, line_power, seed)


@njit(cache=True)
def subtract_lines_lmcmc_numba(data_fft: np.ndarray, smooth: np.ndarray,
                               line_power: np.ndarray, seed: int) -> np.ndarray:
    """Numba version of lmcmc subtraction, preserving the same MCMC update."""

    np.random.seed(seed)
    cleaned = data_fft.copy()
    n2 = min(cleaned.size, smooth.size, line_power.size)
    tiny = np.finfo(np.float64).tiny
    for i in range(1, n2):
        SM = smooth[i]
        SL = line_power[i]
        total = SM + SL
        if total <= 0.0:
            continue
        frac = SL / total
        if frac > 1.0e-2:
            dr = data_fft[i].real
            di = data_fft[i].imag
            hrx = frac * dr
            hix = frac * di
            sigma2 = 0.25 * SM * SL / max(total, tiny)
            sigma = math.sqrt(max(sigma2, 0.0))
            x = dr - hrx
            y = di - hix
            logLx = -2.0 * (x * x + y * y) / max(SM, tiny)
            for _ in range(1, 1000):
                hry = hrx + sigma * np.random.normal()
                hiy = hix + sigma * np.random.normal()
                x = dr - hry
                y = di - hiy
                logLy = -2.0 * (x * x + y * y) / max(SM, tiny)
                alpha = math.log(max(np.random.random(), tiny))
                if alpha < logLy - logLx:
                    logLx = logLy
                    hrx = hry
                    hix = hiy
            cleaned[i] = (dr - hrx) + 1j * (di - hix)
    return cleaned


@njit(cache=True)
def whiten_line_subtracted_numba(real_data: np.ndarray, imag_data: np.ndarray,
                                 sbase: np.ndarray, sline: np.ndarray,
                                 snf: np.ndarray, imin: int, imax: int,
                                 t_obs: float, seed: int) -> np.ndarray:
    """Build C BWtest whitelsf output from scaled Fourier data."""

    np.random.seed(seed)
    n_half = real_data.size - 1
    white = np.zeros((n_half - 1, 3), dtype=np.float64)
    gaussian_sigma = math.sqrt(0.5)
    tiny = np.finfo(np.float64).tiny
    for i in range(1, n_half):
        white[i - 1, 0] = i / t_obs
        if i < imin or i >= imax:
            white[i - 1, 1] = gaussian_sigma * np.random.normal()
            white[i - 1, 2] = gaussian_sigma * np.random.normal()
            continue

        j = i - imin
        dr = real_data[i]
        di = imag_data[i]
        hrx = 0.0
        hix = 0.0
        frac = sline[j] / max(snf[j], tiny)
        if frac > 1.0e-2:
            SM = 2.0 * sbase[j]
            SL = 2.0 * sline[j]
            total = SM + SL
            hrx = frac * dr
            hix = frac * di
            sigma2 = 0.25 * SM * SL / max(total, tiny)
            sigma = math.sqrt(max(sigma2, 0.0))
            x = dr - hrx
            y = di - hix
            logLx = -2.0 * (x * x + y * y) / max(SM, tiny)
            for _ in range(1, 1000):
                hry = hrx + sigma * np.random.normal()
                hiy = hix + sigma * np.random.normal()
                x = dr - hry
                y = di - hiy
                logLy = -2.0 * (x * x + y * y) / max(SM, tiny)
                alpha = math.log(max(np.random.random(), tiny))
                if alpha < logLy - logLx:
                    logLx = logLy
                    hrx = hry
                    hix = hiy

        scale = math.sqrt(max(sbase[j], tiny))
        white[i - 1, 1] = (dr - hrx) / scale
        white[i - 1, 2] = (di - hix) / scale
    return white


@njit(cache=True)
def line_subtracted_scaled_numba(real_data: np.ndarray, imag_data: np.ndarray,
                                 sbase: np.ndarray, sline: np.ndarray,
                                 snf: np.ndarray, imin: int, imax: int,
                                 seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract the sampled line contribution in BayesWave-scaled FFT units."""

    np.random.seed(seed)
    n_half = real_data.size - 1
    out_real = real_data.copy()
    out_imag = imag_data.copy()
    tiny = np.finfo(np.float64).tiny
    for i in range(1, n_half):
        if i < imin or i >= imax:
            continue

        j = i - imin
        dr = real_data[i]
        di = imag_data[i]
        hrx = 0.0
        hix = 0.0
        frac = sline[j] / max(snf[j], tiny)
        if frac > 1.0e-2:
            SM = 2.0 * sbase[j]
            SL = 2.0 * sline[j]
            total = SM + SL
            hrx = frac * dr
            hix = frac * di
            sigma2 = 0.25 * SM * SL / max(total, tiny)
            sigma = math.sqrt(max(sigma2, 0.0))
            x = dr - hrx
            y = di - hix
            logLx = -2.0 * (x * x + y * y) / max(SM, tiny)
            for _ in range(1, 1000):
                hry = hrx + sigma * np.random.normal()
                hiy = hix + sigma * np.random.normal()
                x = dr - hry
                y = di - hiy
                logLy = -2.0 * (x * x + y * y) / max(SM, tiny)
                alpha = math.log(max(np.random.random(), tiny))
                if alpha < logLy - logLx:
                    logLx = logLy
                    hrx = hry
                    hix = hiy

        out_real[i] = dr - hrx
        out_imag[i] = di - hix
    return out_real, out_imag


def scaled_rfft_to_time(real_data: np.ndarray, imag_data: np.ndarray,
                        dt: float, n: int) -> np.ndarray:
    """Convert BayesWave-scaled rFFT coefficients back to tapered time-domain strain.

    The returned time series is in strain units, but it is still the data that
    went through BWtest's Tukey window before the FFT.  We intentionally do not
    divide by the window on the way back because that would make the roll-off
    endpoints numerically fragile.
    """

    scaled_fft = real_data + 1j * imag_data
    return np.fft.irfft(scaled_fft * (math.sqrt(2.0) / dt), n=n)


def cleaned_time_from_scaled_freq(cleaned_freq_data: np.ndarray, dt: float, n: int) -> np.ndarray:
    """Reconstruct the wavelet-cleaned time series written into BWtest's fdata."""

    n_half = n // 2
    cleaned_real = np.zeros(n_half + 1, dtype=np.float64)
    cleaned_imag = np.zeros(n_half + 1, dtype=np.float64)
    cleaned_real[1:n_half] = cleaned_freq_data[2:n:2]
    cleaned_imag[1:n_half] = cleaned_freq_data[3:n + 1:2]
    return scaled_rfft_to_time(cleaned_real, cleaned_imag, dt, n)


def _psd_for_rfft(psd_one_sided: np.ndarray, n: int) -> np.ndarray:
    """Return a PSD array with explicit DC and Nyquist slots for an rfft."""

    n_half = n // 2
    if psd_one_sided.size < n_half:
        raise ValueError("PSD array is too short for the time series")
    psd = np.empty(n_half + 1, dtype=np.float64)
    psd[:n_half] = psd_one_sided[:n_half]
    psd[n_half] = psd_one_sided[n_half - 1] if n_half > 0 else psd_one_sided[0]
    return np.maximum(psd, np.finfo(float).tiny)


def whitened_rfft_with_psd(data: np.ndarray, psd_one_sided: np.ndarray,
                           dt: float, window_power_correction: float) -> np.ndarray:
    """Return Denoise.py-normalized whitened Fourier coefficients.

    Denoise.py performs the Q-scan on half-complex FFT coefficients divided by
    the square root of the raw spectral estimate. In BWtest units this is
    equivalent to using the BayesWave-scaled FFT and dividing by
    ``sqrt(2*PSD_internal)``. Since ``BWpsd = (4*W/T)*PSD_internal``, the raw
    NumPy FFT is scaled by ``dt*sqrt(W/(T*BWpsd))``. The corresponding display
    time series is obtained by inverse transforming these coefficients and
    multiplying by ``sqrt(2*N)``.
    """

    n = data.size
    n_half = n // 2
    t_obs = n * dt
    psd = _psd_for_rfft(psd_one_sided, n)
    white_fft = np.fft.rfft(data)
    white_fft[0] = 0.0
    white_fft[n_half] = 0.0
    white_fft[1:n_half] *= dt * math.sqrt(window_power_correction / t_obs) / np.sqrt(psd[1:n_half])
    return white_fft


def whiten_time_series_with_psd(data: np.ndarray, psd_one_sided: np.ndarray,
                                dt: float, window_power_correction: float) -> np.ndarray:
    """Whiten a time series with the final physical one-sided PSD.

    The returned time series follows the Denoise.py display convention:
    inverse-transform the Denoise-normalized Fourier coefficients and multiply
    by ``sqrt(2*N)``. For stationary Gaussian data, the interior samples have
    approximately unit variance up to edge tapering.
    """

    n = data.size
    white_fft = whitened_rfft_with_psd(data, psd_one_sided, dt, window_power_correction)
    return np.fft.irfft(white_fft, n=n) * math.sqrt(2.0 * n)


def compute_glitch_time_products(raw_time_data: np.ndarray, cleaned_freq_data: np.ndarray,
                                 final_psd: np.ndarray, dt: float,
                                 window_power_correction: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the wavelet-cleaning glitch, whitened time series, and whitened FFT."""

    cleaned_time = cleaned_time_from_scaled_freq(cleaned_freq_data, dt, raw_time_data.size)
    glitch_time = raw_time_data - cleaned_time
    whitened_glitch_fft = whitened_rfft_with_psd(glitch_time, final_psd, dt, window_power_correction)
    whitened_glitch = np.fft.irfft(whitened_glitch_fft, n=raw_time_data.size) * math.sqrt(2.0 * raw_time_data.size)
    return glitch_time, whitened_glitch, whitened_glitch_fft


@njit(cache=True)
def feature_flood_fill(img: np.ndarray, x: int, y: int, new_clr: int) -> None:
    """Connected-component fill used by Denoise.py's feature clustering."""

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
                if 0 <= xx < m and 0 <= yy < n and img[xx, yy] == prev:
                    img[xx, yy] = new_clr
                    qx[rear] = xx
                    qy[rear] = yy
                    rear += 1


@njit(cache=True)
def feature_cluster_core(freqs: np.ndarray, tfD: np.ndarray, tfR: np.ndarray,
                         scale: float, t_obs: float, snr_thresh: float) -> Tuple[
                             np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             int, float, int, float, float, float, float, float
                         ]:
    """Denoise.py clustering and cluster-SNR cut for whitened Q-transform pixels."""

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
            if flag == 1 and tfD[j, i] > FEATURE_WARM:
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
        feature_flood_fill(live2, xpix, ypix, new_clr)

    cf = new_clr - 1
    total_sig = np.zeros(nt, dtype=np.float64)
    if cf <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return total_sig, empty, empty, empty, empty, hot0, float(hot), -1, 0.0, 0.0, 0.0, 0.0, 0.0

    DT = np.zeros((cf, nt), dtype=np.float64)
    dt = t_obs / float(nt)
    imin = int(T_RISE / t_obs * float(nt))
    imax = int((t_obs - T_RISE) / t_obs * float(nt))
    for j in range(nf):
        sqf = scale * math.sqrt(freqs[j])
        for i in range(imin, imax):
            if live2[j, i] > 0:
                jj = live2[j, i] - 2
                DT[jj, i] += sqf * tfR[j, i]

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
        if csnr[j] > snr_thresh:
            for i in range(nt):
                total_sig[i] += DT[j, i]

    return (
        total_sig, csnr, peak_t, peak_f, peak_amp, hot0, float(hot), jmax,
        smax, tmin, tmax, fmin if ifmin < nf else 0.0, fmax if ifmax > 0 else 0.0
    )


def feature_q_frequencies(t_obs: float, fmax: float) -> np.ndarray:
    """Frequency layers for the Denoise.py-style Q-scan feature clustering."""

    fmin = 1.0 / t_obs
    if fmax <= fmin:
        return np.array([fmin], dtype=np.float64)
    octaves = max(1, int(np.rint(math.log(fmax / fmin) / math.log(2.0))))
    nf = FEATURE_QSCAN_SUBSCALE * octaves + 1
    dx = math.log(2.0) / float(FEATURE_QSCAN_SUBSCALE)
    return np.exp(math.log(fmin) + np.arange(nf, dtype=np.float64) * dx)


def extract_whitened_feature(whitened_glitch_fft: np.ndarray, t_obs: float,
                             snr_thresh: float) -> np.ndarray:
    """Cluster the whitened glitch and return the significant whitened feature.

    ``whitened_glitch_fft`` must be the Denoise-normalized rfft coefficients.
    It is smaller than ``np.fft.rfft(whitened_glitch_time)`` by ``sqrt(2*N)``;
    using the display time series here would inflate the cluster SNR.
    """

    n = 2 * (whitened_glitch_fft.size - 1)
    n_half = n // 2
    fmax = (n // 2 - 1) / t_obs
    freqs = feature_q_frequencies(t_obs, fmax)
    print(f"feature frequency layers = {freqs.size:d} fmin {freqs[0]:e} fmax {freqs[-1]:e}")

    whitened_fft = whitened_glitch_fft.copy()
    whitened_fft[0] = 0.0
    whitened_fft[n_half] = 0.0
    if n_half > 1:
        white_var = float(np.mean(2.0 * (whitened_fft[1:n_half].real ** 2 +
                                         whitened_fft[1:n_half].imag ** 2)))
        print(f"feature white variance {white_var:e}")
    tfD, tfR = transform_c(whitened_fft, freqs, Q_S, t_obs, n)
    scale = getscale(freqs, Q_S, t_obs, fmax, n)
    total_sig, csnr, peak_t, peak_f, _peak_amp, hot0, hot, jmax, smax, tmin, tmax, fmin, fmax_found = feature_cluster_core(
        freqs, tfD, tfR, scale, t_obs, snr_thresh
    )

    print(f"{hot0:d} initial hot pixels")
    print(f"{int(hot):d} hot pixels")
    print(f"{csnr.shape[0]:d} clusters")
    for j in range(csnr.shape[0]):
        print(f"SNR of cluster {j:d} = {csnr[j]:f} ({peak_t[j]:.2f}, {peak_f[j]:.2f})")
    if jmax >= 0:
        print(f"Loudest cluster is #{jmax:d} SNR = {smax:f}")
    else:
        print("No clusters found")
    print(f"tmin {tmin:f} tmax {tmax:f} fmin {fmin:f} fmax {fmax_found:f}")
    significant = int(np.sum(csnr > snr_thresh))
    print(f"{significant:d} significant cluster(s)")
    print(f"Feature SNR = {math.sqrt(float(np.dot(total_sig, total_sig))):f}")
    return total_sig


def write_whitened_line_subtracted_files(bayesline: BayesLineParams,
                                         original_fft: np.ndarray,
                                         cleaned_freq_data: np.ndarray,
                                         dt: float,
                                         rng: np.random.Generator) -> None:
    """Write the optional C-style whitelsf diagnostic files."""

    assert bayesline.data is not None
    assert bayesline.Sbase is not None and bayesline.Sline is not None and bayesline.Snf is not None
    data = bayesline.data
    n = data.n
    n_half = n // 2
    background = bayesline_background_psd(bayesline)

    cleaned_real = np.zeros(n_half + 1, dtype=np.float64)
    cleaned_imag = np.zeros(n_half + 1, dtype=np.float64)
    cleaned_real[1:n_half] = cleaned_freq_data[2:n:2]
    cleaned_imag[1:n_half] = cleaned_freq_data[3:n + 1:2]

    original_scaled = original_fft * (dt / math.sqrt(2.0))
    original_real = original_scaled.real.astype(np.float64, copy=True)
    original_imag = original_scaled.imag.astype(np.float64, copy=True)

    seed_noglitch = int(rng.integers(1, 2**31 - 1))
    white_noglitch = whiten_line_subtracted_numba(
        cleaned_real, cleaned_imag, background, bayesline.Sline, bayesline.Snf,
        data.imin, data.imax, data.t_obs, seed_noglitch
    )
    np.savetxt("whitelsf_noglitch.dat", white_noglitch)

    seed_raw = int(rng.integers(1, 2**31 - 1))
    white_raw = whiten_line_subtracted_numba(
        original_real, original_imag, background, bayesline.Sline, bayesline.Snf,
        data.imin, data.imax, data.t_obs, seed_raw
    )
    np.savetxt("whitelsf.dat", white_raw)


def write_time_domain_line_subtracted_files(bayesline: BayesLineParams,
                                            times: np.ndarray,
                                            original_fft: np.ndarray,
                                            cleaned_freq_data: np.ndarray,
                                            dt: float,
                                            rng: np.random.Generator) -> None:
    """Write unwhitened time-domain line-subtracted diagnostic files.

    The two outputs are time-domain counterparts to the optional whitelsf files:

    * line_subtracted_time.dat is built from the original tapered FFT data with
      the sampled BayesLine line model subtracted.
    * line_subtracted_noglitch_time.dat is built from the wavelet-cleaned FFT
      data used for PSD estimation, again with the line model subtracted.

    These files are not whitened. They are also not de-windowed: the Tukey
    roll-off applied before the FFT remains present at both ends of the segment.
    """

    assert bayesline.data is not None
    assert bayesline.Sbase is not None and bayesline.Sline is not None and bayesline.Snf is not None
    data = bayesline.data
    n = data.n
    n_half = n // 2
    background = bayesline_background_psd(bayesline)

    cleaned_real = np.zeros(n_half + 1, dtype=np.float64)
    cleaned_imag = np.zeros(n_half + 1, dtype=np.float64)
    cleaned_real[1:n_half] = cleaned_freq_data[2:n:2]
    cleaned_imag[1:n_half] = cleaned_freq_data[3:n + 1:2]

    original_scaled = original_fft * (dt / math.sqrt(2.0))
    original_real = original_scaled.real.astype(np.float64, copy=True)
    original_imag = original_scaled.imag.astype(np.float64, copy=True)

    seed_raw = int(rng.integers(1, 2**31 - 1))
    raw_real, raw_imag = line_subtracted_scaled_numba(
        original_real, original_imag, background, bayesline.Sline, bayesline.Snf,
        data.imin, data.imax, seed_raw
    )
    raw_time = scaled_rfft_to_time(raw_real, raw_imag, dt, n)
    np.savetxt("line_subtracted_time.dat", np.column_stack((times, raw_time)))
    print("Wrote line_subtracted_time.dat; WARNING: Tukey roll-off remains in the time-domain data")

    seed_noglitch = int(rng.integers(1, 2**31 - 1))
    noglitch_real, noglitch_imag = line_subtracted_scaled_numba(
        cleaned_real, cleaned_imag, background, bayesline.Sline, bayesline.Snf,
        data.imin, data.imax, seed_noglitch
    )
    noglitch_time = scaled_rfft_to_time(noglitch_real, noglitch_imag, dt, n)
    np.savetxt("line_subtracted_noglitch_time.dat", np.column_stack((times, noglitch_time)))
    print("Wrote line_subtracted_noglitch_time.dat; WARNING: Tukey roll-off remains in the time-domain data")


def write_line_subtracted_frequency_file(bayesline: BayesLineParams,
                                         original_fft: np.ndarray,
                                         dt: float,
                                         rng: np.random.Generator,
                                         output_file: str = "freq_nolines.dat") -> None:
    """Write original tapered FFT data after final fair-draw Lorentzian subtraction."""

    assert bayesline.data is not None
    assert bayesline.Sbase is not None and bayesline.Sline is not None and bayesline.Snf is not None
    data = bayesline.data
    n = data.n
    n_half = n // 2
    background = bayesline_background_psd(bayesline)

    original_scaled = original_fft * (dt / math.sqrt(2.0))
    original_real = original_scaled.real.astype(np.float64, copy=True)
    original_imag = original_scaled.imag.astype(np.float64, copy=True)

    seed = int(rng.integers(1, 2**31 - 1))
    out_real, out_imag = line_subtracted_scaled_numba(
        original_real, original_imag, background, bayesline.Sline, bayesline.Snf,
        data.imin, data.imax, seed
    )

    freqs = np.arange(1, n_half, dtype=np.float64) / data.t_obs
    np.savetxt(output_file, np.column_stack((freqs, out_real[1:n_half], out_imag[1:n_half])))
    print(f"Wrote {output_file} from original tapered data with final Lorentzian lines subtracted")


def write_glitch_time_domain_files(times: np.ndarray, raw_time_data: np.ndarray,
                                   cleaned_freq_data: np.ndarray, final_psd: np.ndarray,
                                   dt: float, window_power_correction: float) -> None:
    """Write the wavelet-cleaning excess and the same excess whitened by BWpsd.

    ``raw_time_data`` is the time series after BWtest's preprocessing window
    choice, and ``cleaned_freq_data`` is the BayesWave-scaled FFT of the
    wavelet-cleaned data produced during BayesLine burn-in. Their difference is
    the time-domain excess removed by the wavelet denoiser.
    """

    glitch_time, whitened_glitch, _whitened_glitch_fft = compute_glitch_time_products(
        raw_time_data, cleaned_freq_data, final_psd, dt, window_power_correction
    )

    np.savetxt("glitch_time.dat", np.column_stack((times, glitch_time)))
    np.savetxt("whitened_glitch_time.dat", np.column_stack((times, whitened_glitch)))
    print("Wrote glitch_time.dat and whitened_glitch_time.dat")


def write_whitened_feature_file(times: np.ndarray, raw_time_data: np.ndarray,
                                cleaned_freq_data: np.ndarray, final_psd: np.ndarray,
                                dt: float, window_power_correction: float,
                                snr_thresh: float) -> None:
    """Write the clustered, SNR-thresholded whitened glitch feature."""

    _glitch_time, _whitened_glitch, whitened_glitch_fft = compute_glitch_time_products(
        raw_time_data, cleaned_freq_data, final_psd, dt, window_power_correction
    )
    feature = extract_whitened_feature(whitened_glitch_fft, raw_time_data.size * dt, snr_thresh)
    np.savetxt("whitened_feature_time.dat", np.column_stack((times, feature)))
    print("Wrote whitened_feature_time.dat")


def complex_geometric_sum(theta: np.ndarray, start: int, count: int) -> np.ndarray:
    """Sum exp(i*theta*n) for n=start...start+count-1, vectorized in theta."""

    theta = np.asarray(theta, dtype=np.float64)
    out = np.zeros(theta.shape, dtype=np.complex128)
    if count <= 0:
        return out
    center = start + 0.5 * (count - 1)
    half_theta = 0.5 * theta
    denom = np.sin(half_theta)
    near = np.abs(denom) < 1.0e-12
    if np.any(near):
        out[near] = count * np.exp(1j * theta[near] * center)
    if np.any(~near):
        far = ~near
        out[far] = (
            np.exp(1j * theta[far] * center)
            * np.sin(0.5 * count * theta[far])
            / denom[far]
        )
    return out


def tukey_window_dft_offsets(offsets: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Exact DFT of BWtest's Tukey window at fractional bin offsets.

    For a complex sinusoid at frequency ``f0``, the rfft coefficient at bin
    ``k`` after applying the Tukey window is this function evaluated at
    ``offset = f0*Tobs - k``. This avoids doing a full FFT for every trial
    sinusoid in the line-by-line sinefit diagnostic.
    """

    offsets = np.asarray(offsets, dtype=np.float64)
    theta = TPI * offsets / float(n)
    imin = int(alpha * (n - 1.0) / 2.0)
    imax = int((n - 1.0) * (1.0 - alpha / 2.0))
    nwin = n - imax

    if imin <= 0 or nwin <= 0:
        return complex_geometric_sum(theta, 0, n)

    total = np.zeros(offsets.shape, dtype=np.complex128)

    # Leading roll-off: 0.5*(1 - cos(pi*n/imin)).
    beta = math.pi / float(imin)
    total += 0.5 * complex_geometric_sum(theta, 0, imin)
    total -= 0.25 * complex_geometric_sum(theta + beta, 0, imin)
    total -= 0.25 * complex_geometric_sum(theta - beta, 0, imin)

    # Flat section, including the samples at imin and imax, matching tukey_inplace.
    mid_start = imin
    mid_count = max(0, imax - imin + 1)
    total += complex_geometric_sum(theta, mid_start, mid_count)

    # Trailing roll-off: 0.5*(1 + cos(pi*(n-imax)/nwin)).
    tail_start = imax + 1
    tail_count = max(0, n - tail_start)
    if tail_count > 0:
        beta = math.pi / float(nwin)
        phase = -beta * imax
        total += 0.5 * complex_geometric_sum(theta, tail_start, tail_count)
        total += 0.25 * np.exp(1j * phase) * complex_geometric_sum(theta + beta, tail_start, tail_count)
        total += 0.25 * np.exp(-1j * phase) * complex_geometric_sum(theta - beta, tail_start, tail_count)

    return total


def sinefit(bayesline: BayesLineParams, n: int, alpha: float,
            output_file: str = "sinefit.dat",
            scan_points: int = SINEFIT_SCAN_POINTS) -> np.ndarray:
    """Match each reconstructed Lorentzian line to a Tukey-windowed sinusoid.

    The comparison is intentionally model-only for this first pass: the
    reconstructed line is the same local windowed-Lorentzian power profile used
    in the BayesLine line model, and the sinusoid is represented by the power
    profile of a Tukey-windowed pure tone. Inner products use the final
    broadband non-line PSD as the local noise model.
    """

    if bayesline.data is None or bayesline.lines_x is None:
        raise RuntimeError("BayesLine state is not initialized")
    lines = bayesline.lines_x
    if bayesline.Sbase is None:
        raise RuntimeError("Smooth PSD is not available for sinefit weighting")
    if lines.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")
    if scan_points < 1:
        raise ValueError("scan_points must be positive")

    nlines = int(lines.n)
    rows = np.zeros((nlines, 6), dtype=np.float64)
    if nlines == 0:
        np.savetxt(
            output_file,
            rows,
            header="line_frequency line_width sinusoid_frequency line_SNR residual_SNR mismatch",
        )
        print(f"Wrote {output_file} with 0 sinefit matches")
        return rows

    data = bayesline.data
    t_obs = data.t_obs
    n_half = n // 2
    smooth = np.maximum(bayesline_background_psd(bayesline), np.finfo(float).tiny)
    order = np.argsort(lines.f[:nlines])
    scan_axis = np.array([0.0], dtype=np.float64)
    if scan_points > 1:
        scan_axis = np.linspace(-1.0, 1.0, scan_points, dtype=np.float64)

    out_row = 0
    for idx in order:
        f_line = float(lines.f[idx])
        nu = float(lines.nu[idx])
        amp = float(lines.a[idx])
        if not (math.isfinite(f_line) and math.isfinite(nu)) or f_line <= 0.0 or nu <= 0.0:
            rows[out_row] = (f_line, nu, f_line, 0.0, 0.0, 1.0)
            out_row += 1
            continue

        k_center, lorentz_power = local_lorentzian_template(
            t_obs, lines.nf, lines.nnu, lines.wdth, lines.lnumin, lines.lnumax,
            lines.ltemplate, f_line, amp, nu
        )
        start = k_center - lines.wdth // 2
        bins_all = start + np.arange(lines.wdth, dtype=np.int64)
        valid = (
            (bins_all > 0) & (bins_all < n_half)
            & (bins_all >= data.imin) & (bins_all < data.imax)
        )
        if not np.any(valid):
            rows[out_row] = (f_line, nu, f_line, 0.0, 0.0, 1.0)
            out_row += 1
            continue

        bins_int = bins_all[valid]
        bins = bins_int.astype(np.float64)
        lorentz = np.maximum(lorentz_power[valid], 0.0)
        weight = 1.0 / (smooth[bins_int - data.imin] * smooth[bins_int - data.imin])
        lorentz_norm = float(np.dot(weight * lorentz, lorentz))
        line_snr = math.sqrt(max(lorentz_norm, 0.0))
        if lorentz_norm <= 0.0:
            rows[out_row] = (f_line, nu, f_line, 0.0, 0.0, 1.0)
            out_row += 1
            continue

        best_freq = f_line
        best_match = 0.0
        half_scan = 0.5 * nu
        for fs in f_line + half_scan * scan_axis:
            if fs <= 0.0 or fs >= n_half / t_obs:
                continue
            offsets = fs * t_obs - bins
            sine_power = np.abs(tukey_window_dft_offsets(offsets, n, alpha)) ** 2
            sine_norm = float(np.dot(weight * sine_power, sine_power))
            if sine_norm <= 0.0:
                continue
            match = float(np.dot(weight * lorentz, sine_power) / math.sqrt(lorentz_norm * sine_norm))
            if match > best_match:
                best_match = min(match, 1.0)
                best_freq = float(fs)

        residual_snr = line_snr * math.sqrt(max(0.0, 1.0 - best_match * best_match))
        rows[out_row] = (
            f_line, nu, best_freq, line_snr, residual_snr,
            max(0.0, 1.0 - best_match)
        )
        out_row += 1

    np.savetxt(
        output_file,
        rows,
        header="line_frequency line_width sinusoid_frequency line_SNR residual_SNR mismatch",
    )
    print(f"Wrote {output_file} with {nlines:d} sinefit matches")
    return rows


def tukeyed_real_sinusoid_scaled_templates(frequency: float, bins: np.ndarray,
                                           n: int, t_obs: float,
                                           alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return scaled rfft templates for unit cosine and sine at one frequency."""

    offsets_plus = frequency * t_obs - bins
    offsets_minus = -frequency * t_obs - bins
    wplus = tukey_window_dft_offsets(offsets_plus, n, alpha)
    wminus = tukey_window_dft_offsets(offsets_minus, n, alpha)
    scale = (t_obs / float(n)) / math.sqrt(2.0)
    cos_template = 0.5 * (wplus + wminus) * scale
    sin_template = (wplus - wminus) / (2.0j) * scale
    return cos_template, sin_template


def fit_tukeyed_sinusoid_to_band(data_band: np.ndarray, weight: np.ndarray,
                                 frequency: float, bins: np.ndarray,
                                 n: int, t_obs: float, alpha: float) -> Tuple[float, float, np.ndarray, float]:
    """Fit cosine/sine coefficients for a Tukey-windowed sinusoid in one band."""

    cos_template, sin_template = tukeyed_real_sinusoid_scaled_templates(
        frequency, bins, n, t_obs, alpha
    )
    cc = float(np.real(np.sum(weight * np.conj(cos_template) * cos_template)))
    ss = float(np.real(np.sum(weight * np.conj(sin_template) * sin_template)))
    cs = float(np.real(np.sum(weight * np.conj(cos_template) * sin_template)))
    rc = float(np.real(np.sum(weight * np.conj(cos_template) * data_band)))
    rs = float(np.real(np.sum(weight * np.conj(sin_template) * data_band)))

    det = cc * ss - cs * cs
    if det <= np.finfo(float).tiny:
        empty = np.zeros_like(data_band)
        return 0.0, 0.0, empty, 0.0

    ccoef = (rc * ss - rs * cs) / det
    scoef = (cc * rs - cs * rc) / det
    model = ccoef * cos_template + scoef * sin_template
    snr2 = max(float(np.real(np.sum(weight * np.conj(model) * model))), 0.0)
    return ccoef, scoef, model, snr2


def write_sine_subtracted_frequency_file(bayesline: BayesLineParams,
                                         original_fft: np.ndarray,
                                         dt: float,
                                         n: int, alpha: float,
                                         output_file: str = "freq_nosine.dat",
                                         scan_points: int = SINEFIT_SCAN_POINTS) -> None:
    """Write original tapered FFT data with best-fit Tukeyed sinusoids subtracted."""

    if bayesline.data is None or bayesline.lines_x is None:
        raise RuntimeError("BayesLine state is not initialized")
    if bayesline.Sbase is None:
        raise RuntimeError("Smooth PSD is not available for sinusoid subtraction")
    lines = bayesline.lines_x
    if lines.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")

    data = bayesline.data
    t_obs = data.t_obs
    n_half = n // 2
    original_scaled = original_fft * (dt / math.sqrt(2.0))
    residual = np.zeros(n_half + 1, dtype=np.complex128)
    residual[1:n_half] = original_scaled[1:n_half]
    smooth = np.maximum(bayesline_background_psd(bayesline), np.finfo(float).tiny)
    scan_axis = np.array([0.0], dtype=np.float64)
    if scan_points > 1:
        scan_axis = np.linspace(-1.0, 1.0, scan_points, dtype=np.float64)

    nfit = 0
    order = np.argsort(lines.f[:lines.n])
    for idx in order:
        f_line = float(lines.f[idx])
        nu = float(lines.nu[idx])
        if not (math.isfinite(f_line) and math.isfinite(nu)) or f_line <= 0.0 or nu <= 0.0:
            continue

        k_center, _line_shape = local_lorentzian_template(
            t_obs, lines.nf, lines.nnu, lines.wdth, lines.lnumin, lines.lnumax,
            lines.ltemplate, f_line, 1.0, nu
        )
        start = k_center - lines.wdth // 2
        bins_all = start + np.arange(lines.wdth, dtype=np.int64)
        valid = (
            (bins_all > 0) & (bins_all < n_half)
            & (bins_all >= data.imin) & (bins_all < data.imax)
        )
        if not np.any(valid):
            continue

        bins_int = bins_all[valid]
        bins = bins_int.astype(np.float64)
        weight = 1.0 / smooth[bins_int - data.imin]
        data_band = residual[bins_int]
        best_model = np.zeros_like(data_band)
        best_snr2 = -1.0
        half_scan = 0.5 * nu

        for fs in f_line + half_scan * scan_axis:
            if fs <= 0.0 or fs >= n_half / t_obs:
                continue
            _ccoef, _scoef, model, snr2 = fit_tukeyed_sinusoid_to_band(
                data_band, weight, float(fs), bins, n, t_obs, alpha
            )
            if snr2 > best_snr2:
                best_snr2 = snr2
                best_model = model

        if best_snr2 > 0.0:
            residual[bins_int] -= best_model
            nfit += 1

    freqs = np.arange(1, n_half, dtype=np.float64) / t_obs
    np.savetxt(output_file, np.column_stack((freqs, residual[1:n_half].real, residual[1:n_half].imag)))
    print(f"Wrote {output_file} after subtracting {nfit:d} fitted Tukeyed sinusoids")


def lineget_candidates(power: np.ndarray, smooth: np.ndarray, freqs: np.ndarray,
                       istart: int, iend: int, max_lines: int,
                       candidate_threshold: float) -> np.ndarray:
    """Find local periodogram peaks above a staged power threshold."""

    return lineget_candidates_numba(power, smooth, freqs, istart, iend, max_lines, candidate_threshold)


@njit(cache=True)
def lineget_candidates_numba(power: np.ndarray, smooth: np.ndarray, freqs: np.ndarray,
                             istart: int, iend: int, max_lines: int,
                             candidate_threshold: float) -> np.ndarray:
    """Compiled local peak finder for the C lineget scan."""

    start = max(3, istart)
    stop = min(iend, power.size - 3)
    found = np.empty(max_lines, dtype=np.float64)
    count = 0
    for i in range(start, stop):
        if power[i] / max(smooth[i], np.finfo(np.float64).tiny) > candidate_threshold:
            if power[i] > power[i - 1] and power[i] > power[i + 1]:
                found[count] = freqs[i]
                count += 1
                if count >= max_lines:
                    break
    return found[:count]


def lookup_unit(line: LorentzianParams, n2: int, t_obs: float, f0: float, nu: float,
                amp: float = 1.0) -> np.ndarray:
    """Return one full-grid windowed Lorentzian from the lookup table."""

    if line.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")
    return lookup_line_delta(np.zeros(n2, dtype=np.float64), 0, t_obs, line.nf, line.nnu,
                             line.wdth, line.lnumin, line.lnumax, line.ltemplate,
                             f0, amp, nu, 1.0)


@njit(cache=True)
def lpeak_windowed_numba(power: np.ndarray, smooth: np.ndarray, n: int, spread: int,
                         f0: float, nu: float, t_obs: float, nf: int, nnu: int,
                         wdth: int, lnumin: float, lnumax: float,
                         ltemplate: np.ndarray) -> Tuple[float, float]:
    """Compiled C Lpeak using only the local lookup window."""

    k = int(round(f0 * t_obs))
    y = (f0 - k / t_obs) * t_obs
    if y < -0.5:
        k -= 1
    if y > 0.5:
        k += 1
    y = (f0 - k / t_obs) * t_obs

    if 1 < k < n // 2 - 2:
        if y < 0.0:
            peak = (-y) * (power[k - 1] - smooth[k - 1]) + (1.0 + y) * (power[k] - smooth[k])
        else:
            peak = y * (power[k + 1] - smooth[k + 1]) + (1.0 - y) * (power[k] - smooth[k])
    else:
        peak = 0.0
    if peak < 0.0:
        peak = 0.0

    k, ii, jj, yl, zl = lorentzian_lookup_params(t_obs, nf, nnu, lnumin, lnumax, f0, nu)
    logL = 0.0
    j = k - wdth // 2
    tiny = np.finfo(np.float64).tiny
    ilo = wdth // 2 - spread
    ihi = wdth // 2 + spread
    if ilo < 0:
        ilo = 0
    if ihi >= wdth:
        ihi = wdth - 1
    for i in range(ilo, ihi + 1):
        idx = i + j
        if 0 < idx < n // 2:
            s0 = max(smooth[idx], tiny)
            lt_i = lorentzian_lookup_value(ltemplate, ii, jj, yl, zl, peak, i)
            s1 = max(smooth[idx] + lt_i, tiny)
            logL += -math.log(s1) - power[idx] / s1
            logL += math.log(s0) + power[idx] / s0
    return logL, peak


@njit(cache=True)
def best_lpeak_for_candidate_numba(fpeak: float, power: np.ndarray, smooth: np.ndarray,
                                   n: int, spread: int, t_obs: float, nf: int,
                                   nnu: int, wdth: int, lnumin: float, lnumax: float,
                                   ltemplate: np.ndarray, freq_offsets: np.ndarray,
                                   nu_grid: np.ndarray) -> Tuple[float, float, float, float]:
    """Search the same startup frequency/nu grid as C lorentzfit for one peak."""

    best_logL = -1.0e40
    best_f = fpeak
    best_nu = nu_grid[0]
    best_amp = 0.0
    for ii in range(freq_offsets.size):
        fx = fpeak + freq_offsets[ii]
        for jj in range(nu_grid.size):
            logL, amp = lpeak_windowed_numba(
                power, smooth, n, spread, fx, nu_grid[jj], t_obs, nf, nnu, wdth, lnumin, lnumax, ltemplate
            )
            if logL > best_logL:
                best_logL = logL
                best_f = fx
                best_nu = nu_grid[jj]
                best_amp = amp
    return best_logL, best_f, best_nu, best_amp


def lpeak_windowed(line: LorentzianParams, power: np.ndarray, smooth: np.ndarray, n: int,
                   spread: int, f0: float, nu: float, t_obs: float) -> Tuple[float, float]:
    """C Lpeak: choose amplitude from local excess and score a trial line."""

    if line.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")
    return lpeak_windowed_numba(power, smooth, n, spread, f0, nu, t_obs, line.nf, line.nnu,
                                line.wdth, line.lnumin, line.lnumax, line.ltemplate)


@njit(cache=True)
def cut_lorentz_windowed_numba(smooth: np.ndarray, line_model: np.ndarray,
                               power: np.ndarray, f0: float, nu: float, amp: float,
                               t_obs: float, nf: int, nnu: int, wdth: int,
                               lnumin: float, lnumax: float, numin: float, numax: float,
                               ltemplate: np.ndarray, freq_offsets: np.ndarray,
                               log_offsets: np.ndarray, linecut: float) -> Tuple[float, float, float, float]:
    """Compiled C cut_lorentz equivalent, mutating line_model in-place."""

    n2 = power.size
    old_k, old_ii, old_jj, old_y, old_z = lorentzian_lookup_params(
        t_obs, nf, nnu, lnumin, lnumax, f0, nu
    )
    imin = old_k - wdth // 2
    imax = old_k + wdth // 2
    if imin < 1:
        imin = 1
    if imax > n2:
        imax = n2
    local_n = imax - imin
    base = np.empty(local_n, dtype=np.float64)
    for i in range(local_n):
        base[i] = line_model[imin + i]

    old_start = old_k - wdth // 2
    for i in range(wdth):
        idx = old_start + i
        if imin <= idx < imax:
            base[idx - imin] -= lorentzian_lookup_value(ltemplate, old_ii, old_jj, old_y, old_z, amp, i)

    tiny = np.finfo(np.float64).tiny
    logLx = 0.0
    for i in range(local_n):
        idx = imin + i
        s = smooth[idx] + base[i]
        if s > 0.0:
            logLx += -(math.log(s) + power[idx] / s)

    lognu = math.log(max(nu, tiny))
    logamp = math.log(max(amp, tiny))
    best_logL = -1.0e60
    best_f = f0
    best_nu = nu
    best_amp = amp

    for ii in range(freq_offsets.size):
        fx = f0 + freq_offsets[ii]
        for jj in range(log_offsets.size):
            nux = math.exp(lognu + log_offsets[jj])
            if nux >= numin and nux <= numax:
                k, ii, jj_lookup, yl, zl = lorentzian_lookup_params(
                    t_obs, nf, nnu, lnumin, lnumax, fx, nux
                )
                trial_start = k - wdth // 2
                for kk in range(log_offsets.size):
                    ax = math.exp(logamp + log_offsets[kk])
                    logLy = 0.0
                    for i in range(local_n):
                        idx = imin + i
                        sline = base[i]
                        offset = idx - trial_start
                        if 0 <= offset < wdth:
                            sline += lorentzian_lookup_value(ltemplate, ii, jj_lookup, yl, zl, ax, offset)
                        s = smooth[idx] + sline
                        if s > 0.0:
                            logLy += -(math.log(s) + power[idx] / s)
                    if logLy > best_logL:
                        best_logL = logLy
                        best_f = fx
                        best_nu = nux
                        best_amp = ax

    dlogL = best_logL - logLx
    for i in range(local_n):
        line_model[imin + i] = base[i]

    if dlogL >= linecut:
        add_lookup_line_inplace(line_model, t_obs, nf, nnu, wdth, lnumin, lnumax,
                                ltemplate, best_f, best_amp, best_nu, 1.0)
        return dlogL, best_f, best_nu, best_amp
    return dlogL, f0, nu, amp


def cut_lorentz_windowed(line: LorentzianParams, smooth: np.ndarray, line_model: np.ndarray,
                         power: np.ndarray, f0: float, nu: float, amp: float,
                         t_obs: float, linecut: float = LINECUT) -> Tuple[float, float, float, float, np.ndarray]:
    """C cut_lorentz: refine one candidate and remove it if it is not needed."""

    if line.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")
    freq_offsets = 0.01 * np.arange(-5, 6, dtype=np.float64) / t_obs
    log_offsets = np.arange(-4, 5, dtype=np.float64) / 8.0
    dlogL, best_f, best_nu, best_amp = cut_lorentz_windowed_numba(
        smooth, line_model, power, f0, nu, amp, t_obs, line.nf, line.nnu, line.wdth,
        line.lnumin, line.lnumax, line.numin, line.numax, line.ltemplate, freq_offsets, log_offsets,
        linecut
    )
    return dlogL, best_f, best_nu, best_amp, line_model


def lorentzfit_four_pass(line: LorentzianParams, data_fft: np.ndarray, freqs: np.ndarray,
                         t_obs: float, istart: int, iend: int, max_lines: int,
                         rng: np.random.Generator, linecut: float = LINECUT,
                         line_proposal: Optional[np.ndarray] = None,
                         line_proposal_lorentzfit: Optional[np.ndarray] = None) -> np.ndarray:
    """Four-pass C lorentzfit startup using windowed lines and lmcmc cleanup."""

    if line.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")
    n2 = data_fft.size - 1
    n = 2 * n2
    dhold = data_fft.copy()
    dprime = data_fft.copy()
    PR = 2.0 * np.abs(dhold[:n2]) ** 2
    lnumin = line.lnumin
    lnumax = line.lnumax
    lnu1 = min(math.log(2.0e-4), lnumax)
    lnu2 = min(math.log(2.0e-3), lnumax)
    lf = np.zeros(max_lines, dtype=np.float64)
    lnu = np.zeros(max_lines, dtype=np.float64)
    lamp = np.zeros(max_lines, dtype=np.float64)
    lgL = np.zeros(max_lines, dtype=np.float64)
    nltotal = 0
    line_model = np.zeros(n2, dtype=np.float64)
    freq_offsets = np.arange(-100, 101, dtype=np.float64) / (200.0 * t_obs)
    cut_freq_offsets = 0.01 * np.arange(-5, 6, dtype=np.float64) / t_obs
    cut_log_offsets = np.arange(-4, 5, dtype=np.float64) / 8.0

    for m in range(4):
        lnm = lnu1 if m == 0 else (lnu2 if m == 1 else lnumax)
        # The staged power threshold limits early passes to only the loudest
        # periodogram excesses. The late-pass Lpeak likelihood threshold is
        # decoupled from LINEMUL so the RJMCMC can be seeded with weaker line
        # candidates without changing medspecspline's PSD construction.
        candidate_threshold = 100.0 if m == 0 else (30.0 if m == 1 else LINEMUL)
        likelihood_threshold = 100.0 if m == 0 else (30.0 if m == 1 else LINETHRESH)
        nu_grid = np.exp(lnumin + (lnm - lnumin) * np.arange(21, dtype=np.float64) / 20.0)
        power = 2.0 * np.abs(dprime[:n2]) ** 2
        PG, _, SM = medspecspline_power(power, 1.0 / t_obs, n)
        candidates = lineget_candidates(PG, SM, freqs[:n2], istart, iend, max_lines, candidate_threshold)
        print(
            f"There are {candidates.size} line candidates above power threshold {candidate_threshold:g} "
            f"(Lpeak threshold {likelihood_threshold:g})"
        )

        added = 0
        for fpeak in candidates:
            if nltotal + added >= max_lines:
                break
            best_logL, best_f, best_nu, best_amp = best_lpeak_for_candidate_numba(
                float(fpeak), PG, SM, n, 2, t_obs, line.nf, line.nnu, line.wdth,
                line.lnumin, line.lnumax, line.ltemplate, freq_offsets, nu_grid
            )
            if best_logL >= likelihood_threshold and best_amp > 0.0:
                idx = nltotal + added
                lf[idx] = best_f
                lnu[idx] = best_nu
                lamp[idx] = best_amp
                lgL[idx] = best_logL
                if line_proposal is not None:
                    prop_idx = int(best_f * t_obs)
                    if 0 <= prop_idx < min(line_proposal.size, n2):
                        line_proposal[prop_idx] = max(line_proposal[prop_idx], LINE_PROPOSAL_BOOST)
                        if line_proposal_lorentzfit is not None:
                            line_proposal_lorentzfit[prop_idx] = max(
                                line_proposal_lorentzfit[prop_idx], LINE_PROPOSAL_BOOST
                            )
                add_lookup_line_inplace(line_model, t_obs, line.nf, line.nnu, line.wdth,
                                        line.lnumin, line.lnumax, line.ltemplate,
                                        best_f, best_amp, best_nu, 1.0)
                added += 1
        nltotal += added

        if nltotal > 0:
            order = np.argsort(lgL[:nltotal])[::-1]
            new_lf = np.zeros(max_lines, dtype=np.float64)
            new_lnu = np.zeros(max_lines, dtype=np.float64)
            new_lamp = np.zeros(max_lines, dtype=np.float64)
            new_lgL = np.zeros(max_lines, dtype=np.float64)
            kept = 0
            for idx in order:
                iidx = int(idx)
                dlogL, fnew, nunew, ampnew = cut_lorentz_windowed_numba(
                    SM, line_model, PR, lf[iidx], lnu[iidx], lamp[iidx], t_obs,
                    line.nf, line.nnu, line.wdth, line.lnumin, line.lnumax,
                    line.numin, line.numax, line.ltemplate, cut_freq_offsets, cut_log_offsets,
                    linecut
                )
                if dlogL >= linecut and kept < max_lines:
                    new_lf[kept] = fnew
                    new_lnu[kept] = nunew
                    new_lamp[kept] = ampnew
                    new_lgL[kept] = lgL[iidx]
                    kept += 1
            lf, lnu, lamp, lgL = new_lf, new_lnu, new_lamp, new_lgL
            nltotal = kept

        cleaned = subtract_lines_lmcmc(dhold[:n2], SM, line_model, rng)
        dprime = dhold.copy()
        dprime[:n2] = cleaned

    line.n = min(nltotal, line.size)
    for i in range(line.n):
        line.f[i] = lf[i]
        line.nu[i] = lnu[i]
        line.a[i] = lamp[i]
        line.q[i] = line.f[i] / max(line.nu[i], np.finfo(float).tiny)
    return dprime


# ---------------------------------------------------------------------------
# BayesLine setup, burn-in, and proposal helpers
# ---------------------------------------------------------------------------
#
# These routines follow the public BayesLine C entry points. BayesLineSetup
# creates the state and loads the lookup table. BayesLineBurnin calls blstart,
# assembles the initial PSD components, writes diagnostics, and sets priors for
# the long RJMCMC.
def CubicSplineGSL(imin: int, imax: int, n: int, x: np.ndarray, y: np.ndarray, xint: np.ndarray, yint: np.ndarray) -> None:
    """Compatibility wrapper for the C cubic-spline branch."""

    yint[imin:imax] = np.interp(xint[imin:imax], x[:n], y[:n], left=y[0], right=y[n - 1])


def AkimaSplineGSL(imin: int, imax: int, n: int, x: np.ndarray, y: np.ndarray, xint: np.ndarray, yint: np.ndarray) -> None:
    """Compatibility wrapper for the C Akima-spline branch."""

    yint[imin:imax] = akima_eval_array(x[:n], y[:n], n, xint[imin:imax])


def spline_eval_array(points: np.ndarray, values: np.ndarray, xq: np.ndarray, spline_flag: int) -> np.ndarray:
    """Evaluate either Akima or linear interpolation according to SplineFlag."""

    if spline_flag == 1:
        return akima_eval_array(points, values, points.size, xq)
    return np.interp(xq, points, values, left=values[0], right=values[-1])


def spline_eval_one(points: np.ndarray, values: np.ndarray, xq: float, spline_flag: int) -> float:
    """Scalar version of spline_eval_array for proposal densities."""

    if spline_flag == 1:
        return float(akima_eval_one(points, values, points.size, xq))
    return float(np.interp(xq, points, values, left=values[0], right=values[-1]))


def spline_proposal_update(prop_points: np.ndarray, prop_sdata: np.ndarray,
                           freq: np.ndarray, data: DataParams,
                           sbase: np.ndarray, fixed_psd: np.ndarray, snx: np.ndarray,
                           power: np.ndarray, logLx: float, spline_flag: int,
                           changed_index: int,
                           compute_loglike: bool = True,
                           multiplicity: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Optional[float], int, int]:
    """Build a proposed PSD after a spline move, using local Akima updates.

    The Akima branch mirrors BayesLine.c: changing one spline knot only affects
    the frequency bins bracketed by three neighboring knots on either side.
    Cubic/linear interpolation is kept on the existing full-band path.
    """

    if spline_flag != 1 or changed_index < 0:
        sbase_prop = np.exp(spline_eval_array(prop_points, prop_sdata, freq, spline_flag))
        return sbase_prop, sbase_prop + fixed_psd, None, 0, freq.size

    imin, imax = getrangeakima_numba(changed_index, prop_points.size, prop_points,
                                     data.t_obs, data.ncut)
    sbase_prop = sbase.copy()
    sn_prop = snx.copy()
    if imax > imin:
        sbase_prop[imin:imax] = np.exp(
            akima_eval_array(prop_points, prop_sdata, prop_points.size, freq[imin:imax])
        )
        sn_prop[imin:imax] = sbase_prop[imin:imax] + fixed_psd[imin:imax]
    logLy_local = None
    if compute_loglike:
        logLy_local = logLx + delta_loglike_range(power, sn_prop, snx, imin, imax, multiplicity)
    return sbase_prop, sn_prop, logLy_local, imin, imax


def create_dataParams(freq: np.ndarray, n: int, t_obs: float, max_lines: int, fmin: float, fmax: float) -> DataParams:
    """Create the active frequency-band metadata used by BayesLine."""

    del max_lines
    df = 1.0 / t_obs
    imin = max(1, int(math.floor(fmin * t_obs)))
    imax = min(n // 2, int(math.floor(fmax * t_obs)))
    ncut = max(0, imax - imin)
    return DataParams(n=n, t_obs=t_obs, fmin=fmin, fmax=fmax, df=df, imin=imin, imax=imax,
                      ncut=ncut, flow=freq[imin], fhigh=freq[imax - 1])


def create_lorentzianParams(size: int) -> LorentzianParams:
    """Allocate a Lorentzian line container."""

    return LorentzianParams(size=size)


def create_gaussianBumpParams(size: int) -> GaussianBumpParams:
    """Allocate a broad Gaussian bump container."""

    return GaussianBumpParams(size=size)


def full_spectrum_spline(Sline: np.ndarray, data: DataParams, lines: LorentzianParams) -> None:
    """Fill Sline with the current full line model over the active band."""

    if lines.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")
    Sline[:] = sline_from_lookup(data.ncut, data.imin, data.t_obs, lines.nf, lines.nnu, lines.wdth,
                                 lines.lnumin, lines.lnumax, lines.ltemplate,
                                 lines.f, lines.a, lines.nu, lines.n)


def loglike(power: np.ndarray, sn: np.ndarray, multiplicity: float = 1.0) -> float:
    """Python wrapper around the numba likelihood kernel."""

    return float(loglike_numba(power, sn, multiplicity))


def logprior_model(priors: BayesLinePriors, sn: np.ndarray,
                   sbase: np.ndarray, sline: np.ndarray,
                   sgauss: np.ndarray) -> float:
    """Evaluate active PSD/component priors for the current model."""

    val = 0.0
    if priors.lower is not None and priors.upper is not None:
        val += float(logprior_bounds_numba(priors.lower, priors.upper, sn))
    if priors.use_context_psd_prior and priors.mean is not None and priors.sigma is not None:
        val += float(logprior_logpsd_gaussian_numba(priors.mean, priors.sigma, sn))
    if priors.use_context_component_prior:
        if priors.smooth_mean is not None and priors.smooth_sigma is not None:
            val += float(logprior_smooth_component_numba(
                priors.smooth_mean, priors.smooth_sigma, sbase
            ))
        if (priors.component_reference is not None
                and priors.line_log1p_mean is not None
                and priors.line_log1p_sigma is not None):
            val += float(logprior_ratio_component_numba(
                priors.component_reference, priors.line_log1p_mean,
                priors.line_log1p_sigma, sline
            ))
        if (priors.component_reference is not None
                and priors.gaussian_log1p_mean is not None
                and priors.gaussian_log1p_sigma is not None):
            val += float(logprior_ratio_component_numba(
                priors.component_reference, priors.gaussian_log1p_mean,
                priors.gaussian_log1p_sigma, sgauss
            ))
    return val


def has_local_psd_prior(priors: BayesLinePriors) -> bool:
    """Return True when a local proposal can update the prior incrementally."""

    return ((priors.lower is not None and priors.upper is not None)
            or (priors.use_context_psd_prior and priors.mean is not None and priors.sigma is not None)
            or priors.use_context_component_prior)


def delta_logprior_range(priors: BayesLinePriors,
                         sn_new: np.ndarray, sn_old: np.ndarray,
                         sbase_new: np.ndarray, sbase_old: np.ndarray,
                         sline_new: np.ndarray, sline_old: np.ndarray,
                         sgauss_new: np.ndarray, sgauss_old: np.ndarray,
                         ilow: int, ihigh: int) -> float:
    """Local PSD-prior difference for proposals that only touch a small band."""

    val = 0.0
    if priors.lower is not None and priors.upper is not None:
        val += float(delta_logprior_bounds_range_numba(
            priors.lower, priors.upper, sn_new, sn_old, ilow, ihigh
        ))
    if priors.use_context_psd_prior and priors.mean is not None and priors.sigma is not None:
        val += float(delta_logprior_logpsd_gaussian_range_numba(
            priors.mean, priors.sigma, sn_new, sn_old, ilow, ihigh
        ))
    if priors.use_context_component_prior:
        if priors.smooth_mean is not None and priors.smooth_sigma is not None:
            val += float(delta_logprior_smooth_component_range_numba(
                priors.smooth_mean, priors.smooth_sigma,
                sbase_new, sbase_old, ilow, ihigh
            ))
        if (priors.component_reference is not None
                and priors.line_log1p_mean is not None
                and priors.line_log1p_sigma is not None):
            val += float(delta_logprior_ratio_component_range_numba(
                priors.component_reference, priors.line_log1p_mean,
                priors.line_log1p_sigma, sline_new, sline_old, ilow, ihigh
            ))
        if (priors.component_reference is not None
                and priors.gaussian_log1p_mean is not None
                and priors.gaussian_log1p_sigma is not None):
            val += float(delta_logprior_ratio_component_range_numba(
                priors.component_reference, priors.gaussian_log1p_mean,
                priors.gaussian_log1p_sigma, sgauss_new, sgauss_old, ilow, ihigh
            ))
    return val


def rjdraw(model: float, sp: float, prange: float, pmin: float, rng: np.random.Generator) -> float:
    """Mixture draw used for reversible-jump spline amplitude proposals."""

    if rng.random() < 0.4:
        return pmin + prange * rng.random()
    return model + sp * rng.normal()


def rjden(model: float, ref: float, sp: float, prange: float) -> float:
    """Log proposal density matching rjdraw."""

    u = (ref - model) / sp
    return math.log(0.4 / prange + 0.6 / (sp * math.sqrt(TPI)) * math.exp(-0.5 * u * u))


def spline_spacing_birth_log_factor(current_n: int, flow: float, fhigh: float, dfmin: float) -> float:
    """RJ factor for the normalized ordered-knot prior with minimum spacing.

    Birth proposes a new knot uniformly over the full frequency interval and
    rejects configurations that violate dfmin.  The normalized prior volume of
    K ordered knots with fixed endpoints and gap dfmin is proportional to
    (L - (K - 1) dfmin)^(K - 2)/(K - 2)!, so birth/death moves need the volume
    ratio below.  It is unity when dfmin is zero.
    """

    if dfmin <= 0.0:
        return 0.0
    span = fhigh - flow
    if current_n < 2 or span <= 0.0:
        return -math.inf
    old_free = span - float(current_n - 1) * dfmin
    new_free = span - float(current_n) * dfmin
    if old_free <= 0.0 or new_free <= 0.0:
        return -math.inf
    return (math.log(span)
            + float(current_n - 2) * math.log(old_free)
            - float(current_n - 1) * math.log(new_free))


def draw_line_frequency(fprop: np.ndarray, data: DataParams, rng: np.random.Generator) -> float:
    """Draw a line frequency from the burn-in line proposal histogram."""

    weights = np.maximum(fprop[:data.ncut], 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return data.flow + (data.fhigh - data.flow) * rng.random()
    idx = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
    idx = min(max(idx, 0), data.ncut - 1)
    return data.flow + (idx + rng.random()) / data.t_obs


def line_proposal_log_density(f: float, fprop: np.ndarray, data: DataParams) -> float:
    """Look up the log density of a proposed line frequency."""

    idx = int((f - data.flow) * data.t_obs)
    idx = min(max(idx, 0), data.ncut - 1)
    p = max(float(fprop[idx]), np.finfo(float).tiny)
    return math.log(p)


def gaussian_bump_amplitude_bounds(priors: BayesLinePriors, sbase: np.ndarray,
                                   flow: float, t_obs: float, ncut: int,
                                   f0: float) -> Tuple[float, float]:
    """Return the local log-uniform amplitude bounds for a Gaussian bump."""

    ji = min(max(int((f0 - flow) * t_obs), 0), ncut - 1)
    if priors.lower is not None:
        lower = max(float(priors.lower[ji]), np.finfo(float).tiny)
    else:
        lower = max(float(sbase[ji]) / 10.0, np.finfo(float).tiny)
    return lower, lower * 100.0


def draw_gaussian_bump_from_prior(priors: BayesLinePriors, sbase: np.ndarray,
                                  flow: float, fhigh: float, t_obs: float,
                                  ncut: int, sigma_min: float, sigma_max: float,
                                  rng: np.random.Generator) -> Tuple[float, float, float]:
    """Draw one broad Gaussian bump from the RJ prior."""

    f0 = flow + (fhigh - flow) * rng.random()
    amin, amax = gaussian_bump_amplitude_bounds(priors, sbase, flow, t_obs, ncut, f0)
    amp = math.exp(math.log(amin) + math.log(amax / amin) * rng.random())
    sigma = math.exp(math.log(sigma_min) + math.log(sigma_max / sigma_min) * rng.random())
    return f0, amp, sigma


def bayesline_background_psd(bayesline: BayesLineParams) -> np.ndarray:
    """Return the non-line PSD background: spline plus optional Gaussian bumps."""

    if bayesline.Sbase is None:
        raise RuntimeError("Smooth PSD is not available")
    if bayesline.Sgauss is None:
        return bayesline.Sbase
    return bayesline.Sbase + bayesline.Sgauss


def BayesLineSetup(bptr: BayesLineParams, freqData: np.ndarray, fmin: float, fmax: float, deltaT: float, Tobs: float) -> None:
    """Allocate BayesLine state and attach the duration-specific line lookup."""

    bptr.maxBLLines = line_array_size_for_duration(Tobs)
    bptr.maxBLBumps = gaussian_bump_array_size_for_duration(Tobs)
    n = freqData.size
    freq = np.arange(n // 2, dtype=np.float64) / Tobs
    power = np.zeros(n // 2, dtype=np.float64)
    z = freqData.reshape(-1, 2)
    power[1:] = 2.0 * (z[1:n // 2, 0] ** 2 + z[1:n // 2, 1] ** 2)

    data = create_dataParams(freq, n, Tobs, bptr.maxBLLines, fmin, fmax)
    bptr.data = data
    bptr.freq = freq[data.imin:data.imax].copy()
    bptr.power = power[data.imin:data.imax].copy()
    bptr.spow = bptr.power.copy()
    bptr.sfreq = bptr.freq.copy()
    bptr.Sbase = robust_smooth(bptr.power)
    bptr.Sline = np.zeros_like(bptr.Sbase)
    bptr.Sgauss = np.zeros_like(bptr.Sbase)
    bptr.Snf = bptr.Sbase + bptr.Sline + bptr.Sgauss
    bptr.Sna = bptr.Snf.copy()
    bptr.fa = bptr.freq.copy()
    bptr.lines_x = create_lorentzianParams(bptr.maxBLLines)
    bptr.bumps_x = create_gaussianBumpParams(bptr.maxBLBumps)
    attach_lorentzian_lookup(bptr.lines_x, data)
    knot_step = max(1, int(round(data.fgrid * Tobs)))
    knot_idx = np.arange(0, data.ncut, knot_step, dtype=np.int64)
    if knot_idx.size == 0 or knot_idx[-1] != data.ncut - 1:
        knot_idx = np.append(knot_idx, data.ncut - 1)
    bptr.spline = SplineParams(points=bptr.freq[knot_idx], data=np.log(np.maximum(bptr.Sbase[knot_idx], np.finfo(float).tiny)))
    bptr.spline_x = SplineParams(points=bptr.spline.points.copy(), data=bptr.spline.data.copy())
    bptr.maxSplineKnots = bptr.spline.n
    _ = deltaT


def adapt_post_startup_model_caps(bayesline: BayesLineParams) -> None:
    """Set RJMCMC line/knot caps from the initialized model dimensions."""

    assert bayesline.lines_x is not None
    assert bayesline.spline_x is not None

    initialized_lines = int(bayesline.lines_x.n)
    initialized_knots = int(bayesline.spline_x.n)
    line_cap = 2 * initialized_lines
    spline_cap = 2 * initialized_knots

    bayesline.lines_x.size = line_cap
    bayesline.maxBLLines = line_cap
    bayesline.maxSplineKnots = spline_cap
    print(
        "Adaptive RJMCMC caps from startup: "
        f"max lines {line_cap} (startup {initialized_lines}), "
        f"max spline knots {spline_cap} (startup {initialized_knots})"
    )


def BayesLineInitialize(bayesline: BayesLineParams) -> None:
    """Initialize combined PSD arrays before burn-in or sampling."""

    if bayesline.Sbase is not None:
        if bayesline.Sline is None:
            bayesline.Sline = np.zeros_like(bayesline.Sbase)
        if bayesline.Sgauss is None:
            bayesline.Sgauss = np.zeros_like(bayesline.Sbase)
        bayesline.Snf = bayesline.Sbase + bayesline.Sline + bayesline.Sgauss


def set_power_from_scaled_frequency_data(bayesline: BayesLineParams, freqData: np.ndarray) -> None:
    """Set likelihood power from BayesLine's post-cleaning dt/sqrt(2) Fourier data.

    BayesLineSetup sees raw FFT packing and initializes power with 2*|d_f|^2.
    BayesLineBurnin then replaces freqData with the C/BayesWave scaled Fourier
    convention and updates power to |d_f|^2.  The --expand target stage skips
    Burnin, so it must make that replacement explicitly after setup.
    """

    if bayesline.data is None:
        raise RuntimeError("BayesLine data are not initialized")
    data = bayesline.data
    bayesline.power = (
        freqData[2 * data.imin:2 * data.imax:2] ** 2
        + freqData[2 * data.imin + 1:2 * data.imax + 1:2] ** 2
    ).copy()
    bayesline.spow = bayesline.power.copy()


def blstart(line: LorentzianParams, data: np.ndarray, residual: np.ndarray, n: int, dt: float,
            fmin: float, fstep: float, Nsp: list[int], dspline: np.ndarray,
            pspline: np.ndarray, max_lines: int, linecut: float = LINECUT,
            rng: Optional[np.random.Generator] = None,
            line_proposal: Optional[np.ndarray] = None,
            line_proposal_medspec: Optional[np.ndarray] = None,
            line_proposal_lorentzfit: Optional[np.ndarray] = None) -> None:
    """C blstart equivalent.

    The ordering is important: wavelet-clean the time series, write that cleaned
    FFT into residual/freqData, then use a separate line-subtracted copy only to
    initialize the smooth Akima spline and line model.
    """

    freqs = np.fft.rfftfreq(n, dt)
    if rng is None:
        rng = np.random.default_rng(1234)
    cleaned_time = bayesline_clean_time(data, dt)
    fft = np.fft.rfft(cleaned_time)
    # C blstart writes BayesWave-scaled residual Fourier coefficients back to
    # freqData: residual = FFT(cleaned_data) * sqrt(Tobs^2 / (2*N^2)).
    scale = dt / math.sqrt(2.0)
    scaled_fft = fft * scale
    istart = max(1, int(fmin * n * dt))
    iend = n // 2 - 1
    residual[0] = 0.0
    residual[1] = 0.0
    residual[2:n:2] = scaled_fft[1:n // 2].real
    residual[3:n + 1:2] = scaled_fft[1:n // 2].imag
    if line_proposal is not None:
        raw_power = 2.0 * np.abs(scaled_fft[:n // 2]) ** 2
        _, _, _, proposal_full = medspecspline_power(
            raw_power, 1.0 / (n * dt), n, return_line_proposal=True
        )
        ncopy = min(line_proposal.size, proposal_full.size)
        line_proposal[:ncopy] = proposal_full[:ncopy]
        if line_proposal_medspec is not None:
            ncopy = min(line_proposal_medspec.size, proposal_full.size)
            line_proposal_medspec[:ncopy] = proposal_full[:ncopy]
    # lorentzfit adds any Lpeak-refined startup candidates to this same
    # proposal map before cut_lorentz prunes weak lines.
    line_cleaned_fft = lorentzfit_four_pass(line, scaled_fft, freqs, n * dt,
                                            istart, iend, max_lines, rng, linecut,
                                            line_proposal, line_proposal_lorentzfit)
    line_cleaned_power = 2.0 * np.abs(line_cleaned_fft[:n // 2]) ** 2
    _, _, SM = medspecspline_power(line_cleaned_power, 1.0 / (n * dt), n)
    splinef, splineA = splinestart_akima(n, istart, iend, fstep, SM, freqs[:n // 2], n * dt, rng)

    knot_count = min(dspline.size, splinef.size)
    Nsp[0] = knot_count
    dspline[:knot_count] = splineA[:knot_count]
    pspline[:knot_count] = splinef[:knot_count]


def BayesLineBurnin(bayesline: BayesLineParams, timeData: np.ndarray, freqData: np.ndarray,
                    ifo: str, fprop: np.ndarray, SplineFlag: int,
                    write_start: bool = False, output_psd_scale: float = 1.0,
                    linecut: float = LINECUT,
                    startup_rng: Optional[np.random.Generator] = None) -> None:
    """Run the C burn-in/startup path and optionally write startup diagnostics."""

    del ifo, SplineFlag
    BayesLineInitialize(bayesline)
    assert bayesline.data is not None and bayesline.lines_x is not None
    data = bayesline.data
    Nsp = [0]
    dspline = np.zeros(max(4, data.ncut), dtype=np.float64)
    pspline = np.zeros_like(dspline)
    line_proposal_full = np.ones(data.n // 2, dtype=np.float64)
    line_proposal_medspec_full = np.ones(data.n // 2, dtype=np.float64)
    line_proposal_lorentzfit_full = np.ones(data.n // 2, dtype=np.float64)
    blstart(bayesline.lines_x, timeData, freqData, data.n, data.t_obs / data.n, data.fmin,
            data.fgrid, Nsp, dspline, pspline, bayesline.maxBLLines, linecut, startup_rng,
            line_proposal_full, line_proposal_medspec_full, line_proposal_lorentzfit_full)
    imax = data.imax
    imin = data.imin
    bayesline.power = (freqData[2 * imin:2 * imax:2] ** 2 +
                       freqData[2 * imin + 1:2 * imax + 1:2] ** 2).copy()
    bayesline.spow = bayesline.power.copy()
    bayesline.spline = SplineParams(points=pspline[:Nsp[0]].copy(), data=dspline[:Nsp[0]].copy())
    bayesline.spline_x = SplineParams(points=bayesline.spline.points.copy(), data=bayesline.spline.data.copy())
    bayesline.Sbase = np.exp(spline_eval_array(bayesline.spline_x.points, bayesline.spline_x.data,
                                               bayesline.freq, 1))
    full_spectrum_spline(bayesline.Sline, data, bayesline.lines_x)
    if bayesline.Sgauss is None or bayesline.Sgauss.size != bayesline.Sbase.size:
        bayesline.Sgauss = np.zeros_like(bayesline.Sbase)
    else:
        bayesline.Sgauss.fill(0.0)
    if bayesline.bumps_x is not None:
        bayesline.bumps_x.n = 0
    bayesline.Snf = bayesline.Sbase + bayesline.Sline + bayesline.Sgauss
    # The startup spline and line finder follow C medspecspline, which builds
    # models from the one-sided power 2*|d_f|^2.  The long RJMCMC state and
    # BWpsd.dat convention use the half-complex likelihood PSD units, so startup
    # model diagnostics need an extra factor of 1/2 when converted to the final
    # physical one-sided PSD scale.  The periodogram diagnostic keeps the same
    # convention as periodogram.dat: its expectation is 2*PSD.
    startup_model_scale = 0.5 * output_psd_scale
    if write_start:
        np.savetxt(
            BL_START_FILENAME,
            np.column_stack((
                bayesline.freq,
                2.0 * output_psd_scale * bayesline.power,
                startup_model_scale * bayesline.Sbase,
                startup_model_scale * bayesline.Sline,
            )),
        )
    # Proposal only: this data-derived histogram guides line birth and
    # global-relocation moves toward medspecspline excess bins. The Hastings
    # ratio uses the same fprop density, so this does not become a prior or
    # double-count the data in the target posterior.
    fprop[:data.ncut] = line_proposal_full[data.imin:data.imax]
    fsum = float(np.sum(fprop[:data.ncut]))
    if fsum > 0.0:
        fprop[:data.ncut] /= fsum
        source_norm = fsum
    else:
        fprop[:data.ncut] = 1.0 / max(1, data.ncut)
        source_norm = float(max(1, data.ncut))
    medspec_source = line_proposal_medspec_full[data.imin:data.imax] / source_norm
    lorentzfit_source = line_proposal_lorentzfit_full[data.imin:data.imax] / source_norm
    np.savetxt(
        "line_proposal.dat",
        np.column_stack((bayesline.freq, fprop[:data.ncut], medspec_source, lorentzfit_source)),
        header="frequency normalized_line_proposal medspec_source_scaled lorentzfit_source_scaled",
    )
    print(f"There are {bayesline.lines_x.n} line candidates")
    print(f"Total lines {bayesline.lines_x.n}")

    if bayesline.lines_x.n > 0:
        amin = float(np.min(bayesline.lines_x.a[:bayesline.lines_x.n]))
        amax = float(np.max(bayesline.lines_x.a[:bayesline.lines_x.n]))
    else:
        amin = float(np.min(np.maximum(bayesline.power - bayesline.Sbase, np.finfo(float).tiny)))
        amax = float(np.max(np.maximum(bayesline.power - bayesline.Sbase, np.finfo(float).tiny)))
    bayesline.priors.LAmin = max(0.1 * amin, np.finfo(float).tiny)
    bayesline.priors.LAmax = max(10.0 * amax, bayesline.priors.LAmin * 10.0)
    bayesline.priors.sigma = bayesline.Snf.copy()
    bayesline.priors.mean = bayesline.Snf.copy()
    bayesline.priors.lower = bayesline.Sbase / 10.0
    bayesline.priors.upper = (bayesline.Sbase + 100.0 * bayesline.Sline) * 10.0
    bayesline.priors.upper = np.maximum(bayesline.priors.upper, bayesline.priors.lower * 100.0)

    np.savetxt("start_psd_bw.dat", np.column_stack((bayesline.freq, startup_model_scale * bayesline.Snf)))
    np.savetxt("spline.dat", np.column_stack((bayesline.spline.points,
                                              startup_model_scale * np.exp(bayesline.spline.data))))


def BayesLineLorentzSplineMCMC(bayesline: BayesLineParams, heat: float, steps: int,
                               priorFlag: int, dan: list[float], fprop: np.ndarray,
                               SplineFlag: int,
                               psd_samples: Optional[np.ndarray] = None,
                               spline_samples: Optional[np.ndarray] = None,
                               gaussian_samples: Optional[np.ndarray] = None,
                               count_trace: Optional[np.ndarray] = None,
                               line_trim: bool = False,
                               gaussian_bumps: bool = False,
                               debug_acceptance: bool = False) -> int:
    """Main reversible-jump sampler for spline knots, Lorentzians, and optional Gaussian bumps."""

    assert bayesline.data is not None
    assert bayesline.lines_x is not None
    assert bayesline.spline_x is not None
    assert bayesline.power is not None
    assert bayesline.freq is not None
    assert bayesline.Sbase is not None
    assert bayesline.Sline is not None
    assert bayesline.Snf is not None

    rng = bayesline.rng
    data = bayesline.data
    lines = bayesline.lines_x
    bumps = bayesline.bumps_x
    spline = bayesline.spline_x
    freq = bayesline.freq
    power = bayesline.power
    priors = bayesline.priors
    likelihood_multiplicity = max(float(bayesline.power_multiplicity), np.finfo(float).tiny)

    flow = data.flow
    fhigh = data.fhigh
    t_obs = data.t_obs
    ncut = data.ncut
    tmax = lines.size
    spline_tmax = bayesline.maxSplineKnots if bayesline.maxSplineKnots > 0 else spline.n
    gaussian_enabled = bool(gaussian_bumps)
    if gaussian_enabled and bumps is None:
        raise RuntimeError("Gaussian bumps were requested but no bump container is allocated")
    sigma_min = float(bayesline.gaussianSigmaMin)
    sigma_max = float(bayesline.gaussianSigmaMax)
    if gaussian_enabled and (sigma_min <= 0.0 or sigma_max <= sigma_min):
        raise ValueError("Gaussian bump sigma bounds must satisfy 0 < sigma_min < sigma_max")
    if lines.ltemplate is None:
        raise RuntimeError("Lorentzian lookup table has not been loaded")
    numin = lines.numin
    numax = lines.numax
    lnumin = math.log(numin)
    lnumax = math.log(numax)
    lAmin = math.log(max(priors.LAmin, np.finfo(float).tiny))
    lAmax = math.log(max(priors.LAmax, priors.LAmin * 10.0))

    sbase = np.exp(spline_eval_array(spline.points, spline.data, freq, SplineFlag))
    sline = sline_from_lookup(ncut, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                              lines.lnumin, lines.lnumax, lines.ltemplate,
                              lines.f, lines.a, lines.nu, lines.n)
    if gaussian_enabled and bumps is not None:
        sgauss = gaussian_bumps_from_params(freq, bumps)
    else:
        sgauss = np.zeros(ncut, dtype=np.float64)
        if bumps is not None and not gaussian_enabled:
            bumps.n = 0
    snx = sbase + sline + sgauss
    logLx = 1.0 if bayesline.constantLogLFlag else loglike(power, snx, likelihood_multiplicity)
    logPsx = logprior_model(priors, snx, sbase, sline, sgauss) if priorFlag == 1 else 0.0

    ac = np.zeros(8, dtype=np.int64)
    cc = np.ones(8, dtype=np.int64)
    proposal_attempts = np.zeros(8, dtype=np.int64)
    proposal_valid = np.zeros(8, dtype=np.int64)
    proposal_invalid = np.zeros(8, dtype=np.int64)
    fweights = np.maximum(fprop[:ncut], np.finfo(float).tiny)
    if np.sum(fweights) <= 0.0:
        fweights[:] = 1.0
    fprop[:ncut] = fweights / np.sum(fweights)
    shiftval = max(1.0 / t_obs, min(DFMIN, DFMIN * rng.random()))
    sample_count = 0
    sample_target = 0 if psd_samples is None else int(psd_samples.shape[0])
    sample_start = steps // 2
    post_burnin_steps = max(1, steps - sample_start)
    sample_stride = max(1, post_burnin_steps // sample_target) if sample_target > 0 else 1

    for mc in range(steps):
        logpx = logpy = logqx = logqy = 0.0
        lbl = 0
        accepted = False
        check = False
        typ = -1

        prop_points = spline.points
        prop_sdata = spline.data
        prop_nlines = lines.n
        prop_f = lines.f[:lines.n]
        prop_a = lines.a[:lines.n]
        prop_nu = lines.nu[:lines.n]
        prop_nbump = 0 if bumps is None else bumps.n
        prop_bf = np.empty(0, dtype=np.float64) if bumps is None else bumps.f[:bumps.n]
        prop_ba = np.empty(0, dtype=np.float64) if bumps is None else bumps.a[:bumps.n]
        prop_bs = np.empty(0, dtype=np.float64) if bumps is None else bumps.sigma[:bumps.n]
        sn_prop = None
        sbase_prop = sbase
        sline_prop = sline
        sgauss_prop = sgauss
        logLy_local = None
        spline_dimension_log_factor = 0.0
        proposal_ilow = 0
        proposal_ihigh = ncut
        spline_changed_index = -1

        if gaussian_enabled and bumps is not None and bumps.size > 0:
            proposal_selector = rng.random()
            choose_spline = proposal_selector < (1.0 / 3.0)
            choose_bump = proposal_selector >= (2.0 / 3.0)
        else:
            choose_spline = rng.random() > 0.5
            choose_bump = False

        if choose_spline:
            alpha = rng.random()
            if alpha > 0.5:
                lbl = 0
                if rng.random() > 0.5:
                    typ = 5
                    if spline.n >= 7 and spline.n < spline_tmax:
                        newfreq = flow + (fhigh - flow) * rng.random()
                        mdl = spline_eval_one(spline.points, spline.data, newfreq, SplineFlag)
                        ji = min(max(int((newfreq - flow) * t_obs), 0), ncut - 1)
                        lower = max(priors.lower[ji], np.finfo(float).tiny) if priors.lower is not None else max(sbase[ji] / 10.0, np.finfo(float).tiny)
                        sp = abs(math.log(lower)) * 1.0e-3
                        sp = max(sp, 1.0e-6)
                        prange = math.log(lower * 100.0) - math.log(lower)
                        newval = rjdraw(mdl, sp, prange, math.log(lower), rng)
                        spline_dimension_log_factor = spline_spacing_birth_log_factor(
                            spline.n, flow, fhigh, DFMIN
                        )
                        if not math.isfinite(spline_dimension_log_factor):
                            check = True
                        prop_points = np.append(spline.points, newfreq)
                        prop_sdata = np.append(spline.data, newval)
                        order = np.argsort(prop_points)
                        prop_points = prop_points[order]
                        prop_sdata = prop_sdata[order]
                        spline_changed_index = int(np.searchsorted(prop_points, newfreq, side="left"))
                        if spline_changed_index >= prop_points.size:
                            spline_changed_index = prop_points.size - 1
                        logqy = rjden(mdl, newval, sp, prange)
                        logpy = -math.log(prange)
                    else:
                        check = True
                else:
                    typ = 6
                    if spline.n > 7:
                        ki = int(rng.integers(1, spline.n - 1))
                        newfreq = float(spline.points[ki])
                        prop_points = np.delete(spline.points, ki)
                        prop_sdata = np.delete(spline.data, ki)
                        spline_changed_index = min(ki, prop_points.size - 1)
                        mdl = spline_eval_one(prop_points, prop_sdata, newfreq, SplineFlag)
                        ji = min(max(int((newfreq - flow) * t_obs), 0), ncut - 1)
                        lower = max(priors.lower[ji], np.finfo(float).tiny) if priors.lower is not None else max(sbase[ji] / 10.0, np.finfo(float).tiny)
                        sp = max(abs(math.log(lower)) * 1.0e-3, 1.0e-6)
                        prange = math.log(lower * 100.0) - math.log(lower)
                        reverse_birth_factor = spline_spacing_birth_log_factor(
                            spline.n - 1, flow, fhigh, DFMIN
                        )
                        if not math.isfinite(reverse_birth_factor):
                            check = True
                        spline_dimension_log_factor = -reverse_birth_factor
                        logqx = rjden(mdl, float(spline.data[ki]), sp, prange)
                        logpx = -math.log(prange)
                    else:
                        check = True
            elif alpha > 0.3 and spline.n > 2:
                lbl = 1
                typ = 7
                ki = int(rng.integers(1, spline.n - 1))
                prop_points = spline.points.copy()
                prop_sdata = spline.data.copy()
                newfreq = prop_points[ki] + (shiftval if rng.random() > 0.5 else -shiftval)
                if newfreq <= prop_points[ki - 1] or newfreq >= prop_points[ki + 1]:
                    check = True
                else:
                    prop_points[ki] = newfreq
                    order = np.argsort(prop_points)
                    prop_points = prop_points[order]
                    prop_sdata = prop_sdata[order]
                    spline_changed_index = int(np.searchsorted(prop_points, newfreq, side="left"))
                    if spline_changed_index >= prop_points.size:
                        spline_changed_index = prop_points.size - 1
            else:
                lbl = 2
                typ = 4
                ii = int(rng.integers(0, spline.n))
                prop_points = spline.points.copy()
                prop_sdata = spline.data.copy()
                e1 = 0.0005
                a = rng.random()
                if a > 0.8:
                    e1 = 0.002
                elif a > 0.6:
                    e1 = 0.005
                elif a > 0.4:
                    e1 = 0.05
                if rng.random() > 0.5:
                    prop_sdata[ii] += rng.normal(0.0, e1)
                    if 0 < ii < spline.n - 1:
                        prop_points[ii] += rng.normal(0.0, e1)
                    spline_changed_index = ii
                else:
                    if 0 < ii < spline.n - 1:
                        newfreq = flow + (fhigh - flow) * rng.random()
                        if newfreq <= prop_points[ii - 1] or newfreq >= prop_points[ii + 1]:
                            check = True
                        else:
                            prop_points[ii] = newfreq
                    if not check:
                        newfreq = float(prop_points[ii])
                        ji = min(max(int((prop_points[ii] - flow) * t_obs), 0), ncut - 1)
                        lower = max(priors.lower[ji], np.finfo(float).tiny) if priors.lower is not None else max(sbase[ji] / 10.0, np.finfo(float).tiny)
                        prop_sdata[ii] = math.log(lower) + math.log(100.0) * rng.random()
                        spline_changed_index = ii

            if not check:
                if np.any(np.diff(prop_points) < DFMIN):
                    check = True
                if prop_points[0] < flow or prop_points[-1] > fhigh:
                    check = True
                if priors.lower is not None:
                    idx = np.clip(((prop_points - flow) * t_obs).astype(np.int64), 0, ncut - 1)
                    low = np.log(np.maximum(priors.lower[idx], np.finfo(float).tiny))
                    high = np.log(np.maximum(priors.lower[idx] * 100.0, np.finfo(float).tiny))
                    if np.any(prop_sdata < low) or np.any(prop_sdata > high):
                        check = True
                if not check:
                    fixed_psd = sline + sgauss if gaussian_enabled else sline
                    sbase_prop, sn_prop, logLy_local, proposal_ilow, proposal_ihigh = spline_proposal_update(
                        prop_points, prop_sdata, freq, data, sbase, fixed_psd, snx,
                        power, logLx, SplineFlag, spline_changed_index,
                        compute_loglike=not bayesline.constantLogLFlag,
                        multiplicity=likelihood_multiplicity
                    )

        elif choose_bump:
            assert bumps is not None
            if rng.random() > 0.5:
                lbl = 6
                if rng.random() > 0.5:
                    typ = 8
                    if bumps.n < bumps.size:
                        nf, na, ns = draw_gaussian_bump_from_prior(
                            priors, sbase, flow, fhigh, t_obs, ncut, sigma_min, sigma_max, rng
                        )
                        prop_nbump = bumps.n + 1
                        prop_bf = np.empty(prop_nbump, dtype=np.float64)
                        prop_ba = np.empty(prop_nbump, dtype=np.float64)
                        prop_bs = np.empty(prop_nbump, dtype=np.float64)
                        prop_bf[:bumps.n] = bumps.f[:bumps.n]
                        prop_ba[:bumps.n] = bumps.a[:bumps.n]
                        prop_bs[:bumps.n] = bumps.sigma[:bumps.n]
                        prop_bf[bumps.n] = nf
                        prop_ba[bumps.n] = na
                        prop_bs[bumps.n] = ns
                        sn_prop = snx.copy()
                        sgauss_prop = sgauss.copy()
                        ilow, ihigh = add_gaussian_bump_band_inplace(sn_prop, freq, nf, na, ns, 1.0)
                        add_gaussian_bump_band_inplace(sgauss_prop, freq, nf, na, ns, 1.0)
                        proposal_ilow = ilow
                        proposal_ihigh = ihigh
                        if ihigh <= ilow:
                            check = True
                        elif not bayesline.constantLogLFlag:
                            logLy_local = logLx + delta_loglike_range(
                                power, sn_prop, snx, ilow, ihigh, likelihood_multiplicity
                            )
                    else:
                        check = True
                else:
                    typ = 9
                    if bumps.n > 0:
                        ki = int(rng.integers(0, bumps.n))
                        of = float(bumps.f[ki])
                        oa = float(bumps.a[ki])
                        os = float(bumps.sigma[ki])
                        prop_nbump = bumps.n - 1
                        prop_bf = np.empty(prop_nbump, dtype=np.float64)
                        prop_ba = np.empty(prop_nbump, dtype=np.float64)
                        prop_bs = np.empty(prop_nbump, dtype=np.float64)
                        if ki > 0:
                            prop_bf[:ki] = bumps.f[:ki]
                            prop_ba[:ki] = bumps.a[:ki]
                            prop_bs[:ki] = bumps.sigma[:ki]
                        if ki < prop_nbump:
                            prop_bf[ki:] = bumps.f[ki + 1:bumps.n]
                            prop_ba[ki:] = bumps.a[ki + 1:bumps.n]
                            prop_bs[ki:] = bumps.sigma[ki + 1:bumps.n]
                        sn_prop = snx.copy()
                        sgauss_prop = sgauss.copy()
                        ilow, ihigh = add_gaussian_bump_band_inplace(sn_prop, freq, of, oa, os, -1.0)
                        add_gaussian_bump_band_inplace(sgauss_prop, freq, of, oa, os, -1.0)
                        proposal_ilow = ilow
                        proposal_ihigh = ihigh
                        if ihigh <= ilow:
                            check = True
                        elif not bayesline.constantLogLFlag:
                            logLy_local = logLx + delta_loglike_range(
                                power, sn_prop, snx, ilow, ihigh, likelihood_multiplicity
                            )
                    else:
                        check = True
            else:
                lbl = 7
                typ = 10
                if bumps.n > 0:
                    prop_bf = bumps.f[:bumps.n].copy()
                    prop_ba = bumps.a[:bumps.n].copy()
                    prop_bs = bumps.sigma[:bumps.n].copy()
                    ii = int(rng.integers(0, bumps.n))
                    oldf = float(prop_bf[ii])
                    olda = float(prop_ba[ii])
                    olds = float(prop_bs[ii])
                    if rng.random() > 0.8:
                        newf, newa, news = draw_gaussian_bump_from_prior(
                            priors, sbase, flow, fhigh, t_obs, ncut, sigma_min, sigma_max, rng
                        )
                    else:
                        beta = rng.random()
                        if beta > 0.9:
                            scale = 1.0
                        elif beta > 0.6:
                            scale = 0.4
                        elif beta > 0.3:
                            scale = 0.15
                        else:
                            scale = 0.05
                        newf = oldf + rng.normal(0.0, max(0.25 / t_obs, 0.5 * olds) * scale)
                        newa = olda * math.exp(rng.normal(0.0, scale))
                        news = olds * math.exp(rng.normal(0.0, 0.5 * scale))
                    if newf < flow or newf > fhigh or news < sigma_min or news > sigma_max:
                        check = True
                    if not check:
                        amin, amax = gaussian_bump_amplitude_bounds(priors, sbase, flow, t_obs, ncut, newf)
                        if newa < amin or newa > amax:
                            check = True
                    if not check:
                        prop_bf[ii] = newf
                        prop_ba[ii] = newa
                        prop_bs[ii] = news
                        sn_prop = snx.copy()
                        sgauss_prop = sgauss.copy()
                        ilowx, ihighx = add_gaussian_bump_band_inplace(sn_prop, freq, oldf, olda, olds, -1.0)
                        add_gaussian_bump_band_inplace(sgauss_prop, freq, oldf, olda, olds, -1.0)
                        ilowy, ihighy = add_gaussian_bump_band_inplace(sn_prop, freq, newf, newa, news, 1.0)
                        add_gaussian_bump_band_inplace(sgauss_prop, freq, newf, newa, news, 1.0)
                        if ihighx <= ilowx:
                            proposal_ilow = ilowy
                            proposal_ihigh = ihighy
                        elif ihighy <= ilowy:
                            proposal_ilow = ilowx
                            proposal_ihigh = ihighx
                        else:
                            proposal_ilow = min(ilowx, ilowy)
                            proposal_ihigh = max(ihighx, ihighy)
                        if proposal_ihigh <= proposal_ilow:
                            check = True
                        elif not bayesline.constantLogLFlag:
                            logLy_local = logLx + delta_loglike_range(
                                power, sn_prop, snx, proposal_ilow, proposal_ihigh, likelihood_multiplicity
                            )
                else:
                    check = True

        else:
            if tmax == 0:
                check = True
            elif rng.random() > 0.5:
                lbl = 3
                if rng.random() > 0.5:
                    typ = 2
                    if lines.n < tmax:
                        nf = draw_line_frequency(fprop, data, rng)
                        na = math.exp(lAmin + (lAmax - lAmin) * rng.random())
                        nn = math.exp(lnumin + (lnumax - lnumin) * rng.random())
                        prop_nlines = lines.n + 1
                        prop_f = np.empty(prop_nlines, dtype=np.float64)
                        prop_a = np.empty(prop_nlines, dtype=np.float64)
                        prop_nu = np.empty(prop_nlines, dtype=np.float64)
                        prop_f[:lines.n] = lines.f[:lines.n]
                        prop_a[:lines.n] = lines.a[:lines.n]
                        prop_nu[:lines.n] = lines.nu[:lines.n]
                        prop_f[lines.n] = nf
                        prop_a[lines.n] = na
                        prop_nu[lines.n] = nn
                        logqy = line_proposal_log_density(nf, fprop, data)
                        logpy = -math.log(float(ncut))
                        sn_prop = snx.copy()
                        sline_prop = sline.copy()
                        ilow, ihigh = add_lookup_line_band_inplace(
                            sn_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate, nf, na, nn, 1.0
                        )
                        add_lookup_line_band_inplace(
                            sline_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate, nf, na, nn, 1.0
                        )
                        proposal_ilow = ilow
                        proposal_ihigh = ihigh
                        if not bayesline.constantLogLFlag:
                            logLy_local = logLx + delta_loglike_range(
                                power, sn_prop, snx, ilow, ihigh, likelihood_multiplicity
                            )
                    else:
                        check = True
                else:
                    typ = 3
                    if lines.n > 0:
                        ki = int(rng.integers(0, lines.n))
                        of = prop_f[ki]
                        oa = prop_a[ki]
                        on = prop_nu[ki]
                        logqx = line_proposal_log_density(of, fprop, data)
                        logpx = -math.log(float(ncut))
                        prop_nlines = lines.n - 1
                        prop_f = np.empty(prop_nlines, dtype=np.float64)
                        prop_a = np.empty(prop_nlines, dtype=np.float64)
                        prop_nu = np.empty(prop_nlines, dtype=np.float64)
                        if ki > 0:
                            prop_f[:ki] = lines.f[:ki]
                            prop_a[:ki] = lines.a[:ki]
                            prop_nu[:ki] = lines.nu[:ki]
                        if ki < prop_nlines:
                            prop_f[ki:] = lines.f[ki + 1:lines.n]
                            prop_a[ki:] = lines.a[ki + 1:lines.n]
                            prop_nu[ki:] = lines.nu[ki + 1:lines.n]
                        sn_prop = snx.copy()
                        sline_prop = sline.copy()
                        ilow, ihigh = add_lookup_line_band_inplace(
                            sn_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate, of, oa, on, -1.0
                        )
                        add_lookup_line_band_inplace(
                            sline_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate, of, oa, on, -1.0
                        )
                        proposal_ilow = ilow
                        proposal_ihigh = ihigh
                        if not bayesline.constantLogLFlag:
                            logLy_local = logLx + delta_loglike_range(
                                power, sn_prop, snx, ilow, ihigh, likelihood_multiplicity
                            )
                    else:
                        check = True
            else:
                lbl = 5
                if lines.n > 0:
                    prop_f = lines.f[:lines.n].copy()
                    prop_a = lines.a[:lines.n].copy()
                    prop_nu = lines.nu[:lines.n].copy()
                    ii = int(rng.integers(0, lines.n))
                    oldf = prop_f[ii]
                    olda = prop_a[ii]
                    oldnu = prop_nu[ii]
                    if rng.random() > 0.8:
                        lbl = 4
                        typ = 0
                        newf = draw_line_frequency(fprop, data, rng)
                        prop_f[ii] = newf
                        prop_nu[ii] = math.exp(lnumin + (lnumax - lnumin) * rng.random())
                        prop_a[ii] = math.exp(lAmin + (lAmax - lAmin) * rng.random())
                        logqy = line_proposal_log_density(newf, fprop, data)
                        logpy = line_proposal_log_density(oldf, fprop, data)
                    else:
                        typ = 1
                        a = rng.random()
                        if a > 0.9:
                            beta = 10.0
                        elif a > 0.6:
                            beta = 1.0
                        elif a > 0.3:
                            beta = 0.1
                        else:
                            beta = 0.01
                        prop_a[ii] *= math.exp(rng.normal(0.0, beta * 0.5))
                        prop_nu[ii] += rng.normal(0.0, beta * 1.0e-3)
                        prop_f[ii] += rng.normal(0.0, beta * 0.1 / t_obs)
                    if prop_a[ii] < priors.LAmin or prop_a[ii] > priors.LAmax:
                        check = True
                    if prop_f[ii] < flow or prop_f[ii] > fhigh:
                        check = True
                    if prop_nu[ii] < numin or prop_nu[ii] > numax:
                        check = True
                    if not check:
                        sn_prop = snx.copy()
                        sline_prop = sline.copy()
                        ilowx, ihighx = add_lookup_line_band_inplace(
                            sn_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate, oldf, olda, oldnu, -1.0
                        )
                        add_lookup_line_band_inplace(
                            sline_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate, oldf, olda, oldnu, -1.0
                        )
                        ilowy, ihighy = add_lookup_line_band_inplace(
                            sn_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate,
                            prop_f[ii], prop_a[ii], prop_nu[ii], 1.0
                        )
                        add_lookup_line_band_inplace(
                            sline_prop, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                            lines.lnumin, lines.lnumax, lines.ltemplate,
                            prop_f[ii], prop_a[ii], prop_nu[ii], 1.0
                        )
                        ilow = min(ilowx, ilowy)
                        ihigh = max(ihighx, ihighy)
                        proposal_ilow = ilow
                        proposal_ihigh = ihigh
                        if not bayesline.constantLogLFlag:
                            logLy_local = logLx + delta_loglike_range(
                                power, sn_prop, snx, ilow, ihigh, likelihood_multiplicity
                            )
                else:
                    check = True

        proposal_positive = False
        if not check and sn_prop is not None:
            proposal_positive = positive_range_numba(sn_prop, proposal_ilow, proposal_ihigh)
        if typ >= 0 and 0 <= lbl < proposal_attempts.size:
            proposal_attempts[lbl] += 1
            if not check and sn_prop is not None and proposal_positive:
                proposal_valid[lbl] += 1
            else:
                proposal_invalid[lbl] += 1

        if not check and sn_prop is not None and proposal_positive:
            if bayesline.constantLogLFlag:
                logLy = 1.0
            elif logLy_local is not None:
                logLy = logLy_local
            else:
                logLy = loglike(power, sn_prop, likelihood_multiplicity)
            if priorFlag == 1:
                if has_local_psd_prior(priors) and logLy_local is not None:
                    logPsy = logPsx + delta_logprior_range(
                        priors, sn_prop, snx,
                        sbase_prop, sbase, sline_prop, sline, sgauss_prop, sgauss,
                        proposal_ilow, proposal_ihigh
                    )
                else:
                    logPsy = logprior_model(priors, sn_prop, sbase_prop, sline_prop, sgauss_prop)
            else:
                logPsy = 0.0
            if line_trim:
                logpy -= LINE_COUNT_PRIOR_ZETA * float(prop_nlines)
                logpx -= LINE_COUNT_PRIOR_ZETA * float(lines.n)
            logH = ((logLy - logLx) * heat + logpy - logqy - logpx + logqx
                    + spline_dimension_log_factor)
            if priorFlag != 0:
                logH += logPsy - logPsx
            cc[lbl] += 1
            if math.log(max(rng.random(), np.finfo(float).tiny)) < logH:
                accepted = True
                ac[lbl] += 1
                logLx = logLy
                logPsx = logPsy
                snx = sn_prop
                sbase = sbase_prop
                sline = sline_prop
                sgauss = sgauss_prop
                if typ >= 8:
                    assert bumps is not None
                    bumps.n = prop_nbump
                    if prop_nbump > 0:
                        bumps.f[:prop_nbump] = prop_bf
                        bumps.a[:prop_nbump] = prop_ba
                        bumps.sigma[:prop_nbump] = prop_bs
                elif typ >= 4:
                    spline.points = prop_points.copy()
                    spline.data = prop_sdata.copy()
                else:
                    lines.n = prop_nlines
                    lines.f[:prop_nlines] = prop_f
                    lines.a[:prop_nlines] = prop_a
                    lines.nu[:prop_nlines] = prop_nu
                    lines.q[:prop_nlines] = lines.f[:prop_nlines] / np.maximum(lines.nu[:prop_nlines], np.finfo(float).tiny)

        if count_trace is not None and mc < count_trace.shape[0]:
            count_trace[mc, 0] = mc
            count_trace[mc, 1] = lines.n
            count_trace[mc, 2] = spline.n
            if count_trace.shape[1] > 3:
                count_trace[mc, 3] = 0 if bumps is None else bumps.n

        if sample_count < sample_target and mc >= sample_start and (mc - sample_start) % sample_stride == 0:
            psd_samples[sample_count, :] = snx
            if spline_samples is not None:
                spline_samples[sample_count, :] = sbase
            if gaussian_samples is not None:
                gaussian_samples[sample_count, :] = sgauss
            sample_count += 1

        if mc % 1000 == 0:
            rates = [ac[i] / cc[i] for i in range(8)]
            bump_count = 0 if bumps is None else bumps.n
            if gaussian_enabled:
                print(f"{mc} {lines.n} {spline.n} {bump_count} {logLx:.6e} " + " ".join(f"{r:.6f}" for r in rates))
            else:
                print(f"{mc} {lines.n} {spline.n} {bump_count} {logLx:.6e} " + " ".join(f"{r:.6f}" for r in rates[:6]))

    bayesline.spline_x = SplineParams(points=spline.points.copy(), data=spline.data.copy())
    if bayesline.spline is not None:
        bayesline.spline = SplineParams(points=spline.points.copy(), data=spline.data.copy())
    # Rebuild the final PSD from the accepted parameters, matching the C
    # end-of-chain Akima interpolation/full line-model refresh.
    final_sbase = np.exp(spline_eval_array(spline.points, spline.data, freq, SplineFlag))
    final_sline = sline_from_lookup(ncut, data.imin, t_obs, lines.nf, lines.nnu, lines.wdth,
                                    lines.lnumin, lines.lnumax, lines.ltemplate,
                                    lines.f, lines.a, lines.nu, lines.n)
    if gaussian_enabled and bumps is not None:
        final_sgauss = gaussian_bumps_from_params(freq, bumps)
    else:
        final_sgauss = np.zeros(ncut, dtype=np.float64)
    final_snf = final_sbase + final_sline + final_sgauss
    bayesline.Sbase = final_sbase
    bayesline.Sline = final_sline
    bayesline.Sgauss = final_sgauss
    bayesline.Snf = final_snf
    ratio = power / np.maximum(final_snf, np.finfo(float).tiny)
    dan[0] = float(np.max(ratio))
    if debug_acceptance:
        print("MCMC proposal diagnostics by label")
        for ii in range(8):
            print(
                f"  label {ii}: attempts {proposal_attempts[ii]} "
                f"valid {proposal_valid[ii]} invalid {proposal_invalid[ii]} "
                f"accepted {ac[ii]}"
            )
    return sample_count


def BayesLineRJMCMC(bayesline: BayesLineParams, freqData: np.ndarray, psd: np.ndarray,
                    invpsd: np.ndarray, splinePSD: np.ndarray, N: int, cycle: int,
                    beta: float, priorFlag: int, fprop: np.ndarray, SplineFlag: int,
                    psd_samples: Optional[np.ndarray] = None,
                    spline_samples: Optional[np.ndarray] = None,
                    gaussian_samples: Optional[np.ndarray] = None,
                    count_trace: Optional[np.ndarray] = None,
                    line_trim: bool = False,
                    gaussian_bumps: bool = False,
                    debug_acceptance: bool = False) -> int:
    """Public C-style RJMCMC wrapper that returns full-band PSD arrays."""

    del freqData
    dan = [0.0]
    collected = BayesLineLorentzSplineMCMC(
        bayesline, beta, cycle, priorFlag, dan, fprop, SplineFlag,
        psd_samples=psd_samples, spline_samples=spline_samples,
        gaussian_samples=gaussian_samples,
        count_trace=count_trace,
        line_trim=line_trim,
        gaussian_bumps=gaussian_bumps,
        debug_acceptance=debug_acceptance
    )
    output_state = bayesline
    assert output_state.data is not None and output_state.Snf is not None and output_state.Sbase is not None
    data = output_state.data
    psd.fill(0.0)
    splinePSD.fill(0.0)
    psd[data.imin:data.imax] = output_state.Snf
    splinePSD[data.imin:data.imax] = output_state.Sbase
    psd[:data.imin] = 1.0
    splinePSD[:data.imin] = 1.0
    psd[data.imax:N // 2] = 1.0
    splinePSD[data.imax:N // 2] = 1.0
    invpsd[:] = 1.0 / np.maximum(psd, np.finfo(float).tiny)
    return collected


def _require_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("data provided does not lead to 2^n samples")


def valid_power_of_two_frequency(value: float) -> Tuple[bool, int]:
    nearest = int(round(value))
    if not math.isclose(value, float(nearest), rel_tol=0.0, abs_tol=1.0e-9):
        return False, nearest
    if nearest <= 0 or (nearest & (nearest - 1)) != 0:
        return False, nearest
    return True, nearest


def segment_bounds(trigger_time: float, duration: float) -> Tuple[float, float]:
    end = trigger_time + TRIGGER_OFFSET_FROM_END
    start = end - duration
    return start, end


def expanded_fetch_bounds(trigger_time: float, analysis_duration: float,
                          expand: int) -> Tuple[float, float]:
    """Return the union needed for the target segment and symmetric context."""

    analysis_start, analysis_end = segment_bounds(trigger_time, analysis_duration)
    before_chunks = expand // 2
    after_chunks = expand - before_chunks
    context_start = analysis_start - analysis_duration * float(before_chunks)
    context_end = analysis_end + analysis_duration * float(after_chunks)
    return min(analysis_start, context_start), max(analysis_end, context_end)


def is_power_of_two_int(value: int) -> bool:
    """Return True when ``value`` is a positive power of two."""

    return value > 0 and (value & (value - 1)) == 0


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


def crop_regular_arrays(times: np.ndarray, strain: np.ndarray, start: float,
                        end: float, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
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
                          end: float) -> Tuple[np.ndarray, np.ndarray]:
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
    if times.size < 2:
        raise RuntimeError("not enough samples to resample")
    if times.size != strain.size:
        raise RuntimeError("GWPy returned mismatched time and strain arrays")

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


def fetch_ligo_data_bounds(start: float, end: float, ifo: str, source: str,
                           channel: Optional[str], output_file: Optional[str],
                           target_rate: float, padding: float, resampler: str,
                           trigger_time: Optional[float] = None) -> str:
    """Fetch, resample, crop to explicit bounds, and save strain data for BWtest."""

    outfile = output_file if output_file is not None else f"{ifo}.txt"
    fetch_start = start - padding
    fetch_end = end + padding

    print(f"Fetching {ifo} data from {fetch_start:.6f} to {fetch_end:.6f}")
    if trigger_time is not None:
        print(f"Trigger time {trigger_time:.6f}")
    print(f"Target sample rate after {resampler} resampling = {target_rate:g} Hz")

    try:
        if source == "open":
            from gwosc.timeline import get_segments
            from gwpy.timeseries import TimeSeries

            segments = get_segments(f"{ifo}_DATA", int(math.floor(fetch_start)), int(math.ceil(fetch_end)))
            if len(segments) == 0:
                raise RuntimeError(f"no {ifo} open-data segment covers the requested time")
            series = TimeSeries.fetch_open_data(ifo, fetch_start, fetch_end)
        else:
            from gwpy.timeseries import TimeSeries

            channel_name = channel if channel is not None else f"{ifo}:GDS-CALIB_STRAIN"
            print(f"Using frame channel {channel_name}")
            series = TimeSeries.get(channel_name, fetch_start, fetch_end)
    except ImportError as exc:
        raise RuntimeError("gwpy/gwosc are required for trigger-time fetching") from exc

    print(f"Original sample rate: {series.sample_rate}")
    original_rate = gwpy_sample_rate_hz(series)
    if math.isclose(original_rate, target_rate, rel_tol=1.0e-6, abs_tol=1.0e-6):
        print("Sample rate already matches target; skipping resampling")
        cropped = series.crop(start, end)
        print(f"Cropped span: {cropped.span}")
        times = np.asarray(cropped.times.value, dtype=np.float64)
        strain = np.asarray(cropped.value, dtype=np.float64)
        if times.size != strain.size:
            raise RuntimeError("GWPy returned mismatched time and strain arrays")
    elif resampler == "lal":
        print("Resampling with LALSuite ResampleREAL8TimeSeries")
        times, strain = lal_resample_and_crop(series, target_rate, start, end)
    elif resampler == "gwpy":
        print("Resampling with GWPy")
        resampled = series.resample(target_rate)
        print(f"Resampled sample rate: {resampled.sample_rate}")
        cropped = resampled.crop(start, end)
        print(f"Cropped span: {cropped.span}")
        times = np.asarray(cropped.times.value, dtype=np.float64)
        strain = np.asarray(cropped.value, dtype=np.float64)
        if times.size != strain.size:
            raise RuntimeError("GWPy returned mismatched time and strain arrays")
    else:
        raise RuntimeError(f"unknown resampler {resampler!r}")

    print(f"Output samples: {strain.size}")
    np.savetxt(outfile, np.column_stack((times, strain)), fmt="%.18e")
    print(f"Fetched data written to {outfile}")
    return outfile


def fetch_ligo_data(trigger_time: float, duration: float, ifo: str, source: str,
                    channel: Optional[str], output_file: Optional[str],
                    target_rate: float, padding: float, resampler: str) -> str:
    """Fetch, resample, crop, and save the standard trigger-offset analysis segment."""

    start, end = segment_bounds(trigger_time, duration)
    print(f"Trigger time {trigger_time:.6f} is {TRIGGER_OFFSET_FROM_END:.1f} seconds before segment end")
    return fetch_ligo_data_bounds(
        start, end, ifo, source, channel, output_file,
        target_rate, padding, resampler, trigger_time=trigger_time
    )


def read_frame(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a two-column strain file: GPS time and strain.

    The full file is allowed to be non-power-of-two because segmented modes can
    contain context plus target data. Each FFT segment is checked after cropping.
    """

    arr = np.loadtxt(filename, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{filename} must contain at least two columns")
    time = arr[:, 0].copy()
    strain = arr[:, 1].copy()
    return time, strain


def infer_regular_dt(times: np.ndarray) -> float:
    """Infer the sample spacing from a regular GPS-time array."""

    if times.size < 2:
        raise ValueError("not enough time samples")
    diffs = np.diff(times)
    dt = float(np.median(diffs))
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("input times are not increasing")
    return dt


def crop_arrays_by_time(times: np.ndarray, strain: np.ndarray,
                        start: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """Crop regularly sampled arrays by GPS start time and duration."""

    dt = infer_regular_dt(times)
    sample_count = int(round(duration / dt))
    if sample_count <= 0:
        raise ValueError("requested crop has no samples")
    start_index = int(round((start - times[0]) / dt))
    end_index = start_index + sample_count
    if start_index < 0 or end_index > strain.size:
        raise ValueError(
            f"not enough data for requested segment [{start:.6f}, {start + duration:.6f}]"
        )
    out_times = times[start_index:end_index].copy()
    out_strain = strain[start_index:end_index].copy()
    _require_power_of_two(out_strain.size)
    return out_times, out_strain


def prepare_frequency_segment(times: np.ndarray, strain: np.ndarray,
                              duration: float, no_tukey: bool) -> Tuple[
                                  np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float
                              ]:
    """Prepare one analysis/PSD segment: optional Tukey window, FFT, and BW fdata."""

    n = strain.size
    _require_power_of_two(n)
    t_obs = float(duration)
    dt = t_obs / n
    data = strain.copy()
    alpha = 2.0 * T_RISE / t_obs
    window_power_correction = tukey_power_correction(n, alpha)
    if not no_tukey:
        tukey_inplace(data, alpha)
    dataf_c = np.fft.rfft(data)
    fdata = np.zeros(n, dtype=np.float64)
    fdata[2:n:2] = dataf_c[1:n // 2].real
    fdata[3:n + 1:2] = dataf_c[1:n // 2].imag
    return times.copy(), data, dataf_c, fdata, dt, alpha, window_power_correction


def cleaned_scaled_fdata_from_windowed(data: np.ndarray, dt: float) -> np.ndarray:
    """Return BayesLine-scaled Fourier data after wavelet cleaning one segment."""

    n = data.size
    cleaned_time = bayesline_clean_time(data, dt)
    cleaned_fft = np.fft.rfft(cleaned_time) * (dt / math.sqrt(2.0))
    fdata = np.zeros(n, dtype=np.float64)
    fdata[2:n:2] = cleaned_fft[1:n // 2].real
    fdata[3:n + 1:2] = cleaned_fft[1:n // 2].imag
    return fdata


def full_power_from_scaled_fdata(fdata: np.ndarray) -> np.ndarray:
    """Return internal BayesLine power from dt/sqrt(2)-scaled packed FFT data."""

    n = fdata.size
    power = np.zeros(n // 2, dtype=np.float64)
    power[1:] = fdata[2:n:2] ** 2 + fdata[3:n + 1:2] ** 2
    return power


def build_same_grid_context_average_power(context_times: np.ndarray,
                                          context_strain: np.ndarray,
                                          duration: float,
                                          chunk_count: int,
                                          no_tukey: bool) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Average independently cleaned powers from same-duration context chunks.

    The chunks all have the target duration, so the averaged power lives on the
    same Fourier grid and uses the same windowed-Lorentzian lookup as the
    target analysis segment.  The context RJMCMC supplies the statistical SNR
    boost through ``power_multiplicity = chunk_count``.
    """

    if chunk_count < 1:
        raise ValueError("context chunk count must be positive")
    if context_strain.size % chunk_count != 0:
        raise ValueError("context segment cannot be split into equal chunks")
    chunk_n = context_strain.size // chunk_count
    _require_power_of_two(chunk_n)
    if context_times.size != context_strain.size:
        raise ValueError("context time/strain arrays have different lengths")

    power_sum = np.zeros(chunk_n // 2, dtype=np.float64)
    raw_power_sum = np.zeros(chunk_n // 2 - 1, dtype=np.float64)
    dt = duration / chunk_n
    window_power_correction = 1.0
    alpha = 2.0 * T_RISE / duration
    for m in range(chunk_count):
        lo = m * chunk_n
        hi = lo + chunk_n
        _times, windowed, dataf_c, _raw_fdata, dt, alpha, window_power_correction = prepare_frequency_segment(
            context_times[lo:hi], context_strain[lo:hi], duration, no_tukey
        )
        cleaned_fdata = cleaned_scaled_fdata_from_windowed(windowed, dt)
        power_sum += full_power_from_scaled_fdata(cleaned_fdata)
        raw_power_sum += (
            2.0 * dt * dt * window_power_correction
            * np.abs(dataf_c[1:chunk_n // 2]) ** 2 / duration
        )

    return (
        power_sum / float(chunk_count),
        raw_power_sum / float(chunk_count),
        dt,
        alpha,
        window_power_correction,
    )


def final_line_parameter_rows(bayesline: BayesLineParams, psd_scale: float) -> np.ndarray:
    """Return final fair-draw Lorentzian peak parameters."""

    if bayesline.data is None or bayesline.lines_x is None:
        raise RuntimeError("BayesLine state is incomplete")
    lines = bayesline.lines_x
    if lines.n == 0:
        return np.empty((0, 5), dtype=np.float64)

    rows = np.zeros((lines.n, 5), dtype=np.float64)
    tiny = np.finfo(float).tiny
    for k in range(lines.n):
        f0 = float(lines.f[k])
        nu = float(lines.nu[k])
        a_peak_internal = float(lines.a[k])
        rows[k, 0] = f0
        rows[k, 1] = nu
        rows[k, 2] = f0 / max(nu, tiny)
        rows[k, 3] = a_peak_internal
        rows[k, 4] = psd_scale * a_peak_internal
    order = np.argsort(rows[:, 0])
    return rows[order]


def write_final_line_parameters(bayesline: BayesLineParams, psd_scale: float,
                                filename: str = "line_parameters.dat") -> None:
    """Write final fair-draw Lorentzian parameters for cross-duration checks."""

    rows = final_line_parameter_rows(bayesline, psd_scale)
    np.savetxt(
        filename,
        rows,
        header="f0_Hz nu_Hz Q a_peak_internal a_peak_physical",
    )
    print(f"Wrote {filename} with final fair-draw Lorentzian line parameters")


def clone_lorentzian_lookup_container(template: LorentzianParams, size: int) -> LorentzianParams:
    """Allocate a Lorentzian container while preserving an attached target lookup."""

    out = LorentzianParams(size=size)
    out.nf = template.nf
    out.nnu = template.nnu
    out.wdth = template.wdth
    out.imin = template.imin
    out.numin = template.numin
    out.numax = template.numax
    out.lnumin = template.lnumin
    out.lnumax = template.lnumax
    out.ltemplate = template.ltemplate
    return out


def copy_context_state_to_target(source: BayesLineParams, target: BayesLineParams,
                                 spline_flag: int, gaussian_enabled: bool) -> None:
    """Copy a same-duration context posterior state into the target state.

    The current ``--expand`` design stacks independent target-duration context
    chunks, so the context and target BayesLine grids have identical frequency
    spacing, lookup tables, Tukey roll-off, and internal PSD units.  No line
    peak or area rescaling is applied here; copied parameters are rebuilt on the
    target arrays only to attach them to the target likelihood.
    """

    if source.spline_x is None or source.lines_x is None:
        raise RuntimeError("source BayesLine state is incomplete")
    if target.data is None or target.freq is None or target.lines_x is None:
        raise RuntimeError("target BayesLine state is incomplete")
    if source.data is None or source.freq is None:
        raise RuntimeError("source BayesLine data are incomplete")
    if source.data.ncut != target.data.ncut or not np.allclose(source.freq, target.freq, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("context and target grids differ; same-grid state copy is invalid")

    target_points = source.spline_x.points.copy()
    target_spline_data = source.spline_x.data.copy()
    target.spline_x = SplineParams(target_points, target_spline_data)
    target.spline = SplineParams(target_points.copy(), target_spline_data.copy())
    sbase = np.exp(spline_eval_array(target_points, target_spline_data, target.freq, spline_flag))

    nlines = source.lines_x.n
    if target.lines_x.size < nlines:
        target.lines_x = clone_lorentzian_lookup_container(target.lines_x, nlines)
    target.lines_x.n = nlines
    if nlines > 0:
        target.lines_x.f[:nlines] = source.lines_x.f[:nlines]
        target.lines_x.nu[:nlines] = source.lines_x.nu[:nlines]
        target.lines_x.a[:nlines] = source.lines_x.a[:nlines]
        target.lines_x.q[:nlines] = target.lines_x.f[:nlines] / np.maximum(target.lines_x.nu[:nlines], np.finfo(float).tiny)
    sline = sline_from_lookup(
        target.data.ncut, target.data.imin, target.data.t_obs,
        target.lines_x.nf, target.lines_x.nnu, target.lines_x.wdth,
        target.lines_x.lnumin, target.lines_x.lnumax, target.lines_x.ltemplate,
        target.lines_x.f, target.lines_x.a, target.lines_x.nu, target.lines_x.n
    )

    if gaussian_enabled and source.bumps_x is not None:
        nbump = source.bumps_x.n
        if target.bumps_x is None or target.bumps_x.size < nbump:
            target.bumps_x = create_gaussianBumpParams(max(nbump, 1))
        target.bumps_x.n = nbump
        if nbump > 0:
            target.bumps_x.f[:nbump] = source.bumps_x.f[:nbump]
            target.bumps_x.a[:nbump] = source.bumps_x.a[:nbump]
            target.bumps_x.sigma[:nbump] = source.bumps_x.sigma[:nbump]
    elif target.bumps_x is not None:
        target.bumps_x.n = 0
    if gaussian_enabled and target.bumps_x is not None:
        sgauss = gaussian_bumps_from_params(target.freq, target.bumps_x)
    else:
        sgauss = np.zeros(target.data.ncut, dtype=np.float64)

    target.Sbase = sbase
    target.Sline = sline
    target.Sgauss = sgauss
    target.Snf = sbase + sline + sgauss
    target.Sna = target.Snf.copy()
    target.rng = source.rng


def ensure_lorentzian_capacity(lines: LorentzianParams, capacity: int) -> LorentzianParams:
    """Return a Lorentzian container with at least ``capacity`` storage slots."""

    capacity = max(int(capacity), int(lines.n), 1)
    if lines.f.size >= capacity:
        lines.size = capacity
        return lines

    out = clone_lorentzian_lookup_container(lines, capacity)
    ncopy = min(lines.n, capacity)
    out.n = ncopy
    if ncopy > 0:
        out.f[:ncopy] = lines.f[:ncopy]
        out.a[:ncopy] = lines.a[:ncopy]
        out.nu[:ncopy] = lines.nu[:ncopy]
        out.q[:ncopy] = lines.q[:ncopy]
    return out


def ensure_gaussian_bump_capacity(bumps: GaussianBumpParams, capacity: int) -> GaussianBumpParams:
    """Return a Gaussian-bump container with at least ``capacity`` storage slots."""

    capacity = max(int(capacity), int(bumps.n), 1)
    if bumps.f.size >= capacity:
        bumps.size = capacity
        return bumps

    out = GaussianBumpParams(size=capacity)
    ncopy = min(bumps.n, capacity)
    out.n = ncopy
    if ncopy > 0:
        out.f[:ncopy] = bumps.f[:ncopy]
        out.a[:ncopy] = bumps.a[:ncopy]
        out.sigma[:ncopy] = bumps.sigma[:ncopy]
    return out


def set_line_amplitude_prior_from_state(bayesline: BayesLineParams) -> None:
    """Set log-uniform line-amplitude prior bounds from the current state."""

    if bayesline.lines_x is None or bayesline.power is None or bayesline.Sbase is None:
        raise RuntimeError("BayesLine state is incomplete")
    if bayesline.Sgauss is None:
        background = bayesline.Sbase
    else:
        background = bayesline.Sbase + bayesline.Sgauss

    if bayesline.lines_x.n > 0:
        amin = float(np.min(bayesline.lines_x.a[:bayesline.lines_x.n]))
        amax = float(np.max(bayesline.lines_x.a[:bayesline.lines_x.n]))
    else:
        excess = np.maximum(bayesline.power - background, np.finfo(float).tiny)
        amin = float(np.min(excess))
        amax = float(np.max(excess))
    bayesline.priors.LAmin = max(0.1 * amin, np.finfo(float).tiny)
    bayesline.priors.LAmax = max(10.0 * amax, bayesline.priors.LAmin * 10.0)


def adapt_context_target_model_caps(bayesline: BayesLineParams,
                                    gaussian_enabled: bool = False) -> None:
    """Set target-stage caps to twice the transferred context final dimensions."""

    if bayesline.lines_x is None or bayesline.spline_x is None:
        raise RuntimeError("BayesLine target state is incomplete")

    initialized_lines = int(bayesline.lines_x.n)
    initialized_knots = int(bayesline.spline_x.n)
    line_cap = max(2, 2 * initialized_lines)
    spline_cap = max(14, 2 * initialized_knots)

    bayesline.lines_x = ensure_lorentzian_capacity(bayesline.lines_x, line_cap)
    bayesline.maxBLLines = line_cap
    bayesline.maxSplineKnots = spline_cap

    if gaussian_enabled and bayesline.bumps_x is not None:
        initialized_bumps = int(bayesline.bumps_x.n)
        bump_cap = max(2, 2 * initialized_bumps)
        bayesline.bumps_x = ensure_gaussian_bump_capacity(bayesline.bumps_x, bump_cap)
        bayesline.maxBLBumps = bump_cap
        print(
            "Context-prior target caps: "
            f"max lines {line_cap} (context {initialized_lines}), "
            f"max spline knots {spline_cap} (context {initialized_knots}), "
            f"max Gaussian bumps {bump_cap} (context {initialized_bumps})"
        )
    else:
        print(
            "Context-prior target caps: "
            f"max lines {line_cap} (context {initialized_lines}), "
            f"max spline knots {spline_cap} (context {initialized_knots})"
        )


def install_context_psd_prior(target: BayesLineParams,
                              context_psd_samples: np.ndarray,
                              context_spline_samples: np.ndarray,
                              context_gaussian_samples: Optional[np.ndarray],
                              sample_count: int,
                              inflate: float,
                              sigma_floor: float,
                              output_psd_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the target-stage Gaussian log-PSD prior from context PSD samples."""

    if target.data is None or target.freq is None or target.Snf is None or target.Sbase is None:
        raise RuntimeError("target BayesLine state is incomplete")
    tiny = np.finfo(float).tiny
    ncut = target.data.ncut

    if sample_count > 0:
        active_psd = np.maximum(context_psd_samples[:sample_count, :ncut], tiny)
        log_samples = np.log(active_psd)
        log_center = np.median(log_samples, axis=0)
        if sample_count > 1:
            log_sigma = np.std(log_samples, axis=0, ddof=1)
        else:
            log_sigma = np.zeros(ncut, dtype=np.float64)
        prior_mean = np.exp(log_center)

        smooth_samples = np.maximum(context_spline_samples[:sample_count, :ncut], tiny)
        if context_gaussian_samples is not None:
            gaussian_component_samples = np.maximum(context_gaussian_samples[:sample_count, :ncut], 0.0)
        else:
            gaussian_component_samples = np.zeros_like(smooth_samples)
        line_component_samples = np.maximum(active_psd - smooth_samples - gaussian_component_samples, 0.0)

        log_smooth_samples = np.log(smooth_samples)
        smooth_log_center = np.median(log_smooth_samples, axis=0)
        if sample_count > 1:
            smooth_log_sigma = np.std(log_smooth_samples, axis=0, ddof=1)
        else:
            smooth_log_sigma = np.zeros(ncut, dtype=np.float64)
        smooth_mean = np.exp(smooth_log_center)

        line_log1p_samples = np.log1p(line_component_samples / np.maximum(smooth_mean[None, :], tiny))
        line_log1p_mean = np.median(line_log1p_samples, axis=0)
        if sample_count > 1:
            line_log1p_sigma = np.std(line_log1p_samples, axis=0, ddof=1)
        else:
            line_log1p_sigma = np.zeros(ncut, dtype=np.float64)

        gaussian_log1p_samples = np.log1p(gaussian_component_samples / np.maximum(smooth_mean[None, :], tiny))
        gaussian_log1p_mean = np.median(gaussian_log1p_samples, axis=0)
        if sample_count > 1:
            gaussian_log1p_sigma = np.std(gaussian_log1p_samples, axis=0, ddof=1)
        else:
            gaussian_log1p_sigma = np.zeros(ncut, dtype=np.float64)

        prior_background = smooth_mean + np.median(gaussian_component_samples, axis=0)
    else:
        print("WARNING: no context PSD samples were collected; using final context state as prior center")
        prior_mean = np.maximum(target.Snf.copy(), tiny)
        smooth_mean = np.maximum(target.Sbase.copy(), tiny)
        smooth_log_sigma = np.zeros(ncut, dtype=np.float64)
        if target.Sline is None:
            line_component = np.zeros(ncut, dtype=np.float64)
        else:
            line_component = np.maximum(target.Sline.copy(), 0.0)
        line_log1p_mean = np.log1p(line_component / np.maximum(smooth_mean, tiny))
        line_log1p_sigma = np.zeros(ncut, dtype=np.float64)
        if target.Sgauss is None:
            gaussian_component = np.zeros(ncut, dtype=np.float64)
        else:
            gaussian_component = np.maximum(target.Sgauss.copy(), 0.0)
        gaussian_log1p_mean = np.log1p(gaussian_component / np.maximum(smooth_mean, tiny))
        gaussian_log1p_sigma = np.zeros(ncut, dtype=np.float64)
        prior_background = np.maximum(smooth_mean + gaussian_component, tiny)
        log_sigma = np.zeros(ncut, dtype=np.float64)

    prior_sigma = np.maximum(log_sigma * inflate, sigma_floor)
    target.priors.mean = prior_mean
    target.priors.sigma = prior_sigma
    # The Gaussian context prior is centered on the total PSD, but the spline
    # birth/update proposals use ``priors.lower`` as their local amplitude scale.
    # That proposal scale must follow the copied smooth spline itself, just
    # like the ordinary startup path uses Sbase/10.  Near a loud line,
    # total_psd/10 can sit above the smooth spline and cause every spline move to
    # fail the proposal-bound check before the likelihood or context prior gets a
    # vote.  The explicit log-PSD prior below still constrains the total model
    # around prior_mean.
    target.priors.lower = np.maximum(target.Sbase / 10.0, tiny)
    target.priors.upper = np.maximum(prior_mean * 100.0, target.priors.lower * 100.0)
    target.priors.use_context_psd_prior = False
    target.priors.use_context_component_prior = True
    target.priors.smooth_mean = smooth_mean
    target.priors.smooth_sigma = np.maximum(smooth_log_sigma * inflate, sigma_floor)
    target.priors.component_reference = smooth_mean
    target.priors.line_log1p_mean = line_log1p_mean
    target.priors.line_log1p_sigma = np.maximum(line_log1p_sigma * inflate, sigma_floor)
    if context_gaussian_samples is not None:
        target.priors.gaussian_log1p_mean = gaussian_log1p_mean
        target.priors.gaussian_log1p_sigma = np.maximum(gaussian_log1p_sigma * inflate, sigma_floor)
    else:
        target.priors.gaussian_log1p_mean = None
        target.priors.gaussian_log1p_sigma = None
    set_line_amplitude_prior_from_state(target)

    np.savetxt(
        "context_psd_prior.dat",
        np.column_stack((target.freq, output_psd_scale * prior_mean, prior_sigma)),
        header="frequency context_prior_median_psd log_psd_sigma",
    )
    component_columns = [
        target.freq,
        output_psd_scale * smooth_mean,
        target.priors.smooth_sigma,
        line_log1p_mean,
        target.priors.line_log1p_sigma,
    ]
    component_header = "frequency smooth_median_psd smooth_log_sigma line_log1p_mean line_log1p_sigma"
    if context_gaussian_samples is not None:
        component_columns.extend([
            gaussian_log1p_mean,
            target.priors.gaussian_log1p_sigma,
        ])
        component_header += " gaussian_log1p_mean gaussian_log1p_sigma"
    np.savetxt(
        "context_component_prior.dat",
        np.column_stack(component_columns),
        header=component_header,
    )
    print(
        "Installed context component priors on target run: "
        f"log-sigma floor {sigma_floor:g}, inflation {inflate:g}, "
        f"samples {sample_count}"
    )
    print("Wrote context_psd_prior.dat")
    print("Wrote context_component_prior.dat")
    return prior_mean, prior_background, prior_sigma


def build_context_line_proposal(target: BayesLineParams, fprop: np.ndarray,
                                context_prior_mean: np.ndarray,
                                context_background: np.ndarray) -> None:
    """Construct the target-stage line proposal from the context posterior."""

    if target.data is None or target.freq is None or target.lines_x is None:
        raise RuntimeError("target BayesLine state is incomplete")
    data = target.data
    ncut = data.ncut
    weights = np.ones(ncut, dtype=np.float64)
    prior_source = np.ones(ncut, dtype=np.float64)
    final_line_source = np.ones(ncut, dtype=np.float64)

    if context_prior_mean.size >= ncut and context_background.size >= ncut:
        ratio = context_prior_mean[:ncut] / np.maximum(context_background[:ncut], np.finfo(float).tiny)
        prior_source[ratio > LINE_PROPOSAL_THRESH] = LINE_PROPOSAL_BOOST
        weights = np.maximum(weights, prior_source)

    for k in range(target.lines_x.n):
        idx = int(round((float(target.lines_x.f[k]) - data.flow) * data.t_obs))
        if 0 <= idx < ncut:
            final_line_source[idx] = LINE_PROPOSAL_BOOST
            weights[idx] = max(weights[idx], LINE_PROPOSAL_BOOST)

    total = float(np.sum(weights))
    if total <= 0.0:
        weights[:] = 1.0 / max(1, ncut)
        total = 1.0

    fprop.fill(0.0)
    fprop[:ncut] = weights / total
    np.savetxt(
        "line_proposal.dat",
        np.column_stack((target.freq, fprop[:ncut], prior_source / total, final_line_source / total)),
        header="frequency normalized_line_proposal context_psd_excess_scaled context_final_line_scaled",
    )
    print("Wrote line_proposal.dat for the target context-prior run")


def decimation_factor_for_nyquist(data_nyquist: float, requested_nyquist: float) -> int:
    """Return the legacy integer downsampling factor for an existing file."""

    if math.isclose(data_nyquist, requested_nyquist, rel_tol=1.0e-6, abs_tol=1.0e-6):
        return 1

    ratio = data_nyquist / requested_nyquist
    nearest = int(round(ratio))
    if nearest >= 1 and math.isclose(ratio, float(nearest), rel_tol=1.0e-5, abs_tol=1.0e-6):
        dec = nearest
    else:
        dec = int(ratio)

    if dec > 8:
        dec = 8
    if dec < 1:
        dec = 1
    return dec


def detector_label_from_args(args: argparse.Namespace) -> str:
    """Return the detector label used to tag fetched-data output files."""

    if args.channel is not None and ":" in args.channel:
        detector = args.channel.split(":", 1)[0]
    else:
        detector = args.ifo
    return detector.strip().upper()


def tagged_output_name(base_name: str, input_spec: InputSpec) -> str:
    """Decorate main output filenames for fetched trigger-time data."""

    if not input_spec.output_tag:
        return base_name
    stem, suffix = base_name.rsplit(".", 1)
    return f"{stem}_{input_spec.output_tag}.{suffix}"


def plot_prior_recovery_histograms(count_trace: np.ndarray, line_max: int, spline_max: int,
                                   gaussian_max: Optional[int] = None,
                                   burnin_fraction: float = 0.5,
                                   plot_file: str = "prior_recovery_histograms.pdf") -> None:
    """Write dimension histograms from a prior-recovery run."""

    if count_trace.size == 0:
        print("warning: prior-recovery trace is empty; skipping histogram plot")
        return
    if burnin_fraction < 0.0 or burnin_fraction >= 1.0:
        print("warning: invalid prior-recovery histogram burn-in fraction; skipping histogram plot")
        return

    start = int(math.floor(burnin_fraction * count_trace.shape[0]))
    samples = count_trace[start:]
    if samples.size == 0:
        print("warning: no prior-recovery samples remain after histogram burn-in cut")
        return

    line_counts = samples[:, 1].astype(np.int64)
    spline_counts = samples[:, 2].astype(np.int64)
    has_gaussian_counts = count_trace.shape[1] > 3
    gaussian_counts = samples[:, 3].astype(np.int64) if has_gaussian_counts else None

    def integer_histogram(values: np.ndarray, lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observed_lo = int(np.min(values))
        observed_hi = int(np.max(values))
        lo = min(lo, observed_lo)
        hi = max(hi, observed_hi)
        centers = np.arange(lo, hi + 1, dtype=np.int64)
        bins = np.arange(lo, hi + 2, dtype=np.float64) - 0.5
        counts, _ = np.histogram(values, bins=bins)
        prob = counts.astype(np.float64) / max(1, int(np.sum(counts)))
        return centers, counts, prob

    line_x, line_hist, line_prob = integer_histogram(line_counts, 1, line_max)
    spline_x, spline_hist, spline_prob = integer_histogram(spline_counts, 7, spline_max)
    gaussian_x = gaussian_hist = gaussian_prob = None
    if gaussian_counts is not None:
        gaussian_hi = gaussian_max if gaussian_max is not None else int(np.max(gaussian_counts))
        gaussian_x, gaussian_hist, gaussian_prob = integer_histogram(
            gaussian_counts, 0, max(0, gaussian_hi)
        )
    np.savetxt(
        "prior_recovery_line_histogram.dat",
        np.column_stack((line_x, line_prob, line_hist)),
        fmt=["%d", "%.18e", "%d"],
        header="n_lines probability count",
    )
    np.savetxt(
        "prior_recovery_spline_histogram.dat",
        np.column_stack((spline_x, spline_prob, spline_hist)),
        fmt=["%d", "%.18e", "%d"],
        header="n_spline probability count",
    )
    written_files = [
        "prior_recovery_line_histogram.dat",
        "prior_recovery_spline_histogram.dat",
    ]
    if gaussian_x is not None and gaussian_hist is not None and gaussian_prob is not None:
        np.savetxt(
            "prior_recovery_gaussian_histogram.dat",
            np.column_stack((gaussian_x, gaussian_prob, gaussian_hist)),
            fmt=["%d", "%.18e", "%d"],
            header="n_gaussian_bumps probability count",
        )
        written_files.append("prior_recovery_gaussian_histogram.dat")

    try:
        import os
        import tempfile

        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bwtest_matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable; wrote histogram data files but not {plot_file}: {exc}")
        return

    npanels = 3 if gaussian_x is not None and gaussian_prob is not None else 2
    fig_height = 9.5 if npanels == 3 else 7.0
    fig, axes = plt.subplots(npanels, 1, figsize=(9.0, fig_height), constrained_layout=True)
    axes = np.atleast_1d(axes)
    panels = [
        (axes[0], line_x, line_prob, "number of Lorentzian lines", "P(n_lines)"),
        (axes[1], spline_x, spline_prob, "number of Akima spline knots", "P(n_spline)"),
    ]
    if npanels == 3:
        panels.append(
            (axes[2], gaussian_x, gaussian_prob, "number of Gaussian PSD bumps", "P(n_gaussian)")
        )
    for ax, xval, prob, xlabel, ylabel in panels:
        ax.bar(xval, prob, width=0.85, color="0.45", edgecolor="0.2", linewidth=0.2)
        if xval.size > 0:
            ax.axhline(1.0 / float(xval.size), color="tab:red", linestyle="--", linewidth=1.0,
                       label="uniform over plotted support")
            ax.set_xlim(float(xval[0]) - 0.75, float(xval[-1]) + 0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="0.9", linewidth=0.7)
        ax.legend(loc="best", frameon=False)
    fig.suptitle(f"Prior recovery histograms, samples {start} to {count_trace.shape[0] - 1}")
    fig.savefig(plot_file)
    plt.close(fig)
    print(f"Wrote {plot_file}")
    print("Wrote " + ", ".join(written_files))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=Path(argv[0]).name,
        description="Fetch a LIGO strain segment by trigger time, or run BWtest on an existing two-column strain file.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="either an existing two-column strain file, or GPS trigger_time duration",
    )
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
        "--duration",
        "--analysis-duration",
        type=float,
        default=None,
        help="desired analysis duration in seconds for an existing input file",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=1,
        help="power-of-two count of target-duration context chunks split before/after the target; default 1 keeps the original one-segment path",
    )
    parser.add_argument(
        "--context_prior_inflate",
        "--context-prior-inflate",
        type=float,
        default=None,
        help="inflate the context log-PSD sample standard deviation by this factor; default sqrt(--expand)",
    )
    parser.add_argument(
        "--context_prior_sigma_floor",
        "--context-prior-sigma-floor",
        type=float,
        default=CONTEXT_PRIOR_LOG_SIGMA_FLOOR,
        help="minimum log-PSD sigma for the --expand context prior, default %(default)g",
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
        "--fmin",
        type=float,
        default=DEFAULT_ANALYSIS_FMIN,
        help="low-frequency analysis cutoff in Hz, default %(default)g",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional fetched strain text file name, default <IFO>.txt",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=2.0,
        help="seconds fetched on each side before resampling, then cropped away; frame/GWOSC inputs only, default %(default)g",
    )
    parser.add_argument(
        "--resampler",
        choices=("lal", "gwpy"),
        default=None,
        help="frame/GWOSC resampler; default is lal for --source cluster and gwpy for --source open",
    )
    parser.add_argument(
        "--no-tukey",
        "--no_tukey",
        "--input-already-tukeyed",
        action="store_true",
        help=(
            "skip applying a new Tukey window to the input; intended for data "
            "already tapered with BWtest's Tukey roll-off. The line model and "
            "output scaling still assume one standard Tukey window."
        ),
    )
    parser.add_argument(
        "--write_bl_start",
        "--write-bl-start",
        action="store_true",
        help=f"write BayesLine startup diagnostic to {BL_START_FILENAME}",
    )
    parser.add_argument(
        "--writewhite",
        "--write-white",
        action="store_true",
        help="write C-style whitelsf.dat and whitelsf_noglitch.dat diagnostics",
    )
    parser.add_argument(
        "--write_line_subtracted_time",
        "--write-line-subtracted-time",
        "--writelinesubtime",
        action="store_true",
        help="write unwhitened no-lines diagnostics: freq_nolines.dat plus time-domain outputs before and after glitch cleaning",
    )
    parser.add_argument(
        "--write_line_params",
        "--write-line-params",
        "--writelineparams",
        action="store_true",
        help="write final fair-draw Lorentzian parameters to line_parameters.dat",
    )
    parser.add_argument(
        "--write_glitch",
        "--write-glitch",
        "--writeglitch",
        action="store_true",
        help="write the time-domain wavelet-cleaning glitch and the same glitch whitened by the final BWpsd",
    )
    parser.add_argument(
        "--feature",
        action="store_true",
        help="cluster the whitened glitch with the Denoise.py SNR cut and write whitened_feature_time.dat",
    )
    parser.add_argument(
        "--feature-snr-thresh",
        "--feature_snr_thresh",
        type=float,
        default=FEATURE_SNR_THRESH,
        help="cluster SNR threshold used with --feature, default %(default)g",
    )
    parser.add_argument(
        "--sinefit",
        "--sine-fit",
        action="store_true",
        help="write sinefit.dat and freq_nosine.dat for Tukeyed-sinusoid line diagnostics/subtraction",
    )
    parser.add_argument(
        "--linetrim",
        "--line-trim",
        action="store_true",
        help=(
            f"turn on the exp(-ZETA*nlines) line-count prior with ZETA={LINE_COUNT_PRIOR_ZETA:g}; "
            "off by default, but on in the C BayesLine sampler"
        ),
    )
    parser.add_argument(
        "--linecut",
        "--line-cut",
        type=float,
        default=LINECUT,
        help="cut_lorentz delta-log-likelihood retention threshold, default %(default)g",
    )
    parser.add_argument(
        "--gaussian_bumps",
        "--gaussian-bumps",
        action="store_true",
        help=(
            "turn on broad Gaussian PSD bumps in the RJMCMC; the model starts "
            "with zero bumps and proposes reversible-jump birth/death/update moves"
        ),
    )
    parser.add_argument(
        "--gaussian_bump_sigma_min",
        "--gaussian-bump-sigma-min",
        type=float,
        default=GAUSSIAN_BUMP_SIGMA_MIN,
        help="minimum Gaussian-bump width in Hz, default %(default)g",
    )
    parser.add_argument(
        "--gaussian_bump_sigma_max",
        "--gaussian-bump-sigma-max",
        type=float,
        default=GAUSSIAN_BUMP_SIGMA_MAX,
        help="maximum Gaussian-bump width in Hz, default %(default)g",
    )
    parser.add_argument(
        "--prior_recovery",
        "--prior-recovery",
        action="store_true",
        help="run the RJMCMC with constant likelihood, disable soft count/PSD priors, and write prior_recovery_counts.dat",
    )
    parser.add_argument(
        "--prior_recovery_plot_burnin",
        "--prior-recovery-plot-burnin",
        type=float,
        default=0.5,
        help="fraction of prior-recovery count trace to discard before histogramming, default %(default)g",
    )
    parser.add_argument(
        "--rjmcmc_steps",
        "--rjmcmc-steps",
        "--mcmc-steps",
        type=int,
        default=100000,
        help="number of Lorentzian+spline RJMCMC iterations, default %(default)d",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for startup and RJMCMC stochastic steps, default %(default)d",
    )
    parser.add_argument(
        "--psd_samples",
        "--psd-samples",
        "--median-psd-samples",
        type=int,
        default=200,
        help="number of second-half RJMCMC PSD states to store for the median PSD, default %(default)d",
    )
    parser.add_argument(
        "--psd_range",
        "--psd-range",
        "--write-psd-range",
        action="store_true",
        help="write psd_range.dat with frequency, MCMC-sample mean PSD, and PSD standard deviation",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="print wall-clock timing for the main BWtest phases",
    )
    parser.add_argument(
        "--debug_acceptance",
        "--debug-acceptance",
        action="store_true",
        help="print proposal attempt/valid/invalid/accepted counts for MCMC move labels",
    )
    return parser.parse_args(argv[1:])


def resolve_input_file(args: argparse.Namespace, nyquist: int) -> InputSpec:
    if args.input_file is not None:
        analysis_duration = args.duration
        if args.inputs:
            if len(args.inputs) == 1 and analysis_duration is None:
                try:
                    analysis_duration = float(args.inputs[0])
                except ValueError as exc:
                    raise ValueError("--file accepts at most one positional analysis duration") from exc
            else:
                print("warning: --file was supplied, so positional inputs are ignored")
        if analysis_duration is not None and analysis_duration <= 0.0:
            raise ValueError("analysis duration must be positive")
        return InputSpec(args.input_file, fetched_from_frame=False, analysis_duration=analysis_duration)

    if len(args.inputs) == 1:
        try:
            float(args.inputs[0])
        except ValueError:
            pass
        else:
            if not Path(args.inputs[0]).exists():
                raise ValueError("provide both trigger_time and duration, or use --file INPUT")
        analysis_duration = args.duration
        if analysis_duration is not None and analysis_duration <= 0.0:
            raise ValueError("analysis duration must be positive")
        return InputSpec(args.inputs[0], fetched_from_frame=False, analysis_duration=analysis_duration)

    if len(args.inputs) == 2:
        first = Path(args.inputs[0])
        if first.exists():
            if args.duration is not None:
                raise ValueError("analysis duration was supplied both positionally and with --duration")
            try:
                analysis_duration = float(args.inputs[1])
            except ValueError as exc:
                raise ValueError("file input with two positional arguments must be INPUT_FILE duration") from exc
            if analysis_duration <= 0.0:
                raise ValueError("analysis duration must be positive")
            return InputSpec(args.inputs[0], fetched_from_frame=False, analysis_duration=analysis_duration)

        try:
            trigger_time = float(args.inputs[0])
            duration = float(args.inputs[1])
        except ValueError as exc:
            raise ValueError("positional inputs must be either one file name or trigger_time duration") from exc
        if duration <= 0.0:
            raise ValueError("segment duration must be positive")
        if args.padding < 0.0:
            raise ValueError("padding must be non-negative")
        target_rate = 2.0 * float(nyquist)
        if args.expand > 1:
            start, end = expanded_fetch_bounds(trigger_time, duration, args.expand)
            filename = fetch_ligo_data_bounds(
                start, end, args.ifo, args.source, args.channel,
                args.output, target_rate, args.padding, args.resampler,
                trigger_time=trigger_time
            )
        else:
            filename = fetch_ligo_data(
                trigger_time, duration, args.ifo, args.source, args.channel,
                args.output, target_rate, args.padding, args.resampler
            )
        trigger_label = int(math.floor(trigger_time + 0.5))
        output_tag = f"{detector_label_from_args(args)}_{trigger_label}"
        return InputSpec(
            filename,
            fetched_from_frame=True,
            output_tag=output_tag,
            trigger_time=trigger_time,
            analysis_duration=duration,
        )

    raise ValueError("provide a frame file, --file INPUT, or trigger_time duration")


def main(argv: list[str]) -> int:
    """Driver matching BWtest.c: preprocess data, run BayesLine, write outputs."""

    args = parse_args(argv)
    timing_marks: list[Tuple[str, float]] = []
    timing_start = time.perf_counter()
    timing_last = timing_start

    def timing_mark(label: str) -> None:
        nonlocal timing_last
        if not args.timing:
            return
        now = time.perf_counter()
        timing_marks.append((label, now - timing_last))
        timing_last = now

    ok, nyquist = valid_power_of_two_frequency(args.nyquist)
    if not ok:
        print(f"warning: requested Nyquist frequency {args.nyquist:g} Hz is not a power of two")
        return 1
    if args.expand < 1 or not is_power_of_two_int(int(args.expand)):
        print("warning: --expand must be a positive power of two")
        return 1
    args.expand = int(args.expand)
    if args.context_prior_inflate is not None and (
            not math.isfinite(args.context_prior_inflate) or args.context_prior_inflate <= 0.0):
        print("warning: --context-prior-inflate must be positive and finite")
        return 1
    if (not math.isfinite(args.context_prior_sigma_floor)
            or args.context_prior_sigma_floor <= 0.0):
        print("warning: --context-prior-sigma-floor must be positive and finite")
        return 1
    if not math.isfinite(args.fmin) or args.fmin <= 0.0 or args.fmin >= float(nyquist):
        print(f"warning: requested fmin {args.fmin:g} Hz must be finite and below Nyquist {nyquist:g} Hz")
        return 1
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0.0):
        print("warning: --duration must be positive and finite")
        return 1
    if args.psd_samples < 1:
        print("warning: --psd-samples must be at least 1")
        return 1
    if args.rjmcmc_steps < 1:
        print("warning: --rjmcmc-steps must be at least 1")
        return 1
    if args.seed < 0:
        print("warning: --seed must be non-negative")
        return 1
    if not math.isfinite(args.linecut) or args.linecut < 0.0:
        print("warning: --linecut must be finite and non-negative")
        return 1
    if (not math.isfinite(args.prior_recovery_plot_burnin)
            or args.prior_recovery_plot_burnin < 0.0
            or args.prior_recovery_plot_burnin >= 1.0):
        print("warning: --prior-recovery-plot-burnin must be in the range [0, 1)")
        return 1
    if args.prior_recovery and args.expand > 1:
        print("warning: --prior-recovery is only supported for --expand 1")
        return 1
    if not math.isfinite(args.feature_snr_thresh) or args.feature_snr_thresh < 0.0:
        print("warning: --feature-snr-thresh must be finite and non-negative")
        return 1
    if (not math.isfinite(args.gaussian_bump_sigma_min)
            or not math.isfinite(args.gaussian_bump_sigma_max)
            or args.gaussian_bump_sigma_min <= 0.0
            or args.gaussian_bump_sigma_max <= args.gaussian_bump_sigma_min):
        print("warning: Gaussian bump widths must satisfy 0 < sigma_min < sigma_max")
        return 1
    if args.resampler is None:
        args.resampler = "gwpy" if args.source == "open" else "lal"

    fmin = float(args.fmin)
    fmax = float(nyquist)
    try:
        input_spec = resolve_input_file(args, nyquist)
        timeX, dataX = read_frame(input_spec.filename)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"warning: {exc}")
        return 1
    timing_mark("read_input")

    expand_factor = int(args.expand)
    expanded_psd = expand_factor > 1
    file_duration_requested = (not input_spec.fetched_from_frame
                               and input_spec.analysis_duration is not None)
    segment_crop_requested = expanded_psd or file_duration_requested
    ND = timeX.size
    print(f"Number of points in input data = {ND}")

    if segment_crop_requested and input_spec.analysis_duration is None:
        print("warning: an explicit analysis duration is required for this segmented input mode")
        return 1

    if segment_crop_requested:
        analysis_duration = float(input_spec.analysis_duration)
        psd_duration = analysis_duration * float(expand_factor)
        context_before_chunks = expand_factor // 2 if expanded_psd else 0
        context_after_chunks = expand_factor - context_before_chunks if expanded_psd else 0
        context_before_duration = analysis_duration * float(context_before_chunks)
        context_after_duration = analysis_duration * float(context_after_chunks)
        if expanded_psd and psd_duration > 256.0:
            print(f"WARNING: context/prior segment is {psd_duration:g} seconds, which exceeds 256 seconds")

        dt_in = infer_regular_dt(timeX)
        input_duration = dt_in * ND
        fny = 1.0 / (2.0 * dt_in)
        print(f"Input duration {input_duration:f} s, dt {dt_in:e}, Nyquist {fny:f}, requested {fmax:f}")
        sample_rate_matches_request = math.isclose(fny, fmax, rel_tol=1.0e-6, abs_tol=1.0e-6)
        if sample_rate_matches_request:
            fny = fmax
        if fmax > fny:
            print(f"warning: requested Nyquist {fmax:g} Hz exceeds data Nyquist {fny:g} Hz")
            return 1

        dec = decimation_factor_for_nyquist(fny, fmax)
        print(f"Down sample = {dec}")
        if input_spec.fetched_from_frame and dec > 1:
            expected_rate = 2.0 * fmax
            actual_rate = 2.0 * fny
            print(
                f"warning: fetched frame data should have been resampled to {expected_rate:g} Hz, "
                f"but the saved data imply {actual_rate:g} Hz"
            )
            return 1

        if input_spec.fetched_from_frame:
            print("Skipping Butterworth decimation for externally resampled frame input")
        elif dec == 1:
            if sample_rate_matches_request:
                print("Sample rate already matches requested Nyquist; skipping Butterworth decimation")
            else:
                print("Skipping Butterworth decimation; legacy integer downsampling factor is 1")
        else:
            fmn = 1.0 / analysis_duration
            fmx = fmax
            dataX = bwbpf_numba(dataX, 1, 8, 1.0 / dt_in, fmx, fmn)
            dataX = bwbpf_numba(dataX, -1, 8, 1.0 / dt_in, fmx, fmn)

        full_times = timeX[::dec].copy()
        full_data = dataX[::dec].copy()
        available_start = float(full_times[0])
        full_dt = infer_regular_dt(full_times)
        available_end = available_start + full_dt * full_times.size
        if input_spec.fetched_from_frame:
            if input_spec.trigger_time is None:
                print("warning: fetched-data trigger time is missing")
                return 1
            trigger_time = float(input_spec.trigger_time)
            analysis_start, analysis_end = segment_bounds(trigger_time, analysis_duration)
        else:
            if expanded_psd:
                # A plain file has no trigger metadata or guaranteed naming
                # convention.  For the two-stage symmetric context run, assume
                # the target segment sits in the middle of the provided data
                # with equal context duration before and after it.
                target_center = available_start + 0.5 * input_duration
                analysis_start = target_center - 0.5 * analysis_duration
                analysis_end = analysis_start + analysis_duration
                trigger_time = analysis_end - TRIGGER_OFFSET_FROM_END
                print(
                    "WARNING: user-supplied --expand input assumes the target "
                    f"analysis segment is centered in the file with "
                    f"{context_before_chunks} context chunks before and "
                    f"{context_after_chunks} after (effective trigger GPS "
                    f"{trigger_time:.6f})"
                )
            else:
                # Keep the historical file+duration behavior for --expand 1:
                # treat the file center as the trigger so the analysis segment
                # keeps the usual trigger+4 end time.
                trigger_time = float(timeX[0]) + 0.5 * input_duration
                analysis_start, analysis_end = segment_bounds(trigger_time, analysis_duration)
                print(
                    "WARNING: user-supplied analysis duration assumes the trigger "
                    f"is at the center of the provided data, GPS {trigger_time:.6f}"
                )

        if expanded_psd:
            context_before_start = analysis_start - context_before_duration
            context_before_end = analysis_start
            context_after_start = analysis_end
            context_after_end = analysis_end + context_after_duration
            psd_start = context_before_start
            psd_end = context_after_end
        else:
            context_before_start = context_after_start = analysis_start
            context_before_end = context_after_end = analysis_end
            psd_start, psd_end = analysis_start, analysis_end
        needed_start = min(analysis_start, psd_start)
        needed_end = max(analysis_end, psd_end)
        tol = 0.5 * full_dt
        if needed_start < available_start - tol or needed_end > available_end + tol:
            print(
                "warning: not enough data provided for requested analysis/PSD segments; "
                f"need [{needed_start:.6f}, {needed_end:.6f}] "
                f"but file covers [{available_start:.6f}, {available_end:.6f}]"
            )
            return 1

        if expanded_psd:
            print(
                f"Using analysis segment [{analysis_start:.6f}, {analysis_end:.6f}] "
                f"with context/prior before [{context_before_start:.6f}, {context_before_end:.6f}] "
                f"and after [{context_after_start:.6f}, {context_after_end:.6f}]"
            )
        else:
            print(
                f"Using requested analysis duration {analysis_duration:g} s from input file; "
                f"analysis/PSD segment [{analysis_start:.6f}, {analysis_end:.6f}]"
            )
        try:
            # The fit segment drives BayesLine initialization.  With --expand,
            # the symmetric context is split into target-duration chunks; the
            # nearest preceding chunk initializes the context sampler, and the
            # averaged cleaned powers from all chunks replace the likelihood
            # power before the context RJMCMC starts.
            if expanded_psd:
                before_times_raw, before_data_raw = crop_arrays_by_time(
                    full_times, full_data, context_before_start, context_before_duration
                )
                after_times_raw, after_data_raw = crop_arrays_by_time(
                    full_times, full_data, context_after_start, context_after_duration
                )
                context_times_raw = np.concatenate((before_times_raw, after_times_raw))
                context_data_raw = np.concatenate((before_data_raw, after_data_raw))
            else:
                context_times_raw, context_data_raw = crop_arrays_by_time(full_times, full_data, psd_start, psd_duration)
            analysis_times_raw, analysis_data_raw = crop_arrays_by_time(
                full_times, full_data, analysis_start, analysis_duration
            )
        except ValueError as exc:
            print(f"warning: {exc}")
            return 1

        Tobs = analysis_duration
        if expanded_psd:
            if context_data_raw.size % expand_factor != 0:
                print("warning: context segment cannot be split into equal target-duration chunks")
                return 1
            context_chunk_n = context_data_raw.size // expand_factor
            if context_chunk_n != analysis_data_raw.size:
                print(
                    "warning: context chunk sample count does not match target analysis sample count "
                    f"({context_chunk_n} != {analysis_data_raw.size})"
                )
                return 1
            fit_times_raw = before_times_raw[-context_chunk_n:].copy()
            fit_data_raw = before_data_raw[-context_chunk_n:].copy()
            fit_Tobs = Tobs
        else:
            fit_times_raw = context_times_raw
            fit_data_raw = context_data_raw
            fit_Tobs = Tobs
        print(f"Number of points used in analysis = {analysis_data_raw.size}")
        if expanded_psd:
            print(
                f"Number of points used in context/prior segment = {context_data_raw.size} "
                f"({expand_factor} chunks of {fit_data_raw.size})"
            )
        else:
            print(f"Number of points used for PSD estimation = {fit_data_raw.size}")
    else:
        print(f"Number of points in data = {ND}")
        dt = (timeX[-1] - timeX[0]) / ND
        Tobs = float(round(ND * dt))
        dt = Tobs / ND
        fny = 1.0 / (2.0 * dt)
        print(f"{Tobs:f} {dt:e} {fny:f} {fmax:f}")
        sample_rate_matches_request = math.isclose(fny, fmax, rel_tol=1.0e-6, abs_tol=1.0e-6)
        if sample_rate_matches_request:
            fny = fmax
        if fmax > fny:
            print(f"warning: requested Nyquist {fmax:g} Hz exceeds data Nyquist {fny:g} Hz")
            return 1

        dec = decimation_factor_for_nyquist(fny, fmax)
        print(f"Down sample = {dec}")

        if input_spec.fetched_from_frame and dec > 1:
            expected_rate = 2.0 * fmax
            actual_rate = 2.0 * fny
            print(
                f"warning: fetched frame data should have been resampled to {expected_rate:g} Hz, "
                f"but the saved data imply {actual_rate:g} Hz"
            )
            return 1

        N = ND // dec
        try:
            _require_power_of_two(N)
        except ValueError as exc:
            print(f"warning: {exc} after downsampling by {dec}")
            return 1
        print(f"Number of points used in analysis = {N}")

        if input_spec.fetched_from_frame:
            print("Skipping Butterworth decimation for externally resampled frame input")
        elif dec == 1:
            if sample_rate_matches_request:
                print("Sample rate already matches requested Nyquist; skipping Butterworth decimation")
            else:
                print("Skipping Butterworth decimation; legacy integer downsampling factor is 1")
        else:
            fmn = 1.0 / Tobs
            fmx = fmax
            dataX = bwbpf_numba(dataX, 1, 8, 1.0 / dt, fmx, fmn)
            dataX = bwbpf_numba(dataX, -1, 8, 1.0 / dt, fmx, fmn)

        analysis_times_raw = timeX[::dec][:N].copy()
        analysis_data_raw = dataX[::dec][:N].copy()
        fit_times_raw = analysis_times_raw
        fit_data_raw = analysis_data_raw
        context_times_raw = fit_times_raw
        context_data_raw = fit_data_raw
        fit_Tobs = Tobs

    N = analysis_data_raw.size
    fit_N = fit_data_raw.size

    if args.no_tukey:
        print("WARNING: --no-tukey set; BWtest will not apply a new Tukey window to the input data")
        print("WARNING: line model still uses the standard Tukey-windowed Lorentzian lookup")
        print("WARNING: output scaling still applies the usual one-window Tukey power correction")

    times, data, dataf_c, fdata, dt, alpha, window_power_correction = prepare_frequency_segment(
        analysis_times_raw, analysis_data_raw, Tobs, args.no_tukey
    )
    context_average_power_full: Optional[np.ndarray] = None
    context_raw_power_avg: Optional[np.ndarray] = None
    if expanded_psd:
        # In the normal path, BayesLineBurnin wavelet-cleans fdata in place. In
        # the expanded path burn-in initializes from the nearest preceding
        # context chunk, so we perform the same wavelet-cleaning step separately
        # for the target analysis segment.
        fit_times, fit_data, fit_dataf_c, fit_fdata, fit_dt, fit_alpha, fit_window_power_correction = prepare_frequency_segment(
            fit_times_raw, fit_data_raw, fit_Tobs, args.no_tukey
        )
        fdata = cleaned_scaled_fdata_from_windowed(data, dt)
        print(
            "Building same-grid context power from "
            f"{context_before_chunks} chunks before and {context_after_chunks} after the target"
        )
        context_average_power_full, context_raw_power_avg, stack_dt, stack_alpha, stack_window_power_correction = (
            build_same_grid_context_average_power(
                context_times_raw, context_data_raw, Tobs, expand_factor, args.no_tukey
            )
        )
        if (not math.isclose(stack_dt, fit_dt, rel_tol=1.0e-12, abs_tol=0.0)
                or not math.isclose(stack_window_power_correction, fit_window_power_correction,
                                    rel_tol=1.0e-12, abs_tol=0.0)
                or not math.isclose(stack_alpha, fit_alpha, rel_tol=1.0e-12, abs_tol=0.0)):
            print("warning: inconsistent context chunk preprocessing parameters")
            return 1
    else:
        fit_times = times
        fit_data = data
        fit_dataf_c = dataf_c
        fit_fdata = fdata
        fit_dt = dt
        fit_alpha = alpha
        fit_window_power_correction = window_power_correction
    output_psd_scale = 4.0 * window_power_correction / Tobs
    fit_output_psd_scale = 4.0 * fit_window_power_correction / fit_Tobs
    raw_freq = np.arange(1, N // 2, dtype=np.float64) / Tobs
    raw_power = 2.0 * dt * dt * window_power_correction * np.abs(dataf_c[1:N // 2]) ** 2 / Tobs
    np.savetxt("periodogram_raw.dat", np.column_stack((raw_freq, raw_power)))
    if expanded_psd:
        fit_raw_freq = np.arange(1, N // 2, dtype=np.float64) / Tobs
        if context_raw_power_avg is None:
            fit_raw_power = (
                2.0 * fit_dt * fit_dt * fit_window_power_correction
                * np.abs(fit_dataf_c[1:fit_N // 2]) ** 2 / fit_Tobs
            )
        else:
            fit_raw_power = context_raw_power_avg
        np.savetxt("periodogram_raw_psd.dat", np.column_stack((fit_raw_freq, fit_raw_power)))
    timing_mark("preprocess_fft")

    psd = np.zeros(N // 2, dtype=np.float64)
    invpsd = np.zeros(N // 2, dtype=np.float64)
    splinePSD = np.zeros(N // 2, dtype=np.float64)
    fprop = np.zeros(fit_N // 2, dtype=np.float64)

    fit_bptr = BayesLineParams()
    run_seed = int(args.seed)
    fit_bptr.rng = np.random.default_rng(run_seed)
    startup_rng = np.random.default_rng(run_seed)
    BayesLineSetup(fit_bptr, fit_fdata, fmin, fmax, fit_dt, fit_Tobs)
    fit_bptr.gaussianSigmaMin = float(args.gaussian_bump_sigma_min)
    fit_bptr.gaussianSigmaMax = float(args.gaussian_bump_sigma_max)
    target_bptr: Optional[BayesLineParams] = None
    if expanded_psd:
        target_bptr = BayesLineParams()
        # The target setup loads/generates the requested-duration line lookup.
        # The expanded context run now uses same-duration chunks, so the context
        # and target grids share the same lookup. The target container still
        # holds the analysis-segment likelihood and receives the context state
        # and posterior samples before its own RJMCMC stage.
        BayesLineSetup(target_bptr, fdata, fmin, fmax, dt, Tobs)
        set_power_from_scaled_frequency_data(target_bptr, fdata)
        target_bptr.gaussianSigmaMin = float(args.gaussian_bump_sigma_min)
        target_bptr.gaussianSigmaMax = float(args.gaussian_bump_sigma_max)
        assert fit_bptr.data is not None and fit_bptr.lines_x is not None
        assert target_bptr.data is not None and target_bptr.lines_x is not None
        print(
            f"Expanded same-grid context-prior run: target T={Tobs:g} s, "
            f"context chunks={expand_factor}, context likelihood multiplicity={expand_factor}"
        )
    print(f"BayesLine line array size = {fit_bptr.maxBLLines}")
    print(f"BayesLine startup linecut = {args.linecut:g}")
    if args.gaussian_bumps:
        print(
            "Gaussian bump RJMCMC enabled: "
            f"array size = {fit_bptr.maxBLBumps}, "
            f"sigma range = [{fit_bptr.gaussianSigmaMin:g}, {fit_bptr.gaussianSigmaMax:g}] Hz"
        )
    timing_mark("BayesLineSetup")
    BayesLineBurnin(
        fit_bptr, fit_data, fit_fdata, "H1", fprop, 1,
        write_start=args.write_bl_start,
        output_psd_scale=fit_output_psd_scale,
        linecut=args.linecut,
        startup_rng=startup_rng,
    )
    if expanded_psd:
        if context_average_power_full is None or fit_bptr.data is None:
            print("warning: internal error, stacked context power was not built")
            return 1
        fit_bptr.power = context_average_power_full[fit_bptr.data.imin:fit_bptr.data.imax].copy()
        fit_bptr.spow = fit_bptr.power.copy()
        fit_bptr.power_multiplicity = float(expand_factor)
        print(
            "Context RJMCMC using averaged cleaned power from "
            f"{expand_factor} same-duration chunks with multiplicity {fit_bptr.power_multiplicity:g}"
        )
    else:
        fit_bptr.power_multiplicity = 1.0
    adapt_post_startup_model_caps(fit_bptr)
    timing_mark("BayesLineBurnin")

    assert fit_bptr.data is not None and fit_bptr.Sbase is not None and fit_bptr.Snf is not None
    imin = int(fit_bptr.data.fmin * fit_Tobs)
    startup_model_psd_scale = 0.5 * fit_output_psd_scale
    p_freq = np.arange(imin, fit_N // 2, dtype=np.float64) / fit_Tobs
    if expanded_psd:
        p_pow = 2.0 * fit_output_psd_scale * fit_bptr.power
    else:
        p_pow = fit_output_psd_scale * 2.0 * (fit_fdata[2 * imin:fit_N:2] ** 2 +
                                              fit_fdata[2 * imin + 1:fit_N + 1:2] ** 2)
    model_len = p_freq.size
    np.savetxt("periodogram.dat", np.column_stack((p_freq, p_pow[:model_len],
                                                   startup_model_psd_scale * fit_bptr.Sbase[:model_len],
                                                   startup_model_psd_scale * fit_bptr.Snf[:model_len])))
    if expanded_psd:
        np.savetxt("fprop_context.dat", np.column_stack((fit_bptr.freq, fprop[:fit_bptr.data.ncut])))
    else:
        np.savetxt("fprop.dat", np.column_stack((fit_bptr.freq, fprop[:fit_bptr.data.ncut])))
    timing_mark("startup_diagnostics")

    rjmcmc_steps = args.rjmcmc_steps
    trace_columns = 4 if args.gaussian_bumps else 3
    spline_prior_max = fit_bptr.maxSplineKnots
    prior_flag = 1
    line_trim_enabled = args.linetrim
    if args.prior_recovery:
        fit_bptr.constantLogLFlag = 1
        prior_flag = 0
        line_trim_enabled = False
        print("Prior recovery mode: RJMCMC likelihood is constant; soft PSD and line-count priors are disabled")
        if args.linetrim:
            print("Prior recovery mode: ignoring --linetrim so line counts are not exponentially biased")

    if expanded_psd:
        if target_bptr is None or target_bptr.data is None:
            print("warning: internal error, target BayesLine state was not initialized")
            return 1
        context_sample_ncut = fit_bptr.data.ncut
        context_psd_samples = np.empty((args.psd_samples, context_sample_ncut), dtype=np.float64)
        context_spline_samples = np.empty_like(context_psd_samples)
        context_gaussian_samples = np.empty_like(context_psd_samples) if args.gaussian_bumps else None
        collected_context_samples = BayesLineRJMCMC(
            fit_bptr, fit_fdata, psd, invpsd, splinePSD, N, rjmcmc_steps, 1.0, prior_flag, fprop, 1,
            psd_samples=context_psd_samples, spline_samples=context_spline_samples,
            gaussian_samples=context_gaussian_samples,
            count_trace=None,
            line_trim=line_trim_enabled,
            gaussian_bumps=args.gaussian_bumps,
            debug_acceptance=args.debug_acceptance
        )
        copy_context_state_to_target(fit_bptr, target_bptr, 1, args.gaussian_bumps)
        timing_mark("context_RJMCMC")

        context_prior_inflate = (math.sqrt(float(expand_factor))
                                 if args.context_prior_inflate is None
                                 else float(args.context_prior_inflate))
        context_prior_mean, context_background, _context_prior_sigma = install_context_psd_prior(
            target_bptr, context_psd_samples, context_spline_samples, context_gaussian_samples,
            collected_context_samples, context_prior_inflate,
            float(args.context_prior_sigma_floor), output_psd_scale
        )
        adapt_context_target_model_caps(target_bptr, args.gaussian_bumps)
        target_fprop = np.zeros(N // 2, dtype=np.float64)
        build_context_line_proposal(target_bptr, target_fprop, context_prior_mean, context_background)

        psd_samples = np.empty((args.psd_samples, target_bptr.data.ncut), dtype=np.float64)
        spline_samples = np.empty_like(psd_samples)
        gaussian_samples = np.empty_like(psd_samples) if args.gaussian_bumps else None
        collected_psd_samples = BayesLineRJMCMC(
            target_bptr, fdata, psd, invpsd, splinePSD, N, rjmcmc_steps, 1.0, prior_flag, target_fprop, 1,
            psd_samples=psd_samples, spline_samples=spline_samples,
            gaussian_samples=gaussian_samples,
            count_trace=None,
            line_trim=line_trim_enabled,
            gaussian_bumps=args.gaussian_bumps,
            debug_acceptance=args.debug_acceptance
        )
        bptr = target_bptr
        fprop = target_fprop
        timing_mark("BayesLineRJMCMC")
        count_trace = None
    else:
        sample_ncut = fit_bptr.data.ncut
        psd_samples = np.empty((args.psd_samples, sample_ncut), dtype=np.float64)
        spline_samples = np.empty_like(psd_samples)
        gaussian_samples = np.empty_like(psd_samples) if args.gaussian_bumps else None
        count_trace = np.empty((rjmcmc_steps, trace_columns), dtype=np.int64) if args.prior_recovery else None
        collected_psd_samples = BayesLineRJMCMC(
            fit_bptr, fit_fdata, psd, invpsd, splinePSD, N, rjmcmc_steps, 1.0, prior_flag, fprop, 1,
            psd_samples=psd_samples, spline_samples=spline_samples,
            gaussian_samples=gaussian_samples,
            count_trace=count_trace,
            line_trim=line_trim_enabled,
            gaussian_bumps=args.gaussian_bumps,
            debug_acceptance=args.debug_acceptance
        )
        bptr = fit_bptr
        timing_mark("BayesLineRJMCMC")
    if count_trace is not None:
        np.savetxt(
            "prior_recovery_counts.dat",
            count_trace,
            fmt="%d",
            header="iteration n_lines n_spline" + (" n_gaussian_bumps" if args.gaussian_bumps else ""),
        )
        print("Wrote prior_recovery_counts.dat with iteration, n_lines, n_spline")
        plot_prior_recovery_histograms(
            count_trace,
            fit_bptr.maxBLLines,
            spline_prior_max,
            gaussian_max=fit_bptr.maxBLBumps if args.gaussian_bumps else None,
            burnin_fraction=args.prior_recovery_plot_burnin,
        )

    bw_freq = np.arange(N // 2, dtype=np.float64) / Tobs
    freq_out = np.zeros((N // 2, 3), dtype=np.float64)
    freq_out[:, 0] = bw_freq
    # BayesLine samples the PSD in BayesWave internal units.  With the
    # dt/sqrt(2)-scaled Fourier coefficients used in the sampler, the internal
    # PSD is Tobs/4 times the physical one-sided PSD.  Convert the final
    # user-facing files back to duration-independent PSD units while scaling the
    # frequency-domain data consistently for ADtest.py's Re/sqrt(2*PSD) check.
    psd_scale = output_psd_scale
    freq_scale = math.sqrt(16.0 * window_power_correction / Tobs)
    freq_out[1:, 1] = freq_scale * fdata[2:N:2]
    freq_out[1:, 2] = freq_scale * fdata[3:N + 1:2]

    fairdraw_psd_out = psd_scale * psd.copy()
    if fairdraw_psd_out.size > 1:
        fairdraw_psd_out[0] = fairdraw_psd_out[1]
    fairdraw_smooth_out = psd_scale * splinePSD.copy()
    fairdraw_gaussian = np.zeros_like(psd)
    if bptr.Sgauss is not None:
        fairdraw_gaussian[bptr.data.imin:bptr.data.imax] = bptr.Sgauss
    fairdraw_gaussian_out = psd_scale * fairdraw_gaussian
    fairdraw_line_out = fairdraw_psd_out - fairdraw_smooth_out - fairdraw_gaussian_out
    np.savetxt("BWfairdrawpsd.dat", np.column_stack((bw_freq, fairdraw_psd_out)))
    if args.gaussian_bumps:
        np.savetxt("BWfairdrawpsd_components.dat",
                   np.column_stack((bw_freq, fairdraw_psd_out, fairdraw_smooth_out,
                                    fairdraw_line_out, fairdraw_gaussian_out)))
    else:
        np.savetxt("BWfairdrawpsd_components.dat",
                   np.column_stack((bw_freq, fairdraw_psd_out, fairdraw_smooth_out, fairdraw_line_out)))

    median_psd = psd.copy()
    median_spline = splinePSD.copy()
    median_gaussian = fairdraw_gaussian.copy()
    average_psd = psd.copy()
    if collected_psd_samples > 0:
        active_samples = psd_samples[:collected_psd_samples]
        median_psd.fill(1.0)
        median_spline.fill(1.0)
        median_gaussian.fill(0.0)
        average_psd.fill(1.0)
        median_psd[bptr.data.imin:bptr.data.imax] = np.median(active_samples, axis=0)
        median_spline[bptr.data.imin:bptr.data.imax] = np.median(spline_samples[:collected_psd_samples], axis=0)
        if gaussian_samples is not None:
            median_gaussian[bptr.data.imin:bptr.data.imax] = np.median(
                gaussian_samples[:collected_psd_samples], axis=0
            )
        average_psd[bptr.data.imin:bptr.data.imax] = np.mean(active_samples, axis=0)
        if collected_psd_samples != args.psd_samples:
            print(f"warning: collected {collected_psd_samples} PSD samples, requested {args.psd_samples}")
    if args.prior_recovery:
        print("Prior recovery mode: skipped average-PSD likelihood diagnostic")
    else:
        average_model_active = np.maximum(average_psd[bptr.data.imin:bptr.data.imax], np.finfo(float).tiny)
        average_summand = -(bptr.power / average_model_active + np.log(average_model_active))
        average_loglike = float(np.sum(average_summand))
        np.savetxt(
            "BW_likelihood_summand.dat",
            np.column_stack((bptr.freq, average_summand)),
            header="frequency log_likelihood_summand",
        )
        print("Wrote BW_likelihood_summand.dat for the average PSD model")
        print(f"Average PSD model log likelihood = {average_loglike:.12e} ({average_loglike / bptr.data.ncut:.12e} per frequency bin)")

    median_psd_out = psd_scale * median_psd
    if median_psd_out.size > 1:
        median_psd_out[0] = median_psd_out[1]
    median_smooth_out = psd_scale * median_spline
    median_gaussian_out = psd_scale * median_gaussian
    median_line_out = median_psd_out - median_smooth_out - median_gaussian_out
    bwpsd_filename = tagged_output_name("BWpsd.dat", input_spec)
    frequency_data_filename = tagged_output_name("frequency_data.dat", input_spec)
    if input_spec.output_tag:
        print(f"Writing tagged main outputs: {bwpsd_filename}, {frequency_data_filename}")
    np.savetxt(bwpsd_filename, np.column_stack((bw_freq, median_psd_out)))
    if args.gaussian_bumps:
        np.savetxt("BWpsd_components.dat",
                   np.column_stack((bw_freq, median_psd_out, median_smooth_out,
                                    median_line_out, median_gaussian_out)))
    else:
        np.savetxt("BWpsd_components.dat", np.column_stack((bw_freq, median_psd_out, median_smooth_out, median_line_out)))
    if args.psd_range:
        std_psd = np.zeros_like(psd)
        if collected_psd_samples > 0:
            ddof = 1 if collected_psd_samples > 1 else 0
            std_psd[bptr.data.imin:bptr.data.imax] = np.std(active_samples, axis=0, ddof=ddof)
        mean_psd_out = psd_scale * average_psd
        std_psd_out = psd_scale * std_psd
        if mean_psd_out.size > 1:
            mean_psd_out[0] = mean_psd_out[1]
            std_psd_out[0] = std_psd_out[1]
        np.savetxt("psd_range.dat", np.column_stack((bw_freq, mean_psd_out, std_psd_out)))
        print("Wrote psd_range.dat with MCMC-sample mean PSD and standard deviation")
    if args.write_line_params:
        line_params_filename = tagged_output_name("line_parameters.dat", input_spec)
        write_final_line_parameters(bptr, psd_scale, line_params_filename)
    np.savetxt(frequency_data_filename, freq_out)
    if args.write_glitch:
        write_glitch_time_domain_files(times, data, fdata, median_psd_out, dt, window_power_correction)
    if args.feature:
        write_whitened_feature_file(
            times, data, fdata, median_psd_out, dt, window_power_correction,
            args.feature_snr_thresh
        )
    if args.writewhite:
        write_whitened_line_subtracted_files(bptr, dataf_c, fdata, dt, bptr.rng)
    if args.write_line_subtracted_time:
        write_line_subtracted_frequency_file(bptr, dataf_c, dt, bptr.rng)
        write_time_domain_line_subtracted_files(bptr, times, dataf_c, fdata, dt, bptr.rng)
    if args.sinefit:
        sinefit(bptr, N, alpha)
        write_sine_subtracted_frequency_file(bptr, dataf_c, dt, N, alpha)
    timing_mark("write_outputs")

    if args.timing:
        total = time.perf_counter() - timing_start
        print("Timing summary")
        for label, elapsed in timing_marks:
            print(f"  {label:22s} {elapsed:10.3f} s")
        print(f"  {'total':22s} {total:10.3f} s")
        for label, elapsed in timing_marks:
            if label == "BayesLineRJMCMC" and rjmcmc_steps > 0:
                print(f"  {'RJMCMC per 100K':22s} {elapsed * 100000.0 / rjmcmc_steps:10.3f} s")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
