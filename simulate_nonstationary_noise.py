#!/usr/bin/env python3
"""Simulate non-stationary Gaussian noise in the time domain.

The default example is intended to look like a LIGO-style one-sided PSD with a
slow time-dependent modulation,

    S(f,t) = S_f(f) S_t(t).

For this separable case the simulation is fast: draw stationary colored Fourier
noise, transform it to the time domain, and multiply by sqrt(S_t).  The script
also includes a structured fast path for the built-in nonseparable phase-band
example, using a short harmonic expansion and FFTs.  A slow exact path for a
general dynamic spectrum S(f,t) is included for small tests and checks, but it
scales like N^2 and is not intended for long LIGO-length series.  If numba is
installed, the exact direct sum can be JIT accelerated for the built-in dynamic
examples; SciPy FFTs are used automatically when scipy is installed.

Edit example_frequency_psd(), example_time_modulation(), or
example_dynamic_psd() to define a new model.  For real-valued output in general
mode, the dynamic spectrum should satisfy S(-f,t)=S(f,t).

Simulation methods
------------------

1. Fast separable path

   For S(f,t)=S_f(f)S_t(t), the covariance square root factors into a stationary
   coloring in frequency and an amplitude modulation in time.  The code draws
   Hermitian Fourier coefficients with variance set by S_f(f), transforms them
   to the time domain, and multiplies by sqrt(S_t(t)).  This costs one inverse
   FFT and is the default path:

       python3 simulate_nonstationary_noise.py

2. Short-FFT harmonic path

   Some nonseparable spectra are still structured enough to simulate quickly.
   The built-in phase-band example has

       S(f,t) = S_f(f) [1 + A G(f) cos(2 pi t/tau + psi(f))].

   The fast engine expands the square-root modulation
   sqrt(1 + A G(f) cos theta) in a short Fourier series in theta.  Each retained
   harmonic is then simulated with a small number of FFTs.  The real requirement
   is low harmonic rank in time:

       sqrt(S(f,t)/S_f(f)) ~= sum_{h=-H}^{H} a_h(f) exp(i h Omega t)

   with modest H.  Slow, smooth modulation is one way to get this, but the
   modulation does not have to be slow in an absolute sense if only a few
   harmonics are present.  A limited frequency band G(f) is useful for modeling
   localized non-stationarity and can make the coefficient functions cleaner, but
   it is not the main speed requirement.  Broad-band low-harmonic modulation is
   still fast, while sharp transient time dependence usually requires many
   harmonics or a different structured basis.  Increase --harmonic-count and
   --harmonic-quadrature to check convergence:

       python3 simulate_nonstationary_noise.py --mode general --general-engine harmonic

3. Exact direct-sum paths

   The exact dynamic-spectrum construction sums over all basis frequencies and
   costs O(N^2).  The pure NumPy implementation calls example_dynamic_psd(), so
   it is easy to edit but only suitable for small tests.  The numba engine can
   accelerate the same direct sum for heavier checks, but only for spectra that
   have been written in jit-able form.  Numba cannot compile arbitrary Python
   functions or callbacks such as a user-edited example_dynamic_psd().  To add a
   new accelerated dynamic spectrum, mirror its formula inside
   _simulate_general_builtin_numba() or write a new @njit kernel with explicit
   scalar math.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from scipy import fft as scipy_fft
except Exception:  # pragma: no cover - optional dependency
    scipy_fft = None

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional dependency
    njit = None
    prange = range


def example_frequency_psd(f_hz: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """One-sided example PSD S_f(f) in strain^2/Hz.

    The absolute frequency is used so the same function can be evaluated on the
    two-sided FFT frequency grid.
    """

    f = np.abs(f_hz)
    low = args.f0 / (f + args.fx)
    return args.s0 * (
        low**4
        + args.a * low**2.5
        + args.b
        + args.c * (f / args.f0) ** 2
    )


def example_time_modulation(t_s: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Dimensionless positive time modulation S_t(t)."""

    return 1.0 + args.mod_amp * np.cos(2.0 * np.pi * t_s / args.mod_period)


def example_dynamic_psd(
    f_hz: np.ndarray,
    t_s: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """General one-sided dynamic spectrum S(f,t).

    This default is the separable example.  Replace this body with any positive
    dynamic spectrum for non-separable simulations.  The inputs may be arrays
    with broadcast-compatible shapes.  For a real time series, keep the dynamic
    spectrum even in frequency.
    """

    base = example_frequency_psd(f_hz, args)
    if args.dynamic_example == "separable":
        return base * example_time_modulation(t_s, args)
    if args.dynamic_example == "phase-band":
        f_abs = np.abs(f_hz)
        envelope = np.exp(-0.5 * ((f_abs - args.phase_band_fc) / args.phase_band_sigma_f) ** 2)
        phase = (
            2.0 * np.pi * t_s / args.mod_period
            + 2.0 * np.pi * (f_abs - args.phase_band_fc) / args.phase_band_delta_f_phase
        )
        multiplier = 1.0 + args.mod_amp * envelope * np.cos(phase)
        return base * multiplier
    raise ValueError(f"unknown dynamic example {args.dynamic_example}")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_time_grid(duration: float, sample_rate: float) -> tuple[np.ndarray, float]:
    n_float = duration * sample_rate
    n = int(round(n_float))
    if not np.isclose(n_float, n, rtol=0.0, atol=1.0e-9):
        raise ValueError("--duration * --sample-rate must be an integer")
    dt = 1.0 / sample_rate
    return np.arange(n, dtype=float) * dt, dt


def inverse_fft(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """IFFT using NumPy or SciPy when available."""

    backend = args.fft_backend
    if backend == "auto":
        backend = "scipy" if scipy_fft is not None else "numpy"
    if backend == "scipy":
        if scipy_fft is None:
            raise RuntimeError("SciPy FFT backend requested, but scipy is not available")
        try:
            return scipy_fft.ifft(values, workers=args.fft_workers)
        except TypeError:
            return scipy_fft.ifft(values)
    return np.fft.ifft(values)


def one_sided_to_internal_spectrum(
    psd_one_sided: np.ndarray,
    sample_rate: float,
    convention: str,
) -> np.ndarray:
    """Convert a one-sided PSD to the DFT covariance convention.

    With NumPy's unnormalized FFT, a stationary real time series with one-sided
    PSD S_1(f) has E[|X_k|^2] = N fs S_1(f_k) / 2 for positive-frequency bins.
    The finite-data covariance convention used here writes this as
    E[|X_k|^2] = N S_internal[k], so S_internal = fs S_1 / 2.
    """

    if convention == "one-sided":
        return 0.5 * sample_rate * psd_one_sided
    if convention == "internal":
        return psd_one_sided
    raise ValueError(f"unknown PSD convention {convention}")


def hermitian_unit_fourier_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return Hermitian Fourier coefficients with unit variance per bin."""

    eta = np.zeros(n, dtype=np.complex128)
    eta[0] = rng.normal()
    if n % 2 == 0:
        eta[n // 2] = rng.normal()
        pos = np.arange(1, n // 2)
    else:
        pos = np.arange(1, (n + 1) // 2)
    eta[pos] = (rng.normal(size=len(pos)) + 1j * rng.normal(size=len(pos))) / math.sqrt(2.0)
    eta[-pos] = np.conj(eta[pos])
    return eta


if njit is not None:

    @njit(parallel=True, fastmath=True)
    def _simulate_general_builtin_numba(
        times: np.ndarray,
        freqs: np.ndarray,
        eta_re: np.ndarray,
        eta_im: np.ndarray,
        sample_rate: float,
        psd_is_one_sided: int,
        dynamic_code: int,
        s0: float,
        f0: float,
        fx: float,
        a: float,
        b: float,
        c: float,
        mod_amp: float,
        mod_period: float,
        phase_band_fc: float,
        phase_band_sigma_f: float,
        phase_band_delta_f_phase: float,
    ) -> np.ndarray:
        n = len(times)
        out = np.empty(n, dtype=np.float64)
        norm = 1.0 / math.sqrt(n)
        two_pi = 2.0 * math.pi

        for i in prange(n):
            t = times[i]
            acc = 0.0
            for k in range(n):
                f_signed = freqs[k]
                f = abs(f_signed)
                low = f0 / (f + fx)
                base = s0 * (low**4 + a * low**2.5 + b + c * (f / f0) ** 2)
                if dynamic_code == 0:
                    modulation = 1.0 + mod_amp * math.cos(two_pi * t / mod_period)
                else:
                    x = (f - phase_band_fc) / phase_band_sigma_f
                    envelope = math.exp(-0.5 * x * x)
                    phase = (
                        two_pi * t / mod_period
                        + two_pi * (f - phase_band_fc) / phase_band_delta_f_phase
                    )
                    modulation = 1.0 + mod_amp * envelope * math.cos(phase)

                internal = base * modulation
                if psd_is_one_sided == 1:
                    internal *= 0.5 * sample_rate
                amp = math.sqrt(internal)
                angle = two_pi * f_signed * t
                acc += amp * (eta_re[k] * math.cos(angle) - eta_im[k] * math.sin(angle))
            out[i] = norm * acc
        return out


def check_positive(name: str, values: np.ndarray) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive")


def simulate_separable(
    rng: np.random.Generator,
    times: np.ndarray,
    sample_rate: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Fast simulation for S(f,t)=S_f(f)S_t(t)."""

    n = len(times)
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    psd_f = example_frequency_psd(freqs, args)
    modulation = example_time_modulation(times, args)
    check_positive("S_f(f)", psd_f)
    check_positive("S_t(t)", modulation)

    internal_spectrum = one_sided_to_internal_spectrum(psd_f, sample_rate, args.psd_convention)
    eta = hermitian_unit_fourier_noise(rng, n)
    stationary_f = np.sqrt(internal_spectrum) * eta
    stationary_t = math.sqrt(n) * inverse_fft(stationary_f, args).real
    return np.sqrt(modulation) * stationary_t


def phase_band_sqrt_coefficients(
    b_values: np.ndarray,
    harmonics: int,
    quadrature: int,
    chunk_size: int,
) -> np.ndarray:
    """Fourier coefficients of sqrt(1 + B cos theta) for each B."""

    if harmonics < 0:
        raise ValueError("--harmonic-count must be non-negative")
    if quadrature <= 2 * harmonics:
        raise ValueError("--harmonic-quadrature should exceed 2 * --harmonic-count")
    if chunk_size <= 0:
        raise ValueError("--harmonic-coeff-chunk-size must be positive")

    theta = 2.0 * np.pi * np.arange(quadrature, dtype=float) / quadrature
    cos_theta = np.cos(theta)
    cos_h_theta = np.vstack([np.cos(h * theta) for h in range(harmonics + 1)])
    coeffs = np.empty((harmonics + 1, len(b_values)), dtype=float)

    for start in range(0, len(b_values), chunk_size):
        stop = min(start + chunk_size, len(b_values))
        q = np.sqrt(1.0 + b_values[start:stop, None] * cos_theta[None, :])
        coeffs[:, start:stop] = (cos_h_theta @ q.T) / quadrature
    return coeffs


def simulate_phase_band_harmonic(
    rng: np.random.Generator,
    times: np.ndarray,
    sample_rate: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Fast structured simulation for the phase-band nonseparable example."""

    n = len(times)
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    psd_f = example_frequency_psd(freqs, args)
    check_positive("S_f(f)", psd_f)
    internal_spectrum = one_sided_to_internal_spectrum(psd_f, sample_rate, args.psd_convention)
    eta = hermitian_unit_fourier_noise(rng, n)
    base = np.sqrt(internal_spectrum) * eta

    f_abs = np.abs(freqs)
    envelope = np.exp(-0.5 * ((f_abs - args.phase_band_fc) / args.phase_band_sigma_f) ** 2)
    b_values = args.mod_amp * envelope
    coeffs = phase_band_sqrt_coefficients(
        b_values,
        args.harmonic_count,
        args.harmonic_quadrature,
        args.harmonic_coeff_chunk_size,
    )
    psi = 2.0 * np.pi * (f_abs - args.phase_band_fc) / args.phase_band_delta_f_phase
    omega_t = 2.0 * np.pi * times / args.mod_period

    output = math.sqrt(n) * inverse_fft(base * coeffs[0], args)
    for h in range(1, args.harmonic_count + 1):
        phase_factor = np.exp(1j * h * psi)
        positive = math.sqrt(n) * inverse_fft(base * coeffs[h] * phase_factor, args)
        negative = math.sqrt(n) * inverse_fft(base * coeffs[h] * np.conj(phase_factor), args)
        output += positive * np.exp(1j * h * omega_t)
        output += negative * np.exp(-1j * h * omega_t)
    return output.real


def simulate_general_numba(
    rng: np.random.Generator,
    times: np.ndarray,
    sample_rate: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Exact direct-sum simulation for built-in examples using numba."""

    if njit is None:
        raise RuntimeError("numba engine requested, but numba is not available")
    n = len(times)
    if n > args.general_max_samples and not args.allow_large_general:
        raise ValueError(
            f"numba direct mode is exact but scales like N^2 and N={n} exceeds "
            f"--general-max-samples={args.general_max_samples}; lower N, pass "
            "--allow-large-general, or use --general-engine=harmonic for phase-band"
        )

    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    psd_f = example_frequency_psd(freqs, args)
    check_positive("S_f(f)", psd_f)
    eta = hermitian_unit_fourier_noise(rng, n)
    dynamic_code = 0 if args.dynamic_example == "separable" else 1
    return _simulate_general_builtin_numba(
        times,
        freqs,
        eta.real,
        eta.imag,
        sample_rate,
        1 if args.psd_convention == "one-sided" else 0,
        dynamic_code,
        args.s0,
        args.f0,
        args.fx,
        args.a,
        args.b,
        args.c,
        args.mod_amp,
        args.mod_period,
        args.phase_band_fc,
        args.phase_band_sigma_f,
        args.phase_band_delta_f_phase,
    )


def simulate_general_slow(
    rng: np.random.Generator,
    times: np.ndarray,
    sample_rate: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Exact but slow simulation for a general dynamic spectrum S(f,t)."""

    n = len(times)
    if n > args.general_max_samples and not args.allow_large_general:
        raise ValueError(
            f"general mode scales like N^2 and N={n} exceeds --general-max-samples="
            f"{args.general_max_samples}; use separable mode, lower N, or pass "
            "--allow-large-general"
        )

    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    eta = hermitian_unit_fourier_noise(rng, n)
    output = np.zeros(n, dtype=np.complex128)

    for start in range(0, n, args.general_chunk_size):
        stop = min(start + args.general_chunk_size, n)
        f_block = freqs[start:stop, None]
        dynamic = example_dynamic_psd(f_block, times[None, :], args)
        check_positive("S(f,t)", dynamic)
        internal = one_sided_to_internal_spectrum(dynamic, sample_rate, args.psd_convention)
        phase = np.exp(2j * np.pi * f_block * times[None, :])
        output += np.sum(np.sqrt(internal) * eta[start:stop, None] * phase, axis=0)

    return (output / math.sqrt(n)).real


def choose_general_engine(args: argparse.Namespace) -> str:
    if args.general_engine != "auto":
        return args.general_engine
    if args.dynamic_example == "phase-band":
        return "harmonic"
    if args.dynamic_example == "separable":
        return "separable-fast"
    return "numpy"


def simulate_general(
    rng: np.random.Generator,
    times: np.ndarray,
    sample_rate: float,
    args: argparse.Namespace,
) -> np.ndarray:
    engine = choose_general_engine(args)
    if engine == "separable-fast":
        return simulate_separable(rng, times, sample_rate, args)
    if engine == "harmonic":
        if args.dynamic_example != "phase-band":
            raise ValueError("--general-engine=harmonic is only implemented for --dynamic-example=phase-band")
        return simulate_phase_band_harmonic(rng, times, sample_rate, args)
    if engine == "numba":
        return simulate_general_numba(rng, times, sample_rate, args)
    if engine == "numpy":
        return simulate_general_slow(rng, times, sample_rate, args)
    raise ValueError(f"unknown general engine {engine}")


def write_time_series(path: Path, times: np.ndarray, noise: np.ndarray, args: argparse.Namespace) -> None:
    header = "\n".join(
        [
            "non-stationary Gaussian noise realization",
            f"mode {args.mode}",
            f"general_engine {choose_general_engine(args) if args.mode == 'general' else 'not_used'}",
            f"fft_backend {args.fft_backend}",
            f"fft_workers {args.fft_workers}",
            f"duration_s {args.duration}",
            f"sample_rate_hz {args.sample_rate}",
            f"n_samples {len(times)}",
            f"seed {args.seed}",
            f"psd_convention {args.psd_convention}",
            f"S0 {args.s0}",
            f"f0_hz {args.f0}",
            f"fx_hz {args.fx}",
            f"a {args.a}",
            f"b {args.b}",
            f"c {args.c}",
            f"modulation_amplitude {args.mod_amp}",
            f"modulation_period_s {args.mod_period}",
            f"dynamic_example {args.dynamic_example}",
            f"phase_band_fc_hz {args.phase_band_fc}",
            f"phase_band_sigma_f_hz {args.phase_band_sigma_f}",
            f"phase_band_delta_f_phase_hz {args.phase_band_delta_f_phase}",
            f"harmonic_count {args.harmonic_count}",
            f"harmonic_quadrature {args.harmonic_quadrature}",
            "columns time_s noise",
        ]
    )
    np.savetxt(path, np.column_stack((times, noise)), header=header)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate time-domain non-stationary Gaussian noise."
    )
    parser.add_argument("--mode", choices=("separable", "general"), default="separable")
    parser.add_argument(
        "--dynamic-example",
        choices=("separable", "phase-band"),
        default="phase-band",
        help="dynamic-spectrum example used by --mode=general",
    )
    parser.add_argument("--duration", type=positive_float, default=256.0, help="duration in seconds")
    parser.add_argument("--sample-rate", type=positive_float, default=2048.0, help="sample rate in Hz")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--output", type=Path, default=Path("nonstationary_noise.dat"))
    parser.add_argument(
        "--psd-convention",
        choices=("one-sided", "internal"),
        default="one-sided",
        help="interpret example PSD as a one-sided PSD in strain^2/Hz or as the internal DFT-bin spectrum",
    )
    parser.add_argument(
        "--fft-backend",
        choices=("auto", "numpy", "scipy"),
        default="auto",
        help="FFT backend for fast paths; auto prefers scipy when available",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=-1,
        help="worker count for scipy.fft; ignored by numpy",
    )

    parser.add_argument("--s0", type=positive_float, default=1.0e-46)
    parser.add_argument("--f0", type=positive_float, default=150.0)
    parser.add_argument("--fx", type=nonnegative_float, default=10.0)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--mod-amp", type=float, default=0.25)
    parser.add_argument("--mod-period", type=positive_float, default=30.0)
    parser.add_argument(
        "--phase-band-fc",
        type=positive_float,
        default=150.0,
        help="center frequency in Hz for the nonseparable phase-band modulation",
    )
    parser.add_argument(
        "--phase-band-sigma-f",
        type=positive_float,
        default=50.0,
        help="Gaussian frequency width in Hz for the nonseparable phase-band modulation",
    )
    parser.add_argument(
        "--phase-band-delta-f-phase",
        type=positive_float,
        default=120.0,
        help="frequency interval in Hz over which the modulation phase changes by 2 pi",
    )

    parser.add_argument(
        "--general-max-samples",
        type=int,
        default=8192,
        help="safety limit for exact numpy/numba direct modes unless --allow-large-general is set",
    )
    parser.add_argument(
        "--general-engine",
        choices=("auto", "numpy", "numba", "harmonic", "separable-fast"),
        default="auto",
        help="engine for --mode=general; auto uses harmonic for phase-band",
    )
    parser.add_argument("--general-chunk-size", type=int, default=64)
    parser.add_argument(
        "--harmonic-count",
        type=int,
        default=8,
        help="number of positive harmonics for the phase-band harmonic engine",
    )
    parser.add_argument(
        "--harmonic-quadrature",
        type=int,
        default=256,
        help="quadrature samples used to compute sqrt-modulation Fourier coefficients",
    )
    parser.add_argument(
        "--harmonic-coeff-chunk-size",
        type=int,
        default=4096,
        help="frequency bins per chunk when computing harmonic coefficients",
    )
    parser.add_argument("--allow-large-general", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(args.mod_amp) >= 1.0:
        raise ValueError("--mod-amp must have absolute value less than one")
    if args.general_max_samples <= 0:
        raise ValueError("--general-max-samples must be positive")
    if args.general_chunk_size <= 0:
        raise ValueError("--general-chunk-size must be positive")
    if args.harmonic_count < 0:
        raise ValueError("--harmonic-count must be non-negative")
    if args.harmonic_quadrature <= 2 * args.harmonic_count:
        raise ValueError("--harmonic-quadrature should exceed 2 * --harmonic-count")
    if args.harmonic_coeff_chunk_size <= 0:
        raise ValueError("--harmonic-coeff-chunk-size must be positive")

    times, _ = build_time_grid(args.duration, args.sample_rate)
    rng = np.random.default_rng(args.seed)

    if args.mode == "separable":
        noise = simulate_separable(rng, times, args.sample_rate, args)
    else:
        noise = simulate_general(rng, times, args.sample_rate, args)

    write_time_series(args.output, times, noise, args)
    print(f"wrote {len(times)} samples to {args.output}")
    if args.mode == "general":
        print(f"general engine: {choose_general_engine(args)}")
    print(f"mean={np.mean(noise):.6e}, std={np.std(noise):.6e}")


if __name__ == "__main__":
    main()
