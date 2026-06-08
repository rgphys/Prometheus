"""
Shot Noise Extension for Prometheus
====================================

Provides signal-to-noise ratio (SNR) estimation and photon shot-noise
injection for Prometheus transmission spectra.  Works with any simulation
output — no assumptions are made about the absorbing species, wavelength
regime, or atmospheric scenario.

Three SNR "source" modes are supported:

1. **Constant** — A single SNR/bin value applied uniformly across the
   wavelength grid (e.g. from an ETC run at target conditions).

2. **Tabulated** — A wavelength-dependent SNR table (e.g. from an ETC PDF)
   that is linearly interpolated onto the simulation grid and then scaled
   to account for differences in target magnitude and exposure time via
   the photon-noise scaling law.

3. **High-resolution JSON** — A per-pixel SNR curve from an ETC JSON
   export (e.g. ESO UVES ETC).  Parsed, interpolated, and scaled in the
   same way as mode 2.

4. **CSV** — A two-column CSV file (wavelength in nm, SNR) used directly
   without any scaling.  The SNR curve is linearly interpolated onto the
   simulation grid.

The module is intentionally decoupled from the Prometheus simulation
classes: it operates on plain NumPy arrays of wavelength and flux so
that it can be slotted into any post-processing pipeline.

Usage overview
--------------
>>> from Prometheus.pythonScripts.shotNoise import SNRModel, apply_shot_noise
>>>
>>> snr_model = SNRModel.constant(snr_per_bin=847.0)
>>> noisy_spec, sigma = apply_shot_noise(wavelength, spectrum, snr_model)

Calculators can be found here: https://www.eso.org/observing/etc/
"""

import csv
import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# SNR Scaling Law
# ──────────────────────────────────────────────────────────────────────────────
def scale_snr(baseline_snr: float,
              baseline_mag: float,
              baseline_time_hrs: float,
              target_mag: float,
              transit_duration_hrs: float,
              num_bins: int) -> float:
    """Scale a baseline ETC SNR to per-bin observing conditions.

    Applies the standard photon shot-noise scaling law:

        SNR_target = SNR_baseline * √(F_target / F_baseline) * √(t_bin / t_baseline)

    where the flux ratio is derived from the magnitude difference and the
    per-bin exposure time is ``transit_duration_hrs / num_bins``.

    Parameters
    ----------
    baseline_snr : float
        SNR measured (or reported) by the ETC at baseline conditions.
    baseline_mag : float
        Magnitude of the reference source used in the ETC run.
    baseline_time_hrs : float
        Exposure time of the ETC run, in hours.
    target_mag : float
        Magnitude of the actual science target.
    transit_duration_hrs : float
        Total in-transit duration, in hours.
    num_bins : int
        Number of orbital-phase bins the transit is divided into.

    Returns
    -------
    float
        Scaled SNR per wavelength bin per orbital-phase bin.
    """
    flux_ratio = 10 ** ((baseline_mag - target_mag) / 2.5)
    time_ratio = (transit_duration_hrs / num_bins) / baseline_time_hrs
    return baseline_snr * math.sqrt(flux_ratio) * math.sqrt(time_ratio)


# ──────────────────────────────────────────────────────────────────────────────
# SNR Model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TransitParams:
    """Observing parameters needed by the SNR scaling law.

    Only required when the SNR source is a tabulated or JSON curve that
    must be rescaled to actual target conditions.  Ignored for the
    *constant* SNR mode.

    Attributes
    ----------
    target_mag : float
        Apparent magnitude of the science target in the relevant band.
    transit_duration_hrs : float
        Total in-transit duration in hours.
    num_bins : int
        Number of orbital-phase bins.
    """
    target_mag: float
    transit_duration_hrs: float
    num_bins: int


@dataclass
class SNRModel:
    """Wavelength-dependent (or constant) SNR model.

    Do not instantiate directly — use the three class-method constructors
    :meth:`constant`, :meth:`from_table`, or :meth:`from_uves_json`.

    Attributes
    ----------
    mode : str
        One of ``"constant"``, ``"table"``, or ``"json"``.
    _snr_value : float or None
        Uniform SNR (mode ``"constant"`` only).
    _wav_nm, _snr_arr : ndarray or None
        Sorted reference SNR curve (modes ``"table"`` and ``"json"``).
    _baseline_mag, _baseline_time_hrs : float or None
        ETC reference conditions for scaling.
    transit_params : TransitParams or None
        Target observing conditions (required for ``"table"``/``"json"``).
    """
    mode: str
    _snr_value: Optional[float] = None
    _wav_nm: Optional[np.ndarray] = field(default=None, repr=False)
    _snr_arr: Optional[np.ndarray] = field(default=None, repr=False)
    _baseline_mag: Optional[float] = None
    _baseline_time_hrs: Optional[float] = None
    transit_params: Optional[TransitParams] = None

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def constant(cls, snr_per_bin: float) -> "SNRModel":
        """Create a model with a single, wavelength-independent SNR.

        Parameters
        ----------
        snr_per_bin : float
            Signal-to-noise ratio per wavelength bin.

        Returns
        -------
        SNRModel
        """
        return cls(mode="constant", _snr_value=snr_per_bin)

    @classmethod
    def from_table(cls,
                   wav_nm: list | np.ndarray,
                   snr: list | np.ndarray,
                   baseline_mag: float,
                   baseline_time_hrs: float,
                   transit_params: TransitParams) -> "SNRModel":
        """Create a model from a wavelength–SNR table.

        The table is linearly interpolated onto the simulation grid and
        then rescaled via :func:`scale_snr`.

        Parameters
        ----------
        wav_nm : array-like
            Reference wavelengths in **nanometres**.
        snr : array-like
            Corresponding baseline SNR values.
        baseline_mag : float
            Magnitude of the ETC reference source.
        baseline_time_hrs : float
            ETC exposure time in hours.
        transit_params : TransitParams
            Actual observing conditions for rescaling.

        Returns
        -------
        SNRModel
        """
        idx = np.argsort(wav_nm)
        return cls(
            mode="table",
            _wav_nm=np.asarray(wav_nm, dtype=float)[idx],
            _snr_arr=np.asarray(snr, dtype=float)[idx],
            _baseline_mag=baseline_mag,
            _baseline_time_hrs=baseline_time_hrs,
            transit_params=transit_params,
        )

    @classmethod
    def from_json(cls,
                       json_path: str,
                       transit_params: TransitParams,
                       correction_factor: float = 1.0) -> "SNRModel":
        """Create a model from a UVES ETC JSON export.

        The JSON is expected to follow the ESO ETC v2 output schema::

            data.orders[].detectors[].wavelength   (metres)
            data.orders[].detectors[].plots.snr.snr
            input.target.brightness.mag
            input.timesnr.DET1.WIN1.UIT1           (seconds)

        Parameters
        ----------
        json_path : str
            Path to the JSON file.
        transit_params : TransitParams
            Actual observing conditions for rescaling.
        correction_factor : float, optional
            Multiplicative correction applied to all SNR values (e.g.
            telescope area ratio).  Defaults to 1.0 (no correction).

        Returns
        -------
        SNRModel
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        all_wav, all_snr = [], []
        for order in data["data"]["orders"]:
            det = order["detectors"][0]
            wav_m = np.array(det["wavelength"])
            snr_vals = np.array(det["plots"]["snr"]["snr"])
            valid = (snr_vals > 0) & np.isfinite(snr_vals)
            all_wav.extend((wav_m[valid] * 1e9).tolist())
            all_snr.extend((snr_vals[valid] * correction_factor).tolist())

        idx = np.argsort(all_wav)
        baseline_mag = data["input"]["target"]["brightness"]["mag"]
        baseline_time = data["input"]["timesnr"]["DET1.WIN1.UIT1"] / 3600.0

        return cls(
            mode="json",
            _wav_nm=np.array(all_wav)[idx],
            _snr_arr=np.array(all_snr)[idx],
            _baseline_mag=baseline_mag,
            _baseline_time_hrs=baseline_time,
            transit_params=transit_params,
        )

    @classmethod
    def from_csv(cls, csv_path: str) -> "SNRModel":
        """Create a model from a two-column CSV file.

        The CSV must have columns for wavelength (nm) and SNR, with no
        scaling applied — the SNR values are used directly.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.  Expected columns: wavelength (nm), SNR.

        Returns
        -------
        SNRModel
        """
        wav, snr = [], []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                try:
                    w, s = float(row[0]), float(row[1])
                except (ValueError, IndexError):
                    continue  # skip header or malformed rows
                wav.append(w)
                snr.append(s)

        idx = np.argsort(wav)
        return cls(
            mode="csv",
            _wav_nm=np.array(wav)[idx],
            _snr_arr=np.array(snr)[idx],
        )

    # ── Evaluation ───────────────────────────────────────────────────────

    def snr_at(self, wavelength_nm: float) -> float:
        """Return the SNR per bin at a single wavelength.

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometres.

        Returns
        -------
        float
            SNR per bin.
        """
        if self.mode == "constant":
            return self._snr_value

        if self.mode == "csv":
            return float(np.interp(wavelength_nm, self._wav_nm, self._snr_arr))

        baseline_snr = float(np.interp(wavelength_nm, self._wav_nm, self._snr_arr))
        return scale_snr(
            baseline_snr=baseline_snr,
            baseline_mag=self._baseline_mag,
            baseline_time_hrs=self._baseline_time_hrs,
            target_mag=self.transit_params.target_mag,
            transit_duration_hrs=self.transit_params.transit_duration_hrs,
            num_bins=self.transit_params.num_bins,
        )

    def snr_array(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Return an SNR array matching an entire wavelength grid.

        Parameters
        ----------
        wavelength_nm : ndarray
            Wavelength grid in nanometres.

        Returns
        -------
        ndarray
            SNR per bin at each wavelength point.
        """
        if self.mode == "constant":
            return np.full_like(wavelength_nm, self._snr_value, dtype=float)

        if self.mode == "csv":
            return np.interp(wavelength_nm, self._wav_nm, self._snr_arr)

        baseline_snr = np.interp(wavelength_nm, self._wav_nm, self._snr_arr)
        tp = self.transit_params
        flux_ratio = 10 ** ((self._baseline_mag - tp.target_mag) / 2.5)
        time_ratio = (tp.transit_duration_hrs / tp.num_bins) / self._baseline_time_hrs
        return baseline_snr * math.sqrt(flux_ratio) * math.sqrt(time_ratio)


# ──────────────────────────────────────────────────────────────────────────────
# Noise Application
# ──────────────────────────────────────────────────────────────────────────────
def sigma_from_snr(snr: float | np.ndarray) -> np.ndarray:
    """Convert SNR to 1-sigma fractional noise level.

    Parameters
    ----------
    snr : float or ndarray
        Signal-to-noise ratio (must be > 0).

    Returns
    -------
    ndarray
        1-sigma noise amplitude (dimensionless, same units as the flux).
    """
    return np.where(np.asarray(snr) > 0, 1.0 / np.asarray(snr), 0.0)


def apply_shot_noise(wavelength_nm: np.ndarray,
                     spectrum: np.ndarray,
                     snr_model: SNRModel,
                     seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian shot noise to a transmission spectrum.

    Parameters
    ----------
    wavelength_nm : ndarray
        Wavelength grid in nanometres.
    spectrum : ndarray
        Clean transmission spectrum (dimensionless, typically close to 1).
    snr_model : SNRModel
        Noise model that provides SNR as a function of wavelength.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noisy_spectrum : ndarray
        Spectrum with Gaussian noise added.
    sigma : ndarray
        1-sigma noise level at each wavelength point (useful for error
        bars and confidence bands).
    """
    rng = np.random.default_rng(seed)
    sigma = sigma_from_snr(snr_model.snr_array(wavelength_nm))
    noise = rng.normal(0.0, sigma)
    return spectrum + noise, sigma
