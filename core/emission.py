"""
Emission physics for Prometheus.

The base radiative transfer in :mod:`gasProperties` is pure extinction: every
chord is attenuated by Beer-Lambert, ``I = I_star * exp(-tau)``.  That is the
correct answer only when the gas is a perfect absorber.  Real exospheric gas
also *puts photons back* into the line of sight, by two mechanisms:

1. **Resonance scattering of starlight.**  An atom in a resonance line (Na D,
   K D, Mg, Ca II, ...) absorbs a stellar photon and re-emits it in a random
   direction.  A fraction of those photons lands in the observer's beam.  This
   is the mechanism that makes Io's sodium cloud glow, and it is the reason a
   transiting exomoon cloud shows *less* absorption than the pure-extinction
   answer (the scattered photons partially refill the line core) while an
   off-limb cloud shows net *emission* against the dark sky.

2. **Thermal (LTE) emission.**  Any opacity in a gas at temperature ``T`` also
   emits, with source function ``B_lambda(T)`` (Kirchhoff's law).  This matters
   for the dense, warm parts of a hydrostatic atmosphere and for molecular
   bands.

This module holds the *physics primitives* (source functions, geometric
dilution, the g-factor); the line-of-sight integration that uses them lives in
``Atmosphere.getLOSopticalDepthAndEmission_Batch``.

Formal solution
---------------
For a chord running along ``+x`` with the observer at ``x = -inf``, the
emergent specific intensity is

    I(lambda) = I_star(lambda) * exp(-tau_total)
                + sum_i  j_i(lambda) * dx * exp(-tau_i->obs)

where ``j`` is the volume emissivity [erg s^-1 cm^-3 sr^-1 cm^-1] and
``tau_i->obs`` is the optical depth between cell ``i`` and the observer.  The
first term is what Prometheus computed before; this module supplies ``j``.

Source functions
----------------
*Resonance scattering* (single scattering, isotropic phase function):

    j_scat(lambda) = n_abs * sigma(lambda) * W(r) * I_star(lambda)

``W(r) = Omega_star / 4pi`` is the geometric dilution factor, i.e. the mean
intensity at the parcel is ``J = W * I_star``, and the scattering emissivity of
isotropically re-radiating material is ``j = n * sigma * J``.  Note this uses
*exactly the same* ``sigma(lambda)`` as the absorption path, so the emission is
automatically consistent with the extinction and carries the same Voigt shape,
the same Doppler shifts, and the same line list.

*Thermal emission* (LTE):

    j_therm(lambda) = n_abs * sigma(lambda) * B_lambda(T)

Approximations, stated explicitly
---------------------------------
* **Single scattering.**  Photons are scattered at most once.  Valid while the
  line-centre optical depth of the cloud is <~ 1, which is the regime of every
  exospheric cloud model in this repository.  Multiple scattering would
  *increase* the emission, so the single-scattering answer is a lower bound on
  the fill-in.
* **Isotropic phase function.**  Resonance scattering of a J=1/2 -> 3/2 line is
  very nearly isotropic; for the dipole (J=1/2 -> 1/2) component the true phase
  function deviates by at most ~10% from isotropic.
* **Coherent scattering in the observer frame.**  The re-emitted wavelength is
  taken to equal the absorbed wavelength in the observer frame.  This is
  *exact* in the forward-scattering limit (star directly behind the parcel,
  observer directly in front), because then the line-of-sight projection of the
  scattering atom's velocity is identical for the incoming and outgoing photon.
  That is precisely the in-transit geometry, so the in-transit fill-in is
  handled exactly.  For large scattering angles (an off-limb cloud seen at
  quadrature) the frequency redistribution across the thermal width is
  approximate at the level of the Doppler width itself.
* **No self-shielding of the incident beam.**  ``I_star`` illuminating a parcel
  is not attenuated by intervening gas.  Same optically-thin justification as
  single scattering; ``EmissionModel.self_shielding`` is reserved for a future
  implementation and currently must be ``False``.

Created 2026-09-11.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import constants as const


# --------------------------------------------------------------------------
# Source functions
# --------------------------------------------------------------------------

def planck_lambda(wavelength: np.ndarray, T: Any) -> np.ndarray:
    """Planck function ``B_lambda(T)`` in cgs.

    Args:
        wavelength (np.ndarray): Wavelength [cm], any shape.
        T: Temperature [K].  Scalar, or an array broadcastable against
            ``wavelength``.

    Returns:
        np.ndarray: Specific intensity [erg s^-1 cm^-2 cm^-1 sr^-1].
    """
    wav = np.asarray(wavelength, dtype=float)
    T_ = np.asarray(T, dtype=float)
    # expm1 keeps the Rayleigh-Jeans limit accurate.
    x = const.h * const.c / (wav * const.k_B * np.maximum(T_, 1e-30))
    return (2.0 * const.h * const.c ** 2 / wav ** 5) / np.expm1(np.minimum(x, 700.0))


def dilution_factor(r: np.ndarray, R_star: float) -> np.ndarray:
    """Geometric dilution factor ``W = Omega_star / (4 pi)`` at distance ``r``.

    The mean intensity of starlight at a parcel a distance ``r`` from the centre
    of a uniformly bright sphere of radius ``R_star`` is ``J = W * I_star``, with

        W(r) = 0.5 * (1 - sqrt(1 - (R_star/r)^2)),

    which tends to the familiar ``R_star^2 / (4 r^2)`` for ``r >> R_star`` and
    to 1/2 at the stellar surface.  Inside the star the formula is clamped to
    1/2; such cells are unphysical anyway and carry no gas in practice.

    Args:
        r (np.ndarray): Distance from the stellar centre [cm], any shape.
        R_star (float): Stellar radius [cm].

    Returns:
        np.ndarray: Dilution factor, dimensionless, in [0, 0.5].
    """
    r_ = np.asarray(r, dtype=float)
    ratio = np.clip(R_star / np.maximum(r_, R_star), 0.0, 1.0)
    return 0.5 * (1.0 - np.sqrt(1.0 - ratio ** 2))


# --------------------------------------------------------------------------
# Stellar illumination
# --------------------------------------------------------------------------

class StellarIntensity:
    """The stellar surface intensity in *physical* cgs units.

    Prometheus' internal ``Fstar`` is either a PHOENIX surface intensity (real
    cgs) or a flat ``1.0`` placeholder, because only the ratio ``F_in / F_out``
    matters for a transmission spectrum.  Emission breaks that scale invariance:
    the emissivity is an absolute number of erg, so it has to be compared
    against an absolute stellar intensity.  This class provides that, and also
    the conversion factor back into whatever internal units the transit
    normalisation is using.

    Args:
        star (Any): A :class:`celestialBodies.Star`.
        wavelength (np.ndarray): The simulation wavelength grid [cm]; used only
            for the blackbody fallback bookkeeping.
        disk_average (bool): Multiply by the limb-darkening disk-average factor
            ``1 - u1/3 - u2/6`` so that the intensity used to *illuminate* the
            gas is the disk-averaged one rather than the disk-centre value.

    Attributes:
        is_phoenix (bool): True if a PHOENIX spectrum is attached to the star.
        internal_scale (np.ndarray): Multiply a physical cgs intensity by this
            to express it in the same units as ``Transit``'s ``F_out``.
    """

    def __init__(self, star: Any, wavelength: np.ndarray,
                 disk_average: bool = True):
        self.star = star
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.is_phoenix = getattr(star, 'Fstar_function', None) is not None

        u1 = getattr(star, 'CLV_u1', 0.0) or 0.0
        u2 = getattr(star, 'CLV_u2', 0.0) or 0.0
        self.disk_factor = (1.0 - u1 / 3.0 - u2 / 6.0) if disk_average else 1.0

        if self.is_phoenix:
            self._x = star.Fstar_function.x
            self._y = star.Fstar_function.y
            # Internal units already are physical cgs surface intensity.
            self.internal_scale = np.ones_like(self.wavelength)
        else:
            # Flat star: the internal unit is "disk-centre intensity = 1", so a
            # physical intensity has to be divided by the physical disk-centre
            # intensity.  ASSUMPTION (not a measurement): the photosphere
            # radiates as a blackbody at T_eff.
            self._x = None
            self._y = None
            I_ref = planck_lambda(self.wavelength, star.T_eff)
            self.internal_scale = 1.0 / I_ref

    def __call__(self, wavelength: np.ndarray) -> np.ndarray:
        """Surface intensity [erg s^-1 cm^-2 cm^-1 sr^-1] at given wavelengths.

        Args:
            wavelength (np.ndarray): Wavelength [cm].  The last axis must be
                monotonically non-decreasing (true of every Doppler-shifted
                grid in Prometheus) when a PHOENIX spectrum is used.

        Returns:
            np.ndarray: Intensity, same shape as ``wavelength``.
        """
        if self.is_phoenix:
            # Local import: avoids a circular import at module load time.
            from .gasProperties import n_interp_log
            I = n_interp_log(np.ascontiguousarray(wavelength, dtype=float),
                             self._x, self._y, 0.0)
        else:
            I = planck_lambda(wavelength, self.star.T_eff)
        return I * self.disk_factor


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class EmissionModel:
    """Which emission terms to include in the line-of-sight integration.

    Passing an instance of this to :class:`gasProperties.Atmosphere` switches
    the transit from pure Beer-Lambert extinction to the full formal solution
    with a source term.  Leaving it as ``None`` reproduces the previous
    behaviour bit-for-bit.

    Attributes:
        resonant_scattering (bool): Single scattering of starlight by the
            atomic/ionic line opacity.  This is the exomoon-cloud term.
        thermal (bool): LTE thermal emission, ``j = n sigma B_lambda(T)``.
            Only density models that carry a temperature (the
            ``CollisionalAtmosphere`` family) contribute.
        molecular (bool): Include molecular constituents in the emission terms
            as well as in the extinction.  Off by default because it forces the
            expensive per-cell molecular interpolation.
        aerosol_scattering (bool): Single scattering of starlight by aerosol /
            haze opacity, with an isotropic phase function and a grey single-
            scattering albedo.  Off by default: real aerosols are strongly
            forward-scattering, so the isotropic assumption is much weaker here
            than it is for a resonance line.
        aerosol_albedo (float): Single-scattering albedo of the aerosol
            opacity, splitting its extinction into a scattering share
            ``albedo`` and a true-absorption share ``1 - albedo``.  The
            scattering share redirects starlight; the absorption share emits
            thermally under ``thermal``, as Kirchhoff's law requires.  The
            default 1.0 is a pure scatterer, which neither absorbs nor emits;
            set it to 0.0 for a purely absorbing grey opacity.  A modelling
            ASSUMPTION, not a measurement.  Atomic, ionic and molecular line
            opacity is always treated as pure absorption.
        stellar_doppler (bool): Sample the stellar spectrum at the wavelength
            seen in the *parcel's* frame, i.e. Doppler-shifted by the parcel's
            radial velocity with respect to the star.  This is what makes gas
            sitting in a deep stellar Fraunhofer core scatter weakly and gas
            moving a few km/s out of the core scatter strongly.  Has no effect
            for a flat/blackbody star (featureless illumination).
        self_shielding (bool): Attenuate the incident starlight along the
            star-to-parcel path.  Not implemented; must be False.
    """

    resonant_scattering: bool = True
    thermal: bool = False
    molecular: bool = False
    aerosol_scattering: bool = False
    aerosol_albedo: float = 1.0
    stellar_doppler: bool = True
    self_shielding: bool = False

    def __post_init__(self):
        if self.self_shielding:
            raise NotImplementedError(
                "Self-shielding of the incident stellar beam is not "
                "implemented; the emission kernel assumes an optically thin "
                "illumination path.")
        if not (self.resonant_scattering or self.thermal
                or self.aerosol_scattering):
            raise ValueError(
                "EmissionModel has no active source term; pass emission=None "
                "for a pure-extinction transit.")
        if not 0.0 <= self.aerosol_albedo <= 1.0:
            raise ValueError("aerosol_albedo must lie in [0, 1].")

    @property
    def needs_atoms(self) -> bool:
        """True if atomic/ionic constituents carry a source term."""
        return self.resonant_scattering or self.thermal

    @property
    def needs_aerosols(self) -> bool:
        """True if scattering constituents carry a source term."""
        return self.aerosol_scattering


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def g_factor(constituent: Any, star: Any, r: float,
             wavelength: Optional[np.ndarray] = None,
             v_radial: float = 0.0) -> float:
    """Photon scattering rate per atom (the ``g``-factor) [photons s^-1].

    This is a *diagnostic*: the line-of-sight integration never needs it, since
    it works directly with ``sigma(lambda) * J(lambda)``.  It is the standard
    number quoted for resonance-scattering clouds (e.g. Io's sodium), and is
    useful for comparing a model against the planetary-science literature.

    It is computed by integrating the constituent's own absorption cross
    section against the diluted stellar photon flux,

        g = int sigma(lambda) * 4 pi J(lambda) * lambda / (h c) dlambda,

    so it inherits the same line list, oscillator strengths and Voigt profile as
    the rest of the simulation.  No literature value is assumed anywhere.

    Args:
        constituent (Any): An :class:`gasProperties.AtmosphericConstituent`
            with its lookup function already attached.
        star (Any): The :class:`celestialBodies.Star` providing the
            illumination (PHOENIX if attached, blackbody at ``T_eff``
            otherwise).
        r (float): Distance of the parcel from the stellar centre [cm].
        wavelength (Optional[np.ndarray]): Integration grid [cm].  Defaults to
            the grid the constituent's cross-section lookup was built on, which
            already resolves every line.
        v_radial (float): Radial velocity of the parcel away from the star
            [cm/s].  Positive = receding, which blueshifts the part of the
            stellar spectrum the parcel samples.

    Returns:
        float: The g-factor in photons per second per atom.
    """
    if wavelength is None:
        wavelength = np.asarray(constituent.lookupFunction.x, dtype=float)
    wavelength = np.sort(np.asarray(wavelength, dtype=float))

    sigma = constituent.getSigmaAbs(wavelength[np.newaxis, :])[0]
    I_star = StellarIntensity(star, wavelength)
    lam_star = wavelength * const.calculateDopplerShift(v_radial)
    J = dilution_factor(r, star.R) * I_star(lam_star[np.newaxis, :])[0]

    photon_flux = 4.0 * np.pi * J * wavelength / (const.h * const.c)
    return float(np.trapezoid(sigma * photon_flux, wavelength))
