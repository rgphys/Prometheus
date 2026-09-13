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
For a chord running along ``x`` with the observer at ``x = +inf`` (the
Prometheus convention: mid-transit puts the planet at ``x = +a``), the emergent
specific intensity is

    I(lambda) = I_star(lambda) * exp(-tau_total)
                + sum_i  j_i(lambda) * dx * exp(-tau_i->obs)

where ``j`` is the volume emissivity [erg s^-1 cm^-3 sr^-1 cm^-1] and
``tau_i->obs`` is the optical depth between cell ``i`` and the observer, i.e.
summed over the cells at larger ``x``.  The
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

When both are active, line opacity uses the two-level-atom source function
``S = (1 - eps) J + eps B`` with an explicit thermalisation probability
``eps`` (:attr:`EmissionModel.line_thermalisation`); the two terms are never
simply added, which would exceed both physical limits.  The complete rules,
including the aerosol albedo split, live in :func:`source_weights`.

Validation status
-----------------
Validated against real data: the bare-surface eclipse path (Planck source,
stellar normalisation, band averaging) on the 55 Cnc e MIRI spectrum and on
TRAPPIST-1 b/c and LHS 3844 b (``Tests/EmissionCalibration``,
``Tests/EmissionReleaseCheck``).  Validated by internal consistency only
(Kirchhoff's law, sign tests, solver agreement): thermal emission from gas,
molecular emission, and **resonance scattering**, for which no published
measurement isolating the term has yet been compared.

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

import warnings
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

class BlackbodyStarWarning(UserWarning):
    """Emission or an eclipse is using a blackbody at ``T_eff`` as the star.

    Not an error: it is a stated modelling assumption.  But eclipse depth scales
    as ``1 / I_star``, and real photospheres depart strongly from a blackbody in
    the infrared (measured: 55 Cnc 25% too bright at 7-12 um, TRAPPIST-1 ~50%
    too bright at 15 um, biasing a recovered brightness temperature by up to
    ~100 K; see ``Tests/EmissionReleaseCheck``).  Silence it deliberately with
    ``warnings.simplefilter('ignore', BlackbodyStarWarning)``.
    """


def check_spectrum_coverage(star: Any, wavelength: np.ndarray) -> None:
    """Raise if the star's attached spectrum does not cover ``wavelength``.

    The interpolator clamps to the edge value outside its range, which would
    silently turn a truncated spectrum (e.g. PHOENIX HiRes, which stops near
    5.5 um) into a flat, wrong star.

    Args:
        star (Any): A :class:`celestialBodies.Star` with ``Fstar_function`` set.
        wavelength (np.ndarray): Wavelengths that will be evaluated [cm].

    Raises:
        ValueError: If any wavelength lies outside the tabulated range.
    """
    x = np.asarray(star.Fstar_function.x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    w = np.asarray(wavelength, dtype=float)
    tol = 1e-9 * hi
    if w.min() < lo - tol or w.max() > hi + tol:
        raise ValueError(
            f"The stellar spectrum covers {lo * 1e4:.4f}-{hi * 1e4:.4f} um but "
            f"{w.min() * 1e4:.4f}-{w.max() * 1e4:.4f} um is requested.  Outside "
            "its range the spectrum would be clamped to its edge value.  Attach "
            "a spectrum that covers the grid (Star.addFstarFunctionFromArrays "
            "with BT-Settl/ATLAS or a measured spectrum).")


class StellarIntensity:
    """The disk-averaged stellar surface intensity in *physical* cgs units.

    Prometheus' internal ``Fstar`` is either a PHOENIX spectrum (surface flux
    divided by pi, see :meth:`celestialBodies.Star.getSpectrum`) or a flat
    ``1.0`` placeholder, because only the ratio ``F_in / F_out`` matters for a
    transmission spectrum.  Emission breaks that scale invariance: the
    emissivity is an absolute number of erg, so it has to be compared against
    an absolute stellar intensity.  This class provides that, and the factor
    that converts it into the units of ``Transit``'s ``F_out``.

    Two quantities are kept strictly apart:

    * The **physical** intensity returned by ``__call__`` is the disk-averaged
      surface intensity ``F_surface / pi``.  That is what illuminates a distant
      parcel (``J = W * I``) and what normalises an eclipse depth
      (``F_star = pi R_star^2 * I``).  Limb darkening does not change it: the
      stellar flux is fixed and limb darkening only redistributes it over the
      disk.
    * The **internal** unit of the transit chord sum.  ``Transit`` treats
      ``Fstar`` as the *disk-centre* intensity and multiplies it by the CLV
      profile, so its disk-integrated ``F_out`` is ``pi R^2 * Fstar * D``, with
      ``D`` the disk-average of the CLV profile.  ``internal_scale`` carries
      that factor ``D`` so emission lands in the same units as ``F_out``.

    Args:
        star (Any): A :class:`celestialBodies.Star`.
        wavelength (np.ndarray): The simulation wavelength grid [cm].  An
            attached spectrum must cover it (``ValueError`` otherwise); with no
            spectrum a :class:`BlackbodyStarWarning` is issued.

    Attributes:
        is_tabulated (bool): True if a stellar spectrum (PHOENIX, or one
            attached with ``addFstarFunctionFromArrays``) is attached.
        disk_factor (np.ndarray): Disk average of the CLV profile the transit
            chord sum uses, ``2 * int_0^1 CLV(mu) mu dmu``, per wavelength.
        internal_scale (np.ndarray): Multiply a physical cgs intensity by this
            to express it in the same units as ``Transit``'s ``F_out``.
    """

    #: Gauss-Legendre nodes for the disk average of a tabulated CLV profile.
    _N_MU = 64

    def __init__(self, star: Any, wavelength: np.ndarray):
        self.star = star
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.is_tabulated = getattr(star, 'Fstar_function', None) is not None
        self.disk_factor = self._clvDiskFactor(star, self.wavelength)

        if self.is_tabulated:
            check_spectrum_coverage(star, self.wavelength)
            self._x = star.Fstar_function.x
            self._y = star.Fstar_function.y
            # Internal Fstar is the tabulated intensity itself.
            self.internal_scale = self.disk_factor
        else:
            warnings.warn(
                f"No stellar spectrum attached: the star is modelled as a "
                f"blackbody at T_eff = {star.T_eff:.0f} K.  In the infrared "
                "this can bias eclipse depths by tens of percent; attach a "
                "spectrum with Star.addFstarFunction or "
                "Star.addFstarFunctionFromArrays.",
                BlackbodyStarWarning, stacklevel=3)
            # Flat star: internal Fstar = 1 at disk centre.  ASSUMPTION (not a
            # measurement): the photosphere radiates as a blackbody at T_eff,
            # i.e. a disk-averaged intensity B_lambda(T_eff).
            self._x = None
            self._y = None
            self.internal_scale = self.disk_factor / planck_lambda(
                self.wavelength, star.T_eff)

    @classmethod
    def _clvDiskFactor(cls, star: Any, wavelength: np.ndarray) -> np.ndarray:
        """Disk average of the CLV profile used by the transit chord sum."""
        clv_function = getattr(star, 'CLV_function', None)
        if clv_function is None:
            u1 = getattr(star, 'CLV_u1', 0.0) or 0.0
            u2 = getattr(star, 'CLV_u2', 0.0) or 0.0
            return np.full_like(wavelength, 1.0 - u1 / 3.0 - u2 / 6.0)
        nodes, weights = np.polynomial.legendre.leggauss(cls._N_MU)
        mu = 0.5 * (nodes + 1.0)
        profile = np.asarray(clv_function(mu, wavelength), dtype=float)
        return 2.0 * np.sum((0.5 * weights * mu)[:, np.newaxis] * profile,
                            axis=0)

    def __call__(self, wavelength: np.ndarray) -> np.ndarray:
        """Disk-averaged surface intensity [erg s^-1 cm^-2 cm^-1 sr^-1].

        Args:
            wavelength (np.ndarray): Wavelength [cm].  The last axis must be
                monotonically non-decreasing (true of every Doppler-shifted
                grid in Prometheus) when a tabulated spectrum is used.

        Returns:
            np.ndarray: Intensity, same shape as ``wavelength``.
        """
        if self.is_tabulated:
            # Local import: avoids a circular import at module load time.
            from .gasProperties import n_interp_log
            return n_interp_log(np.ascontiguousarray(wavelength, dtype=float),
                                self._x, self._y, 0.0)
        return planck_lambda(wavelength, self.star.T_eff)

    @property
    def is_phoenix(self) -> bool:
        """Backwards-compatible alias of :attr:`is_tabulated`."""
        return self.is_tabulated


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

    Source functions
    ----------------
    Every opacity source ``i`` contributes extinction ``kappa_i`` and an
    emissivity ``kappa_i * S_i``; :func:`source_weights` is the single
    definition, shared by the chord kernel and :class:`eclipse.Eclipse1D`.

    * **Line opacity** (atoms, ions, and molecules when ``molecular``) follows
      the two-level-atom source function

          S = (1 - eps) * J + eps * B_lambda(T),

      where ``J`` is the diluted stellar mean intensity and ``eps`` the
      photon-destruction (thermalisation) probability.  ``resonant_scattering``
      alone is ``eps = 0``; ``thermal`` alone is ``eps = 1``; both together
      require ``line_thermalisation`` to be given explicitly, because the two
      limits are not additive.  Where a density model carries no temperature
      (the exosphere family) there is no ``B``, and line opacity scatters with
      weight ``1 - eps``.
    * **Continuum opacity** (collision-induced absorption, H-) is true
      absorption: ``S = B`` under ``thermal``, nothing otherwise.
    * **Aerosol opacity** splits by the single-scattering albedo ``omega``:
      ``S = omega * J`` (if ``aerosol_scattering``) ``+ (1 - omega) * B`` (if
      ``thermal``).

    Attributes:
        resonant_scattering (bool): Single scattering of starlight by line
            opacity.  This is the exomoon-cloud term.
        thermal (bool): LTE thermal emission from line opacity and from the
            absorbing share of aerosol opacity.  Only density models that carry
            a temperature (the ``CollisionalAtmosphere`` family) contribute.
        molecular (bool): Include molecular constituents in the emission terms
            as well as in the extinction.  Off by default because it forces the
            expensive per-cell molecular interpolation.
        aerosol_scattering (bool): Single scattering of starlight by aerosol /
            haze opacity, with an isotropic phase function.  Off by default:
            real aerosols are strongly forward-scattering, so the isotropic
            assumption is much weaker here than it is for a resonance line.
        aerosol_albedo (float): Single-scattering albedo of the aerosol
            opacity.  The default 1.0 is a pure scatterer, which neither absorbs
            nor emits; set it to 0.0 for a purely absorbing grey opacity.  A
            modelling ASSUMPTION, not a measurement.
        line_thermalisation (Optional[float]): Photon-destruction probability
            ``eps`` of line opacity, in [0, 1].  Required when both
            ``resonant_scattering`` and ``thermal`` are on; not accepted
            otherwise.  A modelling ASSUMPTION, not a measurement: it is the
            ratio of collisional de-excitation to total de-excitation, and
            depends on the density and the line.
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
    line_thermalisation: Optional[float] = None
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
        both = self.resonant_scattering and self.thermal
        if both and self.line_thermalisation is None:
            raise ValueError(
                "resonant_scattering and thermal are both on, so line opacity "
                "needs S = (1 - eps) J + eps B: pass line_thermalisation=eps "
                "in [0, 1].  The two limits are not additive.  Use "
                "resonant_scattering=False for pure LTE emission (eps = 1).")
        if not both and self.line_thermalisation is not None:
            raise ValueError(
                "line_thermalisation only applies when resonant_scattering "
                "and thermal are both on.")
        if self.line_thermalisation is not None and not (
                0.0 <= self.line_thermalisation <= 1.0):
            raise ValueError("line_thermalisation must lie in [0, 1].")

    @property
    def epsilon(self) -> float:
        """Photon-destruction probability of line opacity."""
        if self.line_thermalisation is not None:
            return float(self.line_thermalisation)
        return 1.0 if self.thermal else 0.0

    @property
    def needs_atoms(self) -> bool:
        """True if atomic/ionic constituents carry a source term."""
        return self.resonant_scattering or self.thermal

    @property
    def needs_aerosols(self) -> bool:
        """True if scattering constituents carry a source term."""
        return self.aerosol_scattering or (self.thermal
                                           and self.aerosol_albedo < 1.0)


def source_weights(em: EmissionModel, constituent: Any,
                   has_temperature: bool) -> tuple:
    """Weights of ``J`` and ``B`` in one constituent's source function.

    The single definition of the source-function rules in
    :class:`EmissionModel`, used by every solver so they cannot drift apart.

    Args:
        em (EmissionModel): The emission configuration.
        constituent (Any): An atomic, molecular or scattering constituent.
        has_temperature (bool): Whether the density model supplies a
            temperature, i.e. whether ``B_lambda(T)`` exists.

    Returns:
        Tuple[float, float]: ``(w_J, w_B)`` such that the constituent's source
        function is ``S = w_J * J + w_B * B``.  ``(0, 0)`` means pure
        extinction.
    """
    if getattr(constituent, 'isContinuum', False):
        # CIA and H- are true absorption in LTE: thermal emission only.
        return 0.0, (1.0 if (em.thermal and has_temperature) else 0.0)
    if getattr(constituent, 'isScatterer', False):
        omega = em.aerosol_albedo
        w_J = omega if em.aerosol_scattering else 0.0
        w_B = (1.0 - omega) if (em.thermal and has_temperature) else 0.0
        return w_J, w_B
    if getattr(constituent, 'isMolecule', False) and not em.molecular:
        return 0.0, 0.0
    eps = em.epsilon
    w_J = (1.0 - eps) if em.resonant_scattering else 0.0
    w_B = eps if (em.thermal and has_temperature) else 0.0
    return w_J, w_B


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
