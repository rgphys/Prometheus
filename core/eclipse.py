"""
Secondary-eclipse (dayside thermal emission) geometry for Prometheus.

Prometheus is built around transmission: chords through a stellar disk, and a
flux ratio ``F_in / F_out`` that is bounded by 1.  A secondary eclipse is the
other observable — the planet's own emission, measured as
``F_planet / F_star`` when the planet disappears behind the star — and it needs
a different integration domain: a *planet*-centred disk rather than a
star-centred one.

That is all this module adds.  The radiative transfer itself is the same
machinery: :meth:`gasProperties.Atmosphere.getLOSopticalDepthAndEmission_Batch`
supplies the optical depth and the emission of any gas above the surface, and
:mod:`emission` supplies the source functions.

What is computed
----------------
The flux a distant observer receives from a source of specific intensity ``I``
over projected area ``A`` is ``f = ∫ I dA / d²``, so the eclipse depth is

    depth(λ) = ∫_planet I_p(λ) dA / (π R_star² · I_star(λ))

with ``I_star`` the disk-averaged stellar surface intensity.  For a bare,
uniform, opaque dayside this reduces to the textbook

    depth(λ) = epsilon · (R_p / R_star)² · B_λ(T_day) / I_star(λ),

which the quadrature here reproduces to the accuracy of its ray grid.

Each ray at impact parameter ``b`` from the planet centre carries

    I_ray = epsilon · B_λ(T_surf(b)) · exp(-tau_above)   +   I_em

where ``tau_above`` is the optical depth of the gas between the surface and the
observer and ``I_em`` is that gas's own emission.  Rays with ``b > R_p`` miss
the solid body: they have no surface term and see the limb of the atmosphere
only.

Geometry and phase
------------------
The rays are placed at orbital phase ``pi``, i.e. superior conjunction, where
Prometheus' star-centred frame puts the planet at ``x = -a`` with zero
line-of-sight velocity — the correct kinematics for an eclipse, and the correct
star-planet distance for the illumination term.  The star is *not* added to any
ray: the stellar contribution enters only through the denominator.  This module
therefore models the planet's own emission; it does not model the ingress and
egress light curve.

Scope
-----
The dayside temperature structure is an *input*, not something Prometheus
derives: there is no energy-balance or heat-redistribution solver here.  Give it
a temperature map (a measured brightness temperature, or an
instantaneous-reradiation profile) and it returns the eclipse depth that map
implies.  That is exactly what is needed to compare a model against a measured
eclipse depth or brightness temperature.

Created 2026-09-11.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import constants as const
from . import emission as emis
from . import geometryHandler as geom


@dataclass
class DaysideSurface:
    """The opaque lower boundary of the planet, seen at secondary eclipse.

    At superior conjunction the sub-observer point coincides with the
    substellar point, so a ray at impact parameter ``b`` strikes the surface at
    an angle ``theta`` from the substellar point with ``sin(theta) = b / R_p``.

    Attributes:
        T_day (float): Dayside temperature [K].  Under ``profile='uniform'``
            this is the disk temperature, which is precisely what an observed
            brightness temperature means, so that is the mode to use when
            comparing against one.  Under ``profile='instant'`` it is the
            substellar temperature.
        profile (str): ``'uniform'`` — the whole visible disk at ``T_day``;
            ``'instant'`` — instantaneous re-radiation, ``T = T_day ·
            cos(theta)**0.25``, the no-redistribution limit.
        emissivity (float): Grey surface emissivity.  1.0 (a blackbody surface)
            is the standard assumption for a magma ocean; lower values scale the
            depth linearly.  A modelling ASSUMPTION, not a measurement.
        T_floor (float): Temperature floor [K] applied to ``'instant'`` so the
            terminator does not go to zero.  Contributes negligibly to the disk
            integral; it only keeps the Planck function finite.
    """

    T_day: float
    profile: str = 'uniform'
    emissivity: float = 1.0
    T_floor: float = 1.0

    def temperature(self, b: np.ndarray, R_p: float) -> np.ndarray:
        """Surface temperature seen by rays at impact parameter ``b``.

        Args:
            b (np.ndarray): Impact parameter from the planet centre [cm].
            R_p (float): Planet radius [cm].

        Returns:
            np.ndarray: Temperature [K], same shape as ``b``.
        """
        if self.profile == 'uniform':
            return np.full_like(np.asarray(b, dtype=float), self.T_day)
        if self.profile == 'instant':
            mu = np.sqrt(np.clip(1.0 - (np.asarray(b, dtype=float) / R_p) ** 2,
                                 0.0, 1.0))
            return np.maximum(self.T_day * mu ** 0.25, self.T_floor)
        raise ValueError(f"Unknown surface profile {self.profile!r}; "
                         "use 'uniform' or 'instant'.")

    def intensity(self, wavelength: np.ndarray, b: np.ndarray,
                  R_p: float) -> np.ndarray:
        """Emergent surface intensity ``epsilon · B_lambda(T(b))``.

        Args:
            wavelength (np.ndarray): Wavelength grid (n_wav,) [cm].
            b (np.ndarray): Impact parameters (n_rays,) [cm].
            R_p (float): Planet radius [cm].

        Returns:
            np.ndarray: Intensity (n_rays, n_wav) in cgs.
        """
        T = self.temperature(b, R_p)[:, np.newaxis]
        return self.emissivity * emis.planck_lambda(
            np.asarray(wavelength)[np.newaxis, :], T)


class Eclipse:
    """Secondary-eclipse depth of a planet's dayside.

    Args:
        planet (Any): A :class:`celestialBodies.Planet`.
        wavelength (np.ndarray): Wavelength grid (n_wav,) [cm], increasing.
        surface (Optional[DaysideSurface]): The opaque dayside.  ``None`` models
            a planet with no solid/liquid boundary, leaving only the gas.
        atmosphere (Optional[Any]): A :class:`gasProperties.Atmosphere` built
            with an :class:`emission.EmissionModel`, for gas above the surface.
            ``None`` is a bare body.
        R_top (Optional[float]): Outer radius of the ray grid [cm].  Defaults to
            the planet radius when there is no atmosphere; give it explicitly to
            include the limb of an extended atmosphere.
        b_steps (int): Radial ray samples from 0 to ``R_top``.
        phi_steps (int): Azimuthal ray samples.  1 is correct and cheapest for
            an azimuthally symmetric dayside.
        x_border (Optional[float]): Half-length of the line-of-sight
            integration through the atmosphere [cm].
        x_steps (int): Cells along the line of sight.

    Attributes:
        stellarIntensity (emission.StellarIntensity): The denominator's stellar
            surface intensity, PHOENIX if attached to the star and a blackbody
            at ``T_eff`` otherwise.
    """

    def __init__(self, planet: Any, wavelength: np.ndarray,
                 surface: Optional[DaysideSurface] = None,
                 atmosphere: Optional[Any] = None,
                 R_top: Optional[float] = None,
                 b_steps: int = 60, phi_steps: int = 1,
                 x_border: Optional[float] = None, x_steps: int = 40):
        if surface is None and atmosphere is None:
            raise ValueError("An eclipse needs something that emits: pass a "
                             "DaysideSurface, an Atmosphere, or both.")
        self.planet = planet
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.surface = surface
        self.atmosphere = atmosphere
        self.R_top = float(R_top) if R_top is not None else float(planet.R)
        if self.R_top < planet.R:
            raise ValueError("R_top must be at least the planet radius.")
        self.b_steps = int(b_steps)
        self.phi_steps = int(phi_steps)
        self.x_border = (float(x_border) if x_border is not None
                         else 10.0 * float(planet.R))
        self.x_steps = int(x_steps)
        self.orbphase = np.pi          # superior conjunction
        self.stellarIntensity = emis.StellarIntensity(planet.hostStar,
                                                      self.wavelength)

    #  ray grid
    def rayGrid(self):
        """The planet-centred ray grid.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: impact parameters ``b``
            (n_rays,) [cm], azimuths ``theta`` (n_rays,) [rad], and the
            quadrature weights ``b · db · dtheta`` (n_rays,) [cm^2] whose sum is
            the projected area of the ray grid.
        """
        db = self.R_top / self.b_steps
        b = (np.arange(self.b_steps) + 0.5) * db
        dtheta = 2.0 * np.pi / self.phi_steps
        theta = (np.arange(self.phi_steps) + 0.5) * dtheta
        B, TH = np.meshgrid(b, theta, indexing='ij')
        return B.ravel(), TH.ravel(), (B * db * dtheta).ravel()

    def emergentIntensity(self) -> np.ndarray:
        """Emergent intensity of every ray through the planet.

        Returns:
            np.ndarray: Specific intensity (n_rays, n_wav) in cgs.
        """
        b, theta, _ = self.rayGrid()
        R_p = float(self.planet.R)
        n_wav = len(self.wavelength)

        hits = b < R_p
        # Where the ray meets the surface.  The observer is at x = -inf, so the
        # near face of the planet is the smaller-x root.
        x_p = float(self.planet.a) * np.cos(self.orbphase)
        x_surface = np.where(hits,
                             x_p - np.sqrt(np.clip(R_p ** 2 - b ** 2, 0.0, None)),
                             np.inf)

        I = np.zeros((len(b), n_wav))
        if self.surface is not None:
            I[hits] = self.surface.intensity(self.wavelength, b[hits], R_p)

        if self.atmosphere is None:
            return I

        # Map the planet-centred rays into the star-centred frame the transfer
        # kernel works in.  At superior conjunction the planet sits on the
        # x axis, so its sky-plane offset is zero and (b, theta) is simply the
        # sky position relative to the star's centre.
        y_p = float(self.planet.a) * np.sin(self.orbphase)
        y = y_p + b * np.sin(theta)
        z = b * np.cos(theta)
        rho = np.sqrt(y ** 2 + z ** 2)
        phi = np.arctan2(y, z)
        orb = np.full_like(b, self.orbphase)

        x_grid = np.linspace(x_p - self.x_border, x_p + self.x_border,
                             self.x_steps, endpoint=False) \
            + self.x_border / self.x_steps
        delta_x = 2.0 * self.x_border / self.x_steps

        tau, I_em, tau_vis = self.atmosphere.getLOSopticalDepthAndEmission_Batch(
            x_grid, phi, rho, orb, self.wavelength, delta_x,
            self.stellarIntensity, x_block=x_surface, return_visible_tau=True)

        # The surface shines through whatever gas lies in front of it; the gas
        # adds its own emission on top.
        return I * np.exp(-tau_vis) + I_em

    def depth(self) -> np.ndarray:
        """Eclipse depth ``F_planet / F_star``.

        Returns:
            np.ndarray: Depth as a fraction (multiply by 1e6 for ppm), shape
            ``(n_wav,)``.
        """
        _, _, weights = self.rayGrid()
        I = self.emergentIntensity()
        F_planet = (weights[:, np.newaxis] * I).sum(axis=0)
        I_star = self.stellarIntensity(self.wavelength[np.newaxis, :])[0]
        F_star = np.pi * self.planet.hostStar.R ** 2 * I_star
        return F_planet / F_star

    #  reductions
    def bandDepth(self, lower_cm: float, upper_cm: float,
                  weights: Optional[np.ndarray] = None) -> float:
        """Band-averaged eclipse depth between two wavelengths.

        The average is weighted by the stellar flux, which is what a
        broadband photometric eclipse depth measures: the band depth is the
        ratio of the band-integrated planet flux to the band-integrated stellar
        flux, not the unweighted mean of the depth spectrum.

        Args:
            lower_cm (float): Lower edge of the band [cm].
            upper_cm (float): Upper edge of the band [cm].
            weights (Optional[np.ndarray]): Optional throughput on the
                wavelength grid; combined with the stellar weighting.

        Returns:
            float: Band-averaged depth as a fraction.
        """
        wav = self.wavelength
        sel = (wav >= lower_cm) & (wav <= upper_cm)
        if not sel.any():
            raise ValueError("The band does not overlap the wavelength grid.")
        w = self.stellarIntensity(wav[np.newaxis, :])[0]
        if weights is not None:
            w = w * np.asarray(weights, dtype=float)
        return float(np.trapezoid(self.depth()[sel] * w[sel], wav[sel])
                     / np.trapezoid(w[sel], wav[sel]))

    def brightnessTemperature(self, depth: np.ndarray = None) -> np.ndarray:
        """Temperature of a uniform blackbody disk giving this depth.

        Inverts ``depth = (R_p/R_star)² · B_lambda(T) / I_star(lambda)`` for
        ``T``, which is the standard definition of a brightness temperature and
        makes a model directly comparable to a published one.

        Args:
            depth (np.ndarray): Depth spectrum to invert; defaults to this
                eclipse's own.

        Returns:
            np.ndarray: Brightness temperature [K] per wavelength.
        """
        if depth is None:
            depth = self.depth()
        wav = self.wavelength
        I_star = self.stellarIntensity(wav[np.newaxis, :])[0]
        B = np.asarray(depth) * I_star \
            * (self.planet.hostStar.R / self.planet.R) ** 2
        return inverse_planck(wav, B)


class Eclipse1D:
    """Secondary eclipse of a spherically symmetric atmosphere, on radial layers.

    :class:`Eclipse` integrates along a shared, uniform Cartesian line-of-sight
    grid.  That is the right tool for an extended exosphere, and the wrong one
    for a bound atmosphere: the scale height of a rocky planet's atmosphere is
    ~1e-3 of its radius, the surface sits at a different ``x`` for every ray, so
    a single uniform grid has to resolve the scale height across the whole
    planet.  Convergence needs several thousand cells and ~1.5 s per spectrum,
    and an under-resolved run is smooth and wrong rather than obviously broken.

    For a spherically symmetric atmosphere none of that is necessary.  The
    opacity depends only on radius, so the cross section is evaluated once per
    radial layer instead of once per (ray, cell), and each ray's path length
    through a layer follows in closed form from the chord geometry:

        ds_k(b) = sqrt(r_{k+1}^2 - b^2) - sqrt(max(r_k^2 - b^2, 0))

    traversed once for a ray that ends on the surface and twice for one that
    misses it.  That is exact -- there is no line-of-sight discretisation error
    left -- and it is 20-50x faster, which is what makes a retrieval practical.

    The disk integral uses Gauss-Legendre quadrature in ``mu = cos(theta)``,
    where the integrand is smooth, plus a separate set of rays across the thin
    limb annulus above the solid body.

    Args:
        planet (Any): A :class:`celestialBodies.Planet`.
        wavelength (np.ndarray): Wavelength grid (n_wav,) [cm], increasing.
        surface (Optional[DaysideSurface]): The opaque lower boundary.
        density_model (Optional[Any]): One Prometheus density model from the
            ``CollisionalAtmosphere`` family, with its constituents attached.
            ``None`` models a bare body.
        emission (Optional[Any]): An :class:`emission.EmissionModel`.  Required
            when ``density_model`` is given.  ``stellar_doppler`` has no effect
            here: a static 1-D profile has no velocity field.
        n_layers (int): Radial layers between the surface and ``R_top``.
        R_top (Optional[float]): Top of the atmosphere [cm].  Defaults to the
            radius at which the model's density has fallen by ``exp(-n_scale)``
            from its surface value, found by bisection.
        n_scale (float): Number of density e-foldings spanned when ``R_top`` is
            chosen automatically.
        n_mu (int): Gauss-Legendre nodes across the solid disk.
        n_limb (int): Rays across the limb annulus above the solid body, where
            the atmosphere glows against the sky.  Set to 0 to integrate the
            solid disk only, which is what makes this directly comparable to
            :class:`Eclipse` run with ``R_top = R_p``.

    Attributes:
        r_edges (np.ndarray): Layer boundaries [cm], ``(n_layers + 1,)``.
        r_mid (np.ndarray): Layer midpoints [cm].
        T_layer, n_layer (np.ndarray): Temperature [K] and total number density
            [cm^-3] at the midpoints.
    """

    def __init__(self, planet: Any, wavelength: np.ndarray,
                 surface: Optional[DaysideSurface] = None,
                 density_model: Optional[Any] = None,
                 emission: Optional[Any] = None,
                 n_layers: int = 160, R_top: Optional[float] = None,
                 n_scale: float = 25.0, n_mu: int = 12, n_limb: int = 12):
        if surface is None and density_model is None:
            raise ValueError("An eclipse needs something that emits: pass a "
                             "DaysideSurface, a density model, or both.")
        if density_model is not None and emission is None:
            raise ValueError("A density model needs an EmissionModel to "
                             "contribute a source term.")
        self.planet = planet
        self.wavelength = np.ascontiguousarray(wavelength, dtype=float)
        self.surface = surface
        self.model = density_model
        self.em = emission
        self.orbphase = np.pi
        self.stellarIntensity = emis.StellarIntensity(planet.hostStar,
                                                      self.wavelength)
        self.n_mu, self.n_limb = int(n_mu), int(n_limb)

        R_p = float(planet.R)
        if density_model is None:
            self.R_top = R_p
            self.r_edges = np.array([R_p])
        else:
            self.R_top = (float(R_top) if R_top is not None
                          else self._findTop(n_scale))
            self.r_edges = np.linspace(R_p, self.R_top, int(n_layers) + 1)
        self.r_mid = (0.5 * (self.r_edges[1:] + self.r_edges[:-1])
                      if len(self.r_edges) > 1 else np.array([]))
        if density_model is not None:
            self.n_layer = self._sampleRadially(
                density_model.calculateNumberDensity, self.r_mid)
            if getattr(density_model, 'isNonIsothermal', False):
                self.T_layer = self._sampleRadially(
                    density_model.calculateTemperature, self.r_mid)
            else:
                self.T_layer = np.full_like(self.r_mid,
                                            float(density_model.T))

    #  radial sampling
    def _sampleRadially(self, fn, r: np.ndarray) -> np.ndarray:
        """Evaluates a density-model method on a purely radial ray.

        The model API is written in the star-centred chord frame.  At superior
        conjunction the planet sits on the x axis, so a point at radius ``r``
        from the planet centre is reached with ``rho = 0`` and
        ``x = x_p - r``.

        Args:
            fn (Callable): ``calculateNumberDensity`` or ``calculateTemperature``.
            r (np.ndarray): Radii from the planet centre [cm].

        Returns:
            np.ndarray: The sampled quantity, same shape as ``r``.
        """
        x_p = float(self.planet.a) * np.cos(self.orbphase)
        out = fn(np.asarray(x_p - r, dtype=float), 0.0, 0.0, self.orbphase)
        return np.asarray(out, dtype=float).reshape(-1)

    def _findTop(self, n_scale: float) -> float:
        """Radius where the density has dropped ``n_scale`` e-foldings."""
        R_p = float(self.planet.R)
        hi = R_p
        n0 = self._sampleRadially(self.model.calculateNumberDensity,
                                  np.array([R_p * (1 + 1e-9)]))[0]
        target = n0 * np.exp(-n_scale)
        for _ in range(80):
            hi *= 1.02
            n_hi = self._sampleRadially(self.model.calculateNumberDensity,
                                        np.array([hi]))[0]
            if n_hi <= target or n_hi <= 0.0:
                break
        lo = R_p
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            n_mid = self._sampleRadially(self.model.calculateNumberDensity,
                                         np.array([mid]))[0]
            if n_mid > target:
                lo = mid
            else:
                hi = mid
        return hi

    #  opacity
    def layerOpacity(self) -> np.ndarray:
        """Extinction coefficient per layer, ``sum_i n_i sigma_i(lambda)``.

        Returns:
            np.ndarray: ``(n_layers, n_wav)`` in cm^-1.
        """
        n_lay, n_wav = len(self.r_mid), len(self.wavelength)
        kappa = np.zeros((n_lay, n_wav))
        if self.model is None:
            return kappa
        P = np.clip(self.n_layer * const.k_B * self.T_layer, 1e-30, None)
        for c in self.model.constituents:
            n_abs = self.n_layer * c.chi
            if c.isMolecule:
                from .gasProperties import (_bilinear_PT_interp_Tvec,
                                            n_interp_linear_rows)
                sigma_native = _bilinear_PT_interp_Tvec(
                    P, self.T_layer, c.P_grid, c.T_grid, c.sigma_grid_log,
                    c.lookupOffset)
                wav_rows = np.ascontiguousarray(
                    np.tile(self.wavelength, (n_lay, 1)))
                sigma = n_interp_linear_rows(wav_rows, c.wav_grid, sigma_native)
            elif getattr(c, 'isScatterer', False):
                sigma = np.tile(c.getSigmaAbs(self.wavelength), (n_lay, 1))
            else:
                sigma = c.getSigmaAbs(
                    np.ascontiguousarray(np.tile(self.wavelength, (n_lay, 1))))
            kappa += n_abs[:, np.newaxis] * sigma
        return kappa

    def layerSource(self) -> np.ndarray:
        """Source function per layer, ``(n_layers, n_wav)`` in cgs intensity.

        Thermal emission uses ``B_lambda(T)``; resonance/aerosol scattering uses
        the diluted stellar intensity at the planet's orbital distance.  Both are
        weighted the same way as in the chord kernel, so the two solvers agree.
        """
        n_lay, n_wav = len(self.r_mid), len(self.wavelength)
        if self.model is None or n_lay == 0:
            return np.zeros((n_lay, n_wav))
        S = np.zeros((n_lay, n_wav))
        if self.em.thermal:
            S += emis.planck_lambda(self.wavelength[np.newaxis, :],
                                    self.T_layer[:, np.newaxis])
        if self.em.resonant_scattering or self.em.aerosol_scattering:
            W = emis.dilution_factor(float(self.planet.a),
                                     self.planet.hostStar.R)
            I_star = self.stellarIntensity(self.wavelength[np.newaxis, :])[0]
            S += W * I_star[np.newaxis, :]
        return S

    #  geometry
    def _rayGrid(self):
        """Impact parameters and disk-integration weights.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: impact parameters ``b``
            [cm], area weights [cm^2] summing to the projected area, and a
            boolean flag marking rays that terminate on the surface.
        """
        R_p = float(self.planet.R)
        # Solid disk: F = 2 pi R_p^2 int_0^1 I(mu) mu dmu, Gauss-Legendre in mu.
        nodes, weights = np.polynomial.legendre.leggauss(self.n_mu)
        mu = 0.5 * (nodes + 1.0)
        w_mu = 0.5 * weights
        b_disk = R_p * np.sqrt(np.clip(1.0 - mu ** 2, 0.0, 1.0))
        a_disk = 2.0 * np.pi * R_p ** 2 * w_mu * mu
        if self.model is None or self.R_top <= R_p:
            return b_disk, a_disk, np.ones_like(b_disk, dtype=bool)
        # Limb annulus, integrated in b directly.
        edges = np.linspace(R_p, self.R_top, self.n_limb + 1)
        b_limb = 0.5 * (edges[1:] + edges[:-1])
        a_limb = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        return (np.concatenate([b_disk, b_limb]),
                np.concatenate([a_disk, a_limb]),
                np.concatenate([np.ones(len(b_disk), bool),
                                np.zeros(len(b_limb), bool)]))

    @staticmethod
    def _pathLengths(b: float, r_edges: np.ndarray) -> np.ndarray:
        """Chord length through each spherical shell, for one impact parameter.

        Args:
            b (float): Impact parameter [cm].
            r_edges (np.ndarray): Shell boundaries [cm], increasing.

        Returns:
            np.ndarray: One-sided path length in each shell [cm]; zero for
            shells the ray does not reach.
        """
        z = np.sqrt(np.clip(r_edges ** 2 - b ** 2, 0.0, None))
        return np.diff(z)

    #  transfer
    def emergentIntensity(self) -> np.ndarray:
        """Emergent intensity for every ray, ``(n_rays, n_wav)`` in cgs."""
        b, _, hits = self._rayGrid()
        R_p = float(self.planet.R)
        n_wav = len(self.wavelength)
        I = np.zeros((len(b), n_wav))
        kappa = self.layerOpacity()
        S = self.layerSource()
        I_surf = (self.surface.intensity(self.wavelength, b, R_p)
                  if self.surface is not None else np.zeros((len(b), n_wav)))

        for j in range(len(b)):
            ds = self._pathLengths(b[j], self.r_edges)
            if hits[j]:
                # Integrate along the direction of propagation, which starts at
                # the surface and works outward toward the observer: shell 0 is
                # the one touching the surface.  The formal solution is a
                # sequential recursion, so this order is not optional.
                seg = [(k, ds[k]) for k in range(len(ds)) if ds[k] > 0.0]
                base = I_surf[j]
            else:
                # Limb ray: down the near side to the tangent point, then back
                # out the far side.  Shells below the tangent are not reached.
                near = [(k, ds[k]) for k in range(len(ds) - 1, -1, -1)
                        if ds[k] > 0.0]
                far = [(k, d) for k, d in reversed(near)]
                seg = near + far
                base = np.zeros(n_wav)
            out = base
            for k, d in seg:
                dtau = kappa[k] * d
                w = np.where(dtau > 1e-8,
                             -np.expm1(-np.where(dtau > 1e-8, dtau, 1.0))
                             / np.where(dtau > 1e-8, dtau, 1.0),
                             1.0 - 0.5 * dtau)
                out = out * np.exp(-dtau) + S[k] * dtau * w
            I[j] = out
        return I

    def depth(self) -> np.ndarray:
        """Eclipse depth ``F_planet / F_star``, shape ``(n_wav,)``."""
        _, area, _ = self._rayGrid()
        F_p = (area[:, np.newaxis] * self.emergentIntensity()).sum(axis=0)
        I_star = self.stellarIntensity(self.wavelength[np.newaxis, :])[0]
        return F_p / (np.pi * self.planet.hostStar.R ** 2 * I_star)

    def bandDepth(self, lower_cm: float, upper_cm: float,
                  depth: Optional[np.ndarray] = None) -> float:
        """Stellar-flux-weighted band average of the depth spectrum."""
        wav = self.wavelength
        sel = (wav >= lower_cm) & (wav <= upper_cm)
        if not sel.any():
            raise ValueError("The band does not overlap the wavelength grid.")
        d = self.depth() if depth is None else depth
        w = self.stellarIntensity(wav[np.newaxis, :])[0]
        return float(np.trapezoid(d[sel] * w[sel], wav[sel])
                     / np.trapezoid(w[sel], wav[sel]))


def inverse_planck(wavelength: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Temperature whose Planck function equals ``B`` at ``wavelength``.

    Args:
        wavelength (np.ndarray): Wavelength [cm].
        B (np.ndarray): Specific intensity [erg s^-1 cm^-2 cm^-1 sr^-1].

    Returns:
        np.ndarray: Brightness temperature [K]; NaN where ``B <= 0``.
    """
    wav = np.asarray(wavelength, dtype=float)
    B_ = np.asarray(B, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = 2.0 * const.h * const.c ** 2 / (wav ** 5 * B_)
        T = const.h * const.c / (wav * const.k_B * np.log1p(arg))
    return np.where(B_ > 0.0, T, np.nan)


def band_brightness_temperature(planet: Any, wavelength: np.ndarray,
                                depth: float, lower_cm: float,
                                upper_cm: float) -> float:
    """Brightness temperature implied by a *band-averaged* eclipse depth.

    Solves for the uniform dayside temperature whose stellar-flux-weighted band
    depth equals the measured one, which is how a broadband brightness
    temperature is defined.  Use this to turn a published eclipse depth into the
    temperature a Prometheus model has to reproduce.

    Args:
        planet (Any): The :class:`celestialBodies.Planet`.
        wavelength (np.ndarray): Wavelength grid covering the band [cm].
        depth (float): Measured band-averaged depth as a fraction.
        lower_cm (float): Lower band edge [cm].
        upper_cm (float): Upper band edge [cm].

    Returns:
        float: Brightness temperature [K].
    """
    from scipy.optimize import brentq

    def residual(T):
        ecl = Eclipse(planet, wavelength,
                      surface=DaysideSurface(T_day=T, profile='uniform'),
                      b_steps=24)
        return ecl.bandDepth(lower_cm, upper_cm) - depth

    return float(brentq(residual, 10.0, 20000.0, xtol=1e-3))
