"""
This file defines classes for celestial bodies (Star, Planet, Moon)
and a utility class to load pre-defined systems from data files.
These classes encapsulate the physical properties and orbital mechanics
required for transit simulations.

Created on 16. September 2022 by Andrea Gebek.
"""

import csv
import os
import pathlib
import shutil
import urllib.request as request
from contextlib import closing
from typing import Any, Callable, Optional, Tuple

import astropy.io.fits as fits
import numpy as np
from scipy.interpolate import interp1d

from . import constants as const
from . import geometryHandler as geom


class Star:
    """Represents a star with its physical and observational properties.

    This class stores stellar parameters, handles the retrieval of stellar spectra,
    and calculates effects like center-to-limb variation (CLV) and the
    Rossiter-McLaughlin (RM) effect.

    Attributes:
        R (float): Stellar radius in cm.
        M (float): Stellar mass in g.
        T_eff (float): Effective temperature in Kelvin.
        log_g (float): Logarithm of the surface gravity (log10(cm/s^2)).
        Z (float): Metallicity [Fe/H].
        alpha (float): Alpha-element enhancement [alpha/Fe].
        CLV_u1 (float): Linear limb-darkening coefficient.
        CLV_u2 (float): Quadratic limb-darkening coefficient.
        vsiniStarrot (float): Projected stellar rotational velocity in cm/s.
        phiStarrot (float): Azimuthal angle of the stellar rotation axis in radians.
        Fstar_function (Optional[Callable]): An interpolation function for the
            stellar flux as a function of wavelength.
    """

    def __init__(self, R: float, dR: float, M: float, dM: float, T_eff: float, dT_eff: float, log_g: float, dlog_g: float, Z: float, dZ: float, alpha: float) -> None:
        """Initializes the Star object with its fundamental properties.

        Args:
            R (float): Stellar radius in cm.
            M (float): Stellar mass in g.
            T_eff (float): Effective temperature in Kelvin.
            log_g (float): Logarithm of the surface gravity (log10(cm/s^2)).
            Z (float): Metallicity [Fe/H].
            alpha (float): Alpha-element enhancement [alpha/Fe].
        """
        self.R: float = R
        self.dR: float = dR
        self.M: float = M
        self.dM: float = dM
        self.T_eff: float = T_eff
        self.dT_eff: float = dT_eff
        self.log_g: float = log_g
        self.dlog_g: float = dlog_g
        self.Z: float = Z
        self.dZ: float = dZ
        self.alpha: float = alpha
        self.CLV_u1: float = 0.
        self.CLV_u2: float = 0.
        self.CLV_function: Optional[Callable[[Any, Any], Any]] = None
        self.vsiniStarrot: float = 0.
        self.phiStarrot: float = 0.
        self.Fstar_function: Optional[Callable[[Any], Any]] = None

    def addCLVparameters(self, CLV_u1: float, CLV_u2: float) -> None:
        """Adds quadratic limb-darkening coefficients to the star.

        Args:
            CLV_u1 (float): The linear limb-darkening coefficient.
            CLV_u2 (float): The quadratic limb-darkening coefficient.
        """
        self.CLV_u1 = CLV_u1
        self.CLV_u2 = CLV_u2

    def addCLVfunction(
            self, CLV_function: Callable[[Any, Any], Any]) -> None:
        """Adds a wavelength-dependent center-to-limb variation profile.

        Takes precedence over the quadratic ``addCLVparameters``
        coefficients in the transit chord sum.  Only the shape in mu (at
        fixed wavelength) matters: the chord sum normalises per wavelength,
        so any per-wavelength scaling of the profile cancels.

        Args:
            CLV_function: Callable ``f(mu, wavelength)`` with ``mu`` an
                array of cosines of the heliocentric angle, shape (N,),
                and ``wavelength`` an array in cm, shape (W,).  Must
                return the relative intensity ``I(mu, wavelength)`` with
                shape (N, W).
        """
        self.CLV_function = CLV_function

    def addStellarContamination(self, spot_fraction: float = 0.0,
                                spot_temp: Optional[float] = None,
                                fac_fraction: float = 0.0,
                                fac_temp: Optional[float] = None) -> None:
        """Adds unocculted-spot/facula contamination (transit light source effect).

        Models the wavelength-dependent transit-depth bias from active regions
        outside the transit chord (Sing+2011; Rackham+2018).  This is a
        disk-integrated stellar effect applied to the final transit depth, NOT a
        center-to-limb (chord) effect, so it is distinct from the CLV machinery.

        Args:
            spot_fraction (float): Unocculted dark-spot covering fraction [0,1).
            spot_temp (Optional[float]): Spot temperature [K]. Defaults to
                0.86*T_eff (a typical cool-spot contrast) if spots are present.
            fac_fraction (float): Unocculted bright-facula covering fraction.
            fac_temp (Optional[float]): Facula temperature [K]. Defaults to
                T_eff + 100 K if faculae are present.
        """
        self.spot_fraction: float = spot_fraction
        self.spot_temp: float = spot_temp if spot_temp is not None else 0.86 * self.T_eff
        self.fac_fraction: float = fac_fraction
        self.fac_temp: float = fac_temp if fac_temp is not None else self.T_eff + 100.0

    def stellarContaminationFactor(self, wavelength: np.ndarray) -> np.ndarray:
        """Transit-depth contamination factor epsilon(lambda) (TLSE).

        ``D_observed = D_clean * epsilon``, with
        ``epsilon = 1/(1 - sum_k f_k (1 - B(lambda,T_k)/B(lambda,T_phot)))``
        over active regions k (Planck blackbody surface brightnesses).

        Args:
            wavelength (np.ndarray): Wavelength grid [cm].

        Returns:
            np.ndarray: Multiplicative depth factor, same shape as wavelength.
        """
        h_planck = 6.626e-27
        hc_k = h_planck * const.c / const.k_B

        def planck(T):
            return 1.0 / (wavelength ** 5 * np.expm1(hc_k / (wavelength * T)))
        Bp = planck(self.T_eff)
        dimming = np.zeros_like(wavelength)
        if getattr(self, 'spot_fraction', 0.0) > 0:
            dimming = dimming + self.spot_fraction * (1.0 - planck(self.spot_temp) / Bp)
        if getattr(self, 'fac_fraction', 0.0) > 0:
            dimming = dimming + self.fac_fraction * (1.0 - planck(self.fac_temp) / Bp)
        return 1.0 / (1.0 - dimming)

    def addRMparameters(self, vsiniStarrot: float, phiStarrot: float) -> None:
        """Adds Rossiter-McLaughlin effect parameters to the star.

        Args:
            vsiniStarrot (float): Projected stellar rotational velocity in cm/s.
            phiStarrot (float): Azimuthal angle of the stellar rotation axis
                in radians.
        """
        self.vsiniStarrot = vsiniStarrot
        self.phiStarrot = phiStarrot

    def getSurfaceVelocity(self, phi: float, rho: float) -> float:
        """Calculates the line-of-sight velocity of a point on the stellar surface.

        This is used for calculating the Rossiter-McLaughlin effect.

        Args:
            phi (float): The azimuthal angle on the stellar disk in radians.
            rho (float): The projected radial distance from the star's center in cm.

        Returns:
            float: The line-of-sight velocity at the specified point in cm/s.
        """
        v_los = self.vsiniStarrot * rho / \
            self.R * np.cos(phi - self.phiStarrot)
        return v_los

    @staticmethod
    def round_to_grid(grid: np.ndarray, value: float) -> float:
        """Finds the value in a grid closest to a given value.

        Args:
            grid (np.ndarray): The array of grid points.
            value (float): The value to match.

        Returns:
            float: The grid point closest to the input value.
        """
        diff = np.subtract(value, grid)
        arg = np.argmin(np.abs(diff))
        return grid[arg]

    def getSpectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Queries a PHOENIX photosphere model, either from disk or from the PHOENIX website
        if the spectrum hasn't been downloaded before, and return the wavelength and flux arrays.

        The function rounds the stellar parameters (effective temperature, surface gravity,
        metallicity, and alpha enhancement) to the nearest available grid values, constructs
        the appropriate URLs for the PHOENIX model FITS files, downloads them, reads the data,
        and returns the wavelength and flux arrays in cgs units.

        Returns
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - Wavelength array (in cm)
            - Flux array (in cgs units, divided by pi)
        """
        # Acceptable PHOENIX grid values for each stellar parameter.
        T_grid = np.concatenate(
            (np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)))
        log_g_grid = np.arange(0, 6.5, 0.5)
        Z_grid = np.concatenate(
            (np.arange(-4, -1, 1), np.arange(-1.5, 1.5, 0.5)))
        alpha_grid = np.arange(0, 1.6, 0.2)-0.2

        T_a = Star.round_to_grid(T_grid, self.T_eff)
        log_g_a = Star.round_to_grid(log_g_grid, self.log_g)
        Z_a = Star.round_to_grid(Z_grid, self.Z)
        alpha_a = Star.round_to_grid(alpha_grid, self.alpha)

        # This is where phoenix spectra are located.
        root = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'

        # Build the PHOENIX URL path from rounded grid parameters.
        z_string = '{:.1f}'.format(float(Z_a))
        if Z_a > 0:
            z_string = '+' + z_string
        elif Z_a == 0:
            z_string = '-' + z_string
        else:
            z_string = z_string
        a_string = ''
        if alpha_a > 0:
            a_string = '.Alpha=+'+'{:.2f}'.format(float(alpha_a))
        if alpha_a < 0:
            a_string = '.Alpha='+'{:.2f}'.format(float(alpha_a))
        t_string = str(int(T_a))
        if T_a < 10000:
            t_string = '0'+t_string
        g_string = '-'+'{:.2f}'.format(float(log_g_a))

        # PHOENIX FTP URLs for the wavelength grid and the spectrum.
        waveurl = root+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
        specurl = root+'PHOENIX-ACES-AGSS-COND-2011/Z'+z_string+a_string+'/lte' + \
            t_string+g_string+z_string+a_string+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

        # Local cache paths for the downloaded FITS files.
        cache_dir = os.path.join(os.path.dirname(__file__), "phoenix_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        wavename = os.path.join(cache_dir, 'WAVE.fits')
        specname = os.path.join(cache_dir, f'lte{t_string}{g_string}{z_string}{a_string}.fits')

        # Download once; skip if already cached.
        if not os.path.exists(wavename):
            print(f"Downloading WAVE.fits (this will only happen once)...")
            with closing(request.urlopen(waveurl)) as r:
                with open(wavename, 'wb') as f:
                    shutil.copyfileobj(r, f)

        if not os.path.exists(specname):
            print(f"Downloading stellar spectrum {specname} (this will only happen once)...")
            with closing(request.urlopen(specurl)) as r:
                with open(specname, 'wb') as f:
                    shutil.copyfileobj(r, f)

        F = fits.getdata(specname)
        w = fits.getdata(wavename)

        return (w * 1e-8, F / np.pi)
        # PHOENIX outputs pi*H_lambda (Eddington flux); dividing by pi gives surface intensity.
        # Only the shape matters here: the factor cancels in the F_in/F_out transit ratio.

    def calculateCLV(self, rho: float) -> float:
        """Calculates the center-to-limb variation (CLV) factor for a given radius.

        This uses a quadratic limb-darkening law.

        Args:
            rho (float): The projected radial distance from the star's center in cm.

        Returns:
            float: The CLV intensity factor, normalized to 1 at the center.
        """
        arg = 1. - np.sqrt(1. - rho**2 / self.R**2)
        return 1. - self.CLV_u1 * arg - self.CLV_u2 * arg**2

    def calculateRM(self, phi: float, rho: float, wavelength: np.ndarray) -> np.ndarray:
        """Calculates the Doppler-shifted stellar flux from a point on the star's surface.

        Args:
            phi (float): The azimuthal angle on the stellar disk in radians.
            rho (float): The projected radial distance from the star's center in cm.
            wavelength (np.ndarray): The array of wavelengths at which to calculate the flux.

        Returns:
            np.ndarray: The Doppler-shifted flux at the given location.
        """
        v_los = self.getSurfaceVelocity(phi, rho)
        shift = const.calculateDopplerShift(v_los)
        F_shifted = 10.**self.Fstar_function(wavelength / shift)
        return F_shifted

    def addFstarFunction(self, wavelength: np.ndarray) -> None:
        """Creates and stores an interpolation function for the stellar spectrum.

        This function fetches the PHOENIX spectrum, selects the relevant wavelength
        range, and creates a 1D interpolation function for `log10(flux)` vs.
        `wavelength`.

        Args:
            wavelength (np.ndarray): The wavelength grid for the simulation, used to
                determine the required range of the stellar spectrum.
        """
        PHOENIX_output = self.getSpectrum()
        w_star = PHOENIX_output[0]
        w_max = np.max(wavelength) * \
            const.calculateDopplerShift(-self.vsiniStarrot)
        w_min = np.min(wavelength) * \
            const.calculateDopplerShift(self.vsiniStarrot)
        SEL = (w_star >= w_min) * (w_star <= w_max)
        minArg = max(min(np.argwhere(SEL)).item() - 1, 0)
        maxArg = max(np.argwhere(SEL)).item() + 2
        w_starSEL = w_star[minArg:maxArg]
        F_0 = PHOENIX_output[1][minArg:maxArg]
        Fstar_function = interp1d(w_starSEL, np.log10(F_0), kind='linear')
        self.Fstar_function = Fstar_function

    def getFstarIntegrated(self, wavelength: np.ndarray, grid: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the stellar flux integrated over the entire disk.

        This method computes the total flux from the star, considering CLV and RM effects.
        It also calculates the flux from the upper part of the star, which is used
        for normalization in some contexts.

        Args:
            wavelength (np.ndarray): The array of wavelengths for the calculation.
            grid (Any): The spatial grid object (`geometryHandler.Grid`), used for
                discretization.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The total integrated stellar flux.
                - The integrated flux from the upper portion of the stellar disk.
        """
        if self.vsiniStarrot == 0.:
            FstarIntegrated = np.pi * self.R**2 * \
                (1. - self.CLV_u1 / 3. - self.CLV_u2 / 6.) * \
                np.ones_like(wavelength)
            upperTerm = 0.5 * (-self.CLV_u2 * self.R**2 -
                               self.CLV_u1 * self.R**2 + self.R**2)
            term1 = -4. * self.R**2 * self.CLV_u1 * \
                (1. - grid.rho_border**2 / self.R**2)**1.5
            term2 = self.R**2 * self.CLV_u2 * (6 * grid.rho_border**2 / self.R**2 + 8. * (
                1. - grid.rho_border**2 / self.R**2)**1.5 - 3. * (self.R**2 - grid.rho_border**2)**2 / self.R**4)
            lowerTerm = 1. / 12. * \
                (term1 - term2 - 6. * self.CLV_u1 *
                 grid.rho_border**2 + 6. * grid.rho_border**2)
            FstarUpper = 2. * np.pi * \
                (upperTerm - lowerTerm) * np.ones_like(wavelength)
        else:
            phiArray = grid.constructPhiAxis()
            delta_phi = grid.getDeltaPhi()
            rhoArray = grid.constructRhoAxis()
            delta_rho = grid.getDeltaRho()
            FstarIntegrated = np.zeros_like(wavelength)
            for phi in phiArray:
                for rho in rhoArray:
                    Fstar = self.calculateRM(phi, rho, wavelength)
                    Fstar *= self.calculateCLV(rho)
                    FstarIntegrated += Fstar * delta_phi * delta_rho * rho
                    FstarUpper = np.zeros_like(wavelength)
        return FstarIntegrated, FstarUpper

    def getFstar(self, phi: float, rho: float, wavelength: np.ndarray) -> np.ndarray:
        """Calculates the stellar flux from a specific point on the disk.

        If stellar rotation (`vsiniStarrot`) is non-zero, this includes the RM effect.
        It always includes the CLV effect.

        Args:
            phi (float): The azimuthal angle on the stellar disk in radians.
            rho (float): The projected radial distance from the star's center in cm.
            wavelength (np.ndarray): The array of wavelengths.

        Returns:
            np.ndarray: The flux array from the specified point on the stellar disk.
        """
        if self.vsiniStarrot == 0.:
            Fstar = np.ones_like(wavelength) * self.calculateCLV(rho)
        else:
            Fstar = self.calculateRM(phi, rho, wavelength)
            Fstar *= self.calculateCLV(rho)
        return Fstar


class Planet:
    """Represents a planet and its orbital properties.

    Attributes:
        name (str): The name of the planet.
        R (float): The radius of the planet in cm.
        M (float): The mass of the planet in g.
        a (float): The semi-major axis of the planet's orbit in cm.
        hostStar (Star): The host star object.
        transitDuration (float): The duration of the transit in hours.
        orbitalPeriod (float): The orbital period of the planet in days.
    """

    def __init__(self, name: str, R: float, M: float, a: float, hostStar: Star, transitDuration: float, orbitalPeriod: float, b:float) -> None:
        """Initializes the Planet object.

        Args:
            name (str): The name of the planet.
            R (float): The radius of the planet in cm.
            M (float): The mass of the planet in g.
            a (float): The semi-major axis of the planet's orbit in cm.
            hostStar (Star): The host star object.
            transitDuration (float): The duration of the transit in hours.
            orbitalPeriod (float): The orbital period of the planet in days.
            b (float): The impact parameter.
        """
        self.name: str = name
        self.R: float = R
        self.M: float = M
        self.a: float = a
        self.hostStar: Star = hostStar
        self.transitDuration: float = transitDuration
        self.orbitalPeriod: float = orbitalPeriod
        self.b: float = b

    def getPosition(self, orbphase: float) -> Tuple[float, float]:
        """Calculates the planet's (x, y) coordinates for a given orbital phase.

        Assumes a circular orbit viewed edge-on. The observer is along the x-axis.

        Args:
            orbphase (float): The orbital phase in radians (0 at mid-transit).

        Returns:
            Tuple[float, float]: The x and y coordinates of the planet in cm.
        """
        x_p = self.a * np.cos(orbphase)
        y_p = self.a * np.sin(orbphase)
        return x_p, y_p

    def getLOSvelocity(self, orbphase: float) -> float:
        """Calculates the planet's line-of-sight velocity.

        Assumes a circular orbit.

        Args:
            orbphase (float): The orbital phase in radians (0 at mid-transit).

        Returns:
            float: The line-of-sight velocity in cm/s.
        """
        v_los = -np.sin(orbphase) * np.sqrt(const.G * self.hostStar.M / self.a)
        return v_los

    def getDistanceFromPlanet(self, x: np.ndarray, phi, rho, orbphase) -> np.ndarray:
        """Calculates the 3D distance from a point in space to the planet's center.

        The point is defined in a cylindrical coordinate system (x, phi, rho)
        relative to the observer's line of sight through the star's center.

        Supports both scalar and batch (array) inputs for phi, rho, orbphase.
        When inputs are arrays of shape (n_chords,), x has shape (n_x,), the
        result is broadcast to shape (n_chords, n_x).

        Args:
            x (np.ndarray): Array of coordinates along the line of sight in cm, shape (n_x,).
            phi: Azimuthal angle(s) on the sky plane in radians. Scalar or (n_chords,).
            rho: Projected radial distance(s) from the star's center in cm. Scalar or (n_chords,).
            orbphase: Planet's orbital phase(s) in radians. Scalar or (n_chords,).

        Returns:
            np.ndarray: Distance(s) from the point(s) to the planet's center in cm.
                        Shape (n_x,) for scalar inputs, (n_chords, n_x) for array inputs.
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_p, y_p = self.getPosition(orbphase)
        # For batch inputs: phi/rho/orbphase are (n_chords,), x is (n_x,).
        # Expand body-position scalars/vectors to (n_chords, 1) and x to (1, n_x)
        # so the subtraction broadcasts correctly to (n_chords, n_x).
        x_    = np.asarray(x)
        x_p_  = np.asarray(x_p)
        y_p_  = np.asarray(y_p)
        y_    = np.asarray(y)
        z_    = np.asarray(z)
        if x_p_.ndim > 0:                          # batch mode
            x_    = x_[np.newaxis, :]              # (1, n_x)
            x_p_  = x_p_[:, np.newaxis]            # (n_chords, 1)
            y_p_  = y_p_[:, np.newaxis]
            y_    = y_[:, np.newaxis]
            z_    = z_[:, np.newaxis]
        r_fromPlanet = np.sqrt((x_ - x_p_)**2 + (y_ - y_p_)**2 + z_**2)
        return r_fromPlanet

    def getTorusCoords(self, x, phi, rho, orbphase):
        """Calculates coordinates relative to the planet for a torus model.

        Supports both scalar and batch (array) inputs for phi, rho, orbphase.
        When inputs are arrays of shape (n_chords,), x has shape (n_x,), the
        results are broadcast to shape (n_chords, n_x).

        Args:
            x: The coordinate(s) along the line of sight in cm, shape (n_x,).
            phi: Azimuthal angle(s) in radians. Scalar or (n_chords,).
            rho: Projected radial distance(s) in cm. Scalar or (n_chords,).
            orbphase: Planet's orbital phase(s) in radians. Scalar or (n_chords,).

        Returns:
            Tuple (a, z):
                a: Cylindrical radius from planet's axis. Shape (n_x,) or (n_chords, n_x).
                z: Vertical distance from orbital plane. Shape (n_x,) or (n_chords, n_x).
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_p, y_p = self.getPosition(orbphase)
        x_    = np.asarray(x)
        x_p_  = np.asarray(x_p)
        y_p_  = np.asarray(y_p)
        y_    = np.asarray(y)
        z_    = np.asarray(z)
        if x_p_.ndim > 0:                          # batch mode
            x_    = x_[np.newaxis, :]
            x_p_  = x_p_[:, np.newaxis]
            y_p_  = y_p_[:, np.newaxis]
            y_    = y_[:, np.newaxis]
            z_    = z_[:, np.newaxis]
        a = np.sqrt((x_ - x_p_)**2 + (y_ - y_p_)**2)
        return a, z_


class Moon:
    """Represents a moon orbiting a planet.

    Attributes:
        midTransitOrbphase (float): The orbital phase of the moon (relative to
            its planet) at the time of the planet's mid-transit. In radians.
        R (float): The radius of the moon in cm.
        a (float): The semi-major axis of the moon's orbit around the planet in cm.
        hostPlanet (Planet): The host planet object.
    """

    def __init__(self, midTransitOrbphase: float, R: float, a: float, hostPlanet: Planet) -> None:
        """Initializes the Moon object.

        Args:
            midTransitOrbphase (float): The orbital phase of the moon relative to
                its planet at the time of the planet's mid-transit, in radians.
            R (float): The radius of the moon in cm.
            a (float): The semi-major axis of the moon's orbit around the planet in cm.
            hostPlanet (Planet): The host planet object.
        """
        self.midTransitOrbphase: float = midTransitOrbphase
        self.R: float = R
        self.a: float = a
        self.hostPlanet: Planet = hostPlanet

    def getOrbphase(self, orbphase: float) -> float:
        """Calculates the moon's orbital phase around its planet.

        This is scaled by the relative orbital periods of the planet and moon.

        Args:
            orbphase (float): The orbital phase of the host planet in radians.

        Returns:
            float: The orbital phase of the moon around its planet in radians.
        """
        a_p = np.float64(self.hostPlanet.a)
        M_p = np.float64(self.hostPlanet.M)
        a_m = np.float64(self.a)
        M_s = np.float64(self.hostPlanet.hostStar.M)
        period_ratio = np.sqrt((a_p**3 * M_p) / (a_m**3 * M_s))
        orbphase_moon = self.midTransitOrbphase + np.float64(orbphase) * period_ratio
        return orbphase_moon

    def getPosition(self, orbphase: float) -> Tuple[float, float]:
        """Calculates the moon's (x, y) coordinates in the star's frame of reference.

        Args:
            orbphase (float): The orbital phase of the host planet in radians.

        Returns:
            Tuple[float, float]: The x and y coordinates of the moon in cm.
        """
        orbphase_moon = self.getOrbphase(orbphase)
        x_p, y_p = self.hostPlanet.getPosition(orbphase)
        x_moon = x_p + self.a * np.cos(orbphase_moon)
        y_moon = y_p + self.a * np.sin(orbphase_moon)
        return x_moon, y_moon

    def getLOSvelocity(self, orbphase: float) -> float:
        """Calculates the moon's total line-of-sight velocity.

        This is the sum of the planet's velocity and the moon's orbital velocity
        around the planet.

        Args:
            orbphase (float): The orbital phase of the host planet in radians.

        Returns:
            float: The moon's line-of-sight velocity in cm/s.
        """
        v_los_planet = self.hostPlanet.getLOSvelocity(orbphase)
        orbphase_moon = self.getOrbphase(orbphase)
        v_los = v_los_planet - \
            np.sin(orbphase_moon) * \
            np.sqrt(const.G * self.hostPlanet.M / self.a)
        return v_los

    def getDistanceFromMoon(self, x, phi, rho, orbphase) -> np.ndarray:
        """Calculates the 3D distance from a point in space to the moon's center.

        Supports both scalar and batch (array) inputs for phi, rho, orbphase.
        When inputs are arrays of shape (n_chords,), x has shape (n_x,), the
        result is broadcast to shape (n_chords, n_x).

        Args:
            x: Array of coordinates along the line of sight in cm, shape (n_x,).
            phi: Azimuthal angle(s) on the sky plane in radians. Scalar or (n_chords,).
            rho: Projected radial distance(s) from the star's center in cm. Scalar or (n_chords,).
            orbphase: Host planet's orbital phase(s) in radians. Scalar or (n_chords,).

        Returns:
            np.ndarray: Distance(s) to the moon in cm.
                        Shape (n_x,) for scalar inputs, (n_chords, n_x) for array inputs.
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        x_moon, y_moon = self.getPosition(orbphase)
        x_     = np.asarray(x)
        x_m_   = np.asarray(x_moon)
        y_m_   = np.asarray(y_moon)
        y_     = np.asarray(y)
        z_     = np.asarray(z)
        if x_m_.ndim > 0:                          # batch mode
            x_     = x_[np.newaxis, :]             # (1, n_x)
            x_m_   = x_m_[:, np.newaxis]           # (n_chords, 1)
            y_m_   = y_m_[:, np.newaxis]
            y_     = y_[:, np.newaxis]
            z_     = z_[:, np.newaxis]
        r_fromMoon = np.sqrt((x_ - x_m_)**2 + (y_ - y_m_)**2 + z_**2)
        return r_fromMoon


class AvailablePlanets:
    """A utility class to load and manage pre-defined planet and star data.

    This class reads data from 'stars.csv' and 'planets.csv' to populate
    a list of known exoplanetary systems.

    Attributes:
        stars (dict[str, Star]): A dictionary mapping star names to Star objects.
        planetList (list[Planet]): A list of available Planet objects.
    """

    def __init__(self) -> None:
        """Initializes the class by loading data from CSV files."""
        cwd = pathlib.Path(__file__).parent.resolve()
        self.stars: dict[str, Star] = {}
        with open(os.path.join(cwd, '../Resources/stars.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                R = float(row['R_sun']) * const.R_sun
                dR = float(row['dR_sun']) * const.R_sun
                M = float(row['M_sun']) * const.M_sun
                dM = float(row['dM_sun']) * const.M_sun
                T_eff = float(row['T_eff'])
                dT_eff = float(row['dT_eff'])
                log_g = float(row['log_g'])
                dlog_g = float(row['dlog_g'])
                Z = float(row['Fe_H'])
                dZ = float(row['dFe_H'])
                alpha = 0
                self.stars[name] = Star(
                    R, dR, M, dM, T_eff, dT_eff, log_g, dlog_g, Z, dZ, alpha)
        self.planetList: list[Planet] = []
        with open(os.path.join(cwd, '../Resources/planets.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                R = float(row['R_J']) * const.R_J
                M = float(row['M_J']) * const.M_J
                a = float(row['a_AU']) * const.AU
                transitDuration = float(row['transitDuration'])
                hostStarName = row['hostStar']
                hostStar = self.stars.get(hostStarName)
                orbitalPeriod = float(row['P'])
                b = float(row['b'])
                b*= hostStar.R
                if hostStar is not None:
                    planet = Planet(name, R, M, a, hostStar,
                                    transitDuration, orbitalPeriod, b)
                    self.planetList.append(planet)
                else:
                    print(
                        f"Warning: Host star {hostStarName} not found for planet {name}")

    def listPlanetNames(self) -> list[str]:
        """Returns a list of names of all available planets.

        Returns:
            list[str]: A list of planet names.
        """
        planetNames: list[str] = []
        for planet in self.planetList:
            planetNames.append(planet.name)
        return planetNames

    def findPlanet(self, namePlanet: str) -> Optional[Planet]:
        """Finds a planet by its name.

        Args:
            namePlanet (str): The name of the planet to find.

        Returns:
            Optional[Planet]: The Planet object if found, otherwise None.
        """
        for planet in self.planetList:
            if planet.name == namePlanet:
                return planet
        print('System', namePlanet, 'was not found.')
        return None


"""
Convenience constructors and moon orbital-mechanics helpers.

These wrap the body constructors above with research-sensible defaults so a
system can be set up in a couple of lines, and collect the moon:planet
orbital-mechanics relations used to place a moon for transit (e.g. the optimal
mid-transit phase that maximises the moon's sky-plane displacement).  All
lengths are CGS.
"""


def find_planet(name: str) -> Planet:
    """Looks up a pre-defined :class:`Planet` by name.

    Args:
        name (str): Planet name as it appears in ``Resources/planets.csv``.

    Returns:
        Planet: The matching planet object.

    Raises:
        ValueError: If no planet with that name is available.
    """
    planet = AvailablePlanets().findPlanet(name)
    if planet is None:
        raise ValueError(f"Planet {name!r} not found in Prometheus resources.")
    return planet


def make_moon(planet: Planet, a_over_Rp: float = 1.7, R: Optional[float] = None,
              midTransitOrbphase: float = 0.375 * 2.0 * np.pi) -> 'Moon':
    """Builds a :class:`Moon` orbiting ``planet``.

    Args:
        planet (Planet): The host planet.
        a_over_Rp (float): Moon semi-major axis in planet radii (default 1.7).
        R (Optional[float]): Moon radius [cm] (default Io's radius).
        midTransitOrbphase (float): Moon orbital phase at the planet's
            mid-transit [rad].

    Returns:
        Moon: The constructed moon object.
    """
    R = const.R_Io if R is None else R
    return Moon(midTransitOrbphase=midTransitOrbphase, R=R,
                a=a_over_Rp * planet.R, hostPlanet=planet)


#  Optimal moon phase theory (see Tests/midtransit_phase_proof.tex)
#
# The moon's sky-plane offset during transit is
#     y_m(theta) = a_p sin(theta) + a_m sin(theta_0 + N theta),
# with N the moon:planet mean-motion ratio.  Writing eps = a_m/a_p and
# expanding the lightcurve L = f(y_m/a_p) to first order in eps, the only
# time-antisymmetric (i.e. detectable against any symmetric bare-planet
# model) term is  eps sin(theta_0) f'(theta) cos(N theta), so every
# asymmetry observable is proportional to sin(theta_0) and maximised at
# quadrature.  The peak displacement admits an exact all-orders optimum at
# quadrature corrected by the moon's own motion during the displacement.


def mean_motion_ratio(planet: Planet, a_over_Rp: float = 1.7) -> float:
    """Moon:planet mean-motion ratio ``N = sqrt(a_p^3 M_p / (a_m^3 M_star))``.

    Args:
        planet (Planet): The host planet.
        a_over_Rp (float): Moon semi-major axis in planet radii.

    Returns:
        float: The dimensionless mean-motion ratio.
    """
    a_m = a_over_Rp * planet.R
    return float(np.sqrt((planet.a**3 * planet.M) /
                         (a_m**3 * planet.hostStar.M)))


def optimal_midtransit_phase(planet: Planet, a_over_Rp: float = 1.7,
                             branch: str = 'late') -> float:
    """Moon phase at mid-transit maximising the lightcurve peak shift [rad].

    Exact closed form (all orders in a_m/a_p, any monotone cloud profile):
    the moon must reach maximum sky-plane elongation exactly as its cloud
    crosses the stellar disk centre, which displaces the absorption peak by
    the maximum possible +/- arcsin(a_m/a_p) of planet phase.

        theta0* = 3*pi/2 - N*arcsin(a_m/a_p)   (branch='late',  peak after
                                                mid-transit, trailing moon)
        theta0* =   pi/2 + N*arcsin(a_m/a_p)   (branch='early', peak before
                                                mid-transit, leading moon)

    Both branches also sit on the flat |sin(theta0)| plateau of the
    detectability (antisymmetric-RMS) curve, within <1% of its maximum.

    Args:
        planet (Planet): The host planet.
        a_over_Rp (float): Moon semi-major axis in planet radii.
        branch (str): 'late' or 'early' peak displacement.

    Returns:
        float: The optimal mid-transit moon phase [rad].
    """
    N = mean_motion_ratio(planet, a_over_Rp)
    delta = np.arcsin(a_over_Rp * planet.R / planet.a)
    if branch == 'late':
        return float(3 * np.pi / 2 - N * delta)
    if branch == 'early':
        return float(np.pi / 2 + N * delta)
    raise ValueError(f"branch must be 'late' or 'early', got {branch!r}")


def max_peak_shift_minutes(planet: Planet, a_over_Rp: float = 1.7) -> float:
    """Maximum achievable lightcurve peak displacement [minutes].

    Delta_t = (T_p / 2 pi) arcsin(a_m/a_p): the time the planet takes to
    traverse one moon-orbit radius in the sky.  This bound is attained at
    :func:`optimal_midtransit_phase`.

    Args:
        planet (Planet): The host planet.
        a_over_Rp (float): Moon semi-major axis in planet radii.

    Returns:
        float: The maximum peak shift [minutes].
    """
    delta = np.arcsin(a_over_Rp * planet.R / planet.a)
    return float(delta / (2 * np.pi) * planet.orbitalPeriod * 24.0 * 60.0)