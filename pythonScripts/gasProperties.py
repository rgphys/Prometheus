"""
This file defines classes and functions related to the properties of the
gaseous medium in an exoplanet's atmosphere or exosphere. It includes
various models for number density distributions (e.g., barometric, power-law),
classes for atomic and molecular absorbers, and the main `Transit` class that
orchestrates the simulation of a transit light curve.

Created on 19. October 2021 by Andrea Gebek.
"""

import os
from copy import deepcopy
from typing import Any, Callable, List, Tuple, Union

import h5py
import numpy as np
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter as gauss
from scipy.special import erf, lambertw, voigt_profile

from . import constants as const
from . import geometryHandler as geom
from . import memoryHandler as memutil

lineListPath: str = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + '/Resources/LineList.txt'

LINE_LIST = np.loadtxt(lineListPath, dtype=str,
                              usecols=(0, 1, 2, 3, 4), skiprows=1)
molecularLookupPath: str = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + '/Resources/molecularResources/'

@njit(parallel=True, fastmath=True)
def n_interp_log(x_targets, x_grid, y_grid_log, offset):
    """
    Numba-accelerated linear interpolation with O(N+M) two-pointer scan.
    Assumes the last axis of x_targets is monotonically non-decreasing,
    which holds for all Doppler-shifted wavelength grids in Prometheus.
    """
    shape = x_targets.shape
    n_last = shape[-1]
    n_rows = 1
    for d in range(len(shape) - 1):
        n_rows *= shape[d]

    flat = x_targets.reshape(n_rows, n_last)
    out = np.empty((n_rows, n_last), dtype=np.float64)
    m = len(x_grid)

    for row in prange(n_rows):
        j = 0
        for i in range(n_last):
            xi = flat[row, i]
            # Advance pointer — valid because input rows are monotonic
            while j < m - 2 and x_grid[j + 1] < xi:
                j += 1
            if xi <= x_grid[0]:
                log_val = y_grid_log[0]
            elif xi >= x_grid[m - 1]:
                log_val = y_grid_log[m - 1]
            else:
                t = (xi - x_grid[j]) / (x_grid[j + 1] - x_grid[j])
                log_val = y_grid_log[j] + t * (y_grid_log[j + 1] - y_grid_log[j])
            out[row, i] = 10.0 ** log_val - offset

    return out.reshape(shape)


@njit(parallel=True, fastmath=True)
def n_interp_linear_rows(x_targets, x_grid, y_grid_2d):
    """
    Row-wise 1D linear interpolation in *linear* space with O(N+M)
    two-pointer scan.  Each row of y_grid_2d is interpolated independently
    onto the corresponding row of x_targets.

    x_targets: (n_rows, n_out) — monotonically increasing along axis 1
    x_grid:    (M,) — sorted reference grid (shared across all rows)
    y_grid_2d: (n_rows, M) — per-row function values on x_grid
    Returns:   (n_rows, n_out)
    """
    n_rows = x_targets.shape[0]
    n_out = x_targets.shape[1]
    m = len(x_grid)
    out = np.empty((n_rows, n_out), dtype=np.float64)

    for row in prange(n_rows):
        j = 0
        for i in range(n_out):
            xi = x_targets[row, i]
            while j < m - 2 and x_grid[j + 1] < xi:
                j += 1
            if xi <= x_grid[0]:
                out[row, i] = y_grid_2d[row, 0]
            elif xi >= x_grid[m - 1]:
                out[row, i] = y_grid_2d[row, m - 1]
            else:
                t = (xi - x_grid[j]) / (x_grid[j + 1] - x_grid[j])
                out[row, i] = y_grid_2d[row, j] + t * (y_grid_2d[row, j + 1] - y_grid_2d[row, j])

    return out


@njit(parallel=True, fastmath=True)
def _bilinear_PT_interp(P_vals, T_val, P_grid, T_grid, sigma_grid_log, lookupOffset):
    """
    Bilinear interpolation over the (P, T) dimensions of a molecular
    cross-section lookup table, returning sigma in *linear* space on the
    native wavelength grid.  Avoids the cost of a full 3-D scattered
    RegularGridInterpolator evaluation.

    P_vals:         (n_points,)
    T_val:          float (scalar, constant per scenario)
    P_grid:         (n_P,) — sorted pressure axis
    T_grid:         (n_T,) — sorted temperature axis
    sigma_grid_log: (n_P, n_T, n_wav_native) — log10(sigma + offset)
    lookupOffset:   float

    Returns: (n_points, n_wav_native)
    """
    n_points = len(P_vals)
    n_P = len(P_grid)
    n_T = len(T_grid)
    n_wav = sigma_grid_log.shape[2]
    out = np.empty((n_points, n_wav), dtype=np.float64)

    # T bracket (constant for all points)
    ti = np.searchsorted(T_grid, T_val) - 1
    if ti < 0:
        ti = 0
    if ti >= n_T - 1:
        ti = n_T - 2
    t_T = (T_val - T_grid[ti]) / (T_grid[ti + 1] - T_grid[ti])
    w_T0 = 1.0 - t_T
    w_T1 = t_T

    for p in prange(n_points):
        P_val = P_vals[p]
        # P bracket with clamping
        if P_val <= P_grid[0]:
            pi = 0
            t_P = 0.0
        elif P_val >= P_grid[n_P - 1]:
            pi = n_P - 2
            t_P = 1.0
        else:
            pi = np.searchsorted(P_grid, P_val) - 1
            if pi < 0:
                pi = 0
            if pi >= n_P - 1:
                pi = n_P - 2
            t_P = (P_val - P_grid[pi]) / (P_grid[pi + 1] - P_grid[pi])

        w00 = (1.0 - t_P) * w_T0
        w10 = t_P * w_T0
        w01 = (1.0 - t_P) * w_T1
        w11 = t_P * w_T1

        for w in range(n_wav):
            log_val = (w00 * sigma_grid_log[pi, ti, w]
                     + w10 * sigma_grid_log[pi + 1, ti, w]
                     + w01 * sigma_grid_log[pi, ti + 1, w]
                     + w11 * sigma_grid_log[pi + 1, ti + 1, w])
            out[p, w] = 10.0 ** log_val - lookupOffset

    return out

class CollisionalAtmosphere:
    """Base class for a collisional atmosphere with a defined temperature and pressure.

    Attributes:
        T (float): Temperature of the atmosphere in Kelvin.
        P_0 (float): Reference pressure at the base of the atmosphere in cgs units.
        constituents (List[Union['AtmosphericConstituent', 'MolecularConstituent']]):
            List of absorbing species in the atmosphere.
        hasMoon (bool): Flag indicating if a moon is present in this model.
    """

    def __init__(self, T: float, P_0: float):
        """Initializes the CollisionalAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
        """
        self.T: float = T
        self.P_0: float = P_0
        self.constituents: List[Union['AtmosphericConstituent',
                                      'MolecularConstituent']] = []
        self.hasMoon: bool = False

    def getReferenceNumberDensity(self) -> float:
        """Calculates the number density at the reference pressure and temperature.

        Returns:
            float: The reference number density (n_0) in cm^-3.
        """
        n_0 = self.P_0 / (const.k_B * self.T)
        return n_0

    def getVelDispersion(self, m: float) -> float:
        """Calculates the thermal velocity dispersion for a given particle mass.

        Args:
            m (float): Mass of the particle in grams.

        Returns:
            float: The 1D thermal velocity dispersion (sigma_v) in cm/s.
        """
        sigma_v = np.sqrt(self.T * const.k_B / m)
        return sigma_v

    def addConstituent(self, speciesName: str, chi: float) -> None:
        """Adds an atomic or ionic constituent to the atmosphere.

        Args:
            speciesName (str): The name of the species (e.g., 'NaI').
            chi (float): The mixing ratio of this species.
        """
        species = const.AvailableSpecies().findSpecies(speciesName)
        m = species.mass
        sigma_v = self.getVelDispersion(m)
        constituent = AtmosphericConstituent(species, chi, sigma_v)
        self.constituents.append(constituent)

    def addMolecularConstituent(self, speciesName: str, chi: float) -> None:
        """Adds a molecular constituent to the atmosphere.

        Args:
            speciesName (str): The name of the molecule.
            chi (float): The mixing ratio of this molecule.
        """
        constituent = MolecularConstituent(speciesName, chi)
        self.constituents.append(constituent)

    def addScatteringConstituent(self, scattererType: str, paramsDict: dict) -> None:
        """Adds a continuum scattering/aerosol constituent to the atmosphere.

        Args:
            scattererType (str): One of `SCATTERER_TYPES`.
            paramsDict (dict): Parameter dictionary from the setup file.
        """
        self.constituents.append(makeScatteringConstituent(scattererType, paramsDict))


class BarometricAtmosphere(CollisionalAtmosphere):
    """An isothermal atmosphere with a density profile following the barometric formula.

    Attributes:
        mu (float): The mean molecular weight of the atmosphere in grams.
        planet (Any): The Planet object this atmosphere belongs to.
    """

    def __init__(self, T: float, P_0: float, mu: float, planet: Any):
        """Initializes the BarometricAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
            mu (float): The mean molecular weight of the atmosphere in grams.
            planet (Any): The Planet object this atmosphere belongs to.
        """
        super().__init__(T, P_0)
        self.mu: float = mu
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = BarometricAtmosphere.getReferenceNumberDensity(self)
        H = const.k_B * self.T * self.planet.R**2 / \
            (const.G * self.mu * self.planet.M)
        n = n_0 * np.exp((self.planet.R - r) / H) * \
            np.heaviside(r - self.planet.R, 1.)
        return n


class HydrostaticAtmosphere(CollisionalAtmosphere):
    """An isothermal atmosphere in hydrostatic equilibrium.

    Attributes:
        mu (float): The mean molecular weight of the atmosphere in grams.
        planet (Any): The Planet object this atmosphere belongs to.
    """

    def __init__(self, T: float, P_0: float, mu: float, planet: Any):
        """Initializes the HydrostaticAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
            mu (float): The mean molecular weight of the atmosphere in grams.
            planet (Any): The Planet object this atmosphere belongs to.
        """
        super().__init__(T, P_0)
        self.mu: float = mu
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = HydrostaticAtmosphere.getReferenceNumberDensity(self)
        Jeans_0 = const.G * self.mu * self.planet.M / \
            (const.k_B * self.T * self.planet.R)
        Jeans = const.G * self.mu * self.planet.M / \
            (const.k_B * self.T * r) * np.heaviside(r - self.planet.R, 1.)
        n = n_0 * np.exp(Jeans - Jeans_0)
        return n


class PowerLawAtmosphere(CollisionalAtmosphere):
    """An atmosphere with a density profile following a power law.

    Attributes:
        q (float): The power-law index for the density profile.
        planet (Any): The Planet object this atmosphere belongs to.
    """

    def __init__(self, T: float, P_0: float, q: float, planet: Any):
        """Initializes the PowerLawAtmosphere.

        Args:
            T (float): Temperature of the atmosphere in Kelvin.
            P_0 (float): Reference pressure at the base in cgs units.
            q (float): The power-law index.
            planet (Any): The Planet object this atmosphere belongs to.
        """
        super().__init__(T, P_0)
        self.q: float = q
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density at each x coordinate (cm^-3).
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = PowerLawAtmosphere.getReferenceNumberDensity(self)
        n = n_0 * (self.planet.R / r)**self.q * \
            np.heaviside(r - self.planet.R, 1.)
        return n


class EvaporativeExosphere:
    """Base class for an exosphere model normalized by total particle number.

    Attributes:
        N (float): Total number of particles in the exosphere.
        hasMoon (bool): Flag indicating if a moon is present in this model.
    """

    def __init__(self, N: float):
        """Initializes the EvaporativeExosphere.

        Args:
            N (float): Total number of particles in the exosphere.
        """
        self.N: float = N
        self.hasMoon: bool = False

    def addConstituent(self, speciesName: str, sigma_v: float) -> None:
        """Adds an atomic or ionic constituent to the exosphere.

        Note: An evaporative exosphere can only have one constituent.

        Args:
            speciesName (str): The name of the species (e.g., 'NaI').
            sigma_v (float): The velocity dispersion of the species in cm/s.
        """
        species = const.AvailableSpecies().findSpecies(speciesName)
        constituent = AtmosphericConstituent(species, 1., sigma_v)
        self.constituents: List[AtmosphericConstituent] = [constituent]

    def addMolecularConstituent(self, speciesName: str, T: float) -> None:
        """Adds a molecular constituent to the exosphere.

        Note: An evaporative exosphere can only have one constituent.

        Args:
            speciesName (str): The name of the molecule.
            T (float): The pseudo-temperature for the molecular cross-sections.
        """
        constituent = MolecularConstituent(speciesName, 1.0)
        self.constituents: List[MolecularConstituent] = [constituent]
        self.T: float = T

    def addScatteringConstituent(self, scattererType: str, paramsDict: dict) -> None:
        """Adds a continuum scattering/aerosol constituent to the exosphere.

        Unlike the atomic/molecular constituents of an evaporative exosphere
        (which replace the single absorber), scattering constituents are
        appended so they can coexist with another absorber.

        Args:
            scattererType (str): One of `SCATTERER_TYPES`.
            paramsDict (dict): Parameter dictionary from the setup file.
        """
        if not hasattr(self, 'constituents') or self.constituents is None:
            self.constituents = []
        self.constituents.append(makeScatteringConstituent(scattererType, paramsDict))


class PowerLawExosphere(EvaporativeExosphere):
    """An exosphere with a density profile following a power law.

    Attributes:
        q (float): The power-law index for the density profile.
        planet (Any): The Planet object this exosphere belongs to.
    """

    def __init__(self, N: float, q: float, planet: Any):
        """Initializes the PowerLawExosphere.

        Args:
            N (float): Total number of particles in the exosphere.
            q (float): The power-law index.
            planet (Any): The Planet object this exosphere belongs to.
        """
        super().__init__(N)
        self.q: float = q
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi, rho, orbphase) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Supports both scalar and batch (array) inputs for phi, rho, orbphase.
        When inputs are arrays of shape (n_chords,), returns (n_chords, n_x).

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi: Azimuthal angle(s) on the sky plane (radians). Scalar or (n_chords,).
            rho: Projected radial distance(s) from star's center (cm). Scalar or (n_chords,).
            orbphase: Planet's orbital phase(s) (radians). Scalar or (n_chords,).

        Returns:
            np.ndarray: Number density. Shape (n_x,) for scalar, (n_chords, n_x) for batch.
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.planet.R**3) * self.N
        n = n_0 * (self.planet.R / r)**self.q * \
            np.heaviside(r - self.planet.R, 1.)
        return n


class MoonExosphere(EvaporativeExosphere):
    """An exosphere sourced from a moon, with a power-law density profile.

    Attributes:
        q (float): The power-law index for the density profile.
        moon (Any): The Moon object this exosphere belongs to.
        planet (Any): The host planet of the moon.
    """

    def __init__(self, N: float, q: float, moon: Any):
        """Initializes the MoonExosphere.

        Args:
            N (float): Total number of particles in the exosphere.
            q (float): The power-law index.
            moon (Any): The Moon object this exosphere belongs to.
        """
        super().__init__(N)
        self.q: float = q
        self.moon: Any = moon
        self.hasMoon: bool = True
        self.planet: Any = moon.hostPlanet

    def calculateNumberDensity(self, x: np.ndarray, phi, rho, orbphase) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Supports both scalar and batch (array) inputs for phi, rho, orbphase.
        When inputs are arrays of shape (n_chords,), returns (n_chords, n_x).

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi: Azimuthal angle(s) on the sky plane (radians). Scalar or (n_chords,).
            rho: Projected radial distance(s) from star's center (cm). Scalar or (n_chords,).
            orbphase: Planet's orbital phase(s) (radians). Scalar or (n_chords,).

        Returns:
            np.ndarray: Number density. Shape (n_x,) for scalar, (n_chords, n_x) for batch.
        """
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * self.N
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n


class TidallyHeatedMoon(EvaporativeExosphere):
    """A moon exosphere with a variable source rate dependent on orbital phase.

    This model is designed to simulate phenomena like volcanic activity on a
    tidally heated moon, where the outgassing rate changes with orbital position.

    Attributes:
        q (float): The power-law index for the density profile.
        moon (Any): The Moon object this exosphere belongs to.
        planet (Any): The host planet of the moon.
        N_function (Callable[[float], float]): An interpolation function that
            returns the total number of particles as a function of the moon's
            orbital phase.
    """

    def __init__(self, q: float, moon: Any):
        """Initializes the TidallyHeatedMoon model.

        Args:
            q (float): The power-law index for the density profile.
            moon (Any): The Moon object this exosphere belongs to.
        """
        self.q: float = q
        self.moon: Any = moon
        self.hasMoon: bool = True
        self.planet: Any = moon.hostPlanet

    def addSourceRateFunction(self, filename: str, tau_photoionization: float, mass_absorber: float) -> None:
        """Loads a source rate profile and creates an interpolation function.

        The total number of particles `N` at any time is calculated as
        `M_dot * tau / m`, where M_dot is the mass loss rate, tau is the
        photoionization lifetime, and m is the particle mass.

        Args:
            filename (str): Path to the file containing the mass loss rate (M_dot)
                as a function of the moon's orbital phase.
            tau_photoionization (float): The photoionization lifetime of the
                absorbing species in seconds.
            mass_absorber (float): The mass of a single absorbing particle in grams.
        """
        Mdot = np.loadtxt(filename)
        Mdot = np.concatenate((Mdot, Mdot[::-1]))
        phi_moon = np.linspace(0., 2. * np.pi, len(Mdot))
        N_function = interp1d(phi_moon, np.log10(
            Mdot * tau_photoionization / mass_absorber))
        self.N_function: Callable[[float], float] = N_function

    def calculateAbsorberNumber(self, orbphase: float) -> float:
        """Calculates the total number of absorbers for a given planetary orbital phase.

        Args:
            orbphase (float): The planet's orbital phase in radians.

        Returns:
            float: The total number of particles `N` in the exosphere.
        """
        orbphase_moon = self.moon.getOrbphase(orbphase) % (2. * np.pi)
        N = 10**self.N_function(orbphase_moon)
        return N

    def calculateNumberDensity(self, x: np.ndarray, phi, rho, orbphase) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Supports both scalar and batch (array) inputs for phi, rho, orbphase.
        When inputs are arrays of shape (n_chords,), returns (n_chords, n_x).

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi: Azimuthal angle(s) on the sky plane (radians). Scalar or (n_chords,).
            rho: Projected radial distance(s) from star's center (cm). Scalar or (n_chords,).
            orbphase: Planet's orbital phase(s) (radians). Scalar or (n_chords,).

        Returns:
            np.ndarray: Number density. Shape (n_x,) for scalar, (n_chords, n_x) for batch.
        """
        N = self.calculateAbsorberNumber(orbphase)
        r = self.moon.getDistanceFromMoon(x, phi, rho, orbphase)
        # N may be a scalar or (n_chords,); broadcast to match r
        N_ = np.asarray(N)
        if N_.ndim > 0:
            N_ = N_[:, np.newaxis]     # (n_chords, 1) to broadcast with (n_chords, n_x)
        n_0 = (self.q - 3.) / (4. * np.pi * self.moon.R**3) * N_
        n = n_0 * (self.moon.R / r)**self.q * np.heaviside(r - self.moon.R, 1.)
        return n


class TorusExosphere(EvaporativeExosphere):
    """An exosphere model of a neutral gas torus around a planet.

    The density profile is Gaussian in both the radial and vertical directions
    of the torus.

    Attributes:
        a_torus (float): The radius of the torus centerline in cm.
        v_ej (float): The ejection velocity of particles, which determines the
            torus scale height, in cm/s.
        planet (Any): The Planet object this torus surrounds.
    """

    def __init__(self, N: float, a_torus: float, v_ej: float, planet: Any):
        """Initializes the TorusExosphere.

        Args:
            N (float): Total number of particles in the torus.
            a_torus (float): The radius of the torus centerline in cm.
            v_ej (float): The ejection velocity of particles in cm/s.
            planet (Any): The Planet object this torus surrounds.
        """
        super().__init__(N)
        self.a_torus: float = a_torus
        self.v_ej: float = v_ej
        self.planet: Any = planet

    def calculateNumberDensity(self, x: np.ndarray, phi, rho, orbphase) -> np.ndarray:
        """Calculates the number density at a given point in space.

        Supports both scalar and batch (array) inputs for phi, rho, orbphase.
        When inputs are arrays of shape (n_chords,), returns (n_chords, n_x).

        Args:
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi: Azimuthal angle(s) on the sky plane (radians). Scalar or (n_chords,).
            rho: Projected radial distance(s) from star's center (cm). Scalar or (n_chords,).
            orbphase: Planet's orbital phase(s) (radians). Scalar or (n_chords,).

        Returns:
            np.ndarray: Number density. Shape (n_x,) for scalar, (n_chords, n_x) for batch.
        """
        a, z = self.planet.getTorusCoords(x, phi, rho, orbphase)
        v_orbit = np.sqrt(const.G * self.planet.M / self.a_torus)
        H_torus = self.a_torus * self.v_ej / v_orbit
        n_a = np.exp(-((a - self.a_torus) / (4. * H_torus))**2)
        n_z = np.exp(-(z / H_torus)**2)
        term1 = 8. * H_torus**2 * np.exp(-self.a_torus**2 / (16. * H_torus**2))
        term2 = 2. * np.sqrt(np.pi) * self.a_torus * H_torus * \
            (erf(self.a_torus / (4. * H_torus)) + 1.)
        n_0 = 1. / (2. * np.pi**1.5 * H_torus * (term1 + term2)) * self.N
        n = n_0 * np.multiply(n_a, n_z)
        return n


class SerpensExosphere(EvaporativeExosphere):
    """An exosphere model based on particle data from the SERPENS simulation code.

    This class loads particle positions from a SERPENS output file, histograms
    them onto a 3D grid, and creates an interpolation function for the
    number density.

    Attributes:
        filename (str): Path to the SERPENS output file.
        planet (Any): The Planet object this exosphere belongs to.
        sigmaSmoothing (float): The sigma for Gaussian smoothing of the
            histogrammed density grid.
        InterpolatedDensity (Callable): A 3D interpolation function for number density.
    """

    def __init__(self, filename: str, N: float, planet: Any, sigmaSmoothing: float):
        """Initializes the SerpensExosphere.

        Args:
            filename (str): Path to the SERPENS output file.
            N (float): Total number of particles to scale the simulation to.
            planet (Any): The Planet object this exosphere belongs to.
            sigmaSmoothing (float): The sigma for Gaussian smoothing in grid units.
        """
        super().__init__(N)
        self.filename: str = filename
        self.planet: Any = planet
        self.sigmaSmoothing: float = sigmaSmoothing

    def addInterpolatedDensity(self, spatialGrid: Any) -> None:
        """Loads SERPENS data and creates the density interpolation function.

        Args:
            spatialGrid (Any): The `geometryHandler.Grid` object defining the
                simulation grid.
        """
        serpensOutput = np.loadtxt(self.filename) * 1e2
        particlePos = serpensOutput[:, 0:3]
        xBins = spatialGrid.constructXaxis(midpoints=False)
        yBins = np.linspace(-spatialGrid.rho_border,
                            spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        zBins = np.linspace(-spatialGrid.rho_border,
                            spatialGrid.rho_border, 2 * int(spatialGrid.rho_steps) + 1)
        cellVolume = (xBins[1] - xBins[0]) * \
            (yBins[1] - yBins[0]) * (zBins[1] - zBins[0])
        n_histogram = np.histogramdd(particlePos, bins=[xBins, yBins, zBins])[
            0] * self.N / (np.size(particlePos, axis=0) * cellVolume)
        if self.sigmaSmoothing > 0.:
            n_histogram = gauss(n_histogram, sigma=self.sigmaSmoothing)
        print('Sum over all particles, potentially smoothed with a Gaussian:', np.sum(
            n_histogram) * cellVolume)
        xPoints = spatialGrid.constructXaxis()
        yPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(
            spatialGrid.rho_steps), endpoint=False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        zPoints = np.linspace(-spatialGrid.rho_border, spatialGrid.rho_border, 2 * int(
            spatialGrid.rho_steps), endpoint=False) + 2. * spatialGrid.rho_border / (4. * spatialGrid.rho_steps)
        x, y, z = np.meshgrid(xPoints, yPoints, zPoints, indexing='ij')
        SEL = ((y**2 + z**2) > self.planet.R**2) * \
            ((y**2 + z**2) < self.planet.hostStar.R**2)
        print('Sum over all particles outside of the planetary disk but inside the stellar disk:', np.sum(
            n_histogram[SEL]) * cellVolume)
        n_function = RegularGridInterpolator(
            (xPoints, yPoints, zPoints), n_histogram,
            bounds_error=False, fill_value=0.0)
        self.InterpolatedDensity: Callable[[
            np.ndarray], np.ndarray] = n_function

    def calculateNumberDensity(self, x: np.ndarray, phi, rho, orbphase) -> np.ndarray:
        """Calculates number density using the pre-computed interpolation function.

        Fully vectorized over a batch of chords: ``phi`` and ``rho`` may be
        scalars (one chord) or 1-D arrays of length ``n_chords``.  This mirrors
        the batched contract of the other density models so the SERPENS
        exosphere works with the vectorized ``getLOSopticalDepth_Batch`` path.

        Args:
            x (np.ndarray): Coordinates along the line of sight (cm), shape (n_x,).
            phi (float | np.ndarray): Azimuthal sky-plane angle(s) (radians).
            rho (float | np.ndarray): Projected radial distance(s) from the star
                centre (cm).
            orbphase (float): Planet's orbital phase (radians); unused here as the
                SERPENS density field is static in the transit frame.

        Returns:
            np.ndarray: Number density (cm^-3).  Shape ``(n_x,)`` for scalar
            ``phi``/``rho``, or ``(n_chords, n_x)`` for batched input.
        """
        scalar_in = np.ndim(phi) == 0 and np.ndim(rho) == 0
        x = np.atleast_1d(x)
        phi_arr = np.atleast_1d(phi)
        rho_arr = np.atleast_1d(rho)
        y, z = geom.Grid.getCartesianFromCylinder(phi_arr, rho_arr)   # (n_chords,)
        n_chords, n_x = phi_arr.size, x.size
        X = np.broadcast_to(x[None, :], (n_chords, n_x))
        Y = np.broadcast_to(np.atleast_1d(y)[:, None], (n_chords, n_x))
        Z = np.broadcast_to(np.atleast_1d(z)[:, None], (n_chords, n_x))
        coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        n = self.InterpolatedDensity(coords).reshape(n_chords, n_x)
        return n[0] if scalar_in else n


class RadialWindExosphere(EvaporativeExosphere):
    """Radially expanding planetary wind with a beta-law velocity profile.

    The number density follows from mass continuity:
        n(r) = Mdot / (4π r² v(r) μ_particle)
    with the beta-law velocity profile:
        v(r) = v_terminal * max(1 − r_inner/r, 0)^beta.

    Two velocity laws are supported, selected by ``wind_model``:

    * ``'beta'`` (default) — a parametrized (beta-law) wind.  Setting beta ≈ 1
      and v_terminal near the local sound speed gives a first-order
      approximation to a thermally driven Parker wind.
    * ``'parker'`` — the exact **isothermal Parker wind** transonic solution,
      obtained in closed form via the Lambert-W function (see
      :meth:`_wind_velocity_parker`).  This has no free ``beta``/``v_terminal``/
      ``v_base`` knobs; the profile is fixed by the wind temperature ``T`` and
      the planet mass, with ``Mdot`` setting only the density normalization.

      A trace heavy species (e.g. Na) does not drive its own transonic wind —
      its sonic point lies far outside the line-forming region — but is instead
      *advected* by the bulk (light, H/He) outflow.  Pass ``wind_mu`` to set the
      mean particle mass that fixes the Parker **dynamics** (sound speed, sonic
      radius, hence v(r)) to that of the bulk gas, while ``mu`` continues to set
      the **density normalization** n = Mdot / (4π r² v μ) for the tracer.  When
      ``wind_mu`` is omitted the dynamics use ``mu`` (a self-driven wind).

    A *modified* beta velocity law is used (for ``wind_model='beta'``) so the
    outflow is launched with a
    finite base speed ``v_base`` at ``r_inner``:

        v(r) = v_base + (v_terminal − v_base) · max(1 − r_inner/r, 0)^beta.

    The finite base velocity represents the (subsonic, ≈ sound-speed) wind
    launch speed of a transonic outflow.  Physically it sets the wind base
    density ρ_base = Ṁ / (4π r_inner² v_base μ) and removes the unphysical
    density divergence that a pure beta law (v → 0 at r_inner) produces through
    mass continuity — no arbitrary post-hoc density/velocity floor is required.

    When Doppler orbital motion is enabled, :meth:`calculateLOSVelocity` is
    called automatically by :meth:`Atmosphere.getLOSopticalDepth_Batch` to
    apply a position-dependent (per-x) Doppler shift to atomic absorbers.

    Attributes:
        Mdot (float): Mass loss rate [g/s].
        mu (float): Mean particle mass [g].
        v_terminal (float): Terminal wind speed [cm/s].
        beta (float): Beta-law exponent (default 1.0).
        r_inner (float): Inner wind boundary [cm]; defaults to planet.R.
        r_outer (float or None): Optional outer cutoff [cm].
        v_base (float): Wind launch speed at r_inner [cm/s].  Defaults to
            v_terminal × 1e-3 (a small but finite subsonic base speed); set it
            to the sound speed at the base for a transonic, Parker-like wind.
        planet (Any): Host planet object.
    """

    def __init__(self, Mdot: float, mu: float, v_terminal: float = None,
                 beta: float = 1.0, r_inner: float = None,
                 r_outer: float = None, v_base: float = None,
                 wind_model: str = 'beta', T: float = None,
                 planet: Any = None, wind_mu: float = None):
        super().__init__(1.0)  # N placeholder; density set by mass continuity
        self.Mdot: float = Mdot
        self.mu: float = mu
        self.beta: float = beta
        self.planet: Any = planet
        self.r_inner: float = r_inner if r_inner is not None else planet.R
        self.r_outer: float = r_outer
        self.wind_model: str = wind_model

        if wind_model == 'parker':
            if T is None:
                raise ValueError("wind_model='parker' requires a wind "
                                 "temperature T [K].")
            if planet is None:
                raise ValueError("wind_model='parker' requires a planet "
                                 "(for the gravitational sonic point).")
            self.T: float = T
            # Parker dynamics are set by the bulk escaping gas mean mass
            # ``wind_mu`` (defaults to ``mu`` for a self-driven wind); a trace
            # species keeps its own ``mu`` for the density normalization only.
            self.wind_mu: float = wind_mu if wind_mu is not None else mu
            # Isothermal sound speed and sonic-point radius (bulk dynamics).
            self.c_s: float = np.sqrt(const.k_B * T / self.wind_mu)
            self.r_c: float = const.G * planet.M * self.wind_mu / (2.0 * const.k_B * T)
            # v_terminal/v_base are unused by the Parker solution.
            self.v_terminal: float = v_terminal
            self.v_base: float = None
        elif wind_model == 'beta':
            if v_terminal is None:
                raise ValueError("wind_model='beta' requires v_terminal [cm/s].")
            self.v_terminal: float = v_terminal
            # Finite launch speed; default to 0.1% of terminal speed.  Guard
            # against non-positive values, which would re-introduce the density
            # divergence.
            v_base = v_base if v_base is not None else v_terminal * 1e-3
            self.v_base: float = v_base if v_base > 0. else v_terminal * 1e-3
        else:
            raise ValueError(f"Unknown wind_model {wind_model!r}; "
                             "use 'beta' or 'parker'.")

    def _wind_velocity(self, r: np.ndarray) -> np.ndarray:
        """Wind speed at radius ``r``, dispatched by :attr:`wind_model`."""
        if self.wind_model == 'parker':
            return self._wind_velocity_parker(r)
        # Modified beta-law velocity with a finite base speed:
        # v(r) = v_base + (v_terminal − v_base) · max(1 − r_inner/r, 0)^beta,
        # so v(r_inner) = v_base > 0 and the mass-continuity number density
        # stays finite everywhere in the wind.
        r_safe = np.where(r > 0, r, self.r_inner)
        arg = np.clip(1.0 - self.r_inner / r_safe, 0.0, None)
        return self.v_base + (self.v_terminal - self.v_base) * arg ** self.beta

    def _wind_velocity_parker(self, r: np.ndarray) -> np.ndarray:
        """Isothermal Parker-wind speed via the Lambert-W solution [cm/s].

        The steady, isothermal, spherically symmetric momentum + continuity
        equations reduce to the dimensionless transonic relation

            (v/c_s)² − ln[(v/c_s)²] = 4 ln(r/r_c) + 4 r_c/r − 3 ≡ D(r),

        with isothermal sound speed c_s = √(k_B T / μ) and sonic radius
        r_c = G M μ / (2 k_B T).  Writing y = (v/c_s)² this is y − ln y = D,
        whose closed form is

            y = −W_b(−e^{−D}),

        where W is the Lambert-W function: the principal branch (b = 0) gives
        the subsonic solution for r < r_c, and the W₋₁ branch gives the
        supersonic solution for r > r_c.  The argument −e^{−D} ∈ [−1/e, 0)
        because D ≥ 1 (minimum at the sonic point), so the solution is real
        everywhere.

        Because the wind is launched at a finite, sub-sonic speed, the
        mass-continuity number density n ∝ 1/(r² v) stays finite with no
        free base-velocity floor.
        """
        r_safe = np.where(r > 0, r, self.r_inner)
        lam = r_safe / self.r_c
        D = 4.0 * np.log(lam) + 4.0 / lam - 3.0
        # Argument of W lies in [-1/e, 0).  Floor it at the next double toward
        # zero from the branch point -1/e: scipy's lambertw returns NaN exactly
        # at -1/e, and any numerical overshoot just below it is unphysical.
        arg = np.maximum(-np.exp(-D), np.nextafter(-1.0 / np.e, 0.0))
        # lambertw is not vectorized over the branch index, so evaluate both
        # real branches and select per cell (subsonic inside r_c, supersonic
        # outside).
        w0 = np.real(lambertw(arg, 0))
        wm1 = np.real(lambertw(arg, -1))
        w = np.where(r_safe > self.r_c, wm1, w0)
        return self.c_s * np.sqrt(np.maximum(-w, 0.0))

    def calculateNumberDensity(self, x: np.ndarray, phi, rho, orbphase) -> np.ndarray:
        """Number density from mass continuity n = Mdot / (4π r² v(r) μ).

        Supports scalar and batch (array) phi, rho, orbphase inputs.

        Args:
            x (np.ndarray): LOS coordinates (n_x,) [cm].
            phi: Azimuthal angle(s) [rad].  Scalar or (n_chords,).
            rho: Projected radius (radii) [cm].  Scalar or (n_chords,).
            orbphase: Orbital phase(s) [rad].  Scalar or (n_chords,).

        Returns:
            np.ndarray: Number density (n_x,) or (n_chords, n_x) [cm⁻³].
        """
        r = self.planet.getDistanceFromPlanet(x, phi, rho, orbphase)
        v = self._wind_velocity(r)
        # Guard r=0 in intermediate division; the mask below zeroes these cells.
        r_safe = np.where(r > 0, r, 1.0)
        n = self.Mdot / (4.0 * np.pi * r_safe ** 2 * v * self.mu)
        mask = np.heaviside(r - self.r_inner, 0.0)
        if self.r_outer is not None:
            mask *= np.heaviside(self.r_outer - r, 0.0)
        return n * mask

    def calculateLOSVelocity(self, x_grid: np.ndarray, phi_batch: np.ndarray,
                              rho_batch: np.ndarray,
                              orbphase_batch: np.ndarray) -> np.ndarray:
        """LOS projection of the radial outflow velocity (n_chords, n_x) [cm/s].

        The sign convention matches :meth:`celestialBodies.Planet.getLOSvelocity`:
        the returned value is the velocity component along +x, i.e. **positive
        for gas moving away from the observer (redshift)**, since the observer
        sits at x = −∞.  This is essential because the result is summed with the
        bulk orbital velocity ``v_bulk`` (also in the +x convention) inside
        :meth:`Atmosphere.getLOSopticalDepth_Batch` before the Doppler shift is
        applied.  The bulk orbital velocity is NOT included here.

        For a radial outflow the gas velocity vector is v_radial · r̂, whose +x
        component is v_radial · (x − x_p)/r.  Gas behind the planet (x > x_p,
        dx > 0) recedes from the observer (redshift); near-side gas (dx < 0)
        approaches (blueshift).

        Args:
            x_grid (np.ndarray): LOS positions (n_x,) [cm].
            phi_batch (np.ndarray): Azimuthal angles (n_chords,) [rad].
            rho_batch (np.ndarray): Projected radii (n_chords,) [cm].
            orbphase_batch (np.ndarray): Orbital phases (n_chords,) [rad].

        Returns:
            np.ndarray: LOS velocity field (n_chords, n_x) [cm/s].
        """
        r = self.planet.getDistanceFromPlanet(x_grid, phi_batch, rho_batch, orbphase_batch)
        v_radial = self._wind_velocity(r)  # (n_chords, n_x)
        x_p = self.planet.a * np.cos(orbphase_batch)        # (n_chords,)
        dx = x_grid[np.newaxis, :] - x_p[:, np.newaxis]    # (n_chords, n_x); positive = behind planet
        r_safe = np.where(r > 0, r, 1.0)
        # +x component of the radial outflow (away-from-observer = redshift > 0),
        # matching the sign convention of planet.getLOSvelocity.
        return v_radial * dx / r_safe


"""
Calculate absorption cross sections
"""


class AtmosphericConstituent:
    """Represents an atomic or ionic absorbing species.

    This class handles the calculation of absorption cross-sections for a given
    species by computing Voigt profiles for its spectral lines.

    Attributes:
        isMolecule (bool): Flag indicating this is not a molecule.
        species (Any): The `constants.Species` object.
        chi (float): The mixing ratio of this species.
        sigma_v (float): The velocity dispersion in cm/s.
        lookupFunction (Callable): An interpolation function for the absorption
            cross-section (`log10(sigma_abs)`) vs. wavelength.
    """

    def __init__(self, species: Any, chi: float, sigma_v: float):
        """Initializes the AtmosphericConstituent.

        Args:
            species (Any): The `constants.Species` object.
            chi (float): The mixing ratio of this species.
            sigma_v (float): The velocity dispersion in cm/s.
        """
        self.isMolecule: bool = False
        self.species: Any = species
        self.chi: float = chi
        self.sigma_v: float = sigma_v
        self.wavelengthGridRefinement: float = 10.
        self.wavelengthGridExtension: float = 0.01
        self.lookupOffset: float = 1e-50

    def getLineParameters(self, wavelength: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieves spectral line parameters from a data file.

        Reads the line list file and returns the parameters for the lines of this
        species that fall within the specified wavelength range.

        Args:
            wavelength (np.ndarray): A 2-element array with the min and max
                wavelengths of interest [cm].

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - line_wavelength (np.ndarray): Wavelengths of the lines [cm].
                - line_gamma (np.ndarray): Damping parameters (Gamma) of the lines.
                - line_f (np.ndarray): Oscillator strengths (f-values) of the lines.
        """
        line_wavelength = LINE_LIST[:, 2]
        line_A = LINE_LIST[:, 3]
        line_f = LINE_LIST[:, 4]
        SEL_COMPLETE = (line_wavelength != '') * \
            (line_A != '') * (line_f != '')
        SEL_SPECIES = (LINE_LIST[:, 0] == self.species.element) * \
            (LINE_LIST[:, 1] == self.species.ionizationState)
        line_wavelength = line_wavelength[SEL_SPECIES *
                                          SEL_COMPLETE].astype(float) * 1e-8
        line_gamma = line_A[SEL_SPECIES *
                            SEL_COMPLETE].astype(float) / (4. * np.pi)
        line_f = line_f[SEL_SPECIES * SEL_COMPLETE].astype(float)
        SEL_WAVELENGTH = (line_wavelength > min(wavelength)) * \
            (line_wavelength < max(wavelength))
        return line_wavelength[SEL_WAVELENGTH], line_gamma[SEL_WAVELENGTH], line_f[SEL_WAVELENGTH]

    def calculateVoigtProfile(self, wavelength: np.ndarray) -> np.ndarray:
        """Calculates the total absorption cross-section from all spectral lines.

        The profile is a sum of Voigt profiles for each line of the species.

        Args:
            wavelength (np.ndarray): Wavelength grid to calculate the profile on [cm].

        Returns:
            np.ndarray: The absorption cross-section (sigma_abs) in cm^2.
        """
        line_wavelength, line_gamma, line_f = self.getLineParameters(
            wavelength)
        sigma_abs = np.zeros_like(wavelength)
        for idx in range(len(line_wavelength)):
            lineProfile = voigt_profile(
                const.c / wavelength - const.c / line_wavelength[idx], self.sigma_v / line_wavelength[idx], line_gamma[idx])
            sigma_abs += np.pi * \
                (const.e)**2 / (const.m_e * const.c) * \
                line_f[idx] * lineProfile
        return sigma_abs

    def constructLookupFunction(self, wavelengthGrid: 'WavelengthGrid') -> Callable[[np.ndarray], np.ndarray]:
        """Creates an interpolation function for the absorption cross-section.

        Calculates the Voigt profile on a refined grid and creates a 1D
        interpolator for `log10(sigma_abs)` to speed up subsequent calculations.

        Args:
            wavelengthGrid (WavelengthGrid): The wavelength grid object for the simulation.

        Returns:
            Callable[[np.ndarray], np.ndarray]: The interpolation function.
        """
        wavelengthGridRefined = deepcopy(wavelengthGrid)
        wavelengthGridRefined.resolutionHigh /= self.wavelengthGridRefinement
        wavelengthGridRefined.lower_w *= (1. - self.wavelengthGridExtension)
        wavelengthGridRefined.upper_w *= (1. + self.wavelengthGridExtension)
        wavelengthRefined = wavelengthGridRefined.constructWavelengthGridSingle(
            self)
        sigma_abs = self.calculateVoigtProfile(wavelengthRefined)
        lookupFunction = interp1d(wavelengthRefined, np.log10(
            sigma_abs + self.lookupOffset), bounds_error=False, fill_value=np.log10(self.lookupOffset))
        return lookupFunction

    def addLookupFunctionToConstituent(self, wavelengthGrid: 'WavelengthGrid') -> None:
        """Constructs and attaches the lookup function to the object instance.

        Args:
            wavelengthGrid (WavelengthGrid): The simulation's wavelength grid object.
        """
        lookupFunction = self.constructLookupFunction(wavelengthGrid)
        self.lookupFunction: Callable[[
            np.ndarray], np.ndarray] = lookupFunction

    def getSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        # wavelength can now be a 2D or 3D array (batch, x, wavelength)
        # We pass the underlying numpy arrays from the interp1d object
        return n_interp_log(
            wavelength, 
            self.lookupFunction.x, 
            self.lookupFunction.y, 
            self.lookupOffset
        )


class MolecularConstituent:
    """Represents a molecular absorbing species.

    This class handles the retrieval of pre-computed molecular cross-sections
    from HDF5 files, which are functions of pressure, temperature, and wavelength.

    Attributes:
        isMolecule (bool): Flag indicating this is a molecule.
        moleculeName (str): Name of the molecule.
        chi (float): Mixing ratio of the molecule.
        lookupFunction (Callable): A multi-dimensional interpolation function for
            the absorption cross-section.
    """

    def __init__(self, moleculeName: str, chi: float):
        """Initializes the MolecularConstituent.

        Args:
            moleculeName (str): The name of the molecule, corresponding to the
                HDF5 filename (without extension).
            chi (float): The mixing ratio of this molecule.
        """
        self.isMolecule: bool = True
        self.lookupOffset: float = 1e-50
        self.moleculeName: str = moleculeName
        self.chi: float = chi

    def constructLookupFunction(self, wavelengthGrid=None) -> Callable[[np.ndarray], np.ndarray]:
        """Creates an interpolation function from an HDF5 cross-section file.

        Reads pressure, temperature, wavelength, and cross-section data from
        an HDF5 file.  The raw grid arrays are stored on the object for the
        fast decomposed (bilinear P-T + 1-D wavelength) interpolation path
        used by getLOSopticalDepth_Batch.  A RegularGridInterpolator is also
        kept for backward compatibility with getSigmaAbs.

        If wavelengthGrid is supplied the stored wav_grid and sigma_grid_log
        are sliced to the simulation wavelength range plus a 1 % Doppler
        margin.  This reduces the per-chord bilinear kernel from O(76k) to
        O(N_sim) wavelength points — a large speedup for narrow grids.

        Returns:
            Callable[[np.ndarray], np.ndarray]: The interpolation function.
        """
        _DOPPLER_MARGIN = 0.01  # 1 % covers ~3000 km/s, well beyond any orbital speed

        with h5py.File(molecularLookupPath + self.moleculeName + '.h5', 'r+') as f:
            P = f['p'][:] * 10.
            T = f['t'][:]
            wav_full = 1. / f['bin_edges'][:][::-1]

            if wavelengthGrid is not None:
                wav_min = wavelengthGrid.lower_w * (1.0 - _DOPPLER_MARGIN)
                wav_max = wavelengthGrid.upper_w * (1.0 + _DOPPLER_MARGIN)
                mask = (wav_full >= wav_min) & (wav_full <= wav_max)
                if mask.sum() >= 2:
                    # Contiguous index range so h5py reads only the needed slab
                    lo, hi = int(np.argmax(mask)), int(len(mask) - np.argmax(mask[::-1]))
                    n_full = len(wav_full)
                    # sigma_abs is stored reversed relative to bin_edges, so the
                    # wavelength slice [lo:hi] maps to original indices [n-hi:n-lo]
                    sigma_abs = f['xsecarr'][:, :, n_full - hi:n_full - lo][:, :, ::-1]
                    wavelength = wav_full[lo:hi]
                else:
                    sigma_abs = f['xsecarr'][:][:, :, ::-1]
                    wavelength = wav_full
            else:
                sigma_abs = f['xsecarr'][:][:, :, ::-1]
                wavelength = wav_full

        sigma_log = np.log10(sigma_abs + self.lookupOffset)

        # Store raw grids for the decomposed interpolation path
        self.P_grid = np.ascontiguousarray(P)
        self.T_grid = np.ascontiguousarray(T)
        self.wav_grid = np.ascontiguousarray(wavelength)
        self.sigma_grid_log = np.ascontiguousarray(sigma_log)

        # Keep a RegularGridInterpolator for the legacy getSigmaAbs path
        lookupFunction = RegularGridInterpolator(
            (P, T, wavelength), sigma_log,
            bounds_error=False,
            fill_value=np.log10(self.lookupOffset))
        return lookupFunction

    def addLookupFunctionToConstituent(self, wavelengthGrid=None) -> None:
        """Constructs and attaches the lookup function to the object instance.

        Args:
            wavelengthGrid: Optional WavelengthGrid.  When provided, the stored
                cross-section table is sliced to the simulation wavelength range
                so the per-chord Numba kernel operates on a much smaller array.
        """
        lookupFunction = self.constructLookupFunction(wavelengthGrid)
        self.lookupFunction: Callable[[
            np.ndarray], np.ndarray] = lookupFunction

    def getSigmaAbs(self, P: np.ndarray, T: float, wavelength: np.ndarray) -> np.ndarray:
        # wavelength is (n_chords, n_wav)
        # P is (n_chords, n_x)
        n_chords, n_wav = wavelength.shape
        n_x = P.shape[1]
        total_points = n_chords * n_x * n_wav

        # Use broadcast views (no memory copy) then flatten via indices
        # P needs shape (n_chords, n_x, n_wav) -> flatten
        # wav needs shape (n_chords, n_x, n_wav) -> flatten
        # Instead of np.repeat (which copies), use np.broadcast_to (zero-copy views)
        P_view = np.broadcast_to(P[:, :, np.newaxis], (n_chords, n_x, n_wav))
        wav_view = np.broadcast_to(wavelength[:, np.newaxis, :], (n_chords, n_x, n_wav))

        # Reshape views to flat arrays — .reshape on broadcast arrays triggers a copy,
        # but only one copy each instead of repeat + flatten (two copies)
        PFlattened = np.clip(P_view.reshape(total_points), a_min=1e-4, a_max=None)
        wavelengthFlattened = wav_view.reshape(total_points)
        TFlattened = np.full(total_points, T)

        inputArray = np.column_stack([PFlattened, TFlattened, wavelengthFlattened])

        # Free intermediates before the lookup allocates more memory
        del PFlattened, TFlattened, wavelengthFlattened

        sigma_absFlattened = 10**self.lookupFunction(inputArray) - self.lookupOffset
        del inputArray

        # Return as 3D array: (n_chords, n_x, n_wav)
        return sigma_absFlattened.reshape(n_chords, n_x, n_wav)


# Reserved species keys identifying continuum scattering/aerosol constituents.
# These are matched in prometheus.py and setup.py to distinguish scattering
# sources from atomic/ionic species and molecular HDF5 lookup tables.
SCATTERER_TYPES: Tuple[str, ...] = ('RayleighHaze', 'GrayCloud', 'PowerLawAerosol', 'TabulatedAerosol')


class ScatteringConstituent:
    """Base class for continuum scattering / aerosol opacity sources.

    Scattering constituents contribute a wavelength-only extinction
    cross-section that acts per host-gas particle. The extincting number
    density is the host scenario's gas number density scaled by the abundance
    ``chi`` (a particle-to-gas ratio). In transit transmission, photons
    scattered out of the line of sight are lost from the beam, so the
    extinction cross-section is added directly to the Beer-Lambert optical
    depth — no scattering phase function or multiple scattering is modelled.

    Attributes:
        isMolecule (bool): Always False (kept for branch compatibility with the
            existing atomic/molecular optical-depth code paths).
        isScatterer (bool): Flag identifying this as a scattering source.
        chi (float): Abundance of the scattering particles relative to the host
            gas number density (particle-to-gas ratio).
        P_top (Optional[float]): If set, the opacity only applies where the
            local gas pressure is >= P_top (cgs, barye), confining the cloud
            below a cloud-top pressure. Requires a temperature-bearing host
            scenario (e.g. barometric/hydrostatic); ignored otherwise.
    """

    def __init__(self, chi: float, P_top: Union[float, None] = None):
        """Initializes the ScatteringConstituent.

        Args:
            chi (float): Particle-to-gas abundance ratio.
            P_top (Optional[float]): Cloud-top pressure in cgs (barye). The
                opacity is confined to where the local pressure exceeds this
                value. Defaults to None (no confinement).
        """
        self.isMolecule: bool = False
        self.isScatterer: bool = True
        self.chi: float = chi
        self.P_top: Union[float, None] = P_top
        self._sigma_cache: dict = {}

    def calculateSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        """Returns the extinction cross-section [cm^2] on a wavelength grid.

        Must be implemented by subclasses.

        Args:
            wavelength (np.ndarray): Wavelength grid [cm].

        Returns:
            np.ndarray: Extinction cross-section per particle [cm^2].
        """
        raise NotImplementedError

    def addLookupFunctionToConstituent(self, wavelengthGrid: 'WavelengthGrid' = None) -> None:
        """No-op kept for interface symmetry with atomic/molecular constituents.

        Scattering cross-sections are smooth and analytic, so no precomputed
        lookup table is required; `getSigmaAbs` evaluates them directly (with
        light caching).
        """
        return None

    def getSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        """Returns the (cached) extinction cross-section on a wavelength grid.

        Args:
            wavelength (np.ndarray): Wavelength grid [cm].

        Returns:
            np.ndarray: Extinction cross-section per particle [cm^2], same
                shape as `wavelength`.
        """
        key = wavelength.shape, wavelength.tobytes()
        sigma = self._sigma_cache.get(key)
        if sigma is None:
            sigma = self.calculateSigmaAbs(wavelength)
            self._sigma_cache[key] = sigma
        return sigma


class RayleighHaze(ScatteringConstituent):
    """Parametrized Rayleigh-scattering haze.

    The extinction cross-section follows a power law in wavelength:
    ``sigma(lambda) = sigma_ref * (lambda_ref / lambda) ** slope``. A slope of
    4 corresponds to pure Rayleigh scattering; smaller values describe
    flatter, more aerosol-like hazes.

    Attributes:
        sigma_ref (float): Reference cross-section at `lambda_ref` [cm^2].
        lambda_ref (float): Reference wavelength [cm].
        slope (float): Power-law exponent (4 for pure Rayleigh).
    """

    def __init__(self, chi: float = 1.0, sigma_ref: float = 5.31e-27,
                 lambda_ref: float = 4000e-8, slope: float = 4.0,
                 P_top: Union[float, None] = None):
        """Initializes the RayleighHaze.

        Args:
            chi (float): Particle-to-gas abundance ratio. Defaults to 1.0.
            sigma_ref (float): Reference cross-section at `lambda_ref` [cm^2].
                Defaults to 5.31e-27 (~H2 Rayleigh at 4000 A).
            lambda_ref (float): Reference wavelength [cm]. Defaults to 4000 A.
            slope (float): Power-law exponent. Defaults to 4.0.
            P_top (Optional[float]): Cloud-top pressure [barye]. Defaults to None.
        """
        super().__init__(chi, P_top)
        self.sigma_ref: float = sigma_ref
        self.lambda_ref: float = lambda_ref
        self.slope: float = slope

    def calculateSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        """Computes the Rayleigh power-law cross-section.

        Args:
            wavelength (np.ndarray): Wavelength grid [cm].

        Returns:
            np.ndarray: Extinction cross-section [cm^2].
        """
        return self.sigma_ref * (self.lambda_ref / wavelength) ** self.slope


class GrayCloud(ScatteringConstituent):
    """Gray (wavelength-independent) cloud opacity.

    Provides a flat extinction cross-section per host-gas particle. Combined
    with the optional `P_top` confinement of the base class, this can represent
    either a gray haze (no confinement) or an opaque cloud deck below a
    cloud-top pressure.

    Attributes:
        sigma_gray (float): Wavelength-independent cross-section [cm^2].
    """

    def __init__(self, chi: float = 1.0, sigma_gray: float = 1e-10,
                 P_top: Union[float, None] = None):
        """Initializes the GrayCloud.

        Args:
            chi (float): Particle-to-gas abundance ratio. Defaults to 1.0.
            sigma_gray (float): Wavelength-independent cross-section [cm^2].
                Defaults to 1e-10.
            P_top (Optional[float]): Cloud-top pressure [barye]. Defaults to None.
        """
        super().__init__(chi, P_top)
        self.sigma_gray: float = sigma_gray

    def calculateSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        """Computes the gray (constant) cross-section.

        Args:
            wavelength (np.ndarray): Wavelength grid [cm].

        Returns:
            np.ndarray: Extinction cross-section [cm^2], constant across `wavelength`.
        """
        return np.full_like(wavelength, self.sigma_gray, dtype=np.float64)


class PowerLawAerosol(ScatteringConstituent):
    """Aerosol with a user-specified Ångström extinction exponent (alpha).

    The extinction cross-section follows:
        σ(λ) = σ_ref × (λ / λ_ref)^(−alpha)

    This is the standard Ångström aerosol parameterization.  alpha = 4
    recovers pure Rayleigh scattering; typical tropospheric aerosols have
    alpha ≈ 1–2; values < 1 approach the gray (wavelength-independent) limit.

    Scattering is treated as extinction out of the beam — no phase function or
    multiple scattering is modelled.

    Unlike :class:`RayleighHaze` (which uses the alternative notation
    σ = σ_ref × (λ_ref/λ)^slope), this class exposes the exponent as ``alpha``
    with the conventional sign, making it unambiguous in publications.

    Attributes:
        sigma_ref (float): Reference cross-section at lambda_ref [cm²].
        lambda_ref (float): Reference wavelength [cm].
        alpha (float): Ångström exponent.
    """

    def __init__(self, chi: float = 1.0, sigma_ref: float = 1e-25,
                 lambda_ref: float = 5500e-8, alpha: float = 2.0,
                 P_top: Union[float, None] = None):
        super().__init__(chi, P_top)
        self.sigma_ref: float = sigma_ref
        self.lambda_ref: float = lambda_ref
        self.alpha: float = alpha

    def calculateSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        return self.sigma_ref * (wavelength / self.lambda_ref) ** (-self.alpha)


class TabulatedAerosol(ScatteringConstituent):
    """Aerosol with cross-sections loaded from a two-column CSV file.

    The CSV must have columns: wavelength [Angstrom], sigma [cm²] (no header,
    or comment lines starting with ``#``).  Values are linearly interpolated
    within the tabulated range.

    Outside the table the extinction is, by default, held at the nearest edge
    value (``extrapolate='edge'``).  A measured aerosol cross-section does not
    physically vanish just because the table ends, so dropping straight to
    σ = 0 would introduce a spurious opacity cliff at the table boundaries.
    Pass ``extrapolate='zero'`` to recover the previous hard-cutoff behaviour.

    Scattering is treated as extinction out of the beam — no phase function or
    multiple scattering is modelled.

    Attributes:
        filepath (str): Path to the CSV cross-section table.
        extrapolate (str): Out-of-range behaviour, ``'edge'`` or ``'zero'``.
    """

    def __init__(self, chi: float = 1.0, filepath: str = '',
                 extrapolate: str = 'edge',
                 P_top: Union[float, None] = None):
        super().__init__(chi, P_top)
        self.filepath: str = filepath
        if extrapolate not in ('edge', 'zero'):
            raise ValueError(
                f"extrapolate must be 'edge' or 'zero', got {extrapolate!r}")
        self.extrapolate: str = extrapolate
        data = np.loadtxt(filepath, delimiter=',', comments='#')
        wav_A, sigma_raw = data[:, 0], data[:, 1]
        idx = np.argsort(wav_A)
        self._wav_cm: np.ndarray = wav_A[idx] * 1e-8
        self._sigma_table: np.ndarray = sigma_raw[idx]

    def calculateSigmaAbs(self, wavelength: np.ndarray) -> np.ndarray:
        if self.extrapolate == 'zero':
            return np.interp(wavelength, self._wav_cm, self._sigma_table,
                             left=0.0, right=0.0)
        # 'edge': np.interp already holds the nearest edge value when left/right
        # are not supplied.
        return np.interp(wavelength, self._wav_cm, self._sigma_table)


def makeScatteringConstituent(scattererType: str, paramsDict: dict) -> ScatteringConstituent:
    """Factory for scattering constituents from a setup-file parameter dict.

    Args:
        scattererType (str): One of `SCATTERER_TYPES` ('RayleighHaze', 'GrayCloud').
        paramsDict (dict): Parameter dictionary from the setup file.

    Returns:
        ScatteringConstituent: The constructed scattering constituent.

    Raises:
        ValueError: If `scattererType` is not recognized.
    """
    chi = paramsDict.get('chi', 1.0)
    P_top = paramsDict.get('P_top', None)
    if scattererType == 'RayleighHaze':
        return RayleighHaze(
            chi=chi,
            sigma_ref=paramsDict.get('sigma_ref', 5.31e-27),
            lambda_ref=paramsDict.get('lambda_ref', 4000e-8),
            slope=paramsDict.get('slope', 4.0),
            P_top=P_top,
        )
    elif scattererType == 'GrayCloud':
        return GrayCloud(
            chi=chi,
            sigma_gray=paramsDict.get('sigma_gray', 1e-10),
            P_top=P_top,
        )
    elif scattererType == 'PowerLawAerosol':
        return PowerLawAerosol(
            chi=chi,
            sigma_ref=paramsDict.get('sigma_ref', 1e-25),
            lambda_ref=paramsDict.get('lambda_ref', 5500e-8),
            alpha=paramsDict.get('alpha', 2.0),
            P_top=P_top,
        )
    elif scattererType == 'TabulatedAerosol':
        return TabulatedAerosol(
            chi=chi,
            filepath=paramsDict['filepath'],
            extrapolate=paramsDict.get('extrapolate', 'edge'),
            P_top=P_top,
        )
    raise ValueError(f"Unknown scattering constituent type: {scattererType}")


class Atmosphere:
    """Manages all atmospheric/exospheric density distributions for a simulation.

    This class aggregates multiple density models (e.g., a barometric atmosphere
    plus a moon exosphere) and calculates the total optical depth along a line of sight.

    Attributes:
        densityDistributionList (List[Any]): A list of density distribution
            model objects (e.g., `BarometricAtmosphere`, `PowerLawExosphere`).
        hasOrbitalDopplerShift (bool): Flag indicating whether to include
            Doppler shifts from orbital motion.
    """

    def __init__(self, densityDistributionList: List[Any], hasOrbitalDopplerShift: bool):
        """Initializes the Atmosphere object.

        Args:
            densityDistributionList (List[Any]): A list of density model objects.
            hasOrbitalDopplerShift (bool): Flag for including orbital Doppler shifts.
        """
        self.densityDistributionList: List[Any] = densityDistributionList
        self.hasOrbitalDopplerShift: bool = hasOrbitalDopplerShift

    @staticmethod
    def getAbsorberNumberDensity(densityDistribution: Any, chi: float, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the number density of a specific absorbing species.

        Args:
            densityDistribution (Any): The density model object.
            chi (float): The mixing ratio of the absorber.
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The number density of the absorber at each x coordinate [cm^-3].
        """
        n_total = densityDistribution.calculateNumberDensity(
            x, phi, rho, orbphase)
        n_abs = n_total * chi
        return n_abs

    def getAbsorberVelocityField(self, densityDistribution: Any, x: np.ndarray, phi: float, rho: float, orbphase: float) -> np.ndarray:
        """Calculates the line-of-sight velocity of the gas.

        Args:
            densityDistribution (Any): The density model object.
            x (np.ndarray): Array of coordinates along the line of sight (cm).
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): Planet's orbital phase (radians).

        Returns:
            np.ndarray: The line-of-sight velocity at each x coordinate [cm/s].
        """
        v_los = np.zeros_like(x)
        if self.hasOrbitalDopplerShift:
            if not densityDistribution.hasMoon:
                v_los += densityDistribution.planet.getLOSvelocity(orbphase)
            else:
                v_los += densityDistribution.moon.getLOSvelocity(orbphase)
        return v_los

    def getLOSopticalDepth_Batch(self, x_grid, phi_batch, rho_batch, orbphase_batch, wavelength, delta_x):
        """Calculates optical depth for a batch of chords.

        Fully vectorized — no Python loops over individual chords.
        n_tot has shape (n_chords, n_x), shifted_wav has shape (n_chords, n_wav).

        Args:
            x_grid (np.ndarray): Line-of-sight grid, shape (n_x,).
            phi_batch (np.ndarray): Azimuthal angles, shape (n_chords,).
            rho_batch (np.ndarray): Projected radii, shape (n_chords,).
            orbphase_batch (np.ndarray): Orbital phases, shape (n_chords,).
            wavelength (np.ndarray): Wavelength grid, shape (n_wav,).
            delta_x (float): Step size along line of sight.

        Returns:
            np.ndarray: Optical depths, shape (n_chords, n_wav).
        """
        n_chords = len(phi_batch)
        n_wav = len(wavelength)
        total_tau = np.zeros((n_chords, n_wav))

        for dist_model in self.densityDistributionList:
            has_wind_velocity = hasattr(dist_model, 'calculateLOSVelocity')

            # --- bulk orbital velocity (one value per chord) ---
            if self.hasOrbitalDopplerShift:
                if not dist_model.hasMoon:
                    v_bulk = dist_model.planet.getLOSvelocity(orbphase_batch)  # (n_chords,)
                else:
                    v_bulk = dist_model.moon.getLOSvelocity(orbphase_batch)    # (n_chords,)
            else:
                v_bulk = np.zeros(n_chords)

            shifts = const.calculateDopplerShift(-v_bulk)                       # (n_chords,)
            shifted_wav = shifts[:, np.newaxis] * wavelength[np.newaxis, :]    # (n_chords, n_wav)

            # --- per-x velocity field for wind models ---
            # Compute only the (n_chords, n_x) shift factors, NOT the full
            # (n_chords, n_x, n_wav) tensor — the expansion is done lazily
            # inside the per-x loop (Optimization 3).
            if has_wind_velocity and self.hasOrbitalDopplerShift:
                v_wind = dist_model.calculateLOSVelocity(
                    x_grid, phi_batch, rho_batch, orbphase_batch
                )  # (n_chords, n_x)
                v_total_field = v_wind + v_bulk[:, np.newaxis]                  # (n_chords, n_x)
                shifts_field = const.calculateDopplerShift(-v_total_field)      # (n_chords, n_x)
            else:
                shifts_field = None

            # --- density: fully vectorized, returns (n_chords, n_x) ---
            n_tot = dist_model.calculateNumberDensity(
                x_grid, phi_batch, rho_batch, orbphase_batch
            )  # (n_chords, n_x)
            n_x_local = n_tot.shape[1]

            for constituent in dist_model.constituents:
                if constituent.isMolecule:
                    # --- Optimizations 2 + 4: decomposed P-T interpolation ---
                    # 1) bilinear-interpolate over (P, T) per x-step on native wav grid
                    # 2) accumulate the weighted column on the native grid
                    # 3) 1-D interpolate the result onto the Doppler-shifted grid
                    P = n_tot * const.k_B * dist_model.T                        # (n_chords, n_x)
                    P_clamped = np.clip(P, 1e-4, None)
                    n_abs = n_tot * constituent.chi                              # (n_chords, n_x)

                    n_wav_native = len(constituent.wav_grid)
                    sigma_eff = np.zeros((n_chords, n_wav_native))

                    for xi in range(n_x_local):
                        sigma_xi = _bilinear_PT_interp(
                            P_clamped[:, xi], dist_model.T,
                            constituent.P_grid, constituent.T_grid,
                            constituent.sigma_grid_log, constituent.lookupOffset
                        )                                                        # (n_chords, n_wav_native)
                        sigma_eff += n_abs[:, xi, np.newaxis] * sigma_xi

                    sigma_eff *= delta_x
                    total_tau += n_interp_linear_rows(
                        shifted_wav, constituent.wav_grid, sigma_eff)

                elif getattr(constituent, 'isScatterer', False):
                    # Continuum scattering: cross-section wavelength-only, no Doppler
                    n_abs = n_tot * constituent.chi                              # (n_chords, n_x)
                    if constituent.P_top is not None and hasattr(dist_model, 'T'):
                        P = n_tot * const.k_B * dist_model.T
                        n_abs = np.where(P >= constituent.P_top, n_abs, 0.0)
                    col_density = np.sum(n_abs, axis=1) * delta_x               # (n_chords,)
                    sigma = constituent.getSigmaAbs(wavelength)                  # (n_wav,)
                    total_tau += col_density[:, np.newaxis] * sigma[np.newaxis, :]

                else:  # atoms
                    if shifts_field is not None:
                        # --- Optimization 3: per-x loop for wind models ---
                        # Instead of materializing the full (C, X, W) shifted
                        # wavelength tensor, iterate over the x-axis and expand
                        # only a (C, W) slice at a time.
                        n_abs = n_tot * constituent.chi                          # (n_chords, n_x)
                        for xi in range(n_x_local):
                            wav_xi = shifts_field[:, xi, np.newaxis] * wavelength[np.newaxis, :]  # (n_chords, n_wav)
                            sigma_xi = constituent.getSigmaAbs(wav_xi)           # (n_chords, n_wav)
                            total_tau += n_abs[:, xi, np.newaxis] * sigma_xi * delta_x
                    else:
                        # Fast path: single bulk Doppler shift per chord
                        col_density = np.sum(n_tot * constituent.chi, axis=1) * delta_x
                        unique_shifts, inverse = np.unique(shifts, return_inverse=True)
                        cache_key = (
                            unique_shifts.shape,
                            unique_shifts.tobytes(),
                            wavelength.shape,
                            wavelength.tobytes(),
                        )
                        sigma_cache = getattr(constituent, "_batch_sigma_cache", {})
                        if cache_key not in sigma_cache:
                            unique_shifted_wav = unique_shifts[:, np.newaxis] * wavelength[np.newaxis, :]
                            sigma_cache[cache_key] = constituent.getSigmaAbs(unique_shifted_wav)
                            constituent._batch_sigma_cache = sigma_cache
                        sigma = sigma_cache[cache_key][inverse]
                        total_tau += col_density[:, np.newaxis] * sigma

        return total_tau


class WavelengthGrid:
    """Creates and manages the wavelength grid for the simulation.

    The grid can be non-uniform, with higher resolution around specified
    spectral lines and lower resolution elsewhere.

    Attributes:
        lower_w (float): The lower bound of the wavelength range [cm].
        upper_w (float): The upper bound of the wavelength range [cm].
        widthHighRes (float): The width of the high-resolution region around
            each spectral line [cm].
        resolutionLow (float): The step size for the low-resolution parts of the grid [cm].
        resolutionHigh (float): The step size for the high-resolution parts of the grid [cm].
    """

    def __init__(self, lower_w: float, upper_w: float, widthHighRes: float, resolutionLow: float, resolutionHigh: float):
        """Initializes the WavelengthGrid.

        Args:
            lower_w (float): Lower wavelength bound [cm].
            upper_w (float): Upper wavelength bound [cm].
            widthHighRes (float): High-resolution region width [cm].
            resolutionLow (float): Low-resolution step size [cm].
            resolutionHigh (float): High-resolution step size [cm].
        """
        self.lower_w: float = lower_w
        self.upper_w: float = upper_w
        self.widthHighRes: float = widthHighRes
        self.resolutionLow: float = resolutionLow
        self.resolutionHigh: float = resolutionHigh

    def arangeWavelengthGrid(self, linesList: List[float]) -> np.ndarray:
        """Constructs a non-uniform wavelength grid.

        Creates a grid with high resolution around the specified line centers
        and low resolution in between.

        Args:
            linesList (List[float]): A list of spectral line center wavelengths [cm].

        Returns:
            np.ndarray: The constructed wavelength grid [cm].
        """
        peaks = np.sort(np.unique(linesList))
        diff = np.concatenate(([np.inf], np.diff(peaks), [np.inf]))
        if len(peaks) == 0:
            print(
                'WARNING: No absorption lines from atoms/ions in the specified wavelength range!')
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        HighResBorders: Tuple[List[float], List[float]] = ([], [])
        for idx, peak in enumerate(peaks):
            if diff[idx] > self.widthHighRes:
                HighResBorders[0].append(peak - self.widthHighRes / 2.)
            if diff[idx + 1] > self.widthHighRes:
                HighResBorders[1].append(peak + self.widthHighRes / 2.)
        grid: List[np.ndarray] = []
        for idx in range(len(HighResBorders[0])):
            grid.append(
                np.arange(HighResBorders[0][idx], HighResBorders[1][idx], self.resolutionHigh))
            if idx == 0:
                if self.lower_w < HighResBorders[0][0]:
                    grid.append(
                        np.arange(self.lower_w, HighResBorders[0][0], self.resolutionLow))
                if len(HighResBorders[0]) == 1 and self.upper_w > HighResBorders[1][-1]:
                    grid.append(
                        np.arange(HighResBorders[1][0], self.upper_w, self.resolutionLow))
            elif idx == len(HighResBorders[0]) - 1:
                grid.append(np.arange(
                    HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
                if self.upper_w > HighResBorders[1][-1]:
                    grid.append(
                        np.arange(HighResBorders[1][-1], self.upper_w, self.resolutionLow))
            else:
                grid.append(np.arange(
                    HighResBorders[1][idx - 1], HighResBorders[0][idx], self.resolutionLow))
        wavelengthGrid = np.sort(np.concatenate(grid))
        return wavelengthGrid

    def constructWavelengthGridSingle(self, constituent: AtmosphericConstituent) -> np.ndarray:
        """Constructs a wavelength grid for a single atomic/ionic constituent.

        Args:
            constituent (AtmosphericConstituent): The constituent to get lines from.

        Returns:
            np.ndarray: The constructed wavelength grid [cm].
        """
        linesList = constituent.getLineParameters(
            np.array([self.lower_w, self.upper_w]))[0]
        return self.arangeWavelengthGrid(linesList)

    def constructWavelengthGrid(self, densityDistributionList: List[Any]) -> np.ndarray:
        """Constructs a wavelength grid for all atomic/ionic species in the atmosphere.

        Molecular opacities are continuous and do not influence the grid construction.

        Args:
            densityDistributionList (List[Any]): List of all density models.

        Returns:
            np.ndarray: The final wavelength grid for the simulation [cm].
        """
        linesList: List[float] = []
        for densityDistribution in densityDistributionList:
            for constituent in densityDistribution.constituents:
                if constituent.isMolecule or getattr(constituent, 'isScatterer', False):
                    continue
                lines_w = constituent.getLineParameters(
                    np.array([self.lower_w, self.upper_w]))[0]
                linesList.extend(lines_w)
        if len(linesList) == 0:
            return np.arange(self.lower_w, self.upper_w, self.resolutionLow)
        return self.arangeWavelengthGrid(linesList)


class Transit:
    """Main class to orchestrate a transit simulation.

    This class combines the atmospheric model, wavelength grid, and spatial grid
    to calculate the transit light curve or transmission spectrum.

    Attributes:
        atmosphere (Atmosphere): The atmosphere object containing all gas properties.
        wavelengthGrid (WavelengthGrid): The wavelength grid object.
        spatialGrid (Any): The `geometryHandler.Grid` object.
        planet (Any): The primary `Planet` object of the simulation.
        wavelength (np.ndarray): The wavelength array for the simulation.
    """

    def __init__(self, atmosphere: Atmosphere, wavelengthGrid: WavelengthGrid, spatialGrid: geom.Grid):
        """Initializes the Transit simulation object.

        Args:
            atmosphere (Atmosphere): The atmosphere object.
            wavelengthGrid (WavelengthGrid): The wavelength grid object.
            spatialGrid (Any): The `geometryHandler.Grid` object.
        """
        self.atmosphere: Atmosphere = atmosphere
        self.wavelengthGrid: WavelengthGrid = wavelengthGrid
        self.spatialGrid: geom.Grid = spatialGrid
        self.planet: Any = self.atmosphere.densityDistributionList[0].planet

    def addWavelength(self) -> None:
        """Constructs and stores the wavelength grid for the simulation."""
        wavelength = self.wavelengthGrid.constructWavelengthGrid(
            self.atmosphere.densityDistributionList)
        self.wavelength: np.ndarray = wavelength

    def checkBlock(self, phi: float, rho: float, orbphase: float) -> bool:
        """Checks if a line of sight is blocked by the opaque body of a planet or moon.

        Args:
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): The planet's orbital phase (radians).

        Returns:
            bool: True if the line of sight is blocked, False otherwise.
        """
        y, z = geom.Grid.getCartesianFromCylinder(phi, rho)
        y_p = self.planet.getPosition(orbphase)[1]
        blockingPlanet = (np.sqrt((y - y_p)**2 + z**2) < self.planet.R)
        if blockingPlanet:
            return True
        for densityDistribution in self.atmosphere.densityDistributionList:
            if densityDistribution.hasMoon:
                moon = densityDistribution.moon
                y_moon = moon.getPosition(orbphase)[1]
                blockingMoon = ((y - y_moon)**2 + z**2 < moon.R)
                if blockingMoon:
                    return True
        return False

    def evaluateChord(self, phi: float, rho: float, orbphase: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the transmitted flux along a single line of sight (chord).

        This involves getting the unattenuated stellar flux, calculating the
        optical depth through the atmosphere, and applying the Beer-Lambert law.

        Args:
            phi (float): Azimuthal angle on the sky plane (radians).
            rho (float): Projected radial distance from star's center (cm).
            orbphase (float): The planet's orbital phase (radians).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - F_in (np.ndarray): The attenuated flux received by the observer.
                - F_out (np.ndarray): The unattenuated flux from that point on the star.
        """
        if self.planet.hostStar.Fstar_function is not None:
            Fstar = self.planet.hostStar.getFstar(phi, rho, self.wavelength)
        else:
            Fstar = np.ones_like(self.wavelength)
        F_out = rho * Fstar
        if self.checkBlock(phi, rho, orbphase):
            F_in = np.zeros_like(self.wavelength)
            return F_in, F_out
        x = self.spatialGrid.constructXaxis()
        delta_x = self.spatialGrid.getDeltaX()
        tau = self.atmosphere.getLOSopticalDepth_Batch(
            x,
            np.array([phi]),
            np.array([rho]),
            np.array([orbphase]),
            self.wavelength,
            delta_x,
        )[0]
        F_in = rho * Fstar * np.exp(-tau)
        return F_in, F_out

    def sumOverChords(self, max_memory_gb: float = 2.0) -> np.ndarray:
        chordGrid = self.spatialGrid.getChordGrid()
        n_wav = len(self.wavelength)
        n_orb = self.spatialGrid.orbphase_steps
        
        F_in_sum = np.zeros((n_orb, n_wav))
        F_out_sum = np.zeros((n_orb, n_wav))
        
        phi = chordGrid[:, 0]
        rho = chordGrid[:, 1]
        orb = chordGrid[:, 2]
        
        y = rho * np.sin(phi)
        z = rho * np.cos(phi)
        
        orb_axis = self.spatialGrid.constructOrbphaseAxis()
        orb_indices = np.abs(orb[:, None] - orb_axis).argmin(axis=1)

        star = self.planet.hostStar
        mu_term = np.sqrt(np.clip(1. - rho**2 / star.R**2, 0, 1))
        clv = 1. - star.CLV_u1 * (1. - mu_term) - star.CLV_u2 * (1. - mu_term)**2
        
        v_star = star.vsiniStarrot * rho / star.R * np.cos(phi - star.phiStarrot)
        star_shifts = const.calculateDopplerShift(v_star)
        
        has_molecules = any(
            any(c.isMolecule for c in dist.constituents)
            for dist in self.atmosphere.densityDistributionList
        )

        # With Optimizations 2-4, both the molecular and per-x wind paths
        # avoid materialising the full (C, X, W) tensor.  Only molecules
        # still require a moderately larger per-chord allocation (for the
        # native-wavelength-grid intermediates).
        heavy_path = has_molecules

        x_grid = self.spatialGrid.constructXaxis()
        n_x = len(x_grid)

        from . import memoryHandler as mem
        batch_size = mem.calculate_optimal_chunk_size(
            len(chordGrid),
            n_wav,
            n_x,             # Pass n_x here
            max_memory_gb,
            is_molecular=heavy_path
        )
        
        x_grid = self.spatialGrid.constructXaxis()
        delta_x = self.spatialGrid.getDeltaX()

        for i in range(0, len(chordGrid), batch_size):
            idx = slice(i, i + batch_size)
            
            # --- MODIFIED SECTION ---
            # If no spectrum is loaded, assume a flat star (flux = 1.0)
            if star.Fstar_function is None:
                F_star_batch = np.ones((len(phi[idx]), n_wav))
            else:
                star_wav_shifted = self.wavelength[None, :] / star_shifts[idx, None]
                F_star_batch = n_interp_log(
                    star_wav_shifted, 
                    star.Fstar_function.x, 
                    star.Fstar_function.y, 
                    0.0
                )
            # ------------------------
            
            F_star_batch *= clv[idx, None]

            y_p = self.planet.a * np.sin(orb[idx])
            is_blocked = (np.sqrt((y[idx] - y_p)**2 + z[idx]**2) < self.planet.R)
            for densityDistribution in self.atmosphere.densityDistributionList:
                if densityDistribution.hasMoon:
                    moon = densityDistribution.moon
                    y_moon = moon.getPosition(orb[idx])[1]
                    is_blocked |= ((y[idx] - y_moon)**2 + z[idx]**2 < moon.R**2)
            
            F_out = rho[idx, None] * F_star_batch
            F_in = np.zeros_like(F_out)
            
            active = ~is_blocked
            if np.any(active):
                tau = self.atmosphere.getLOSopticalDepth_Batch(
                    x_grid, phi[idx][active], rho[idx][active], 
                    orb[idx][active], self.wavelength, delta_x
                )
                F_in[active] = F_out[active] * np.exp(-tau)

            if batch_size == len(chordGrid):
                F_in_sum = F_in.reshape(
                    self.spatialGrid.phi_steps * self.spatialGrid.rho_steps,
                    n_orb,
                    n_wav,
                ).sum(axis=0)
                F_out_sum = F_out.reshape(
                    self.spatialGrid.phi_steps * self.spatialGrid.rho_steps,
                    n_orb,
                    n_wav,
                ).sum(axis=0)
            else:
                np.add.at(F_in_sum, orb_indices[idx], F_in)
                np.add.at(F_out_sum, orb_indices[idx], F_out)

        return F_in_sum / F_out_sum
