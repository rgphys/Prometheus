"""
This file defines the `Grid` class, which handles the creation and management
of the spatial and temporal grids used for the transit simulation. This includes
the line-of-sight integration axis (x), the sky-plane coordinates (rho, phi),
and the orbital phase axis.

Created on 18. October 2021 by Andrea Gebek.
"""

from typing import Tuple

import numpy as np


class Grid:
    """Manages the spatial and temporal grids for the simulation.

    This class stores the discretization parameters and provides methods to
    construct the coordinate axes and a flattened grid of all points to be
    evaluated in the simulation.

    The coordinate system is defined as follows:
    - The observer is at x = -infinity.
    - The star is at the origin (0,0,0).
    - The x-axis is the line of sight through the star's center.
    - The y-z plane is the plane of the sky.
    - `rho` is the radial distance from the origin in the y-z plane.
    - `phi` is the azimuthal angle in the y-z plane.

    Attributes:
        x_midpoint (float): The center of the integration chord along the x-axis [cm].
        x_border (float): The half-length of the integration chord [cm].
        x_steps (int): The number of steps along the x-axis.
        rho_border (float): The maximum radius on the sky plane to consider (e.g., stellar radius) [cm].
        rho_steps (int): The number of steps along the rho-axis.
        phi_steps (int): The number of steps along the phi-axis.
        orbphase_border (float): The maximum absolute orbital phase to simulate [radians].
        orbphase_steps (int): The number of orbital phase steps.
    """

    def __init__(
        self,
        x_midpoint: float,
        x_border: float,
        x_steps: int,
        rho_border: float,
        rho_steps: int,
        phi_steps: int,
        orbphase_border: float,
        orbphase_steps: int
    ) -> None:
        """Initializes the Grid object with all discretization parameters.

        Args:
            x_midpoint (float): The center of the integration chord along the x-axis [cm].
            x_border (float): The half-length of the integration chord [cm].
            x_steps (int): The number of steps along the x-axis.
            rho_border (float): The maximum radius on the sky plane [cm].
            rho_steps (int): The number of steps along the rho-axis.
            phi_steps (int): The number of steps along the phi-axis.
            orbphase_border (float): The max orbital phase to simulate [radians].
            orbphase_steps (int): The number of orbital phase steps.
        """
        self.x_midpoint = x_midpoint
        self.x_border = x_border
        self.x_steps = x_steps
        self.rho_border = rho_border
        self.rho_steps = rho_steps
        self.phi_steps = phi_steps
        self.orbphase_border = orbphase_border
        self.orbphase_steps = orbphase_steps

    @staticmethod
    def getCartesianFromCylinder(phi: float, rho: float) -> Tuple[float, float]:
        """Converts cylindrical coordinates on the sky plane to Cartesian (y, z).

        Args:
            phi (float): Azimuthal angle in radians.
            rho (float): Radial distance in cm.

        Returns:
            Tuple[float, float]: The y and z coordinates in cm.
        """
        y = rho * np.sin(phi)
        z = rho * np.cos(phi)
        return y, z

    def getDeltaX(self) -> float:
        """Calculates the step size along the line-of-sight (x) axis.

        Returns:
            float: The step size delta_x in cm.
        """
        return 2. * self.x_border / float(self.x_steps)

    def getDeltaRho(self) -> float:
        """Calculates the step size in the radial (rho) direction.

        Returns:
            float: The step size delta_rho in cm.
        """
        return self.rho_border / float(self.rho_steps)

    def getDeltaPhi(self) -> float:
        """Calculates the step size in the angular (phi) direction.

        Returns:
            float: The step size delta_phi in radians.
        """
        return 2. * np.pi / float(self.phi_steps)

    def constructXaxis(self, midpoints: bool = True) -> np.ndarray:
        """Constructs the array of points along the line-of-sight (x) axis.

        Args:
            midpoints (bool, optional): If True, returns cell midpoints. If False,
                returns cell edges. Defaults to True.

        Returns:
            np.ndarray: The array of x-coordinates in cm.
        """
        if midpoints:  # Gas cell midpoints
            x_axis = np.linspace(
                self.x_midpoint - self.x_border,
                self.x_midpoint + self.x_border,
                int(self.x_steps),
                endpoint=False
            ) + self.x_border / float(self.x_steps)
        else:  # Return an array with x_steps + 1 entries, marking the cell edges
            x_axis = np.linspace(
                self.x_midpoint - self.x_border,
                self.x_midpoint + self.x_border,
                int(self.x_steps) + 1
            )
        return x_axis

    def constructRhoAxis(self, midpoints: bool = True) -> np.ndarray:
        """Constructs the array of points along the radial (rho) axis.

        Args:
            midpoints (bool, optional): If True, returns cell midpoints. If False,
                returns cell edges. Defaults to True.

        Returns:
            np.ndarray: The array of rho-coordinates in cm.
        """
        if midpoints:
            rho_axis = np.linspace(
                0., self.rho_border, int(self.rho_steps), endpoint=False
            ) + 0.5 * self.rho_border / float(self.rho_steps)
        else:
            rho_axis = np.linspace(
                0., self.rho_border, int(self.rho_steps) + 1
            )
        return rho_axis

    def constructPhiAxis(self, midpoints: bool = True) -> np.ndarray:
        """Constructs the array of points along the angular (phi) axis.

        Args:
            midpoints (bool, optional): If True, returns cell midpoints. If False,
                returns cell edges. Defaults to True.

        Returns:
            np.ndarray: The array of phi-coordinates in radians.
        """
        if midpoints:
            phi_axis = np.linspace(
                0, 2 * np.pi, int(self.phi_steps), endpoint=False
            ) + np.pi / float(self.phi_steps)
        else:
            phi_axis = np.linspace(
                0, 2 * np.pi, int(self.phi_steps) + 1
            )
        return phi_axis

    def constructOrbphaseAxis(self) -> np.ndarray:
        """Constructs the array of orbital phases to be simulated.

        Returns:
            np.ndarray: The array of orbital phases in radians, centered on 0.
        """
        orbphase_axis = np.linspace(
            -self.orbphase_border, self.orbphase_border, int(self.orbphase_steps)
        )
        return orbphase_axis

    def getChordGrid(self) -> np.ndarray:
        """Creates a flattened grid of all (phi, rho, orbphase) coordinates.

        This is useful for iterating through all lines of sight (chords) and
        orbital phases that need to be evaluated in the simulation.

        Returns:
            np.ndarray: A 2D array where each row is a unique combination of
            (phi, rho, orbphase). The shape is (phi_steps * rho_steps *
            orbphase_steps, 3).
        """
        phi_axis = self.constructPhiAxis()
        rho_axis = self.constructRhoAxis()
        orbphase_axis = self.constructOrbphaseAxis()
        phiGrid, rhoGrid, orbphaseGrid = np.meshgrid(
            phi_axis, rho_axis, orbphase_axis, indexing='ij'
        )
        chordGrid = np.stack(
            (phiGrid.flatten(), rhoGrid.flatten(), orbphaseGrid.flatten()), axis=-1
        )
        return chordGrid


def spatial_grid(planet, x_border_Rp: float = 12.0, x_steps: int = 25,
                 rho_steps: int = 60, phi_steps: int = 30,
                 orbphase_window: float = 0.0, orbphase_steps: int = 1,
                 rho_border=None) -> 'Grid':
    """Builds a :class:`Grid` with defaults tuned for an extended exosphere.

    The line-of-sight chord is centred on the planet's orbital distance and the
    sky-plane integration radius defaults to the stellar radius (so the transit
    depth normalises by the full stellar disk).

    Args:
        planet: A planet object exposing ``R``, ``a`` and ``hostStar.R``.
        x_border_Rp (float): Half-length of the LOS chord, in planet radii.
        x_steps (int): Number of steps along the line-of-sight axis.
        rho_steps (int): Number of steps along the sky-plane radial axis.
        phi_steps (int): Number of steps along the azimuthal axis.
        orbphase_window (float): Half-window of orbital phase [rad] (0 -> a
            single phase at mid-transit; >0 -> a lightcurve).
        orbphase_steps (int): Number of orbital-phase samples.
        rho_border (Optional[float]): Sky-plane integration radius [cm]
            (default: the stellar radius; do not shrink it, as the depth
            normalisation is by the stellar disk).

    Returns:
        Grid: The constructed spatial/temporal grid.
    """
    rho_border = planet.hostStar.R if rho_border is None else rho_border
    return Grid(
        x_midpoint=planet.a, x_border=x_border_Rp * planet.R, x_steps=x_steps,
        rho_border=rho_border, rho_steps=rho_steps, phi_steps=phi_steps,
        orbphase_border=orbphase_window, orbphase_steps=orbphase_steps)


def orbphase_window_from_hours(planet, half_window_hours: float) -> float:
    """Converts a +/- time half-window [hours] to an orbital-phase half-window [rad].

    Args:
        planet: A planet object exposing ``orbitalPeriod`` [days].
        half_window_hours (float): Half-window around mid-transit [hours].

    Returns:
        float: The corresponding orbital-phase half-window [rad].
    """
    period_hours = planet.orbitalPeriod * 24.0
    return (half_window_hours / period_hours) * 2.0 * np.pi