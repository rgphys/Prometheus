"""
This file stores physical and astronomical constants in cgs units.
It also defines classes for managing atomic and ionic species used in
the simulations.

Created on 2. June 2021 by Andrea Gebek.
"""

from typing import List, Optional

import numpy as np

e = 4.803e-10   # Elementary charge
m_e = 9.109e-28
c = 2.998e10
G = 6.674*10**(-8)
k_B = 1.381*10**(-16)
amu = 1.661*10**(-24)
R_J = 7.1492e9        # Jupiter equatorial radius
M_J = 1.898e30      #Jupiter mass
M_E = 5.974e27      #Earth mass
R_sun = 6.96e10     # Solar radius
M_sun = 1.988e33    # Solar mass
R_Io = 1.822e8      # Io radius
euler_mascheroni = 0.57721
AU = 1.496e13   # Conversion of one astronomical unit into cm

# Na D doublet rest wavelengths [Angstrom], VACUUM values to match the vacuum
# wavelengths in Resources/LineList.txt (Na D2 = 5891.583 A). The air values
# (5889.95 / 5895.92) sit ~1.6 A off the computed lines, enough to push a narrow
# bandpass off the absorption feature entirely.
NA_D2_ANG = 5891.583   # Na D2 (3s -> 3p 2P_3/2), vacuum
NA_D1_ANG = 5897.558   # Na D1 (3s -> 3p 2P_1/2), vacuum

"""
Generally useful functions
"""
def calculateDopplerShift(v: float) -> float:
    """Calculates the relativistic Doppler shift factor.

    This factor multiplied by the rest wavelength gives the observed wavelength.

    Args:
        v (float): The line-of-sight velocity in cm/s. Positive for motion
            toward the observer (blueshift): the returned factor is < 1, so
            observed = factor * rest is shorter than the rest wavelength.

    Returns:
        float: The dimensionless Doppler shift factor.
    """
    beta = v / c
    shift = np.sqrt((1. - beta) / (1. + beta))
    return shift


"""
Available atoms/ions with their atomic masses
"""

class Species:
    """Represents a single atomic or ionic species.

    Attributes:
        name (str): The common name of the species (e.g., 'NaI', 'SiII').
        element (str): The chemical symbol of the element (e.g., 'Na', 'Si').
        ionizationState (str): The ionization state as a string (e.g., '1' for
            neutral, '2' for singly ionized).
        mass (float): The mass of the species in grams.
    """
    def __init__(self, name: str, element: str, ionizationState: str, mass: float) -> None:
        """Initializes a Species object.

        Args:
            name (str): The common name of the species.
            element (str): The chemical symbol of the element.
            ionizationState (str): The ionization state as a string.
            mass (float): The mass of the species in grams.
        """
        self.name: str = name
        self.element: str = element
        self.ionizationState: str = ionizationState
        self.mass: float = mass

class SpeciesCollection:
    """A container for a list of Species objects.

    Provides methods for finding, listing, and adding species.

    Attributes:
        speciesList (List[Species]): A list of Species objects.
    """
    def __init__(self, speciesList: Optional[List[Species]] = None) -> None:
        """Initializes the SpeciesCollection.

        Args:
            speciesList (Optional[List[Species]]): An optional initial list of
                Species objects. Defaults to an empty list.
        """
        if speciesList is None:
            self.speciesList: List[Species] = []
        else:
            self.speciesList: List[Species] = speciesList

    def findSpecies(self, nameSpecies: str) -> Optional[Species]:
        """Finds a species in the collection by its name.

        Args:
            nameSpecies (str): The name of the species to find.

        Returns:
            Optional[Species]: The Species object if found, otherwise None.
        """
        for species in self.speciesList:
            if species.name == nameSpecies:
                return species
        print('Species', nameSpecies, 'was not found.')
        return None

    def listSpeciesNames(self) -> List[str]:
        """Returns a list of names of all species in the collection.

        Returns:
            List[str]: A list of species names.
        """
        names: List[str] = []
        for species in self.speciesList:
            names.append(species.name)
        return names

    def addSpecies(self, species: Species) -> None:
        """Adds a Species object to the collection.

        Args:
            species (Species): The species to add.
        """
        self.speciesList.append(species)


class AvailableSpecies(SpeciesCollection):
    """A pre-populated collection of common astrophysical species.

    Inherits from SpeciesCollection and initializes with a default set of
    atoms and ions relevant for exoplanet atmosphere studies.
    """
    def __init__(self) -> None:
        """Initializes and populates the list of available species."""
        NaI = Species('NaI', 'Na', '1', 22.99 * amu)
        KI = Species('KI', 'K', '1', 39.0983 * amu)
        SiI = Species('SiI', 'Si', '1', 28.0855 * amu)
        SiII = Species('SiII', 'Si', '2', 28.0855 * amu)
        SiIII = Species('SiIII', 'Si', '3', 28.0855 * amu)
        SiIV = Species('SiIV', 'Si', '4', 28.0855 * amu)
        MgI = Species('MgI', 'Mg', '1', 24.305 * amu)
        MgII = Species('MgII', 'Mg', '2', 24.305 * amu)
        AlI = Species('AlI', 'Al', '1', 26.9815 * amu)
        CaI = Species('CaI', 'Ca', '1', 40.078 * amu)
        CaII = Species('CaII', 'Ca', '2', 40.078 * amu)
        TiI = Species('TiI', 'Ti', '1', 47.867 * amu)
        TiII = Species('TiII', 'Ti', '2', 47.867 * amu)
        CrI = Species('CrI', 'Cr', '1', 51.9961 * amu)
        MnI = Species('MnI', 'Mn', '1', 54.938 * amu)
        FeI = Species('FeI', 'Fe', '1', 55.845 * amu)
        CoI = Species('CoI', 'Co', '1', 58.933 * amu)
        NiI = Species('NiI', 'Ni', '1', 58.6934 * amu)
        OI = Species('OI', 'O', '1', 15.999 * amu)
        CII = Species('CII', 'C', '2', 12.011 * amu)
        SIII = Species('SIII', 'S', '3', 32.06 * amu)
        SIV = Species('SIV', 'S', '4', 32.06 * amu)

        self.speciesList: List[Species] = [NaI, KI, SiI, SiII, SiIII, SiIV, MgI, MgII, AlI, CaI, CaII, TiI, TiII, CrI, MnI, FeI, CoI, NiI, OI, CII, SIII, SIV]