"""
This script runs an interactive command-line Q&A session to gather all
necessary parameters for a PROMETHEUS simulation. The collected settings
are then saved into a JSON file, which can be used to run the main
simulation script.

Created on 2. June 2021 by Andrea Gebek.
"""

import numpy as np
import json
import sys
import pythonScripts.constants as const
import pythonScripts.celestialBodies as bodies


class NumericalQuestion:
    """A helper class to ask the user for a numerical input with validation.

    This class handles prompting the user, checking if the input is a valid
    number, and ensuring it falls within a specified range.

    Attributes:
        questionText (str): The question to ask the user.
        lowerBorder (float): The lower bound of the acceptable range.
        upperBorder (float): The upper bound of the acceptable range.
        unit (float): A conversion factor to apply to the user's input.
        roundBorders (bool): Whether to display borders as rounded integers.
        digits (int): Number of decimal places for displaying borders.
        acceptLowerBorder (bool): Whether the lower border is an inclusive value.
        acceptUpperBorder (bool): Whether the upper border is an inclusive value.
    """

    def __init__(self, questionText, lowerBorder, upperBorder, unit, roundBorders=True, digits=0, acceptLowerBorder=True, acceptUpperBorder=True):
        """Initializes the NumericalQuestion object.

        Args:
            questionText (str): The text of the question.
            lowerBorder (float): The minimum acceptable value.
            upperBorder (float): The maximum acceptable value.
            unit (float): A conversion factor to apply to the input value.
            roundBorders (bool): If True, display borders as scientific notation integers.
            digits (int): Number of digits for scientific notation display.
            acceptLowerBorder (bool): If True, the lower border is inclusive.
            acceptUpperBorder (bool): If True, the upper border is inclusive.
        """
        self.questionText = questionText
        self.lowerBorder = lowerBorder
        self.upperBorder = upperBorder
        self.unit = unit
        self.roundBorders = roundBorders
        self.digits = digits
        self.acceptLowerBorder = acceptLowerBorder
        self.acceptUpperBorder = acceptUpperBorder

    def readValue(self):
        """Prompts the user for a numerical value and validates it.

        The method repeatedly asks the question until a valid number within the
        defined range is entered.

        Returns:
            float: The validated numerical value, multiplied by the unit factor.
        """

        while True:

            if self.acceptLowerBorder:
                brackl = ' ['
            else:
                brackl = ' ('

            if self.acceptUpperBorder:
                brackr = '] '
            else:
                brackr = ') '

            if self.roundBorders:
                string = input(self.questionText + brackl + '{:.0e}'.format(
                    self.lowerBorder) + ', ' + '{:.0e}'.format(self.upperBorder) + brackr).replace(' ', '')

            elif self.digits > 0:
                string = input(self.questionText + brackl + '{:.{n}e}'.format(
                    self.lowerBorder, n=self.digits) + ', ' + '{:.{n}e}'.format(self.upperBorder, n=self.digits) + brackr).replace(' ', '')

            else:
                string = input(self.questionText + brackl + '{:e}'.format(
                    self.lowerBorder) + ', ' + '{:e}'.format(self.upperBorder) + brackr).replace(' ', '')

            if string == '':
                print('You actually do have to enter something!')
                continue

            value = float(string)

            if value > self.lowerBorder and value < self.upperBorder:
                return value * self.unit

            elif value == self.lowerBorder and self.acceptLowerBorder:
                return value * self.unit

            elif value == self.upperBorder and self.acceptUpperBorder:
                return value * self.unit

            else:
                print('The value you entered is not in the appropriate interval.')
                continue


class TextQuestion:
    """A helper class to ask the user for a text input with validation.

    This class can handle simple text input or multiple-choice questions
    where the answer must be one of the provided options.

    Attributes:
        questionText (str): The question to ask the user.
        options (list or None): A list of valid string options for the answer.
            If None, any non-empty string is accepted.
    """

    def __init__(self, questionText, boolQuestion=False, options=None):
        """Initializes the TextQuestion object.

        Args:
            questionText (str): The text of the question.
            boolQuestion (bool): If True, sets options to ['yes', 'no'].
            options (list, optional): A list of valid string answers.
        """
        self.questionText = questionText
        if boolQuestion:
            self.options = ['yes', 'no']
        else:
            self.options = options

    def readStr(self):
        """Prompts the user for a text value and validates it.

        The method repeatedly asks the question until a valid string is entered.
        If it is a boolean question, it returns True for 'yes' and False for 'no'.

        Returns:
            str or bool: The validated string answer, or a boolean for 'yes'/'no' questions.
        """

        while True:

            if self.options is None:
                string = input(self.questionText + ' ').replace(' ', '')

            else:
                string = input(self.questionText + ' ' +
                               str(self.options).replace("'", '') + ' ').replace(' ', '')

            if string == '':
                print('You actually do have to enter something!')
                continue

            elif self.options is not None and string not in self.options:
                print('Please select a valid option.')
                continue

            else:
                if string == 'yes':
                    return True
                elif string == 'no':
                    return False
                else:
                    return string


"""
Fundamentals
"""

QCLV = TextQuestion(
    'Do you want to take center-to-limb variations into account?', True)
QRM = TextQuestion('Do you want to take the Rossiter-McLaughlin-Effect into account (note that this means that you \
have to provide additional information about the host star to specifiy its spectrum)?', True)
QOrbitalMotion = TextQuestion(
    'Do you want to consider the Doppler shifts due to planetary/exomoon orbital motion?', True)


def setupFundamentals():
    """Conducts a Q&A session for fundamental simulation settings.

    Asks about including center-to-limb variation, the Rossiter-McLaughlin
    effect, and Doppler shifts from orbital motion.

    Returns:
        dict: A dictionary containing the fundamental settings.
    """

    print(r"""
    *******  *******           ,/MMM8&&&.         ****     **** ******** ********** **      ** ******** **     **  ********
    /**////**/**////**     _...MMMMM88&&&&..._    /**/**   **/**/**///// /////**/// /**     /**/**///// /**    /** **////// 
    /**   /**/**   /**   .:'''MMMMM88&&&&&&''':.  /**//** ** /**/**          /**    /**     /**/**      /**    /**/**       
    /******* /*******   :     MMMMM88&&&&&&     : /** //***  /**/*******     /**    /**********/******* /**    /**/*********
    /**////  /**///**    ':...MMMMM88&&&&&&....:  /**  //*   /**/**////      /**    /**//////**/**////  /**    /**////////**
    /**      /**  //**      `''MMMMM88&&&&'''`    /**   /    /**/**          /**    /**     /**/**      /**    /**       /**
    /**      /**   //**         'MMM8&&&'         /**        /**/********    /**    /**     /**/********//*******  ******** 
    //       //     //                            //         // ////////     //     //      // ////////  ///////  ////////  
    """)

    print('\nWelcome to PROMETHEUS! First, define if you want to make some fundamental simplifications.\n')

    fundamentalsDict = {'ExomoonSource': False, 'DopplerPlanetRotation': False}

    fundamentalsDict['CLV_variations'] = QCLV.readStr()
    fundamentalsDict['RM_effect'] = QRM.readStr()
    fundamentalsDict['DopplerOrbitalMotion'] = QOrbitalMotion.readStr()

    return fundamentalsDict


"""
Scenarios for the spatial distribution of the medium
"""

QScenario = TextQuestion('Enter the name of the scenario for the number density profile or \
0 to stop adding absorption sources:', False, ['barometric', 'hydrostatic', 'powerLaw', 'exomoon', 'torus', 'serpens', 'radialWind', '0'])
QTemperature = NumericalQuestion(
    'Enter the temperature of the atmosphere in Kelvin:', 1., 1e6, 1.)
QPressure = NumericalQuestion(
    'Enter the pressure at the reference radius in bar:', 1e-15, 1e6, 1e6)
Qmu = NumericalQuestion(
    'Enter the mean molecular weight of the atmosphere in atomic mass units:', 0.05, 500., const.amu)
QpowerLaw = NumericalQuestion(
    'Enter the power law index for the escaping atmosphere:', 3., 20., 1., acceptLowerBorder=False)
QpowerLawNormalization = TextQuestion('Do you want to normalize the number density profile via (total) pressure or via number \
of absorbing atoms at the base of the wind?', False, ['pressure', 'number'])
QpowerLawMoon = NumericalQuestion(
    'Enter the power law index for the exomoon exosphere:', 3., 20., 1.)
QtorusAxis = NumericalQuestion(
    'Enter the distance between the center of the torus and the center of the exoplanet in Jovian radii:', 0.01, 1000, const.R_J)
QtorusEjectionSpeed = NumericalQuestion(
    'Enter the ejection velocity (which sets the torus scale height) of the particles from the torus in km/s:', 1e-2, 1e3, 1e5)
QserpensPath = TextQuestion(
    'Enter the full path to the serpens input txt file, including the filename with the .txt ending:')
QwindMdot = NumericalQuestion(
    'Enter the mass loss rate of the radial wind in g/s:', 1e-10, 1e30, 1.)
QwindMu = NumericalQuestion(
    'Enter the mean particle mass of the wind in atomic mass units:', 0.05, 500., const.amu)
QwindModel = TextQuestion(
    "Choose the wind velocity model ('beta' = parametrized beta-law, \
'parker' = exact isothermal Parker wind):", False, ['beta', 'parker'])
QwindTemperature = NumericalQuestion(
    'Enter the (isothermal) wind temperature in Kelvin:', 1., 1e6, 1.)
QwindVterminal = NumericalQuestion(
    'Enter the terminal (reference) wind velocity in km/s:', 0.1, 1e5, 1e5)
QwindBeta = NumericalQuestion(
    'Enter the beta-law exponent for the wind velocity profile (1 = linear, 2 = quadratic):', 0.1, 10., 1.)
QwindRinner = NumericalQuestion(
    'Enter the inner wind boundary in planetary radii (0 = use planet radius):', 0., 100., 1.)
QwindRouter = TextQuestion('Do you want to set an outer cutoff radius for the wind?', True)


def setupScenario(fundamentalsDict):
    """Conducts a Q&A session for atmosphere/exosphere density models.

    Allows the user to add one or more density distribution models (e.g.,
    barometric, power-law) and specify their parameters.

    Args:
        fundamentalsDict (dict): The dictionary of fundamental settings, which
            may be updated if a moon scenario is chosen.

    Returns:
        dict: A dictionary of scenario models and their parameters.
    """

    print('\nNow, specifiy the spatial distribution of the absorbing medium, i.e. the structure of the atmosphere/exosphere.\n')

    scenarioDict = {}

    while True:

        scenario_name = QScenario.readStr()

        if scenario_name == '0':
            break

        params = {}

        if scenario_name == 'barometric' or scenario_name == 'hydrostatic':

            params['T'] = QTemperature.readValue()
            params['P_0'] = QPressure.readValue()
            params['mu'] = Qmu.readValue()

            if scenario_name == 'barometric':
                QScenario.options.remove('hydrostatic')
            else:
                QScenario.options.remove('barometric')

        elif scenario_name == 'powerLaw':

            params['q_esc'] = QpowerLaw.readValue()

            if QpowerLawNormalization.readStr() == 'pressure':

                params['P_0'] = QPressure.readValue()
                params['T'] = QTemperature.readValue()

        elif scenario_name == 'exomoon':

            params['q_moon'] = QpowerLawMoon.readValue()

            fundamentalsDict['ExomoonSource'] = True

        elif scenario_name == 'torus':

            params['a_torus'] = QtorusAxis.readValue()
            params['v_ej'] = QtorusEjectionSpeed.readValue()

        elif scenario_name == 'serpens':

            params['serpensPath'] = QserpensPath.readStr()

        elif scenario_name == 'radialWind':

            params['Mdot'] = QwindMdot.readValue()
            params['mu'] = QwindMu.readValue()
            params['wind_model'] = QwindModel.readStr()
            if params['wind_model'] == 'parker':
                # Parker wind shape is set by temperature + planet mass; no
                # beta/v_terminal knobs are needed.
                params['T'] = QwindTemperature.readValue()
            else:
                params['v_terminal'] = QwindVterminal.readValue()
                params['beta'] = QwindBeta.readValue()
            r_inner_Rp = QwindRinner.readValue()
            # Store in planet radii; converted to cm in prometheus.py using planet.R
            params['r_inner_Rp'] = r_inner_Rp  # 0 → use planet.R (handled in prometheus.py)
            if QwindRouter.readStr():
                params['r_outer'] = NumericalQuestion(
                    'Enter the outer wind cutoff radius in planetary radii:', 1., 1e6, 1.).readValue()

        scenarioDict[scenario_name] = params

        QScenario.options.remove(scenario_name)

    if len(scenarioDict) == 0:
        print('You have not added any absorption sources! Your loss. PROMETHEUS exits now.')
        sys.exit()

    return scenarioDict


"""
Architecture
"""

Qsystem = TextQuestion('Enter the name of the exoplanetary system',
                       False, bodies.AvailablePlanets().listPlanetNames())
QRstar = NumericalQuestion(
    'Enter the radius of the host star in solar radii:', 1e-5, 1e5, const.R_sun)
QMstar = NumericalQuestion(
    'Enter the mass of the host star in solar masses:', 1e-5, 1e10, const.M_sun)
QRplanet = NumericalQuestion(
    'Enter the radius of the exoplanet in Jupiter radii:', 1e-5, 1e5, const.R_J)
QMplanet = NumericalQuestion(
    'Enter the mass of the exoplanet in Jupiter masses:', 1e-5, 1e3, const.M_J)
QorbitalRadiusPlanet = NumericalQuestion(
    'Enter the orbital distance between planet and star in AU:', 1e-5, 1e3, const.AU)
QTstar = NumericalQuestion(
    'Enter the effective temperature of the star in Kelvin:', 2300., 12000., 1., roundBorders=False)
QloggStar = NumericalQuestion(
    'Enter the logarithmic value of the surface gravity in log10(cm/s^2):', 0., 6., 1., roundBorders=False)
QmetallicityStar = NumericalQuestion(
    'Enter the metallicity of the star [Fe/H]:', -4., 1., 1., roundBorders=False)
QalphaStar = NumericalQuestion(
    'Enter the alpha-enhancement of the star [alpha/Fe]:', -0.2, 1.2, 1., roundBorders=False)
QRMvsini = NumericalQuestion(
    'Enter the maximum velocity of the stellar rotation (v*sin(i)) in km/s:', 1e-2, 1e4, 1e5)
QRMazimuth = NumericalQuestion('Enter the angle at which the maximum velocity towards the observer lies on the stellar disk \
in degrees (measured in the canonical prometheus coordinate system of the angular coordinate)', 0., 360., np.pi / 180., roundBorders=False)
QCLVu1 = NumericalQuestion(
    'Enter the first (linear) coefficient for limb darkening:', -1., 1., 1.)
QCLVu2 = NumericalQuestion(
    'Enter the second (quadratic) coefficient for limb darkening:', -1., 1., 1.)
QRmoon = NumericalQuestion(
    'Enter the radius of the moon in Io radii:', 1e-3, 1e3, const.R_Io)
QorbphaseMoon = NumericalQuestion('Enter the orbital phase of the moon when the planet is transiting. A moon orbital phase of 0 corresponds to the moon sitting \
between the planet and the observer, 0.25 means that the exomoon is located to the right of the planet when viewed from the observer.', -0.5, 0.5, 2. * np.pi)


def setupArchitecture(fundamentalsDict):
    """Conducts a Q&A session for the system architecture parameters.

    Asks for the planet name and, depending on fundamental settings, parameters
    for the RM effect, CLV, and any exomoons.

    Args:
        fundamentalsDict (dict): The dictionary of fundamental settings.

    Returns:
        dict: A dictionary of architectural parameters.
    """

    print('\nProvide parameters related to the architecture of the system.\n')

    architectureDict = {}

    planetName = Qsystem.readStr()

    architectureDict['planetName'] = planetName

    planet = bodies.AvailablePlanets().findPlanet(planetName)

    if fundamentalsDict['RM_effect']:

        architectureDict['vsini'] = QRMvsini.readValue()
        architectureDict['azimuth_starrot'] = QRMazimuth.readValue()

    if fundamentalsDict['CLV_variations']:

        architectureDict['u1'] = QCLVu1.readValue()
        architectureDict['u2'] = QCLVu2.readValue()

    if fundamentalsDict['ExomoonSource']:

        architectureDict['R_moon'] = QRmoon.readValue()
        architectureDict['a_moon'] = NumericalQuestion('Enter the orbital distance between the exomoon and the planet in planetary radii (measured from the centers of the bodies):',
                                                       1, planet.a / planet.R, planet.R, roundBorders=False).readValue()
        architectureDict['starting_orbphase_moon'] = QorbphaseMoon.readValue()

    return architectureDict


"""
Specify the absorption species.
"""

Qmolecule = TextQuestion('Enter the name of the molecular absorber:')
Qsigmav = NumericalQuestion(
    'Enter the pseudo-thermal velocity dispersion (sigma_v = sqrt(k_B * T / m)) in km/s:', 1e-3, 1e5, 1e5)
QMoleculesTemperature = NumericalQuestion(
    'Enter the pseudo-temperature for the molecular absorber in K:', 100., 3400., 1.)


def setupRayleighHaze(key_scenario):
    """Collects parameters for a parametrized Rayleigh-scattering haze.

    Args:
        key_scenario (str): The name of the host density scenario.

    Returns:
        tuple[str, dict]: The reserved species key ('RayleighHaze') and its
            parameter dictionary.
    """
    params = {}
    params['chi'] = NumericalQuestion('Enter the abundance (particle-to-gas number ratio) of the Rayleigh haze in the ' +
                                      key_scenario + ' scenario:', 0., 1e30, 1., acceptLowerBorder=False).readValue()
    params['sigma_ref'] = NumericalQuestion(
        'Enter the reference Rayleigh extinction cross-section (sigma_ref) in cm^2 '
        '(Lecavelier des Etangs et al. 2008 H2 baseline: 5.31e-27 at lambda_ref=3500 A):', 0., 1., 1., acceptLowerBorder=False).readValue()
    params['lambda_ref'] = NumericalQuestion(
        'Enter the reference wavelength (lambda_ref) for the Rayleigh cross-section in Angstrom '
        '(use 3500 to pair with the 5.31e-27 H2 baseline):', 500, 55000, 1e-8, roundBorders=False).readValue()
    params['slope'] = NumericalQuestion(
        'Enter the Rayleigh slope exponent (4 corresponds to pure Rayleigh scattering):', 0., 20., 1.).readValue()
    return 'RayleighHaze', params


def setupGrayCloud(key_scenario):
    """Collects parameters for a gray (wavelength-independent) cloud/aerosol.

    Args:
        key_scenario (str): The name of the host density scenario.

    Returns:
        tuple[str, dict]: The reserved species key ('GrayCloud') and its
            parameter dictionary.
    """
    params = {}
    params['chi'] = NumericalQuestion('Enter the abundance (particle-to-gas number ratio) of the gray cloud in the ' +
                                      key_scenario + ' scenario:', 0., 1e30, 1., acceptLowerBorder=False).readValue()
    params['sigma_gray'] = NumericalQuestion(
        'Enter the gray (wavelength-independent) extinction cross-section in cm^2:', 0., 1., 1., acceptLowerBorder=False).readValue()
    if TextQuestion('Do you want to confine the cloud below a cloud-top pressure (opaque deck)?', True).readStr():
        params['P_top'] = NumericalQuestion(
            'Enter the cloud-top pressure in cgs units (barye):', 0., 1e12, 1., acceptLowerBorder=False).readValue()
    return 'GrayCloud', params


def setupPowerLawAerosol(key_scenario):
    """Collects parameters for a power-law aerosol with user-specified Ångström exponent.

    Args:
        key_scenario (str): The name of the host density scenario.

    Returns:
        tuple[str, dict]: The reserved key ('PowerLawAerosol') and its parameter dict.
    """
    params = {}
    params['chi'] = NumericalQuestion('Enter the abundance (particle-to-gas number ratio) of the aerosol in the ' +
                                      key_scenario + ' scenario:', 0., 1e30, 1., acceptLowerBorder=False).readValue()
    params['sigma_ref'] = NumericalQuestion(
        'Enter the reference extinction cross-section (sigma_ref) in cm^2:', 0., 1., 1., acceptLowerBorder=False).readValue()
    params['lambda_ref'] = NumericalQuestion(
        'Enter the reference wavelength (lambda_ref) in Angstrom:', 500, 55000, 1e-8, roundBorders=False).readValue()
    params['alpha'] = NumericalQuestion(
        'Enter the Angstrom exponent alpha (4 = Rayleigh, ~2 = typical aerosol, ~0 = gray):', 0., 20., 1.).readValue()
    if TextQuestion('Do you want to confine the aerosol below a cloud-top pressure?', True).readStr():
        params['P_top'] = NumericalQuestion(
            'Enter the cloud-top pressure in cgs units (barye):', 0., 1e12, 1., acceptLowerBorder=False).readValue()
    return 'PowerLawAerosol', params


def setupTabulatedAerosol(key_scenario):
    """Collects parameters for a tabulated aerosol loaded from a CSV file.

    The CSV must have columns: wavelength [Angstrom], sigma [cm^2].

    Args:
        key_scenario (str): The name of the host density scenario.

    Returns:
        tuple[str, dict]: The reserved key ('TabulatedAerosol') and its parameter dict.
    """
    params = {}
    params['chi'] = NumericalQuestion('Enter the abundance (particle-to-gas number ratio) of the tabulated aerosol in the ' +
                                      key_scenario + ' scenario:', 0., 1e30, 1., acceptLowerBorder=False).readValue()
    params['filepath'] = TextQuestion(
        'Enter the full path to the aerosol cross-section CSV file (columns: wavelength[A], sigma[cm^2]):').readStr()
    if TextQuestion('Do you want to confine the aerosol below a cloud-top pressure?', True).readStr():
        params['P_top'] = NumericalQuestion(
            'Enter the cloud-top pressure in cgs units (barye):', 0., 1e12, 1., acceptLowerBorder=False).readValue()
    return 'TabulatedAerosol', params


def setupSpecies(scenarioDict):
    """Conducts a Q&A session for the absorbing species.

    For each defined density scenario, this function asks the user to add
    one or more atomic, ionic, or molecular absorbers and specify their
    abundances or other properties.

    Args:
        scenarioDict (dict): The dictionary of defined density scenarios.

    Returns:
        dict: A nested dictionary containing the species and their parameters
              for each scenario.
    """

    print('\nSpecify the absorbing species and their abundances.\n')

    speciesDict = {}

    for key_scenario in scenarioDict.keys():

        hasTemperature = 'T' in scenarioDict[key_scenario].keys()
        isWindScenario = (key_scenario == 'radialWind')
        # Continuum scattering sources are offered for collisional scenarios where
        # a cloud-top pressure can be defined.  PowerLawAerosol and TabulatedAerosol
        # are also available for collisional scenarios.
        if hasTemperature:
            absorberOptions = ['atom', 'molecule', 'rayleigh', 'gray', 'powerlawaerosol', 'tabulatedaerosol']
        else:
            absorberOptions = ['atom', 'molecule']
        QabsorberType = TextQuestion('Which absorber type do you want to add in the ' + key_scenario +
                                     ' scenario (atom includes ions; rayleigh/gray/powerlawaerosol/tabulatedaerosol are continuum scattering sources)?',
                                     False, absorberOptions)
        PossibleAbsorbers = const.AvailableSpecies().listSpeciesNames()

        speciesDict[key_scenario] = {}

        # Scenarios that incorporate a temperature
        if hasTemperature:

            while True:

                absorberType = QabsorberType.readStr()
                params = {}

                if absorberType == 'atom':

                    key_species = TextQuestion('Enter the name of the absorbing species you want to consider for the ' +
                                               key_scenario + ' scenario:', False, PossibleAbsorbers).readStr()
                    PossibleAbsorbers.remove(key_species)
                    params['chi'] = NumericalQuestion('Enter the mixing ratio of ' + key_species + ' in the ' +
                                                      key_scenario + ' scenario:', 0., 1., 1., acceptLowerBorder=False).readValue()

                elif absorberType == 'molecule':
                    key_species = Qmolecule.readStr()
                    params['chi'] = NumericalQuestion('Enter the mixing ratio of ' + key_species + ' in the ' +
                                                      key_scenario + ' scenario:', 0., 1., 1., acceptLowerBorder=False).readValue()

                elif absorberType == 'rayleigh':
                    key_species, params = setupRayleighHaze(key_scenario)

                elif absorberType == 'gray':
                    key_species, params = setupGrayCloud(key_scenario)

                elif absorberType == 'powerlawaerosol':
                    key_species, params = setupPowerLawAerosol(key_scenario)

                else:  # tabulatedaerosol
                    key_species, params = setupTabulatedAerosol(key_scenario)

                speciesDict[key_scenario][key_species] = params

                QstopAbsorbers = TextQuestion(
                    'Do you want to add another absorber in the ' + key_scenario + ' scenario?', True)

                if not QstopAbsorbers.readStr():
                    break

        else:  # Evaporative scenarios (no temperature), only one absorbing species

            absorberType = QabsorberType.readStr()
            params = {}

            if absorberType == 'atom':
                key_species = TextQuestion('Enter the name of the absorbing species you want to consider for the ' +
                                           key_scenario + ' scenario:', False, PossibleAbsorbers).readStr()
                params['sigma_v'] = Qsigmav.readValue()

            else:
                key_species = Qmolecule.readStr()
                params['T'] = QMoleculesTemperature.readValue()

            # radialWind derives particle density from mass continuity (Mdot/v/r²);
            # Nparticles is not used and not asked for.
            if not isWindScenario:
                params['Nparticles'] = NumericalQuestion('Enter the number of ' + key_species + ' particles in the ' +
                                                         key_scenario + ' scenario:', 0., 1e50, 1., acceptLowerBorder=False).readValue()

            speciesDict[key_scenario][key_species] = params

    return speciesDict


"""
Grid parameters
"""

QlowerWavelengthBorder = NumericalQuestion(
    'Enter the lower wavelength border in Angstrom:', 500, 55000, 1e-8, roundBorders=False)
QxBins = NumericalQuestion(
    'Enter the number of bins for the spatial discretization along the chord (x-direction):', 2., 1e6, 1.)
QphiBins = NumericalQuestion(
    'Enter the number of bins for the spatial discretization for the polar coordinate (phi-direction):', 1., 1e4, 1.)
QrhoBins = NumericalQuestion(
    'Enter the number of bins for the spatial discretization in radial direction (rho-direction):', 2., 1e6, 1.)
QorbphaseBorder = NumericalQuestion(
    'Enter the orbital phase at which the light curve calculation starts and stops:', 0., 0.5, 2. * np.pi)
QorbphaseBins = NumericalQuestion(
    'Enter the number of bins for the orbital phase discretization:', 1., 1e4, 1.)


def setupGrid(architectureDict):
    """Conducts a Q&A session for grid discretization parameters.

    Asks for parameters defining the wavelength grid, the spatial integration
    grid (x, phi, rho), and the orbital phase grid.

    Args:
        architectureDict (dict): The dictionary of architectural parameters,
            used to provide context and default values.

    Returns:
        dict: A dictionary of grid parameters.
    """

    print('\nAlmost done! Specify the discretization parameters for the wavelength and spatial grids. \n')

    planet = bodies.AvailablePlanets().findPlanet(
        architectureDict['planetName'])

    gridsDict = {}

    gridsDict['lower_w'] = QlowerWavelengthBorder.readValue()
    gridsDict['upper_w'] = NumericalQuestion(
        'Enter the upper wavelength border in Angstrom:', gridsDict['lower_w'] * 1e8, 55000, 1e-8, roundBorders=False).readValue()
    gridsDict['resolutionLow'] = NumericalQuestion('Enter the resolution of the coarse wavelength grid in Angstrom:',
                                                   1e-6, (gridsDict['upper_w'] - gridsDict['lower_w']) * 1e8 / 2., 1e-8, roundBorders=False).readValue()
    gridsDict['widthHighRes'] = NumericalQuestion('Enter the bandwidth (centered on each absorption line) over which to consider a higher resolution in Angstrom:',
                                                  1e-6, (gridsDict['upper_w'] - gridsDict['lower_w']) * 1e8 / 2., 1e-8, roundBorders=False).readValue()
    gridsDict['resolutionHigh'] = NumericalQuestion(
        'Enter the resolution of the fine wavelength grid in Angstrom:', 1e-6, gridsDict['widthHighRes'] * 1e8, 1e-8, roundBorders=False).readValue()

    # Hardcoded default option is the orbital radius of the planetary orbit around the host star
    gridsDict['x_midpoint'] = planet.a
    gridsDict['x_border'] = NumericalQuestion('Enter the half chord length (x-direction) for the numerical integration along the x-axis in planetary radii:',
                                              0., (gridsDict['x_midpoint'] - planet.hostStar.R) / planet.R, planet.R, roundBorders=False).readValue()
    gridsDict['x_steps'] = QxBins.readValue()

    gridsDict['phi_steps'] = QphiBins.readValue()
    gridsDict['rho_steps'] = QrhoBins.readValue()
    # Hardcoded default option is the radius of the host star
    gridsDict['upper_rho'] = planet.hostStar.R
    gridsDict['orbphase_border'] = QorbphaseBorder.readValue()
    gridsDict['orbphase_steps'] = QorbphaseBins.readValue()

    return gridsDict


"""
Write parameter dictionary and store it as json file
"""


def createSetupFile(PATH):
    """Orchestrates the entire setup process and writes the results to a file.

    This function calls all the individual `setup...` functions in sequence,
    compiles the results into a single dictionary, and saves it as a JSON file
    named by the user.

    Args:
        PATH (str): The base path where the 'setupFiles' directory is located.
    """

    inputFileName = TextQuestion(
        'Write the name of this setup file (without file name ending):').readStr()

    fundamentalsDict = setupFundamentals()
    scenarioDict = setupScenario(fundamentalsDict)
    architectureDict = setupArchitecture(fundamentalsDict)
    speciesDict = setupSpecies(scenarioDict)
    gridsDict = setupGrid(architectureDict)

    parameters = {'Fundamentals': fundamentalsDict, 'Architecture': architectureDict,
                  'Scenarios': scenarioDict, 'Species': speciesDict, 'Grids': gridsDict}

    with open(PATH + '/setupFiles/' + inputFileName + '.txt', 'w') as outfile:
        json.dump(parameters, outfile)

    print('\n\nAll parameters are stored! To run PROMETHEUS, type <python prometheus.py ' +
          inputFileName + '>.\n\n')

    print(r"""
    *******  *******           ,/MMM8&&&.         ****     **** ******** ********** **      ** ******** **     **  ********
    /**////**/**////**     _...MMMMM88&&&&..._    /**/**   **/**/**///// /////**/// /**     /**/**///// /**    /** **////// 
    /**   /**/**   /**   .:'''MMMMM88&&&&&&''':.  /**//** ** /**/**          /**    /**     /**/**      /**    /**/**       
    /******* /*******   :     MMMMM88&&&&&&     : /** //***  /**/*******     /**    /**********/******* /**    /**/*********
    /**////  /**///**    ':...MMMMM88&&&&&&....:  /**  //*   /**/**////      /**    /**//////**/**////  /**    /**////////**
    /**      /**  //**      `''MMMMM88&&&&'''`    /**   /    /**/**          /**    /**     /**/**      /**    /**       /**
    /**      /**   //**         'MMM8&&&'         /**        /**/********    /**    /**     /**/********//*******  ******** 
    //       //     //                            //         // ////////     //     //      // ////////  ///////  ////////  
    """)
