# Examples

These worked examples are simplified and annotated versions of the figure scripts that ship with the project (`fig*.py`). Each builds a simulation purely from Python objects — no configuration files. All examples assume the `pythonScripts` package is importable (see [getting-started.md](getting-started.md#installation)).

Common imports:

```python
import numpy as np
import pythonScripts.gasProperties as gasprop
import pythonScripts.celestialBodies as bodies
import pythonScripts.geometryHandler as geom
import pythonScripts.constants as const
```

---

## 1. Hydrostatic transit spectrum with atoms (+ a haze)

*Based on the optical segment of `fig1_model_zoo.py`.* A hydrostatic WASP-39 b atmosphere with Na I and K I lines on a Rayleigh-haze continuum, integrated across the optical.

```python
planet = bodies.AvailablePlanets().findPlanet('WASP-39b')

MU     = 2.3 * const.amu     # mean molecular weight (mass, g)
P_0    = 1e4                 # base pressure (barye)
T_ATM  = 1100.0             # K
CHI_NA = 3e-7               # Na mixing ratio
CHI_K  = 1e-7               # K mixing ratio

# Wavelength grid: 0.35–1.0 µm, fine sampling in the alkali line cores.
wg = gasprop.WavelengthGrid(
    lower_w=0.35e-4, upper_w=1.0e-4,
    widthHighRes=8e-8, resolutionLow=4e-7, resolutionHigh=1e-9,
)

# Hydrostatic atmosphere with two atomic absorbers + a Rayleigh haze.
atm = gasprop.HydrostaticAtmosphere(T_ATM, P_0, MU, planet)
for name, chi in [('NaI', CHI_NA), ('KI', CHI_K)]:
    atm.addConstituent(name, chi)
    atm.constituents[-1].addLookupFunctionToConstituent(wg)   # atoms: build Voigt lookup
atm.addScatteringConstituent(
    'RayleighHaze',
    {'chi': 1.0, 'sigma_ref': 1.5e-22, 'lambda_ref': 4000e-8, 'slope': 4.0},
)

# Finely resolved geometry (thin annulus around the planet).
s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=14.0 * planet.R, x_steps=80,
    rho_border=planet.hostStar.R, rho_steps=400, phi_steps=20,
    orbphase_border=0.0, orbphase_steps=1,
)

# Continuum-dominated, so orbital Doppler shifts can be left off (faster).
sim = gasprop.Transit(
    gasprop.Atmosphere([atm], hasOrbitalDopplerShift=False), wg, s_grid)
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.45)[0]

wav_um   = sim.wavelength * 1e4
depth_ppm = (1.0 - R) * 1e6
# Na I D appears near 0.5894 µm, K I near 0.7684 µm, on a λ^-4 haze slope.
```

**Notes**

- Atoms require `addLookupFunctionToConstituent(wg)`; the haze (a scatterer) does not.
- `atm.constituents[-1]` is the constituent just added — the lookup is attached per-constituent.
- The fine `rho_steps=400` is needed because the absorbing annulus is thin relative to the stellar disk; it converges the chord sum.

---

## 2. Molecular transmission spectrum (JWST near-IR)

*Based on `fig8_molecular_jwst-8.py`.* H₂O, CO₂, CO, and SO₂ in a hydrostatic atmosphere across 0.6–5.3 µm — the NIRSpec/PRISM grasp.

```python
planet = bodies.AvailablePlanets().findPlanet('WASP-39b')

MU    = 2.3 * const.amu
P_0   = 1e4
T_ATM = 1100.0

# Roughly WASP-39 b-like mixing ratios; SO2 is a photochemical product.
MOLECULES = [('H2O', 1e-3), ('CO2', 1e-4), ('CO', 3e-4), ('SO2', 3e-6)]

# Molecular opacity is smooth — a uniform, coarse grid suffices
# (high == low resolution here).
wg = gasprop.WavelengthGrid(
    lower_w=0.6e-4, upper_w=5.3e-4,
    widthHighRes=1e-7, resolutionLow=4e-6, resolutionHigh=4e-6,
)

s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=15.0 * planet.R, x_steps=50,
    rho_border=planet.hostStar.R, rho_steps=150, phi_steps=40,
    orbphase_border=0.0, orbphase_steps=1,
)

atm = gasprop.HydrostaticAtmosphere(T_ATM, P_0, MU, planet)
for name, chi in MOLECULES:
    atm.addMolecularConstituent(name, chi)
    # Passing wg slices the cross-section table to the sim range (faster).
    atm.constituents[-1].addLookupFunctionToConstituent(wg)

sim = gasprop.Transit(
    gasprop.Atmosphere([atm], hasOrbitalDopplerShift=False), wg, s_grid)
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.45)[0]

wav_um   = sim.wavelength * 1e4
depth_ppm = (1.0 - R) * 1e6
# Diagnostic bands: H2O 1.4/1.9 µm, CO2 4.3 µm, CO 4.7 µm, SO2 4.0 µm.
```

**Notes**

- Use `addMolecularConstituent` (not `addConstituent`). The molecule name must match an HDF5 file in `Resources/molecularResources/` (e.g. `H2O.h5`).
- Passing `wg` to `addLookupFunctionToConstituent` slices the `(P, T, λ)` table to the simulation range plus a small Doppler margin — a large speedup for narrow grids.
- To inspect each species' contribution, build a separate atmosphere with a single molecule and re-run (as `fig8` does).

---

## 3. Io-analogue Na torus (exosphere)

*Based on `fig3_io_torus-3.py`.* A circumplanetary neutral-Na gas torus around WASP-49 b, producing a Na D transmission signature. Exospheres are normalized by a **total particle number** `N`, and their single atomic constituent takes a **velocity dispersion** (cm/s), not a mixing ratio.

```python
planet = bodies.AvailablePlanets().findPlanet('WASP-49b')

R_J     = const.R_J
A_TORUS = 5.9 * R_J     # Io-like orbital distance (cm)
V_EJ    = 1e5           # ejection velocity (cm/s) — sets torus scale height
N_TORUS = 1e33          # total Na atoms
SIGMA_V = 2e6           # thermal velocity dispersion (cm/s)

# Narrow grid around the Na D doublet (~5890 Å = 5.89e-5 cm).
wg = gasprop.WavelengthGrid(
    lower_w=5.880e-5, upper_w=5.910e-5,
    widthHighRes=4e-8, resolutionLow=3e-9, resolutionHigh=2e-10,
)

s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=12.0 * planet.R, x_steps=28,
    rho_border=planet.hostStar.R, rho_steps=22, phi_steps=30,
    orbphase_border=0.0, orbphase_steps=1,
)

torus = gasprop.TorusExosphere(N=N_TORUS, a_torus=A_TORUS,
                               v_ej=V_EJ, planet=planet)
torus.addConstituent('NaI', SIGMA_V)        # 2nd arg is sigma_v (cm/s), not chi
torus.constituents[-1].addLookupFunctionToConstituent(wg)

# Torus kinematics matter → enable orbital Doppler shifting.
atmos = gasprop.Atmosphere([torus], hasOrbitalDopplerShift=True)
sim = gasprop.Transit(atmos, wg, s_grid)
sim.addWavelength()

# Use a realistic stellar spectrum (call AFTER addWavelength).
planet.hostStar.addFstarFunction(sim.wavelength)

R = sim.sumOverChords(max_memory_gb=4.0)[0]

wav_ang = sim.wavelength * 1e8           # cm → Å
# Continuum-normalize to read excess absorption:
cont = np.median(R[(wav_ang < 5884) | (wav_ang > 5902)])
excess_pct = (R / cont - 1.0) * 100.0
```

**Notes**

- For an exosphere, `addConstituent(speciesName, sigma_v)` takes a velocity dispersion in cm/s; the absolute amount of gas is set by `N`.
- `hasOrbitalDopplerShift=True` is appropriate here because the line shape and the orbital velocity matter at this resolution.
- `addFstarFunction` requires PHOENIX spectra to be available; omit it (flat star) if you only need the line shape relative to a normalized continuum.

---

## 4. Combining a lower atmosphere with an escaping wind

An `Atmosphere` can hold several density models at once. This pattern layers a hazy hydrostatic lower atmosphere beneath a radially escaping Parker wind that carries a trace Na absorber (cf. the wind physics in `fig6_atmospheric_escape-6.py`).

```python
planet = bodies.AvailablePlanets().findPlanet('HD189733b')

NA_MASS = 22.99 * const.amu     # tracer mass (density normalization)
MU_BULK = 1.3  * const.amu      # bulk H/He gas mass (Parker dynamics)

wg = gasprop.WavelengthGrid(
    lower_w=5.884e-5, upper_w=5.896e-5,
    widthHighRes=5e-8, resolutionLow=3e-9, resolutionHigh=1.2e-10,
)
s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=40.0 * planet.R, x_steps=55,
    rho_border=planet.hostStar.R, rho_steps=28, phi_steps=28,
    orbphase_border=0.0, orbphase_steps=1,
)

# Escaping wind: exact isothermal Parker solution. mu sets the tracer density
# normalization; wind_mu sets the (bulk-gas) wind dynamics.
wind = gasprop.RadialWindExosphere(
    Mdot=1.0e3, mu=NA_MASS, wind_model='parker', T=9.0e3, wind_mu=MU_BULK,
    r_inner=2.0 * planet.R, r_outer=40.0 * planet.R, planet=planet,
)
wind.addConstituent('NaI', 1e6)             # sigma_v in cm/s
wind.constituents[-1].addLookupFunctionToConstituent(wg)

# Hazy hydrostatic lower atmosphere providing a continuum floor.
haze_atm = gasprop.HydrostaticAtmosphere(T=2500.0, P_0=1e6,
                                         mu=2.3 * const.amu, planet=planet)
haze_atm.addScatteringConstituent(
    'RayleighHaze',
    {'chi': 1.0, 'sigma_ref': 1e-19, 'lambda_ref': 4000e-8, 'slope': 4.0})

# Both models in one atmosphere. Position-dependent wind Doppler needs the flag.
atmos = gasprop.Atmosphere([wind, haze_atm], hasOrbitalDopplerShift=True)
sim = gasprop.Transit(atmos, wg, s_grid)
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.8)[0]
```

**Notes**

- `wind_model='parker'` requires `T` and `planet`; `v_terminal`/`beta` are unused. `wind_mu` separates the wind *dynamics* (bulk gas) from the tracer *density* normalization (`mu`).
- With `hasOrbitalDopplerShift=True`, the wind's per-cell line-of-sight velocity produces the blue/red line asymmetry and kinematic broadening characteristic of escape.
- The list order in `Atmosphere([...])` does not matter physically; optical depths from all models are summed.

---

## Producing a light curve or phase-resolved map

Any of the above becomes a light curve by setting `orbphase_steps > 1` and a non-zero `orbphase_border` on the `Grid`. `sumOverChords` then returns `R` with shape `(orbphase_steps, n_wavelength)`:

```python
N_PHASES = 21
orbphase_border = (2.6 / (planet.orbitalPeriod * 24.0)) * 2.0 * np.pi   # ±2.6 h
s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=40.0 * planet.R, x_steps=55,
    rho_border=planet.hostStar.R, rho_steps=28, phi_steps=28,
    orbphase_border=orbphase_border, orbphase_steps=N_PHASES,
)
# ... build sim as before ...
R_2D = sim.sumOverChords(max_memory_gb=0.8)     # (N_PHASES, n_wavelength)
phases_rad = s_grid.constructOrbphaseAxis()
```

Each row `R_2D[i]` is the spectrum at orbital phase `phases_rad[i]`; integrating a row over a line bandpass yields a light-curve point, and stacking rows gives a velocity-vs-phase transmission map.
