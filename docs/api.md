# API Reference

Prometheus has no network/HTTP API. "API" here means the three programmatic
surfaces a user works with:

1. [Building a simulation in Python](#building-a-simulation-in-python) — using
   the classes directly, without a setup file.
2. [The setup-file JSON schema](#setup-file-json-schema) — the document
   `prometheus.py` consumes, produced by the `setup` wizard.
3. [The shot-noise API](#shot-noise-api) — post-processing a clean spectrum.

---

## Building a simulation in Python

The forward model can be assembled directly from the classes in
`pythonScripts/gasProperties.py`, `celestialBodies.py`, and `geometryHandler.py`.
This is what `mainRetrieval.py` does. The general recipe:

```python
import numpy as np
import pythonScripts.celestialBodies as bodies
import pythonScripts.gasProperties as gasprop
import pythonScripts.geometryHandler as geom
import pythonScripts.constants as const

# 1. Pick a pre-defined system (from Resources/planets.csv + stars.csv)
planet = bodies.AvailablePlanets().findPlanet('WASP-49b')

# 2. Wavelength grid (all in cm):
#    lower border, upper border, high-res window width, coarse res, fine res
wavelengthGrid = gasprop.WavelengthGrid(5888e-8, 5900e-8, 2e-8, 5e-9, 2e-10)

# 3. Spatial grid:
#    x_midpoint (= planet.a), x_border, x_steps,
#    rho_border (= R_star), rho_steps, phi_steps,
#    orbphase_border [rad], orbphase_steps
spatialGrid = geom.Grid(planet.a, 5. * planet.R, 30,
                        planet.hostStar.R, 40, 60, 0.25, 25)

# 4. Build a scenario and add an absorber
scenario = gasprop.HydrostaticAtmosphere(T=1000., P_0=1e-3, mu=2.3 * const.amu,
                                         planet=planet)
scenario.addConstituent('NaI', chi=1e-6)          # atomic, mixing ratio
scenario.constituents[-1].addLookupFunctionToConstituent(wavelengthGrid)

# 5. Aggregate and run
atmos = gasprop.Atmosphere([scenario], hasOrbitalDopplerShift=True)
transit = gasprop.Transit(atmos, wavelengthGrid, spatialGrid)
transit.addWavelength()

R = transit.sumOverChords(max_memory_gb=2.0)      # (n_orbphase, n_wav)
wavelength = wavelengthGrid.constructWavelengthGrid([scenario])  # (n_wav,) [cm]
orbphase = spatialGrid.constructOrbphaseAxis()                   # (n_orbphase,) [rad]
```

### Adding different constituent types

```python
# Atomic / ionic on a collisional scenario (mixing ratio chi):
scenario.addConstituent('KI', chi=1e-7)

# Molecular (requires Resources/molecularResources/H2O.h5):
scenario.addMolecularConstituent('H2O', chi=1e-4)

# Continuum scattering / aerosol (paramsDict mirrors the setup file):
scenario.addScatteringConstituent('RayleighHaze',
    {'chi': 1.0, 'sigma_ref': 5.31e-27, 'lambda_ref': 4000e-8, 'slope': 4.0})

# Always attach the lookup after adding a constituent:
scenario.constituents[-1].addLookupFunctionToConstituent(wavelengthGrid)
```

For **evaporative** scenarios the atomic absorber takes a velocity dispersion
`sigma_v` (cm/s) instead of `chi`, and a molecular absorber takes a
pseudo-temperature `T` (K):

```python
exo = gasprop.PowerLawExosphere(N=1e30, q=3.34, planet=planet)
exo.addConstituent('NaI', sigma_v=1e6)            # 10 km/s dispersion
exo.constituents[-1].addLookupFunctionToConstituent(wavelengthGrid)
```

### Stellar effects

```python
star = planet.hostStar
star.addCLVparameters(0.34, 0.28)                 # quadratic limb darkening
star.addRMparameters(vsiniStarrot=3e5, phiStarrot=0.0)   # optional RM (cm/s, rad)
star.addFstarFunction(transit.wavelength)         # load + cache PHOENIX spectrum
```

If `addFstarFunction` is never called, the star is treated as flat (`F = 1`).

### Single-chord evaluation

```python
F_in, F_out = transit.evaluateChord(phi=0.0, rho=0.5 * star.R, orbphase=0.0)
transmission = F_in / F_out
```

---

## Setup-file JSON schema

`prometheus.py` reads `../setupFiles/<name>.txt`, a JSON document with five
top-level keys. The wizard writes cgs values (after unit conversion). Below is
the schema with representative values.

### `Fundamentals`
```json
{
  "ExomoonSource": false,
  "DopplerPlanetRotation": false,
  "CLV_variations": false,
  "RM_effect": false,
  "DopplerOrbitalMotion": true
}
```
Only `DopplerOrbitalMotion` is read directly by `prometheus.py` (passed to
`Atmosphere`). `ExomoonSource` is set when a moon scenario is chosen. CLV/RM
flags are consumed when constructing the stellar treatment.

### `Architecture`
```json
{ "planetName": "WASP-49b" }
```
Plus, when applicable: `starting_orbphase_moon`, `R_moon`, `a_moon` (for the
`exomoon` scenario), and RM/CLV parameters when those fundamentals are enabled.

### `Scenarios`
A dict keyed by scenario name. Presence of a `T` key marks a **collisional**
scenario (enables molecular `chi`, scattering, cloud-top `P_top`); its absence
marks an **evaporative** scenario.

```json
{
  "hydrostatic": { "T": 1000.0, "P_0": 1.0e-3, "mu": 3.82e-24 },
  "barometric":  { "T": 1000.0, "P_0": 1.0e-3, "mu": 3.82e-24 },
  "powerLaw":    { "q_esc": 4.0, "P_0": 1.0e-3, "T": 1000.0 },
  "exomoon":     { "q_moon": 3.34 },
  "torus":       { "a_torus": 7.15e10, "v_ej": 1.0e5 },
  "serpens":     { "serpensPath": "/abs/path/to/serpens_output.txt" },
  "radialWind":  {
    "Mdot": 1.0e10, "mu": 3.82e-24,
    "wind_model": "beta",
    "v_terminal": 1.0e6, "beta": 1.0,
    "r_inner_Rp": 0.0,
    "r_outer": 10.0
  }
}
```

`radialWind` notes (parsed in `prometheus.py`):
- `r_inner_Rp` is in **planet radii**; `0` (or absent) → use `planet.R`.
- `r_outer` is in **planet radii** and optional.
- `wind_model: "parker"` replaces `v_terminal`/`beta` with a wind temperature
  `T` (Parker dynamics are fixed by `T` and planet mass; `Mdot` sets density).

### `Species`
A dict keyed by scenario name, then by species/source key. The value's keys
depend on the absorber type and scenario family.

```json
{
  "hydrostatic": {
    "NaI":          { "chi": 1.0e-6 },
    "H2O":          { "chi": 1.0e-4 },
    "RayleighHaze": { "chi": 1.0, "sigma_ref": 5.31e-27, "lambda_ref": 4000e-8, "slope": 4.0 },
    "GrayCloud":    { "chi": 1.0, "sigma_gray": 1.0e-10, "P_top": 1.0e-3 },
    "PowerLawAerosol": { "chi": 1.0, "sigma_ref": 1.0e-25, "lambda_ref": 5500e-8, "alpha": 2.0 },
    "TabulatedAerosol": { "chi": 1.0, "filepath": "Resources/aerosols/tholin.csv", "extrapolate": "edge" }
  },
  "powerLaw": {
    "NaI": { "sigma_v": 1.0e6, "Nparticles": 1.0e32 }
  }
}
```

Dispatch rules (see `prometheus.py` and `gasProperties.SCATTERER_TYPES`):
- Key in `AvailableSpecies` → atomic/ionic. Collisional uses `chi`; evaporative
  uses `sigma_v`.
- Key in `SCATTERER_TYPES` → scattering source (collisional scenarios only).
- Otherwise → molecule (requires `Resources/molecularResources/<key>.h5`).
  Collisional uses `chi`; evaporative uses `T`.
- Evaporative scenarios allow exactly **one** atomic/molecular absorber, plus
  any number of appended scattering sources. They carry `Nparticles` (except
  `radialWind`, whose density comes from mass continuity).

### `Grids`
```json
{
  "lower_w": 5888e-8, "upper_w": 5900e-8,
  "widthHighRes": 2e-8, "resolutionLow": 5e-9, "resolutionHigh": 2e-10,
  "x_midpoint": 5.79e11, "x_border": 8.6e9, "x_steps": 30,
  "upper_rho": 7.2e10, "rho_steps": 40, "phi_steps": 60,
  "orbphase_border": 1.5708, "orbphase_steps": 25
}
```
Wavelengths in cm; `x_*`/`rho` in cm; `orbphase_border` in radians; step counts
are integers.

---

## Shot-noise API

`pythonScripts/shotNoise.py` injects photon shot noise into a clean spectrum.

### `SNRModel` constructors

```python
from pythonScripts.shotNoise import SNRModel, TransitParams, apply_shot_noise

# (a) constant SNR per bin (e.g. a single ETC number)
m = SNRModel.constant(snr_per_bin=847.0)

# (b) wavelength-dependent table, rescaled to target conditions
tp = TransitParams(target_mag=11.0, transit_duration_hrs=2.14, num_bins=25)
m = SNRModel.from_table(wav_nm=[400, 500, 600], snr=[500, 800, 700],
                        baseline_mag=8.0, baseline_time_hrs=1.0,
                        transit_params=tp)

# (c) ESO ETC v2 JSON export, rescaled
m = SNRModel.from_json("uves_etc.json", transit_params=tp, correction_factor=1.0)

# (d) two-column CSV (wavelength nm, SNR), used as-is
m = SNRModel.from_csv("snr_curve.csv")
```

### Scaling law

Modes (b) and (c) rescale the baseline SNR to per-bin observing conditions:

```
SNR_target = SNR_baseline · √(F_target/F_baseline) · √(t_bin/t_baseline)
F_target/F_baseline = 10^((mag_baseline − mag_target) / 2.5)
t_bin = transit_duration_hrs / num_bins
```

### Evaluation and injection

```python
import numpy as np

wavelength_nm = wavelength_cm * 1e7        # convert cm → nm
sigma_arr = m.snr_array(wavelength_nm)     # per-bin SNR over the grid
noisy_spectrum, sigma = apply_shot_noise(wavelength_nm, clean_spectrum, m, seed=0)
```

`apply_shot_noise` returns the noisy spectrum and the 1-σ noise level
(`1/SNR`, zeroed where SNR ≤ 0), suitable for error bars.
