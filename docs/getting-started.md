# Getting started

This page covers installation, dependencies, and a complete first simulation built entirely from Python objects.

---

## Requirements

- **Python 3.11+** (3.13 recommended)
- The scientific stack listed below

The core dependencies imported by the package are:

| Package | Used for |
|---|---|
| `numpy` | Arrays, broadcasting, the vectorized chord sum |
| `scipy` | `interp1d` / `RegularGridInterpolator`, `voigt_profile`, `erf`, `lambertw` |
| `numba` | JIT-compiled interpolation kernels (`n_interp_log`, `_bilinear_PT_interp`) |
| `h5py` | Reading molecular cross-section tables (`.h5`) |
| `astropy` | Unit/constant helpers and spectrum retrieval |
| `psutil` | Querying available RAM for memory-aware batching |

`matplotlib` and `pandas` are not required to run a simulation, but the example figure scripts use them for plotting and line-list inspection.

---

## Installation

Clone the repository and install the dependencies into a virtual environment:

```bash
git clone https://github.com/rgphys/Prometheus.git
cd Prometheus

python -m venv env
source env/bin/activate            # Windows: env\Scripts\activate

pip install numpy scipy numba h5py astropy psutil
# optional, for the example figures:
pip install matplotlib pandas
```

Prometheus is imported as the `pythonScripts` package from the repository root. The simplest way to make it importable is to run your scripts from the repo root, or to prepend the repo root to `sys.path`:

```python
import sys, os
sys.path.insert(0, '/path/to/Prometheus')   # directory containing pythonScripts/

import pythonScripts.gasProperties as gasprop
import pythonScripts.celestialBodies as bodies
import pythonScripts.geometryHandler as geom
import pythonScripts.constants as const
```

If you prefer an editable install, add a `pyproject.toml`/`setup.cfg` that exposes the `pythonScripts` package and run `pip install -e .`; the import paths above remain unchanged.

### Data resources

The package reads its physics data from the `Resources/` directory at the repo root:

- `Resources/LineList.txt` — atomic/ionic line list (element, ionization state, vacuum wavelength, Aₖᵢ, fᵢₖ).
- `Resources/molecularResources/*.h5` — molecular cross-section tables (one HDF5 file per molecule, e.g. `H2O.h5`).
- `Resources/planets.csv`, `Resources/stars.csv` — the built-in planet/star catalog read by `AvailablePlanets`.

These are resolved relative to the package location automatically; no configuration is needed.

---

## Units

Prometheus works internally in **cgs**. The conventions you will encounter most often:

- Wavelengths in **cm** (`5890 Å = 5.890e-5 cm`, `1 µm = 1e-4 cm`).
- Pressures in **barye** (cgs; `1 bar = 1e6 barye`).
- Number densities in **cm⁻³**, lengths in **cm**, velocities in **cm/s**, masses in **g**.
- Mean molecular weight `mu` is a **mass in grams**, so write it as a multiple of `const.amu` (e.g. `2.3 * const.amu`).

Helpful constants live in `pythonScripts.constants`: `amu`, `k_B`, `G`, `c`, `R_J`, `M_J`, `R_sun`, `M_sun`, `R_Io`, `AU`.

---

## Your first simulation

The pattern is always the same four objects:

1. a **planet** (carries its host star),
2. one or more **density models** wrapped in an `Atmosphere`,
3. a **`WavelengthGrid`**,
4. a **`Grid`** (spatial/geometry),

assembled into a **`Transit`** that you run with `sumOverChords`.

```python
import numpy as np
import pythonScripts.gasProperties as gasprop
import pythonScripts.celestialBodies as bodies
import pythonScripts.geometryHandler as geom
import pythonScripts.constants as const

# 1. Planet from the built-in catalog (also loads its host star).
planet = bodies.AvailablePlanets().findPlanet('WASP-39b')

# 2. Wavelength grid: 0.35–1.0 µm, fine sampling near lines.
#    Bounds are in cm; 0.35 µm = 0.35e-4 cm.
wg = gasprop.WavelengthGrid(
    lower_w=0.35e-4, upper_w=1.0e-4,
    widthHighRes=8e-8,        # width of the fine-sampled region around each line [cm]
    resolutionLow=4e-7,       # coarse step away from lines [cm]
    resolutionHigh=1e-9,      # fine step inside line cores [cm]
)

# 3. Hydrostatic atmosphere with a Na I absorber.
#    T [K], P_0 [barye], mu [g].
atm = gasprop.HydrostaticAtmosphere(T=1100.0, P_0=1e4,
                                    mu=2.3 * const.amu, planet=planet)
atm.addConstituent('NaI', 3e-7)                       # species name, mixing ratio
atm.constituents[-1].addLookupFunctionToConstituent(wg)  # build the opacity lookup

# 4. Spatial grid (line-of-sight x, sky-plane rho/phi, orbital phase).
s_grid = geom.Grid(
    x_midpoint=planet.a,              # chord centered on the planet [cm]
    x_border=14.0 * planet.R,         # half-length of the integration chord [cm]
    x_steps=80,
    rho_border=planet.hostStar.R,     # integrate out to the stellar limb [cm]
    rho_steps=400,
    phi_steps=20,
    orbphase_border=0.0,              # single mid-transit phase
    orbphase_steps=1,
)

# Assemble and run.
atmosphere = gasprop.Atmosphere([atm], hasOrbitalDopplerShift=False)
sim = gasprop.Transit(atmosphere, wg, s_grid)
sim.addWavelength()                   # builds sim.wavelength from the line list

R = sim.sumOverChords(max_memory_gb=0.5)   # shape (orbphase_steps, n_wavelength)

wavelength_um = sim.wavelength * 1e4
transit_depth_ppm = (1.0 - R[0]) * 1e6

print(f'{len(sim.wavelength)} wavelength points, '
      f'peak depth {transit_depth_ppm.max():.0f} ppm')
```

### What each step does

- **`AvailablePlanets().findPlanet(name)`** returns a `Planet` object with `R`, `M`, `a` (semi-major axis) and a `hostStar`. Use `AvailablePlanets().listPlanetNames()` to see the catalog.
- **`WavelengthGrid`** stores the sampling parameters. The actual array is built later by `Transit.addWavelength`, which places fine sampling (`resolutionHigh`) within `widthHighRes` of every atomic line in range and coarse sampling (`resolutionLow`) elsewhere. Molecular and continuum opacities are smooth and do not add grid points.
- **`addConstituent(name, chi)`** adds an atomic/ionic absorber with mixing ratio `chi`. **`addLookupFunctionToConstituent(wg)`** must be called afterward to precompute the Voigt cross-section interpolator — atoms only. (`atm.constituents[-1]` is the constituent you just added.)
- **`Grid`** discretizes the integration. `x` is the line of sight, `rho`/`phi` are sky-plane polar coordinates, and `orbphase` is the orbital-phase axis (use `orbphase_steps > 1` for a light curve).
- **`Atmosphere([...], hasOrbitalDopplerShift=...)`** wraps one or more density models. Set the flag `True` for high-resolution work where orbital Doppler shifts matter.
- **`sumOverChords(max_memory_gb=...)`** runs the simulation and returns `R` of shape `(orbphase_steps, n_wavelength)`. `max_memory_gb` caps the per-batch RAM footprint.

### Adding a stellar spectrum

By default the stellar disk is treated as a flat, limb-darkened continuum (`F_star = 1`), which keeps the chord sum fast. To use a realistic PHOENIX stellar spectrum (needed for accurate high-resolution line work), call this **after** `addWavelength`:

```python
sim.addWavelength()
planet.hostStar.addFstarFunction(sim.wavelength)   # PHOENIX spectrum on the sim grid
R = sim.sumOverChords(max_memory_gb=4.0)
```

You can also attach limb-darkening (`star.addCLVparameters(u1, u2)`) and Rossiter–McLaughlin rotation (`star.addRMparameters(vsini, phi_rot)`) to the `planet.hostStar` object.

---

Continue to **[examples.md](examples.md)** for molecular and torus runs, or **[architecture.md](architecture.md)** for how the optical depth is computed under the hood.
