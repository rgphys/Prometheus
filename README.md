# Prometheus

PRObing Mass loss in Exoplanetary Transits with Hydrostatic, Evaporative and User-defined Scenarios. PROMETHEUS is a radiative transfer tool to compute lightcurves and transmission spectra of an object transiting its host star, typically an exoplanet. The code calculates the amount of absorption during the transit for gaseous media in arbitrary geometry. PROMETHEUS supports various density profiles beyond the canonical hydrostatic (barometric) law for dense atmospheres, such as the outgassed cloud of an exomoon, a circumplanetary torus, or a radially expanding wind. For these tenuous exospheres, line absorption by various atoms and ions is considered (with line lists from NIST). Additionally, it is possible to model absorption by molecules based on ExoMOL lookup tables.

---

## Features

- **Multiple atmosphere/exosphere models:**
  - Barometric atmosphere
  - Hydrostatic atmosphere
  - Power-law atmosphere / exosphere
  - Exomoon exosphere
  - Torus exosphere
  - SERPENS particle simulation integration
  - Radial wind exosphere (beta-law velocity profile)
- **Multi-species support** — atoms, ions, molecules, and continuum scatterers
- **Continuum opacity sources:**
  - Rayleigh/power-law haze (`RayleighHaze`) — parametrized by reference cross-section and wavelength slope
  - Gray cloud deck (`GrayCloud`) — wavelength-independent opacity with optional cloud-top pressure cutoff
  - Power-law aerosol (`PowerLawAerosol`) — Ångström exponent parameterization
  - Tabulated aerosol (`TabulatedAerosol`) — cross-sections from a user-supplied CSV file
- **Doppler orbital motion** correction (bulk) and **position-dependent wind velocity** Doppler shifts for radial wind models
- **Vectorized computation** via NumPy broadcasting and Numba-accelerated interpolation; no Python loops over individual chords
- **Memory-aware batching** via `memoryHandler.py` — chords are processed in chunks sized to stay within a configurable RAM limit
- **Configurable memory limits** via `--max-memory` flag
- **High/low resolution wavelength grids** for flexible spectral sampling

> **Note:** The core computation path is fully vectorized using NumPy and Numba JIT kernels. Python `multiprocessing` is imported but not used in the active computation path; the import is a legacy artifact.

---

## Requirements

- Python 3.13+
- See `../requirements.txt` for the full dependency list (numpy, scipy, astropy, h5py, numba, etc.)

---

## Installation

```bash
git clone https://github.com/CrazeXD/Prometheus.git
cd Prometheus
# Activate the shared venv from the repo root
source ../env/bin/activate
```

---

## Usage

### 1. Create a Setup File

Run the interactive setup script to generate a JSON configuration file:

```bash
python prometheus.py setup
```

This creates a `.txt` setup file under `../setupFiles/`. The setup wizard will prompt for:
- Scenario type and parameters
- Absorbing species (atoms, ions, molecules, Rayleigh haze, gray cloud, power-law aerosol, tabulated aerosol)
- Spatial and wavelength grid resolution
- Orbital architecture

### 2. Run the Forward Model

```bash
python prometheus.py <setup_name>
```

Optionally, limit RAM usage (default is 2 GB):

```bash
python prometheus.py <setup_name> --max-memory 4.0
```

### 3. Output

Results are saved to `../output/<setup_name>.txt` with the format:

- **Row 1:** Orbital phases (in units of full orbit)
- **Remaining rows:** Wavelength [cm] (column 1), Transit depth R(orbital phase, wavelength) (remaining columns)

---

## Project Structure

```
Prometheus/
├ pythonScripts/
│   ├ setup.py           # Interactive setup file generator
│   ├ gasProperties.py   # Atmosphere/exosphere models, scattering, transit computation
│   ├ celestialBodies.py # Planet & moon definitions
│   ├ geometryHandler.py # Spatial grid and chord geometry
│   ├ memoryHandler.py   # Memory-aware chunk processing
│   └ constants.py       # Physical constants & available species
├ Resources/             # Cross-section and species data
│   ├ molecularResources/ # HDF5 molecular cross-sections
│   ├ aerosols/           # Precomputed aerosol Mie extinction tables
│   └ optical_constants/  # Published optical constants
└ docs/                  # Documentation
```

Setup files are stored in `../setupFiles/` (gitignored) and outputs in `../output/`.

---

## Atmospheric Scenarios

| Scenario | Description | Key parameters |
|---|---|---|
| `barometric` | Isothermal atmosphere with exponential pressure profile | T, P_0, mu |
| `hydrostatic` | Hydrostatic equilibrium atmosphere | T, P_0, mu |
| `powerLaw` | Power-law density profile for tenuous exospheres | q_esc, optional P_0/T |
| `exomoon` | Exosphere sourced from an orbiting moon | q_moon, N_particles |
| `torus` | Neutral gas torus around the planet | a_torus, v_ej, N_particles |
| `serpens` | Interpolated density from SERPENS particle simulation output | path, N_particles |
| `radialWind` | Radially expanding beta-law wind (density from mass continuity) | Mdot, mu, v_terminal, beta, r_inner, r_outer |

### Radial Wind Details

The `radialWind` scenario implements a beta-law velocity profile:

```
v(r) = v_terminal × max(1 − r_inner / r, 0)^beta
n(r) = Mdot / (4π r² v(r) μ)
```

This is a parametrized wind law, **not** a full isothermal Parker wind solver. To approximate a thermally driven Parker wind, set `beta ≈ 1` and `v_terminal` near the local sound speed. The density diverges at `r = r_inner` where `v → 0`; a floor of `v_terminal × 1e-3` regularizes the inner boundary.

When Doppler orbital motion is enabled, the wind's radially varying LOS velocity is applied per grid cell (per-chord × per-x), giving asymmetric blueshift (front face) and redshift (back face) signatures in the absorption line profiles.

---

## Opacity Sources

Each scenario can include any combination of absorbing and scattering constituents:

| Type | Key | Description |
|---|---|---|
| Atomic / ionic line absorber | species name (e.g. `NaI`, `KI`, `CaII`) | Voigt-profile line absorption from NIST line lists |
| Molecular absorber | molecule name (e.g. `H2O`, `SO2`) | Pre-computed cross-sections from ExoMOL HDF5 tables |
| Rayleigh / power-law haze | `RayleighHaze` | σ(λ) = σ_ref · (λ_ref/λ)^slope; slope = 4 for pure Rayleigh |
| Gray cloud deck | `GrayCloud` | Wavelength-independent σ_gray; optional cloud-top pressure cutoff |
| Power-law aerosol | `PowerLawAerosol` | σ(λ) = σ_ref · (λ/λ_ref)^(−alpha); alpha in Ångström convention |
| Tabulated aerosol | `TabulatedAerosol` | σ(λ) from user-supplied CSV (wavelength [Å], sigma [cm²]) |

All scattering/aerosol constituents contribute extinction to the Beer-Lambert optical depth by treating scattered photons as lost from the beam. **No scattering phase function or multiple scattering is modelled.** Cross-sections are smooth and analytic (or tabulated), so no Doppler shifting is applied to continuum opacities.

The `PowerLawAerosol` and `GrayCloud` cloud-top pressure confinement (`P_top`) is currently only supported for collisional (temperature-bearing) scenarios; it is ignored in evaporative exosphere contexts.

---

## Computation Architecture

The core computation in `Transit.sumOverChords` is fully vectorized:

1. Chords are batched into memory-sized chunks by `memoryHandler.calculate_optimal_chunk_size`.
2. For each chunk, `Atmosphere.getLOSopticalDepth_Batch` computes optical depths with **no Python loops over individual chords**:
   - Atomic absorbers: column density factorization + cached Doppler-shifted cross-sections via Numba `n_interp_log`.
   - Molecular absorbers: pressure–temperature–wavelength 3D interpolation via `scipy.RegularGridInterpolator`, summed with `np.einsum`.
   - Aerosol/haze scatterers: analytic or tabulated σ(λ) × column density.
   - Wind models: per-(chord, x) Doppler shifts applied via broadcasting when `calculateLOSVelocity` is present.

Typical RAM usage per chunk: `n_chords × n_x × n_wav × 64 bytes` (molecular path) or `n_chords × n_wav × 16 bytes` (atomic path).

---

## Running Smoke Tests

A self-contained smoke test script is included that does not require network access or PHOENIX spectra:

```bash
cd Prometheus/
python test_smoke.py
```

The script tests: hydrostatic NaI absorption, Rayleigh haze slope, gray cloud flat extinction, `PowerLawAerosol` slope, `RadialWindExosphere` densities, and `evaluateChord` compatibility.

---

## License

This project is licensed under the GPL-3.0 License. See [LICENSE](LICENSE) for details.
