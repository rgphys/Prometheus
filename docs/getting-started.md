# Getting Started

This guide covers installation, the directory layout Prometheus expects, how to
build your first setup file, how to run the forward model, and how to read the
output.

---

## 1. Requirements

Prometheus is a pure-Python code that relies on the scientific Python stack.

| Package | Used for |
|---|---|
| `numpy` | All array math and broadcasting |
| `scipy` | `RegularGridInterpolator`, `interp1d`, `voigt_profile`, `erf`, `lambertw`, Gaussian smoothing |
| `numba` | JIT-compiled interpolation kernels (`@njit(parallel=True)`) |
| `h5py` | Reading molecular cross-section HDF5 tables |
| `astropy` | Reading PHOENIX stellar-spectrum FITS files and unit handling |
| `psutil` | Querying available system memory for chord batching |
| `matplotlib` | Plotting (used by `mainRetrieval.py`) |
| `astroquery` | (Optional) building NIST line lists via `Resources/astroquery_retrieval.py` |

A recent CPython (3.10+) is recommended; the code is developed and tested
against CPython 3.12. Numba must support the installed Python version.

### Installing dependencies

```bash
python -m venv env
source env/bin/activate
pip install numpy scipy numba h5py astropy psutil matplotlib astroquery
```

> The in-repo `README.md` references a shared `../requirements.txt` and a
> shared `../env/` virtual environment that live **outside** the cloned
> repository in the original author's working tree. They are not part of the
> repository. Create your own environment as shown above.

---

## 2. Getting the code

```bash
git clone https://github.com/rgphys/Prometheus.git
cd Prometheus
```

---

## 3. Directory layout

Prometheus computes paths relative to its own location. In `prometheus.py`:

```python
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

`PATH` is the directory **that encloses** the cloned `Prometheus/` folder — i.e.
the cloned repo's parent. Setup files and outputs are read from / written to
sibling directories of the repo:

```
<PATH>/
├── Prometheus/              # the cloned repository
│   ├── prometheus.py
│   ├── mainRetrieval.py
│   ├── pythonScripts/
│   ├── Resources/
│   └── docs/
├── setupFiles/              # JSON setup files (created by `setup`)   ← sibling
└── output/                  # result text files                       ← sibling
```

**Before your first run, create the sibling directories** if they do not exist:

```bash
mkdir -p ../setupFiles ../output
```

If you prefer to keep everything self-contained, you can also create
`setupFiles/` and `output/` and adjust `PATH`, but the default behavior is to
use the parent directory.

### Resources shipped with the repo

| Path | Contents |
|---|---|
| `Resources/LineList.txt` | NIST atomic/ionic line list (element, ion stage, vacuum wavelength [Å], `A_ki`, `f`-value) |
| `Resources/planets.csv` | Pre-defined planets (radius, mass, semi-major axis, host-star key, transit duration, period, impact parameter) |
| `Resources/stars.csv` | Pre-defined host stars (radius, mass, `T_eff`, `log g`, `[Fe/H]` and uncertainties) |
| `Resources/aerosols/*.csv` | Tabulated aerosol extinction cross-sections (wavelength [Å], σ [cm²]) |
| `Resources/optical_constants/` | Published refractive-index tables used to generate the aerosol CSVs |
| `Resources/astroquery_retrieval.py` | Helper to regenerate line-list entries from the NIST ADS database |
| `Resources/molecularResources/` | Expected location of molecular cross-section HDF5 files (`<molecule>.h5`); **not shipped** — supply your own |

---

## 4. Building a setup file

Run the interactive wizard:

```bash
python prometheus.py setup
```

The wizard (implemented in `pythonScripts/setup.py`) walks you through five
blocks and writes a single JSON document to `../setupFiles/<name>.txt`:

1. **Fundamentals** — toggle center-to-limb variation (CLV), the
   Rossiter–McLaughlin (RM) effect, and Doppler shifts from orbital motion.
2. **Scenarios** — one or more density distributions (`barometric`,
   `hydrostatic`, `powerLaw`, `exomoon`, `torus`, `serpens`, `radialWind`) and
   their parameters.
3. **Architecture** — the planetary system (chosen from `planets.csv`) plus any
   RM/CLV/exomoon parameters.
4. **Species** — for each scenario, the absorbers: atoms/ions, molecules, and —
   for temperature-bearing (collisional) scenarios — continuum scattering
   sources (`rayleigh`, `gray`, `powerlawaerosol`, `tabulatedaerosol`).
5. **Grids** — wavelength range and resolution (coarse + fine), spatial grid
   (`x`, `rho`, `phi` step counts), and orbital-phase range and step count.

The wizard validates every numeric entry against physical bounds and converts
the user-facing units (Å, bar, km/s, planetary/solar/Jovian radii, …) into the
cgs units used internally.

See [api.md](api.md#setup-file-json-schema) for the full JSON schema if you
prefer to hand-write or programmatically generate setup files.

---

## 5. Running the forward model

```bash
python prometheus.py <name>
```

`<name>` is the setup-file name without the `.txt` extension. Optionally cap the
RAM used for chord batching (default 2.0 GB):

```bash
python prometheus.py <name> --max-memory 4.0
```

On the first run that requires a PHOENIX stellar spectrum (i.e. when CLV/RM is
enabled and `Fstar_function` is built), Prometheus downloads the relevant
FITS files from the Göttingen PHOENIX archive and caches them under
`pythonScripts/phoenix_cache/`. If no spectrum is loaded, the star is treated as
spatially flat (`F = 1`), which is the configuration exercised by the default
fundamentals path.

When the run finishes, Prometheus prints the elapsed time and the maximum and
minimum flux decrease (in percent) across the grid.

---

## 6. Output format

Results are written to `../output/<name>.txt` via `numpy.savetxt`, with a
3-line header. The layout is:

- **Row 1** — orbital phases in units of full orbits (i.e. radians / 2π). The
  first entry of this row is `NaN` (a placeholder occupying the wavelength
  column).
- **Subsequent rows** — column 1 is the **wavelength in cm**; the remaining
  columns are the transit depth `R(phase, wavelength)` for each phase.

Schematically:

```
NaN        phase_0   phase_1   ...   phase_{P-1}
λ_0[cm]    R(0,0)    R(0,1)    ...   R(0,P-1)
λ_1[cm]    R(1,0)    R(1,1)    ...   R(1,P-1)
 ...
```

To load and plot a transmission spectrum at a single phase:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../output/myrun.txt')
phases = data[0, 1:]            # orbital phases [full orbits]
wavelength_cm = data[1:, 0]    # wavelength [cm]
R = data[1:, 1:]               # transit depth, shape (n_wav, n_phase)

plt.plot(wavelength_cm * 1e8, R[:, 0])  # phase index 0, x-axis in Angstrom
plt.xlabel('Wavelength [Å]')
plt.ylabel('Transit depth R')
plt.show()
```

---

## 7. Optional: adding observational noise

The `pythonScripts/shotNoise.py` module post-processes a clean spectrum by
injecting photon shot noise based on an SNR model (constant, tabulated, ESO ETC
JSON, or CSV). It operates on plain NumPy arrays and is independent of the
simulation classes:

```python
from pythonScripts.shotNoise import SNRModel, apply_shot_noise

snr_model = SNRModel.constant(snr_per_bin=847.0)
noisy_spectrum, sigma = apply_shot_noise(wavelength_nm, spectrum, snr_model, seed=0)
```

See [api.md](api.md#shot-noise-api) for all constructors and the scaling law.

---

## 8. A scripted (non-interactive) example

`mainRetrieval.py` at the repo root is a worked example that builds a
WASP-49 b + exomoon sodium-cloud simulation entirely in Python (no setup file),
runs the forward model, extracts a Na D light curve, and plots it. It is the
best reference for using Prometheus as a library — see
[api.md](api.md#building-a-simulation-in-python).
