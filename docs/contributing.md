# Contributing

Prometheus is a research code, and contributions — new density models, additional opacity sources, performance work, and documentation — are welcome. This page describes how to set up a development environment and the conventions the code follows.

---

## Development setup

Prometheus has no build step: it is a plain Python package (`pythonScripts`) imported from the repository root. To work on it, clone the repo and install the runtime dependencies into a virtual environment.

```bash
git clone https://github.com/rgphys/Prometheus.git
cd Prometheus

python -m venv env
source env/bin/activate            # Windows: env\Scripts\activate

pip install numpy scipy numba h5py astropy psutil
# for running the example figure scripts:
pip install matplotlib pandas
```

The package is imported as `pythonScripts.*` (see [getting-started.md](getting-started.md#installation)). Run your scripts from the repository root, or prepend the repo root to `sys.path`, so that `import pythonScripts.gasProperties` resolves. The physics data in `Resources/` is located relative to the package automatically — no configuration is required.

---

## Running the examples

The `fig*.py` scripts at the repository root are the de-facto integration tests: each builds a full simulation programmatically and exercises a different code path (atomic lines, molecular bands, tori, escaping winds, light curves). Running one end-to-end is the quickest way to confirm a change has not broken anything.

```bash
python fig1_model_zoo.py
python fig8_molecular_jwst-8.py
```

Some figures (`fig2`, `fig4`, `fig9`, `fig10`, `fig11`) additionally depend on the separate `mnemosyne`/`dishoom` packages, which sit on top of Prometheus and are not part of this repository; the Prometheus-only figures (`fig1`, `fig3`, `fig6`, `fig7`, `fig8`) run against `pythonScripts.*` alone.

When you add a feature, the lightest-weight check is to build a small simulation directly from Python objects — a single planet, one density model, a narrow `WavelengthGrid`, and a coarse `Grid` — and confirm `Transit.sumOverChords` returns a sensible spectrum. Keep `max_memory_gb` small (e.g. `0.1–0.5`) for quick iterations.

---

## Code style and conventions

- **Units are cgs everywhere.** Wavelengths in cm, pressures in barye, lengths in cm, velocities in cm/s, masses in g, temperatures in K. Mean molecular weight is a mass in grams (a multiple of `const.amu`). Keep new code in cgs and convert only at the presentation layer.
- **The API is programmatic.** A simulation is assembled from constructed Python objects — there is no configuration-file layer or command-line interface, and contributions should preserve that. New functionality should be exposed as classes/methods that compose with `Atmosphere` and `Transit`.
- **Stay vectorized.** The optical-depth kernel is fully vectorized over chords (no Python loop over individual lines of sight), and the hot interpolation paths are Numba `@njit(parallel=True, fastmath=True)` kernels. New density models implement a vectorized `calculateNumberDensity` that accepts batched `(n_chords,)` / `(n_chords, n_x)` inputs; avoid materializing the full `(chord, x, wavelength)` tensor.
- **Respect the memory budget.** Heavy allocations should flow through the memory-aware batching in `memoryHandler.py` rather than assuming all chords fit in RAM at once. If a new path changes per-chord memory, update `estimate_chord_memory` accordingly.
- **Match the surrounding style.** Follow the naming and structure already present in the module you are editing (e.g. `gasProperties.py`). Keep public constructor argument names stable, since they are part of the documented API.

---

## Adding a new density model

A density model holds a reference to its `Planet` (or `Moon`) and exposes a vectorized number-density method that the optical-depth kernel calls. To add one:

1. Subclass the appropriate base (`CollisionalAtmosphere` for mixing-ratio constituents, `EvaporativeExosphere` for `N`-normalized single-constituent models) so it inherits the constituent-adding API.
2. Implement the vectorized `calculateNumberDensity` for your profile.
3. If the model carries kinematics (like an escaping wind), expose `calculateLOSVelocity` so the kernel applies a position-dependent Doppler shift when `hasOrbitalDopplerShift=True`.
4. Confirm it composes inside `Atmosphere([...])` alongside the existing models and runs through `Transit.sumOverChords`.

See [architecture.md](architecture.md) for the object model and the optical-depth kernel, and [api-reference.md](api-reference.md) for the existing constructor signatures to mirror.

---

## Adding physics data

- **Atomic / ionic lines** come from `Resources/LineList.txt` (element, ionization state, vacuum wavelength, Aₖᵢ, fᵢₖ); a new species also needs an entry in the `AvailableSpecies` catalog in `constants.py` (mass and ionization state).
- **Molecular cross sections** are one HDF5 file per molecule in `Resources/molecularResources/<name>.h5`, tabulated on a `(pressure, temperature, wavelength)` grid; the filename must match the name passed to `addMolecularConstituent`.
- **Planets / stars** are rows in `Resources/planets.csv` / `Resources/stars.csv`, read by `AvailablePlanets`.

---

## Submitting changes

1. Branch from `main`.
2. Make your change, and verify at least one relevant `fig*.py` (or a small programmatic script) still produces a sensible spectrum.
3. Keep commits focused and write a clear message describing the physics or behavior that changed.
4. Open a pull request against `rgphys/Prometheus` describing the motivation, the approach, and how you validated it.

---

## Where to go next

- **[architecture.md](architecture.md)** — how the modules fit together and how a spectrum is computed.
- **[api-reference.md](api-reference.md)** — constructor signatures and methods.
- **[examples.md](examples.md)** — runnable end-to-end scripts.
