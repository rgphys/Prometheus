# Contributing

This guide explains how to extend Prometheus: the repository conventions, how to
add a new density scenario or opacity source (the two most common extensions),
how to validate your changes, and the pull-request workflow.

Prometheus is a research code. There is **no automated test suite, linter, or
CI pipeline in the repository** — validation is done by running the forward
model and inspecting results. Keep that in mind: the burden of verifying a
change is on the contributor and the reviewer.

---

## 1. Repository layout recap

```
Prometheus/
├── prometheus.py            # CLI driver / entry point
├── mainRetrieval.py         # scripted library-usage example
├── pythonScripts/
│   ├── constants.py         # cgs constants, Species / AvailableSpecies
│   ├── celestialBodies.py   # Star, Planet, Moon, AvailablePlanets
│   ├── geometryHandler.py   # Grid (spatial/temporal discretization)
│   ├── gasProperties.py     # scenarios, constituents, kernels, Transit
│   ├── memoryHandler.py     # chord-batch sizing
│   ├── setup.py             # interactive setup wizard
│   └── shotNoise.py         # SNR / shot-noise post-processing
├── Resources/               # line list, planet/star tables, aerosol CSVs
└── docs/                    # this documentation
```

See [architecture.md](architecture.md) for the object model and data flow, and
[modules.md](modules.md) for the per-class reference.

---

## 2. Code style

The code base does not enforce a formatter, but follow the conventions already
present so diffs stay readable:

- **Units are cgs everywhere internally.** Convert user-facing units (Å, bar,
  km/s, planetary/solar/Jovian radii) only at the boundary — the setup wizard
  (`setup.py`) and the parsing in `prometheus.py`. Never let a non-cgs value
  reach a physics method.
- **Naming.** Classes use `CamelCase` (`HydrostaticAtmosphere`), methods and
  functions use `camelCase` (`calculateNumberDensity`, `getLOSopticalDepth_Batch`)
  to match the existing surface. Numba kernels are prefixed with `n_` or `_`
  (`n_interp_log`, `_bilinear_PT_interp`).
- **Vectorize, don't loop over chords.** The hot path operates on batched arrays
  of shape `(n_chords, n_x)` / `(n_chords, n_wav)`. New density or velocity
  methods must accept batched `phi/rho/orbphase` and broadcast against `x`, the
  same way `Planet.getDistanceFromPlanet` does. A per-chord Python loop on the
  optical-depth path is a regression.
- **Keep cgs constants in `constants.py`.** Don't hard-code physical constants
  inline.
- Docstrings are short and describe the physics / shape contract of the return
  value, not the implementation line by line.

---

## 3. Adding a new density scenario

A scenario is a number-density model. Decide which family it belongs to:

- **Collisional** (`CollisionalAtmosphere` base): carries `T` and `P_0`, density
  normalized by `n_0 = P_0/(k_B T)`. Supports continuum scattering with
  cloud-top pressure confinement.
- **Evaporative** (`EvaporativeExosphere` base): normalized by a particle number
  `N` or by mass continuity. Allows exactly one atomic/molecular absorber plus
  any number of scattering sources.

Then, in `pythonScripts/gasProperties.py`:

1. **Subclass the right base** and implement
   `calculateNumberDensity(self, x, phi, rho, orbphase)` returning an array of
   shape `(n_chords, n_x)` (and `(n_x,)` for scalar chord input). Use the
   batched distance helpers on `Planet` / `Moon`
   (`getDistanceFromPlanet`, `getDistanceFromMoon`, `getTorusCoords`) so
   broadcasting is handled for you.
2. **(Optional) position-dependent Doppler.** If the scenario has a bulk LOS
   velocity field that varies along `x` (like the radial wind), implement
   `calculateLOSVelocity(x_grid, phi_batch, rho_batch, orbphase_batch)`
   returning `(n_chords, n_x)`. The kernel will detect it and take the wind path
   (per-cell wavelength shift). Without it, the scenario uses only the per-chord
   bulk shift from the planet/moon.
3. **Wire up parsing** in `prometheus.py`: add a branch that recognizes the new
   scenario key in `Scenarios` and constructs your class with the parsed
   parameters. Convert units here.
4. **Wire up the wizard** in `setup.py`: add a `setupScenario` option and the
   numeric questions (with bounds and unit conversion) for its parameters so
   `setup` can produce a valid setup file.
5. **Document it**: add a row to the scenario tables in
   [architecture.md §2](architecture.md#2-object-model) and
   [modules.md](modules.md#gaspropertiespy), and the JSON example in
   [api.md](api.md#scenarios).

---

## 4. Adding a new opacity source

There are three constituent kinds. Pick the one that matches.

### Atomic / ionic line absorber
Add the species to `AvailableSpecies` in `constants.py` (name, element,
ionization stage, mass in g) and make sure `Resources/LineList.txt` contains its
lines (use `Resources/astroquery_retrieval.py` to pull and filter NIST lines).
No new class is needed — `AtmosphericConstituent` handles any species in the
list.

### Molecule
No code change is required to support a new molecule — supply its cross-section
table as `Resources/molecularResources/<MoleculeName>.h5` with datasets `p`,
`t`, `bin_edges`, and `xsecarr` (log-σ on a `(P, T, λ)` grid). Any species key
that is neither in `AvailableSpecies` nor in `SCATTERER_TYPES` is dispatched to
`MolecularConstituent`, which slices the table to the simulation window.

### Continuum scattering / aerosol
1. Subclass `ScatteringConstituent` in `gasProperties.py` and implement
   `calculateSigmaAbs(self, wavelength)` returning `σ(λ)` in cm² (wavelength-only;
   no Doppler shift is applied to continuum sources).
2. Register the key in `SCATTERER_TYPES` and add a case to
   `makeScatteringConstituent(scattererType, paramsDict)`.
3. If it should honor cloud-top confinement, accept a `P_top` argument (only
   meaningful on collisional hosts — it is ignored in evaporative contexts where
   no pressure is defined).
4. Add a `setup*` helper in `setup.py` and a parsing branch in `prometheus.py`,
   then document the parameters in [api.md](api.md#species) and the table in
   [architecture.md §2](architecture.md#constituent-classes-opacity-sources).

---

## 5. Validating a change

Because there is no test suite, validate empirically:

1. **Smoke-run the forward model.** Build a minimal setup file (or use
   `mainRetrieval.py`) and run

   ```bash
   python prometheus.py <name> --max-memory 2.0
   ```

   Confirm it completes and the printed max/min flux decrease is physically
   sane (e.g. `R ≤ 1`, absorption deepest near line centers).

2. **Check the single-chord path.** `Transit.evaluateChord(phi, rho, orbphase)`
   returns `(F_in, F_out)` and is the cheapest way to sanity-check a new
   scenario or constituent in isolation:

   ```python
   F_in, F_out = transit.evaluateChord(phi=0.0, rho=0.5 * star.R, orbphase=0.0)
   print(F_in / F_out)        # transmission for this chord
   ```

3. **Cross-check batched vs. single-chord.** A batched `sumOverChords` result
   must be consistent with summing `evaluateChord` over the same grid — this is
   the main correctness invariant when touching the kernel.

4. **Watch memory.** If you add intermediates on the optical-depth path, update
   `memoryHandler.estimate_chord_memory` so batch sizing still keeps peak RAM
   under `--max-memory`. Materializing a `(n_chords, n_x, n_wav)` tensor is the
   classic way to blow the budget — the kernel deliberately avoids it.

5. **Conservation checks.** For the radial wind, number density should follow
   mass continuity `n(r) = Mdot/(4π r² v(r) μ)`; verify a new velocity law gives
   a finite, positive density across `[r_inner, r_outer]`.

---

## 6. Branching and commits

- Branch off `main` with a short descriptive name
  (`feature/parker-wind`, `fix/clv-edge`, `docs/api-schema`).
- Keep commits focused; write imperative, descriptive messages
  (the existing history uses messages like
  *"Enhance memory estimation, add radial wind parameters, …"*).
- Do not commit large binaries. Molecular `.h5` tables, PHOENIX FITS files, and
  the `phoenix_cache/` directory are intentionally **not** tracked (see
  `.gitignore`); ship instructions for obtaining them instead of the files.
- Do not commit `__pycache__/`, virtual environments, or `output/` artifacts.

---

## 7. Pull-request process

1. Push your branch and open a PR against `main`.
2. In the description, state:
   - **what** physics/behavior changed and **why**,
   - the **setup file or script** you used to validate it (so a reviewer can
     reproduce),
   - a before/after of the relevant spectrum or light curve when the change
     affects results,
   - any change to memory characteristics or run time on the hot path.
3. Note any new dependency explicitly — the dependency list lives in
   [getting-started.md](getting-started.md#1-requirements), not in a packaged
   `requirements.txt` in the repo, so additions must be called out.
4. Because there is no CI, the reviewer reproduces your validation run. Make that
   easy: include the exact command and any small input files.

---

## 8. Where things live (quick map for contributors)

| If you want to change… | Edit |
|---|---|
| A physical constant or supported atom/ion | `pythonScripts/constants.py` |
| Star / planet / moon geometry or stellar treatment | `pythonScripts/celestialBodies.py` |
| The spatial / phase grid | `pythonScripts/geometryHandler.py` |
| A density scenario, opacity source, or the kernel | `pythonScripts/gasProperties.py` |
| Chord-batch memory sizing | `pythonScripts/memoryHandler.py` |
| The interactive setup wizard | `pythonScripts/setup.py` |
| Shot-noise post-processing | `pythonScripts/shotNoise.py` |
| Setup-file parsing / the CLI | `prometheus.py` |
| Planet/star tables or the line list | `Resources/` |
