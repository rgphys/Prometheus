# Architecture

This document describes how Prometheus is structured internally: the coordinate
system and geometry, the object model, the data flow from setup file to output,
the vectorized optical-depth kernel that does the heavy lifting, and the
key design decisions and their trade-offs.

All quantities are in **cgs units** internally. User-facing units (Å, bar,
km/s, Jovian/solar/Io radii, …) are converted at the setup boundary.

---

## 1. Geometry and coordinate system

Prometheus ray-traces the stellar disk. The coordinate frame
(defined in `geometryHandler.Grid`) is:

- The **observer** sits at `x = −∞`.
- The **star** is centered at the origin `(0, 0, 0)`.
- The **x-axis** is the line of sight, pointing from the observer through the
  star.
- The **y–z plane** is the plane of the sky.
- A sky-plane point is described in polar coordinates `(rho, phi)`:
  `y = rho·sin(phi)`, `z = rho·cos(phi)`.

A **chord** is a single line of sight: a fixed `(rho, phi)` and orbital phase,
integrated along `x`. The full simulation evaluates a Cartesian product of
`phi × rho × orbital phase`, flattened by `Grid.getChordGrid()` into an
`(N_chord, 3)` array of `(phi, rho, orbphase)` rows.

### The four discretization axes

| Axis | Meaning | Built by | Step counts |
|---|---|---|---|
| `x` | line-of-sight integration coordinate | `constructXaxis` | `x_steps`, half-length `x_border`, centered on `x_midpoint` (= planet semi-major axis `a`) |
| `rho` | sky-plane radius (0 → stellar radius) | `constructRhoAxis` | `rho_steps`, upper bound `rho_border` (= `R_star`) |
| `phi` | sky-plane azimuth (0 → 2π) | `constructPhiAxis` | `phi_steps` |
| orbphase | orbital phase (−border → +border) | `constructOrbphaseAxis` | `orbphase_steps`, in radians (centered on mid-transit = 0) |

All spatial axes use **cell-midpoint** sampling by default (the integration
treats each cell as a rectangle of width `delta_x`, `delta_rho`, `delta_phi`).

### Body positions

`celestialBodies.Planet.getPosition(orbphase)` places the planet on a circular,
edge-on orbit: `x_p = a·cos(orbphase)`, `y_p = a·sin(orbphase)`. The
line-of-sight velocity is `v_los = −sin(orbphase)·√(G·M_star/a)` (positive =
receding = redshift). A `Moon` orbits its planet with a period ratio derived
from Kepler's third law (`getOrbphase`) and adds its orbital velocity to the
planet's.

`getDistanceFromPlanet`, `getDistanceFromMoon`, and `getTorusCoords` all support
**batched broadcasting**: given `x` of shape `(n_x,)` and `phi/rho/orbphase` of
shape `(n_chords,)`, they return arrays of shape `(n_chords, n_x)`. This is the
backbone of the vectorized density evaluation.

---

## 2. Object model

```
Transit
 ├── Atmosphere
 │    └── densityDistributionList: [ <scenario>, ... ]
 │         each scenario (e.g. HydrostaticAtmosphere, RadialWindExosphere, ...)
 │           ├── planet  (and .moon for moon-sourced scenarios)
 │           └── constituents: [ AtmosphericConstituent | MolecularConstituent | ScatteringConstituent, ... ]
 ├── WavelengthGrid
 └── spatialGrid: geometryHandler.Grid
       └── planet.hostStar: Star
```

### Scenario classes (number-density models)

Two families, distinguished by whether they carry a temperature:

**Collisional** (`CollisionalAtmosphere` base, has `T` and `P_0`):
`BarometricAtmosphere`, `HydrostaticAtmosphere`, `PowerLawAtmosphere`. Density
is normalized by a reference number density `n_0 = P_0 / (k_B T)`. Only these
support continuum scattering with cloud-top pressure confinement.

**Evaporative** (`EvaporativeExosphere` base, normalized by particle number `N`
or by mass continuity): `PowerLawExosphere`, `MoonExosphere`,
`TidallyHeatedMoon`, `TorusExosphere`, `SerpensExosphere`,
`RadialWindExosphere`. These represent tenuous exospheres.

Every scenario implements `calculateNumberDensity(x, phi, rho, orbphase)`
returning `(n_chords, n_x)` (or `(n_x,)` for scalar input). Some also implement
`calculateLOSVelocity(...)` (only `RadialWindExosphere`) for
position-dependent Doppler shifts.

### Constituent classes (opacity sources)

| Class | `isMolecule` | `isScatterer` | Cross-section source |
|---|---|---|---|
| `AtmosphericConstituent` | False | — | Voigt profiles summed over NIST lines, cached as `log10 σ(λ)` interpolant |
| `MolecularConstituent` | True | — | HDF5 table `σ(P, T, λ)`, decomposed bilinear-PT + 1-D-λ interpolation |
| `ScatteringConstituent` (subclasses below) | False | True | Analytic or tabulated `σ(λ)`, no Doppler shift |

Scattering subclasses: `RayleighHaze`, `GrayCloud`, `PowerLawAerosol`,
`TabulatedAerosol`. The reserved keys are in `SCATTERER_TYPES`.

---

## 3. Data flow

### Setup → objects (`prometheus.py`)

1. Load the JSON setup file into `Fundamentals / Scenarios / Architecture /
   Species / Grids` dictionaries.
2. `bodies.AvailablePlanets().findPlanet(name)` looks up the planet (and its
   host star) from `planets.csv` / `stars.csv`.
3. Build the `WavelengthGrid` and the spatial `Grid`.
4. For each scenario key, instantiate the matching scenario class.
5. For each scenario, add its constituents. The dispatch is by key:
   - name in `AvailableSpecies` → atomic/ionic `AtmosphericConstituent`
     (collisional gets a mixing ratio `chi`; evaporative gets `sigma_v`),
   - name in `SCATTERER_TYPES` → `ScatteringConstituent` via
     `makeScatteringConstituent`,
   - otherwise → `MolecularConstituent` (collisional uses `chi`; evaporative
     uses a pseudo-temperature `T`).
   Each constituent then builds its wavelength-sliced lookup via
   `addLookupFunctionToConstituent(wavelengthGrid)`.
6. Wrap the scenarios in an `Atmosphere(scenarioList, DopplerOrbitalMotion)` and
   then a `Transit`.

### Objects → result (`Transit.sumOverChords`)

1. Flatten the `(phi, rho, orbphase)` chord grid.
2. Precompute, per chord: CLV factor, stellar-rotation Doppler shift, and the
   orbital-phase bin index.
3. Estimate an **optimal batch size** (`memoryHandler.calculate_optimal_chunk_size`)
   from the RAM budget, the wavelength count, `n_x`, and whether the heavy
   (molecular) path is active.
4. For each batch of chords:
   - Compute the (possibly flat) stellar flux `F_star_batch` and apply CLV.
   - Determine which chords are **blocked** by the opaque planet/moon disk.
   - For active (unblocked) chords, call
     `Atmosphere.getLOSopticalDepth_Batch(...)` to get `τ` of shape
     `(n_active, n_wav)`.
   - Apply Beer–Lambert: `F_in = F_out · exp(−τ)`, with `F_out = rho · F_star`.
   - Accumulate `F_in` and `F_out` into per-orbital-phase sums (the `rho`
     weighting is the polar-coordinate area element).
5. Return `R = F_in_sum / F_out_sum` of shape `(n_orbphase, n_wav)`.

`evaluateChord` is a single-chord convenience wrapper around the same kernel,
used for testing and by retrieval scripts.

---

## 4. The optical-depth kernel

`Atmosphere.getLOSopticalDepth_Batch(x_grid, phi_batch, rho_batch,
orbphase_batch, wavelength, delta_x)` is the computational core. It is **fully
vectorized — there is no Python loop over individual chords.** It iterates over
scenarios and, within each scenario, over constituents, accumulating
`total_tau` of shape `(n_chords, n_wav)`.

For each scenario it computes once:
- the bulk per-chord LOS velocity `v_bulk` (planet or moon),
- the per-chord Doppler-shifted wavelength grid `shifted_wav` (`n_chords, n_wav`),
- for wind models with `calculateLOSVelocity`, the per-(chord, x) total velocity
  field and its Doppler shift factor `shifts_field` (`n_chords, n_x`),
- the number density `n_tot` of shape `(n_chords, n_x)`.

Then per constituent, one of three branches runs:

**Atomic / ionic** (`AtmosphericConstituent`):
- *Fast path* (no wind field): factorize as column density × cross-section.
  Compute `col_density = Σ_x n·chi·delta_x` per chord, look up the
  cross-section once per **unique** bulk Doppler shift (cached by a
  `(shift, wavelength)` key), and broadcast back. This exploits the fact that
  the cross-section depends only on the per-chord shift, not on `x`.
- *Wind path* (`shifts_field` present): loop over `x` and, for each step,
  shift the wavelength grid by that cell's velocity and look up the
  cross-section — accumulating `n_abs·σ·delta_x`. This applies the
  position-dependent wind Doppler shift without ever materializing the full
  `(n_chords, n_x, n_wav)` tensor.

**Molecular** (`MolecularConstituent`):
- Compute pressure `P = n_tot·k_B·T` (clamped) per `(chord, x)`.
- Loop over `x`; at each step do a **bilinear interpolation over (P, T)** on the
  table's native wavelength grid (`_bilinear_PT_interp`, Numba), and accumulate
  the abundance-weighted column `sigma_eff` on that native grid.
- After the loop, do a single **1-D interpolation** (`n_interp_linear_rows`,
  Numba) from the native grid onto the Doppler-shifted wavelength grid.
- This "decomposed" scheme replaces a full 3-D scattered `RegularGridInterpolator`
  evaluation over `(n_chords·n_x·n_wav)` points and avoids the `(C, X, W)`
  tensor.

**Scattering / aerosol** (`ScatteringConstituent`):
- Cross-section is wavelength-only; no Doppler shift.
- `col_density = Σ_x n·chi·delta_x` per chord (optionally masked to cells where
  `P ≥ P_top` for cloud-top confinement, requires a temperature-bearing host).
- `tau += col_density ⊗ σ(λ)`.

### Numba kernels

| Kernel | Role |
|---|---|
| `n_interp_log` | Row-wise 1-D interpolation of `log10 σ(λ)` with a two-pointer O(N+M) scan; returns `10^(log) − offset`. Used for atomic cross-sections and the (flat-or-PHOENIX) stellar flux. Requires monotonic target rows — true for all Doppler-shifted grids. |
| `n_interp_linear_rows` | Row-wise 1-D interpolation in **linear** space (per-row `y`); maps the molecular `sigma_eff` from native to shifted grid. |
| `_bilinear_PT_interp` | Bilinear (P, T) interpolation on the molecular log-σ table at fixed `T`, returning σ on the native wavelength grid. |

All three are `@njit(parallel=True, fastmath=True)` and parallelize over rows
(chords / points) with `prange`.

---

## 5. Memory-aware batching

`memoryHandler` sizes chord batches so peak RAM stays under the budget
(`--max-memory`, default 2 GB; `None` → 50 % of available RAM via `psutil`).

- `estimate_chord_memory(num_wavelengths, n_x, is_molecular)` budgets
  ~640 bytes/output-wavelength for the molecular path (dominated by
  native-grid intermediates) and 16 bytes/wavelength for the atomic path, with a
  2× overhead buffer.
- `calculate_optimal_chunk_size(...)` divides the available bytes by the
  per-chord estimate, clamped to `[1, total_chords]`.

When the whole grid fits in one batch, `sumOverChords` reshapes and sums
directly; otherwise it uses `np.add.at` scatter-adds keyed by the orbital-phase
bin index.

---

## 6. Spectral lines, wavelength grid, and Doppler shifts

- **Line list**: `LineList.txt` (tab-separated NIST export) provides element,
  ionization stage, vacuum wavelength [Å], Einstein `A_ki`, and oscillator
  strength `f`. `AtmosphericConstituent.getLineParameters` filters lines to the
  species and wavelength window; the damping parameter is `γ = A_ki / (4π)`.
- **Voigt profile**: `calculateVoigtProfile` sums
  `π e²/(m_e c) · f · V(...)` over lines, with Gaussian width set by the thermal
  velocity dispersion `sigma_v` and Lorentzian width `γ`. It is evaluated on a
  refined grid (10× finer, 1 % extended) and cached as a `log10 σ` interpolant
  for speed and to avoid log-of-zero (`lookupOffset = 1e-50`).
- **Adaptive wavelength grid**: `WavelengthGrid.arangeWavelengthGrid` builds a
  non-uniform grid — fine resolution (`resolutionHigh`) in `widthHighRes`-wide
  windows around each line center, coarse (`resolutionLow`) elsewhere. Molecular
  and scattering opacities are smooth and do not influence grid construction.
- **Doppler shifts**: the relativistic factor
  `√((1−β)/(1+β))` (`constants.calculateDopplerShift`) is applied to the
  wavelength grid. Bulk orbital motion gives one shift per chord; the radial
  wind additionally gives a per-cell shift via `calculateLOSVelocity`. Stellar
  rotation (RM) shifts the stellar flux per chord.

---

## 7. Stellar treatment

`celestialBodies.Star`:
- **Flat star** (default when no spectrum is loaded): `F = 1` everywhere; only
  the geometric `rho` weighting and CLV apply.
- **PHOENIX spectrum**: `getSpectrum` rounds `(T_eff, log g, [Fe/H], α)` to the
  PHOENIX grid, downloads and caches the FITS files, and builds a `log10 F(λ)`
  interpolant restricted to the simulation's wavelength window (± rotational
  broadening).
- **CLV**: quadratic limb darkening `1 − u1·(1−μ) − u2·(1−μ)²`.
- **RM effect**: surface LOS velocity `v·sin(i)·(rho/R)·cos(phi − phi_rot)`
  Doppler-shifts the local stellar flux. With rotation off, the integrated
  stellar flux has a closed form; with rotation on it is summed over the disk.

---

## 8. Radial wind details

`RadialWindExosphere` supports two velocity laws (`wind_model`):

- **`'beta'`** — a modified beta law with a finite launch speed:
  `v(r) = v_base + (v_terminal − v_base)·max(1 − r_inner/r, 0)^beta`. The finite
  `v_base` (default `v_terminal·1e-3`) removes the density divergence a pure
  beta law would create at `r_inner` through mass continuity.
- **`'parker'`** — the exact isothermal Parker-wind transonic solution in closed
  form via the Lambert-W function. The dimensionless relation
  `(v/c_s)² − ln[(v/c_s)²] = 4 ln(r/r_c) + 4 r_c/r − 3` is solved as
  `y = −W_b(−e^{−D})`, selecting the principal branch (subsonic, `r < r_c`) or
  the `W₋₁` branch (supersonic, `r > r_c`). A trace heavy species can be
  *advected* by the bulk light gas by passing `wind_mu` (sets the dynamics)
  separately from `mu` (sets the density normalization).

In both cases the number density follows mass continuity
`n(r) = Mdot / (4π r² v(r) μ)`, masked to `[r_inner, r_outer]`.

---

## 9. Design decisions and trade-offs

- **Vectorization over multiprocessing.** The hot path uses NumPy broadcasting
  and Numba JIT kernels, not `multiprocessing`. `import multiprocessing` appears
  in the codebase but is not used on the active computation path — it is a
  legacy artifact.
- **Extinction, not scattering.** All scattering/aerosol sources contribute
  extinction to the Beer–Lambert optical depth (photons scattered out of the
  beam are lost). There is **no scattering phase function and no multiple
  scattering**. Because continuum cross-sections are smooth, no Doppler shift is
  applied to them.
- **Decomposed molecular interpolation.** Splitting the 3-D `σ(P, T, λ)` lookup
  into a per-x bilinear (P, T) step plus a single 1-D λ remap avoids
  materializing the `(chord, x, wavelength)` tensor and is the dominant speedup
  for narrow grids (the table is also sliced to the simulation window ± a 1 %
  Doppler margin).
- **Caching.** Atomic cross-sections are cached per unique bulk Doppler shift,
  and the Voigt-profile interpolant is precomputed once per species. The
  `lookupOffset = 1e-50` lets cross-sections be stored and interpolated in
  log space without log-of-zero.
- **Cloud-top confinement is collisional-only.** `P_top` confinement for
  `GrayCloud` / `PowerLawAerosol` requires a temperature-bearing host scenario;
  it is ignored in evaporative exosphere contexts (no pressure is defined).
