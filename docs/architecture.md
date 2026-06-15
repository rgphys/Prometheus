# Architecture

This page describes how Prometheus is organized and how a transmission spectrum is computed, from the planet object through to the final flux ratio.

---

## Modules

All code lives in the `pythonScripts` package:

| Module | Responsibility |
|---|---|
| `constants.py` | Physical constants (cgs), `Species`, and `AvailableSpecies` (the atomic/ionic catalog with masses and ionization states). |
| `celestialBodies.py` | `Star`, `Planet`, `Moon`, and `AvailablePlanets` (loads the planet/star catalog from `Resources/*.csv`). Handles orbital positions, line-of-sight velocities, limb darkening, Rossiter–McLaughlin rotation, and PHOENIX stellar spectra. |
| `geometryHandler.py` | `Grid` — the spatial/temporal discretization: the line-of-sight axis `x`, the sky-plane polar coordinates `(rho, phi)`, and the orbital-phase axis. |
| `gasProperties.py` | The bulk of the physics: density models (atmospheres and exospheres), absorber/scatterer constituents, the `Atmosphere` aggregator, the `WavelengthGrid`, the Numba optical-depth kernels, and the `Transit` orchestrator. |
| `memoryHandler.py` | Memory-aware batching: estimates per-chord memory and picks a chunk size that fits within a RAM budget. |

The example figure scripts (`fig*.py`) import only from `pythonScripts.*` for the Prometheus physics; some additionally use the separate `mnemosyne`/`dishoom` packages, which sit on top of Prometheus and are not part of this repository.

---

## Object model and data flow

A simulation is built bottom-up from independent objects, then run:

```
                Planet  ──┐  (carries hostStar: Star)
                          │
  density model(s) ───────┤   HydrostaticAtmosphere / TorusExosphere /
   + constituents         │   RadialWindExosphere / ...
   (atoms, molecules,     │     .addConstituent(...) / .addMolecularConstituent(...)
    scatterers)           │     .addScatteringConstituent(...)
                          │     constituent.addLookupFunctionToConstituent(wg)
                          ▼
              Atmosphere([models], hasOrbitalDopplerShift)
                          │
   WavelengthGrid ────────┤
   Grid (geometry) ───────┤
                          ▼
                       Transit
                          │  .addWavelength()      → builds Transit.wavelength
                          │  .sumOverChords(...)    → R(orbphase, wavelength)
                          ▼
                  R = Σ F_in / Σ F_out
```

The flow:

1. **Planet → density model.** A density model (e.g. `HydrostaticAtmosphere`) holds a reference to its `Planet` and knows how to return a number density `n(x, phi, rho, orbphase)` at any point.
2. **Constituents → density model.** Each absorbing/scattering species is attached to a density model. Atomic and molecular constituents must have their opacity lookup precomputed via `addLookupFunctionToConstituent`.
3. **Density models → `Atmosphere`.** One or more density models are wrapped in an `Atmosphere`, which also carries the global `hasOrbitalDopplerShift` flag and owns the optical-depth computation.
4. **`WavelengthGrid`.** Stores the sampling parameters. `Transit.addWavelength()` invokes `WavelengthGrid.constructWavelengthGrid`, which scans every atomic line in range and produces a non-uniform grid: fine near lines, coarse elsewhere.
5. **`Grid`.** Defines the chords. `getChordGrid()` returns the flattened set of `(phi, rho, orbphase)` triples to evaluate.
6. **`Transit.sumOverChords`.** Batches the chords, computes optical depth per batch, applies Beer–Lambert attenuation against the (optionally PHOENIX) stellar surface flux, and accumulates `F_in`/`F_out` per orbital phase.

---

## Geometry and coordinate system

`geometryHandler.Grid` uses a star-centered frame (defined in its docstring):

- The observer is at `x = -∞`; the star sits at the origin.
- The `x`-axis is the line of sight through the star's center; the `y`–`z` plane is the sky plane.
- `rho` is the radial distance from the origin in the sky plane; `phi` is the azimuthal angle.

The integration chord runs along `x`, centered at `x_midpoint` (typically `planet.a`) with half-length `x_border` and `x_steps` cells. The sky-plane sampling spans `rho ∈ [0, rho_border]` (usually out to the stellar radius) with `rho_steps × phi_steps` chords. The orbital-phase axis spans `[-orbphase_border, +orbphase_border]` with `orbphase_steps` points; a single mid-transit spectrum uses `orbphase_border=0, orbphase_steps=1`, while a light curve or phase-resolved map uses many phases.

`getChordGrid()` builds the Cartesian product of these axes into a flat `(N_chords, 3)` array of `(phi, rho, orbphase)`.

---

## The optical-depth kernel

The heart of the code is `Atmosphere.getLOSopticalDepth_Batch(x_grid, phi_batch, rho_batch, orbphase_batch, wavelength, delta_x)`. It is **fully vectorized over chords** — there is no Python loop over individual lines of sight. It returns optical depth of shape `(n_chords, n_wav)` and is built around three constituent paths.

For each density model in the atmosphere it first computes:

- the per-chord bulk Doppler shift from the planet's (or moon's) orbital line-of-sight velocity, applied to the shared wavelength grid → `shifted_wav` of shape `(n_chords, n_wav)`;
- for wind models (those exposing `calculateLOSVelocity`), an additional per-cell velocity field `(n_chords, n_x)` that produces a position-dependent Doppler shift;
- the number density `n_tot` of shape `(n_chords, n_x)` via the model's vectorized `calculateNumberDensity`.

Then, per constituent:

**Atomic / ionic absorbers.**
The cross section is a sum of Voigt line profiles, precomputed once on a refined grid and stored as a `log10(sigma)` interpolator. In the common case (a single bulk Doppler shift per chord), the column density `Σ n · χ · Δx` is factored out and the cross section is evaluated only on the *unique* shifted grids, with the results cached on the constituent (`_batch_sigma_cache`) — so repeated phases reuse the same cross sections. When a wind velocity field is present, the code instead loops over the `x`-axis and expands only a `(n_chords, n_wav)` slice at a time, so it never materializes the full `(chord, x, wavelength)` tensor.

**Molecular absorbers.**
Cross sections are tabulated on a `(pressure, temperature, wavelength)` grid read from HDF5. Rather than a single 3-D interpolation over the whole tensor, the molecular path is **decomposed** (this is the main optimization over the original code):

1. For each `x`-cell, a Numba-compiled **bilinear interpolation in `(P, T)`** (`_bilinear_PT_interp`) evaluates the cross section on the molecule's *native* wavelength grid for all chords at once.
2. The weighted column `Σ_x n·χ·σ` is accumulated on that native grid.
3. A single Numba 1-D interpolation (`n_interp_linear_rows`) maps the accumulated column from the native grid onto the per-chord Doppler-shifted grid.

This avoids building the `(chord, x, wavelength)` tensor and replaces the expensive `RegularGridInterpolator` call in the inner loop. The native cross-section table is also sliced to the simulation wavelength range (plus a ~1% Doppler margin) at load time, shrinking the per-chord kernel from tens of thousands of wavelength points to `O(N_sim)`.

**Continuum scatterers / aerosols.**
The extinction cross section depends only on wavelength (no Doppler shift). The column density `Σ n·χ·Δx` is multiplied by `sigma(wavelength)` and added to the optical depth. Cloud-top confinement (`P_top`) zeroes the contribution where the local pressure is below the threshold, for temperature-bearing (collisional) host models.

### The Numba interpolation kernels

Two JIT kernels do the heavy interpolation, both exploiting the fact that Doppler-shifted wavelength grids are monotonic:

- **`n_interp_log(x_targets, x_grid, y_grid_log, offset)`** — interpolates `log10(sigma)` and returns `10**value - offset`. Uses a two-pointer scan that is `O(N + M)` per row instead of a binary search per point, and runs in parallel over rows (`prange`). Used for atomic cross sections and for resampling the PHOENIX stellar spectrum.
- **`n_interp_linear_rows(x_targets, x_grid, y_grid_2d)`** — row-wise linear interpolation in linear space, used to map the accumulated molecular column onto each chord's shifted grid.

Both are decorated `@njit(parallel=True, fastmath=True)`.

---

## Memory-aware batching

`Transit.sumOverChords` does not evaluate all chords at once. It asks `memoryHandler.calculate_optimal_chunk_size` for a batch size given a RAM budget (`max_memory_gb`):

- `get_available_memory` clamps the requested budget to the actually-available system RAM (via `psutil`); passing `max_memory_gb=None` uses 50% of available RAM.
- `estimate_chord_memory` estimates bytes per chord. The molecular path is the heavy one (it allocates native-wavelength-grid intermediates per chord, budgeted at ~640 bytes per output wavelength point); the atomic path is light (~16 bytes per wavelength point). A 2× buffer covers Python/NumPy overhead.
- The chord grid is then processed in slices of that size.

For each batch, the code: resamples the stellar surface flux (flat or PHOENIX, optionally Doppler-shifted by stellar rotation and scaled by limb darkening), masks chords blocked by the opaque planet/moon disk, computes optical depth for the unblocked chords, applies `F_in = F_out · exp(-τ)`, and accumulates `F_in`/`F_out` into per-orbital-phase sums via `np.add.at`. The returned `R = F_in_sum / F_out_sum` has shape `(orbphase_steps, n_wavelength)`.

This keeps peak memory bounded regardless of how finely the spatial and wavelength grids are sampled, which is what makes high-resolution, large-grid runs feasible.

---

## Radial-wind physics

`RadialWindExosphere` models an escaping, radially expanding outflow. The density follows from mass continuity,

```
n(r) = Mdot / (4π r² v(r) μ),
```

with two selectable velocity laws (`wind_model`):

- **`'beta'`** — a modified β-law, `v(r) = v_base + (v_terminal - v_base)·max(1 - r_inner/r, 0)^β`. The finite launch speed `v_base` (default `1e-3 · v_terminal`) represents a subsonic wind base and removes the unphysical density divergence a pure β-law produces as `v → 0` at `r_inner`. Setting `β ≈ 1` with `v_terminal` near the sound speed gives a first-order approximation to a thermally driven wind.
- **`'parker'`** — the **exact isothermal Parker wind** transonic solution, obtained in closed form via the Lambert-W function (`scipy.special.lambertw`). It has no free `β`/`v_terminal` knobs: the profile is fixed by the wind temperature `T` and the planet mass through the sound speed `c_s = sqrt(k_B T / wind_mu)` and the sonic radius `r_c = G M wind_mu / (2 k_B T)`. `Mdot` sets only the density normalization.

A key feature is the **tracer/bulk separation** for the Parker model. A heavy trace species (e.g. Na) does not drive its own transonic wind, so its dynamics are set by the *bulk* light gas: pass `wind_mu` (the mean particle mass of the H/He outflow) to fix the velocity profile, while `mu` (the tracer mass) sets only the continuity normalization. When `wind_mu` is omitted, the dynamics use `mu` (a self-driven wind).

When orbital Doppler shifting is enabled, `RadialWindExosphere.calculateLOSVelocity` is called automatically by the optical-depth kernel to apply a **position-dependent** Doppler shift along each chord. The near (approaching) and far (receding) faces of the outflow produce a blue/red asymmetry and kinematic broadening that a single bulk shift cannot capture — this is the physics exercised in `fig6`.

---

## Where to go next

- **[api-reference.md](api-reference.md)** — exact constructor signatures and methods.
- **[examples.md](examples.md)** — runnable end-to-end scripts.
