# Architecture

This page describes how Prometheus is organized and how a transmission spectrum is computed, from the planet object through to the final flux ratio.

---

## Modules

All code lives in the `core` package:

| Module | Responsibility |
|---|---|
| `constants.py` | Physical constants (cgs), `Species`, and `AvailableSpecies` (the atomic/ionic catalog with masses and ionization states). |
| `celestialBodies.py` | `Star`, `Planet`, `Moon`, and `AvailablePlanets` (loads the planet/star catalog from `Resources/*.csv`). Handles orbital positions, line-of-sight velocities, limb darkening, Rossiter–McLaughlin rotation, and PHOENIX stellar spectra. |
| `geometryHandler.py` | `Grid` — the spatial/temporal discretization: the line-of-sight axis `x`, the sky-plane polar coordinates `(rho, phi)`, and the orbital-phase axis. |
| `gasProperties.py` | The bulk of the physics: density models (atmospheres and exospheres), absorber/scatterer constituents, the `Atmosphere` aggregator, the `WavelengthGrid`, the Numba optical-depth kernels, and the `Transit` orchestrator. |
| `emission.py` | Emission physics: the resonance-scattering and thermal source functions, geometric dilution, the absolute stellar illumination, the `EmissionModel` switchboard, and the `g_factor` diagnostic. |
| `eclipse.py` | Secondary-eclipse geometry: a planet-centred ray grid, an opaque dayside boundary, and the `F_planet / F_star` reduction. |
| `memoryHandler.py` | Memory-aware batching: estimates per-chord memory and picks a chunk size that fits within a RAM budget. |

The example figure scripts (`fig*.py`) import only from `core.*` for the Prometheus physics; some additionally use the separate `mnemosyne`/`dishoom` packages, which sit on top of Prometheus and are not part of this repository.

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
              Atmosphere([models], hasOrbitalDopplerShift,
                         emission=EmissionModel(...))   # optional
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


## Emission

By default Prometheus solves pure extinction: the only thing gas does to a
chord is remove photons from it. Passing an `emission.EmissionModel` to
`Atmosphere` switches to the full formal solution, in which gas also *adds*
photons:

```
I = I_star · exp(-tau_total)  +  Σ_i  j_i · Δx · exp(-tau_(i→obs))
```

`Transit` still owns the first term (it carries the limb darkening and the
stellar rotation); `Atmosphere.getLOSopticalDepthAndEmission_Batch` returns
`tau_total` and the sum.

### Source functions

**Resonance scattering of starlight** — the exomoon-cloud term. An atom absorbs
a stellar photon and re-radiates it isotropically; the fraction landing in the
observer's beam is set by how much of the sky the star covers at the parcel:

```
j_scat(λ) = n_abs · sigma(λ) · W(r) · I_star(λ_star)
W(r) = Omega_star / 4π = 0.5 · (1 - sqrt(1 - (R_star/r)²))    → R_star²/4r²  for r ≫ R_star
```

`sigma(λ)` is *the same cross section the extinction path uses* — the same line
list, oscillator strengths, Voigt profile and Doppler shifts — so the emission
can never be inconsistent with the absorption, and no separate emission line
data is needed.

`λ_star` is the wavelength the parcel samples the stellar spectrum at, i.e. the
parcel-frame wavelength Doppler-shifted by the parcel's **radial velocity with
respect to the star** (`EmissionModel.stellar_doppler`, on by default). This is
what makes gas sitting in a deep stellar Fraunhofer core scatter weakly, while
gas that has Doppler-shifted out of the core sees the full continuum and
scatters strongly. Wind models supply that velocity through
`calculateRadialVelocityFromStar`; other models are assumed to move
perpendicular to the star direction (exact for a circular orbit).

**Thermal (LTE) emission** — `j_therm(λ) = n_abs · sigma(λ) · B_λ(T)`,
Kirchhoff's law, for density models that carry a temperature.

**Aerosol scattering** — same isotropic single-scattering form. The albedo
splits aerosol extinction into a scattering share, which redirects starlight,
and a true-absorption share, which emits thermally; Kirchhoff's law then holds
for every constituent rather than only for the line opacity. The default
albedo of 1 is a pure scatterer that neither absorbs nor emits. The scattering
term itself is off by default: real aerosols are strongly forward-scattering,
so the isotropic assumption is much weaker here than for a resonance line.

### Why the cell loop is inverted

`getLOSopticalDepthAndEmission_Batch` puts the loop over line-of-sight cells on
the **outside**, unlike `getLOSopticalDepth_Batch`. It has to: the attenuation
applied to a cell's emission is the optical depth between that cell and the
observer summed over *every* species, so all constituents' per-cell
contributions must be in hand at the same time. Everything stays vectorised
over chords and wavelength, so the peak allocation is still `(n_chords, n_wav)`
and never the `(chord, x, wavelength)` tensor. The trade-off is that the
factored `column × sigma` fast path for atoms is unavailable, so an emission run
costs roughly what a wind run costs. Each cell's emission is integrated exactly for a source function that is
constant across the cell, `S · (1 - exp(-dtau))`, rather than with a midpoint
weight. The two agree while cells are thin, but only the exact form survives a
cell becoming optically thick — which is what makes an opaque layer in LTE
re-emit exactly what it absorbs. That is checked directly in
`Tests/EmissionCalibration`, where an opaque grey layer at the surface
temperature leaves the eclipse depth unchanged at every optical depth from 0.01
to 99 and at every grid resolution.

### Absolute units

Emissivity is an absolute number of erg, so it cannot be compared against the
flat `Fstar = 1` placeholder the transmission path is free to use.
`emission.StellarIntensity` resolves this: it returns the PHOENIX surface
intensity when one is attached, and a blackbody at `T_eff` otherwise (an
explicit modelling assumption), plus the `internal_scale` factor that converts
a physical cgs intensity back into whatever units `F_out` is accumulated in.
When emission is active, `run_transit` also widens the retained PHOENIX
wavelength window by `illumination_velocity` (default 100 km/s), because the
spectrum is now sampled in the *parcel's* frame rather than the observer's.

### Geometry

Two things change in `Transit.sumOverChords`:

- Chords outside the stellar limb are given zero photospheric flux. With the
  default `rho_border = R_star` every chord is on-disk and this is a no-op; it
  matters once the sky-plane grid is widened past the limb to capture the
  off-limb glow of a cloud seen against the dark sky.
- Chords blocked by an opaque body are still integrated. The body hides only
  the gas *behind* it, so `x_block` (the body's `x` coordinate) truncates the
  emission integral while the transmitted term stays zero.

`R = Σ F_in / Σ F_out` can therefore exceed 1 where the gas puts back more light
than it removes. `sumOverChords(return_components=True)` splits `R` into its
`transmission` and `emission` parts, and `TransitResult.fill_in_fraction()`
reports what fraction of the pure-extinction line absorption the scattered
photons refill.

### Stated approximations

Single scattering; isotropic phase function; coherent scattering in the
observer frame (exact in the forward-scattering limit, i.e. exactly the
in-transit geometry); no self-shielding of the incident stellar beam. All four
are documented at the top of `core/emission.py` with the regime each is valid
in. Multiple scattering would only *increase* the emission, so the answer here
is a lower bound on the fill-in.


## Secondary eclipses

`emission.py` supplies source functions; `eclipse.py` supplies the other
observable. A transit integrates chords over a *star*-centred sky plane and
returns a ratio bounded by 1. An eclipse integrates rays over a *planet*-centred
disk and returns

```
depth(λ) = ∫_planet I_p(λ) dA / (π R_star² · I_star(λ))
```

Each ray at impact parameter `b` carries
`I = epsilon · B_λ(T_surf(b)) · exp(-tau_above) + I_em`, where `tau_above` is
the optical depth between the surface and the observer — supplied by the
emission kernel's `return_visible_tau` — and `I_em` is the overlying gas's own
emission. Rays with `b > R_p` miss the solid body and see only the limb. For a
bare uniform dayside this reduces to `epsilon · (R_p/R_star)² · B_λ(T)/I_star`,
which the quadrature reproduces to ~1e-15.

Rays are placed at orbital phase `pi`, where Prometheus' frame puts the planet
at `x = -a` with zero line-of-sight velocity — the right kinematics and the
right star-planet distance for the illumination term. The star enters only
through the denominator.

The dayside temperature is an **input**: there is no energy-balance or
heat-redistribution solver. `DaysideSurface` offers a uniform disk (which is
exactly what a measured brightness temperature means) or the
instantaneous-reradiation profile `T = T_sub · cos(theta)**0.25`. The ingress
and egress light curve is not modelled.

### Stellar spectra beyond PHOENIX

The PHOENIX HiRes grid Prometheus downloads stops near 5.5 µm, so it cannot
supply the denominator for any mid-infrared work.
`Star.addFstarFunctionFromArrays(wavelength, intensity)` attaches an arbitrary
spectrum instead — a grid that reaches further (BT-Settl, ATLAS9) or a measured
one. This matters more than it sounds: a blackbody at `T_eff` over-predicts a
G8V photosphere's 6–12 µm surface brightness by ~13%, because that continuum
forms high in the atmosphere where the gas is cooler than `T_eff`. Since
eclipse depth scales as `1/I_star`, using the blackbody fallback inflates a
recovered brightness temperature by nearly 200 K. Do not use it for mid-IR
emission work.

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
