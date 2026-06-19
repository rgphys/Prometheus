# API reference

All public classes live in the `pythonScripts` package. Constructor signatures and argument names below are taken directly from the source. Units are **cgs** throughout: wavelengths in cm, pressures in barye, lengths in cm, velocities in cm/s, masses in g, temperatures in K.

```python
import pythonScripts.gasProperties as gasprop
import pythonScripts.celestialBodies as bodies
import pythonScripts.geometryHandler as geom
import pythonScripts.constants as const
```

---

## `celestialBodies`

### `AvailablePlanets()`

Loads the built-in planet/star catalog from `Resources/planets.csv` and `Resources/stars.csv`.

| Method | Returns |
|---|---|
| `findPlanet(namePlanet: str) -> Planet \| None` | The `Planet` with that name (prints a warning and returns `None` if absent). |
| `listPlanetNames() -> list[str]` | All catalog planet names. |

```python
planet = bodies.AvailablePlanets().findPlanet('WASP-39b')
```

Catalog includes (at time of writing): `WASP-49b`, `HD189733b`, `55-Cancri-e`, `WASP-39b`, `KELT-20b`, `KELT-9b`, `WASP-121b`, `WASP-189b`, `WASP-21b`, `WASP-69b`, `WASP-76b`, `WASP-79b`, `LHS-1140b`.

### `Planet`

```python
Planet(name, R, M, a, hostStar, transitDuration, orbitalPeriod, b)
```

Usually obtained from `AvailablePlanets`, not constructed directly. Key attributes: `R` (radius, cm), `M` (mass, g), `a` (semi-major axis, cm), `hostStar` (a `Star`), `orbitalPeriod` (days), `transitDuration`, `b` (impact parameter, cm).

Selected methods: `getPosition(orbphase)`, `getLOSvelocity(orbphase)` (cm/s), `getDistanceFromPlanet(x, phi, rho, orbphase)`, `getTorusCoords(x, phi, rho, orbphase)`. All accept scalar or batched `(n_chords,)` inputs.

### `Star`

```python
Star(R, dR, M, dM, T_eff, dT_eff, log_g, dlog_g, Z, dZ, alpha)
```

Obtained via `planet.hostStar`. Attributes include `R`, `M`, `T_eff`, `log_g`, `Z`, and (initially zero) `CLV_u1`, `CLV_u2`, `vsiniStarrot`, `phiStarrot`, `Fstar_function`.

| Method | Purpose |
|---|---|
| `addFstarFunction(wavelength)` | Fetch a PHOENIX spectrum and build a `log10(flux)` interpolator over the simulation wavelength grid. Call **after** `Transit.addWavelength()`. |
| `addCLVparameters(CLV_u1, CLV_u2)` | Quadratic limb-darkening coefficients. |
| `addRMparameters(vsiniStarrot, phiStarrot)` | Rossiter–McLaughlin: projected rotation velocity (cm/s) and rotation-axis azimuth (rad). |

If `addFstarFunction` is never called, the star is treated as a flat continuum (`F_star = 1`), which is faster.

### `Moon`

```python
Moon(midTransitOrbphase, R, a, hostPlanet)
```

`midTransitOrbphase` (rad) is the moon's orbital phase relative to the planet at the planet's mid-transit; `R` (cm), `a` (moon semi-major axis around the planet, cm), `hostPlanet` (a `Planet`). Used with `MoonExosphere` / `TidallyHeatedMoon`.

### Convenience constructors & moon orbital mechanics

Module-level helpers that wrap the constructors above with research-sensible defaults.

```python
find_planet(name) -> Planet                 # like AvailablePlanets().findPlanet but RAISES if absent
make_moon(planet, a_over_Rp=1.7, R=None, midTransitOrbphase=0.375*2*pi) -> Moon
```

`make_moon` places a moon at `a_over_Rp · planet.R`; `R` defaults to `const.R_Io`.

Moon:planet orbital-mechanics relations (used to place a moon for transit; all take `a_over_Rp`, so they can be evaluated *before* a `Moon` exists):

```python
mean_motion_ratio(planet, a_over_Rp=1.7) -> float        # N = sqrt(a_p^3 M_p / (a_m^3 M_*))
optimal_midtransit_phase(planet, a_over_Rp=1.7, branch='late') -> float   # rad; maximises peak shift
max_peak_shift_minutes(planet, a_over_Rp=1.7) -> float   # max lightcurve peak displacement [min]
```

`optimal_midtransit_phase` returns the moon phase at mid-transit that maximises the moon's sky-plane displacement (`branch='late'` → peak after mid-transit; `'early'` → before). See `Tests/midtransit_phase_proof.tex`.

---

## `geometryHandler`

### `Grid`

```python
Grid(x_midpoint, x_border, x_steps,
     rho_border, rho_steps, phi_steps,
     orbphase_border, orbphase_steps)
```

| Parameter | Meaning |
|---|---|
| `x_midpoint` | Center of the line-of-sight integration chord (cm); typically `planet.a`. |
| `x_border` | Half-length of the chord (cm); e.g. `14 * planet.R`. |
| `x_steps` | Number of cells along the line of sight. |
| `rho_border` | Maximum sky-plane radius (cm); typically `planet.hostStar.R`. |
| `rho_steps` | Number of radial steps. |
| `phi_steps` | Number of azimuthal steps. |
| `orbphase_border` | Maximum absolute orbital phase to simulate (rad); `0.0` for a single mid-transit spectrum. |
| `orbphase_steps` | Number of orbital-phase points; `1` for a single spectrum, `>1` for a light curve/map. |

Useful methods: `getChordGrid()` (flattened `(N,3)` of `(phi, rho, orbphase)`), `constructXaxis()`, `constructRhoAxis()`, `constructPhiAxis()`, `constructOrbphaseAxis()`, `getDeltaX()`, `getDeltaRho()`, `getDeltaPhi()`.

### Grid convenience builders

```python
spatial_grid(planet, x_border_Rp=12.0, x_steps=25, rho_steps=60, phi_steps=30,
             orbphase_window=0.0, orbphase_steps=1, rho_border=None) -> Grid
orbphase_window_from_hours(planet, half_window_hours) -> float   # hours → rad half-window
```

`spatial_grid` builds a `Grid` tuned for an extended exosphere: chord centred on `planet.a`, and `rho_border` defaults to the **stellar radius** (depth normalises by the stellar disk — do not shrink it). `orbphase_window` is the orbital-phase half-window (0 → single mid-transit phase; `>0` with `orbphase_steps>1` → a lightcurve), conveniently produced from a `±hours` window by `orbphase_window_from_hours`.

---

## `gasProperties`

### `WavelengthGrid`

```python
WavelengthGrid(lower_w, upper_w, widthHighRes, resolutionLow, resolutionHigh)
```

| Parameter | Meaning |
|---|---|
| `lower_w`, `upper_w` | Wavelength bounds (cm). |
| `widthHighRes` | Width of the fine-sampled region centered on each atomic line (cm). |
| `resolutionLow` | Step size away from lines (cm). |
| `resolutionHigh` | Step size inside the high-resolution regions (cm). |

The grid array itself is built by `Transit.addWavelength()` → `constructWavelengthGrid`: fine sampling near atomic lines, coarse elsewhere. If there are no atomic lines in range, a uniform `resolutionLow` grid is returned. Molecular/continuum opacities do not add grid points.

Convenience builders (bounds in Å):

```python
na_d_grid(lower_ang=5880.0, upper_ang=5910.0, widthHighRes=4e-8,
          resolutionLow=3e-9, resolutionHigh=2e-10) -> WavelengthGrid   # Na D doublet window
line_grid(center_ang, half_window_ang=15.0, **kwargs) -> WavelengthGrid # centred on any line
```

---

### Collisional atmospheres

All share the `CollisionalAtmosphere` base, which provides the constituent-adding API:

| Method | Adds |
|---|---|
| `addConstituent(speciesName, chi)` | Atomic/ionic absorber with mixing ratio `chi`. Look up the species mass automatically and compute the thermal velocity dispersion. |
| `addMolecularConstituent(speciesName, chi)` | Molecular absorber with mixing ratio `chi` (cross sections from `Resources/molecularResources/<name>.h5`). |
| `addScatteringConstituent(scattererType, paramsDict)` | Continuum scatterer/aerosol (see [Scattering constituents](#scattering-constituents)). |

After adding an **atomic or molecular** constituent, call `atm.constituents[-1].addLookupFunctionToConstituent(wg)` to precompute its opacity lookup. (Scatterers do not require this.)

The three collisional models:

```python
BarometricAtmosphere(T, P_0, mu, planet)     # isothermal exponential profile
HydrostaticAtmosphere(T, P_0, mu, planet)    # hydrostatic equilibrium (Jeans term)
PowerLawAtmosphere(T, P_0, q, planet)        # n ∝ (R_p / r)^q
```

- `T` — temperature (K).
- `P_0` — reference (base) pressure (barye). Reference number density is `n_0 = P_0 / (k_B T)`.
- `mu` — mean molecular weight as a **mass in grams** (e.g. `2.3 * const.amu`).
- `q` — power-law index (`PowerLawAtmosphere` only).
- `planet` — a `Planet`.

---

### Evaporative exospheres

All share the `EvaporativeExosphere` base, normalized by a total particle number `N`. **An evaporative exosphere holds exactly one constituent.** Its constituent API differs from the collisional one:

| Method | Adds |
|---|---|
| `addConstituent(speciesName, sigma_v)` | The single atomic/ionic absorber. The second argument is the **velocity dispersion `sigma_v` (cm/s)**, not a mixing ratio (mixing ratio is fixed to 1; `N` sets the absolute amount). |
| `addMolecularConstituent(speciesName, T)` | A single molecular absorber with a pseudo-temperature `T`. |
| `addScatteringConstituent(scattererType, paramsDict)` | A continuum scatterer. |

The models:

```python
PowerLawExosphere(N, q, planet)
MoonExosphere(N, q, moon)                 # sourced from a Moon; sets hasMoon=True
TidallyHeatedMoon(q, moon)                # phase-dependent source rate (see below)
TorusExosphere(N, a_torus, v_ej, planet)
SerpensExosphere(filename, N, planet, sigmaSmoothing)
RadialWindExosphere(...)                  # see below
```

- `N` — total particle number.
- `q` — power-law index.
- `a_torus` — torus centerline radius (cm); `v_ej` — ejection velocity (cm/s), which sets the torus scale height `H = a_torus · v_ej / v_orbit`.
- `moon` — a `Moon`.
- `SerpensExosphere` also requires `exo.addInterpolatedDensity(spatialGrid)` to histogram the SERPENS particle file onto the grid before running.

**`TidallyHeatedMoon`** additionally needs a source-rate profile:

```python
exo = gasprop.TidallyHeatedMoon(q, moon)
exo.addSourceRateFunction(filename, tau_photoionization, mass_absorber)
```

`N` at each phase is computed as `Mdot(phase) · tau_photoionization / mass_absorber`.

#### `RadialWindExosphere`

```python
RadialWindExosphere(Mdot, mu, v_terminal=None, beta=1.0,
                    r_inner=None, r_outer=None, v_base=None,
                    wind_model='beta', T=None, planet=None, wind_mu=None)
```

| Parameter | Meaning |
|---|---|
| `Mdot` | Mass-loss rate (g/s); sets the density normalization. |
| `mu` | Tracer mean particle mass (g) for the continuity normalization. |
| `v_terminal` | Terminal speed (cm/s). **Required for `wind_model='beta'`.** |
| `beta` | β-law exponent (default 1.0). |
| `r_inner` | Inner boundary (cm); defaults to `planet.R`. |
| `r_outer` | Optional outer cutoff (cm). |
| `v_base` | Launch speed at `r_inner` (cm/s); defaults to `1e-3 · v_terminal`. |
| `wind_model` | `'beta'` (default) or `'parker'` (exact isothermal Parker wind). |
| `T` | Wind temperature (K). **Required for `wind_model='parker'`.** |
| `planet` | Host planet (required for `'parker'`; needed generally). |
| `wind_mu` | Bulk-gas mean mass (g) fixing the Parker **dynamics**; defaults to `mu`. Use it to advect a trace species in a light (H/He) outflow. |

For `wind_model='parker'`, the object exposes `c_s` (sound speed), `r_c` (sonic radius), and `_wind_velocity(r)` (the analytic velocity profile). When orbital Doppler shifting is on, `calculateLOSVelocity(...)` is invoked automatically to apply a position-dependent Doppler shift. See [architecture.md](architecture.md#radial-wind-physics).

#### Scenario builders

Factory functions that construct an evaporative exosphere **and** attach its atomic constituent + line-lookup in one call (so the returned object is ready to drop into `Atmosphere`):

```python
moon_exosphere_scenario(N, q, moon, species, sigma_v, wavelengthGrid) -> MoonExosphere
powerlaw_exosphere_scenario(N, q, planet, species, sigma_v, wavelengthGrid) -> PowerLawExosphere
radial_wind_scenario(Mdot, planet, species, sigma_v, wavelengthGrid,
                     mu=None, wind_model='parker', T=1e4, wind_mu=None,
                     v_terminal=None, beta=1.0, v_base=None,
                     r_inner=None, r_outer=None) -> RadialWindExosphere
```

`species` is a key like `'NaI'`; `sigma_v` the velocity dispersion (cm/s). For `radial_wind_scenario`, `mu` defaults to the species' own atomic mass.

---

### Absorber / scatterer constituents

You normally create these via the `add*Constituent` methods above rather than directly; this section documents what they are.

#### `AtmosphericConstituent` (atoms / ions)

Represents an atomic/ionic absorber. Computes a Voigt-profile cross section from the line list and stores a `log10(sigma)` interpolator.

- `addLookupFunctionToConstituent(wavelengthGrid)` — **must be called** before running, after the constituent is added. Precomputes the cross-section interpolator on a refined grid.
- `getSigmaAbs(wavelength)` — cross section (cm²) at the given wavelengths (supports batched arrays).

#### `MolecularConstituent`

Represents a molecule. Reads its `(P, T, wavelength)` cross-section table from `Resources/molecularResources/<moleculeName>.h5`.

- `addLookupFunctionToConstituent(wavelengthGrid=None)` — **must be called** before running. If a `wavelengthGrid` is passed, the stored table is sliced to the simulation range (plus a 1% Doppler margin) for a large speedup on narrow grids.
- `getSigmaAbs(P, T, wavelength)` — cross section (cm²).

#### Scattering constituents

Added via `addScatteringConstituent(scattererType, paramsDict)`, where `scattererType` is one of `SCATTERER_TYPES`: `'RayleighHaze'`, `'GrayCloud'`, `'PowerLawAerosol'`, `'TabulatedAerosol'`. The `paramsDict` keys map to the constructor arguments below (all optional except where noted; defaults shown).

```python
RayleighHaze(chi=1.0, sigma_ref=5.31e-27, lambda_ref=4000e-8, slope=4.0, P_top=None)
#   sigma(λ) = sigma_ref * (lambda_ref / λ) ** slope   (slope=4 → pure Rayleigh)

GrayCloud(chi=1.0, sigma_gray=1e-10, P_top=None)
#   sigma(λ) = sigma_gray  (wavelength-independent)

PowerLawAerosol(chi=1.0, sigma_ref=1e-25, lambda_ref=5500e-8, alpha=2.0, P_top=None)
#   sigma(λ) = sigma_ref * (λ / lambda_ref) ** (-alpha)   (Ångström convention)

TabulatedAerosol(chi=1.0, filepath='', extrapolate='edge', P_top=None)
#   sigma(λ) from a 2-column CSV: wavelength [Å], sigma [cm²]
```

- `chi` — particle-to-gas abundance ratio.
- `P_top` — optional cloud-top pressure (barye); opacity applies only where local gas pressure ≥ `P_top` (collisional/temperature-bearing host models only).
- Scattering is treated as extinction out of the beam; no phase function or multiple scattering is modeled, and no Doppler shift is applied to continuum opacity.

Example dict form:

```python
atm.addScatteringConstituent('RayleighHaze',
    {'chi': 1.0, 'sigma_ref': 1.5e-22, 'lambda_ref': 4000e-8, 'slope': 4.0})
```

---

### `Atmosphere`

```python
Atmosphere(densityDistributionList, hasOrbitalDopplerShift)
```

Wraps a list of one or more density models and owns the optical-depth computation.

- `densityDistributionList` — e.g. `[atm]` or `[hydrostatic, wind]`.
- `hasOrbitalDopplerShift` — `True` to apply orbital (and, for wind models, position-dependent) Doppler shifts; `False` for fast continuum-dominated runs.

Key method (called internally by `Transit`): `getLOSopticalDepth_Batch(x_grid, phi_batch, rho_batch, orbphase_batch, wavelength, delta_x) -> (n_chords, n_wav)`.

---

### `Transit`

```python
Transit(atmosphere, wavelengthGrid, spatialGrid)
```

The orchestrator. `atmosphere` is an `Atmosphere`, `wavelengthGrid` a `WavelengthGrid`, `spatialGrid` a `Grid`. The primary planet is read from `atmosphere.densityDistributionList[0].planet`.

| Method | Purpose |
|---|---|
| `addWavelength()` | Build and store `Transit.wavelength` (cm) from the atomic line list and grid parameters. Call before `sumOverChords`. |
| `sumOverChords(max_memory_gb=2.0) -> np.ndarray` | Run the simulation. Returns `R` of shape `(orbphase_steps, n_wavelength)`. `R[i]` is the flux ratio at orbital phase `i`; `1 - R[i]` is the transit depth. |
| `evaluateChord(phi, rho, orbphase)` | Lower-level single-chord evaluation returning `(F_in, F_out)`. |
| `checkBlock(phi, rho, orbphase)` | Whether a chord is blocked by the opaque planet/moon disk. |

`max_memory_gb` caps the per-batch RAM footprint; pass `None` to use 50% of available system RAM.

Typical end-of-run reduction:

```python
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.5)
depth_ppm = (1.0 - R[0]) * 1e6          # single mid-transit spectrum
wavelength_um = sim.wavelength * 1e4
```

### `run_transit` and `TransitResult`

A one-call wrapper around the `Atmosphere` → `Transit` → `sumOverChords` reduction, returning a `TransitResult` with spectrum/lightcurve accessors:

```python
run_transit(scenarios, wavelengthGrid, spatialGrid,
            hasOrbitalDopplerShift=True, use_phoenix_star=True,
            max_memory_gb=4.0) -> TransitResult
```

- `scenarios` — list of density distributions (e.g. from the scenario builders); the host planet is taken from the first.
- `use_phoenix_star=False` uses a flat star (much faster; fine for relative depths).

`TransitResult` (a dataclass) carries `wavelength_cm`, `R_2D` (`(n_phase, n_wav)`), `orbphase`, `planet`, with:

| Member | Returns |
|---|---|
| `wavelength_ang` / `wavelength_um` | Wavelength axis in Å / µm. |
| `spectrum()` | Phase-collapsed `R(λ)` (median over phase). |
| `spectrum_normalized()` | Spectrum / its continuum max. |
| `transit_depth(line_window_ang=…, continuum_exclude_ang=…, mode='peak')` | Excess absorption fraction vs continuum (`mode='peak'` or `'mean'`). |
| `lightcurve(line_window_ang=…, continuum_exclude_ang=…, mode='mean')` | Band line/continuum vs phase (needs `orbphase_steps>1`). |

Window defaults are centred on the vacuum Na D2 line (`const.NA_D2_ANG`).

```python
scen = gasprop.moon_exosphere_scenario(N=5e33, q=3.34, moon=moon,
                                       species='NaI', sigma_v=2e6, wavelengthGrid=wg)
res  = gasprop.run_transit([scen], wg, sg, use_phoenix_star=False)
depth_pct = res.transit_depth() * 100.0
```

---

## `constants`

Physical constants (cgs): `e`, `m_e`, `c`, `G`, `k_B`, `amu`, `R_J`, `M_J`, `M_E`, `R_sun`, `M_sun`, `R_Io`, `AU`. Na D doublet rest wavelengths (Å, **vacuum**, matching `LineList.txt`): `NA_D2_ANG` (5891.583), `NA_D1_ANG` (5897.558).

- `calculateDopplerShift(v)` — relativistic Doppler factor for line-of-sight velocity `v` (cm/s).
- `AvailableSpecies().findSpecies(name)` / `.listSpeciesNames()` — the atomic/ionic catalog. Built-in species include `NaI`, `KI`, `SiI`–`SiIV`, `MgI`/`MgII`, `AlI`, `CaI`/`CaII`, `TiI`/`TiII`, `CrI`, `MnI`, `FeI`, `CoI`, `NiI`, `OI`, `CII`, `SIII`, `SIV`.
