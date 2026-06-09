# Module Reference

Per-module documentation of every class and the significant functions in the
Prometheus code base. Units are cgs unless noted. Method signatures match the
source.

Module index:

- [`prometheus.py`](#prometheuspy) — entry point / driver
- [`mainRetrieval.py`](#mainretrievalpy) — scripted example
- [`pythonScripts/constants.py`](#constantspy) — constants and species
- [`pythonScripts/celestialBodies.py`](#celestialbodiespy) — Star, Planet, Moon
- [`pythonScripts/geometryHandler.py`](#geometryhandlerpy) — spatial/temporal grid
- [`pythonScripts/gasProperties.py`](#gaspropertiespy) — density models, opacity, transit
- [`pythonScripts/memoryHandler.py`](#memoryhandlerpy) — batch sizing
- [`pythonScripts/setup.py`](#setuppy) — interactive setup wizard
- [`pythonScripts/shotNoise.py`](#shotnoisepy) — SNR / shot-noise post-processing
- [`Resources/astroquery_retrieval.py`](#resourcesastroquery_retrievalpy) — line-list builder

---

## `prometheus.py`

The command-line driver. With the argument `setup`, it runs the interactive
wizard (`setup.createSetupFile`). Otherwise it treats `sys.argv[1]` as a
setup-file name, parses the optional `--max-memory <GB>` flag (default 2.0),
loads `../setupFiles/<name>.txt`, constructs the object graph (planet,
wavelength grid, spatial grid, scenario list with constituents, `Atmosphere`,
`Transit`), runs `Transit.sumOverChords`, and writes the `(phase, wavelength, R)`
grid to `../output/<name>.txt`. `PATH` is the parent of the repository folder.

---

## `mainRetrieval.py`

A self-contained, non-interactive example (forward model + light-curve
extraction, **not** an actual inverse retrieval despite the file name). It builds
a WASP-49 b system, a `TidallyHeatedMoon` sodium exosphere with a phase-dependent
source rate, runs `sumOverChords`, filters the result to the Na D doublet, and
plots a normalized light curve with matplotlib. Note it references file paths and
the `addSourceRateFunction` workflow specific to the original author's machine —
read it as an API usage demonstration.

---

## `constants.py`

Physical/astronomical constants in cgs (`e`, `m_e`, `c`, `G`, `k_B`, `amu`,
`R_J`, `M_J`, `M_E`, `R_sun`, `M_sun`, `R_Io`, `AU`, …).

- **`calculateDopplerShift(v) -> factor`** — relativistic Doppler factor
  `√((1−β)/(1+β))`, `β = v/c`. Multiply a rest wavelength by this to get the
  observed wavelength. Positive `v` = receding = redshift.
- **`Species(name, element, ionizationState, mass)`** — one atom/ion (e.g.
  `NaI`, element `Na`, stage `'1'`, mass in g).
- **`SpeciesCollection`** — container with `findSpecies(name)`,
  `listSpeciesNames()`, `addSpecies(species)`.
- **`AvailableSpecies(SpeciesCollection)`** — pre-populated list of 22 supported
  atoms/ions: `NaI, KI, SiI–IV, MgI/II, AlI, CaI/II, TiI/II, CrI, MnI, FeI, CoI,
  NiI, OI, CII, SIII, SIV`.

---

## `celestialBodies.py`

### `Star`
Stellar parameters and spectrum handling.
- `addCLVparameters(u1, u2)` — quadratic limb-darkening coefficients.
- `addRMparameters(vsini, phi_rot)` — projected rotation velocity and axis angle.
- `getSurfaceVelocity(phi, rho)` — LOS velocity of a disk point (for RM).
- `getSpectrum()` — round `(T_eff, log g, [Fe/H], α)` to the PHOENIX grid,
  download/cache the FITS files, return `(wavelength [cm], flux/π)`.
- `addFstarFunction(wavelength)` — build a `log10 F(λ)` interpolant over the
  simulation window (± rotation broadening) and store it as `Fstar_function`.
- `calculateCLV(rho)` — quadratic limb-darkening intensity factor.
- `calculateRM(phi, rho, wavelength)` — Doppler-shifted local stellar flux.
- `getFstarIntegrated(wavelength, grid)` — disk-integrated flux (closed form
  when not rotating; summed otherwise).
- `getFstar(phi, rho, wavelength)` — local flux (CLV, plus RM if rotating).

### `Planet`
Name, radius `R`, mass `M`, semi-major axis `a`, host `Star`, transit duration,
period, impact parameter `b`.
- `getPosition(orbphase) -> (x_p, y_p)` — circular edge-on orbit.
- `getLOSvelocity(orbphase)` — `−sin(orbphase)·√(G·M_star/a)`.
- `getDistanceFromPlanet(x, phi, rho, orbphase)` — 3-D distance to planet center;
  **batched** to `(n_chords, n_x)`.
- `getTorusCoords(x, phi, rho, orbphase) -> (a_cyl, z)` — cylindrical coords for
  the torus model; batched.

### `Moon`
`midTransitOrbphase`, radius `R`, semi-major axis `a` (around the planet), host
`Planet`.
- `getOrbphase(orbphase)` — moon phase from the planet phase via the Kepler
  period ratio.
- `getPosition(orbphase)` — moon `(x, y)` in the star frame.
- `getLOSvelocity(orbphase)` — planet velocity + moon orbital velocity.
- `getDistanceFromMoon(x, phi, rho, orbphase)` — batched distance to the moon.

### `AvailablePlanets`
Loads `Resources/stars.csv` and `Resources/planets.csv` into `Star`/`Planet`
objects. `listPlanetNames()`, `findPlanet(name)`.

---

## `geometryHandler.py`

### `Grid`
Holds the discretization parameters and builds the axes.
- `getCartesianFromCylinder(phi, rho) -> (y, z)` (static).
- `getDeltaX / getDeltaRho / getDeltaPhi` — cell sizes.
- `constructXaxis / constructRhoAxis / constructPhiAxis(midpoints=True)` — axes,
  either cell midpoints or edges.
- `constructOrbphaseAxis()` — orbital phases in `[−border, +border]`.
- `getChordGrid()` — flattened `(phi·rho·orbphase, 3)` array of all chords.

---

## `gasProperties.py`

The largest module: density models, opacity sources, the atmosphere aggregator,
and the transit driver.

### Numba kernels
- `n_interp_log(x_targets, x_grid, y_grid_log, offset)` — row-wise 1-D
  interpolation of `log10 y`, returning `10^log − offset`. Two-pointer O(N+M)
  scan; requires monotonic target rows.
- `n_interp_linear_rows(x_targets, x_grid, y_grid_2d)` — row-wise 1-D
  interpolation in linear space with a per-row `y`.
- `_bilinear_PT_interp(P_vals, T_val, P_grid, T_grid, sigma_grid_log, offset)` —
  bilinear `(P, T)` interpolation of the molecular log-σ table at fixed `T`,
  returning σ on the native wavelength grid.

### Collisional scenarios

**`CollisionalAtmosphere`** (base): `T`, `P_0`, `constituents`, `hasMoon=False`.
- `getReferenceNumberDensity()` → `P_0/(k_B T)`.
- `getVelDispersion(m)` → `√(T k_B / m)`.
- `addConstituent(name, chi)` — atomic/ionic (mixing ratio).
- `addMolecularConstituent(name, chi)` — molecular.
- `addScatteringConstituent(type, paramsDict)` — continuum source.

| Class | `calculateNumberDensity(x, phi, rho, orbphase)` |
|---|---|
| `BarometricAtmosphere(T, P_0, mu, planet)` | `n_0·exp((R_p − r)/H)·H(r − R_p)`, `H = k_B T R_p²/(G μ M_p)` |
| `HydrostaticAtmosphere(T, P_0, mu, planet)` | `n_0·exp(Jeans(r) − Jeans_0)` (hydrostatic equilibrium) |
| `PowerLawAtmosphere(T, P_0, q, planet)` | `n_0·(R_p/r)^q·H(r − R_p)` |

### Evaporative scenarios

**`EvaporativeExosphere`** (base): `N`, `hasMoon=False`.
- `addConstituent(name, sigma_v)` — single atomic absorber (velocity dispersion).
- `addMolecularConstituent(name, T)` — single molecular absorber, stores pseudo-`T`.
- `addScatteringConstituent(type, paramsDict)` — appended (coexists with absorber).

| Class | Density model |
|---|---|
| `PowerLawExosphere(N, q, planet)` | `n_0·(R_p/r)^q`, `n_0 = (q−3)/(4π R_p³)·N` |
| `MoonExosphere(N, q, moon)` | as above but centered on the moon (`hasMoon=True`) |
| `TidallyHeatedMoon(q, moon)` | moon exosphere with phase-dependent particle number `N(orbphase)`; `addSourceRateFunction(file, tau_photoionization, mass)` loads a mass-loss-rate curve |
| `TorusExosphere(N, a_torus, v_ej, planet)` | Gaussian-in-(radius, height) torus; scale height `H = a_torus·v_ej/v_orbit` |
| `SerpensExosphere(file, N, planet, sigmaSmoothing)` | density histogrammed from SERPENS particles; `addInterpolatedDensity(grid)` builds a 3-D interpolant |
| `RadialWindExosphere(Mdot, mu, ...)` | mass-continuity wind; see below |

**`RadialWindExosphere`** — `n(r) = Mdot/(4π r² v(r) μ)`, masked to
`[r_inner, r_outer]`.
- `_wind_velocity(r)` dispatches by `wind_model`.
- `_wind_velocity_parker(r)` — exact isothermal Parker solution via Lambert-W.
- `calculateLOSVelocity(x_grid, phi_batch, rho_batch, orbphase_batch)` — the
  `+x` projection of the radial outflow, `(n_chords, n_x)`, used for
  position-dependent Doppler shifts. Constructor knobs: `v_terminal`, `beta`,
  `v_base`, `r_inner`, `r_outer`, `wind_model ∈ {'beta','parker'}`, `T`,
  `wind_mu`.

### Opacity-source constituents

**`AtmosphericConstituent(species, chi, sigma_v)`** — atomic/ionic.
- `getLineParameters(wavelength)` → `(line_λ, γ, f)` for this species in range.
- `calculateVoigtProfile(wavelength)` → summed Voigt cross-section [cm²].
- `constructLookupFunction / addLookupFunctionToConstituent(wavelengthGrid)` —
  build a refined-grid `log10 σ` interpolant.
- `getSigmaAbs(wavelength)` — evaluate via `n_interp_log` (supports 2-/3-D input).

**`MolecularConstituent(moleculeName, chi)`** — molecular.
- `constructLookupFunction(wavelengthGrid=None)` — read `<molecule>.h5`
  (`p`, `t`, `bin_edges`, `xsecarr`), slice to the simulation window ± 1 %, store
  `P_grid/T_grid/wav_grid/sigma_grid_log` for the fast path and a
  `RegularGridInterpolator` for the legacy `getSigmaAbs`.
- `getSigmaAbs(P, T, wavelength)` — legacy 3-D `(n_chords, n_x, n_wav)` lookup.

**`ScatteringConstituent(chi, P_top=None)`** (base, `isScatterer=True`) —
continuum extinction per host-gas particle. `getSigmaAbs(wavelength)` caches
`calculateSigmaAbs(wavelength)`.

| Subclass | `σ(λ)` |
|---|---|
| `RayleighHaze(chi, sigma_ref=5.31e-27, lambda_ref=4000e-8, slope=4, P_top)` | `σ_ref·(λ_ref/λ)^slope` |
| `GrayCloud(chi, sigma_gray=1e-10, P_top)` | constant `σ_gray` |
| `PowerLawAerosol(chi, sigma_ref=1e-25, lambda_ref=5500e-8, alpha=2, P_top)` | `σ_ref·(λ/λ_ref)^(−alpha)` (Ångström convention) |
| `TabulatedAerosol(chi, filepath, extrapolate='edge', P_top)` | linearly interpolated from a 2-col CSV (λ [Å], σ [cm²]); `'edge'` holds, `'zero'` cuts off |

`makeScatteringConstituent(scattererType, paramsDict)` — factory; `SCATTERER_TYPES
= ('RayleighHaze', 'GrayCloud', 'PowerLawAerosol', 'TabulatedAerosol')`.

### Aggregation and driver

**`Atmosphere(densityDistributionList, hasOrbitalDopplerShift)`**
- `getAbsorberNumberDensity(dist, chi, x, phi, rho, orbphase)` (static).
- `getAbsorberVelocityField(dist, x, phi, rho, orbphase)`.
- `getLOSopticalDepth_Batch(x_grid, phi_batch, rho_batch, orbphase_batch,
  wavelength, delta_x)` — the vectorized optical-depth kernel; returns
  `τ (n_chords, n_wav)`. (See [architecture.md §4](architecture.md#4-the-optical-depth-kernel).)

**`WavelengthGrid(lower_w, upper_w, widthHighRes, resolutionLow, resolutionHigh)`**
- `arangeWavelengthGrid(linesList)` — non-uniform grid, fine near lines.
- `constructWavelengthGridSingle(constituent)` / `constructWavelengthGrid(list)` —
  build the grid from all atomic/ionic lines (molecular/scattering ignored).

**`Transit(atmosphere, wavelengthGrid, spatialGrid)`**
- `addWavelength()` — construct and store the wavelength array.
- `checkBlock(phi, rho, orbphase)` — is this chord behind the opaque planet/moon?
- `evaluateChord(phi, rho, orbphase) -> (F_in, F_out)` — single-chord transmission.
- `sumOverChords(max_memory_gb=2.0) -> R` — batched full simulation; returns
  `(n_orbphase, n_wav)`.

---

## `memoryHandler.py`

Memory-aware batch sizing.
- `get_available_memory(max_memory_gb=2.0)` — bytes available (or 50 % of system
  RAM if `None`).
- `estimate_chord_memory(num_wavelengths, n_x, is_molecular=True)` — per-chord
  byte estimate (~640 B/λ molecular, 16 B/λ atomic, ×2 buffer).
- `calculate_optimal_chunk_size(total_chords, num_wavelengths, n_x,
  max_memory_gb=2.0, is_molecular=True)` — clamp to `[1, total_chords]`.
- `chunk_array(array, chunk_size, axis=0)`, `chunk_indices(total_items,
  chunk_size)` — generic chunking helpers.

---

## `setup.py`

The interactive wizard. `NumericalQuestion` and `TextQuestion` validate input
(bounds, units, allowed options, yes/no). The block functions are
`setupFundamentals`, `setupScenario`, `setupArchitecture`,
`setupRayleighHaze`/`setupGrayCloud`/`setupPowerLawAerosol`/`setupTabulatedAerosol`,
`setupSpecies`, `setupGrid`. `createSetupFile(PATH)` runs them in order and
writes the JSON to `../setupFiles/<name>.txt`. See
[api.md](api.md#setup-file-json-schema) for the resulting schema.

---

## `shotNoise.py`

SNR estimation and Gaussian shot-noise injection, decoupled from the simulation
(operates on NumPy arrays of wavelength [nm] and flux).
- `scale_snr(...)` — photon-noise scaling law (magnitude + exposure-time ratios).
- `TransitParams(target_mag, transit_duration_hrs, num_bins)` — dataclass.
- `SNRModel` — constructors `constant(snr_per_bin)`, `from_table(...)`,
  `from_json(...)` (ESO ETC v2 schema), `from_csv(...)`. Evaluators `snr_at(λ)`
  and `snr_array(λ)`.
- `sigma_from_snr(snr)` — 1/SNR (0 where SNR ≤ 0).
- `apply_shot_noise(wavelength_nm, spectrum, snr_model, seed=None) ->
  (noisy_spectrum, sigma)`.

See [api.md](api.md#shot-noise-api) for usage.

---

## `Resources/astroquery_retrieval.py`

A standalone helper that queries the NIST line database (`astroquery.nist`) for
selected ions, filters by oscillator strength, and writes new entries in the
tab-separated `LineList.txt` format. Run it manually to extend the line list.
