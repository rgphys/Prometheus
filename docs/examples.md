# Examples

These worked examples are simplified and annotated versions of the figure scripts that ship with the project (`fig*.py`). Each builds a simulation purely from Python objects — no configuration files. All examples assume the `core` package is importable (see [getting-started.md](getting-started.md#installation)).

Common imports:

```python
import numpy as np
import core.gasProperties as gasprop
import core.celestialBodies as bodies
import core.geometryHandler as geom
import core.constants as const
```

---

## 1. Hydrostatic transit spectrum with atoms (+ a haze)

*Based on the optical segment of `fig1_model_zoo.py`.* A hydrostatic WASP-39 b atmosphere with Na I and K I lines on a Rayleigh-haze continuum, integrated across the optical.

```python
planet = bodies.AvailablePlanets().findPlanet('WASP-39b')

MU     = 2.3 * const.amu     # mean molecular weight (mass, g)
P_0    = 1e4                 # base pressure (barye)
T_ATM  = 1100.0             # K
CHI_NA = 3e-7               # Na mixing ratio
CHI_K  = 1e-7               # K mixing ratio

# Wavelength grid: 0.35–1.0 µm, fine sampling in the alkali line cores.
wg = gasprop.WavelengthGrid(
    lower_w=0.35e-4, upper_w=1.0e-4,
    widthHighRes=8e-8, resolutionLow=4e-7, resolutionHigh=1e-9,
)

# Hydrostatic atmosphere with two atomic absorbers + a Rayleigh haze.
atm = gasprop.HydrostaticAtmosphere(T_ATM, P_0, MU, planet)
for name, chi in [('NaI', CHI_NA), ('KI', CHI_K)]:
    atm.addConstituent(name, chi)
    atm.constituents[-1].addLookupFunctionToConstituent(wg)   # atoms: build Voigt lookup
atm.addScatteringConstituent(
    'RayleighHaze',
    {'chi': 1.0, 'sigma_ref': 1.5e-22, 'lambda_ref': 4000e-8, 'slope': 4.0},
)

# Finely resolved geometry (thin annulus around the planet).
s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=14.0 * planet.R, x_steps=80,
    rho_border=planet.hostStar.R, rho_steps=400, phi_steps=20,
    orbphase_border=0.0, orbphase_steps=1,
)

# Continuum-dominated, so orbital Doppler shifts can be left off (faster).
sim = gasprop.Transit(
    gasprop.Atmosphere([atm], hasOrbitalDopplerShift=False), wg, s_grid)
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.45)[0]

wav_um   = sim.wavelength * 1e4
depth_ppm = (1.0 - R) * 1e6
# Na I D appears near 0.5894 µm, K I near 0.7684 µm, on a λ^-4 haze slope.
```

**Notes**

- Atoms require `addLookupFunctionToConstituent(wg)`; the haze (a scatterer) does not.
- `atm.constituents[-1]` is the constituent just added — the lookup is attached per-constituent.
- The fine `rho_steps=400` is needed because the absorbing annulus is thin relative to the stellar disk; it converges the chord sum.

---

## 2. Molecular transmission spectrum (JWST near-IR)

*Based on `fig8_molecular_jwst-8.py`.* H₂O, CO₂, CO, and SO₂ in a hydrostatic atmosphere across 0.6–5.3 µm — the NIRSpec/PRISM grasp.

```python
planet = bodies.AvailablePlanets().findPlanet('WASP-39b')

MU    = 2.3 * const.amu
P_0   = 1e4
T_ATM = 1100.0

# Roughly WASP-39 b-like mixing ratios; SO2 is a photochemical product.
MOLECULES = [('H2O', 1e-3), ('CO2', 1e-4), ('CO', 3e-4), ('SO2', 3e-6)]

# Molecular opacity is smooth — a uniform, coarse grid suffices
# (high == low resolution here).
wg = gasprop.WavelengthGrid(
    lower_w=0.6e-4, upper_w=5.3e-4,
    widthHighRes=1e-7, resolutionLow=4e-6, resolutionHigh=4e-6,
)

s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=15.0 * planet.R, x_steps=50,
    rho_border=planet.hostStar.R, rho_steps=150, phi_steps=40,
    orbphase_border=0.0, orbphase_steps=1,
)

atm = gasprop.HydrostaticAtmosphere(T_ATM, P_0, MU, planet)
for name, chi in MOLECULES:
    atm.addMolecularConstituent(name, chi)
    # Passing wg slices the cross-section table to the sim range (faster).
    atm.constituents[-1].addLookupFunctionToConstituent(wg)

sim = gasprop.Transit(
    gasprop.Atmosphere([atm], hasOrbitalDopplerShift=False), wg, s_grid)
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.45)[0]

wav_um   = sim.wavelength * 1e4
depth_ppm = (1.0 - R) * 1e6
# Diagnostic bands: H2O 1.4/1.9 µm, CO2 4.3 µm, CO 4.7 µm, SO2 4.0 µm.
```

**Notes**

- Use `addMolecularConstituent` (not `addConstituent`). The molecule name must match an HDF5 file in `Resources/molecularResources/` (e.g. `H2O.h5`).
- Passing `wg` to `addLookupFunctionToConstituent` slices the `(P, T, λ)` table to the simulation range plus a small Doppler margin — a large speedup for narrow grids.
- To inspect each species' contribution, build a separate atmosphere with a single molecule and re-run (as `fig8` does).

---

## 3. Io-analogue Na torus (exosphere)

*Based on `fig3_io_torus-3.py`.* A circumplanetary neutral-Na gas torus around WASP-49 b, producing a Na D transmission signature. Exospheres are normalized by a **total particle number** `N`, and their single atomic constituent takes a **velocity dispersion** (cm/s), not a mixing ratio.

```python
planet = bodies.AvailablePlanets().findPlanet('WASP-49b')

R_J     = const.R_J
A_TORUS = 5.9 * R_J     # Io-like orbital distance (cm)
V_EJ    = 1e5           # ejection velocity (cm/s) — sets torus scale height
N_TORUS = 1e33          # total Na atoms
SIGMA_V = 2e6           # thermal velocity dispersion (cm/s)

# Narrow grid around the Na D doublet (~5890 Å = 5.89e-5 cm).
wg = gasprop.WavelengthGrid(
    lower_w=5.880e-5, upper_w=5.910e-5,
    widthHighRes=4e-8, resolutionLow=3e-9, resolutionHigh=2e-10,
)

s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=12.0 * planet.R, x_steps=28,
    rho_border=planet.hostStar.R, rho_steps=22, phi_steps=30,
    orbphase_border=0.0, orbphase_steps=1,
)

torus = gasprop.TorusExosphere(N=N_TORUS, a_torus=A_TORUS,
                               v_ej=V_EJ, planet=planet)
torus.addConstituent('NaI', SIGMA_V)        # 2nd arg is sigma_v (cm/s), not chi
torus.constituents[-1].addLookupFunctionToConstituent(wg)

# Torus kinematics matter → enable orbital Doppler shifting.
atmos = gasprop.Atmosphere([torus], hasOrbitalDopplerShift=True)
sim = gasprop.Transit(atmos, wg, s_grid)
sim.addWavelength()

# Use a realistic stellar spectrum (call AFTER addWavelength).
planet.hostStar.addFstarFunction(sim.wavelength)

R = sim.sumOverChords(max_memory_gb=4.0)[0]

wav_ang = sim.wavelength * 1e8           # cm → Å
# Continuum-normalize to read excess absorption:
cont = np.median(R[(wav_ang < 5884) | (wav_ang > 5902)])
excess_pct = (R / cont - 1.0) * 100.0
```

**Notes**

- For an exosphere, `addConstituent(speciesName, sigma_v)` takes a velocity dispersion in cm/s; the absolute amount of gas is set by `N`.
- `hasOrbitalDopplerShift=True` is appropriate here because the line shape and the orbital velocity matter at this resolution.
- `addFstarFunction` requires PHOENIX spectra to be available; omit it (flat star) if you only need the line shape relative to a normalized continuum.

---

## 4. Combining a lower atmosphere with an escaping wind

An `Atmosphere` can hold several density models at once. This pattern layers a hazy hydrostatic lower atmosphere beneath a radially escaping Parker wind that carries a trace Na absorber (cf. the wind physics in `fig6_atmospheric_escape-6.py`).

```python
planet = bodies.AvailablePlanets().findPlanet('HD189733b')

NA_MASS = 22.99 * const.amu     # tracer mass (density normalization)
MU_BULK = 1.3  * const.amu      # bulk H/He gas mass (Parker dynamics)

wg = gasprop.WavelengthGrid(
    lower_w=5.884e-5, upper_w=5.896e-5,
    widthHighRes=5e-8, resolutionLow=3e-9, resolutionHigh=1.2e-10,
)
s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=40.0 * planet.R, x_steps=55,
    rho_border=planet.hostStar.R, rho_steps=28, phi_steps=28,
    orbphase_border=0.0, orbphase_steps=1,
)

# Escaping wind: exact isothermal Parker solution. mu sets the tracer density
# normalization; wind_mu sets the (bulk-gas) wind dynamics.
wind = gasprop.RadialWindExosphere(
    Mdot=1.0e3, mu=NA_MASS, wind_model='parker', T=9.0e3, wind_mu=MU_BULK,
    r_inner=2.0 * planet.R, r_outer=40.0 * planet.R, planet=planet,
)
wind.addConstituent('NaI', 1e6)             # sigma_v in cm/s
wind.constituents[-1].addLookupFunctionToConstituent(wg)

# Hazy hydrostatic lower atmosphere providing a continuum floor.
haze_atm = gasprop.HydrostaticAtmosphere(T=2500.0, P_0=1e6,
                                         mu=2.3 * const.amu, planet=planet)
haze_atm.addScatteringConstituent(
    'RayleighHaze',
    {'chi': 1.0, 'sigma_ref': 1e-19, 'lambda_ref': 4000e-8, 'slope': 4.0})

# Both models in one atmosphere. Position-dependent wind Doppler needs the flag.
atmos = gasprop.Atmosphere([wind, haze_atm], hasOrbitalDopplerShift=True)
sim = gasprop.Transit(atmos, wg, s_grid)
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.8)[0]
```

**Notes**

- `wind_model='parker'` requires `T` and `planet`; `v_terminal`/`beta` are unused. `wind_mu` separates the wind *dynamics* (bulk gas) from the tracer *density* normalization (`mu`).
- With `hasOrbitalDopplerShift=True`, the wind's per-cell line-of-sight velocity produces the blue/red line asymmetry and kinematic broadening characteristic of escape.
- The list order in `Atmosphere([...])` does not matter physically; optical depths from all models are summed.

---

## Producing a light curve or phase-resolved map

Any of the above becomes a light curve by setting `orbphase_steps > 1` and a non-zero `orbphase_border` on the `Grid`. `sumOverChords` then returns `R` with shape `(orbphase_steps, n_wavelength)`:

```python
N_PHASES = 21
orbphase_border = (2.6 / (planet.orbitalPeriod * 24.0)) * 2.0 * np.pi   # ±2.6 h
s_grid = geom.Grid(
    x_midpoint=planet.a, x_border=40.0 * planet.R, x_steps=55,
    rho_border=planet.hostStar.R, rho_steps=28, phi_steps=28,
    orbphase_border=orbphase_border, orbphase_steps=N_PHASES,
)
# ... build sim as before ...
R_2D = sim.sumOverChords(max_memory_gb=0.8)     # (N_PHASES, n_wavelength)
phases_rad = s_grid.constructOrbphaseAxis()
```

Each row `R_2D[i]` is the spectrum at orbital phase `phases_rad[i]`; integrating a row over a line bandpass yields a light-curve point, and stacking rows gives a velocity-vs-phase transmission map.

---

## 5. Resonance-scattering emission from an exomoon cloud

Everything above solves pure extinction. Passing an `EmissionModel` adds the
photons the gas scatters *back* into the beam: in transit they partially refill
the line core, and outside the stellar limb the cloud glows against the dark
sky. See [architecture.md](architecture.md#emission) for the radiative transfer
and its stated approximations.

```python
import core.emission as emis

planet = bodies.AvailablePlanets().findPlanet('WASP-49b')
moon   = bodies.Moon(midTransitOrbphase=0.0, R=const.R_Io,
                     a=5.0 * planet.R, hostPlanet=planet)

wg   = gasprop.na_d_grid(5885.0, 5905.0)
scen = gasprop.moon_exosphere_scenario(N=5e33, q=3.34, moon=moon,
                                       species='NaI', sigma_v=2e6,
                                       wavelengthGrid=wg)

# Widen the sky plane past the stellar limb so the off-limb glow is captured.
# Chords outside the limb receive no photospheric flux, so the transmission
# spectrum is unaffected -- they only add emission.
sg = geom.spatial_grid(planet, rho_steps=120, phi_steps=40,
                       rho_border=2.0 * planet.hostStar.R)

model = emis.EmissionModel(resonant_scattering=True)
res = gasprop.run_transit([scen], wg, sg, use_phoenix_star=True,
                          emission=model)

depth = res.transit_depth() * 100.0          # % excess absorption, net of fill-in
glow  = res.emission_spectrum().max()        # peak emission, in stellar-flux units
print(f"{depth:.3f} % absorption, {glow*1e6:.1f} ppm peak emission, "
      f"{res.fill_in_fraction()*100:.2f} % of the absorption refilled")
```

For a hot Jupiter the fill-in is small — the star subtends only
`W = (R_star/a)²/4 ≈ 4e-3` of the sky at the planet, so scattering returns only
~0.03% of what absorption removes for the Io-like cloud above (measured in
`Tests/EmissionCalibration/fig_emission_spectra.py`). That is a useful result in
itself: it is the quantitative justification for treating a transit as pure
absorption. The emission term matters where the denominator is small — off the
stellar disk, out of transit, and at high spectral resolution.

To see the two effects separately:

```python
atm = gasprop.Atmosphere([scen], hasOrbitalDopplerShift=True, emission=model)
sim = gasprop.Transit(atm, wg, sg); sim.addWavelength()
sim.planet.hostStar.addFstarFunction(sim.wavelength, extra_velocity=1e7)
parts = sim.sumOverChords(return_components=True)
# parts['R'] == parts['transmission'] + parts['emission']
```

**The stellar line trap.** With `stellar_doppler=True` (the default) each parcel
samples the stellar spectrum in *its own* frame. Gas at rest relative to the
star sits inside the stellar Na D core and is barely illuminated; gas that has
been Doppler-shifted out of the core sees the full continuum and scatters far
more. The `g_factor` diagnostic makes this concrete:

```python
c = scen.constituents[0]
star = planet.hostStar
star.addFstarFunction(np.array([5885e-8, 5905e-8]), extra_velocity=2e6)
for v_kms in (0.0, 5.0, 15.0):
    g = emis.g_factor(c, star, planet.a, v_radial=v_kms * 1e5)
    print(f"v_rad = {v_kms:5.1f} km/s -> g = {g:8.2f} photons/s/atom")
```

Only `RadialWindExosphere` supplies a star-radial velocity field to the transfer
(`calculateRadialVelocityFromStar`); other models are treated as moving
perpendicular to the star direction, which is exact for a circular orbit.

**Thermal emission** is the other source term, off by default:

```python
model = emis.EmissionModel(resonant_scattering=False, thermal=True)
```

It uses `B_λ(T)` from the density model's temperature field, so only the
`CollisionalAtmosphere` family contributes. At optical wavelengths against a
main-sequence host it is negligible; it becomes relevant in the infrared and for
hot, dense gas.

---

## 6. Secondary eclipse of a lava planet

Emission's other observable: the planet's own dayside flux, `F_p/F_*`, rather
than the starlight it blocks. This uses a planet-centred ray grid
(`eclipse.py`) instead of the star-centred chord grid.

```python
import core.eclipse as ecl

planet = bodies.AvailablePlanets().findPlanet('55-Cancri-e')
star   = planet.hostStar

# The PHOENIX HiRes grid stops near 5.5 um, so attach a spectrum that reaches
# the mid-IR.  Skipping this falls back to a blackbody at T_eff, which
# over-predicts a G8V photosphere at 6-12 um by ~13%.
w_cm, I_cgs = np.loadtxt('btsettl.csv', delimiter=',', skiprows=1, unpack=True)
star.addFstarFunctionFromArrays(w_cm, I_cgs)

wav = np.linspace(6.0e-4, 12.0e-4, 400)          # 6-12 um, in cm
e = ecl.Eclipse(planet, wav,
                surface=ecl.DaysideSurface(T_day=2000.0, profile='uniform'))

print(e.bandDepth(6.3e-4, 11.8e-4) * 1e6, 'ppm')   # band-averaged, ppm
```

Going the other way — turning a *measured* eclipse depth into the temperature a
model has to reproduce:

```python
T_b = ecl.band_brightness_temperature(planet, wav, 110e-6, 6.3e-4, 11.8e-4)
```

Add gas above the surface by passing an `Atmosphere` built with an
`EmissionModel`; rays then carry
`epsilon·B_λ(T_surf)·exp(-tau_above) + I_gas`, and rays outside the solid body
see only the limb:

```python
atmos = gasprop.Atmosphere([layer], hasOrbitalDopplerShift=False,
                           emission=emis.EmissionModel(resonant_scattering=False,
                                                       thermal=True,
                                                       aerosol_albedo=0.0))
e = ecl.Eclipse(planet, wav, surface=surface, atmosphere=atmos,
                R_top=1.2 * planet.R)
```

Note `aerosol_albedo=0.0`: aerosol extinction is split by the albedo into a
scattering share and a true-absorption share, and only the absorbing share
emits thermally. A grey opacity meant as an absorber needs albedo 0, or it will
remove flux without putting any back.

The dayside temperature is an input here, not a prediction — Prometheus has no
energy-balance solver. `Tests/EmissionCalibration/` runs this against the
JWST/MIRI eclipse of 55 Cnc e and works through the error budget.
