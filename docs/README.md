# Prometheus

**PRO**bing **M**ass loss in **E**xoplanetary **T**ransits with **H**ydrostatic, **E**vaporative, and **U**ser-defined **S**cenarios.

Prometheus is a forward radiative-transfer code for **transmission spectroscopy** of transiting bodies. Given a planet (or moon), a model for how absorbing gas is distributed around it, and a wavelength range, Prometheus computes the wavelength-dependent transit depth — the fraction of stellar flux removed from the beam as starlight passes through the intervening gas.

It is designed for the regime where the absorbing medium is *not* limited to a dense hydrostatic atmosphere: outgassed exomoon clouds, circumplanetary tori, and radially escaping planetary winds are all first-class density models, alongside the canonical barometric/hydrostatic atmosphere.

---

## What it computes

For a given orbital phase and wavelength grid, Prometheus integrates the Beer–Lambert optical depth along every line of sight (chord) that crosses the stellar disk, attenuates the local stellar surface flux, and sums over the disk:

```
R(λ, orbphase) = Σ_chords F_in(λ) / Σ_chords F_out(λ)
```

where `F_out = ρ · F_star` is the unobscured stellar flux for a chord and `F_in = F_out · exp(−τ)` is the transmitted flux. The result `R` is the in-transit/out-of-transit flux ratio; `1 − R` is the excess absorption (transit depth) at each wavelength. Running over an array of orbital phases produces a transit light curve or a phase-resolved transmission map.

The opacity along each chord can combine:

- **Atomic / ionic line absorption** — Voigt profiles built from NIST line lists (Na I, K I, Fe I, Mg I/II, Ca I/II, Ti I/II, and more).
- **Molecular band absorption** — pre-computed ExoMOL/HITEMP cross-section tables (functions of pressure, temperature, and wavelength) for species such as H₂O, CO₂, CO, and SO₂.
- **Continuum scattering / aerosols** — parametrized Rayleigh haze, gray cloud decks, Ångström power-law aerosols, and tabulated Mie cross-sections.

Doppler shifts from the planet's orbital motion (bulk) and from radial-wind kinematics (position-dependent, per line-of-sight cell) are applied to line absorbers, so the code captures the velocity broadening and blue/red asymmetry of escaping atmospheres.

---

## Density models

| Model | Class | Use case |
|---|---|---|
| Barometric atmosphere | `BarometricAtmosphere` | Isothermal exponential profile |
| Hydrostatic atmosphere | `HydrostaticAtmosphere` | Hydrostatic equilibrium (with Jeans term) |
| Power-law atmosphere | `PowerLawAtmosphere` | Tenuous power-law profile |
| Power-law exosphere | `PowerLawExosphere` | Particle-normalized power-law cloud |
| Moon exosphere | `MoonExosphere` | Outgassed cloud sourced from an orbiting moon |
| Tidally-heated moon | `TidallyHeatedMoon` | Phase-dependent volcanic source rate |
| Torus exosphere | `TorusExosphere` | Io-analogue circumplanetary gas torus |
| SERPENS exosphere | `SerpensExosphere` | Density interpolated from SERPENS particle output |
| Radial-wind exosphere | `RadialWindExosphere` | Escaping wind (β-law or exact isothermal Parker) |

Multiple density models can be combined in a single simulation (e.g. a hydrostatic lower atmosphere beneath an escaping wind).

---

## Who it's for

Prometheus is built for exoplanet and planetary-science researchers who need a fast, scriptable forward model for transmission spectra — whether to interpret high-resolution alkali-line observations, fit JWST molecular spectra, or model the kinematic signatures of atmospheric escape and volcanic exomoons.

The API is **fully programmatic**: you build a simulation by constructing Python objects directly — a planet, one or more density models, a wavelength grid, and a spatial grid — and call a method to run it. There is no configuration-file layer and no command-line wizard; everything is a Python object you can introspect, subclass, and embed in a larger pipeline (parameter sweeps, retrievals, MCMC likelihoods).

---

## A minimal run

```python
import pythonScripts.gasProperties as gasprop
import pythonScripts.celestialBodies as bodies
import pythonScripts.geometryHandler as geom
import pythonScripts.constants as const

planet = bodies.AvailablePlanets().findPlanet('WASP-39b')

wg = gasprop.WavelengthGrid(0.35e-4, 1.0e-4,
                            widthHighRes=8e-8,
                            resolutionLow=4e-7, resolutionHigh=1e-9)

atm = gasprop.HydrostaticAtmosphere(T=1100.0, P_0=1e4,
                                    mu=2.3 * const.amu, planet=planet)
atm.addConstituent('NaI', 3e-7)
atm.constituents[-1].addLookupFunctionToConstituent(wg)

s_grid = geom.Grid(x_midpoint=planet.a, x_border=14.0 * planet.R, x_steps=80,
                   rho_border=planet.hostStar.R, rho_steps=400, phi_steps=20,
                   orbphase_border=0.0, orbphase_steps=1)

sim = gasprop.Transit(gasprop.Atmosphere([atm], hasOrbitalDopplerShift=False),
                      wg, s_grid)
sim.addWavelength()
R = sim.sumOverChords(max_memory_gb=0.5)[0]

transit_depth_ppm = (1.0 - R) * 1e6   # vs sim.wavelength (cm)
```

---

## Documentation map

- **[getting-started.md](getting-started.md)** — installation, dependencies, and a complete first run.
- **[architecture.md](architecture.md)** — how the modules fit together and how a spectrum is computed (data flow, the Numba optical-depth kernel, memory-aware batching, radial-wind physics).
- **[api-reference.md](api-reference.md)** — constructor signatures and methods for every public class.
- **[examples.md](examples.md)** — annotated worked examples: hydrostatic alkali spectrum, molecular JWST spectrum, and an Io-analogue torus.
- **[contributing.md](contributing.md)** — development setup and code style.

---

## License

GPL-3.0. See [LICENSE](../LICENSE).
