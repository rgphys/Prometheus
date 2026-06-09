# Prometheus Documentation

**PRO**bing **M**ass loss in **E**xoplanetary **T**ransits with **H**ydrostatic,
**E**vaporative and **U**ser-defined **S**cenarios.

Prometheus is a forward radiative-transfer tool that computes transmission
spectra and light curves of an object — typically an exoplanet — transiting its
host star. The code integrates the line-of-sight absorption produced by a
gaseous medium of arbitrary geometry and returns the wavelength- and
orbital-phase-resolved transit depth.

This `docs/` folder is the reference manual for the code base. It is written for
a mixed physics / scientific-computing audience: it assumes familiarity with
exoplanet transit observations and basic radiative transfer, but explains the
software architecture and APIs in full.

---

## What Prometheus does

Given a description of a star, a transiting body, and a model for the gas around
that body, Prometheus computes

```
R(orbital phase, wavelength) = F_in / F_out
```

the ratio of the in-transit stellar flux (attenuated by the intervening gas) to
the out-of-transit flux. `R = 1` means full transmission; `R < 1` means
absorption. The result is a 2-D grid over orbital phase and wavelength from
which both **transmission spectra** (fixed phase, varying wavelength) and
**light curves** (fixed wavelength band, varying phase) can be extracted.

The absorbing medium is not restricted to a classical hydrostatic atmosphere.
Prometheus supports tenuous, non-spherical and dynamically expanding
distributions that are typical of mass-loss studies:

- dense hydrostatic / barometric atmospheres,
- power-law exospheres,
- an exomoon-sourced neutral gas cloud,
- a circumplanetary neutral gas torus,
- particle distributions imported from the **SERPENS** Monte-Carlo code,
- a radially expanding wind (parametrized beta-law **or** an exact isothermal
  Parker-wind solution).

Absorption is computed for atomic and ionic lines (Voigt profiles from NIST
line lists), molecular bands (pre-computed ExoMOL-style cross-section tables),
and several continuum scattering / aerosol sources.

---

## Why it exists

Atmospheric escape and exospheres leave their fingerprint on high-resolution
transmission spectra: sodium and potassium doublets, metastable helium, metal
lines, and broad scattering slopes. Interpreting these signals requires a
forward model that can:

1. Handle **non-hydrostatic, non-spherical** geometries (clouds, tori, winds)
   that a 1-D plane-parallel atmosphere code cannot represent.
2. Resolve **individual spectral lines** with proper Voigt profiles and
   Doppler shifts from bulk orbital motion and position-dependent wind
   velocities.
3. Run fast enough that it can be embedded in retrievals.

Prometheus targets exactly this niche. It originated as a research code in the
exoplanet group of Andrea Gebek and has since been extended with additional
scenarios (radial winds), continuum opacity sources (haze, clouds, aerosols),
vectorized / Numba-accelerated kernels, memory-aware batching, and a
post-processing shot-noise module.

---

## High-level architecture

A run flows through four layers:

```
            ┌─────────────────────────────────────────────┐
   setup    │  setup.py  →  JSON setup file (setupFiles/)  │
            └─────────────────────────────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
   build    │  prometheus.py  parses JSON, constructs:     │
            │    • Planet / Star            (celestialBodies)
            │    • WavelengthGrid, Grid     (gasProperties, geometryHandler)
            │    • scenario objects + constituents (gasProperties)
            │    • Atmosphere, Transit      (gasProperties)
            └─────────────────────┬───────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
  compute   │  Transit.sumOverChords()                     │
            │    • chord grid → memory-sized batches       │
            │    • Atmosphere.getLOSopticalDepth_Batch()   │
            │    • Beer–Lambert: F_in = F_out · exp(−τ)    │
            └─────────────────────┬───────────────────────┘
                                  │
            ┌─────────────────────▼───────────────────────┐
  output    │  output/<setup>.txt  (phases + R grid)       │
            │  optional: shotNoise.py post-processing      │
            └──────────────────────────────────────────────┘
```

The geometry is a ray-tracing scheme: the star is at the origin, the observer at
`x = −∞`, and the sky plane is the `(y, z)` plane parametrized in polar
coordinates `(rho, phi)`. Each `(rho, phi)` pair defines a **chord** — a line of
sight through the system — and the code integrates the gas number density along
`x` for every chord and every orbital phase.

---

## Documentation map

| Document | Contents |
|---|---|
| [getting-started.md](getting-started.md) | Installation, dependencies, directory layout, first run, output format |
| [architecture.md](architecture.md) | Coordinate system, data flow, the batched optical-depth kernel, design decisions |
| [modules.md](modules.md) | Per-module / per-class / per-function reference |
| [api.md](api.md) | Programmatic API: building a simulation in Python, the setup-file JSON schema, the shot-noise API |
| [contributing.md](contributing.md) | Branching, code style, adding scenarios/opacity sources, testing, PR process |

---

## Quick reference

```bash
# 1. interactively build a setup file (writes ../setupFiles/<name>.txt)
python prometheus.py setup

# 2. run the forward model (reads ../setupFiles/<name>.txt,
#    writes ../output/<name>.txt)
python prometheus.py <name>

# 3. optional: cap RAM (GB) used by the chord batching
python prometheus.py <name> --max-memory 4.0
```

---

## License

Prometheus is distributed under the **GNU General Public License v3.0**. See the
[`LICENSE`](../LICENSE) file at the repository root.
