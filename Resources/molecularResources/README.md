# Molecular opacity tables

Prometheus reads molecular absorption from ExoMol cross-section tables in this
directory (`MolecularConstituent`, `pythonScripts/gasProperties.py`). Each
`<molecule>.h5` holds sigma(P, T, lambda) at **R = 15000** over **0.3–50 µm**,
on a **22-point pressure** grid (1e-5–100 bar) × **27-point temperature** grid
(100–3400 K). These are **cross sections**, not correlated-k tables.

Provenance: **ExoMolOP** (Chubb et al. 2021, A&A 646, A21), TauREx `.h5`
format, downloaded verbatim from [exomol.com](https://www.exomol.com). Line
lists per molecule: H2O POKAZATEL, CO Li2015, CO2 UCL-4000, CH4 MM,
SO2 ExoAmes, H2S AYT2, SiO2 OYT3.

## The files are not in git

Each file is ~365 MB (~2.5 GB total), so `*.h5` is git-ignored and fetched on
demand:

```bash
source ../../../env/bin/activate     # for h5py validation (optional but recommended)
python fetch_opacities.py            # download any missing/invalid tables
python fetch_opacities.py --verify-only
python fetch_opacities.py --list
python fetch_opacities.py --only H2O,CO --force
```

The fetcher validates each download structurally (HDF5 keys, `mol_name`,
R≈15000, P/T grid shapes) and records a SHA-256 in `checksums.sha256` on first
success; later runs verify against it. Downloads resume if interrupted.

## Adding a molecule

- **On ExoMolOP already?** Add an entry to `MANIFEST` in `fetch_opacities.py`
  (molecule, iso slug, dataset, remote filename, size) and re-run.
- **Not on ExoMolOP?** Generate the table with **ExoCross**
  (Yurchenko et al. 2018) from the ExoMOL line list, using the same
  `R15000_0.3-50mu` grid, drop the `.h5` here named `<molecule>.h5`, and add a
  `MANIFEST` entry (URL unused for locally generated files).
