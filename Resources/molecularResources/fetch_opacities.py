#!/usr/bin/env python3
"""Fetch the ExoMol molecular opacity cross-section tables Prometheus needs.

The ``*.h5`` files in this directory are ExoMolOP (Chubb et al. 2021,
A&A 646, A21) TauREx-format cross sections: absorption cross section
sigma(P, T, lambda) sampled at R = lambda/dlambda = 15000 over 0.3-50 um,
on a 22-point pressure grid (1e-5 .. 100 bar) x 27-point temperature grid
(100 .. 3400 K).  They are ~365 MB each (~2.5 GB total), which is too large
to commit, so they are git-ignored and fetched on demand with this script.

Source of truth
---------------
Files are downloaded verbatim from the ExoMol database.  The URL template
(ExoMolOP paper, Sect. 7.2) is::

    https://www.exomol.com/db/<molecule>/<iso>/<dataset>/<filename>

The exact filenames were verified against the live server; each remote
Content-Length matches the file size recorded in ``MANIFEST`` below.  Note
the water file uses ``__R15000`` (double underscore) where every other
molecule uses ``.R15000`` -- this asymmetry is on the ExoMol side.

Prometheus loads these as ``<molecule>.h5`` (see MolecularConstituent in
``core/gasProperties.py``), so each download is saved under its
short molecule name, e.g. ``1H2-16O__POKAZATEL...h5`` -> ``H2O.h5``.

Usage
-----
    python fetch_opacities.py                 # fetch any missing/invalid file
    python fetch_opacities.py --only H2O,CO   # subset
    python fetch_opacities.py --force         # re-download even if present
    python fetch_opacities.py --verify-only   # validate on disk, download nothing
    python fetch_opacities.py --list          # print the manifest and exit

Validation is structural (HDF5 keys, mol_name, R~15000, P/T grid shapes)
rather than a hardcoded checksum: the first successful download of each file
records its SHA-256 in ``checksums.sha256`` next to the data, and later runs
verify against that recorded hash.  No checksum is invented by this script.

To ADD a molecule not on ExoMolOP, generate the table yourself with ExoCross
(Yurchenko et al. 2018) from the ExoMOL line list using the same
R15000_0.3-50mu grid, drop the ``.h5`` in here, and add a MANIFEST entry.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://www.exomol.com/db"
CHECKSUM_FILE = os.path.join(HERE, "checksums.sha256")

# molecule -> (exomol_molecule, iso_slug, dataset, remote_filename, expected_bytes)
# expected_bytes are the live-server Content-Length values (verified 2026-07-03);
# treated as advisory -- an ExoMol re-release may change them without the data
# being wrong, so a size mismatch warns but structural validation decides.
MANIFEST: dict[str, tuple[str, str, str, str, int]] = {
    "H2O":  ("H2O",  "1H2-16O",  "POKAZATEL", "1H2-16O__POKAZATEL__R15000_0.3-50mu.xsec.TauREx.h5", 365310072),
    "CO":   ("CO",   "12C-16O",  "Li2015",    "12C-16O__Li2015.R15000_0.3-50mu.xsec.TauREx.h5",     365310264),
    "CO2":  ("CO2",  "12C-16O2", "UCL-4000",  "12C-16O2__UCL-4000.R15000_0.3-50mu.xsec.TauREx.h5",  365310264),
    "CH4":  ("CH4",  "12C-1H4",  "MM",        "12C-1H4__MM.R15000_0.3-50mu.xsec.TauREx.h5",         365310264),
    "SO2":  ("SO2",  "32S-16O2", "ExoAmes",   "32S-16O2__ExoAmes.R15000_0.3-50mu.xsec.TauREx.h5",   365310264),
    "H2S":  ("H2S",  "1H2-32S",  "AYT2",      "1H2-32S__AYT2.R15000_0.3-50mu.xsec.TauREx.h5",       365310264),
    "SiO2": ("SiO2", "28Si-16O2", "OYT3",     "28Si-16O2__OYT3.R15000_0.3-50mu.xsec.TauREx.h5",     365310264),
}


def url_for(mol: str) -> str:
    exomol, iso, dataset, fname, _ = MANIFEST[mol]
    return f"{BASE}/{exomol}/{iso}/{dataset}/{fname}"


def dest_for(mol: str) -> str:
    return os.path.join(HERE, f"{mol}.h5")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums() -> dict[str, str]:
    out: dict[str, str] = {}
    if os.path.exists(CHECKSUM_FILE):
        with open(CHECKSUM_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                digest, name = line.split(None, 1)
                out[name.strip()] = digest
    return out


def record_checksum(mol: str, digest: str) -> None:
    sums = load_checksums()
    sums[f"{mol}.h5"] = digest
    with open(CHECKSUM_FILE, "w") as f:
        f.write("# SHA-256 of ExoMol opacity files, recorded on first successful fetch.\n")
        f.write("# Regenerate a line by deleting it and re-running fetch_opacities.py.\n")
        for name in sorted(sums):
            f.write(f"{sums[name]}  {name}\n")


def validate(mol: str, path: str, expected_bytes: int) -> tuple[bool, str]:
    """Structural + checksum validation. Returns (ok, message)."""
    if not os.path.exists(path):
        return False, "missing"
    size = os.path.getsize(path)
    notes = []
    if size != expected_bytes:
        notes.append(f"size {size} != expected {expected_bytes}")

    # Recorded-checksum check (authoritative once established).
    recorded = load_checksums().get(f"{mol}.h5")
    if recorded is not None:
        if sha256_of(path) != recorded:
            return False, f"sha256 mismatch vs checksums.sha256 ({'; '.join(notes) or 'ok size'})"

    # Structural check via h5py if available; otherwise fall back to size only.
    try:
        import h5py
        import numpy as np
    except ImportError:
        msg = "size-only (h5py unavailable)"
        if recorded is not None:
            msg = "sha256 OK; " + msg
        return True, msg + (f"; WARN {'; '.join(notes)}" if notes else "")

    try:
        with h5py.File(path, "r") as f:
            required = {"xsecarr", "bin_edges", "p", "t", "mol_name"}
            missing = required - set(f.keys())
            if missing:
                return False, f"HDF5 missing datasets: {sorted(missing)}"
            name = f["mol_name"][()][0]
            name = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            if name != MANIFEST[mol][0]:
                return False, f"mol_name '{name}' != '{MANIFEST[mol][0]}'"
            xs = f["xsecarr"].shape
            if len(xs) != 3 or xs[0] != len(f["p"]) or xs[1] != len(f["t"]):
                return False, f"xsecarr shape {xs} inconsistent with p/t grids"
            be = f["bin_edges"][:]
            R = np.median(be[1:] / np.diff(be))
            if not (14000 < R < 16000):
                return False, f"resolution R~{R:.0f} not ~15000 (truncated?)"
    except Exception as exc:  # noqa: BLE001 -- report any HDF5 read failure
        return False, f"HDF5 read error: {exc}"

    tag = "sha256 OK" if recorded is not None else "structure OK"
    return True, tag + (f"; WARN {'; '.join(notes)}" if notes else "")


def download(mol: str) -> None:
    url = url_for(mol)
    dest = dest_for(mol)
    part = dest + ".part"
    tool = shutil.which("curl") or shutil.which("wget")
    if tool is None:
        raise RuntimeError("neither curl nor wget found on PATH")
    print(f"  downloading {mol}  <-  {url}")
    if "curl" in tool:
        cmd = [tool, "-L", "--fail", "--retry", "5", "--retry-delay", "3",
               "--retry-connrefused", "-C", "-", "-o", part, url]
    else:  # wget
        cmd = [tool, "-c", "-O", part, url]
    subprocess.run(cmd, check=True)
    os.replace(part, dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch ExoMol opacity tables for Prometheus.")
    ap.add_argument("--only", help="comma-separated molecule subset, e.g. H2O,CO")
    ap.add_argument("--force", action="store_true", help="re-download even if present and valid")
    ap.add_argument("--verify-only", action="store_true", help="validate files on disk; download nothing")
    ap.add_argument("--list", action="store_true", help="print the manifest (molecule -> URL) and exit")
    args = ap.parse_args(argv)

    if args.list:
        for mol in MANIFEST:
            print(f"{mol:5s} {url_for(mol)}")
        return 0

    mols = list(MANIFEST)
    if args.only:
        mols = [m.strip() for m in args.only.split(",") if m.strip()]
        unknown = [m for m in mols if m not in MANIFEST]
        if unknown:
            ap.error(f"unknown molecule(s): {unknown}. Known: {list(MANIFEST)}")

    failures = 0
    for mol in mols:
        dest = dest_for(mol)
        expected = MANIFEST[mol][4]
        ok, msg = validate(mol, dest, expected)

        if args.verify_only:
            print(f"[{'OK ' if ok else 'BAD'}] {mol:5s} {msg}")
            failures += 0 if ok else 1
            continue

        if ok and not args.force:
            print(f"[skip] {mol:5s} present, {msg}")
            continue

        try:
            download(mol)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"[FAIL] {mol:5s} download error: {exc}", file=sys.stderr)
            failures += 1
            continue

        ok, msg = validate(mol, dest, expected)
        if not ok:
            print(f"[FAIL] {mol:5s} downloaded but invalid: {msg}", file=sys.stderr)
            failures += 1
            continue
        if load_checksums().get(f"{mol}.h5") is None:
            digest = sha256_of(dest)
            record_checksum(mol, digest)
            print(f"[ OK ] {mol:5s} {msg}; recorded sha256 {digest[:12]}...")
        else:
            print(f"[ OK ] {mol:5s} {msg}")

    if failures:
        print(f"\n{failures} file(s) failed.", file=sys.stderr)
        return 1
    print("\nAll requested opacity files present and valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
