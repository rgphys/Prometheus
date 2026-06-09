"""Generate aerosol extinction cross-section tables from REAL optical constants.

For each aerosol species this reads a published complex refractive index
(n, k)(lambda) table from ``optical_constants/`` and runs Mie theory
(``miepython``) over a log-normal particle-size distribution to produce a
per-particle extinction cross-section sigma_ext(lambda) [cm^2].  The output is
one CSV per species (two columns: wavelength [Angstrom], sigma [cm^2]; '#'
comment header) — the same format consumed by Prometheus' ``TabulatedAerosol``
constituent and the ``aerosol_zoo.py`` overview figure.

These are NOT illustrative analytic shapes: the spectral character of every
curve (slope sign, plateau, absorption bands) comes directly from laboratory /
literature optical constants, with only the particle size assumed.

Optical-constant sources (file -> reference):
  * H2SO4    H2SO4_75_Palmer_1975.ri   75 wt% sulphuric acid, Palmer & Williams
             (1975), Appl. Opt. 14, 208 (Venus cloud droplets).  [Oxford ARIA]
  * water_ice ICE_Warren_2008.ri        Warren & Brandt (2008), JGR 113, D14220
             (Earth cirrus ice).  [Oxford ARIA]
  * tholin   Titan_tholin.dat          Khare et al. (1984), Icarus 60, 127
             (Titan organic haze analogue).  [LX-MIE, Kitzmann & Heng 2018]
  * ch4_haze CH4(s)_Martonchik.csv      solid CH4, Martonchik et al. (1994)
             (Neptune stratospheric condensate haze).  [VIRGA]
  * mars_dust sand_koepke_1997.ri       quartz + clay mineral dust, Koepke et al.
             (1997) GADS/OPAC (Mars silicate-dust analogue).  [Oxford ARIA]
  * silicate astrosil_D03.txt           Draine (2003) astronomical silicate.
             [B. Draine, Princeton]

Per-species particle effective radius r_eff (geometric-median of the log-normal,
geometric width sigma_g) sets the dominant size mode:
  H2SO4 ~1 um droplets; water ice ~12 um crystals; tholins ~0.05 um aggregates;
  CH4 haze ~0.1 um; fine silicate dust ~0.25 um (Mars-analogue reddening mode);
  astro-silicate ~0.2 um (ISM grain).  The two large gray modes (droplets, ice)
  and the sub-micron reddening/blue modes bracket the optical regimes.
"""

import os
import numpy as np
import miepython

HERE = os.path.dirname(os.path.abspath(__file__))
OC = os.path.join(os.path.dirname(HERE), 'optical_constants')

# Output wavelength grid: 0.30-5.0 um, log-spaced (Angstrom).
lam_um = np.geomspace(0.30, 5.0, 80)
lam_A = lam_um * 1e4

UM_TO_CM = 1e-4            # 1 um in cm
CM2_PER_UM2 = UM_TO_CM ** 2   # um^2 -> cm^2


#  refractive-index loaders (one per native file format) 
def _read_lines(path):
    """Read a possibly latin-1 / CRLF text file as a list of clean lines."""
    with open(path, encoding='latin-1') as f:
        return [ln.rstrip('\r\n') for ln in f]


def load_aria(path):
    """Oxford ARIA '.ri': '#'-comment header with a '#FORMAT=WAVN|WAVL N K' line.

    WAVN = wavenumber [cm^-1] (-> wavelength = 1e4 / wavn um); WAVL = wavelength
    [um].  Returns (wl_um, n, k) sorted by ascending wavelength.
    """
    fmt = None
    rows = []
    for ln in _read_lines(path):
        s = ln.strip()
        if not s:
            continue
        if s.startswith('#'):
            up = s.upper().replace(' ', '')
            if 'FORMAT=' in up:
                fmt = 'WAVN' if 'FORMAT=WAVN' in up else 'WAVL'
            continue
        parts = s.replace(',', ' ').split()
        try:
            rows.append([float(p) for p in parts[:3]])
        except ValueError:
            continue
    a = np.array(rows)
    x, n, k = a[:, 0], a[:, 1], a[:, 2]
    wl = 1e4 / x if fmt == 'WAVN' else x
    idx = np.argsort(wl)
    return wl[idx], n[idx], k[idx]


def load_lxmie(path):
    """LX-MIE '.dat': 3 '#'-comment header lines, columns wl_um, n, k (tab)."""
    rows = []
    for ln in _read_lines(path):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        try:
            rows.append([float(p) for p in parts[:3]])
        except ValueError:
            continue
    a = np.array(rows)
    idx = np.argsort(a[:, 0])
    return a[idx, 0], a[idx, 1], a[idx, 2]


def load_draine(path):
    """Draine 'callindex.out': cols wave(um), eps1-1, eps2, Re(n)-1, Im(n).

    Data begin after the header row containing 'wave(um)'; n = col4 + 1,
    k = col5.
    """
    rows = []
    started = False
    for ln in _read_lines(path):
        if 'wave(um)' in ln:
            started = True
            continue
        if not started:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            wl = float(parts[0]); n = float(parts[3]) + 1.0; k = float(parts[4])
        except ValueError:
            continue
        rows.append([wl, n, k])
    a = np.array(rows)
    idx = np.argsort(a[:, 0])
    return a[idx, 0], a[idx, 1], a[idx, 2]


def load_ch4(path):
    """VIRGA 'CH4(s)_Martonchik.csv': freq_cm-1, nr90, ni90, nr30, ni30, wl_um.

    Use the 90 K columns; recover wavelength from the frequency where the
    wavelength field is blank.
    """
    rows = []
    for ln in _read_lines(path)[1:]:          # skip the header row
        parts = ln.split(',')
        if len(parts) < 6:
            continue
        try:
            freq = float(parts[0]); n = float(parts[1]); k = float(parts[2])
        except ValueError:
            continue
        wl = parts[5].strip()
        wl_um = float(wl) if wl else (1e4 / freq if freq > 0 else np.nan)
        if np.isfinite(wl_um):
            rows.append([wl_um, n, k])
    a = np.array(rows)
    idx = np.argsort(a[:, 0])
    return a[idx, 0], a[idx, 1], a[idx, 2]


#  Mie integration over a log-normal size distribution 
def mie_sigma_ext(n_lam, k_lam, r_eff_um, sigma_g=1.6, n_r=40):
    """Per-particle extinction cross-section [cm^2] vs the output grid.

    Averages the Mie extinction cross-section over a log-normal number-size
    distribution (geometric-median radius ``r_eff_um``, geometric width
    ``sigma_g``); the spread smooths the single-size Mie interference ripples
    into the kind of broad curve a real polydisperse cloud produces.

    Args:
        n_lam, k_lam: real / imaginary refractive index on ``lam_um``.
        r_eff_um: geometric-median particle radius [um].
        sigma_g: geometric standard deviation of the log-normal.
        n_r: number of radius quadrature points.

    Returns:
        sigma_ext(lam_um) [cm^2], the number-weighted mean per particle.
    """
    ln_sg = np.log(sigma_g)
    radii = np.geomspace(r_eff_um / sigma_g ** 3, r_eff_um * sigma_g ** 3, n_r)
    # Log-normal number weights (median = r_eff_um).
    w = np.exp(-0.5 * (np.log(radii / r_eff_um) / ln_sg) ** 2) / radii
    w /= w.sum()
    m = n_lam - 1j * k_lam                              # miepython convention
    sig_um2 = np.zeros_like(lam_um)
    for r, wr in zip(radii, w):
        x = 2.0 * np.pi * r / lam_um                   # size parameter
        qext = miepython.efficiencies_mx(m, x)[0]
        sig_um2 += wr * qext * np.pi * r ** 2           # um^2
    return sig_um2 * CM2_PER_UM2                        # cm^2


# (file_key, loader, source_file, r_eff_um)
SPECIES = [
    ('h2so4',     load_aria,   'H2SO4_75_Palmer_1975.ri', 1.0),
    ('water_ice', load_aria,   'ICE_Warren_2008.ri',      12.0),
    ('tholin',    load_lxmie,  'Titan_tholin.dat',        0.05),
    ('ch4_haze',  load_ch4,    'CH4s_Martonchik.csv',     0.1),
    ('mars_dust', load_aria,   'sand_koepke_1997.ri',     0.25),
    ('silicate',  load_draine, 'astrosil_D03.txt',        0.2),
]


def main():
    for key, loader, src, r_eff in SPECIES:
        wl, n, k = loader(os.path.join(OC, src))
        # Interpolate (n, k) onto the output grid (edge-clamped outside range).
        n_lam = np.interp(lam_um, wl, n)
        k_lam = np.clip(np.interp(lam_um, wl, k), 0.0, None)
        sig = mie_sigma_ext(n_lam, k_lam, r_eff)
        path = os.path.join(HERE, f'{key}.csv')
        with open(path, 'w') as f:
            f.write(f'# {key} aerosol extinction cross-section (Mie; '
                    f'r_eff={r_eff} um)\n')
            f.write(f'# optical constants: {src}\n')
            f.write('# wavelength_Angstrom, sigma_cm2\n')
            for a, s in zip(lam_A, sig):
                f.write(f'{a:.1f}, {s:.6e}\n')
        print(f'  wrote {key}.csv  ({sig.min():.2e}-{sig.max():.2e} cm^2)'
              f'  [{src}]')


if __name__ == '__main__':
    main()
