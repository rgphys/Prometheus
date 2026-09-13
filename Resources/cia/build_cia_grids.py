"""
Build the compact CIA grids Prometheus reads (cia_<pair>.npz) from HITRAN.

Source: HITRAN collision-induced absorption database, "main" files
    https://hitran.org/data/CIA/main/H2-H2_2011.cia
    https://hitran.org/data/CIA/main/H2-He_2011.cia
The raw files (25 MB and 147 MB) are git-ignored; this script downloads any that
are missing, then writes log10 k on a coarser regular wavenumber grid and
reports the interpolation error against the native tables.

Grid choice: native spacing is 1 cm^-1.  CIA bands are pressure-broadened
collisional features tens to hundreds of cm^-1 wide, so a coarser step loses
little; the script measures that loss rather than assuming it.

Run:  python build_cia_grids.py [--step 2] [--Tmax 5000]
"""
import argparse, json, os, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
from Prometheus.core.continuum import read_hitran_cia

FILES = {'H2-H2': 'H2-H2_2011.cia', 'H2-He': 'H2-He_2011.cia'}
URL = 'https://hitran.org/data/CIA/main/{}'
FLOOR = 1e-60        # k <= 0 in the tables (far wings) is stored at this floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=float, default=2.0, help='wavenumber step [cm^-1]')
    ap.add_argument('--Tmax', type=float, default=5000.0)
    a = ap.parse_args()
    report = {}
    for pair, fname in FILES.items():
        raw = os.path.join(HERE, fname)
        if not os.path.exists(raw):
            print('downloading', URL.format(fname))
            urllib.request.urlretrieve(URL.format(fname), raw)
        T, nu, k = read_hitran_cia(raw)
        keep = T <= a.Tmax
        T, k = T[keep], k[keep]
        nu_c = np.arange(nu[0], nu[-1] + 0.5 * a.step, a.step)
        logk = np.log10(np.clip(k, FLOOR, None))
        logk_c = np.vstack([np.interp(nu_c, nu, row) for row in logk])
        # interpolation error on the native grid, where the absorption matters
        back = np.vstack([10 ** np.interp(nu, nu_c, row) for row in logk_c])
        strong = k > 1e-3 * k.max(axis=1, keepdims=True)
        rel = np.abs(back[strong] / k[strong] - 1)
        # band-integrated error over 100 cm^-1 bins
        edges = np.arange(nu[0], nu[-1], 100.0)
        idx = np.digitize(nu, edges)
        band = []
        for row_k, row_b in zip(k, back):
            s_k = np.bincount(idx, row_k); s_b = np.bincount(idx, row_b)
            m = s_k > 1e-3 * s_k.max()
            band.append(np.abs(s_b[m] / s_k[m] - 1).max())
        out = os.path.join(HERE, f'cia_{pair}.npz')
        np.savez_compressed(out, T=T.astype(np.float32), nu=nu_c.astype(np.float64),
                            log10k=logk_c.astype(np.float32))
        report[pair] = dict(source=URL.format(fname), n_T=int(len(T)), T_range_K=[float(T[0]), float(T[-1])],
                            nu_range_cm1=[float(nu_c[0]), float(nu_c[-1])], step_cm1=a.step,
                            pointwise_rel_error_median=float(np.median(rel)),
                            pointwise_rel_error_p99=float(np.percentile(rel, 99)),
                            band100cm1_rel_error_max=float(max(band)),
                            size_MB=round(os.path.getsize(out) / 1e6, 2))
        print(pair, json.dumps(report[pair]))
    json.dump(dict(description='Compact HITRAN CIA grids for Prometheus (log10 k, cm^5 molecule^-2)',
                   references=['HITRAN CIA database, main 2011 files for H2-H2 and H2-He; see the '
                               'HITRAN CIA page for the underlying computations'],
                   built_by='build_cia_grids.py', grids=report),
              open(os.path.join(HERE, 'PROVENANCE.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
