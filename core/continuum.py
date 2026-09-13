"""
Continuum opacity: collision-induced absorption and the negative hydrogen ion.

Line-by-line molecular tables leave out the two continua that set the
photosphere of an H2-dominated atmosphere:

* **Collision-induced absorption (CIA)** by H2-H2 and H2-He pairs.  Its
  absorption coefficient scales with the product of the two number densities,
  ``kappa = n_A n_B k(nu, T)``, so it grows as P^2 and dominates deep down.
  Data: the HITRAN CIA database (``Resources/cia``), H2-H2 and H2-He 2011 files.
* **The negative hydrogen ion H-**, bound-free photo-detachment below
  1.6419 um and free-free absorption at all wavelengths, from the fits of
  John (1988, A&A 193, 189).  Coefficients are transcribed from the paper itself
  (ADS scan, Tables 2 and 3a/3b, Eqs. 3-6), not from third-party code: two
  widely used implementations disagree with the paper (one in the bound-free
  polynomial, both with single-digit typos in Table 3a).  The implementation is
  verified against John's own Table 1 of total coefficients.

Both are *true absorption* in LTE, so they emit thermally and never scatter;
see :func:`emission.source_weights`.

A continuum constituent is recognised by ``isContinuum = True`` and supplies
``absorptionCoefficient(n_tot, T, wavelength) -> kappa [cm^-1]``, evaluated per
cell from the local total number density and temperature.  That is the only
interface the transfer kernels need.

Created 2026-09-13.
"""

import os
from typing import Tuple

import numpy as np
from numba import njit

from . import constants as const

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'Resources', 'cia')

# ---------------------------------------------------------------------------
# H- (John 1988)
# ---------------------------------------------------------------------------

#: Photo-detachment threshold [um] (John 1988, Eq. 3 and text).
HMINUS_LAMBDA0_UM = 1.6419

#: Table 2 of John (1988): C_n, n = 1..6, for f(lambda) in Eq. (5).
HMINUS_BF_C = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])

#: Table 3a of John (1988): free-free parameters for lambda > 0.3645 um.
#: Rows n = 1..6; columns A_n, B_n, C_n, D_n, E_n, F_n.
HMINUS_FF_3A = np.array([
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520],
    [-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390],
    [2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380],
    [-696.2710, -1841.4000, 8624.9700, -10051.5300, 4400.0670, -655.0200],
    [88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850],
])

#: Table 3b of John (1988): free-free parameters for 0.1823 < lambda < 0.3645 um.
HMINUS_FF_3B = np.array([
    [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
    [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600],
    [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
    [115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
])

#: Validity of the free-free fit stated by John (1988): 0.5 <= 5040/T <= 3.6.
HMINUS_FF_THETA_RANGE = (0.5, 3.6)


def hminus_bound_free(wavelength_um: np.ndarray, T: np.ndarray) -> np.ndarray:
    """H- photo-detachment coefficient per unit electron pressure per H atom.

    John (1988) Eqs. (3)-(5): ``k_bf = 0.750 T^-5/2 exp(alpha/(lambda0 T))
    (1 - exp(-alpha/(lambda T))) sigma_lambda``, with
    ``sigma = 1e-18 lambda^3 (1/lambda - 1/lambda0)^1.5 f(lambda)`` and
    ``f = sum_n C_n (1/lambda - 1/lambda0)^((n-1)/2)``.  The paper prints
    ``alpha = 1.439e8``, which is hc/k in Angstrom K; with lambda in microns,
    as Eqs. (4)-(5) require, alpha = hc/k = 1.439e4 um K.  Checked against the
    paper's Table 1.

    Args:
        wavelength_um (np.ndarray): Wavelength [um].
        T (np.ndarray): Temperature [K], broadcastable against wavelength.

    Returns:
        np.ndarray: k_bf [cm^4 dyn^-1]; zero beyond the threshold and below
        the 0.125 um limit of the fit.
    """
    lam = np.asarray(wavelength_um, dtype=float)
    T = np.asarray(T, dtype=float)
    alpha = const.h * const.c / const.k_B * 1e4          # um K
    x = np.clip(1.0 / lam - 1.0 / HMINUS_LAMBDA0_UM, 0.0, None)
    f = sum(C * x ** ((n - 1) / 2.0) for n, C in enumerate(HMINUS_BF_C, start=1))
    sigma = 1e-18 * lam ** 3 * x ** 1.5 * f
    k = (0.750 * T ** -2.5 * np.exp(alpha / (HMINUS_LAMBDA0_UM * T))
         * (1.0 - np.exp(-alpha / (lam * T))) * sigma)
    return np.where((lam > 0.125) & (lam <= HMINUS_LAMBDA0_UM), k, 0.0)


def hminus_free_free(wavelength_um: np.ndarray, T: np.ndarray) -> np.ndarray:
    """H- free-free coefficient per unit electron pressure per H atom.

    John (1988) Eq. (6) with Tables 3a (lambda > 0.3645 um) and 3b
    (0.1823 < lambda <= 0.3645 um).  The fit is stated valid for
    0.5 <= 5040/T <= 3.6, i.e. 1400 <= T <= 10080 K; outside that range this
    function raises rather than extrapolate.

    Args:
        wavelength_um (np.ndarray): Wavelength [um].
        T (np.ndarray): Temperature [K], broadcastable against wavelength.

    Returns:
        np.ndarray: k_ff [cm^4 dyn^-1]; zero below 0.1823 um.

    Raises:
        ValueError: If any temperature is outside the fit's validity.
    """
    lam = np.asarray(wavelength_um, dtype=float)
    T = np.asarray(T, dtype=float)
    theta = 5040.0 / T
    lo, hi = HMINUS_FF_THETA_RANGE
    if np.any((theta < lo - 1e-9) | (theta > hi + 1e-9)):
        raise ValueError(
            f"H- free-free (John 1988) is valid for {5040 / hi:.0f}-{5040 / lo:.0f} K "
            f"(5040/T in [{lo}, {hi}]); got T = {T.min():.0f}-{T.max():.0f} K.")
    lam, theta = np.broadcast_arrays(lam, theta)

    def series(tab):
        out = np.zeros_like(lam)
        for n, (A, B, C, D, E, F) in enumerate(tab, start=1):
            out = out + theta ** ((n + 1) / 2.0) * (
                lam ** 2 * A + B + C / lam + D / lam ** 2 + E / lam ** 3 + F / lam ** 4)
        return 1e-29 * out

    return np.where(lam > 0.3645, series(HMINUS_FF_3A),
                    np.where(lam > 0.1823, series(HMINUS_FF_3B), 0.0))


class HMinusConstituent:
    """H- continuum, bound-free plus free-free (John 1988).

    John's coefficients are per unit electron pressure per neutral H atom and
    already include the Saha population of H-, so the absorption coefficient is

        kappa = (k_bf + k_ff) * n_H * P_e,   P_e = n_e k_B T.

    Prometheus has no equilibrium chemistry: the atomic-hydrogen and electron
    volume mixing ratios are INPUTS, not derived.  They are modelling
    assumptions unless taken from a chemistry calculation.

    Args:
        chi_H (float): Volume mixing ratio of atomic H.
        chi_e (float): Volume mixing ratio of free electrons.
    """

    isMolecule = False
    isScatterer = False
    isContinuum = True

    def __init__(self, chi_H: float, chi_e: float):
        self.name = 'H-'
        self.chi_H = float(chi_H)
        self.chi_e = float(chi_e)
        self.chi = 1.0

    def absorptionCoefficient(self, n_tot: np.ndarray, T: np.ndarray,
                              wavelength: np.ndarray) -> np.ndarray:
        """kappa [cm^-1] with shape broadcast(n_tot[..., None], wavelength).

        Args:
            n_tot (np.ndarray): Total number density [cm^-3], shape (...).
            T (np.ndarray): Temperature [K], same shape as ``n_tot``.
            wavelength (np.ndarray): Wavelength [cm], shape (n_wav,).
        """
        n = np.asarray(n_tot, dtype=float)[..., np.newaxis]
        TT = np.broadcast_to(np.asarray(T, dtype=float), np.shape(n_tot))[..., np.newaxis]
        lam_um = np.asarray(wavelength, dtype=float) * 1e4
        live = n > 0.0
        T_eval = np.where(live, TT, 5040.0)                 # empty cells never raise
        k = hminus_bound_free(lam_um, T_eval) + hminus_free_free(lam_um, T_eval)
        P_e = self.chi_e * n * const.k_B * TT
        return np.where(live, k * (self.chi_H * n) * P_e, 0.0)


# ---------------------------------------------------------------------------
# CIA (HITRAN)
# ---------------------------------------------------------------------------

def read_hitran_cia(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a HITRAN ``.cia`` file into a (T, wavenumber) grid.

    Blocks sharing the modal wavenumber range are kept; each block header gives
    the pair, nu_min, nu_max, number of points and temperature.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: temperatures (n_T,) [K],
        wavenumbers (n_nu,) [cm^-1], binary absorption coefficients
        (n_T, n_nu) [cm^5 molecule^-2].
    """
    lines = open(path).read().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        h = lines[i]
        if not h.strip():
            i += 1
            continue
        nu_min, nu_max = float(h[20:30]), float(h[30:40])
        npts, T = int(h[40:47]), float(h[47:54])
        data = np.array([l.split()[:2] for l in lines[i + 1:i + 1 + npts]], dtype=float)
        blocks.append(((nu_min, nu_max, npts), T, data))
        i += 1 + npts
    from collections import Counter
    key = Counter(b[0] for b in blocks).most_common(1)[0][0]
    sel = sorted([b for b in blocks if b[0] == key], key=lambda b: b[1])
    nu = sel[0][2][:, 0]
    return (np.array([b[1] for b in sel]), nu, np.vstack([b[2][:, 1] for b in sel]))


@njit(cache=True)
def _cia_blend(tab, i_lo, t, fine_idx, fine_u, out):
    """k on a fine grid from log10 k tabulated on a coarse grid.

    For each layer, blends two tabulated temperature rows (weights t), then
    interpolates log10 k linearly onto the fine points (coarse index + weight).
    Coarse points with no data are -inf and give k = 0.
    """
    n_l = i_lo.shape[0]
    n_f = fine_idx.shape[0]
    n_c = tab.shape[1]
    row = np.empty(n_c)
    for l in range(n_l):
        a = 1.0 - t[l]
        for c in range(n_c):
            x0 = tab[i_lo[l], c]
            x1 = tab[i_lo[l] + 1, c]
            row[c] = a * x0 + t[l] * x1 if (x0 > -1e300 and x1 > -1e300) else -np.inf
        for f in range(n_f):
            c = fine_idx[f]
            y0 = row[c]
            y1 = row[c + 1]
            if y0 > -1e300 and y1 > -1e300:
                out[l, f] = 10.0 ** (y0 + fine_u[f] * (y1 - y0))
            else:
                out[l, f] = 0.0


class CIAConstituent:
    """Collision-induced absorption of one molecular pair.

    ``kappa = (chi_A n)(chi_B n) k(nu, T)``, interpolated bilinearly in T and
    wavenumber on ``log10 k`` from the compact grid in ``Resources/cia``
    (built from HITRAN by ``Resources/cia/build_cia_grids.py``).  Beyond the
    tabulated wavenumber range the coefficient is zero; a temperature outside
    the tabulated range raises rather than extrapolate.

    Args:
        pair (str): ``'H2-H2'`` or ``'H2-He'``.
        chi_A (float): Volume mixing ratio of the first species.
        chi_B (float): Volume mixing ratio of the second species.
    """

    isMolecule = False
    isScatterer = False
    isContinuum = True

    def __init__(self, pair: str, chi_A: float, chi_B: float):
        path = os.path.join(RESOURCES, f'cia_{pair}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run Resources/cia/build_cia_grids.py.")
        d = np.load(path)
        self.name = pair
        self.pair = pair
        self.chi_A, self.chi_B = float(chi_A), float(chi_B)
        self.chi = 1.0
        self.T_grid = d['T'].astype(float)
        self.nu_grid = d['nu'].astype(float)
        self.logk = d['log10k'].astype(float)                  # (n_T, n_nu)

    #: Coarse-grid resolving power used for 1-D wavelength grids.  The HITRAN
    #: tables are sampled at 1 cm^-1 and CIA features are tens of cm^-1 wide,
    #: so evaluating on a lambda/dlambda = 3000 grid and interpolating is
    #: exact to the precision the validation measures (see continuum tests).
    COARSE_R = 3000.0

    def _coarseFor(self, wavelength: np.ndarray):
        """Coarse wavelength grid, its log10-k table, and the fine-point map."""
        key = (wavelength.shape, float(wavelength[0]), float(wavelength[-1]), float(wavelength.sum()))
        cache = getattr(self, '_coarse_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        lo, hi = float(wavelength.min()), float(wavelength.max())
        n_c = max(int(np.ceil(np.log(hi / lo) * self.COARSE_R)) + 2, 2)
        coarse = np.geomspace(lo, hi, n_c)
        tab = self._gridFor(coarse)
        idx = np.clip(np.searchsorted(coarse, wavelength) - 1, 0, n_c - 2)
        u = np.clip((wavelength - coarse[idx]) / (coarse[idx + 1] - coarse[idx]), 0.0, 1.0)
        val = (np.ascontiguousarray(tab), idx.astype(np.int64), u)
        self._coarse_cache = (key, val)
        return val

    def _gridFor(self, wavelength: np.ndarray) -> np.ndarray:
        """log10 k interpolated onto ``wavelength`` for every tabulated T, cached.

        The wavenumber interpolation depends only on the grid, so for repeated
        calls on one 1-D grid (a retrieval) it is done once; each call then only
        interpolates in temperature.  Out-of-range wavenumbers map to -inf (k = 0).
        """
        key = (wavelength.shape, float(wavelength[0]), float(wavelength[-1]), float(wavelength.sum()))
        cache = getattr(self, '_grid_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        nu = 1.0 / wavelength
        inside = (nu >= self.nu_grid[0]) & (nu <= self.nu_grid[-1])
        tab = np.full((len(self.T_grid), len(wavelength)), -np.inf)
        for i in range(len(self.T_grid)):
            tab[i, inside] = np.interp(nu[inside], self.nu_grid, self.logk[i])
        self._grid_cache = (key, tab)
        return tab

    def binaryCoefficient(self, T: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        """k(nu, T) [cm^5 molecule^-2], shape T.shape + (n_wav,)."""
        T = np.asarray(T, dtype=float)
        if T.size and (T.min() < self.T_grid[0] - 1e-9 or T.max() > self.T_grid[-1] + 1e-9):
            raise ValueError(
                f"{self.pair} CIA is tabulated for {self.T_grid[0]:.0f}-{self.T_grid[-1]:.0f} K; "
                f"got {T.min():.0f}-{T.max():.0f} K.")
        wavelength = np.asarray(wavelength, dtype=float)
        if wavelength.ndim == 1 and T.ndim == 1:
            tab, idx, u = self._coarseFor(wavelength)
            i = np.clip(np.searchsorted(self.T_grid, T) - 1, 0, len(self.T_grid) - 2).astype(np.int64)
            t = np.clip((T - self.T_grid[i]) / (self.T_grid[i + 1] - self.T_grid[i]), 0.0, 1.0)
            out = np.empty((len(T), len(wavelength)))
            _cia_blend(tab, i, t, idx, u, out)
            return out
        nu = 1.0 / np.asarray(wavelength, dtype=float)                 # cm^-1
        inside = (nu >= self.nu_grid[0]) & (nu <= self.nu_grid[-1])
        j = np.clip(np.searchsorted(self.nu_grid, nu) - 1, 0, len(self.nu_grid) - 2)
        u = np.clip((nu - self.nu_grid[j]) / (self.nu_grid[j + 1] - self.nu_grid[j]), 0.0, 1.0)
        i = np.clip(np.searchsorted(self.T_grid, T) - 1, 0, len(self.T_grid) - 2)
        t = np.clip((T - self.T_grid[i]) / (self.T_grid[i + 1] - self.T_grid[i]), 0.0, 1.0)
        L = self.logk
        i_, t_ = i[..., np.newaxis], t[..., np.newaxis]
        lo = L[i_, j] * (1 - u) + L[i_, j + 1] * u
        hi = L[i_ + 1, j] * (1 - u) + L[i_ + 1, j + 1] * u
        return np.where(inside, 10.0 ** (lo * (1 - t_) + hi * t_), 0.0)

    def absorptionCoefficient(self, n_tot: np.ndarray, T: np.ndarray,
                              wavelength: np.ndarray) -> np.ndarray:
        """kappa [cm^-1], shape n_tot.shape + (n_wav,)."""
        n = np.asarray(n_tot, dtype=float)
        TT = np.broadcast_to(np.asarray(T, dtype=float), n.shape)
        live = n > 0.0
        T_eval = np.where(live, TT, self.T_grid[0])
        k = self.binaryCoefficient(T_eval, wavelength)
        return np.where(live[..., np.newaxis],
                        (self.chi_A * self.chi_B) * (n ** 2)[..., np.newaxis] * k, 0.0)
