"""Generate a tree diagram of Prometheus density scenarios (top-down).

Renders the scenario class hierarchy (gasProperties.py) as a top-down tree:
root -> two family base classes -> leaf scenarios, with RadialWind's two
velocity-law sub-models hanging below it. Each leaf is annotated with its
governing physics (density/velocity law) and constructor parameters.
Output: scenario_tree.png next to this script.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# --- palette ---------------------------------------------------------------
C_ROOT = "#2f3640"
C_COLL = "#1f6f8b"     # collisional family
C_EVAP = "#b8860b"     # evaporative family
C_LEAF_C = "#dff1f6"
C_LEAF_E = "#fbf2da"
C_WIND = "#fff4df"
TXT = "#1b1b1b"
SUB = "#555"

LINE_DY = 3.3          # vertical pitch between text lines (data units)
PAD = 2.3

fig, ax = plt.subplots(figsize=(34, 12.5))
ax.set_xlim(0, 272)
ax.set_ylim(28, 128)
ax.axis("off")


def box(cx, cy, w, h, fc, ec, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))


def node_top(cx, top_y, w, lines, fc, ec, box_lines=None):
    """Draw a node whose TOP edge is at top_y. lines: (text,fs,weight,style,color).

    box_lines pads the box to a fixed line count (text stays top-aligned) so a
    row of nodes with differing content can share one common bottom edge.
    """
    h = max(len(lines), box_lines or 0) * LINE_DY + 2 * PAD
    cy = top_y - h / 2
    box(cx, cy, w, h, fc, ec)
    y = top_y - PAD - LINE_DY / 2
    for text, fs, wt, st, col in lines:
        ax.text(cx, y, text, ha="center", va="center", fontsize=fs,
                fontweight=wt, fontstyle=st, color=col, zorder=4)
        y -= LINE_DY
    return top_y - h          # bottom y


def node_mid(cx, cy, w, lines, fc, ec):
    n = len(lines)
    h = n * LINE_DY + 2 * PAD
    node_top(cx, cy + h / 2, w, lines, fc, ec)
    return cy + h / 2, cy - h / 2   # top, bottom


def vconnect(px, py_bot, cx, cy_top, color):
    """Orthogonal top-down connector: parent bottom -> child top."""
    my = (py_bot + cy_top) / 2
    ax.add_line(Line2D([px, px, cx, cx], [py_bot, my, my, cy_top],
                       color=color, lw=1.4, zorder=1, solid_capstyle="round"))


# line-builder shortcuts
def T(t, fs=10.5, wt="bold", st="normal", c=TXT): return (t, fs, wt, st, c)
def E(t, c=TXT):                                   return (t, 11.5, "normal", "normal", c)
def D(t, c=SUB):                                   return (t, 9.5, "normal", "italic", c)
def P(t, c):                                       return ("params:  " + t, 10, "normal", "normal", c)

# --- leaf content ----------------------------------------------------------
coll = [
    [T("BarometricAtmosphere"), D("isothermal"),
     E(r"$n=n_0\,e^{(R_p-r)/H}$"), E(r"$H=k_B T R_p^2/G\mu M_p$"), P(r"$T,\ P_0,\ \mu$", C_COLL)],
    [T("HydrostaticAtmosphere"), D("isothermal, hydrostatic eq."),
     E(r"$n=n_0\,e^{\lambda(r)-\lambda(R_p)}$"), E(r"$\lambda=G\mu M_p/k_B T r$"), P(r"$T,\ P_0,\ \mu$", C_COLL)],
    [T("NonIsothermalHydrostatic"), T("Atmosphere"), D("power-law T-P"),
     E(r"$T(P)=T_{ref}(P/P_0)^{\beta}$"), P(r"$T_{ref},\ P_0,\ \mu,\ \beta$", C_COLL)],
    [T("PowerLawAtmosphere"), D("tenuous"),
     E(r"$n=n_0\,(R_p/r)^{q}$"), P(r"$T,\ P_0,\ q$", C_COLL)],
]
evap = [
    [T("PowerLawExosphere"),
     E(r"$n=\frac{(q-3)N}{4\pi R_p^3}(R_p/r)^{q}$"), P(r"$N,\ q$", "#8a6500")],
    [T("MoonExosphere"), D("sourced at moon"),
     E(r"$n=\frac{(q-3)N}{4\pi R_m^3}(R_m/r)^{q}$"), P(r"$N,\ q$", "#8a6500")],
    [T("TidallyHeatedMoon"), D("phase-variable source"),
     E(r"moon power-law"), E(r"$\dot N=\dot N(\phi_{orb})$"), P(r"$q$", "#8a6500")],
    [T("TorusExosphere"), D("Gaussian ring"),
     E(r"$n\propto e^{-((a-a_t)/4H)^2}e^{-(z/H)^2}$"), E(r"$H=a_t v_{ej}/v_{orb}$"),
     P(r"$N,\ a_{torus},\ v_{ej}$", "#8a6500")],
    [T("SerpensExosphere"), D("MC histogram, smoothed"),
     E(r"$n(\vec r)$ from SERPENS particles"), P(r"file$,\ N,\ \sigma_{smooth}$", "#8a6500")],
    [T("RadialWindExosphere"), D("mass continuity"),
     E(r"$n=\frac{\dot M}{4\pi r^2 v(r)\,\mu}$"), P(r"$\dot M,\ \mu$", "#8a6500"),
     D(r"velocity law  $\downarrow$", C_EVAP)],
]

# --- geometry --------------------------------------------------------------
W, PITCH = 22, 25
xC = [14 + i * PITCH for i in range(4)]                 # collisional columns
xE0 = xC[-1] + PITCH + 11                                # family gap
xE = [xE0 + i * PITCH for i in range(6)]                 # evaporative columns

COLL_FX = sum(xC) / len(xC)
EVAP_FX = sum(xE) / len(xE)
ROOT_X = (COLL_FX + EVAP_FX) / 2

LEAF_TOP = 84

# --- root + families -------------------------------------------------------
_, root_bot = node_mid(ROOT_X, 112, 18, [
    ("PROMETHEUS", 15, "bold", "normal", "white"),
    ("density scenarios", 11, "normal", "normal", "white"),
    (r"$n(\vec r)$", 13, "normal", "normal", "white")], C_ROOT, C_ROOT)

coll_ftop, coll_fbot = node_mid(COLL_FX, 97, 44, [
    ("CollisionalAtmosphere", 13, "bold", "normal", "white"),
    ("dense; defined by T, P  ·  barometric base class", 10.5, "normal", "normal", "white")],
    C_COLL, C_COLL)
evap_ftop, evap_fbot = node_mid(EVAP_FX, 97, 44, [
    ("EvaporativeExosphere", 13, "bold", "normal", "white"),
    ("tenuous; normalized by N particles", 10.5, "normal", "normal", "white")],
    C_EVAP, C_EVAP)

vconnect(ROOT_X, root_bot, COLL_FX, coll_ftop, C_COLL)
vconnect(ROOT_X, root_bot, EVAP_FX, evap_ftop, C_EVAP)

# --- leaves (uniform height so the row shares one bottom edge) --------------
LEAF_LINES = max(len(l) for l in coll + evap)

for x, lines in zip(xC, coll):
    node_top(x, LEAF_TOP, W, lines, C_LEAF_C, C_COLL, box_lines=LEAF_LINES)
    vconnect(COLL_FX, coll_fbot, x, LEAF_TOP, C_COLL)

wind_bottom = None
for x, lines in zip(xE, evap):
    bot = node_top(x, LEAF_TOP, W, lines, C_LEAF_E, C_EVAP, box_lines=LEAF_LINES)
    vconnect(EVAP_FX, evap_fbot, x, LEAF_TOP, C_EVAP)
    if lines[0][0] == "RadialWindExosphere":
        wind_bottom, wind_x = bot, x

# --- radial-wind velocity-model children (below the RadialWind leaf) -------
wind_models = [
    (wind_x - 11, [T("wind_model='beta'", 10.5), D("CAK (1975)"),
                   E(r"$v=v_b+(v_\infty-v_b)$"), E(r"$\times(1-R_{in}/r)^{\beta}$"),
                   P(r"$v_\infty,\ \beta,\ r_{in}$", "#8a6500")]),
    (wind_x + 11, [T("wind_model='parker'", 10.5), D("Parker (1958)"),
                   D("transonic (Lambert-W)"),
                   E(r"$c_s=\sqrt{k_B T/\mu}$"), P(r"$T,\ M_p$", "#8a6500")]),
]
child_top = wind_bottom - 5
wind_lines = max(len(l) for _, l in wind_models)
for x, lines in wind_models:
    node_top(x, child_top, W, lines, C_WIND, C_EVAP, box_lines=wind_lines)
    vconnect(wind_x, wind_bottom, x, child_top, C_EVAP)

ax.text(ROOT_X, 127, "Prometheus density scenarios",
        ha="center", va="top", fontsize=22, fontweight="bold", color=TXT)

out = __file__.rsplit("/", 1)[0] + "/scenario_tree.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", out)
