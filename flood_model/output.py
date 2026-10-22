# ##########################################################
# ## FLOOD MODEL: OUTPUT SCRIPT FOR THE POLICY SCENARIOS ###
# ##########################################################
# Tables and figures for the policy scenarios solved by main.py (benchmark,
# no risk misperception, subsidised self-protection, public protection,
# subsidised insurance, all combined; and, with main.py's FACTORIAL, the
# other combinations of these four levers), read from the outputs main.py
# saves to Output/flood_model/policy_scenarios/simulations. Run as a script
# or cell by cell; results go to Output/flood_model/policy_scenarios:
#     maps/benchmark (1)-(5e), maps/scenarios (6)-(12),
#     distributions/benchmark (13)-(18), distributions/scenarios (19)-(24),
#     tables (25)-(27b, scenario summary, exposure), bars (28)-(29a),
#     interactions (pairwise interactions, Shapley values).
# Conventions follow ../revision_output_clean.py (FOOTNOTE comments, 3D bars
# over a basemap at z = 0, tables in Excel), with LaTeX copies of the tables.
# Bars either stack categories (additive quantities: households, housing,
# total damages) or overlay them (levels: rents, damages per household):
# an overlaid bar rises to the lowest category value, then adds the surplus
# of each higher category, in increasing order. 3D maps of levels (rents,
# damages per household) instead show one panel per housing type, with the
# level as bar height and the number of households as bar color.

# %% PREAMBLE

import json
import sys
import time
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

try:
    HERE = Path(__file__).resolve().parent
except NameError:                     # Interactive Window: no __file__
    HERE = Path.cwd()
    if not (HERE / "solver.py").exists():
        HERE = HERE / "flood_model"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
import data
import floods

## PATHS

PATH_SCEN = config.OUTPUT / "policy_scenarios"
PATH_SIMUL = PATH_SCEN / "simulations"
FOLDERS = {key: PATH_SCEN / sub for key, sub in [
    ("maps_bench", "maps/benchmark"), ("maps_scen", "maps/scenarios"),
    ("dist_bench", "distributions/benchmark"),
    ("dist_scen", "distributions/scenarios"),
    ("tables", "tables"), ("bars", "bars"),
    ("interactions", "interactions")]}
for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

## LABELS AND COLORS (validated with the dataviz palette checker)

SCEN = json.loads((PATH_SIMUL / "scenarios.json").read_text())
SINGLES = ["no_misperception", "subsid_protec", "public_protec",
           "subsid_insur"]
SCENARIOS = [s for s in ["benchmark"] + SINGLES + ["combined"]
             if s in SCEN["scenarios"]]
# Other combinations of two or three levers (main.py, FACTORIAL)
FACTORIAL = [s for s in SCEN["scenarios"] if s not in SCENARIOS]
SCEN_LABELS = {"benchmark": "Benchmark",
               "no_misperception": "No risk misperception",
               "subsid_protec": "Subsidised self-protection",
               "public_protec": "Public protection",
               "subsid_insur": "Subsidised insurance",
               "combined": "All combined"}
SCEN_SHORT = {"benchmark": "Bench.", "no_misperception": "No RM",
              "subsid_protec": "Subs. SP", "public_protec": "Public P.",
              "subsid_insur": "Subs. ins.", "combined": "Combined",
              "sum": "Sum singles", "interaction": "Interaction"}
# Benchmark in primary ink, then categorical slots in fixed order (line
# styles as secondary encoding: more than three hues overlap)
SCEN_COLORS = {"benchmark": "#0b0b0b", "no_misperception": "#2a78d6",
               "subsid_protec": "#eb6834", "public_protec": "#1baf7a",
               "subsid_insur": "#eda100", "combined": "#e87ba4"}
SCEN_STYLES = {"benchmark": "-", "no_misperception": "--",
               "subsid_protec": "-.", "public_protec": ":",
               "subsid_insur": (0, (5, 1, 1, 1, 1, 1)), "combined": "-"}

HTYPES = ["Formal private", "Informal backyards", "Informal settlements",
          "Formal subsidised"]
HTYPES_SHORT = ["FP", "IB", "IS", "FS"]
HTYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INCGRPS = ["Low income", "Mid-low income", "Mid-high income", "High income"]
INCGRPS_SHORT = ["Low", "M-low", "M-high", "High"]
INCGRP_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal
UNPROTECTED = "#b5b3ad"
LEVEL_LABELS = ["Unprotected", "1 level (15 cm)", "2 levels (30 cm)",
                "3 levels (45 cm)"]
LEVEL_COLORS = [UNPROTECTED, "#f09a72", "#d9541f", "#8f3412"]  # ordinal
CC_COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LAND, URBAN = "#e4e2da", "#c9c6bb"
# Damages per household above this share of mean income (%) are capped in
# the plots and marked with a triangle (heat maps: the hottest color)
DAMAGE_CAP = 25
# Sequential heat scale for bar colors (households in the level maps),
# without its palest end, which vanishes on the basemap
HEAT = LinearSegmentedColormap.from_list(
    "heat", plt.get_cmap("YlOrRd")(np.linspace(0.15, 1, 256)))

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
BOX = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID,
           alpha=0.9)
CAP_HANDLE = Line2D([], [], marker="^", color=INK, linestyle="none",
                    markersize=6, label=f"Above {DAMAGE_CAP}% (capped)")

## LOAD INPUTS

param = {**config.PARAM, **SCEN["param"]}
p, inputs = data.prepare_inputs(param)
y_bar = inputs["average_income"]                  # ȳ_i, rands/year
alpha, q0 = p["alpha"], p["q0"]

## GEOGRAPHY: cell centroids in longitude / latitude, and a simple 2D
## basemap (municipality and urban footprint)

geo_grid = gpd.read_file(config.DATA / "data_Cape_Town" / "grid_reference_500.shp")
grid_csv = pd.read_csv(config.DATA / "data_Cape_Town"
                       / "grid_NEDUM_Cape_Town_500.csv", sep=";")
centroids = geo_grid.geometry.centroid.to_crs("EPSG:4326")
LON, LAT = centroids.x.to_numpy(), centroids.y.to_numpy()
N_CELLS = len(LON)
CELLS = np.arange(N_CELLS)
# Size of a 500 m cell in degrees (bars cover a share of it, see map3d)
DLON = 500 / (111320 * np.cos(np.deg2rad(LAT.mean())))
DLAT = 500 / 110950
BOUNDS = geo_grid.to_crs("EPSG:4326").total_bounds        # lon/lat extent

# Basemap raster on the 500 m lattice of the grid (projected metres): the
# municipality in light gray, cells with more than 5% urban land darker,
# transparent outside; corners converted to longitude / latitude
col = np.round((geo_grid["X_MIN"] - geo_grid["X_MIN"].min()) / 500).astype(int)
row = np.round((geo_grid["Y_MIN"] - geo_grid["Y_MIN"].min()) / 500).astype(int)
raster = np.zeros((row.max() + 2, col.max() + 2, 4))
urban_cell = grid_csv["urban"].to_numpy() > 0.05 * 250000
raster[row, col] = np.where(urban_cell[:, None],
                            matplotlib.colors.to_rgba(URBAN),
                            matplotlib.colors.to_rgba(LAND))
corner_x, corner_y = np.meshgrid(
    geo_grid["X_MIN"].min() + 500 * np.arange(col.max() + 2),
    geo_grid["Y_MIN"].min() + 500 * np.arange(row.max() + 2))
corners = gpd.GeoSeries(gpd.points_from_xy(corner_x.ravel(), corner_y.ravel()),
                        crs=geo_grid.crs).to_crs("EPSG:4326")
BASE_LON = corners.x.to_numpy().reshape(corner_x.shape)
BASE_LAT = corners.y.to_numpy().reshape(corner_x.shape)
BASE_RGBA = raster

## LOAD SIMULATION RESULTS


def load_simulation_data(var, name):
    prefix = "" if var == "mask_self_protec" else "initial_state_"
    path = PATH_SIMUL / f"{prefix}{var}_{name}.npy"
    if var == "developer_tax" and not path.exists():
        return np.zeros(3)        # [t, losses, revenue]: no misperception
    return np.load(path)


VARIABLES = ["households", "dwelling_size", "rent", "housing_supply",
             "utility", "mask_self_protec", "damages_contents",
             "damages_structure", "damages_reimbursed",
             "damages_informal_levels", "ex_post_u_post", "ex_post_z_post",
             "ex_post_price", "net_income", "landlord_revenue",
             "surplus_damages", "error", "solver_cells", "developer_tax"]


def load_scenario(label):
    name = SCEN["scenarios"][label]["name"]
    return derive({var: load_simulation_data(var, name) for var in VARIABLES})

###############################################################################

# %% DERIVED VARIABLES

print("DERIVED VARIABLES")


def derive(d):
    """Add per-scenario variables used by the plots and tables."""
    hh = np.nan_to_num(d["households"])
    d["hh"] = hh
    d["hh_type"], d["hh_group"] = hh.sum(1), hh.sum(0)
    # Housing consumed per household (m2): FP dwelling size, shacks, and for
    # RDP owners the house and the part of the backyard not rented out
    mu = np.nan_to_num(d["housing_supply"][1]) / 1e6
    cons = np.nan_to_num(d["dwelling_size"]).copy()
    cons[1:3] = p["shack_size"]
    cons[3] = p["RDP_size"] + p["backyard_size"] * (1 - mu)
    d["consumption"] = cons
    d["rent_clean"] = np.nan_to_num(d["rent"])
    # Actual expected damages (rands/year): gross, and borne net of payouts
    d["dmg"] = d["damages_contents"] + d["damages_structure"]
    d["dmg_net"] = d["dmg"] - d["damages_reimbursed"]
    d["level"] = np.nan_to_num(d["mask_self_protec"]).astype(int)
    return d


sim = {label: load_scenario(label) for label in SCENARIOS}
B = sim["benchmark"]
N_TOTAL = B["hh"].sum()

## EXPECTED FLOOD DEPTH FROM INPUTS
## Expected annual depth averaged over the cell, E[p_T d_T] (cm), with the
## deepest flood type in each event (same states and probabilities as
## floods.py). Per housing type, frequent pluvial events are drained
## (floods.PLUVIAL_SAFE), so that a positive depth means positive expected
## damages before public or self-protection.


def expected_depth(climate_change, htype=None):
    opts = {"coastal": 1, "climate_change": climate_change}
    depth, prop = floods.flood_maps(inputs, opts, p)
    event = np.nan_to_num(prop * depth)            # (types, periods, cells)
    if htype is not None:
        safe = np.isin(floods.RETURN_PERIODS, floods.PLUVIAL_SAFE[htype])
        event[floods.FLOOD_TYPES.index("pluvial"), safe] = 0
    states = floods.state_damages(event.max(0))
    return 100 * floods.expected_damage(
        floods.state_probabilities(opts, p), states)


DEPTH_CELL = {cc: expected_depth(cc) for cc in (0, 1)}
# By housing type (formal, backyard, informal, subsidized), without CC
DEPTH = np.array([expected_depth(0, h) for h in
                  ("formal", "backyard", "informal", "subsidized")])
# Exposed households: expected depth above EXPOSURE_THRESHOLD (cm/year). In
# the benchmark, a zero threshold (any positive expected damage, as in
# main.py) counts 86% of households as exposed, most of them at negligible
# depths; 0.1 cm/year keeps 32% of households and 97.6% of damages
EXPOSURE_THRESHOLD = 0.1
EXPOSED = DEPTH > EXPOSURE_THRESHOLD
EXP3 = EXPOSED[:, None, :]                    # (types, groups, cells)


def exposed_share(d, types=(0, 1, 2, 3), groups=(0, 1, 2, 3)):
    """% of the households of these housing types and groups exposed."""
    hh = d["hh"][np.ix_(list(types), list(groups))]
    n = hh.sum()
    return 100 * (hh * EXP3[list(types)]).sum() / n if n > 0 else np.nan


def settler_shares(d):
    """% of exposed settlers protected by sandbags, % of settlers exposed."""
    settlers = d["hh_type"][2]
    exposed = (settlers * EXPOSED[2]).sum()
    protected = (settlers * EXPOSED[2] * (d["level"] >= 1)).sum()
    return 100 * protected / exposed, 100 * exposed / settlers.sum()

###############################################################################

# %% PLOTTING HELPERS (3D MAPS, OVERLAID OR STACKED BARS)

print("PLOTTING HELPERS")

ELEV, AZIM = 38, -62


def segments(v, overlay):
    """Bar segments for values v (categories, items) >= 0: (category,
    bottom, height), each (categories, items), one row per drawing rank.
    Stacked: categories in their fixed order. Overlaid: per item, the lowest
    value from zero, then the surplus of each higher category (increasing
    order), so that the bar height is the highest value."""
    k, n = v.shape
    if not overlay:
        cat = np.repeat(np.arange(k)[:, None], n, 1)
        return cat, np.cumsum(v, 0) - v, v
    order = np.argsort(v, 0, kind="stable")
    sv = np.take_along_axis(v, order, 0)
    bottom = np.vstack([np.zeros(n), sv[:-1]])
    return order, bottom, sv - bottom


def cap_values(v, cap):
    """Values capped at cap, and items with a value above it."""
    v = np.clip(np.nan_to_num(v), 0, None)
    if cap is None:
        return v, np.zeros(v.shape[1], bool)
    return np.minimum(v, cap), (v > cap).any(0)


def bar_top(values, overlay=False, cap=None):
    v, _ = cap_values(np.atleast_2d(values), cap)
    cat, bottom, height = segments(v, overlay)
    return (bottom + height).max(initial=0)


def add_basemap(ax, zorder=2):
    """Municipality and urban footprint at z = 0, and the CBD."""
    surf = ax.plot_surface(BASE_LON, BASE_LAT, np.zeros_like(BASE_LON),
                           facecolors=BASE_RGBA, rstride=1, cstride=1,
                           shade=False, linewidth=0, antialiased=False)
    surf.set_zorder(zorder)
    ax.scatter([18.4241], [-33.9249], [0], s=6, color=INK2, zorder=zorder,
               depthshade=False)
    ax.text(18.4241, -33.9249 + 0.02, 0, "CBD", fontsize=6, color=INK2,
            zorder=zorder)


def map3d(ax, values, colors, zlabel, title=None, zlim=None, labels=None,
          legend=True, level=False, overlay=False, cap=None, heat=None,
          norm=None):
    """3D bars per cell over the basemap, for values (categories, cells),
    stacked or overlaid (segments). level: thinner, more transparent bars
    seen from higher up, for level variables whose bars form walls. cap:
    values above it are capped and marked with a triangle. heat: (cells,)
    values coloring single bars on the HEAT scale with norm (opaque; values
    beyond its ends take its end colors). All segments go into one bar3d
    call, so that matplotlib depth-sorts their faces."""
    v, over = cap_values(np.atleast_2d(values), cap)
    keep = np.any(v > 0, 0)
    lon, lat, v, over = LON[keep], LAT[keep], v[:, keep], over[keep]
    alpha, size = (0.7, 0.55) if level else (0.9, 0.9)
    if heat is not None:
        alpha = 0.9
    add_basemap(ax, zorder=2)
    cat, bottom, height = segments(v, overlay)
    m = height > 0
    if m.any():
        cells = np.nonzero(m)[1]
        if heat is None:
            rgba = np.array([matplotlib.colors.to_rgba(c, alpha)
                             for c in colors])[cat[m]]
        else:
            h = np.nan_to_num(np.asarray(heat, dtype=float)[keep][cells])
            rgba = HEAT(norm(np.maximum(h, norm.vmin)), alpha=alpha)
        dx, dy = size * DLON, size * DLAT
        ax.bar3d(lon[cells] - dx / 2, lat[cells] - dy / 2, bottom[m], dx, dy,
                 height[m], color=rgba, shade=True, edgecolor="none",
                 zorder=3)
    if over.any():
        ax.scatter(lon[over], lat[over], np.full(over.sum(), cap), marker="^",
                   s=12, color=INK, depthshade=False, zorder=4)
    if zlim is None:
        zlim = (0, max((bottom + height).max(initial=0), 1e-9))
    ax.set_xlim(BOUNDS[0], BOUNDS[2])
    ax.set_ylim(BOUNDS[1], BOUNDS[3])
    ax.set_zlim(*zlim)
    km_x = (BOUNDS[2] - BOUNDS[0]) * 111.32 * np.cos(np.deg2rad(LAT.mean()))
    km_y = (BOUNDS[3] - BOUNDS[1]) * 110.95
    ax.set_box_aspect((km_x, km_y, 0.55 * km_y), zoom=0.92)
    ax.view_init(elev=52 if level else ELEV, azim=AZIM)
    ax.set_xlabel("Longitude", labelpad=2, fontsize=8)
    ax.set_ylabel("Latitude", labelpad=2, fontsize=8)
    ax.set_zlabel(zlabel, labelpad=6, fontsize=8)
    ax.tick_params(axis="both", labelsize=7, pad=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
    if title:
        ax.set_title(title, pad=0)
    if legend and labels:
        ax.legend(handles=legend_handles(colors, labels, cap),
                  loc="upper left", fontsize=7, handlelength=1,
                  borderaxespad=0)
    return zlim


def heat_colorbar(fig, rect, norm, label, extend="neither", handles=None):
    """Horizontal colorbar of the HEAT scale in fig coordinates rect (x0, y0,
    width, height); handles: legend entries placed to its right."""
    cax = fig.add_axes(rect)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm, HEAT), cax=cax,
                       orientation="horizontal", extend=extend)
    bar.set_label(label, fontsize=8, color=INK2)
    bar.ax.tick_params(labelsize=7)
    if isinstance(norm, LogNorm):
        ticks = [m * 10.0 ** e for e in range(-3, 5) for m in (1, 3)]
        bar.set_ticks([v for v in ticks if norm.vmin <= v <= norm.vmax])
        bar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
        bar.ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    bar.outline.set_edgecolor(GRID)
    if handles:
        cax.legend(handles=handles,
                   loc="center left", bbox_to_anchor=(1.06, 0.5), fontsize=7,
                   handlelength=1)
    return bar


def legend_handles(colors, labels, cap=None):
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    return handles + ([CAP_HANDLE] if cap is not None else [])


def bars2d(ax, x, widths, values, colors, overlay=False, cap=None):
    """2D bars at x for values (categories, items), stacked or overlaid;
    values above cap are capped and marked with a triangle. Returns the
    highest bar."""
    v, over = cap_values(np.atleast_2d(values), cap)
    cat, bottom, height = segments(v, overlay)
    for r in range(v.shape[0]):
        for c in range(v.shape[0]):
            m = (cat[r] == c) & (height[r] > 0)
            if m.any():
                ax.bar(x[m], height[r][m], width=widths[m], bottom=bottom[r][m],
                       color=colors[c], edgecolor="white", linewidth=0.3)
    if over.any():
        ax.scatter(x[over], np.full(over.sum(), cap), marker="^", s=22,
                   color=INK, zorder=5, clip_on=False)
    return (bottom + height).max(initial=0)


def _fmt(x, pos=None):
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.1f}"
    return f"{x:.2g}"


def new3d(fig, *pos):
    return fig.add_subplot(*pos, projection="3d", computed_zorder=False)


def save(fig, folder, filename):
    # OneDrive briefly locks files it is syncing (OSError 22): retry
    for attempt in range(5):
        try:
            fig.savefig(FOLDERS[folder] / filename)
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(3)
    plt.close(fig)
    print("  saved", FOLDERS[folder].relative_to(PATH_SCEN) / filename)


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

###############################################################################

# %% MAP VARIABLES (bar heights by category, per scenario)

print("MAP VARIABLES")


def map_values(d, kind):
    """Bar heights (categories, cells) of map `kind` for scenario d (rent,
    damages_relative: bar heights of the heat maps, by housing type)."""
    if kind == "households_type":                     # (1)
        return d["hh_type"]
    if kind == "households_group":                    # (2)
        return d["hh_group"]
    if kind == "housing":                             # (3), hectares
        return d["hh_type"] * d["consumption"] / 1e4
    if kind == "rent":                                # (4), without FS
        return np.where(d["hh_type"][:3] > 0, d["rent_clean"][:3], 0)
    if kind == "damages_type":                        # (5b), M rands
        return d["dmg"].sum(1) / 1e6
    if kind == "damages_relative":                    # (5c), % of income
        income = (d["hh"] * y_bar[None, :, None]).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = 100 * d["dmg_net"].sum(1) / income
        return np.where(d["hh_type"] > 0, np.nan_to_num(rel), 0)
    if kind == "protection":                          # (5d), households
        return np.array([d["hh_type"][2] * (d["level"] == k)
                         for k in range(4)])
    raise ValueError(kind)


MAPS = {
    # kind: (z label, colors, labels, title, overlay, cap, level)
    "households_type": ("Number of households", HTYPE_COLORS, HTYPES,
                        "Households by housing type", False, None, False),
    "households_group": ("Number of households", INCGRP_COLORS, INCGRPS,
                         "Households by income group", False, None, False),
    "housing": ("Housing (ha of floor space)", HTYPE_COLORS, HTYPES,
                "Housing quantity by housing type", False, None, False),
    "damages_type": ("Damages (M ZAR/year)", HTYPE_COLORS, HTYPES,
                     "Expected flood damages by housing type", False, None,
                     False),
    "protection": ("Informal settlement households", LEVEL_COLORS,
                   LEVEL_LABELS, "Self-protection of settlers", False, None,
                   False),
}


def protection_box(d):
    protected, exposed = settler_shares(d)
    return (f"Exposed settlers using sandbags: {protected:5.1f}%\n"
            f"Settlers exposed to floods:      {exposed:5.1f}%")


def draw_map(ax, d, kind, title=None, zlim=None, legend=True, values=None):
    zlabel, colors, labels, default_title, overlay, cap, level = MAPS[kind]
    if values is None:
        values = map_values(d, kind)
    return map3d(ax, values, colors, zlabel, title=title or default_title,
                 zlim=zlim, labels=labels, legend=legend, level=level,
                 overlay=overlay, cap=cap)


## HEAT MAPS: one panel per housing type, bar height = a level variable
## (rent, damages per household) with the same z scale in all panels and
## scenarios, bar color = households of that housing type in the cell on the
## HEAT scale (log), the same in all panels and scenarios

HEAT_MAPS = {
    # kind: (housing types, z label, cap)
    "rent": ([0, 1, 2], "Rent (ZAR/m²/year)", None),
    "damages_relative": ([0, 1, 2, 3],
                         "Damages per household (% of mean income)",
                         DAMAGE_CAP),
}
# Households of one housing type per cell: from below 1 to about 7,000
HH_NORM = LogNorm(10, 10000, clip=True)
HH_LABEL = "Households of the housing type in the cell (log scale)"
HEAT_ZLIM = {kind: (0, max(bar_top(map_values(sim[s], kind)[t], cap=cap)
                           for s in SCENARIOS for t in types))
             for kind, (types, zlabel, cap) in HEAT_MAPS.items()}


def draw_heat(ax, d, kind, t, title=None):
    types, zlabel, cap = HEAT_MAPS[kind]
    return map3d(ax, map_values(d, kind)[t], None, zlabel,
                 title=title or f"{HTYPES[t]}", zlim=HEAT_ZLIM[kind],
                 legend=False, level=True, cap=cap, heat=d["hh_type"][t],
                 norm=HH_NORM)


def heat_colorbar_kind(fig, rect, kind):
    cap = HEAT_MAPS[kind][2]
    return heat_colorbar(fig, rect, HH_NORM, HH_LABEL, "both",
                         [CAP_HANDLE] if cap is not None else None)


def heat_figure(d, kind, suptitle):
    """One panel per housing type of HEAT_MAPS[kind], colorbar below; 2x2
    grids spaced like figure 5."""
    types = HEAT_MAPS[kind][0]
    if len(types) == 3:
        fig = plt.figure(figsize=(19, 7.2))
        axes = [new3d(fig, 1, 3, k + 1) for k in range(3)]
        fig.subplots_adjust(wspace=0.02, bottom=0.12, top=0.95)
        rect = (0.3, 0.05, 0.4, 0.025)
    else:
        fig = plt.figure(figsize=(15, 13.2))
        axes = [new3d(fig, 2, 2, k + 1) for k in range(4)]
        fig.subplots_adjust(wspace=0.02, hspace=0.04, bottom=0.07, top=0.97)
        rect = (0.3, 0.03, 0.4, 0.011)
    for ax, t in zip(axes, types):
        draw_heat(ax, d, kind, t)
    heat_colorbar_kind(fig, rect, kind)
    fig.suptitle(suptitle, fontweight="bold", y=0.995)
    return fig


def panel_colorbar(fig, ax, kind):
    """Colorbar under one panel of a grid (call after subplots_adjust)."""
    pos = ax.get_position()
    return heat_colorbar_kind(
        fig, (pos.x0 + 0.2 * pos.width, pos.y0 - 0.03, 0.45 * pos.width,
              0.008), kind)

###############################################################################

# %% FIGURES 1-5: BENCHMARK 3D MAPS OF THE CITY STRUCTURE

print("FIGURES 1-5: BENCHMARK 3D MAPS")

## FOOTNOTE: (1)-(2) Number of households per 500 m grid cell in the benchmark
## (risk misperception, sandbags at full cost), stacked by housing type (1)
## or income group (2). (3) Housing quantity, i.e. floor space occupied in
## hectares (households x housing per household: FP dwelling size; IB and IS
## 14 m2 shacks; FS the 40 m2 RDP house plus the part of the 70 m2 backyard not
## rented out), stacked by housing type. (4) One panel per rented housing type
## (FP, IB, IS): annual rent per m2 (bar height, same z scale in all panels and
## scenarios), colored by the number of households of that type in the cell
## (log scale, same in all panels and scenarios; below 10 or above 10,000:
## end colors). (5) Maps (1)-(3) and the FP panel of (4). Basemap: City of
## Cape Town (light) and urban footprint (dark).

FILES_1_3 = {"households_type": "fig01_map_households_by_type.png",
             "households_group": "fig02_map_households_by_group.png",
             "housing": "fig03_map_housing_quantity.png"}
for kind, filename in FILES_1_3.items():
    fig = plt.figure(figsize=(9, 7.5))
    draw_map(new3d(fig, 111), B, kind, title=MAPS[kind][3] + " (benchmark)")
    save(fig, "maps_bench", filename)

fig = heat_figure(B, "rent", "Rent by housing type, colored by number of "
                  "households (benchmark)")
save(fig, "maps_bench", "fig04_map_rent.png")

fig = plt.figure(figsize=(15, 12.5))
for k, kind in enumerate(FILES_1_3):
    draw_map(new3d(fig, 2, 2, k + 1), B, kind,
             title=f"({k + 1}) {MAPS[kind][3]}")
ax = new3d(fig, 2, 2, 4)
draw_heat(ax, B, "rent", 0, title="(4) Formal private rent, colored by "
          "households")
fig.subplots_adjust(wspace=0.02, hspace=0.04)
panel_colorbar(fig, ax, "rent")
save(fig, "maps_bench", "fig05_maps_city_structure_2x2.png")

###############################################################################

# %% FIGURES 5a-5e: BENCHMARK 3D MAPS OF FLOOD RISKS

print("FIGURES 5a-5e: BENCHMARK FLOOD MAPS")

## FOOTNOTE: (5a) Expected annual flood depth averaged over each cell (cm),
## from the FATHOM fluvial and pluvial and DELTARES coastal maps (deepest
## flood type in each event), without climate change, with the surplus under
## climate change (fluvial and pluvial probabilities x2, 2050 sea-level-rise
## coastal maps) stacked on top; cells with land available for housing (the
## shares of households exposed are in table_exposure). Climate change does
## not double the expected depth where floods are frequent: the model
## interpolates depth linearly in the exceedance probability P, from zero at
## P = 1 up to the 5-year event, and doubling probabilities moves that event
## from 0.2 to 0.4 while the zero anchor stays at P = 1, so the frequent-flood
## triangle shrinks from 0.4 to 0.3 times the 5-year depth while the rest
## doubles (a cell flooded to the same depth at every return period goes from
## 0.6 to 0.7 times that depth, +17%). Cells flooded only by rare events
## double. (5b) Actual expected damages to structures and contents (M
## ZAR/year), stacked by housing type (formal structures attributed to their
## occupants). (5c) One panel per housing type: expected damages borne per
## household (net of insurance payouts) in % of the occupants' mean income
## (bar height, same z scale in all panels and scenarios, capped at 25%:
## triangles), colored by the number of households of that type in the cell
## (as in (4)). (5d) Number of informal settlement households by
## sandbag levels chosen (gray: unprotected). (5e) Maps (5a), (5b), the FP
## panel of (5c) and (5d).

solver_cells = B["solver_cells"] > 0
d0, d1 = DEPTH_CELL[0], DEPTH_CELL[1]
DEPTH_VALUES = np.where(
    solver_cells, np.array([np.minimum(d0, d1), np.maximum(d1 - d0, 0)]), 0)
MAPS["depth"] = ("Expected flood depth (cm/year)", CC_COLORS,
                 ["No climate change", "Surplus with climate change"],
                 "Expected flood depth", False, None, False)
m = solver_cells & (d0 > EXPOSURE_THRESHOLD)
print(f"  depth with CC / without, cells above 0.1 cm/year: median "
      f"{np.median(d1[m] / d0[m]):.2f}, ratio of sums {d1[m].sum() / d0[m].sum():.2f}; "
      f"cells above 10 cm/year: median "
      f"{np.median((d1 / np.maximum(d0, 1e-12))[m & (d0 > 10)]):.2f}")

FLOOD_MAPS = {"depth": "fig05a_map_flood_depth_cc.png",
              "damages_type": "fig05b_map_damages_by_type.png",
              "damages_relative": "fig05c_map_damages_relative.png",
              "protection": "fig05d_map_self_protection.png"}
for kind, filename in FLOOD_MAPS.items():
    if kind == "damages_relative":
        fig = heat_figure(B, kind, "Damages per household by housing type, "
                          "colored by number of households (benchmark)")
    else:
        fig = plt.figure(figsize=(9, 7.5))
        draw_map(new3d(fig, 111), B, kind,
                 values=DEPTH_VALUES if kind == "depth" else None,
                 title=MAPS[kind][3]
                 + ("" if kind == "depth" else " (benchmark)"))
    save(fig, "maps_bench", filename)

fig = plt.figure(figsize=(15, 12.5))
for k, (kind, tag) in enumerate(zip(FLOOD_MAPS, "abcd")):
    ax = new3d(fig, 2, 2, k + 1)
    if kind == "damages_relative":
        draw_heat(ax, B, kind, 0, title="(5c) Damages per formal private "
                  "household, colored by households")
        heat_ax = ax
    else:
        draw_map(ax, B, kind, values=DEPTH_VALUES if kind == "depth" else None,
                 title=f"(5{tag}) {MAPS[kind][3]}")
fig.subplots_adjust(wspace=0.02, hspace=0.04)
panel_colorbar(fig, heat_ax, "damages_relative")
save(fig, "maps_bench", "fig05e_maps_flood_risks_2x2.png")

###############################################################################

# %% FIGURES 6-12: 3D MAPS FOR EACH SCENARIO

print("FIGURES 6-12: SCENARIO MAPS")

## FOOTNOTE: Maps (1)-(3), (5b) and (5d) in each scenario, with the same z
## scale in all panels (see the footnotes of the benchmark maps). (9a)-(9c):
## the panels of map (4) for FP, IB and IS, one figure per housing type, with
## the same z and color scales as (4). (11a)-(11f): map (5c) for each
## scenario, one figure per scenario, with the same z and color scales as
## (5c).

SCENARIO_MAPS = {"households_type": "fig06_scenarios_households_by_type.png",
                 "households_group": "fig07_scenarios_households_by_group.png",
                 "housing": "fig08_scenarios_housing_quantity.png",
                 "damages_type": "fig10_scenarios_damages_by_type.png",
                 "protection": "fig12_scenarios_self_protection.png"}

for kind, filename in SCENARIO_MAPS.items():
    fig = plt.figure(figsize=(16.5, 11))
    zlabel, colors, labels, title, overlay, cap, level = MAPS[kind]
    top = max(bar_top(map_values(sim[s], kind), overlay, cap)
              for s in SCENARIOS)
    for k, s in enumerate(SCENARIOS):
        draw_map(new3d(fig, 2, 3, k + 1), sim[s], kind, title=SCEN_LABELS[s],
                 zlim=(0, top), legend=False)
    fig.legend(handles=legend_handles(colors, labels, cap), loc="lower center",
               ncol=len(labels) + (cap is not None), fontsize=10,
               title=title, title_fontproperties={"weight": "bold"})
    fig.subplots_adjust(wspace=0.0, hspace=0.02, bottom=0.07)
    save(fig, "maps_scen", filename)

RENT_FILES = ["fig09a_scenarios_rent_formal_private.png",
              "fig09b_scenarios_rent_backyards.png",
              "fig09c_scenarios_rent_settlements.png"]
for t, filename in zip(HEAT_MAPS["rent"][0], RENT_FILES):
    fig = plt.figure(figsize=(16.5, 11))
    for k, s in enumerate(SCENARIOS):
        draw_heat(new3d(fig, 2, 3, k + 1), sim[s], "rent", t,
                  title=SCEN_LABELS[s])
    fig.subplots_adjust(wspace=0.0, hspace=0.02, bottom=0.09)
    heat_colorbar_kind(fig, (0.3, 0.035, 0.4, 0.012), "rent")
    fig.suptitle(f"{HTYPES[t]}: rent, colored by number of households",
                 fontweight="bold", y=0.96)
    save(fig, "maps_scen", filename)

for s, tag in zip(SCENARIOS, "abcdef"):
    fig = heat_figure(sim[s], "damages_relative", "Damages per household by "
                      "housing type, colored by number of households "
                      f"({SCEN_LABELS[s]})")
    save(fig, "maps_scen", f"fig11{tag}_{s}_damages_relative.png")

###############################################################################

# %% FIGURES 13-18: BENCHMARK DISTRIBUTIONS ACROSS FLOOD DEPTH

print("FIGURES 13-18: BENCHMARK DISTRIBUTIONS")

## FOOTNOTE: Distributions over the expected annual flood depth of each
## household's dwelling (cm, log scale; per housing type, frequent pluvial
## floods are drained in formal housing, backyards and RDP premises), for
## exposed households (expected depth above 0.1 cm/year), by income group,
## with the same y axis in all panels. (13) Number of households, stacked by
## housing type, and the share of exposed households in the group. (14) Mean
## dwelling size in formal private housing (groups 2-4). (15) Mean rent by
## housing type (FP, IB, IS), overlaid: each bar rises to the lowest mean,
## then adds the surplus of each higher one. (16) Total expected damages by
## housing type (M ZAR/year), overlaid. (17) Mean damages borne per household
## by housing type (net of insurance payouts, % of the group's mean income),
## overlaid and capped at 25% (triangles); box: exposed households whose
## damages exceed the cap. (18) Number of exposed informal settlement
## households by sandbag levels (gray: unprotected).

BINS = np.logspace(np.log10(EXPOSURE_THRESHOLD),
                   np.log10(DEPTH.max() * 1.0001), 33)
CENTERS = np.sqrt(BINS[:-1] * BINS[1:])
WIDTHS = np.diff(BINS) * 0.92


def binned(weights):
    """Sum of weights (types, cells) by depth bin, (types, bins), with each
    housing type at its own (drained) depth; exposed households only."""
    out = np.zeros((4, len(BINS) - 1))
    for h in range(4):
        idx = np.digitize(DEPTH[h], BINS) - 1
        ok = EXPOSED[h] & (idx >= 0) & (idx < len(BINS) - 1)
        out[h] = np.bincount(idx[ok], weights=np.nan_to_num(weights[h][ok]),
                             minlength=len(BINS) - 1)
    return out


def _share(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0)


def dist_values(d, kind, i, total=False):
    """Values of distribution `kind` for income group i by depth bin:
    (categories, bins) by housing type (or sandbag level), or (bins,) over
    all of them if total (protection: protected households)."""
    hh = d["hh"][:, i]
    if kind == "households":
        v = binned(hh) / 1e3
        return v.sum(0) if total else v
    if kind == "consumption":                          # FP only
        num, den = binned(hh * d["consumption"])[:1], binned(hh)[:1]
        return _share(num, den)[0] if total else _share(num, den)
    if kind == "rent":                                 # FP, IB, IS
        num, den = binned(hh * d["rent_clean"])[:3], binned(hh)[:3]
        return _share(num.sum(0), den.sum(0)) if total else _share(num, den)
    if kind == "damages":
        v = binned(d["dmg"][:, i]) / 1e6
        return v.sum(0) if total else v
    if kind == "damages_relative":
        num, den = binned(d["dmg_net"][:, i]), binned(hh)
        if total:
            return 100 * _share(num.sum(0), den.sum(0)) / y_bar[i]
        return 100 * _share(num, den) / y_bar[i]
    if kind == "protection":                           # settlers by level
        v = np.array([binned(hh * (d["level"] == k)[None, :])[2]
                      for k in range(4)])
        return v[1:].sum(0) if total else v
    raise ValueError(kind)


DISTS = {
    # kind: (file stem, groups, colors, labels, overlay, cap, y label, title)
    "households": ("13_households", [0, 1, 2, 3], HTYPE_COLORS, HTYPES,
                   False, None, "Households (thousands)", "Households"),
    "consumption": ("14_housing_consumption", [1, 2, 3], HTYPE_COLORS[:1],
                    None, True, None, "Formal dwelling size (m²)",
                    "Formal private dwelling size"),
    "rent": ("15_rent", [0, 1, 2, 3], HTYPE_COLORS[:3], HTYPES[:3], True,
             None, "Rent (ZAR/m²/year)", "Rent by housing type (overlaid)"),
    "damages": ("16_damages", [0, 1, 2, 3], HTYPE_COLORS, HTYPES, True,
                None, "Damages (M ZAR/year)",
                "Expected damages by housing type (overlaid)"),
    "damages_relative": ("17_damages_relative", [0, 1, 2, 3], HTYPE_COLORS,
                         HTYPES, True, DAMAGE_CAP,
                         "Damages per household (% of mean income)",
                         "Damages per household by housing type (overlaid)"),
    "protection": ("18_self_protection", [0], LEVEL_COLORS, LEVEL_LABELS,
                   False, None, "Exposed settlement households",
                   "Self-protection of exposed settlers"),
}


def style_hist(ax, ylabel, title, bins=None,
               xlabel="Expected flood depth (cm/year, log scale)"):
    bins = BINS if bins is None else bins
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)


def dist_figure(groups):
    n = len(groups)
    nrow, ncol = (2, 2) if n == 4 else (1, n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12 if n > 1 else 7,
                                                  4.2 * nrow),
                             squeeze=False, sharey=True)
    for ax in axes.flat:
        ax.yaxis.set_tick_params(labelleft=True)
    return fig, axes, nrow


def above_cap_box(d, i):
    """Exposed households of group i whose damages exceed DAMAGE_CAP."""
    hh = d["hh"][:, i] * EXPOSED
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = 100 * d["dmg_net"][:, i] / d["hh"][:, i] / y_bar[i]
    n_above = hh[np.nan_to_num(rel) > DAMAGE_CAP].sum()
    return (f"Above {DAMAGE_CAP}%: {n_above:,.0f} households "
            f"({100 * n_above / max(hh.sum(), 1):.1f}% of exposed)")


for kind, (stem, groups, colors, labels, overlay, cap, ylabel,
           title) in DISTS.items():
    fig, axes, nrow = dist_figure(groups)
    top = max(bar_top(dist_values(B, kind, i), overlay, cap) for i in groups)
    for ax, i in zip(axes.flat, groups):
        bars2d(ax, CENTERS, WIDTHS, dist_values(B, kind, i), colors, overlay,
               cap)
        style_hist(ax, ylabel, INCGRPS[i])
        ax.set_ylim(0, 1.18 * top)
        note = None
        if kind == "households":
            note = (f"Exposed: {exposed_share(B, groups=[i]):.1f}% of "
                    f"{B['hh_group'][i].sum():,.0f} households")
        elif kind == "damages_relative":
            note = above_cap_box(B, i)
        elif kind == "protection":
            note = protection_box(B)
        if note:
            ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top",
                    fontsize=8, color=INK2, bbox=BOX)
    fig.suptitle(f"{title} across expected flood depth (benchmark)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if nrow > 1 else 0.84))
    if labels:
        fig.legend(handles=legend_handles(colors, labels, cap),
                   loc="upper left", ncol=len(labels) + (cap is not None),
                   fontsize=8,
                   bbox_to_anchor=(0.005, 0.965 if nrow > 1 else 0.925))
    save(fig, "dist_bench", f"fig{stem}_benchmark.png")

###############################################################################

# %% FIGURES 19-24b: DISTRIBUTIONS ACROSS FLOOD DEPTH, BY SCENARIO

print("FIGURES 19-24b: SCENARIO DISTRIBUTIONS")

## FOOTNOTE: Same distributions as (13)-(18), for the benchmark and the four
## single-policy scenarios, with the same y axis in all panels: step lines,
## the benchmark in black, drawn on top. (19)-(24): by income group, over all
## housing types (24: exposed settlers protected by sandbags). (19b)-(24b):
## by housing type, over all income groups (20b: FP dwelling size and FS
## house plus the part of the backyard not rented out; IB and IS shacks are
## all 14 m2; 21b: FP, IB and IS; 24b: settlements). (19), (19b) Vertical
## lines: mean expected depth of each distribution. (23), (23b) Damages
## borne in % of the occupants' mean income, capped at 25% (triangles).
## Depth is the expected flood depth before public or self-protection, so the
## distributions only move with relocation.

DIST_SCENARIOS = [s for s in SCENARIOS if s != "combined"]
SCEN_FILES = {"households": "fig19_households", "consumption":
              "fig20_housing_consumption", "rent": "fig21_rent",
              "damages": "fig22_damages", "damages_relative":
              "fig23_damages_relative", "protection": "fig24_self_protection"}
SCEN_YLABELS = {"rent": "Rent (ZAR/m²/year)",
                "protection": "Protected exposed settlers"}
# By housing type: (housing types, y label, title)
TYPE_DISTS = {
    "households": ([0, 1, 2, 3], "Households (thousands)", "Households"),
    "consumption": ([0, 3], "Dwelling size (m²)", "Dwelling size"),
    "rent": ([0, 1, 2], "Rent (ZAR/m²/year)", "Rent"),
    "damages": ([0, 1, 2, 3], "Damages (M ZAR/year)", "Expected damages"),
    "damages_relative": ([0, 1, 2, 3],
                         "Damages per household (% of mean income)",
                         "Damages per household"),
    "protection": ([2], "Protected exposed settlers",
                   "Self-protection of exposed settlers"),
}


def mean_depth(d, i):
    """Household-weighted mean expected depth of exposed group-i households."""
    w = d["hh"][:, i] * EXPOSED
    return (w * DEPTH).sum() / w.sum()


def mean_depth_type(d, t):
    """Mean expected depth of exposed households in housing type t."""
    w = d["hh_type"][t] * EXPOSED[t]
    return (w * DEPTH[t]).sum() / w.sum()


def dist_values_type(d, kind, t):
    """Values of distribution `kind` for housing type t by depth bin, over
    all income groups."""
    hh = d["hh_type"]
    if kind == "households":
        return binned(hh)[t] / 1e3
    if kind in ("consumption", "rent"):
        level = d["consumption"] if kind == "consumption" else d["rent_clean"]
        return _share(binned(hh * level)[t], binned(hh)[t])
    if kind == "damages":
        return binned(d["dmg"].sum(1))[t] / 1e6
    if kind == "damages_relative":
        income = (d["hh"] * y_bar[None, :, None]).sum(1)
        return 100 * _share(binned(d["dmg_net"].sum(1))[t], binned(income)[t])
    if kind == "protection":
        return binned(hh * (d["level"] >= 1)[None, :])[t]
    raise ValueError(kind)


def scenario_dists(panels, titles, value, ylabel, cap, mean, suptitle,
                   filename, bins=None, xlabel=None, refline=None):
    """Step lines of value(d, panel) by bin (depth bins unless bins) for
    DIST_SCENARIOS, one panel per item of panels, same y axis; mean(d,
    panel): vertical lines; refline: (x, label) of a dotted reference line."""
    bins = BINS if bins is None else bins
    xlabel = xlabel or "Expected flood depth (cm/year, log scale)"
    fig, axes, nrow = dist_figure(panels)
    values = {(s, p): value(sim[s], p) for s in DIST_SCENARIOS for p in panels}
    top = max(np.minimum(v, cap if cap else np.inf).max()
              for v in values.values()) or 1
    for ax, p, title in zip(axes.flat, panels, titles):
        for s in DIST_SCENARIOS[1:] + ["benchmark"]:  # benchmark on top
            v, over = cap_values(values[(s, p)][None, :], cap)
            ax.step(bins, np.r_[v[0], v[0][-1]], where="post",
                    color=SCEN_COLORS[s], linestyle=SCEN_STYLES[s],
                    linewidth=1.2, zorder=20 if s == "benchmark" else 3,
                    label=SCEN_LABELS[s])
            if cap and over.any():
                ax.scatter(CENTERS[over], np.full(over.sum(), cap),
                           marker="^", s=22, color=SCEN_COLORS[s], zorder=21,
                           clip_on=False)
        if mean:
            for s in DIST_SCENARIOS:
                ax.axvline(mean(sim[s], p), color=SCEN_COLORS[s],
                           linestyle=SCEN_STYLES[s], linewidth=1.1,
                           zorder=22)
        if refline:
            ax.axvline(refline[0], color=MUTED, linestyle=":", linewidth=1,
                       zorder=1)
        style_hist(ax, ylabel, title, bins, xlabel)
        ax.set_ylim(0, 1.1 * top)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = [labels.index(SCEN_LABELS[s]) for s in DIST_SCENARIOS]
    handles, labels = [handles[k] for k in order], [labels[k] for k in order]
    if refline:
        handles.append(Line2D([], [], color=MUTED, linestyle=":"))
        labels.append(refline[1])
    if cap:
        handles.append(CAP_HANDLE)
        labels.append(CAP_HANDLE.get_label())
    if mean:
        handles.append(Line2D([], [], color=INK2, linewidth=1))
        labels.append("Vertical lines: mean depth")
    fig.suptitle(suptitle, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92 if nrow > 1 else 0.8))
    fig.legend(handles, labels, loc="upper left", ncol=4, fontsize=8,
               bbox_to_anchor=(0.005, 0.965 if nrow > 1 else 0.93))
    save(fig, "dist_scen", filename)


for kind, stem in SCEN_FILES.items():
    groups, cap = DISTS[kind][1], DISTS[kind][5]
    title = (DISTS[kind][7].replace(" (overlaid)", "")
             .replace(" by housing type", ""))
    scenario_dists(
        groups, [INCGRPS[i] for i in groups],
        lambda d, i: dist_values(d, kind, i, total=True),
        SCEN_YLABELS.get(kind, DISTS[kind][6]), cap,
        mean_depth if kind == "households" else None,
        f"{title} across expected flood depth, by scenario and income group",
        f"{stem}_scenarios.png")
    types, ylabel, title = TYPE_DISTS[kind]
    scenario_dists(
        types, [HTYPES[t] for t in types],
        lambda d, t: dist_values_type(d, kind, t), ylabel, DISTS[kind][5],
        mean_depth_type if kind == "households" else None,
        f"{title} across expected flood depth, by scenario and housing type",
        stem.replace("_", "b_", 1) + "_scenarios_by_type.png")

###############################################################################

# %% FIGURES 18c, 24c-24d: HOUSEHOLDS ACROSS DAMAGES PER HOUSEHOLD

print("FIGURES 18c, 24c-24d: HOUSEHOLDS ACROSS DAMAGES PER HOUSEHOLD")

## FOOTNOTE: Number of households (thousands) across their expected damages
## per household: damages borne (net of insurance payouts) in % of their
## group's mean income, log scale, by cell, housing type and income group
## (below 0.001% in the first bin). Households without expected damages are
## left out (share in the notes of 18c). (18c) Benchmark, by income group,
## stacked by housing type. (24c) Scenarios (benchmark and single policies)
## by income group; (24d) by housing type; step lines, benchmark in black.
## Same y axis in all panels; dotted line: 25% of income.

DMG_BINS = np.logspace(-3, np.log10(300), 34)
DMG_CENTERS = np.sqrt(DMG_BINS[:-1] * DMG_BINS[1:])
DMG_WIDTHS = np.diff(DMG_BINS) * 0.92
DMG_XLABEL = "Expected damages per household (% of mean income, log scale)"


def damages_per_household(d):
    """Damages borne per household in % of the group's mean income, (types,
    groups, cells); 0 without households."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = 100 * d["dmg_net"] / d["hh"] / y_bar[None, :, None]
    return np.where(d["hh"] > 0, np.nan_to_num(rel), 0)


def binned_damages(d, types, groups):
    """Households of these housing types and income groups (lists) by bin
    of damages per household, (bins,), in thousands."""
    rel = damages_per_household(d)[np.ix_(types, groups)]
    hh = d["hh"][np.ix_(types, groups)]
    ok = rel > 0
    idx = np.clip(np.digitize(rel[ok], DMG_BINS) - 1, 0, len(DMG_BINS) - 2)
    return np.bincount(idx, weights=hh[ok], minlength=len(DMG_BINS) - 1) / 1e3


def no_damage_note(d, i):
    hh = d["hh"][:, i]
    zero = hh[(damages_per_household(d)[:, i] <= 0) & (hh > 0)].sum()
    return (f"Without expected damages: {zero:,.0f} households "
            f"({100 * zero / hh.sum():.1f}%)")


ALL4 = [0, 1, 2, 3]
fig, axes, nrow = dist_figure(ALL4)
values = {i: np.array([binned_damages(B, [t], [i]) for t in range(4)])
          for i in ALL4}
top = max(bar_top(v) for v in values.values())
for ax, i in zip(axes.flat, ALL4):
    bars2d(ax, DMG_CENTERS, DMG_WIDTHS, values[i], HTYPE_COLORS)
    ax.axvline(DAMAGE_CAP, color=MUTED, linestyle=":", linewidth=1, zorder=1)
    style_hist(ax, "Households (thousands)", INCGRPS[i], DMG_BINS, DMG_XLABEL)
    ax.set_ylim(0, 1.18 * top)
    ax.text(0.02, 0.97, no_damage_note(B, i), transform=ax.transAxes,
            va="top", fontsize=8, color=INK2, bbox=BOX)
fig.suptitle("Households across expected damages per household (benchmark)",
             fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.legend(handles=legend_handles(HTYPE_COLORS, HTYPES)
           + [Line2D([], [], color=MUTED, linestyle=":", label="25% of income")],
           loc="upper left", ncol=5, fontsize=8, bbox_to_anchor=(0.005, 0.965))
save(fig, "dist_bench", "fig18c_households_over_damages_benchmark.png")

for panels, titles, value, by, filename in (
        (ALL4, INCGRPS, lambda d, i: binned_damages(d, ALL4, [i]),
         "income group", "fig24c_households_over_damages_scenarios.png"),
        (ALL4, HTYPES, lambda d, t: binned_damages(d, [t], ALL4),
         "housing type",
         "fig24d_households_over_damages_scenarios_by_type.png")):
    scenario_dists(panels, titles, value, "Households (thousands)", None,
                   None, "Households across expected damages per household, "
                   f"by scenario and {by}", filename, DMG_BINS, DMG_XLABEL,
                   (DAMAGE_CAP, "25% of income"))

###############################################################################

# %% WELFARE, LANDLORDS AND SCENARIO METRICS

print("WELFARE AND METRICS")

## Welfare rows: low-income households outside and inside RDP housing, and
## groups 2-4; masks over (housing types, income groups)
ROWS = ["Low income, non-beneficiaries", "Low income, RDP owners",
        "Mid-low income", "Mid-high income", "High income"]
ROW_MASKS = []
for r in range(5):
    mask = np.zeros((4, 4), bool)
    if r == 0:
        mask[:3, 0] = True
    elif r == 1:
        mask[3, 0] = True
    else:
        mask[:, r - 1] = True
    ROW_MASKS.append(mask)
ROW_INCOME = np.array([y_bar[0], y_bar[0], y_bar[1], y_bar[2], y_bar[3]])
ROW_SHARE = np.array([100 * (B["hh"] * m[:, :, None]).sum() / N_TOTAL
                      for m in ROW_MASKS])
ROW_LABELS = [f"{name} ({share:.1f}% of households)"
              for name, share in zip(ROWS, ROW_SHARE)]


def row_weights(d, r, exposed):
    w = d["hh"] * ROW_MASKS[r][:, :, None]
    return w * EXP3 if exposed else w


def mean_utility(u, w):
    ok = (w > 0) & np.isfinite(u)
    return np.sum(w[ok] * u[ok]) / w[ok].sum()


def welfare(d, exposed=False):
    """Welfare of each row in scenario d relative to the benchmark (on its
    exposed households if exposed): n, benchmark households; u, mean
    ex-post utility in d; du, its % change; ev, equivalent variation per
    benchmark household (rands/year); share, % of the row exposed in d.

    EV as in scenarios.welfare_table, at benchmark prices and locations,
    with the ratio r of mean ex-post utilities (of exposed households if
    exposed): FP (ỹ - R q0)(r - 1); IB, IS c z_post (r^(1/α) - 1); RDP
    owners, with their own ratio in each cell."""
    res = {key: np.zeros(5) for key in ("n", "u", "du", "ev", "share")}
    for r in range(5):
        wb, ws = row_weights(B, r, exposed), row_weights(d, r, exposed)
        res["n"][r] = wb.sum()
        res["share"][r] = 100 * (d["hh"] * ROW_MASKS[r][:, :, None]
                                 * EXP3).sum() / row_weights(d, r, False).sum()
        ub = mean_utility(B["ex_post_u_post"], wb)
        us = mean_utility(d["ex_post_u_post"], ws)
        res["u"][r], res["du"][r] = us, 100 * (us / ub - 1)
        if d is B:
            continue
        if r == 1:                                     # RDP owners
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = d["ex_post_u_post"][3, 0] / B["ex_post_u_post"][3, 0]
                ev = np.nan_to_num(B["ex_post_price"][3, 0]
                                   * B["ex_post_z_post"][3, 0]
                                   * (ratio ** (1 / alpha) - 1))
            res["ev"][r] = np.sum(wb[3, 0] * ev) / wb[3, 0].sum()
            continue
        i, ratio = (0 if r == 0 else r - 1), us / ub
        ev = np.zeros((4, N_CELLS))
        ev[0] = (B["net_income"][i] - np.nan_to_num(B["rent"][0]) * q0) \
            * (ratio - 1)
        for h in (1, 2):
            ev[h] = B["ex_post_price"][h, i] * B["ex_post_z_post"][h, i] \
                * (ratio ** (1 / alpha) - 1)
        w = wb[:, i]
        res["ev"][r] = np.sum(np.where(w > 0, w * np.nan_to_num(ev), 0)) \
            / w.sum()
    return res


def landlords(d):
    """Yearly revenues (rands) of absentee landlords: formal gross (land
    rents developers would pay before compensating their unanticipated
    losses, i.e. δ P + t per m2 of developed land), formal net (gross minus
    the expected compensation of developers' losses and of RDP owners'
    uncovered damages, the utility-floor transfers), and slumlords' rents
    from informal settlements."""
    t, losses, revenue = d["developer_tax"]
    gross = d["landlord_revenue"][0].sum() + revenue
    net = gross - losses - d["surplus_damages"].sum()
    return gross, net, d["landlord_revenue"][1].sum()


SUMMARY_ROWS = ["Damages (M ZAR)", "Damages, exposed (% of income)",
                "Exposed households (%)", "Exposed settlers protected (%)",
                "Public cost of PS and SI (M ZAR)",
                "Developers' losses (M ZAR)", "Uncovered RDP damages (M ZAR)"]
SHORT_ROWS = ["Low inc., non-benef.", "Low inc., RDP owners", "Mid-low inc.",
              "Mid-high inc.", "High inc."]


def metrics(d, label):
    """Scalar outcomes of scenario d: for all households (levels of the
    summary rows, welfare changes relative to the benchmark) and for exposed
    households."""
    info = SCEN["scenarios"][label]
    budget = info["budget"] or [0, 0, 0]
    exposed_hh = d["hh"] * EXP3
    income = (exposed_hh * y_bar[None, :, None]).sum()
    out = {
        "Damages (M ZAR)": d["dmg"].sum() / 1e6,
        "Damages, exposed (% of income)":
            100 * (d["dmg_net"] * EXP3).sum() / income,
        "Exposed households (%)": 100 * exposed_hh.sum() / d["hh"].sum(),
        "Exposed settlers protected (%)": settler_shares(d)[0],
        "Public cost of PS and SI (M ZAR)": (budget[0] + budget[1]) / 1e6,
        "Developers' losses (M ZAR)": d["developer_tax"][1] / 1e6,
        "Uncovered RDP damages (M ZAR)": d["surplus_damages"].sum() / 1e6,
    }
    out["Aggregate expenses (M ZAR)"] = sum(out[k] for k in SUMMARY_ROWS[4:])
    exp = {"Damages of exposed households (M ZAR)":
           (d["dmg"] * EXP3).sum() / 1e6,
           "Damages, exposed (% of income)":
           out["Damages, exposed (% of income)"]}
    for target, w in ((out, welfare(d)), (exp, welfare(d, exposed=True))):
        if target is exp:
            for r, name in enumerate(SHORT_ROWS):
                target[f"Exposed, {name} (% of group)"] = w["share"][r]
        for r, name in enumerate(SHORT_ROWS):
            target[f"Utility, {name} (%)"] = w["du"][r]
        for r, name in enumerate(SHORT_ROWS):
            target[f"EV, {name} (ZAR)"] = w["ev"][r]
        target["Utility, population-weighted (%)"] = \
            np.sum(w["n"] * w["du"]) / w["n"].sum()
        target["EV, population-weighted (ZAR)"] = \
            np.sum(w["n"] * w["ev"]) / w["n"].sum()
    net = (landlords(d)[1] - landlords(B)[1]) / N_TOTAL
    out["Landlords' net revenue (ZAR/household)"] = net
    out["EV + landlords' net revenue (ZAR/household)"] = \
        out["EV, population-weighted (ZAR)"] + net
    return out, exp


METRICS = {s: metrics(sim[s], s) for s in SCENARIOS}

###############################################################################

# %% TABLES 25-27b

print("TABLES 25-27b")


def latex_table(df, path, fmt):
    """Write a booktabs tabular: df has a (block, row) MultiIndex, and
    fmt(block, row, column, value) gives the cell text."""
    cols = list(df.columns)
    esc = lambda s: str(s).replace("%", r"\%")
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join([""] + [esc(c) for c in cols]) + r" \\", r"\midrule"]
    for k, block in enumerate(df.index.get_level_values(0).unique()):
        if k:
            lines.append(r"\addlinespace")
        if block:
            lines.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}"
                         rf"{{\textit{{{esc(block)}}}}} \\")
        for row, values in df.loc[block].iterrows():
            cells = [fmt(block, row, c, v) for c, v in zip(cols, values)]
            lines.append((r"\quad " if block else "") + esc(row) + " & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _num(v, digits=1, sign=False):
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"


def add_sum_interaction(row, level):
    """Sum of single-policy changes and interaction (combined change minus
    that sum); level: the scenario columns hold levels, not changes."""
    base = row["benchmark"] if level else 0
    row["sum"] = sum(row[s] - base for s in SINGLES)
    row["interaction"] = row["combined"] - base - row["sum"]
    return row


COLS_LEVELS = ["benchmark"] + SCENARIOS[1:]
COLS_SCEN = COLS_LEVELS + ["sum", "interaction"]

## TABLE 25: BENCHMARK DAMAGES AND EXPOSURE BY HOUSING TYPE AND INCOME GROUP

## FOOTNOTE: Benchmark. Damages: actual expected annual damages to structures
## and contents (M ZAR/year; formal structures attributed to occupants).
## Damages for exposed households: damages borne (net of insurance payouts) as
## a % of the exposed households' mean income. Exposed: households with
## expected flood depth above 0.1 cm/year at their dwelling (before
## protection; EXPOSURE_THRESHOLD). Self-protected: share of exposed
## households protected by sandbags (only settlers can protect). Dashes: no
## households.

METRICS25 = ["Damages (M ZAR)", "Damages, exposed (% of income)",
             "Exposed households (%)", "Exposed households (thousands)",
             "Self-protected (% of exposed)"]


def exposure_stats(d, types, groups):
    """Metrics of METRICS25 over housing types and income groups (lists)."""
    hh = d["hh"][np.ix_(types, groups)]
    exp = EXPOSED[types][:, None, :]
    n, n_exp = hh.sum(), (hh * exp).sum()
    income = (hh * exp * y_bar[groups][None, :, None]).sum()
    dmg = d["dmg"][np.ix_(types, groups)].sum()
    dmg_net = (d["dmg_net"][np.ix_(types, groups)] * exp).sum()
    prot = 0.
    if 2 in types:
        k = types.index(2)
        prot = (hh[k] * exp[k] * (d["level"] >= 1)).sum()
    if n <= 0:
        return [np.nan] * 5
    return [dmg / 1e6, 100 * dmg_net / income if income else np.nan,
            100 * n_exp / n, n_exp / 1e3,
            100 * prot / n_exp if n_exp else np.nan]


COLS = INCGRPS + ["All"]
rows = {}
for t, label in enumerate(HTYPES + ["All housing types"]):
    types = [t] if t < 4 else [0, 1, 2, 3]
    for j, col in enumerate(COLS):
        groups = [j] if j < 4 else [0, 1, 2, 3]
        stats = exposure_stats(B, types, groups)
        if t in (0, 1, 3):                    # no self-protection
            stats[4] = np.nan
        for m, v in zip(METRICS25, stats):
            rows.setdefault((label, m), {})[col] = v
table25 = pd.DataFrame(rows).T[COLS]
table25.to_excel(FOLDERS["tables"] / "table25_damages_exposure_benchmark.xlsx",
                 float_format="%.3f")
DIGITS = dict(zip(METRICS25, [1, 2, 1, 1, 1]))
latex_table(table25, FOLDERS["tables"] / "table25_damages_exposure_benchmark.tex",
            lambda block, row, c, v: _num(v, DIGITS[row]))

## TABLE 26: CHANGES ACROSS SCENARIOS BY INCOME GROUP

## FOOTNOTE: Benchmark levels (first column) and changes relative to the
## benchmark of the metrics of Table 25, summed over housing types: damages in
## % changes, shares in percentage points, exposed households in thousands.
## Sum of singles: sum of the changes of the four single-policy scenarios.
## Interaction: combined scenario minus that sum.

METRICS26 = ["Damages (M ZAR; changes in %)",
             "Damages, exposed (% of income; changes in pp)",
             "Exposed households (%; changes in pp)",
             "Exposed households (thousands)",
             "Self-protected (% of exposed; changes in pp)"]
rows = {}
for j, grp in enumerate(INCGRPS + ["All"]):
    groups = [j] if j < 4 else [0, 1, 2, 3]
    stats = {s: exposure_stats(sim[s], [0, 1, 2, 3], groups)
             for s in SCENARIOS}
    for m, metric in enumerate(METRICS26):
        base = stats["benchmark"][m]
        row = {"benchmark": base}
        for s in SCENARIOS[1:]:
            v = stats[s][m]
            row[s] = 100 * (v / base - 1) if m == 0 else v - base
        rows[(metric, grp)] = add_sum_interaction(row, level=False)
table26 = pd.DataFrame(rows).T[COLS_SCEN]
table26.columns = [SCEN_SHORT[c] for c in COLS_SCEN]
table26.to_excel(FOLDERS["tables"] / "table26_damages_exposure_scenarios.xlsx",
                 float_format="%.3f")
DIGITS26 = dict(zip(METRICS26, [1, 3, 2, 2, 1]))
latex_table(table26, FOLDERS["tables"] / "table26_damages_exposure_scenarios.tex",
            lambda block, row, c, v: _num(v, DIGITS26[block],
                                          sign=c != "Bench."))

## TABLES 27 AND 27b: WELFARE (ALL HOUSEHOLDS; EXPOSED HOUSEHOLDS)

## FOOTNOTE: Ex-post utilities (with actual instead of perceived damages,
## accounting.ex_post): benchmark levels (first column; household-weighted
## means, RDP owners' mean over cells) and % changes. EV: equivalent
## variation per benchmark household, at benchmark prices and locations (see
## welfare), in ZAR/year and in % of the group's mean income ȳ (benchmark
## column: ȳ). Labels: share of all households. Taxes financing PS and SI
## (groups 3-4) are included. Landlords: gross revenue, land rents developers
## would pay before compensating their unanticipated losses (δ P + t per m2
## of developed land), and net revenue, minus the expected compensation of
## developers' losses and of RDP owners' uncovered damages (utility-floor
## transfers); slumlords: rents of informal settlements; all per city
## household (benchmark: levels; scenarios: changes). Aggregate:
## population-weighted means of utility changes and of EVs, and the EV mean
## plus the change in landlords' net revenue per household (slumlords
## excluded). Public protection is free in the model: its aggregate gain is
## the largest annual cost at which it breaks even. Changes below about 0.3%
## for groups 1-2 are within solver noise (group 1's equilibrium utility is
## pinned by lumpy cells). Table 27b: the same for exposed households only
## (benchmark exposed households for EVs; each scenario's exposed households
## for utilities), with the % of each group exposed in each scenario.


def welfare_table(exposed):
    res = {s: welfare(sim[s], exposed) for s in SCENARIOS}
    rows = {}
    for r, label in enumerate(ROW_LABELS):
        if exposed:
            rows[(label, "Exposed (% of group)")] = {
                s: res[s]["share"][r] for s in SCENARIOS}
        rows[(label, "Utility")] = {"benchmark": res["benchmark"]["u"][r],
                                    **{s: res[s]["du"][r]
                                       for s in SCENARIOS[1:]}}
        rows[(label, "EV (ZAR/year)")] = {
            "benchmark": ROW_INCOME[r],
            **{s: res[s]["ev"][r] for s in SCENARIOS[1:]}}
        rows[(label, "EV (% of income)")] = {
            "benchmark": np.nan,
            **{s: 100 * res[s]["ev"][r] / ROW_INCOME[r]
               for s in SCENARIOS[1:]}}
    if not exposed:
        acc = {s: np.array(landlords(sim[s])) / N_TOTAL for s in SCENARIOS}
        for k, (block, name) in enumerate(
                [("Landlords", "Gross revenue (ZAR/household/year)"),
                 ("Landlords", "Net revenue (ZAR/household/year)"),
                 ("Slumlords", "Revenue (ZAR/household/year)")]):
            rows[(block, name)] = {"benchmark": acc["benchmark"][k],
                                   **{s: acc[s][k] - acc["benchmark"][k]
                                      for s in SCENARIOS[1:]}}
    n = res["benchmark"]["n"]
    agg = "Aggregate (exposed)" if exposed else "Aggregate"
    rows[(agg, "Utility (%, population-weighted)")] = {
        "benchmark": np.nan, **{s: np.sum(n * res[s]["du"]) / n.sum()
                                for s in SCENARIOS[1:]}}
    rows[(agg, "EV (ZAR/household/year)")] = {
        "benchmark": np.sum(n * ROW_INCOME) / n.sum(),
        **{s: np.sum(n * res[s]["ev"]) / n.sum() for s in SCENARIOS[1:]}}
    if not exposed:
        rows[(agg, "EV + landlords' net revenue (ZAR/household/year)")] = {
            "benchmark": np.nan,
            **{s: rows[(agg, "EV (ZAR/household/year)")][s]
               + rows[("Landlords", "Net revenue (ZAR/household/year)")][s]
               for s in SCENARIOS[1:]}}
    table = pd.DataFrame(rows).T[COLS_LEVELS]
    table.columns = [SCEN_SHORT[c] for c in COLS_LEVELS]
    return table


def fmt_welfare(block, row, c, v):
    if c == "Bench." or row.startswith("Exposed (%"):
        return _num(v, 0 if np.isfinite(v) and abs(v) > 100 else 1)
    return _num(v, 3 if row.startswith(("Utility", "EV (%")) else 1,
                sign=True)


for exposed, stem in ((False, "table27_welfare"),
                      (True, "table27b_welfare_exposed")):
    table = welfare_table(exposed)
    table.to_excel(FOLDERS["tables"] / f"{stem}.xlsx", float_format="%.3f")
    latex_table(table, FOLDERS["tables"] / f"{stem}.tex", fmt_welfare)
    print(table.round(3).to_string())
    if not exposed:
        table27 = table

## TABLE: SCENARIO SUMMARY

## FOOTNOTE: Levels by scenario. Damages: actual expected annual damages
## (gross). Damages, exposed: damages
## borne by exposed households (net of insurance payouts) as a % of their
## mean income. Exposed: expected depth above 0.1 cm/year. Public cost: free
## sandbags and insurance payouts, financed by lump-sum taxes on groups 3-4
## (public protection is free in the model). Developers' losses: their
## unanticipated structure losses under risk misperception, financed by a
## tax on developed land. Uncovered RDP damages: utility-floor transfers to
## RDP owners, financed by landlords.

rows = {name: {s: METRICS[s][0][name] for s in SCENARIOS}
        for name in SUMMARY_ROWS}
table_summary = pd.DataFrame(rows).T[COLS_LEVELS]
table_summary.columns = [SCEN_SHORT[c] for c in COLS_LEVELS]
table_summary.to_excel(FOLDERS["tables"] / "table_scenario_summary.xlsx",
                       float_format="%.3f")
latex_table(pd.concat({"": table_summary}),
            FOLDERS["tables"] / "table_scenario_summary.tex",
            lambda block, row, c, v: _num(v, 2 if "% of income" in row else 1))
print(table_summary.round(2).to_string())

## TABLE: EXPOSURE BY HOUSING TYPE AND INCOME GROUP, BY SCENARIO

## FOOTNOTE: % of the households of each housing type and income group
## exposed (expected flood depth above 0.1 cm/year at their dwelling, before
## protection), by scenario; informal settlements: also the % of exposed
## settlers using sandbags. Dashes: no households (groups 3-4 have no access
## to backyards and settlements, and RDP owners are all in group 1).

rows = {}
for t, block in enumerate(HTYPES + ["All housing types"]):
    types = [t] if t < 4 else [0, 1, 2, 3]
    for j, grp in enumerate(INCGRPS + ["All"]):
        groups = [j] if j < 4 else [0, 1, 2, 3]
        rows[(block, grp)] = {s: exposed_share(sim[s], types, groups)
                              for s in SCENARIOS}
    if t == 2:
        rows[(block, "Exposed using sandbags")] = {
            s: settler_shares(sim[s])[0] for s in SCENARIOS}
table_exposure = pd.DataFrame(rows).T[COLS_LEVELS]
table_exposure.columns = [SCEN_SHORT[c] for c in COLS_LEVELS]
table_exposure.to_excel(FOLDERS["tables"] / "table_exposure.xlsx",
                        float_format="%.3f")
latex_table(table_exposure, FOLDERS["tables"] / "table_exposure.tex",
            lambda block, row, c, v: _num(v, 1))
print(table_exposure.round(1).to_string())

###############################################################################

# %% FIGURES 28-29a: HOUSING TYPES AND DAMAGES ACROSS SCENARIOS

print("FIGURES 28-29a: SCENARIO BARS")

## FOOTNOTE: By income group and for the whole city, with the same y axis in
## all panels. (28) Number of households by housing type, stacked. (29)
## Actual expected damages (M ZAR/year) by housing type, stacked; diamonds:
## damages net of insurance payouts where they differ. (29a) Damages borne by
## exposed households (net of payouts) in % of their mean income, by housing
## type, overlaid (each bar rises to the lowest value, then adds the surplus of
## each higher one); pies: share of the group's households exposed.


def scenario_bar_values(kind, groups):
    """(housing types, scenarios) values for income groups `groups`."""
    out = np.zeros((4, len(SCENARIOS)))
    for k, s in enumerate(SCENARIOS):
        d = sim[s]
        if kind == "households":
            out[:, k] = d["hh"][:, groups].sum((1, 2)) / 1e3
        elif kind == "damages":
            out[:, k] = d["dmg"][:, groups].sum((1, 2)) / 1e6
        else:                                          # % of income, exposed
            num = (d["dmg_net"][:, groups] * EXP3).sum((1, 2))
            den = (d["hh"][:, groups] * EXP3
                   * y_bar[groups][None, :, None]).sum((1, 2))
            out[:, k] = 100 * _share(num, den)
    return out


def scenario_bars(kind, filename, ylabel, title):
    panels = [[i] for i in range(4)] + [[0, 1, 2, 3]]
    overlay = kind == "damages_relative"
    values = [scenario_bar_values(kind, g) for g in panels]
    top = max(bar_top(v, overlay) for v in values)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    x = np.arange(len(SCENARIOS))
    for k, (ax, groups) in enumerate(zip(axes.flat, panels)):
        bars2d(ax, x, np.full(len(x), 0.62), values[k], HTYPE_COLORS, overlay)
        if kind == "damages":
            net = np.array([sim[s]["dmg_net"][:, groups].sum()
                            for s in SCENARIOS]) / 1e6
            gross = values[k].sum(0)
            diff = np.abs(net - gross) > 1e-3 * max(gross.max(), 1e-9)
            ax.scatter(x[diff], net[diff], marker="D", s=22, color=INK,
                       zorder=5)
        if overlay:                    # pies: share of households exposed
            band = blended_transform_factory(ax.transData, ax.transAxes)
            for j, s in enumerate(SCENARIOS):
                share = exposed_share(sim[s], groups=groups)
                pie = ax.inset_axes([j - 0.33, 0.8, 0.66, 0.13],
                                    transform=band)
                pie.pie([share, 100 - share], colors=[INK2, GRID],
                        startangle=90, counterclock=False)
                pie.set_aspect("equal")
                ax.text(j, 0.95, f"{share:.0f}%", transform=band,
                        ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [SCEN_SHORT[s] for s in SCENARIOS], rotation=35,
                      ha="right", fontsize=8)
        ax.set_title(INCGRPS[k] if k < 4 else "All households")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
        style_axes(ax)
    axes.flat[0].set_ylim(0, top / (0.76 if overlay else 0.95))
    handles = legend_handles(HTYPE_COLORS, HTYPES)
    if kind == "damages":
        handles.append(Line2D([], [], marker="D", color=INK, linestyle="none",
                              label="Net of insurance payouts"))
    if overlay:
        handles.append(Patch(facecolor=INK2,
                             label="Pies: share of households exposed"))
    axes.flat[5].axis("off")
    axes.flat[5].legend(handles=handles, loc="center", fontsize=9,
                        title=title, title_fontproperties={"weight": "bold"})
    fig.tight_layout()
    save(fig, "bars", filename)


scenario_bars("households", "fig28_households_by_type_scenarios.png",
              "Households (thousands)", "Households by housing type")
scenario_bars("damages", "fig29_damages_by_type_scenarios.png",
              "Damages (M ZAR/year)", "Expected damages by housing type")
scenario_bars("damages_relative", "fig29a_damages_relative_scenarios.png",
              "Damages of exposed (% of income)",
              "Damages of exposed households\nby housing type (overlaid)")

###############################################################################

# %% POLICY INTERACTIONS (FULL FACTORIAL, IF RUN)

print("POLICY INTERACTIONS")

## FOOTNOTE: For each outcome Y, with Y(S) its value when the set S of policy
## levers is on (No RM: no risk misperception; Subs. SP: free sandbags; Public
## P.: public protection; Subs. ins.: subsidised insurance, coverage 0.79 in
## every combination) and Y(0) the benchmark: single effects Y(j) - Y(0);
## pairwise interactions Y(jk) - Y(j) - Y(k) + Y(0); Shapley values, each
## lever's mean marginal effect over all orders of introduction (they add up
## to the combined effect Y(all) - Y(0)). An interaction with the sign of the
## two single effects' sum makes the levers complements (they reinforce each
## other), the opposite sign substitutes (they overlap). Outcomes as in the
## scenario summary and welfare tables; aggregate expenses: public cost of
## PS and SI + developers' losses + uncovered RDP damages; aggregate EV:
## population-weighted EV + change in landlords' net revenue per household.
## The exposed-household Shapley table uses the welfare of exposed households
## (Table 27b) and the % of each group exposed.

LEVERS = ["RM0", "PS1", "PP1", "SI1"]
LEVER_SHORT = dict(zip(LEVERS, ["No RM", "Subs. SP", "Public P.", "Subs. ins."]))


def scenario_of(subset):
    """Scenario label for a set of levers."""
    subset = [lever for lever in LEVERS if lever in subset]
    if not subset:
        return "benchmark"
    if len(subset) == 1:
        return SINGLES[LEVERS.index(subset[0])]
    if len(subset) == 4:
        return "combined"
    return "+".join(subset)


if FACTORIAL:
    Y, Y_EXP = {}, {}
    for r in range(5):
        for subset in combinations(LEVERS, r):
            label = scenario_of(subset)
            if label in METRICS:
                Y[frozenset(subset)], Y_EXP[frozenset(subset)] = METRICS[label]
            else:
                Y[frozenset(subset)], Y_EXP[frozenset(subset)] = metrics(
                    load_scenario(label), label)
    empty, n = frozenset(), len(LEVERS)

    def effect(Ys, subset, y):
        return Ys[frozenset(subset)][y] - Ys[empty][y]

    def shapley(Ys, names):
        rows = {}
        for y in names:
            row = {}
            for j in LEVERS:
                others = [k for k in LEVERS if k != j]
                row[LEVER_SHORT[j]] = sum(
                    factorial(r) * factorial(n - r - 1) / factorial(n)
                    * (Ys[frozenset(S) | {j}][y] - Ys[frozenset(S)][y])
                    for r in range(n) for S in combinations(others, r))
            rows[y] = row
        return pd.DataFrame(rows).T

    def fmt_inter(block, row, c, v):
        return _num(v, 0 if "(ZAR" in row else 2 if "(%" in row else 1,
                    sign=True)

    names = list(Y[empty])
    rows = {}
    for y in names:
        row = {LEVER_SHORT[j]: effect(Y, [j], y) for j in LEVERS}
        for j, k in combinations(LEVERS, 2):
            row[f"{LEVER_SHORT[j]} x {LEVER_SHORT[k]}"] = (
                effect(Y, [j, k], y) - effect(Y, [j], y) - effect(Y, [k], y))
        row["Combined"] = effect(Y, LEVERS, y)
        row["Sum of singles"] = sum(effect(Y, [j], y) for j in LEVERS)
        rows[y] = row
    table_inter = pd.DataFrame(rows).T
    shapley_names = [y for y in names if y != "Aggregate expenses (M ZAR)"]

    SECTIONS = ["Damages, exposure and costs", "Utility changes (%)",
                "EV changes (ZAR/household/year)", "Aggregate changes"]

    def sectioned(df):
        """Rows of a Shapley table in SECTIONS, with the section's prefix
        and unit dropped from the row names."""
        blocks = {}
        for y, row in df.iterrows():
            if "population-weighted" in y or y.startswith(
                    ("Landlords", "EV + ")):
                block, name = SECTIONS[3], y
            elif y.startswith("Utility, "):
                block, name = SECTIONS[1], y[9:].removesuffix(" (%)")
            elif y.startswith("EV, "):
                block, name = SECTIONS[2], y[4:].removesuffix(" (ZAR)")
            else:
                block, name = SECTIONS[0], y
            blocks.setdefault(block, {})[name] = row
        return pd.concat({b: pd.DataFrame(blocks[b]).T for b in SECTIONS
                          if b in blocks})

    def fmt_shapley(block, row, c, v):
        if block == SECTIONS[1] or row.startswith("Utility"):
            return _num(v, 3, sign=True)
        if block == SECTIONS[2] or "(ZAR" in row:
            return _num(v, 0, sign=True)
        return _num(v, 2 if "(%" in row else 1, sign=True)

    tables = {"table_pairwise_interactions": pd.concat({"": table_inter}),
              "table_shapley": sectioned(shapley(Y, shapley_names)),
              "table_shapley_exposed": sectioned(
                  shapley(Y_EXP, list(Y_EXP[empty])))}
    for stem, df in tables.items():
        df.to_excel(FOLDERS["interactions"] / f"{stem}.xlsx",
                    float_format="%.3f")
        latex_table(df, FOLDERS["interactions"] / f"{stem}.tex",
                    fmt_inter if stem == "table_pairwise_interactions"
                    else fmt_shapley)
        print(df.round(2).to_string())

    ## FIGURE: interaction matrices (diagonal: single effects; off-diagonal:
    ## pairwise interactions). Colors are oriented so that blue improves the
    ## outcome (fewer damages, exposed households or expenses; more welfare
    ## or protection): on the diagonal, a beneficial single effect; off the
    ## diagonal, an interaction that adds to the benefits (complements when
    ## both levers help), red one that takes away from them (substitutes)
    DIVERGING = LinearSegmentedColormap.from_list(
        "red_blue", ["#b02a2a", "#f09a72", "#f0efec", "#86b6ef", "#184f95"])
    KEY = {"Damages (M ZAR)": -1, "Exposed households (%)": -1,
           "Exposed settlers protected (%)": 1,
           "EV + landlords' net revenue (ZAR/household)": 1,
           "Damages, exposed (% of income)": -1,
           "Aggregate expenses (M ZAR)": -1}
    TITLES = {"EV + landlords' net revenue (ZAR/household)":
              "Aggregate EV: population-weighted EV\n+ landlords' net revenue "
              "(ZAR/household)"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (y, better) in zip(axes.flat, KEY.items()):
        M = np.full((n, n), np.nan)
        for a, j in enumerate(LEVERS):
            M[a, a] = effect(Y, [j], y)
            for b, k in enumerate(LEVERS):
                if b > a:
                    M[a, b] = M[b, a] = (effect(Y, [j, k], y)
                                         - effect(Y, [j], y)
                                         - effect(Y, [k], y))
        lim = np.nanmax(np.abs(M)) or 1
        ax.imshow(better * M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
        digits = 2 if lim < 1 else 1 if lim < 10 else 0
        for a in range(n):
            for b in range(n):
                v = M[a, b]
                ax.text(b, a, _num(v, digits, sign=True), ha="center",
                        va="center", fontsize=9,
                        color="white" if abs(v) > 0.6 * lim else INK,
                        fontweight="bold" if a == b else "normal")
        ticks = [LEVER_SHORT[j] for j in LEVERS]
        ax.set_xticks(range(n), ticks, rotation=30, ha="right")
        ax.set_yticks(range(n), ticks)
        ax.set_title(f"{TITLES.get(y, y)}\ncombined "
                     f"{_num(effect(Y, LEVERS, y), digits, True)}, sum of "
                     f"singles "
                     f"{_num(sum(effect(Y, [j], y) for j in LEVERS), digits, True)}",
                     fontsize=9)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("Single effects (diagonal, bold) and pairwise interactions "
                 "(off-diagonal) relative to the benchmark\nBlue: improves "
                 "the outcome (off-diagonal: levers reinforce each other); "
                 "red: worsens it (off-diagonal: levers overlap)",
                 fontweight="bold", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "interactions", "fig_interaction_matrices.png")

###############################################################################

# %% DIAGNOSTICS FOR THE COMMENTS

print("DIAGNOSTICS")

## No risk misperception: housing costs and damages of formal households, and
## who bears developers' losses (benchmark -> no misperception)
R = sim["no_misperception"]
for i in (1, 2, 3):
    bills = [np.sum(d["hh"][0, i] * d["rent_clean"][0] * d["consumption"][0])
             / 1e6 for d in (B, R)]
    damages = [d["dmg"][:, i].sum() / 1e6 for d in (B, R)]
    sizes = [np.sum(d["hh"][0, i] * d["consumption"][0]) / d["hh"][0, i].sum()
             for d in (B, R)]
    print(f"  {INCGRPS[i]:>15s}: rent bill {bills[0]:,.0f} -> {bills[1]:,.0f} "
          f"M ZAR ({bills[1] - bills[0]:+,.0f}), damages {damages[0]:.0f} -> "
          f"{damages[1]:.0f} M ({damages[1] - damages[0]:+.0f}), dwelling "
          f"{sizes[0]:.1f} -> {sizes[1]:.1f} m2")
print("  formal floor space (M m2):",
      [round(np.sum(d["hh"][0] * d["consumption"][0]) / 1e6, 2) for d in (B, R)],
      "developers' losses (M ZAR):",
      [round(d["developer_tax"][1] / 1e6, 1) for d in (B, R)],
      "landlords [gross, net, slum] (M ZAR):",
      [np.round(np.array(landlords(d)) / 1e6, 1).tolist() for d in (B, R)])
print("  equilibrium utilities:", B["utility"].round(1), "->",
      R["utility"].round(1))
for s in SCENARIOS:
    print(f"  {SCEN_LABELS[s]:>28s}: max population error "
          f"{100 * np.abs(sim[s]['error']).max():.2f}%, insurance share "
          f"{SCEN['scenarios'][s]['insurance_share']}")

# %%
