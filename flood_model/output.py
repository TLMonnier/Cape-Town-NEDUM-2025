# ##########################################################
# ## FLOOD MODEL: OUTPUT SCRIPT FOR THE POLICY SCENARIOS ###
# ##########################################################
# Tables and figures for the policy scenarios solved by main.py (benchmark,
# no risk misperception, subsidised self-protection, public protection,
# subsidised insurance, all combined; and, with main.py's FACTORIAL, the
# other combinations of these four levers), read from the outputs main.py
# saves to Output/flood_model/policy_scenarios/simulations. Run as a script
# or cell by cell; results go to Output/flood_model/policy_scenarios, with
# the number of each figure and table in the PDF (pdf/figures_tables.tex) as
# file prefix:
#     maps/inputs (figures 1-2), maps/benchmark (3-7), maps/scenarios
#     (13-21), distributions/benchmark (8-12), distributions/scenarios
#     (22-26), bars (27-29), interactions (figure 30, tables 11-13), tables
#     (tables 1-10).
# Figures have no titles (they are added in the PDF built from the outputs);
# subplots keep a short label (income group, housing type, scenario).
# Damages: absolute damages cover structures and contents (formal
# structures attributed to their occupants); damages as a share of income
# only cover the damages households pay for (net of insurance payouts):
# contents, settlers' shacks, RDP houses, and backyard shacks, paid by the
# RDP owners of the cell; formal private structures are borne by developers.
# Expected flood depth is before any protection investment (sandbags,
# public protection).

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
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                               LogNorm, Normalize, TwoSlopeNorm)
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
    ("maps_inputs", "maps/inputs"), ("maps_bench", "maps/benchmark"),
    ("maps_scen", "maps/scenarios"),
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
              "subsid_insur": "Subs. ins.", "combined": "Combined"}
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
# Damages per household as a share of mean income: color scales and axes
# stop at this level (%), without marking the observations above it
DAMAGE_CAP = 25
# Sequential heat scale for bar colors, without its palest end, which
# vanishes on the basemap
HEAT = LinearSegmentedColormap.from_list(
    "heat", plt.get_cmap("YlOrRd")(np.linspace(0.15, 1, 256)))
# Diverging scale centred on 1 for the amenity index
DIVERGING = LinearSegmentedColormap.from_list(
    "red_blue", ["#b02a2a", "#f09a72", "#f0efec", "#86b6ef", "#184f95"])
# Income groups in green (wage circles over the amenity map), light to dark
INCGRP_GREENS = ["#a6dba0", "#5aae61", "#1b7837", "#00441b"]
# Sequential green scale for commuting costs, without its palest end
GREENS = LinearSegmentedColormap.from_list(
    "greens", plt.get_cmap("Greens")(np.linspace(0.12, 1, 256)))

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.labelpad": 6,
    "xtick.major.pad": 4, "ytick.major.pad": 4,
    "legend.frameon": False, "legend.borderaxespad": 0.3,
    "legend.title_fontsize": 8,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.06})
BOX = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID,
           alpha=0.95)
WHITE_LEGEND = dict(frameon=True, facecolor="white", edgecolor=GRID,
                    framealpha=1)
DIAMOND = dict(marker="D", markersize=3.6, linewidth=1.1)

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
CBD_LONLAT = (18.4241, -33.9249)

# Basemap raster on the 500 m lattice of the grid (projected metres): the
# municipality in light gray, cells with more than 5% urban land darker,
# transparent outside; corners converted to longitude / latitude
col = np.round((geo_grid["X_MIN"] - geo_grid["X_MIN"].min()) / 500).astype(int)
row = np.round((geo_grid["Y_MIN"] - geo_grid["Y_MIN"].min()) / 500).astype(int)
CELL_ROW, CELL_COL = row, col          # (row and col are reused below)


def raster_of(rgba_cells):
    """Basemap raster from one RGBA color per grid cell."""
    out = np.zeros((row.max() + 2, col.max() + 2, 4))
    out[row, col] = rgba_cells
    return out


urban_cell = grid_csv["urban"].to_numpy() > 0.05 * 250000
BASE_RGBA = raster_of(np.where(urban_cell[:, None],
                               matplotlib.colors.to_rgba(URBAN),
                               matplotlib.colors.to_rgba(LAND)))
corner_x, corner_y = np.meshgrid(
    geo_grid["X_MIN"].min() + 500 * np.arange(col.max() + 2),
    geo_grid["Y_MIN"].min() + 500 * np.arange(row.max() + 2))
corners = gpd.GeoSeries(gpd.points_from_xy(corner_x.ravel(), corner_y.ravel()),
                        crs=geo_grid.crs).to_crs("EPSG:4326")
BASE_LON = corners.x.to_numpy().reshape(corner_x.shape)
BASE_LAT = corners.y.to_numpy().reshape(corner_x.shape)

## LOAD SIMULATION RESULTS


def load_simulation_data(var, name):
    prefix = "" if var == "mask_self_protec" else "initial_state_"
    path = PATH_SIMUL / f"{prefix}{var}_{name}.npy"
    if var == "developer_tax" and not path.exists():
        return np.zeros(3)        # [t, losses, revenue]: no misperception
    if var == "rdp_vacant" and not path.exists():
        return np.zeros(N_CELLS)  # runs saved before the RDP choice option
    return np.load(path)


VARIABLES = ["households", "dwelling_size", "rent", "housing_supply",
             "utility", "mask_self_protec", "damages_contents",
             "damages_structure", "damages_reimbursed",
             "damages_informal_levels", "ex_post_u_post", "ex_post_z_post",
             "ex_post_price", "net_income", "landlord_revenue",
             "surplus_damages", "error", "solver_cells", "developer_tax",
             "rdp_vacant"]


def load_scenario(label):
    info = SCEN["scenarios"][label]
    return derive({var: load_simulation_data(var, info["name"])
                   for var in VARIABLES}, info)

###############################################################################

# %% DERIVED VARIABLES

print("DERIVED VARIABLES")


def perceived_factor(opts):
    """Share k of actual damages agents perceive (solver.perception)."""
    if not opts["agents_anticipate_floods"]:
        return 0.
    return p["risk_internaliz"] if opts["risk_misperc"] else 1.


def derive(d, info):
    """Add per-scenario variables used by the plots and tables."""
    opts = info["options"]
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
    # Actual expected damages (rands/year): gross (structures and contents),
    # and net of insurance payouts
    d["dmg"] = d["damages_contents"] + d["damages_structure"]
    d["dmg_net"] = d["dmg"] - d["damages_reimbursed"]
    # Damages households pay for (net of payouts): no formal private
    # structures (developers); backyard shacks (net of payouts, same share s
    # for the insured groups 1-2) move from their tenants to the RDP owners
    # of the cell, who own them
    s = (info.get("insurance_share") or 0) if opts["subsid_insur"] else 0
    borne = d["dmg_net"].copy()
    borne[0] -= d["damages_structure"][0]
    shacks = (1 - s) * d["damages_structure"][1]
    borne[1] -= shacks
    borne[3, 0] += shacks.sum(0)
    d["dmg_borne"] = borne
    # Developers' unanticipated losses, the part of formal structure damages
    # they do not perceive: (1 - k) x actual damages (groups, cells)
    d["dev_loss"] = (1 - perceived_factor(opts)) * d["damages_structure"][0]
    # Total mean income of the occupants of each housing type and cell
    d["occ_income"] = (hh * y_bar[None, :, None]).sum(1)
    d["level"] = np.nan_to_num(d["mask_self_protec"]).astype(int)
    # RDP choice (RC1): cells whose RDP units were declined
    d["vacant"] = d["rdp_vacant"] > 0
    return d


sim = {label: load_scenario(label) for label in SCENARIOS}
B = sim["benchmark"]
N_TOTAL = B["hh"].sum()
print("  developers' losses (M ZAR), from structure damages vs solver:",
      [(round(sim[s]["dev_loss"].sum() / 1e6, 1),
        round(sim[s]["developer_tax"][1] / 1e6, 1)) for s in SCENARIOS])

## EXPECTED FLOOD DEPTH FROM INPUTS
## Expected annual depth averaged over the cell, E[p_T d_T] (cm), with the
## deepest flood type in each event (same states and probabilities as
## floods.py), before any protection investment (sandbags, public
## protection). Per housing type, frequent pluvial events are drained
## (floods.PLUVIAL_SAFE), so that a positive depth means positive expected
## damages before protection.


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
# the benchmark, a zero threshold counts most households as exposed, most of
# them at negligible depths; 0.1 cm/year keeps about a third of households
# and almost all damages. Most exposed: above MOST_EXPOSED_THRESHOLD
EXPOSURE_THRESHOLD = 0.1
MOST_EXPOSED_THRESHOLD = 1.0
EXPOSED = DEPTH > EXPOSURE_THRESHOLD
MOST_EXPOSED = DEPTH > MOST_EXPOSED_THRESHOLD
EXP3 = EXPOSED[:, None, :]                    # (types, groups, cells)
SUBSAMPLES = {"all": None, "exposed": EXPOSED, "most": MOST_EXPOSED}
SUBSAMPLE_NAMES = {
    "all": "All households",
    "exposed": f"Exposed households (above {EXPOSURE_THRESHOLD:g} cm/year)",
    "most": ("Most exposed households (above "
             f"{MOST_EXPOSED_THRESHOLD:g} cm/year)")}


def exposed_share(d, types=(0, 1, 2, 3), groups=(0, 1, 2, 3), mask=EXPOSED):
    """% of the households of these housing types and groups in mask."""
    hh = d["hh"][np.ix_(list(types), list(groups))]
    n = hh.sum()
    return (100 * (hh * mask[list(types)][:, None, :]).sum() / n
            if n > 0 else np.nan)


def mean_depth_exposed(d, types=(0, 1, 2, 3), groups=(0, 1, 2, 3)):
    """Household-weighted mean expected depth (cm/year) of exposed
    households."""
    w = d["hh"][np.ix_(list(types), list(groups))] \
        * EXPOSED[list(types)][:, None, :]
    n = w.sum()
    return ((w * DEPTH[list(types)][:, None, :]).sum() / n if n > 0
            else np.nan)


def settler_shares(d):
    """% of exposed settlers protected by sandbags, % of settlers exposed."""
    settlers = d["hh_type"][2]
    exposed = (settlers * EXPOSED[2]).sum()
    protected = (settlers * EXPOSED[2] * (d["level"] >= 1)).sum()
    return 100 * protected / exposed, 100 * exposed / settlers.sum()

###############################################################################

# %% PLOTTING HELPERS (3D MAPS, LINES AND BARS)

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


def to_log(v, vmin):
    """Bar heights on a log scale: decades above vmin (0 at or below it)."""
    v = np.nan_to_num(np.asarray(v, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > vmin, np.log10(np.maximum(v, vmin) / vmin), 0)


def decade_below(v):
    return 10.0 ** np.floor(np.log10(v))


def _range(arrays, low_pct=1, decades=3):
    """Log range: from the decade below the low_pct percentile of positive
    values, but at most `decades` decades below the maximum, to the max."""
    pos = np.concatenate([a[a > 0] for a in arrays])
    return (max(decade_below(np.percentile(pos, low_pct)),
                decade_below(pos.max()) / 10 ** decades), pos.max())


def bar_top(values, overlay=False, log=None):
    v = np.clip(np.nan_to_num(np.atleast_2d(values)), 0, None)
    if log is not None:
        v = to_log(v, log)
    cat, bottom, height = segments(v, overlay)
    return (bottom + height).max(initial=0)


def add_basemap(ax, base=None, zorder=2):
    """Municipality and urban footprint (or another raster) at z = 0."""
    surf = ax.plot_surface(BASE_LON, BASE_LAT, np.zeros_like(BASE_LON),
                           facecolors=BASE_RGBA if base is None else base,
                           rstride=1, cstride=1, shade=False, linewidth=0,
                           antialiased=False)
    surf.set_zorder(zorder)


def add_cbd(ax):
    """CBD drawn after the bars, so that it stays visible."""
    ax.scatter(*CBD_LONLAT, [0], s=14, color=INK, edgecolor="white",
               linewidth=0.6, zorder=50, depthshade=False)
    ax.text(CBD_LONLAT[0] + 0.012, CBD_LONLAT[1] + 0.012, 0, "CBD",
            fontsize=7, color=INK, zorder=51,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85))


def map3d(ax, values, colors=None, zlabel="", zlim=None, labels=None,
          legend=True, legend_title=None, level=False, overlay=False,
          heat=None, norm=None, cmap=None, log=None, lon=None, lat=None,
          base=None, footprint=None):
    """3D bars over the basemap, for values (categories, items) at the
    cells (or at lon, lat), stacked or overlaid (segments). level: thinner,
    more transparent bars seen from higher up, for level variables whose
    bars form walls. log: bar heights on a log scale, in decades above log
    (the lowest value shown). heat: (items,) values coloring single bars
    with cmap (default HEAT) and norm (values beyond its ends take its end
    colors). base: basemap raster. All segments go into one bar3d call, so
    that matplotlib depth-sorts their faces; the CBD and the legend (white
    box) are drawn on top."""
    lon = LON if lon is None else np.asarray(lon)
    lat = LAT if lat is None else np.asarray(lat)
    v = np.clip(np.nan_to_num(np.atleast_2d(
        np.asarray(values, dtype=float))), 0, None)
    if log is not None:
        v = to_log(v, log)
    keep = np.any(v > 0, 0)
    lon, lat, v = lon[keep], lat[keep], v[:, keep]
    alpha, size = (0.75, 0.55) if level else (0.9, 0.9)
    if footprint:
        size = footprint
    if heat is not None:
        alpha = 0.9
    add_basemap(ax, base, zorder=2)
    cat, bottom, height = segments(v, overlay)
    m = height > 0
    if m.any():
        cells = np.nonzero(m)[1]
        if heat is None:
            rgba = np.array([matplotlib.colors.to_rgba(c, alpha)
                             for c in colors])[cat[m]]
        else:
            h = np.nan_to_num(np.asarray(heat, dtype=float)[keep][cells])
            rgba = (cmap or HEAT)(norm(np.clip(h, norm.vmin, norm.vmax)),
                                  alpha=alpha)
        dx, dy = size * DLON, size * DLAT
        ax.bar3d(lon[cells] - dx / 2, lat[cells] - dy / 2, bottom[m], dx, dy,
                 height[m], color=rgba, shade=True, edgecolor="none",
                 zorder=3)
    add_cbd(ax)
    if zlim is None:
        zlim = (0, max((bottom + height).max(initial=0), 1e-9))
    ax.set_xlim(BOUNDS[0], BOUNDS[2])
    ax.set_ylim(BOUNDS[1], BOUNDS[3])
    ax.set_zlim(*zlim)
    km_x = (BOUNDS[2] - BOUNDS[0]) * 111.32 * np.cos(np.deg2rad(LAT.mean()))
    km_y = (BOUNDS[3] - BOUNDS[1]) * 110.95
    ax.set_box_aspect((km_x, km_y, 0.55 * km_y), zoom=0.88)
    ax.view_init(elev=52 if level else ELEV, azim=AZIM)
    ax.set_xlabel("Longitude", labelpad=6, fontsize=8)
    ax.set_ylabel("Latitude", labelpad=6, fontsize=8)
    ax.set_zlabel(zlabel, labelpad=10, fontsize=8)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.tick_params(axis="z", labelsize=7, pad=4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    if log is not None:                 # 1-2-5 ticks above log
        steps = [np.log10(m * 10.0 ** e) for e in range(9) for m in (1, 2, 5)]
        ax.zaxis.set_major_locator(ticker.FixedLocator(
            [z for z in steps if z <= zlim[1] * 1.0001]))
        ax.zaxis.set_major_formatter(ticker.FuncFormatter(
            lambda z, _: _fmt(log * 10 ** z)))
    else:
        ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.zaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
    if legend and labels:
        leg = ax.legend(handles=legend_handles(colors, labels),
                        loc="upper left", fontsize=7, handlelength=1,
                        title=legend_title, **WHITE_LEGEND)
        leg.set_zorder(100)
    return zlim


def colorbar_below(fig, axes, norm, label, cmap=None, width=0.45,
                   height=0.012, gap=0.035):
    """Horizontal colorbar centred under axes (call after the layout)."""
    boxes = [ax.get_position() for ax in np.atleast_1d(axes)]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    cax = fig.add_axes((x0 + (1 - width) / 2 * (x1 - x0), y0 - gap,
                        width * (x1 - x0), height))
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap or HEAT),
                       cax=cax, orientation="horizontal")
    bar.set_label(label, fontsize=8, color=INK2, labelpad=4)
    bar.ax.tick_params(labelsize=7)
    if isinstance(norm, LogNorm):
        ticks = [m * 10.0 ** e for e in range(-4, 7) for m in (1, 3)]
        bar.set_ticks([v for v in ticks if norm.vmin <= v <= norm.vmax])
        bar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
        bar.ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        bar.ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        bar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    bar.outline.set_edgecolor(GRID)
    return bar


## 2D MAPS: cells drawn on their 500 m lattice (corners in longitude /
## latitude)


def cell_raster(v):
    """Masked (rows, cols) raster of per-cell values, for pcolormesh."""
    out = np.full((CELL_ROW.max() + 1, CELL_COL.max() + 1), np.nan)
    out[CELL_ROW, CELL_COL] = v
    return np.ma.masked_invalid(out)


def map2d(ax, heat=None, norm=None, cmap=None):
    """2D map: municipality and urban footprint, then heat (cells,) values
    (NaN: not drawn) with cmap and norm, and the CBD on top."""
    ax.pcolormesh(BASE_LON, BASE_LAT, cell_raster(urban_cell.astype(float)),
                  cmap=ListedColormap([LAND, URBAN]), vmin=0, vmax=1,
                  shading="flat", zorder=1, rasterized=True)
    mesh = None
    if heat is not None:
        mesh = ax.pcolormesh(BASE_LON, BASE_LAT, cell_raster(heat),
                             cmap=cmap, norm=norm, shading="flat", zorder=2,
                             rasterized=True)
    ax.scatter(*CBD_LONLAT, s=16, color=INK, edgecolor="white",
               linewidth=0.6, zorder=50)
    ax.annotate("CBD", CBD_LONLAT, xytext=(5, 5), textcoords="offset points",
                fontsize=7, color=INK, zorder=51,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))
    ax.set_xlim(BOUNDS[0], BOUNDS[2])
    ax.set_ylim(BOUNDS[1], BOUNDS[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(LAT.mean())))
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    for side in ax.spines.values():
        side.set_edgecolor(GRID)
    return mesh


def colorbar_right(fig, ax, mappable, label, extend="neither", fmt=None):
    """Vertical colorbar next to ax, with plain number ticks."""
    bar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02,
                       extend=extend)
    bar.set_label(label, fontsize=8, color=INK2, labelpad=4)
    bar.ax.tick_params(labelsize=7)
    bar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt or _fmt))
    bar.outline.set_edgecolor(GRID)
    return bar


def legend_handles(colors, labels):
    return [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]


def bars2d(ax, x, widths, values, colors, overlay=False):
    """2D bars at x for values (categories, items), stacked or overlaid.
    Returns the highest bar."""
    v = np.clip(np.nan_to_num(np.atleast_2d(values)), 0, None)
    cat, bottom, height = segments(v, overlay)
    for r in range(v.shape[0]):
        for c in range(v.shape[0]):
            m = (cat[r] == c) & (height[r] > 0)
            if m.any():
                ax.bar(x[m], height[r][m], width=widths[m], bottom=bottom[r][m],
                       color=colors[c], edgecolor="white", linewidth=0.3)
    return (bottom + height).max(initial=0)


def _share(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0)


def positive(v):
    """Values for a log axis: NaN where not positive (missing)."""
    v = np.asarray(v, dtype=float)
    return np.where(v > 0, v, np.nan)


def connected(ax, x, v, **kw):
    """Connected diamonds over the finite values of v: the line joins the
    observations on either side of missing ones."""
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    return ax.plot(np.asarray(x)[ok], v[ok], **{**DIAMOND, **kw})


def _fmt(x, pos=None):
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.2g}"


def log_axis(ax, axis="y"):
    """Log scale with plain number labels."""
    getattr(ax, f"set_{axis}scale")("log")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(ticker.FuncFormatter(_fmt))
    target.set_minor_formatter(ticker.NullFormatter())


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


def grid3d(n, ncol=None):
    """Figure with n 3D panels: one, 1 x 3, 2 x 2 or 2 x 3, tightly packed."""
    shape = {1: (1, 1), 3: (1, 3), 4: (2, 2), 6: (2, 3)}[n]
    size = {1: (8.6, 6.6), 3: (19, 6.2), 4: (13, 11.6), 6: (18, 10.4)}[n]
    fig = plt.figure(figsize=size)
    axes = [new3d(fig, *shape, k + 1) for k in range(n)]
    # Room on the right of each panel for its z label
    fig.subplots_adjust(left=0, right=0.9, top=1,
                        bottom=0.07 if n > 1 else 0,
                        wspace=0.08 if n > 1 else 0, hspace=0.08)
    return fig, axes

###############################################################################

# %% TABLES 1-2: INCOME GROUPS AND MODEL ASSUMPTIONS (DATA)

print("TABLES 1-2: DATA")


def write_table(df, stem, fmt, folder="tables"):
    df.to_excel(FOLDERS[folder] / f"{stem}.xlsx", float_format="%.3f")
    latex_table(df, FOLDERS[folder] / f"{stem}.tex", fmt)


def latex_table(df, path, fmt):
    """Write a booktabs tabular: df has a (block, row) MultiIndex, and
    fmt(block, row, column, value) gives the cell text."""
    cols = list(df.columns)
    esc = lambda s: (str(s).replace("%", r"\%").replace("&", r"\&")
                     .replace("²", r"\textsuperscript{2}")
                     .replace("ȳ", r"$\bar{y}$").replace(">", r"$>$"))
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join([""] + [esc(c) for c in cols]) + r" \\", r"\midrule"]
    for k, block in enumerate(df.index.get_level_values(0).unique()):
        if k:
            lines.append(r"\addlinespace")
        if block:
            lines.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}"
                         rf"{{\textit{{{esc(block)}}}}} \\")
        for row, values in df.loc[block].iterrows():
            cells = [esc(fmt(block, row, c, v)) for c, v in zip(cols, values)]
            lines.append((r"\quad " if block else "") + esc(row) + " & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _num(v, digits=1, sign=False):
    if v is None or isinstance(v, str):
        return v or "--"
    if not np.isfinite(v):
        return "--"
    if round(v, digits) == 0:            # no "-0.000"
        v = 0.
    return f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"


## TABLE 1: INCOME GROUPS (DATA)

## FOOTNOTE: 2011 Census (Income_distribution_2011.csv): annual household
## income brackets grouped into the four model income groups (households
## without income excluded); mean income: household-weighted mean of the
## bracket medians. Employment rate: exogenous annual employment rate of
## each group (half the number of employed workers per two-worker
## household, calibrated in Pfeiffer et al. from educational attainment).
## Population share: share of households (with income) in each group.

brackets = pd.read_csv(config.DATA / "data_Cape_Town"
                       / "Income_distribution_2011.csv")
groups_of = np.array(p["income_distribution"])
hh_groups = np.array([brackets.Households_nb[groups_of == j].sum()
                      for j in range(1, 5)])
EMPLOYMENT = 100 * np.array([1.14, 1.94, 1.92, 1.94]) / 2
rows = {}
for j in range(4):
    sub = brackets[groups_of == j + 1]
    low, high = sub.INC_min.min(), sub.INC_max.max()
    rows[("", INCGRPS[j])] = {
        "Annual income range (ZAR, 2011)":
            f"{low:,.0f} - {high:,.0f}" if j < 3 else f"{low:,.0f} and above",
        "Mean income (ZAR, 2011)": y_bar[j],
        "Employment rate (%)": EMPLOYMENT[j],
        "Population share (%)": 100 * hh_groups[j] / hh_groups.sum()}
table0a = pd.DataFrame(rows).T
table0a.index.names = ["", "Income group"]
write_table(table0a, "table01_income_groups",
            lambda b, r, c, v: v if isinstance(v, str)
            else _num(v, 0 if "ZAR" in c else 1))

## TABLE 2: MODEL ASSUMPTIONS BY HOUSING TYPE

## FOOTNOTE: Eligible groups: income groups with access to each housing
## type (1: poorest). Location: endogenous within the areas available for
## each type (backyards of RDP premises; informal settlement areas tolerated
## in 2011); RDP units are exogenous, but low-income households may decline
## them (RDP choice). Dwelling size: formal private dwellings at least the
## minimum lot size; RDP house and backyard; shacks. Price: rents per m2
## (RDP housing is free). Population share: 2011 households by housing type
## (SAL data).

totals = np.array([config.TOTAL_FORMAL, config.TOTAL_RDP,
                   config.TOTAL_BACKYARD, config.TOTAL_INFORMAL])
table0b = pd.DataFrame({
    "Eligible groups": ["1, 2, 3, 4", "1", "1, 2", "1, 2"],
    "Location": ["Endogenous", "Exogenous", "Endogenous within defined areas",
                 "Endogenous within defined areas"],
    "Dwelling size": [
        f"Endogenous (min. {p['mini_lot_size']:.0f} m²)",
        f"Fixed ({p['RDP_size']} m² + {p['backyard_size']} m² backyard)",
        f"Fixed ({p['shack_size']} m²)", f"Fixed ({p['shack_size']} m²)"],
    "Price": ["Endogenous", "Free", "Endogenous", "Endogenous"],
    "Population share (%)": [f"{v:.1f}" for v in 100 * totals / totals.sum()]},
    index=pd.MultiIndex.from_product(
        [[""], ["Formal private (FP)", "Formal subsidised (FS)",
                "Informal backyards (IB)", "Informal settlements (IS)"]]))
write_table(table0b, "table02_model_assumptions", lambda b, r, c, v: v)
print(table0a.to_string(), "\n", table0b.to_string())

###############################################################################

# %% FIGURES 1-2: INPUT MAPS (LAND, JOBS, AMENITIES, WAGES, COMMUTING)

print("FIGURES 1-2: INPUT MAPS")

## FOOTNOTE: (1) Basemap: cells with land available for each housing type
## (indicator, whatever the share of the cell), drawn on top of each other:
## informal settlements in front, then informal backyards, then formal
## private (formal subsidised not shown); light gray: other cells of the
## municipality. Bars: jobs by income group in each of the 185 job centres
## (transport zones with more than 2,500 jobs, 2013 TAZ data rescaled to the
## number of households of each group), stacked. (2a) Amenity index A(x)
## (mean 1, diverging scale centred on 1). Circles: calibrated annual wage of
## each income group in each job centre, concentric, with a diameter
## proportional to log(wage / 10,000 ZAR); darker for richer groups, lighter
## circles on top. (2b) Annual commuting costs (expected income minus
## expected income net of commuting costs, both from the transport model) as
## a % of the group's mean income, in each cell where households live in the
## benchmark, one panel per income group with its own color scale (colors
## stop at the 99th percentile).

# Job centres (calibration.sub.import_employment_data)
taz = pd.read_csv(config.DATA / "data_Cape_Town" / "TAZ_amp_2013_proj_centro2.csv")
jobs12 = np.array([np.zeros(len(taz))] + [taz.Ink1 / 3] * 3 + [taz.Ink2 / 2] * 2
                  + [taz.Ink3 / 3] * 3 + [taz.Ink4 / 3] * 3)
x_km, y_km = taz.X / 1000, taz.Y / 1000
centres = jobs12.sum(0) > 2500
centres &= ~((x_km > -10) | (y_km > -3719) | ((x_km > -20) & (y_km > -3765))
             | taz.TZ2013.isin([1010, 1012, 1394, 1499, 4703]))
jobs = np.array([jobs12[groups_of == j + 1].sum(0) for j in range(4)])[:, centres]
jobs = jobs * hh_groups[:, None] / jobs.sum(1, keepdims=True)
wages = np.load(config.DATA / "precalculated_inputs" / "incomeCentersKeep.npy").T
pts = gpd.GeoSeries(gpd.points_from_xy(taz.X[centres], taz.Y[centres]),
                    crs=geo_grid.crs).to_crs("EPSG:4326")
JOB_LON, JOB_LAT = pts.x.to_numpy(), pts.y.to_numpy()
print(f"  {centres.sum()} job centres, wages for {wages.shape[1]}")

# (1) land availability basemap + jobs
land = inputs["coeff_land"]
rgba = np.where(np.isfinite(grid_csv["X"].to_numpy())[:, None],
                matplotlib.colors.to_rgba(LAND), 0)
for t in (0, 1, 2):                        # FP, then IB, then IS in front
    rgba = np.where((land[t] > 0)[:, None],
                    matplotlib.colors.to_rgba(HTYPE_COLORS[t]), rgba)
fig, (ax,) = grid3d(1)
map3d(ax, jobs, INCGRP_COLORS, "Jobs (households)", labels=INCGRPS,
      legend_title="Jobs by income group", lon=JOB_LON, lat=JOB_LAT,
      base=raster_of(rgba), footprint=1.4)
fig.legend(handles=legend_handles(HTYPE_COLORS[:3], HTYPES[:3]),
           title="Land available for", loc="upper right", fontsize=7,
           bbox_to_anchor=(0.98, 0.93), **WHITE_LEGEND)
save(fig, "maps_inputs", "fig01_map_land_jobs.png")

# (2a) amenity heat map + concentric wage circles
amen = inputs["amenities"]
amen_norm = TwoSlopeNorm(1, np.nanpercentile(amen, 1), np.nanpercentile(amen, 99))
WAGE_FLOOR, CIRCLE_MAX = 1e4, 18             # ZAR; largest diameter (pt)


def circle_size(w):
    """Marker area (pt^2): diameter proportional to log(w / WAGE_FLOOR)."""
    diameter = CIRCLE_MAX * np.log10(np.asarray(w) / WAGE_FLOOR) \
        / np.log10(np.nanmax(wages) / WAGE_FLOOR)
    return np.maximum(diameter, 0) ** 2


fig, ax = plt.subplots(figsize=(7.2, 7.6))
mesh = map2d(ax, amen, amen_norm, DIVERGING)
for j in (3, 2, 1, 0):                       # richest (largest) below
    ax.scatter(JOB_LON, JOB_LAT, s=circle_size(wages[j]),
               facecolor=matplotlib.colors.to_rgba(INCGRP_GREENS[j], 0.4),
               edgecolor=matplotlib.colors.to_rgba(INCGRP_GREENS[j], 0.9),
               linewidth=0.5, zorder=10 + 3 - j)
colorbar_right(fig, ax, mesh, "Amenity index (mean 1)", extend="both",
               fmt=lambda v, _: f"{v:.2f}")
groups_leg = ax.legend(handles=[
    Line2D([], [], marker="o", linestyle="", markersize=8,
           markerfacecolor=matplotlib.colors.to_rgba(c, 0.5),
           markeredgecolor=c, label=l)
    for c, l in zip(INCGRP_GREENS, INCGRPS)], title="Wage by income group",
    loc="upper left", fontsize=7, **WHITE_LEGEND)
groups_leg.set_zorder(100)
ax.add_artist(groups_leg)
sizes_leg = ax.legend(handles=[
    Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(circle_size(w)),
           markerfacecolor="none", markeredgecolor=INK2, label=f"{w:,.0f}")
    for w in (2e4, 5e4, 2e5, 8e5)], title="Annual wage (ZAR)",
    loc="lower right", fontsize=7, labelspacing=1.5, borderpad=0.9, handletextpad=1.2,
    **WHITE_LEGEND)
sizes_leg.set_zorder(100)
save(fig, "maps_inputs", "fig02a_map_amenities_wages.png")

# (2b) commuting costs, one panel and color scale per income group
gross = np.load(config.DATA / "precalculated_transport"
                / "GRID_averageIncome_0.npy")
COMMUTING = 100 * (gross - inputs["net_income"]) / y_bar[:, None]
lived = B["hh"].sum((0, 1)) > 1
fig, axes = plt.subplots(2, 2, figsize=(11.5, 12.4))
for j, ax in enumerate(axes.flat):
    v = np.where(lived, COMMUTING[j], np.nan)
    norm = Normalize(np.nanmin(v), np.nanpercentile(v, 99))
    mesh = map2d(ax, v, norm, GREENS)
    ax.set_title(INCGRPS[j], pad=3)
    colorbar_right(fig, ax, mesh, "Commuting costs (% of mean income)",
                   extend="max")
fig.tight_layout(h_pad=1.5, w_pad=1.5)
save(fig, "maps_inputs", "fig02b_map_commuting_costs.png")

###############################################################################

# %% MAP VARIABLES (bar heights by category, per scenario)

print("MAP VARIABLES")


def map_values(d, kind):
    """Bar heights (categories, cells) of count map `kind` for scenario d."""
    if kind == "households_type":                     # (3a)
        return d["hh_type"]
    if kind == "households_group":                    # (3b)
        return d["hh_group"]
    if kind == "housing":                             # hectares (commented out)
        return d["hh_type"] * d["consumption"] / 1e4
    if kind == "damages_type":                        # (5b), M rands
        return d["dmg"].sum(1) / 1e6
    if kind == "protection":                          # (7), households
        return np.array([d["hh_type"][2] * (d["level"] == k)
                         for k in range(4)])
    raise ValueError(kind)


MAPS = {
    # kind: (z label, colors, labels, legend title)
    "households_type": ("Number of households", HTYPE_COLORS, HTYPES,
                        "Housing type"),
    "households_group": ("Number of households", INCGRP_COLORS, INCGRPS,
                         "Income group"),
    "housing": ("Housing (ha of floor space)", HTYPE_COLORS, HTYPES,
                "Housing type"),
    "damages_type": ("Expected damages (M ZAR/year)", HTYPE_COLORS, HTYPES,
                     "Housing type"),
    "protection": ("Informal settlement households", LEVEL_COLORS,
                   LEVEL_LABELS, "Sandbag levels"),
}


def draw_map(ax, d, kind, zlim=None, legend=True, values=None):
    zlabel, colors, labels, title = MAPS[kind]
    if values is None:
        values = map_values(d, kind)
    return map3d(ax, values, colors, zlabel, zlim=zlim, labels=labels,
                 legend=legend, legend_title=title)


## LEVEL MAPS: one panel per housing type, bar height = a level variable,
## bar color = another level variable on a heat scale. Housing costs: linear
## scales specific to each housing type (the same in all scenarios);
## damages: log scales, the same in all panels and scenarios

LEVEL_MAPS = {
    # kind: (housing types, z label, color label, log scales)
    "housing_cost": ([0, 1, 2], "Housing expenditure (% of mean income)",
                     "Rent (ZAR/m²/year)", False),
    "damages": ([0, 1, 2, 3], "Expected damages (thousand ZAR/year, log scale)",
                "Damages paid per household (% of mean income, log scale)",
                True),
}


def level_values(d, kind):
    """(bar heights, bar colors), each (housing types, cells)."""
    hh = d["hh_type"]
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "housing_cost":
            # Rent x housing consumed per household / occupants' mean income
            cost = 100 * d["rent_clean"] * d["consumption"] * hh \
                / d["occ_income"]
            ok = (hh > 0) & (d["rent_clean"] > 0)
            return (np.where(ok, np.nan_to_num(cost), 0),
                    np.where(ok, d["rent_clean"], 0))
        if kind == "damages":
            rel = 100 * d["dmg_borne"].sum(1) / d["occ_income"]
            return (np.where(hh > 0, d["dmg"].sum(1) / 1e3, 0),
                    np.where(hh > 0, np.nan_to_num(rel), 0))
    raise ValueError(kind)


# Scales by (kind, housing type): log floor of the bar heights (None:
# linear), z limits and color norm
LEVEL_LOG, LEVEL_NORM, LEVEL_ZLIM = {}, {}, {}
for kind, (types, _, _, log) in LEVEL_MAPS.items():
    vals = [level_values(sim[s], kind) for s in SCENARIOS]
    if log:
        floor = _range([v[0][types] for v in vals])[0]
        zlim = (0, max(bar_top(v[0][t], log=floor) for v in vals
                       for t in types))
        # Colors from 0.01% to the cap (unmarked beyond either end)
        norm = LogNorm(0.01, DAMAGE_CAP, clip=True)
        for t in types:
            LEVEL_LOG[kind, t], LEVEL_ZLIM[kind, t] = floor, zlim
            LEVEL_NORM[kind, t] = norm
        continue
    for t in types:
        heights = np.concatenate([v[0][t] for v in vals])
        heat = np.concatenate([v[1][t][v[1][t] > 0] for v in vals])
        LEVEL_LOG[kind, t] = None
        LEVEL_ZLIM[kind, t] = (0, heights.max())
        LEVEL_NORM[kind, t] = Normalize(heat.min(), heat.max())


def draw_level(ax, d, kind, t):
    heights, heat = level_values(d, kind)
    return map3d(ax, heights[t], None, LEVEL_MAPS[kind][1],
                 zlim=LEVEL_ZLIM[kind, t], legend=False, level=True,
                 heat=heat[t], norm=LEVEL_NORM[kind, t],
                 log=LEVEL_LOG[kind, t])


def level_figure(panels, kind, labels, filename, folder):
    """One 3D panel per (scenario, housing type) item; one colorbar below
    all panels when they share the color scale, else one below each."""
    fig, axes = grid3d(len(panels))
    for ax, (d, t), label in zip(axes, panels, labels):
        draw_level(ax, d, kind, t)
        ax.set_title(label, pad=-4)
    norms = [LEVEL_NORM[kind, t] for d, t in panels]
    if all(norm is norms[0] for norm in norms):
        colorbar_below(fig, axes, norms[0], LEVEL_MAPS[kind][2])
    else:
        for ax, norm in zip(axes, norms):
            colorbar_below(fig, ax, norm, LEVEL_MAPS[kind][2], width=0.7)
    save(fig, folder, filename)

###############################################################################

# %% FIGURES 3-7: BENCHMARK 3D MAPS

print("FIGURES 3-7: BENCHMARK 3D MAPS")

## FOOTNOTE: (3a)-(3b) Number of households per 500 m grid cell in the
## benchmark (risk misperception, sandbags at full cost, RDP choice), stacked
## by housing type (3a) or income group (3b). (4) One panel per rented
## housing type (FP, IB, IS): annual housing expenditure per household (rent
## per m2 times floor space: FP dwelling size, 14 m2 shacks) as a % of the
## occupants' mean income (bar height), colored by the rent per m2; linear
## scales specific to each panel, the same in all scenarios (figures 15-17).
## Basemap: City of Cape Town (light) and urban footprint (dark).

FILES_3 = {"households_type": "fig03a_map_households_by_type.png",
           "households_group": "fig03b_map_households_by_group.png"}
for kind, filename in FILES_3.items():
    fig, (ax,) = grid3d(1)
    draw_map(ax, B, kind)
    save(fig, "maps_bench", filename)

# Housing quantity and the 2x2 summary of the city structure: commented out
# fig, (ax,) = grid3d(1)
# draw_map(ax, B, "housing")
# save(fig, "maps_bench", "map_housing_quantity.png")

level_figure([(B, t) for t in LEVEL_MAPS["housing_cost"][0]], "housing_cost",
             [HTYPES[t] for t in LEVEL_MAPS["housing_cost"][0]],
             "fig04_map_housing_cost_rent.png", "maps_bench")

## FOOTNOTE: (5a) Expected annual flood depth averaged over each cell
## (cm/year, before any protection investment), from the FATHOM fluvial and
## pluvial and DELTARES coastal maps (deepest flood type in each event),
## without climate change, with the increase under climate change (fluvial
## and pluvial probabilities x2, 2050 sea-level-rise coastal maps) stacked on
## top; cells with land available for housing. (5b) Actual expected damages
## to structures and contents (M ZAR/year), stacked by housing type (formal
## structures attributed to their occupants). (6) One panel per housing
## type: actual expected damages to structures and contents in the cell
## (thousand ZAR/year, bar height, log scale, same z scale in all panels and
## scenarios), colored by the damages households pay for per household (net
## of insurance payouts; FP: contents only; IB: contents only, shacks paid by
## the RDP owners of the cell; FS: houses, contents and backyard shacks) as a
## % of the occupants' mean income (log scale, colors stop at 25%). (7)
## Number of informal settlement households by sandbag levels chosen (gray:
## unprotected).

solver_cells = B["solver_cells"] > 0
d0, d1 = DEPTH_CELL[0], DEPTH_CELL[1]
DEPTH_VALUES = np.where(
    solver_cells, np.array([np.minimum(d0, d1), np.maximum(d1 - d0, 0)]), 0)
MAPS["depth"] = ("Expected flood depth (cm/year)", CC_COLORS,
                 ["No climate change", "Increase with climate change"],
                 "Expected flood depth")

for kind, filename in {"depth": "fig05a_map_flood_depth_cc.png",
                       "damages_type": "fig05b_map_damages_by_type.png",
                       "protection": "fig07_map_self_protection.png"}.items():
    fig, (ax,) = grid3d(1)
    draw_map(ax, B, kind, values=DEPTH_VALUES if kind == "depth" else None)
    save(fig, "maps_bench", filename)

level_figure([(B, t) for t in LEVEL_MAPS["damages"][0]], "damages",
             [HTYPES[t] for t in LEVEL_MAPS["damages"][0]],
             "fig06_map_damages.png", "maps_bench")

# 2x2 summary of the flood maps: commented out

###############################################################################

# %% FIGURES 13-21: 3D MAPS FOR EACH SCENARIO

print("FIGURES 13-21: SCENARIO MAPS")

## FOOTNOTE: (13), (14) Maps (5b) and (7) in each scenario, with the same z
## scale in all panels. (15)-(17) Map (4) for FP, IB and IS, one figure per
## housing type with one panel per scenario, with the same scales as the
## housing type's panel in (4). (18)-(21) Map (6) for FP, IB, IS and FS, one
## figure per housing type with one panel per scenario, with the same scales
## as (6).

# Households by type, by group, housing quantity: commented out
SCENARIO_MAPS = {"damages_type": "fig13_scenarios_damages_by_type.png",
                 "protection": "fig14_scenarios_self_protection.png"}

for kind, filename in SCENARIO_MAPS.items():
    fig, axes = grid3d(len(SCENARIOS))
    zlabel, colors, labels, title = MAPS[kind]
    top = max(bar_top(map_values(sim[s], kind)) for s in SCENARIOS)
    for ax, s in zip(axes, SCENARIOS):
        draw_map(ax, sim[s], kind, zlim=(0, top), legend=False)
        ax.set_title(SCEN_LABELS[s], pad=-4)
    fig.legend(handles=legend_handles(colors, labels), loc="lower center",
               ncol=len(labels), fontsize=9, title=title,
               bbox_to_anchor=(0.5, 0.0), **WHITE_LEGEND)
    save(fig, "maps_scen", filename)

for t, number in zip(LEVEL_MAPS["housing_cost"][0], (15, 16, 17)):
    level_figure([(sim[s], t) for s in SCENARIOS], "housing_cost",
                 [SCEN_LABELS[s] for s in SCENARIOS],
                 f"fig{number}_scenarios_housing_cost_{HTYPES_SHORT[t]}.png",
                 "maps_scen")

for t, number in zip(LEVEL_MAPS["damages"][0], (18, 19, 20, 21)):
    level_figure([(sim[s], t) for s in SCENARIOS], "damages",
                 [SCEN_LABELS[s] for s in SCENARIOS],
                 f"fig{number}_scenarios_damages_{HTYPES_SHORT[t]}.png",
                 "maps_scen")

###############################################################################

# %% FIGURES 8-12: BENCHMARK DISTRIBUTIONS ACROSS FLOOD DEPTH AND RENTS

print("FIGURES 8-11: BENCHMARK DISTRIBUTIONS")

## FOOTNOTE: Distributions over the expected annual flood depth of each
## household's dwelling (cm/year, before any protection investment, log
## scale; per housing type, frequent pluvial floods are drained in formal
## housing, backyards and RDP premises), for exposed households (expected
## depth above 0.1 cm/year), by income group. (8) Number of households,
## stacked by housing type, and the share of exposed households in the
## group, with the same y axis in all panels. (9) Total actual expected
## damages to structures and contents (M ZAR/year) by housing type. (10)
## Mean damages households pay for per household (net of insurance
## payouts; FP: contents only; IB: contents only, shacks paid by RDP
## owners; FS: houses, contents and backyard shacks) as a % of the group's
## mean income, by housing type (axis cut at 25%). (9)-(10): connected
## diamonds over the bins with households, linear y axes specific to each
## panel. (11) Number of exposed low-income informal settlement households
## by sandbag levels (gray: unprotected).

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
    # Damages: missing (NaN) in bins without households
    if kind == "damages":
        v, n = binned(d["dmg"][:, i]) / 1e6, binned(hh)
        if total:
            v, n = v.sum(0), n.sum(0)
        return np.where(n > 0, v, np.nan)
    if kind == "damages_relative":
        num, n = binned(d["dmg_borne"][:, i]), binned(hh)
        if total:
            num, n = num.sum(0), n.sum(0)
        return np.where(n > 0, 100 * _share(num, n) / y_bar[i], np.nan)
    if kind == "protection":                           # settlers by level
        v = np.array([binned(hh * (d["level"] == k)[None, :])[2]
                      for k in range(4)])
        return v[1:].sum(0) if total else v
    raise ValueError(kind)


DISTS = {
    # kind: (file stem, groups, colors, labels, legend title, style, y label)
    # style: "stacked" bars (same y axis in all panels), or "lines"
    # (connected diamonds, linear y axes specific to each panel)
    "households": ("08_households", [0, 1, 2, 3], HTYPE_COLORS, HTYPES,
                   "Housing type", "stacked", "Households (thousands)"),
    "damages": ("09_damages", [0, 1, 2, 3], HTYPE_COLORS, HTYPES,
                "Housing type", "lines", "Damages (M ZAR/year)"),
    "damages_relative": ("10_damages_relative", [0, 1, 2, 3], HTYPE_COLORS,
                         HTYPES, "Housing type", "lines",
                         "Damages paid per household\n(% of mean income)"),
    "protection": ("11_self_protection", [0], LEVEL_COLORS, LEVEL_LABELS,
                   "Sandbag levels", "stacked",
                   "Exposed settlement households"),
}
# Formal dwelling size and rent across flood depth: commented out


def lines_ylim(ax, values, cap=None):
    """Linear y axis from 0 to the highest finite value (or cap)."""
    top = np.nanmax(np.concatenate([np.ravel(v) for v in values]))
    top = 1.06 * top if np.isfinite(top) and top > 0 else 1
    ax.set_ylim(0, min(top, cap) if cap else top)


def style_hist(ax, ylabel, label=None, bins=None,
               xlabel="Expected flood depth (cm/year, log scale)",
               log_x=True):
    bins = BINS if bins is None else bins
    if log_x:
        log_axis(ax, "x")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.set_title(label, pad=3)
    style_axes(ax)


def dist_figure(n, sharey=True):
    nrow, ncol = (2, 2) if n == 4 else (1, n)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11 if n > 1 else 6.4,
                                                  3.7 * nrow),
                             squeeze=False, sharey=sharey)
    for ax in axes.flat:
        ax.yaxis.set_tick_params(labelleft=True)
    return fig, axes


def legend_above(fig, handles, title, ncol=None):
    """Figure legend in rows above the panels, with room reserved for it
    (lays the figure out again)."""
    ncol = ncol or len(handles)
    rows = -(-len(handles) // ncol)
    height = (0.3 + 0.22 * rows) / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - height), h_pad=1.2, w_pad=1.2)
    fig.legend(handles=handles, loc="upper left", title=title, ncol=ncol,
               fontsize=8, bbox_to_anchor=(0.005, 1.0), alignment="left")


for kind, (stem, groups, colors, labels, title, style,
           ylabel) in DISTS.items():
    fig, axes = dist_figure(len(groups), sharey=style == "stacked")
    values = {i: dist_values(B, kind, i) for i in groups}
    if style == "stacked":
        top = max(bar_top(v) for v in values.values())
    for ax, i in zip(axes.flat, groups):
        if style == "stacked":
            bars2d(ax, CENTERS, WIDTHS, values[i], colors)
            ax.set_ylim(0, 1.15 * top)
        else:
            for t in range(4):
                connected(ax, CENTERS, values[i][t], color=colors[t])
            lines_ylim(ax, [values[i]], DAMAGE_CAP
                       if kind == "damages_relative" else None)
        style_hist(ax, ylabel, INCGRPS[i] if len(groups) > 1 else None)
        if kind == "households":
            ax.text(0.02, 0.97,
                    f"Exposed: {exposed_share(B, groups=[i]):.1f}% of "
                    f"{B['hh_group'][i].sum():,.0f} households",
                    transform=ax.transAxes, va="top", fontsize=8, color=INK2,
                    bbox=BOX)
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    handles = legend_handles(colors, labels) if style == "stacked" else [
        Line2D([], [], color=c, label=l, **DIAMOND)
        for c, l in zip(colors, labels)]
    if kind == "protection":
        # Legend and exposure shares outside the plot area, on the right
        protected, exposed = settler_shares(B)
        ax = axes.flat[0]
        ax.legend(handles=handles, title=title, loc="upper left",
                  bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.text(1.03, 0.35, f"Exposed settlers using sandbags: "
                f"{protected:.1f}%\nSettlers exposed to floods: "
                f"{exposed:.1f}%", transform=ax.transAxes, va="top",
                fontsize=8, color=INK2, bbox=BOX)
    else:
        legend_above(fig, handles, title)
    save(fig, "dist_bench", f"fig{stem}_benchmark.png")

# Households across damages per household: commented out

###############################################################################

# %% FIGURES 22-25: DISTRIBUTIONS ACROSS FLOOD DEPTH, BY SCENARIO

print("FIGURES 22-25: SCENARIO DISTRIBUTIONS")

## FOOTNOTE: Same distributions as (8)-(11), for the benchmark and the four
## single-policy scenarios, by income group, over all housing types:
## connected diamonds at the bin centres, the benchmark in black, drawn on
## top. (22) Households, with the same y axis in all panels; vertical lines:
## mean expected depth of each distribution. (23) Damages. (24) Damages paid
## per household as a % of the group's mean income (axis cut at 25%).
## (23)-(24): bins with households only, linear y axes specific to each
## panel. (25) Exposed settlers protected by sandbags. Depth is the expected
## flood depth before any protection investment, so the distributions only
## move with relocation.

DIST_SCENARIOS = [s for s in SCENARIOS if s != "combined"]
SCEN_FILES = {"households": "fig22_households", "damages": "fig23_damages",
              "damages_relative": "fig24_damages_relative",
              "protection": "fig25_self_protection"}
SCEN_YLABELS = {"protection": "Protected exposed settlers"}
# By housing type, dwelling size and rent across flood depth: commented out


def mean_depth(d, i):
    """Household-weighted mean expected depth of exposed group-i households."""
    return mean_depth_exposed(d, groups=[i])


def scenario_handles(extra=()):
    return [Line2D([], [], color=SCEN_COLORS[s], linestyle=SCEN_STYLES[s],
                   label=SCEN_LABELS[s], **DIAMOND)
            for s in DIST_SCENARIOS] + list(extra)


def scenario_dists(panels, labels, value, ylabel, filename, bins=None,
                   xlabel=None, log_y=False, cap=None, mean=None,
                   sharey=True, log_x=True, extra=()):
    """Connected diamonds of value(d, panel) by bin (depth bins unless
    bins(panel)) for DIST_SCENARIOS, one panel per item of panels; mean(d,
    panel): vertical lines; cap: top of the y axis (unmarked)."""
    fig, axes = dist_figure(len(panels), sharey=sharey)
    for ax, panel, label in zip(axes.flat, panels, labels):
        b = BINS if bins is None else bins(panel)
        x = np.sqrt(b[:-1] * b[1:]) if log_x else (b[:-1] + b[1:]) / 2
        values = []
        for s in DIST_SCENARIOS[1:] + ["benchmark"]:  # benchmark on top
            v = value(sim[s], panel)
            values.append(v)
            connected(ax, x, positive(v) if log_y else v,
                      color=SCEN_COLORS[s], linestyle=SCEN_STYLES[s],
                      zorder=20 if s == "benchmark" else 3)
        if mean:
            for s in DIST_SCENARIOS:
                ax.axvline(mean(sim[s], panel), color=SCEN_COLORS[s],
                           linestyle=SCEN_STYLES[s], linewidth=1, zorder=22)
        style_hist(ax, ylabel, label, b,
                   xlabel or "Expected flood depth (cm/year, log scale)",
                   log_x)
        if log_y:
            log_axis(ax)
        elif sharey:
            ax.set_ylim(bottom=0)
        else:
            lines_ylim(ax, values, cap)
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    handles = scenario_handles(extra)
    if mean:
        handles.append(Line2D([], [], color=INK2, linewidth=1,
                              label="Vertical lines: mean depth"))
    legend_above(fig, handles, "Scenario", ncol=3)
    save(fig, "dist_scen", filename)


for kind, stem in SCEN_FILES.items():
    stem_b, groups, colors, labels, title, style, ylabel = DISTS[kind]
    scenario_dists(
        groups, [INCGRPS[i] for i in groups],
        lambda d, i: dist_values(d, kind, i, total=True),
        SCEN_YLABELS.get(kind, ylabel), f"{stem}_scenarios.png",
        cap=DAMAGE_CAP if kind == "damages_relative" else None,
        mean=mean_depth if kind == "households" else None,
        sharey=style == "stacked")

# Households across damages per household: commented out

###############################################################################

# %% FIGURES 12, 26: HOUSEHOLDS ACROSS RENTS

print("FIGURES 12, 26: HOUSEHOLDS ACROSS RENTS")

## FOOTNOTE: Number of households (thousands) across the annual rent per m2
## of their dwelling, by income group, in the rented housing types (FP, IB,
## IS; RDP housing is free), all households. Bins and axes are specific to
## each panel, from the lowest to the highest rent of the group; mid-low
## income: to the 99.5th percentile of its rents (household-weighted),
## leaving out a few hundred households in settlements and backyards with
## higher rents. (12) Benchmark, stacked by housing type.
## (26) Benchmark and single-policy scenarios, over all rented housing
## types: connected diamonds at the bin centres, the benchmark in black.

RENT_XLABEL = "Annual rent (ZAR/m²)"
# Household-weighted quantile of rents at the axis end, by income group
RENT_TOP = {0: 1, 1: 0.995, 2: 1, 3: 1}


def rent_bins(i, scenarios=("benchmark",)):
    """30 bins from the lowest rent paid by group i in these scenarios to
    the RENT_TOP[i] quantile of these rents (household-weighted)."""
    rents, weights = [], []
    for s in scenarios:
        ok = sim[s]["hh"][:3, i] > 0
        rents.append(sim[s]["rent_clean"][:3][ok])
        weights.append(sim[s]["hh"][:3, i][ok])
    rents, weights = np.concatenate(rents), np.concatenate(weights)
    order = np.argsort(rents)
    cum = np.cumsum(weights[order]) / weights.sum()
    top = rents[order][min(np.searchsorted(cum, RENT_TOP[i]), len(cum) - 1)]
    return np.linspace(rents.min(), top * 1.0001, 31)


def rent_hist(d, i, bins):
    """Households of group i by rent bin, (FP/IB/IS, bins), thousands."""
    out = np.zeros((3, len(bins) - 1))
    for t in range(3):
        idx = np.digitize(d["rent_clean"][t], bins) - 1
        ok = (d["hh"][t, i] > 0) & (idx >= 0) & (idx < len(bins) - 1)
        out[t] = np.bincount(idx[ok], weights=d["hh"][t, i][ok],
                             minlength=len(bins) - 1) / 1e3
    return out


fig, axes = dist_figure(4, sharey=False)
for ax, i in zip(axes.flat, range(4)):
    b = rent_bins(i)
    bars2d(ax, (b[:-1] + b[1:]) / 2, np.full(len(b) - 1, np.diff(b)[0] * 0.92),
           rent_hist(B, i, b), HTYPE_COLORS[:3])
    style_hist(ax, "Households (thousands)", INCGRPS[i], b, RENT_XLABEL,
               log_x=False)
fig.tight_layout(h_pad=1.2, w_pad=1.2)
legend_above(fig, legend_handles(HTYPE_COLORS[:3], HTYPES[:3]), "Housing type")
save(fig, "dist_bench", "fig12_households_over_rent_benchmark.png")

RENT_BINS = {i: rent_bins(i, DIST_SCENARIOS) for i in range(4)}
scenario_dists(range(4), INCGRPS,
               lambda d, i: rent_hist(d, i, RENT_BINS[i]).sum(0),
               "Households (thousands)",
               "fig26_households_over_rent_scenarios.png",
               bins=lambda i: RENT_BINS[i], xlabel=RENT_XLABEL, sharey=False,
               log_x=False)

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
SHORT_ROWS = ["Low inc., non-benef.", "Low inc., RDP owners", "Mid-low inc.",
              "Mid-high inc.", "High inc."]


def row_weights(d, r, mask=None):
    w = d["hh"] * ROW_MASKS[r][:, :, None]
    return w if mask is None else w * mask[:, None, :]


def mean_utility(u, w):
    ok = (w > 0) & np.isfinite(u)
    return np.sum(w[ok] * u[ok]) / w[ok].sum() if ok.any() else np.nan


def welfare(d, mask=None):
    """Welfare of each row in scenario d relative to the benchmark, on the
    households of the subsample mask ((types, cells), None: all): n,
    benchmark households of the subsample; u, mean ex-post utility in d; du,
    its % change; ev, equivalent variation per benchmark household
    (rands/year), so that sum(n x ev) is the aggregate monetary gain of the
    subsample; share, % of the row in the subsample in d; comp, % of the
    subsample's households in the row in d.

    EV as in scenarios.welfare_table, at benchmark prices and locations,
    with the ratio r of mean ex-post utilities (over the subsample): FP
    (ỹ - R q0)(r - 1); IB, IS c z_post (r^(1/α) - 1); RDP owners, with their
    own ratio in each cell (at zero benchmark consumption, c z with z giving
    their utility in d).

    RDP choice (RC1), as in scenarios.welfare_table: the low-income
    non-beneficiaries of the benchmark end up in d's market or in the RDP
    units vacant in the benchmark only, so u in d (first row) averages over
    both; benchmark owners whose unit is vacant in d get the mean ex-post
    utility of all these households in d (in the subsample or not)."""
    res = {key: np.zeros(5) for key in ("n", "u", "du", "ev", "share",
                                        "comp")}
    M = 1 if mask is None else mask[:, None, :]
    # RC1: d's owners of units vacant in the benchmark (zeros without RC),
    # whose ex-post utility is up to A(x) (accounting.ex_post)
    newcomers = np.zeros_like(d["hh"])
    newcomers[3, 0] = d["hh"][3, 0] * (B["vacant"] & ~d["vacant"])
    u_post = d["ex_post_u_post"].copy()
    u_post[3, 0] = u_post[3, 0] * inputs["amenities"]
    pool = mean_utility(u_post, row_weights(d, 0) + newcomers)
    n_sub = (d["hh"] * M).sum()
    for r in range(5):
        wb, ws = row_weights(B, r, mask), row_weights(d, r, mask)
        res["n"][r] = wb.sum()
        res["share"][r] = 100 * ws.sum() / row_weights(d, r).sum()
        res["comp"][r] = 100 * ws.sum() / n_sub
        ub = mean_utility(B["ex_post_u_post"], wb)
        us = mean_utility(d["ex_post_u_post"], ws)
        if r == 0 and newcomers.any():
            us = mean_utility(u_post, ws + newcomers * M)
        res["u"][r], res["du"][r] = us, 100 * (us / ub - 1)
        if d is B:
            continue
        if r == 1:                                     # RDP owners
            with np.errstate(invalid="ignore", divide="ignore",
                             over="ignore"):
                target = np.where(d["vacant"], pool / inputs["amenities"],
                                  d["ex_post_u_post"][3, 0])
                ratio = target / B["ex_post_u_post"][3, 0]
                # At zero consumption (floor): EV = c z, with z giving the
                # target utility at the benchmark backyard share
                z_b, c_b = B["ex_post_z_post"][3, 0], B["ex_post_price"][3, 0]
                mu_b = np.nan_to_num(B["housing_supply"][1]) / 1e6
                housing = (p["RDP_size"] + p["backyard_size"] * (1 - mu_b)
                           - q0) ** p["beta"]
                ev = np.nan_to_num(np.where(
                    z_b > 0, c_b * z_b * (ratio ** (1 / alpha) - 1),
                    c_b * (target / housing) ** (1 / alpha)))
            res["ev"][r] = (np.sum(wb[3, 0] * ev) / wb[3, 0].sum()
                            if wb[3, 0].sum() > 0 else 0.)
            continue
        i, ratio = (0 if r == 0 else r - 1), us / ub
        ev = np.zeros((4, N_CELLS))
        ev[0] = (B["net_income"][i] - np.nan_to_num(B["rent"][0]) * q0) \
            * (ratio - 1)
        for h in (1, 2):
            ev[h] = B["ex_post_price"][h, i] * B["ex_post_z_post"][h, i] \
                * (ratio ** (1 / alpha) - 1)
        w = wb[:, i]
        res["ev"][r] = (np.sum(np.where(w > 0, w * np.nan_to_num(ev), 0))
                        / w.sum() if w.sum() > 0 else 0.)
    return res


def landlords(d):
    """Yearly revenues (rands) of absentee landlords: formal gross (land
    rents developers would pay before compensating their unanticipated
    losses, i.e. δ P + t per m2 of developed land), formal net (gross minus
    the expected compensation of developers' losses), and slumlords' rents
    from informal settlements."""
    t, losses, revenue = d["developer_tax"]
    gross = d["landlord_revenue"][0].sum() + revenue
    net = gross - losses - d["surplus_damages"].sum()
    return gross, net, d["landlord_revenue"][1].sum()


SUMMARY_ROWS = ["Aggregate damages (M ZAR)", "o.w. developer losses (M ZAR)",
                "Damages, exposed (% of income)", "Exposed households (%)",
                "Mean expected flood depth, exposed (cm/year)",
                "Exposed settlers protected (%)",
                "Public cost of PS and SI (M ZAR)"]
AGG_EV = "Aggregate EV (M ZAR)"
AGG_EV_LANDLORDS = "Aggregate EV + landlords' net revenue (M ZAR)"
# Per household of the subsample (benchmark households)
AGG_EV_HH = "Aggregate EV (ZAR/household)"
AGG_EV_LANDLORDS_HH = "Aggregate EV + landlords' net revenue (ZAR/household)"


def metrics(d, label):
    """Scalar outcomes of scenario d: levels of the summary rows, and welfare
    changes relative to the benchmark by subsample ("<subsample>|<kind>|
    <row>" keys; aggregate EVs in M ZAR)."""
    info = SCEN["scenarios"][label]
    budget = info["budget"] or [0, 0, 0]
    exposed_hh = d["hh"] * EXP3
    income = (exposed_hh * y_bar[None, :, None]).sum()
    out = {
        "Aggregate damages (M ZAR)": d["dmg"].sum() / 1e6,
        "o.w. developer losses (M ZAR)": d["dev_loss"].sum() / 1e6,
        "Damages, exposed (% of income)":
            100 * (d["dmg_borne"] * EXP3).sum() / income,
        "Exposed households (%)": 100 * exposed_hh.sum() / d["hh"].sum(),
        "Mean expected flood depth, exposed (cm/year)":
            mean_depth_exposed(d),
        "Exposed settlers protected (%)": settler_shares(d)[0],
        "Public cost of PS and SI (M ZAR)": (budget[0] + budget[1]) / 1e6,
    }
    for sub, mask in SUBSAMPLES.items():
        w = welfare(d, mask)
        for r, name in enumerate(SHORT_ROWS):
            out[f"{sub}|Utility (%)|{name}"] = w["du"][r]
            out[f"{sub}|EV (ZAR/household/year)|{name}"] = w["ev"][r]
        out[f"{sub}|{AGG_EV}"] = np.sum(w["n"] * w["ev"]) / 1e6
        out[f"{sub}|{AGG_EV_HH}"] = np.sum(w["n"] * w["ev"]) / w["n"].sum()
    out["Landlords' net revenue (M ZAR)"] = \
        (landlords(d)[1] - landlords(B)[1]) / 1e6
    out[f"all|{AGG_EV_LANDLORDS}"] = \
        out[f"all|{AGG_EV}"] + out["Landlords' net revenue (M ZAR)"]
    out[f"all|{AGG_EV_LANDLORDS_HH}"] = \
        1e6 * out[f"all|{AGG_EV_LANDLORDS}"] / N_TOTAL
    return out


METRICS = {s: metrics(sim[s], s) for s in SCENARIOS}

###############################################################################

# %% TABLES 3, 5-9: DAMAGES, EXPOSURE AND WELFARE

print("TABLES 3, 5-9")

COLS_LEVELS = ["benchmark"] + SCENARIOS[1:]
# Labels with the benchmark population shares
TYPE_SHARE = 100 * B["hh_type"].sum(1) / N_TOTAL
GROUP_SHARE = 100 * B["hh_group"].sum(1) / N_TOTAL
HTYPES_PCT = [f"{h} ({v:.1f}%)" for h, v in zip(HTYPES, TYPE_SHARE)]
INCGRPS_PCT = [f"{g} ({v:.1f}%)" for g, v in zip(INCGRPS, GROUP_SHARE)]

## TABLE 3: BENCHMARK DAMAGES AND EXPOSURE BY HOUSING TYPE AND INCOME GROUP

## FOOTNOTE: Benchmark. Damages: actual expected annual damages to structures
## and contents (M ZAR/year; formal structures attributed to occupants), of
## which developers' losses, the formal structure damages they do not
## perceive (financed by a tax on developed land). Damages, exposed: damages
## households pay for (net of insurance payouts; no formal private
## structures; backyard shacks paid by the RDP owners of the cell) as a % of
## the exposed households' mean income. Exposed: households with expected
## flood depth above 0.1 cm/year at their dwelling, before any protection
## investment. Mean expected flood depth, exposed: household-weighted,
## exposed households. Labels: % of all households in the benchmark. Dashes:
## no households, or not applicable.

METRICS25 = [("Damages (M ZAR)", "pct", 1),
             ("o.w. developer losses (M ZAR)", "pct", 1),
             ("Damages, exposed (% of income)", "pp", 3),
             ("Exposed households (%)", "pp", 1),
             ("Mean expected flood depth, exposed (cm/year)", "abs", 3)]


def exposure_stats(d, types, groups):
    """Metrics of METRICS25 over housing types and income groups (lists)."""
    hh = d["hh"][np.ix_(types, groups)]
    exp = EXPOSED[types][:, None, :]
    n, n_exp = hh.sum(), (hh * exp).sum()
    income = (hh * exp * y_bar[groups][None, :, None]).sum()
    dmg = d["dmg"][np.ix_(types, groups)].sum()
    borne = (d["dmg_borne"][np.ix_(types, groups)] * exp).sum()
    dev = d["dev_loss"][groups].sum() if 0 in types else np.nan
    if n <= 0:
        return [np.nan] * len(METRICS25)
    return [dmg / 1e6, dev / 1e6, 100 * borne / income if income else np.nan,
            100 * n_exp / n, mean_depth_exposed(d, types, groups)]


COLS = INCGRPS_PCT + ["All"]
rows = {}
for t, label in enumerate(HTYPES_PCT + ["All housing types"]):
    types = [t] if t < 4 else [0, 1, 2, 3]
    for j, col in enumerate(COLS):
        groups = [j] if j < 4 else [0, 1, 2, 3]
        for (m, _, _), v in zip(METRICS25, exposure_stats(B, types, groups)):
            rows.setdefault((label, m), {})[col] = v
table03 = pd.DataFrame(rows).T[COLS]
DIGITS = {m: dg for m, _, dg in METRICS25}
write_table(table03, "table03_damages_exposure_benchmark",
            lambda block, row, c, v: _num(v, DIGITS[row]))

## TABLE 6: CHANGES ACROSS SCENARIOS BY INCOME GROUP

## FOOTNOTE: Benchmark levels (first column) and changes relative to the
## benchmark of the metrics of Table 3, over all housing types: damages and
## developers' losses in % changes, shares in percentage points, mean depth
## in cm/year. Labels: % of all households in the benchmark.

UNITS = {"pct": "changes in %", "pp": "changes in pp", "abs": "changes"}
rows = {}
for j, grp in enumerate(INCGRPS_PCT + ["All"]):
    groups = [j] if j < 4 else [0, 1, 2, 3]
    stats = {s: exposure_stats(sim[s], [0, 1, 2, 3], groups)
             for s in SCENARIOS}
    for m, (metric, change, _) in enumerate(METRICS25):
        base = stats["benchmark"][m]
        row = {"benchmark": base}
        for s in SCENARIOS[1:]:
            v = stats[s][m]
            if change == "pct":
                row[s] = 100 * (v / base - 1) if base else np.nan
            else:
                row[s] = v - base
        block = f"{metric[:metric.rfind(' (')]} ({metric[metric.rfind('(') + 1:-1]}; {UNITS[change]})" \
            if "(" in metric else metric
        rows[(block, grp)] = row
table06 = pd.DataFrame(rows).T[COLS_LEVELS]
table06.columns = [SCEN_SHORT[c] for c in COLS_LEVELS]
DIGITS26 = {f"{m[:m.rfind(' (')]} ({m[m.rfind('(') + 1:-1]}; {UNITS[ch]})": dg
            for m, ch, dg in METRICS25}
write_table(table06, "table06_damages_exposure_scenarios",
            lambda block, row, c, v: _num(v, DIGITS26[block],
                                          sign=c != "Bench."))

## TABLES 7, 8 AND 9: WELFARE (ALL, EXPOSED AND MOST EXPOSED HOUSEHOLDS)

## FOOTNOTE: Ex-post utilities (with actual instead of perceived damages):
## benchmark levels (first column; household-weighted means, RDP owners'
## mean over cells) and % changes. EV: equivalent variation per benchmark
## household of each row, at benchmark prices and locations, in ZAR/year
## (benchmark column: the row's mean income ȳ). Labels: share of all
## households. Taxes financing PS and SI (groups 3-4) are included.
## Landlords: gross revenue, land rents developers would pay before
## compensating their unanticipated losses, and net revenue, after that
## compensation; slumlords: rents of informal settlements (ZAR/year per
## household of the city; benchmark: levels; scenarios: changes). Total:
## aggregate monetary gain, the sum of EVs weighted by the number of
## households of each row plus the change in landlords' net revenue per
## household times the number of households (slumlords excluded), M
## ZAR/year, and that gain minus the reduction in aggregate damages. Public
## protection is free in the model: its aggregate gain is the largest annual
## cost at which it breaks even. Changes below about 0.3% for groups 1-2 are
## within solver noise. (8) The same for exposed households (expected depth
## above 0.1 cm/year; benchmark exposed households for EVs, each scenario's
## exposed households for utilities), with the % of each row exposed in each
## scenario; total: sum of EVs weighted by each row's share of the benchmark
## subsample (ZAR/household/year). (9) The same for the most exposed
## households (above 1 cm/year), with each row's share of the subsample and
## the subsample's share of all households in each scenario.


def welfare_table(sub):
    mask = SUBSAMPLES[sub]
    res = {s: welfare(sim[s], mask) for s in SCENARIOS}
    rows = {}
    labels = ROWS if sub == "most" else ROW_LABELS
    for r, label in enumerate(labels):
        if sub == "exposed":
            rows[(label, "Exposed (% of row)")] = {
                s: res[s]["share"][r] for s in SCENARIOS}
        if sub == "most":
            rows[(label, "Share of subsample (%)")] = {
                s: res[s]["comp"][r] for s in SCENARIOS}
        rows[(label, "Utility")] = {"benchmark": res["benchmark"]["u"][r],
                                    **{s: res[s]["du"][r]
                                       for s in SCENARIOS[1:]}}
        rows[(label, "EV (ZAR/household/year)")] = {
            "benchmark": ROW_INCOME[r],
            **{s: res[s]["ev"][r] for s in SCENARIOS[1:]}}
    if sub == "all":
        # Revenues per household of the city (ZAR/year)
        acc = {s: np.array(landlords(sim[s])) / N_TOTAL for s in SCENARIOS}
        net = "Net revenue (ZAR/household/year)"
        for k, (block, name) in enumerate(
                [("Landlords", "Gross revenue (ZAR/household/year)"),
                 ("Landlords", net),
                 ("Slumlords", "Revenue (ZAR/household/year)")]):
            rows[(block, name)] = {"benchmark": acc["benchmark"][k],
                                   **{s: acc[s][k] - acc["benchmark"][k]
                                      for s in SCENARIOS[1:]}}
        gain = {s: (np.sum(res[s]["n"] * res[s]["ev"])
                    + rows[("Landlords", net)][s] * N_TOTAL) / 1e6
                for s in SCENARIOS[1:]}
        cut = {s: (sim["benchmark"]["dmg"].sum() - sim[s]["dmg"].sum()) / 1e6
               for s in SCENARIOS[1:]}
        rows[("Total", "Aggregate monetary gain (M ZAR/year)")] = {
            "benchmark": np.nan, **gain}
        # Gain beyond the avoided damages (the ratio flips sign when
        # damages rise)
        rows[("Total", "Minus the reduction in aggregate damages "
                       "(M ZAR/year)")] = {
            "benchmark": np.nan, **{s: gain[s] - cut[s] for s in gain}}
    else:
        rows[("Total", "Sum of EVs, weighted by subsample shares "
                       "(ZAR/household/year)")] = {
            "benchmark": np.nan,
            **{s: np.sum(res[s]["n"] * res[s]["ev"]) / res[s]["n"].sum()
               for s in SCENARIOS[1:]}}
    if sub == "most":
        rows[("Total", "Subsample (% of all households)")] = {
            s: exposed_share(sim[s], mask=mask) for s in SCENARIOS}
    table = pd.DataFrame(rows).T[COLS_LEVELS]
    table.columns = [SCEN_SHORT[c] for c in COLS_LEVELS]
    return table


def fmt_welfare(block, row, c, v):
    if row.startswith(("Exposed (%", "Share of", "Subsample")):
        return _num(v, 1)
    if c == "Bench.":
        return _num(v, 0 if np.isfinite(v) and abs(v) > 100 else 1)
    return _num(v, 3 if row.startswith("Utility") else 1, sign=True)


for sub, stem in (("all", "table07_welfare"),
                  ("exposed", "table08_welfare_exposed"),
                  ("most", "table09_welfare_most_exposed")):
    table = welfare_table(sub)
    write_table(table, stem, fmt_welfare)
    print(table.round(3).to_string())

## TABLE 5: SCENARIO SUMMARY

## FOOTNOTE: Levels by scenario. Aggregate damages: actual expected annual
## damages to structures and contents, of which developers' losses (formal
## structure damages developers do not perceive under risk misperception,
## financed by a tax on developed land). Damages, exposed: damages households
## pay for (net of insurance payouts; no formal private structures) as a % of
## the exposed households' mean income. Exposed: expected depth above 0.1
## cm/year, before any protection investment. Mean expected flood depth,
## exposed: household-weighted, exposed households. Public cost: free
## sandbags and
## insurance payouts, financed by lump-sum taxes on groups 3-4 (public
## protection is free in the model).

rows = {name: {s: METRICS[s][name] for s in SCENARIOS} for name in SUMMARY_ROWS}
table_summary = pd.DataFrame(rows).T[COLS_LEVELS]
table_summary.columns = [SCEN_SHORT[c] for c in COLS_LEVELS]
write_table(pd.concat({"": table_summary}), "table05_scenario_summary",
            lambda block, row, c, v: _num(
                v, 3 if "(% of income)" in row or "depth" in row else 1))
print(table_summary.round(2).to_string())

# Exposure by housing type and income group, by scenario: removed

###############################################################################

# %% TABLES 4, 10: RENT REGRESSIONS

print("TABLES 4, 10: RENT REGRESSIONS")

## FOOTNOTE: OLS regressions of the log of the annual rent per m2 across
## cells (500 m grid cells with households of the housing type), with
## heteroskedasticity-robust (HC1) standard errors in parentheses; * p < 0.1,
## ** p < 0.05, *** p < 0.01. Expected flood depth (cm/year, before any
## protection investment, at the housing type's drainage level) and its
## square; log mean income net of commuting costs of the occupants
## (household-weighted, net of lump-sum taxes; backyard revenues accrue to
## RDP owners, who pay no rent, so they enter no rent regression); log
## amenity index; log formal dwelling size (FP); log number of households of
## the housing type in the cell; share of the cell available to the housing
## type. The informal disamenity indices are constants (one per informal
## type) and are absorbed by the constant (Table 4) or the housing type
## dummies (Table 10). (4) Benchmark, by housing type. (10) FP, IB and IS
## pooled, with housing type dummies (FP: reference), by scenario. Equilibrium
## relations, not causal effects.

REG_VARS = ["Expected flood depth (cm/year)", "Expected flood depth squared",
            "Log net income", "Log amenity index", "Log dwelling size",
            "Log households", "Available land share"]


def reg_data(d, t):
    """(log rent, regressors (cells, REG_VARS)) for housing type t."""
    hh, R = d["hh_type"][t], d["rent_clean"][t]
    with np.errstate(divide="ignore", invalid="ignore"):
        income = (d["hh"][t] * d["net_income"]).sum(0) / hh
        X = np.column_stack([DEPTH[t], DEPTH[t] ** 2, np.log(income),
                             np.log(inputs["amenities"]),
                             np.log(np.maximum(d["consumption"][t], 1e-9)),
                             np.log(hh), inputs["coeff_land"][t]])
    ok = (hh > 0) & (R > 0) & np.isfinite(X).all(1)
    return np.log(R[ok]), X[ok]


def ols(y, X, names):
    """OLS with HC1 standard errors: (coefficients, s.e.) Series, n, R2."""
    X = np.column_stack([X, np.ones(len(y))])
    names = names + ["Constant"]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - k)
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return (pd.Series(beta, names), pd.Series(np.sqrt(np.diag(V)), names),
            n, r2)


def reg_column(results):
    beta, se, n, r2 = results
    col = {}
    for name in beta.index:
        z = abs(beta[name] / se[name]) if se[name] > 0 else 0
        stars = "***" if z > 2.576 else "**" if z > 1.96 else "*" \
            if z > 1.645 else ""
        col[(name, "coef")] = f"{beta[name]:.4f}{stars}"
        col[(name, "se")] = f"({se[name]:.4f})"
    col[("Observations", "")] = f"{n:,}"
    col[("R²", "")] = f"{r2:.3f}"
    return col


def reg_table(columns):
    df = pd.DataFrame(columns)
    order = []
    for name in REG_VARS + ["Informal backyards", "Informal settlements",
                            "Constant"]:
        order += [(name, "coef"), (name, "se")]
    order += [("Observations", ""), ("R²", "")]
    df = df.reindex([o for o in order if o in df.index]).fillna("")
    df.index = pd.MultiIndex.from_tuples(
        [("", name if kind != "se" else "") for name, kind in df.index])
    return df


def reg_latex(df, stem):
    """Regression tables: coefficient rows, then standard error rows."""
    df.to_excel(FOLDERS["tables"] / f"{stem}.xlsx")
    cols = list(df.columns)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & ".join([""] + [str(c) for c in cols]) + r" \\", r"\midrule"]
    for (_, name), values in df.iterrows():
        if name == "Observations":
            lines.append(r"\midrule")
        name = name.replace("²", r"\textsuperscript{2}")
        lines.append(name + " & " + " & ".join(
            str(v).replace("*", r"$^{*}$").replace(r"$^{*}$$^{*}$", r"$^{**}$")
            for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FOLDERS["tables"] / f"{stem}.tex").write_text("\n".join(lines),
                                                   encoding="utf-8")


cols04 = {}
for t in (0, 1, 2):
    y, X = reg_data(B, t)
    keep = [k for k in range(len(REG_VARS))
            if not (t > 0 and REG_VARS[k] == "Log dwelling size")]
    cols04[HTYPES[t]] = reg_column(ols(y, X[:, keep],
                                       [REG_VARS[k] for k in keep]))
table04 = reg_table(cols04)
reg_latex(table04, "table04_rent_regressions_by_type")


def pooled(d):
    ys, Xs = [], []
    for t in (0, 1, 2):
        y, X = reg_data(d, t)
        dummies = np.zeros((len(y), 2))
        if t:
            dummies[:, t - 1] = 1
        ys.append(y)
        Xs.append(np.column_stack([X, dummies]))
    return np.concatenate(ys), np.vstack(Xs)


table10 = reg_table({SCEN_SHORT[s]: reg_column(ols(
    *pooled(sim[s]), REG_VARS + ["Informal backyards",
                                 "Informal settlements"]))
    for s in SCENARIOS})
reg_latex(table10, "table10_rent_regressions_scenarios")
print(table04.to_string(), "\n", table10.to_string())

###############################################################################

# %% FIGURES 27-29: HOUSING TYPES AND DAMAGES ACROSS SCENARIOS

print("FIGURES 27-29: SCENARIO LINES")

## FOOTNOTE: By income group and for the whole city: connected diamonds
## across scenarios, by housing type and over all housing types (black).
## (27) Number of households (linear y axes specific to each panel). (28)
## Actual expected damages to structures and contents (M ZAR/year). (29)
## Damages exposed households pay for (net of insurance payouts; no formal
## private structures; backyard shacks paid by RDP owners) as a % of their
## mean income; pies: share of the group's households exposed. (28)-(29):
## same y axis (log scale) in all panels.


def scenario_line_values(kind, groups):
    """(housing types + all, scenarios) values for income groups `groups`."""
    out = np.zeros((5, len(SCENARIOS)))
    for k, s in enumerate(SCENARIOS):
        d = sim[s]
        if kind == "households":
            v = d["hh"][:, groups].sum((1, 2)) / 1e3
            out[:, k] = np.append(v, v.sum())
        elif kind == "damages":
            v = d["dmg"][:, groups].sum((1, 2)) / 1e6
            out[:, k] = np.append(v, v.sum())
        else:                                          # % of income, exposed
            num = (d["dmg_borne"][:, groups] * EXP3).sum((1, 2))
            den = (d["hh"][:, groups] * EXP3
                   * y_bar[groups][None, :, None]).sum((1, 2))
            out[:, k] = 100 * _share(np.append(num, num.sum()),
                                     np.append(den, den.sum()))
    return out


def scenario_lines(kind, filename, ylabel, log=True):
    """Log: same log y axis in all panels; else linear y axes specific to
    each panel."""
    panels = [[i] for i in range(4)] + [[0, 1, 2, 3]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=log)
    x = np.arange(len(SCENARIOS))
    for k, (ax, groups) in enumerate(zip(axes.flat, panels)):
        v = scenario_line_values(kind, groups)
        for t in range(4):
            connected(ax, x, positive(v[t]), color=HTYPE_COLORS[t])
        connected(ax, x, positive(v[4]), color=INK)
        if log:
            log_axis(ax)
        else:
            lines_ylim(ax, [positive(v)])
        if kind == "damages_relative":   # pies: share of households exposed
            band = blended_transform_factory(ax.transData, ax.transAxes)
            for j, s in enumerate(SCENARIOS):
                share = exposed_share(sim[s], groups=groups)
                pie = ax.inset_axes([j - 0.3, 1.0, 0.6, 0.12],
                                    transform=band)
                pie.pie([share, 100 - share], colors=[INK2, GRID],
                        startangle=90, counterclock=False)
                pie.set_aspect("equal")
        ax.set_xticks(x, [SCEN_SHORT[s] for s in SCENARIOS], rotation=35,
                      ha="right", fontsize=8)
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.set_title(INCGRPS[k] if k < 4 else "All households", pad=3
                     if kind != "damages_relative" else 26)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_tick_params(labelleft=True)
        style_axes(ax)
    handles = [Line2D([], [], color=c, label=l, **DIAMOND)
               for c, l in zip(HTYPE_COLORS + [INK],
                               HTYPES + ["All housing types"])]
    if kind == "damages_relative":
        handles.append(Patch(facecolor=INK2,
                             label="Pies: share of households exposed"))
    axes.flat[5].axis("off")
    axes.flat[5].legend(handles=handles, loc="center left", fontsize=9,
                        title="Housing type")
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save(fig, "bars", filename)


scenario_lines("households", "fig27_households_by_type_scenarios.png",
               "Households (thousands)", log=False)
scenario_lines("damages", "fig28_damages_by_type_scenarios.png",
               "Damages (M ZAR/year, log scale)")
scenario_lines("damages_relative", "fig29_damages_relative_scenarios.png",
               "Damages paid by exposed\n(% of income, log scale)")

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
## other), the opposite sign substitutes (they overlap). (11) Pairwise
## interactions of the scenario summary outcomes and aggregate EVs (M
## ZAR/year). (12)-(13) Shapley values, one panel per subsample (all,
## exposed above 0.1 cm/year, most exposed above 1 cm/year households;
## header: share of all households in the benchmark), for each row of the
## welfare tables: (12) change in ex-post utility (%); (13) EV (ZAR per
## benchmark household of the row and subsample) and aggregate EV per
## household of the subsample (EVs weighted by the rows' shares of the
## benchmark subsample, plus the change in landlords' net revenue per
## household for all households). (30) Interaction matrices.

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
    Y = {}
    for r in range(5):
        for subset in combinations(LEVERS, r):
            label = scenario_of(subset)
            Y[frozenset(subset)] = (METRICS[label] if label in METRICS
                                    else metrics(load_scenario(label), label))
    empty, n = frozenset(), len(LEVERS)

    def effect(subset, y):
        return Y[frozenset(subset)][y] - Y[empty][y]

    def shapley(names):
        rows = {}
        for y in names:
            row = {}
            for j in LEVERS:
                others = [k for k in LEVERS if k != j]
                row[LEVER_SHORT[j]] = sum(
                    factorial(r) * factorial(n - r - 1) / factorial(n)
                    * (Y[frozenset(S) | {j}][y] - Y[frozenset(S)][y])
                    for r in range(n) for S in combinations(others, r))
            row["Combined"] = effect(LEVERS, y)
            rows[y] = row
        return pd.DataFrame(rows).T

    ## Pairwise interactions: summary outcomes and aggregate EVs
    INTER_NAMES = SUMMARY_ROWS + [f"{sub}|{AGG_EV}" for sub in SUBSAMPLES] \
        + [f"all|{AGG_EV_LANDLORDS}"]

    def pretty(y):
        if "|" not in y:
            return y
        sub, *rest = y.split("|")
        return f"{rest[-1]}, {SUBSAMPLE_NAMES[sub].split(' (')[0].lower()}"

    rows = {}
    for y in INTER_NAMES:
        row = {LEVER_SHORT[j]: effect([j], y) for j in LEVERS}
        for j, k in combinations(LEVERS, 2):
            row[f"{LEVER_SHORT[j]} x {LEVER_SHORT[k]}"] = (
                effect([j, k], y) - effect([j], y) - effect([k], y))
        row["Combined"] = effect(LEVERS, y)
        row["Sum of singles"] = sum(effect([j], y) for j in LEVERS)
        rows[pretty(y)] = row
    table_inter = pd.concat({"": pd.DataFrame(rows).T})
    write_table(table_inter, "table11_pairwise_interactions",
                lambda b, row, c, v: _num(v, 3 if "(% of" in row or "depth"
                                          in row else 1, sign=True),
                folder="interactions")

    ## Shapley values, three panels: (12) utility by row; (13) EV by row and
    ## aggregate EV per household of the subsample
    for kind, stem in (("Utility (%)", "table12_shapley_utility"),
                       ("EV (ZAR/household/year)", "table13_shapley_ev")):
        blocks = {}
        for sub in SUBSAMPLES:
            share = (100 if sub == "all"
                     else exposed_share(B, mask=SUBSAMPLES[sub]))
            header = f"{SUBSAMPLE_NAMES[sub]}, {share:.1f}% of households"
            names = [f"{sub}|{kind}|{name}" for name in SHORT_ROWS]
            labels = list(ROWS)
            if kind.startswith("EV"):
                names.append(f"all|{AGG_EV_LANDLORDS_HH}" if sub == "all"
                             else f"{sub}|{AGG_EV_HH}")
                labels.append("Aggregate EV + landlords' net revenue"
                              if sub == "all" else "Aggregate EV")
            df = shapley(names)
            df.index = labels
            blocks[header] = df
        table_shapley = pd.concat(blocks)
        write_table(table_shapley, stem,
                    lambda b, row, c, v, k=kind: _num(
                        v, 3 if k.startswith("Utility") else 1, sign=True),
                    folder="interactions")
        print(table_shapley.round(3).to_string())

    ## FIGURE: interaction matrices (diagonal: single effects; off-diagonal:
    ## pairwise interactions). Colors are oriented so that blue improves the
    ## outcome (fewer damages, exposed households or lower depth; more welfare
    ## or protection): on the diagonal, a beneficial single effect; off the
    ## diagonal, an interaction that adds to the benefits (complements when
    ## both levers help), red one that takes away from them (substitutes)
    KEY = {"Aggregate damages (M ZAR)": (-1, "Aggregate damages (M ZAR/year)"),
           "Exposed households (%)": (-1, "Exposed households (%)"),
           "Exposed settlers protected (%)":
               (1, "Exposed settlers protected (%)"),
           "Mean expected flood depth, exposed (cm/year)":
               (-1, "Mean expected flood depth, exposed (cm/year)"),
           f"all|{AGG_EV}": (1, "Aggregate EV, all households (M ZAR/year)"),
           f"exposed|{AGG_EV}":
               (1, "Aggregate EV, exposed households (M ZAR/year)"),
           f"most|{AGG_EV}":
               (1, "Aggregate EV, most exposed households (M ZAR/year)")}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    for ax, (y, (better, label)) in zip(axes.flat, KEY.items()):
        M = np.full((n, n), np.nan)
        for a, j in enumerate(LEVERS):
            M[a, a] = effect([j], y)
            for b, k in enumerate(LEVERS):
                if b > a:
                    M[a, b] = M[b, a] = (effect([j, k], y) - effect([j], y)
                                         - effect([k], y))
        lim = np.nanmax(np.abs(M)) or 1
        ax.imshow(better * M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
        digits = 3 if lim < 0.1 else 2 if lim < 1 else 1 if lim < 10 else 0
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
        ax.set_title(f"{label}\ncombined "
                     f"{_num(effect(LEVERS, y), digits, True)}, sum of "
                     f"singles "
                     f"{_num(sum(effect([j], y) for j in LEVERS), digits, True)}",
                     fontsize=8.5, pad=4)
        for side in ax.spines.values():
            side.set_visible(False)
    axes.flat[7].axis("off")
    axes.flat[7].legend(handles=[
        Patch(facecolor="#184f95", label="Improves the outcome\n(off-diagonal:"
              " levers reinforce\neach other)"),
        Patch(facecolor="#b02a2a", label="Worsens the outcome\n(off-diagonal:"
              " levers overlap)"),
        Patch(facecolor="white", edgecolor=INK2,
              label="Diagonal (bold): single effects\nOff-diagonal: pairwise "
              "interactions")], loc="center left", fontsize=9)
    fig.tight_layout(h_pad=2, w_pad=1.5)
    save(fig, "interactions", "fig30_interaction_matrices.png")

###############################################################################

# %% DIAGNOSTICS

print("DIAGNOSTICS")
for s in SCENARIOS:
    d = sim[s]
    print(f"  {SCEN_LABELS[s]:>28s}: max population error "
          f"{100 * np.abs(d['error']).max():.2f}%, insurance share "
          f"{SCEN['scenarios'][s]['insurance_share']}, vacant RDP units "
          f"{d['rdp_vacant'].sum():,.0f}, uncovered RDP damages "
          f"{d['surplus_damages'].sum():,.0f}")
for sub in ("all", "exposed", "most"):
    print(f"  aggregate EV ({sub}, M ZAR):",
          {SCEN_SHORT[s]: round(METRICS[s][f"{sub}|{AGG_EV}"], 1)
           for s in SCENARIOS[1:]})

# %%
