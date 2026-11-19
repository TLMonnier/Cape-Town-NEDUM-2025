# ##########################################################
# ## FLOOD MODEL: OUTPUT SCRIPT FOR THE POLICY SCENARIOS ###
# ##########################################################
# Tables and figures for the policy scenarios solved by main.py (benchmark,
# public protection, subsidised self-protection, subsidised insurance, no
# risk misperception, all combined; with main.py's FACTORIAL, the other
# combinations of these four levers; and a check run without any
# self-protection), read from Output/policy_scenarios/simulations.
#
# Results go to Output/policy_scenarios/<section>/ (user, 2026-11-19), with
# section = main, appendix or online_appendix, and in each:
#     figures/  figures of the PDF report;
#     tables/   tables of the PDF report (.tex and .xlsx);
#     panels/   each subplot of a multi-panel figure as its own figure;
#     builds/   figures that add housing types, income groups or scenarios
#               one at a time (same axes as the full figure).
# Files are named by content (numbers only appear in the PDF). Outputs with
# damages or welfare also come in an ex-ante version ("_exante": damages
# agents anticipate in equilibrium, equilibrium utilities), next to the
# ex-post one (actual damages, utilities after the true damages are
# realised). The script also writes pdf/figures_tables.tex, with captions
# and notes taken from the registry below (compile it with pdflatex).
#
# Damages: household damages exclude formal private structures (borne by
# developers), reported separately as structure damages; damages as a share
# of income only cover the damages households pay for (net of insurance
# payouts): contents, settlers' shacks, RDP houses, and backyard shacks,
# paid by the RDP owners of the cell. Expected flood depth is before any
# protection investment (sandbags, public protection).

# %% PREAMBLE

import json
import os
import re
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
                               Normalize, TwoSlopeNorm)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy.spatial import cKDTree

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

T0 = time.time()

## PATHS

PATH_SCEN = config.ROOT / "Output" / "policy_scenarios"
PATH_SIMUL = PATH_SCEN / "simulations"
PATH_PDF = PATH_SCEN / "pdf"
SECTIONS = {"main": "main", "appendix": "appendix",
            "online": "online_appendix"}
SECTION_TITLES = {"main": "Main body", "appendix": "Appendix",
                  "online": "Online appendix"}
KINDS = ("figures", "tables", "panels", "builds")
for _sec in SECTIONS.values():
    for _kind in KINDS:
        (PATH_SCEN / _sec / _kind).mkdir(parents=True, exist_ok=True)
PATH_PDF.mkdir(parents=True, exist_ok=True)


def folder(section, kind):
    return PATH_SCEN / SECTIONS[section] / kind


# Panels/ and builds/ variants (environment OUTPUT_VARIANTS=0 skips them,
# for faster runs)
VARIANTS = os.environ.get("OUTPUT_VARIANTS", "1") == "1"
# Only render the figures whose file stem matches this regular expression
# (environment OUTPUT_ONLY; all are still registered for the PDF)
ONLY = os.environ.get("OUTPUT_ONLY")

## LABELS AND COLORS (validated with the dataviz palette checker)

SCEN = json.loads((PATH_SIMUL / "scenarios.json").read_text())
# Display order (user, 2026-11-19): benchmark, public protection, subsidised
# self-protection, subsidised insurance, no risk misperception, combined
SINGLES = ["public_protec", "subsid_protec", "subsid_insur",
           "no_misperception"]
SCENARIOS = [s for s in ["benchmark"] + SINGLES + ["combined"]
             if s in SCEN["scenarios"]]
# Check runs, outside the scenario comparisons
CHECKS = [s for s in ["no_self_protec"] if s in SCEN["scenarios"]]
# Other combinations of two or three levers (main.py, FACTORIAL)
FACTORIAL = [s for s in SCEN["scenarios"] if s not in SCENARIOS + CHECKS]
SCEN_LABELS = {"benchmark": "Benchmark",
               "public_protec": "Public protection",
               "subsid_protec": "Subsidised self-protection",
               "subsid_insur": "Subsidised insurance",
               "no_misperception": "No risk misperception",
               "combined": "All combined",
               "no_self_protec": "No self-protection"}
SCEN_SHORT = {"benchmark": "Bench.", "public_protec": "Public P.",
              "subsid_protec": "Subs. SP", "subsid_insur": "Subs. ins.",
              "no_misperception": "No RM", "combined": "Combined"}
# Benchmark in primary ink, then categorical slots in the display order
# (line styles as secondary encoding: more than three hues overlap)
SCEN_COLORS = {"benchmark": "#0b0b0b", "public_protec": "#2a78d6",
               "subsid_protec": "#eb6834", "subsid_insur": "#1baf7a",
               "no_misperception": "#eda100", "combined": "#e87ba4"}
SCEN_STYLES = {"benchmark": "-", "public_protec": "--",
               "subsid_protec": "-.", "subsid_insur": ":",
               "no_misperception": (0, (5, 1, 1, 1, 1, 1)), "combined": "-"}

HTYPES = ["Formal private", "Informal backyards", "Informal settlements",
          "Formal subsidised"]
HTYPES_SHORT = ["FP", "IB", "IS", "FS"]
HTYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INCGRPS = ["Low income", "Mid-low income", "Mid-high income", "High income"]
INCGRP_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal
FLOOD_LABELS = ["Fluvial", "Pluvial", "Coastal"]
FLOOD_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]
UNPROTECTED = "#b5b3ad"
LEVEL_LABELS = ["Unprotected", "1 level (15 cm)", "2 levels (30 cm)",
                "3 levels (45 cm)"]
LEVEL_COLORS = [UNPROTECTED, "#f09a72", "#d9541f", "#8f3412"]  # ordinal
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LAND, URBAN = "#e4e2da", "#c9c6bb"
# Sequential heat scale, without its palest end (vanishes on the basemap)
HEAT = LinearSegmentedColormap.from_list(
    "heat", plt.get_cmap("YlOrRd")(np.linspace(0.15, 1, 256)))
# Diverging scale centred on 1 for the amenity index
DIVERGING = LinearSegmentedColormap.from_list(
    "red_blue", ["#b02a2a", "#f09a72", "#f0efec", "#86b6ef", "#184f95"])
# Sequential scales without their palest end
GREENS = LinearSegmentedColormap.from_list(
    "greens", plt.get_cmap("Greens")(np.linspace(0.12, 1, 256)))
BLUES = LinearSegmentedColormap.from_list(
    "blues", plt.get_cmap("Blues")(np.linspace(0.15, 1, 256)))
PURPLES = LinearSegmentedColormap.from_list(
    "purples", plt.get_cmap("Purples")(np.linspace(0.15, 1, 256)))

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
# Size of a 500 m cell in degrees (bars cover a share of it, see map3d)
DLON = 500 / (111320 * np.cos(np.deg2rad(LAT.mean())))
DLAT = 500 / 110950
BOUNDS = geo_grid.to_crs("EPSG:4326").total_bounds        # lon/lat extent
CBD_LONLAT = (18.4241, -33.9249)

# Basemap raster on the 500 m lattice of the grid (projected metres): the
# municipality in light gray, cells with more than 5% urban land darker,
# transparent outside; corners converted to longitude / latitude
CELL_COL = np.round((geo_grid["X_MIN"] - geo_grid["X_MIN"].min())
                    / 500).astype(int).to_numpy()
CELL_ROW = np.round((geo_grid["Y_MIN"] - geo_grid["Y_MIN"].min())
                    / 500).astype(int).to_numpy()


def raster_of(rgba_cells):
    """Basemap raster from one RGBA color per grid cell."""
    out = np.zeros((CELL_ROW.max() + 2, CELL_COL.max() + 2, 4))
    out[CELL_ROW, CELL_COL] = rgba_cells
    return out


urban_cell = grid_csv["urban"].to_numpy() > 0.05 * 250000
BASE_CELLS = np.where(urban_cell[:, None], matplotlib.colors.to_rgba(URBAN),
                      matplotlib.colors.to_rgba(LAND))
BASE_RGBA = raster_of(BASE_CELLS)
corner_x, corner_y = np.meshgrid(
    geo_grid["X_MIN"].min() + 500 * np.arange(CELL_COL.max() + 2),
    geo_grid["Y_MIN"].min() + 500 * np.arange(CELL_ROW.max() + 2))
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
    if var == "mask_self_protec" and not path.exists():
        return np.full(N_CELLS, np.nan)              # SP0
    return np.load(path)


VARIABLES = ["households", "dwelling_size", "rent", "housing_supply",
             "utility", "mask_self_protec", "damages_contents",
             "damages_structure", "damages_reimbursed",
             "damages_informal_levels", "ex_post_u", "ex_post_z",
             "ex_post_u_post", "ex_post_z_post", "ex_post_price",
             "net_income", "landlord_revenue", "surplus_damages", "error",
             "solver_cells", "developer_tax", "rdp_vacant"]


def perceived_factor(opts):
    """Share k of actual damages agents perceive (solver.perception)."""
    if not opts["agents_anticipate_floods"]:
        return 0.
    return p["risk_internaliz"] if opts["risk_misperc"] else 1.


def derive(d, label):
    """Add per-scenario variables used by the plots and tables."""
    info = SCEN["scenarios"][label]
    opts = info["options"]
    d["label"], d["opts"], d["view"] = label, opts, "post"
    d["post"] = d
    d["budget"] = info["budget"] or [0, 0, 0]
    d["k"] = perceived_factor(opts)
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
    # household damages (all but formal private structures, which
    # developers bear) and formal private structures (by occupant group)
    d["dmg"] = d["damages_contents"] + d["damages_structure"]
    d["dmg_struct"] = np.zeros_like(d["dmg"])
    d["dmg_struct"][0] = d["damages_structure"][0]
    d["dmg_hh"] = d["dmg"] - d["dmg_struct"]
    # Damages households pay for (net of payouts): no formal private
    # structures (developers); backyard shacks (net of payouts, same share s
    # for the insured groups 1-2) move from their tenants to the RDP owners
    # of the cell, who own them
    s = (info.get("insurance_share") or 0) if opts["subsid_insur"] else 0
    borne = d["dmg"] - d["damages_reimbursed"]
    borne[0] -= d["damages_structure"][0]
    shacks = (1 - s) * d["damages_structure"][1]
    borne[1] -= shacks
    borne[3, 0] += shacks.sum(0)
    d["dmg_borne"] = borne
    # Developers' unanticipated losses, the part of formal structure damages
    # they do not perceive: (1 - k) x actual damages (groups, cells)
    d["dev_loss"] = (1 - d["k"]) * d["damages_structure"][0]
    d["level"] = np.nan_to_num(d["mask_self_protec"]).astype(int)
    # RDP choice (RC1): cells whose RDP units were declined
    d["vacant"] = d["rdp_vacant"] > 0
    return d


def load_scenario(label):
    name = SCEN["scenarios"][label]["name"]
    return derive({var: load_simulation_data(var, name)
                   for var in VARIABLES}, label)


DAMAGE_KEYS = ("dmg", "dmg_hh", "dmg_struct", "dmg_borne",
               "damages_contents", "damages_structure", "damages_reimbursed",
               "damages_informal_levels")


def exante(d):
    """Ex-ante (equilibrium) view of scenario d: the damages agents
    anticipate, k x actual damages (k = 0.36 under risk misperception, 1
    otherwise; insured households perceive k (1 - s) x actual damages, which
    is k x the damages they bear), no unanticipated developer losses, and
    equilibrium instead of ex-post consumption, utilities and prices of the
    composite good (1 + γ k ρ borne, used for equivalent variations)."""
    e = dict(d)
    for key in DAMAGE_KEYS:
        e[key] = d["k"] * d[key]
    e["dev_loss"] = np.zeros_like(d["dev_loss"])
    e["ex_post_u_post"], e["ex_post_z_post"] = d["ex_post_u"], d["ex_post_z"]
    e["ex_post_price"] = 1 + d["k"] * (d["ex_post_price"] - 1)
    e["view"] = "ante"
    return e


SIM = {"post": {s: load_scenario(s) for s in SCENARIOS + CHECKS}}
SIM["ante"] = {s: exante(d) for s, d in SIM["post"].items()}
VIEWS = ("post", "ante")
VIEW_SUFFIX = {"post": "", "ante": "_exante"}
VIEW_CAPTION = {"post": "", "ante": " (ex ante)"}
VIEW_NOTE = {
    "post": "",
    "ante": (" Ex-ante version: damages agents anticipate in equilibrium "
             "(36\\% of actual damages under risk misperception, all of them "
             "without; insured households anticipate the uninsured share), "
             "without developers' unanticipated losses, and equilibrium "
             "utilities instead of utilities after the actual damages are "
             "realised.")}
sim = SIM["post"]
B = sim["benchmark"]
N_TOTAL = B["hh"].sum()
print("  developers' losses (M ZAR), from structure damages vs solver:",
      [(round(sim[s]["dev_loss"].sum() / 1e6, 1),
        round(sim[s]["developer_tax"][1] / 1e6, 1)) for s in SCENARIOS])

## EXPECTED FLOOD DEPTH FROM INPUTS
## Expected annual depth averaged over the cell, E[p_T d_T] (cm), with the
## deepest flood type in each event (same states and probabilities as
## floods.py), before any protection investment (sandbags, public
## protection), without climate change. Per housing type, frequent pluvial
## events are drained (floods.PLUVIAL_SAFE), so that a positive depth means
## positive expected damages before protection.

DEPTH_OPTS = {"coastal": 1, "climate_change": 0}


def expected_depth(htype=None, by_type=False):
    """(cells,) expected depth (cm/year); by_type: (flood types, cells)
    contributions of each flood type (the deepest in each event), which
    add up to the expected depth (expectations are linear in the events)."""
    depth, prop = floods.flood_maps(inputs, DEPTH_OPTS, p)
    event = np.nan_to_num(prop * depth)            # (types, periods, cells)
    if htype is not None:
        safe = np.isin(floods.RETURN_PERIODS, floods.PLUVIAL_SAFE[htype])
        event[floods.FLOOD_TYPES.index("pluvial"), safe] = 0
    proba = floods.state_probabilities(DEPTH_OPTS, p)
    best = event.max(0)
    if not by_type:
        return 100 * floods.expected_damage(proba, floods.state_damages(best))
    deepest = event.argmax(0)
    return np.array([100 * floods.expected_damage(
        proba, floods.state_damages(np.where(deepest == t, best, 0)))
        for t in range(len(event))])


DEPTH_CELL = expected_depth()
DEPTH_FLOOD_TYPES = expected_depth(by_type=True)
assert np.allclose(DEPTH_FLOOD_TYPES.sum(0), DEPTH_CELL)
# By housing type (formal, backyard, informal, subsidized)
DEPTH = np.array([expected_depth(h) for h in
                  ("formal", "backyard", "informal", "subsidized")])
# Exposed households: expected depth above EXPOSURE_THRESHOLD (cm/year): a
# zero threshold counts most households as exposed, most of them at
# negligible depths; 0.1 cm/year keeps about a third of households and
# almost all damages. Most exposed: above MOST_EXPOSED_THRESHOLD (user,
# 2026-11-19: 5 cm/year)
EXPOSURE_THRESHOLD = 0.1
MOST_EXPOSED_THRESHOLD = 5.0
EXPOSED = DEPTH > EXPOSURE_THRESHOLD
MOST_EXPOSED = DEPTH > MOST_EXPOSED_THRESHOLD
SUBSAMPLES = {"all": None, "exposed": EXPOSED, "most": MOST_EXPOSED}
SUBSAMPLE_NAMES = {
    "all": "All households",
    "exposed": f"Exposed households (above {EXPOSURE_THRESHOLD:g} cm/year)",
    "most": ("Most exposed households (above "
             f"{MOST_EXPOSED_THRESHOLD:g} cm/year)")}
MOST_TXT = f"{MOST_EXPOSED_THRESHOLD:g}\\,cm/year"


def share_of_all(d, mask=EXPOSED, types=(0, 1, 2, 3), groups=(0, 1, 2, 3)):
    """% of all households of d in these housing types and groups and in
    mask ((types, cells); None: everywhere)."""
    hh = d["hh"][np.ix_(list(types), list(groups))]
    if mask is not None:
        hh = hh * mask[list(types)][:, None, :]
    return 100 * hh.sum() / d["hh"].sum()


def mean_depth(d, mask=EXPOSED, types=(0, 1, 2, 3), groups=(0, 1, 2, 3)):
    """Household-weighted mean expected depth (cm/year) of the households
    of these types and groups in mask."""
    w = d["hh"][np.ix_(list(types), list(groups))] \
        * mask[list(types)][:, None, :]
    n = w.sum()
    return ((w * DEPTH[list(types)][:, None, :]).sum() / n if n > 0
            else np.nan)


def wquantile(values, weights, q):
    """Weighted quantile (lower: the smallest value whose cumulated weight
    reaches q)."""
    values, weights = np.ravel(values), np.ravel(weights)
    ok = weights > 0
    if not ok.any():
        return np.nan
    order = np.argsort(values[ok])
    v, w = values[ok][order], weights[ok][order]
    cum = np.cumsum(w) / w.sum()
    return v[min(np.searchsorted(cum, q), len(v) - 1)]


def settler_shares(d, mask=EXPOSED):
    """% of settlers in mask protected by sandbags, % of settlers in mask."""
    settlers = d["hh_type"][2]
    n = (settlers * mask[2]).sum()
    protected = (settlers * mask[2] * (d["level"] >= 1)).sum()
    return (100 * protected / n if n else np.nan,
            100 * n / settlers.sum())

###############################################################################

# %% PLOTTING HELPERS (3D MAPS, 2D MAPS, LINES AND BARS)

print("PLOTTING HELPERS")

ELEV, AZIM = 38, -62


def segments(v):
    """Stacked bar segments for values v (categories, items) >= 0:
    (category, bottom, height), each (categories, items)."""
    k, n = v.shape
    cat = np.repeat(np.arange(k)[:, None], n, 1)
    return cat, np.cumsum(v, 0) - v, v


def bar_top(values):
    v = np.clip(np.nan_to_num(np.atleast_2d(values)), 0, None)
    return v.sum(0).max(initial=0)


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
          legend=True, legend_title=None, level=False, heat=None, norm=None,
          cmap=None, lon=None, lat=None, base=None, footprint=None):
    """3D bars over the basemap, for values (categories, items) at the
    cells (or at lon, lat), stacked. level: thinner, more transparent bars
    seen from higher up, for level variables whose bars form walls. heat:
    (items,) values coloring single bars with cmap (default HEAT) and norm
    (values beyond its ends take its end colors). zlim: bars above it are
    cut at it. base: basemap raster. All segments go into one bar3d call,
    so that matplotlib depth-sorts their faces; the CBD and the legend
    (white box) are drawn on top."""
    lon = LON if lon is None else np.asarray(lon)
    lat = LAT if lat is None else np.asarray(lat)
    v = np.clip(np.nan_to_num(np.atleast_2d(
        np.asarray(values, dtype=float))), 0, None)
    if zlim is not None:                    # cut stacks at the top
        top = np.cumsum(v, 0)
        v = np.diff(np.minimum(np.vstack([np.zeros(v.shape[1]), top]),
                               zlim[1]), axis=0)
    keep = np.any(v > 0, 0)
    lon, lat, v = lon[keep], lat[keep], v[:, keep]
    alpha_bar, size = (0.75, 0.55) if level else (0.9, 0.9)
    if footprint:
        size = footprint
    if heat is not None:
        alpha_bar = 0.9
    add_basemap(ax, base, zorder=2)
    cat, bottom, height = segments(v)
    m = height > 0
    if m.any():
        cells = np.nonzero(m)[1]
        if heat is None:
            rgba = np.array([matplotlib.colors.to_rgba(c, alpha_bar)
                             for c in colors])[cat[m]]
        else:
            h = np.nan_to_num(np.asarray(heat, dtype=float)[keep][cells])
            rgba = (cmap or HEAT)(norm(np.clip(h, norm.vmin, norm.vmax)),
                                  alpha=alpha_bar)
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
                   height=0.012, gap=0.035, extend="neither"):
    """Horizontal colorbar centred under axes (call after the layout)."""
    boxes = [ax.get_position() for ax in np.atleast_1d(axes)]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    cax = fig.add_axes((x0 + (1 - width) / 2 * (x1 - x0), y0 - gap,
                        width * (x1 - x0), height))
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap or HEAT),
                       cax=cax, orientation="horizontal", extend=extend)
    bar.set_label(label, fontsize=8, color=INK2, labelpad=4)
    bar.ax.tick_params(labelsize=7)
    bar.ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    bar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    bar.outline.set_edgecolor(GRID)
    return bar


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


def line_handles(colors, labels, styles=None):
    styles = styles or ["-"] * len(colors)
    return [Line2D([], [], color=c, linestyle=s, label=l, **DIAMOND)
            for c, l, s in zip(colors, labels, styles)]


def bars2d(ax, x, widths, values, colors):
    """2D stacked bars at x for values (categories, items)."""
    v = np.clip(np.nan_to_num(np.atleast_2d(values)), 0, None)
    bottom = np.cumsum(v, 0) - v
    for c in range(v.shape[0]):
        m = v[c] > 0
        if m.any():
            ax.bar(np.asarray(x)[m], v[c][m], width=np.asarray(widths)[m],
                   bottom=bottom[c][m], color=colors[c], edgecolor="white",
                   linewidth=0.3)


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


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def grid3d(n):
    """Figure with n 3D panels: one, 1 x 3, 2 x 2 or 2 x 3, tightly packed."""
    shape = {1: (1, 1), 3: (1, 3), 4: (2, 2), 6: (2, 3)}[n]
    size = {1: (8.6, 6.6), 3: (19, 6.2), 4: (13, 11.6), 6: (18, 10.4)}[n]
    fig = plt.figure(figsize=size)
    axes = [new3d(fig, *shape, k + 1) for k in range(n)]
    # Room on the right of each panel for its z label
    fig.subplots_adjust(left=0, right=0.87 if n > 1 else 0.9, top=1,
                        bottom=0.07 if n > 1 else 0,
                        wspace=0.12 if n > 1 else 0, hspace=0.08)
    return fig, axes


def grid2d_maps(n):
    """Figure with n 2D map panels (1, 3, 4 or 6)."""
    shape = {1: (1, 1), 3: (1, 3), 4: (2, 2), 6: (2, 3)}[n]
    size = {1: (6.6, 6.6), 3: (17, 6.4), 4: (11.5, 12.4),
            6: (17, 12.4)}[n]
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False)
    return fig, list(axes.flat)


def dist_figure(n):
    """Figure with n 2D panels: 1 x n up to 3, 2 x 2 for 4."""
    nrow, ncol = (2, 2) if n == 4 else (1, n)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(11 if n > 1 else 6.4, 3.7 * nrow),
                             squeeze=False)
    return fig, list(axes.flat)


def legend_above(fig, handles, title, ncol=None):
    """Figure legend in rows above the panels, with room reserved for it
    (lays the figure out again)."""
    ncol = ncol or len(handles)
    rows = -(-len(handles) // ncol)
    height = (0.3 + 0.22 * rows) / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - height), h_pad=1.2, w_pad=1.2)
    fig.legend(handles=handles, loc="upper left", title=title, ncol=ncol,
               fontsize=8, bbox_to_anchor=(0.005, 1.0), alignment="left")

## SAVING AND THE PDF REGISTRY
## Each output of the PDF is registered with its section, subsection,
## position (the PDF follows the user's order of 2026-11-19 within each
## section, whatever the order of the code), caption and note.

REGISTRY = []


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def save_fig(fig, path):
    # The tight bounding box ignores the z labels of 3D axes, which stick
    # out on the right: an invisible anchor keeps them in the image
    boxes = [ax.get_position().x1 for ax in fig.axes if ax.name == "3d"]
    if boxes:
        fig.text(min(max(boxes) + 0.1, 1.0), 0.5, ".", alpha=0)
    # OneDrive briefly locks files it is syncing (OSError 22): retry
    for attempt in range(5):
        try:
            fig.savefig(path)
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(3)
    plt.close(fig)


def register(section, kind, stem, order, sub, caption, note, width=None):
    REGISTRY.append(dict(section=section, kind=kind, stem=stem, order=order,
                         sub=sub, caption=caption, note=note, width=width))


def figure_set(stem, section, n, make, draw, finish=None, names=None,
               builds=(), doc=None):
    """A figure, its panels and its builds.

    make(n) -> (fig, axes) lays out n panels; draw(ax, k, vis) draws panel
    k, where vis maps a build key to the number of visible items (absent:
    all); finish(fig, axes, panels, vis) adds legends and colorbars. Saves
    the full figure to <section>/figures (and registers it for the PDF if
    doc = (order, subsection, caption, note[, width])), each panel alone to
    <section>/panels if n > 1 (names: panel labels), and for each build
    (key, item names, first count) the figures with the first 0, 1, ...
    items visible to <section>/builds."""

    def render(panels, vis, path):
        fig, axes = make(len(panels))
        for ax, k in zip(axes, panels):
            draw(ax, k, vis)
        if finish:
            finish(fig, axes, panels, vis)
        save_fig(fig, path)

    if ONLY and not re.search(ONLY, stem):
        if doc:
            register(section, "figure", stem, *doc)
        return
    render(list(range(n)), {}, folder(section, "figures") / f"{stem}.png")
    print(f"  {SECTIONS[section]}/figures/{stem}.png "
          f"({time.time() - T0:.0f} s)")
    if doc:
        register(section, "figure", stem, *doc)
    if not VARIANTS:
        return
    if n > 1:
        for k in range(n):
            render([k], {}, folder(section, "panels")
                   / f"{stem}__{slug(names[k]) if names else k}.png")
    for key, items, first in builds:
        for j in range(first, len(items) + 1):
            last = slug(items[j - 1]) if j else "none"
            render(list(range(n)), {key: j}, folder(section, "builds")
                   / f"{stem}__{key}{j}_{last}.png")

###############################################################################

# %% TABLE HELPERS AND TABLES 1-2 (DATA)

print("TABLES 1-2: DATA")


def write_table(df, section, stem, fmt, doc=None, resize=True):
    """Booktabs table (.tex) and Excel copy in <section>/tables; doc =
    (order, subsection, caption, note) registers it for the PDF."""
    path = folder(section, "tables")
    df.to_excel(path / f"{stem}.xlsx", float_format="%.4f")
    latex_table(df, path / f"{stem}.tex", fmt)
    if doc:
        register(section, "table", stem, *doc[:4],
                 width=doc[4] if len(doc) > 4 else resize)
    print(f"  {SECTIONS[section]}/tables/{stem}.tex")


def esc(s):
    return (str(s).replace("%", r"\%").replace("&", r"\&")
            .replace("²", r"\textsuperscript{2}")
            .replace("ȳ", r"$\bar{y}$").replace(">", r"$>$"))


def latex_table(df, path, fmt):
    """Write a booktabs tabular: df has a (block, row) MultiIndex, and
    fmt(block, row, column, value) gives the cell text."""
    cols = list(df.columns)
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


def _num(v, digits=1, sign=False, dash_zero=False):
    """Number with `digits` decimals; values that would round to zero keep
    two significant digits (so that small positive damages do not show as
    0.0); exact zeros as dashes if dash_zero; NaN as dashes."""
    if v is None or isinstance(v, str):
        return v or "--"
    if not np.isfinite(v) or (dash_zero and v == 0):
        return "--"
    if round(v, digits) == 0 and v > 0 and not sign:
        return f"<{10.0 ** -digits:.{digits}f}"
    if round(v, digits) == 0 and v != 0:
        digits = min(int(-np.floor(np.log10(abs(v)))) + 1, 8)
    elif round(v, digits) == 0:
        v = 0.
    return f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"


## TABLE: INCOME GROUPS (DATA)

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
table_income = pd.DataFrame(rows).T
write_table(table_income, "appendix", "table_income_groups",
            lambda b, r, c, v: v if isinstance(v, str)
            else _num(v, 0 if "ZAR" in c else 1),
            doc=(10, "Data and model", "Income groups",
                 "2011 Census: annual household income brackets grouped "
                 "into the four model income groups (households without "
                 "income excluded). Mean income: household-weighted mean of "
                 "the bracket medians. Employment rate: exogenous annual "
                 "employment rate of each group (half the number of employed "
                 "workers per two-worker household, calibrated from "
                 "educational attainment). Population share: share of "
                 "households with income in each group."))

## TABLE: MODEL ASSUMPTIONS BY HOUSING TYPE

totals = np.array([config.TOTAL_FORMAL, config.TOTAL_RDP,
                   config.TOTAL_BACKYARD, config.TOTAL_INFORMAL])
table_assumptions = pd.DataFrame({
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
write_table(table_assumptions, "appendix", "table_model_assumptions",
            lambda b, r, c, v: v,
            doc=(20, "Data and model", "Model assumptions by housing type",
                 "Eligible groups: income groups with access to each housing "
                 "type (1: poorest). Location: endogenous within the areas "
                 "available for each type (backyards of RDP premises; "
                 "informal settlement areas tolerated in 2011). RDP units "
                 "are located exogenously, but low-income households may "
                 "decline them when their utility there, without renting out "
                 "their backyard, is below that of the rest of their group. "
                 "Dwelling size: formal private dwellings of at least the "
                 "minimum lot size; RDP house and backyard; shacks. Price: "
                 "annual rents per m\\textsuperscript{2} (RDP housing is "
                 "free). Population share: 2011 households by housing type "
                 "(SAL data)."))

###############################################################################

# %% INPUT MAPS (JOBS OVER POPULATION, LAND, AMENITIES, NET INCOMES)

print("INPUT MAPS")

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
pts = gpd.GeoSeries(gpd.points_from_xy(taz.X[centres], taz.Y[centres]),
                    crs=geo_grid.crs).to_crs("EPSG:4326")
JOB_LON, JOB_LAT = pts.x.to_numpy(), pts.y.to_numpy()
N_CENTRES = int(centres.sum())
print(f"  {N_CENTRES} job centres")


def make3d_bottom(n, bottom=0.1):
    """3D grid with room for a colorbar below (single panels)."""
    fig, axes = grid3d(n)
    if n == 1:
        fig.subplots_adjust(bottom=bottom)
    return fig, axes


## Jobs by income group over the number of households in the benchmark
## (heat map at z = 0)

HH_CELL = B["hh"].sum((0, 1))
HH_NORM = Normalize(0, np.percentile(HH_CELL[HH_CELL > 1], 99))
_cells = BASE_CELLS.copy()
_live = HH_CELL > 1
_cells[_live] = HEAT(HH_NORM(np.minimum(HH_CELL[_live], HH_NORM.vmax)))
BASE_HH = raster_of(_cells)
JOBS_TOP = bar_top(jobs)


def draw_jobs(ax, k, vis):
    v = jobs.copy()
    v[vis.get("groups", 4):] = 0
    map3d(ax, v, INCGRP_COLORS, "Jobs (households)", zlim=(0, JOBS_TOP),
          labels=INCGRPS, legend_title="Jobs by income group", lon=JOB_LON,
          lat=JOB_LAT, base=BASE_HH, footprint=1.4)


figure_set("map_jobs_by_income_group_over_households", "appendix", 1,
           make3d_bottom, draw_jobs,
           finish=lambda fig, axes, panels, vis: colorbar_below(
               fig, axes, HH_NORM, "Number of households in the benchmark "
               "(basemap)", extend="max", gap=0.02),
           builds=[("groups", INCGRPS, 0)],
           doc=(30, "Data and model",
                "Jobs by income group overlaid on residential population",
                f"Bars: jobs by income group in each of the {N_CENTRES} job "
                "centres (transport zones with more than 2,500 jobs; 2013 "
                "transport-zone data rescaled to the number of households "
                "of each group), stacked. Basemap: number of households per "
                "500\\,m grid cell in the benchmark equilibrium (colors stop "
                "at the 99th percentile); light gray: other cells of the "
                "municipality, darker gray: urban footprint."))

## Land available for each housing type (share of the cell area)

LAND_SHARE = 100 * inputs["coeff_land"][:3]


def draw_land(ax, k, vis):
    v = np.where(LAND_SHARE[k] > 0, LAND_SHARE[k], np.nan)
    norm = Normalize(0, np.nanmax(v))
    mesh = map2d(ax, v, norm, BLUES)
    ax.set_title(HTYPES[k], pad=3)
    colorbar_right(ax.figure, ax, mesh, "Available land (% of cell area)")


def finish_maps(fig, axes, panels, vis):
    fig.tight_layout(h_pad=1.5, w_pad=1.5)


figure_set("map_land_availability_by_housing_type", "appendix", 3,
           grid2d_maps, draw_land, finish_maps, names=HTYPES[:3],
           doc=(35, "Data and model",
                "Land available by housing type",
                "Share of the area of each 500\\,m grid cell available for "
                "formal private housing, informal backyards (in RDP "
                "premises) and informal settlements (areas tolerated in "
                "2011); cells without available land not shown. Basemap: "
                "City of Cape Town (light) and urban footprint (dark)."))

## Amenity index

amen = inputs["amenities"]
AMEN_NORM = TwoSlopeNorm(1, np.nanpercentile(amen, 1),
                         np.nanpercentile(amen, 99))


def draw_amenities(ax, k, vis):
    mesh = map2d(ax, amen, AMEN_NORM, DIVERGING)
    colorbar_right(ax.figure, ax, mesh, "Amenity index (mean 1)",
                   extend="both", fmt=lambda v, _: f"{v:.2f}")


figure_set("map_amenity_index", "appendix", 1, grid2d_maps, draw_amenities,
           finish_maps,
           doc=(40, "Data and model", "Amenity index",
                "Calibrated amenity index (mean 1, diverging scale centred "
                "on 1; colors stop at the 1st and 99th percentiles)."))

## Income net of commuting costs, by income group (ZAR/year)

LIVED = HH_CELL > 1
NET_INCOME = inputs["net_income"]


def draw_net_income(ax, k, vis):
    v = np.where(LIVED & np.isfinite(NET_INCOME[k]), NET_INCOME[k], np.nan)
    norm = Normalize(np.nanpercentile(v, 1), np.nanpercentile(v, 99))
    mesh = map2d(ax, v, norm, GREENS)
    ax.set_title(INCGRPS[k], pad=3)
    colorbar_right(ax.figure, ax, mesh,
                   "Income net of commuting costs (ZAR/year)", extend="both")


figure_set("map_income_net_of_commuting_by_income_group", "appendix", 4,
           grid2d_maps, draw_net_income, finish_maps, names=INCGRPS,
           doc=(41, "Data and model",
                "Income net of commuting costs by income group",
                "Expected annual income net of commuting costs of each "
                "income group (transport model, ZAR/year), in each cell "
                "where households live in the benchmark; one panel and "
                "color scale per income group (colors stop at the 1st and "
                "99th percentiles)."))

###############################################################################

# %% MAP VARIABLES AND SCALES

print("MAP VARIABLES")

SOLVER_CELLS = B["solver_cells"] > 0


def map_values(d, kind):
    """Bar heights (categories, cells) of count map `kind` for scenario d."""
    if kind == "households_type":
        return d["hh_type"]
    if kind == "households_group":
        return d["hh_group"]
    if kind == "damages_type":                        # M rands/year
        return d["dmg"].sum(1) / 1e6
    if kind == "protection":                          # households
        return np.array([d["hh_type"][2] * (d["level"] == k)
                         for k in range(4)])
    if kind == "depth":                               # cm/year
        return np.where(SOLVER_CELLS, DEPTH_FLOOD_TYPES, 0)
    raise ValueError(kind)


MAPS = {
    # kind: (z label, colors, labels, legend title, build key)
    "households_type": ("Number of households", HTYPE_COLORS, HTYPES,
                        "Housing type", "types"),
    "households_group": ("Number of households", INCGRP_COLORS, INCGRPS,
                         "Income group", "groups"),
    "damages_type": ("Expected damages (M ZAR/year)", HTYPE_COLORS, HTYPES,
                     "Housing type", "types"),
    "protection": ("Informal settlement households", LEVEL_COLORS,
                   LEVEL_LABELS, "Sandbag levels", None),
    "depth": ("Expected flood depth (cm/year)", FLOOD_COLORS, FLOOD_LABELS,
              "Flood type", None),
}


def count_map_set(stem, section, kind, scen, view="post", doc=None):
    """Count maps (stacked bars) for the scenarios scen (one panel each),
    with builds over the categories (housing types, income groups)."""
    zlabel, colors, labels, title, key = MAPS[kind]
    S = SIM[view]
    values = {s: map_values(S[s], kind) for s in scen}
    top = max(bar_top(v) for v in values.values())

    def draw(ax, k, vis):
        v = values[scen[k]].copy()
        if key:
            v[vis.get(key, len(v)):] = 0
        map3d(ax, v, colors, zlabel, zlim=(0, top), legend=False)
        if len(scen) > 1:
            ax.set_title(SCEN_LABELS[scen[k]], pad=-4)

    def finish(fig, axes, panels, vis):
        handles = legend_handles(colors, labels)
        if len(axes) == 1:
            leg = axes[0].legend(handles=handles, loc="upper left",
                                 fontsize=7, handlelength=1, title=title,
                                 **WHITE_LEGEND)
            leg.set_zorder(100)
        else:
            fig.legend(handles=handles, loc="lower center",
                       ncol=len(labels), fontsize=9, title=title,
                       bbox_to_anchor=(0.5, 0.0), **WHITE_LEGEND)

    figure_set(stem + VIEW_SUFFIX[view], section, len(scen),
               lambda n: grid3d(n), draw, finish,
               names=[SCEN_LABELS[s] for s in scen],
               builds=[(key, labels, 0)] if key else (), doc=doc)


## LEVEL MAPS: one panel per housing type (or scenario), bar height = a
## level variable, bar color = another one on a heat scale. Linear scales
## (user, 2026-11-19), specific to each housing type, the same in all
## scenarios (one set of scales per view).
##   damages: height = expected damages in the cell (thousand ZAR/year),
##            color = damages paid per household (% of mean income);
##   burden:  height = damages paid per household (% of mean income), no
##            color scale (bars cut at the 99.5th percentile);
##   rent:    2D heat map of the rent per m2 (no bars).

LEVEL_LABEL = {
    "damages": ("Expected damages (thousand ZAR/year)",
                "Damages paid per household (% of mean income)"),
    "burden": ("Damages paid per household (% of mean income)", None),
    "rent": (None, "Annual rent (ZAR/m²)"),
}


def level_values(d, kind):
    """(bar heights, heat values), each (housing types, cells), zero where
    the housing type has no households."""
    hh = d["hh_type"]
    occ_income = (d["hh"] * y_bar[None, :, None]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        burden = np.nan_to_num(100 * d["dmg_borne"].sum(1) / occ_income)
    ok = hh > 0
    if kind == "damages":
        return np.where(ok, d["dmg"].sum(1) / 1e3, 0), np.where(ok, burden, 0)
    if kind == "burden":
        return np.where(ok, burden, 0), None
    if kind == "rent":
        return None, np.where(ok & (d["rent_clean"] > 0), d["rent_clean"], 0)
    raise ValueError(kind)


# Scales by (view, kind, housing type): z limits, color norm, extend
LEVEL_ZLIM, LEVEL_NORM, LEVEL_EXTEND = {}, {}, {}
for view in VIEWS:
    for kind in ("damages", "burden", "rent"):
        if kind == "rent" and view == "ante":
            continue
        vals = [level_values(SIM[view][s], kind) for s in SCENARIOS]
        for t in range(4):
            w = np.concatenate([SIM[view][s]["hh_type"][t] for s in SCENARIOS])
            if kind != "rent":
                h = np.concatenate([v[0][t] for v in vals])
                top = (wquantile(h, w * (h > 0), 0.995) if kind == "burden"
                       else h.max())
                LEVEL_ZLIM[view, kind, t] = (0, max(top, 1e-9))
            if kind != "burden":
                c = np.concatenate([v[1][t] for v in vals])
                if not (c > 0).any():
                    continue
                vmax = wquantile(c, w * (c > 0), 0.99)
                vmin = 0 if kind == "damages" else c[c > 0].min()
                LEVEL_NORM[view, kind, t] = Normalize(vmin, vmax)
                LEVEL_EXTEND[view, kind, t] = ("max" if c.max() > vmax
                                               else "neither")


def level_map_set(stem, section, kind, panels, view="post", doc=None):
    """3D level maps for panels = [(scenario, housing type)]; one colorbar
    below each panel if their scales differ, else one below all."""
    S = SIM[view]
    zlabel, clabel = LEVEL_LABEL[kind]
    vals = {s: level_values(S[s], kind) for s in {s for s, t in panels}}
    by_type = len({t for s, t in panels}) > 1

    def draw(ax, k, vis):
        s, t = panels[k]
        heights, heat = vals[s]
        map3d(ax, heights[t], [HTYPE_COLORS[t]], zlabel,
              zlim=LEVEL_ZLIM[view, kind, t], legend=False, level=True,
              heat=None if heat is None else heat[t],
              norm=LEVEL_NORM.get((view, kind, t)))
        ax.set_title(HTYPES[t] if by_type else SCEN_LABELS[s], pad=-4)

    def finish(fig, axes, idx, vis):
        if len(axes) == 1:
            fig.subplots_adjust(bottom=0.1)
        if kind == "burden":
            return
        keys = [(view, kind, panels[k][1]) for k in idx]
        if len(set(keys)) == 1:
            colorbar_below(fig, axes, LEVEL_NORM[keys[0]], clabel,
                           extend=LEVEL_EXTEND[keys[0]],
                           gap=0.02 if len(axes) == 1 else 0.035)
        else:
            for ax, key in zip(axes, keys):
                colorbar_below(fig, ax, LEVEL_NORM[key], clabel, width=0.7,
                               extend=LEVEL_EXTEND[key], gap=0.015)

    def make(n):
        fig, axes = grid3d(n)
        if n > 1 and by_type and kind != "burden":
            # Room for a colorbar under each panel, and for the z labels
            fig.set_size_inches(13.5, 13)
            fig.subplots_adjust(right=0.83, wspace=0.16, hspace=0.24,
                                bottom=0.07)
        elif n > 1:
            fig.subplots_adjust(right=0.87, wspace=0.12)
        return fig, axes

    figure_set(stem + VIEW_SUFFIX[view], section, len(panels), make, draw,
               finish, names=[HTYPES[t] if by_type else SCEN_LABELS[s]
                              for s, t in panels], doc=doc)


def rent_map_set(stem, section, panels, doc=None):
    """2D rent heat maps for panels = [(scenario, housing type)], one color
    scale per housing type (the same in all scenarios)."""
    by_type = len({t for s, t in panels}) > 1

    def draw(ax, k, vis):
        s, t = panels[k]
        heat = level_values(sim[s], "rent")[1][t]
        mesh = map2d(ax, np.where(heat > 0, heat, np.nan),
                     LEVEL_NORM["post", "rent", t], HEAT)
        ax.set_title(HTYPES[t] if by_type else SCEN_LABELS[s], pad=3)
        colorbar_right(ax.figure, ax, mesh, LEVEL_LABEL["rent"][1],
                       extend=LEVEL_EXTEND["post", "rent", t])

    figure_set(stem, section, len(panels), grid2d_maps, draw, finish_maps,
               names=[HTYPES[t] if by_type else SCEN_LABELS[s]
                      for s, t in panels], doc=doc)

###############################################################################

# %% BENCHMARK MAPS

print("BENCHMARK MAPS")

BASEMAP_NOTE = (" Basemap: City of Cape Town (light) and urban footprint "
                "(dark).")
BENCH_NOTE = ("Benchmark: risk misperception (households and developers "
              "perceive 36\\% of actual flood risks), sandbags at full cost "
              "(up to three levels of 15\\,cm, 50 ZAR per level and year), "
              "RDP choice.")
DAMAGE_PAID_NOTE = (
    "Damages paid: damages households pay for, net of insurance payouts "
    "(formal private: contents only, structures are borne by developers; "
    "informal backyards: contents only, shacks are paid by the RDP owners "
    "of the cell; formal subsidised: houses, contents and backyard "
    "shacks), as a \\% of the occupants' mean income.")

for kind, stem in (("households_type", "map_households_by_housing_type"),
                   ("households_group", "map_households_by_income_group")):
    count_map_set(stem, "main", kind, ["benchmark"])
register("main", "figure", None, 10, "Benchmark equilibrium",
         "Households in the benchmark",
         "Number of households per 500\\,m grid cell, stacked by housing "
         "type (a) or income group (b). " + BENCH_NOTE + BASEMAP_NOTE)
REGISTRY[-1]["label"] = "map_households"
REGISTRY[-1]["parts"] = [("map_households_by_housing_type",
                          "By housing type"),
                         ("map_households_by_income_group",
                          "By income group")]

count_map_set("map_expected_flood_depth_by_flood_type", "main", "depth",
              ["benchmark"])
for view in VIEWS:
    count_map_set("map_damages_by_housing_type", "main", "damages_type",
                  ["benchmark"], view)
register("main", "figure", None, 20, "Benchmark equilibrium",
         "Expected flood depth and damages in the benchmark",
         "(a) Expected annual flood depth averaged over each cell (cm/year, "
         "before any protection investment, without climate change), from "
         "the FATHOM fluvial and pluvial and DELTARES coastal maps, stacked "
         "by flood type: in each event, the depth of the deepest flood type "
         "counts towards that type, so that the stack is the expected depth "
         "of the cell; solver cells. (b) Actual expected damages to "
         "structures and contents (M ZAR/year), stacked by housing type; "
         "formal structures are attributed to their occupants."
         + BASEMAP_NOTE)
REGISTRY[-1]["label"] = "map_damages_by_housing_type"
REGISTRY[-1]["parts"] = [("map_expected_flood_depth_by_flood_type",
                          "Expected flood depth"),
                         ("map_damages_by_housing_type",
                          "Expected damages by housing type")]
register("main", "figure", "map_damages_by_housing_type_exante", 21,
         "Benchmark equilibrium",
         "Expected damages by housing type in the benchmark (ex ante)",
         "As Figure~\\ref{fig:map_damages_by_housing_type}(b), with the "
         "damages agents anticipate." + VIEW_NOTE["ante"], width=0.6)

for view, order in zip(VIEWS, (30, 31)):
    level_map_set("map_damages_and_damages_paid_by_housing_type", "main",
                  "damages", [("benchmark", t) for t in range(4)], view,
                  doc=(order, "Benchmark equilibrium",
                       "Expected damages and damages paid per household in "
                       "the benchmark" + VIEW_CAPTION[view],
                       "One panel per housing type. Bar height: actual "
                       "expected damages to structures and contents in the "
                       "cell (thousand ZAR/year). Color: damages paid per "
                       "household. " + DAMAGE_PAID_NOTE + " Linear scales "
                       "specific to each panel, the same in all scenarios "
                       "(colors stop at the 99th percentile of households, "
                       "over all scenarios)." + VIEW_NOTE[view], 0.92))
    level_map_set("map_damages_paid_by_housing_type", "appendix", "burden",
                  [("benchmark", t) for t in range(4)], view,
                  doc=(55 + (view == "ante"), "Benchmark equilibrium",
                       "Damages paid per household in the benchmark"
                       + VIEW_CAPTION[view],
                       "One panel per housing type. Bar height: damages "
                       "paid per household. " + DAMAGE_PAID_NOTE
                       + " Linear scales specific to each panel; bars are "
                       "cut at the 99.5th percentile of households (over "
                       "all scenarios)." + VIEW_NOTE[view], 0.92))

rent_map_set("map_rents_by_housing_type", "appendix",
             [("benchmark", t) for t in range(3)],
             doc=(50, "Benchmark equilibrium", "Rents in the benchmark",
                  "One panel per rented housing type: annual rent per "
                  "m\\textsuperscript{2} in each cell with households of "
                  "the type. Linear color scales specific to each panel, "
                  "the same in all scenarios (colors stop at the 99th "
                  "percentile of households over all scenarios)."
                  + BASEMAP_NOTE, 1.0))

count_map_set("map_self_protection", "appendix", "protection", ["benchmark"],
              doc=(60, "Benchmark equilibrium",
                   "Self-protection of informal settlers in the benchmark",
                   "Number of informal settlement households by sandbag "
                   "levels chosen (gray: unprotected)." + BASEMAP_NOTE,
                   0.6))

###############################################################################

# %% SCENARIO MAPS

print("SCENARIO MAPS")

SCEN_ORDER_NOTE = ("Scenarios: benchmark, public protection of every "
                   "dwelling up to 45\\,cm (Public P.), free sandbags for "
                   "settlers (Subs.\\ SP), subsidised insurance of the two "
                   "poorest groups (Subs.\\ ins.), no risk misperception "
                   "(No RM), and all four combined.")

for view, order in zip(VIEWS, (50, 51)):
    count_map_set("scenarios_map_damages_by_housing_type", "online",
                  "damages_type", SCENARIOS, view,
                  doc=(order, "Policy scenarios",
                       "Expected damages by housing type, by scenario"
                       + VIEW_CAPTION[view],
                       "As Figure~\\ref{fig:map_damages_by_housing_type}(b) "
                       "in each scenario, with the same scale in all panels. "
                       + SCEN_ORDER_NOTE + VIEW_NOTE[view], 1.0))
count_map_set("scenarios_map_self_protection", "online", "protection",
              SCENARIOS,
              doc=(60, "Policy scenarios",
                   "Self-protection of informal settlers, by scenario",
                   "As Figure~\\ref{fig:map_self_protection} in each "
                   "scenario, with the same scale in all panels.", 1.0))
for t in range(3):
    rent_map_set(f"scenarios_map_rents_{slug(HTYPES[t])}", "online",
                 [(s, t) for s in SCENARIOS],
                 doc=(70 + t, "Policy scenarios",
                      f"Rents by scenario: {HTYPES[t].lower()}",
                      f"As the {HTYPES[t].lower()} panel of "
                      "Figure~\\ref{fig:map_rents_by_housing_type} in each "
                      "scenario, with the same color scale.", 1.0))
for t in range(4):
    for view in VIEWS:
        level_map_set(f"scenarios_map_damages_{slug(HTYPES[t])}", "online",
                      "damages", [(s, t) for s in SCENARIOS], view,
                      doc=(80 + 2 * t + (view == "ante"), "Policy scenarios",
                           "Expected damages and damages paid per household "
                           f"by scenario: {HTYPES[t].lower()}"
                           + VIEW_CAPTION[view],
                           f"As the {HTYPES[t].lower()} panel of Figure~"
                           "\\ref{fig:map_damages_and_damages_paid_by_"
                           f"housing_type{VIEW_SUFFIX[view]}}} in each "
                           "scenario, with the same scales."
                           + (" Backyard shacks are paid by the RDP owners "
                              "of the cell." if t == 1 else "")
                           + VIEW_NOTE[view], 1.0))

###############################################################################

# %% DISTRIBUTIONS ACROSS FLOOD DEPTH AND RENTS

print("DISTRIBUTIONS")

BINS = np.logspace(np.log10(EXPOSURE_THRESHOLD),
                   np.log10(DEPTH.max() * 1.0001), 33)
CENTERS = np.sqrt(BINS[:-1] * BINS[1:])
WIDTHS = np.diff(BINS) * 0.92
DEPTH_XLABEL = "Expected flood depth (cm/year, log scale)"
# Low-income households live in these housing types (no formal private)
LOW_TYPES = [t for t in range(4) if B["hh"][t, 0].sum() > 0]


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
    # kind: (groups, colors, labels, legend title, style, y label)
    # style: "stacked" bars (same y axis in all panels), or "lines"
    # (connected diamonds, linear y axes specific to each panel)
    "households": ([0, 1, 2, 3], HTYPE_COLORS, HTYPES, "Housing type",
                   "stacked", "Households (thousands)"),
    "damages": ([0, 1, 2, 3], HTYPE_COLORS, HTYPES, "Housing type",
                "lines", "Damages (M ZAR/year)"),
    "damages_relative": ([0, 1, 2, 3], HTYPE_COLORS, HTYPES, "Housing type",
                         "lines",
                         "Damages paid per household\n(% of mean income)"),
    "protection": ([0], LEVEL_COLORS, LEVEL_LABELS, "Sandbag levels",
                   "stacked", "Exposed settlement households"),
}
# Panels with a log y axis (user, 2026-11-19: low-income damages paid)
LOG_PANELS = {"damages_relative": {0}}


def log_label(label):
    """'Y (unit)' -> 'Y (unit, log scale)'."""
    return (label[:-1] + ", log scale)" if label.endswith(")")
            else label + " (log scale)")


def set_ylim(ax, values, log=False, shared_top=None):
    """Linear y axis from 0 to the highest finite value (or shared_top),
    or log axis over the positive values."""
    v = np.concatenate([np.ravel(x) for x in values])
    v = v[np.isfinite(v)]
    if log:
        log_axis(ax)
        v = v[v > 0]
        if len(v):
            ax.set_ylim(v.min() / 1.6, v.max() * 1.6)
        return
    top = shared_top if shared_top is not None else (v.max() if len(v)
                                                     else 1)
    ax.set_ylim(0, 1.06 * top if top > 0 else 1)


def style_hist(ax, ylabel, label=None, bins=None, xlabel=DEPTH_XLABEL,
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


def bench_dist_set(stem, section, kind, d, view="post", doc=None):
    """Benchmark distribution across depth, one panel per income group,
    stacked bars or lines by category, with builds over housing types."""
    groups, colors, labels, title, style, ylabel = DISTS[kind]
    values = {i: dist_values(d, kind, i) for i in groups}
    top = max(bar_top(v) for v in values.values())
    log_panels = LOG_PANELS.get(kind, set())
    ncat = len(labels)

    def draw(ax, k, vis):
        i = groups[k]
        j = vis.get("types", ncat)
        if style == "stacked":
            v = values[i].copy()
            v[j:] = 0
            bars2d(ax, CENTERS, WIDTHS, v, colors)
            ax.set_ylim(0, 1.15 * top)
        else:
            log = i in log_panels
            for t in range(j):
                connected(ax, CENTERS, positive(values[i][t]) if log
                          else values[i][t], color=colors[t])
            set_ylim(ax, [values[i]], log)
        style_hist(ax, log_label(ylabel) if (i in log_panels
                                             and style == "lines") else ylabel,
                   INCGRPS[i] if len(groups) > 1 else None)
        if kind == "households":
            ax.text(0.02, 0.97,
                    f"Exposed: {share_of_all(d, groups=[i]) / share_of_all(d, None, groups=[i]) * 100:.1f}% of "
                    f"{d['hh_group'][i].sum():,.0f} households",
                    transform=ax.transAxes, va="top", fontsize=8, color=INK2,
                    bbox=BOX)

    def finish(fig, axes, panels, vis):
        handles = legend_handles(colors, labels) if style == "stacked" \
            else line_handles(colors, labels)
        if kind == "protection":
            fig.tight_layout()
            protected, exposed = settler_shares(d)
            ax = axes[0]
            ax.legend(handles=handles, title=title, loc="upper left",
                      bbox_to_anchor=(1.02, 1), fontsize=8)
            ax.text(1.03, 0.35, f"Exposed settlers using sandbags: "
                    f"{protected:.1f}%\nSettlers exposed to floods: "
                    f"{exposed:.1f}%", transform=ax.transAxes, va="top",
                    fontsize=8, color=INK2, bbox=BOX)
        else:
            legend_above(fig, handles, title)

    figure_set(stem + VIEW_SUFFIX[view], section, len(groups), dist_figure,
               draw, finish, names=[INCGRPS[i] for i in groups],
               builds=[("types", labels, 0)] if kind != "protection" else (),
               doc=doc)


DEPTH_NOTE = ("Expected annual flood depth of each household's dwelling "
              "(cm/year, before any protection investment, log scale; per "
              "housing type, frequent pluvial floods are drained in formal "
              "housing, backyards and RDP premises); exposed households "
              f"(expected depth above {EXPOSURE_THRESHOLD:g}\\,cm/year).")

bench_dist_set("dist_households_over_depth", "main", "households", B,
               doc=(40, "Benchmark equilibrium",
                    "Exposed households across expected flood depth",
                    "Exposed households by income group, stacked by housing "
                    "type; boxes: share of the group's households exposed. "
                    + DEPTH_NOTE + " Same axes in all panels.", 0.9))
for view in VIEWS:
    bench_dist_set("dist_damages_over_depth", "online", "damages",
                   SIM[view]["benchmark"], view,
                   doc=(10 + (view == "ante"), "Benchmark equilibrium",
                        "Expected damages across expected flood depth"
                        + VIEW_CAPTION[view],
                        "Total actual expected damages to structures and "
                        "contents (M ZAR/year) of exposed households by "
                        "income group and housing type: connected diamonds "
                        "at the centre of each depth bin with households of "
                        "the housing type. " + DEPTH_NOTE + " Linear y axes "
                        "specific to each panel." + VIEW_NOTE[view], 0.9))
    for s, section, order in (("benchmark", "main", 50),
                              ("no_self_protec", "online", 20)):
        if s not in SIM[view]:
            continue
        check = s != "benchmark"
        bench_dist_set("dist_damages_paid_over_depth"
                       + ("_no_self_protection" if check else ""), section,
                       "damages_relative", SIM[view][s], view,
                       doc=(order + (view == "ante"), "Benchmark equilibrium",
                            "Damages paid per household across expected "
                            "flood depth" + (", without self-protection"
                                             if check else "")
                            + VIEW_CAPTION[view],
                            ("Benchmark without any self-protection (no "
                             "sandbags), as a check that informal settlers' "
                             "damages paid then increase with expected "
                             "flood depth. " if check else "")
                            + "Mean damages paid per exposed household "
                            "(net of insurance payouts; formal private "
                            "structures excluded; backyard shacks paid by "
                            "RDP owners) as a \\% of the group's mean "
                            "income, by income group and housing type: "
                            "connected diamonds at the centre of each depth "
                            "bin with households of the housing type. "
                            + DEPTH_NOTE + " Linear y axes specific to each "
                            "panel, log scale for low-income households."
                            + VIEW_NOTE[view], 0.9))
bench_dist_set("dist_self_protection_over_depth", "appendix", "protection", B,
               doc=(70, "Benchmark equilibrium",
                    "Self-protection of exposed low-income settlers",
                    "Exposed low-income informal settlement households by "
                    "sandbag levels chosen, across expected flood depth "
                    "(gray: unprotected). Box: shares over all informal "
                    "settlement households. " + DEPTH_NOTE, 0.75))

## ACROSS SCENARIOS: connected diamonds by scenario (benchmark on top), one
## panel per income group (or, for low-income households, per housing
## type), with builds adding the scenarios one at a time

DIST_SCENARIOS = [s for s in SCENARIOS if s != "combined"]


def scenario_handles(scen=DIST_SCENARIOS):
    return [Line2D([], [], color=SCEN_COLORS[s], linestyle=SCEN_STYLES[s],
                   label=SCEN_LABELS[s], **DIAMOND) for s in scen]


def scen_dist_set(stem, section, panels, labels, value, ylabel, view="post",
                  bins=None, xlabel=DEPTH_XLABEL, mean=None, sharey=False,
                  log_x=True, log_panels=(), doc=None):
    """value(d, panel) by bin (depth bins unless bins(panel)) for the
    DIST_SCENARIOS; mean(d, panel): vertical lines; same y axes in all
    builds (and panels if sharey)."""
    S = SIM[view]
    values = {(k, s): value(S[s], panel) for k, panel in enumerate(panels)
              for s in DIST_SCENARIOS}
    top = np.nanmax([np.nanmax(np.r_[v, 0]) for v in values.values()])

    def draw(ax, k, vis):
        b = BINS if bins is None else bins(panels[k])
        x = np.sqrt(b[:-1] * b[1:]) if log_x else (b[:-1] + b[1:]) / 2
        log = k in log_panels
        shown = DIST_SCENARIOS[:vis.get("scenarios", len(DIST_SCENARIOS))]
        for s in shown[::-1]:                     # benchmark drawn on top
            v = values[k, s]
            connected(ax, x, positive(v) if log else v,
                      color=SCEN_COLORS[s], linestyle=SCEN_STYLES[s],
                      zorder=20 if s == "benchmark" else 3)
        if mean:
            for s in shown:
                ax.axvline(mean(S[s], panels[k]), color=SCEN_COLORS[s],
                           linestyle=SCEN_STYLES[s], linewidth=1, zorder=22)
        style_hist(ax, log_label(ylabel) if log else ylabel, labels[k],
                   b, xlabel, log_x)
        set_ylim(ax, [values[k, s] for s in DIST_SCENARIOS], log,
                 top if sharey else None)

    def finish(fig, axes, idx, vis):
        handles = scenario_handles()
        if mean:
            handles.append(Line2D([], [], color=INK2, linewidth=1,
                                  label="Vertical lines: mean depth"))
        legend_above(fig, handles, "Scenario", ncol=3)

    figure_set(stem + VIEW_SUFFIX[view], section, len(panels), dist_figure,
               draw, finish, names=labels,
               builds=[("scenarios", [SCEN_LABELS[s] for s in DIST_SCENARIOS],
                        1)], doc=doc)


SCEN_DIST_NOTE = (" Benchmark and the four single-policy scenarios: "
                  "connected diamonds at the centre of each bin, the "
                  "benchmark in black, drawn on top.")
LOW_NOTE = (" Low-income households only, one panel per housing type "
            "where they live. Log axes leave out bins with zero damages "
            "(e.g. protected settlers under subsidised self-protection).")
DIST_SPECS = {
    # kind: (stem, caption, note, y label, order, mean lines, shared y)
    "households": ("dist_households_over_depth",
                   "Exposed households across expected flood depth",
                   "As Figure~\\ref{fig:dist_households_over_depth}, over "
                   "all housing types. Vertical lines: mean expected depth "
                   "of exposed households in each scenario. Depth is "
                   "measured before any protection investment, so the "
                   "distributions only move with relocation.",
                   "Households (thousands)", 90, True, True),
    "damages": ("dist_damages_over_depth",
                "Expected damages across expected flood depth",
                "As Figure~\\ref{fig:dist_damages_over_depth}, over all "
                "housing types. Linear y axes specific to each panel.",
                "Damages (M ZAR/year)", 100, False, False),
    "damages_relative": ("dist_damages_paid_over_depth",
                         "Damages paid per household across expected flood "
                         "depth",
                         "As Figure~\\ref{fig:dist_damages_paid_over_depth},"
                         " over all housing types. Linear y axes specific "
                         "to each panel, log scale for low-income "
                         "households.",
                         "Damages paid per household\n(% of mean income)",
                         110, False, False),
}

for kind, (stem, caption, note, ylabel, order, with_mean,
           sharey) in DIST_SPECS.items():
    for view in (VIEWS if kind != "households" else ("post",)):
        dv = VIEW_NOTE[view] if kind != "households" else ""
        off = 1 if view == "ante" else 0
        # By income group, over all housing types
        scen_dist_set(
            "scenarios_" + stem, "appendix", list(range(4)), INCGRPS,
            lambda d, i, kind=kind: dist_values(d, kind, i, total=True),
            ylabel, view, sharey=sharey,
            mean=(lambda d, i: mean_depth(d, groups=[i])) if with_mean
            else None,
            log_panels={0} if kind == "damages_relative" else (),
            doc=(order + off, "Policy scenarios",
                 caption + ", by scenario" + VIEW_CAPTION[view],
                 note + SCEN_DIST_NOTE + dv, 0.9))
        # Low-income households by housing type
        scen_dist_set(
            "scenarios_" + stem + "_low_income", "appendix", LOW_TYPES,
            [HTYPES[t] for t in LOW_TYPES],
            lambda d, t, kind=kind: dist_values(d, kind, 0)[t], ylabel, view,
            sharey=sharey,
            mean=(lambda d, t: mean_depth(d, types=[t], groups=[0]))
            if with_mean else None,
            log_panels=(set(range(len(LOW_TYPES)))
                        if kind == "damages_relative" else ()),
            doc=(order + 2 + off, "Policy scenarios",
                 caption + ", low-income households by housing type, by "
                 "scenario" + VIEW_CAPTION[view],
                 note.replace("over all housing types", "for low-income "
                              "households")
                 + (LOW_NOTE if kind == "damages_relative" else LOW_NOTE.split(" Log")[0]) + SCEN_DIST_NOTE + dv, 0.9))

scen_dist_set("scenarios_dist_self_protection_over_depth", "appendix", [0],
              [None], lambda d, i: dist_values(d, "protection", 0,
                                               total=True),
              "Protected exposed settlers",
              doc=(120, "Policy scenarios",
                   "Self-protection of exposed low-income settlers, by "
                   "scenario",
                   "Exposed low-income informal settlement households "
                   "protected by sandbags, across expected flood depth."
                   + SCEN_DIST_NOTE, 0.75))

## HOUSEHOLDS ACROSS RENTS

RENT_XLABEL = "Annual rent (ZAR/m²)"
# Household-weighted quantile of rents at the axis end, by income group
RENT_TOP = {0: 1, 1: 0.995, 2: 1, 3: 1}


def rent_bins(i, scenarios=("benchmark",), types=(0, 1, 2)):
    """30 bins from the lowest rent paid by group i in these housing types
    and scenarios to the RENT_TOP[i] quantile of these rents
    (household-weighted)."""
    rents, weights = [], []
    for s in scenarios:
        ok = sim[s]["hh"][list(types), i] > 0
        rents.append(sim[s]["rent_clean"][list(types)][ok])
        weights.append(sim[s]["hh"][list(types), i][ok])
    rents, weights = np.concatenate(rents), np.concatenate(weights)
    top = wquantile(rents, weights, RENT_TOP[i])
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


RENT_BENCH_BINS = {i: rent_bins(i) for i in range(4)}
RENT_HIST = {i: rent_hist(B, i, RENT_BENCH_BINS[i]) for i in range(4)}


def draw_rent_hist(ax, k, vis):
    b = RENT_BENCH_BINS[k]
    v = RENT_HIST[k].copy()
    v[vis.get("types", 3):] = 0
    bars2d(ax, (b[:-1] + b[1:]) / 2, np.full(len(b) - 1, np.diff(b)[0] * 0.92),
           v, HTYPE_COLORS[:3])
    ax.set_ylim(0, 1.06 * bar_top(RENT_HIST[k]))
    style_hist(ax, "Households (thousands)", INCGRPS[k], b, RENT_XLABEL,
               log_x=False)


RENT_NOTE = ("Households of each income group across the annual rent per "
             "m\\textsuperscript{2} of their dwelling, in the rented housing "
             "types (RDP housing is free); all households. Bins and axes are "
             "specific to each panel, from the lowest to the highest rent of "
             "the group; mid-low income: up to the 99.5th percentile of its "
             "rents (household-weighted), leaving out a few hundred "
             "households in settlements and backyards with higher rents.")
figure_set("dist_households_over_rent", "online", 4, dist_figure,
           draw_rent_hist,
           lambda fig, axes, panels, vis: legend_above(
               fig, legend_handles(HTYPE_COLORS[:3], HTYPES[:3]),
               "Housing type"),
           names=INCGRPS, builds=[("types", HTYPES[:3], 0)],
           doc=(30, "Benchmark equilibrium", "Households across rents",
                RENT_NOTE + " Stacked by housing type.", 0.9))

RENT_BINS = {i: rent_bins(i, DIST_SCENARIOS) for i in range(4)}
scen_dist_set("scenarios_dist_households_over_rent", "appendix", list(range(4)),
              INCGRPS, lambda d, i: rent_hist(d, i, RENT_BINS[i]).sum(0),
              "Households (thousands)", bins=lambda i: RENT_BINS[i],
              xlabel=RENT_XLABEL, log_x=False,
              doc=(130, "Policy scenarios", "Households across rents, by "
                   "scenario",
                   "As Figure~\\ref{fig:dist_households_over_rent}, over "
                   "all rented housing types. Bins and axes are specific to "
                   "each panel (mid-low income: up to the 99.5th percentile "
                   "of its rents over these scenarios)." + SCEN_DIST_NOTE,
                   0.9))
LOW_RENT_TYPES = [t for t in LOW_TYPES if t < 3]
RENT_LOW_BINS = {t: rent_bins(0, DIST_SCENARIOS, (t,)) for t in LOW_RENT_TYPES}
scen_dist_set("scenarios_dist_households_over_rent_low_income", "appendix",
              LOW_RENT_TYPES, [HTYPES[t] for t in LOW_RENT_TYPES],
              lambda d, t: rent_hist(d, 0, RENT_LOW_BINS[t])[t],
              "Households (thousands)", bins=lambda t: RENT_LOW_BINS[t],
              xlabel=RENT_XLABEL, log_x=False,
              doc=(132, "Policy scenarios", "Low-income households across "
                   "rents, by housing type, by scenario",
                   "Low-income households across the annual rent per "
                   "m\\textsuperscript{2} of their dwelling, one panel per "
                   "rented housing type where they live; bins and axes "
                   "specific to each panel." + SCEN_DIST_NOTE, 0.9))

###############################################################################

# %% WELFARE, LANDLORDS AND SCENARIO METRICS

print("WELFARE AND METRICS")

## Welfare rows (user, 2026-11-19): low-income households in informal
## settlements, in informal backyards and in RDP housing (no low-income
## households live in formal private housing), then groups 2-4; as
## (name, short name, housing types, income group)
ROW_SPECS = [("Low income, informal settlements", "Low inc., IS", [2], 0),
             ("Low income, informal backyards", "Low inc., IB", [1], 0),
             ("Low income, RDP owners", "Low inc., RDP", [3], 0),
             ("Mid-low income", "Mid-low inc.", [0, 1, 2, 3], 1),
             ("Mid-high income", "Mid-high inc.", [0, 1, 2, 3], 2),
             ("High income", "High inc.", [0, 1, 2, 3], 3)]
assert B["hh"][0, 0].sum() < 1, "low-income FP households: add a row"
ROWS = [r[0] for r in ROW_SPECS]
SHORT_ROWS = [r[1] for r in ROW_SPECS]
NR = len(ROWS)
RDP_ROW = 2
ROW_MASKS = []
for name, short, types, i in ROW_SPECS:
    m = np.zeros((4, 4), bool)
    m[types, i] = True
    ROW_MASKS.append(m)
ROW_GROUP = np.array([r[3] for r in ROW_SPECS])
ROW_INCOME = y_bar[ROW_GROUP]
ROW_SHARE = np.array([100 * (B["hh"] * m[:, :, None]).sum() / N_TOTAL
                      for m in ROW_MASKS])
ROW_LABELS = [f"{name} ({share:.1f}% of households)"
              for name, share in zip(ROWS, ROW_SHARE)]


def mean_utility(u, w):
    ok = (w > 0) & np.isfinite(u)
    return np.sum(w[ok] * u[ok]) / w[ok].sum() if ok.any() else np.nan


def welfare(d, base, mask=None):
    """Welfare of each row in scenario d relative to base (the benchmark of
    the same view), on the households of the subsample mask ((types,
    cells), None: all): n, base households of the row in the subsample;
    n_d, the same in d; u, mean utility in d (ex post or in equilibrium,
    depending on the view); du, its % change; ev, equivalent variation per
    base household (rands/year), so that sum(n x ev) is the aggregate
    monetary gain of the subsample.

    EV as in scenarios.welfare_table, at benchmark prices and locations,
    with the ratio r of mean utilities of the row (over the subsample): FP
    (ỹ - R q0)(r - 1); IB, IS c z (r^(1/α) - 1); RDP owners, with their own
    ratio in each cell (at zero benchmark consumption, c z with z giving
    their utility in d).

    RDP choice (RC1): base owners whose unit is vacant in d get the mean
    utility in d of the low-income market households and of d's owners of
    units vacant in the base only (newcomers, whose utility is up to A(x)
    in accounting.ex_post)."""
    res = {key: np.zeros(NR) for key in ("n", "n_d", "u", "du", "ev")}
    M = 1 if mask is None else mask[:, None, :]
    newcomers = np.zeros_like(d["hh"])
    newcomers[3, 0] = d["hh"][3, 0] * (base["vacant"] & ~d["vacant"])
    u_d = d["ex_post_u_post"].copy()
    u_d[3, 0] = u_d[3, 0] * inputs["amenities"]
    market = np.zeros_like(d["hh"])
    market[:3, 0] = d["hh"][:3, 0]
    pool = mean_utility(u_d, market + newcomers)
    for r, (name, short, types, i) in enumerate(ROW_SPECS):
        wb = base["hh"] * ROW_MASKS[r][:, :, None] * M
        ws = d["hh"] * ROW_MASKS[r][:, :, None] * M
        res["n"][r], res["n_d"][r] = wb.sum(), ws.sum()
        ub = mean_utility(base["ex_post_u_post"], wb)
        us = mean_utility(d["ex_post_u_post"], ws)
        res["u"][r], res["du"][r] = us, 100 * (us / ub - 1)
        if d is base:
            continue
        if r == RDP_ROW:
            with np.errstate(invalid="ignore", divide="ignore",
                             over="ignore"):
                target = np.where(d["vacant"], pool / inputs["amenities"],
                                  d["ex_post_u_post"][3, 0])
                ratio = target / base["ex_post_u_post"][3, 0]
                # At zero consumption (floor): EV = c z, with z giving the
                # target utility at the benchmark backyard share
                z_b = base["ex_post_z_post"][3, 0]
                c_b = base["ex_post_price"][3, 0]
                mu_b = np.nan_to_num(base["housing_supply"][1]) / 1e6
                housing = (p["RDP_size"] + p["backyard_size"] * (1 - mu_b)
                           - q0) ** p["beta"]
                ev = np.nan_to_num(np.where(
                    z_b > 0, c_b * z_b * (ratio ** (1 / alpha) - 1),
                    c_b * (target / housing) ** (1 / alpha)))
            res["ev"][r] = (np.sum(wb[3, 0] * ev) / wb[3, 0].sum()
                            if wb[3, 0].sum() > 0 else 0.)
            continue
        ratio = us / ub
        ev = np.zeros((4, N_CELLS))
        ev[0] = (base["net_income"][i] - np.nan_to_num(base["rent"][0]) * q0) \
            * (ratio - 1)
        for h in (1, 2):
            ev[h] = base["ex_post_price"][h, i] * base["ex_post_z_post"][h, i] \
                * (ratio ** (1 / alpha) - 1)
        w = wb[:, i]
        res["ev"][r] = (np.sum(np.where(w > 0, w * np.nan_to_num(ev), 0))
                        / w.sum() if w.sum() > 0 else 0.)
    return res


def landlords(d):
    """Yearly revenues (rands) of absentee landlords: legitimate (formal
    land; ex post: gross of developers' losses, the land rents developers
    would pay before compensating them, i.e. δ P + t per m2 of developed
    land; ex ante: the equilibrium land rents, after the developers' tax
    they anticipate), illegitimate (slumlords' rents from informal
    settlements), and developers' losses (ex post: the expected
    compensation of their unanticipated losses, plus the RDP floor's
    uncovered damages; ex ante: none, they are not anticipated)."""
    t, losses, revenue = d["developer_tax"]
    if d["view"] == "ante":
        return d["landlord_revenue"][0].sum(), d["landlord_revenue"][1].sum(), 0.
    return (d["landlord_revenue"][0].sum() + revenue,
            d["landlord_revenue"][1].sum(),
            losses + d["surplus_damages"].sum())


def landlords_net(d):
    legit, illegit, losses = landlords(d)
    return legit - losses


## DAMAGE AND EXPOSURE STATISTICS (tables 3, 5, 6)

def stat_rows(sub, types=(0, 1, 2, 3)):
    """(label, key, digits) of the damage and exposure rows, for the
    subsample sub ("exposed" or "most")."""
    word = "exposed" if sub == "exposed" else "most exposed"
    rows = [("Household damages (M ZAR/year)", "hh_dmg", 2)]
    if 0 in types:
        rows.append(("Structure damages, formal private (M ZAR/year)",
                     "struct", 2))
    rows += [(f"Damages paid, {word} (% of income)", "rel", 3),
             (f"Mean expected flood depth, {word} (cm/year)", "depth", 2),
             (f"Damages paid, 9th decile of {word} (% of income)", "p90_rel",
              3),
             (f"Expected flood depth, 9th decile of {word} (cm/year)",
              "p90_depth", 2),
             (f"{word.capitalize()} households (% of all households)",
              "share", 2)]
    return rows


def exposure_stats(d, types, groups, sub="exposed"):
    """Damage and exposure statistics of the households of these housing
    types and income groups (lists). Damages: all households of the
    category (sub = "exposed") or those in the subsample only ("most");
    shares of income, depth and deciles: households in the subsample."""
    mask = SUBSAMPLES[sub]
    ix = np.ix_(types, groups)
    hh = d["hh"][ix]
    m = mask[types][:, None, :]
    inside = m if sub != "exposed" else 1
    n_sub = (hh * m).sum()
    income = (hh * m * y_bar[groups][None, :, None]).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.nan_to_num(100 * d["dmg_borne"][ix] / hh
                            / y_bar[groups][None, :, None])
    depth = np.broadcast_to(DEPTH[types][:, None, :], hh.shape)
    out = {"hh_dmg": (d["dmg_hh"][ix] * inside).sum() / 1e6,
           "struct": (d["dmg_struct"][ix] * inside).sum() / 1e6,
           "share": 100 * n_sub / d["hh"].sum()}
    if n_sub <= 0:
        return {**out, "rel": np.nan, "depth": np.nan, "p90_rel": np.nan,
                "p90_depth": np.nan}
    out.update(rel=100 * (d["dmg_borne"][ix] * m).sum() / income,
               depth=(hh * m * depth).sum() / n_sub,
               p90_rel=wquantile(rel, hh * m, 0.9),
               p90_depth=wquantile(depth, hh * m, 0.9))
    return out


def public_cost(d, mask=None):
    """Public cost (rands/year) of free sandbags (PS) and insurance payouts
    (SI) in scenario d, on the households of the subsample mask (None: all;
    then the public budget)."""
    d = d["post"]                           # actual payouts in both views
    if mask is None:
        return d["budget"][0] + d["budget"][1]
    cost = 0.
    if d["opts"]["subsid_protec"]:
        cost += p["sandbag_course_cost"] * (d["hh_type"][2] * d["level"]
                                            * mask[2]).sum()
    if d["opts"]["subsid_insur"]:
        cost += (d["damages_reimbursed"] * mask[:, None, :]).sum()
    return cost


assert np.isclose(public_cost(sim["subsid_protec"],
                              np.ones((4, N_CELLS), bool)),
                  public_cost(sim["subsid_protec"]))


def targeted_share(d, label, sub):
    """% of all households targeted by the policy of scenario label, within
    the subsample (exposed: all households; most: most exposed): exposed
    households (No RM, Public P.), exposed settlers (Subs. SP), low and
    mid-low income households (Subs. ins.), and their union (Combined)."""
    if label == "benchmark":
        return np.nan
    mask = SUBSAMPLES[sub]
    within = np.ones((4, N_CELLS), bool) if sub == "exposed" else mask
    target = np.zeros((4, 4, N_CELLS), bool)
    if label in ("no_misperception", "public_protec", "combined"):
        target |= (mask & within)[:, None, :]
    if label in ("subsid_protec",):
        target[2] |= (mask[2] & within[2])[None, :]
    if label in ("subsid_insur", "combined"):
        target[:, :2] |= within[:, None, :]
    return 100 * (d["hh"] * target).sum() / d["hh"].sum()


def summary_values(S, label, sub="exposed"):
    """Rows of the scenario summary (table 5) for scenario label of the
    view S, over all households (sub = "exposed") or the most exposed."""
    d, base = S[label], S["benchmark"]
    st = exposure_stats(d, [0, 1, 2, 3], [0, 1, 2, 3], sub)
    out = {key: st[key] for _, key, _ in stat_rows(sub)}
    mask = SUBSAMPLES[sub]
    out["protected"] = settler_shares(d, mask)[0]
    cost = public_cost(d, None if sub == "exposed" else mask)
    out["cost"] = cost / 1e6
    if label == "benchmark" or cost <= 0:
        out["efficiency"] = np.nan
    else:
        sb = exposure_stats(base, [0, 1, 2, 3], [0, 1, 2, 3], sub)
        cut = (sb["hh_dmg"] + sb["struct"] - st["hh_dmg"] - st["struct"])
        out["efficiency"] = 1e6 * cut / cost
    out["targeted"] = targeted_share(d, label, sub)
    return out


def summary_rows(sub):
    word = "exposed" if sub == "exposed" else "most exposed"
    return stat_rows(sub) + [
        (f"{word.capitalize()} settlers protected (%)", "protected", 1),
        ("Public cost of PS and SI (M ZAR/year)", "cost", 2),
        ("Cost efficiency (damage reduction per ZAR spent)", "efficiency", 2),
        ("Targeted population (% of all households)", "targeted", 2)]


AGG_EV = "Aggregate EV (M ZAR)"
AGG_EV_LANDLORDS = "Aggregate EV + landlords' net revenue (M ZAR)"
AGG_EV_HH = "Aggregate EV (ZAR/household)"
# Outcomes of the pairwise interactions (not the ratios: cost efficiency,
# targeted population)
INTER_KEYS = ["hh_dmg", "struct", "rel", "depth", "p90_rel", "p90_depth",
              "share", "protected", "cost"]


def metrics(S, label):
    """Scalar outcomes of scenario label in view S: summary rows (keys of
    summary_rows("exposed")), and welfare changes relative to the
    benchmark by subsample ("<subsample>|<kind>|<row>" keys)."""
    d, base = S[label], S["benchmark"]
    out = dict(summary_values(S, label))
    for sub, mask in SUBSAMPLES.items():
        w = welfare(d, base, mask)
        for r, name in enumerate(SHORT_ROWS):
            out[f"{sub}|Utility (%)|{name}"] = w["du"][r]
            out[f"{sub}|EV (ZAR/household/year)|{name}"] = w["ev"][r]
        out[f"{sub}|{AGG_EV}"] = np.sum(w["n"] * w["ev"]) / 1e6
        out[f"{sub}|{AGG_EV_HH}"] = np.sum(w["n"] * w["ev"]) / w["n"].sum()
    out["landlords"] = (landlords_net(d) - landlords_net(base)) / 1e6
    out[f"all|{AGG_EV_LANDLORDS}"] = out[f"all|{AGG_EV}"] + out["landlords"]
    return out


METRICS = {view: {s: metrics(SIM[view], s) for s in SCENARIOS}
           for view in VIEWS}

###############################################################################

# %% DAMAGE, EXPOSURE AND WELFARE TABLES

print("DAMAGE, EXPOSURE AND WELFARE TABLES")

COLS_SCEN = [SCEN_SHORT[s] for s in SCENARIOS]
DAMAGE_TABLE_NOTE = (
    "Household damages: actual expected annual damages to the contents of "
    "all dwellings and to the structures households own (shacks, RDP "
    "houses), M ZAR/year; structure damages, formal private: damages to "
    "developers' structures (attributed to their occupants' income group). "
    "Damages paid: damages households pay for (net of insurance payouts; "
    "no formal private structures; backyard shacks paid by the RDP owners "
    "of the cell) as a \\% of the mean income of the {word} households "
    "(mean) or per household as a \\% of its group's mean income (9th "
    "decile: 90th percentile across {word} households). Expected flood "
    "depth at the dwelling, before any protection investment; {word}: "
    "{define}. Dashes: no households, zero, or not applicable.")
DEFINE = {"exposed": f"expected depth above {EXPOSURE_THRESHOLD:g}\\,cm/year",
          "most": f"expected depth above {MOST_TXT}"}


def damage_note(sub):
    word = "exposed" if sub == "exposed" else "most exposed"
    return DAMAGE_TABLE_NOTE.format(word=word, define=DEFINE[sub])


## BENCHMARK DAMAGES AND EXPOSURE BY HOUSING TYPE AND INCOME GROUP
## (user, 2026-11-19: no population shares in labels; exposed households as
## a % of all households, last in each block; 9th deciles; household and
## formal private structure damages instead of developers' losses)

COLS_GROUPS = INCGRPS + ["All"]


def damage_table(S, sub):
    rows = {}
    for t, block in enumerate(HTYPES + ["All housing types"]):
        types = [t] if t < 4 else [0, 1, 2, 3]
        for j, col in enumerate(COLS_GROUPS):
            groups = [j] if j < 4 else [0, 1, 2, 3]
            st = exposure_stats(S["benchmark"], types, groups, sub)
            for label, key, _ in stat_rows(sub, types):
                rows.setdefault((block, label), {})[col] = st[key]
    return pd.DataFrame(rows).T[COLS_GROUPS]


def digits_of(rows):
    return {label: dg for label, _, dg in rows}


for sub, section, order in (("exposed", "main", 60), ("most", "appendix", 75)):
    digits = digits_of(stat_rows(sub))
    for view in VIEWS:
        write_table(
            damage_table(SIM[view], sub), section,
            "table_damages_exposure_benchmark"
            + ("_above_5cm" if sub == "most" else "") + VIEW_SUFFIX[view],
            lambda block, row, c, v, dg=digits: _num(v, dg[row],
                                                     dash_zero=True),
            doc=(order + (view == "ante"), "Benchmark equilibrium",
                 "Damages and exposure in the benchmark, by housing type and "
                 "income group" + ("" if sub == "exposed" else
                                   f", households above {MOST_TXT}")
                 + VIEW_CAPTION[view],
                 ("" if sub == "exposed" else
                  f"Households with an expected flood depth above {MOST_TXT}"
                  " only (damages included). ")
                 + damage_note(sub) + VIEW_NOTE[view]))

## SCENARIO SUMMARY (user, 2026-11-19: rows as in the damage table, cost
## efficiency, targeted population)

for sub, section, order in (("exposed", "main", 70), ("most", "appendix", 77)):
    rows_s = summary_rows(sub)
    digits = digits_of(rows_s)
    for view in VIEWS:
        vals = {s: summary_values(SIM[view], s, sub) for s in SCENARIOS}
        table = pd.DataFrame({label: {s: vals[s][key] for s in SCENARIOS}
                              for label, key, _ in rows_s}).T[SCENARIOS]
        table.columns = COLS_SCEN
        write_table(
            pd.concat({"": table}), section,
            "table_scenario_summary" + ("_above_5cm" if sub == "most" else "")
            + VIEW_SUFFIX[view],
            lambda block, row, c, v, dg=digits: _num(v, dg[row],
                                                     dash_zero=True),
            doc=(order + (view == "ante"), "Policy scenarios",
                 "Scenario summary" + ("" if sub == "exposed" else
                                       f", households above {MOST_TXT}")
                 + VIEW_CAPTION[view],
                 "Levels by scenario. " + ("" if sub == "exposed" else
                                           f"Households with an expected "
                                           f"flood depth above {MOST_TXT} "
                                           "only. ")
                 + damage_note(sub) + " Settlers protected: share of the "
                 + ("exposed" if sub == "exposed" else "most exposed")
                 + " informal settlement households using sandbags. Public "
                 "cost: free sandbags and insurance payouts"
                 + ("" if sub == "exposed" else " to these households")
                 + ", financed by lump-sum taxes on groups 3--4 (public "
                 "protection is free in the model). Cost efficiency: "
                 "reduction in household and structure damages relative to "
                 "the benchmark per ZAR of public cost (dashes: no public "
                 "cost; combined: also includes the free levers). Targeted "
                 "population: exposed households (Public P., No RM), "
                 "exposed informal settlement households (Subs.\\ SP), low "
                 "and mid-low income households (Subs.\\ ins.), their union "
                 "(Combined)" + ("" if sub == "exposed" else ", within the "
                                 "subsample") + ", as a \\% of all "
                 "households of the scenario." + VIEW_NOTE[view],
                 0.9 if sub == "exposed" else True))

## CHANGES ACROSS SCENARIOS BY INCOME GROUP (user, 2026-11-19: changes in
## absolute terms, in the units of each row; rows as in the damage table)

for sub, order in (("exposed", 80), ("most", 82)):
    rows_s = stat_rows(sub)
    digits = digits_of(rows_s)
    for view in VIEWS:
        S = SIM[view]
        rows = {}
        for j, grp in enumerate(INCGRPS + ["All"]):
            groups = [j] if j < 4 else [0, 1, 2, 3]
            stats = {s: exposure_stats(S[s], [0, 1, 2, 3], groups, sub)
                     for s in SCENARIOS}
            for label, key, _ in rows_s:
                base = stats["benchmark"][key]
                rows[(label, grp)] = {
                    "benchmark": base,
                    **{s: stats[s][key] - base for s in SCENARIOS[1:]}}
        table = pd.DataFrame(rows).T[SCENARIOS]
        table.columns = COLS_SCEN
        write_table(
            table, "appendix", "table_damages_exposure_scenarios"
            + ("_above_5cm" if sub == "most" else "") + VIEW_SUFFIX[view],
            lambda block, row, c, v, dg=digits: _num(
                v, dg[block], sign=c != "Bench.", dash_zero=True),
            doc=(order + (view == "ante"), "Policy scenarios",
                 "Damages and exposure across scenarios, by income group"
                 + ("" if sub == "exposed" else
                    f", households above {MOST_TXT}") + VIEW_CAPTION[view],
                 "Benchmark levels (first column) and changes relative to "
                 "the benchmark (scenario minus benchmark, in the units of "
                 "each block) of the statistics of Table~\\ref{tab:table_"
                 "damages_exposure_benchmark" + ("_above_5cm" if sub == "most"
                                                 else "") + VIEW_SUFFIX[view]
                 + "}, over all housing types. " + damage_note(sub)
                 + VIEW_NOTE[view], 0.9))

## WELFARE (ALL, EXPOSED AND MOST EXPOSED HOUSEHOLDS)
## (user, 2026-11-19: low-income households split by housing type; one
## landlords block with legitimate and illegitimate revenues and
## developers' losses; average EV over all households; subsample shares of
## all households)

WELFARE_NOTE = (
    "Utilities: benchmark levels (first column; household-weighted means; "
    "RDP owners: mean over cells, up to the amenity index) and \\% changes; "
    "ex post, with actual instead of perceived damages. EV: equivalent "
    "variation per benchmark household of each row, at benchmark prices "
    "and locations, in ZAR/year (benchmark column: the row's mean income). "
    "Taxes financing the subsidies (groups 3--4) are included. Changes "
    "below about 0.3\\% for low and mid-low income households are within "
    "solver noise.")
LANDLORD_NOTE = (
    " Landlords, in ZAR/year per household of the city (benchmark: levels; "
    "scenarios: changes): legitimate revenues, the formal land rents "
    "developers would pay before compensating their unanticipated losses; "
    "illegitimate revenues, rents of informal settlements; developers' "
    "losses, the expected compensation of their unanticipated losses. "
    "Aggregate monetary gain: EVs weighted by the number of households of "
    "each row, plus the change in landlords' legitimate revenues net of "
    "developers' losses (illegitimate revenues excluded), M ZAR/year. "
    "Public protection is free in the model: its aggregate gain is the "
    "largest annual cost at which it breaks even.")
LANDLORD_NOTE_ANTE = (
    " Landlords, in ZAR/year per household of the city (benchmark: levels; "
    "scenarios: changes): legitimate revenues, the equilibrium formal land "
    "rents (after the developers' tax, which developers anticipate); "
    "illegitimate revenues, rents of informal settlements; developers' "
    "losses are not anticipated. Aggregate monetary gain: EVs weighted by "
    "the number of households of each row, plus the change in legitimate "
    "revenues, M ZAR/year.")


def welfare_table(S, sub):
    mask = SUBSAMPLES[sub]
    base = S["benchmark"]
    res = {s: welfare(S[s], base, mask) for s in SCENARIOS}
    rows = {}
    labels = ROW_LABELS if sub == "all" else ROWS
    word = "Exposed" if sub == "exposed" else "Most exposed"
    for r, label in enumerate(labels):
        if sub != "all":
            rows[(label, f"{word} (% of all households)")] = {
                s: 100 * res[s]["n_d"][r] / S[s]["hh"].sum()
                for s in SCENARIOS}
        rows[(label, "Utility")] = {"benchmark": res["benchmark"]["u"][r],
                                    **{s: res[s]["du"][r]
                                       for s in SCENARIOS[1:]}}
        rows[(label, "EV (ZAR/household/year)")] = {
            "benchmark": ROW_INCOME[r],
            **{s: res[s]["ev"][r] for s in SCENARIOS[1:]}}
    n_b = res["benchmark"]["n"]
    if sub == "all":
        rows[("All households", "Average EV (ZAR/household/year)")] = {
            "benchmark": np.sum(n_b * ROW_INCOME) / n_b.sum(),
            **{s: np.sum(res[s]["n"] * res[s]["ev"]) / res[s]["n"].sum()
               for s in SCENARIOS[1:]}}
        acc = {s: np.array(landlords(S[s])) / N_TOTAL for s in SCENARIOS}
        names = ["Legitimate revenues (ZAR/household/year)",
                 "Illegitimate revenues (ZAR/household/year)",
                 "Developers' losses (ZAR/household/year)"]
        for k, name in enumerate(names):
            rows[("Landlords", name)] = {
                "benchmark": acc["benchmark"][k],
                **{s: acc[s][k] - acc["benchmark"][k]
                   for s in SCENARIOS[1:]}}
        rows[("Total", "Aggregate monetary gain (M ZAR/year)")] = {
            "benchmark": np.nan,
            **{s: (np.sum(res[s]["n"] * res[s]["ev"])
                   + (landlords_net(S[s]) - landlords_net(base))) / 1e6
               for s in SCENARIOS[1:]}}
    else:
        rows[("Total", "Sum of EVs, weighted by subsample shares "
                       "(ZAR/household/year)")] = {
            "benchmark": np.nan,
            **{s: np.sum(res[s]["n"] * res[s]["ev"]) / res[s]["n"].sum()
               for s in SCENARIOS[1:]}}
        rows[("Total", "Subsample (% of all households)")] = {
            s: res[s]["n_d"].sum() / S[s]["hh"].sum() * 100
            for s in SCENARIOS}
    table = pd.DataFrame(rows).T[SCENARIOS]
    table.columns = COLS_SCEN
    return table


def fmt_welfare(block, row, c, v):
    if "% of all" in row:
        return _num(v, 2, dash_zero=True)
    if block == "Landlords" and v == 0:
        return "--"
    if c == "Bench.":
        return _num(v, 0 if np.isfinite(v) and abs(v) > 100 else 1)
    if block == "Landlords" and v == 0:
        return "--"
    return _num(v, 3 if row.startswith("Utility") else 1, sign=True)


for sub, stem, order in (("all", "table_welfare", 80),
                         ("exposed", "table_welfare_exposed", 90),
                         ("most", "table_welfare_most_exposed", 100)):
    for view in VIEWS:
        table = welfare_table(SIM[view], sub)
        note = WELFARE_NOTE if view == "post" else WELFARE_NOTE.replace(
            "ex post, with actual instead of perceived damages",
            "in equilibrium, with the damages agents anticipate")
        if sub == "all":
            note += (" Average EV: EVs weighted by the number of households "
                     "of each row (benchmark column: mean income)."
                     + (LANDLORD_NOTE if view == "post"
                        else LANDLORD_NOTE_ANTE))
        else:
            word = "exposed" if sub == "exposed" else "most exposed"
            note = (f"As Table~\\ref{{tab:table_welfare{VIEW_SUFFIX[view]}}},"
                    f" for the {word} households ("
                    + DEFINE[sub] + "): benchmark subsample for EVs, each "
                    "scenario's subsample for utilities, with the number of "
                    f"{word} households of each row as a \\% of all "
                    "households in each scenario. Total: sum of the rows' "
                    "EVs weighted by their shares of the benchmark "
                    "subsample, i.e.\\ the mean EV per household of the "
                    "subsample (ZAR/year), and the subsample's share of all "
                    "households in each scenario.")
        write_table(table, "main", stem + VIEW_SUFFIX[view], fmt_welfare,
                    doc=(order + (view == "ante"), "Policy scenarios",
                         {"all": "Welfare, all households",
                          "exposed": "Welfare, exposed households",
                          "most": "Welfare, most exposed households "
                                  f"(above {MOST_TXT})"}[sub]
                         + VIEW_CAPTION[view], note + VIEW_NOTE[view]))
        print(table.round(3).to_string())

###############################################################################

# %% RENT REGRESSIONS

print("RENT REGRESSIONS")


def ols(y, X, names, w=None):
    """(Weighted) OLS with HC1 standard errors: (coefficients, s.e.)
    Series, n, R2."""
    X = np.column_stack([X, np.ones(len(y))])
    names = names + ["Constant"]
    sw = np.ones(len(y)) if w is None else np.sqrt(w / w.mean())
    Xw, yw = X * sw[:, None], y * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    e = yw - Xw @ beta
    n, k = X.shape
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    V = XtXi @ (Xw.T * e ** 2) @ Xw @ XtXi * n / (n - k)
    r2 = 1 - e @ e / np.sum((yw - np.average(y, weights=sw ** 2) * sw) ** 2)
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


def reg_table(columns, order_vars):
    df = pd.DataFrame(columns)
    order = []
    for name in order_vars + ["Constant"]:
        order += [(name, "coef"), (name, "se")]
    order += [("Observations", ""), ("R²", "")]
    df = df.reindex([o for o in order if o in df.index]).fillna("")
    return df


def reg_latex(df, section, stem, extra=()):
    """Regression tables: coefficient rows, then standard error rows;
    extra: (label, values) rows after R2."""
    df.to_excel(folder(section, "tables") / f"{stem}.xlsx")
    cols = list(df.columns)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & ".join([""] + [str(c) for c in cols]) + r" \\", r"\midrule"]
    for (name, kind), values in df.iterrows():
        if name == "Observations":
            lines.append(r"\midrule")
        label = "" if kind == "se" else name.replace("²", r"\textsuperscript{2}")
        lines.append(label + " & " + " & ".join(
            re.sub(r"(\*+)", lambda m: f"$^{{{m.group(1)}}}$", str(v))
            for v in values) + r" \\")
    for label, values in extra:
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (folder(section, "tables") / f"{stem}.tex").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"  {SECTIONS[section]}/tables/{stem}.tex")


REG_NOTE = ("OLS regressions of the log of the annual model rent per "
            "m\\textsuperscript{2} across 500\\,m grid cells with "
            "households of the housing type; heteroskedasticity-robust "
            "(HC1) standard errors in parentheses; $^{*}$ $p<0.1$, $^{**}$ "
            "$p<0.05$, $^{***}$ $p<0.01$. Expected flood depth: cm/year, "
            "before any protection investment, at the housing type's "
            "drainage level. Equilibrium relations, not causal effects.")

## FORMAL PRIVATE RENTS ON DATA AND ESTIMATION COVARIATES (main body;
## user, 2026-11-19): data dwelling size (2011 sub-place average of formal
## dwellings, from the nearest sub-place centroid), log income net of
## commuting costs (transport model, averaged with the 2011 income
## distribution of the sub-place) and log calibrated amenity index;
## weighted by the number of formal private households in column 3

_mat = __import__("scipy.io", fromlist=["io"]).loadmat(
    config.DATA / "precalculated_inputs" / "data.mat")["data"]
SP_XY = 1000 * np.column_stack([_mat["spX"][0, 0].ravel(),
                                _mat["spY"][0, 0].ravel()])
SP_SIZE = _mat["spDwellingSize"][0, 0].ravel()
SP_INCOME = _mat["sp2011IncomeDistributionNClass"][0, 0].astype(float)
SP_DIST, SP_NEAREST = cKDTree(SP_XY).query(
    np.column_stack([grid_csv["X"], grid_csv["Y"]]))
with np.errstate(invalid="ignore", divide="ignore"):
    SP_SHARES = SP_INCOME / SP_INCOME.sum(1, keepdims=True)
DATA_SIZE = SP_SIZE[SP_NEAREST]
DATA_NET_INCOME = (SP_SHARES[SP_NEAREST].T * NET_INCOME).sum(0)
FP_VARS = ["Expected flood depth (cm/year)", "Log dwelling size (data)",
           "Log net income", "Log amenity index"]
hh_fp = B["hh_type"][0]
with np.errstate(divide="ignore", invalid="ignore"):
    X_fp = np.column_stack([DEPTH[0], np.log(DATA_SIZE),
                            np.log(DATA_NET_INCOME), np.log(amen)])
    y_fp = np.log(B["rent_clean"][0])
ok_fp = (hh_fp > 0) & (B["rent_clean"][0] > 0) & np.isfinite(X_fp).all(1) \
    & (SP_DIST < 5000)
cols_fp = {
    "(1)": reg_column(ols(y_fp[ok_fp], X_fp[ok_fp][:, :2], FP_VARS[:2])),
    "(2)": reg_column(ols(y_fp[ok_fp], X_fp[ok_fp], FP_VARS)),
    "(3)": reg_column(ols(y_fp[ok_fp], X_fp[ok_fp], FP_VARS,
                          w=hh_fp[ok_fp]))}
reg_latex(reg_table(cols_fp, FP_VARS), "main",
          "table_rent_regression_formal_private_data_covariates",
          extra=[("Weights", ["--", "--", "Households"])])
register("main", "table",
         "table_rent_regression_formal_private_data_covariates", 65,
         "Benchmark equilibrium",
         "Formal private rents, flood risks and data covariates",
         REG_NOTE.replace("with households of the housing type",
                          "with formal private households in the "
                          "benchmark") +
         " Covariates from data and estimation rather than the model: "
         "dwelling size, 2011 mean size of formal dwellings in the nearest "
         "sub-place (centroid within 5\\,km; log); net income, income net "
         "of commuting costs of each income group from the transport "
         "model, averaged with the 2011 income distribution of the "
         "sub-place (log); calibrated amenity index (log). Column 3 weighs "
         "cells by their number of formal private households.", 0.7)

## MODEL COVARIATES, BY HOUSING TYPE AND POOLED BY SCENARIO (online
## appendix)

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


cols = {}
for t in (0, 1, 2):
    y, X = reg_data(B, t)
    keep = [k for k in range(len(REG_VARS))
            if not (t > 0 and REG_VARS[k] == "Log dwelling size")]
    cols[HTYPES[t]] = reg_column(ols(y, X[:, keep],
                                     [REG_VARS[k] for k in keep]))
reg_latex(reg_table(cols, REG_VARS), "online",
          "table_rent_regressions_by_housing_type")
MODEL_REG_NOTE = (
    " Regressors from the model: expected flood depth and its square; log "
    "mean income net of commuting costs of the occupants (household-"
    "weighted, net of lump-sum taxes; backyard revenues accrue to RDP "
    "owners, who pay no rent, so they enter no rent regression); log "
    "amenity index; log formal dwelling size; log number of households of "
    "the housing type in the cell; share of the cell available to the "
    "housing type. Informal disamenity indices are constants, absorbed by "
    "the constant (or the housing type dummies). Formal rents are bid "
    "rents that depend on income and dwelling size in closed form, hence "
    "the near-perfect fit.")
register("online", "table", "table_rent_regressions_by_housing_type", 40,
         "Benchmark equilibrium",
         "Rent regressions in the benchmark, by housing type",
         REG_NOTE + MODEL_REG_NOTE, 0.85)


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


POOLED_VARS = REG_VARS + ["Informal backyards", "Informal settlements"]
reg_latex(reg_table({SCEN_SHORT[s]: reg_column(ols(*pooled(sim[s]),
                                                   POOLED_VARS))
                     for s in SCENARIOS}, POOLED_VARS), "online",
          "table_rent_regressions_scenarios")
register("online", "table", "table_rent_regressions_scenarios", 100,
         "Policy scenarios",
         "Rent regressions, all rented housing types, by scenario",
         "As Table~\\ref{tab:table_rent_regressions_by_housing_type}, with "
         "formal private, informal backyard and informal settlement cells "
         "pooled and housing type dummies (formal private: reference), in "
         "each scenario.", True)

###############################################################################

# %% HOUSING TYPES AND DAMAGES ACROSS SCENARIOS (STACKED BARS)

print("SCENARIO BARS")


def bar_values(S, kind, groups, sub="exposed"):
    """(housing types, scenarios) values for income groups `groups`; sub:
    households in the subsample only (damages), or the subsample whose
    damages paid are shown (damages_relative). damages_relative: each
    type's contribution, its damages paid over the income of all the
    subsample's households of these groups, so that the stack is the
    damages paid by the subsample as a % of its income."""
    mask = SUBSAMPLES[sub]
    out = np.zeros((4, len(SCENARIOS)))
    for k, s in enumerate(SCENARIOS):
        d = S[s]
        if kind == "households":
            out[:, k] = d["hh"][:, groups].sum((1, 2)) / 1e3
        elif kind == "damages":
            inside = mask[:, None, :] if sub == "most" else 1
            out[:, k] = (d["dmg"][:, groups] * inside).sum((1, 2)) / 1e6
        else:
            m = mask[:, None, :]
            num = (d["dmg_borne"][:, groups] * m).sum((1, 2))
            den = (d["hh"][:, groups] * m
                   * y_bar[groups][None, :, None]).sum()
            out[:, k] = 100 * num / den if den > 0 else 0
    return out


PANEL_GROUPS = [[i] for i in range(4)] + [[0, 1, 2, 3]]
PANEL_NAMES = INCGRPS + ["All households"]


def make_bars(n):
    if n == 1:
        fig, ax = plt.subplots(figsize=(6.6, 3.9))
        return fig, [ax]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    fig._legend_ax = axes.flat[5]
    return fig, list(axes.flat[:n])


def scen_bars_set(stem, section, kind, ylabel, view="post", sub="exposed",
                  doc=None):
    """Stacked bars by housing type across scenarios, one panel per income
    group and for all households, with linear y axes specific to each
    panel; damages paid: pies of the share of the group's households in
    the subsample. Builds over housing types and scenarios."""
    S = SIM[view]
    values = [bar_values(S, kind, g, sub) for g in PANEL_GROUPS]
    x = np.arange(len(SCENARIOS))
    pies = kind == "damages_relative"

    def draw(ax, k, vis):
        nt, ns = vis.get("types", 4), vis.get("scenarios", len(SCENARIOS))
        v = values[k][:, :ns].copy()
        v[nt:] = 0
        bars2d(ax, x[:ns], np.full(ns, 0.62), v, HTYPE_COLORS)
        top = values[k].sum(0).max()
        ax.set_ylim(0, 1.08 * top if top > 0 else 1)
        if pies:
            band = blended_transform_factory(ax.transData, ax.transAxes)
            for j, s in enumerate(SCENARIOS[:ns]):
                share = (share_of_all(S[s], SUBSAMPLES[sub],
                                      groups=PANEL_GROUPS[k])
                         / share_of_all(S[s], None, groups=PANEL_GROUPS[k])
                         * 100)
                pie = ax.inset_axes([j - 0.3, 1.0, 0.6, 0.12],
                                    transform=band)
                pie.pie([share, 100 - share], colors=[INK2, GRID],
                        startangle=90, counterclock=False)
                pie.set_aspect("equal")
        ax.set_xticks(x, [SCEN_SHORT[s] for s in SCENARIOS], rotation=35,
                      ha="right", fontsize=8)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_title(PANEL_NAMES[k], pad=26 if pies else 3)
        ax.set_ylabel(ylabel)
        style_axes(ax)

    def finish(fig, axes, panels, vis):
        handles = legend_handles(HTYPE_COLORS, HTYPES)
        if pies:
            handles.append(Patch(facecolor=INK2, label=(
                "Pies: share of the group's households "
                + ("exposed" if sub == "exposed" else "most exposed"))))
        if len(axes) == 1:
            fig.tight_layout()
            axes[0].legend(handles=handles, loc="upper left",
                           bbox_to_anchor=(1.02, 1), fontsize=8,
                           title="Housing type")
        else:
            fig._legend_ax.axis("off")
            fig._legend_ax.legend(handles=handles, loc="center left",
                                  fontsize=9, title="Housing type")
            fig.tight_layout(h_pad=1.5, w_pad=1.0)

    figure_set(stem + ("_above_5cm" if sub == "most" else "")
               + VIEW_SUFFIX[view], section, len(PANEL_GROUPS),
               make_bars, draw, finish, names=PANEL_NAMES,
               builds=[("types", HTYPES, 0),
                       ("scenarios", [SCEN_SHORT[s] for s in SCENARIOS], 1)],
               doc=doc)


BARS_NOTE = (" By income group and for the whole city, stacked by housing "
             "type, across scenarios; linear y axes specific to each "
             "panel.")
scen_bars_set("scenarios_bars_households_by_housing_type", "online",
              "households", "Households (thousands)",
              doc=(90, "Policy scenarios",
                   "Households by housing type across scenarios",
                   "Number of households (thousands)." + BARS_NOTE, 0.85))
for sub in ("exposed", "most"):
    off = 0 if sub == "exposed" else 2
    for view in VIEWS:
        o = off + (view == "ante")
        within = ("" if sub == "exposed" else
                  f" of the most exposed households (above {MOST_TXT})")
        scen_bars_set(
            "scenarios_bars_damages_by_housing_type", "main", "damages",
            "Damages (M ZAR/year)", view, sub,
            doc=(110 + o, "Policy scenarios",
                 "Expected damages across scenarios"
                 + ("" if sub == "exposed" else
                    f", households above {MOST_TXT}") + VIEW_CAPTION[view],
                 "Actual expected damages to structures and contents "
                 f"(M ZAR/year){within}." + BARS_NOTE + VIEW_NOTE[view],
                 0.85))
        scen_bars_set(
            "scenarios_bars_damages_paid_by_housing_type", "main",
            "damages_relative", "Damages paid\n(% of income)", view, sub,
            doc=(120 + o, "Policy scenarios",
                 "Damages paid by "
                 + ("exposed" if sub == "exposed" else "most exposed")
                 + " households across scenarios" + VIEW_CAPTION[view],
                 "Damages "
                 + ("exposed" if sub == "exposed" else "most exposed")
                 + " households pay for (net of insurance payouts; no "
                 "formal private structures; backyard shacks paid by RDP "
                 "owners) as a \\% of their mean income; each housing "
                 "type's segment is its damages paid over the income of "
                 "all these households of the group, so that the stack is "
                 "the group's total. Pies: share of the group's households "
                 + ("exposed" if sub == "exposed" else
                    f"above {MOST_TXT}") + "." + BARS_NOTE + VIEW_NOTE[view],
                 0.85))

###############################################################################

# %% POLICY INTERACTIONS (FULL FACTORIAL, IF RUN)

print("POLICY INTERACTIONS")

# Levers in display order (scenario names follow main.py's lever order)
LEVERS = ["PP1", "PS1", "SI1", "RM0"]
MAIN_LEVERS = ["RM0", "PS1", "PP1", "SI1"]
LEVER_SHORT = dict(zip(LEVERS, ["Public P.", "Subs. SP", "Subs. ins.",
                                "No RM"]))
LEVER_SCEN = dict(zip(LEVERS, SINGLES))
INTER_NOTE = (
    "For each outcome $Y$, with $Y(S)$ its value when the set $S$ of policy "
    "levers is on (Public P.: public protection; Subs.\\ SP: free sandbags; "
    "Subs.\\ ins.: subsidised insurance, coverage 0.79 in every "
    "combination; No RM: no risk misperception) and $Y(\\emptyset)$ the "
    "benchmark: single effects $Y(j) - Y(\\emptyset)$; pairwise "
    "interactions $Y(jk) - Y(j) - Y(k) + Y(\\emptyset)$; Shapley values, "
    "each lever's mean marginal effect over all orders of introduction, "
    "which add up to the combined effect $Y(\\text{all}) - "
    "Y(\\emptyset)$.")


def scenario_of(subset):
    """Scenario label for a set of levers."""
    subset = [lever for lever in MAIN_LEVERS if lever in subset]
    if not subset:
        return "benchmark"
    if len(subset) == 1:
        return LEVER_SCEN[subset[0]]
    if len(subset) == 4:
        return "combined"
    return "+".join(subset)


if FACTORIAL:
    FACT_SIM = {"post": {}, "ante": {}}
    for label in FACTORIAL:
        d = load_scenario(label)
        FACT_SIM["post"][label], FACT_SIM["ante"][label] = d, exante(d)
    n_lev = len(LEVERS)
    for view in VIEWS:
        S = {**SIM[view], **FACT_SIM[view]}
        Y = {}
        for r in range(5):
            for subset in combinations(LEVERS, r):
                label = scenario_of(subset)
                Y[frozenset(subset)] = (METRICS[view][label]
                                        if label in METRICS[view]
                                        else metrics(S, label))
        empty = frozenset()

        def effect(subset, y, Y=Y):
            return Y[frozenset(subset)][y] - Y[empty][y]

        def shapley(names, Y=Y):
            rows = {}
            for y in names:
                row = {}
                for j in LEVERS:
                    others = [k for k in LEVERS if k != j]
                    row[LEVER_SHORT[j]] = sum(
                        factorial(r) * factorial(n_lev - r - 1)
                        / factorial(n_lev)
                        * (Y[frozenset(S_) | {j}][y] - Y[frozenset(S_)][y])
                        for r in range(n_lev)
                        for S_ in combinations(others, r))
                row["Combined"] = Y[frozenset(LEVERS)][y] - Y[empty][y]
                rows[y] = row
            return pd.DataFrame(rows).T

        ## Pairwise interactions: summary outcomes and aggregate EVs
        names = dict((key, label) for label, key, _ in summary_rows("exposed")
                     if key in INTER_KEYS)
        names.update({f"{sub}|{AGG_EV}": f"{AGG_EV}, "
                      + SUBSAMPLE_NAMES[sub].split(" (")[0].lower()
                      for sub in SUBSAMPLES})
        names[f"all|{AGG_EV_LANDLORDS}"] = (
            "Aggregate EV + landlords' net revenue (M ZAR), all households")
        rows = {}
        for y, label in names.items():
            row = {LEVER_SHORT[j]: effect([j], y) for j in LEVERS}
            for j, k in combinations(LEVERS, 2):
                row[f"{LEVER_SHORT[j]} x {LEVER_SHORT[k]}"] = (
                    effect([j, k], y) - effect([j], y) - effect([k], y))
            row["Combined"] = effect(LEVERS, y)
            row["Sum of singles"] = sum(effect([j], y) for j in LEVERS)
            rows[label] = row
        write_table(pd.concat({"": pd.DataFrame(rows).T}), "online",
                    "table_pairwise_interactions" + VIEW_SUFFIX[view],
                    lambda b, row, c, v: _num(
                        v, 3 if "(% of" in row or "depth" in row else 2,
                        sign=True),
                    doc=(110 + (view == "ante"), "Policy interactions",
                         "Single effects and pairwise interactions"
                         + VIEW_CAPTION[view],
                         INTER_NOTE + " Outcomes of the scenario summary "
                         "(Table~\\ref{tab:table_scenario_summary"
                         + VIEW_SUFFIX[view] + "}) and aggregate EVs of all, "
                         "exposed and most exposed households (M ZAR/year; "
                         "all households: also with the change in "
                         "landlords' net revenue). An interaction with the "
                         "sign of the two single effects' sum makes the "
                         "levers complements, the opposite sign "
                         "substitutes." + VIEW_NOTE[view]))

        ## Shapley values (user, 2026-11-19: low-income rows by housing
        ## type, most exposed above 5 cm/year, no landlords in the EV table)
        for kind, stem, order in (("Utility (%)", "table_shapley_utility",
                                   130),
                                  ("EV (ZAR/household/year)",
                                   "table_shapley_ev", 140)):
            blocks = {}
            for sub in SUBSAMPLES:
                share = (100 if sub == "all"
                         else share_of_all(B, SUBSAMPLES[sub]))
                header = f"{SUBSAMPLE_NAMES[sub]}, {share:.1f}% of households"
                keys = [f"{sub}|{kind}|{name}" for name in SHORT_ROWS]
                labels = list(ROWS)
                if kind.startswith("EV"):
                    keys.append(f"{sub}|{AGG_EV_HH}")
                    labels.append("Aggregate EV")
                df = shapley(keys)
                df.index = labels
                blocks[header] = df
            table = pd.concat(blocks)
            write_table(table, "main", stem + VIEW_SUFFIX[view],
                        lambda b, row, c, v, k=kind: _num(
                            v, 3 if k.startswith("Utility") else 1,
                            sign=True),
                        doc=(order + (view == "ante"), "Policy interactions",
                             ("Shapley decomposition of utility changes"
                              if kind.startswith("Utility") else
                              "Shapley decomposition of equivalent "
                              "variations") + VIEW_CAPTION[view],
                             INTER_NOTE + " One panel per subsample: all "
                             "households, exposed households (expected "
                             f"depth above {EXPOSURE_THRESHOLD:g}\\,cm/year) "
                             f"and most exposed households (above "
                             f"{MOST_TXT}); headers give the subsample's "
                             "share of all households in the benchmark. "
                             + ("Change in the mean utility of each row of "
                                "the welfare tables (\\%)."
                                if kind.startswith("Utility") else
                                "EV of each row of the welfare tables "
                                "(ZAR/year per benchmark household of the "
                                "row and subsample). Aggregate EV: EVs "
                                "weighted by the rows' shares of the "
                                "benchmark subsample, i.e.\\ per household "
                                "of the subsample (ZAR/year), without "
                                "landlords' revenues.")
                             + VIEW_NOTE[view]))
            print(table.round(3).to_string())

        ## Interaction matrices (diagonal: single effects; off-diagonal:
        ## pairwise interactions), colors oriented so that blue improves
        ## the outcome
        KEY = [("hh_dmg", -1, "Household damages (M ZAR/year)"),
               ("struct", -1, "Structure damages, formal private "
                              "(M ZAR/year)"),
               ("share", -1, "Exposed households (% of all households)"),
               ("protected", 1, "Exposed settlers protected (%)"),
               ("depth", -1, "Mean expected flood depth, exposed "
                             "(cm/year)"),
               (f"all|{AGG_EV}", 1, "Aggregate EV, all households "
                                    "(M ZAR/year)"),
               (f"exposed|{AGG_EV}", 1, "Aggregate EV, exposed households "
                                        "(M ZAR/year)"),
               (f"most|{AGG_EV}", 1, "Aggregate EV, most exposed "
                                     "households (M ZAR/year)")]

        def make_matrices(n):
            if n == 1:
                fig, ax = plt.subplots(figsize=(4.6, 4.4))
                return fig, [ax]
            fig, axes = plt.subplots(3, 3, figsize=(13.5, 13))
            fig._legend_ax = axes.flat[8]
            return fig, list(axes.flat[:n])

        def draw_matrix(ax, k, vis, effect=effect):
            y, better, label = KEY[k]
            M = np.full((n_lev, n_lev), np.nan)
            for a, j in enumerate(LEVERS):
                M[a, a] = effect([j], y)
                for b, kk in enumerate(LEVERS):
                    if b > a:
                        M[a, b] = M[b, a] = (effect([j, kk], y)
                                             - effect([j], y)
                                             - effect([kk], y))
            lim = np.nanmax(np.abs(M)) or 1
            ax.imshow(better * M, cmap=DIVERGING,
                      norm=TwoSlopeNorm(0, -lim, lim))
            digits = 3 if lim < 0.1 else 2 if lim < 1 else 1 if lim < 10 \
                else 0
            for a in range(n_lev):
                for b in range(n_lev):
                    v = M[a, b]
                    ax.text(b, a, _num(round(v, digits), digits, sign=True), ha="center",
                            va="center", fontsize=9,
                            color="white" if abs(v) > 0.6 * lim else INK,
                            fontweight="bold" if a == b else "normal")
            ticks = [LEVER_SHORT[j] for j in LEVERS]
            ax.set_xticks(range(n_lev), ticks, rotation=30, ha="right")
            ax.set_yticks(range(n_lev), ticks)
            ax.set_title(f"{label}\ncombined "
                         f"{_num(round(effect(LEVERS, y), digits), digits, True)}, sum of "
                         "singles "
                         f"{_num(round(sum(effect([j], y) for j in LEVERS), digits), digits, True)}",
                         fontsize=8.5, pad=4)
            for side in ax.spines.values():
                side.set_visible(False)

        def finish_matrices(fig, axes, panels, vis):
            if len(axes) == 1:
                fig.tight_layout()
                return
            fig._legend_ax.axis("off")
            fig._legend_ax.legend(handles=[
                Patch(facecolor="#184f95", label="Improves the outcome\n"
                      "(off-diagonal: levers reinforce\neach other)"),
                Patch(facecolor="#b02a2a", label="Worsens the outcome\n"
                      "(off-diagonal: levers overlap)"),
                Patch(facecolor="white", edgecolor=INK2,
                      label="Diagonal (bold): single effects\nOff-diagonal:"
                      " pairwise interactions")], loc="center left",
                fontsize=9)
            fig.tight_layout(h_pad=2, w_pad=1.5)

        figure_set("interaction_matrices" + VIEW_SUFFIX[view], "online",
                   len(KEY), make_matrices, draw_matrix, finish_matrices,
                   names=[k[2].split(" (")[0] for k in KEY],
                   doc=(120 + (view == "ante"), "Policy interactions",
                        "Single effects and pairwise interactions"
                        + VIEW_CAPTION[view],
                        "Diagonal (bold): single effects relative to the "
                        "benchmark; off-diagonal: pairwise interactions. "
                        "Colors are oriented so that blue improves the "
                        "outcome (fewer damages or exposed households, "
                        "lower mean depth of exposed households; more "
                        "protection or welfare): off the diagonal, blue "
                        "means that the levers reinforce each other, red "
                        "that they overlap. Aggregate EV: sum of EVs over "
                        "each subsample, without landlords' revenues "
                        "(M ZAR/year)." + VIEW_NOTE[view], 1.0))

###############################################################################

# %% DIAGNOSTICS

print("DIAGNOSTICS")
for s in SCENARIOS + CHECKS:
    d = sim[s]
    print(f"  {SCEN_LABELS[s]:>28s}: max population error "
          f"{100 * np.abs(d['error']).max():.2f}%, insurance share "
          f"{SCEN['scenarios'][s]['insurance_share']}, vacant RDP units "
          f"{d['rdp_vacant'].sum():,.0f}")
for view in VIEWS:
    for sub in SUBSAMPLES:
        print(f"  aggregate EV ({view}, {sub}, M ZAR):",
              {SCEN_SHORT[s]: round(METRICS[view][s][f"{sub}|{AGG_EV}"], 1)
               for s in SCENARIOS[1:]})

###############################################################################

# %% PDF REPORT (pdf/figures_tables.tex, from the registry)

print("PDF REPORT")

INTRO = {
    "main": (BENCH_NOTE + " Exposed households live where the expected "
             f"flood depth exceeds {EXPOSURE_THRESHOLD:g}\\,cm/year; the "
             f"most exposed, where it exceeds {MOST_TXT}. Expected flood "
             "depth is measured before any protection investment. "
             + SCEN_ORDER_NOTE + " Ex-ante versions use the damages agents "
             "anticipate in equilibrium and equilibrium utilities; the "
             "others, the actual damages and utilities once they are "
             "realised."),
    "appendix": "", "online": ""}
TEX_HEAD = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage[section]{placeins}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\captionsetup[sub]{font=footnotesize,skip=2pt}
\graphicspath{{../}}
\setlength{\parindent}{0pt}
\newcommand{\note}[1]{\par\vspace{3pt}\begin{minipage}{\linewidth}\footnotesize\textit{Notes:} #1\end{minipage}}

\title{Flood risks and policy scenarios in Cape Town:\\ tables and figures}
\date{}

\begin{document}
\maketitle
\tableofcontents
\clearpage
"""


def tex_entry(e):
    sec = SECTIONS[e["section"]]
    label = e.get("label") or e["stem"]
    if e["kind"] == "table":
        width = e["width"]
        size = (r"\textwidth" if width is True or width is None
                else rf"{width}\textwidth")
        body = (rf"\resizebox{{{size}}}{{!}}{{\input{{../{sec}/tables/"
                rf"{e['stem']}.tex}}}}")
        return "\n".join([r"\begin{table}[!htbp]", r"\centering",
                          rf"\caption{{{e['caption']}}}",
                          rf"\label{{tab:{label}}}", body,
                          rf"\note{{{e['note']}}}", r"\end{table}", ""])
    if e.get("parts"):
        parts = []
        for k, (stem, sub) in enumerate(e["parts"]):
            parts += [r"\begin{subfigure}{0.49\textwidth}",
                      rf"\includegraphics[width=\textwidth]{{{sec}/figures/"
                      rf"{stem}.png}}", rf"\caption{{{sub}}}",
                      r"\end{subfigure}" + (r"\hfill" if k == 0 else "")]
        body = "\n".join(parts)
    else:
        body = (rf"\includegraphics[width={e['width'] or 0.9}\textwidth]"
                rf"{{{sec}/figures/{e['stem']}.png}}")
    return "\n".join([r"\begin{figure}[!htbp]", r"\centering", body,
                      rf"\caption{{{e['caption']}}}",
                      rf"\label{{fig:{label}}}", rf"\note{{{e['note']}}}",
                      r"\end{figure}", ""])


def write_tex():
    lines = [TEX_HEAD]
    for section in SECTIONS:
        entries = sorted([e for e in REGISTRY if e["section"] == section],
                         key=lambda e: e["order"])
        lines += ["% " + "=" * 69,
                  rf"\section{{{SECTION_TITLES[section]}}}", ""]
        if INTRO[section]:
            lines += [INTRO[section], ""]
        sub = None
        for e in entries:
            if e["sub"] != sub:
                sub = e["sub"]
                lines += [r"\FloatBarrier", rf"\subsection{{{sub}}}", ""]
            lines.append(tex_entry(e))
        lines.append(r"\clearpage")
    lines.append(r"\end{document}")
    (PATH_PDF / "figures_tables.tex").write_text("\n".join(lines),
                                                 encoding="utf-8")
    print(f"  pdf/figures_tables.tex: {len(REGISTRY)} entries")


orders = [(e["section"], e["order"]) for e in REGISTRY]
assert len(orders) == len(set(orders)), "duplicate PDF positions"
write_tex()
print(f"Done in {time.time() - T0:.0f} s")

# %%
