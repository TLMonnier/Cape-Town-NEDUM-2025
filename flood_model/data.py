"""Load model inputs (main.tex Sections 3 and 5).

Raw files are read once and cached into a single .npz (_cache/), which is
refreshed automatically when a source file is newer. prepare_inputs then
derives the solver inputs: calibrated parameters, interest and agricultural
rents, minimum lot size, target households per income group, land available
per housing type L_h(x) and height limits.
"""
import numpy as np
import pandas as pd
import scipy.io
from scipy.interpolate import interp1d

import config
from floods import DELTARES_RETURN_PERIODS, RETURN_PERIODS

CACHE_VERSION = 3              # bump when _read_raw() changes
N_PIXELS = 24014
AREA_PIXEL = 0.25e6            # m2 (500m x 500m grid cells)
CBD = (-53267.944572790904 / 1000, -3754855.1309322729 / 1000)  # km

_PRECALC = config.DATA / "precalculated_inputs"
_CT = config.DATA / "data_Cape_Town"
# FATHOM fluvial (undefended) and pluvial maps
_FATHOM_FILES = [config.DATA / "flood_maps" / f"{t}_{rp}yr.xlsx"
                 for t in ("FU", "P") for rp in RETURN_PERIODS]
# DELTARES coastal maps (MERITDEM), without / with sea-level rise
_DELTARES_FILES = [config.DATA / "flood_maps" / f"C_MERITDEM_{cc}_{rp:04d}.xlsx"
                   for cc in (0, 1) for rp in DELTARES_RETURN_PERIODS]
_FLOOD_FILES = _FATHOM_FILES + _DELTARES_FILES
_SOURCES = [
    _PRECALC / f for f in (
        "calibratedUtility_beta.npy", "calibratedUtility_q0.npy",
        "calibratedHousing_b.npy", "calibratedHousing_kappa.npy",
        "param_amenity_settlement.npy", "param_amenity_backyard.npy",
        "calibratedAmenities.npy", "data.mat")
] + [
    _CT / "grid_NEDUM_Cape_Town_500.csv",
    _CT / "Income_distribution_2011.csv",
    _CT / "Scenarios" / "Scenario_interest_rate_1.csv",
    config.DATA / "precalculated_transport" / "GRID_incomeNetOfCommuting_0.npy",
] + _FLOOD_FILES


def _read_raw():
    """Read raw input files into a flat dict of arrays."""
    raw = {}
    # Calibrated parameters (main.tex Sections 5.2-5.3) and amenities A(x)
    for key, f in [("beta", "calibratedUtility_beta"),
                   ("q0", "calibratedUtility_q0"),
                   ("coeff_b", "calibratedHousing_b"),
                   ("coeff_A", "calibratedHousing_kappa"),
                   ("pocket_informal", "param_amenity_settlement"),
                   ("pocket_backyard", "param_amenity_backyard"),
                   ("amenities", "calibratedAmenities")]:
        raw[key] = np.load(_PRECALC / f"{f}.npy")

    # RDP housing per grid cell, and Small Place (SP) census data
    mat = scipy.io.loadmat(_PRECALC / "data.mat")["data"]
    for key, field in [("rdp_count", "gridCountRDPfromGV"),
                       ("rdp_area", "gridAreaRDPfromGV"),
                       ("sp_dwelling_size", "spDwellingSize"),
                       ("sp_backyard", "spInformalBackyard"),
                       ("sp_informal", "spInformalSettlement"),
                       ("sp_total", "spTotalDwellings")]:
        raw[key] = mat[field][0][0].squeeze()

    # Grid: coordinates (m), and land (m2) that is urban, available for
    # construction within the urban edge, and in informal settlements
    grid = pd.read_csv(_CT / "grid_NEDUM_Cape_Town_500.csv", sep=";")
    for col in ("X", "Y", "urban", "unconstrained_UE", "informal"):
        raw["grid_" + col] = grid[col].to_numpy()

    # 2011 households and median income per income bracket
    inc = pd.read_csv(_CT / "Income_distribution_2011.csv")
    raw["hh_bracket"] = inc.Households_nb.to_numpy()
    raw["inc_bracket"] = inc.INC_med.to_numpy()

    rates = pd.read_csv(_CT / "Scenarios" / "Scenario_interest_rate_1.csv",
                        sep=";").dropna(subset=["real_interest_rate"])
    raw["rate_year"] = rates.Year_interest_rate.to_numpy()
    raw["rate"] = rates.real_interest_rate.to_numpy()

    # Expected income net of commuting costs ỹ_i(x) (main.tex Section 5.1),
    # (income groups, cells)
    raw["net_income"] = np.load(
        config.DATA / "precalculated_transport"
        / "GRID_incomeNetOfCommuting_0.npy")

    # Flood maps: (flood type [FU, P] or climate change [0, 1],
    # return period, pixel)
    for prefix, files, n_rp in [
            ("flood", _FATHOM_FILES, len(RETURN_PERIODS)),
            ("coastal", _DELTARES_FILES, len(DELTARES_RETURN_PERIODS))]:
        maps = [pd.read_excel(f) for f in files]
        shape = (2, n_rp, N_PIXELS)
        raw[prefix + "_depth"] = np.array(
            [m.flood_depth for m in maps]).reshape(shape)
        raw[prefix + "_prop"] = np.array(
            [m.prop_flood_prone for m in maps]).reshape(shape)
    return raw


def load_raw(refresh=False):
    """Return raw inputs, re-reading source files only if they changed."""
    cache = config.CACHE / f"raw_inputs_v{CACHE_VERSION}.npz"
    if (not refresh and cache.exists() and cache.stat().st_mtime
            >= max(f.stat().st_mtime for f in _SOURCES)):
        with np.load(cache) as f:
            return dict(f)
    print("Reading raw input files (cached afterwards)...")
    raw = _read_raw()
    config.CACHE.mkdir(exist_ok=True)
    np.savez(cache, **raw)
    return raw


def prepare_inputs(param, refresh=False):
    """Return (updated param, inputs dict) for the equilibrium solver."""
    raw = load_raw(refresh)
    p = dict(param)
    # Utility U = z^α (q - q0)^β A(x) B_h, β = 1 - α (main.tex (1)), and
    # housing production s = κ k^(1-a) (main.tex (3)), with
    # coeff_A = κ and coeff_b = 1 - a the capital elasticity
    for key in ("beta", "q0", "coeff_b", "coeff_A"):
        p[key] = raw[key].item()
    p["alpha"] = 1 - p["beta"]
    p["coeff_a"] = 1 - p["coeff_b"]
    # Informal disamenity indices B_IS, B_IB (main.tex Section 5.3)
    p["pocket_informal"] = raw["pocket_informal"].item()
    p["pocket_backyard"] = raw["pocket_backyard"].item()

    # Real interest rate δ: average over 2008-2010, negative values excluded
    # (main.tex Section 5.4, outdated, says a 4-year average)
    rates = interp1d(raw["rate_year"] - 2011, raw["rate"])(np.arange(-3, 0))
    rates[rates < 0] = np.nan
    p["interest_rate"] = np.nanmean(rates) / 100

    # Agricultural rent R_A: formal rent at which developers can just pay
    # the agricultural land price P_A (zero profit in main.tex (4)),
    #     R_A = (δ P_A)^a (ρ + δ)^(1-a) / (κ a^a (1-a)^(1-a)).
    # No formal housing is built where R_FP < R_A (city edge, main.tex
    # Section 4.6, condition (ii))
    a, b, r = p["coeff_a"], p["coeff_b"], p["interest_rate"]
    p["agricultural_rent"] = (
        p["agricultural_price_baseline"] ** a * r ** a
        * (p["depreciation_rate"] + r) ** b
        / (p["coeff_A"] * b ** b * a ** a))

    # Minimum formal lot size q_min (main.tex Section 4.2): smallest average
    # dwelling size among Small Places with less than 10% informal dwellings
    tot = raw["sp_total"]
    ok = tot != 0
    informal_share = (raw["sp_informal"][ok] + raw["sp_backyard"][ok]) / tot[ok]
    p["mini_lot_size"] = np.nanmin(
        raw["sp_dwelling_size"][ok][informal_share < 0.1])

    # Income groups: 12 data brackets -> 4 model groups
    groups = np.array(p["income_distribution"])
    hh, inc = raw["hh_bracket"], raw["inc_bracket"]
    households = np.array([hh[groups == j].sum() for j in range(1, 5)])
    average_income = np.array(
        [np.sum(inc[groups == j] * hh[groups == j]) for j in range(1, 5)]
        ) / households
    mean_income = np.sum(hh * inc) / np.sum(hh)
    # Initial step size of the utility iteration (solver.solve)
    p["convergence_factor"] = (
        0.01 * (np.nanmean(average_income) / mean_income) ** 0.01)

    # Target households N_i per group (main.tex Section 4.6, condition (iii))
    # in the endogenous housing types FP, IB, IS: rescale the census
    # distribution to the total population, then remove all RDP households
    # from the poorest group (only group 1 gets RDP housing)
    target = households * (config.POPULATION / households.sum())
    target[0] = max(target[0] - config.TOTAL_RDP, 0)

    # Land available for each housing type, L_h(x), as a share of the cell
    # area (main.tex Section 3, "Socioeconomic data"). Each RDP plot has a
    # house (RDP_size) and a backyard (backyard_size).
    # - FS: land covered by RDP houses. The footprint is divided by
    #   max_land_use to get the gross land taken by RDP houses, which is
    #   removed from land available for FP; the net footprint is used in
    #   coeff_land[3]
    rdp_area_share = (raw["rdp_area"] * p["RDP_size"]
                      / (p["backyard_size"] + p["RDP_size"])
                      / AREA_PIXEL) / p["max_land_use"]
    # - IB: backyards of RDP plots, at most the urban area of the cell
    backyard_share = np.fmax(np.fmin(
        raw["grid_urban"] / AREA_PIXEL,
        raw["rdp_area"] * p["backyard_size"]
        / (p["backyard_size"] + p["RDP_size"]) / AREA_PIXEL), 0)
    # - IS: tolerated informal settlement areas (exogenous)
    informal_share = raw["grid_informal"] / AREA_PIXEL
    # - FP: remaining land within the urban edge
    private = (raw["grid_unconstrained_UE"] / AREA_PIXEL - backyard_share
               - informal_share - rdp_area_share) * p["max_land_use"]
    private[private < 0] = 0
    # Only part of the land can be built on (roads, open spaces...)
    coeff_land = np.array([
        private,
        backyard_share * p["max_land_use_backyard"],
        informal_share * p["max_land_use_settlement"],
        rdp_area_share * p["max_land_use"]])

    # Height limit on formal housing supply, as a maximum floor area ratio:
    # 80 within historic_radius km of the CBD, 10 elsewhere (converted to
    # m2 of floor per km2 of land, the unit of housing supply)
    dist = (((raw["grid_X"] / 1000 - CBD[0]) ** 2
             + (raw["grid_Y"] / 1000 - CBD[1]) ** 2) ** 0.5)
    housing_limit = 1e6 * np.where(dist <= p["historic_radius"],
                                   p["limit_height_center"],
                                   p["limit_height_out"])

    rdp_count = raw["rdp_count"].astype(float)
    inputs = {
        # A(x), normalised to a mean of 1 ("centered around one")
        "amenities": raw["amenities"] / np.nanmean(raw["amenities"]),
        "net_income": raw["net_income"],
        "coeff_land": coeff_land,
        "housing_limit": housing_limit,
        # Exogenous RDP households N_1^FS(x), rescaled to the 2011 total
        "households_RDP": rdp_count * config.TOTAL_RDP / rdp_count.sum(),
        "target": target,
        "average_income": average_income,
        "flood_depth": raw["flood_depth"],
        "flood_prop": raw["flood_prop"],
        "coastal_depth": raw["coastal_depth"],
        "coastal_prop": raw["coastal_prop"],
    }
    return p, inputs
