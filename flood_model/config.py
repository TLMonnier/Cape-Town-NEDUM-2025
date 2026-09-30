"""Paths, options and parameters of the flood model.

Options and parameters that are not flood-related are hard-wired to the
values used in the legacy flood_script.py (urban edge, no new RDP housing,
informal land constrained, no amenity upgrading / subsidies / eviction /
incremental housing), and simplified away in the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # cape_town_perso/
DATA = ROOT / "Data"
OUTPUT = ROOT / "Output" / "flood_model"
CACHE = Path(__file__).resolve().parent / "_cache"

# Flood-related switches (0/1)
OPTIONS = {
    # AF: households internalise expected flood damages in their bids
    "agents_anticipate_floods": 1,
    # CC: fluvial/pluvial flood probabilities are multiplied by
    # param["risk_increase"], coastal floods use sea-level-rise maps
    "climate_change": 0,
    # CO: coastal floods (DELTARES) on top of fluvial and pluvial floods
    "coastal": 1,
    # RM: perceived damages = param["risk_internaliz"] x actual damages
    "risk_misperc": 0,
    # SP: informal settlers may buy sandbag protection (up to 15cm)
    "self_protec": 0,
    # RA: households maximise CRRA expected utility over flood states (bid
    # rents, protection choice, backyard supply); developers risk neutral
    "risk_avers": 0,
}

PARAM = {
    # Flood-related parameters
    "risk_internaliz": 0.36,       # share of flood risk perceived under RM
    "sandbag_course_cost": 250,    # annual cost of protection (rands)
    "CRRA": 0.2772,                # relative risk aversion under RA
    "risk_increase": 2,            # probability multiplier under CC
    "protec_depth": 0.15,          # flood depth (m) stopped by sandbags
    # Solver
    "max_iter": 200,
    "precision": 0.001,
    "utility_init": [1200, 4800, 16000, 77000],
    # Return the iterate with the lowest max abs error (False: last iterate)
    "return_best": True,
    # Stop after this many iterations without improvement while errors flip
    # sign (None: run to max_iter; early stops tend to miss later, better
    # iterates, as the step size keeps shrinking)
    "patience": None,
    # Structural parameters (see legacy inputs/parameters_and_options.py)
    "depreciation_rate": 0.025,
    "shack_size": 14,              # m2, informal dwelling
    "RDP_size": 40,                # m2, formal subsidized dwelling
    "backyard_size": 70,           # m2, RDP backyard
    "informal_structure_value": 3000,
    "subsidized_structure_value": 127000,
    "fraction_z_dwellings": 0.27,  # share of composite good exposed to floods
    "max_land_use": 0.7,
    "max_land_use_backyard": 0.45,
    "max_land_use_settlement": 0.4,
    "historic_radius": 6,          # km
    "limit_height_center": 80,
    "limit_height_out": 10,
    "agricultural_price_baseline": 807.2,
    "threshold": 130,              # m2, formal dwellings above have 2 floors
    # Map from the 12 income brackets in the data to the 4 model groups
    "income_distribution": [0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4],
}

# 2011 household counts per housing type (SAL data, hard-coded in legacy
# inputs.data.import_macro_data)
TOTAL_RDP = 194258
# TOTAL_FORMAL = 821028 - TOTAL_RDP
TOTAL_FORMAL = 810818 - TOTAL_RDP
TOTAL_BACKYARD = 91132
TOTAL_INFORMAL = 143765
POPULATION = TOTAL_FORMAL + TOTAL_BACKYARD + TOTAL_INFORMAL + TOTAL_RDP
# Formal (brick) vs informal (shack) backyard structures, used to weight
# backyard structural damages
BACKYARDS_FORMAL = 16216
BACKYARDS_INFORMAL = 74916

# Housing market access by income group (poorest to richest)
ACCESS = {
    "formal": [1, 1, 1, 1],
    "backyard": [1, 1, 0, 0],
    "informal": [1, 1, 0, 0],
}


def simulation_name(options, tag=""):
    return "simul_AF{agents_anticipate_floods}_CC{climate_change}" \
           "_CO{coastal}_RM{risk_misperc}_SP{self_protec}" \
           "_RA{risk_avers}".format(**options) + tag
