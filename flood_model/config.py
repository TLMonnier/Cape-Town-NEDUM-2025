"""Paths, options and parameters of the flood model.

Notation follows the working paper main.tex (Section 4, "Model"). Code names
map to the paper's symbols as follows (values from the tables "Chosen
parameters" and "Estimated parameters" of main.tex Section 5):

    code (PARAM / data.prepare_inputs)    paper     value
    alpha, beta = 1 - alpha                α, 1-α    0.747, 0.253
    q0                                     q_0       3.97 m2
    amenities (data)                       A(x)      mean 1
    pocket_informal, pocket_backyard       B_IS, B_IB  0.78, 0.80
    net_income (data)                      ỹ_i(x)    rands/year
    fraction_z_dwellings                   γ         0.27
    depreciation_rate                      ρ         0.025
    interest_rate                          δ         0.043
    coeff_A                                κ         0.031
    coeff_a, coeff_b = 1 - coeff_a         a, 1-a    0.758, 0.242
    shack_size                             q_I       14 m2
    informal_structure_value               v_I       3,000 rands
    RDP_size + backyard_size               q_FS      40 + 70 = 110 m2
    backyard_size                          Y         70 m2
    subsidized_structure_value             v_FS      127,000 rands
    mini_lot_size                          q_min     31.6 m2
    agricultural_price_baseline            P_A       807.2 rands/m2
    coeff_land (data)                      L_h(x)    share of cell area

Housing types h: FP formal private ("formal" in the code), IB informal
backyards ("backyard"), IS informal settlements ("informal"), FS formal
subsidized, i.e. RDP housing ("RDP" or "subsidized"). Income groups
i = 1..4 in the paper are indexed 0..3 in the code.

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
    # AF: households and developers anticipate floods, i.e. internalise
    # expected flood damages (main.tex Section 6.1 compares AF1 to AF0)
    "agents_anticipate_floods": 1,
    # CC: climate change. Fluvial and pluvial flood probabilities are
    # multiplied by param["risk_increase"], and coastal floods use
    # sea-level-rise maps (main.tex Sections 3 and 6.2)
    "climate_change": 0,
    # CO: coastal floods (DELTARES) on top of fluvial and pluvial floods
    "coastal": 1,
    # RM: risk misperception. Agents perceive param["risk_internaliz"] x
    # actual damages (not in main.tex)
    "risk_misperc": 0,
    # SP: self-protection. Informal settlers may buy sandbags that stop
    # floods up to param["protec_depth"] (not in main.tex)
    "self_protec": 0,
}

PARAM = {
    # Flood-related parameters
    "risk_internaliz": 0.36,       # RM: share of flood risk perceived
    "sandbag_course_cost": 250,    # SP: annual cost of protection (rands)
    "protec_depth": 0.15,          # SP: flood depth (m) stopped by sandbags
    "risk_increase": 2,            # CC: flood probability multiplier
    # Solver (main.tex Section 4.6, "Determination of the equilibrium")
    "max_iter": 200,
    # Convergence when |simulated / target households - 1| <= precision for
    # every income group
    "precision": 0.001,
    "utility_init": [1200, 4800, 16000, 77000],
    # Return the iterate with the lowest max abs error (False: the last
    # iterate, as in the legacy code)
    "return_best": True,
    # Structural parameters (main.tex Table "Chosen parameters")
    "depreciation_rate": 0.025,    # ρ
    "shack_size": 14,              # q_I (m2), informal dwelling
    "RDP_size": 40,                # m2, RDP house (without its backyard)
    "backyard_size": 70,           # Y (m2), RDP backyard
    "informal_structure_value": 3000,         # v_I (rands)
    "subsidized_structure_value": 127000,     # v_FS (rands)
    "fraction_z_dwellings": 0.27,  # γ, share of composite good exposed
    "agricultural_price_baseline": 807.2,     # P_A (rands/m2)
    # Land use and height limits (main.tex Section 3, "Socioeconomic data":
    # land availability and height restrictions), see data.prepare_inputs
    "max_land_use": 0.7,
    "max_land_use_backyard": 0.45,
    "max_land_use_settlement": 0.4,
    "historic_radius": 6,          # km
    "limit_height_center": 80,
    "limit_height_out": 10,
    # Formal private dwellings above this size (m2) have two floors, and a
    # lower structural damage function (not in main.tex)
    "threshold": 130,
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

# Housing market access by income group (poorest to richest): main.tex
# Section 4.1, groups 3 and 4 are never observed in informal housing (and
# only group 1 gets RDP housing)
ACCESS = {
    "formal": [1, 1, 1, 1],
    "backyard": [1, 1, 0, 0],
    "informal": [1, 1, 0, 0],
}


def simulation_name(options, tag=""):
    return "simul_AF{agents_anticipate_floods}_CC{climate_change}" \
           "_CO{coastal}_RM{risk_misperc}_SP{self_protec}".format(
               **options) + tag
