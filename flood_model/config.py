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
    "risk_misperc": 1,
    # SP: self-protection (not in main.tex), as the number of sandbag levels
    # informal settlers can build: 0 (none), 1, 2 or 3. Each household
    # chooses k = 0, ..., SP levels; k levels raise its floor by
    # k x param["sandbag_height"] and cost k x param["sandbag_course_cost"]
    "self_protec": 3,
    # PS: subsidised self-protection (not in main.tex, requires SP >= 1).
    # Sandbags are free for settlers, and the scheme is financed by
    # lump-sum taxes on TAXED_GROUPS, proportional to their mean income
    # (budget balanced in equilibrium, see solver.solve)
    "subsid_protec": 0,
    # SI: subsidised insurance (not in main.tex), 0 or 1. Households of
    # INSURED_GROUPS (groups 1-2) are reimbursed a share s of the damages
    # to their own assets (contents, settlers' shacks, RDP houses and
    # backyard shacks; not developers' structures) at no premium, financed
    # like PS. s is the share of settlers' damages averted by sandbags
    # bought at full cost, at their locations, in the same scenario without
    # insurance (run.insurance_terms). Insured households only internalise
    # uninsured damages; settlers may still buy sandbags at full cost (SP).
    "subsid_insur": 0,
    # PP: public protection (not in main.tex), 0 or 1. Every dwelling, in
    # every housing type, is protected against every flood type up to
    # H = param["public_protection_height"]: the water depth falls to
    # max(d - H, 0), like a raised floor. Free (counterfactual, no cost);
    # settlers' sandbags (SP) raise their floor further above H
    "public_protec": 0,
}

PARAM = {
    # Flood-related parameters
    "risk_internaliz": 0.36,       # RM: share of flood risk perceived
    "sandbag_course_cost": 25,    # SP: annual cost per level (rands)
    "sandbag_height": 0.15,        # SP: floor elevation per level (m)
    "public_protection_height": 0.45,   # PP: protection height H (m)
    # AF0, RM1: developers' unanticipated structure losses are financed by a
    # lump-sum tax on developers, anticipated ex ante (solver.solve; False
    # in the legacy check)
    "developer_loss_tax": True,
    # RDP owners' utility floor: the government pays the part of their own
    # expected flood damages that would push their utility below u_1, the
    # equilibrium utility of group 1, financed by absentee landlords
    # (solver.Markets.rdp_transfer, accounting.ex_post; False in the legacy
    # check and in flood_model_CRRA)
    "rdp_utility_floor": True,
    # PS, SI, developers' tax: budget balance, relative tolerance and max
    # fixed-point rounds
    "budget_tol": 0.001,
    "max_iter_budget": 10,
    # SI: coverage share s; None: computed from the reference run without
    # insurance (run.insurance_terms)
    "insurance_share": None,
    # SI: sandbag levels in the reference run if the scenario has SP0
    "insurance_reference_levels": 3,
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
# PS: income groups paying the lump-sum taxes that finance free sandbags
# (the two richest, only found in formal housing)
TAXED_GROUPS = (2, 3)
# SI: income groups covered by the subsidised insurance (the two poorest;
# must include every group with access to backyards and settlements)
INSURED_GROUPS = (0, 1)


def simulation_name(options, tag=""):
    """Output name; "_PP1" is only appended under public protection, so
    that names without it are unchanged."""
    return "simul_AF{agents_anticipate_floods}_CC{climate_change}" \
           "_CO{coastal}_RM{risk_misperc}_SP{self_protec}" \
           "_PS{subsid_protec}_SI{subsid_insur}".format(**options) \
           + ("_PP1" if options.get("public_protec") else "") + tag
