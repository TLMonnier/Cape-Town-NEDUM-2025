# ############################################
# ## FLOOD MODEL: INTERACTIVE MAIN SCRIPT ###
# ############################################
# Run cell by cell (Shift+Enter) in the VS Code Interactive Window, or as a
# plain script. Edit options and parameters in the second cell.

# %% Setup (re-run after editing the flood_model modules)

import importlib
import sys
from pathlib import Path

import numpy as np

try:
    HERE = Path(__file__).resolve().parent
except NameError:                     # Interactive Window: no __file__
    HERE = Path.cwd()
    if not (HERE / "solver.py").exists():
        HERE = HERE / "flood_model"
if not (HERE / "solver.py").exists():
    raise RuntimeError("Working directory must be python/ or python/"
                       "flood_model/ (use os.chdir), not " + str(Path.cwd()))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# (Re)load the package modules so that code edits are taken into account
_modules = []
for _name in ("config", "floods", "data", "accounting", "solver", "run",
              "scenarios"):
    _mod = sys.modules.get(_name)
    _file = getattr(_mod, "__file__", None)
    if _mod is not None and (not _file
                             or Path(_file).resolve().parent != HERE):
        del sys.modules[_name]        # same name, other package
        _mod = None
    _modules.append(importlib.reload(_mod) if _mod
                    else importlib.import_module(_name))
config, floods, data, accounting, solver, run, scenarios = _modules

# %% Options and parameters

# Flood options (0/1)
options = dict(config.OPTIONS)
# AF: agents internalise expected flood damages
options["agents_anticipate_floods"] = 1
# CC: fluvial/pluvial probabilities x param["risk_increase"], coastal
# floods from sea-level-rise maps
options["climate_change"] = 0
# CO: coastal floods (DELTARES) on top of fluvial and pluvial floods
options["coastal"] = 1
# RM: perceived damages = param["risk_internaliz"] x actual damages
# NB: need to introduce tax on developers to balance budget when option is selected
options["risk_misperc"] = 1
# SP: number of sandbag levels settlers can build (0: no self-protection,
# 1, 2 or 3); each level raises the floor by param["sandbag_height"] (15 cm)
# and costs param["sandbag_course_cost"] a year

# Maybe add a word on why it is hard to price insurance for poor households
options["self_protec"] = 3
# PS: free sandbags for settlers (requires SP >= 1), financed by lump-sum
# taxes on the two richest groups, proportional to their mean income

# Some "moral hazard" wrt location choice (expected)
options["subsid_protec"] = 0
# SI: subsidised insurance of groups 1-2's own assets (0/1), financed like
# PS: coverage share s = share of settlers' damages averted by sandbags at
# full cost, at their locations, in the same scenario without insurance
# (insured agents only internalise uninsured damages; sandbags still
# possible)

# NB: here, 2 is defined wrt income gains for the whole group (not just exposed population),
# hence leads to lower coverage...
# We do find some "moral hazard" wrt self-protection take-up (still relevant)?
# Even more for formal households!
# No utility changes wrt risk-based insurance when rich households need to pay taxes
# Note that keeping self-protection allows informal damages to decrease (not increase)
# NB: baseline should be defined for subsidized self-protection vs. basic self-protection
# (not no self-protection)
# NB: do loop for all coverages?
options["subsid_insur"] = 1
# PP: free public protection of every dwelling up to
# param["public_protection_height"] (45 cm): water depth max(d - 0.45, 0);
# sandbags (SP) add on top
options["public_protec"] = 0

# Land value capture vs mandatory insurance when introducing public protection?

# Parameters (any entry of config.PARAM can be overridden here)
param = dict(config.PARAM)
param["risk_internaliz"] = 0.36
param["sandbag_course_cost"] = 25      # config default: 250
param["risk_increase"] = 2
param["max_iter"] = 1000
# Solver returns the iterate with the lowest error (False: last iterate)
param["return_best"] = True

# Suffix for output file names, to keep parameter variants apart
# (names only encode options), e.g. "_SB0" for a zero sandbag cost
tag = ""

name = config.simulation_name(options, tag)
print(name)

# %% Inputs and flood damages (raw files are cached after the first run)

p, inputs = data.prepare_inputs(param)
# SI1: coverage share s from a reference run without insurance (sandbags at
# full cost), unless set in param["insurance_share"]
p = run.insurance_terms(p, inputs, options)
damages = floods.compute_damages(inputs, options, p)

# %% Equilibrium

markets = solver.Markets(p, inputs, damages, options)
outputs, converged = solver.solve(markets)

# %% Quick look

hh = outputs["households"]           # housing type x income group x cell
# Sandbag levels chosen in each cell (0: none)
level = np.nan_to_num(outputs["mask_self_protec"]).astype(int)
protected = level >= 1
print("Utility by income group:", np.round(outputs["utility"], 1))
print("Error by income group (%):", np.round(100 * outputs["error"], 2))
for t, label in enumerate(["formal", "backyard", "informal", "RDP"]):
    print(f"{label:>9s} households: {np.nansum(hh[t]):>10,.0f}  by group",
          np.round(np.nansum(hh[t], 1)).astype(int))
print(f"Protected settler households: {np.nansum(hh[2][:, protected]):,.0f}"
      f" ({protected.sum()} protected cells)")
if "budget" in outputs:              # PS or SI
    sandbags, payouts, revenue = outputs["budget"]
    rate = 100 * outputs["tax"] / inputs["average_income"]   # same τ
    print(f"Public budget: sandbags {sandbags:,.0f} + insurance "
          f"{payouts:,.0f} rands/year, tax revenue {revenue:,.0f}")
    if "insurance_share" in outputs:
        print(f"Insurance coverage share of groups 1-2: "
              f"{outputs['insurance_share'][0]:.4f}")
    for i in config.TAXED_GROUPS:
        print(f"  group {i + 1} lump-sum tax: {outputs['tax'][i]:,.1f} "
              f"rands/year per household ({rate[i]:.4f}% of mean income)")

# Flood type with the highest expected damage (damages["flood_type"] holds
# one mask per damage series, also for state damages and protection)
selected = damages["flood_type"]["expected"]["structure_informal_settlements"]
print("Cells by selected flood type (settlement structures):",
      {kind: int(np.sum(selected == i))
       for i, kind in enumerate(floods.FLOOD_TYPES)},
      "| settler households:",
      {kind: int(np.nansum(hh[2][:, selected == i]))
       for i, kind in enumerate(floods.FLOOD_TYPES)})

# %% Self-protection take-up among settlers exposed to floods

# Exposed: cells with positive actual expected damage (without protection)
# to settlement structures or contents
exposed = ((damages["expected"]["structure_informal_settlements"] > 0)
           | (damages["expected"]["contents_informal"] > 0))
settlers_exposed = np.nansum(hh[2][:, exposed])
labels = ["No protection"] + [
    f"{k} level{'s' if k > 1 else ''} ({k * 100 * p['sandbag_height']:.0f} cm)"
    for k in range(1, options["self_protec"] + 1)]
take_up = np.array([np.nansum(hh[2][:, exposed & (level == k)])
                    for k in range(len(labels))])
share = 100 * take_up / max(settlers_exposed, 1)
print(f"Exposed settlers: {settlers_exposed:,.0f} households")
for label, n, s in zip(labels, take_up, share):
    print(f"  {label:>16s}: {s:5.1f}%  ({n:,.0f} households)")

if options["self_protec"]:
    import matplotlib.pyplot as plt

    # Ordinal blue ramp for 1-3 levels (light to dark), gray for none
    colors = ["#b5b3ad"] + ["#86b6ef", "#2a78d6", "#184f95"][
        :options["self_protec"]]
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(labels)))
    y = np.arange(len(labels))[::-1]           # "No protection" on top
    ax.barh(y, share, height=0.6, color=colors)
    for yi, s, n in zip(y, share, take_up):
        ax.text(s + 1, yi, f"{s:.1f}%  ({n:,.0f})", va="center",
                fontsize=9, color="#52514e")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of exposed settler households (%)", color="#52514e")
    ax.set_title(f"Sandbag levels chosen by exposed settlers\n{name}",
                 fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    plt.show()

# %% Comparison with the baseline (same options, no self-protection or
# insurance): damages, welfare, relocation and housing markets

# NB: need to clarify basleine definition (with or without risk mispereception?)
# Also add tables that focus on exposed population (which is also changing)?
# Which mean income is used for housing type?
# How is EV computed? Smaller than utility changes but makes sense wrt income gains...
# How are RDP owners dealt with? No moves for group 4 (or counted from 0)?
# Check computation of ex-post utilities

base_options = run.baseline_options(options)
base_damages = floods.compute_damages(inputs, base_options, p)
base_markets = solver.Markets(p, inputs, base_damages, base_options)
base_outputs, _ = solver.solve(base_markets, verbose=False)

scenarios.damage_table(accounting.household_damages(base_markets, base_outputs),
                       accounting.household_damages(markets, outputs),
                       inputs["average_income"])
scenarios.welfare_table(base_markets, base_outputs, markets, outputs)
scenarios.relocation_table(base_outputs, outputs)

# %% Save outputs and flood type masks (to Output/flood_model)

run.save(outputs, options, tag, damages=damages)
print("Saved", name, "to", config.OUTPUT)

# %%
