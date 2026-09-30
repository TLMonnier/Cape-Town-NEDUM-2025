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
for _name in ("config", "floods", "data", "solver", "run"):
    _mod = sys.modules.get(_name)
    _file = getattr(_mod, "__file__", None)
    if _mod is not None and (not _file
                             or Path(_file).resolve().parent != HERE):
        del sys.modules[_name]        # same name, other package
        _mod = None
    _modules.append(importlib.reload(_mod) if _mod
                    else importlib.import_module(_name))
config, floods, data, solver, run = _modules

# %% Options and parameters

# Flood options (0/1)
options = dict(config.OPTIONS)
# AF: households internalise expected flood damages
options["agents_anticipate_floods"] = 1
# CC: fluvial/pluvial probabilities x param["risk_increase"], coastal
# floods from sea-level-rise maps
options["climate_change"] = 0
# CO: coastal floods (DELTARES) on top of fluvial and pluvial floods
options["coastal"] = 1
# RM: perceived damages = param["risk_internaliz"] x actual damages
options["risk_misperc"] = 0
# SP: settlers may buy sandbag protection (floods <= 0.15m)
options["self_protec"] = 0
# RA: households maximise CRRA expected utility over flood states (bid
# rents, protection, backyard supply); about 1 min instead of 2 s
options["risk_avers"] = 0

# Parameters (any entry of config.PARAM can be overridden here)
param = dict(config.PARAM)
param["risk_internaliz"] = 0.36
param["sandbag_course_cost"] = 0      # config default: 250
param["CRRA"] = 0.2772
param["risk_increase"] = 2
param["max_iter"] = 1000
# Solver returns the iterate with the lowest error; optional early stop
# after `patience` non-improving iterations with flipping errors
param["return_best"] = True
param["patience"] = None

# Suffix for output file names, to keep parameter variants apart
# (names only encode options), e.g. "_SB0" for a zero sandbag cost
tag = ""

name = config.simulation_name(options, tag)
print(name)

# %% Inputs and flood damages (raw files are cached after the first run)

p, inputs = data.prepare_inputs(param)
damages = floods.compute_damages(inputs, options, p)

# %% Equilibrium

outputs, converged = solver.solve(p, inputs, damages, options)

# %% Quick look

hh = outputs["households"]           # housing type x income group x cell
protected = np.nan_to_num(outputs["mask_self_protec"]) == 1
print("Utility by income group:", np.round(outputs["utility"], 1))
print("Error by income group (%):", np.round(100 * outputs["error"], 2))
for t, label in enumerate(["formal", "backyard", "informal", "RDP"]):
    print(f"{label:>9s} households: {np.nansum(hh[t]):>10,.0f}  by group",
          np.round(np.nansum(hh[t], 1)).astype(int))
print(f"Protected settler households: {np.nansum(hh[2][:, protected]):,.0f}"
      f" ({protected.sum()} protected cells)")

# Flood type with the highest expected damage (damages["flood_type"] holds
# one mask per damage series, also for state damages and protection)
selected = damages["flood_type"]["expected"]["structure_informal_settlements"]
print("Cells by selected flood type (settlement structures):",
      {kind: int(np.sum(selected == i))
       for i, kind in enumerate(floods.FLOOD_TYPES)},
      "| settler households:",
      {kind: int(np.nansum(hh[2][:, selected == i]))
       for i, kind in enumerate(floods.FLOOD_TYPES)})

# %% Save outputs and flood type masks (to Output/flood_model)

run.save(outputs, options, tag, damages=damages)
print("Saved", name, "to", config.OUTPUT)

# %%
