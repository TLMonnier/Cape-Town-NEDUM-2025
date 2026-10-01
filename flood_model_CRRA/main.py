# ##########################################################
# ## FLOOD MODEL WITH RISK AVERSION: INTERACTIVE SCRIPT ###
# ##########################################################
# Run cell by cell (Shift+Enter) in the VS Code Interactive Window, or as a
# plain script. Edit options and parameters in the second cell. Uses the
# modules of ../flood_model (inputs, damages, solver) and crra.py here.

# %% Setup (re-run after editing the flood_model or flood_model_CRRA modules)

import importlib
import sys
from pathlib import Path

import numpy as np

try:
    HERE = Path(__file__).resolve().parent
except NameError:                     # Interactive Window: no __file__
    HERE = Path.cwd()
    if not (HERE / "crra.py").exists():
        HERE = HERE / "flood_model_CRRA"
if not (HERE / "crra.py").exists():
    raise RuntimeError("Working directory must be python/ or python/"
                       "flood_model_CRRA/ (use os.chdir), not "
                       + str(Path.cwd()))
BASE = HERE.parent / "flood_model"
for _path in (BASE, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# (Re)load the modules so that code edits are taken into account (crra after
# solver, so that CRRAMarkets subclasses the reloaded solver.Markets)
_modules = []
for _name, _dir in [("config", BASE), ("floods", BASE), ("data", BASE),
                    ("solver", BASE), ("run", BASE), ("crra", HERE),
                    ("run_crra", HERE)]:
    _mod = sys.modules.get(_name)
    _file = getattr(_mod, "__file__", None)
    if _mod is not None and (not _file
                             or Path(_file).resolve().parent != _dir):
        del sys.modules[_name]        # same name, other package
        _mod = None
    _modules.append(importlib.reload(_mod) if _mod
                    else importlib.import_module(_name))
config, floods, data, solver, run, crra, run_crra = _modules

# %% Options and parameters

# Flood options (0/1), as in flood_model
options = dict(config.OPTIONS)
# AF: agents internalise flood damages
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

# Parameters (any entry of config.PARAM or crra.PARAM can be overridden)
param = {**config.PARAM, **crra.PARAM}
param["CRRA"] = 0.2772                # relative risk aversion θ
param["risk_internaliz"] = 0.36
param["sandbag_course_cost"] = 0      # config default: 250
param["risk_increase"] = 2
param["max_iter"] = 1000
# Solver returns the iterate with the lowest error (False: last iterate)
param["return_best"] = True

# Suffix for output file names, to keep parameter variants apart
# (names only encode options), e.g. "_CRRA2" for param["CRRA"] = 2
tag = ""

name = config.simulation_name(options, tag)
print(name)

# %% Inputs and flood damages (raw files are cached after the first run)

p, inputs = data.prepare_inputs(param)
damages = floods.compute_damages(inputs, options, p)

# %% Equilibrium (about 1 min)

markets = crra.CRRAMarkets(p, inputs, damages, options)
outputs, converged = solver.solve(markets)

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

# %% Save outputs and flood type masks (to Output/flood_model_CRRA)

run.save(outputs, options, tag, folder=crra.OUTPUT, damages=damages)
print("Saved", name, "to", crra.OUTPUT)

# %%
