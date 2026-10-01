"""Check flood_model against the legacy code (python/inputs, python/equilibrium).

Runs the legacy flood_script.py pipeline in-process, by default with the bug
fixes of flood_model applied in memory (the legacy files are not modified),
then the new model in legacy mode, and compares all outputs. Legacy mode means:
no coastal floods, flood types combined by the max of expected damages (legacy)
instead of per return period, and the last iterate returned instead of the
best one. The legacy code also gets config.TOTAL_FORMAL. With the default
fixes (see CLAUDE.md), both codes then agree exactly (to machine precision
under RM). The legacy code runs with risk_avers = 0: risk aversion lives in
../flood_model_CRRA and has no legacy equivalent.
"conv" is not a flood_model fix: it applies the damping the legacy solver
intended (worse convergence), for experiments only.

Usage (from any directory):
    conda run -n nedum-2025 python flood_model/tools/check_legacy.py 1001
    ... check_legacy.py 1101 --fixes pluvial      # subset of fixes
CONFIG digits are the AF, CC, RM, SP options.
"""
import argparse
import copy
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path(__file__).resolve().parents[1]
LEGACY = NEW.parent
FIXES = ("pluvial", "rm_backyard", "backyard_cap", "backyard_loss", "conv")
OUT_KEYS = ["utility", "error", "simulated_jobs", "households_housing_types",
            "household_centers", "households", "dwelling_size",
            "housing_supply", "rent", "rent_matrix", "capital_land",
            "average_income", "limit_city", "mask_self_protec"]

sys.path[:0] = [str(NEW), str(LEGACY)]
import config  # noqa: E402  (flood_model)
import floods  # noqa: E402
import run  # noqa: E402


def legacy_combine(by_type, proba):
    """Legacy combination of flood types: max of expected damages, and max
    of state damages, across types."""
    states = [floods.state_damages(d) for d in by_type]
    expected = [floods.expected_damage(proba, s) for s in states]
    return (floods.max_over_types(states)[0],
            floods.max_over_types(expected)[0],
            np.zeros(by_type[0].shape, dtype=np.int8))


def patched(module, replacements):
    """Copy of a module with source replacements (legacy file untouched)."""
    src = open(module.__file__, encoding="utf-8").read()
    for old, new in replacements:
        assert src.count(old) == 1, (module.__name__, old)
        src = src.replace(old, new)
    m = types.ModuleType(module.__name__ + "_patched")
    m.__file__ = module.__file__
    exec(compile(src, module.__file__, "exec"), m.__dict__)
    return m


def run_legacy(options, fixes):
    cwd = os.getcwd()
    os.chdir(LEGACY)                     # legacy paths are relative ('..')
    try:
        return _run_legacy(options, fixes)
    finally:
        os.chdir(cwd)


def _run_legacy(options_new, fixes):
    import inputs.data as inpdt
    import inputs.parameters_and_options as inpprm
    import equilibrium.compute_equilibrium as eqcmp
    import equilibrium.sub.compute_outputs as eqout
    import equilibrium.sub.functions_solver as eqsol

    folder = '../Data/'
    precalc, data_ct = folder + 'precalculated_inputs/', folder + 'data_Cape_Town/'
    options = inpprm.import_options()
    options.update(urban_edge=1, informal_land_constrained=1,
                   new_RDP_housing=0, amenity_upgrading=0, poor_subsidies=0,
                   eviction=0, incremental_housing=0, risk_avers=0,
                   **options_new)
    param = inpprm.import_param(precalc, options)
    param.update(risk_internaliz=0.36, sandbag_course_cost=250, max_iter=200)

    # Same 2011 formal housing total as flood_model
    data_repl = [("    total_formal = 821028 - total_RDP\n",
                  f"    total_formal = {config.TOTAL_FORMAL + config.TOTAL_RDP}"
                  " - total_RDP\n")]
    if "backyard_cap" in fixes:
        data_repl.append((
            "    coeff_land_backyard = np.fmin(urban, area_backyard)\n",
            "    coeff_land_backyard = np.fmin(urban, area_backyard)\n"
            "    area_backyard = coeff_land_backyard\n"))
    data_mod = patched(inpdt, data_repl)

    grid, _ = inpdt.import_grid(data_ct)
    amenities = inpdt.import_amenities(precalc, options)
    interest_rate, population, housing_type_data, _, _ = \
        data_mod.import_macro_data(param, data_ct + 'Scenarios/', folder)
    access = inpdt.import_hypothesis_housing_type()
    mean_income, hh_class, average_income, *_ = \
        inpdt.import_income_classes_data(param, data_ct)
    data_rdp, housing_types_sp, data_sp, mp_grid, formal_density, *_ = \
        inpdt.import_households_data(precalc)
    housing_types = pd.read_excel(folder + 'housing_types_grid_sal.xlsx')
    housing_types[np.isnan(housing_types)] = 0
    (spline_RDP, spline_estimate_RDP, spline_land_RDP, spline_land_backyard,
     spline_land_informal, spline_land_constraints, _
     ) = data_mod.import_land_use(grid, options, param, data_rdp,
                                  housing_types, housing_type_data, data_ct,
                                  folder)
    coeff_land = inpdt.import_coeff_land(
        spline_land_constraints, spline_land_backyard, spline_land_informal,
        spline_land_RDP, param, options, 29)
    param, min_supply, agri_rent = inpprm.import_construction_parameters(
        param, grid, housing_types_sp, data_sp["dwelling_size"], mp_grid,
        formal_density, coeff_land, interest_rate, options)
    net_income = np.load(folder + 'precalculated_transport/'
                         'GRID_incomeNetOfCommuting_0.npy')

    if options["agents_anticipate_floods"]:
        cfd = inpdt.compute_fraction_capital_destroyed
        if "pluvial" in fixes:           # stop in-place edits of flood maps
            inpdt.compute_fraction_capital_destroyed = \
                lambda d, *a: cfd({k: v.copy() for k, v in d.items()}, *a)
        try:
            (fcd, fcd_protec, *_, tables, tables_protec, proba
             ) = inpdt.import_full_floods_data(options, param, folder)
        finally:
            inpdt.compute_fraction_capital_destroyed = cfd
    else:                                # legacy script fails here
        cols = ["structure_formal_2", "structure_formal_1",
                "structure_subsidized_2", "structure_subsidized_1",
                "contents_formal", "contents_informal", "contents_backyard",
                "structure_backyards", "structure_informal_settlements"]
        fcd = pd.DataFrame({c: np.zeros(24014) for c in cols})
        fcd_protec = fcd.copy()
        tables = {c: np.zeros((11, 24014)) for c in cols}
        tables_protec = copy.deepcopy(tables)
        proba = [0.8] + [0] * 10

    sol_repl = []
    if "backyard_loss" in fixes:
        sol_repl.append(("    housing_supply = mu\n",
                         "    housing_supply = mu\n"
                         "    housing_supply[R <= Z_IB] = 0\n"))
    if "rm_backyard" in fixes:
        sol_repl.append((
            "(param[\"depreciation_rate\"] + fraction_capital_destroyed"
            ".structure_backyards + interest_rate)",
            "(param[\"depreciation_rate\"] + (param[\"risk_internaliz\"] if "
            "options[\"risk_misperc\"] == 1 else 1) * fraction_capital_"
            "destroyed.structure_backyards + interest_rate)"))
    m_out = patched(eqout, [])
    m_out.eqsol = patched(eqsol, sol_repl)
    eq_repl = []
    if "conv" in fixes:
        eq_repl.append(("(total_simulated_jobs[index_iteration, :] + 100)",
                        "(total_simulated_jobs[index_iteration - 1, :] + 100)"))
    m_eq = patched(eqcmp, eq_repl)
    m_eq.eqout = m_out
    res = m_eq.compute_equilibrium(
        fcd, fcd_protec, tables, tables_protec, proba, amenities, param,
        inpdt.import_housing_limit(grid, param), population, hh_class,
        spline_RDP(29), coeff_land, net_income, grid, options, agri_rent,
        interest_rate, spline_estimate_RDP(29), average_income, mean_income,
        access, min_supply, param["coeff_A"], None)
    return {k: np.asarray(v) for k, v in zip(OUT_KEYS, res)}


def compare(new, ref):
    worst = 0
    for key in OUT_KEYS:
        a, b = np.asarray(new[key], float), np.asarray(ref[key], float)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        ok = np.isfinite(b)
        rel = (np.max(np.abs(a[ok] - b[ok])) / max(np.max(np.abs(b[ok])), 1e-300)
               if ok.any() else 0)
        worst = max(worst, rel, 0 if same_nan else 1)
        print(f"  {key:26s} max rel diff {rel:.2e}"
              + ("" if same_nan else "  NaN PATTERN DIFFERS"))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="AF CC RM SP digits, e.g. 1001")
    parser.add_argument("--fixes", nargs="*", default=list(FIXES[:-1]),
                        choices=FIXES)
    args = parser.parse_args()
    keys = ["agents_anticipate_floods", "climate_change", "risk_misperc",
            "self_protec"]
    if len(args.config) != len(keys):
        sys.exit("CONFIG must have 4 digits (AF CC RM SP), e.g. 1001")
    options = {k: int(c) for k, c in zip(keys, args.config)}

    print(f"Legacy run with fixes {args.fixes}...")
    ref = run_legacy(options, set(args.fixes))
    print("New model run in legacy mode...")
    floods.combine_flood_types = legacy_combine
    new, *_ = run.run_model({**options, "coastal": 0},
                            {"return_best": False})
    worst = compare(new, ref)
    print("IDENTICAL" if worst == 0 else f"Max relative difference {worst:.2e}")


if __name__ == "__main__":
    main()
