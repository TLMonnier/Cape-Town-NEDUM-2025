"""Run the flood model: python run.py [--af 1 --cc 0 --co 1 --rm 0 --sp 1 ...]

Solves the 2011 static equilibrium for the chosen flood options and saves
outputs as <OUTPUT>/initial_state_<variable>_<simulation name>.npy, and the
flood type selected in each cell as <OUTPUT>/flood_type_<name>.npz.
"""
import argparse
import time

import numpy as np

import accounting
import config
import data
import floods
import solver

FLAGS = {"af": "agents_anticipate_floods", "cc": "climate_change",
         "co": "coastal", "rm": "risk_misperc", "sp": "self_protec",
         "ps": "subsid_protec", "si": "subsid_insur", "pp": "public_protec"}
CHOICES = {"sp": (0, 1, 2, 3)}


def run_model(options=None, param=None, refresh_cache=False, verbose=True,
              markets_class=solver.Markets):
    """Return (outputs dict, converged flag, full param dict, damages).

    options and param override config.OPTIONS and config.PARAM."""
    options = {**config.OPTIONS, **(options or {})}
    p, inputs = data.prepare_inputs({**config.PARAM, **(param or {})},
                                    refresh_cache)
    p = insurance_terms(p, inputs, options, markets_class, verbose)
    damages = floods.compute_damages(inputs, options, p)
    markets = markets_class(p, inputs, damages, options)
    outputs, converged = solver.solve(markets, verbose)
    return outputs, converged, p, damages


def baseline_options(options):
    """Same scenario without self-protection, insurance or public
    protection (SP0 PS0 SI0 PP0)."""
    return {**options, "self_protec": 0, "subsid_protec": 0,
            "subsid_insur": 0, "public_protec": 0}


def insurance_terms(p, inputs, options, markets_class=solver.Markets,
                    verbose=True):
    """Return p with the insurance coverage share s (SI1), if not set,
    computed from a reference run without insurance: the same options and
    parameters with SI0 and PS0 (sandbags at full cost), and SP sandbag
    levels (p["insurance_reference_levels"] if SP0). With D the settlers'
    total actual damages in that equilibrium, at the sandbag levels they
    chose, and D_0 their damages without sandbags (same households and
    locations),
        s = 1 - D / D_0,
    the share of exposed settlers' damages averted by (unsubsidised)
    self-protection. s depends on the scenario (AF, CC, RM, sandbag cost,
    ...); solver.solve reports it in outputs["insurance_share"].
    """
    if not options["subsid_insur"] or p["insurance_share"] is not None:
        return p
    reference = {**options, "subsid_insur": 0, "subsid_protec": 0,
                 "self_protec": (options["self_protec"]
                                 or p["insurance_reference_levels"])}
    mk = markets_class(p, inputs,
                       floods.compute_damages(inputs, reference, p), reference)
    outputs, _ = solver.solve(mk, verbose=False)
    bare = {**outputs,
            "mask_self_protec": np.zeros_like(outputs["mask_self_protec"])}
    D, D_0 = (acc["contents"][2].sum() + acc["structure"][2].sum()
              for acc in (accounting.household_damages(mk, outputs),
                          accounting.household_damages(mk, bare)))
    p = dict(p)
    p["insurance_share"] = float(1 - D / D_0) if D_0 > 0 else 0.
    if verbose:
        print(f"Insurance SI1: coverage share s = {p['insurance_share']:.4f} "
              f"(settlers' damages {D:,.0f} rands/year with sandbags, "
              f"{D_0:,.0f} without, SP{reference['self_protec']} at "
              f"{p['sandbag_course_cost']:g} rands per level)")
    return p


def save(outputs, options, tag="", folder=config.OUTPUT, damages=None):
    """Save outputs (and flood type masks if damages are given)."""
    folder.mkdir(parents=True, exist_ok=True)
    name = config.simulation_name(options, tag)
    for key, value in outputs.items():
        prefix = "" if key == "mask_self_protec" else "initial_state_"
        np.save(folder / f"{prefix}{key}_{name}.npy", value)
    if damages is not None:
        # Keys "<level>__<series>": level "expected[_protec<k>]" gives the
        # dominant type of each cell (with k sandbag levels),
        # "return_periods[_protec<k>]" the type selected in each event;
        # values index "flood_types" (-1: no damage)
        masks = {f"{level}__{series}": mask
                 for level, by_series in damages["flood_type"].items()
                 for series, mask in by_series.items()}
        np.savez(folder / f"flood_type_{name}.npz",
                 flood_types=np.array(floods.FLOOD_TYPES), **masks)
    return name


def build_parser(description=__doc__):
    """Command-line options: flood options and main parameters."""
    parser = argparse.ArgumentParser(description=description)
    for flag, key in FLAGS.items():
        # --sp: number of sandbag levels
        parser.add_argument("--" + flag, dest=key, type=int,
                            choices=CHOICES.get(flag, (0, 1)),
                            default=config.OPTIONS[key])
    for key in ("risk_internaliz", "sandbag_course_cost", "risk_increase"):
        parser.add_argument("--" + key, type=float, default=config.PARAM[key])
    parser.add_argument("--max_iter", type=int, default=config.PARAM["max_iter"])
    parser.add_argument("--refresh_cache", action="store_true",
                        help="re-read raw input files")
    return parser


def main():
    args = vars(build_parser().parse_args())
    options = {k: args.pop(k) for k in config.OPTIONS}
    refresh = args.pop("refresh_cache")
    start = time.time()
    outputs, converged, _, damages = run_model(options, args, refresh)
    name = save(outputs, options, damages=damages)
    print(f"Saved {name} to {config.OUTPUT} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
