"""Run the flood model: python run.py [--af 1 --cc 0 --co 1 --rm 0 --sp 1 ...]

Solves the 2011 static equilibrium for the chosen flood options and saves
outputs as <OUTPUT>/initial_state_<variable>_<simulation name>.npy, and the
flood type selected in each cell as <OUTPUT>/flood_type_<name>.npz.
"""
import argparse
import time

import numpy as np

import config
import data
import floods
import solver

FLAGS = {"af": "agents_anticipate_floods", "cc": "climate_change",
         "co": "coastal", "rm": "risk_misperc", "sp": "self_protec"}


def run_model(options=None, param=None, refresh_cache=False, verbose=True):
    """Return (outputs dict, converged flag, full param dict, damages).

    options and param override config.OPTIONS and config.PARAM."""
    options = {**config.OPTIONS, **(options or {})}
    p, inputs = data.prepare_inputs({**config.PARAM, **(param or {})},
                                    refresh_cache)
    damages = floods.compute_damages(inputs, options, p)
    markets = solver.Markets(p, inputs, damages, options)
    outputs, converged = solver.solve(markets, verbose)
    return outputs, converged, p, damages


def save(outputs, options, tag="", folder=config.OUTPUT, damages=None):
    """Save outputs (and flood type masks if damages are given)."""
    folder.mkdir(parents=True, exist_ok=True)
    name = config.simulation_name(options, tag)
    for key, value in outputs.items():
        prefix = "" if key == "mask_self_protec" else "initial_state_"
        np.save(folder / f"{prefix}{key}_{name}.npy", value)
    if damages is not None:
        # Keys "<level>__<series>": level "expected[_protec]" gives the
        # dominant type of each cell, "return_periods[_protec]" the type
        # selected in each event; values index "flood_types" (-1: no damage)
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
        parser.add_argument("--" + flag, dest=key, type=int, choices=(0, 1),
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
