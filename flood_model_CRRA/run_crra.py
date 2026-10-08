"""Run the flood model with risk-averse (CRRA) households:
python run_crra.py [--CRRA 0.2772 --af 1 --cc 0 --co 1 --rm 0 --sp 1 ...]

Same options, inputs and outputs as ../flood_model/run.py, saved to
Output/flood_model_CRRA under the same file names.
"""
import time

import crra                  # first: puts ../flood_model on sys.path
import config                # flood_model modules from here on
import run


def run_model(options=None, param=None, refresh_cache=False, verbose=True):
    """Return (outputs dict, converged flag, full param dict, damages).

    options and param override config.OPTIONS and config.PARAM + crra.PARAM.
    Insurance reference runs (SI1) are also risk averse.
    """
    return run.run_model(options, {**crra.PARAM, **(param or {})},
                         refresh_cache, verbose, crra.CRRAMarkets)


def main():
    parser = run.build_parser(__doc__)
    parser.add_argument("--CRRA", type=float, default=crra.PARAM["CRRA"])
    args = vars(parser.parse_args())
    options = {k: args.pop(k) for k in config.OPTIONS}
    refresh = args.pop("refresh_cache")
    start = time.time()
    outputs, converged, _, damages = run_model(options, args, refresh)
    name = run.save(outputs, options, folder=crra.OUTPUT, damages=damages)
    print(f"Saved {name} to {crra.OUTPUT} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
