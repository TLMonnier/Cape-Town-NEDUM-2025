"""Check flood_model_CRRA against flood_model with degenerate flood states.

When every flood state carries the expected damage (D̄_s = ρ for all s),
households face no risk, and the CRRA model must reproduce the risk-neutral
flood_model whatever the risk aversion. The check compares both Markets
classes at given utilities (initial ones, and the risk-neutral equilibrium):
bid rents, supply, households and protection choice should agree at rounding
level (about 1e-15).

Excluded: backyard supply in cells where RDP owners' income net of their
house costs, ỹ_1 - (ρ + ρ^struct_FS) v_FS, is not positive. Their composite
good is negative whatever they rent out, so neither model is defined there:
flood_model applies main.tex (10) anyway (often μ = 1), and CRRAMarkets
rents out nothing if θ >= 1 (V = -inf everywhere) or follows the odd
extension of utility if θ < 1. These cells are counted separately.

The equilibria are compared for information only: the solver paths can
split when they go through these cells, and the best-iterate rule then
picks iterates with nearly equal errors (differences of order 1e-5).

Usage (from any directory):
    conda run -n nedum-2025 python flood_model_CRRA/tools/check_degenerate.py
    ... check_degenerate.py 11011 --sandbag_course_cost 100 --CRRA 2
CONFIG digits are the AF, CC, CO, RM, SP options (SP: 0-3 sandbag levels;
default 10103).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crra  # noqa: E402  (puts ../flood_model on sys.path)
import config  # noqa: E402
import data  # noqa: E402
import floods  # noqa: E402
import solver  # noqa: E402


def degenerate(damages):
    """Damages with every state damage replaced by the expected damage."""
    def flat(expected, states):
        return {n: np.broadcast_to(expected[n], v.shape).copy()
                for n, v in states.items()}

    out = dict(damages)
    out["states"] = flat(damages["expected"], damages["states"])
    out["states_protec"] = {
        k: flat(damages["expected_protec"][k], s)
        for k, s in damages["states_protec"].items()}
    return out


def rel_diff(a, b):
    """Max relative difference over finite values (inf if NaN patterns
    differ)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    ok = np.isfinite(b)
    if not ok.any():
        return 0.
    return np.max(np.abs(a[ok] - b[ok])) / max(np.max(np.abs(b[ok])), 1e-300)


def compare_markets(ra, rn, u, defined):
    """Print and return the max relative difference between both Markets
    classes at utilities u, over cells where RDP owners are defined."""
    worst = 0.
    for kind in solver.TYPES:
        a, b = ra.solve_market(kind, u), rn.solve_market(kind, u)
        cells = defined if kind == "backyard" else slice(None)
        for key in ("rent_matrix", "supply", "households", "mask"):
            d = rel_diff(a[key][..., cells], b[key][..., cells])
            worst = max(worst, d)
            print(f"  {kind:9s} {key:12s} max rel diff {d:.1e}")
        if kind == "backyard":
            differ = ~np.isclose(a["supply"], b["supply"]) & ~defined
            print(f"  backyard supply differs in {differ.sum()} of the "
                  f"{(~defined).sum()} excluded cells, "
                  f"{(differ & (ra.land[1] > 0)).sum()} with backyard land")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="10103",
                        help="AF CC CO RM SP digits")
    parser.add_argument("--sandbag_course_cost", type=float, default=25)
    parser.add_argument("--CRRA", type=float, default=crra.PARAM["CRRA"])
    args = parser.parse_args()
    keys = ["agents_anticipate_floods", "climate_change", "coastal",
            "risk_misperc", "self_protec"]
    if len(args.config) != len(keys):
        sys.exit("CONFIG must have 5 digits (AF CC CO RM SP), e.g. 10103")
    options = {**config.OPTIONS,
               **{k: int(c) for k, c in zip(keys, args.config)}}

    p, inputs = data.prepare_inputs({
        **config.PARAM, **crra.PARAM, "CRRA": args.CRRA,
        "sandbag_course_cost": args.sandbag_course_cost})
    damages = degenerate(floods.compute_damages(inputs, options, p))
    rn = solver.Markets(p, inputs, damages, options)
    ra = crra.CRRAMarkets(p, inputs, damages, options)
    v = p["subsidized_structure_value"]
    defined = (rn.y[0] - p["depreciation_rate"] * v
               - rn.dmg["structure_subsidized_1"] * v) > 0

    out_rn, _ = solver.solve(rn, verbose=False)
    worst = 0.
    for label, u in [("initial", np.array(p["utility_init"], float)),
                     ("risk-neutral equilibrium", out_rn["utility"])]:
        print(f"Markets at the {label} utilities:")
        worst = max(worst, compare_markets(ra, rn, u, defined))
    print("OK" if worst < 1e-12
          else f"DIFFERENT (max rel diff {worst:.1e})")

    print("Equilibria (for information, see the docstring):")
    out_ra, _ = solver.solve(ra, verbose=False)
    for key in ("utility", "households", "rent", "housing_supply",
                "mask_self_protec"):
        print(f"  {key:16s} max rel diff "
              f"{rel_diff(out_ra[key], out_rn[key]):.1e}")


if __name__ == "__main__":
    main()
