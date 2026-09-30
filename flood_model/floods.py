"""Expected flood damages (fraction of capital destroyed) per grid cell.

Flood maps give, for each return period, the flood-prone share of each
pixel and the maximum flood depth there. Damage in a pixel for a given return
period is share x depth-damage function(depth). Between two return periods,
damage is interpolated linearly in annual exceedance probability, which
defines 11 flood states (the first one from annual probability 1, with zero
damage, to 1/5; the last one keeping the 1000-yr damage beyond 1/1000).
State damage is the average damage over the state's probability interval,
and expected damage the probability-weighted sum of state damages.

Flood types: fluvial (FATHOM, undefended), pluvial (FATHOM) and coastal
(DELTARES, MERITDEM). Coastal maps are aligned on FATHOM return periods by
taking, for each FATHOM return period, the closest smaller or equal DELTARES
return period. Flood types are combined by taking, for each return period
(event), the maximum damage across types (floods spill over rather than pile
up), before computing state and expected damages. `flood_type` records the
type selected for each return period, and the dominant type of each cell,
defined as the type with the highest expected damage on its own.

Climate change multiplies FATHOM exceedance probabilities by risk_increase
(dummy scenario), so the FATHOM T-yr map stands for an event of return
period T / risk_increase. Coastal floods instead use DELTARES sea-level-rise
maps (RCP8.5, 2050), aligned on these effective return periods so that their
probabilities are not scaled twice.
"""
import numpy as np

import config

RETURN_PERIODS = np.array([5, 10, 20, 50, 75, 100, 200, 250, 500, 1000])
DELTARES_RETURN_PERIODS = np.array([0, 2, 5, 10, 25, 50, 100, 250])
FLOOD_TYPES = ("fluvial", "pluvial", "coastal")
NO_FLOOD = -1            # flood_type code where all flood types do no damage

# Depth (m) -> fraction of capital destroyed
DAMAGE_FUNCTIONS = {
    # de Villiers et al. (2007), house contents
    "contents": ([0, 0.1, 0.3, 0.6, 1.2, 1.5, 2.4, 10],
                 [0, 0.06, 0.15, 0.35, 0.77, 0.95, 1, 1]),
    # Englhardt et al. (2019): wooden buildings (shacks)
    "type2": ([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
              [0, 0.45, 0.65, 0.82, 0.95, 1, 1, 1, 1, 1, 1, 1]),
    # one-floor unreinforced masonry (formal backyards)
    "type3a": ([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
               [0, 0.4, 0.55, 0.7, 0.78, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81,
                0.81]),
    # one-floor reinforced masonry (formal private and subsidized)
    "type4a": ([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
               [0, 0.31, 0.45, 0.55, 0.62, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                0.65]),
    # two-floor reinforced masonry (formal private above threshold size)
    "type4b": ([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
               [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.62, 0.64, 0.65,
                0.65]),
}

# Pluvial return periods against which housing types are protected by
# stormwater design (CCT Minimum Standards 2014, Govender 2011)
PLUVIAL_SAFE = {"formal": (5, 10, 20), "subsidized": (5, 10),
                "backyard": (5, 10), "informal": ()}

# Damage series: name -> (damage function, housing type)
SERIES = {
    "contents_formal": ("contents", "formal"),
    "contents_backyard": ("contents", "backyard"),
    "contents_informal": ("contents", "informal"),
    "structure_formal_1": ("type4a", "formal"),
    "structure_formal_2": ("type4b", "formal"),
    "structure_subsidized_1": ("type4a", "subsidized"),
    "structure_informal_settlements": ("type2", "informal"),
    "structure_informal_backyards": ("type2", "backyard"),
    "structure_formal_backyards": ("type3a", "backyard"),
}
# Only informal settlers can protect themselves
PROTEC_SERIES = ("contents_informal", "structure_informal_settlements")


def state_probabilities(options, param):
    """Probabilities (summing to 1) of the 11 flood states."""
    f = param["risk_increase"] if options["climate_change"] else 1
    exceed = np.r_[1, 1 / RETURN_PERIODS * f, 0]
    return exceed[:-1] - exceed[1:]


def coastal_return_periods(options, param):
    """DELTARES return period used for each FATHOM return period."""
    f = param["risk_increase"] if options["climate_change"] else 1
    idx = np.searchsorted(DELTARES_RETURN_PERIODS, RETURN_PERIODS / f,
                          side="right") - 1
    return DELTARES_RETURN_PERIODS[idx]


def flood_maps(inputs, options, param):
    """Depth and flood-prone share, (flood types, return periods, pixels)."""
    depth, prop = list(inputs["flood_depth"]), list(inputs["flood_prop"])
    if options["coastal"]:
        rp = np.searchsorted(DELTARES_RETURN_PERIODS,
                             coastal_return_periods(options, param))
        cc = options["climate_change"]       # sea-level-rise maps under CC
        depth.append(inputs["coastal_depth"][cc][rp])
        prop.append(inputs["coastal_prop"][cc][rp])
    return np.array(depth), np.array(prop)


def _rp_damages(depth, prop, damage_function, safe_rps, protec_depth):
    """(return periods, pixels) damages for one flood type."""
    prop = np.where(np.isin(RETURN_PERIODS, safe_rps)[:, None], 0, prop)
    if protec_depth is not None:
        prop = np.where(depth <= protec_depth, 0, prop)
    return prop * np.interp(depth, *damage_function)


def _states(dmg):
    """(11, pixels) state damages from (return periods, pixels) damages."""
    dmg = np.concatenate([np.zeros_like(dmg[:1]), dmg, dmg[-1:]])
    return 0.5 * (dmg[:-1] + dmg[1:])


def _expected(proba, states):
    return sum(p * s for p, s in zip(proba, states))


def _max_over_types(values):
    """Max over flood types (first axis), and index of the selected type
    in FLOOD_TYPES (first one if tied, NO_FLOOD if no damage)."""
    values = np.asarray(values)
    selected = values.argmax(0).astype(np.int8)
    best = values.max(0)
    selected[best == 0] = NO_FLOOD
    return best, selected


def combine_flood_types(by_type, proba):
    """Return (state damages, expected damages, flood type per return
    period) from damages by flood type and return period.

    In each event (return period), the most damaging flood type applies."""
    rp_damages, rp_type = _max_over_types(by_type)
    states = _states(rp_damages)
    return states, _expected(proba, states), rp_type


def compute_damages(inputs, options, param):
    """Return expected and state damages for each series.

    Output dict:
        expected: name -> (pixels,) expected fraction of capital destroyed
        states: name -> (11, pixels) state damages (used under risk aversion)
        expected_protec, states_protec: same with sandbag protection
            (informal settlement series only)
        flood_type: index in FLOOD_TYPES of the flood type (NO_FLOOD if no
            damage), for each series:
            expected[_protec]: dominant type of the cell (highest expected
                damage on its own)
            return_periods[_protec]: (10, pixels) type selected in each event
        proba: (11,) state probabilities
    """
    depth, prop = flood_maps(inputs, options, param)
    types = [FLOOD_TYPES[t] for t in range(len(depth))]
    proba = state_probabilities(options, param)
    out = {"proba": proba, "expected": {}, "expected_protec": {},
           "states": {}, "states_protec": {}}
    out["flood_type"] = {k: {} for k in (
        "expected", "expected_protec", "return_periods",
        "return_periods_protec")}
    for name, (fun, htype) in SERIES.items():
        for protec in (False, True):
            if protec and name not in PROTEC_SERIES:
                continue
            suffix = "_protec" if protec else ""
            if options["agents_anticipate_floods"]:
                by_type = [_rp_damages(
                    depth[t], prop[t], DAMAGE_FUNCTIONS[fun],
                    PLUVIAL_SAFE[htype] if kind == "pluvial" else (),
                    param["protec_depth"] if protec else None)
                    for t, kind in enumerate(types)]
            else:
                by_type = [np.zeros((len(RETURN_PERIODS), depth.shape[-1]))]
            (out["states" + suffix][name], out["expected" + suffix][name],
             out["flood_type"]["return_periods" + suffix][name]
             ) = combine_flood_types(by_type, proba)
            out["flood_type"]["expected" + suffix][name] = _max_over_types(
                [_expected(proba, _states(d)) for d in by_type])[1]

    # Backyard structures: weighted average of brick and shack structures
    w_f, w_i = config.BACKYARDS_FORMAL, config.BACKYARDS_INFORMAL
    for key in ("expected", "states"):
        d = out[key]
        d["structure_backyards"] = (
            (w_f * d.pop("structure_formal_backyards")
             + w_i * d.pop("structure_informal_backyards")) / (w_f + w_i))
    return out
