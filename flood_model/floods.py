"""Flood damages: expected fraction of capital destroyed per grid cell.

main.tex Sections 3 ("Flood data") and 4.1. Floods enter the model only
through expected annual damage rates, i.e. expected fractions of capital
destroyed per year, which act as extra depreciation rates:
    ρ^content_h(x): housing contents, a share γ of the composite good z,
        which costs (1 + γ ρ^content_h(x)) per unit (budget constraint (2));
    ρ^struct_h(x): housing structures, whose annual cost per rand of value
        becomes ρ + ρ^struct_h(x) (+ δ when financed): paid by developers
        for FP (profit (4)), and by households for FS, IS, IB (budget
        constraint (2)).
This module computes the *actual* damages. Agents' perception of them (AF,
RM options) is applied in solver.Markets.

For each damage series (damage function x housing type, SERIES below) and
grid cell x:

1. Event damages. For each flood type and return period T (an event that
   occurs once every T years on average), flood maps give the flood-prone
   share p_T(x) of the cell and the flood depth d_T(x) there. With f the
   depth-damage function of the series, event T destroys a fraction
       D_T(x) = p_T(x) f(d_T(x))
   of the capital in the cell. Two corrections:
   - drainage (main.tex Section 3): frequent pluvial events do no damage in
     housing types served by stormwater drains (PLUVIAL_SAFE);
   - public protection (PP option, not in main.tex): every dwelling, of
     every housing type, is protected up to H = public_protection_height
     (45 cm) against every flood type, which lowers the water depth to
     max(d_T(x) - H, 0) (H = 0 without PP);
   - sandbags (SP option): settlers protected by k levels of sandbags
     (k = 1, ..., options["self_protec"]) raise their floor by a further
     h_k = k x sandbag_height (15, 30 or 45 cm), so that the water depth
     inside the dwelling becomes max(d_T(x) - H - h_k, 0):
         D_T^k(x) = p_T(x) f(max(d_T(x) - H - h_k, 0)).
2. Flood types. In each event T, the most damaging flood type applies:
   D_T = max over types of D_T^type (water spills over to other areas rather
   than piling up). main.tex (outdated) says the maximum *depth* across
   types: it is the same rule only where types flood the same share of the
   cell.
3. Expected damage. Event T has annual exceedance probability P = 1/T.
   Damage is interpolated linearly in P between the 10 return periods
   T_1, ..., T_10 = 5, ..., 1000 years, from zero damage at P = 1 (the yearly
   flood is harmless), and kept at the 1000-yr damage for P < 1/1000. The
   expected annual damage is the integral over P in [0, 1] (trapezoidal
   rule) and the 11 terms of the sum are "flood states" s:
       ρ(x) = Σ_{s=0..10} π_s D̄_s(x),  π_s = P_s - P_{s+1},
       D̄_s(x) = (D_s(x) + D_{s+1}(x)) / 2,
   with P_s = 1/T_s, P_0 = 1, D_0 = 0, P_11 = 0 and D_11 = D_10.

Flood types: fluvial (FATHOM, undefended maps), pluvial (FATHOM) and coastal
(DELTARES, MERITDEM elevation model). Coastal maps come with other return
periods (DELTARES_RETURN_PERIODS): each FATHOM return period T uses the
closest smaller or equal DELTARES return period (conservative).

Climate change (CC, main.tex Section 3): FATHOM annual probabilities are
multiplied by risk_increase = 2, i.e. P_s = 2/T_s (the T-yr map stands for an
event of return period T/2). Coastal floods instead use the DELTARES
sea-level-rise maps (RCP8.5, 2050), aligned on these effective return
periods T/2 so that coastal probabilities are not scaled twice.
"""
import numpy as np

import config

RETURN_PERIODS = np.array([5, 10, 20, 50, 75, 100, 200, 250, 500, 1000])
DELTARES_RETURN_PERIODS = np.array([0, 2, 5, 10, 25, 50, 100, 250])
FLOOD_TYPES = ("fluvial", "pluvial", "coastal")
NO_FLOOD = -1            # flood_type code where all flood types do no damage

# Depth-damage functions f: depth (m) -> fraction of capital destroyed
# (main.tex Section 3)
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

# Pluvial return periods (years) against which housing types are protected
# by stormwater drains (CCT Minimum Standards 2014, Govender 2011): formal
# housing, including RDP premises and their backyards. As a result, contents
# damages differ across housing types (main.tex Section 3, outdated, says
# they do not).
PLUVIAL_SAFE = {"formal": (5, 10, 20), "subsidized": (5, 10),
                "backyard": (5, 10), "informal": ()}

# Damage series: name -> (damage function, housing type)
SERIES = {
    # ρ^content_h, h = FP, IB, IS, FS (RDP owners' contents damages do not
    # affect their choices, see solver.Markets.backyard_supply; they enter
    # damage accounting)
    "contents_formal": ("contents", "formal"),
    "contents_backyard": ("contents", "backyard"),
    "contents_informal": ("contents", "informal"),
    "contents_subsidized": ("contents", "subsidized"),
    # ρ^struct_FP, borne by developers: one-floor dwellings up to
    # param["threshold"] m2, two floors above (not in main.tex)
    "structure_formal_1": ("type4a", "formal"),
    "structure_formal_2": ("type4b", "formal"),
    # ρ^struct_FS: RDP houses (40 m2, one floor), borne by their owners
    "structure_subsidized_1": ("type4a", "subsidized"),
    # ρ^struct_IS: shacks in informal settlements, borne by settlers
    "structure_informal_settlements": ("type2", "informal"),
    # ρ^struct_IB: backyard structures, borne by RDP owners. Shacks and brick
    # structures are averaged into "structure_backyards" (compute_damages)
    "structure_informal_backyards": ("type2", "backyard"),
    "structure_formal_backyards": ("type3a", "backyard"),
}
# Series with sandbag protection (SP): only informal settlers can protect
# (both their contents and their shack)
PROTEC_SERIES = ("contents_informal", "structure_informal_settlements")


def state_probabilities(options, param):
    """Probabilities π_s = P_s - P_{s+1} of the 11 flood states (sum to 1).

    Under CC, exceedance probabilities P_s = risk_increase / T_s."""
    f = param["risk_increase"] if options["climate_change"] else 1
    exceed = np.r_[1, 1 / RETURN_PERIODS * f, 0]
    return exceed[:-1] - exceed[1:]


def coastal_return_periods(options, param):
    """DELTARES return period used for each FATHOM return period: the
    closest smaller or equal one to the effective return period T (T /
    risk_increase under CC). E.g. without CC, T 5->5, 10->10, 20->10, ...,
    1000->250, so that the 0 and 2-yr maps are unused."""
    f = param["risk_increase"] if options["climate_change"] else 1
    idx = np.searchsorted(DELTARES_RETURN_PERIODS, RETURN_PERIODS / f,
                          side="right") - 1
    return DELTARES_RETURN_PERIODS[idx]


def flood_maps(inputs, options, param):
    """Depth d_T(x) and flood-prone share p_T(x), each of shape (flood
    types, return periods, cells), flood types as in FLOOD_TYPES (fluvial,
    pluvial, and coastal if options["coastal"])."""
    depth, prop = list(inputs["flood_depth"]), list(inputs["flood_prop"])
    if options["coastal"]:
        rp = np.searchsorted(DELTARES_RETURN_PERIODS,
                             coastal_return_periods(options, param))
        cc = options["climate_change"]       # sea-level-rise maps under CC
        depth.append(inputs["coastal_depth"][cc][rp])
        prop.append(inputs["coastal_prop"][cc][rp])
    return np.array(depth), np.array(prop)


def event_damages(depth, prop, damage_function, safe_rps=(),
                  floor_height=0):
    """Event damages D_T(x) = p_T(x) f(max(d_T(x) - h, 0)), (return
    periods, cells), for one flood type, with h the protection height
    (public protection H plus sandbags h_k; 0 without protection). Events
    with return periods in safe_rps do no damage."""
    prop = np.where(np.isin(RETURN_PERIODS, safe_rps)[:, None], 0, prop)
    if floor_height:
        depth = np.maximum(depth - floor_height, 0)
    return prop * np.interp(depth, *damage_function)


def state_damages(event):
    """State damages D̄_s = (D_s + D_{s+1}) / 2, (11, cells), from event
    damages D_T, (10, cells), with D_0 = 0 and D_11 = D_10."""
    event = np.concatenate([np.zeros_like(event[:1]), event, event[-1:]])
    return 0.5 * (event[:-1] + event[1:])


def expected_damage(proba, states):
    """Expected annual damage ρ = Σ_s π_s D̄_s."""
    return sum(p * s for p, s in zip(proba, states))


def max_over_types(values):
    """Max over flood types (first axis), and index of the selected type
    in FLOOD_TYPES (first one if tied, NO_FLOOD if no damage)."""
    values = np.asarray(values)
    selected = values.argmax(0).astype(np.int8)
    best = values.max(0)
    selected[best == 0] = NO_FLOOD
    return best, selected


def combine_flood_types(by_type, proba):
    """Return (state damages, expected damages, flood type selected in each
    event) from event damages by flood type, (types, return periods,
    cells): in each event, the most damaging flood type applies."""
    event, event_type = max_over_types(by_type)
    states = state_damages(event)
    return states, expected_damage(proba, states), event_type


def compute_damages(inputs, options, param):
    """Return actual expected and state damages for each series.

    Output dict:
        expected: name -> (cells,) expected damage ρ(x)
        states: name -> (11, cells) state damages D̄_s(x)
        expected_protec, states_protec: k -> same with k sandbag levels,
            k = 1, ..., options["self_protec"] (PROTEC_SERIES only)
    All damages include public protection under PP.
        proba: (11,) state probabilities π_s
        flood_type: index in FLOOD_TYPES of the flood type (NO_FLOOD if no
            damage), for each series:
            expected[_protec<k>]: dominant type of each cell (the type with
                the highest expected damage on its own)
            return_periods[_protec<k>]: (10, cells) type selected in each
                event
    """
    if options["self_protec"] not in (0, 1, 2, 3):
        raise ValueError("self_protec is the number of sandbag levels, "
                         "0 to 3")
    depth, prop = flood_maps(inputs, options, param)
    proba = state_probabilities(options, param)
    levels = range(1, options["self_protec"] + 1)
    # PP: public protection height H, below sandbags
    public = (param["public_protection_height"]
              if options.get("public_protec") else 0)
    out = {"proba": proba, "expected": {}, "states": {},
           "expected_protec": {k: {} for k in levels},
           "states_protec": {k: {} for k in levels}}
    out["flood_type"] = {f"{kind}{sfx}": {}
                         for sfx in [""] + [f"_protec{k}" for k in levels]
                         for kind in ("expected", "return_periods")}
    for name, (fun, htype) in SERIES.items():
        for k in [0] + list(levels if name in PROTEC_SERIES else []):
            by_type = [event_damages(
                depth[t], prop[t], DAMAGE_FUNCTIONS[fun],
                PLUVIAL_SAFE[htype] if kind == "pluvial" else (),
                public + k * param["sandbag_height"])
                for t, kind in enumerate(FLOOD_TYPES[:len(depth)])]
            states, expected, rp_type = combine_flood_types(by_type, proba)
            if k == 0:
                out["states"][name], out["expected"][name] = states, expected
            else:
                out["states_protec"][k][name] = states
                out["expected_protec"][k][name] = expected
            sfx = f"_protec{k}" if k else ""
            out["flood_type"]["return_periods" + sfx][name] = rp_type
            out["flood_type"]["expected" + sfx][name] = max_over_types(
                [expected_damage(proba, state_damages(d))
                 for d in by_type])[1]

    # Backyard structures (ρ^struct_IB): average of brick and shack
    # structures, weighted by their 2011 counts
    w_f, w_i = config.BACKYARDS_FORMAL, config.BACKYARDS_INFORMAL
    for key in ("expected", "states"):
        d = out[key]
        d["structure_backyards"] = (
            (w_f * d.pop("structure_formal_backyards")
             + w_i * d.pop("structure_informal_backyards")) / (w_f + w_i))
    return out
