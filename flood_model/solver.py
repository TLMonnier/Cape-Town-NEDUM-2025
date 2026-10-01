"""Static spatial equilibrium with flood risks (main.tex Section 4).

Equation numbers refer to main.tex (h: housing type, i: income group,
x: grid cell, u: utility level; see config.py for code names):
    (1)  utility  U = z^α (q - q0)^(1-α) A(x) B_h,  B_FP = B_FS = 1
    (2)  budget constraint
             ỹ_i(x) + 1{h=FS} μ(x) Y (R_IB(x)
                                       - (ρ + ρ^struct_IB(x) + δ) v_I / q_I)
             = (1 + γ ρ^content_h(x)) z + q_h R_h(x)
               + 1{h=IS,FS} (ρ + ρ^struct_h(x)) v_h + 1{h=IS} δ v_h
    (3)  formal housing production per unit of land  s_FP(k) = κ k^(1-a)
    (4)  developers' profit
             Π(x, k) = R_FP(x) s_FP(k) - (ρ + ρ^struct_FP(x) + δ) k - δ P(x)
    (5)  formal dwelling size Q*(x, i | u), implicitly defined by
             u = (α ỹ_i(x) / (1 + γ ρ^content_FP(x)))^α
                 (Q* - q0) / (Q* - α q0)^α A(x)
         and Q_FP(x, i, u) = max(q_min, Q*(x, i | u))
    (6)  ψ_i^FP(x, u) = (1 - α) ỹ_i(x) / (Q_FP(x, i, u) - α q0)
    (7)  ψ_i^IS(x, u) = [ỹ_i(x) - (ρ + δ + ρ^struct_IS(x)) v_I
                         - (1 + γ ρ^content_IS(x)) z_IS(x, u)] / q_I
    (8)  ψ_i^IB(x, u) = [ỹ_i(x) - (1 + γ ρ^content_IB(x)) z_IB(x, u)] / q_I
         with z_h(x, u) = (u / ((q_I - q0)^(1-α) A(x) B_h))^(1/α)
    (9)  s_FP(x) = κ^(1/a)
                   ((1 - a) R_FP(x) / (ρ + ρ^struct_FP(x) + δ))^((1-a)/a)
    (10) μ(x) = α (q_FS - q0) / Y
                - (1 - α) (ỹ_1(x) - (ρ + ρ^struct_FS(x)) v_FS)
                  / (Y [R_IB(x) - (ρ + ρ^struct_IB(x) + δ) v_I / q_I])
    (11) N_i^h(x) = s_h(x) L_h(x) / Q_h(x, i, u)

Structure:
- Markets: for given utility levels u = (u_1, ..., u_4), the 3 endogenous
  housing markets h = FP, IB, IS. In each grid cell, each income group bids
  ψ_i^h(x, u), the cell goes to the highest bidder, whose bid sets the market
  rent R_h(x) = max_i ψ_i^h(x, u), housing supply s_h(x) and the number of
  households N_i^h(x) (main.tex Sections 4.2-4.5).
- solve: adjusts u until each group's simulated number of households
  matches its target N_i (main.tex Section 4.6).
- FS (RDP) housing is exogenous: its households are added to the outputs at
  the end (_export).

Flood risks enter through the expected damage rates ρ^content_h(x) and
ρ^struct_h(x) computed in floods.py, as perceived by agents: k x actual
damages, with k = 0 if agents do not anticipate floods (AF0),
k = risk_internaliz under risk misperception (RM1), and k = 1 otherwise.
Households are risk neutral: they value damages at their expected value, as
if fully insured at an actuarially fair premium (see ../flood_model_CRRA for
risk aversion).

Self-protection (SP option, not in main.tex): informal settlers can pay
sandbag_course_cost a year so that floods up to protec_depth do no damage.
Each group protects in a cell if that raises its bid rent ψ_i^IS.

Units: rents in rands per m2 per year; housing supply s_h in m2 of floor
per km2 of *available* land; L_h(x) as a share of the cell area (0.25 km2).
"""
import numpy as np

import config

TYPES = ("formal", "backyard", "informal")      # h = FP, IB, IS

# Dwelling sizes (m2) at which the left side of (5) is tabulated
Q_GRID = np.concatenate((
    [10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3),
     10 ** (-2), 10 ** (-1)],
    np.arange(0.11, 0.15, 0.01), np.arange(0.15, 1.15, 0.05),
    np.arange(1.2, 3.1, 0.1), np.arange(3.5, 13.1, 0.25),
    np.arange(15, 60, 0.5), np.arange(60, 100, 2.5), np.arange(110, 210, 10),
    [250, 300, 500, 1000, 2000, 200000, 1000000, 10 ** 12]))


def perception(options, p):
    """Factor k such that perceived damages = k x actual damages: 0 if
    agents do not anticipate floods (AF0), risk_internaliz under risk
    misperception (RM1), 1 otherwise."""
    if not options["agents_anticipate_floods"]:
        return 0
    return p["risk_internaliz"] if options["risk_misperc"] else 1


class Markets:
    """The 3 endogenous housing markets, on the cells where housing can be.

    Arrays are (income groups, cells) or (cells,), restricted to cells with
    some land available (more than 1% of the cell, all housing types
    together) and positive net income for some group: about 4,000 of the
    24,014 cells.
    """

    def __init__(self, p, inputs, damages, options):
        self.p, self.inputs, self.options = p, inputs, options
        self.sel = sel = ((np.sum(inputs["coeff_land"], 0) > 0.01)
                          & (np.nanmax(inputs["net_income"], 0) > 0))
        self.y = inputs["net_income"][:, sel]           # ỹ_i(x)
        self.amenities = inputs["amenities"][sel]       # A(x)
        self.land = inputs["coeff_land"][:3, sel]       # L_h(x), h = FP, IB, IS
        self.housing_limit = inputs["housing_limit"][sel]
        self.cells = np.arange(sel.sum())
        self.no_access = {k: np.array(v) == 0
                          for k, v in config.ACCESS.items()}

        # Expected damages ρ as perceived by agents
        k = perception(options, p)
        self.dmg = {n: k * v[sel] for n, v in damages["expected"].items()}
        self.dmg_protec = {n: k * v[sel]
                           for n, v in damages["expected_protec"].items()}

        # Price of the composite good for FP households, 1 + γ ρ^content_FP:
        # contents worth γ z are destroyed at the expected rate ρ^content
        self.formal_price = 1 + (p["fraction_z_dwellings"]
                                 * self.dmg["contents_formal"])

        # Table of f(Q) = (Q - q0) / (Q - α q0)^α, the left side of (5) up
        # to the factor (α ỹ / (1 + γ ρ^content))^α A(x), sorted by f(Q) to
        # invert it by interpolation
        with np.errstate(invalid="ignore"):
            f = (Q_GRID - p["q0"]) / (Q_GRID - p["alpha"] * p["q0"]) \
                ** p["alpha"]
        ok = np.isfinite(f)
        order = np.argsort(f[ok], kind="mergesort")
        self.f_grid, self.q_grid = f[ok][order], Q_GRID[ok][order]

    # --- Bid rents ψ_i^h(x, u), (groups, cells) -----------------------------

    def formal_bids(self, u):
        """Bid rents ψ_i^FP (6) and dwelling sizes Q_FP (5).

        FP households choose q and z: with price c = 1 + γ ρ^content for z,
        maximising (1) subject to (2), c z + R q = ỹ, gives the Stone-Geary
        demands R (q - q0) = (1 - α) (ỹ - R q0) and c z = α (ỹ - R q0), hence
        R = (1 - α) ỹ / (q - α q0), i.e. (6). Plugging both into (1) gives
        (5), which is solved for q given u: f(Q*) = u c^α / (A (α ỹ)^α).
        """
        p = self.p
        y = np.where(self.y > 0, self.y, np.nan)
        left_side = ((u[:, None] / self.amenities[None, :])
                     * (self.formal_price[None, :] ** p["alpha"])
                     / ((p["alpha"] * y) ** p["alpha"]))
        size = np.interp(left_side, self.f_grid, self.q_grid,
                         left=np.nan, right=np.nan)
        # Developers do not build below the legal minimum lot size
        size = np.maximum(size, p["mini_lot_size"])
        with np.errstate(invalid="ignore"):
            R = p["beta"] * y / (size - (p["alpha"] * p["q0"]))
        return _clean(R), size

    def _composite_good(self, u, pocket):
        """z_h(x, u): composite good yielding utility u in a shack of fixed
        size q_I, from (1): u = z^α (q_I - q0)^(1-α) A(x) B_h."""
        p = self.p
        return ((u[:, None]
                 / (self.amenities[None, :] * pocket
                    * ((p["shack_size"] - p["q0"]) ** p["beta"])))
                ** (1 / p["alpha"]))

    def backyard_bids(self, u):
        """Bid rents ψ_i^IB (8), for dwellings of fixed size q_I."""
        p = self.p
        R = self.backyard_rent(self._composite_good(u, p["pocket_backyard"]))
        return R, np.full(R.shape, float(p["shack_size"]))

    def backyard_rent(self, z):
        """(8): tenants spend (1 + γ ρ^content_IB) z on the composite good
        and pay the rest of their income as rent for q_I m2. RDP owners
        bear the costs of the shack structures (backyard_supply)."""
        p = self.p
        R = (1 / p["shack_size"]) * (
            self.y - ((1 + self.dmg["contents_backyard"][None, :]
                       * p["fraction_z_dwellings"]) * z))
        return _clean(R, self.no_access["backyard"])

    def informal_bids(self, u):
        """Bid rents ψ_i^IS (7), dwelling sizes (q_I) and self-protection.

        Under SP, each group bids its highest bid with or without sandbags.
        Returns the protection choice of the top bidder in each cell (NaN
        without SP).
        """
        p = self.p
        z = self._composite_good(u, p["pocket_informal"])
        R = self.informal_rent(z, protected=False)
        size = np.full(R.shape, float(p["shack_size"]))
        if not self.options["self_protec"]:
            return R, size, np.full(len(self.cells), np.nan)
        R_protec = self.informal_rent(z, protected=True)
        protec = R_protec > R
        R = np.where(protec, R_protec, R)
        top = R.argmax(0)
        mask = protec[top, self.cells] & (R[top, self.cells] > 0)
        return R, size, mask

    def informal_rent(self, z, protected):
        """(7): settlers also pay for their shack, (ρ + δ) v_I for capital
        costs and ρ^struct_IS v_I for expected flood damages, plus the
        sandbag cost if protected (with damages net of protection)."""
        p = self.p
        dmg = self.dmg_protec if protected else self.dmg
        cost = p["sandbag_course_cost"] if protected else 0
        structure_cost = (p["informal_structure_value"]
                          * (p["interest_rate"] + p["depreciation_rate"]))
        R = (1 / p["shack_size"]) * (
            self.y - cost
            - ((1 + dmg["contents_informal"][None, :]
                * p["fraction_z_dwellings"]) * z)
            - structure_cost
            - (dmg["structure_informal_settlements"][None, :]
               * p["informal_structure_value"]))
        return _clean(R, self.no_access["informal"])

    # --- Housing supply s_h(x), m2 per km2 of available land ----------------

    def formal_supply(self, R, size):
        """(9): developers choose capital per unit of land k to maximise (4),
        κ (1-a) R k^(-a) = ρ + ρ^struct_FP + δ, and supply s_FP = κ k^(1-a).

        Structure damages use the two-floor damage function for dwellings
        above param["threshold"] m2. No housing is built where R < R_A, the
        agricultural rent (city edge, Section 4.6 condition (ii)), and
        supply is capped by the height limit.
        """
        p = self.p
        destroyed = np.where(size > p["threshold"],
                             self.dmg["structure_formal_2"],
                             self.dmg["structure_formal_1"])
        supply = (1000000 * (p["coeff_A"] ** (1 / p["coeff_a"]))
                  * ((p["coeff_b"] / (p["interest_rate"]
                                      + p["depreciation_rate"] + destroyed))
                     ** (p["coeff_b"] / p["coeff_a"]))
                  * (R ** (p["coeff_b"] / p["coeff_a"])))
        supply[R < p["agricultural_rent"]] = 0
        return np.minimum(supply, self.housing_limit)

    def backyard_supply(self, R):
        """(10): share μ of their backyard Y that RDP owners (group 1) rent
        out, times 1e6 m2/km2 (shacks cover all the rented space).

        Owners maximise (1) with z (1 + γ ρ^content) = ỹ_1 - (ρ + ρ^struct_FS)
        v_FS + μ Y (R - Z), where Z = (ρ + ρ^struct_IB + δ) v_I / q_I is the
        cost of the shacks per m2, and q = q_FS - μ Y. The first-order
        condition gives (10), which does not depend on ρ^content.

        If R <= Z, renting out loses money: utility decreases with μ, so
        μ = 0 (the first-order condition then has no feasible solution; the
        legacy code still applied (10) and rented out everything). μ is
        bounded to [0, 1].

        Not handled: where ỹ_1 - (ρ + ρ^struct_FS) v_FS <= 0 (heavily
        flooded RDP cells), the owners' composite good is negative whatever
        μ, so (10) is not defined; it is applied anyway (usually μ = 1).
        """
        p = self.p
        rdp_value = p["subsidized_structure_value"]
        shack_cost = ((p["depreciation_rate"]
                       + self.dmg["structure_backyards"]
                       + p["interest_rate"])
                      * (p["informal_structure_value"] / p["shack_size"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            share = ((p["alpha"]
                      * (p["RDP_size"] + p["backyard_size"] - p["q0"])
                      / (p["backyard_size"]))
                     - (p["beta"]
                        * (self.y[0] - (self.dmg["structure_subsidized_1"]
                                        * rdp_value)
                           - (p["depreciation_rate"] * rdp_value))
                        / (p["backyard_size"] * (R - shack_cost))))
        share = np.where(R > shack_cost, share, 0)
        return 1000000 * np.maximum(np.minimum(share, 1), 0)

    # --- Market clearing for given utilities --------------------------------

    def solve_market(self, kind, u):
        """Allocate each cell of market `kind` to the highest bidder, and
        return households (groups, cells), rent, supply and dwelling size of
        the cell, the bid rent matrix, and the protection choice."""
        mask = np.full(len(self.cells), np.nan)
        if kind == "formal":
            R_mat, size = self.formal_bids(u)
        elif kind == "backyard":
            R_mat, size = self.backyard_bids(u)
        else:
            R_mat, size, mask = self.informal_bids(u)

        # Market rent R_h(x) = max_i ψ_i^h(x, u) (main.tex Section 4.3)
        top = R_mat.argmax(0)
        R, size = R_mat[top, self.cells], size[top, self.cells]

        if kind == "formal":
            supply = self.formal_supply(R, size)
        elif kind == "backyard":
            supply = self.backyard_supply(R)
        else:               # one m2 of floor per m2 of settlement land
            supply = 1000000 * np.ones(len(R))
        supply[R == 0] = 0                  # no positive bid

        # (11): households N = s L / Q, all from the top bidder, with L the
        # available share of the cell (0.25 km2)
        with np.errstate(invalid="ignore"):
            density = supply / size
        density[np.isnan(density)] = 0
        winner = (np.arange(len(R_mat))[:, None] == top) & (R > 0)
        households = (density * self.land[TYPES.index(kind)] * 0.25
                      )[None, :] * winner
        if kind == "formal":
            # Reported rent: at least the agricultural rent
            R = np.maximum(R, self.p["agricultural_rent"])
        return {"households": households, "rent": R, "supply": supply,
                "size": size, "rent_matrix": R_mat, "mask": mask}


def _clean(R, no_access=None):
    """Bid rents: zero where negative or undefined, or without access."""
    R = np.where(np.isnan(R) | (R < 0), 0, R)
    if no_access is not None:
        R[no_access] = 0
    return R


def solve(mk, verbose=True):
    """Iterate on utility levels until population targets are met.

    main.tex Section 4.6: starting from p["utility_init"], each iteration
    solves the 3 markets for the current u (conditions (i)-(ii)), compares
    the simulated households of each group, N_i^sim = Σ_h Σ_x N_i^h(x), to
    the target N_i (condition (iii)), and updates
        log u_i <- log u_i + λ_k log((N_i^sim + 10) / (N_i + 10)),
    with 10% larger steps upwards. Too many households of group i means that
    its utility is too low: raising u_i lowers its bids and raises formal
    dwelling sizes, hence lowers N_i^sim. Iterations stop when every
    |N_i^sim / N_i - 1| <= p["precision"], or after p["max_iter"].

    Exact market clearing is often out of reach: a few large cells at a
    near-tie between income groups flip from one group to the other. Unless
    p["return_best"] is False (legacy behaviour), the iterate with the lowest
    max abs error is returned.

    Returns (outputs on the full grid, converged flag).
    """
    p, target = mk.p, mk.inputs["target"]
    max_iter = p["max_iter"]

    def evaluate(u):
        res = [mk.solve_market(kind, u) for kind in TYPES]
        # Households by (housing type FP/IB/IS, income group)
        jobs = np.array([np.sum(r["households"], 1) for r in res])
        total = np.sum(jobs, 0)
        return res, jobs, total, total / target - 1

    def step(total, factor):
        diff = np.log((total + 10) / (target + 10)) * factor
        diff[diff > 0] = diff[diff > 0] * 1.1
        return diff

    u = np.array(p["utility_init"], dtype=float)
    res, jobs, total, error = evaluate(u)
    diff = step(total, p["convergence_factor"])
    best = (np.max(np.abs(error)), 0, res, jobs, u, error)
    it = 0
    while it < max_iter - 1 and np.max(np.abs(error)) > p["precision"]:
        it += 1
        u = np.exp(np.log(u) + diff)
        # Step size λ_k: about convergence_factor / 1.5, shrinking linearly
        # to 40% of that over max_iter iterations. The legacy code meant to
        # damp steps by the current error but read a zero total instead,
        # hence 100 / (target + 100) here; this constant damping converges
        # more often than the intended one, so it is kept.
        factor = (p["convergence_factor"]
                  / (1 + 0.5 * np.abs(100 / (target + 100) - 1))
                  * (1 - 0.6 * it / max_iter))
        res, jobs, total, error = evaluate(u)
        diff = step(total, factor)
        if np.max(np.abs(error)) < best[0]:
            best = (np.max(np.abs(error)), it, res, jobs, u, error)
    n_iter = it + 1
    if p["return_best"]:
        _, it, res, jobs, u, error = best
    converged = np.max(np.abs(error)) <= p["precision"]
    if verbose:
        print(f"{'Converged' if converged else 'NOT converged'}: iterate "
              f"{it + 1} of {n_iter}, max abs error "
              f"{np.max(np.abs(error)):.2e} (precision {p['precision']})")
    return _export(mk, res, u, error, jobs), converged


def _export(mk, res, u, error, jobs):
    """Map solver outputs back to the full grid and add RDP housing.

    Housing types are ordered FP, IB, IS, FS (formal, backyard, informal,
    RDP) and income groups poorest to richest. Outputs (cells: the 24,014
    grid cells):
        utility: (4,) u_i
        error: (4,) N_i^sim / N_i - 1
        simulated_jobs: (3, 4) households by endogenous housing type, group
        households: (4, 4, cells) N_i^h(x), by housing type and group
        households_housing_types, household_centers: (4, cells) sums of
            households over groups and over housing types
        dwelling_size: (4, cells) Q_h(x), m2 (NaN where no bidder)
        housing_supply: (4, cells) s_h(x), m2 per km2 of available land
        rent: (4, cells) R_h(x), rands/m2/year (NaN for RDP)
        rent_matrix: (3, 4, cells) bid rents ψ_i^h(x)
        capital_land: (4, cells) capital per unit of land, k = (s / κ)^(1/(1-a))
        average_income: (4,) mean income per group
        limit_city: (1, 4, 4, cells) households > 1
        mask_self_protec: (cells,) top bidder in informal settlements
            protects (NaN without SP)
    """
    p, inputs, sel = mk.p, mk.inputs, mk.sel
    n = len(sel)
    land_rdp = inputs["coeff_land"][3]
    hh_rdp = inputs["households_RDP"]

    households = np.zeros((4, 4, n))
    for t, r in enumerate(res):
        households[t][:, sel] = r["households"]
    households[3, 0] = hh_rdp            # RDP: poorest group only

    def grid(key, fill):
        out = np.full((3, n), fill, dtype=float)
        out[:, sel] = [r[key] for r in res]
        return out

    size = grid("size", 0)
    size[size <= 0] = np.nan
    size_rdp = p["RDP_size"] * (land_rdp > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # RDP floor space per unit of available land (legacy formula)
        supply_rdp = (p["RDP_size"] / (p["RDP_size"] + p["backyard_size"])
                      * p["RDP_size"] * hh_rdp / (land_rdp * 0.25))
    supply_rdp[np.isnan(supply_rdp)] = 0
    supply = np.vstack([grid("supply", 0), supply_rdp])

    rent_matrix = np.full((3, 4, n), np.nan)
    rent_matrix[:, :, sel] = [r["rent_matrix"] for r in res]
    mask = np.zeros(n)
    mask[sel] = res[2]["mask"]

    return {
        "utility": u,
        "error": error,
        "simulated_jobs": jobs,
        "households_housing_types": households.sum(1),
        "household_centers": households.sum(0),
        "households": households,
        "dwelling_size": np.vstack([size, size_rdp]),
        "housing_supply": supply,
        "rent": np.vstack([grid("rent", np.nan), np.full(n, np.nan)]),
        "rent_matrix": rent_matrix,
        "capital_land": (supply / 1000000 / p["coeff_A"])
        ** (1 / p["coeff_b"]),
        "average_income": inputs["average_income"],
        "limit_city": np.array([households > 1]),
        "mask_self_protec": mask,
    }
