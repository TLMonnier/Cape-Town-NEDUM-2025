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

Subsidised insurance (SI1, not in main.tex): households of the insured
groups (config.INSURED_GROUPS, groups 1-2) are reimbursed a share s
(p["insurance_share"], run.insurance_terms) of the damages to their own
assets: contents in every housing type, settlers' shacks, RDP houses and
RDP owners' backyard shacks. Formal developers' structures are not insured.
Insured households only bear and perceive (1 - s) k x actual damages (moral
hazard). Backyards, settlements and RDP housing only host insured groups,
so their damages are scaled by 1 - s for everyone; in formal housing, the
composite good price is 1 + γ (1 - s_i) ρ^content_FP per group i.

Self-protection (SP option, not in main.tex): informal settlers choose a
number of sandbag levels k = 0, ..., SP (at most 3). k levels cost
k c_SP a year (c_SP = sandbag_course_cost) and raise the floor by
k x sandbag_height, which lowers expected damages to ρ^k (floods.py). In
each cell, each group picks the k that maximises its bid rent ψ_i^IS(k)
(the lowest k if tied), i.e. the protection level that is best for it.

RDP owners' utility floor (p["rdp_utility_floor"], not in main.tex): RDP
owners (group 1) do not move, so their utility
    U_FS(x) = z^α (q_FS - μ(x) Y - q0)^(1-α) A(x)
is not equalised with u_1. Where their own expected flood damages
D(x) = ρ^struct_FS(x) v_FS + γ ρ^content_FS(x) z (as perceived, net of
insurance) would push U_FS(x) below u_1, the government pays them the
smallest lump-sum transfer T(x) that lifts their utility to u_1, and at most
their damages: T(x) <= D(x). Owners anticipate T(x): it adds to ỹ_1(x) in
their budget (2) and in (10), but they still pay 1 + γ ρ^content_FS at the
margin (Markets.rdp_transfer). T is financed by absentee landlords, so it
does not feed back on the equilibrium; solve reports it with the landlords'
revenues.

Units: rents in rands per m2 per year; housing supply s_h in m2 of floor
per km2 of *available* land; L_h(x) as a share of the cell area (0.25 km2).
"""
import copy

import numpy as np

import accounting
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


# Damage series borne by formal developers (structures) or by FP households
# of any income group (contents): not scaled by the insured share 1 - s in
# Markets.dmg (FP contents insurance enters formal_price by group). All
# other series are borne by insured groups only (backyards, settlements,
# RDP owners).
FORMAL_SERIES = ("structure_formal_1", "structure_formal_2", "contents_formal")


def coverage(options, p):
    """(4,) share s_i of their own damages reimbursed to households of group
    i: p["insurance_share"] for config.INSURED_GROUPS under SI1, else 0."""
    s = np.zeros(4)
    if options["subsid_insur"]:
        s[list(config.INSURED_GROUPS)] = p["insurance_share"]
    return s


def perception(options, p):
    """Factor k such that agents perceive k x actual damages: 0 if agents
    do not anticipate floods (AF0), risk_internaliz under risk misperception
    (RM1), 1 otherwise (before insurance, see Markets.factor)."""
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
        self.damages = damages                          # actual damages
        self.sel = sel = ((np.sum(inputs["coeff_land"], 0) > 0.01)
                          & (np.nanmax(inputs["net_income"], 0) > 0))
        self.y = inputs["net_income"][:, sel]           # ỹ_i(x)
        self.y_pretax = self.y
        self.tax = np.zeros(4)                          # lump sums T_i
        # Developers: lump-sum tax t per m2 of land and agricultural rent
        self.developer_tax, self.agricultural_rent = 0., p["agricultural_rent"]
        self.amenities = inputs["amenities"][sel]       # A(x)
        self.land = inputs["coeff_land"][:3, sel]       # L_h(x), h = FP, IB, IS
        self.housing_limit = inputs["housing_limit"][sel]
        self.rdp_cells = inputs["households_RDP"][sel] > 0
        self.cells = np.arange(sel.sum())
        self.no_access = {k: np.array(v) == 0
                          for k, v in config.ACCESS.items()}

        # Price of a sandbag level for settlers: 0 when subsidised (PS)
        if options["subsid_protec"] and not options["self_protec"]:
            raise ValueError("subsid_protec requires self_protec >= 1")
        self.sandbag_price = (0 if options["subsid_protec"]
                              else p["sandbag_course_cost"])

        # Insurance (SI1): share s_i of own damages reimbursed, (4,)
        if options["subsid_insur"] and p["insurance_share"] is None:
            raise ValueError("SI1 needs param['insurance_share'] "
                             "(see run.insurance_terms)")
        self.coverage = coverage(options, p)
        if options["subsid_insur"] and not all(
                self.coverage[i] == self.coverage[0]
                for i in range(4) if config.ACCESS["backyard"][i]
                or config.ACCESS["informal"][i]):
            raise ValueError("config.INSURED_GROUPS must include group 1 "
                             "and every group with access to backyards and "
                             "settlements")

        # Expected damages ρ as perceived by agents and borne net of
        # insurance: k x actual damages for formal series, k (1 - s) x
        # actual damages for the others (insured groups only)
        k = perception(options, p)
        s = self.coverage[0]
        self.factor = {"formal": k, "own": k * (1 - s) if s else k}
        self.dmg = {n: self.perceived_factor(n) * v[sel]
                    for n, v in damages["expected"].items()}
        # With `level` sandbag levels (dmg_protec[level], level >= 1)
        self.dmg_protec = {level: {n: self.perceived_factor(n) * v[sel]
                                   for n, v in d.items()}
                           for level, d in damages["expected_protec"].items()}

        # Price of the composite good for FP households of group i,
        # 1 + γ (1 - s_i) ρ^content_FP, (groups, cells): contents worth γ z
        # are destroyed at the expected rate ρ^content, a share s_i of which
        # is reimbursed
        self.formal_price = 1 + (p["fraction_z_dwellings"]
                                 * (1 - self.coverage)[:, None]
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

    def perceived_factor(self, name):
        """Factor applied to the actual damage series `name` for the damages
        agents perceive and bear: k for formal series (FORMAL_SERIES), and
        k (1 - s) for series borne by insured groups only (SI1)."""
        return self.factor["formal" if name in FORMAL_SERIES else "own"]

    def set_tax(self, tax):
        """Lump-sum taxes T_i (rands a year, (4,)), paid by every household
        of group i wherever it lives: net income becomes ỹ_i(x) - T_i."""
        self.tax = np.asarray(tax, dtype=float)
        self.y = self.y_pretax - self.tax[:, None]

    def set_developer_tax(self, t):
        """Lump-sum tax t on developers (rands per m2 of developed land a
        year), financing their unanticipated structure losses (see solve).
        Capital choice (9) is unchanged, but zero profit (4) now requires
        R s - (ρ + ρ^struct + δ) k - t >= δ P_A at the city edge, i.e.
        R >= R_A(t) = (δ P_A + t)^a (ρ + δ)^(1-a) / (κ a^a (1-a)^(1-a))
        (data.prepare_inputs gives R_A(0))."""
        p = self.p
        self.developer_tax = t
        if not t:
            self.agricultural_rent = p["agricultural_rent"]
            return
        a, b, r = p["coeff_a"], p["coeff_b"], p["interest_rate"]
        self.agricultural_rent = (
            (p["agricultural_price_baseline"] * r + t) ** a
            * (p["depreciation_rate"] + r) ** b
            / (p["coeff_A"] * b ** b * a ** a))

    # --- Bid rents ψ_i^h(x, u), (groups, cells) -----------------------------

    def formal_bids(self, u):
        """Bid rents ψ_i^FP (6) and dwelling sizes Q_FP (5).

        FP households choose q and z: with price c = 1 + γ ρ^content for z,
        maximising (1) subject to (2), c z + R q = ỹ, gives the Stone-Geary
        demands R (q - q0) = (1 - α) (ỹ - R q0) and c z = α (ỹ - R q0), hence
        R = (1 - α) ỹ / (q - α q0), i.e. (6). Plugging both into (1) gives
        (5), which is solved for q given u: f(Q*) = u c^α / (A (α ỹ)^α).
        The price c = formal_price varies by group under insurance (SI1).
        """
        p = self.p
        y = np.where(self.y > 0, self.y, np.nan)
        left_side = ((u[:, None] / self.amenities[None, :])
                     * (self.formal_price ** p["alpha"])
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

        Under SP, each group bids max_k ψ_i^IS(k) over sandbag levels
        k = 0, ..., SP (the lowest k if tied). Returns the level chosen by
        the top bidder in each cell (0 where nobody bids; NaN without SP).
        """
        p = self.p
        z = self._composite_good(u, p["pocket_informal"])
        R = self.informal_rent(z, level=0)
        size = np.full(R.shape, float(p["shack_size"]))
        if not self.options["self_protec"]:
            return R, size, np.full(len(self.cells), np.nan)
        R_levels = np.array([R] + [self.informal_rent(z, level)
                                   for level in self.dmg_protec])
        level = R_levels.argmax(0)                  # (groups, cells)
        R = R_levels.max(0)
        top = R.argmax(0)
        mask = np.where(R[top, self.cells] > 0, level[top, self.cells], 0)
        return R, size, mask

    def informal_rent(self, z, level):
        """(7) with `level` sandbag levels: settlers also pay for their
        shack, (ρ + δ) v_I for capital costs and ρ^struct_IS v_I for
        expected flood damages, plus level x c_SP for sandbags (with damages
        ρ^level net of protection; c_SP = 0 when subsidised):
            ψ_i^IS(k) = [ỹ_i - k c_SP - (1 + γ ρ^content_IS,k) z_IS
                         - (ρ + δ) v_I - ρ^struct_IS,k v_I] / q_I."""
        p = self.p
        dmg = self.dmg_protec[level] if level else self.dmg
        cost = level * self.sandbag_price
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
        supply[R < self.agricultural_rent] = 0
        return np.minimum(supply, self.housing_limit)

    def backyard_supply(self, R, transfer=0):
        """(10): share μ of their backyard Y that RDP owners (group 1) rent
        out, times 1e6 m2/km2 (shacks cover all the rented space).

        Owners maximise (1) with z (1 + γ ρ^content) = ỹ_1 - (ρ + ρ^struct_FS)
        v_FS + T + μ Y (R - Z), where Z = (ρ + ρ^struct_IB + δ) v_I / q_I is
        the cost of the shacks per m2, q = q_FS - μ Y, and T the lump-sum
        transfer of the utility floor (rdp_transfer; 0 without it). The
        first-order condition gives (10), with ỹ_1 + T in place of ỹ_1,
        which does not depend on ρ^content.

        If R <= Z, renting out loses money: utility decreases with μ, so
        μ = 0 (the first-order condition then has no feasible solution; the
        legacy code still applied (10) and rented out everything). μ is
        bounded to [0, 1].

        Without the utility floor: where ỹ_1 - (ρ + ρ^struct_FS) v_FS <= 0
        (heavily flooded RDP cells), the owners' composite good is negative
        whatever μ, so (10) is not defined; it is applied anyway (usually
        μ = 1). The floor's transfer makes their composite good positive.
        """
        p = self.p
        rdp_value = p["subsidized_structure_value"]
        shack_cost = self.shack_cost()
        with np.errstate(divide="ignore", invalid="ignore"):
            share = ((p["alpha"]
                      * (p["RDP_size"] + p["backyard_size"] - p["q0"])
                      / (p["backyard_size"]))
                     - (p["beta"]
                        * (self.y[0] - (self.dmg["structure_subsidized_1"]
                                        * rdp_value)
                           - (p["depreciation_rate"] * rdp_value) + transfer)
                        / (p["backyard_size"] * (R - shack_cost))))
        share = np.where(R > shack_cost, share, 0)
        return 1000000 * np.maximum(np.minimum(share, 1), 0)

    def shack_cost(self):
        """Z = (ρ + ρ^struct_IB + δ) v_I / q_I: annual cost per m2 of the
        shacks RDP owners build in their backyard."""
        p = self.p
        return ((p["depreciation_rate"] + self.dmg["structure_backyards"]
                 + p["interest_rate"])
                * (p["informal_structure_value"] / p["shack_size"]))

    def rdp_consumption(self, share, R, transfer=0):
        """Composite good z of RDP owners renting out a share μ of their
        backyard at rent R, from their budget (2) with the floor's transfer T
        (rdp_transfer; 0 without it):
            (1 + γ ρ^content_FS) z = ỹ_1 - (ρ + ρ^struct_FS) v_FS + T
                                     + μ Y (R - Z).
        Damages are perceived and net of insurance (SI1)."""
        p = self.p
        V, B = p["subsidized_structure_value"], p["backyard_size"]
        g = p["fraction_z_dwellings"] * self.dmg["contents_subsidized"]
        house = self.dmg["structure_subsidized_1"] * V
        income = (self.y[0] - p["depreciation_rate"] * V
                  + share * B * (R - self.shack_cost()) + transfer)
        return (income - house) / (1 + g)

    # --- RDP owners' utility floor ------------------------------------------

    def rdp_utility(self, R, transfer):
        """Utility of RDP owners (1) at their optimal backyard share, for a
        backyard rent R and a lump-sum transfer T:
            U_FS = z^α (q_FS - μ Y - q0)^(1-α) A(x)   (NaN if z <= 0),
        and their composite good z."""
        p = self.p
        share = self.backyard_supply(R, transfer) / 1000000
        z = self.rdp_consumption(share, R, transfer)
        with np.errstate(invalid="ignore"):
            U = (np.where(z > 0, z, np.nan) ** p["alpha"]
                 * (p["RDP_size"] + p["backyard_size"] * (1 - share)
                    - p["q0"]) ** p["beta"] * self.amenities)
        return U, z

    def rdp_borne(self, z):
        """Own expected flood damages borne by RDP owners (as perceived, net
        of insurance), D = ρ^struct_FS v_FS + γ ρ^content_FS z. Backyard
        shacks are a cost of renting out (Z), not counted."""
        p = self.p
        return (self.dmg["structure_subsidized_1"]
                * p["subsidized_structure_value"]
                + p["fraction_z_dwellings"] * self.dmg["contents_subsidized"]
                * z)

    def rdp_transfer(self, R, u_1):
        """Lump-sum transfer T(x) of the utility floor (module docstring),
        per RDP household, for backyard rent R and group 1's utility u_1.

        T = 0 where U_FS(x, T = 0) >= u_1 (or without RDP households, or
        p["rdp_utility_floor"] off). Elsewhere, T is the smallest transfer
        such that U_FS(x, T) >= u_1, or T = D(T) if that comes first
        (transfers never exceed the owners' damages). Both U_FS(T) and
        T - D(T) increase with T (D rises by at most γ ρ / (1 + γ ρ) < 1 per
        rand of T), so T is found by bisection on
        [0, ρ^struct_FS v_FS + γ ρ^content_FS max(ỹ_1 - ρ v_FS + Y (R - Z)^+, 0) + 1],
        whose upper end satisfies T > D(T). Owners re-choose μ for each T.
        The bisection only runs on the cells below the floor, until T is
        known within 1e-6 rands.
        """
        T = np.zeros(len(R))
        if not self.p["rdp_utility_floor"]:
            return T
        U, _ = self.rdp_utility(R, T)
        need = self.rdp_cells & ~(U >= u_1)          # incl. z <= 0 (NaN)
        if not need.any():
            return T
        p, sub, R = self.p, self._subset(need), R[need]
        V, B = p["subsidized_structure_value"], p["backyard_size"]
        low = np.zeros(len(R))
        high = (sub.dmg["structure_subsidized_1"] * V
                + p["fraction_z_dwellings"] * sub.dmg["contents_subsidized"]
                * np.maximum(sub.y[0] - p["depreciation_rate"] * V
                             + B * np.maximum(R - sub.shack_cost(), 0), 0)
                + 1)
        while np.max(high - low) > 1e-6:
            mid = (low + high) / 2
            U, z = sub.rdp_utility(R, mid)
            more = ~(U >= u_1) & (mid < sub.rdp_borne(z))
            low, high = np.where(more, mid, low), np.where(more, high, mid)
        T[need] = high
        return T

    def _subset(self, cells):
        """Shallow copy of these markets restricted to `cells` (boolean over
        the solver cells), for the RDP owners' methods."""
        sub = copy.copy(self)
        sub.y = self.y[:, cells]
        sub.amenities = self.amenities[cells]
        sub.dmg = {n: v[cells] for n, v in self.dmg.items()}
        sub.rdp_cells = self.rdp_cells[cells]
        sub.cells = np.arange(cells.sum())
        return sub

    # --- Market clearing for given utilities --------------------------------

    def solve_market(self, kind, u):
        """Allocate each cell of market `kind` to the highest bidder, and
        return households (groups, cells), rent, supply and dwelling size of
        the cell, the bid rent matrix, the protection choice, and the RDP
        owners' floor transfer (rdp_transfer; zeros outside backyards)."""
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

        transfer = np.zeros(len(R))
        if kind == "formal":
            supply = self.formal_supply(R, size)
        elif kind == "backyard":
            # RDP owners' utility floor at group 1's current utility u_1
            transfer = self.rdp_transfer(R, u[0])
            supply = self.backyard_supply(R, transfer)
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
            R = np.maximum(R, self.agricultural_rent)
        return {"households": households, "rent": R, "supply": supply,
                "size": size, "rent_matrix": R_mat, "mask": mask,
                "transfer": transfer}


def _clean(R, no_access=None):
    """Bid rents: zero where negative or undefined, or without access."""
    R = np.where(np.isnan(R) | (R < 0), 0, R)
    if no_access is not None:
        R[no_access] = 0
    return R


def solve(mk, verbose=True):
    """Solve the equilibrium (_solve_budgets), and add the RDP owners'
    utility floor costs and the absentee landlords' revenues, which finance
    them (no feedback on the equilibrium: landlords live outside the city).

    Returns (outputs on the full grid, converged flag), with outputs from
    _export and _solve_budgets, plus (rands a year):
        rdp_transfer: (2, cells) floor transfers per RDP household: [0]
            anticipated in equilibrium, with perceived damages
            (Markets.rdp_transfer); [1] ex-post top-up, so that ex-post
            utility with actual damages stays >= u_1 at the equilibrium
            backyard share and spending (accounting.ex_post; 0 when damages
            are perceived correctly)
        surplus_damages: (2,) totals of both rows over RDP households
        landlord_revenue: (2, cells) absentee landlords' revenues from
            formal land and informal settlements
            (accounting.landlord_revenue)
    """
    outputs, converged = _solve_budgets(mk, verbose)
    sel = mk.sel
    if mk.p["rdp_utility_floor"]:
        outputs["rdp_transfer"][1, sel] = \
            accounting.ex_post(mk, outputs)["rdp_topup"]
    hh = outputs["households"][3, 0]
    outputs["surplus_damages"] = np.sum(outputs["rdp_transfer"] * hh, 1)
    revenue = np.zeros((2, len(sel)))
    revenue[:, sel] = accounting.landlord_revenue(mk, outputs)
    outputs["landlord_revenue"] = revenue
    if verbose and mk.p["rdp_utility_floor"]:
        floor = outputs["rdp_transfer"] > 0
        total = outputs["surplus_damages"].sum()
        print(f"RDP utility floor: {hh[floor[0]].sum():,.0f} households "
              f"({floor[0].sum()} cells) get {outputs['surplus_damages'][0]:,.0f}"
              f" rands/year in equilibrium, {hh[floor[1]].sum():,.0f} "
              f"({floor[1].sum()} cells) {outputs['surplus_damages'][1]:,.0f} "
              f"ex post; landlords' revenues {revenue.sum() / 1e6:,.1f} M "
              f"rands/year (formal {revenue[0].sum() / 1e6:,.1f} M, "
              f"settlements {revenue[1].sum() / 1e6:,.1f} M), i.e. a "
              f"{100 * total / revenue.sum():.4f}% levy")
    return outputs, converged


def _solve_budgets(mk, verbose=True):
    """Solve the equilibrium, balancing public schemes (PS, SI) and the
    developers' loss tax when they apply.

    Public schemes (not in main.tex), possibly combined:
    - PS: sandbags are free for settlers, and the government pays
      c_SP Σ_x N^IS(x) k(x), with k(x) the number of levels chosen by the
      settlers of cell x;
    - SI1: insurance of groups 1-2's own assets (share s) at no premium,
      the government pays the reimbursements (accounting.insurance_payouts).
    Their total cost C is financed by lump-sum taxes on the richest groups
    i in config.TAXED_GROUPS (3 and 4 in main.tex, found in formal housing
    only), proportional to their exogenous mean income ȳ_i:
        T_i = τ ȳ_i,  with  Σ_i N_i T_i = C,  i.e.  τ = C / Σ_i N_i ȳ_i,
    N_i the (closed-city) number of households of group i. Taxes lower
    these groups' net income everywhere (Markets.set_tax).

    Developers' loss tax (p["developer_loss_tax"], when damages are
    misperceived: AF0 or RM1): developers bear actual structure damages but
    build on perceived ones. Their unanticipated losses
    L = Σ_x (ρ^struct_actual - ρ^struct_perceived)(x) K(x), with K(x) the
    capital in cell x (developers are not insured), are financed by a
    lump-sum tax t per m2 of developed land, t = L / developed land, which
    developers anticipate (Markets.set_developer_tax), so that zero profit
    holds ex post.

    C and L depend on the equilibrium, which depends on T and t: they are
    balanced by fixed point iteration (T = t = 0, equilibrium, C and L,
    T(C) and t(L), equilibrium, ...) until C and L change by less than
    p["budget_tol"] (relative) or after p["max_iter_budget"] rounds.

    Returns (outputs on the full grid, converged flag). With a public
    scheme, outputs also hold "tax" (T_i, (4,)) and "budget" (sandbag
    subsidy, insurance payouts, tax revenue; rands a year), and under SI1
    "insurance_share" (s, (1,)); with the developers' tax, "developer_tax" (t, losses L, tax revenue). converged
    also requires balanced budgets.
    """
    p, inputs, opt = mk.p, mk.inputs, mk.options
    public = bool(opt["subsid_protec"] or opt["subsid_insur"])
    developers = bool(p["developer_loss_tax"] and (
        not opt["agents_anticipate_floods"] or opt["risk_misperc"]))
    if not (public or developers):
        return _solve_utilities(mk, verbose)
    taxed = np.isin(np.arange(4), config.TAXED_GROUPS)
    base = np.sum((inputs["target"] * inputs["average_income"])[taxed])
    tax, t = np.zeros(4), 0.
    previous = None
    for k in range(p["max_iter_budget"]):
        mk.set_tax(tax)
        mk.set_developer_tax(t)
        outputs, converged = _solve_utilities(mk, verbose)
        sandbags = 0.
        if opt["subsid_protec"]:
            settlers = np.nansum(outputs["households"][2], 0)
            sandbags = p["sandbag_course_cost"] * np.sum(
                settlers * np.nan_to_num(outputs["mask_self_protec"]))
        payouts = (accounting.insurance_payouts(mk, outputs)
                   if opt["subsid_insur"] else 0.)
        cost, revenue = sandbags + payouts, np.sum(inputs["target"] * tax)
        losses, land = (accounting.developer_losses(mk, outputs)
                        if developers else (0., 0.))
        dev_revenue = t * land
        if verbose:
            print(f"  budget round {k + 1}: public cost {cost:,.0f} "
                  f"(sandbags {sandbags:,.0f}, insurance {payouts:,.0f}) vs "
                  f"taxes {revenue:,.0f}; developers' losses {losses:,.0f} "
                  f"vs tax {dev_revenue:,.0f} rands/year")
        current = np.array([cost, losses])
        done = (previous is not None and np.all(
            np.abs(current - previous) <= p["budget_tol"] * current))
        previous = current
        if done or (cost == revenue == 0 and losses == dev_revenue == 0):
            break
        tax = np.where(taxed, cost / base * inputs["average_income"], 0)
        t = losses / land if land else 0.
    gaps = (abs(cost - revenue) / max(cost, 1),
            abs(losses - dev_revenue) / max(losses, 1))
    balanced = max(gaps) <= p["budget_tol"]
    if verbose:
        print(f"Budgets {'balanced' if balanced else 'NOT balanced'} "
              f"(relative gaps {gaps[0]:.1e}, {gaps[1]:.1e}): taxes "
              f"{np.round(mk.tax[taxed])} rands/year on groups "
              f"{np.flatnonzero(taxed)}, developers' tax "
              f"{mk.developer_tax:.3f} rands/m2/year")
    if public:
        outputs["tax"] = mk.tax.copy()
        outputs["budget"] = np.array([sandbags, payouts, revenue])
    if opt["subsid_insur"]:
        outputs["insurance_share"] = np.array([p["insurance_share"]])
    if developers:
        outputs["developer_tax"] = np.array([mk.developer_tax, losses,
                                             dev_revenue])
    return outputs, converged and balanced


def _solve_utilities(mk, verbose=True):
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
        mask_self_protec: (cells,) number of sandbag levels chosen by the
            top bidder in informal settlements (0: none; NaN without SP)
        rdp_transfer: (2, cells) RDP owners' floor transfer per household,
            rands/year: [0] in equilibrium (Markets.rdp_transfer), [1]
            ex-post top-up (zeros here, filled by solve)
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
    transfer = np.zeros((2, n))
    transfer[0, sel] = res[1]["transfer"]

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
        "rdp_transfer": transfer,
    }
