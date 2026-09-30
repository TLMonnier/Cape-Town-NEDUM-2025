"""Static spatial equilibrium with flood risks.

Utility levels of the 4 income groups are adjusted until the simulated number
of households per group matches its target. For given utilities, each of the
3 endogenous housing markets (formal private, informal backyards in RDP
premises, informal settlements) allocates each grid cell to the highest
bidding income group, whose bid rent and dwelling size pin down housing
supply and population. Formal subsidized (RDP) housing is exogenous and only
added to outputs at the end. See the NEDUM technical documentation for the
underlying formulas.

Utility is U = A z^alpha (q - q0)^beta, with z the composite good.

Risk neutrality (RA0): households value expected damages, a certain budget
z (1 + f E[D]) = y - q R - ... (i.e. perfect, actuarially fair insurance).

Risk aversion (RA1): in each flood state i, the (preference-free) budget
gives the composite good z_i, e.g. z_i (1 + f D_i) = y - q R. Households
maximise E[V(U_i)] with V the CRRA utility V(x) = x^(1-CRRA)/(1-CRRA), and
utility levels u are certainty equivalents, V^-1(E[V(U_i)]). As A and q do
not vary across states, this amounts to comparing certainty-equivalent
consumption CE[z_i] = phi^-1(V^-1(E[V(phi(z_i))])), phi(z) = z^alpha, with
the certain z* yielding u. Negative consumption (which can arise with
additive structure damages) is valued with the odd extensions
phi(z) = -(-z)^alpha and V(x) = -(-x)^(1-CRRA)/(1-CRRA) if CRRA < 1, and
with V = -inf if CRRA >= 1.
- Formal housing: only contents are risky, so CE[z_i] = (y - q R) / c* with
  c* = 1 / CE[1 / (1 + f D_i)], and the dwelling size condition keeps its
  certain form with price c*.
- Backyard tenants and informal settlers: bid rents solve CE[z_i(R)] = z* by
  safeguarded Newton (CE[z_i(R)] is decreasing in R; one step when only
  contents are risky).
- RDP owners renting out backyards: the share maximising E[V(U_i)] is found
  on a grid, then refined by golden-section search (with negative
  consumption states the objective need not be concave).
- Formal developers stay risk neutral (expected structure damages).
"""
import numpy as np

import config


def odd_power(x, e):
    """sign(x) |x|^e: power function extended to negative x."""
    return np.sign(x) * np.abs(x) ** e


def crra(x, gamma):
    """CRRA utility x^(1-gamma)/(1-gamma) (log x if gamma = 1). For x <= 0:
    -(-x)^(1-gamma)/(1-gamma) if gamma < 1, -inf otherwise."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if gamma < 1:
            return odd_power(x, 1 - gamma) / (1 - gamma)
        v = np.log(x) if gamma == 1 else x ** (1 - gamma) / (1 - gamma)
        return np.where(x > 0, v, -np.inf)


def crra_inverse(v, gamma):
    """Inverse of crra (-inf maps to 0 if gamma >= 1)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if gamma < 1:
            return odd_power(v * (1 - gamma), 1 / (1 - gamma))
        if gamma == 1:
            return np.exp(v)
        return (v * (1 - gamma)) ** (1 / (1 - gamma))

TYPES = ("formal", "backyard", "informal")

# Dwelling sizes (m2) at which the formal demand condition is tabulated
Q_GRID = np.concatenate((
    [10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3),
     10 ** (-2), 10 ** (-1)],
    np.arange(0.11, 0.15, 0.01), np.arange(0.15, 1.15, 0.05),
    np.arange(1.2, 3.1, 0.1), np.arange(3.5, 13.1, 0.25),
    np.arange(15, 60, 0.5), np.arange(60, 100, 2.5), np.arange(110, 210, 10),
    [250, 300, 500, 1000, 2000, 200000, 1000000, 10 ** 12]))


class Markets:
    """Housing markets restricted to the selected grid cells."""

    def __init__(self, p, inputs, damages, options, sel):
        self.p, self.options = p, options
        self.y = inputs["net_income"][:, sel]
        self.amenities = inputs["amenities"][sel]
        self.land = inputs["coeff_land"][:3, sel]
        self.housing_limit = inputs["housing_limit"][sel]
        self.pix = np.arange(sel.sum())
        self.no_access = {k: np.array(v) == 0
                          for k, v in config.ACCESS.items()}

        # Damages as perceived by agents
        k = p["risk_internaliz"] if options["risk_misperc"] else 1
        self.dmg = {n: k * v[sel] for n, v in damages["expected"].items()}
        self.dmg_protec = {n: k * v[sel]
                           for n, v in damages["expected_protec"].items()}
        self.states = {n: k * v[:, sel] for n, v in damages["states"].items()}
        self.states_protec = {n: k * v[:, sel]
                              for n, v in damages["states_protec"].items()}
        self.proba = damages["proba"]
        # Price of the composite good in formal housing, including contents
        # damages: 1 + f E[D] (RA0) or its certainty equivalent c* (RA1)
        f = p["fraction_z_dwellings"]
        if options["risk_avers"]:
            self.formal_price = 1 / self.ce_consumption(
                1 / (1 + f * self.states["contents_formal"]))
        else:
            self.formal_price = 1 + (f * self.dmg["contents_formal"])

        # Inverse of the formal demand condition: left side -> dwelling size
        with np.errstate(invalid="ignore"):
            f = (Q_GRID - p["q0"]) / (Q_GRID - p["alpha"] * p["q0"]) \
                ** p["alpha"]
        ok = np.isfinite(f)
        order = np.argsort(f[ok], kind="mergesort")
        self.f_grid, self.q_grid = f[ok][order], Q_GRID[ok][order]

    # --- Risk aversion: expected CRRA utility over flood states -------------

    def expected_utility(self, U):
        """E[V(U_i)], flood states on the first axis."""
        return np.tensordot(self.proba, crra(U, self.p["CRRA"]), 1)

    def ce_consumption(self, z):
        """Certain composite good yielding the same expected CRRA utility as
        state-contingent z_i (states on the first axis). State-invariant
        utility factors (A, q) cancel out."""
        alpha = self.p["alpha"]
        U = crra_inverse(self.expected_utility(odd_power(z, alpha)),
                         self.p["CRRA"])
        return odd_power(U, 1 / alpha)

    # --- Bid rents (4 groups x cells) and dwelling sizes --------------------

    def formal_bids(self, u):
        p, y = self.p, self.y
        y_pos = np.where(y < 0, np.nan, y)
        left_side = ((u[:, None] / self.amenities[None, :])
                     * (self.formal_price[None, :] ** p["alpha"])
                     / ((p["alpha"] * y_pos) ** p["alpha"]))
        size = np.interp(left_side, self.f_grid, self.q_grid,
                         left=np.nan, right=np.nan)
        size = np.maximum(size, p["mini_lot_size"])
        with np.errstate(invalid="ignore"):
            R = p["beta"] * y / (size - (p["alpha"] * p["q0"]))
        R[y < 0] = 0
        return R, size

    def _composite_good(self, u, pocket):
        """Composite good consumption yielding utility u in a shack."""
        p = self.p
        return ((u[:, None]
                 / (self.amenities[None, :] * pocket
                    * ((p["shack_size"] - p["q0"]) ** p["beta"])))
                ** (1 / p["alpha"]))

    def backyard_bids(self, u):
        p = self.p
        z = self._composite_good(u, p["pocket_backyard"])
        if self.options["risk_avers"]:
            R = self.crra_bids("backyard", z)
        else:
            R = (1 / p["shack_size"]) * (
                self.y - ((1 + self.dmg["contents_backyard"][None, :]
                           * p["fraction_z_dwellings"]) * z))
            R[self.no_access["backyard"]] = 0
        return R, np.full(R.shape, float(p["shack_size"]))

    def informal_bids(self, u):
        """Bid rents, dwelling sizes and self-protection choice."""
        p, opt = self.p, self.options
        z = self._composite_good(u, p["pocket_informal"])
        structure_cost = (p["informal_structure_value"]
                          * (p["interest_rate"] + p["depreciation_rate"]))

        def bid(protected):
            cost = p["sandbag_course_cost"] if protected else 0
            if opt["risk_avers"]:
                return self.crra_bids("informal", z, protected)
            dmg = self.dmg_protec if protected else self.dmg
            R = (1 / p["shack_size"]) * (
                self.y - cost
                - ((1 + dmg["contents_informal"][None, :]
                    * p["fraction_z_dwellings"]) * z)
                - structure_cost
                - (dmg["structure_informal_settlements"][None, :]
                   * p["informal_structure_value"]))
            return _clean(R, self.no_access["informal"])

        R = bid(False)
        size = np.full(R.shape, float(p["shack_size"]))
        if not opt["self_protec"]:
            return R, size, np.full(len(self.pix), np.nan)

        # Each group protects if it raises its (risk-neutral or certainty-
        # equivalent) bid rent
        R_protec = bid(True)
        protec = R_protec > R
        R = np.where(protec, R_protec, R)
        top = R.argmax(0)
        mask = protec[top, self.pix] & (R[top, self.pix] > 0)
        return R, size, mask

    def crra_bids(self, market, z_target, protected=False):
        """Risk-averse (non state-contingent) bid rents, 4 groups x cells.

        In flood state i, the composite good is
        z_i(R) = (net_i - q R) / (1 + f D^c_i), with net_i = y for backyard
        tenants, and net_i = y - cost - s (r + d) - D^s_i s in settlements
        (structure damage D^s, sandbag cost if protected). The bid rent
        solves CE[z_i(R)] = z_target, the certain composite good yielding
        utility u. Only groups with access and a positive highest
        state-contingent bid can bid a positive rent.
        """
        p = self.p
        states = self.states_protec if protected else self.states
        g, c = np.nonzero(np.repeat(~self.no_access[market][:, None],
                                    len(self.pix), 1))
        price = (1 + p["fraction_z_dwellings"]
                 * states[f"contents_{market}"][:, c])
        net = np.broadcast_to(self.y[g, c], price.shape)
        if market == "informal":
            s = p["informal_structure_value"]
            net = ((self.y[g, c]
                    - (p["sandbag_course_cost"] if protected else 0)
                    - s * (p["interest_rate"] + p["depreciation_rate"]))[None]
                   - (states["structure_informal_settlements"][:, c] * s))
        zt = z_target[g, c]
        keep = np.max(net - price * zt[None], 0) > 0
        R = np.zeros(z_target.shape)
        R[g[keep], c[keep]] = self._ce_bids(
            net[:, keep], price[:, keep], zt[keep], p["shack_size"])
        return _clean(R)

    def _ce_bids(self, net, price, z_target, q, n_iter=30, tol=1e-9):
        """Solve CE[(net_i - q R) / price_i] = z_target for R (states x
        pairs) by Newton's method, safeguarded by bisection.

        CE decreases with R, so the solution lies between the lowest and
        highest state-contingent bids (net_i - price_i z_target) / q. Newton
        starts from the highest; it is exact in one step when net_i is
        state-invariant (CE is then linear in R), and converges monotonically
        otherwise when CE is concave (e.g. power means of order <= 1 for
        positive consumption).
        """
        R_state = (net - price * z_target[None]) / q
        low, high = R_state.min(0), R_state.max(0)
        R = high.copy()
        # d CE / d R = E[psi'(z_i) dz_i/dR] / psi'(CE), psi = V o phi,
        # psi'(z) proportional to |z|^(alpha (1 - CRRA) - 1)
        e = self.p["alpha"] * (1 - self.p["CRRA"]) - 1
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _ in range(n_iter):
                z = (net - q * R[None]) / price
                ce = self.ce_consumption(z)
                gap = ce - z_target
                low = np.where(gap > 0, R, low)
                high = np.where(gap <= 0, R, high)
                slope = (-q * np.tensordot(self.proba, np.abs(z) ** e / price,
                                           1) / np.abs(ce) ** e)
                new = R - gap / slope
                new = np.where(np.isfinite(new) & (new >= low) & (new <= high),
                               new, (low + high) / 2)
                done = np.abs(new - R) <= tol * np.maximum(1, np.abs(R))
                R = new
                if done.all():
                    break
        return R

    # --- Housing supply per km2 of available land ---------------------------

    def formal_supply(self, R, size):
        p = self.p
        # Two-floor damage function above the size threshold
        destroyed = np.where(
            size > p["threshold"], self.dmg["structure_formal_2"],
            np.where(size <= p["threshold"],
                     self.dmg["structure_formal_1"], 1))
        with np.errstate(invalid="ignore"):
            supply = (1000000 * (p["coeff_A"] ** (1 / p["coeff_a"]))
                      * ((p["coeff_b"] / (p["interest_rate"]
                                          + p["depreciation_rate"]
                                          + destroyed))
                         ** (p["coeff_b"] / p["coeff_a"]))
                      * (R ** (p["coeff_b"] / p["coeff_a"])))
        supply[R < p["agricultural_rent"]] = 0
        supply[np.isnan(supply)] = 0
        supply[supply < 0] = 0
        return np.minimum(supply, self.housing_limit)

    def backyard_supply(self, R):
        """Share of RDP backyards rented out by (poorest) RDP owners.

        Owners rent out a share of their backyard, where they build shacks,
        trading own housing space for rent net of shack costs. They bear
        damages to their house and to the shacks. RDP houses (40m2) are below
        the two-floor threshold.
        """
        p = self.p
        if self.options["risk_avers"]:
            return 1000000 * self._crra_backyard_share(R)
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
        # The first-order condition is a minimum when renting out loses
        # money (legacy code then rented out everything)
        share = np.where(R > shack_cost, share, 0)
        return 1000000 * np.maximum(np.minimum(share, 1), 0)

    def _crra_backyard_share(self, R):
        """Backyard share m maximising the owners' expected CRRA utility.

        State i: U_i = phi(z_i) (H - B m)^beta, with composite good
        z_i = y - (d + D^h_i) V + B m (R - Z_i), B the backyard size, V the
        RDP house value, Z_i = (d + D^s_i + r) s / q the shack cost per m2
        and H - B m own housing. Only cells with a positive rent matter.
        """
        p = self.p
        B, V = p["backyard_size"], p["subsidized_structure_value"]
        income = (self.y[0] - p["depreciation_rate"] * V
                  - self.states["structure_subsidized_1"] * V)
        margin = R - ((p["depreciation_rate"]
                       + self.states["structure_backyards"]
                       + p["interest_rate"])
                      * (p["informal_structure_value"] / p["shack_size"]))
        # z_i is linear in m: positive on [0, 1] iff positive at both ends
        regular = ((np.min(income, 0) > 0)
                   & (np.min(income + B * margin, 0) > 0))
        share = np.zeros(len(R))
        for cells, solve in [(regular & (R > 0), self._share_newton),
                             (~regular & (R > 0), self._share_search)]:
            if cells.any():
                share[cells] = solve(income[:, cells], margin[:, cells])
        return share

    def _share_newton(self, income, margin, n_iter=30, tol=1e-12):
        """Optimal share when consumption is positive in all states.

        The log of the certainty-equivalent utility,
        beta log(H - B m) + alpha log CE[z_i(m)], is concave in m (power
        means of order <= 1 of linear functions are concave), so its slope
        g = -beta B / h + alpha B S1 / S0, with S_k = E[z_i^(a-k) m_i^k] and
        a = alpha (1 - CRRA), decreases: safeguarded Newton on [0, 1].
        """
        p = self.p
        alpha, beta, B = p["alpha"], p["beta"], p["backyard_size"]
        H = p["RDP_size"] + B - p["q0"]
        a = alpha * (1 - p["CRRA"])

        def slope(m, inc, mar):
            z = inc + B * m * mar
            z2 = z ** (a - 2)
            S0, S1, S2 = (np.tensordot(self.proba, v, 1)
                          for v in (z2 * z * z, z2 * z * mar, z2 * mar * mar))
            h = H - B * m
            return (-beta * B / h + alpha * B * S1 / S0,
                    -beta * B ** 2 / h ** 2
                    + alpha * B ** 2 * ((a - 1) * S2 * S0 - a * S1 ** 2)
                    / S0 ** 2)

        n = income.shape[1]
        share = np.where(slope(np.zeros(n), income, margin)[0] <= 0, 0.,
                         np.where(slope(np.ones(n), income, margin)[0] >= 0,
                                  1., np.nan))
        inner = np.isnan(share)
        inc, mar = income[:, inner], margin[:, inner]
        low, high = np.zeros(inner.sum()), np.ones(inner.sum())
        m = np.full(inner.sum(), 0.5)
        for _ in range(n_iter):
            g, dg = slope(m, inc, mar)
            low, high = np.where(g > 0, m, low), np.where(g > 0, high, m)
            new = m - g / dg
            new = np.where(np.isfinite(new) & (new >= low) & (new <= high),
                           new, (low + high) / 2)
            done = np.abs(new - m) <= tol
            m = new
            if done.all():
                break
        share[inner] = m
        return share

    def _share_search(self, income, margin, n_grid=100, n_iter=30):
        """Optimal share when consumption can be negative in some state.

        With the odd extension of utility, E[V(U_i)] has a cusp wherever
        some z_i changes sign and can have several local maxima (e.g. on
        both sides of a cusp). It is evaluated on a grid over [0, 1] that
        includes these sign changes, so that the objective is smooth between
        consecutive points, then refined by golden-section search on the
        intervals on both sides of the best point (precision ~1e-8)."""
        p = self.p
        B = p["backyard_size"]
        H = p["RDP_size"] + B - p["q0"]

        def objective(m):
            """E[V(U_i)] for shares m (cells, or points x cells)."""
            m = m[..., None, :] if m.ndim == 2 else m
            U = (odd_power(income + B * m * margin, p["alpha"])
                 * (H - B * m) ** p["beta"])
            v = np.tensordot(crra(U, p["CRRA"]), self.proba, (-2, 0))
            return np.where(np.isnan(v), -np.inf, v)

        n = income.shape[1]
        cols = np.arange(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            cusps = -income / (B * margin)
        points = np.sort(np.concatenate([
            np.repeat(np.linspace(0, 1, n_grid + 1)[:, None], n, 1),
            np.where((cusps > 0) & (cusps < 1), cusps, np.nan)]), 0)
        values = objective(points)
        best = np.argmax(values, 0)
        share, value = points[best, cols], values[best, cols]
        anchor = share
        ratio = (np.sqrt(5) - 1) / 2
        for side in (-1, 1):
            other = points[np.clip(best + side, 0, len(points) - 1), cols]
            other = np.where(np.isnan(other), anchor, other)
            low, high = np.minimum(anchor, other), np.maximum(anchor, other)
            for _ in range(n_iter):
                left = high - ratio * (high - low)
                right = low + ratio * (high - low)
                go_left = objective(left) >= objective(right)
                low = np.where(go_left, low, left)
                high = np.where(go_left, right, high)
            candidate = (low + high) / 2
            better = objective(candidate) > value
            share = np.where(better, candidate, share)
            value = np.where(better, objective(candidate), value)
        return share

    # --- Market clearing for given utilities --------------------------------

    def solve_market(self, kind, u):
        mask = np.full(len(self.pix), np.nan)
        if kind == "formal":
            R_mat, size = self.formal_bids(u)
            R_mat = _clean(R_mat)
        elif kind == "backyard":
            R_mat, size = self.backyard_bids(u)
            R_mat = _clean(R_mat)
        else:
            R_mat, size, mask = self.informal_bids(u)

        # Each cell goes to the highest bidder
        top = R_mat.argmax(0)
        limit = ((R_mat == R_mat.max(0)) & (self.y > 0) & (R_mat > 0))
        R, size = R_mat[top, self.pix], size[top, self.pix]

        if kind == "formal":
            supply = self.formal_supply(R, size)
        elif kind == "backyard":
            supply = self.backyard_supply(R)
        else:
            supply = 1000000 * np.ones(len(R))
        supply[R == 0] = 0

        with np.errstate(invalid="ignore"):
            density = supply / size * (limit.sum(0) > 0)
        density[np.isnan(density)] = 0
        households = (density * self.land[TYPES.index(kind)] * 0.25
                      )[None, :] * limit
        if kind == "formal":
            R = np.maximum(R, self.p["agricultural_rent"])
        return {"households": households, "rent": R, "supply": supply,
                "size": size, "rent_matrix": R_mat, "mask": mask}


def _clean(R, no_access=None):
    R = np.where(np.isnan(R) | (R < 0), 0, R)
    if no_access is not None:
        R[no_access] = 0
    return R


def solve(p, inputs, damages, options, verbose=True):
    """Iterate on utility levels until population targets are met.

    Exact market clearing is often out of reach: a few large cells at a
    near-tie between income groups flip from one group to the other. Unless
    p["return_best"] is False (legacy behaviour), the iterate with the lowest
    max abs error is returned. With p["patience"] set, iterations also stop
    once the best error has not improved for that many iterations while some
    group's error keeps changing sign (flipping).
    """
    sel = ((np.sum(inputs["coeff_land"], 0) > 0.01)
           & (np.nanmax(inputs["net_income"], 0) > 0))
    mk = Markets(p, inputs, damages, options, sel)
    target = inputs["target"]
    max_iter = p["max_iter"]

    def evaluate(u):
        res = [mk.solve_market(kind, u) for kind in TYPES]
        jobs = np.array([np.sum(r["households"], 1) for r in res])
        return res, jobs, np.sum(jobs, 0)

    def step(total, factor):
        diff = np.log((total + 10) / (target + 10)) * factor
        diff[diff > 0] = diff[diff > 0] * 1.1
        return diff

    u = np.array(p["utility_init"], dtype=float)
    res, jobs, total = evaluate(u)
    diff = step(total, p["convergence_factor"])
    error = total / target - 1
    history = [error]
    best = (np.max(np.abs(error)), 0, res, jobs, u, error)
    it = 0
    while it < max_iter - 1 and np.max(np.abs(error)) > p["precision"]:
        it += 1
        u = np.exp(np.log(u) + diff)
        # Damped steps, shrinking over iterations. The legacy code meant to
        # damp by the current error but read a zero total instead; this
        # constant damping (about 1/1.5) converges more often than the
        # intended one, so it is kept (bit-identical to legacy).
        factor = (p["convergence_factor"]
                  / (1 + 0.5 * np.abs(100 / (target + 100) - 1))
                  * (1 - 0.6 * it / max_iter))
        res, jobs, total = evaluate(u)
        diff = step(total, factor)
        error = total / target - 1
        history.append(error)
        if np.max(np.abs(error)) < best[0]:
            best = (np.max(np.abs(error)), it, res, jobs, u, error)
        elif p["patience"] and it - best[1] >= p["patience"]:
            signs = np.sign(history[best[1]:])
            if (np.diff(signs, axis=0) != 0).sum(0).max() >= 2:
                break
    n_iter = it + 1
    if p["return_best"]:
        _, it, res, jobs, u, error = best
    converged = np.max(np.abs(error)) <= p["precision"]
    if verbose:
        print(f"{'Converged' if converged else 'NOT converged'}: iterate "
              f"{it + 1} of {n_iter}, max abs error "
              f"{np.max(np.abs(error)):.2e} (precision {p['precision']})")
    return _export(p, inputs, sel, res, u, error, jobs), converged


def _export(p, inputs, sel, res, u, error, jobs):
    """Map solver outputs back to the full grid and add RDP housing."""
    n = len(sel)
    land_rdp = inputs["coeff_land"][3]
    hh_rdp = inputs["households_RDP"]

    households = np.zeros((4, 4, n))
    for t, r in enumerate(res):
        households[t][:, sel] = r["households"]
    households[3, 0] = hh_rdp

    def grid(key, fill):
        out = np.full((3, n), fill, dtype=float)
        out[:, sel] = [r[key] for r in res]
        return out

    size = grid("size", 0)
    size[size <= 0] = np.nan
    size_rdp = p["RDP_size"] * (land_rdp > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # RDP supply per unit of available land (house only, no backyard)
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
