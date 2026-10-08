"""Risk-averse households: CRRA expected utility over flood states.

Extends the risk-neutral model in ../flood_model, whose inputs, flood
damages and solver are reused (this module puts ../flood_model on sys.path).
CRRAMarkets only overrides how households bid for housing and how RDP owners
rent out their backyards. Equation numbers refer to ../flood_model/main.tex.

Flood states. floods.py splits each year into 11 flood states s, with
probability π_s and damage D̄_s(x) (fraction of capital destroyed); the
expected damage is ρ(x) = Σ_s π_s D̄_s(x). In flood_model, households value
damages at ρ, as if fully insured at an actuarially fair premium. Here they
are uninsured: the budget constraint (2) holds state by state, and the
composite good z_s they are left with varies across states. For instance,
for backyard tenants paying rent R,
    (1 + γ D̄^content_s(x)) z_s = ỹ_i(x) - q_I R.

Preferences. With state utility U_s = z_s^α (q - q0)^(1-α) A(x) B_h (1),
households maximise
    E[V(U_s)] = Σ_s π_s V(U_s),  V(U) = U^(1-θ) / (1-θ),
with relative risk aversion θ = param["CRRA"], and utility levels u are
certainty equivalents: u = V^(-1)(E[V(U_s)]). Since A, B_h and q do not vary
across states, a dwelling gives utility u iff the certainty-equivalent
composite good
    CE[z_s] = φ^(-1)(V^(-1)(E[V(φ(z_s))])),  φ(z) = z^α,
            = (Σ_s π_s z_s^(α(1-θ)))^(1 / (α(1-θ)))   (a power mean),
equals the certain composite good z yielding u (z_h(x, u) in solver.py).
Negative consumption (possible when structure damages are subtracted from
income) is valued with the odd extensions φ(z) = -(-z)^α and
V(x) = -(-x)^(1-θ) / (1-θ) if θ < 1, and with V = -inf if θ >= 1. θ = 1
(log utility) is not handled by the backyard supply below.

Household choices (developers stay risk neutral and keep using ρ):
- FP households: only contents are at risk, z_s = (ỹ - q R) / (1 + γ D̄_s),
  so CE[z_s] = (ỹ - q R) / c* with c* = 1 / CE[1 / (1 + γ D̄^content_s)].
  The risk-neutral formulas (5)-(6) apply with c* instead of 1 + γ ρ.
- IB tenants and IS settlers: the bid rent ψ solves CE[z_s(ψ)] = z_h(x, u),
  replacing (7)-(8), with z_s(R) = (net_s - q_I R) / (1 + γ D̄^content_s),
  net_s = ỹ for tenants and, for settlers,
  net_s = ỹ - (ρ + δ) v_I - D̄^struct_s v_I (- k c_SP with k sandbag
  levels under SP). Under SP, settlers choose the number of levels that
  maximises their risk-averse bid.
- RDP owners: the backyard share μ maximises E[V(U_s)] with
  U_s = φ(z_s) (q_FS - μ Y - q0)^(1-α) A(x), replacing (10), where
  z_s = ỹ_1 - (ρ + D̄^struct_FS,s) v_FS + μ Y (R - Z_s) and
  Z_s = (ρ + D̄^struct_IB,s + δ) v_I / q_I. Their contents damages are
  ignored, as in flood_model (where they do not affect μ).

Check: with degenerate states (D̄_s = ρ in every state), this model must
reproduce flood_model (tools/check_degenerate.py).
"""
import sys
from pathlib import Path

import numpy as np

_FLOOD_MODEL = Path(__file__).resolve().parents[1] / "flood_model"
if str(_FLOOD_MODEL) not in sys.path:
    sys.path.insert(0, str(_FLOOD_MODEL))

import config  # noqa: E402  (flood_model)
import solver  # noqa: E402

# Parameters added to config.PARAM
PARAM = {
    "CRRA": 0.2772,                # θ, relative risk aversion
    # RDP owners' utility floor (flood_model solver docstring): not
    # implemented for risk-averse owners
    "rdp_utility_floor": False,
}
OUTPUT = config.ROOT / "Output" / "flood_model_CRRA"


def odd_power(x, e):
    """sign(x) |x|^e: power function extended to negative x."""
    return np.sign(x) * np.abs(x) ** e


def crra(x, gamma):
    """CRRA utility V(x) = x^(1-gamma)/(1-gamma) (log x if gamma = 1). For
    x <= 0: -(-x)^(1-gamma)/(1-gamma) if gamma < 1, -inf otherwise."""
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


class CRRAMarkets(solver.Markets):
    """Housing markets with risk-averse households (see module docstring)."""

    def __init__(self, p, inputs, damages, options):
        super().__init__(p, inputs, damages, options)
        if p["rdp_utility_floor"]:
            raise NotImplementedError("the RDP owners' utility floor is "
                                      "only implemented for risk-neutral "
                                      "agents (set rdp_utility_floor False)")
        # State damages D̄_s as perceived by agents and borne net of
        # insurance (SI1 reimburses a share s in every state), (11, cells):
        # factors k or k (1 - s) as in solver.Markets.dmg
        sel = self.sel
        self.proba = damages["proba"]
        self.states = {n: self.perceived_factor(n) * v[:, sel]
                       for n, v in damages["states"].items()}
        # With `level` sandbag levels (states_protec[level], level >= 1)
        self.states_protec = {
            level: {n: self.perceived_factor(n) * v[:, sel]
                    for n, v in d.items()}
            for level, d in damages["states_protec"].items()}
        # FP households of group i: certainty-equivalent price of the
        # composite good, c*_i = 1 / CE[1 / (1 + γ (1 - s_i) D̄^content_s)],
        # instead of 1 + γ (1 - s_i) ρ^content, (groups, cells)
        f = p["fraction_z_dwellings"]
        self.formal_price = np.array([
            1 / self.ce_consumption(
                1 / (1 + f * (1 - s) * self.states["contents_formal"]))
            for s in self.coverage])

    # --- Expected CRRA utility over flood states ----------------------------

    def expected_utility(self, U):
        """E[V(U_s)], flood states on the first axis."""
        return np.tensordot(self.proba, crra(U, self.p["CRRA"]), 1)

    def ce_consumption(self, z):
        """CE[z_s]: certain composite good yielding the same expected CRRA
        utility as state-contingent z_s (states on the first axis).
        State-invariant utility factors (A, B_h, q) cancel out."""
        alpha = self.p["alpha"]
        U = crra_inverse(self.expected_utility(odd_power(z, alpha)),
                         self.p["CRRA"])
        return odd_power(U, 1 / alpha)

    # --- Bid rents ----------------------------------------------------------

    def backyard_rent(self, z):
        return self.crra_bids("backyard", z)

    def informal_rent(self, z, level):
        return self.crra_bids("informal", z, level)

    def crra_bids(self, market, z_target, level=0):
        """Risk-averse bid rents (groups, cells), solving
        CE[z_s(R)] = z_target.

        In flood state s, the composite good is
        z_s(R) = (net_s - q R) / (1 + γ D̄^content_s), with net_s = ỹ for
        backyard tenants, and net_s = ỹ - k c_SP - (ρ + δ) v_I
        - D̄^struct_s v_I in settlements with k = `level` sandbag levels
        (state damages net of protection). Only groups with access and a
        positive highest state-contingent bid can bid a positive rent.
        """
        p = self.p
        states = self.states_protec[level] if level else self.states
        g, c = np.nonzero(np.repeat(~self.no_access[market][:, None],
                                    len(self.cells), 1))
        price = (1 + p["fraction_z_dwellings"]
                 * states[f"contents_{market}"][:, c])
        net = np.broadcast_to(self.y[g, c], price.shape)
        if market == "informal":
            s = p["informal_structure_value"]
            net = ((self.y[g, c]
                    - level * self.sandbag_price
                    - s * (p["interest_rate"] + p["depreciation_rate"]))[None]
                   - (states["structure_informal_settlements"][:, c] * s))
        zt = z_target[g, c]
        keep = np.max(net - price * zt[None], 0) > 0
        R = np.zeros(z_target.shape)
        R[g[keep], c[keep]] = self._ce_bids(
            net[:, keep], price[:, keep], zt[keep], p["shack_size"])
        return solver._clean(R)

    def _ce_bids(self, net, price, z_target, q, n_iter=30, tol=1e-9):
        """Solve CE[(net_s - q R) / price_s] = z_target for R (states x
        pairs) by Newton's method, safeguarded by bisection.

        CE decreases with R, so the solution lies between the lowest and
        highest state-contingent bids (net_s - price_s z_target) / q. Newton
        starts from the highest; it is exact in one step when net_s is
        state-invariant (CE is then linear in R), and converges monotonically
        otherwise when CE is concave (e.g. power means of order <= 1 for
        positive consumption).
        """
        R_state = (net - price * z_target[None]) / q
        low, high = R_state.min(0), R_state.max(0)
        R = high.copy()
        # d CE / d R = E[ψ'(z_s) dz_s/dR] / ψ'(CE), ψ = V o φ,
        # ψ'(z) proportional to |z|^(α (1 - θ) - 1)
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

    # --- Backyard supply by RDP owners --------------------------------------

    def backyard_supply(self, R, transfer=0):
        # transfer: always 0 (no utility floor under risk aversion)
        return 1000000 * self._crra_backyard_share(R)

    def _crra_backyard_share(self, R):
        """Backyard share μ maximising the owners' expected CRRA utility.

        State s: U_s = φ(z_s) (H - Y μ)^β, with H = q_FS - q0 and composite
        good z_s = income_s + Y μ margin_s, where
        income_s = ỹ_1 - (ρ + D̄^struct_FS,s) v_FS and margin_s = R - Z_s.
        Only cells with a positive rent matter.
        """
        p = self.p
        B, V = p["backyard_size"], p["subsidized_structure_value"]
        income = (self.y[0] - p["depreciation_rate"] * V
                  - self.states["structure_subsidized_1"] * V)
        margin = R - ((p["depreciation_rate"]
                       + self.states["structure_backyards"]
                       + p["interest_rate"])
                      * (p["informal_structure_value"] / p["shack_size"]))
        # z_s is linear in μ: positive on [0, 1] iff positive at both ends
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
        β log(H - Y μ) + α log CE[z_s(μ)], is concave in μ (power means of
        order <= 1 of linear functions are concave), so its slope
        g = -β Y / h + α Y S1 / S0, with S_k = E[z_s^(a-k) margin_s^k] and
        a = α (1 - θ), decreases: safeguarded Newton on [0, 1].
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

        With the odd extension of utility, E[V(U_s)] has a cusp wherever
        some z_s changes sign and can have several local maxima (e.g. on
        both sides of a cusp). It is evaluated on a grid over [0, 1] that
        includes these sign changes, so that the objective is smooth between
        consecutive points, then refined by golden-section search on the
        intervals on both sides of the best point (precision ~1e-8)."""
        p = self.p
        B = p["backyard_size"]
        H = p["RDP_size"] + B - p["q0"]

        def objective(m):
            """E[V(U_s)] for shares m (cells, or points x cells)."""
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
