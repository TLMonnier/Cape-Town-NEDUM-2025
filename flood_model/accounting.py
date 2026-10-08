"""Damage accounting: actual expected annual flood damages, insurance
payouts, developers' losses, ex-post utilities (with the RDP owners' floor
top-up) and absentee landlords' revenues, for an equilibrium (main.tex
Section 6 reports such damages).

Damages are *actual* expected annual damages (rands a year), whatever
agents perceive (AF, RM) or insurance reimburses (SI). In housing type h,
for the households of group i in cell x:
- contents: γ ρ^content_h(x) z, with z the composite good they consume:
    FP: from utility (1), z = (u_i / (A(x) (Q - q0)^(1-α)))^(1/α), with Q
        their dwelling size;
    IB, IS: z_h(x, u_i) (solver.Markets._composite_good);
    FS (RDP owners): from their budget (solver.Markets.rdp_consumption);
- structures: ρ^struct_h(x) x structure value:
    FP: developers' capital in the cell, k L_FP(x) (legacy
        outputs.flood_outputs.compute_formal_structure_cost), with the
        one- or two-floor damage function by dwelling size;
    IB: one shack (v_I) per household, owned by the RDP owner;
    IS: one shack (v_I) per household;
    FS: one RDP house (v_FS) per household.
Damages are attributed to the dwelling's occupants. Settlers' damages use
the number of sandbag levels chosen in their cell. Insurance (SI1) covers a
share s of the damages to the own assets of insured groups (groups 1-2):
contents, and structures except developers' capital (backyard shacks are
owned by RDP owners, of group 1).

Under risk aversion (flood_model_CRRA), z is the certainty-equivalent
composite good.
"""
import numpy as np


TYPES = ("formal", "backyard", "informal", "RDP")


def household_damages(mk, outputs):
    """Return a dict of (4 housing types, 4 groups, solver cells) arrays:
        households: number of households
        contents, structure: total actual damages (rands a year)
        own: damages to the households' own assets (all but developers'
            capital), the base of insurance
        reimbursed: insurance payouts, s_i x own (SI1; zero otherwise)
    """
    p, sel = mk.p, mk.sel
    d = {n: v[sel] for n, v in mk.damages["expected"].items()}
    alpha, beta, q0 = p["alpha"], p["beta"], p["q0"]
    gamma, v_I = p["fraction_z_dwellings"], p["informal_structure_value"]
    hh = outputs["households"][:, :, sel]
    u = outputs["utility"]
    contents, structure = np.zeros_like(hh), np.zeros_like(hh)

    # FP: contents of households, capital of developers
    Q = outputs["dwelling_size"][0][sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (u[:, None] / (mk.amenities * (Q - q0) ** beta)) ** (1 / alpha)
        contents[0] = np.nan_to_num(hh[0] * gamma * d["contents_formal"] * z)
        capital = outputs["capital_land"][0][sel] * mk.land[0] * 250000
        destroyed = np.where(Q > p["threshold"], d["structure_formal_2"],
                             d["structure_formal_1"])
        share = np.nan_to_num(hh[0] / hh[0].sum(0))
    structure[0] = share * capital * destroyed

    # IB: tenants' contents, shacks
    z = mk._composite_good(u, p["pocket_backyard"])
    contents[1] = hh[1] * gamma * d["contents_backyard"] * z
    structure[1] = hh[1] * d["structure_backyards"] * v_I

    # IS: with the sandbag level chosen in each cell (0 without SP)
    level = np.nan_to_num(outputs["mask_self_protec"][sel]).astype(int)
    d_IS = {n: d[n].copy() for n in ("contents_informal",
                                      "structure_informal_settlements")}
    for k, dk in mk.damages["expected_protec"].items():
        for n in d_IS:
            d_IS[n] = np.where(level == k, dk[n][sel], d_IS[n])
    z = mk._composite_good(u, p["pocket_informal"])
    contents[2] = hh[2] * gamma * d_IS["contents_informal"] * z
    structure[2] = hh[2] * d_IS["structure_informal_settlements"] * v_I

    # FS: RDP owners (group 1), renting out a share μ of their backyard
    mu = outputs["housing_supply"][1][sel] / 1000000
    R = np.nan_to_num(outputs["rent"][1][sel])
    z = mk.rdp_consumption(mu, R, outputs["rdp_transfer"][0][sel])
    contents[3, 0] = hh[3, 0] * gamma * d["contents_subsidized"] * z
    structure[3, 0] = (hh[3, 0] * d["structure_subsidized_1"]
                       * p["subsidized_structure_value"])

    own = contents.copy()
    own[1:] += structure[1:]
    return {"households": hh, "contents": contents, "structure": structure,
            "own": own, "reimbursed": mk.coverage[None, :, None] * own}


def insurance_payouts(mk, outputs):
    """Total insurance payouts (rands a year) in an equilibrium."""
    return household_damages(mk, outputs)["reimbursed"].sum()


def developer_losses(mk, outputs):
    """Developers' unanticipated structure losses and developed land.

    Returns (L, land): L = Σ_x (ρ_actual - ρ_perceived)(x) K(x) in rands a
    year, with K(x) the capital in cell x (developers are not insured); land
    is the FP land with positive supply (m2)."""
    p, sel = mk.p, mk.sel
    Q = outputs["dwelling_size"][0][sel]
    two_floors = Q > p["threshold"]
    act = mk.damages["expected"]
    actual = np.where(two_floors, act["structure_formal_2"][sel],
                      act["structure_formal_1"][sel])
    perceived = np.where(two_floors, mk.dmg["structure_formal_2"],
                         mk.dmg["structure_formal_1"])
    land = mk.land[0] * 250000 * (outputs["housing_supply"][0][sel] > 0)
    capital = outputs["capital_land"][0][sel] * land
    return np.sum((actual - perceived) * capital), land.sum()


def ex_post(mk, outputs):
    """Ex-post consumption and utility, with actual instead of perceived
    damages (identical to the equilibrium when agents perceive damages
    correctly).

    Households keep their equilibrium location, dwelling and rent, and the
    budget they planned for the composite good and their own damages,
    S = z + B_perceived(z). Ex post, z_post solves z + B_actual(z) = S, with
    B the damages they bear: contents γ ρ^content z, plus their own
    structures (settlers' shacks; RDP owners' houses and backyard shacks),
    net of insurance payouts (SI1: (1 - s_i) x damages). FP households only
    bear
    contents damages: developers' unanticipated losses are financed by the
    developers' tax (solver.solve). Utility scales with z^α at fixed
    dwelling size: u_post = u (z_post / z)^α.

    RDP owners' utility floor (p["rdp_utility_floor"], solver docstring):
    their planned budget includes the equilibrium transfer T. If ex post
    their utility z_post^α (q_FS - μ Y - q0)^(1-α) A(x) falls below u_1, the
    government tops it up: z_post rises to
        z_floor = (u_1 / ((q_FS - μ Y - q0)^(1-α) A(x)))^(1/α),
    or to the composite good with all their own (house and contents)
    damages paid if that is lower, at a cost
    T_post = z_post' + B_actual(z_post') - (S + T) (rdp_topup).

    Returns a dict of (4 housing types, 4 groups, solver cells) arrays:
    z, z_post, u (equilibrium utility; RDP owners: up to A(x)), u_post, and
    price, the actual price of z borne at the margin, 1 + γ ρ_borne (used
    for equivalent variations); and rdp_topup, (solver cells,) the ex-post
    floor top-up per RDP household (rands a year; 0 without the floor).
    """
    p, sel = mk.p, mk.sel
    alpha, beta, q0 = p["alpha"], p["beta"], p["q0"]
    gamma, v_I = p["fraction_z_dwellings"], p["informal_structure_value"]
    # Actual and perceived damages borne, net of insurance: formal contents
    # by group, (groups, cells); other household series: insured groups
    uninsured = 1 - mk.coverage[:, None]
    s = mk.coverage[0]
    expected = mk.damages["expected"]
    actual = {n: (1 - s) * v[sel] for n, v in expected.items()}
    actual["contents_formal"] = uninsured * expected["contents_formal"][sel]
    perceived = dict(mk.dmg)
    perceived["contents_formal"] = uninsured * mk.dmg["contents_formal"]
    u_eq, hh = outputs["utility"], outputs["households"][:, :, sel]
    shape = hh.shape
    z, z_post, u, u_post, price = (np.zeros(shape) for _ in range(5))

    def borne(rho_c, D, zz):
        """Damages borne at consumption zz (contents rate rho_c, own
        structure damage D)."""
        return gamma * rho_c * zz + D

    def solve_z(S, rho_c, D):
        """z such that z + borne(z) = S."""
        return (S - D) / (1 + gamma * rho_c)

    # Sandbag levels for settlers
    level = np.nan_to_num(outputs["mask_self_protec"][sel]).astype(int)

    def informal(d, protec, sel_cells=True):
        out = {}
        for n in ("contents_informal", "structure_informal_settlements"):
            v = d[n].copy()
            for k, dk in protec.items():
                v = np.where(level == k, dk[n][sel] if sel_cells else dk[n],
                             v)
            out[n] = v
        return out

    act_IS = informal(actual, {k: {n: (1 - s) * v for n, v in dk.items()}
                               for k, dk in
                               mk.damages["expected_protec"].items()})
    per_IS = informal(perceived, mk.dmg_protec, sel_cells=False)

    Q = outputs["dwelling_size"][0][sel]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z_types = [
            (u_eq[:, None] / (mk.amenities * (Q - q0) ** beta)) ** (1 / alpha),
            mk._composite_good(u_eq, p["pocket_backyard"]),
            mk._composite_good(u_eq, p["pocket_informal"])]
        specs = [  # (perceived contents, actual contents, perceived D, actual D)
            (perceived["contents_formal"], actual["contents_formal"], 0, 0),
            (perceived["contents_backyard"], actual["contents_backyard"], 0, 0),
            (per_IS["contents_informal"], act_IS["contents_informal"],
             per_IS["structure_informal_settlements"] * v_I,
             act_IS["structure_informal_settlements"] * v_I)]
        for h in range(3):
            rp, ra, Dp, Da = specs[h]
            S = z_types[h] + borne(rp, Dp, z_types[h])
            z[h], z_post[h] = z_types[h], solve_z(S, ra, Da)
            u[h] = np.broadcast_to(u_eq[:, None], shape[1:])
            u_post[h] = u[h] * (z_post[h] / z[h]) ** alpha
            price[h] = 1 + gamma * ra

        # RDP owners (group 1): available budget A = ỹ_1 - ρ v_FS
        # + μ Y (R - (ρ + δ) v_I / q_I) for z and own damages, plus the
        # floor's equilibrium transfer T
        V, B, qI = p["subsidized_structure_value"], p["backyard_size"], \
            p["shack_size"]
        mu = outputs["housing_supply"][1][sel] / 1000000
        R = np.nan_to_num(outputs["rent"][1][sel])
        T = outputs["rdp_transfer"][0][sel]
        A = (mk.y[0] - p["depreciation_rate"] * V
             + mu * B * (R - (p["depreciation_rate"] + p["interest_rate"])
                         * v_I / qI))
        shacks = actual["structure_backyards"] * (mu * B / qI * v_I)
        rho_c = actual["contents_subsidized"]
        D = actual["structure_subsidized_1"] * V + shacks
        z_rdp = mk.rdp_consumption(mu, R, T)
        z_post_rdp = solve_z(A + T, rho_c, D)
        H = p["RDP_size"] + B - q0
        topup = np.zeros(len(T))
        if p["rdp_utility_floor"]:
            # Composite good at utility u_1, at most the one with all own
            # damages paid (z + shack damages = A)
            z_floor = (u_eq[0] / (mk.amenities * (H - mu * B) ** beta)) \
                ** (1 / alpha)
            target = np.minimum(z_floor, A - shacks)
            low = mk.rdp_cells & (z_post_rdp < target * (1 - 1e-10))
            z_post_rdp = np.where(low, target, z_post_rdp)
            topup = np.where(low, z_post_rdp + borne(rho_c, D, z_post_rdp)
                             - (A + T), 0)
        u_rdp = (np.where(z_rdp > 0, z_rdp, np.nan) ** alpha
                 * (H - mu * B) ** beta)
        z[3, 0], z_post[3, 0], u[3, 0] = z_rdp, z_post_rdp, u_rdp
        u_post[3, 0] = u_rdp * (z_post_rdp / z_rdp) ** alpha
        price[3, 0] = 1 + gamma * rho_c
    return {"households": hh, "z": z, "z_post": z_post, "u": u,
            "u_post": u_post, "price": price, "rdp_topup": topup}


def landlord_revenue(mk, outputs):
    """Revenues of absentee landlords per solver cell (rands a year), (2,
    cells):
    [0] formal land: zero-profit land rent from (4), Π = 0 gives
        δ P(x) = R s_FP - (ρ + ρ^struct_FP + δ) k - t per m2 of developed
        land, i.e. a R s_FP - t where (9) holds (not where the height limit
        caps supply), with perceived damages (developers' unanticipated
        losses are financed by their tax t, solver.solve);
    [1] informal settlements: rents R_IS q_I N^IS(x).
    Backyard rents go to RDP owners, and RDP land to nobody."""
    p, sel = mk.p, mk.sel
    supply = outputs["housing_supply"][0][sel]
    Q = outputs["dwelling_size"][0][sel]
    destroyed = np.where(Q > p["threshold"], mk.dmg["structure_formal_2"],
                         mk.dmg["structure_formal_1"])
    cost = p["interest_rate"] + p["depreciation_rate"] + destroyed
    land = mk.land[0] * 250000 * (supply > 0)        # m2
    with np.errstate(invalid="ignore"):
        formal = land * (outputs["rent"][0][sel] * supply / 1000000
                         - cost * outputs["capital_land"][0][sel]
                         - mk.developer_tax)
    informal = (np.nan_to_num(outputs["rent"][2][sel]) * p["shack_size"]
                * outputs["households"][2][:, sel].sum(0))
    return np.array([np.nan_to_num(formal), informal])
