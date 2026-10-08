"""Quick-look comparisons of a scenario with its baseline (same options
without self-protection or insurance, run.baseline_options): flood damages,
relocation, and housing markets."""
import numpy as np

import accounting

HOUSING = ("formal", "backyard", "informal", "RDP")


def damage_table(acc_base, acc_scen, average_income):
    """Print actual expected annual damages (accounting.household_damages)
    per housing type and per income group, in the baseline and in the
    scenario:
    - in absolute terms: millions of rands a year, and % change;
    - in relative income terms: Σ D / Σ N ȳ_i, damages as a % of the
      households' (exogenous) mean income, and change in percentage points.
    Under insurance, the scenario's damages net of payouts follow.
    """
    insured = acc_scen["reimbursed"].any()

    def totals(acc, axes, net=False):
        damage = acc["contents"] + acc["structure"]
        if net:
            damage = damage - acc["reimbursed"]
        income = acc["households"] * average_income[None, :, None]
        D, Y = damage.sum(axes), income.sum(axes)
        return np.append(D, D.sum()), np.append(Y, Y.sum())

    header = (f"{'':>12s} {'base':>8s} {'scen':>8s} {'change':>8s} |"
              f" {'base':>7s} {'scen':>7s} {'change':>8s}")
    if insured:
        header += f" | {'net':>8s} {'net':>7s}"
    for title, axes, labels in [
            ("housing type", (1, 2), HOUSING + ("total",)),
            ("income group", (0, 2),
             tuple(f"group {i + 1}" for i in range(4)) + ("total",))]:
        D_b, Y_b = totals(acc_base, axes)
        D_s, Y_s = totals(acc_scen, axes)
        D_n, _ = totals(acc_scen, axes, net=True)
        print(f"Flood damages by {title}: M rands/year | % of mean income"
              + (" | net of insurance (M, %)" if insured else ""))
        print(header)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, label in enumerate(labels):
                line = (f"{label:>12s} {D_b[k] / 1e6:8.2f} {D_s[k] / 1e6:8.2f}"
                        f" {100 * (D_s[k] / D_b[k] - 1):+7.1f}% |"
                        f" {100 * D_b[k] / Y_b[k]:6.3f}% "
                        f"{100 * D_s[k] / Y_s[k]:6.3f}% "
                        f"{100 * (D_s[k] / Y_s[k] - D_b[k] / Y_b[k]):+6.3f}pp")
                if insured:
                    line += (f" | {D_n[k] / 1e6:8.2f} "
                             f"{100 * D_n[k] / Y_s[k]:6.3f}%")
                print(line)


def welfare_table(mk_base, out_base, mk_scen, out_scen):
    """Print utility levels and equivalent variations (EV) per income group.

    Utilities: equilibrium levels u_i (non-RDP households), and ex-post
    household-weighted means with actual instead of perceived damages
    (accounting.ex_post; identical when damages are perceived correctly,
    lower under AF0 or RM1).

    EV is the change in annual income that would give each baseline
    household the scenario's ex-post utility, holding baseline rents,
    prices and locations fixed (Hicksian, at baseline prices), with
    r_i = ū_i,scen / ū_i,base the ratio of ex-post group means:
    - FP: indirect utility is A α^α (1-α)^(1-α) (ỹ - R q0) / (c^α R^(1-α)),
      linear in the supernumerary income ỹ - R q0: EV = (ỹ - R q0)(r_i - 1);
    - IB, IS (fixed size q_I): u is proportional to z^α, so z_post must rise
      to z_post r_i^(1/α), at price c = 1 + γ ρ_borne:
      EV = c z_post (r_i^(1/α) - 1);
    - RDP owners (utility not equalised across cells): as IB, IS with
      their own ex-post utility ratio in each cell, at their baseline
      backyard share.
    Means are per baseline household, in rands a year and as a % of the
    group's mean income ȳ_i. Taxes (PS, SI) are included through utility.
    """
    p, sel = mk_base.p, mk_base.sel
    alpha = p["alpha"]
    eb, es = accounting.ex_post(mk_base, out_base), \
        accounting.ex_post(mk_scen, out_scen)

    def group_mean(e, key):          # non-RDP households
        w = e["households"][:3]
        return (w * e[key][:3]).sum((0, 2)) / w.sum((0, 2))

    u_b, u_s = out_base["utility"], out_scen["utility"]
    post_b, post_s = group_mean(eb, "u_post"), group_mean(es, "u_post")
    ratio = (post_s / post_b)[:, None]
    hh = eb["households"]
    ev = np.zeros_like(hh)
    R = np.nan_to_num(out_base["rent"][0][sel])
    ev[0] = (mk_base.y - R * p["q0"]) * (ratio - 1)
    for h in (1, 2):
        ev[h] = eb["price"][h] * eb["z_post"][h] * (ratio ** (1 / alpha) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rdp_ratio = es["u_post"][3, 0] / eb["u_post"][3, 0]
        ev[3, 0] = np.nan_to_num(eb["price"][3, 0] * eb["z_post"][3, 0]
                                 * (rdp_ratio ** (1 / alpha) - 1))
    total = (hh * ev).sum((0, 2))
    n = hh.sum((0, 2))
    y_bar = mk_base.inputs["average_income"]
    print("Utility (equilibrium; ex post with actual damages) and "
          "equivalent variation (EV, at baseline prices):")
    for i in range(4):
        print(f"  group {i + 1}: utility {u_b[i]:,.1f} -> {u_s[i]:,.1f} "
              f"({100 * (u_s[i] / u_b[i] - 1):+.3f}%); ex post "
              f"{post_b[i]:,.1f} -> {post_s[i]:,.1f} "
              f"({100 * (post_s[i] / post_b[i] - 1):+.3f}%); mean EV "
              f"{total[i] / n[i]:+,.1f} rands/year "
              f"({100 * total[i] / n[i] / y_bar[i]:+.3f}% of mean income)")
    w = np.where(np.isfinite(rdp_ratio), hh[3, 0], 0)
    print(f"  of which RDP owners: mean EV "
          f"{(hh[3, 0] * ev[3, 0]).sum() / hh[3, 0].sum():+,.1f} rands/year,"
          f" mean ex-post utility change "
          f"{100 * (np.nansum(w * rdp_ratio) / w.sum() - 1):+.3f}%")
    print(f"  total EV: {total.sum() / 1e6:+,.2f} M rands/year")


def relocation_table(out_base, out_scen):
    """Print the share of households that changed location, and average
    changes in rents, housing supply and housing consumption.

    Movers: half the sum over cells (and housing types) of |N_scen - N_base|
    per income group (per housing type: over cells and groups), divided by
    baseline households. As households of a group are interchangeable, this
    is the smallest number of moves consistent with both distributions.

    Markets, per endogenous housing type: household-weighted averages in each
    run of the rent (rands/m2/year), housing supply (m2 per km2 of
    available land) and dwelling size (m2), and their % change.
    """
    hb, hs = out_base["households"], out_scen["households"]
    moved = 0.5 * np.abs(hs - hb)
    print("Households that changed location (cell or housing type):")
    for i in range(4):
        print(f"  group {i + 1}: {100 * moved[:, i].sum() / hb[:, i].sum():5.2f}%"
              f" ({moved[:, i].sum():,.0f} households)")
    print(f"  total:   {100 * moved.sum() / hb.sum():5.2f}%"
          f" ({moved.sum():,.0f} households)")
    for h, label in enumerate(HOUSING):
        print(f"  in {label} housing: "
              f"{100 * moved[h].sum() / max(hb[h].sum(), 1):5.2f}%")

    print("Household-weighted averages (baseline -> scenario):")
    for h, label in enumerate(HOUSING[:3]):
        parts = []
        for key, name in [("rent", "rent"), ("housing_supply", "supply"),
                          ("dwelling_size", "size")]:
            means = [np.nansum(out[key][h] * out["households"][h].sum(0))
                     / out["households"][h].sum()
                     for out in (out_base, out_scen)]
            parts.append(f"{name} {means[0]:,.1f} -> {means[1]:,.1f} "
                         f"({100 * (means[1] / means[0] - 1):+.2f}%)")
        print(f"  {label:>8s}: " + "; ".join(parts))
