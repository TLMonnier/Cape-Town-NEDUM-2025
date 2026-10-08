# flood_model_CRRA

Risk-averse version of `../flood_model`: households maximise CRRA expected
utility over the 11 flood states instead of valuing damages at their
expected value. Split from `flood_model` on 2026-10-01 (previously its RA
option).

It is an extension, not a copy: inputs (`data`), damages (`floods`),
parameters (`config`), the utility iteration (`solver.solve`) and the
risk-neutral parts of the markets come from `../flood_model`, which
`crra.py` puts on `sys.path`. Changes there (bug fixes, new inputs) apply
here too. See `../flood_model/CLAUDE.md` for the shared model, options and
outputs, and `crra.py`'s docstring for the CRRA model.

## Running

```
conda run -n nedum-2025 python flood_model_CRRA/run_crra.py               # default options
conda run -n nedum-2025 python flood_model_CRRA/run_crra.py --sp 1 --sandbag_course_cost 25 --CRRA 2
conda run -n nedum-2025 python flood_model_CRRA/tools/check_degenerate.py   # RA = RN without risk
```
Interactive use: `main.py` (`# %%` cells; from `python/` or
`flood_model_CRRA/`). From Python: `run_crra.run_model(options, param)`
returns `(outputs, converged, param, damages)`, as `flood_model`.

Outputs: same files and names as `flood_model`
(`simul_AF?_CC?_CO?_RM?_SP?_PS?` + tag), in `../../Output/flood_model_CRRA/`.
The names do not encode the CRRA coefficient: use `tag` (e.g. `_CRRA2`).
Runs take about 40-80 s (vs 2 s for `flood_model`).

## Layout

- `crra.py`: `PARAM` (CRRA coefficient θ, default 0.2772), `OUTPUT`, CRRA utility helpers, and `CRRAMarkets(solver.Markets)`, which overrides `formal_price`, `backyard_rent`, `informal_rent` and `backyard_supply`.
- `run_crra.py`: `run_model` (calls `run.run_model` with `CRRAMarkets`, so insurance reference runs are risk averse too) and CLI (flood_model's parser plus `--CRRA`).
- Insurance (SI1, redefined on 2026-10-08, see `../flood_model/CLAUDE.md`): insured groups' state damages are scaled by 1 - s (`Markets.perceived_factor`; the insurance pays a share s in every state), and the FP certainty-equivalent price is computed per group, c*_i = 1 / CE[1 / (1 + γ (1 - s_i) D̄_s)]. Damage accounting and ex-post utilities use certainty-equivalent consumption. SI1 at RM0: s = 0.8465 as in flood_model (2026-10-08).
- RDP owners' utility floor (flood_model, 2026-10-08): not implemented under risk aversion. `crra.PARAM` sets `rdp_utility_floor` False, `CRRAMarkets` raises NotImplementedError if it is True, and `backyard_supply(R, transfer=0)` ignores the transfer. `solver.solve` still returns `landlord_revenue`, `surplus_damages` (zeros) and `rdp_transfer` (zeros). `check_degenerate.py` now builds its parameters from `config.PARAM` and `crra.PARAM` (OK on 2026-10-08, config 10103).
- `main.py`: cell script for the Interactive Window. Its setup cell reloads the flood_model modules from `../flood_model` and `crra`, `run_crra` from here (in that order).
- `tools/check_degenerate.py`: with every state damage set to the expected damage, compares `CRRAMarkets` to `solver.Markets` at fixed utilities and in equilibrium.

## Model

- Households maximise E[V(U_s)] with V(U) = U^(1-θ)/(1-θ) and U_s = A z_s^α (q - q0)^β over the 11 flood states. Utility levels are certainty equivalents, and equilibrium rents are the risk-averse bids.
  - Formal housing and backyard tenants have closed forms: 1 + γ E[D] becomes (E[(1 + γ D_s)^-a])^(-1/a), with a = α (1 - θ).
  - Settlements, where structure damage is additive, solve CE[z_s(R)] = z by safeguarded Newton (`_ce_bids`), which also drives the protection choice.
  - RDP owners' backyard share maximises their expected utility: safeguarded Newton when consumption is positive in all states, grid plus golden-section search otherwise (odd extension of utility for negative consumption). Their contents damages are ignored, as in flood_model.
  - Formal developers stay risk neutral. θ = 1 is not handled.
- The legacy RA (flood_script) doubled state damages, ignored CC in state probabilities, used CRRA only for the protection choice and kept risk-neutral rents: this formulation has no legacy equivalent.

## Verification

- 2026-10-01 split: outputs are bit-identical to the previous `flood_model` RA1 runs (SP1 sandbag 25; CC1 RM1 SP1 sandbag 100).
- `tools/check_degenerate.py` (2026-10-01, configs 10101 with CRRA 0.2772 and 11011 with CRRA 2): at fixed utilities, all markets agree to 1e-15, except backyard supply in cells where RDP owners' net income is negative (see Known issues). The equilibria differ by up to 1e-5 (utilities) because the solver paths split when they go through such a cell (cell 2001 at iteration 27 for 11011), and the best-iterate rule then picks another near-tied iterate. With `return_best = False`, final utilities differ by 5e-9.

## Known issues

- **RDP owners with negative net income** (see `../flood_model/CLAUDE.md`): their composite good is negative for any backyard share. With CRRA >= 1, V = -inf for every share and the search returns 0; with CRRA < 1, the odd extension can return 1 even when renting out loses money (R < Z), because shrinking own housing makes the negative utility less negative. flood_model applies (10) instead. These cells currently have no backyard households in equilibrium.
- Odd extension in general: utility for negative consumption has no economic meaning; it only keeps the objective defined.

- Sandbag levels (2026-10-01, SP 0-3 with raised floors, see `../flood_model/CLAUDE.md`): `check_degenerate.py 10103` passes at the market level, including the chosen levels.

Effects on results (CRRA 0.2772 vs RN, AF1 CC0 CO1 RM0 SP1, sandbag 250, unless stated; computed before the split, with the former one-level barrier rule):
- Allocation: settlements -3.4%, backyards +3.4%, mean settlement rent +2.5%, utilities within 0.1%.
- Protection take-up is unchanged at every positive sandbag cost: 0 at 250, 2,467 settler households at 100, 3,498 at 50, 7,595 at 25 (same with CC1).
- At zero cost, take-up is lower (79,453 vs 84,351), because there are fewer settlers.
- Stronger risk aversion barely helps. With CRRA 2, take-up is unchanged (2,467 at a cost of 100, 7,595 at 25). With CRRA 5, it is unchanged at 100 and only rises to 8,581 (6.3% of settlers) at 25.
- Why: sandbags only stop shallow, frequent, low-loss floods, whereas risk aversion mostly penalises rare deep floods.

## Editing notes

- Per iteration, the slow parts are the settlement bids (Newton on CE, twice under SP) and the backyard share search for cells with possibly negative consumption.
- Keep the override points in sync with `flood_model/solver.py` (see its CLAUDE.md editing notes), and rerun `tools/check_degenerate.py` after changes on either side.
