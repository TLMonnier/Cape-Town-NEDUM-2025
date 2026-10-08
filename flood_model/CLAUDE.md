# flood_model
- **Welfare comparisons are noisy for groups 1-2**: because of the lumpy-cell problem below, group 1's scenario utility often lands on the same value (1,821.9) whatever the policy, with population errors of +-0.5%; utility changes and EVs below about 0.3% for these groups are within solver noise (checked 2026-10-01 with 1,000 iterations).

Simplified, vectorised rewrite of the NEDUM Cape Town static equilibrium used
by `../flood_script.py` (JUE revision): 2011 equilibrium with flood risks,
risk misperception and self-protection (sandbags), for risk-neutral agents.
Risk aversion lives in `../flood_model_CRRA`, which reuses this package.
It replaces `flood_script.py` + `inputs/`, `equilibrium/` (about 5,900 lines)
with about 700 lines, and runs in about 2 s once inputs are cached, against
about 1-2 min for the legacy code.

`main.tex` is the (legacy) working paper. Docstrings and comments refer to
its equations (1)-(11) and sections, and `config.py` maps code names to the
paper's symbols. Where they differ, the code is right and main.tex is
outdated (user, 2026-10-01): 3-year (not 4-year) interest rate average, max
damage (not max depth) across flood types per event, contents damages
varying across housing types (drainage), q_FS = RDP house + backyard.
These spots are marked "outdated" in the comments.

## Running

The `nedum-2025` env's `python.exe` does not start directly from Claude's
shell: use `conda run`.

Interactive use (replaces `flood_script.py`): open `main.py` in VS Code, edit
the options/parameters cell, and run the `# %%` cells in the Interactive
Window. The setup cell reloads the package modules, so code edits are picked
up without restarting the kernel. Use `tag` to keep parameter variants apart
in output names.

```
conda run -n nedum-2025 python flood_model/run.py                 # default options
conda run -n nedum-2025 python flood_model/run.py --cc 1 --co 0 --rm 1 --sp 3 --sandbag_course_cost 50
conda run -n nedum-2025 python flood_model/tools/check_legacy.py 1001   # verify vs legacy (AF CC RM SP)
```
From Python: `run.run_model(options, param)` returns
`(outputs, converged, param, damages)`.

Outputs go to `../../Output/flood_model/initial_state_<var>_<name>.npy` and
`mask_self_protec_<name>.npy` with `name = simul_AF?_CC?_CO?_RM?_SP?_PS?_SI?[_PP1]` (`_PP1` only under public protection):
same variables, shapes and conventions as the legacy `Output/flood_output`
files (without the unused `_PS/_SI/_T` name parts, plus `_CO` for coastal
floods; the `_RA0` suffix was dropped on 2026-10-01). `flood_type_<name>.npz`
holds the flood type masks:
- keys `expected[_protec<k>]__<series>`: the dominant type of each cell (with k sandbag levels);
- keys `return_periods[_protec<k>]__<series>`: the type selected in each event (10 x cells);
- values index the `flood_types` key (fluvial, pluvial, coastal), and -1 means no flood damage.

## Layout

- `config.py`: paths (relative to this file), flood options (`OPTIONS`), parameters (`PARAM`) with their main.tex symbols, hard-coded 2011 totals, housing access by income group.
- `data.py`: reads raw inputs (caches them in `_cache/raw_inputs_v<N>.npz`, refreshed automatically when a source file is newer; bump `CACHE_VERSION` if `_read_raw` changes) and derives the solver inputs: land availability `coeff_land` (L_h(x)), target households, interest and agricultural rents, minimum lot size.
- `floods.py`: actual expected and state damages per cell (FATHOM fluvial undefended + pluvial, and DELTARES coastal aligned on FATHOM return periods; 10 return periods -> 11 flood states), and flood type masks.
- `solver.py`: `perception` (factor applied to actual damages), `Markets` (selected cells; bid rents, dwelling sizes, housing supply and households per housing type for given utilities; RDP owners' floor transfer `rdp_transfer`) and `solve(markets)` (budget fixed point `_solve_budgets`, utility iteration, best iterate, export to full grid with RDP housing added, floor costs and landlords' revenues).
- `accounting.py`: actual damages per household (`household_damages`: contents from consumption z, structures from capital/shack/house values, attributed to occupants), insurance payouts, developers' unanticipated losses (`developer_losses`), ex-post consumption and utility with actual damages (`ex_post`, incl. the RDP floor top-up), and absentee landlords' revenues (`landlord_revenue`).
- `scenarios.py`: quick-look comparisons with the baseline (`run.baseline_options`: SP0 PS0 SI0 PP0): `damage_table` (by housing type and group, M rands and % of mean income ȳ_i, net of insurance), `welfare_table` (equilibrium and ex-post utilities, equivalent variation at baseline prices), `relocation_table` (movers = half the sum of |ΔN| over cells and housing types; household-weighted rents, supply, dwelling sizes).
- `run.py`: CLI (`build_parser`, also used by `flood_model_CRRA`), `run_model` (with `markets_class`), `insurance_terms` and `save`.
- `main.py`: cell script for the Interactive Window (works without `__file__`, from `python/` or `flood_model/` as working directory).
- `tools/check_legacy.py`: runs the legacy code in-process (with the bug fixes below patched in memory; legacy files untouched) and this package in legacy mode, and compares all outputs.

## Model conventions

- 24,014 grid cells of 0.25 km²; the solver only works on cells with land share > 1% and positive net income (about 4,000 cells).
- Income groups 0-3 (poorest to richest; 1-4 in main.tex). Housing types: 0 formal private (FP), 1 informal backyards in RDP premises (IB), 2 informal settlements (IS), 3 formal subsidized (FS: RDP, exogenous, group 0 only). Backyards and settlements are open to groups 0-1 only.
- Rents are annual rands per m²; housing supply is m² per km² of *available* land; `coeff_land` is the available share of each cell.
- Damages: expected annual share of capital destroyed. Contents damage raises the price of the composite good to 1 + γ ρ^content (γ = `fraction_z_dwellings`); structure damage enters as a capital cost (formal: developers; backyards: RDP owner for its house plus shack cost; settlements: households).
- `floods.compute_damages` always returns actual damages; agents perceive k x actual damages (`solver.perception`): k = 0 under AF0, `risk_internaliz` under RM1, 1 otherwise. So under AF0 the returned damages and flood type masks are the actual ones (legacy and earlier versions returned zeros). Under insurance (SI1), insured groups bear and perceive k (1 - s) x actual damages (`Markets.perceived_factor`, `Markets.factor`).
- Flood types are combined per event: for each return period, the most damaging type applies (bath-tub logic), then state and expected damages are computed. The dominant type flag of a cell is the type with the highest expected damage on its own.
- Coastal floods (option CO, default on; absent from the legacy code): DELTARES MERITDEM maps, with each FATHOM return period T mapped to the closest smaller or equal DELTARES return period (0, 2, 5, 10, 25, 50, 100, 250). So T 5->5, 10->10, 20->10, ..., 1000->250, and the 0/2-yr maps are unused. Under CC, coastal uses the sea-level-rise maps (2050, RCP8.5) aligned on T / risk_increase, because FATHOM probabilities are scaled by risk_increase (T 5->2, 20->10, 200->100, ...): climate change is not counted twice. The pluvial stormwater correction does not apply to coastal floods; sandbag protection does.
- SP (0-3, since 2026-10-01) is the number of sandbag levels settlers can build. Each group in each settlement cell chooses k = 0, ..., SP levels to maximise its bid rent (lowest k if tied). k levels raise the floor by k x `sandbag_height` (15 cm), so that the inside water depth is max(d - 0.15 k, 0) for structure and contents, and cost k x `sandbag_course_cost` a year. `damages["expected_protec"][k]` and `["states_protec"][k]` hold damages with k levels; `mask_self_protec` is the top bidder's k (NaN without SP). Before 2026-10-01, SP1 was a barrier: floods up to 15 cm did no damage, deeper ones full damage (legacy rule, still used by `tools/check_legacy.py`).
- PP (public_protec, 0/1, 2026-10-08, user decisions): free public protection (counterfactual, no cost or tax) of every dwelling, in every housing type and damage series (developers' structures too), against every flood type, up to H = `public_protection_height` (0.45 m): the water depth falls to max(d - H, 0), like a raised floor (`floods.compute_damages`; drainage and the per-event max over flood types apply as before). Settlers' sandbags add on top: k levels give max(d - H - 0.15 k, 0). Agents perceive the protected damages (k x actual as usual). `run.baseline_options` also sets PP0.
- PS (subsid_protec, 2026-10-01; requires SP >= 1): sandbags are free for settlers (they choose levels at zero cost), and the government pays C = c_SP Σ_x N^IS(x) k(x). C is financed by lump-sum taxes T_i = τ ȳ_i on `config.TAXED_GROUPS` (groups 3-4, formal only), with ȳ_i their exogenous mean income (`average_income`) and τ = C / Σ_i N_i ȳ_i (N_i = targets), so Σ_i N_i T_i = C. Taxes lower ỹ_i(x) everywhere (`Markets.set_tax`). `solver.solve` balances the budget by fixed point (T = 0, equilibrium, C, T(C), ...; `budget_tol` 1e-3, `max_iter_budget` 10) and adds outputs `tax` (4,) and `budget` [sandbags, insurance, revenue]. With SP3, c_SP 25: C = 3.2 M rands/year, T = 3 and 14 rands/year (0.0018% of mean income), balanced in 2 rounds.
- SI (subsid_insur, 0/1; redefined on 2026-10-08, user decision, replacing the former SI1-3): subsidised insurance of `config.INSURED_GROUPS` (groups 1-2) at no premium, financed with the same lump-sum taxes on groups 3-4 as PS (costs add up). Insured households are reimbursed a share s of the damages to their own assets: contents in every housing type, settlers' shacks, RDP houses and RDP owners' backyard shacks; formal developers' structures are not insured (so the developers' loss tax is unaffected). They only bear and perceive (1 - s) k x actual damages (moral hazard): in `Markets`, all series except `solver.FORMAL_SERIES` are scaled by 1 - s (backyards, settlements and RDP housing only host insured groups; `__init__` checks this), and `formal_price` is per group, 1 + γ (1 - s_i) ρ^content_FP (groups, cells). Settlers may still buy sandbags at full cost. s = 1 - D / D_0 (`run.insurance_terms`), the share of settlers' damages averted by sandbags in a reference run of the same scenario without insurance (SI0, PS0, same SP or `insurance_reference_levels` if SP0, same sandbag cost), with D at their chosen levels and D_0 without sandbags, same households and locations. s changes across scenarios (0.847 RM0, 0.789 RM1, at SP3 and 25 rands per level) and is returned in `outputs["insurance_share"]`. Payouts = s x own damages of groups 1-2 (`accounting.household_damages`). Outputs `tax`, `budget` [sandbags, insurance, tax revenue], `insurance_share`.
- Developers' loss tax (`developer_loss_tax`, 2026-10-01, user decision): under AF0 or RM1, developers build on perceived damages but bear actual ones; their unanticipated losses Σ (ρ_actual,borne - ρ_perceived) K are financed by a lump-sum tax t per m2 of developed land, anticipated ex ante: R_A(t) = (δ P_A + t)^a (ρ + δ)^(1-a) / (κ a^a (1-a)^(1-a)) moves the city edge (`Markets.set_developer_tax`). t is balanced with the public budget in `solver.solve` (t = 0.50 rands/m2/year, losses 234 M rands/year under RM1). This changes AF0 and RM1 equilibria vs earlier versions; the legacy check turns it off. Output `developer_tax` [t, losses, revenue].
- Ex-post utilities (user, 2026-10-01): with misperceived damages (k != 1), households keep their location, dwelling, rent and planned spending S = z + perceived borne damages, and z_post solves z + actual borne damages = S (owners bear their structures; FP households only contents). Group means of u_post (non-RDP) feed the welfare table; EV uses the ratio of ex-post group means (RDP owners: per-cell ratio).
- RDP owners' utility floor (`rdp_utility_floor`, default True, user decisions of 2026-10-08): RDP owners' utility z^α (q_FS - μY - q0)^(1-α) A(x) never falls below u_1 (group 1's equilibrium utility) because of floods. (a) In equilibrium, with perceived damages: the government pays a lump-sum transfer T(x), the smallest that lifts their optimised utility to u_1, and at most their own damages D = ρ^struct_FS v_FS + γ ρ^content_FS z (perceived, net of insurance; shacks excluded). Owners anticipate T: it enters (10) and their budget, but they still pay 1 + γ ρ^content at the margin (`Markets.rdp_transfer`, bisection on the few cells below the floor). (b) Ex post, with actual damages at fixed μ and spending, a top-up keeps u_post >= u_1, again at most up to their own damages (`accounting.ex_post`). Both are financed by absentee landlords, with no feedback on the equilibrium. Outputs: `rdp_transfer` (2, cells) per RDP household [equilibrium, ex post], `surplus_damages` (2,) totals, `landlord_revenue` (2, cells) [formal land rent R s - (ρ + ρ^struct + δ) k - t (= a R s - t where (9) holds), settlement rents R_IS q_I N]; backyard rents go to RDP owners. At default settings (RM1 SP3): 291 households in 2 cells get 0.63 M rands/year in equilibrium, plus 8.8 M ex post for 880 households in 6 cells; landlords' revenues are 37,020 M (formal 36,774 M, settlements 246 M), so the levy is 0.025%. RM0 SP0: 9.4 M for 880 households in equilibrium, nothing ex post; CC1 RM1: 1.3 M + 12.5 M (0.037% levy). Not implemented in flood_model_CRRA (False there); False in the legacy check.
- Hard-wired (legacy flood_script values): urban edge, no new RDP, informal land constrained, no amenity upgrading, poor subsidies, eviction or incremental housing, undefended fluvial maps, MERITDEM coastal maps, pluvial correction on, housing supply adjusts.

## Differences from the legacy code

Bug fixes (legacy behaviour corrected):

1. **pluvial**: `inputs.data.compute_fraction_capital_destroyed` edited the pluvial maps in place when applying the stormwater correction. The first call (formal) zeroed 5/10/20-yr pluvial risk for *every* later housing type, including informal settlements, which should get no correction.
2. **rm_backyard** (RM only): the backyard shack structure cost used actual, not perceived, damages.
3. **backyard_cap**: `import_land_use` computed `coeff_land_backyard = fmin(urban, area_backyard)` but never used it; the cap now applies (3 cells).
4. **backyard_loss**: with the shack cost Z in backyard supply, the closed-form first-order condition (main.tex (10)) has no feasible solution when the rent R <= Z (renting out loses money, so the optimal share is 0), and the legacy formula then rented out the whole backyard. Supply is now 0 there. With the flood_script settings, this affected 97 cells and 7,478 backyard households (about 10%).

Model changes (not bug fixes):
- Coastal floods (CO).
- Flood types combined per return period (legacy: max of expected damages across types).
- The solver returns the best iterate (legacy: last).
- `TOTAL_FORMAL = 810818 - TOTAL_RDP` (half the SAL sum; user change of 2026-09-30, legacy 821,028).
- Exact ties between top bidders: the cell goes to one group, the poorest (legacy: to each tied group, double counting households). No effect in practice: no exact tie between positive bids at the equilibrium (checked 2026-10-08, RM0 SP0 and RM1 SP3), while about 77 formal cells have bids within 0.1%. The user chose not to change the rule.
- Sandbag levels with raised floors (SP 0-3), subsidised sandbags (PS), insurance (SI), developers' loss tax and ex-post utilities (2026-10-01), and the RDP owners' utility floor with landlords' revenues (2026-10-08), see Model conventions.

Kept on purpose: `compute_equilibrium` meant to damp utility steps by the
current error but read a not-yet-computed (zero) total, giving a constant
damping of about 1/1.5. The intended damping ("conv" in
`tools/check_legacy.py`) converged in 3/24 test scenarios vs. 12/24 for the
constant one, so `solver.solve` writes the constant damping explicitly.

Also removed: the NameError when `agents_anticipate_floods = 0`, the in-place
slicing of damage tables in `compute_equilibrium`, the global `np.seterr`,
the reliance on an indirect `numpy.matlib` import, unused Excel reads, the
risk aversion option (moved to `../flood_model_CRRA` on 2026-10-01) and the
`patience` early stop (unused, it missed later better iterates).

## Verification

`tools/check_legacy.py` (run only on request) runs the legacy code with
fixes 1-4 and `config.TOTAL_FORMAL` patched in memory, against this package
in legacy mode (CO0, max of expected damages, last iterate, one barrier
sandbag level, no developers' loss tax; SP 0 or 1 only). All 14 outputs
were bit-identical for AF CC RM SP = 1001, 1101, 0000 and 1011 after the RA
removal, and 1001 after the sandbag levels change (2026-10-01). Not rerun
after the developers' loss tax.

The 2026-10-01 cleanup left all outputs bit-identical to the previous
version (4 RN configurations, incl. AF0 and CC1 RM1 SP1). After the later
changes of 2026-10-01, outputs without self-protection, insurance or
misperception (SP0 PS0 SI0, AF1 RM0) are still bit-identical; AF0 and RM1
equilibria change through the developers' loss tax.

Insurance redefinition and PP option (2026-10-08): SI0 PP0 outputs (RM1 SP3, RM0 SP0, floor
off) are bit-identical to the version before the RDP floor.
`check_degenerate.py` OK; flood_model_CRRA runs SI1 (s = 0.8465 at RM0).

RDP utility floor (2026-10-08): with `rdp_utility_floor` False, all
previous outputs are bit-identical (RM1 SP3, RM0 SP0, former SI2), plus the new
`rdp_transfer` (zeros), `surplus_damages` and `landlord_revenue`. With the
floor on (RM0 SP0, RM1 SP3, former SI2, AF0, CC1 RM1), no RDP owner is below u_1 in
equilibrium or ex post, and z > 0 everywhere. Transfers never reach the
damage bound, and formal land rents equal a R s - t to 1e-15. The floor
barely moves the equilibrium (utilities within 0.004%, 0-5 movers): every
floor cell has no backyard bidder (R = 0, μ = 0), so the transfer is pure
income. Runtime goes up by about 0.7 s per solve (bisection). `check_degenerate.py` is OK. The legacy check was not rerun.

Effects on results (AF1 CC0 CO1 RM0 SP1, sandbag 250, unless stated; before the 2026-10-01 changes):
- Fix 1 (pluvial), vs unpatched legacy: settlement households -1.2% and settlers' mean expected structural damage doubles (1.8 -> 3.6 per mille). With CC1 RM1: settlements -3.3%, damage 3.8 -> 10.4 per mille.
- Fix 2 (rm_backyard): +3% backyard households under RM.
- Fix 4 (backyard_loss): backyard households 77,717 -> 71,561 (-7.9%), settlements +7,018, poorest group's utility -0.3%.
- Per-return-period max: formal households' mean expected structural damage +1.5%, little else.
- Coastal: dominant type in about 350 cells, where about 5,000 households live. Formal structural damage per household +4%; settlements unaffected.

Policy scenarios (2026-10-01, SP3, sandbag 25 per level, RM0, 200 iterations, vs SP0 PS0 SI0 baseline):
- PS1: settlers' damages -85%, all exposed settlers protect; budget 3.2 M rands/year.
- SI1 (2026-10-08 definition, s = 0.847, floor on): payouts 54.6 M rands/year (48.2 M to group 1, 6.5 M to group 2), taxes 51 and 237 rands/year on groups 3-4 (0.030% of mean income). Moral hazard: exposed settlers protecting 22.1% -> 11.4% vs the uninsured SP3 run, settlers' damages 0.86 -> 2.50 M; group 2's physical damages +54% vs baseline (formal contents, location). EV +102 / +84 / -43 / -195 rands/year for groups 1-4 (+0.52%, +0.15%, -0.025%, -0.025% of mean income), total +10.9 M. No RDP owner needs the utility floor.
- SI1 under RM1 (s = 0.789): payouts 53.6 M, taxes 50 and 233; protecting settlers 16.1% -> 2.5%; EV +93 / +30 / -42 / -187, total -0.6 M. The developers' budget missed `budget_tol` (gap 1.4e-3 after 10 rounds), so `converged` is False.
- The former SI1 (coverage of all damages incl. developers, s from PS1 vs baseline), SI2 (cap) and SI3 (full coverage) were removed on 2026-10-08.
- PP1 (2026-10-08, floor on, vs SP0 PS0 SI0 PP0): RM0 SP0: total damages 436 -> 127 M rands/year (-71%; formal -70%, settlements -85%, RDP -74%); EV +100 / +94 / +302 / +1,293 rands/year for groups 1-4 (+0.51%, +0.16%, +0.18%, +0.16% of mean income), total +362 M; 1.5% of households move (4.0% of settlers); settlement rents -3%. With SP3, exposed settlers fall from 90,540 to 14,238 and only 3.8% of them still protect; results barely change (total EV +362 M). RM1 SP0: damages 563 -> 182 M (-68%), total EV +256 M (ex post). Group 1's utility lands on 1,821.9 again (lumpy cells, see Known issues).

## Known issues

- **Welfare comparisons are noisy for groups 1-2**: because of the lumpy-cell problem below, group 1's scenario utility often lands on the same value (1,821.9) whatever the policy, with population errors of +-0.5%; utility changes and EVs below about 0.3% for these groups are within solver noise (checked 2026-10-01 with 1,000 iterations).
- **The solver often does not converge to `precision` (0.1%)**, e.g. for the flood_script settings, even with 2,000 iterations.
  - Cause: a few lumpy cells at a near-tie between groups 0 and 1 (e.g. informal cell 21397, about 3,400 households) flip between groups, leaving group 0-1 totals 0.1-0.8% off target. Utilities are stable to about 1e-5.
  - Current behaviour: `solve` returns the best iterate (lowest max abs error). Raising `max_iter` to 400 reaches precision in some configurations.
  - Proper fix: allocate near-tied cells fractionally, splitting a cell between groups whose bids are within a tolerance, or a logit allocation of cells across bidders with a large scale parameter. Either makes aggregate demand continuous in utilities.
- **Self-protection take-up among settlers was zero** at `sandbag_course_cost = 250` with the former one-level barrier rule, with or without RM or CC (and under risk aversion, see `../flood_model_CRRA`): the benefit (expected damage of floods at or below 0.15 m, times structure and exposed contents values) was almost always below 250 rands a year. Not rechecked with raised-floor levels. With SP3 and 25 rands per level (2026-10-01, 200 iterations): 22% of exposed settlers protect (8.3% one level, 13.2% two, 0.6% three).
- At zero cost, settlers in cells where sandbags change no damage stay at k = 0 (ties go to the lowest level).
- Coastal alignment on "smaller or equal" return periods is conservative. Frequent coastal flooding (the 0/2-yr DELTARES maps, already about 330 cells at 2 yr) only enters through the linear interpolation from zero damage at annual probability 1 to the 5-yr map.
- The raised floor protects structures as much as contents (same depth reduction for both damage functions).
- `mask_self_protec` flags the top bidder's choice even in cells without settlement land: weight by households when computing take-up.
- Formal bid rents use the unconstrained first-order condition even where the minimum lot size binds.
- RDP housing supply in the outputs keeps the legacy formula, with an extra factor RDP_size / (RDP_size + backyard_size); it does not enter the equilibrium.
- **RDP owners with negative net income** (solved by the utility floor since 2026-10-08; remains with `rdp_utility_floor` False, as in the legacy check and flood_model_CRRA): in heavily flooded RDP cells, ỹ_1 - (ρ + ρ^struct_FS) v_FS <= 0, so their composite good is negative whatever share μ they rent out, and main.tex (10) is not defined. `backyard_supply` applies it anyway (usually μ = 1 when R > Z). This happens in 27 cells (626 RDP households) at default settings, and in 41 cells (842 households) under CC, of which only 3-5 have backyard land, and no backyard households end up there in equilibrium (checked 2026-10-01). With the floor, the transfer makes z > 0 in every cell with RDP households. Cells without RDP households still get (10) with negative income, which has no effect.
- The RDP floor compares owners with u_1 while their backyard share is chosen against perceived damages only. Under misperception, the ex-post top-up is not anticipated (owners do not change μ for it).

## Editing notes

- Keep everything vectorised over (groups x cells); no Python loops over cells.
- `../flood_model_CRRA` subclasses `solver.Markets` and overrides `formal_price` (set in `__init__`), `backyard_rent`, `informal_rent` and `backyard_supply(R, transfer=0)` (it ignores the transfer and refuses `rdp_utility_floor`); it also uses `solver.perception`, `solver._clean`, `Markets.sandbag_price`, `run.run_model`, `run.build_parser`, `run.save` and the `states`/`proba` entries of `floods.compute_damages`. Keep these interfaces, and run `../flood_model_CRRA/tools/check_degenerate.py` after changing them.
- Adding raw input files: list them in `data._SOURCES` (via `_FLOOD_FILES` for maps) and bump `CACHE_VERSION`.
- The order of floating-point operations mirrors the legacy code, so that `tools/check_legacy.py` gives exact equality. A reordering can flip near-tied cells (see above) and produce visible but meaningless diffs.
