# flood_model

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
conda run -n nedum-2025 python flood_model/run.py --cc 1 --co 0 --rm 1 --sandbag_course_cost 500
conda run -n nedum-2025 python flood_model/tools/check_legacy.py 1001   # verify vs legacy (AF CC RM SP)
```
From Python: `run.run_model(options, param)` returns
`(outputs, converged, param, damages)`.

Outputs go to `../../Output/flood_model/initial_state_<var>_<name>.npy` and
`mask_self_protec_<name>.npy` with `name = simul_AF?_CC?_CO?_RM?_SP?`:
same variables, shapes and conventions as the legacy `Output/flood_output`
files (without the unused `_PS/_SI/_T` name parts, plus `_CO` for coastal
floods; the `_RA0` suffix was dropped on 2026-10-01). `flood_type_<name>.npz`
holds the flood type masks:
- keys `expected[_protec]__<series>`: the dominant type of each cell;
- keys `return_periods[_protec]__<series>`: the type selected in each event (10 x cells);
- values index the `flood_types` key (fluvial, pluvial, coastal), and -1 means no flood damage.

## Layout

- `config.py`: paths (relative to this file), flood options (`OPTIONS`), parameters (`PARAM`) with their main.tex symbols, hard-coded 2011 totals, housing access by income group.
- `data.py`: reads raw inputs (caches them in `_cache/raw_inputs_v<N>.npz`, refreshed automatically when a source file is newer; bump `CACHE_VERSION` if `_read_raw` changes) and derives the solver inputs: land availability `coeff_land` (L_h(x)), target households, interest and agricultural rents, minimum lot size.
- `floods.py`: actual expected and state damages per cell (FATHOM fluvial undefended + pluvial, and DELTARES coastal aligned on FATHOM return periods; 10 return periods -> 11 flood states), and flood type masks.
- `solver.py`: `perception` (factor applied to actual damages), `Markets` (selected cells; bid rents, dwelling sizes, housing supply and households per housing type for given utilities) and `solve(markets)` (utility iteration, best iterate, export to full grid with RDP housing added).
- `run.py`: CLI (`build_parser`, also used by `flood_model_CRRA`), `run_model` and `save`.
- `main.py`: cell script for the Interactive Window (works without `__file__`, from `python/` or `flood_model/` as working directory).
- `tools/check_legacy.py`: runs the legacy code in-process (with the bug fixes below patched in memory; legacy files untouched) and this package in legacy mode, and compares all outputs.

## Model conventions

- 24,014 grid cells of 0.25 km²; the solver only works on cells with land share > 1% and positive net income (about 4,000 cells).
- Income groups 0-3 (poorest to richest; 1-4 in main.tex). Housing types: 0 formal private (FP), 1 informal backyards in RDP premises (IB), 2 informal settlements (IS), 3 formal subsidized (FS: RDP, exogenous, group 0 only). Backyards and settlements are open to groups 0-1 only.
- Rents are annual rands per m²; housing supply is m² per km² of *available* land; `coeff_land` is the available share of each cell.
- Damages: expected annual share of capital destroyed. Contents damage raises the price of the composite good to 1 + γ ρ^content (γ = `fraction_z_dwellings`); structure damage enters as a capital cost (formal: developers; backyards: RDP owner for its house plus shack cost; settlements: households).
- `floods.compute_damages` always returns actual damages; agents perceive k x actual damages (`solver.perception`): k = 0 under AF0, `risk_internaliz` under RM1, 1 otherwise. So under AF0 the returned damages and flood type masks are the actual ones (legacy and earlier versions returned zeros), while the equilibrium is unchanged.
- Flood types are combined per event: for each return period, the most damaging type applies (bath-tub logic), then state and expected damages are computed. The dominant type flag of a cell is the type with the highest expected damage on its own.
- Coastal floods (option CO, default on; absent from the legacy code): DELTARES MERITDEM maps, with each FATHOM return period T mapped to the closest smaller or equal DELTARES return period (0, 2, 5, 10, 25, 50, 100, 250). So T 5->5, 10->10, 20->10, ..., 1000->250, and the 0/2-yr maps are unused. Under CC, coastal uses the sea-level-rise maps (2050, RCP8.5) aligned on T / risk_increase, because FATHOM probabilities are scaled by risk_increase (T 5->2, 20->10, 200->100, ...): climate change is not counted twice. The pluvial stormwater correction does not apply to coastal floods; sandbag protection does.
- SP lets each group in settlements choose protection (flood depth <= 0.15 m no longer damages) if it raises its bid rent.
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
- Exact ties between top bidders: the cell goes to one group (legacy: to each tied group, double counting households). No effect in practice.

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

`tools/check_legacy.py` runs the legacy code with fixes 1-4 and
`config.TOTAL_FORMAL` patched in memory, against this package in legacy mode
(CO0, max of expected damages, last iterate). All 14 outputs are
bit-identical for AF CC RM SP = 1001, 1101, 0000 and 1011 (2026-10-01, after
the RA removal). The 2026-10-01 cleanup left all outputs bit-identical to
the previous version (4 RN configurations, incl. AF0 and CC1 RM1 SP1).

Effects on results (AF1 CC0 CO1 RM0 SP1, sandbag 250, unless stated):
- Fix 1 (pluvial), vs unpatched legacy: settlement households -1.2% and settlers' mean expected structural damage doubles (1.8 -> 3.6 per mille). With CC1 RM1: settlements -3.3%, damage 3.8 -> 10.4 per mille.
- Fix 2 (rm_backyard): +3% backyard households under RM.
- Fix 4 (backyard_loss): backyard households 77,717 -> 71,561 (-7.9%), settlements +7,018, poorest group's utility -0.3%.
- Per-return-period max: formal households' mean expected structural damage +1.5%, little else.
- Coastal: dominant type in about 350 cells, where about 5,000 households live. Formal structural damage per household +4%; settlements unaffected.

## Known issues

- **The solver often does not converge to `precision` (0.1%)**, e.g. for the flood_script settings, even with 2,000 iterations.
  - Cause: a few lumpy cells at a near-tie between groups 0 and 1 (e.g. informal cell 21397, about 3,400 households) flip between groups, leaving group 0-1 totals 0.1-0.8% off target. Utilities are stable to about 1e-5.
  - Current behaviour: `solve` returns the best iterate (lowest max abs error). Raising `max_iter` to 400 reaches precision in some configurations.
  - Proper fix: allocate near-tied cells fractionally, splitting a cell between groups whose bids are within a tolerance, or a logit allocation of cells across bidders with a large scale parameter. Either makes aggregate demand continuous in utilities.
- **Self-protection take-up among settlers is zero** at `sandbag_course_cost = 250`, with or without RM or CC (and under risk aversion, see `../flood_model_CRRA`). Protection is chosen only in cells without settlement land.
  - The benefit (expected damage of floods at or below 0.15 m, times structure and exposed contents values) is almost always below 250 rands a year.
- Coastal alignment on "smaller or equal" return periods is conservative. Frequent coastal flooding (the 0/2-yr DELTARES maps, already about 330 cells at 2 yr) only enters through the linear interpolation from zero damage at annual probability 1 to the 5-yr map.
- Protection is all-or-nothing at 0.15 m (no reduced damage above that depth).
- `mask_self_protec` flags the top bidder's choice even in cells without settlement land: weight by households when computing take-up.
- Formal bid rents use the unconstrained first-order condition even where the minimum lot size binds.
- **RDP owners with negative net income**: in heavily flooded RDP cells, ỹ_1 - (ρ + ρ^struct_FS) v_FS <= 0, so their composite good is negative whatever share μ they rent out, and main.tex (10) is not defined. `backyard_supply` applies it anyway (usually μ = 1 when R > Z). This happens in 27 cells (626 RDP households) at default settings, and in 41 cells (842 households) under CC, of which only 3-5 have backyard land, and no backyard households end up there in equilibrium (checked 2026-10-01).
- RDP housing supply in the outputs keeps the legacy formula, with an extra factor RDP_size / (RDP_size + backyard_size); it does not enter the equilibrium.

## Editing notes

- Keep everything vectorised over (groups x cells); no Python loops over cells.
- `../flood_model_CRRA` subclasses `solver.Markets` and overrides `formal_price` (set in `__init__`), `backyard_rent`, `informal_rent` and `backyard_supply`; it also uses `solver.perception`, `solver._clean`, `run.build_parser`, `run.save` and the `states`/`proba` entries of `floods.compute_damages`. Keep these interfaces, and run `../flood_model_CRRA/tools/check_degenerate.py` after changing them.
- Adding raw input files: list them in `data._SOURCES` (via `_FLOOD_FILES` for maps) and bump `CACHE_VERSION`.
- The order of floating-point operations mirrors the legacy code, so that `tools/check_legacy.py` gives exact equality. A reordering can flip near-tied cells (see above) and produce visible but meaningless diffs. Check with the tool after changes.
