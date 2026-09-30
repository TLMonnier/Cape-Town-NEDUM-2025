# flood_model

Simplified, vectorised rewrite of the NEDUM Cape Town static equilibrium used
by `../flood_script.py` (JUE revision): 2011 equilibrium with flood risks,
risk misperception, self-protection (sandbags) and risk aversion.
It replaces `flood_script.py` + `inputs/`, `equilibrium/` (about 5,900 lines)
with about 900 lines, and runs in about 2 s once inputs are cached (about
1 min under risk aversion), against about 1-2 min for the legacy code.

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
conda run -n nedum-2025 python flood_model/run.py --cc 1 --co 0 --rm 1 --ra 1 --sandbag_course_cost 500
conda run -n nedum-2025 python flood_model/tools/check_legacy.py 10010   # verify vs legacy (RA0 only)
```
From Python: `run.run_model(options, param)` returns
`(outputs, converged, param, damages)`.

Outputs go to `../../Output/flood_model/initial_state_<var>_<name>.npy` and
`mask_self_protec_<name>.npy` with `name = simul_AF?_CC?_CO?_RM?_SP?_RA?`:
same variables, shapes and conventions as the legacy `Output/flood_output`
files (without the unused `_PS/_SI/_T` name parts, plus `_CO` for coastal
floods). `flood_type_<name>.npz` holds the flood type masks:
- keys `expected[_protec]__<series>`: the dominant type of each cell;
- keys `return_periods[_protec]__<series>`: the type selected in each event (10 x cells);
- values index the `flood_types` key (fluvial, pluvial, coastal), and -1 means no flood damage.

## Layout

- `config.py`: paths (relative to this file), flood options (`OPTIONS`), parameters (`PARAM`), hard-coded 2011 totals, housing access by income group.
- `data.py`: reads raw inputs (caches them in `_cache/raw_inputs_v<N>.npz`, refreshed automatically when a source file is newer; bump `CACHE_VERSION` if `_read_raw` changes) and derives the solver inputs: land availability `coeff_land`, target households, interest and agricultural rents, minimum lot size.
- `floods.py`: expected and state flood damages per cell (FATHOM fluvial undefended + pluvial, and DELTARES coastal aligned on FATHOM return periods; 10 return periods -> 11 flood states), and flood type masks.
- `solver.py`: `Markets` (bid rents, dwelling sizes, housing supply per housing type for given utilities, RN or CRRA) and `solve` (utility iteration, best iterate, export to full grid with RDP housing added).
- `run.py`: CLI, `run_model` and `save`.
- `main.py`: cell script for the Interactive Window (works without `__file__`, from `python/` or `flood_model/` as working directory).
- `tools/check_legacy.py`: runs the legacy code in-process (with the bug fixes below patched in memory; legacy files untouched) and this package in legacy mode, and compares all outputs.

## Model conventions

- 24,014 grid cells of 0.25 km²; the solver only works on cells with land share > 1% and positive net income (about 4,000 cells).
- Income groups 0-3 (poorest to richest). Housing types: 0 formal private, 1 informal backyards (in RDP premises), 2 informal settlements, 3 formal subsidized (RDP, exogenous, group 0 only). Backyards and settlements are open to groups 0-1 only.
- Rents are annual rands per m²; housing supply is m² per km² of *available* land; `coeff_land` is the available share of each cell.
- Damages: `fraction_capital_destroyed`, i.e. expected annual share destroyed. Contents damage scales the composite good (`fraction_z_dwellings`), structure damage enters as a capital cost (formal: developers; backyards: RDP owner for its house plus shack cost; settlements: households).
- Flood types are combined per event: for each return period, the most damaging type applies (bath-tub logic), then state and expected damages are computed. The dominant type flag of a cell is the type with the highest expected damage on its own.
- Coastal floods (option CO, default on; absent from the legacy code): DELTARES MERITDEM maps, with each FATHOM return period T mapped to the closest smaller or equal DELTARES return period (0, 2, 5, 10, 25, 50, 100, 250). So T 5->5, 10->10, 20->10, ..., 1000->250, and the 0/2-yr maps are unused. Under CC, coastal uses the sea-level-rise maps (2050, RCP8.5) aligned on T / risk_increase, because FATHOM probabilities are scaled by risk_increase (T 5->2, 20->10, 200->100, ...): climate change is not counted twice. The pluvial stormwater correction does not apply to coastal floods; sandbag protection does.
- RM multiplies all perceived damages by `risk_internaliz`. SP lets each group in settlements choose protection (flood depth <= 0.15 m no longer damages) if it raises its bid rent.
- RA: households maximise E[U^(1-CRRA)] with U = A z^alpha (q - q0)^beta over the 11 flood states. Utility levels are certainty equivalents, and equilibrium rents are the risk-averse bids.
  - Formal housing and backyard tenants have closed forms: 1 + f E[D] becomes (E[(1 + f D_i)^-a])^(-1/a), with a = alpha (1 - CRRA).
  - Settlements, where structure damage is additive, use the `crra_bids` bisection, which also drives the protection choice.
  - RDP owners' backyard supply is a bisection on their first-order condition.
  - Formal developers stay risk neutral. CRRA = 1 is not handled.
  - With degenerate states (each state = expected damage), RA reproduces RN to 1e-13 (checked).
- Hard-wired (legacy flood_script values): urban edge, no new RDP, informal land constrained, no amenity upgrading, poor subsidies, eviction or incremental housing, undefended fluvial maps, MERITDEM coastal maps, pluvial correction on, housing supply adjusts.

## Differences from the legacy code

Bug fixes (legacy behaviour corrected):

1. **pluvial**: `inputs.data.compute_fraction_capital_destroyed` edited the pluvial maps in place when applying the stormwater correction. The first call (formal) zeroed 5/10/20-yr pluvial risk for *every* later housing type, including informal settlements, which should get no correction.
2. **rm_backyard** (RM only): the backyard shack structure cost used actual, not perceived, damages.
3. **backyard_cap**: `import_land_use` computed `coeff_land_backyard = fmin(urban, area_backyard)` but never used it; the cap now applies (3 cells).
4. **backyard_loss**: with the shack cost Z in backyard supply, the closed-form first-order condition is a *minimum* when the rent R <= Z, and the legacy formula then rented out the whole backyard. Supply is now 0 there. With the flood_script settings, this affected 97 cells and 7,478 backyard households (about 10%).

The legacy RA also doubled state damages and ignored CC in state
probabilities; RA has since been reformulated (below).

Model changes (not bug fixes):
- Coastal floods (CO).
- Flood types combined per return period (legacy: max of expected damages across types).
- Full CRRA formulation of RA for all household choices, with risk-averse equilibrium rents (legacy: CRRA only for the protection choice, with risk-neutral rents).
- The solver returns the best iterate (legacy: last).
- `TOTAL_FORMAL = 810818 - TOTAL_RDP` (half the SAL sum; user change of 2026-09-30, legacy 821,028).

Kept on purpose: `compute_equilibrium` meant to damp utility steps by the
current error but read a not-yet-computed (zero) total, giving a constant
damping of about 1/1.5. The intended damping ("conv" in
`tools/check_legacy.py`) converged in 3/24 test scenarios vs. 12/24 for the
constant one, so `solver.solve` writes the constant damping explicitly.

Also removed: the NameError when `agents_anticipate_floods = 0`, the in-place
slicing of damage tables in `compute_equilibrium`, the global `np.seterr`,
the reliance on an indirect `numpy.matlib` import, and unused Excel reads.

## Verification

`tools/check_legacy.py` runs the legacy code with fixes 1-4 and
`config.TOTAL_FORMAL` patched in memory, against this package in legacy mode
(CO0, max of expected damages, last iterate). All 14 outputs are
bit-identical for AF CC RM SP RA = 10010, 11010 and 00000 (2026-09-30, after
the changes above). RA1 has no legacy equivalent.

Effects on results (AF1 CC0 CO1 RM0 SP1, sandbag 250, unless stated):
- Fix 1 (pluvial), vs unpatched legacy: settlement households -1.2% and settlers' mean expected structural damage doubles (1.8 -> 3.6 per mille). With CC1 RM1: settlements -3.3%, damage 3.8 -> 10.4 per mille.
- Fix 2 (rm_backyard): +3% backyard households under RM.
- Fix 4 (backyard_loss): backyard households 77,717 -> 71,561 (-7.9%), settlements +7,018, poorest group's utility -0.3%.
- Per-return-period max: formal households' mean expected structural damage +1.5%, little else.
- Coastal: dominant type in about 350 cells, where about 5,000 households live. Formal structural damage per household +4%; settlements unaffected.
- RA (CRRA 0.2772) vs RN:
  - Allocation: settlements -3.4%, backyards +3.4%, mean settlement rent +2.5%, utilities within 0.1%.
  - Protection take-up is unchanged at every positive sandbag cost: 0 at 250, 2,467 settler households at 100, 3,498 at 50, 7,595 at 25 (same with CC1).
  - At zero cost, take-up is lower (79,453 vs 84,351), because there are fewer settlers.
  - Stronger risk aversion barely helps. With CRRA 2, take-up is unchanged (2,467 at a cost of 100, 7,595 at 25). With CRRA 5, it is unchanged at 100 and only rises to 8,581 (6.3% of settlers) at 25.

## Known issues

- **The solver often does not converge to `precision` (0.1%)**, e.g. for the flood_script settings, even with 2,000 iterations.
  - Cause: a few lumpy cells at a near-tie between groups 0 and 1 (e.g. informal cell 21397, about 3,400 households) flip between groups, leaving group 0-1 totals 0.1-0.8% off target. Utilities are stable to about 1e-5.
  - Current behaviour: `solve` returns the best iterate (lowest max abs error).
  - `patience` can stop early once errors keep flipping, but tests showed early stops miss later, better iterates, because the step size keeps shrinking (10010 with max_iter 400: stop at 149 with 0.33% vs 0.06% at 282). It is therefore off by default. Raising `max_iter` to 400 reaches precision in some configurations.
  - Proper fix: allocate near-tied cells fractionally, splitting a cell between groups whose bids are within a tolerance, or a logit allocation of cells across bidders with a large scale parameter. Either makes aggregate demand continuous in utilities.
- **Self-protection take-up among settlers is zero** at `sandbag_course_cost = 250`, with or without RM, CC or RA. Protection is chosen only in cells without settlement land.
  - The benefit (expected damage of floods at or below 0.15 m, times structure and exposed contents values) is almost always below 250 rands a year.
  - Risk aversion does not help: sandbags only stop shallow, frequent, low-loss floods, whereas risk aversion mostly penalises rare deep floods.
- Coastal alignment on "smaller or equal" return periods is conservative. Frequent coastal flooding (the 0/2-yr DELTARES maps, already about 330 cells at 2 yr) only enters through the linear interpolation from zero damage at annual probability 1 to the 5-yr map.
- Protection is all-or-nothing at 0.15 m (no reduced damage above that depth).
- `mask_self_protec` flags the top bidder's choice even in cells without settlement land: weight by households when computing take-up.
- Formal bid rents use the unconstrained first-order condition even where the minimum lot size binds.

## Editing notes

- Keep everything vectorised over (groups x cells); no Python loops over cells.
- RA is the slow path (about 45-75 s). Per iteration, it runs two CRRA bisections for settlements (60 steps, only for pairs whose highest state-contingent bid is positive) and one for backyard supply (40 steps).
- Adding raw input files: list them in `data._SOURCES` (via `_FLOOD_FILES` for maps) and bump `CACHE_VERSION`.
- The order of floating-point operations on the RN path mirrors the legacy code, so that `tools/check_legacy.py` gives exact equality. A reordering can flip near-tied cells (see above) and produce visible but meaningless diffs. Check with the tool after changes.
