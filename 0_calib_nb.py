# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Notebook: run calibration

# ## Preamble
#
# ### Import packages

# region
# We import standard Python libraries
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import Image

# We also import our own packages
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import equilibrium.compute_equilibrium as eqcmp
import equilibrium.functions_dynamic as eqdyn
import calibration.calib_main_func as calmain
import outputs.export_outputs as outexp
# endregion

# ### Define file paths

# This corresponds to the architecture described in the README file
# (introduction tab of the documentation): the data folder is not hosted
# on the Github repository and should be placed in the root folder enclosing
# the repo.

path_code = '..'
path_folder = path_code + '/Data/'
path_precalc_inp = path_folder + 'precalculated_inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_data + 'Scenarios/'
path_outputs = path_code + '/Output/'
path_floods = path_folder + "flood_maps/"
path_input_plots = path_outputs + 'input_plots/'
path_input_tables = path_outputs + 'input_tables/'

# ### Create associated directories if needed

# region
try:
    os.mkdir(path_input_plots)
except OSError as error:
    print(error)

try:
    os.mkdir(path_input_tables)
except OSError as error:
    print(error)
# endregion

# ## Import parameters and options

# ### We import default parameters and options

import inputs.parameters_and_options as inpprm
options = inpprm.import_options()
param = inpprm.import_param(
    path_precalc_inp, options)

# ### We also set custom options for this simulation

# #### We first set options regarding structural assumptions used in the model

# Dummy for taking floods into account in agents' choices
options["agents_anticipate_floods"] = 0
# Dummy for preventing new informal settlement development
options["informal_land_constrained"] = 1

# #### Then we set options regarding flood data used

# Dummy for taking pluvial floods into account (on top of fluvial floods)
options["pluvial"] = 1
# Dummy for reducing pluvial risk for (better protected) formal structures
options["correct_pluvial"] = 1
# Dummy for taking coastal floods into account (on top of fluvial floods)
options["coastal"] = 1
# Digital elevation model to be used with coastal floods (MERITDEM or NASADEM)
# NB: MERITDEM is also the DEM used for fluvial and pluvial flood data
options["dem"] = "MERITDEM"
# Dummy for taking defended (vs. undefended) fluvial flood maps
# NB: FATHOM recommends to use undefended maps due to the high uncertainty
# in infrastructure modelling
options["defended"] = 0
# Dummy for taking sea-level rise into account in coastal flood data
# NB: Projections are up to 2050, based upon IPCC AR5 assessment for the
# RCP 8.5 scenario
options["climate_change"] = 0

# #### We also set options for scenarios on time-moving exogenous variables

# NB: Must be set to 1/2/3 for low/medium/high growth scenario
options["inc_ineq_scenario"] = 2
options["pop_growth_scenario"] = 3
options["fuel_price_scenario"] = 2

# #### Finally, we set options regarding data processing

# Default is set at zero to save computing time
# (data is simply loaded in the model).
#
# NB: this is only needed to create the data for the first time, or when the
# source is changed, so that pre-processed data is updated.

# TAKES SOME TIME
# Dummy for converting small-area-level (SAL) data into grid-level data
# (used for result validation)
options["convert_sal_data"] = 0
# Dummy for computing expected income net of commuting costs on the basis
# of calibrated wages (DYNAMIC)
options["compute_net_income"] = 1

# TAKES TIME
options["location_based_calib"] = 0


# ## Load data

# ### Basic geographic data

import inputs.data as inpdt
grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp, options)
geo_grid = gpd.read_file(path_data + "grid_reference_500.shp")
geo_TAZ = gpd.read_file(path_data + "TAZ_ampp_prod_attr_2013_2032.shp")


# ### Macro data

import inputs.data as inpdt
(interest_rate, population, housing_type_data, total_RDP, backyard_data
 ) = inpdt.import_macro_data(param, path_scenarios, path_folder)


# ### Households and income data

# region
import inputs.data as inpdt

income_class_by_housing_type = inpdt.import_hypothesis_housing_type()

(mean_income, households_per_income_class, average_income, income_mult,
 income_baseline, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

# NB: we create this parameter to maintain money illusion in simulations
# (see eqsim.run_simulation function)
param["income_year_reference"] = mean_income

# Other data at SP (small place) level used for calibration and validation
(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_baseline,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

# Import nb of households per pixel, by housing type (from SAL data)
# NB: RDP housing is included in formal, and there are both formal and informal
# backyards
if options["convert_sal_data"] == 1:
    housing_types = inpdt.import_sal_data(grid, path_folder, path_data,
                                          housing_type_data)
housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0
# endregion

# ### Land use projections

# region
# We import basic projections
import inputs.data as inpdt
(spline_RDP, spline_estimate_RDP, spline_land_RDP,
 spline_land_backyard, spline_land_informal, spline_land_constraints,
 number_properties_RDP) = (
     inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                           housing_type_data, path_data, path_folder)
     )

# We correct areas for each housing type at baseline year for the amount of
# constructible land in each type
import inputs.data as inpdt
coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, options, 0)

# We import housing heigth limits
import inputs.data as inpdt
housing_limit = inpdt.import_housing_limit(grid, param)

# We update parameter vector with construction parameters
# (relies on loaded data) and compute other variables
import inputs.parameters_and_options as inpprm
(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_baseline, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )
# endregion

# ### Import flood data (takes some time when agents anticipate floods)

# region
# If agents anticipate floods, we return output from damage functions
import inputs.data as inpdt
if options["agents_anticipate_floods"] == 1:
    (fraction_capital_destroyed, structural_damages_small_houses,
     structural_damages_medium_houses, structural_damages_large_houses,
     content_damages, structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b
     ) = inpdt.import_full_floods_data(options, param, path_folder)

# Else, we set those outputs as zero
# NB: 24014 is the number of grid pixels
elif options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = pd.DataFrame()
    fraction_capital_destroyed["structure_formal_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_formal_1"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_1"] = np.zeros(24014)
    fraction_capital_destroyed["contents_formal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_informal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_subsidized"] = np.zeros(24014)
    fraction_capital_destroyed["contents_backyard"] = np.zeros(24014)
    fraction_capital_destroyed["structure_backyards"] = np.zeros(24014)
    fraction_capital_destroyed["structure_formal_backyards"] = np.zeros(24014)
    fraction_capital_destroyed["structure_informal_backyards"
                               ] = np.zeros(24014)
    fraction_capital_destroyed["structure_informal_settlements"
                               ] = np.zeros(24014)
# endregion

# ### Import scenarios (for time-moving variables)

import equilibrium.functions_dynamic as eqdyn
(spline_agricultural_price, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_baseline, param, grid, path_scenarios,
                            options)

# ### Import income net of commuting costs (for all time periods)

# region
import inputs.data as inpdt
if options["compute_net_income"] == 1:
    (incomeNetOfCommuting, modalShares, ODflows, averageIncome
     ) = inpdt.import_transport_data(
         grid, param, 0, households_per_income_class, average_income,
         spline_inflation, spline_fuel,
         spline_population_income_distribution, spline_income_distribution,
         path_precalc_inp, path_precalc_transp, 'GRID', options)

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
# endregion


# ## Prepare data

# ### Define dominant income group in each census block (SP)

# region
# In one case, we base our definition on the median income at the SP-level:
# we consider as dominant the group corresponding to the highest income
# threshold is crossed
if options["correct_dominant_incgrp"] == 0:
    data_income_group = np.zeros(len(data_sp["income"]))
    for j in range(0, param["nb_of_income_classes"] - 1):
        data_income_group[data_sp["income"] >
                          threshold_income_distribution[j]] = j+1

# In the other case, we just consider the most numerous income group in each
# census block
elif options["correct_dominant_incgrp"] == 1:
    data_income_group = np.zeros(len(income_distribution))
    for i in range(0, len(income_distribution)):
        data_income_group[i] = np.argmax(income_distribution[i])
# endregion

# Although the second option seems more logical, we take the first one as
# default since we are going to consider median SP prices, and we want
# associated net income to be in line with those values to avoid a sample
# selection bias in our regressions. Indeed, the purpose of such definition
# is to choose the right net income to estimate associated regressions.

# ### Obtain number of formal private housing units per SP

# NB: It is not clear whether RDP are included in SP formal count, and
# if they should be taken out based on imperfect cadastral estimations.
# For reference, we include the two options.

# region
if options["substract_RDP_from_formal"] == 1:
    # We retrieve number of RDP units per SP from grid-level data
    grid_intersect = pd.read_csv(path_data + 'grid_SP_intersect.csv',
                                 sep=';')
    # When pixels are associated to several SPs, we allocate them to the
    # one with the biggest intersection area.
    # NB: it would be more rigorous to split the number of RDP across
    # SPs according to their respective intersection areas, but this is
    # unlikely to change much
    grid_intersect = grid_intersect.groupby('ID_grille').max('Area')
    data_rdp["ID_grille"] = data_rdp.index
    data_rdp["ID_grille"] = data_rdp["ID_grille"] + 1

    rdp_grid = pd.merge(data_rdp, grid_intersect, on="ID_grille",
                        how="outer")
    rdp_sp = rdp_grid.groupby('SP_CODE')['count'].sum()
    rdp_sp = rdp_sp.reset_index()
    rdp_sp = rdp_sp.rename(columns={'SP_CODE': 'sp_code'})
    # We just fill the list with unmatched SPs to get the full SP vector
    rdp_sp_fill = pd.merge(rdp_sp, data_sp['sp_code'], on="sp_code",
                           how="outer")
    rdp_sp_fill['count'] = rdp_sp_fill['count'].fillna(0)
    rdp_sp_fill = rdp_sp_fill.sort_values(by='sp_code')

    data_number_formal = (
        housing_types_sp.total_dwellings_SP_2011
        - housing_types_sp.backyard_SP_2011
        - housing_types_sp.informal_SP_2011
        - rdp_sp_fill['count'])

elif options["substract_RDP_from_formal"] == 0:
    data_number_formal = (
        housing_types_sp.total_dwellings_SP_2011
        - housing_types_sp.backyard_SP_2011
        - housing_types_sp.informal_SP_2011
        )
# endregion

# Given the uncertainty surrounding RDP counts, we take the second option as
# default and prefer to rely on sample selection (see below) to exclude the SPs
# where RDP housing is likely to drive most of our results.

# ### Sample selection

# As the relations we are going to estimate are only true for the
# formal private sector, we exclude SPs in the bottom quintile of property
# prices and for which more than 5% of households are reported to live in
# informal housing (settlements + backyards). We also exclude "rural" SPs
# (i.e., those that are large, with a small share than can be urbanized).
# This is more specific to the regression allowing us to
# estimate parameters of the construction function (fit on building density),
# hence the name of the selected_density variable.

# We also add options to consider other criteria, namely we offer to
# exclude the poorest income group (which is in effect crowded out from the
# formal sector), as well as Mitchell's Plain (as its housing market is
# very specific) and far-away land (where SPs are too big for aggregates to
# be representative of micro conditions).

# region
if options["correct_selected_density"] == 0:
    selected_density = (
        (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
        & (data_number_formal
           > 0.95 * housing_types_sp.total_dwellings_SP_2011)
        & (data_sp["unconstrained_area"]
            < np.nanquantile(data_sp["unconstrained_area"], 0.8))
        & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
        )

elif (options["correct_selected_density"] == 1
      and options["correct_mitchells_plain"] == 0):
    selected_density = (
        (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
        & (data_number_formal
           > 0.95 * housing_types_sp.total_dwellings_SP_2011)
        & (data_sp["unconstrained_area"]
            < np.nanquantile(data_sp["unconstrained_area"], 0.8))
        & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
        & (data_income_group > 0)
        & (data_sp["distance"] < 40)
        )

elif (options["correct_selected_density"] == 1
      and options["correct_mitchells_plain"] == 1):
    selected_density = (
        (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
        & (data_number_formal
           > 0.95 * housing_types_sp.total_dwellings_SP_2011)
        & (data_sp["unconstrained_area"]
            < np.nanquantile(data_sp["unconstrained_area"], 0.8))
        & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
        & (data_income_group > 0)
        & (data_sp["mitchells_plain"] == 0)
        & (data_sp["distance"] < 40)
        )
# endregion

# We pick the second choice as our default since it is more conservative than
# the first, and less ad hoc than the third one.

# ## Calibrate construction function parameters

# We estimate construction function parameters based on a log-linearization
# of the housing market clearing condition
import calibration.calib_main_func as calmain
coeff_b, coeff_a, coeffKappa = calmain.estim_construct_func_param(
    options, param, data_sp, threshold_income_distribution,
    income_distribution, data_rdp, housing_types_sp,
    data_number_formal, data_income_group, selected_density,
    path_data, path_precalc_inp, path_folder)

# NB: The results are automatically saved for later use in simulations.

# region
# We update parameter vector
param["coeff_a"] = coeff_a
param["coeff_b"] = coeff_b
param["coeff_A"] = coeffKappa

# We print the calibrated values
print("coeff_a = " + str(param["coeff_a"]))
print("coeff_b = " + str(param["coeff_b"]))
print("coeff_A = " + str(param["coeff_A"]))

# We save them
# NB: coeff_a = 1 - coeff_b
np.save(path_precalc_inp + 'calibratedHousing_b.npy',
        param["coeff_b"])
np.save(path_precalc_inp + 'calibratedHousing_kappa.npy',
        param["coeff_A"])
# endregion


# ## Calibrate incomes and gravity parameter

# We scan values for the gravity parameter to estimate incomes as a
# function of it. The value range is set by trial and error: the wider the
# range you want to test, the longer. In principle, we should find a value
# within a coarse interval before going to the finer level: this may require
# several iterations if the underlying data changes.
#
# NB: We conduct scanning as it is too long and complex to run a solver directly.

# Then, we select the income-gravity pair that best fits the distribution
# of commuters over distance from the CBD.
#
# NB: We need to proceed in two steps as there is no separate identification
# of the gravity parameter and the net incomes.

# We start by selecting the range over which we want to scan
if options["scan_type"] == "rough":
    list_lambda = [13, 14, 15]
if options["scan_type"] == "normal":
    list_lambda = np.arange(13.8, 14.3, 0.2)
if options["scan_type"] == "fine":
    list_lambda = np.arange(13.94, 13.981, 0.02)
    
# TEST: a bit long due to nb of job centers?
# Adjust numerical parameters for crude scanning? Then, may not be good enough
# DNF to check error terms!
#list_lambda = np.arange(1, 16, 1)

# We then run the function that returns the calibrated outputs
# Note that depending on the precision level and the maximum number of
# iterations, this may take a while
import calibration.calib_main_func as calmain
(incomeCentersKeep, lambdaKeep, cal_avg_income, scoreKeep,
 bhattacharyyaDistances) = (
    calmain.estim_incomes_and_gravity(
        param, grid, list_lambda, households_per_income_class,
        average_income, income_distribution, spline_inflation, spline_fuel,
        spline_population_income_distribution, spline_income_distribution,
        path_data, path_precalc_inp, path_precalc_transp, options)
    )
     
# NB: we just use cal_avg_income for validation wrt average_income

# region
# We update the parameter vector
param["lambda"] = np.array(lambdaKeep)

# We print the calibrated values (when possible)
print("lambda = " + str(param["lambda"]))

# We save them
np.save(path_precalc_inp + 'lambdaKeep.npy',
        param["lambda"])
np.save(path_precalc_inp + 'incomeCentersKeep.npy',
        incomeCentersKeep)
# endregion

# Note that incomes are fitted to reproduce the observed distribution of jobs
# across income groups (in selected job centers), based on the commuting choice
# model.

# ### Let us first visualize inputs

# We import the employment data at the TAZ (transport zone) level
import outputs.export_outputs as outexp
jobsTable, selected_centers = outexp.import_employment_geodata(
    households_per_income_class, param, path_data)

# region
# Then, we visualize number of jobs for income group 1
jobs_poor_2d = pd.merge(geo_TAZ, jobsTable["poor_jobs"],
                        left_index=True, right_index=True)
jobs_poor_2d = pd.merge(jobs_poor_2d, selected_centers,
                        left_index=True, right_index=True)
jobs_poor_2d.drop('geometry', axis=1).to_csv(
    path_input_tables + 'jobs_poor' + '.csv')
jobs_poor_2d_select = jobs_poor_2d[
    (jobs_poor_2d.geometry.bounds.maxy < -3740000)
    & (jobs_poor_2d.geometry.bounds.maxx < -10000)].copy()
jobs_poor_2d_select.loc[
    jobs_poor_2d_select["selected_centers"] == 0, 'jobs_poor'
    ] = 0

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of poor jobs in selected job centers (data)")
jobs_poor_2d_select.plot(column='poor_jobs', ax=ax,
                         cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'jobs_poor_in_selected_centers')
plt.close()

Image(path_input_plots + "jobs_poor_in_selected_centers.png")
# endregion

# region
# Then for income group 2
jobs_midpoor_2d = pd.merge(geo_TAZ, jobsTable["midpoor_jobs"],
                           left_index=True, right_index=True)
jobs_midpoor_2d = pd.merge(jobs_midpoor_2d, selected_centers,
                           left_index=True, right_index=True)
# jobs_midpoor_2d.to_file(path_tables + 'jobs_midpoor' + '.shp')
jobs_midpoor_2d.drop('geometry', axis=1).to_csv(
    path_input_tables + 'jobs_midpoor' + '.csv')
jobs_midpoor_2d_select = jobs_midpoor_2d[
    (jobs_midpoor_2d.geometry.bounds.maxy < -3740000)
    & (jobs_midpoor_2d.geometry.bounds.maxx < -10000)].copy()
jobs_midpoor_2d_select.loc[
    jobs_midpoor_2d_select["selected_centers"] == 0, 'jobs_midpoor'
    ] = 0

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of midpoor jobs in selected job centers (data)")
jobs_midpoor_2d_select.plot(column='midpoor_jobs', ax=ax,
                            cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'jobs_midpoor_in_selected_centers')
plt.close()

Image(path_input_plots + "jobs_midpoor_in_selected_centers.png")
# endregion

# region
# For income group 3
jobs_midrich_2d = pd.merge(geo_TAZ, jobsTable["midrich_jobs"],
                           left_index=True, right_index=True)
jobs_midrich_2d = pd.merge(jobs_midrich_2d, selected_centers,
                           left_index=True, right_index=True)
# jobs_midrich_2d.to_file(path_tables + 'jobs_midrich' + '.shp')
jobs_midrich_2d.drop('geometry', axis=1).to_csv(
    path_input_tables + 'jobs_midrich' + '.csv')
jobs_midrich_2d_select = jobs_midrich_2d[
    (jobs_midrich_2d.geometry.bounds.maxy < -3740000)
    & (jobs_midrich_2d.geometry.bounds.maxx < -10000)].copy()
jobs_midrich_2d_select.loc[
    jobs_midrich_2d_select["selected_centers"] == 0, 'jobs_midrich'
    ] = 0

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of midrich jobs in selected job centers (data)")
jobs_midrich_2d_select.plot(column='midrich_jobs', ax=ax,
                            cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'jobs_midrich_in_selected_centers')
plt.close()

Image(path_input_plots + "jobs_midrich_in_selected_centers.png")
# endregion

# region
# For income group 4
jobs_rich_2d = pd.merge(geo_TAZ, jobsTable["rich_jobs"],
                        left_index=True, right_index=True)
jobs_rich_2d = pd.merge(jobs_rich_2d, selected_centers,
                        left_index=True, right_index=True)
# jobs_rich_2d.to_file(path_tables + 'jobs_rich' + '.shp')
jobs_rich_2d.drop('geometry', axis=1).to_csv(
    path_input_tables + 'jobs_rich' + '.csv')
jobs_rich_2d_select = jobs_rich_2d[
    (jobs_rich_2d.geometry.bounds.maxy < -3740000)
    & (jobs_rich_2d.geometry.bounds.maxx < -10000)].copy()
jobs_rich_2d_select.loc[
    jobs_rich_2d_select["selected_centers"] == 0, 'jobs_rich'
    ] = 0

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of rich jobs in selected job centers (data)")
jobs_rich_2d_select.plot(column='rich_jobs', ax=ax,
                         cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'jobs_rich_in_selected_centers')
plt.close()

Image(path_input_plots + "jobs_rich_in_selected_centers.png")
# endregion

# ### We visualize the associated calibrated outputs

# We first load and rearrange the data that we just computed
income_centers_init = np.load(
    path_precalc_inp + 'incomeCentersKeep.npy')
income_centers_init[income_centers_init < 0] = 0
income_centers_init_merge = pd.DataFrame(income_centers_init)
income_centers_init_merge = income_centers_init_merge.rename(
    columns={income_centers_init_merge.columns[0]: 'poor_income',
             income_centers_init_merge.columns[1]: 'midpoor_income',
             income_centers_init_merge.columns[2]: 'midrich_income',
             income_centers_init_merge.columns[3]: 'rich_income'})
income_centers_init_merge["count"] = income_centers_init_merge.index + 1

# region
# We prepare, execute, and clean the merge with geographic grid
selected_centers_merge = selected_centers.copy()
(selected_centers_merge["count"]
 ) = selected_centers_merge.selected_centers.cumsum()
selected_centers_merge.loc[
    selected_centers_merge.selected_centers == 0, "count"] = 0

income_centers_TAZ = pd.merge(income_centers_init_merge,
                              selected_centers_merge,
                              how='right', on='count')

income_centers_2d = pd.merge(geo_TAZ, income_centers_TAZ,
                             left_index=True, right_index=True)
income_centers_2d.drop('geometry', axis=1).to_csv(
    path_input_tables + 'income_centers_2d' + '.csv')

income_centers_2d_select = income_centers_2d[
    (income_centers_2d.geometry.bounds.maxy < -3740000)
    & (income_centers_2d.geometry.bounds.maxx < -10000)].copy()
# endregion

# region
# We then visualize calibrated incomes for income group 1
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Annual incomes (in R) per job center for poor households")
income_centers_2d_select.plot(column='poor_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'poor_income_in_selected_centers')
plt.close()

Image(path_input_plots + "poor_income_in_selected_centers.png")
# endregion

# region
# For income group 2
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Annual incomes (in R) per job center for midpoor households")
income_centers_2d_select.plot(column='midpoor_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'midpoor_income_in_selected_centers')
plt.close()

Image(path_input_plots + "midpoor_income_in_selected_centers.png")
# endregion

# region
# For income group 3
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Annual incomes (in R) per job center for midrich households")
income_centers_2d_select.plot(column='midrich_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'midrich_income_in_selected_centers')
plt.close()

Image(path_input_plots + "midrich_income_in_selected_centers.png")
# endregion

# region
# For income group 4
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Annual incomes (in R) per job center for rich households")
income_centers_2d_select.plot(column='rich_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_input_plots + 'rich_income_in_selected_centers')
plt.close()

Image(path_input_plots + "rich_income_in_selected_centers.png")
# endregion


# ## Calibrate utility function parameters

# We compute local incomes net of commuting costs at the SP (not grid)
# level that is used in calibration.

# region
# Note that lambda and calibrated incomes have an impact here: from now on,
# we will stop loading precalibrated parameters to rely on the newly
# calibrated parameters that we just saved
options["load_precal_param"] = 0

import inputs.data as inpdt
(incomeNetOfCommuting, *_
 ) = inpdt.import_transport_data(
     grid, param, 0, households_per_income_class, average_income,
     spline_inflation, spline_fuel,
     spline_population_income_distribution, spline_income_distribution,
     path_precalc_inp, path_precalc_transp, 'SP', options)
# endregion

# Then we calibrate utility function parameters based on the maximization
# of a composite likelihood that reproduces the fit on exogenous amenities,
# dwelling sizes, and income sorting.

# NB: Here, we also have an impact from construction parameters and sample
# selection (+ number of formal units)
import calibration.calib_main_func as calmain
(calibratedUtility_beta, calibratedUtility_q0, cal_amenities, u3, u4
 ) = calmain.estim_util_func_param(
     data_number_formal, data_income_group, housing_types_sp, data_sp,
     coeff_a, coeff_b, coeffKappa, interest_rate,
     incomeNetOfCommuting, path_data, path_precalc_inp,
     options, param)

# region
# We update parameter vector
param["beta"] = calibratedUtility_beta
param["q0"] = calibratedUtility_q0

# We print calibrated values (when possible)
print("beta = " + str(param["beta"]))
print("q0 = " + str(param["q0"]))

# ARE AMENITIES GOOD ENOUGH?
# Just check with formal fit? Informal fit will be dealt with disamenities

# We save them
# NB : alpha = 1 - beta
np.save(path_precalc_inp + 'calibratedUtility_beta.npy',
        param["beta"])
np.save(path_precalc_inp + 'calibratedUtility_q0.npy',
        param["q0"])
np.save(path_precalc_inp + 'calibratedAmenities.npy',
        cal_amenities)
# endregion

# TAKES TIME WHEN LOCAL!

# REVIEW FINAL OUTPUTS IN EQUILIBRIUM FUNCTION?

# DOUBLE CHECK SINCE WE CHANGED COST STRUCTURES
options["incremental_housing"] = 1
# param["disam_reduc_fact"] = 1

# Interestingly, baseline model is not so good when introducing formal backayards in the target

# ## Calibrate disamenity index for informal backyards + settlements

# region
# We first need to recompute income net of commuting costs at baseline
# year since calibrated income has changed

import inputs.data as inpdt
(incomeNetOfCommuting, modalShares, ODflows, averageIncome
 ) = inpdt.import_transport_data(
     grid, param, 0, households_per_income_class, average_income,
     spline_inflation, spline_fuel,
     spline_population_income_distribution, spline_income_distribution,
     path_precalc_inp, path_precalc_transp, 'GRID', options)

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
# endregion

# Then, we do the same for the amenity index (calibrated values normalized around 1)
import inputs.data as inpdt
amenities = inpdt.import_amenities(path_precalc_inp, options)

# region
# Let us visualize the calibrated amenity index
import outputs.export_outputs as outexp
amenity_map = outexp.export_map(
    amenities, grid, geo_grid, path_input_plots, 'amenity_map',
    "Amenity score map (from .01 to .99 percentile)",
    path_input_tables,
    ubnd=np.quantile(amenities, 0.9999),
    lbnd=np.quantile(amenities, 0.0001))

Image(path_input_plots + "amenity_map.png")
# endregion

# Note the the local amenity index is only as good as the underlying exogenous amenity covariates used for prediction: this can be updated in future work.

# NB: In what follows, since disamenity index calibration relies on the model fit and is not
# computed a priori (contrary to other parameters), the options set in the
# preamble should be the same as the ones used in the main script, so that
# the calibrated values are in line with the structural assumptions used there.

# ### We start with a general (not location-specific) calibration

# We define a range of disamenity values which we would like to scan,
# and arrange them in a grid
# NB: as for gravity parameter, we set the range by trial and error
#list_amenity_backyard = np.arange(0.55, 0.601, 0.01)
#list_amenity_settlement = np.arange(0.55, 0.601, 0.01)

# TEST: a bit long?
# list_amenity_backyard = np.arange(0.1, 0.9, 0.1)
# list_amenity_settlement = np.arange(0.1, 0.9, 0.1)

# ROUGH: GIVES 0.7
# list_amenity_settlement = np.arange(0.5, 1.01, 0.1)
# list_amenity_backyard = np.arange(0.5, 1.01, 0.1)
# list_amenity_incremental = np.arange(0.5, 1.01, 0.1)

# FINE (a bit long): gives 0.76, 0.78, and 0.8...
# FOR TARGETING TOTAL BACKYARDS!
# list_amenity_settlement = np.arange(0.61, 0.8, 0.01)
# list_amenity_backyard = np.arange(0.61, 0.8, 0.01)
# list_amenity_incremental = np.arange(0.61, 0.8, 0.01)

# Not bad!
# Now test with higher precision and/or finer step
# list_amenity_settlement = np.arange(0.7, 0.9, 0.025)
# list_amenity_backyard = np.arange(0.7, 0.9, 0.025)
# list_amenity_incremental = np.arange(0.7, 0.9, 0.025)
# Good! We have .78, .8 and .84
# list_amenity_settlement = np.arange(0.75, 0.9, 0.01)
# list_amenity_backyard = np.arange(0.75, 0.9, 0.01)
# list_amenity_incremental = np.arange(0.75, 0.9, 0.01)
# Same!!
list_amenity_settlement = np.arange(0.75, 0.8, 0.01)
list_amenity_backyard = np.arange(0.77, 0.82, 0.01)
list_amenity_incremental = np.arange(0.81, 0.86, 0.01)

# NB: need to adjust incremental amenities to have enough people in formal housing
# list_amenity_settlement = 0.7
# list_amenity_backyard = 0.74
# list_amenity_incremental = 1

# list_amenity_backyard = [1]
# list_amenity_settlement = [1]
# housing_type_total = pd.DataFrame(np.array(np.meshgrid(
#     list_amenity_backyard, list_amenity_settlement)).T.reshape(-1, 2))
# housing_type_total.columns = ["param_backyard", "param_settlement"]

housing_type_total = pd.DataFrame(np.array(np.meshgrid(
    list_amenity_backyard, list_amenity_settlement, list_amenity_incremental)).T.reshape(-1, 3))

housing_type_total.columns = ["param_backyard", "param_settlement", "param_incremental"]

housing_type_total = housing_type_total[
   (housing_type_total["param_settlement"]<=housing_type_total["param_backyard"])
   & (housing_type_total["param_backyard"]<=housing_type_total["param_incremental"])]

# We initialize output vector
housing_type_total["formal"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["backyard"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["informal"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["subsidized"] = np.zeros(
    len(housing_type_total.param_backyard))

housing_type_total["backyard_formal"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["backyard_informal"] = np.zeros(
    len(housing_type_total.param_backyard))

# We print the number of total iterations (to have an intuition of how long
# the process will take)
# number_total_iterations = (
#     len(list_amenity_backyard) * len(list_amenity_settlement))
# number_total_iterations = (
#     len(list_amenity_backyard) * len(list_amenity_settlement) * len(list_amenity_incremental))
# print(f"** Calibration: {number_total_iterations} iterations **")

# We are going to compute the initial state equilibrium for each pair of
# parameters, and retain the one that best fits the observed number of
# households in informal settlements + backyards.

# region

# ALLOWS TO SAVE TIME
# param["precision"] = 0.01
param["precision"] = 0.001

options["incremental_housing"] = 1

# iteration_number = 0

import equilibrium.compute_equilibrium as eqcmp
for i in range(0, len(housing_type_total)):
# for i in range(0, len(list_amenity_backyard)):
#     for j in range(0, len(list_amenity_settlement)):
#         for k in range(0, len(list_amenity_incremental)):

    # We set input values
    # param["amenity_backyard"] = list_amenity_backyard[i]
    # param["amenity_settlement"] = list_amenity_settlement[j]
    # param["amenity_incremental"] = list_amenity_incremental[k]
    param["amenity_backyard"] = housing_type_total["param_backyard"].iloc[i]
    param["amenity_settlement"] = housing_type_total["param_settlement"].iloc[i]
    param["amenity_incremental"] = housing_type_total["param_incremental"].iloc[i]
    
    print("amenity_settlement = " + str(param["amenity_settlement"]))
    print("amenity_backyard = " + str(param["amenity_backyard"]))
    print("amenity_incremental = " + str(param["amenity_incremental"]))
    
    param["informal_pockets"] = np.ones(24014) * param["amenity_settlement"]
    param["backyard_pockets"] = np.ones(24014) * param["amenity_backyard"]
    param["incremental_pockets"] = np.ones(24014) * param["amenity_incremental"]  

    # We run the algorithm
    (initial_state_utility,
     initial_state_error,
     initial_state_simulated_jobs,
     initial_state_households_housing_types,
     initial_state_household_centers,
     initial_state_households,
     initial_state_dwelling_size,
     initial_state_housing_supply,
     initial_state_rent,
     initial_state_rent_matrix,
     initial_state_capital_land,
     initial_state_average_income,
     initial_state_limit_city) = eqcmp.compute_equilibrium(
         fraction_capital_destroyed,
         amenities,
         param,
         housing_limit,
         population,
         households_per_income_class,
         total_RDP,
         coeff_land,
         income_net_of_commuting_costs,
         grid,
         options,
         agricultural_rent,
         interest_rate,
         number_properties_RDP,
         average_income,
         mean_income,
         income_class_by_housing_type,
         minimum_housing_supply,
         param["coeff_A"],
         income_baseline)

    # We fill output matrix with the total number of HHs per housing
    # type for given values of backyard and informal amenity parameters
    housing_type_total.iloc[
        (housing_type_total.param_backyard
         == param["amenity_backyard"])
        & (housing_type_total.param_settlement
           == param["amenity_settlement"])
        & (housing_type_total.param_incremental
           == param["amenity_incremental"]),
        3:7] = np.nansum(initial_state_households_housing_types, 1)
    
    # Alternative score...
    backyard_people = initial_state_households_housing_types[1]    
    backyard_hsupply = initial_state_housing_supply[1]/1000000
    formal_backyard_people = np.nansum(backyard_people[backyard_hsupply==2])
    informal_backyard_people = np.nansum(backyard_people) - formal_backyard_people
    # print("formal_backyard_people = " + str(formal_backyard_people))
    # print("informal_backyard_people = " + str(informal_backyard_people))
    
    housing_type_total.iloc[
        (housing_type_total.param_backyard
         == param["amenity_backyard"])
        & (housing_type_total.param_settlement
           == param["amenity_settlement"])
        & (housing_type_total.param_incremental
           == param["amenity_incremental"]),
        7:9] = [formal_backyard_people, informal_backyard_people]
    
    # We update the iteration count and print progress made
    iteration_number = i + 1
    # print(f"iteration {iteration_number}")
    print(f"iteration {iteration_number}/{len(housing_type_total)}")
# endregion

# DO WE NEED AN ALTERNATIVE MSE SCORE?

# region
# We compute the error between simulated and observed number of households
# in each housing type (without RDP, which is exogenously set equal to data)
# distance_share = np.abs(
#     housing_type_total.iloc[:, 3:7] - housing_type_data[None, 0:4])

# We define the score that we want to minimize as the sum of the errors for
# informal backyards and informal settlements
# distance_share_score = (
#     distance_share.iloc[:, 1] + distance_share.iloc[:, 2])

# Alternative
error_informal = (housing_type_total.iloc[:, 5] - housing_type_data[None, 2])**2
error_formal_backyard = (housing_type_total.iloc[:, 7] - backyard_data[0])**2
error_informal_backyard = (housing_type_total.iloc[:, 8] - backyard_data[1])**2

# Do not include everything as formal is essentially a residual
# Need to reweight some errors? Check aggregate numbers at the end
distance_share_score= (
    error_informal + error_formal_backyard + error_informal_backyard)

# We select the arguments associated with the minimum
which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)


calibrated_amenities = housing_type_total.iloc[which, 0:3]
# endregion

# region
# We update parameter vector (after ensuring that param["amenity_backyard"] is
# bigger than param["amenity_settlement"])
param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]
param["amenity_incremental"] = calibrated_amenities[2]

# We print the calibrated values
print(housing_type_total.iloc[which])
print(housing_type_data)
print(backyard_data)

# CHECK BEFORE SAVING

# We save them
np.save(path_precalc_inp + 'param_amenity_backyard.npy',
        param["amenity_backyard"])
np.save(path_precalc_inp + 'param_amenity_settlement.npy',
        param["amenity_settlement"])
np.save(path_precalc_inp + 'param_amenity_incremental.npy',
        param["amenity_incremental"])
# endregion

# DEAL WITH PRECISION PARAMETER FIRST?


# NB: we ensure interior solution

# param["init_util_levels"] = initial_state_utility

# np.save(path_precalc_inp + 'init_util_levels.npy',
#         param["init_util_levels"])

print(np.nansum(initial_state_households_housing_types,1))
print(housing_type_data)

###########################################################################

# TEST // Scanning step also matters!

# PRevious working paper version
# param["amenity_backyard"] = 0.74
# param["amenity_settlement"] = 0.70

# ### Calibrate location-specific disamenity index

# MATTERS WHEN ACCOUNTING FOR STRUCTURES!
# What is in housing_type_data?
# NB: irrelevant in equilibrium function
# options["actual_backyards"] = 1

# TODO: Need more precision or adaptive updates?

# Default is set to 1 but can be changed if we fear overfit of the model
import equilibrium.compute_equilibrium as eqcmp
if options["location_based_calib"] == 1:

    # We start from where we left (to gain time) and compute the
    # equilibrium again

    # We first initialize input values

    index = 0
    index_max = 100
    #index_max = 50
    metrics = np.zeros(index_max)

    param["informal_pockets"] = np.zeros(24014) + param["amenity_settlement"]
    save_param_informal_settlements = np.zeros((index_max, 24014))
    metrics_is = np.zeros(index_max)
    param["backyard_pockets"] = np.zeros(24014) + param["amenity_backyard"]
    save_param_backyards = np.zeros((index_max, 24014))
    metrics_ib = np.zeros(index_max)
    param["incremental_pockets"] = np.zeros(24014) + param["amenity_incremental"]
    save_param_incremental = np.zeros((index_max, 24014))
    metrics_ih = np.zeros(index_max)

    # We run the algorithm
    (initial_state_utility,
     initial_state_error,
     initial_state_simulated_jobs,
     initial_state_households_housing_types,
     initial_state_household_centers,
     initial_state_households,
     initial_state_dwelling_size,
     initial_state_housing_supply,
     initial_state_rent,
     initial_state_rent_matrix,
     initial_state_capital_land,
     initial_state_average_income,
     initial_state_limit_city
     ) = eqcmp.compute_equilibrium(
         fraction_capital_destroyed,
         amenities,
         param,
         housing_limit,
         population,
         households_per_income_class,
         total_RDP,
         coeff_land,
         income_net_of_commuting_costs,
         grid,
         options,
         agricultural_rent,
         interest_rate,
         number_properties_RDP,
         average_income,
         mean_income,
         income_class_by_housing_type,
         minimum_housing_supply,
         param["coeff_A"],
         income_baseline)

    # We set the maximum number of iterations
    number_total_iterations = index_max

    # Then we optimize over the number of households per housing type
    # PER PIXEL, and not just on the aggregate number (to acccount for
    # differing disamenities per location, e.g. eviction probability,
    # infrastructure networks, etc.)

    # To do so, we use granular housing_types variable (from SAL data) instead
    # of aggregate housing_types variable

    for index in range(0, index_max):

        hsupply_backyarding = initial_state_housing_supply[1, :]/1000000        

        # INFORMAL SETTLEMENTS

        # We initialize output vector
        diff_is = np.zeros(24014)
        for i in range(0, 24014):
            # We store the error term
            diff_is[i] = (housing_types.informal_grid[i]
                          - initial_state_households_housing_types[2, :][i]
                          )
            # We apply an empirical reweighting that helps convergence
            adj = (diff_is[i] / (np.nanmax(diff_is) * 100))
            # We increase the amenity score when we underestimate the nb of
            # households
            param["informal_pockets"][i] = param["informal_pockets"][i] + adj
        # We store iteration outcome and prevent extreme sorting from
        # happening due to the amenity score
        metrics_is[index] = sum(np.abs(diff_is))
        param["informal_pockets"][param["informal_pockets"] < 0.01] = 0.01
        param["informal_pockets"][param["informal_pockets"] > 0.99] = 0.99
        save_param_informal_settlements[index, :] = param["informal_pockets"]

        # INFORMAL BACKYARDS

        # We initialize output vector
        diff_ib = np.zeros(24014)
        for i in range(0, 24014):
            # Note that we add an option depending on whether we restrict
            # ourselves to informal backyards (default) or all kinds of
            # backyards (not warranted given the standardized structure
            # assumed in the model)
            # if options["actual_backyards"] == 1:
            #     diff_ib[i] = (
            #         housing_types.backyard_informal_grid[i]
            #         + housing_types.backyard_formal_grid[i]
            #         - initial_state_households_housing_types[1, :][i])
            # elif options["actual_backyards"] == 0:
            #     diff_ib[i] = (
            #         housing_types.backyard_informal_grid[i]
            #         - initial_state_households_housing_types[1, :][i])
            nb_hh_ib = initial_state_households_housing_types[1, :]
            nb_hh_ib[hsupply_backyarding==2] = 0
            diff_ib[i] = (
                housing_types.backyard_informal_grid[i]
                - nb_hh_ib[i])
            # We help convergence and update parameter
            adj = (diff_ib[i] / (np.nanmax(diff_ib) * 100))
            param["backyard_pockets"][i] = (
                param["backyard_pockets"][i] + adj)
        # We store iteration output and prevent extreme sorting
        metrics_ib[index] = sum(np.abs(diff_ib))
        param["backyard_pockets"][param["backyard_pockets"] < 0.01] = 0.01
        param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
        save_param_backyards[index, :] = param["backyard_pockets"]
        
        # INCREMENTAL HOUSING: no need if no specific target!
        diff_ih = np.zeros(24014)
        for i in range(0, 24014):
            # Note that we add an option depending on whether we restrict
            # ourselves to informal backyards (default) or all kinds of
            # backyards (not warranted given the standardized structure
            # assumed in the model)
            # if options["actual_backyards"] == 1:
            #     diff_ih[i] = (
            #         housing_types.backyard_informal_grid[i]
            #         + housing_types.backyard_formal_grid[i]
            #         - initial_state_households_housing_types[1, :][i])
            # elif options["actual_backyards"] == 0:
            #     diff_ih[i] = (
            #         housing_types.backyard_informal_grid[i]
            #         - initial_state_households_housing_types[1, :][i])
            nb_hh_ih = initial_state_households_housing_types[1, :]
            nb_hh_ih[hsupply_backyarding<2] = 0
            diff_ih[i] = (
                housing_types.backyard_formal_grid[i]
                - nb_hh_ih[i])
            # We help convergence and update parameter
            adj = (diff_ih[i] / (np.nanmax(diff_ih) * 100))
            param["incremental_pockets"][i] = (
                param["incremental_pockets"][i] + adj)
        # We store iteration output and prevent extreme sorting
        metrics_ih[index] = sum(np.abs(diff_ih))
        param["incremental_pockets"][param["incremental_pockets"] < 0.01] = 0.01
        param["incremental_pockets"][param["incremental_pockets"] > 0.99] = 0.99
        save_param_incremental[index, :] = param["incremental_pockets"]

        # We retain the sum of the errors as our minimization objective
        metrics[index] = metrics_is[index] + metrics_ib[index] + metrics_ih[index]

        # TODO: preserve parameter order?

        # We run the equilibrium again with updated values of
        # informal/backyard housing disamenity indices, then go to the next
        # iteration

        (initial_state_utility, initial_state_error,
         initial_state_simulated_jobs,
         initial_state_households_housing_types,
         initial_state_household_centers,
         initial_state_households, initial_state_dwelling_size,
         initial_state_housing_supply, initial_state_rent,
         initial_state_rent_matrix, initial_state_capital_land,
         initial_state_average_income, initial_state_limit_city
         ) = eqcmp.compute_equilibrium(
             fraction_capital_destroyed, amenities, param, housing_limit,
             population, households_per_income_class, total_RDP,
             coeff_land, income_net_of_commuting_costs, grid, options,
             agricultural_rent, interest_rate, number_properties_RDP,
             average_income, mean_income, income_class_by_housing_type,
             minimum_housing_supply, param["coeff_A"], income_baseline)

        iteration_number = index + 1

        print(f"iteration {iteration_number}/{number_total_iterations}")

# region
# We pick the set of parameters that minimize the sum of absolute diffs
# between data and simulation
score_min = np.min(metrics)
index_min = np.argmin(metrics)

print(score_min)
print(index_min)

# TODO: Enforce parameter order ex post?

# We update the parameter vector
param["informal_pockets"] = save_param_informal_settlements[index_min]
param["backyard_pockets"] = save_param_backyards[index_min]
param["incremental_pockets"] = save_param_incremental[index_min]

# We save values
np.save(path_precalc_inp + 'param_pockets.npy',
        param["informal_pockets"])
np.save(path_precalc_inp + 'param_backyards.npy',
        param["backyard_pockets"])
np.save(path_precalc_inp + 'param_incremental.npy',
        param["incremental_pockets"])
# endregion

# Can do better!

print(np.nanmin(param["informal_pockets"]))
print(np.nanmean(param["informal_pockets"]))
print(np.nanmax(param["informal_pockets"]))

print(np.nanmin(param["backyard_pockets"]))
print(np.nanmean(param["backyard_pockets"]))
print(np.nanmax(param["backyard_pockets"]))

print(np.nanmin(param["incremental_pockets"]))
print(np.nanmean(param["incremental_pockets"]))
print(np.nanmax(param["incremental_pockets"]))


# NB: we ensure interior solution

# param["init_util_levels"] = initial_state_utility

# np.save(path_precalc_inp + 'init_util_levels.npy',
#         param["init_util_levels"])

print(np.nansum(initial_state_households_housing_types,1))
print(housing_type_data)



