# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:57:30 2022.

@author: monni
"""

# %% Preamble

# IMPORT PACKAGES

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy

import inputs.parameters_and_options as inpprm
import inputs.data as inpdt
import equilibrium.functions_dynamic as eqdyn
import outputs.export_outputs as outexp
import outputs.flood_outputs as outfld
# import outputs.export_outputs_floods as outval


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/Data/'
path_precalc_inp = path_folder + 'precalculated_inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_data + 'Scenarios/'
path_outputs = path_code + '/Output/'
path_floods = path_folder + "flood_maps/"


# IMPORT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(
    path_precalc_inp, options)

# Set custom options for this simulation
#  Dummy for taking floods into account in the utility function
options["agents_anticipate_floods"] = 0
#  Dummy for preventing new informal settlement development
options["informal_land_constrained"] = 0

# More custom options regarding flood model
#  Dummy for taking pluvial floods into account (on top of fluvial floods)
options["pluvial"] = 1
#  Dummy for reducing pluvial risk for (better protected) formal structures
options["correct_pluvial"] = 1
#  Dummy for taking coastal floods into account (on top of fluvial floods)
options["coastal"] = 1
#  Digital elevation to be used with coastal flood data (MERITDEM or NASADEM)
#  NB: MERITDEM is also the DEM used for fluvial and pluvial flood data
options["dem"] = "MERITDEM"
#  We consider undefended flood maps as our default because they are more
#  reliable
options["defended"] = 0
#  Dummy for taking sea-level rise into account in coastal flood data
#  NB: Projections are up to 2050, based upon IPCC AR5 assessment for the
#  RCP 8.5 scenario for coastal (+ dummy scenarios for pluvial / fluvial)
options["climate_change"] = 0

# More custom options regarding scenarios
options["inc_ineq_scenario"] = 2
options["pop_growth_scenario"] = 3
options["fuel_price_scenario"] = 2

# Processing options for this simulation
options["convert_sp_data"] = 0

# TODO: Rewrite scenarios for paper without floods

options["urban_edge"] = 0


# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)

name = ('floods' + str(options["agents_anticipate_floods"])
        + str(options["informal_land_constrained"])
        + '_F' + str(options["defended"])
        + '_P' + str(options["pluvial"]) + str(options["correct_pluvial"])
        + '_C' + str(options["coastal"]) + str(options["climate_change"])
        + '_scenario' + str(options["inc_ineq_scenario"])
        + str(options["pop_growth_scenario"])
        + str(options["fuel_price_scenario"]))

path_plots = path_outputs + name + '/plots/'
path_tables = path_outputs + name + '/tables/'
path_plots_floods = path_plots + 'floods/'
path_tables_floods = path_tables + 'floods/'

year_temp = 0


# %% Load data

print("Import data")

# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
geo_grid = gpd.read_file(path_data + "grid_reference_500.shp")
geo_TAZ = gpd.read_file(path_data + "TAZ_ampp_prod_attr_2013_2032.shp")
amenities = inpdt.import_amenities(path_precalc_inp, options)


# MACRO DATA

(interest_rate, population, housing_type_data, total_RDP
 ) = inpdt.import_macro_data(param, path_scenarios, path_folder)


# HOUSEHOLDS AND INCOME DATA

(mean_income, households_per_income_class, average_income, income_mult,
 income_baseline, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_baseline,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0

# We convert income distribution data (at SP level) to grid dimensions for use
# in income calibration: long to run, uncomment only if needed

if options["convert_sp_data"] == 1:
    print("Convert SP data to grid dimensions - start")
    income_distribution_grid = inpdt.convert_income_distribution(
        income_distribution, grid, path_data, data_sp)
    print("Convert SP data to grid dimensions - end")

income_distribution_grid = np.load(path_data + "income_distrib_grid.npy")


# LAND USE PROJECTIONS

(spline_RDP, spline_estimate_RDP, spline_land_RDP,
 spline_land_backyard, spline_land_informal, spline_land_constraints,
 number_properties_RDP) = (
     inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                           housing_type_data, path_data, path_folder)
     )

#  We correct areas for each housing type at baseline year for the amount of
#  constructible land in each type
coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, 0)

#  We update parameter vector with construction parameters
(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_baseline, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )

# OTHER VALIDATION DATA

data = scipy.io.loadmat(path_precalc_inp + 'data.mat')['data']
data_avg_income = data['gridAverageIncome'][0][0].squeeze()
data_avg_income[np.isnan(data_avg_income)] = 0

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
cal_average_income = np.load(
    path_precalc_transp + 'GRID_averageIncome_0.npy')

# LOAD EQUILIBRIUM DATA

initial_state_utility = np.load(
    path_outputs + name + '/initial_state_utility.npy')
initial_state_error = np.load(
    path_outputs + name + '/initial_state_error.npy')
initial_state_simulated_jobs = np.load(
    path_outputs + name + '/initial_state_simulated_jobs.npy')
initial_state_households_housing_types = np.load(
    path_outputs + name + '/initial_state_households_housing_types.npy')
initial_state_household_centers = np.load(
    path_outputs + name + '/initial_state_household_centers.npy')
initial_state_households = np.load(
    path_outputs + name + '/initial_state_households.npy')
initial_state_dwelling_size = np.load(
    path_outputs + name + '/initial_state_dwelling_size.npy')
initial_state_housing_supply = np.load(
    path_outputs + name + '/initial_state_housing_supply.npy')
initial_state_rent = np.load(
    path_outputs + name + '/initial_state_rent.npy')
initial_state_rent_matrix = np.load(
    path_outputs + name + '/initial_state_rent_matrix.npy')
initial_state_capital_land = np.load(
    path_outputs + name + '/initial_state_capital_land.npy')
initial_state_average_income = np.load(
    path_outputs + name + '/initial_state_average_income.npy')
initial_state_limit_city = np.load(
    path_outputs + name + '/initial_state_limit_city.npy')


# LOAD FLOOD DATA

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

# SCENARIOS

(spline_agricultural_price, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_baseline, param, grid, path_scenarios,
                            options)

# NUMBER OF HOUSEHOLDS

sim_nb_households_tot = np.nansum(initial_state_households_housing_types, 0)
data_nb_households_tot = np.nansum(housing_types[
    ["informal_grid", "backyard_informal_grid", "formal_grid"]
    ], 1)

sim_nb_households_formal = initial_state_households_housing_types[0, :]
data_nb_households_formal = (housing_types["formal_grid"]
                             - initial_state_households_housing_types[3, :])
sim_nb_households_backyard = initial_state_households_housing_types[1, :]
data_nb_households_backyard = housing_types["backyard_informal_grid"]
sim_nb_households_informal = initial_state_households_housing_types[2, :]
data_nb_households_informal = housing_types["informal_grid"]
data_nb_households_rdp = initial_state_households_housing_types[3, :]

sim_nb_households_poor = initial_state_household_centers[0, :]
sim_nb_households_midpoor = initial_state_household_centers[1, :]
sim_nb_households_midrich = initial_state_household_centers[2, :]
sim_nb_households_rich = initial_state_household_centers[3, :]

# FLOOD TYPES

fluviald_floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr',
                   'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
fluvialu_floods = ['FU_5yr', 'FU_10yr', 'FU_20yr', 'FU_50yr', 'FU_75yr',
                   'FU_100yr', 'FU_200yr', 'FU_250yr', 'FU_500yr', 'FU_1000yr']
pluvial_floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr',
                  'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
coastal_floods = ['C_MERITDEM_' + str(options['climate_change']) + '_0000',
                  'C_MERITDEM_' + str(options['climate_change']) + '_0002',
                  'C_MERITDEM_' + str(options['climate_change']) + '_0005',
                  'C_MERITDEM_' + str(options['climate_change']) + '_0010',
                  'C_MERITDEM_' + str(options['climate_change']) + '_0025',
                  'C_MERITDEM_' + str(options['climate_change']) + '_0050',
                  'C_MERITDEM_' + str(options['climate_change']) + '_0100',
                  'C_MERITDEM_' + str(options['climate_change']) + '_0250']

# WE CREATE DIRECTORIES TO STORE OUTPUTS (IF NEEDED)

try:
    os.mkdir(path_plots)
except OSError as error:
    print(error)

try:
    os.mkdir(path_tables)
except OSError as error:
    print(error)

try:
    os.mkdir(path_plots_floods)
except OSError as error:
    print(error)

try:
    os.mkdir(path_tables_floods)
except OSError as error:
    print(error)


# %% Validation: draw maps and figures

print("Static equilibrium validation")

# POPULATION OUTPUTS

# Note that aggregate fit on income groups hold by construction
# Aggregate (and local) fit on housing types is enforced through
# disamenity parameter calibration but is not perfect,
# hence needs to be checked
# NB: through the disamenity parameter, population has in fact been fitted in
# informal backyards and settlements
agg_housing_type_valid = outexp.valid_pop_housing_type(
    initial_state_households_housing_types, housing_type_data,
    'Simulation', 'Data', path_plots, path_tables
    )

# We also validate the fit across housing types and income groups
(agg_FP_income_valid, agg_IB_income_valid, agg_IS_income_valid
 ) = outexp.valid_pop_htype_income(
     initial_state_households, households_per_income_and_housing, 'Simulation',
     'Data', path_plots, path_tables)


#  IN ONE DIMENSION

# Now, we validate overall households density across space
dens_valid_1d = outexp.validation_density(
    grid, initial_state_households_housing_types, housing_types,
    path_plots, path_tables)

# We do the same for total number of households across space,
# housing types and income groups
dist_HH_per_housing_1d = outexp.valid_pop_housing_types(
    grid, initial_state_households_housing_types, housing_types,
    path_plots, path_tables
    )
dist_HH_per_income_1d = outexp.valid_pop_income_groups(
    grid, initial_state_household_centers, income_distribution_grid,
    path_plots, path_tables
    )

# We also plot income groups across space (in 1D) for each housing type,
# even if we cannot validate such output
(dist_HH_per_housing_and_income_1d
 ) = outexp.simul_pop_htype_income(
     grid, initial_state_households, path_plots, path_tables)


#  IN TWO DIMENSIONS

#  Before all, it should be noted that all validation data is disaggregated
#  from SAL level, which is granular enough, but still 5 times coarser than
#  the grid-cell level of analysis: if anything, pixels should be compared
#  in blocks

#  For overall households

total_sim = outexp.export_map(
    sim_nb_households_tot, grid, geo_grid, path_plots,  'total_sim',
    "Total number of households, up to 99.99% quantile (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_tot, 0.9999))
total_data = outexp.export_map(
    data_nb_households_tot, grid, geo_grid, path_plots,  'total_data',
    "Total number of households, up to 99.99% quantile (data)",
    path_tables,
    ubnd=np.nanquantile(data_nb_households_tot, 0.9999))

#  Per housing type

formal_sim = outexp.export_map(
    sim_nb_households_formal, grid, geo_grid, path_plots,  'formal_sim',
    "Number of households in formal private, up to 99.99% quantile"
    + " (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_formal, 0.9999))
formal_data = outexp.export_map(
    data_nb_households_formal, grid, geo_grid, path_plots,  'formal_data',
    "Number of households in formal private, up to 99.99% quantile (data)",
    path_tables,
    ubnd=np.nanquantile(data_nb_households_formal, 0.9999))

backyard_sim = outexp.export_map(
    sim_nb_households_backyard, grid, geo_grid, path_plots,  'backyard_sim',
    "Number of households in informal backyards, up to 99.99% quantile"
    + " (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_backyard, 0.9999))
backyard_data = outexp.export_map(
    data_nb_households_backyard, grid, geo_grid, path_plots,  'backyard_data',
    "Number of households in informal backyards, up to 99.99% quantile (data)",
    path_tables,
    ubnd=np.nanquantile(data_nb_households_backyard, 0.9999))

informal_sim = outexp.export_map(
    sim_nb_households_informal, grid, geo_grid, path_plots,  'informal_sim',
    "Number of households in informal settlements, up to 99.99% quantile"
    + " (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_informal, 0.9999))
informal_data = outexp.export_map(
    data_nb_households_informal, grid, geo_grid, path_plots,  'informal_data',
    "Number of households in informal settlements, up to 99.99% quantile"
    + " (data)",
    path_tables,
    ubnd=np.nanquantile(data_nb_households_informal, 0.9999))

# Note that there is no difference between simulation and validation data
# for RDP housing (purely exogenous)
rdp_sim = outexp.export_map(
    data_nb_households_rdp, grid, geo_grid, path_plots,  'rdp_sim',
    "Number of households in formal subsidized, up to 99.99% quantile (data)",
    path_tables,
    ubnd=np.nanquantile(data_nb_households_rdp, 0.9999))

#  Per income group
#  NB: validation data is set at the SP level, which is less granular than
#  SAL level (25x coarser than grid-cell level). We therefore do not plot it
#  to prevent missinterpretation

poor_sim = outexp.export_map(
    sim_nb_households_poor, grid, geo_grid, path_plots,  'poor_sim',
    "Number of poor households, up to 99.99% quantile (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_poor, 0.9999))
midpoor_sim = outexp.export_map(
    sim_nb_households_midpoor, grid, geo_grid, path_plots,  'midpoor_sim',
    "Number of mid-poor households, up to 99.99% quantile (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_midpoor, 0.9999))
midrich_sim = outexp.export_map(
    sim_nb_households_midrich, grid, geo_grid, path_plots,  'midrich_sim',
    "Number of mid-rich households, up to 99.99% quantile (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_midrich, 0.9999))
rich_sim = outexp.export_map(
    sim_nb_households_rich, grid, geo_grid, path_plots,  'rich_sim',
    "Number of rich households, up to 99.99% quantile (simulation)",
    path_tables,
    ubnd=np.nanquantile(sim_nb_households_rich, 0.9999))


# %% HOUSING SUPPLY OUTPUTS

# By plotting the housing supply per unit of available land, we may check
# whether the bell-shaped curve of urban development holds
avg_hsupply_1d = outexp.valid_housing_supply(
    grid, initial_state_housing_supply, path_plots, path_tables)

# We now consider overall land to recover building density
# NB: plotting the housing supply per unit of available land would be hard to
# interpret since we do not know the exact area that is available within each
# pixel, hence grid cells would not be comparable between each other
housing_supply = initial_state_housing_supply * coeff_land * 0.25
hsupply_noland_1d = outexp.valid_housing_supply_noland(
    grid, housing_supply, path_plots, path_tables)

hsupply_tot = np.nansum(housing_supply, 0)
hsupply_2d_sim = outexp.export_map(
    hsupply_tot, grid, geo_grid, path_plots,  'hsupply_2d_sim',
    "Total housing supply (in m²), up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(hsupply_tot, 0.9999))

# We also define a floor-area ratio with respect to total pixel area (not
# only available land), again for comparability issues
# FAR = np.nansum(housing_supply, 0) / (0.25 * 1000000)
# FAR_2d_sim = outexp.export_map(
#     FAR, grid, geo_grid, path_plots,  'FAR_2d_sim',
#     "Overall floor-area ratio, up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(FAR, 0.9999))

# We repeat the procedure across housing types

hsupply_formal = housing_supply[0, :]
hsupply_formal_2d_sim = outexp.export_map(
    hsupply_formal, grid, geo_grid, path_plots,  'hsupply_formal_2d_sim',
    "Total housing supply in formal private (in m²), up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(hsupply_formal, 0.9999))
# FAR_formal = housing_supply[0, :] / (0.25 * 1000000)
# FAR_formal_2d_sim = outexp.export_map(
#     FAR_formal, grid, geo_grid, path_plots,  'FAR_formal_2d_sim',
#     "Floor-area ratio in formal private, up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(FAR_formal, 0.9999))

# We also compute an habitable floor-area ratio (with respect to available,
# and not total pixel land) for formal private housing, since we have some
# validation data on that dimension
# sim_HFA_dens_formal = initial_state_housing_supply[0, :] / 1000000
# HFA_dens_formal_2d_sim = outexp.export_map(
#     sim_HFA_dens_formal, grid, geo_grid, path_plots,
#     'HFA_dens_formal_2d_sim',
#     "Formal private HFA ratio, up to 99.99% quantile (simulation)",
#     path_tables,
#     ubnd=np.nanquantile(sim_HFA_dens_formal, 0.9999))
# grid_formal_density_HFA[np.isnan(grid_formal_density_HFA)] = 0
# data_HFA_dens_formal = grid_formal_density_HFA
# HFA_dens_formal_2d_data = outexp.export_map(
#     data_HFA_dens_formal, grid, geo_grid,
#     path_plots, 'HFA_dens_formal_2d_data',
#     "Formal private HFA ratio, up to 99.99% quantile (data)",
#     path_tables,
#     ubnd=np.nanquantile(data_HFA_dens_formal, 0.9999))

hsupply_backyard = housing_supply[1, :]
hsupply_backyard_2d_sim = outexp.export_map(
    hsupply_backyard, grid, geo_grid, path_plots, 'hsupply_backyard_2d_sim',
    "Total housing supply in informal backyards (in m²),"
    + " up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(hsupply_backyard, 0.9999))
# FAR_backyard = housing_supply[1, :] / (0.25 * 1000000)
# FAR_backyard_2d_sim = outexp.export_map(
#     FAR_backyard, grid, geo_grid, path_plots, 'FAR_backyard_2d_sim',
#     "Floor-area ratio in informal backyards, up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(FAR_backyard, 0.9999))

hsupply_informal = housing_supply[2, :]
hsupply_informal_2d_sim = outexp.export_map(
    hsupply_informal, grid, geo_grid, path_plots, 'hsupply_informal_2d_sim',
    "Total housing supply in informal settlements (in m²),"
    + " up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(hsupply_informal, 0.9999))
# FAR_informal = housing_supply[2, :] / (0.25 * 1000000)
# FAR_informal_2d_sim = outexp.export_map(
#     FAR_informal, grid, geo_grid, path_plots, 'FAR_informal_2d_sim',
#     "Floor-area ratio in informal settlements, up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(FAR_informal, 0.9999))

hsupply_rdp = housing_supply[3, :]
hsupply_rdp_2d_sim = outexp.export_map(
    hsupply_rdp, grid, geo_grid, path_plots, 'hsupply_rdp_2d_sim',
    "Total housing supply in formal subsidized (in m²), up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(hsupply_rdp, 0.9999))
# FAR_rdp = housing_supply[3, :] / (0.25 * 1000000)
# FAR_rdp_2d_sim = outexp.export_map(
#     FAR_rdp, grid, geo_grid, path_plots, 'FAR_rdp_2d_sim',
#     "Floor-area ratio in formal subsidized, up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(FAR_rdp, 0.9999))

# NB: Since we do not know surface of built land (just of available land),
# we cannot compute building heigth based on dwelling size, since housing
# could be more or less spread out across available land area


# %% HOUSING PRICE OUTPUTS

# First in one dimension
land_price_1d = outexp.simulation_housing_price(
    grid, initial_state_rent, interest_rate, param, center,
    housing_types_sp, path_plots, path_tables, land_price=1)
housing_rent_1d = outexp.simulation_housing_price(
    grid, initial_state_rent, interest_rate, param, center,
    housing_types_sp, path_plots, path_tables, land_price=0)

# NB: check underlying validity/quality of validation data
# housing_price_1d = outexp.valid_housing_price(
#     grid, initial_state_rent, interest_rate, param,
#     housing_types_sp, data_sp,
#     path_plots, path_tables)

# Then in two dimensions
# NB: same remark as before holds for SP-level validation data
# Also note that housing price could just be displayed as the capitalized flow
# of future rents (that is, dividing annual rent by the interest rate)

rent_formal_simul = pd.DataFrame(initial_state_rent[0, :])
rent_formal_simul.loc[sim_nb_households_formal == 0] = 0
housing_price_formal_2d_sim = outexp.export_map(
    rent_formal_simul, grid, geo_grid, path_plots,  'rent_formal_2d_sim',
    "Annual housing rents/m², up to 99.99% quantile (formal private)",
    path_tables,
    ubnd=np.nanquantile(rent_formal_simul, 0.9999))

rent_backyard_simul = pd.DataFrame(initial_state_rent[1, :])
rent_backyard_simul.loc[sim_nb_households_backyard == 0] = 0
housing_price_backyard_2d_sim = outexp.export_map(
    rent_backyard_simul, grid, geo_grid, path_plots,  'rent_backyard_2d_sim',
    "Annual land rents/m², up to 99.99% quantile"
    + " (informal backyards)",
    path_tables,
    ubnd=np.nanquantile(rent_backyard_simul, 0.9999))

rent_informal_simul = pd.DataFrame(initial_state_rent[2, :])
rent_informal_simul.loc[sim_nb_households_informal == 0] = 0
housing_price_informal_2d_sim = outexp.export_map(
    rent_informal_simul, grid, geo_grid, path_plots,  'rent_informal_2d_sim',
    "Annual land rents/m², up to 99.99% quantile"
    + " (informal settlements)",
    path_tables,
    ubnd=np.nanquantile(rent_informal_simul, 0.9999))

land_price = (
    (initial_state_rent[0, :] * param["coeff_A"])
    ** (1 / param["coeff_a"])
    * param["coeff_a"]
    * (param["coeff_b"] / (interest_rate + param["depreciation_rate"]))
    ** (param["coeff_b"] / param["coeff_a"])
    / interest_rate
    )

landprice_formal_simul = pd.DataFrame(land_price)
landprice_formal_simul.loc[sim_nb_households_formal == 0] = 0
land_price_formal_2d_sim = outexp.export_map(
    landprice_formal_simul, grid, geo_grid,
    path_plots, 'landprice_formal_2d_sim',
    "Land prices/m², up to 99.99% quantile (formal private)",
    path_tables,
    ubnd=np.nanquantile(landprice_formal_simul, 0.9999))


# %% DWELLING SIZE OUTPUTS

# We only plot formal private housing since this is the only segment where
# dwelling size is allowed to vary.
# NB: same remark as before for SP-level validation data

# Note that we start getting a lot of nan values around 30km, hence the
# imprecise nature of validation data beyond this distance to the center
dwelling_size_1d = outexp.valid_housing_demand(
    grid, center, initial_state_dwelling_size,
    initial_state_households_housing_types,
    housing_types_sp, data_sp,
    path_plots, path_tables)

formal_dwelling_size = pd.DataFrame(initial_state_dwelling_size[0, :])
formal_dwelling_size.loc[sim_nb_households_formal == 0] = 0

dwelling_size_2d = outexp.export_map(
    formal_dwelling_size, grid, geo_grid,
    path_plots, 'formal_dwellingsize_2d_sim',
    "Simulated dwelling sizes in m², up to 99.99% quantile "
    + "(formal private)",
    path_tables,
    ubnd=np.nanquantile(formal_dwelling_size, 0.9999))


# %% TRANSPORT OUTPUTS

#  Income net of commuting costs

netincome_poor = pd.DataFrame(income_net_of_commuting_costs[0, :])
netincome_poor.loc[sim_nb_households_poor == 0] = 0

netincome_poor_2d_sim = outexp.export_map(
    netincome_poor, grid, geo_grid, path_plots, 'netincome_poor_2d_sim',
    "Theoretical income net of commuting costs for the poor (rands, 2011),"
    + " up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(netincome_poor, 0.9999),
    lbnd=np.nanmin(netincome_poor[netincome_poor > 0]))

netincome_midpoor = pd.DataFrame(income_net_of_commuting_costs[1, :])
netincome_midpoor.loc[sim_nb_households_midpoor == 0] = 0

netincome_midpoor_2d_sim = outexp.export_map(
    netincome_midpoor, grid, geo_grid, path_plots, 'netincome_midpoor_2d_sim',
    "Theoretical income net of commuting costs for the mid-poor (rands, 2011),"
    + " up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(netincome_midpoor, 0.9999),
    lbnd=np.nanmin(netincome_midpoor[netincome_midpoor > 0]))

netincome_midrich = pd.DataFrame(income_net_of_commuting_costs[2, :])
netincome_midrich.loc[sim_nb_households_midrich == 0] = 0

netincome_midrich_2d_sim = outexp.export_map(
    netincome_midrich, grid, geo_grid, path_plots, 'netincome_midrich_2d_sim',
    "Theoretical income net of commuting costs for the mid-rich (rands, 2011),"
    + " up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(netincome_midrich, 0.9999),
    lbnd=np.nanmin(netincome_midrich[netincome_midrich > 0]))

netincome_rich = pd.DataFrame(income_net_of_commuting_costs[3, :])
netincome_rich.loc[sim_nb_households_rich == 0] = 0

netincome_rich_2d_sim = outexp.export_map(
    netincome_rich, grid, geo_grid, path_plots, 'netincome_rich_2d_sim',
    "Theoretical income net of commuting costs for the rich (rands, 2011),"
    + " up to 99.99% quantile",
    path_tables,
    ubnd=np.nanquantile(netincome_rich, 0.9999),
    lbnd=np.nanmin(netincome_rich[netincome_rich > 0]))

# (avg_income_net_of_commuting_1d
#  ) = outexp.plot_income_net_of_commuting_costs(
#      grid, income_net_of_commuting_costs, path_plots, path_tables)


#  Average income

# avgincome_poor = cal_average_income[0, :]
# avgincome_poor_2d_sim = outexp.export_map(
#     avgincome_poor, grid, geo_grid, path_plots, 'avgincome_poor_2d_sim',
#     "Annual income for the poor (rands, 2011), up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(avgincome_poor, 0.9999),
#     lbnd=np.nanmin(avgincome_poor))
# avgincome_midpoor = cal_average_income[1, :]
# avgincome_midpoor_2d_sim = outexp.export_map(
#     avgincome_midpoor, grid, geo_grid, path_plots,
#     'avgincome_midpoor_2d_sim',
#     "Annual income for the mid-poor (rands, 2011), up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(avgincome_midpoor, 0.9999),
#     lbnd=np.nanmin(avgincome_midpoor))
# avgincome_midrich = cal_average_income[2, :]
# avgincome_midrich_2d_sim = outexp.export_map(
#     avgincome_midrich, grid, geo_grid, path_plots,
#     'avgincome_midrich_2d_sim',
#     "Annual income for the mid-rich (rands, 2011), up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(avgincome_midrich, 0.9999),
#     lbnd=np.nanmin(avgincome_midrich))
# avgincome_rich = cal_average_income[3, :]
# avgincome_rich_2d_sim = outexp.export_map(
#     avgincome_rich, grid, geo_grid, path_plots, 'avgincome_rich_2d_sim',
#     "Annual income for the rich (rands, 2011), up to 99.99% quantile",
#     path_tables,
#     ubnd=np.nanquantile(avgincome_rich, 0.9999),
#     lbnd=np.nanmin(avgincome_rich))

# (avg_income_1d
#  ) = outexp.plot_average_income(
#      grid, cal_average_income, path_plots, path_tables)


# %% FLOOD OUPUTS

# We commented this section out as it was made redundant with the use case
# scripts

# We start with aggregate flood exposure validation

# NB: evolution is not necessarily monotonous on the short run because of
# some decreasing flood depths (never proportion of flood-prone area)
# Also note that we focus on fluvial undefended maps, to compare with other
# flood maps

# stats_fluvialu_per_housing_data = outfld.compute_stats_per_housing_type(
#     fluvialu_floods, path_floods, data_nb_households_formal,
#     data_nb_households_rdp, data_nb_households_informal,
#     data_nb_households_backyard, path_tables_floods, 'fluvialu_data')
# stats_fluvialu_per_housing_sim = outfld.compute_stats_per_housing_type(
#     fluvialu_floods, path_floods, sim_nb_households_formal,
#     data_nb_households_rdp,
#     sim_nb_households_informal,
#     sim_nb_households_backyard,
#     path_tables_floods, 'fluvialu_sim')
# outval.validation_flood(
#     stats_fluvialu_per_housing_data, stats_fluvialu_per_housing_sim,
#     'Data', 'Simul', 'fluvialu', path_plots_floods)

# stats_pluvial_per_housing_data = outfld.compute_stats_per_housing_type(
#     pluvial_floods, path_floods, data_nb_households_formal,
#     data_nb_households_rdp, data_nb_households_informal,
#     data_nb_households_backyard, path_tables_floods, 'pluvial_data')
# stats_pluvial_per_housing_sim = outfld.compute_stats_per_housing_type(
#     pluvial_floods, path_floods, sim_nb_households_formal,
#     data_nb_households_rdp,
#     sim_nb_households_informal,
#     sim_nb_households_backyard,
#     path_tables_floods, 'pluvial_sim')
# outval.validation_flood(
#     stats_pluvial_per_housing_data, stats_pluvial_per_housing_sim,
#     'Data', 'Simul', 'pluvial', path_plots_floods)

# stats_coastal_per_housing_data = outfld.compute_stats_per_housing_type(
#     coastal_floods, path_floods, data_nb_households_formal,
#     data_nb_households_rdp, data_nb_households_informal,
#     data_nb_households_backyard, path_tables_floods, 'coastal_data')
# stats_coastal_per_housing_sim = outfld.compute_stats_per_housing_type(
#     coastal_floods, path_floods, sim_nb_households_formal,
#     data_nb_households_rdp,
#     sim_nb_households_informal,
#     sim_nb_households_backyard,
#     path_tables_floods, 'coastal_sim')
# outval.validation_flood_coastal(
#     stats_coastal_per_housing_data, stats_coastal_per_housing_sim,
#     'Data', 'Simul', 'coastal', path_plots_floods)

# We also store the relevant household spatial distribution across flood maps
# for 2D vizualisation
# NB: We do not store household spatial distribution from validation data
# as this would be redundant with prior validation exercises

# fluvialu_floods_dict = outfld.create_flood_dict(
#     fluvialu_floods, path_floods, path_tables_floods,
#     sim_nb_households_poor, sim_nb_households_midpoor,
#     sim_nb_households_midrich, sim_nb_households_rich)
# pluvial_floods_dict = outfld.create_flood_dict(
#     pluvial_floods, path_floods, path_tables_floods,
#     sim_nb_households_poor, sim_nb_households_midpoor,
#     sim_nb_households_midrich, sim_nb_households_rich)
# coastal_floods_dict = outfld.create_flood_dict(
#     coastal_floods, path_floods, path_tables_floods,
#     sim_nb_households_poor, sim_nb_households_midpoor,
#     sim_nb_households_midrich, sim_nb_households_rich)

# Finally, we plot flood severity distribution across 3 selected return periods

# barWidth = 0.1
# transparency = [1, 0.5, 0.25]

# outval.plot_flood_severity_distrib(barWidth, transparency,
#                                    fluvialu_floods_dict, 'FU',
#                                    path_plots_floods, ylim=15000)
# outval.plot_flood_severity_distrib(barWidth, transparency,
#                                    pluvial_floods_dict, 'P',
#                                    path_plots_floods, ylim=90000)
# outval.plot_flood_severity_distrib(barWidth, transparency,
#                                    coastal_floods_dict, 'C_MERITDEM_1',
#                                    path_plots_floods, ylim=1000)


# %% FLOOD DAMAGES

# We first compute calibrated content damages across housing types, and
# damages to formal private structure, as they depend on model outcomes

# NB: We get damages per housing type for one representative household!
content_cost = outfld.compute_content_cost(
    initial_state_households, initial_state_housing_supply,
    income_net_of_commuting_costs, param,
    fraction_capital_destroyed, initial_state_rent,
    initial_state_dwelling_size, interest_rate)

# NB: note that capital is in monetary values
formal_structure_cost = outfld.compute_formal_structure_cost(
        initial_state_capital_land, initial_state_households_housing_types,
        coeff_land)

# Then we run the tables for aggregate damages
# NB: This intrinsically depends on housing types, as the calibrated fraction
# of capital destroyed as a function of maximum flood depth depends on building
# materials used

# NB: we do not use fraction_capital_destroyed directly in the computations
# as we want separate plots for each flood risk, and fraction_capital_destroyed
# already takes into account the maximum across each annualized depreciation
# term

# We re-import flood data to compute ex-post damages in case agents do not
# anticipate floods

(fraction_capital_destroyed, structural_damages_small_houses,
 structural_damages_medium_houses, structural_damages_large_houses,
 content_damages, structural_damages_type1, structural_damages_type2,
 structural_damages_type3a, structural_damages_type3b,
 structural_damages_type4a, structural_damages_type4b
 ) = inpdt.import_full_floods_data(options, param, path_folder)

fluvialu_damages_data = outfld.compute_damages(
    fluvialu_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'fluvialu_data')
fluvialu_damages_sim = outfld.compute_damages(
    fluvialu_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'fluvialu_sim')

pluvial_damages_data = outfld.compute_damages(
    pluvial_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'pluvial_data')
pluvial_damages_sim = outfld.compute_damages(
    pluvial_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'pluvial_sim')

coastal_damages_data = outfld.compute_damages(
    coastal_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'coastal_data')
coastal_damages_sim = outfld.compute_damages(
    coastal_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'coastal_sim')


# We get aggregate validation graphs (deprecated)

# outval.valid_damages(
#     fluvialu_damages_sim, fluvialu_damages_data,
#     path_plots_floods, 'fluvialu', options)
# outval.valid_damages(
#     pluvial_damages_sim, pluvial_damages_data,
#     path_plots_floods, 'pluvial', options)
# outval.valid_damages(
#     coastal_damages_sim, coastal_damages_data,
#     path_plots_floods, 'coastal', options)


# Now in two dimensions

# Note that we do not plot damages associated with spatial household
# distribution from validation data for two reasons. First, since we do not
# have direct validation data on flood damages, we do not consider those plots
# as a strong validation exercise. Then, given that raw flood maps are defined
# at an even more granular level than our analysis grid (between 30 and 100m),
# we do not want to further increase potential measurement error by merging it
# with a map defined at the coarser SAL level

# We first obtain the granular damage tables

fluvialu_damages_2d_sim = outfld.compute_damages_2d(
    fluvialu_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'fluvialu_sim')

pluvial_damages_2d_sim = outfld.compute_damages_2d(
    pluvial_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'pluvial_sim')

coastal_damages_2d_sim = outfld.compute_damages_2d(
    coastal_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables_floods, 'coastal_sim')

# Hence the maps

fluvialu_damages_2d_sim_stacked = np.stack(
    [df for df in fluvialu_damages_2d_sim.values()])
fluvialu_formal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_formal_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 0],
        'fluvialu', 'formal', options)
fluvialu_subsidized_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_subsidized_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 1],
        'fluvialu', 'subsidized', options)
fluvialu_informal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_informal_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 2],
        'fluvialu', 'informal', options)
fluvialu_backyard_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_backyard_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 3],
        'fluvialu', 'backyard', options)
fluvialu_formal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_formal_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 4],
        'fluvialu', 'formal', options)
fluvialu_informal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_informal_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 5],
        'fluvialu', 'informal', options)
fluvialu_backyard_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_backyard_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 6],
        'fluvialu', 'backyard', options)
fluvialu_subsidized_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_subsidized_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 7],
        'fluvialu', 'subsidized', options)

pluvial_damages_2d_sim_stacked = np.stack(
    [df for df in pluvial_damages_2d_sim.values()])
pluvial_formal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_formal_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 0],
        'pluvial', 'formal', options)
pluvial_subsidized_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_subsidized_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 1],
        'pluvial', 'subsidized', options)
pluvial_informal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_informal_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 2],
        'pluvial', 'informal', options)
pluvial_backyard_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_backyard_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 3],
        'pluvial', 'backyard', options)
pluvial_formal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_formal_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 4],
        'pluvial', 'formal', options)
pluvial_informal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_informal_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 5],
        'pluvial', 'informal', options)
pluvial_backyard_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_backyard_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 6],
        'pluvial', 'backyard', options)
pluvial_subsidized_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_subsidized_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 7],
        'pluvial', 'subsidized', options)

coastal_damages_2d_sim_stacked = np.stack(
    [df for df in coastal_damages_2d_sim.values()])
coastal_formal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_formal_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 0],
        'coastal', 'formal', options)
coastal_subsidized_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_subsidized_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 1],
        'coastal', 'subsidized', options)
coastal_informal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_informal_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 2],
        'coastal', 'informal', options)
coastal_backyard_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_backyard_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 3],
        'coastal', 'backyard', options)
coastal_formal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_formal_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 4],
        'coastal', 'formal', options)
coastal_informal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_informal_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 5],
        'coastal', 'informal', options)
coastal_backyard_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_backyard_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 6],
        'coastal', 'backyard', options)
coastal_subsidized_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_subsidized_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 7],
        'coastal', 'subsidized', options)


# We define lists of relevant damage tables for use in subsequent loops

list_sim = [
    fluvialu_backyard_structure_2d_sim,
    fluvialu_backyard_content_2d_sim,
    fluvialu_subsidized_structure_2d_sim,
    fluvialu_subsidized_content_2d_sim,
    fluvialu_informal_structure_2d_sim,
    fluvialu_informal_content_2d_sim,
    fluvialu_formal_structure_2d_sim,
    fluvialu_formal_content_2d_sim,
    pluvial_backyard_structure_2d_sim,
    pluvial_backyard_content_2d_sim,
    pluvial_subsidized_structure_2d_sim,
    pluvial_subsidized_content_2d_sim,
    pluvial_informal_structure_2d_sim,
    pluvial_informal_content_2d_sim,
    pluvial_formal_structure_2d_sim,
    pluvial_formal_content_2d_sim,
    coastal_backyard_structure_2d_sim,
    coastal_backyard_content_2d_sim,
    coastal_subsidized_structure_2d_sim,
    coastal_subsidized_content_2d_sim,
    coastal_informal_structure_2d_sim,
    coastal_informal_content_2d_sim,
    coastal_formal_structure_2d_sim,
    coastal_formal_content_2d_sim]

list_sim_formal = [
    fluvialu_formal_structure_2d_sim,
    fluvialu_formal_content_2d_sim,
    pluvial_formal_structure_2d_sim,
    pluvial_formal_content_2d_sim,
    coastal_formal_structure_2d_sim,
    coastal_formal_content_2d_sim]
list_sim_subsidized = [
    fluvialu_subsidized_structure_2d_sim,
    fluvialu_subsidized_content_2d_sim,
    pluvial_subsidized_structure_2d_sim,
    pluvial_subsidized_content_2d_sim,
    coastal_subsidized_structure_2d_sim,
    coastal_subsidized_content_2d_sim]
list_sim_informal = [
    fluvialu_informal_structure_2d_sim,
    fluvialu_informal_content_2d_sim,
    pluvial_informal_structure_2d_sim,
    pluvial_informal_content_2d_sim,
    coastal_informal_structure_2d_sim,
    coastal_informal_content_2d_sim]
list_sim_backyard = [
    fluvialu_backyard_structure_2d_sim,
    fluvialu_backyard_content_2d_sim,
    pluvial_backyard_structure_2d_sim,
    pluvial_backyard_content_2d_sim,
    coastal_backyard_structure_2d_sim,
    coastal_backyard_content_2d_sim]

# We get the damage maps in absolute values

# NB: coastal damage maps in informal settlements are actually empty
for item in list_sim:
    outexp.export_map(item, grid, geo_grid,
                      path_plots_floods, outexp.retrieve_name(item, -1),
                      "Flood damages (rands, 2011), up to 99.99% quantile",
                      path_tables_floods,
                      ubnd=np.nanquantile(item[item > 0], 0.9999),
                      lbnd=np.nanmin(item))


# Now, let us display economic damages as a share of representative household
# income, to be clearer about the welfare impacts

# We first define a dominant income group in each housing type per location.
# Then, we divide total damages in a given location by the TOTAL number of
# households in the area, and express this quantity as a share of the dominant
# group income: it basically refers to what each household would need to pay
# if only the dominant income group populated the area

# Also note that those outcomes should be interpreted with caution, given that
# net incomes are not the only welfare component in the model. Besides,
# we can express the cost of structural damages as a % of net income in
# formal private areas, but households do not actually bear those costs, which
# are taken into account by developers.

# Apart from setting income equal to nan when there is nobody, this yields
# the same as selected_net_income formulas

income_formal = (
    np.nansum(income_net_of_commuting_costs
              * initial_state_households[0, :, :], 0)
    / np.nansum(initial_state_households[0, :, :], 0))

income_backyard = (
    np.nansum(income_net_of_commuting_costs
              * initial_state_households[1, :, :], 0)
    / np.nansum(initial_state_households[1, :, :], 0))

income_informal = (
    np.nansum(income_net_of_commuting_costs
              * initial_state_households[2, :, :], 0)
    / np.nansum(initial_state_households[2, :, :], 0))

income_subsidized = (
    np.nansum(income_net_of_commuting_costs
              * initial_state_households[3, :, :], 0)
    / np.nansum(initial_state_households[3, :, :], 0))

for item in list_sim_subsidized:
    new_item = (item
                / income_subsidized
                / initial_state_households_housing_types[3, :])
    new_item[np.isnan(new_item)] = 0
    outexp.export_map(
        new_item, grid, geo_grid, path_plots_floods,
        outexp.retrieve_name(item, -1) + '_shareinc',
        "Flood damages (% net income share), up to 99.99% quantile",
        path_tables_floods, ubnd=np.nanquantile(new_item, 0.9999))

for item in list_sim_backyard:
    new_item = (item
                / income_backyard
                / initial_state_households_housing_types[1, :])
    new_item[np.isnan(new_item)] = 0
    outexp.export_map(
        new_item, grid, geo_grid, path_plots_floods,
        outexp.retrieve_name(item, -1) + '_shareinc',
        "Flood damages (% net income share), up to 99.99% quantile",
        path_tables_floods, ubnd=np.nanquantile(new_item, 0.9999))

for item in list_sim_informal:
    new_item = (item
                / income_informal
                / initial_state_households_housing_types[2, :])
    new_item[np.isnan(new_item)] = 0
    outexp.export_map(
        new_item, grid, geo_grid, path_plots_floods,
        outexp.retrieve_name(item, -1) + '_shareinc',
        "Flood damages (% net income share), up to 99.99% quantile",
        path_tables_floods, ubnd=np.nanquantile(new_item, 0.9999))

# We also get an output for formal structure damage as a share of income,
# even if households do not directly pay for it. In any case, we can still
# directly interpret content damages as being paid by households
for item in list_sim_formal:
    new_item = (item
                / income_formal
                / initial_state_households_housing_types[0, :])
    new_item[np.isnan(new_item)] = 0
    outexp.export_map(
        new_item, grid, geo_grid, path_plots_floods,
        outexp.retrieve_name(item, -1) + '_shareinc',
        "Flood damages (% net income share), up to 99.99% quantile",
        path_tables_floods, ubnd=np.nanquantile(new_item, 0.9999))


# We could plot fraction of capital destroyed separately for each
# flood type, but it would be similar due to bath-tub model
# NB: note that content damage function is the same for all housing types
for col in fraction_capital_destroyed.columns:
    value = fraction_capital_destroyed[col]
    outexp.export_map(value, grid, geo_grid,
                      path_plots_floods, col + '_fract_K_destroyed', "",
                      path_tables_floods,
                      ubnd=1)
