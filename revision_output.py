# #########################################
# ## SIMPLIFIED SCRIPT FOR JUE REVISION ###
# #########################################

# WE DO NOT RE-RUN CALIBRATION

# WE THEN FOCUS ON COMPARATIVE STATICS FROM MAIN EQUILIBRIUM SCRIPTS

# ## We import standard Python libraries
import numpy as np
import pandas as pd
# import os
import geopandas as gpd
import matplotlib.pyplot as plt
# mimport seaborn as sns

# ## We also import our own packages
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
# import equilibrium.compute_equilibrium as eqcmp
import outputs.export_outputs as outexp
# import equilibrium.run_simulations as eqsim
# import equilibrium.functions_dynamic as eqdyn

###############################################################################

# THEN WE VISUALIZE ONLY THE OUTPUT WE NEED (put in other script)

# Note that, even though he has access, the second poorest income group is not
# found in informal housing (whereas the poorest income group is crowded out
# of the formal sector)

# ## Define file paths
path_code = '..'
path_folder = path_code + '/Data/'
path_precalc_inp = path_folder + 'precalculated_inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_data + 'Scenarios/'
path_outputs = path_code + '/Output/'
path_input_plots = path_outputs + 'input_plots/'
path_input_tables = path_outputs + 'input_tables/'

# # Parameters and options

# ## Default
options = inpprm.import_options()
param = inpprm.import_param(
    path_precalc_inp, options)

# ## Custom
options["urban_edge"] = 0
# param["year_urban_edge"] = param["baseline_year"]
options["new_RDP_housing"] = 0
# year_begin_RDP?
options["incremental_housing"] = 1

# # ## Output name
# name = ('simul_UE' + str(options["urban_edge"])
#         + '_new_RDP' + str(options["new_RDP_housing"])
#         + '_IH' + str(options["incremental_housing"]))

path_simul = path_outputs + 'revision_output'
path_output_plots = path_simul + '/plots/'
path_output_tables = path_simul + '/tables/'

# ## Basic geographic data
grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp, options)
geo_grid = gpd.read_file(path_data + "grid_reference_500.shp")

# ## Macro data
(interest_rate, population, housing_type_data, total_RDP, backyard_data
 ) = inpdt.import_macro_data(param, path_scenarios, path_folder)

# ## Households and income data
income_class_by_housing_type = inpdt.import_hypothesis_housing_type()
(mean_income, households_per_income_class, average_income, income_mult,
 income_baseline, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_baseline,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0

# ## Land use projections
(spline_RDP, spline_estimate_RDP, spline_land_RDP,
 spline_land_backyard, spline_land_informal, spline_land_constraints,
 number_properties_RDP) = (
     inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                           housing_type_data, path_data, path_folder)
     )

if options["new_RDP_housing"] == 0:
    coeff_land = inpdt.import_coeff_land(
        spline_land_constraints, spline_land_backyard, spline_land_informal,
        spline_land_RDP, param, 0)
    total_RDP = spline_RDP(0)
    number_properties_RDP = spline_estimate_RDP(0)
elif options["new_RDP_housing"] == 1:
    coeff_land = inpdt.import_coeff_land(
        spline_land_constraints, spline_land_backyard, spline_land_informal,
        spline_land_RDP, param, 29)
    number_properties_RDP = spline_estimate_RDP(29)
    total_RDP = spline_RDP(29)

housing_limit = inpdt.import_housing_limit(grid, param)

(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_baseline, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')

# ## Empty flood data (to make function run)
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

# ##Import raw output

simul_UE1_new_RDP0_IH0_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_error = np.load(
    path_simul + '/initial_state_error_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_simulated_jobs = np.load(
    path_simul + '/initial_state_simulated_jobs_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_households_housing_types = np.load(
    path_simul + '/initial_state_households_housing_types_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_household_centers = np.load(
    path_simul + '/initial_state_household_centers_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_rent_matrix = np.load(
    path_simul + '/initial_state_rent_matrix_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_capital_land = np.load(
    path_simul + '/initial_state_capital_land_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_average_income = np.load(
    path_simul + '/initial_state_average_income_simul_UE1_new_RDP0_IH0.npy')
simul_UE1_new_RDP0_IH0_limit_city = np.load(
    path_simul + '/initial_state_limit_city_simul_UE1_new_RDP0_IH0.npy')

simul_UE1_new_RDP0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_error = np.load(
    path_simul + '/initial_state_error_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_simulated_jobs = np.load(
    path_simul + '/initial_state_simulated_jobs_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_households_housing_types = np.load(
    path_simul + '/initial_state_households_housing_types_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_household_centers = np.load(
    path_simul + '/initial_state_household_centers_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_rent_matrix = np.load(
    path_simul + '/initial_state_rent_matrix_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_capital_land = np.load(
    path_simul + '/initial_state_capital_land_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_average_income = np.load(
    path_simul + '/initial_state_average_income_simul_UE1_new_RDP0_IH1.npy')
simul_UE1_new_RDP0_IH1_limit_city = np.load(
    path_simul + '/initial_state_limit_city_simul_UE1_new_RDP0_IH1.npy')

#######

# sim_nb_households_backyard_UE1_new_RDP0_IH0 = simul_UE1_new_RDP0_IH0_households_housing_types[1, :]
# backyard_sim_UE1_new_RDP0_IH0 = outexp.export_map(
#     sim_nb_households_backyard_UE1_new_RDP0_IH0, grid, geo_grid, path_output_plots,  'backyard_sim_UE1_new_RDP0_IH0',
#     "Number of households in informal backyards, up to 99.99% quantile",
#     path_output_tables,
#     ubnd=np.nanquantile(sim_nb_households_backyard_UE1_new_RDP0_IH0, 0.9999))

# sim_nb_households_backyard_UE1_new_RDP0_IH1 = simul_UE1_new_RDP0_IH1_households_housing_types[1, :]
# backyard_sim_UE1_new_RDP0_IH1 = outexp.export_map(
#     sim_nb_households_backyard_UE1_new_RDP0_IH1, grid, geo_grid, path_output_plots,  'backyard_sim_UE1_new_RDP0_IH1',
#     "Number of households in informal backyards, up to 99.99% quantile",
#     path_output_tables,
#     ubnd=np.nanquantile(sim_nb_households_backyard_UE1_new_RDP0_IH1, 0.9999))

#

# We now consider overall land to recover building density
# NB: plotting the housing supply per unit of available land would be hard to
# interpret since we do not know the exact area that is available within each
# pixel, hence grid cells would not be comparable between each other
# hsupply_UE1_new_RDP0_IH0 = simul_UE1_new_RDP0_IH0_housing_supply # * coeff_land * 0.25
# hsupply_UE1_new_RDP0_IH1 = simul_UE1_new_RDP0_IH1_housing_supply # * coeff_land * 0.25

# hsupply_backyard_UE1_new_RDP0_IH0 = hsupply_UE1_new_RDP0_IH0[1, :]/1000000
# hsupply_backyard_UE1_new_RDP0_IH0[simul_UE1_new_RDP0_IH0_households_housing_types[1, :]==0] = 0
# hsupply_backyard_UE1_new_RDP0_IH0_2d_sim = outexp.export_map(
#     hsupply_backyard_UE1_new_RDP0_IH0, grid, geo_grid, path_output_plots,
#     'hsupply_backyard_UE1_new_RDP0_IH0_2d_sim',
#     "Total housing supply in informal backyards (in m²)",
#     path_output_tables,
#     ubnd=2)

# hsupply_backyard_UE1_new_RDP0_IH1 = hsupply_UE1_new_RDP0_IH1[1, :]/1000000
# hsupply_backyard_UE1_new_RDP0_IH1[simul_UE1_new_RDP0_IH1_households_housing_types[1, :]==0] = 0
# hsupply_backyard_UE1_new_RDP0_IH1_2d_sim = outexp.export_map(
#     hsupply_backyard_UE1_new_RDP0_IH1, grid, geo_grid, path_output_plots,
#     'hsupply_backyard_UE1_new_RDP0_IH1_2d_sim',
#     "Total housing supply in informal backyards (in m²)",
#     path_output_tables,
#     ubnd=np.nanquantile(hsupply_backyard_UE1_new_RDP0_IH1, 0.9999))

# hsupply_backyard_UE1_new_RDP0_IH1 = hsupply_UE1_new_RDP0_IH1[1, :]/1000000
# hsupply_backyard_UE1_new_RDP0_IH1[simul_UE1_new_RDP0_IH1_households_housing_types[1, :]==0] = 0
# hsupply_backyard_UE1_new_RDP0_IH1_2d_sim = outexp.export_map(
#     hsupply_backyard_UE1_new_RDP0_IH1, grid, geo_grid, path_output_plots,
#     'hsupply_backyard_UE1_new_RDP0_IH1_2d_sim',
#     "Total housing supply in informal backyards (in m²)",
#     path_output_tables,
#     ubnd=2)

#

# rent_backyard_UE1_new_RDP0_IH0 = simul_UE1_new_RDP0_IH0_rent[1, :]
# rent_backyard_UE1_new_RDP0_IH0[simul_UE1_new_RDP0_IH0_households_housing_types[1, :]==0] = 0
# rent_backyard_UE1_new_RDP0_IH0_2d_sim = outexp.export_map(
#     rent_backyard_UE1_new_RDP0_IH0, grid, geo_grid, path_output_plots,
#     'rent_backyard_UE1_new_RDP0_IH0_2d_sim',
#     "Annual rent in informal backyards (per m²)",
#     path_output_tables,
#     ubnd=np.nanquantile(rent_backyard_UE1_new_RDP0_IH0, 0.9999))

# rent_backyard_UE1_new_RDP0_IH1 = simul_UE1_new_RDP0_IH1_rent[1, :]
# rent_backyard_UE1_new_RDP0_IH1[simul_UE1_new_RDP0_IH1_households_housing_types[1, :]==0] = 0
# rent_backyard_UE1_new_RDP0_IH1_2d_sim = outexp.export_map(
#     rent_backyard_UE1_new_RDP0_IH1, grid, geo_grid, path_output_plots,
#     'rent_backyard_UE1_new_RDP0_IH1_2d_sim',
#     "Annual rent in informal backyards (per m²)",
#     path_output_tables,
#     ubnd=np.nanquantile(rent_backyard_UE1_new_RDP0_IH1, 0.9999))

# Better plots?






#####################################################################

# simul_UE0_utility = np.load(
#     path_simul + '/initial_state_utility_simul_UE0.npy')
# simul_UE0_error = np.load(
#     path_simul + '/initial_state_error_simul_UE0.npy')
# simul_UE0_simulated_jobs = np.load(
#     path_simul + '/initial_state_simulated_jobs_simul_UE0.npy')
# simul_UE0_households_housing_types = np.load(
#     path_simul + '/initial_state_households_housing_types_simul_UE0.npy')
# simul_UE0_household_centers = np.load(
#     path_simul + '/initial_state_household_centers_simul_UE0.npy')
# simul_UE0_households = np.load(
#     path_simul + '/initial_state_households_simul_UE0.npy')
# simul_UE0_dwelling_size = np.load(
#     path_simul + '/initial_state_dwelling_size_simul_UE0.npy')
# simul_UE0_housing_supply = np.load(
#     path_simul + '/initial_state_housing_supply_simul_UE0.npy')
# simul_UE0_rent = np.load(
#     path_simul + '/initial_state_rent_simul_UE0.npy')
# simul_UE0_rent_matrix = np.load(
#     path_simul + '/initial_state_rent_matrix_simul_UE0.npy')
# simul_UE0_capital_land = np.load(
#     path_simul + '/initial_state_capital_land_simul_UE0.npy')
# simul_UE0_average_income = np.load(
#     path_simul + '/initial_state_average_income_simul_UE0.npy')
# simul_UE0_limit_city = np.load(
#     path_simul + '/initial_state_limit_city_simul_UE0.npy')

# simul_UE1_utility = np.load(
#     path_simul + '/initial_state_utility_simul_UE1.npy')
# simul_UE1_error = np.load(
#     path_simul + '/initial_state_error_simul_UE1.npy')
# simul_UE1_simulated_jobs = np.load(
#     path_simul + '/initial_state_simulated_jobs_simul_UE1.npy')
# simul_UE1_households_housing_types = np.load(
#     path_simul + '/initial_state_households_housing_types_simul_UE1.npy')
# simul_UE1_household_centers = np.load(
#     path_simul + '/initial_state_household_centers_simul_UE1.npy')
# simul_UE1_households = np.load(
#     path_simul + '/initial_state_households_simul_UE1.npy')
# simul_UE1_dwelling_size = np.load(
#     path_simul + '/initial_state_dwelling_size_simul_UE1.npy')
# simul_UE1_housing_supply = np.load(
#     path_simul + '/initial_state_housing_supply_simul_UE1.npy')
# simul_UE1_rent = np.load(
#     path_simul + '/initial_state_rent_simul_UE1.npy')
# simul_UE1_rent_matrix = np.load(
#     path_simul + '/initial_state_rent_matrix_simul_UE1.npy')
# simul_UE1_capital_land = np.load(
#     path_simul + '/initial_state_capital_land_simul_UE1.npy')
# simul_UE1_average_income = np.load(
#     path_simul + '/initial_state_average_income_simul_UE1.npy')
# simul_UE1_limit_city = np.load(
#     path_simul + '/initial_state_limit_city_simul_UE1.npy')

# simul_UE0_new_RDP1_utility = np.load(
#     path_simul + '/initial_state_utility_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_error = np.load(
#     path_simul + '/initial_state_error_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_simulated_jobs = np.load(
#     path_simul + '/initial_state_simulated_jobs_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_households_housing_types = np.load(
#     path_simul
#     + '/initial_state_households_housing_types_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_household_centers = np.load(
#     path_simul + '/initial_state_household_centers_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_households = np.load(
#     path_simul + '/initial_state_households_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_dwelling_size = np.load(
#     path_simul + '/initial_state_dwelling_size_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_housing_supply = np.load(
#     path_simul + '/initial_state_housing_supply_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_rent = np.load(
#     path_simul + '/initial_state_rent_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_rent_matrix = np.load(
#     path_simul + '/initial_state_rent_matrix_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_capital_land = np.load(
#     path_simul + '/initial_state_capital_land_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_average_income = np.load(
#     path_simul + '/initial_state_average_income_simul_UE0_new_RDP1.npy')
# simul_UE0_new_RDP1_limit_city = np.load(
#     path_simul + '/initial_state_limit_city_simul_UE0_new_RDP1.npy')

# # dyn_simul_UE0_households_center = np.load(
# #     path_simul + '/simulation_households_center_simul_UE0.npy')
# # dyn_simul_UE0_households_housing_type = np.load(
# #     path_simul + '/simulation_households_housing_type_simul_UE0.npy')
# # dyn_simul_UE0_dwelling_size = np.load(
# #     path_simul + '/simulation_dwelling_size_simul_UE0.npy')
# # dyn_simul_UE0_rent = np.load(
# #     path_simul + '/simulation_rent_simul_UE0.npy')
# # dyn_simul_UE0_households = np.load(
# #     path_simul + '/simulation_households_simul_UE0.npy')
# # dyn_simul_UE0_error = np.load(
# #     path_simul + '/simulation_error_simul_UE0.npy')
# # dyn_simul_UE0_housing_supply = np.load(
# #     path_simul + '/simulation_housing_supply_simul_UE0.npy')
# # dyn_simul_UE0_utility = np.load(
# #     path_simul + '/simulation_utility_simul_UE0.npy')
# # dyn_simul_UE0_deriv_housing = np.load(
# #     path_simul + '/simulation_deriv_housing_simul_UE0.npy')
# # dyn_simul_UE0_T = np.load(
# #     path_simul + '/simulation_T_simul_UE0.npy')
# # dyn_simul_UE0_capital_land = np.load(
# #     path_simul + '/simulation_capital_land_simul_UE0.npy')

# # dyn_simul_UE1_households_center = np.load(
# #     path_simul + '/simulation_households_center_simul_UE1.npy')
# # dyn_simul_UE1_households_housing_type = np.load(
# #     path_simul + '/simulation_households_housing_type_simul_UE1.npy')
# # dyn_simul_UE1_dwelling_size = np.load(
# #     path_simul + '/simulation_dwelling_size_simul_UE1.npy')
# # dyn_simul_UE1_rent = np.load(
# #     path_simul + '/simulation_rent_simul_UE1.npy')
# # dyn_simul_UE1_households = np.load(
# #     path_simul + '/simulation_households_simul_UE1.npy')
# # dyn_simul_UE1_error = np.load(
# #     path_simul + '/simulation_error_simul_UE1.npy')
# # dyn_simul_UE1_housing_supply = np.load(
# #     path_simul + '/simulation_housing_supply_simul_UE1.npy')
# # dyn_simul_UE1_utility = np.load(
# #     path_simul + '/simulation_utility_simul_UE1.npy')
# # dyn_simul_UE1_deriv_housing = np.load(
# #     path_simul + '/simulation_deriv_housing_simul_UE1.npy')
# # dyn_simul_UE1_T = np.load(
# #     path_simul + '/simulation_T_simul_UE1.npy')
# # dyn_simul_UE1_capital_land = np.load(
# #     path_simul + '/simulation_capital_land_simul_UE1.npy')

# # ##Household density
# simul_UE0_nb_households_tot = np.nansum(simul_UE0_households_housing_types, 0)
# simul_UE1_nb_households_tot = np.nansum(simul_UE1_households_housing_types, 0)
# simul_UE0_new_RDP1_nb_households_tot = np.nansum(
#     simul_UE0_new_RDP1_households_housing_types, 0)

# # Grid cell = 500x500m = 25 Ha
# simul_UE0_HHdens_HA = simul_UE0_nb_households_tot/25
# simul_UE1_HHdens_HA = simul_UE1_nb_households_tot/25
# simul_UE0_new_RDP1_HHdens_HA = simul_UE0_new_RDP1_nb_households_tot/25

# simul_UE0_HHdens_HA_discrete = np.zeros(len(simul_UE0_HHdens_HA))
# simul_UE0_HHdens_HA_discrete[(simul_UE0_HHdens_HA > 0)
#                              & (simul_UE0_HHdens_HA <= 10)] = 1
# simul_UE0_HHdens_HA_discrete[(simul_UE0_HHdens_HA > 10)
#                              & (simul_UE0_HHdens_HA <= 20)] = 2
# simul_UE0_HHdens_HA_discrete[(simul_UE0_HHdens_HA > 20)
#                              & (simul_UE0_HHdens_HA <= 50)] = 3
# simul_UE0_HHdens_HA_discrete[(simul_UE0_HHdens_HA > 50)
#                              & (simul_UE0_HHdens_HA <= 100)] = 4
# simul_UE0_HHdens_HA_discrete[(simul_UE0_HHdens_HA > 100)
#                              & (simul_UE0_HHdens_HA <= 200)] = 5
# simul_UE0_HHdens_HA_discrete[simul_UE0_HHdens_HA > 200] = 6

# simul_UE1_HHdens_HA_discrete = np.zeros(len(simul_UE1_HHdens_HA))
# simul_UE1_HHdens_HA_discrete[(simul_UE1_HHdens_HA > 0)
#                              & (simul_UE1_HHdens_HA <= 10)] = 1
# simul_UE1_HHdens_HA_discrete[(simul_UE1_HHdens_HA > 10)
#                              & (simul_UE1_HHdens_HA <= 20)] = 2
# simul_UE1_HHdens_HA_discrete[(simul_UE1_HHdens_HA > 20)
#                              & (simul_UE1_HHdens_HA <= 50)] = 3
# simul_UE1_HHdens_HA_discrete[(simul_UE1_HHdens_HA > 50)
#                              & (simul_UE1_HHdens_HA <= 100)] = 4
# simul_UE1_HHdens_HA_discrete[(simul_UE1_HHdens_HA > 100)
#                              & (simul_UE1_HHdens_HA <= 200)] = 5
# simul_UE1_HHdens_HA_discrete[simul_UE1_HHdens_HA > 200] = 6

# simul_UE0_new_RDP1_HHdens_HA_discrete = np.zeros(
#     len(simul_UE0_new_RDP1_HHdens_HA))
# simul_UE0_new_RDP1_HHdens_HA_discrete[
#     (simul_UE0_new_RDP1_HHdens_HA > 0)
#     & (simul_UE0_new_RDP1_HHdens_HA <= 10)] = 1
# simul_UE0_new_RDP1_HHdens_HA_discrete[
#     (simul_UE0_new_RDP1_HHdens_HA > 10)
#     & (simul_UE0_new_RDP1_HHdens_HA <= 20)] = 2
# simul_UE0_new_RDP1_HHdens_HA_discrete[
#     (simul_UE0_new_RDP1_HHdens_HA > 20)
#     & (simul_UE0_new_RDP1_HHdens_HA <= 50)] = 3
# simul_UE0_new_RDP1_HHdens_HA_discrete[
#     (simul_UE0_new_RDP1_HHdens_HA > 50)
#     & (simul_UE0_new_RDP1_HHdens_HA <= 100)] = 4
# simul_UE0_new_RDP1_HHdens_HA_discrete[
#     (simul_UE0_new_RDP1_HHdens_HA > 100)
#     & (simul_UE0_new_RDP1_HHdens_HA <= 200)] = 5
# simul_UE0_new_RDP1_HHdens_HA_discrete[simul_UE0_new_RDP1_HHdens_HA > 200] = 6

# simul_UE0_HHdens_HA_discrete_map = outexp.discrete_map(
#     simul_UE0_HHdens_HA_discrete, grid, geo_grid, path_output_plots,
#     'simul_UE0_HHdens_HA_discrete_map',
#     "Nb of HHs per Ha (WP scale)", path_output_tables)

# simul_UE1_HHdens_HA_discrete_map = outexp.discrete_map(
#     simul_UE1_HHdens_HA_discrete, grid, geo_grid, path_output_plots,
#     'simul_UE1_HHdens_HA_discrete_map',
#     "Nb of HHs per Ha (WP scale)", path_output_tables)

# simul_UE0_new_RDP1_HHdens_HA_discrete_map = outexp.discrete_map(
#     simul_UE0_new_RDP1_HHdens_HA_discrete, grid, geo_grid, path_output_plots,
#     'simul_UE0_new_RDP1_HHdens_HA_discrete_map',
#     "Nb of HHs per Ha (WP scale)", path_output_tables)


# # ## (Perpetual) land price (/m² of available land) in formal sector

# # NB: do not worry about housing price per se

# # Note that several housing types may co-exist within one cell (but there is
# # one dominant income group for each housing type)

# landprice_formal_simul_UE0 = (
#     (simul_UE0_rent[0, :] * param["coeff_A"])
#     ** (1 / param["coeff_a"])
#     * param["coeff_a"]
#     * (param["coeff_b"] / (interest_rate + param["depreciation_rate"]))
#     ** (param["coeff_b"] / param["coeff_a"])
#     / interest_rate
#     )

# landprice_formal_simul_UE1 = (
#     (simul_UE1_rent[0, :] * param["coeff_A"])
#     ** (1 / param["coeff_a"])
#     * param["coeff_a"]
#     * (param["coeff_b"] / (interest_rate + param["depreciation_rate"]))
#     ** (param["coeff_b"] / param["coeff_a"])
#     / interest_rate
#     )

# landprice_formal_simul_UE0_new_RDP1 = (
#     (simul_UE0_new_RDP1_rent[0, :] * param["coeff_A"])
#     ** (1 / param["coeff_a"])
#     * param["coeff_a"]
#     * (param["coeff_b"] / (interest_rate + param["depreciation_rate"]))
#     ** (param["coeff_b"] / param["coeff_a"])
#     / interest_rate
#     )


# # rent_formal_simul_UE0 = simul_UE0_rent[0, :]
# # rent_formal_simul_UE1 = simul_UE1_rent[0, :]

# simul_UE0_nb_households_formal = simul_UE0_households_housing_types[0, :]
# landprice_formal_simul_UE0[simul_UE0_nb_households_formal == 0] = 0
# # rent_formal_simul_UE0[simul_UE0_nb_households_formal == 0] = 0

# simul_UE1_nb_households_formal = simul_UE1_households_housing_types[0, :]
# landprice_formal_simul_UE1[simul_UE1_nb_households_formal == 0] = 0
# # rent_formal_simul_UE1[simul_UE1_nb_households_formal == 0] = 0

# simul_UE0_new_RDP1_nb_households_formal = (
#     simul_UE0_new_RDP1_households_housing_types[0, :])
# landprice_formal_simul_UE0_new_RDP1[
#     simul_UE0_new_RDP1_nb_households_formal == 0] = 0

# simul_UE0_formal_landprice_discrete = np.zeros(len(landprice_formal_simul_UE0))
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 0)
#                                     & (landprice_formal_simul_UE0 <= 500)] = 1
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 500)
#                                     & (landprice_formal_simul_UE0 <= 1000)] = 2
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 1000)
#                                     & (landprice_formal_simul_UE0 <= 1500)] = 3
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 1500)
#                                     & (landprice_formal_simul_UE0 <= 2000)] = 4
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 2000)
#                                     & (landprice_formal_simul_UE0 <= 3000)] = 5
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 3000)
#                                     & (landprice_formal_simul_UE0 <= 4000)] = 6
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 4000)
#                                     & (landprice_formal_simul_UE0 <= 5000)] = 7
# simul_UE0_formal_landprice_discrete[(landprice_formal_simul_UE0 > 5000)
#                                     & (landprice_formal_simul_UE0 <= 6000)] = 8
# simul_UE0_formal_landprice_discrete[landprice_formal_simul_UE0 > 6000] = 9

# simul_UE1_formal_landprice_discrete = np.zeros(len(landprice_formal_simul_UE1))
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 0)
#                                     & (landprice_formal_simul_UE1 <= 500)] = 1
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 500)
#                                     & (landprice_formal_simul_UE1 <= 1000)] = 2
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 1000)
#                                     & (landprice_formal_simul_UE1 <= 1500)] = 3
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 1500)
#                                     & (landprice_formal_simul_UE1 <= 2000)] = 4
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 2000)
#                                     & (landprice_formal_simul_UE1 <= 3000)] = 5
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 3000)
#                                     & (landprice_formal_simul_UE1 <= 4000)] = 6
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 4000)
#                                     & (landprice_formal_simul_UE1 <= 5000)] = 7
# simul_UE1_formal_landprice_discrete[(landprice_formal_simul_UE1 > 5000)
#                                     & (landprice_formal_simul_UE1 <= 6000)] = 8
# simul_UE1_formal_landprice_discrete[landprice_formal_simul_UE1 > 6000] = 9

# simul_UE0_new_RDP1_formal_landprice_discrete = np.zeros(
#     len(landprice_formal_simul_UE0_new_RDP1))
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 0)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 500)] = 1
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 500)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 1000)] = 2
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 1000)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 1500)] = 3
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 1500)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 2000)] = 4
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 2000)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 3000)] = 5
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 3000)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 4000)] = 6
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 4000)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 5000)] = 7
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     (landprice_formal_simul_UE0_new_RDP1 > 5000)
#     & (landprice_formal_simul_UE0_new_RDP1 <= 6000)] = 8
# simul_UE0_new_RDP1_formal_landprice_discrete[
#     landprice_formal_simul_UE0_new_RDP1 > 6000] = 9

# # simul_UE0_formal_rent_discrete = np.zeros(len(rent_formal_simul_UE0))
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 0)
# #                                     & (rent_formal_simul_UE0 <= 500)] = 1
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 500)
# #                                     & (rent_formal_simul_UE0 <= 1000)] = 2
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 1000)
# #                                     & (rent_formal_simul_UE0 <= 1500)] = 3
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 1500)
# #                                     & (rent_formal_simul_UE0 <= 2000)] = 4
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 2000)
# #                                     & (rent_formal_simul_UE0 <= 3000)] = 5
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 3000)
# #                                     & (rent_formal_simul_UE0 <= 4000)] = 6
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 4000)
# #                                     & (rent_formal_simul_UE0 <= 5000)] = 7
# # simul_UE0_formal_rent_discrete[(rent_formal_simul_UE0 > 5000)
# #                                     & (rent_formal_simul_UE0 <= 6000)] = 8
# # simul_UE0_formal_rent_discrete[rent_formal_simul_UE0 > 6000] = 9

# # simul_UE1_formal_rent_discrete = np.zeros(len(rent_formal_simul_UE1))
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 0)
# #                                     & (rent_formal_simul_UE1 <= 500)] = 1
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 500)
# #                                     & (rent_formal_simul_UE1 <= 1000)] = 2
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 1000)
# #                                     & (rent_formal_simul_UE1 <= 1500)] = 3
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 1500)
# #                                     & (rent_formal_simul_UE1 <= 2000)] = 4
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 2000)
# #                                     & (rent_formal_simul_UE1 <= 3000)] = 5
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 3000)
# #                                     & (rent_formal_simul_UE1 <= 4000)] = 6
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 4000)
# #                                     & (rent_formal_simul_UE1 <= 5000)] = 7
# # simul_UE1_formal_rent_discrete[(rent_formal_simul_UE1 > 5000)
# #                                     & (rent_formal_simul_UE1 <= 6000)] = 8
# # simul_UE1_formal_rent_discrete[rent_formal_simul_UE1 > 6000] = 9

# simul_UE0_formal_landprice_discrete_map = outexp.discrete_map(
#     simul_UE0_formal_landprice_discrete, grid, geo_grid, path_output_plots,
#     'simul_UE0_formal_landprice_discrete_map',
#     "Formal land price per m² (WP scale)", path_output_tables)

# simul_UE1_formal_landprice_discrete_map = outexp.discrete_map(
#     simul_UE1_formal_landprice_discrete, grid, geo_grid, path_output_plots,
#     'simul_UE1_formal_landprice_discrete_map',
#     "Formal land price per m² (WP scale)", path_output_tables)

# simul_UE0_new_RDP1_formal_landprice_discrete_map = outexp.discrete_map(
#     simul_UE0_new_RDP1_formal_landprice_discrete, grid, geo_grid,
#     path_output_plots,
#     'simul_UE0_new_RDP1_formal_landprice_discrete_map',
#     "Formal land price per m² (WP scale)", path_output_tables)

# # simul_UE0_formal_rent_discrete_map = outexp.discrete_map(
# #     simul_UE0_formal_rent_discrete, grid, geo_grid, path_output_plots,
# #     'simul_UE0_formal_rent_discrete_map',
# #     "Formal annual rent per m² (WP scale)", path_output_tables)

# # simul_UE1_formal_rent_discrete_map = outexp.discrete_map(
# #     simul_UE1_formal_rent_discrete, grid, geo_grid, path_output_plots,
# #     'simul_UE1_formal_rent_discrete_map',
# #     "Formal annual per m² (WP scale)", path_output_tables)


# # ## Potential gains from formalizing informal land (no added costs)
# # ## NB: positive gains only?

# formalization_gain_UE0 = (landprice_formal_simul_UE0 * interest_rate
#                           - simul_UE0_rent[2, :])
# # simul_UE0_formalization_gain_discrete_map = outexp.discrete_map(
# #     formalization_gain_UE0, grid, geo_grid, path_output_plots,
# #     'simul_UE0_formalization_gain_discrete_map',
# #     "Static formalization gain per m², no added costs (WP scale)",
# #     path_output_tables)
# simul_UE0_formalization_gain_map = outexp.export_map(
#     formalization_gain_UE0, grid, geo_grid, path_output_plots,
#     'simul_UE0_formalization_gain_discrete_map',
#     "Static formalization gain per m², no added costs (WP scale)",
#     path_output_tables,ubnd=100, lbnd=-100, cmap='coolwarm')

# # Take care
# #landprice_formal_simul_UE1[landprice_formal_simul_UE1==0] = np.nan
# #simul_UE1_rent[2, :][simul_UE1_rent[2, :]==0] = np.nan

# formalization_gain_UE1 = (landprice_formal_simul_UE1 * interest_rate
#                           - simul_UE1_rent[2, :])
# #formalization_gain_UE1[simul_UE1_households_housing_types[2,:]==0] = np.nan
# formalization_gain_UE1[coeff_land[2,:]==0] = np.nan
# formalization_gain_UE1[coeff_land[2,:]==0] = np.nan
# np.nanmean(formalization_gain_UE1[formalization_gain_UE1>0])

# # simul_UE1_formalization_gain_discrete_map = outexp.discrete_map(
# #     formalization_gain_UE1, grid, geo_grid, path_output_plots,
# #     'simul_UE1_formalization_gain_discrete_map',
# #     "Static formalization gain per m², no added costs (WP scale)",
# #     path_output_tables)
# # Refine center of color scale
# simul_UE1_formalization_gain_map = outexp.export_map(
#     formalization_gain_UE1, grid, geo_grid, path_output_plots,
#     'simul_UE1_formalization_gain_discrete_map',
#     "Static formalization gain per m², no added costs (WP scale)",
#     path_output_tables,ubnd=100, lbnd=-100, cmap='coolwarm')
# # On average, gain is lower to cost in Henderson et al.!!

# formalization_gain_UE0_new_RDP1 = (landprice_formal_simul_UE0_new_RDP1
#                                    * interest_rate
#                                    - simul_UE0_new_RDP1_rent[2, :])
# # simul_UE0_new_RDP1_formalization_gain_discrete_map = outexp.discrete_map(
# #     formalization_gain_UE0_new_RDP1, grid, geo_grid, path_output_plots,
# #     'simul_UE0_new_RDP1_formalization_gain_discrete_map',
# #     "Static formalization gain per m², no added costs (WP scale)",
# #     path_output_tables)
# simul_UE0_new_RDP1_formalization_gain_map = outexp.export_map(
#     formalization_gain_UE0_new_RDP1, grid, geo_grid, path_output_plots,
#     'simul_UE0_new_RDP1_formalization_gain_discrete_map',
#     "Static formalization gain per m², no added costs (WP scale)",
#     path_output_tables,ubnd=100, lbnd=-100, cmap='coolwarm')

# # ## Nb of HHs per housing type

# simul_UE0_agg_HH_per_htype = np.nansum(simul_UE0_households_housing_types, 1)
# simul_UE1_agg_HH_per_htype = np.nansum(simul_UE1_households_housing_types, 1)
# simul_UE0_new_RDP1_agg_HH_per_htype = np.nansum(
#     simul_UE0_new_RDP1_households_housing_types, 1)

# data_compar_UE_agg_HH_per_htype = {
#     'Category': ['Formal', 'Formal', 'Backyard', 'Backyard',
#                  'Informal', 'Informal', 'Public', 'Public'],
#     'Group': ['UE1', 'UE0', 'UE1', 'UE0', 'UE1', 'UE0', 'UE1', 'UE0'],
#     'Value': [simul_UE1_agg_HH_per_htype[0], simul_UE0_agg_HH_per_htype[0],
#               simul_UE1_agg_HH_per_htype[1], simul_UE0_agg_HH_per_htype[1],
#               simul_UE1_agg_HH_per_htype[2], simul_UE0_agg_HH_per_htype[2],
#               simul_UE1_agg_HH_per_htype[3], simul_UE0_agg_HH_per_htype[3]]
# }
# df_compar_UE_agg_HH_per_htype = pd.DataFrame(data_compar_UE_agg_HH_per_htype)

# sns.barplot(x='Category', y='Value', hue='Group',
#             data=df_compar_UE_agg_HH_per_htype)
# plt.title('Nb of HHs by housing type and scenario')
# plt.savefig(path_output_plots + 'compar_UE_agg_HH_per_htype')

# # NB: at baseline, we underestimate informal and overestimate formal!

# # Larger initial population drives new sorting across housing markets!
# # Of course, public housing supply (etc.) should also play a role (not here)
# # Need to check how backyarding adjusts with land availability
# # Utility evolution also makes sense!

# data_compar_new_RDP_agg_HH_per_htype = {
#     'Category': ['Formal', 'Formal', 'Backyard', 'Backyard',
#                  'Informal', 'Informal', 'Public', 'Public'],
#     'Group': ['new_RDP1', 'new_RDP0', 'new_RDP1', 'new_RDP0',
#               'new_RDP1', 'new_RDP0', 'new_RDP1', 'new_RDP0'],
#     'Value': [simul_UE0_new_RDP1_agg_HH_per_htype[0],
#               simul_UE0_agg_HH_per_htype[0],
#               simul_UE0_new_RDP1_agg_HH_per_htype[1],
#               simul_UE0_agg_HH_per_htype[1],
#               simul_UE0_new_RDP1_agg_HH_per_htype[2],
#               simul_UE0_agg_HH_per_htype[2],
#               simul_UE0_new_RDP1_agg_HH_per_htype[3],
#               simul_UE0_agg_HH_per_htype[3]]
# }
# df_compar_new_RDP_agg_HH_per_htype = pd.DataFrame(
#     data_compar_new_RDP_agg_HH_per_htype)

# sns.barplot(x='Category', y='Value', hue='Group',
#             data=df_compar_new_RDP_agg_HH_per_htype)
# plt.title('Nb of HHs by housing type and scenario')
# plt.savefig(path_output_plots + 'compar_new_RDP_agg_HH_per_htype')

# # W/o population growth, we just have a switch from informal to formal:
# # backyarding even decreases slightly!
# # W/ population growth (from same baseline), we observe more backyarding
# # Utility evolution still makes sense


# print('Running done')


# # NB: only try to debug unsatisfactory results

# # Take care of code styling and optimization at replication phase

# # Same for visualization: what matters first is result interpretability
