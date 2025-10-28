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
# import matplotlib.pyplot as plt
# import seaborn as sns

# ## We also import our own packages
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import equilibrium.compute_equilibrium as eqcmp
# import outputs.export_outputs as outexp
# import equilibrium.run_simulations as eqsim
# import equilibrium.functions_dynamic as eqdyn


# # Preamble

print("Preamble start")

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
options["urban_edge"] = 1
# param["year_urban_edge"] = param["baseline_year"]
options["new_RDP_housing"] = 0
# year_begin_RDP?
options["incremental_housing"] = 1

# ## Output name
name = ('simul_UE' + str(options["urban_edge"])
        + '_new_RDP' + str(options["new_RDP_housing"])
        + '_IH' + str(options["incremental_housing"]))

path_simul = path_outputs + 'revision_output'
path_output_plots = path_simul + '/plots/'
path_output_tables = path_simul + '/tables/'

# try:
#     os.mkdir(path_simul)
# except OSError as error:
#     print(error)
# try:
#     os.mkdir(path_output_plots)
# except OSError as error:
#     print(error)
# try:
#     os.mkdir(path_output_tables)
# except OSError as error:
#     print(error)

# # Load data

# ## Basic geographic data
grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp, options)
geo_grid = gpd.read_file(path_data + "grid_reference_500.shp")

# ## Macro data
(interest_rate, population, housing_type_data, total_RDP
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

# (spline_agricultural_price, spline_interest_rate,
#  spline_population_income_distribution, spline_inflation,
#  spline_income_distribution, spline_population,
#  spline_income, spline_minimum_housing_supply, spline_fuel
#  ) = eqdyn.import_scenarios(income_baseline, param, grid, path_scenarios,
#                             options)

# ##Equilibrium function
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
     1,
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

# ##Simulation function towards 2040
# (simulation_households_center,
#  simulation_households_housing_type,
#  simulation_dwelling_size,
#  simulation_rent,
#  simulation_households,
#  simulation_error,
#  simulation_housing_supply,
#  simulation_utility,
#  simulation_deriv_housing,
#  simulation_T,
#  simulation_capital_land) = eqsim.run_simulation(
#      30,
#      options,
#      param,
#      grid,
#      initial_state_utility,
#      initial_state_error,
#      initial_state_households,
#      initial_state_households_housing_types,
#      initial_state_housing_supply,
#      initial_state_household_centers,
#      initial_state_average_income,
#      initial_state_rent,
#      initial_state_dwelling_size,
#      initial_state_capital_land,
#      fraction_capital_destroyed,
#      amenities,
#      housing_limit,
#      spline_estimate_RDP,
#      spline_land_constraints,
#      spline_land_backyard,
#      spline_land_RDP,
#      spline_land_informal,
#      income_class_by_housing_type,
#      path_precalc_transp,
#      spline_RDP,
#      spline_agricultural_price,
#      spline_interest_rate,
#      spline_population_income_distribution,
#      spline_inflation,
#      spline_income_distribution,
#      spline_population,
#      spline_income,
#      spline_minimum_housing_supply,
#      spline_fuel,
#      income_baseline
#      )

# ##Export raw output
np.save(path_simul + '/initial_state_utility_' + name + '.npy',
        initial_state_utility)
np.save(path_simul + '/initial_state_error_' + name + '.npy',
        initial_state_error)
np.save(path_simul + '/initial_state_simulated_jobs_' + name + '.npy',
        initial_state_simulated_jobs)
np.save(path_simul + '/initial_state_households_housing_types_' + name +
        '.npy',
        initial_state_households_housing_types)
np.save(path_simul + '/initial_state_household_centers_' + name + '.npy',
        initial_state_household_centers)
np.save(path_simul + '/initial_state_households_' + name + '.npy',
        initial_state_households)
np.save(path_simul + '/initial_state_dwelling_size_' + name + '.npy',
        initial_state_dwelling_size)
np.save(path_simul + '/initial_state_housing_supply_' + name + '.npy',
        initial_state_housing_supply)
np.save(path_simul + '/initial_state_rent_' + name + '.npy',
        initial_state_rent)
np.save(path_simul + '/initial_state_rent_matrix_' + name + '.npy',
        initial_state_rent_matrix)
np.save(path_simul + '/initial_state_capital_land_' + name + '.npy',
        initial_state_capital_land)
np.save(path_simul + '/initial_state_average_income_' + name + '.npy',
        initial_state_average_income)
np.save(path_simul + '/initial_state_limit_city_' + name + '.npy',
        initial_state_limit_city)

# np.save(path_simul + '/simulation_households_center_' + name + '.npy',
#         simulation_households_center)
# np.save(path_simul + '/simulation_households_housing_type_' + name + '.npy',
#         simulation_households_housing_type)
# np.save(path_simul + '/simulation_dwelling_size_' + name + '.npy',
#         simulation_dwelling_size)
# np.save(path_simul + '/simulation_rent_' + name + '.npy',
#         simulation_rent)
# np.save(path_simul + '/simulation_households_' + name + '.npy',
#         simulation_households)
# np.save(path_simul + '/simulation_error_' + name + '.npy',
#         simulation_error)
# np.save(path_simul + '/simulation_housing_supply_' + name + '.npy',
#         simulation_housing_supply)
# np.save(path_simul + '/simulation_utility_' + name + '.npy',
#         simulation_utility)
# np.save(path_simul + '/simulation_deriv_housing_' + name + '.npy',
#         simulation_deriv_housing)
# np.save(path_simul + '/simulation_T_' + name + '.npy',
#         simulation_T)
# np.save(path_simul + '/simulation_capital_land_' + name + '.npy',
#         simulation_capital_land)

print("Preamble done")
