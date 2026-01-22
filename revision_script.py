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
import outputs.export_outputs as outexp
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

# TODO: code loop to do everything in a row?

# ## Custom
options["urban_edge"] = 1
# param["year_urban_edge"] = param["baseline_year"]
options["informal_land_constrained"] = 1
options["new_RDP_housing"] = 0

# TODO: Do not converge well when too far from intiial equilibrium? Seems OKish

# In Harari and Wong, informal amenities go from 1.34 to 1.58 in the center
# (+18%), 1.17 to 1.16 in the middle (-0%), 0.88 to 0.79 in the periphery (-10%),
# or +3% on avg
options["amenity_upgrading"] = 0
options["poor_subsidies"] = 0

# In Harari and Wong, formalization costs go from .88 to .86 (center: -2%),
# .49 to .41 (middle: -8%):, and .78 to .61 (periphery: -17%), or -9% on avg
# We approach that as a redistribution of cell share available for squatting to
# formal development
options["eviction"] = 3

# TODO: rethink max_land_use parameters already at calibration stage?
# Maybe also land use regulations and land rent redistribution?
# Try to do some geographic heterogeneity in counterfactuals?

options["incremental_housing"] = 1

# First, target aggregate distribution
# Then, make it vary across locations
# param["disam_reduc_fact"] = 0.75

# ## Output name
name = ('simul_UE' + str(options["urban_edge"])
        + '_ISconstr' + str(options["informal_land_constrained"])
        + '_RDPnew' + str(options["new_RDP_housing"])
        + '_Aup' + str(options["amenity_upgrading"])
        + '_Psubsid' + str(options["poor_subsidies"])
        + '_Evict' + str(options["eviction"])
        + '_IH' + str(options["incremental_housing"]))

# COMPUTE COMPENSATION COST EX-POST!

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

param = inpprm.import_param(
    path_precalc_inp, options)


# # Load data

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

# REWEIGHT TOTAL BY POPULATION FROM INCOME DATA?

# households_per_income_class = (households_per_income_class
#                                * np.nansum(housing_type_data)
#                                / np.nansum(households_per_income_class))

# housing_type_data = (housing_type_data
#                      * np.nansum(households_per_income_class)
#                      / np.nansum(housing_type_data))
# backyard_data = (backyard_data
#                  * np.nansum(households_per_income_class)
#                  / np.nansum(backyard_data))

# Impact on RDP too...
# housing_type_data = (housing_type_data/2
#                       * np.nansum(households_per_income_class)
#                       / np.nansum(housing_type_data/2))
# backyard_data = (backyard_data/2
#                       * np.nansum(housing_type_data[1])
#                       / np.nansum(backyard_data/2))

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


# coeff_land = inpdt.import_coeff_land(
#     spline_land_constraints, spline_land_backyard, spline_land_informal,
#     spline_land_RDP, param, 0)
# total_RDP = spline_RDP(0)
# number_properties_RDP = spline_estimate_RDP(0)

# We let all effects kick in depending on options: not well defined?
coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, options, 29)
total_RDP = spline_RDP(29)
number_properties_RDP = spline_estimate_RDP(29)

housing_limit = inpdt.import_housing_limit(grid, param)

(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_baseline, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')

# Refer to utility changes in Harari and Wong: avg between +15%, +1% and -8%
# is +3%
# Do it only for non-RDP beneficiaries? Then, need to correct in their utilities!!!
if options["poor_subsidies"]==1:
    # income_net_of_commuting_costs[0] = income_net_of_commuting_costs[0]*1.03
    # income_net_of_commuting_costs[0] = (
    #     income_net_of_commuting_costs[0]
    #     + param["subsidized_structure_value"]
    #     * (9*param["current_rate_public_housing"] + 21*param["future_rate_public_housing"])
    #     / (households_per_income_class[0]*population/sum(households_per_income_class))
    #     )
    # income_net_of_commuting_costs[0] = (
    #     income_net_of_commuting_costs[0]
    #     + param["subsidized_structure_value"]
    #     * param["current_rate_public_housing"]
    #     / (households_per_income_class[0]*population/sum(households_per_income_class))
    #     )
    income_net_of_commuting_costs[0] = (
        income_net_of_commuting_costs[0]
        + param["subsidized_structure_value"]
        * param["current_rate_public_housing"]
        / (households_per_income_class[0]*population/sum(households_per_income_class)-total_RDP)
        )

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


# Scenario test

# print(total_RDP)
# print(np.nansum(number_properties_RDP))
# print(np.nanmean(income_net_of_commuting_costs[0,:]))
# print(np.nanmean(coeff_land,1))

# REDO CALIBRATION TO BETTER FIT HOUSING TYPES???

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


# TAKE CARE TO MUTABLE OBJECTS!

# backyard_hsupply = initial_state_housing_supply[1,:].copy()
# backyard_hsupply = backyard_hsupply/1000000

# formal_backyard_pop = initial_state_households_housing_types[1,:].copy()
# formal_backyard_pop[backyard_hsupply<2] = 0

# informal_backyard_pop = initial_state_households_housing_types[1,:].copy()
# informal_backyard_pop[backyard_hsupply>=2] = 0

# print(np.nansum(formal_backyard_pop))
# print(np.nansum(informal_backyard_pop))
# print(backyard_data)

# ##Household density
simul_nb_households_tot = np.nansum(initial_state_households_housing_types, 0)

# Grid cell = 500x500m = 25 Ha
simul_HHdens_HA = simul_nb_households_tot/25

simul_HHdens_HA_discrete = np.zeros(len(simul_HHdens_HA))
simul_HHdens_HA_discrete[(simul_HHdens_HA > 0)
                              & (simul_HHdens_HA <= 10)] = 1
simul_HHdens_HA_discrete[(simul_HHdens_HA > 10)
                              & (simul_HHdens_HA <= 20)] = 2
simul_HHdens_HA_discrete[(simul_HHdens_HA > 20)
                              & (simul_HHdens_HA <= 50)] = 3
simul_HHdens_HA_discrete[(simul_HHdens_HA > 50)
                              & (simul_HHdens_HA <= 100)] = 4
simul_HHdens_HA_discrete[(simul_HHdens_HA > 100)
                              & (simul_HHdens_HA <= 200)] = 5
simul_HHdens_HA_discrete[simul_HHdens_HA > 200] = 6

simul_HHdens_HA_discrete_map = outexp.discrete_map(
    simul_HHdens_HA_discrete, grid, geo_grid, path_output_plots,
    'simul_HHdens_HA_discrete_map',
    "Nb of HHs per Ha (WP scale)", path_output_tables)


# ## (Perpetual) land price (/m² of available land) in formal sector

# NB: do not worry about housing price per se

# Note that several housing types may co-exist within one cell (but there is
# one dominant income group for each housing type)

landprice_formal_simul = (
    (initial_state_rent[0, :] * param["coeff_A"])
    ** (1 / param["coeff_a"])
    * param["coeff_a"]
    * (param["coeff_b"] / (interest_rate + param["depreciation_rate"]))
    ** (param["coeff_b"] / param["coeff_a"])
    / interest_rate
    )

rent_formal_simul = initial_state_rent[0, :]

simul_nb_households_formal = initial_state_households_housing_types[0, :]
landprice_formal_simul[simul_nb_households_formal == 0] = 0
rent_formal_simul[simul_nb_households_formal == 0] = 0

simul_formal_landprice_discrete = np.zeros(len(landprice_formal_simul))
simul_formal_landprice_discrete[(landprice_formal_simul > 0)
                                    & (landprice_formal_simul <= 500)] = 1
simul_formal_landprice_discrete[(landprice_formal_simul > 500)
                                    & (landprice_formal_simul <= 1000)] = 2
simul_formal_landprice_discrete[(landprice_formal_simul > 1000)
                                    & (landprice_formal_simul <= 1500)] = 3
simul_formal_landprice_discrete[(landprice_formal_simul > 1500)
                                    & (landprice_formal_simul <= 2000)] = 4
simul_formal_landprice_discrete[(landprice_formal_simul > 2000)
                                    & (landprice_formal_simul <= 3000)] = 5
simul_formal_landprice_discrete[(landprice_formal_simul > 3000)
                                    & (landprice_formal_simul <= 4000)] = 6
simul_formal_landprice_discrete[(landprice_formal_simul > 4000)
                                    & (landprice_formal_simul <= 5000)] = 7
simul_formal_landprice_discrete[(landprice_formal_simul > 5000)
                                    & (landprice_formal_simul <= 6000)] = 8
simul_formal_landprice_discrete[landprice_formal_simul > 6000] = 9

# simul_formal_rent_discrete = np.zeros(len(rent_formal_simul))
# simul_formal_rent_discrete[(rent_formal_simul > 0)
#                                     & (rent_formal_simul <= 500)] = 1
# simul_formal_rent_discrete[(rent_formal_simul > 500)
#                                     & (rent_formal_simul <= 1000)] = 2
# simul_formal_rent_discrete[(rent_formal_simul > 1000)
#                                     & (rent_formal_simul <= 1500)] = 3
# simul_formal_rent_discrete[(rent_formal_simul > 1500)
#                                     & (rent_formal_simul <= 2000)] = 4
# simul_formal_rent_discrete[(rent_formal_simul > 2000)
#                                     & (rent_formal_simul <= 3000)] = 5
# simul_formal_rent_discrete[(rent_formal_simul > 3000)
#                                     & (rent_formal_simul <= 4000)] = 6
# simul_formal_rent_discrete[(rent_formal_simul > 4000)
#                                     & (rent_formal_simul <= 5000)] = 7
# simul_formal_rent_discrete[(rent_formal_simul > 5000)
#                                     & (rent_formal_simul <= 6000)] = 8
# simul_formal_rent_discrete[rent_formal_simul > 6000] = 9

simul_formal_landprice_discrete_map = outexp.discrete_map(
    simul_formal_landprice_discrete, grid, geo_grid, path_output_plots,
    'simul_formal_landprice_discrete_map',
    "Formal land price per m² (WP scale)", path_output_tables)

# simul_formal_rent_discrete_map = outexp.discrete_map(
#     simul_formal_rent_discrete, grid, geo_grid, path_output_plots,
#     'simul_formal_rent_discrete_map',
#     "Formal annual rent per m² (WP scale)", path_output_tables)


# TODO: what about FAR regulations?

# TEST FOR WELFARE DECOMPOSITON: what to save?

## Preamble

formal_size = initial_state_dwelling_size[0, :]
backyard_size = initial_state_dwelling_size[1, :]
informal_size = initial_state_dwelling_size[2, :]
rdp_size = initial_state_dwelling_size[3, :]

formal_rent = initial_state_rent[0, :]
backyard_rent = initial_state_rent[1, :]
informal_rent = initial_state_rent[2, :]
rdp_rent = initial_state_rent[3, :]

backyard_supply = initial_state_housing_supply[1,:]/1000000
disam_backyard = param["backyard_pockets"]
disam_backyard[backyard_supply==2] = np.nanmean(param["incremental_pockets"])
disam_informal = param["informal_pockets"]

## Theoretical values before allocation (not as homogeneous as should be?)

utility_temp_formal = (
    (income_net_of_commuting_costs - formal_size[None, :]*formal_rent[None, :])**param["alpha"]
    * (formal_size[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_temp_backyard = (
    (income_net_of_commuting_costs - backyard_size[None, :]*backyard_rent[None, :])**param["alpha"]
    * (backyard_size[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard[None, :]
    )

utility_temp_informal = (
    (income_net_of_commuting_costs - informal_size[None, :]*informal_rent[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal[None, :]
    )

# Size definition? Also limits in hsupply/land availability?
# May create inconsistency with baseline levels? Should not measure? Or separately...
# What about construction costs for backyards!
backyard_supply[np.isnan(backyard_supply)] = 0
backyard_rent[np.isnan(backyard_rent)] = 0

informal_cost = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply * param["backyard_size"] / param["shack_size"])
incremental_cost = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply * param["backyard_size"] / param["RDP_size"])
backyard_cost = informal_cost
backyard_cost[backyard_supply==2] = incremental_cost[backyard_supply==2]

utility_temp_rdp = (
    (income_net_of_commuting_costs
     + backyard_supply[None, :]*param["backyard_size"]*backyard_rent[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

## Allocation: can apply mask to old variables for extensive margin?

agg_alloc_grid = initial_state_households>0

utility_temp_mat = np.array([utility_temp_formal, utility_temp_backyard, utility_temp_informal, utility_temp_rdp])
utility_temp_mat = utility_temp_mat*agg_alloc_grid
utility_temp_mat[utility_temp_mat==0] = np.nan

# Slight variations from min (equilibrium value) to max irrelevant except for
# RDP beneficiaries in poor income group (up to between rich and midrich, but
# more like midpoor on average)

# Averages including RDP for the poor!!!
avg_utility_poor = np.nansum(utility_temp_mat[:,0,:]*initial_state_households[:,0,:])/np.nansum(initial_state_households[:,0,:])
avg_utility_midpoor = np.nansum(utility_temp_mat[:,1,:]*initial_state_households[:,1,:])/np.nansum(initial_state_households[:,1,:])
avg_utility_midrich = np.nansum(utility_temp_mat[:,2,:]*initial_state_households[:,2,:])/np.nansum(initial_state_households[:,2,:])
avg_utility_rich = np.nansum(utility_temp_mat[:,3,:]*initial_state_households[:,3,:])/np.nansum(initial_state_households[:,3,:])

# Now deal with decomposition of interest (do averages ex post?)
# Note that both values and worker allocation change in counterfactuals

# Need to write matrix form with possibility of changing each component?
# Also per housing type?

# Allocation and values need to change at the same time for each component!

# test = np.where(np.isnan(utility_temp_mat) & agg_alloc_grid==1)

