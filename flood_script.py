# #########################################
# ## SIMPLIFIED SCRIPT FOR JUE REVISION ###
# #########################################

# %%

# RE-RUN CALIBRATION!!!

# WE THEN FOCUS ON COMPARATIVE STATICS FROM MAIN EQUILIBRIUM SCRIPTS

# ## We import standard Python libraries

import importlib

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

import equilibrium.sub.compute_outputs as eqout

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
options["eviction"] = 0

# TODO: rethink max_land_use parameters already at calibration stage?
# Maybe also land use regulations and land rent redistribution?
# Try to do some geographic heterogeneity in counterfactuals?

options["incremental_housing"] = 0

# First, target aggregate distribution
# Then, make it vary across locations
# param["disam_reduc_fact"] = 0.75

# ## Output name

#NB: add an option for who bears the structural costs in backyards?

options["agents_anticipate_floods"] = 1

options["climate_change"] = 1

options["risk_misperc"] = 1

# NB: need to compute homogeneous tax ex post wrt damage estimates
# options["subsid_insur"] = 0

options["self_protec"] = 1

options["risk_avers"] = 1

name = ('simul_AF' + str(options["agents_anticipate_floods"]) + '_CC' + str(options["climate_change"])
        + '_RM' + str(options["risk_misperc"]) + '_SP' + str(options["self_protec"]))

# COMPUTE COMPENSATION COST EX-POST!

path_simul = path_outputs + 'flood_output'
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

# SEE JUSTIFICATION
# NB: has to calibrate externally not to be confused with amenity index estimation
# NB: this is based on formal data (but can use additional references to argue for homeneity across income groups / housing types) 

param["risk_internaliz"] = 0.36

# Can go up to three levels, with 15cm elevation each (for IS only)
# NB: get inspiration from redevelopment choice?
# NB: add an option to allow it to be susidized
# NB: check ex post if we recover take-up rates aligned with Visser et al.
param["sandbag_course_cost"] = 1500 # correcting for inflation
# param["sandbag_course_cost"] = 3150

# Matters for self-protection decision???
param["CRRA"] = 0.2772

# Measure who subscribes?

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
    
# transfer_value = (
#     param["subsidized_structure_value"] * param["current_rate_public_housing"]
#     / (households_per_income_class[0]*population/sum(households_per_income_class)-total_RDP)
#     )

# print(np.nanmin(transfer_value))
# print(np.nanmean(transfer_value))
# print(np.nanmax(transfer_value))

# If agents anticipate floods, we return output from damage functions
if options["agents_anticipate_floods"] == 1:
    (fraction_capital_destroyed, fraction_capital_destroyed_protec, structural_damages_small_houses,
     structural_damages_medium_houses, structural_damages_large_houses,
     content_damages, structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b,
     damages_table, damages_protec_table, interval_table_fathom
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

    fraction_capital_destroyed_protec = pd.DataFrame()
    fraction_capital_destroyed_protec["structure_formal_2"] = np.zeros(24014)
    fraction_capital_destroyed_protec["structure_formal_1"] = np.zeros(24014)
    fraction_capital_destroyed_protec["structure_subsidized_2"] = np.zeros(24014)
    fraction_capital_destroyed_protec["structure_subsidized_1"] = np.zeros(24014)
    fraction_capital_destroyed_protec["contents_formal"] = np.zeros(24014)
    fraction_capital_destroyed_protec["contents_informal"] = np.zeros(24014)
    fraction_capital_destroyed_protec["contents_subsidized"] = np.zeros(24014)
    fraction_capital_destroyed_protec["contents_backyard"] = np.zeros(24014)
    fraction_capital_destroyed_protec["structure_backyards"] = np.zeros(24014)
    fraction_capital_destroyed_protec["structure_formal_backyards"] = np.zeros(24014)
    fraction_capital_destroyed_protec["structure_informal_backyards"
                               ] = np.zeros(24014)
    fraction_capital_destroyed_protec["structure_informal_settlements"
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

# importlib.reload(eqout)
param["max_iter"] = 100

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
 initial_state_limit_city,
 mask_self_protec) = eqcmp.compute_equilibrium(
     fraction_capital_destroyed,
     fraction_capital_destroyed_protec,
     damages_table, damages_protec_table, interval_table_fathom,
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
     
# simul_UE0_ISconstr1_RDPnew0_Aup2_Psubsid1_Evict0_IH1_utility = initial_state_utility
# print(simul_UE0_ISconstr1_RDPnew0_Aup2_Psubsid1_Evict0_IH1_utility/simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility)
     
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

np.save(path_simul + '/mask_self_protec_' + name + '.npy',
        mask_self_protec)

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