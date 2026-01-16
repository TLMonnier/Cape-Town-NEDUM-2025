# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:40:06 2026

@author: monni
"""

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
import seaborn as sns

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
# options["urban_edge"] = 1
# param["year_urban_edge"] = param["baseline_year"]
# options["new_RDP_housing"] = 0
# year_begin_RDP?
# options["incremental_housing"] = 1

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

coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, options, 29)
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

# ##Import raw output: do loop?

# name = ('simul_UE' + str(options["urban_edge"])
#         + '_ISconstr' + str(options["informal_land_constrained"])
#         + '_RDPnew' + str(options["new_RDP_housing"])
#         + '_Aup' + str(options["amenity_upgrading"])
#         + '_Psubsid' + str(options["poor_subsidies"])
#         + '_Evict' + str(options["eviction"])
#         + '_IH' + str(options["incremental_housing"]))

# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = np.load(
#     path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_error = np.load(
#     path_simul + '/initial_state_error_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_simulated_jobs = np.load(
#     path_simul + '/initial_state_simulated_jobs_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH11_households_housing_types = np.load(
#     path_simul + '/initial_state_households_housing_types_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_household_centers = np.load(
#     path_simul + '/initial_state_household_centers_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.load(
#     path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.load(
#     path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply = np.load(
#     path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.load(
#     path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent_matrix = np.load(
#     path_simul + '/initial_state_rent_matrix_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_capital_land = np.load(
#     path_simul + '/initial_state_capital_land_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_average_income = np.load(
#     path_simul + '/initial_state_average_income_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
# simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_limit_city = np.load(
#     path_simul + '/initial_state_limit_city_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')


# WELFARE DECOMPOSITON: write functions???

## Load variables

### Dwelling sizes

formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[3, :]

### Market rents

formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[3, :]

### Correct for backyard parameters

backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_housing_supply[1, :]/1000000

# NB: corresponds to baseline (correct in appropriate scenarios)
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = param["backyard_pockets"]
disam_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = param["backyard_pockets"]
disam_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = param["backyard_pockets"]
disam_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = param["backyard_pockets"]
disam_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = param["backyard_pockets"] + 0.1*(1-param["backyard_pockets"])
disam_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = param["backyard_pockets"] + 0.5*(1-param["backyard_pockets"])
disam_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = param["backyard_pockets"] + 1*(1-param["backyard_pockets"])
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = param["backyard_pockets"]
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = param["backyard_pockets"]
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = param["backyard_pockets"]
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = param["backyard_pockets"]

# NB: only works when all values are the same (non-location-specific calibration)
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = (
    np.nanmean(param["incremental_pockets"]))
disam_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = (
    np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = (
    np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"])) + 0.1*(1-np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"])) + 0.5*(1-np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"])) + 1*(1-np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"]))
disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1==2] = ( 
    np.nanmean(param["incremental_pockets"]))

disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = param["informal_pockets"]
disam_informal_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = param["informal_pockets"]
disam_informal_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = param["informal_pockets"]
disam_informal_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = param["informal_pockets"]
disam_informal_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = param["informal_pockets"] + 0.1*(1-param["informal_pockets"])
disam_informal_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = param["informal_pockets"] + 0.5*(1-param["informal_pockets"])
disam_informal_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = param["informal_pockets"] + 1*(1-param["informal_pockets"])
disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = param["informal_pockets"]
disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = param["informal_pockets"]
disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = param["informal_pockets"]
disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = param["informal_pockets"]

backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)] = 0

informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2])

informal_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = informal_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1
backyard_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = ( 
    incremental_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2])

informal_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = informal_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1
backyard_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = ( 
    incremental_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = informal_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1
backyard_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1==2])

informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 * param["backyard_size"] / param["shack_size"])
incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = (
    (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
    * backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 * param["backyard_size"] / param["RDP_size"])
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = informal_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1
backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1==2] = ( 
    incremental_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1==2])

### Adjust remaining parameters

subsid_income_net_of_commuting_costs = (
    income_net_of_commuting_costs[0]
    + param["subsidized_structure_value"]
    * (9*param["current_rate_public_housing"] + 21*param["future_rate_public_housing"])
    / (households_per_income_class[0]*population/sum(households_per_income_class))
    )

# Successively build changes upon baseline utility FOR EACH SCENARIO
# NB: maybe automatize the scenario part?

## USE NEW SPATIAL ALLOCATION FOR EACH VARIABLE TO BE UPDATED (when relevant?)
## Only works with average values, but DNF Jensen inequality: cannot go too much into details

### Define baseline

utility_formal_base = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_base = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_informal_base = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_base = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

#### NB: need to go together when using different allocation matrices (test it)
#### Do we really need allocation grids? Maybe just do extensive margin before adjusting variables (discuss sequence after test)
#### Advantage is that we can be more granular in changes of interest
#### NOTE THAT EXTENSIVE MARGIN IS ON BOTH LOCATION (3D with rent or just 1D with size and aspect, amenities, etc: or just plot distrib) AND HOUSING TYPE

# Idea of graphs is to go further in the analysis!

utility_base_mat = np.array([utility_formal_base, utility_backyard_base, utility_informal_base, utility_rdp_base])
base_utility_poor = (np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:])
                     / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:]))
base_utility_midpoor = (np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:])
                        / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:]))
base_utility_midrich =(np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:])
                       / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:]))
base_utility_rich = (np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:])
                     / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:]))

### No urban edge

#### Not sure if extensive effect mean something...

noUE_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:]))
noUE_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:]))
noUE_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:]))
noUE_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:]))


#### Intensive effects

utility_formal_noUE_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_noUE_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_informal_noUE_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_noUE_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_noUE_housing_eff_mat = np.array([
    utility_formal_noUE_housing_eff, utility_backyard_noUE_housing_eff, utility_informal_noUE_housing_eff, utility_rdp_noUE_housing_eff])
utility_poor_noUE_housing_eff = (
    np.nansum(utility_noUE_housing_eff_mat[:,0,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_noUE_housing_eff = (
    np.nansum(utility_noUE_housing_eff_mat[:,1,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_noUE_housing_eff = (
    np.nansum(utility_noUE_housing_eff_mat[:,2,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_noUE_housing_eff = (
    np.nansum(utility_noUE_housing_eff_mat[:,3,:]*simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:]))


### New IS

#### Not sure if extensive effect mean something...

newIS_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:]))
newIS_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:]))
newIS_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:]))
newIS_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_newIS_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_newIS_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_informal_newIS_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_newIS_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_newIS_housing_eff_mat = np.array([
    utility_formal_newIS_housing_eff, utility_backyard_newIS_housing_eff, utility_informal_newIS_housing_eff, utility_rdp_newIS_housing_eff])
utility_poor_newIS_housing_eff = (
    np.nansum(utility_newIS_housing_eff_mat[:,0,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_newIS_housing_eff = (
    np.nansum(utility_newIS_housing_eff_mat[:,1,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_newIS_housing_eff = (
    np.nansum(utility_newIS_housing_eff_mat[:,2,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_newIS_housing_eff = (
    np.nansum(utility_newIS_housing_eff_mat[:,3,:]*simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,3,:]))


### New RDP

#### Not sure if extensive effect mean something...

newRDP_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,0,:]))
newRDP_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,1,:]))
newRDP_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,2,:]))
newRDP_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_newRDP_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_newRDP_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_informal_newRDP_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_newRDP_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_newRDP_housing_eff_mat = np.array([
    utility_formal_newRDP_housing_eff, utility_backyard_newRDP_housing_eff, utility_informal_newRDP_housing_eff, utility_rdp_newRDP_housing_eff])
utility_poor_newRDP_housing_eff = (
    np.nansum(utility_newRDP_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_newRDP_housing_eff = (
    np.nansum(utility_newRDP_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_newRDP_housing_eff = (
    np.nansum(utility_newRDP_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_newRDP_housing_eff = (
    np.nansum(utility_newRDP_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,3,:]))
























#####

df_utility_changes = pd.DataFrame({
    "No UE": simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    "New IS": simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    "New RDP": simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility,
    "Upgrading": simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility,
    "Subsidies": simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility,
    "Eviction": simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility},
    index=["Poor", "Midpoor", "Midrich", "Rich"])

for col_name in df_utility_changes.columns:
    df_utility_changes[col_name] = (df_utility_changes[col_name]-simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility)/simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility

df_utility_changes.to_excel(path_output_tables + '/df_utility_changes.xlsx', float_format="%.3f")

# styled_df_utility_changes = df_utility_changes.style \
#   .format(precision=2, thousands=".", decimal=",") \
#   .format_index(str.upper, axis=1) \
#   .relabel_index(["Poor", "Mid-poor", "Mid-rich", "Rich"], axis=0)

# WELFARE DECOMPOSITION: add options to load ante and post variables



