# #########################################
# ## SIMPLIFIED SCRIPT FOR JUE REVISION ###
# #########################################

# WE DO NOT RE-RUN CALIBRATION

# WE THEN FOCUS ON COMPARATIVE STATICS FROM MAIN EQUILIBRIUM SCRIPTS

# ## We import standard Python libraries
import numpy as np
import pandas as pd
import os
import geopandas as gpd

# ## We also import our own packages
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import equilibrium.compute_equilibrium as eqcmp
import outputs.export_outputs as outexp


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

# ## Output folder
name = ('simul_UE' + str(options["urban_edge"]))
path_simul = path_outputs + name
path_output_plots = path_simul + '/plots/'
path_output_tables = path_simul + '/tables/'

try:
    os.mkdir(path_simul)
except OSError as error:
    print(error)
try:
    os.mkdir(path_output_plots)
except OSError as error:
    print(error)
try:
    os.mkdir(path_output_tables)
except OSError as error:
    print(error)

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

coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, 0)

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

print("Preamble done")

###############################################################################

# THEN WE VISUALIZE ONLY THE OUTPUT WE NEED

print('Running ' + name)

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

# ##Export raw output
np.save(path_simul + '/initial_state_utility.npy',
        initial_state_utility)
np.save(path_simul + '/initial_state_error.npy',
        initial_state_error)
np.save(path_simul + '/initial_state_simulated_jobs.npy',
        initial_state_simulated_jobs)
np.save(path_simul + '/initial_state_households_housing_types.npy',
        initial_state_households_housing_types)
np.save(path_simul + '/initial_state_household_centers.npy',
        initial_state_household_centers)
np.save(path_simul + '/initial_state_households.npy',
        initial_state_households)
np.save(path_simul + '/initial_state_dwelling_size.npy',
        initial_state_dwelling_size)
np.save(path_simul + '/initial_state_housing_supply.npy',
        initial_state_housing_supply)
np.save(path_simul + '/initial_state_rent.npy',
        initial_state_rent)
np.save(path_simul + '/initial_state_rent_matrix.npy',
        initial_state_rent_matrix)
np.save(path_simul + '/initial_state_capital_land.npy',
        initial_state_capital_land)
np.save(path_simul + '/initial_state_average_income.npy',
        initial_state_average_income)
np.save(path_simul + '/initial_state_limit_city.npy',
        initial_state_limit_city)

# ##Household density
sim_nb_households_tot = np.nansum(initial_state_households_housing_types, 0)

total_sim = outexp.export_map(
    sim_nb_households_tot, grid, geo_grid, path_output_plots, 'total_sim',
    "Total number of households, up to 99.99% quantile (simulation)",
    path_output_tables,
    ubnd=np.nanquantile(sim_nb_households_tot, 0.9999))

# Grid cell = 500x500m = 25 Ha
total_WP = outexp.discrete_map(
    sim_nb_households_tot/25, grid, geo_grid, path_output_plots, 'total_WP',
    "Nb of HHs per Ha (WP scale)", path_output_tables)



# NB: make scales comparable!

print('Running done')







##########################################

# NB: only try to debug unsatisfactory results

# Take care of code styling and optimization at replication phase

# Same for visualization: what matters first is result interpretability
