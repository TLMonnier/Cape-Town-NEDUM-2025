######################################
### OUTPUT SCRIPT FOR JUE REVISION ###
######################################

# PREAMBLE

## IMPORT LIBRARIES

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde

import inputs.data as inpdt
import inputs.parameters_and_options as inpprm

## DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/Data/'
path_precalc_inp = path_folder + 'precalculated_inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_data + 'Scenarios/'
path_outputs = path_code + '/Output/'
path_input_plots = path_outputs + 'input_plots/'
path_input_tables = path_outputs + 'input_tables/'

path_simul = path_outputs + 'revision_output'
path_output_plots = path_simul + '/plots/'
path_output_tables = path_simul + '/tables/'

## LOAD INPUTS

### PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(
    path_precalc_inp, options)

### GEOGRAPHIC INPUTS

grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp, options)
geo_grid = gpd.read_file(path_data + "grid_reference_500.shp")

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')

#### MACRO DATA

(interest_rate, population, housing_type_data, total_RDP, backyard_data
 ) = inpdt.import_macro_data(param, path_scenarios, path_folder)

### HH AND INCOME DATA

income_class_by_housing_type = inpdt.import_hypothesis_housing_type()
(mean_income, households_per_income_class, average_income, income_mult,
 income_baseline, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_baseline,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0

### LAND USE PROJECTIONS

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

# TODO: REMOVE FLOOD MODEL FROM JUE SCRIPT
    
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

# TODO: also reduce the above?

###############################################################################

# TODO: CODE FUNCTIONS ONLY WHEN NEED TO EDIT RELATIVELY SIMPLE COMMANDS BECOMES BURDENSOME
# OR TO PROOF-READ RESULTS! DO AT NEXT STAGE THEN ADD DO SIDE SCRIPT

def load_simulation_data(path_simul, data_type, simulation_name):

    filename = f'/initial_state_{data_type}_{simulation_name}.npy'
    return np.load(path_simul + filename)


def load_multiple_simulation_data(path_simul, data_types, simulation_names):

    results = {}
    
    if isinstance(data_types, str):
        data_types = [data_types] * len(simulation_names)
    
    for sim_name, data_type in zip(simulation_names, data_types):
        results[sim_name] = load_simulation_data(path_simul, data_type, sim_name)
    
    return results

simulation_configs = [
    'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1'
]

# PROCESS SIMULATION RESULTS

## LOAD DATA FOR ALL SIMULATIONS

utility_data = load_multiple_simulation_data(path_simul, 'utility', simulation_configs)
households_data = load_multiple_simulation_data(path_simul, 'households', simulation_configs)
dwelling_size_data = load_multiple_simulation_data(path_simul, 'dwelling_size', simulation_configs)
rent_data = load_multiple_simulation_data(path_simul, 'rent', simulation_configs)
housing_supply_data = load_multiple_simulation_data(path_simul, 'housing_supply', simulation_configs)

## ACCESS LOADED DATA

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[0]]
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[1]]
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[2]]
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[3]]
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[4]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility = utility_data[simulation_configs[5]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility = utility_data[simulation_configs[6]]

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = households_data[simulation_configs[0]]
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = households_data[simulation_configs[1]]
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = households_data[simulation_configs[2]]
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households = households_data[simulation_configs[3]]
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households = households_data[simulation_configs[4]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households = households_data[simulation_configs[5]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households = households_data[simulation_configs[6]]

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = dwelling_size_data[simulation_configs[0]]
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = dwelling_size_data[simulation_configs[1]]
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = dwelling_size_data[simulation_configs[2]]
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size = dwelling_size_data[simulation_configs[3]]
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size = dwelling_size_data[simulation_configs[4]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size = dwelling_size_data[simulation_configs[5]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size = dwelling_size_data[simulation_configs[6]]

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = rent_data[simulation_configs[0]]
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = rent_data[simulation_configs[1]]
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = rent_data[simulation_configs[2]]
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent = rent_data[simulation_configs[3]]
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent = rent_data[simulation_configs[4]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent = rent_data[simulation_configs[5]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent = rent_data[simulation_configs[6]]

simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply = housing_supply_data[simulation_configs[0]]
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply = housing_supply_data[simulation_configs[1]]
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply = housing_supply_data[simulation_configs[2]]
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_housing_supply = housing_supply_data[simulation_configs[3]]
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_housing_supply = housing_supply_data[simulation_configs[4]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_housing_supply = housing_supply_data[simulation_configs[5]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_housing_supply = housing_supply_data[simulation_configs[6]]

## ACCESS INDIVIDUAL VARIABLES FOR POST-PROCESSING

### DWELLING SIZE

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

formal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[3, :]

formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[0, :]
backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[1, :]
informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[2, :]
rdp_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[3, :]

### HOUSING RENTS

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

formal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[3, :]

formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[0, :]
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[1, :]
informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[2, :]
rdp_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[3, :]

### BACKYARD HOUSING SUPPLY

backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_housing_supply[1, :]/1000000
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_housing_supply[1, :]/1000000

## DISTINGUISH BETWEEN BASIC AND REDEVELOPED BACKYARDS

backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)] = 0
backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[np.isnan(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)] = 0
backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[np.isnan(backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)] = 0

###############################################################################

# SCENARIO DEFINITIONS

# TODO: When relevant, show visuals for scenario definition

# TODO: refine scenarios

# TODO: other plots for inputs and annex

# Subsidies: The total yearly amount to be spent by the local government on formal subsidized housing
# according to the City of Cape Town’s Housing Pipeline as contained in its Integrated Human Settlement
# Framework (2013) is redistributed as a direct transfer to low-income non-beneficiaries. The monetary
# value is obtained by multiplying the theoretical number of housing units to be built by the calibrated
# construction costs.

# New RDP: New formal subsidized housing units are built according to the expected completion of the
# City of Cape Town’s Housing Pipeline as of 2040 (conservative estimate). This corresponds to the
# construction of 65,000 units (roughly 2,200 per year) in specific sites, or an increase of the
# public housing supply by roughly 33%.

# New IS: All areas designated by the City of Cape Town as at risk of being squatted become
# available for informal settlements (generous estimate). Note that this only affects some
# fixed share of grid cell areas and does not preclude formal development from occuring at
# the same time. Nor does this imply that these newly available areas will all be settled 
# in equilibrium.

# No UE: The urban edge constraint in place at baseline year is relieved and formal
# development allowed to expand in the rural hinterland. In reality, such expansion
# should be accompanied by an extension of infrastructure networks to make it sustainable,
# but we do not model it and simply extrapolate the transporation costs estimated at
# baseline as a proof of concept.

# Disam. -50pc: The disamenity gap between different forms of informal housing and
# formal housing is reduced by 50%. This stands for aggressive slum upgrading programs
# that would be uniform across space.

# Toler. -50pc: The areas available for informal settlements are reduced by 50% and
# become available for formal private development. This stands for aggressive eviction
# or formalization policies that would be uniform across space.

###############################################################################

# TABLE 1: UTILITY CHANGES

# FOOTNOTE: The table shows the utility change rates per income group in the model for each scenario.
# The utility of the low-income group is split across non-beneficiaries (who share a common utility
# level determined in equilibrium) and beneficiaries (whose utilities are exogenous and heterogeneous
# across space) of formal subsidized housing. The reported values for the beneficiaries correspond
# to population weighted averages. The values reported in the final row are population weighted 
# averages aggregated for all income groups. (Baseline cardinal utility levels are also shown for reference.)

## ADJUST FOR THE DIFFERENCE IN CONSTRUCTION COST

def adjust_backyarding_costs(interest_rate, param, backyard_supply):
    
    informal_cost = (
        (interest_rate + param["depreciation_rate"]) * param["informal_structure_value"]
        * backyard_supply * param["backyard_size"] / param["shack_size"])
    incremental_cost = (
        (interest_rate + param["depreciation_rate"]) * param["subsidized_structure_value"]
        * backyard_supply * param["backyard_size"] / param["RDP_size"])
    backyard_cost = informal_cost
    backyard_cost[backyard_supply==2] = ( 
        incremental_cost[backyard_supply==2])
    
    return backyard_cost

backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = adjust_backyarding_costs(
    interest_rate, param, backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)

backyard_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = adjust_backyarding_costs(
    interest_rate, param, backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)

backyard_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = adjust_backyarding_costs(
    interest_rate, param, backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)

backyard_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = adjust_backyarding_costs(
    interest_rate, param, backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)

backyard_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = adjust_backyarding_costs(
    interest_rate, param, backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)

backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = adjust_backyarding_costs(
    interest_rate, param, backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)

backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = adjust_backyarding_costs(
    interest_rate, param, backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)

## WE COMPUTE THE EX-POST UTILITIES IN RDP TO GET AN AVERAGE ACROSS SPACE (NON-EQUILIBRIUM VALUES)

def compute_rdp_utility(income_net_of_commuting_costs, backyard_supply, param,
                        backyard_rent, backyard_cost, amenities):
    
    utility_rdp = (
        (income_net_of_commuting_costs
         + backyard_supply[None, :]
         * param["backyard_size"]*backyard_rent[None, :]
         - param["subsidized_structure_value"] * param["depreciation_rate"]
         - backyard_cost)**param["alpha"]
        * (param["RDP_size"] + param["backyard_size"] - param["q0"]
           - np.nanmin(backyard_supply[None, :],1)*param["backyard_size"])**param["beta"]
        * amenities[None, :]
        )
    
    return utility_rdp

utility_rdp_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = compute_rdp_utility(
    income_net_of_commuting_costs, backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, param,
    backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, amenities)

utility_rdp_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = compute_rdp_utility(
    income_net_of_commuting_costs, backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, param,
    backyard_rent_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, backyard_cost_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, amenities)

utility_rdp_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = compute_rdp_utility(
    income_net_of_commuting_costs, backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1, param,
    backyard_rent_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1, backyard_cost_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1, amenities)

utility_rdp_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = compute_rdp_utility(
    income_net_of_commuting_costs, backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1, param,
    backyard_rent_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1, backyard_cost_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1, amenities)

utility_rdp_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = compute_rdp_utility(
    income_net_of_commuting_costs, backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1, param,
    backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1, backyard_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1, amenities)

utility_rdp_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = compute_rdp_utility(
    income_net_of_commuting_costs, backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1, param,
    backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1, backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1, amenities)

utility_rdp_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = compute_rdp_utility(
    income_net_of_commuting_costs, backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1, param,
    backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1, backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1, amenities)

## WE CREATE THE ARRAYS USED IN FINAL TABLE

# TODO: Take care to later renaming of variable "population"!

def gen_new_utility_array(simul_households, simul_utility, utility_rdp, population):
    
    agg_households = np.nansum(simul_households,axis=(0,2))
    agg_households[0] = agg_households[0] - np.nansum(simul_households[3,:,:])
    avg_utility = (
        (np.nansum(simul_utility*agg_households)
         + np.nansum(simul_households[3,:,:]*utility_rdp[0, :]))
        / population)
    ext_utility = np.append(
        simul_utility,
        [np.nansum(simul_households[3,:,:]*utility_rdp[0, :])/np.nansum(simul_households[3,:,:]),
        avg_utility])
    
    return ext_utility


ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = gen_new_utility_array(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    utility_rdp_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, population)

ext_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = gen_new_utility_array(
    simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    utility_rdp_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1, population)

ext_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = gen_new_utility_array(
    simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    utility_rdp_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1, population)

ext_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility = gen_new_utility_array(
    simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households, simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility,
    utility_rdp_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1, population)

ext_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility = gen_new_utility_array(
    simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households, simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility,
    utility_rdp_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1, population)

ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility = gen_new_utility_array(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households, simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility,
    utility_rdp_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1, population)

ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility = gen_new_utility_array(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households, simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility,
    utility_rdp_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1, population)

## WE CREATE THE TABLE

df_utility_changes = pd.DataFrame({
    "Subsidies": ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility,
    "New FS": ext_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility,
    "New IS": ext_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    "No UE": ext_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    "Disam. -50pc": ext_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility,
    "Toler. -50pc": ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility},
    index=["Low inc. (no FS)", "Mid-low inc.", "Mid-high inc.", "High inc.", "Low inc. (FS)", "Weighted avg"])

for col_name in df_utility_changes.columns:
    df_utility_changes[col_name] = (df_utility_changes[col_name]-ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility)/ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility

df_utility_changes.to_excel(path_output_tables + '/df_utility_changes.xlsx', float_format="%.3f")

## WE ALSO CREATE A TABLE WITH BASELINE LEVELS FOR REFERENCE

df_utility_baseline = pd.DataFrame({
    "Baseline": ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility},
    index=["Low inc. (no FS)", "Mid-low inc.", "Mid-high inc.", "High inc.", "Low inc. (FS)", "Weighted avg"])

# TODO: check result

df_utility_changes_baseline = pd.concat([df_utility_baseline, df_utility_changes], axis=1)

df_utility_changes_baseline.to_excel(path_output_tables + '/df_utility_changes_baseline.xlsx', float_format="%.3f")

###############################################################################

# WE DEFINE MORE VARIABLES TO BE USED FOR PLOTTING

## WE START BY ADDING REDEVELOPED BACKYARDS TO MAIN OUTCOMES

def gen_new_simul_array(backyard_supply, simul_array,
                        hhs=False, rent=False, dsize=False):

    if hhs==True:    
        basic_backyard_households = np.copy(simul_array[1,:,:])
        increm_backyard_households = np.copy(simul_array[1,:,:])
        basic_backyard_households[backyard_supply==2] = 0
        increm_backyard_households[backyard_supply<2] = 0
        
        new_simul_array = np.array(
            [np.copy(simul_array[0,:,:]),
             basic_backyard_households,
             increm_backyard_households,
             np.copy(simul_array[2,:,:]),
             np.copy(simul_array[3,:,:])]
            )

    if rent==True:
        basic_backyard_households = np.copy(simul_array[1,:])
        increm_backyard_households = np.copy(simul_array[1,:])
        basic_backyard_households[backyard_supply==2] = 0
        increm_backyard_households[backyard_supply<2] = 0
        simul_array[3,:] = 0
        
        new_simul_array = np.array(
            [np.copy(simul_array[0,:]),
             basic_backyard_households,
             increm_backyard_households,
             np.copy(simul_array[2,:]),
             np.copy(simul_array[3,:])]
            )
    
    if dsize==True:
        basic_backyard_households = np.copy(simul_array[1,:])
        increm_backyard_households = np.copy(simul_array[1,:])
        basic_backyard_households[backyard_supply==2] = 0
        increm_backyard_households[backyard_supply<2] = 0
        simul_array[3,:] = param["RDP_size"] # + param["backyard_size"] 
        
        new_simul_array = np.array(
            [np.copy(simul_array[0,:]),
             basic_backyard_households,
             increm_backyard_households,
             np.copy(simul_array[2,:]),
             np.copy(simul_array[3,:])]
            )

    return new_simul_array

## WE LEVERAGE THE ABOVE TO OBTAIN NEW VARIABLES ACROSS SCENARIOS

def gen_new_simul_variables(backyard_supply, simul_households, simul_rent, simul_dwelling_size):

    new_backyard_supply = np.zeros((4,len(backyard_supply)))
    new_backyard_supply[0,:] = np.copy(backyard_supply)
    new_backyard_supply[1,:] = np.copy(backyard_supply)
    new_backyard_supply[2,:] = np.copy(backyard_supply)
    new_backyard_supply[3,:] = np.copy(backyard_supply)
    
    new_simul_households = gen_new_simul_array(
        new_backyard_supply, simul_households, hhs=True)
    
    new_simul_rent = gen_new_simul_array(
        backyard_supply, simul_rent, rent=True)
    
    new_simul_dwelling_size = gen_new_simul_array(
        backyard_supply, simul_dwelling_size, dsize=True)

    return new_simul_households, new_simul_rent, new_simul_dwelling_size

## WE DECLINE THE FUNCTIONS

(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
 new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
 new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size) = gen_new_simul_variables(
        backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size)
     
(new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
 new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
 new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size) = gen_new_simul_variables(
        backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1,
        simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
        simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
        simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size)
     
(new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
 new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
 new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size) = gen_new_simul_variables(
        backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1,
        simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
        simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
        simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size)
     
(new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
 new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent,
 new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size) = gen_new_simul_variables(
        backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1,
        simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
        simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent,
        simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size)
     
(new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households,
 new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent,
 new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size) = gen_new_simul_variables(
        backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1,
        simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households,
        simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent,
        simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size)
     
(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
 new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent,
 new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size) = gen_new_simul_variables(
        backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size)
     
(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households,
 new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent,
 new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size) = gen_new_simul_variables(
        backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent,
        simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size)

## WE ALSO REARRANGE INCOMES TO HAVE A VALUE PER HOUSING TYPE (NOT JUST GROUP)

formal_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
basic_backyard_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
redev_backyard_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
informal_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
rdp_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
for i in range(4):
    formal_income_net_of_commuting_costs[i, :][
        new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0,i,:]==0] = 0
    basic_backyard_income_net_of_commuting_costs[i, :][
        new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[1,i,:]==0] = 0
    redev_backyard_income_net_of_commuting_costs[i, :][
        new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[2,i,:]==0] = 0
    informal_income_net_of_commuting_costs[i, :][
        new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,i,:]==0] = 0
    rdp_income_net_of_commuting_costs[i, :][
        new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[4,i,:]==0] = 0

formal_income_net_of_commuting_costs = np.nansum(formal_income_net_of_commuting_costs, 0)
basic_backyard_income_net_of_commuting_costs = np.nansum(basic_backyard_income_net_of_commuting_costs, 0)
redev_backyard_income_net_of_commuting_costs = np.nansum(redev_backyard_income_net_of_commuting_costs, 0)
informal_income_net_of_commuting_costs = np.nansum(informal_income_net_of_commuting_costs, 0)
rdp_income_net_of_commuting_costs = np.nansum(rdp_income_net_of_commuting_costs, 0)

new_income_net_of_commuting_costs = np.array(
    [formal_income_net_of_commuting_costs,
     basic_backyard_income_net_of_commuting_costs,
     redev_backyard_income_net_of_commuting_costs,
     informal_income_net_of_commuting_costs,
     rdp_income_net_of_commuting_costs]
    )

## WE ALSO RESHAPE AMENITIES FOR PLOTTING

new_amenities = np.tile(amenities, (5,1))

# TODO: ALWAYS THE QUESTION OF VISIBILITY VS AXIS COMAPRISONS

# TODO: ALSO NORM CONSISTENCY ACROSS PLOTS

###############################################################################

# FIGURE 1: HH BREAKDOWN PER HOUSING TYPE

# FOOTNOTE: The figure shows the breakdown of the number households across housing type for each scenario.
# The detail is only shown for the two poor income groups, as the rich two income groups only sort into
# the formal private sector. The breakdown for the population as a whole is shown for reference.
# The y axes are not the same.

## WE STORE THE VALUES TO BE PLOTTED

scenario_data = {
    'Baseline': np.vstack([np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0:2,:], 2).T,
                           np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, (1,2)).T]),
    'Subsidies': np.vstack([np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,0:2,:], 2).T,
                            np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households, (1,2)).T]),
    'New RDP': np.vstack([np.nansum(new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[:,0:2,:], 2).T,
                          np.nansum(new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households, (1,2)).T]),
    'New IS': np.vstack([np.nansum(new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0:2,:], 2).T,
                         np.nansum(new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, (1,2)).T]),
    'No UE': np.vstack([np.nansum(new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[:,0:2,:] ,2).T,
                        np.nansum(new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, (1,2)).T]),
    'Disam. -50pc': np.vstack([np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,0:2,:], 2).T,
                               np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households, (1,2)).T]),
    'Toler. -50pc': np.vstack([np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,0:2,:], 2).T,
                               np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households, (1,2)).T]),
}

## WE DEAL WITH THE AXES DEFINITION

income_groups = ['Low inc.', 'Mid-low inc.', 'All']
housing_types = ['FP', 'IB (basic)', 'IB (redev.)', 'IS', 'FS']
scenarios = list(scenario_data.keys())

## WE WRITE STACKED BARPLOT FUNCTION

fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.suptitle('Distribution of HHs across housing types by scenario', 
             fontsize=16, fontweight='bold', y=0.995)

colors = plt.cm.Set3(np.linspace(0, 1, 5))

for i, income_group in enumerate(income_groups):
    ax = axes[i]
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    bottom = np.zeros(len(scenarios))
    
    for j, housing_type in enumerate(housing_types):
        values = [scenario_data[scenario][i, j] for scenario in scenarios]
        
        ax.bar(x, values, width, label=housing_type, bottom=bottom, color=colors[j])
        bottom += values

    ax.set_ylabel('Number of HHs', fontsize=11, fontweight='bold')
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title(f'{income_group}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if i == 0:
        ax.legend(title='Housing types', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()

plt.savefig(path_output_plots + '/htype_breakdown.png')

plt.show()

###############################################################################

# FIGURES 2-5: BASELINE DISTRIBUTIONS ACROSS KEY VARIABLES

# TODO: Also do aggregate graphs? Show income or htype? Can do both..

# WE DEFINE THE MASTER PLOT FUNCTION

def plot_baseline_distrib_per_incgrp(new_simul_array, new_simul_households,
                                     xlabel, xvar_name, end_option="percentile"):

    n_locations = 24014
    n_housing_types = 5
    n_income_groups = 4

    array = new_simul_array
    population = new_simul_households

    housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]
    income_group_labels = ["Low inc.", "Mid-low inc.", "Mid-high inc.", "High inc."]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for income_idx in range(n_income_groups):
        ax = axes[income_idx]
        
        start = min(array[population[:,income_idx,:]>0])
        
        if end_option=="percentile":
            end = np.nanquantile(array[population[:,income_idx,:]>0], 0.99)
        if end_option=="max":
            end = max(array[population[:,income_idx,:]>0]) + 1
        
        array_bins = np.arange(start, end, (end-start)/100)
        bin_centers = (array_bins[:-1] + array_bins[1:]) / 2

        binned_pop = np.zeros((len(array_bins)-1, n_housing_types))

        for housing_idx in range(n_housing_types):
            for loc_idx in range(n_locations):
                array_value = array[housing_idx, loc_idx]
                pop_value = population[housing_idx, income_idx, loc_idx]
                
                bin_idx = np.digitize(array_value, array_bins) - 1
                if 0 <= bin_idx < len(array_bins) - 1:
                    binned_pop[bin_idx, housing_idx] += pop_value

        bottom = np.zeros(len(array_bins)-1)
        colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
        
        for housing_idx in range(n_housing_types):
            ax.bar(bin_centers, binned_pop[:, housing_idx], 
                   width=array_bins[1]-array_bins[0], 
                   bottom=bottom,
                   label=housing_type_labels[housing_idx],
                   color=colors[housing_idx],
                   edgecolor='white',
                   linewidth=0.5)
            bottom += binned_pop[:, housing_idx]
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Nb of HHs', fontsize=10)
        ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_output_plots + f'/baseline_{xvar_name}_pop_distrib.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return

def plot_baseline_distrib_agg(new_simul_array, new_simul_households,
                              xlabel, xvar_name, end_option="percentile"):

    n_locations = 24014
    n_housing_types = 5

    array = new_simul_array
    population = new_simul_households

    housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]

    fig, ax = plt.subplots(1, figsize=(14, 10))
    
    start = min(array[population>0])
    
    if end_option=="percentile":
        end = np.nanquantile(array[population>0], 0.99)
    if end_option=="max":
        end = max(array[population>0]) + 1
    
    array_bins = np.arange(start, end, (end-start)/100)
    bin_centers = (array_bins[:-1] + array_bins[1:]) / 2

    binned_pop = np.zeros((len(array_bins)-1, n_housing_types))

    for housing_idx in range(n_housing_types):
        for loc_idx in range(n_locations):
            array_value = array[housing_idx, loc_idx]
            pop_value = population[housing_idx, loc_idx]
            
            bin_idx = np.digitize(array_value, array_bins) - 1
            if 0 <= bin_idx < len(array_bins) - 1:
                binned_pop[bin_idx, housing_idx] += pop_value

    bottom = np.zeros(len(array_bins)-1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
    
    for housing_idx in range(n_housing_types):
        ax.bar(bin_centers, binned_pop[:, housing_idx], 
               width=array_bins[1]-array_bins[0], 
               bottom=bottom,
               label=housing_type_labels[housing_idx],
               color=colors[housing_idx],
               edgecolor='white',
               linewidth=0.5)
        bottom += binned_pop[:, housing_idx]
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Nb of HHs')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(path_output_plots + f'/baseline_agg_{xvar_name}_pop_distrib.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return

# WE DECLINE THE FUNCTION ACROSS KEY VARIABLES

## FIGURE 2: RENT

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across housing rent brackets and housing type for each income group. The rents are given in South African
## rands per squared meter per year at 2011 values. Rent brackets are computed so as to split the total
## distribution in 100 and go up to the 99th percentile. The rent for formal subsidized housing is taken as zero.
## The x and y axes are not the same across income groups.

plot_baseline_distrib_per_incgrp(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Annual rent (ZAR/m²)", "rent")

## FIGURE 2: INCOME

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across expected income brackets and housing type for each income group. The incomes are
## net of commuting costs and are given in South African rands per year at 2011 values.
## Income brackets are computed so as to split the total distribution in 100
## and go up to the 99th percentile. The expected income does not include potential added revenues
## from supplying backyard housing or added expenses from construction or maintenance costs.
## It is calibrated at baseline and reflects the intrinsic value of a residential location.
## The x and y axes are not the same across income groups.

plot_baseline_distrib_per_incgrp(
    new_income_net_of_commuting_costs,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Expected income net of commuting costs (ZAR/year)", "income")

## FIGURE 3: DWELLING SIZE

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across dwelling size brackets and housing type for each income group. The dwelling sizes
## are given in squared meters. Size brackets are computed so as to split the total distribution
## in 100 and go up to the maximum value. The dwelling size for formal subsidized housing does not
## include the backyard size and refers only to the size of the dwelling unit itself.
## The x and y axes are not the same across income groups.

plot_baseline_distrib_per_incgrp(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Dwelling size (m²)", "dsize", end_option="max")

## FIGURE 4: AMENITY INDEX

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across amenity brackets and housing type for each income group. The amenity index
## is calibrated at baseline and normalized around one. It  reflects the intrinsic value
## of a residential location. Amenity brackets are computed so as to split the total distribution
## in 100 and go up to the 99th percentile. The x and y axes are not the same across income groups.

plot_baseline_distrib_per_incgrp(
    new_amenities,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Amenity index", "amenity")

## ALSO DO AGGREGATE PLOTS JUST IN CASE

plot_baseline_distrib_agg(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Annual rent (ZAR/m²)", "rent")

plot_baseline_distrib_agg(
    new_income_net_of_commuting_costs,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Expected income net of commuting costs (ZAR/year)", "income")

plot_baseline_distrib_agg(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Dwelling size (m²)", "dsize", end_option="max")

plot_baseline_distrib_agg(
    new_amenities,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Amenity index", "amenity")




########## NOW FOR SCENARIOS

#### Rent

# Example: 5 scenarios, 1000 locations, 5 housing types, 4 income groups
n_scenarios = 5
n_locations = np.shape(new_amenities)[1]
n_housing_types = 5
n_income_groups = 4

# Generate sample data (replace with your actual arrays)

rent = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent])

population = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households])

# Number of points for smooth curve
n_points = 100  # Adjust for smoother/coarser curves

# Labels
scenario_labels = ["Baseline", "Subsidies", "New RDP", "New IS", "No UE"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Colors for scenarios
colors = plt.cm.tab10(np.linspace(0, 1, n_scenarios))

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [3]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # for scenario_idx in [0]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            for housing_idx in range(n_housing_types):
                r = rent[scenario_idx, housing_idx, loc_idx]
                p = population[scenario_idx, housing_idx, income_idx, loc_idx]
                # DOUBLE COUNTING DOES NOT APPEAR AS THERE IS ONLY ONE INCOME GROUP PER HOUSING TYPE IN EACH LOCATION!
                if p > 0:  # Only include locations with population
                    rent_values.append(r)
                    pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                # Use Gaussian KDE for smooth distribution
                kde = gaussian_kde(rent_weighted, bw_method='scott')
                
                # Create smooth x-axis range
                rent_min = rent_values.min()
                
                # rent_max = rent_values.max()
                # rent_max = rent_values[pop_values>1000].max()
                # rent_max = np.nanquantile(rent_values, 0.99)
                
                if income_idx==0:
                    rent_max = 200
                elif income_idx==1:
                    rent_max = 500
                elif income_idx==2:
                    rent_max = 1000
                elif income_idx==3:
                    rent_max = 1400
                
                rent_range = np.linspace(rent_min, rent_max, n_points)
                
                # Evaluate KDE and scale by total population
                density = kde(rent_range)
                # density_scaled = density * pop_values.sum()
                density_scaled = density
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Rent (ZAR/year)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_rent_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!


#### Income

# Example: 5 scenarios, 1000 locations, 5 housing types, 4 income groups
n_scenarios = 5
n_locations = np.shape(new_amenities)[1]
n_housing_types = 5
n_income_groups = 4

# Generate sample data (replace with your actual arrays)

rent = np.stack([
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs])

population = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households])

# Number of points for smooth curve
n_points = 100  # Adjust for smoother/coarser curves

# Labels
scenario_labels = ["Baseline", "Subsidies", "New RDP", "New IS", "No UE"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Colors for scenarios
colors = plt.cm.tab10(np.linspace(0, 1, n_scenarios))

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [3]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # for scenario_idx in [0]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            for housing_idx in range(n_housing_types):
                r = rent[scenario_idx, income_idx, loc_idx]
                p = population[scenario_idx, housing_idx, income_idx, loc_idx]
                # NO DOUBLE COUNTING
                if p > 0:  # Only include locations with population
                    rent_values.append(r)
                    pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                # Use Gaussian KDE for smooth distribution
                kde = gaussian_kde(rent_weighted, bw_method='scott')
                
                # Create smooth x-axis range
                rent_min = rent_values.min()
                rent_max = rent_values.max()

                # rent_max = rent_values[pop_values>1000].max()
                # rent_max = np.nanquantile(rent_values, 0.99)
                
                if income_idx==0:
                    rent_min = 13000
                    rent_max = 19000
                elif income_idx==1:
                    rent_max = 55000
                elif income_idx==2:
                    rent_max = 168000
                elif income_idx==3:
                    rent_min = 755000
                
                rent_range = np.linspace(rent_min, rent_max, n_points)
                
                # Evaluate KDE and scale by total population
                density = kde(rent_range)
                # density_scaled = density * pop_values.sum()
                density_scaled = density
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Income net of commuting (ZAR/year)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_income_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!


#### Amenities (double counting??) Takes longer with correction? Different results?

# Example: 5 scenarios, 1000 locations, 5 housing types, 4 income groups
n_scenarios = 5
n_locations = np.shape(new_amenities)[1]
# n_housing_types = 5
n_income_groups = 4

# Generate sample data (replace with your actual arrays)

rent = np.stack([
    amenities,
    amenities,
    amenities,
    amenities,
    amenities])

population = np.stack([
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 0),
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households, 0),
    np.nansum(new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households, 0),
    np.nansum(new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 0),
    np.nansum(new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 0)])

# Number of points for smooth curve
# MAKES IT EASIER TO VISUALIZE JUMPS IN DISTRIBUTION?
n_points = 100  # Adjust for smoother/coarser curves

# Labels
scenario_labels = ["Baseline", "Subsidies", "New RDP", "New IS", "No UE"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Colors for scenarios
colors = plt.cm.tab10(np.linspace(0, 1, n_scenarios))

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [3]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # for scenario_idx in [0]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            # for housing_idx in range(n_housing_types):
            r = rent[scenario_idx, loc_idx]
            p = population[scenario_idx, income_idx, loc_idx]
            # NO DOUBLE COUNTING? Doesn't change anything
            if p > 0:  # Only include locations with population
                rent_values.append(r)
                pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                # Use Gaussian KDE for smooth distribution
                kde = gaussian_kde(rent_weighted, bw_method='scott')
                
                # Create smooth x-axis range
                rent_min = rent_values.min()
                rent_max = rent_values.max()

                # rent_max = rent_values[pop_values>1000].max()
                # rent_max = np.nanquantile(rent_values, 0.99)
                
                if income_idx==0:
                    rent_max = 1.2
                elif income_idx==1:
                    rent_max = 1.2
                elif income_idx==2:
                    rent_max = 1.2
                elif income_idx==3:
                    rent_min = 0.9
                    rent_max = 1.3
                
                rent_range = np.linspace(rent_min, rent_max, n_points)
                
                # Evaluate KDE and scale by total population
                density = kde(rent_range)
                # density_scaled = density * pop_values.sum()
                density_scaled = density
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Amenity index', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_amenity_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!


#### Dwelling sizes

# Example: 5 scenarios, 1000 locations, 5 housing types, 4 income groups
n_scenarios = 5
n_locations = np.shape(new_amenities)[1]
n_housing_types = 5
n_income_groups = 4

# Generate sample data (replace with your actual arrays)

rent = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size])

population = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households])

# Number of points for smooth curve
n_points = 100  # Adjust for smoother/coarser curves

# Labels
scenario_labels = ["Baseline", "Subsidies", "New RDP", "New IS", "No UE"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Colors for scenarios
colors = plt.cm.tab10(np.linspace(0, 1, n_scenarios))

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [1]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # Mass probability for dwelling size for midpoor in new RDP scenario!
    # for scenario_idx in [2]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            for housing_idx in range(n_housing_types):
                r = rent[scenario_idx, housing_idx, loc_idx]
                p = population[scenario_idx, housing_idx, income_idx, loc_idx]
                if p > 0:  # Only include locations with population
                    rent_values.append(r)
                    pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                # Use Gaussian KDE for smooth distribution
                kde = gaussian_kde(rent_weighted, bw_method='scott')
                
                # Create smooth x-axis range
                rent_min = rent_values.min()
                rent_max = rent_values.max()

                # rent_max = rent_values[pop_values>1000].max()
                # rent_max = np.nanquantile(rent_values, 0.99)
                
                # if income_idx==0:
                #     rent_max = 1.2
                # elif income_idx==1:
                #     rent_max = 1.2
                # elif income_idx==2:
                #     rent_max = 1.2
                # elif income_idx==3:
                #     rent_min = 0.9
                #     rent_max = 1.3
                
                rent_range = np.linspace(rent_min, rent_max, n_points)
                
                # Evaluate KDE and scale by total population
                density = kde(rent_range)
                # density_scaled = density * pop_values.sum()
                density_scaled = density
                
                # Ad hoc correction
                if scenario_idx==2 and income_idx==1:
                    density_scaled[density_scaled>4] = 4
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Dwelling sizes (m²)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_dsize_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!


###### MORE SCENARIOS

#### Rent

# Example: 5 scenarios, 1000 locations, 5 housing types, 4 income groups
n_scenarios = 7
n_locations = np.shape(new_amenities)[1]
n_housing_types = 5
n_income_groups = 4

# Generate sample data (replace with your actual arrays)

rent = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent])

population = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households])

# Number of points for smooth curve
n_points = 100  # Adjust for smoother/coarser curves

# Labels
scenario_labels = ["Baseline", "Disam. -10pc", "Disam. -50pc", "Disam. -100pc", "Toler. -10pc", "Toler. -50pc", "Toler. -100pc"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Colors for scenarios
colors = plt.cm.tab10(np.linspace(0, 1, n_scenarios))

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [3]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # for scenario_idx in [0]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            for housing_idx in range(n_housing_types):
                r = rent[scenario_idx, housing_idx, loc_idx]
                p = population[scenario_idx, housing_idx, income_idx, loc_idx]
                # DOUBLE COUNTING DOES NOT APPEAR AS THERE IS ONLY ONE INCOME GROUP PER HOUSING TYPE IN EACH LOCATION!
                if p > 0:  # Only include locations with population
                    rent_values.append(r)
                    pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                # Use Gaussian KDE for smooth distribution
                kde = gaussian_kde(rent_weighted, bw_method='scott')
                
                # Create smooth x-axis range
                rent_min = rent_values.min()
                rent_max = rent_values.max()
                # rent_max = rent_values[pop_values>1000].max()
                # rent_max = np.nanquantile(rent_values, 0.99)
                
                if income_idx==0:
                    rent_max = 400
                elif income_idx==1:
                    rent_max = 800
                elif income_idx==2:
                    rent_max = 1000
                elif income_idx==3:
                    rent_max = 2500
                
                rent_range = np.linspace(rent_min, rent_max, n_points)
                
                # Evaluate KDE and scale by total population
                density = kde(rent_range)
                # density_scaled = density * pop_values.sum()
                density_scaled = density
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Rent (ZAR/year)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_rent_pop_distrib_upgrading.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!


#### Income

# Generate sample data (replace with your actual arrays)

rent = np.stack([
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs,
    income_net_of_commuting_costs])

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [3]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # for scenario_idx in [0]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            for housing_idx in range(n_housing_types):
                r = rent[scenario_idx, income_idx, loc_idx]
                p = population[scenario_idx, housing_idx, income_idx, loc_idx]
                # NO DOUBLE COUNTING
                if p > 0:  # Only include locations with population
                    rent_values.append(r)
                    pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                # Use Gaussian KDE for smooth distribution
                kde = gaussian_kde(rent_weighted, bw_method='scott')
                
                # Create smooth x-axis range
                rent_min = rent_values.min()
                rent_max = rent_values.max()

                # rent_max = rent_values[pop_values>1000].max()
                # rent_max = np.nanquantile(rent_values, 0.99)
                
                # if income_idx==0:
                #     rent_min = 13000
                #     rent_max = 19000
                # elif income_idx==1:
                #     rent_max = 55000
                # elif income_idx==2:
                #     rent_max = 168000
                # elif income_idx==3:
                #     rent_min = 755000
                
                rent_range = np.linspace(rent_min, rent_max, n_points)
                
                # Evaluate KDE and scale by total population
                density = kde(rent_range)
                # density_scaled = density * pop_values.sum()
                density_scaled = density
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Income net of commuting (ZAR/year)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_income_pop_distrib_upgrading.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!


#### Amenities (double counting??) Takes longer with correction? Different results?

rent = np.stack([
    amenities,
    amenities,
    amenities,
    amenities,
    amenities,
    amenities,
    amenities])

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [3]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # for scenario_idx in [0]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            for housing_idx in range(n_housing_types):
                r = rent[scenario_idx, loc_idx]
                p = population[scenario_idx, housing_idx, income_idx, loc_idx]
                # NO DOUBLE COUNTING? Doesn't change anything
                if p > 0:  # Only include locations with population
                    rent_values.append(r)
                    pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                # Use Gaussian KDE for smooth distribution
                kde = gaussian_kde(rent_weighted, bw_method='scott')
                
                # Create smooth x-axis range
                rent_min = rent_values.min()
                rent_max = rent_values.max()

                # rent_max = rent_values[pop_values>1000].max()
                # rent_max = np.nanquantile(rent_values, 0.99)
                
                # if income_idx==0:
                #     rent_max = 1.2
                # elif income_idx==1:
                #     rent_max = 1.2
                # elif income_idx==2:
                #     rent_max = 1.2
                # elif income_idx==3:
                #     rent_min = 0.9
                #     rent_max = 1.3
                
                rent_range = np.linspace(rent_min, rent_max, n_points)
                
                # Evaluate KDE and scale by total population
                density = kde(rent_range)
                # density_scaled = density * pop_values.sum()
                density_scaled = density
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Amenity index', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_amenity_pop_distrib_upgrading.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!


#### Dwelling sizes


from numpy.linalg import LinAlgError

# Generate sample data (replace with your actual arrays)

rent = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size])


# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# for income_idx in [1]:
for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # Plot distribution for each scenario
    # Mass probability for dwelling size for midpoor in new RDP scenario!
    # for scenario_idx in [0,1,2,4,5,6]:
    for scenario_idx in range(n_scenarios):
        # Collect all rent-population pairs for this scenario and income group
        rent_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            # for housing_idx in [0]:
            for housing_idx in range(n_housing_types):
                r = rent[scenario_idx, housing_idx, loc_idx]
                p = population[scenario_idx, housing_idx, income_idx, loc_idx]
                if p > 0:  # Only include locations with population
                    rent_values.append(r)
                    pop_values.append(p)
        
        if len(rent_values) > 0:
            rent_values = np.array(rent_values)
            pop_values = np.array(pop_values)
            
            # Create smooth continuous distribution using KDE
            # Repeat rent values weighted by population for KDE
            rent_weighted = np.repeat(rent_values, pop_values.astype(int))
            
            if len(rent_weighted) > 1:
                
                try:
                
                    # Use Gaussian KDE for smooth distribution
                    kde = gaussian_kde(rent_weighted, bw_method='scott')
                    
                    # Create smooth x-axis range
                    rent_min = rent_values.min()
                    rent_max = rent_values.max()
    
                    # rent_max = rent_values[pop_values>1000].max()
                    # rent_max = np.nanquantile(rent_values, 0.99)
                    
                    # if income_idx==0:
                    #     rent_max = 1.2
                    # elif income_idx==1:
                    #     rent_max = 1.2
                    # elif income_idx==2:
                    #     rent_max = 1.2
                    # elif income_idx==3:
                    #     rent_min = 0.9
                    #     rent_max = 1.3
                    
                    rent_range = np.linspace(rent_min, rent_max, n_points)
                    
                    # Evaluate KDE and scale by total population
                    density = kde(rent_range)
                    # density_scaled = density * pop_values.sum()
                    density_scaled = density
                
                except LinAlgError:
                
                    # Ad hoc correction
                    if scenario_idx==3 and income_idx==1:
                        density_scaled[density_scaled>1.2] = 1.2
                
                # Plot smooth curve with filled area
                ax.plot(rent_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=1,
                       alpha=0.9)
                
                # Add semi-transparent fill
                ax.fill_between(rent_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    # Formatting
    ax.set_xlabel('Dwelling sizes (m²)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='both', alpha=0.3)
    
plt.tight_layout()
plt.savefig(path_output_plots + '/scenarios_dsize_pop_distrib_upgrading.png', dpi=300, bbox_inches='tight')
plt.show()

# FOR RICH AND MIDRICH, BASELINE IS INDISTINGUISHIBLE FROM SUBSIDIES!!!

# TODO: incorporate increased amenities or subsidy revenues in plots?


# TODO: multiply amentiies by disamenities and add backyard revenues and construction costs to income?

# TODO: Write plot and table functions!!! DNF appropriate options

# TODO: need one average graph for all income groups??? Maybe just for maps?
# But then, not very clear what is happening...

#################################

### NOW DEAL WITH SPACE (baseline then scenarios)

#################################

##### BASELINE

# CLAUDE CODE
# I have an array giving me housing rents in thousands of locations for 5 housing types and an array giving me the population in each location for each housing type across 4 income groups. I also have a geodataframe with geometry for each location. Write the Python code to plot 3D maps for each income group, where in each location bar height represents population and bar color intensity represents housing rent (averaged across housing types with populations in each category used as weights). Add a map background with real geographic information beyond polygon boundaries.

# Let us start with something simpler!
# I have an array giving me housing rents for 5 housing types  in thousands of locations and an array giving me the population for each housing type across 4 income groups in each location. I also have a geodataframe with geometry for each location. Write the Python code to plot 3D maps for each income group, where in each location bar height represents population and bar color intensity represents housing rent (averaged across housing types with populations in each category used as weights).


###

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as ctx
from pyproj import Transformer


gdf = gpd.read_file(path_data + "grid_reference_500.shp")
# gdf = gdf.to_crs('EPSG:4326')

population_array = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households
rent_array = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent

def calculate_weighted_average_rent(population_array, rent_array, income_idx):
    """
    Calculate weighted average rent for a specific income group across housing types
    
    Parameters:
    -----------
    population_array : ndarray
        Array of shape (n_locations, n_housing_types, n_income_groups)
    rent_array : ndarray
        Array of shape (n_locations, n_housing_types) - rent for each housing type
    income_idx : int
        Index of the income group to calculate for
        
    Returns:
    --------
    weighted_rents : ndarray
        Array of shape (n_locations,) with weighted average rent per location
    """
    # Get population for this income group: (n_locations, n_housing_types)
    pop_income = population_array[:, income_idx, :]
    
    # Calculate weighted average rent for each location
    weighted_rents = np.zeros(population_array.shape[2])
    
    for loc_idx in range(population_array.shape[2]):
        total_pop = pop_income[:, loc_idx].sum()
        if total_pop > 0:
            # Weighted average: sum(rent * population) / sum(population)
            weighted_rents[loc_idx] = np.sum(rent_array[:, loc_idx] * pop_income[:, loc_idx]) / total_pop
        else:
            weighted_rents[loc_idx] = 0  # or np.nan if you prefer
    
    return weighted_rents


def add_basemap_to_3d(ax, gdf, alpha=0.3):
    """
    Add a basemap as a surface at z=0 in the 3D plot
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The 3D axis to add the basemap to
    gdf : GeoDataFrame
        GeoDataFrame to determine bounds (should be in EPSG:4326)
    alpha : float
        Transparency of the basemap (0-1)
    """
    # Get bounds in lat/lon (EPSG:4326)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] in lon/lat
    
    # Add some padding to bounds (5%)
    # lon_range = bounds[2] - bounds[0]
    # lat_range = bounds[3] - bounds[1]
    
    
    # SET MANUALLY???
    bounds_padded = [
        bounds[0], #- lon_range * 0.0001,
        bounds[1], #- lat_range * 0.0001,
        bounds[2], #+ lon_range * 0.0001,
        bounds[3] #+ lat_range * 0.0001
    ]
    
    
    # Bounds for display (with desired padding)
    # display_bounds = [
    #     bounds[0],
    #     bounds[1],
    #     bounds[2],
    #     bounds[3]
    # ]
    
    # Convert bounds to Web Mercator for fetching tiles
    transformer_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    # xmin_merc, ymin_merc = transformer_to_merc.transform(bounds, bounds)
    # xmax_merc, ymax_merc = transformer_to_merc.transform(bounds, bounds)
    xmin_merc, ymin_merc = transformer_to_merc.transform(bounds_padded[0], bounds_padded[1])
    xmax_merc, ymax_merc = transformer_to_merc.transform(bounds_padded[2], bounds_padded[3])
    
    # Fetch basemap tiles in Web Mercator
    try:
        basemap, extent_merc = ctx.bounds2img(
            xmin_merc, ymin_merc, xmax_merc, ymax_merc,
            ll=False,  # Not lat/lon, already in Web Mercator
            source=ctx.providers.CartoDB.Positron,  # Light basemap
            zoom='auto'
        )
        
        # extent_merc is [left, right, bottom, top] in Web Mercator
        # We need to convert back to lat/lon for plotting
        
        # Create a grid in Web Mercator space
        n_points_x = basemap.shape[1]
        n_points_y = basemap.shape[0]
        
        x_merc = np.linspace(extent_merc[0], extent_merc[1], n_points_x)
        y_merc = np.linspace(extent_merc[2], extent_merc[3], n_points_y)
        
        # Convert grid points back to lat/lon
        transformer_to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        
        # Create meshgrid and convert each point
        X_latlon = np.zeros((n_points_y, n_points_x))
        Y_latlon = np.zeros((n_points_y, n_points_x))
        
        for i in range(n_points_y):
            for j in range(n_points_x):
                lon, lat = transformer_to_latlon.transform(x_merc[j], y_merc[i])
                X_latlon[i, j] = lon
                Y_latlon[i, j] = lat
        
        # # Crop to display bounds
        # mask_x = (X_latlon[0, :] >= display_bounds[0]) & (X_latlon[0, :] <= display_bounds[2])
        # mask_y = (Y_latlon[:, 0] >= display_bounds[1]) & (Y_latlon[:, 0] <= display_bounds[3])
        
        # # Find indices for cropping
        # x_indices = np.where(mask_x)[0]
        # y_indices = np.where(mask_y)[0]
        
       
        # if len(x_indices) > 0 and len(y_indices) > 0:
        #     x_start, x_end = x_indices[0], x_indices[-1] + 1
        #     y_start, y_end = y_indices[0], y_indices[-1] + 1
            
        #     # Crop all arrays
        #     X_latlon_cropped = X_latlon[y_start:y_end, x_start:x_end]
        #     Y_latlon_cropped = Y_latlon[y_start:y_end, x_start:x_end]
        #     basemap_cropped = basemap[y_start:y_end, x_start:x_end]
        #     Z_cropped = np.zeros_like(X_latlon_cropped)
            
        #     print(f"Cropped basemap shape: {basemap_cropped.shape}")
        #     print(f"Displayed extent (lon/lat): Lon=[{X_latlon_cropped.min():.4f}, {X_latlon_cropped.max():.4f}], Lat=[{Y_latlon_cropped.min():.4f}, {Y_latlon_cropped.max():.4f}]")
            
        #     basemap_flipped = np.flip(basemap_cropped, axis=0)
            
        #     # Plot the cropped basemap
        #     ax.plot_surface(X_latlon_cropped, Y_latlon_cropped, Z_cropped, 
        #                    rstride=10, cstride=10,
        #                    facecolors=basemap_flipped/255.0,
        #                    shade=False,
        #                    alpha=alpha,
        #                    zorder=-1)
        # else:
        #     print("Warning: Cropping failed, displaying full basemap")
        #     Z = np.zeros_like(X_latlon)
        #     ax.plot_surface(X_latlon, Y_latlon, Z, 
        #                    rstride=10, cstride=10,
        #                    facecolors=basemap/255.0,
        #                    shade=False,
        #                    alpha=alpha,
        #                    zorder=-1)
        
        # print("Basemap plotted successfully")        

        Z = np.zeros_like(X_latlon)
        
        # Flip the basemap image vertically to match coordinate system
        basemap_flipped = np.flip(basemap, axis=0)
        
        # Plot the basemap as a surface
        ax.plot_surface(X_latlon, Y_latlon, Z, 
                       rstride=10, cstride=10,
                       facecolors=basemap_flipped/255.0,  # Normalize RGB values
                       shade=False,
                       alpha=alpha,
                       zorder=-1)
        
        print("Basemap loaded successfully")
        print(f"Basemap extent (lon/lat): [{X_latlon.min():.4f}, {X_latlon.max():.4f}, {Y_latlon.min():.4f}, {Y_latlon.max():.4f}]")
        print(f"Data extent (lon/lat): [{bounds[0]:.4f}, {bounds[2]:.4f}, {bounds[1]:.4f}, {bounds[3]:.4f}]")
        
    except Exception as e:
        print(f"Warning: Could not load basemap: {e}")
        print("Continuing without basemap...")

# Just do tests with basemap


def plot_single_income_group_3d(gdf, population_data, rent_data, title='Population Map', 
                                add_colorbar=True, add_basemap=True, basemap_alpha=0.3):
    """
    Plot a single 3D bar map with colors representing weighted average rent
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame with geometry in EPSG:4326 (WGS84)
    population_data : array-like
        1D array with population for each location
    rent_data : array-like
        1D array with weighted average rent for each location
    title : str
        Title for the plot
    add_colorbar : bool
        Whether to add a colorbar showing rent scale
    """
    # Ensure GeoDataFrame is in EPSG:4326
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        print(f"Converting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add basemap first (at z=0)
    if add_basemap:
        add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)
    
    # Get centroids
    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values  # Longitude
    y_coords = centroids.y.values  # Latitude
    
    print(f"Data centroids range - Lon: [{x_coords.min():.4f}, {x_coords.max():.4f}], Lat: [{y_coords.min():.4f}, {y_coords.max():.4f}]")
    
    heights = np.array(population_data)
    rents = np.array(rent_data)
    
    # Filter out locations with zero population
    # WAY QUICKER
    non_zero_mask = heights > 0
    x_coords_filtered = x_coords[non_zero_mask]
    y_coords_filtered = y_coords[non_zero_mask]
    heights_filtered = heights[non_zero_mask]
    rents_filtered = rents[non_zero_mask]
    
    print(f"Plotting {non_zero_mask.sum()} out of {len(heights)} locations (excluding zero population)")
    
    # Bar dimensions in degrees
    lon_range = x_coords.max() - x_coords.min()
    lat_range = y_coords.max() - y_coords.min()
    
    # Scale factor - adjust based on your map extent
    if lon_range < 1:  # City or small region scale
        scale_factor = 0.02
    elif lon_range < 10:  # Metropolitan or state scale
        scale_factor = 0.01
    else:  # Country or larger scale
        scale_factor = 0.005
    
    dx = lon_range * scale_factor  # Width in degrees of longitude
    dy = lat_range * scale_factor  # Depth in degrees of latitude
    
    # Create color map based on rent (filter out zero/invalid rents for normalization)
    # valid_rents = rents[rents > 0]
    # if len(valid_rents) > 0:
    #     vmin = valid_rents.min()
    #     vmax = valid_rents.max()
    if len(rents_filtered) > 0 and rents_filtered.max() > 0:
        vmin = rents_filtered.min()
        vmax = rents_filtered.max()
    else:
        vmin, vmax = 0, 1
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    #cmap = plt.cm.viridis
    cmap = plt.cm.YlOrRd
    # colors = cmap(norm(rents))
    colors = cmap(norm(rents_filtered))
    
    # Handle locations with no population (rent = 0) - make them gray
    colors[rents_filtered == 0] = [0.7, 0.7, 0.7, 0.8]  # Gray color
    # colors[rents == 0] = [0, 0, 0, 0]  # No color (RDP does not appear?)
    
    # # Plot
    # ax.bar3d(x_coords, y_coords, np.zeros_like(heights), 
    #         dx, dy, heights, color=colors, alpha=0.8, 
    #         edgecolor='none', linewidth=0, zorder=1) #zorder=1
    
    # Plot bars (only for non-zero population)
    if len(x_coords_filtered) > 0:
        ax.bar3d(x_coords_filtered, y_coords_filtered, np.zeros_like(heights_filtered), 
                dx, dy, heights_filtered, color=colors, alpha=0.8, 
                edgecolor='none', linewidth=0)
    
    # After plotting basemap and bars, set axis limits to exact data bounds
    # bounds = gdf.total_bounds
    # ax.set_xlim(bounds[0], bounds[2])  # lon min, lon max
    # ax.set_ylim(bounds[1], bounds[3])  # lat min, lat max
    
    # Axis labels for geographic coordinates with increased padding
    ax.set_xlabel('Longitude', labelpad=5)
    ax.set_ylabel('Latitude', labelpad=30)
    ax.set_zlabel('Nb of HHs', labelpad=30)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=0)
    
    # Increase spacing between tick labels and axes
    ax.tick_params(axis='x', pad=5)
    ax.tick_params(axis='y', pad=20)
    ax.tick_params(axis='z', pad=20)
    
    ax.view_init(elev=30, azim=270)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    # if add_colorbar:
    if add_colorbar and len(rents_filtered) > 0:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.01)
        cbar.set_label('Weighted Average Rent (ZAR/m²)', rotation=270, labelpad=20)
    
    # Adjust subplot to make more room
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    plt.show()
    
    return fig

# Significantly slower when adding basemap
# # Or plot single income group
income_idx = 0  # Low income group
population_data = population_array[:, income_idx, :].sum(axis=0)  # Sum across housing types
rent_data = calculate_weighted_average_rent(population_array, rent_array, income_idx)
fig = plot_single_income_group_3d(gdf, population_data, rent_data,
                                  title='Low Income Population Distribution',
                                  add_basemap=True, basemap_alpha=0.4)

# Bounds seem necessary...











########### ALSO DO STACKED BARS FOR SORTING ACROSS HOUSING TYPES




#############################"



### VERY LONG!!! SEE IF WORTH IT

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import contextily as ctx
from shapely.geometry import Polygon, MultiPolygon
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Assuming you have:
# rents: shape (n_locations, 5) - rents for 5 housing types
# population: shape (n_locations, 5, 4) - population for 5 housing types x 4 income groups
# gdf: GeoDataFrame with geometry for each location

# TEST

def calculate_weighted_rent(rents, population, income_group_idx):

    pop = population[:, income_group_idx, :]

    total_pop = pop.sum(axis=0, keepdims=True)
    total_pop = np.where(total_pop == 0, 1, total_pop)
    
    weighted_rents = (rents * pop).sum(axis=0) / total_pop.squeeze()
    
    return weighted_rents

def preprocess_all_data(rents, population, gdf):
    """
    Precompute all data needed for all income groups at once.
    This avoids redundant calculations.
    """
    # Convert to Web Mercator once
    gdf_merc = gdf.to_crs(epsg=3857)
    
    # Get centroids once
    centroids = gdf_merc.geometry.centroid
    x = centroids.x.values
    y = centroids.y.values
    
    # Calculate all weighted rents at once
    all_weighted_rents = np.array([
        calculate_weighted_rent(rents, population, i) for i in range(4)
    ])
    
    # Calculate all total populations at once
    all_total_pop = population.sum(axis=0)  # Sum across housing types for all income groups
    
    # Calculate bar dimensions once
    dx = (x.max() - x.min()) / 100
    dy = (y.max() - y.min()) / 100
    
    return {
        'gdf_merc': gdf_merc,
        'x': x,
        'y': y,
        'dx': dx,
        'dy': dy,
        'weighted_rents': all_weighted_rents,
        'total_pop': all_total_pop
    }

def get_basemap_image(gdf_merc, zoom='auto', source=ctx.providers.CartoDB.Positron):
    """
    Fetch a basemap image for the extent of the GeoDataFrame.
    """
    bounds = gdf_merc.total_bounds
    img, extent = ctx.bounds2img(bounds[0], bounds[1], bounds[2], bounds[3], 
                                 zoom=zoom, source=source)
    return img, extent

# def get_basemap_image(gdf, zoom='auto', source=ctx.providers.OpenStreetMap.Mapnik):
#     """
#     Fetch a basemap image for the extent of the GeoDataFrame.
    
#     Parameters:
#     - gdf: GeoDataFrame with geometry
#     - zoom: zoom level or 'auto'
#     - source: contextily basemap provider
    
#     Returns:
#     - img: numpy array of the basemap image
#     - extent: (left, right, bottom, top) in the GDF's CRS
#     """
#     # Ensure GDF is in Web Mercator (EPSG:3857) for contextily
#     gdf_merc = gdf.to_crs(epsg=3857)
    
#     # Get bounds
#     bounds = gdf_merc.total_bounds  # (minx, miny, maxx, maxy)
    
#     # Fetch basemap
#     img, extent = ctx.bounds2img(bounds[0], bounds[1], bounds[2], bounds[3], 
#                                  zoom=zoom, source=source)
    
#     # Convert extent back to original CRS if needed
#     # For now, we'll work in Web Mercator
#     return img, extent, gdf_merc

def create_polygon_collections(gdf_merc, simplify_tolerance=None):
    """
    Pre-create all polygon collections for faster rendering.
    Optional simplification for large datasets.
    """
    polys = []
    
    for geom in gdf_merc.geometry:
        # Optionally simplify geometry for speed
        if simplify_tolerance:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        
        if geom.geom_type == 'Polygon':
            poly_coords = np.array(geom.exterior.coords)
            verts = [(coord[0], coord[1], 0) for coord in poly_coords]
            polys.append(verts)
        elif geom.geom_type == 'MultiPolygon':
            for poly_geom in geom.geoms:
                poly_coords = np.array(poly_geom.exterior.coords)
                verts = [(coord[0], coord[1], 0) for coord in poly_coords]
                polys.append(verts)
    
    return polys


def plot_3d_rent_map_optimized(preprocessed_data, basemap_img, extent, 
                               polygon_verts, income_group_idx,
                               income_group_name=None, figsize=(16, 12),
                               basemap_alpha=0.6, show_boundaries=True):
    """
    Optimized 3D map plotting using preprocessed data.
    """
    # Extract preprocessed data
    x = preprocessed_data['x']
    y = preprocessed_data['y']
    dx = preprocessed_data['dx']
    dy = preprocessed_data['dy']
    weighted_rents = preprocessed_data['weighted_rents'][income_group_idx]
    total_pop = preprocessed_data['total_pop'][income_group_idx, :]
    
    # Set up figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Add basemap - downsample for speed if image is very large
    max_pixels = 1000
    if basemap_img.shape[0] > max_pixels or basemap_img.shape[1] > max_pixels:
        scale = max_pixels / max(basemap_img.shape[0], basemap_img.shape[1])
        new_height = int(basemap_img.shape[0] * scale)
        new_width = int(basemap_img.shape[1] * scale)
        from scipy.ndimage import zoom
        basemap_img = zoom(basemap_img, (new_height/basemap_img.shape[0], 
                                         new_width/basemap_img.shape[1], 1), order=1)
    
    xx, yy = np.meshgrid(
        np.linspace(extent[0], extent[1], basemap_img.shape[1]),
        np.linspace(extent[2], extent[3], basemap_img.shape[0])
    )
    zz = np.zeros_like(xx)
    
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, 
                    facecolors=basemap_img/255.0, 
                    shade=False, alpha=basemap_alpha, zorder=1)
    
    # Add polygon boundaries if requested
    if show_boundaries and polygon_verts:
        # Create single collection for all polygons (much faster)
        poly_collection = Poly3DCollection(polygon_verts, alpha=0.1, 
                                          facecolors='none', 
                                          edgecolor='black', linewidth=0.8, 
                                          zorder=2)
        ax.add_collection3d(poly_collection)
    
    # Normalize rent values
    norm = Normalize(vmin=np.nanmin(weighted_rents), vmax=np.nanmax(weighted_rents))
    cmap = plt.cm.YlOrRd
    colors = cmap(norm(weighted_rents))
    
    # Plot bars
    z = np.zeros_like(total_pop)
    ax.bar3d(x, y, z, dx, dy, total_pop, color=colors, alpha=0.85, 
             edgecolor='black', linewidth=0.3, zorder=3)
    
    # Set limits
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(0, np.max(total_pop) * 1.1)
    
    # Labels
    ax.set_xlabel('Longitude (Web Mercator)', fontsize=10)
    ax.set_ylabel('Latitude (Web Mercator)', fontsize=10)
    ax.set_zlabel('Population', fontsize=10)
    
    title = f'Population and Weighted Rent'
    if income_group_name:
        title += f' - {income_group_name}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    cbar.set_label('Weighted Average Rent', rotation=270, labelpad=20)
    
    ax.view_init(elev=30, azim=45)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    return fig, ax

# def plot_3d_rent_map(gdf, rents, population, income_group_idx, 
#                      income_group_name=None, figsize=(16, 12),
#                      basemap_source=ctx.providers.CartoDB.Positron,
#                      basemap_alpha=0.6):
#     """
#     Create a 3D map with bar height = population, color = weighted rent, and real basemap.
    
#     Parameters:
#     - gdf: GeoDataFrame with geometry
#     - rents: array of shape (n_locations, 5)
#     - population: array of shape (n_locations, 5, 4)
#     - income_group_idx: index of income group (0-3)
#     - income_group_name: name for plot title
#     - basemap_source: contextily provider for basemap
#     - basemap_alpha: transparency of basemap (0-1)
#     """
#     # Calculate total population for this income group
#     total_pop = population[:, income_group_idx, :].sum(axis=0)
    
#     # Calculate weighted average rent
#     weighted_rents = calculate_weighted_rent(rents, population, income_group_idx)
    
#     # Get basemap
#     basemap_img, extent, gdf_merc = get_basemap_image(gdf, source=basemap_source)
    
#     # Get centroids in Web Mercator projection
#     centroids = gdf_merc.geometry.centroid
#     x = centroids.x.values
#     y = centroids.y.values
    
#     # Set up figure
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Add basemap as image at z=0
#     # Create mesh grid for the basemap
#     xx, yy = np.meshgrid(
#         np.linspace(extent[0], extent[1], basemap_img.shape[1]),
#         np.linspace(extent[2], extent[3], basemap_img.shape[0])
#     )
#     zz = np.zeros_like(xx)
    
#     # Plot the basemap
#     ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, 
#                     facecolors=basemap_img/255.0, 
#                     shade=False, alpha=basemap_alpha, zorder=1)
    
#     # Add polygon boundaries for clarity
#     for idx, geom in enumerate(gdf_merc.geometry):
#         if geom.geom_type == 'Polygon':
#             poly_coords = np.array(geom.exterior.coords)
#             verts = [(coord[0], coord[1], 0) for coord in poly_coords]
#             poly = Poly3DCollection([verts], alpha=0.1, facecolors='none', 
#                                    edgecolor='black', linewidth=0.8, zorder=2)
#             ax.add_collection3d(poly)
#         elif geom.geom_type == 'MultiPolygon':
#             for poly_geom in geom.geoms:
#                 poly_coords = np.array(poly_geom.exterior.coords)
#                 verts = [(coord[0], coord[1], 0) for coord in poly_coords]
#                 poly = Poly3DCollection([verts], alpha=0.1, facecolors='none', 
#                                        edgecolor='black', linewidth=0.8, zorder=2)
#                 ax.add_collection3d(poly)
    
#     # Normalize rent values for coloring
#     norm = Normalize(vmin=np.nanmin(weighted_rents), vmax=np.nanmax(weighted_rents))
#     cmap = plt.cm.YlOrRd  # Yellow to Red colormap
    
#     # Create bars
#     # Bar width/depth - adjust based on your coordinate system
#     dx = np.ones_like(x) * (x.max() - x.min()) / 100
#     dy = np.ones_like(y) * (y.max() - y.min()) / 100
#     dz = total_pop
#     z = np.zeros_like(total_pop)
    
#     # Color based on weighted rent
#     colors = cmap(norm(weighted_rents))
    
#     # Plot bars
#     ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.85, 
#              edgecolor='black', linewidth=0.3, zorder=3)
    
#     # Set axis limits to match basemap extent
#     ax.set_xlim(extent[0], extent[1])
#     ax.set_ylim(extent[2], extent[3])
#     ax.set_zlim(0, np.max(total_pop) * 1.1)
    
#     # Labels and title
#     ax.set_xlabel('Longitude (Web Mercator)', fontsize=10)
#     ax.set_ylabel('Latitude (Web Mercator)', fontsize=10)
#     ax.set_zlabel('Nb of HHs', fontsize=10)
    
#     title = f'Spatial population distribution with rent levels'
#     if income_group_name:
#         title += f' - {income_group_name}'
#     ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
#     # Add colorbar
#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.1)
#     cbar.set_label('Weighted Average Rent', rotation=270, labelpad=20)
    
#     # Adjust viewing angle
#     ax.view_init(elev=30, azim=45)
    
#     # Remove background panes for cleaner look
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
    
#     plt.tight_layout()
    
#     return fig, ax


########################################

# Example usage:
# Assuming your arrays are named: rents, population, gdf

income_group_names = ['Poor', 'Midpoor', 
                      'Midrich', 'Rich']

# Basemap options (choose one):
# ctx.providers.OpenStreetMap.Mapnik - detailed street map
# ctx.providers.CartoDB.Positron - clean, minimal map
# ctx.providers.CartoDB.Voyager - colorful with labels
# ctx.providers.Stamen.Terrain - terrain/topography
# ctx.providers.Esri.WorldImagery - satellite imagery

gdf = gpd.read_file(path_data + "grid_reference_500.shp")
rents = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent
population = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households


# Step 1: Preprocess ALL data once
print("Preprocessing data...")
preprocessed = preprocess_all_data(rents, population, gdf)

# Step 2: Fetch basemap once
print("Fetching basemap...")
basemap_img, extent = get_basemap_image(
    preprocessed['gdf_merc'], 
    source=ctx.providers.CartoDB.Positron
)

# Step 3: Create polygon collections once (optional - set to None to skip boundaries)
print("Creating polygon collections...")
# For very large datasets (>1000 locations), consider simplifying or skipping boundaries
simplify_tol = None  # Set to a value like 100 for simplification
polygon_verts = create_polygon_collections(preprocessed['gdf_merc'], simplify_tol)

# Step 4: Create individual plots efficiently
# WHY LONG? STILL OK?
print("Creating individual plots...")
# for i in range(4):
for i in [0]:
    print(f"  Plotting {income_group_names[i]}...")
    fig, ax = plot_3d_rent_map_optimized(
        preprocessed, basemap_img, extent, polygon_verts, i,
        income_group_name=income_group_names[i]
    )
    
    plt.show()
    
    fig.savefig(f'rent_map_income_group_{i+1}_optimized.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close to free memory
    
    # plt.savefig(f'rent_map_income_group_{i+1}_optimized.png', 
    #             dpi=300, bbox_inches='tight', facecolor='white')
    # plt.close()  # Close to free memory







# Create a 3D map for each income group
# for i in range(4):
for i in [0]:
    fig, ax = plot_3d_rent_map(gdf, rents, population, i, 
                               income_group_name=income_group_names[i],
                               basemap_source=ctx.providers.CartoDB.Positron)
    # plt.savefig(f'rent_map_income_group_{i+1}_basemap.png', 
    #             dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()

# Optional: Create a 2x2 subplot with all income groups
fig = plt.figure(figsize=(20, 16))

# Get basemap once for all subplots
basemap_img, extent, gdf_merc = get_basemap_image(gdf, 
                                                   source=ctx.providers.CartoDB.Positron)

for i in range(4):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    
    # Add basemap
    xx, yy = np.meshgrid(
        np.linspace(extent[0], extent[1], basemap_img.shape[1]),
        np.linspace(extent[2], extent[3], basemap_img.shape[0])
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, 
                    facecolors=basemap_img/255.0, 
                    shade=False, alpha=0.5, zorder=1)
    
    # Add polygon boundaries
    for idx, geom in enumerate(gdf_merc.geometry):
        if geom.geom_type == 'Polygon':
            poly_coords = np.array(geom.exterior.coords)
            verts = [(coord[0], coord[1], 0) for coord in poly_coords]
            poly = Poly3DCollection([verts], alpha=0.1, facecolors='none', 
                                   edgecolor='black', linewidth=0.5, zorder=2)
            ax.add_collection3d(poly)
        elif geom.geom_type == 'MultiPolygon':
            for poly_geom in geom.geoms:
                poly_coords = np.array(poly_geom.exterior.coords)
                verts = [(coord[0], coord[1], 0) for coord in poly_coords]
                poly = Poly3DCollection([verts], alpha=0.1, facecolors='none', 
                                       edgecolor='black', linewidth=0.5, zorder=2)
                ax.add_collection3d(poly)
    
    # Calculate data
    total_pop = population[:, i, :].sum(axis=0)
    weighted_rents = calculate_weighted_rent(rents, population, i)
    
    # Get centroids
    centroids = gdf_merc.geometry.centroid
    x = centroids.x.values
    y = centroids.y.values
    
    # Normalize and color
    norm = Normalize(vmin=np.nanmin(weighted_rents), vmax=np.nanmax(weighted_rents))
    cmap = plt.cm.YlOrRd
    colors = cmap(norm(weighted_rents))
    
    # Bar dimensions
    dx = np.ones_like(x) * (x.max() - x.min()) / 100
    dy = np.ones_like(y) * (y.max() - y.min()) / 100
    dz = total_pop
    z = np.zeros_like(total_pop)
    
    # Plot
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.85, 
             edgecolor='black', linewidth=0.2, zorder=3)
    
    # Set limits
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(0, np.max(population[:, :, i].sum(axis=1)) * 1.1)
    
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)
    ax.set_zlabel('Nb of HHs', fontsize=8)
    ax.set_title(income_group_names[i], fontsize=11, fontweight='bold')
    ax.view_init(elev=30, azim=45)
    
    # Clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

plt.suptitle('Spatial population distribution with rent levels by income group', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('all_income_groups_combined_basemap.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


##### SCENARIO CHANGES
