######################################
### OUTPUT SCRIPT FOR JUE REVISION ###
######################################

# PREAMBLE

## IMPORT LIBRARIES

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
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

############################################

# TODO: MOVE FUNCTIONS IN SIDE SCRIPT

# LOAD SIMULATION RESULTS

def load_simulation_data(path_simul, data_type, simulation_name):
    """
    Load a single simulation data file.
    
    Parameters:
    -----------
    path_simul : str
        Path to simulation outputs
    data_type : str
        Type of data (e.g., 'utility', 'error', 'households')
    simulation_name : str
        Name of the simulation (e.g., 'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1')
    
    Returns:
    --------
    numpy.ndarray
        Loaded data array
    """
    filename = f'/initial_state_{data_type}_{simulation_name}.npy'
    return np.load(path_simul + filename)


def load_multiple_simulation_data(path_simul, data_types, simulation_names):
    """
    Load multiple simulation data files at once.
    
    Parameters:
    -----------
    path_simul : str
        Path to simulation outputs
    data_types : list of str or str
        Type(s) of data to load. If single string, applies to all simulations
    simulation_names : list of str
        List of simulation names
    
    Returns:
    --------
    dict
        Dictionary with simulation_name as keys and loaded data as values
    """
    results = {}
    
    # Handle single data_type for all simulations
    if isinstance(data_types, str):
        data_types = [data_types] * len(simulation_names)
    
    for sim_name, data_type in zip(simulation_names, data_types):
        results[sim_name] = load_simulation_data(path_simul, data_type, sim_name)
    
    return results

# ## Import simulation results using helper functions
simulation_configs = [
    'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr0_RDPnew1_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1',
]

# Load data for all simulations
utility_data = load_multiple_simulation_data(path_simul, 'utility', simulation_configs)

# Access loaded data
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[0]]
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[1]]
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[2]]
simul_UE1_ISconstr1_RDPnew1_Aup2_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[3]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility = utility_data[simulation_configs[4]]
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility = utility_data[simulation_configs[5]]

#############################################

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
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility = np.load(
    path_simul + '/initial_state_utility_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1.npy')

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
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households = np.load(
    path_simul + '/initial_state_households_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1.npy')

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
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size = np.load(
    path_simul + '/initial_state_dwelling_size_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1.npy')

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
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_housing_supply = np.load(
    path_simul + '/initial_state_housing_supply_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1.npy')

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
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1.npy')
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent = np.load(
    path_simul + '/initial_state_rent_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1.npy')


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

# DNF, else changes the source!!!!
subsid_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)

# Needs to be yearly to be realistic...
# subsid_income_net_of_commuting_costs[0] = (
#     income_net_of_commuting_costs[0]
#     + param["subsidized_structure_value"]
#     * (9*param["current_rate_public_housing"] + 21*param["future_rate_public_housing"])
#     / (households_per_income_class[0]*population/sum(households_per_income_class))
#     )
subsid_income_net_of_commuting_costs[0] = (
    income_net_of_commuting_costs[0]
    + param["subsidized_structure_value"]
    * param["current_rate_public_housing"]
    / ((households_per_income_class[0]-total_RDP)*population/sum(households_per_income_class))
    )
# Can also use future

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


### Amenity upgrading (low)

#### Not sure if extensive effect means something...

Aup1_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,0,:]))
Aup1_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,1,:]))
Aup1_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,2,:]))
Aup1_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_Aup1_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

# Take care
utility_backyard_Aup1_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

# Take care
utility_informal_Aup1_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_Aup1_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Aup1_housing_eff_mat = np.array([
    utility_formal_Aup1_housing_eff, utility_backyard_Aup1_housing_eff, utility_informal_Aup1_housing_eff, utility_rdp_Aup1_housing_eff])
utility_poor_Aup1_housing_eff = (
    np.nansum(utility_Aup1_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_Aup1_housing_eff = (
    np.nansum(utility_Aup1_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_Aup1_housing_eff = (
    np.nansum(utility_Aup1_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_Aup1_housing_eff = (
    np.nansum(utility_Aup1_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,3,:]))


utility_formal_Aup1_amenity_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

# Take care
utility_backyard_Aup1_amenity_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
    )

# Take care
utility_informal_Aup1_amenity_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_Aup1_amenity_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Aup1_amenity_eff_mat = np.array([
    utility_formal_Aup1_amenity_eff, utility_backyard_Aup1_amenity_eff, utility_informal_Aup1_amenity_eff, utility_rdp_Aup1_amenity_eff])
utility_poor_Aup1_amenity_eff = (
    np.nansum(utility_Aup1_amenity_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_Aup1_amenity_eff = (
    np.nansum(utility_Aup1_amenity_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_Aup1_amenity_eff = (
    np.nansum(utility_Aup1_amenity_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_Aup1_amenity_eff = (
    np.nansum(utility_Aup1_amenity_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[:,3,:]))


### Amenity upgrading (medium)

#### Not sure if extensive effect means something...

Aup2_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,0,:]))
Aup2_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,1,:]))
Aup2_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,2,:]))
Aup2_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_Aup2_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

# Take care
utility_backyard_Aup2_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

# Take care
utility_informal_Aup2_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_Aup2_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Aup2_housing_eff_mat = np.array([
    utility_formal_Aup2_housing_eff, utility_backyard_Aup2_housing_eff, utility_informal_Aup2_housing_eff, utility_rdp_Aup2_housing_eff])
utility_poor_Aup2_housing_eff = (
    np.nansum(utility_Aup2_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_Aup2_housing_eff = (
    np.nansum(utility_Aup2_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_Aup2_housing_eff = (
    np.nansum(utility_Aup2_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_Aup2_housing_eff = (
    np.nansum(utility_Aup2_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,3,:]))


utility_formal_Aup2_amenity_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

# Take care
utility_backyard_Aup2_amenity_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
    )

# Take care
utility_informal_Aup2_amenity_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_Aup2_amenity_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Aup2_amenity_eff_mat = np.array([
    utility_formal_Aup2_amenity_eff, utility_backyard_Aup2_amenity_eff, utility_informal_Aup2_amenity_eff, utility_rdp_Aup2_amenity_eff])
utility_poor_Aup2_amenity_eff = (
    np.nansum(utility_Aup2_amenity_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_Aup2_amenity_eff = (
    np.nansum(utility_Aup2_amenity_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_Aup2_amenity_eff = (
    np.nansum(utility_Aup2_amenity_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_Aup2_amenity_eff = (
    np.nansum(utility_Aup2_amenity_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[:,3,:]))


### Amenity upgrading (high)

#### Not sure if extensive effect means something...

Aup3_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,0,:]))
Aup3_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,1,:]))
Aup3_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,2,:]))
Aup3_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_Aup3_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

# Take care
utility_backyard_Aup3_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

# Take care
utility_informal_Aup3_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_Aup3_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Aup3_housing_eff_mat = np.array([
    utility_formal_Aup3_housing_eff, utility_backyard_Aup3_housing_eff, utility_informal_Aup3_housing_eff, utility_rdp_Aup3_housing_eff])
utility_poor_Aup3_housing_eff = (
    np.nansum(utility_Aup3_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_Aup3_housing_eff = (
    np.nansum(utility_Aup3_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_Aup3_housing_eff = (
    np.nansum(utility_Aup3_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_Aup3_housing_eff = (
    np.nansum(utility_Aup3_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,3,:]))


utility_formal_Aup3_amenity_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

# Take care
utility_backyard_Aup3_amenity_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
    )

# Take care
utility_informal_Aup3_amenity_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
    )

utility_rdp_Aup3_amenity_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Aup3_amenity_eff_mat = np.array([
    utility_formal_Aup3_amenity_eff, utility_backyard_Aup3_amenity_eff, utility_informal_Aup3_amenity_eff, utility_rdp_Aup3_amenity_eff])
utility_poor_Aup3_amenity_eff = (
    np.nansum(utility_Aup3_amenity_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,0,:]))
utility_midpoor_Aup3_amenity_eff = (
    np.nansum(utility_Aup3_amenity_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,1,:]))
utility_midrich_Aup3_amenity_eff = (
    np.nansum(utility_Aup3_amenity_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,2,:]))
utility_rich_Aup3_amenity_eff = (
    np.nansum(utility_Aup3_amenity_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[:,3,:]))


### Susbidies

#### Not sure if extensive effect mean something...

subsid_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,0,:]))
subsid_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,1,:]))
subsid_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,2,:]))
subsid_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_subsid_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_subsid_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
    )

utility_informal_subsid_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
    )

utility_rdp_subsid_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_subsid_housing_eff_mat = np.array([
    utility_formal_subsid_housing_eff, utility_backyard_subsid_housing_eff, utility_informal_subsid_housing_eff, utility_rdp_subsid_housing_eff])
utility_poor_subsid_housing_eff = (
    np.nansum(utility_subsid_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,0,:]))
utility_midpoor_subsid_housing_eff = (
    np.nansum(utility_subsid_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,1,:]))
utility_midrich_subsid_housing_eff = (
    np.nansum(utility_subsid_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,2,:]))
utility_rich_subsid_housing_eff = (
    np.nansum(utility_subsid_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,3,:]))


utility_formal_subsid_income_eff = (
    (subsid_income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_subsid_income_eff = (
    (subsid_income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
    )

utility_informal_subsid_income_eff = (
    (subsid_income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
    )

# DO NOT WANT TO DOUBLE COUNT BENEFITS
utility_rdp_subsid_income_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_subsid_income_eff_mat = np.array([
    utility_formal_subsid_income_eff, utility_backyard_subsid_income_eff, utility_informal_subsid_income_eff, utility_rdp_subsid_income_eff])
utility_poor_subsid_income_eff = (
    np.nansum(utility_subsid_income_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,0,:]))
utility_midpoor_subsid_income_eff = (
    np.nansum(utility_subsid_income_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,1,:]))
utility_midrich_subsid_income_eff = (
    np.nansum(utility_subsid_income_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,2,:]))
utility_rich_subsid_income_eff = (
    np.nansum(utility_subsid_income_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[:,3,:]))


### Eviction (low)

#### Not sure if extensive effect mean something...

Evict1_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,0,:]))
Evict1_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,1,:]))
Evict1_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,2,:]))
Evict1_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_Evict1_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_Evict1_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
    )

utility_informal_Evict1_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
    )

utility_rdp_Evict1_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Evict1_housing_eff_mat = np.array([
    utility_formal_Evict1_housing_eff, utility_backyard_Evict1_housing_eff, utility_informal_Evict1_housing_eff, utility_rdp_Evict1_housing_eff])
utility_poor_Evict1_housing_eff = (
    np.nansum(utility_Evict1_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,0,:]))
utility_midpoor_Evict1_housing_eff = (
    np.nansum(utility_Evict1_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,1,:]))
utility_midrich_Evict1_housing_eff = (
    np.nansum(utility_Evict1_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,2,:]))
utility_rich_Evict1_housing_eff = (
    np.nansum(utility_Evict1_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[:,3,:]))


### Eviction (medium)

#### Not sure if extensive effect mean something...

Evict2_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,0,:]))
Evict2_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,1,:]))
Evict2_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,2,:]))
Evict2_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_Evict2_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_Evict2_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
    )

utility_informal_Evict2_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
    )

utility_rdp_Evict2_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Evict2_housing_eff_mat = np.array([
    utility_formal_Evict2_housing_eff, utility_backyard_Evict2_housing_eff, utility_informal_Evict2_housing_eff, utility_rdp_Evict2_housing_eff])
utility_poor_Evict2_housing_eff = (
    np.nansum(utility_Evict2_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,0,:]))
utility_midpoor_Evict2_housing_eff = (
    np.nansum(utility_Evict2_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,1,:]))
utility_midrich_Evict2_housing_eff = (
    np.nansum(utility_Evict2_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,2,:]))
utility_rich_Evict2_housing_eff = (
    np.nansum(utility_Evict2_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[:,3,:]))


### Eviction (high)

#### Not sure if extensive effect mean something...

Evict3_utility_poor_alloc_eff = (
    np.nansum(utility_base_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,0,:]))
Evict3_utility_midpoor_alloc_eff = (
    np.nansum(utility_base_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,1,:]))
Evict3_utility_midrich_alloc_eff =(
    np.nansum(utility_base_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,2,:]))
Evict3_utility_rich_alloc_eff = (
    np.nansum(utility_base_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,3,:]))

#### Intensive effects

utility_formal_Evict3_housing_eff = (
    (income_net_of_commuting_costs
     - formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
     * formal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :])**param["alpha"]
    * (formal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :]
    )

utility_backyard_Evict3_housing_eff = (
    (income_net_of_commuting_costs
     - backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
     * backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :])**param["alpha"]
    * (backyard_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
    )

utility_informal_Evict3_housing_eff = (
    (income_net_of_commuting_costs
     - informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
     * informal_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
     - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))**param["alpha"]
    * (informal_size_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :] - param["q0"])**param["beta"]
    * amenities[None, :] * disam_informal_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
    )

utility_rdp_Evict3_housing_eff = (
    (income_net_of_commuting_costs
     + backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
     * param["backyard_size"]*backyard_rent_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :]
     - param["subsidized_structure_value"] * param["depreciation_rate"]
     - backyard_cost_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)**param["alpha"]
    * (param["RDP_size"] + param["backyard_size"] - param["q0"]
       - np.nanmin(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[None, :],1)*param["backyard_size"])**param["beta"]
    * amenities[None, :]
    )

utility_Evict3_housing_eff_mat = np.array([
    utility_formal_Evict3_housing_eff, utility_backyard_Evict3_housing_eff, utility_informal_Evict3_housing_eff, utility_rdp_Evict3_housing_eff])
utility_poor_Evict3_housing_eff = (
    np.nansum(utility_Evict3_housing_eff_mat[:,0,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,0,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,0,:]))
utility_midpoor_Evict3_housing_eff = (
    np.nansum(utility_Evict3_housing_eff_mat[:,1,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,1,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,1,:]))
utility_midrich_Evict3_housing_eff = (
    np.nansum(utility_Evict3_housing_eff_mat[:,2,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,2,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,2,:]))
utility_rich_Evict3_housing_eff = (
    np.nansum(utility_Evict3_housing_eff_mat[:,3,:]*simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,3,:])
    / np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[:,3,:]))


## BUILD TABLE

# Makes no sense? Problem comes from RDP? Cross-check with main utilities
poor_utility_changes = pd.DataFrame({
    "Subsidies": [subsid_utility_poor_alloc_eff, utility_poor_subsid_housing_eff, utility_poor_subsid_income_eff],
    "New RDP": [newRDP_utility_poor_alloc_eff, utility_poor_newRDP_housing_eff, utility_poor_newRDP_housing_eff],
    "New IS": [newIS_utility_poor_alloc_eff, utility_poor_newIS_housing_eff, utility_poor_newIS_housing_eff],
    "No UE": [noUE_utility_poor_alloc_eff, utility_poor_noUE_housing_eff, utility_poor_noUE_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])
midpoor_utility_changes = pd.DataFrame({
    "Subsidies": [subsid_utility_midpoor_alloc_eff, utility_midpoor_subsid_housing_eff, utility_midpoor_subsid_income_eff],
    "New RDP": [newRDP_utility_midpoor_alloc_eff, utility_midpoor_newRDP_housing_eff, utility_midpoor_newRDP_housing_eff],
    "New IS": [newIS_utility_midpoor_alloc_eff, utility_midpoor_newIS_housing_eff, utility_midpoor_newIS_housing_eff],
    "No UE": [noUE_utility_midpoor_alloc_eff, utility_midpoor_noUE_housing_eff, utility_midpoor_noUE_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])
midrich_utility_changes = pd.DataFrame({
    "Subsidies": [subsid_utility_midrich_alloc_eff, utility_midrich_subsid_housing_eff, utility_midrich_subsid_income_eff],
    "New RDP": [newRDP_utility_midrich_alloc_eff, utility_midrich_newRDP_housing_eff, utility_midrich_newRDP_housing_eff],
    "New IS": [newIS_utility_midrich_alloc_eff, utility_midrich_newIS_housing_eff, utility_midrich_newIS_housing_eff],
    "No UE": [noUE_utility_midrich_alloc_eff, utility_midrich_noUE_housing_eff, utility_midrich_noUE_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])
rich_utility_changes = pd.DataFrame({
    "Subsidies": [subsid_utility_rich_alloc_eff, utility_rich_subsid_housing_eff, utility_rich_subsid_income_eff],
    "New RDP": [newRDP_utility_rich_alloc_eff, utility_rich_newRDP_housing_eff, utility_rich_newRDP_housing_eff],
    "New IS": [newIS_utility_rich_alloc_eff, utility_rich_newIS_housing_eff, utility_rich_newIS_housing_eff],
    "No UE": [noUE_utility_rich_alloc_eff, utility_rich_noUE_housing_eff, utility_rich_noUE_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])

for col_name in poor_utility_changes.columns:
    poor_utility_changes[col_name] = (poor_utility_changes[col_name]-base_utility_poor)/base_utility_poor
for col_name in midpoor_utility_changes.columns:
    midpoor_utility_changes[col_name] = (midpoor_utility_changes[col_name]-base_utility_midpoor)/base_utility_midpoor
for col_name in midrich_utility_changes.columns:
    midrich_utility_changes[col_name] = (midrich_utility_changes[col_name]-base_utility_midrich)/base_utility_midrich
for col_name in rich_utility_changes.columns:
    rich_utility_changes[col_name] = (rich_utility_changes[col_name]-base_utility_rich)/base_utility_rich

with pd.ExcelWriter(path_output_tables + '/utility_change_decomp.xlsx') as writer:
    poor_utility_changes.to_excel(writer, float_format="%.3f", sheet_name="poor")
    midpoor_utility_changes.to_excel(writer, float_format="%.3f", sheet_name="midpoor")
    midrich_utility_changes.to_excel(writer, float_format="%.3f", sheet_name="midrich")
    rich_utility_changes.to_excel(writer, float_format="%.3f", sheet_name="rich")

#

poor_utility_changes_upgrading = pd.DataFrame({
    "Disam. -10pc": [Aup1_utility_poor_alloc_eff, utility_poor_Aup1_housing_eff, utility_poor_Aup1_amenity_eff],
    "Disam. -50pc": [Aup2_utility_poor_alloc_eff, utility_poor_Aup2_housing_eff, utility_poor_Aup2_amenity_eff],
    "Disam. -100pc": [Aup3_utility_poor_alloc_eff, utility_poor_Aup3_housing_eff, utility_poor_Aup3_amenity_eff],
    "Toler. -10pc": [Evict1_utility_poor_alloc_eff, utility_poor_Evict1_housing_eff, utility_poor_Evict1_housing_eff],
    "Toler. -50pc": [Evict2_utility_poor_alloc_eff, utility_poor_Evict2_housing_eff, utility_poor_Evict2_housing_eff],
    "Toler. -100pc": [Evict3_utility_poor_alloc_eff, utility_poor_Evict3_housing_eff, utility_poor_Evict3_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])
midpoor_utility_changes_upgrading = pd.DataFrame({
    "Disam. -10pc": [Aup1_utility_midpoor_alloc_eff, utility_midpoor_Aup1_housing_eff, utility_midpoor_Aup1_amenity_eff],
    "Disam. -50pc": [Aup2_utility_midpoor_alloc_eff, utility_midpoor_Aup2_housing_eff, utility_midpoor_Aup2_amenity_eff],
    "Disam. -100pc": [Aup3_utility_midpoor_alloc_eff, utility_midpoor_Aup3_housing_eff, utility_midpoor_Aup3_amenity_eff],
    "Toler. -10pc": [Evict1_utility_midpoor_alloc_eff, utility_midpoor_Evict1_housing_eff, utility_midpoor_Evict1_housing_eff],
    "Toler. -50pc": [Evict2_utility_poor_alloc_eff, utility_poor_Evict2_housing_eff, utility_poor_Evict2_housing_eff],
    "Toler. -100pc": [Evict3_utility_poor_alloc_eff, utility_poor_Evict3_housing_eff, utility_poor_Evict3_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])
midrich_utility_changes_upgrading = pd.DataFrame({
    "Disam. -10pc": [Aup1_utility_midrich_alloc_eff, utility_midrich_Aup1_housing_eff, utility_midrich_Aup1_amenity_eff],
    "Disam. -50pc": [Aup2_utility_midrich_alloc_eff, utility_midrich_Aup2_housing_eff, utility_midrich_Aup2_amenity_eff],
    "Disam. -100pc": [Aup3_utility_midrich_alloc_eff, utility_midrich_Aup3_housing_eff, utility_midrich_Aup3_amenity_eff],
    "Toler. -10pc": [Evict1_utility_midrich_alloc_eff, utility_midrich_Evict1_housing_eff, utility_midrich_Evict1_housing_eff],
    "Toler. -50pc": [Evict2_utility_poor_alloc_eff, utility_poor_Evict2_housing_eff, utility_poor_Evict2_housing_eff],
    "Toler. -100pc": [Evict3_utility_poor_alloc_eff, utility_poor_Evict3_housing_eff, utility_poor_Evict3_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])
rich_utility_changes_upgrading = pd.DataFrame({
    "Disam. -10pc": [Aup1_utility_rich_alloc_eff, utility_rich_Aup1_housing_eff, utility_rich_Aup1_amenity_eff],
    "Disam. -50pc": [Aup2_utility_rich_alloc_eff, utility_rich_Aup2_housing_eff, utility_rich_Aup2_amenity_eff],
    "Disam. -100pc": [Aup3_utility_rich_alloc_eff, utility_rich_Aup3_housing_eff, utility_rich_Aup3_amenity_eff],
    "Toler. -10pc": [Evict1_utility_rich_alloc_eff, utility_rich_Evict1_housing_eff, utility_rich_Evict1_housing_eff],
    "Toler. -50pc": [Evict2_utility_poor_alloc_eff, utility_poor_Evict2_housing_eff, utility_poor_Evict2_housing_eff],
    "Toler. -100pc": [Evict3_utility_poor_alloc_eff, utility_poor_Evict3_housing_eff, utility_poor_Evict3_housing_eff]},
    index=["Ext. margin", "Int. margin", "Dir. effect"])

for col_name in poor_utility_changes_upgrading.columns:
    poor_utility_changes_upgrading[col_name] = (poor_utility_changes_upgrading[col_name]-base_utility_poor)/base_utility_poor
for col_name in midpoor_utility_changes_upgrading.columns:
    midpoor_utility_changes_upgrading[col_name] = (midpoor_utility_changes_upgrading[col_name]-base_utility_midpoor)/base_utility_midpoor
for col_name in midrich_utility_changes_upgrading.columns:
    midrich_utility_changes_upgrading[col_name] = (midrich_utility_changes_upgrading[col_name]-base_utility_midrich)/base_utility_midrich
for col_name in rich_utility_changes_upgrading.columns:
    rich_utility_changes_upgrading[col_name] = (rich_utility_changes_upgrading[col_name]-base_utility_rich)/base_utility_rich

with pd.ExcelWriter(path_output_tables + '/utility_change_decomp_upgrading.xlsx') as writer:
    poor_utility_changes_upgrading.to_excel(writer, float_format="%.3f", sheet_name="poor")
    midpoor_utility_changes_upgrading.to_excel(writer, float_format="%.3f", sheet_name="midpoor")
    midrich_utility_changes_upgrading.to_excel(writer, float_format="%.3f", sheet_name="midrich")
    rich_utility_changes_upgrading.to_excel(writer, float_format="%.3f", sheet_name="rich")

#####

# Consistency issues with poor group definition?
# Other errors... Is decomposition that interesting?
# TODO: Comment out? For now, leverage definition for RDP utilities...

#####

# Weighted sum... Population ratio?

agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility])

# Can make sense for RDP utility to go down if more people are covered...

agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility])

agg_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[0] = agg_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility*agg_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_newRDP_housing_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_newRDP_housing_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility])

agg_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,axis=(0,2))
agg_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0] = agg_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0] - np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])
avg_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility*agg_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households)
     + np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_newIS_housing_eff[0, :]))
    / population)
ext_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = np.append(
    simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    [np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_newIS_housing_eff[0, :])/np.nansum(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]),
    avg_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility])

agg_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,axis=(0,2))
agg_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0] = agg_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0] - np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])
avg_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = (
    (np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility*agg_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households)
     + np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_noUE_housing_eff[0, :]))
    / population)
ext_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = np.append(
    simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    [np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_noUE_housing_eff[0, :])/np.nansum(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:]),
    avg_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility])

df_utility_changes = pd.DataFrame({
    "Subsidies": ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_utility,
    "New RDP": ext_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility,
    "New IS": ext_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility,
    "No UE": ext_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility},
    index=["Poor (no RDP)", "Midpoor", "Midrich", "Rich", "Poor (RDP)", "Weighted avg"])

for col_name in df_utility_changes.columns:
    df_utility_changes[col_name] = (df_utility_changes[col_name]-ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility)/ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility

# Add baseline column for reference? Just refer to bar plots for population breakdown across htypes

df_utility_changes.to_excel(path_output_tables + '/df_utility_changes.xlsx', float_format="%.3f")

###

# TODO: Weird that it all gives the same? Cross-check with breakdown

agg_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility])

agg_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility])

agg_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility])

agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility])

agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility])

agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households = np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households,axis=(0,2))
agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[0] = agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[0] - np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[3,:,:])
avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility = (
    (np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility*agg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households)
     + np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :]))
    / population)
ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility = np.append(
    simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility,
    [np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[3,:,:]*utility_rdp_subsid_income_eff[0, :])/np.nansum(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[3,:,:]),
    avg_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility])

df_utility_changes_upgrading = pd.DataFrame({
    "Disam. -10pc": ext_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility,
    "Disam. -50pc": ext_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility,
    "Disam. -100pc": ext_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_utility,
    "Toler. -10pc": ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_utility,
    "Toler. -50pc": ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_utility,
    "Toler. -100pc": ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_utility},
    index=["Poor (no RDP)", "Midpoor", "Midrich", "Rich", "Poor (RDP)", "Weighted avg"])

for col_name in df_utility_changes_upgrading.columns:
    df_utility_changes_upgrading[col_name] = (df_utility_changes_upgrading[col_name]-ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility)/ext_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility

df_utility_changes_upgrading.to_excel(path_output_tables + '/df_utility_changes_upgrading.xlsx', float_format="%.3f")

# styled_df_utility_changes = df_utility_changes.style \
#   .format(precision=2, thousands=".", decimal=",") \
#   .format_index(str.upper, axis=1) \
#   .relabel_index(["Poor", "Mid-poor", "Mid-rich", "Rich"], axis=0)




# WELFARE DECOMPOSITION: add options to load ante and post variables
# Distinguish redeveloped backyazrds???

# Do it for each scenario (also for other variables at baseline?)

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])]
    )

basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3,:])]
    )

basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:])]
    )

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
    
new_amenities = np.tile(amenities, (5,1))

#
formal_income_net_of_commuting_costs = np.nansum(formal_income_net_of_commuting_costs, 0)
basic_backyard_income_net_of_commuting_costs = np.nansum(basic_backyard_income_net_of_commuting_costs, 0)
redev_backyard_income_net_of_commuting_costs = np.nansum(redev_backyard_income_net_of_commuting_costs, 0)
informal_income_net_of_commuting_costs = np.nansum(informal_income_net_of_commuting_costs, 0)
rdp_income_net_of_commuting_costs = np.nansum(rdp_income_net_of_commuting_costs, 0)
#
new_income_net_of_commuting_costs = np.array(
    [formal_income_net_of_commuting_costs,
     basic_backyard_income_net_of_commuting_costs,
     redev_backyard_income_net_of_commuting_costs,
     informal_income_net_of_commuting_costs,
     rdp_income_net_of_commuting_costs]
    )

###

new_backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)))
new_backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[0,:] = np.copy(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[1,:] = np.copy(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[2,:] = np.copy(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1[3,:] = np.copy(backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
#
basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
#
new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.array(
    [np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
     increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
     np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])]
    )
#
basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[0,:]),
     basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
     increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
     np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3,:])]
    )
#
basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     increm_backyard_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)))
new_backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1)
#
basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
#
new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
     increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
     np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])]
    )
basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
     increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
     np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size[3,:])]
    )

###

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size[3,:])]
    )

new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1 = np.zeros((4,len(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)))
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[0,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[1,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[2,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)
new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1[3,:] = np.copy(backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1)
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[1,:,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[1,:,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[new_backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1<2] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[0,:,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[2,:,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households[3,:,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[3,:] = 0
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_rent[3,:])]
    )
#
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[1,:])
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size = np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[1,:])
basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1==2] = 0
increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[backyard_supply_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1<2] = 0
# Adjust
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[3,:] = param["backyard_size"] + param["RDP_size"]
#
new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size = np.array(
    [np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[0,:]),
     basic_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size,
     increm_backyard_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size,
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[2,:]),
     np.copy(simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_dwelling_size[3,:])]
    )

# Weird results? Has to do with simulation definitions?

# Rows are housing types and columns are income groups, so we transpose
scenario_data = {
    'Baseline': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 2).T,
    'Subsidies': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households, 2).T,
    'New RDP': np.nansum(new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households, 2).T,
    'New IS': np.nansum(new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 2).T,
    'No UE': np.nansum(new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households ,2).T
}

scenario_data_upgrading = {
    'Baseline': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 2).T,
    'Disam. -10pc': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_households, 2).T,
    'Disam. -50pc': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households, 2).T,
    'Disam. -100pc': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1_households, 2).T,
    'Toler. -10pc': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1_households, 2).T,
    'Toler. -50pc': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households, 2).T,
    'Toler. -100pc': np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1_households, 2).T
}

## FIRST GRAPH

# Labels
income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
housing_types = ['FP', 'IB (basic)', 'IB (redev.)', 'IS', 'FS']
scenarios = list(scenario_data.keys())

# Create figure with subplots (4 rows for income groups, 1 column)
fig, axes = plt.subplots(4, 1, figsize=(12, 14))
fig.suptitle('Distribution of HHs across housing types by income group and scenario', 
             fontsize=16, fontweight='bold', y=0.995)

# Colors for housing types: DNF
colors = plt.cm.Set3(np.linspace(0, 1, 5))

# Plot for each income group
for i, income_group in enumerate(income_groups):
    ax = axes[i]
    
    # Bar positions
    x = np.arange(len(scenarios))
    width = 0.6
    
    # Initialize bottom for stacking
    bottom = np.zeros(len(scenarios))
    
    # Stack bars for each housing type
    for j, housing_type in enumerate(housing_types):
        # Extract data for this income group across all scenarios
        values = [scenario_data[scenario][i, j] for scenario in scenarios]
        
        ax.bar(x, values, width, label=housing_type, bottom=bottom, color=colors[j])
        bottom += values
    
    # Formatting
    ax.set_ylabel('Number of HHs', fontsize=11, fontweight='bold')
    ax.set_title(f'{income_group}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend only to the first subplot
    if i == 0:
        ax.legend(title='Housing types', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
# plt.show()

plt.savefig(path_output_plots + '/htype_breakdown.png')


## SECOND GRAPH

# Labels
income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
housing_types = ['FP', 'IB (basic)', 'IB (redev.)', 'IS', 'FS']
scenarios = list(scenario_data_upgrading.keys())

# Create figure with subplots (4 rows for income groups, 1 column)
fig, axes = plt.subplots(4, 1, figsize=(12, 14))
fig.suptitle('Distribution of HHs across housing types by income group and scenario', 
             fontsize=16, fontweight='bold', y=0.995)

# Colors for housing types: DNF
colors = plt.cm.Set3(np.linspace(0, 1, 5))

# Plot for each income group
for i, income_group in enumerate(income_groups):
    ax = axes[i]
    
    # Bar positions
    x = np.arange(len(scenarios))
    width = 0.6
    
    # Initialize bottom for stacking
    bottom = np.zeros(len(scenarios))
    
    # Stack bars for each housing type
    for j, housing_type in enumerate(housing_types):
        # Extract data for this income group across all scenarios
        values = [scenario_data_upgrading[scenario][i, j] for scenario in scenarios]
        
        ax.bar(x, values, width, label=housing_type, bottom=bottom, color=colors[j])
        bottom += values
    
    # Formatting
    ax.set_ylabel('Number of HHs', fontsize=11, fontweight='bold')
    ax.set_title(f'{income_group}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend only to the first subplot
    if i == 0:
        ax.legend(title='Housing types', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
# plt.show()

plt.savefig(path_output_plots + '/htype_breakdown_upgrading.png')

# SOMETHING IS WRONG WITH ALLOCATION OF SECODN GRAPH...


### DENSITY PLOTS


##### Rent

# Example: 1000 locations, 5 housing types, 4 income groups
n_locations = np.shape(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent)[1]
# NEED TO UPDATE WITH REDEV LATER
n_housing_types = 5
n_income_groups = 4

# In example data, location is the first axis: NEED TO CORRECT!
rent = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent
population = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households

# Define rent brackets: NEED TO CHANGE FOR EACH GROUP
# rent_bins = np.arange(0, 1000, 10)  # Adjust as needed
# bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2

# Labels
housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # rent_bins = np.arange(
    #     min(rent[population[:,income_idx,:]>0]), 
    #     max(rent[population[:,income_idx,:]>0]),
    #     10)
    
    start = min(rent[population[:,income_idx,:]>0])
    end = np.nanquantile(rent[population[:,income_idx,:]>0], 0.99)
    
    rent_bins = np.arange(start, end, (end-start)/100)
    bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2
    
    # Initialize arrays to hold binned population for each housing type
    binned_pop = np.zeros((len(rent_bins)-1, n_housing_types))
    
    # Bin the population by rent bracket for each housing type
    for housing_idx in range(n_housing_types):
        for loc_idx in range(n_locations):
            rent_value = rent[housing_idx, loc_idx]
            pop_value = population[housing_idx, income_idx, loc_idx]
            
            # Find which bin this rent falls into
            bin_idx = np.digitize(rent_value, rent_bins) - 1
            if 0 <= bin_idx < len(rent_bins) - 1:
                binned_pop[bin_idx, housing_idx] += pop_value
    
    # Create stacked bar chart
    bottom = np.zeros(len(rent_bins)-1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
    
    for housing_idx in range(n_housing_types):
        ax.bar(bin_centers, binned_pop[:, housing_idx], 
               width=rent_bins[1]-rent_bins[0], 
               bottom=bottom,
               label=housing_type_labels[housing_idx],
               color=colors[housing_idx],
               edgecolor='white',
               linewidth=0.5)
        bottom += binned_pop[:, housing_idx]
    
    # Formatting
    ax.set_xlabel('Rent (ZAR/year)', fontsize=10)
    ax.set_ylabel('Nb of HHs', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(path_output_plots + '/baseline_rent_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Print summary statistics
print("\nSummary Statistics:")
print("-" * 50)
for income_idx in range(n_income_groups):
    total_pop = population[:, income_idx, :].sum()
    print(f"{income_group_labels[income_idx]}: Total Population = {total_pop:,}")
    
# NB: FS does not appear when there is no rent


##### Income

# Example: 1000 locations, 5 housing types, 4 income groups
n_locations = np.shape(new_income_net_of_commuting_costs)[1]
# NEED TO UPDATE WITH REDEV LATER
n_housing_types = 5
n_income_groups = 4

# In example data, location is the first axis: NEED TO CORRECT!
rent = new_income_net_of_commuting_costs
population = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households

# Define rent brackets: NEED TO CHANGE FOR EACH GROUP
# rent_bins = np.arange(0, 1000, 10)  # Adjust as needed
# bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2

# Labels
housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # rent_bins = np.arange(
    #     min(rent[population[:,income_idx,:]>0]), 
    #     max(rent[population[:,income_idx,:]>0]),
    #     10)
    
    start = min(rent[population[:,income_idx,:]>0])
    end = np.nanquantile(rent[population[:,income_idx,:]>0], 0.99)
    
    rent_bins = np.arange(start, end, (end-start)/100)
    bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2
    
    # Initialize arrays to hold binned population for each housing type
    binned_pop = np.zeros((len(rent_bins)-1, n_housing_types))
    
    # Bin the population by rent bracket for each housing type
    for housing_idx in range(n_housing_types):
        for loc_idx in range(n_locations):
            rent_value = rent[housing_idx, loc_idx]
            pop_value = population[housing_idx, income_idx, loc_idx]
            
            # Find which bin this rent falls into
            bin_idx = np.digitize(rent_value, rent_bins) - 1
            if 0 <= bin_idx < len(rent_bins) - 1:
                binned_pop[bin_idx, housing_idx] += pop_value
    
    # Create stacked bar chart
    bottom = np.zeros(len(rent_bins)-1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
    
    for housing_idx in range(n_housing_types):
        ax.bar(bin_centers, binned_pop[:, housing_idx], 
               width=rent_bins[1]-rent_bins[0], 
               bottom=bottom,
               label=housing_type_labels[housing_idx],
               color=colors[housing_idx],
               edgecolor='white',
               linewidth=0.5)
        bottom += binned_pop[:, housing_idx]
    
    # Formatting
    ax.set_xlabel('Income net of commuting (ZAR/year)', fontsize=10)
    ax.set_ylabel('Nb of HHs', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(path_output_plots + '/baseline_income_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Print summary statistics
print("\nSummary Statistics:")
print("-" * 50)
for income_idx in range(n_income_groups):
    total_pop = population[:, income_idx, :].sum()
    print(f"{income_group_labels[income_idx]}: Total Population = {total_pop:,}")
    
# NB: FS does not appear when there is no rent


##### Dwelling size

# Example: 1000 locations, 5 housing types, 4 income groups
n_locations = np.shape(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size)[1]
# NEED TO UPDATE WITH REDEV LATER
n_housing_types = 5
n_income_groups = 4

# In example data, location is the first axis: NEED TO CORRECT!
rent = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size
population = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households

# Define rent brackets: NEED TO CHANGE FOR EACH GROUP
# rent_bins = np.arange(0, 1000, 10)  # Adjust as needed
# bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2

# Labels
housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # rent_bins = np.arange(
    #     min(rent[population[:,income_idx,:]>0]), 
    #     max(rent[population[:,income_idx,:]>0]),
    #     10)
    
    start = min(rent[population[:,income_idx,:]>0])
    # end = np.nanquantile(rent[population[:,income_idx,:]>0], 0.99)
    end = max(rent[population[:,income_idx,:]>0])+10
    
    rent_bins = np.arange(start, end, (end-start)/100)
    bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2
    
    # Initialize arrays to hold binned population for each housing type
    binned_pop = np.zeros((len(rent_bins)-1, n_housing_types))
    
    # Bin the population by rent bracket for each housing type
    for housing_idx in range(n_housing_types):
        for loc_idx in range(n_locations):
            rent_value = rent[housing_idx, loc_idx]
            pop_value = population[housing_idx, income_idx, loc_idx]
            
            # Find which bin this rent falls into
            bin_idx = np.digitize(rent_value, rent_bins) - 1
            if 0 <= bin_idx < len(rent_bins) - 1:
                binned_pop[bin_idx, housing_idx] += pop_value
    
    # Create stacked bar chart
    bottom = np.zeros(len(rent_bins)-1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
    
    for housing_idx in range(n_housing_types):
        ax.bar(bin_centers, binned_pop[:, housing_idx], 
               width=rent_bins[1]-rent_bins[0], 
               bottom=bottom,
               label=housing_type_labels[housing_idx],
               color=colors[housing_idx],
               edgecolor='white',
               linewidth=0.5)
        bottom += binned_pop[:, housing_idx]
    
    # Formatting
    ax.set_xlabel('Dwelling size (m²)', fontsize=10)
    ax.set_ylabel('Nb of HHs', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(path_output_plots + '/baseline_dsize_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Print summary statistics
print("\nSummary Statistics:")
print("-" * 50)
for income_idx in range(n_income_groups):
    total_pop = population[:, income_idx, :].sum()
    print(f"{income_group_labels[income_idx]}: Total Population = {total_pop:,}")
    
# NB: FS does not appear when there is no rent



##### Amenities

# Example: 1000 locations, 5 housing types, 4 income groups
n_locations = np.shape(new_amenities)[1]
# NEED TO UPDATE WITH REDEV LATER
n_housing_types = 5
n_income_groups = 4

# In example data, location is the first axis: NEED TO CORRECT!
rent = new_amenities
population = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households

# Define rent brackets: NEED TO CHANGE FOR EACH GROUP
# rent_bins = np.arange(0, 1000, 10)  # Adjust as needed
# bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2

# Labels
housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]
income_group_labels = ["Poor", "Midpoor", "Midrich", "Rich"]

# Create a figure with subplots for each income group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for income_idx in range(n_income_groups):
    ax = axes[income_idx]
    
    # rent_bins = np.arange(
    #     min(rent[population[:,income_idx,:]>0]), 
    #     max(rent[population[:,income_idx,:]>0]),
    #     10)
    
    start = min(rent[population[:,income_idx,:]>0])
    end = np.nanquantile(rent[population[:,income_idx,:]>0], 0.99)
    
    rent_bins = np.arange(start, end, (end-start)/100)
    bin_centers = (rent_bins[:-1] + rent_bins[1:]) / 2
    
    # Initialize arrays to hold binned population for each housing type
    binned_pop = np.zeros((len(rent_bins)-1, n_housing_types))
    
    # Bin the population by rent bracket for each housing type
    for housing_idx in range(n_housing_types):
        for loc_idx in range(n_locations):
            rent_value = rent[housing_idx, loc_idx]
            pop_value = population[housing_idx, income_idx, loc_idx]
            
            # Find which bin this rent falls into
            bin_idx = np.digitize(rent_value, rent_bins) - 1
            if 0 <= bin_idx < len(rent_bins) - 1:
                binned_pop[bin_idx, housing_idx] += pop_value
    
    # Create stacked bar chart
    bottom = np.zeros(len(rent_bins)-1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
    
    for housing_idx in range(n_housing_types):
        ax.bar(bin_centers, binned_pop[:, housing_idx], 
               width=rent_bins[1]-rent_bins[0], 
               bottom=bottom,
               label=housing_type_labels[housing_idx],
               color=colors[housing_idx],
               edgecolor='white',
               linewidth=0.5)
        bottom += binned_pop[:, housing_idx]
    
    # Formatting
    ax.set_xlabel('Amenity index', fontsize=10)
    ax.set_ylabel('Nb of HHs', fontsize=10)
    ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(path_output_plots + '/baseline_amenity_pop_distrib.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Print summary statistics
print("\nSummary Statistics:")
print("-" * 50)
for income_idx in range(n_income_groups):
    total_pop = population[:, income_idx, :].sum()
    print(f"{income_group_labels[income_idx]}: Total Population = {total_pop:,}")
    
# NB: FS does not appear when there is no rent


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

def calculate_weighted_rent(rents, population, income_group_idx):
    """
    Calculate population-weighted average rent for each location and income group.
    
    Parameters:
    - rents: array of shape (n_locations, 5)
    - population: array of shape (n_locations, 5, 4)
    - income_group_idx: index of income group (0-3)
    
    Returns:
    - weighted_rents: array of shape (n_locations,)
    """
    # Get population for this income group across housing types
    pop = population[:, income_group_idx, :]  # shape: (n_locations, 5)
    
    # Calculate weighted average rent
    # Where population is 0, we'll handle it to avoid division by zero
    total_pop = pop.sum(axis=0, keepdims=True)
    total_pop = np.where(total_pop == 0, 1, total_pop)  # Avoid division by zero
    
    weighted_rents = (rents * pop).sum(axis=0) / total_pop.squeeze()
    
    return weighted_rents

# TEST
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
