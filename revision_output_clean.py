######################################
### OUTPUT SCRIPT FOR JUE REVISION ###
######################################

# %% PREAMBLE

print("PREAMBLE")

## IMPORT LIBRARIES

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde

from joblib import Parallel, delayed
import multiprocessing

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import contextily as ctx
from pyproj import Transformer

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
path_output_plots = path_simul + '/plots/new/'
path_output_tables = path_simul + '/tables/new/'

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

###############################################################################

# %% PROCESS SIMULATION RESULTS

print("PROCESS SIMULATION RESULTS")

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

# %% TABLE 1: UTILITY CHANGES

print("TABLE 1: UTILITY CHANGES")

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

df_utility_changes_baseline = pd.concat([df_utility_baseline, df_utility_changes], axis=1)

df_utility_changes_baseline.to_excel(path_output_tables + '/df_utility_changes_baseline.xlsx', float_format="%.3f")

###############################################################################

# %% WE DEFINE MORE VARIABLES TO BE USED FOR PLOTTING

print("WE DEFINE MORE VARIABLES TO BE USED FOR PLOTTING")

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
        simul_array[3,:] = param["RDP_size"]
        
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

def gen_new_income_per_scenario(income_net_of_commuting_costs, new_simul_households):
    
    formal_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
    basic_backyard_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
    redev_backyard_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
    informal_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
    rdp_income_net_of_commuting_costs = np.copy(income_net_of_commuting_costs)
    
    for i in range(4):
        formal_income_net_of_commuting_costs[i, :][
            new_simul_households[0,i,:]==0] = 0
        basic_backyard_income_net_of_commuting_costs[i, :][
            new_simul_households[1,i,:]==0] = 0
        redev_backyard_income_net_of_commuting_costs[i, :][
            new_simul_households[2,i,:]==0] = 0
        informal_income_net_of_commuting_costs[i, :][
            new_simul_households[3,i,:]==0] = 0
        rdp_income_net_of_commuting_costs[i, :][
            new_simul_households[4,i,:]==0] = 0
        
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
    
    return new_income_net_of_commuting_costs

new_income_net_of_commuting_costs_base = gen_new_income_per_scenario(
    income_net_of_commuting_costs, new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households)

new_income_net_of_commuting_costs_noUE = gen_new_income_per_scenario(
    income_net_of_commuting_costs, new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households)

new_income_net_of_commuting_costs_newIS = gen_new_income_per_scenario(
    income_net_of_commuting_costs, new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households)

new_income_net_of_commuting_costs_newRDP = gen_new_income_per_scenario(
    income_net_of_commuting_costs, new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households)

new_income_net_of_commuting_costs_Aup = gen_new_income_per_scenario(
    income_net_of_commuting_costs, new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households)

new_income_net_of_commuting_costs_subsid = gen_new_income_per_scenario(
    income_net_of_commuting_costs, new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households)

new_income_net_of_commuting_costs_Evict = gen_new_income_per_scenario(
    income_net_of_commuting_costs, new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households)

## WE ALSO RESHAPE AMENITIES FOR PLOTTING

new_amenities = np.tile(amenities, (5,1))

###############################################################################

# %% FIGURE 1: HH BREAKDOWN PER HOUSING TYPE

print("FIGURE 1: HH BREAKDOWN PER HOUSING TYPE")

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

plt.ioff()

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

plt.close(fig)



###############################################################################

# %% FIGURES 2-5: BASELINE DISTRIBUTIONS ACROSS KEY VARIABLES

print("FIGURES 2-5: BASELINE DISTRIBUTIONS ACROSS KEY VARIABLES")

# WE DEFINE THE MASTER PLOT FUNCTION

def plot_baseline_distrib_per_incgrp(new_simul_array, new_simul_households,
                                     xlabel, xvar_name, end_option="percentile"):

    plt.ioff()
    
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
    plt.savefig(path_output_plots + f'/baseline/{xvar_name}/baseline_{xvar_name}_pop_distrib.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    return

def plot_baseline_distrib_agg(new_simul_array, new_simul_households,
                              xlabel, xvar_name, end_option="percentile"):

    plt.ioff()
    
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
    plt.savefig(path_output_plots + f'/baseline/{xvar_name}/baseline_agg_{xvar_name}_pop_distrib.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    return

# WE DECLINE THE FUNCTION ACROSS KEY VARIABLES

## FIGURE 2: RENT

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across housing rent brackets and housing type for each income group. The rents are given in South African
## rands per squared meter per year at 2011 values. Rent brackets are computed so as to split the total
## distribution in 100 and go up to the 99th percentile. The rent for formal subsidized housing is taken as zero.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

plot_baseline_distrib_per_incgrp(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Annual rent (ZAR/m²)", "rent")

plot_baseline_distrib_agg(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Annual rent (ZAR/m²)", "rent")

## FIGURE 3: INCOME

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across expected income brackets and housing type for each income group. The incomes are
## net of commuting costs and are given in South African rands per year at 2011 values.
## Income brackets are computed so as to split the total distribution in 100
## and go up to the 99th percentile. The expected income does not include potential added revenues
## from supplying backyard housing or added expenses from construction or maintenance costs.
## It is calibrated at baseline and reflects the intrinsic value of a residential location.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

plot_baseline_distrib_per_incgrp(
    new_income_net_of_commuting_costs_base,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Expected income net of commuting costs (ZAR/year)", "income")

plot_baseline_distrib_agg(
    new_income_net_of_commuting_costs_base,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Expected income net of commuting costs (ZAR/year)", "income")

## FIGURE 4: DWELLING SIZE

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across dwelling size brackets and housing type for each income group. The dwelling sizes
## are given in squared meters. Size brackets are computed so as to split the total distribution
## in 100 and go up to the maximum value. The dwelling size for formal subsidized housing does not
## include the backyard size and refers only to the size of the dwelling unit itself.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

plot_baseline_distrib_per_incgrp(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Dwelling size (m²)", "dsize", end_option="max")

plot_baseline_distrib_agg(
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Dwelling size (m²)", "dsize", end_option="max")

## FIGURE 5: AMENITY INDEX

## FOOTNOTE: The plot shows the distribution of the number of households at baseline
## across amenity brackets and housing type for each income group. The amenity index
## is calibrated at baseline and normalized around one. It does not incorporate disamenity
## factors related to the nature of housing types and  reflects the intrinsic value
## of a residential location. Amenity brackets are computed so as to split the total distribution
## in 100 and go up to the 99th percentile. The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

plot_baseline_distrib_per_incgrp(
    new_amenities,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    "Amenity index", "amenity")

plot_baseline_distrib_agg(
    new_amenities,
    np.nansum(new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households, 1),
    "Amenity index", "amenity")

###############################################################################

# %% FIGURES 6-9: SCENARIO DISTRIBUTIONS ACROSS KEY VARIABLES (select main scenario to show)

print("FIGURES 6-9: SCENARIO DISTRIBUTIONS ACROSS KEY VARIABLES (select main scenario to show)")

## FIGURE 6: RENT

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the distribution of the number of households
## across housing rent brackets and housing type for each income group. The rents are given in South African
## rands per squared meter per year at 2011 values. Rent brackets are computed so as to split the total
## distribution in 100 and go up to the 99th percentile. The rent for formal subsidized housing is taken as zero.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

## FIGURE 7: INCOME

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the distribution of the number households
## across expected income brackets and housing type for each income group. The incomes are
## net of commuting costs and are given in South African rands per year at 2011 values.
## Income brackets are computed so as to split the total distribution in 100
## and go up to the 99th percentile. The expected income does not include potential added revenues
## from supplying backyard housing or added expenses from construction or maintenance costs.
## It is calibrated at baseline and reflects the intrinsic value of a residential location.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

## FIGURE 8: DWELLING SIZE

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the distribution of the number of households
## across dwelling size brackets and housing type for each income group. The dwelling sizes
## are given in squared meters. Size brackets are computed so as to split the total distribution
## in 100 and go up to the maximum value. The dwelling size for formal subsidized housing does not
## include the backyard size and refers only to the size of the dwelling unit itself.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

## FIGURE 9: AMENITY INDEX

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the distribution of the number of households
## across amenity brackets and housing type for each income group. The amenity index
## is calibrated at baseline and normalized around one. It does not incorporate disamenity
## factors related to the nature of housing types and  reflects the intrinsic value
## of a residential location. Amenity brackets are computed so as to split the total distribution
## in 100 and go up to the 99th percentile. The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)


## WE DEFINE THE MAIN PLOTTING FUNCTIONS

### FIRST FOR EACH INCOME GROUP

def plot_comparison_distrib_per_incgrp(baseline_array, baseline_households,
                                       scenario_array, scenario_households,
                                       xlabel, xvar_name, compar_name,
                                       baseline_label="Baseline",
                                       scenario_label="Scenario",
                                       end_option="percentile"):
        
    plt.ioff()

    n_locations = 24014
    n_housing_types = 5
    n_income_groups = 4
    housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]
    income_group_labels = ["Low inc.", "Mid-low inc.", "Mid-high inc.", "High inc."]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for income_idx in range(n_income_groups):
        ax = axes[income_idx]

        combined_values = np.concatenate([
            baseline_array[baseline_households[:,income_idx,:]>0],
            scenario_array[scenario_households[:,income_idx,:]>0]
        ])
        start = min(combined_values)
        
        if end_option == "percentile":
            end = np.nanquantile(combined_values, 0.99)
        if end_option == "max":
            end = max(combined_values) + 1
        
        array_bins = np.arange(start, end, (end-start)/100)
        bin_centers = (array_bins[:-1] + array_bins[1:]) / 2
        
        binned_pop_baseline = np.zeros((len(array_bins)-1, n_housing_types))
        for housing_idx in range(n_housing_types):
            for loc_idx in range(n_locations):
                array_value = baseline_array[housing_idx, loc_idx]
                pop_value = baseline_households[housing_idx, income_idx, loc_idx]
                
                bin_idx = np.digitize(array_value, array_bins) - 1
                if 0 <= bin_idx < len(array_bins) - 1:
                    binned_pop_baseline[bin_idx, housing_idx] += pop_value
        
        binned_pop_scenario = np.zeros((len(array_bins)-1, n_housing_types))
        for housing_idx in range(n_housing_types):
            for loc_idx in range(n_locations):
                array_value = scenario_array[housing_idx, loc_idx]
                pop_value = scenario_households[housing_idx, income_idx, loc_idx]
                
                bin_idx = np.digitize(array_value, array_bins) - 1
                if 0 <= bin_idx < len(array_bins) - 1:
                    binned_pop_scenario[bin_idx, housing_idx] += pop_value
        
        bottom_baseline = np.zeros(len(array_bins)-1)
        colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
        
        for housing_idx in range(n_housing_types):
            ax.bar(bin_centers, binned_pop_baseline[:, housing_idx], 
                   width=array_bins[1]-array_bins[0], 
                   bottom=bottom_baseline,
                   label=f'{housing_type_labels[housing_idx]} [{baseline_label}]',
                   color=colors[housing_idx],
                   alpha=0.8,
                   edgecolor='white',
                   linewidth=0)
            bottom_baseline += binned_pop_baseline[:, housing_idx]
        
        bottom_scenario = np.zeros(len(array_bins)-1)
        
        for housing_idx in range(n_housing_types):
            ax.bar(bin_centers, binned_pop_scenario[:, housing_idx], 
                   width=array_bins[1]-array_bins[0], 
                   bottom=bottom_scenario,
                   label=f'{housing_type_labels[housing_idx]} [{scenario_label}]',
                   color=colors[housing_idx],
                   alpha=0.4,
                   edgecolor='black',
                   linewidth=0,
                   hatch="...")
            bottom_scenario += binned_pop_scenario[:, housing_idx]
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Nb of HHs', fontsize=10)
        ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_output_plots + f'/compar/{compar_name}/compar_{compar_name}_{xvar_name}_pop_distrib.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    return

### THEN FOR AGGREGATE DISTRIBUTIONS

def plot_comparison_distrib_all_incgrp(baseline_array, baseline_households,
                                       scenario_array, scenario_households,
                                       xlabel, xvar_name, compar_name,
                                       baseline_label="Baseline",
                                       scenario_label="Scenario",
                                       end_option="percentile"):
   
    plt.ioff()
    
    n_locations = 24014
    n_housing_types = 5
    housing_type_labels = ["FP", "IB (basic)", "IB (redev.)", "IS", "FS"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    baseline_valid = baseline_array[baseline_households.sum(axis=1) > 0]
    scenario_valid = scenario_array[scenario_households.sum(axis=1) > 0]
    combined_values = np.concatenate([baseline_valid, scenario_valid])
    start = min(combined_values)
    
    if end_option == "percentile":
        end = np.nanquantile(combined_values, 0.99)
    if end_option == "max":
        end = max(combined_values) + 1
    
    array_bins = np.arange(start, end, (end-start)/100)
    bin_centers = (array_bins[:-1] + array_bins[1:]) / 2

    binned_pop_baseline = np.zeros((len(array_bins)-1, n_housing_types))
    for housing_idx in range(n_housing_types):
        for loc_idx in range(n_locations):
            array_value = baseline_array[housing_idx, loc_idx]

            pop_value = baseline_households[housing_idx, :, loc_idx].sum()
            
            bin_idx = np.digitize(array_value, array_bins) - 1
            if 0 <= bin_idx < len(array_bins) - 1:
                binned_pop_baseline[bin_idx, housing_idx] += pop_value
    
    binned_pop_scenario = np.zeros((len(array_bins)-1, n_housing_types))
    for housing_idx in range(n_housing_types):
        for loc_idx in range(n_locations):
            array_value = scenario_array[housing_idx, loc_idx]

            pop_value = scenario_households[housing_idx, :, loc_idx].sum()
            
            bin_idx = np.digitize(array_value, array_bins) - 1
            if 0 <= bin_idx < len(array_bins) - 1:
                binned_pop_scenario[bin_idx, housing_idx] += pop_value
    
    bottom_baseline = np.zeros(len(array_bins)-1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
    
    for housing_idx in range(n_housing_types):
        ax.bar(bin_centers, binned_pop_baseline[:, housing_idx], 
               width=array_bins[1]-array_bins[0], 
               bottom=bottom_baseline,
               label=f'{housing_type_labels[housing_idx]} [{baseline_label}]',
               color=colors[housing_idx],
                alpha=0.8,
                edgecolor='white',
                linewidth=0)
        bottom_baseline += binned_pop_baseline[:, housing_idx]
    
    bottom_scenario = np.zeros(len(array_bins)-1)
    
    for housing_idx in range(n_housing_types):
        ax.bar(bin_centers, binned_pop_scenario[:, housing_idx], 
               width=array_bins[1]-array_bins[0], 
               bottom=bottom_scenario,
               label=f'{housing_type_labels[housing_idx]} [{scenario_label}]',
               color=colors[housing_idx],
               alpha=0.4,
               edgecolor='black',
               linewidth=0,
               hatch="...")
        bottom_scenario += binned_pop_scenario[:, housing_idx]
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Nb of HHs', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_output_plots + f'/compar/{compar_name}/compar_{compar_name}_{xvar_name}_pop_distrib_all_inc.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    return

## WE DECLINE THE FUNCTIONS FOR EACH OUTCOME OF INTEREST ACROSS SCENARIOS

array_varnames = ['rent', 'dwelling_size', 'income_net_of_commuting_costs', 'amenities']

scenario_names = ['noUE', 'newIS', 'newRDP', 'Aup', 'subsid', 'Evict']

baseline_array_dict = {'rent': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
                       'dwelling_size': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
                       'income_net_of_commuting_costs': new_income_net_of_commuting_costs_base,
                       'amenities': new_amenities}

noUE_array_dict = {'rent': new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
                   'dwelling_size': new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
                   'income_net_of_commuting_costs': new_income_net_of_commuting_costs_noUE,
                   'amenities': new_amenities}

newIS_array_dict = {'rent': new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
                    'dwelling_size': new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
                    'income_net_of_commuting_costs': new_income_net_of_commuting_costs_newIS,
                    'amenities': new_amenities}

newRDP_array_dict = {'rent': new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent,
                     'dwelling_size': new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
                     'income_net_of_commuting_costs': new_income_net_of_commuting_costs_newRDP,
                     'amenities': new_amenities}

Aup_array_dict = {'rent': new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent,
                  'dwelling_size': new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size,
                  'income_net_of_commuting_costs': new_income_net_of_commuting_costs_Aup,
                  'amenities': new_amenities}

subsid_array_dict = {'rent': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent,
                     'dwelling_size': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size,
                     'income_net_of_commuting_costs': new_income_net_of_commuting_costs_subsid,
                     'amenities': new_amenities}

Evict_array_dict = {'rent': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent,
                    'dwelling_size': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size,
                    'income_net_of_commuting_costs': new_income_net_of_commuting_costs_Evict,
                    'amenities': new_amenities}

scenarios_array_dict = {'noUE': noUE_array_dict, 'newIS': newIS_array_dict,
                        'newRDP': newRDP_array_dict, 'Aup': Aup_array_dict,
                        'subsid': subsid_array_dict, 'Evict': Evict_array_dict}

array_label_dict = {'rent': "Annual rent (ZAR/m²)",
                    'dwelling_size': "Dwelling size (m²)",
                    'income_net_of_commuting_costs': "Expected income net of commuting costs (ZAR/year)",
                    'amenities': "Amenity index"}

scenario_labels_dict = {'noUE': "No UE", 'newIS': "New IS", 'newRDP': "New FS",
                        'Aup': "Disam. -50pc", 'subsid': "Subsidies", 'Evict': "Toler. -50pc"}

scenario_hhs_dict = {'noUE': new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
                     'newIS': new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
                     'newRDP': new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households,
                     'Aup': new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households,
                     'subsid': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households,
                     'Evict': new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households}

baseline_households = new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households

## WE EMBED THE PROCESS IN A DEDICATED FUCTION FOR PARALLEL COMPUTING

def process_scenario_variable(scenario_name, array_varname, 
                              baseline_array_dict, baseline_households,
                              scenarios_array_dict, scenario_hhs_dict,
                              array_label_dict, scenario_labels_dict):
    
    end_opt = 'max' if array_varname == 'dwelling_size' else 'percentile'
    
    plot_comparison_distrib_per_incgrp(
        baseline_array_dict[array_varname],
        baseline_households,
        scenarios_array_dict[scenario_name][array_varname],
        scenario_hhs_dict[scenario_name],
        array_label_dict[array_varname], 
        array_varname,
        scenario_name, 
        scenario_label=scenario_labels_dict[scenario_name],
        end_option=end_opt)
    
    plot_comparison_distrib_all_incgrp(
        baseline_array_dict[array_varname],
        baseline_households,
        scenarios_array_dict[scenario_name][array_varname],
        scenario_hhs_dict[scenario_name],
        array_label_dict[array_varname], 
        array_varname,
        scenario_name, 
        scenario_label=scenario_labels_dict[scenario_name],
        end_option=end_opt)
    
    plt.close('all')
    
    return

n_jobs = min(multiprocessing.cpu_count() - 1, len(scenario_names))

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_scenario_variable)(
        scenario_name, array_varname,
        baseline_array_dict, baseline_households,
        scenarios_array_dict, scenario_hhs_dict,
        array_label_dict, scenario_labels_dict)
    for scenario_name in scenario_names
    for array_varname in array_varnames
)

###############################################################################

# %% FIGURES 6-9 bis: WITH DENSITIES

print("FIGURES 6-9 bis: WITH DENSITIES")

## FIGURE 6: RENT

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the density of households
## over housing rent for each income group. The rents are given in South African
## rands per squared meter per year at 2011 values. Densities are estimated using Gaussian kernel with
## Scott's rule for bandwidth factor, and are rendered using 100 equally spaced data points over the whole data range.
## The rent for formal subsidized housing is taken as zero.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

## FIGURE 7: INCOME

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the desnity of households
## over expected income for each income group. The incomes are
## net of commuting costs and are given in South African rands per year at 2011 values.
## Densities are estimated using Gaussian kernel with
## Scott's rule for bandwidth factor, and are rendered using 100 equally spaced data points over the whole data range.
## The expected income does not include potential added revenues
## from supplying backyard housing or added expenses from construction or maintenance costs.
## It is calibrated at baseline and reflects the intrinsic value of a residential location.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

## FIGURE 8: DWELLING SIZE

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the density of households
## over dwelling size for each income group. The dwelling sizes
## are given in squared meters. Densities are estimated using Gaussian kernel with
## Scott's rule for bandwidth factor, and are rendered using 100 equally spaced data points
## over the whole data range. The dwelling size for formal subsidized housing does not
## include the backyard size and refers only to the size of the dwelling unit itself.
## The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

## FIGURE 9: AMENITY INDEX

## FOOTNOTE: The plot compares counterfactual and baseline scenarios for the density of households
## over amenity across for each income group. The amenity index is calibrated at baseline and normalized around one.
## Densities are estimated using Gaussian kernel with
## Scott's rule for bandwidth factor, and are rendered using 100 equally spaced data points over the whole data range.
## The amenity index does not incorporate disamenity
## factors related to the nature of housing types and  reflects the intrinsic value
## of a residential location. The x and y axes are not the same across income groups.
## (The aggregate distribution across income groups is also shown for reference.)

## WE DEFINE ALL PAIRWISE COMPARISONS TO BE USED WITH BASELINE

households_compar_UE = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households])
households_compar_newIS = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households])
households_compar_newRDP = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_households])
households_compar_Aup = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_households])
households_compar_subsid = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_households])
households_compar_Evict = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_households])

rent_compar_UE = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent])
rent_compar_newIS = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent])
rent_compar_newRDP = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_rent])
rent_compar_Aup = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_rent])
rent_compar_subsid = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_rent])
rent_compar_Evict = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_rent])

dwelling_size_compar_UE = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size])
dwelling_size_compar_newIS = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size])
dwelling_size_compar_newRDP = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_dwelling_size])
dwelling_size_compar_Aup = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_dwelling_size])
dwelling_size_compar_subsid = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1_dwelling_size])
dwelling_size_compar_Evict = np.stack([
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_dwelling_size,
    new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1_dwelling_size])

income_compar_UE = np.stack([
    new_income_net_of_commuting_costs_base,
    new_income_net_of_commuting_costs_noUE])
income_compar_newIS = np.stack([
    new_income_net_of_commuting_costs_base,
    new_income_net_of_commuting_costs_newIS])
income_compar_newRDP = np.stack([
    new_income_net_of_commuting_costs_base,
    new_income_net_of_commuting_costs_newRDP])
income_compar_Aup = np.stack([
    new_income_net_of_commuting_costs_base,
    new_income_net_of_commuting_costs_Aup])
income_compar_subsid = np.stack([
    new_income_net_of_commuting_costs_base,
    new_income_net_of_commuting_costs_subsid])
income_compar_Evict = np.stack([
    new_income_net_of_commuting_costs_base,
    new_income_net_of_commuting_costs_Evict])

amenities_compar = np.stack([
    new_amenities,
    new_amenities])

## WE STORE THEM IN DICTIONARIES FOR LOOPING

compar_hhs = {'noUE': households_compar_UE, 'newIS': households_compar_newIS, 'newRDP': households_compar_newRDP,
              'Aup': households_compar_Aup, 'subsid': households_compar_subsid, 'Evict': households_compar_Evict}

compar_rent = {'noUE': rent_compar_UE, 'newIS': rent_compar_newIS, 'newRDP': rent_compar_newRDP,
               'Aup': rent_compar_Aup, 'subsid': rent_compar_subsid, 'Evict': rent_compar_Evict}

compar_dwelling_size = {'noUE': dwelling_size_compar_UE, 'newIS': dwelling_size_compar_newIS, 'newRDP': dwelling_size_compar_newRDP,
                        'Aup': dwelling_size_compar_Aup, 'subsid': dwelling_size_compar_subsid, 'Evict': dwelling_size_compar_Evict}

compar_income_net_of_commuting_costs = {
    'noUE': income_compar_UE, 'newIS': income_compar_newIS, 'newRDP': income_compar_newRDP,
    'Aup': income_compar_Aup, 'subsid': income_compar_subsid, 'Evict': income_compar_Evict}

compar_amenities = {'noUE': amenities_compar, 'newIS': amenities_compar, 'newRDP': amenities_compar,
                    'Aup': amenities_compar, 'subsid': amenities_compar, 'Evict': amenities_compar}

compar_array = {'rent': compar_rent, 'dwelling_size': compar_dwelling_size,
                'income_net_of_commuting_costs': compar_income_net_of_commuting_costs, 'amenities': compar_amenities}

## WE DEFINE THE MAIN PLOTTING FUNCTIONS

def compar_density_plots(new_array_scenarios, new_households_scenarios,
                         n_scenarios, scenario_labels, xlabel, xvar_name,
                         compar_name, income_grp_bounds=None):

    plt.ioff()
    
    n_locations = 24014
    n_housing_types = 5
    n_income_groups = 4

    array = new_array_scenarios

    population = new_households_scenarios

    n_points = 100

    income_group_labels = ["Low inc.", "Mid-low inc.", "Mid-high inc.", "High inc."]

    colors = plt.cm.Paired(np.linspace(0, 1, n_scenarios))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for income_idx in range(n_income_groups):
        ax = axes[income_idx]

        for scenario_idx in range(n_scenarios):

            array_values = []
            pop_values = []
            
            for loc_idx in range(n_locations):
                
                for housing_idx in range(n_housing_types):
                    r = array[scenario_idx, housing_idx, loc_idx]
                    p = population[scenario_idx, housing_idx, income_idx, loc_idx]

                    if p > 0:
                        array_values.append(r)
                        pop_values.append(p)
            
            if len(array_values) > 0:
                array_values = np.array(array_values)
                pop_values = np.array(pop_values)

                array_weighted = np.repeat(array_values, pop_values.astype(int))
                
                if len(array_weighted) > 1:
                    
                    kde = gaussian_kde(array_weighted, bw_method='scott')

                    array_min = array_values.min()
                    array_max = array_values.max()

                    if income_grp_bounds is not None:
                        if income_idx==0:
                            array_min = np.nanmax([income_grp_bounds[0,0], array_min])
                            array_max = np.nanmin([income_grp_bounds[0,1], array_max])
                        elif income_idx==1:
                            array_min = np.nanmax([income_grp_bounds[1,0], array_min])
                            array_max = np.nanmin([income_grp_bounds[1,1], array_max])
                        elif income_idx==2:
                            array_min = np.nanmax([income_grp_bounds[2,0], array_min])
                            array_max = np.nanmin([income_grp_bounds[2,1], array_max])
                        elif income_idx==3:
                            array_min = np.nanmax([income_grp_bounds[3,0], array_min])
                            array_max = np.nanmin([income_grp_bounds[3,1], array_max])
                    
                    array_range = np.linspace(array_min, array_max, n_points)

                    density = kde(array_range)
                    
                    density_scaled = density
                    
                    # Ad hoc correction
                    density_scaled[density_scaled>100] = 4

                    ax.plot(array_range, density_scaled, 
                           label=scenario_labels[scenario_idx],
                           color=colors[scenario_idx],
                           linewidth=1,
                           alpha=0.9)

                    ax.fill_between(array_range, density_scaled, 
                                   alpha=0.2, 
                                   color=colors[scenario_idx])

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(income_group_labels[income_idx], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='both', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(path_output_plots + f'/compar/{compar_name}/dens/dens_compar_{compar_name}_{xvar_name}_pop_distrib.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)

    return

def compar_density_plots_all_incgrp(new_array_scenarios, new_households_scenarios,
                                    n_scenarios, scenario_labels, xlabel, xvar_name,
                                    compar_name, overall_bounds=None):
    
    plt.ioff()
    
    n_locations = 24014
    n_housing_types = 5

    array = new_array_scenarios
    population = new_households_scenarios
    n_points = 100
    colors = plt.cm.Paired(np.linspace(0, 1, n_scenarios))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for scenario_idx in range(n_scenarios):
        array_values = []
        pop_values = []
        
        for loc_idx in range(n_locations):
            for housing_idx in range(n_housing_types):
                r = array[scenario_idx, housing_idx, loc_idx]

                p = population[scenario_idx, housing_idx, :, loc_idx].sum()
                if p > 0:
                    array_values.append(r)
                    pop_values.append(p)
        
        if len(array_values) > 0:
            array_values = np.array(array_values)
            pop_values = np.array(pop_values)
            array_weighted = np.repeat(array_values, pop_values.astype(int))
            
            if len(array_weighted) > 1:
                
                kde = gaussian_kde(array_weighted, bw_method='scott')
                array_min = array_values.min()
                array_max = array_values.max()
                
                if overall_bounds is not None:
                    array_min = np.nanmax([overall_bounds[0], array_min])
                    array_max = np.nanmin([overall_bounds[1], array_max])
                
                array_range = np.linspace(array_min, array_max, n_points)
                density = kde(array_range)
                
                density_scaled = density
                
                # Ad hoc correction
                density_scaled[density_scaled > 100] = 4
                
                ax.plot(array_range, density_scaled, 
                       label=scenario_labels[scenario_idx],
                       color=colors[scenario_idx],
                       linewidth=2,
                       alpha=0.9)
                ax.fill_between(array_range, density_scaled, 
                               alpha=0.2, 
                               color=colors[scenario_idx])
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_output_plots + f'/compar/{compar_name}/dens/dens_compar_{compar_name}_{xvar_name}_pop_distrib_all_inc.png', dpi=300, bbox_inches='tight')

    plt.close(fig)
    
    return

## WE LOOP OVER THE SCENARIOS

def process_scenario_variable_dens(scenario_name, array_varname, 
                                   compar_array, compar_hhs,
                                   array_label_dict, scenario_labels_dict):
    
    compar_density_plots(compar_array[array_varname][scenario_name], compar_hhs[scenario_name],
                         2, ["Baseline", scenario_labels_dict[scenario_name]],
                         array_label_dict[array_varname], array_varname, scenario_name)
    
    compar_density_plots_all_incgrp(compar_array[array_varname][scenario_name], compar_hhs[scenario_name],
                         2, ["Baseline", scenario_labels_dict[scenario_name]],
                         array_label_dict[array_varname], array_varname, scenario_name)
    
    plt.close('all')
    
    return

n_jobs = min(multiprocessing.cpu_count() - 1, len(scenario_names))

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_scenario_variable_dens)(
        scenario_name, array_varname, 
        compar_array, compar_hhs,
        array_label_dict, scenario_labels_dict)
    for scenario_name in scenario_names
    for array_varname in array_varnames
)

###############################################################################

# %%

# FIGURES 10-11: 3D MAPS IN ABSOLUTE VALUES (choose variable and scenario)

## FOOTNOTE: The plot shows the spatial distribution of the number of households per grid cell
## with corresponding variable levels for each income group, at baseline and for given scenario.
## [Choose variable and scenario for full description]
## (The aggregate distribution across income groups is also shown for reference.)
## The z axes and color scales are comparable across scenarios.

## WE COMPUTE AVERAGE ARRAY VALUES ACROSS HOUSING TYPES TO PLOT IN EACH LOCATION

def calculate_weighted_average_rent(population_array, rent_array, income_idx):

    pop_income = population_array[:, income_idx, :]

    weighted_rents = np.zeros(population_array.shape[2])
    
    for loc_idx in range(population_array.shape[2]):
        total_pop = pop_income[:, loc_idx].sum()
        if total_pop > 0:
            weighted_rents[loc_idx] = np.sum(rent_array[:, loc_idx] * pop_income[:, loc_idx]) / total_pop
        else:
            weighted_rents[loc_idx] = 0
    
    return weighted_rents

def calculate_weighted_average_rent_allinc(population_array, rent_array):

    pop_income = population_array.sum(axis=1)

    weighted_rents = np.zeros(population_array.shape[2])
    
    for loc_idx in range(population_array.shape[2]):
        total_pop = pop_income[:, loc_idx].sum()
        if total_pop > 0:
            weighted_rents[loc_idx] = np.sum(rent_array[:, loc_idx] * pop_income[:, loc_idx]) / total_pop
        else:
            weighted_rents[loc_idx] = 0
    
    return weighted_rents

## WE IMPORT BASEMAP FOR PLOTS

def add_basemap_to_3d(ax, gdf, alpha=0.3):

    bounds = gdf.total_bounds

    bounds_padded = [
        bounds[0], 
        bounds[1], 
        bounds[2], 
        bounds[3]
    ]

    transformer_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    xmin_merc, ymin_merc = transformer_to_merc.transform(bounds_padded[0], bounds_padded[1])
    xmax_merc, ymax_merc = transformer_to_merc.transform(bounds_padded[2], bounds_padded[3])

    basemap, extent_merc = ctx.bounds2img(
        xmin_merc, ymin_merc, xmax_merc, ymax_merc,
        ll=False,
        source=ctx.providers.CartoDB.Positron,
        zoom='auto'
    )

    n_points_x = basemap.shape[1]
    n_points_y = basemap.shape[0]
    
    x_merc = np.linspace(extent_merc[0], extent_merc[1], n_points_x)
    y_merc = np.linspace(extent_merc[2], extent_merc[3], n_points_y)

    transformer_to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    X_latlon = np.zeros((n_points_y, n_points_x))
    Y_latlon = np.zeros((n_points_y, n_points_x))
    
    for i in range(n_points_y):
        for j in range(n_points_x):
            lon, lat = transformer_to_latlon.transform(x_merc[j], y_merc[i])
            X_latlon[i, j] = lon
            Y_latlon[i, j] = lat

    Z = np.zeros_like(X_latlon)

    basemap_flipped = np.flip(basemap, axis=0)

    ax.plot_surface(X_latlon, Y_latlon, Z, 
                   rstride=10, cstride=10,
                   facecolors=basemap_flipped/255,
                   shade=False,
                   alpha=alpha,
                   zorder=0)
    
    return

## WE DEFINE PLOTTING FUNCTION FOR ALL INCOME GROUPS

def plot_four_income_groups_3d(elev, gdf, population_data_dict, rent_data_dict,
                                xvar_name, xvar_label, incgrp_names, incgrp_labels, scenario_name,
                                z_bounds, v_bounds,
                                add_colorbar=True, add_basemap=True, basemap_alpha=0.3):

    plt.ioff()
    
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        print(f"Converting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    fig = plt.figure(figsize=(20, 16))
    
    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values 
    y_coords = centroids.y.values 
    
    lon_range = x_coords.max() - x_coords.min()
    lat_range = y_coords.max() - y_coords.min()
    if lon_range < 1:
        scale_factor = 0.02
    elif lon_range < 10:
        scale_factor = 0.01
    else:
        scale_factor = 0.005
    
    dx = lon_range * scale_factor
    dy = lat_range * scale_factor
    
    all_rents = []
    for incgrp_name in incgrp_names:
        rent_data = np.array(rent_data_dict[incgrp_name])
        all_rents.extend(rent_data[rent_data > 0])
    
    for idx, (incgrp_name, incgrp_label) in enumerate(zip(incgrp_names, incgrp_labels)):
        
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d', computed_zorder=False)
        
        if add_basemap:
            add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)
        
        population_data = np.array(population_data_dict[incgrp_name])
        rent_data = np.array(rent_data_dict[incgrp_name])
        
        heights = population_data
        rents = rent_data
        
        non_zero_mask = heights != 0
        
        x_coords_filtered = x_coords[non_zero_mask]
        y_coords_filtered = y_coords[non_zero_mask]
        heights_filtered = heights[non_zero_mask]
        rents_filtered = rents[non_zero_mask]
        
        vmin = v_bounds[idx, 0]
        vmax = v_bounds[idx, 1]
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        cmap = plt.cm.YlOrRd
        
        colors = cmap(norm(rents_filtered))
        colors[rents_filtered == 0] = [0.7, 0.7, 0.7, 0.8]
        
        if len(x_coords_filtered) > 0:
            ax.bar3d(x_coords_filtered, y_coords_filtered, np.zeros_like(heights_filtered), 
                    dx, dy, heights_filtered, color=colors, alpha=0.8, 
                    edgecolor='none', linewidth=0, shade=True, zorder=10)
        
        z_min = z_bounds[idx, 0]
        z_max = z_bounds[idx, 1]
        
        ax.set_zlim(min(0, z_min), max(0, z_max))
        
        ax.set_xlabel('Longitude', labelpad=5)
        ax.set_ylabel('Latitude', labelpad=20)
        ax.set_zlabel('Nb of HHs', labelpad=20)
        ax.set_title(incgrp_label, fontsize=12, fontweight='bold')
        
        ax.tick_params(axis='x', pad=5, labelsize=8)
        ax.tick_params(axis='y', pad=15, labelsize=8)
        ax.tick_params(axis='z', pad=15, labelsize=8)
        
        ax.view_init(elev=elev, azim=270)
        ax.grid(True, alpha=0.3)

        if add_colorbar and len(rents_filtered) > 0:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.05)
            cbar.set_label(xvar_label, rotation=270, labelpad=15, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(path_output_plots + f'maps/{scenario_name}/map_spatial_pop_{xvar_name}_distrib.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return

## ALSO FOR AGGREGATE POPULATION

def plot_single_income_group_3d(elev, gdf, population_data, rent_data,
                                xvar_name, xvar_label, scenario_name,
                                z_bounds_agg, v_bounds_agg,
                                add_colorbar=True, add_basemap=True, basemap_alpha=0.3):

    plt.ioff()
    
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        print(f"Converting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    if add_basemap:
        add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)

    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values 
    y_coords = centroids.y.values 
    
    heights = np.array(population_data)
    rents = np.array(rent_data)
  
    non_zero_mask = heights != 0
    
    x_coords_filtered = x_coords[non_zero_mask]
    y_coords_filtered = y_coords[non_zero_mask]
    heights_filtered = heights[non_zero_mask]
    rents_filtered = rents[non_zero_mask]

    lon_range = x_coords.max() - x_coords.min()
    lat_range = y_coords.max() - y_coords.min()

    if lon_range < 1:
        scale_factor = 0.02
    elif lon_range < 10:
        scale_factor = 0.01
    else:
        scale_factor = 0.005
    
    dx = lon_range * scale_factor
    dy = lat_range * scale_factor
    
    vmin = v_bounds_agg[0]
    vmax = v_bounds_agg[1]
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd
    colors = cmap(norm(rents_filtered))

    colors[rents_filtered == 0] = [0.7, 0.7, 0.7, 0.8]

    if len(x_coords_filtered) > 0:
        ax.bar3d(x_coords_filtered, y_coords_filtered, np.zeros_like(heights_filtered), 
                dx, dy, heights_filtered, color=colors, alpha=0.8, 
                edgecolor='none', linewidth=0, shade=True, zorder=10)

    z_min = z_bounds_agg[0]
    z_max = z_bounds_agg[1]    
    
    ax.set_zlim(min(0, z_min), max(0, z_max))
    
    ax.set_xlabel('Longitude', labelpad=5)
    ax.set_ylabel('Latitude', labelpad=30)
    ax.set_zlabel('Nb of HHs', labelpad=30)

    ax.tick_params(axis='x', pad=5)
    ax.tick_params(axis='y', pad=20)
    ax.tick_params(axis='z', pad=20)
    
    ax.view_init(elev=elev, azim=270)
    ax.grid(True, alpha=0.3)

    if add_colorbar and len(rents_filtered) > 0:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.01)
        cbar.set_label(xvar_label, rotation=270, labelpad=20)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.tight_layout()
    plt.savefig(path_output_plots + f'maps/{scenario_name}/map_spatial_pop_all_{xvar_name}_distrib.png', dpi=300, bbox_inches='tight')

    plt.close(fig)

    return

## WE LOOP OVER THE SCENARIOS

def process_scenario_variable_maps(scenario_name, array_varname, 
                                   population_array, rent_array,
                                   array_label_dict_map):
    
    population_data_dict = {'poor': population_array[scenario_name][:, 0, :].sum(axis=0),
                            'midpoor': population_array[scenario_name][:, 1, :].sum(axis=0),
                            'midrich': population_array[scenario_name][:, 2, :].sum(axis=0),
                            'rich': population_array[scenario_name][:, 3, :].sum(axis=0)}
    
    rent_data_dict = {'poor': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 0),
                      'midpoor': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 1),
                      'midrich': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 2),
                      'rich': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 3)}
    
    plot_four_income_groups_3d(
        70, gdf, population_data_dict, rent_data_dict,
        array_varname, array_label_dict_map[array_varname],
        incgrp_names, incgrp_labels, scenario_name,
        z_bounds, v_bounds_dict[array_varname],
        add_colorbar=True, add_basemap=True, basemap_alpha=0.3)
    
    population_data = population_array[scenario_name].sum(axis=(0,1))
    rent_data = calculate_weighted_average_rent_allinc(
        population_array[scenario_name], rent_array[scenario_name][array_varname])
    
    plot_single_income_group_3d(70, gdf, population_data, rent_data,
                                array_varname, array_label_dict_map[array_varname],
                                scenario_name,
                                z_bounds_agg, v_bounds_agg_dict[array_varname],
                                add_colorbar=True, add_basemap=True, basemap_alpha=0.3)
    
    plt.close('all')
    
    return 

## WE ALSO DEFINE A FUNCTION TO RETRIEVE BOUNDS TO BE USED FOR VALUES IN EACH SCENARIO

def process_scenario_variable_maps_simple(scenario_name, array_varname, 
                                   population_array, rent_array,
                                   array_label_dict_map):
    
    population_data_dict = {'poor': population_array[scenario_name][:, 0, :].sum(axis=0),
                            'midpoor': population_array[scenario_name][:, 1, :].sum(axis=0),
                            'midrich': population_array[scenario_name][:, 2, :].sum(axis=0),
                            'rich': population_array[scenario_name][:, 3, :].sum(axis=0)}
    
    rent_data_dict = {'poor': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 0),
                      'midpoor': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 1),
                      'midrich': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 2),
                      'rich': calculate_weighted_average_rent(
                          population_array[scenario_name], rent_array[scenario_name][array_varname], 3)}
    
    z_bounds = np.zeros((4,2))

    z_bounds[0, :] = [min(population_data_dict['poor']), max(population_data_dict['poor'])]
    z_bounds[1, :] = [min(population_data_dict['midpoor']), max(population_data_dict['midpoor'])]
    z_bounds[2, :] = [min(population_data_dict['midrich']), max(population_data_dict['midrich'])]
    z_bounds[3, :] = [min(population_data_dict['rich']), max(population_data_dict['rich'])]
    
    v_bounds = np.zeros((4,2))
    v_bounds[0, :] = [min(rent_data_dict['poor'][rent_data_dict['poor']>0]),
                      max(rent_data_dict['poor'][rent_data_dict['poor']>0])]
    v_bounds[1, :] = [min(rent_data_dict['midpoor'][rent_data_dict['midpoor']>0]),
                      max(rent_data_dict['midpoor'][rent_data_dict['midpoor']>0])]
    v_bounds[2, :] = [min(rent_data_dict['midrich'][rent_data_dict['midrich']>0]),
                      max(rent_data_dict['midrich'][rent_data_dict['midrich']>0])]
    v_bounds[3, :] = [min(rent_data_dict['rich'][rent_data_dict['rich']>0]),
                      max(rent_data_dict['rich'][rent_data_dict['rich']>0])]

    population_data = population_array[scenario_name].sum(axis=(0,1))
    rent_data = calculate_weighted_average_rent_allinc(
        population_array[scenario_name], rent_array[scenario_name][array_varname])
    
    z_bounds_agg = np.zeros(2)
    z_bounds_agg[0] = min(population_data)
    z_bounds_agg[1] = max(population_data)
    
    v_bounds_agg = np.zeros(2)
    v_bounds_agg[0] = min(rent_data)
    v_bounds_agg[1] = max(rent_data)
    
    return z_bounds, z_bounds_agg, v_bounds, v_bounds_agg

## WE EXECUTE THE FUNCTIONS

### FIRST WE IMPORT THE INPUTS NEEDED

gdf = gpd.read_file(path_data + "grid_reference_500.shp")

incgrp_names = ['poor', 'midpoor', 'midrich', 'rich']
incgrp_labels = ['Low inc.', 'Mid-low inc.', 'Mid-high inc.', 'High inc.']

array_label_dict_map = {'rent': 'Weighted avg annual rent (ZAR/m²)',
                        'dwelling_size': 'Weighted avg dwelling size (m²)',
                        'income_net_of_commuting_costs': 'Expected income net of commuting costs (ZAR/year)',
                        'amenities': 'Amenity index'}

scenario_names_map = ['baseline', 'noUE', 'newIS', 'newRDP', 'Aup', 'subsid', 'Evict']

base_scenario_hhs_dict = {'baseline': baseline_households}
population_array = base_scenario_hhs_dict | scenario_hhs_dict

base_scenario_array_dict = {'baseline': baseline_array_dict}
rent_array = base_scenario_array_dict | scenarios_array_dict

### THEN WE IMPORT THE PLOT VALUE BOUNDS

z_bounds, z_bounds_agg, v_bounds_rent, v_bounds_agg_rent = process_scenario_variable_maps_simple(
    'baseline', 'rent', 
    population_array, rent_array,
    array_label_dict_map)

_, _, v_bounds_dwelling_size, v_bounds_agg_dwelling_size = process_scenario_variable_maps_simple(
    'baseline', 'dwelling_size', 
    population_array, rent_array,
    array_label_dict_map)

_, _, v_bounds_income_net_of_commuting_costs, v_bounds_agg_income_net_of_commuting_costs = process_scenario_variable_maps_simple(
    'baseline', 'income_net_of_commuting_costs', 
    population_array, rent_array,
    array_label_dict_map)

_, _, v_bounds_amenities, v_bounds_agg_amenities = process_scenario_variable_maps_simple(
    'baseline', 'amenities', 
    population_array, rent_array,
    array_label_dict_map)

v_bounds_dict = {
    'rent': v_bounds_rent, 'dwelling_size': v_bounds_dwelling_size,
    'income_net_of_commuting_costs': v_bounds_income_net_of_commuting_costs, 'amenities': v_bounds_amenities}

v_bounds_agg_dict = {
    'rent': v_bounds_agg_rent, 'dwelling_size': v_bounds_agg_dwelling_size,
    'income_net_of_commuting_costs': v_bounds_agg_income_net_of_commuting_costs, 'amenities': v_bounds_agg_amenities}

### THEN WE LAUNCH PARALLEL COMPUTING

n_jobs = min(multiprocessing.cpu_count() - 1, len(scenario_names))

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_scenario_variable_maps)(
        scenario_name, array_varname, 
        population_array, rent_array,
        array_label_dict_map)
    for scenario_name in scenario_names_map
    for array_varname in array_varnames
)

###############################################################################

# %%

# FIGURES 12-13: 3D MAPS OF HOUSING TYPES (choose scenario)

## FOOTNOTE: The plot shows the spatial distribution of the number of households per grid cell
## with distinct colors for each housing type, at baseline and for given scenario.
## Since the same income group can be found in several housing types within one cell
## (but only one income group can be found in each housing type within one cell),
## the color bars are stacked when relevant.
## (The aggregate distribution across income groups is also shown for reference.)
## The z axes are comparable across scenarios.

## WE DEFINE THE MAIN PLOTTING FUNCTION

def plot_four_income_groups_3d_housing_types(elev, gdf, population_by_housing_type,
                                             incgrp_names, incgrp_labels, scenario_name,
                                             z_bounds, housing_type_labels=None,
                                             add_legend=True, add_basemap=True, basemap_alpha=0.3):

    plt.ioff()
    
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        print(f"Converting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    fig = plt.figure(figsize=(20, 16))

    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values 
    y_coords = centroids.y.values 

    lon_range = x_coords.max() - x_coords.min()
    lat_range = y_coords.max() - y_coords.min()
    if lon_range < 1:
        scale_factor = 0.02
    elif lon_range < 10:
        scale_factor = 0.01
    else:
        scale_factor = 0.005
    
    dx = lon_range * scale_factor
    dy = lat_range * scale_factor

    housing_type_colors = plt.cm.Set3(np.linspace(0, 1, 5))

    if housing_type_labels is None:
        housing_type_labels = [f'Type {i+1}' for i in range(5)]

    for idx, (incgrp_name, incgrp_label) in enumerate(zip(incgrp_names, incgrp_labels)):
        
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d', computed_zorder=False)
        
        if add_basemap:
            add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)

        population_data = population_by_housing_type[:, idx, :]

        total_population = population_data.sum(axis=0)  # shape (24014,)

        non_zero_mask = total_population > 0
        
        x_coords_filtered = x_coords[non_zero_mask]
        y_coords_filtered = y_coords[non_zero_mask]
        population_data_filtered = population_data[:, non_zero_mask]
        
        if len(x_coords_filtered) > 0:
            z_bottom = np.zeros_like(x_coords_filtered)

            for housing_type_idx in range(5):
                heights = population_data_filtered[housing_type_idx, :]

                mask_with_pop = heights > 0
                if mask_with_pop.any():
                    ax.bar3d(x_coords_filtered[mask_with_pop], 
                            y_coords_filtered[mask_with_pop], 
                            z_bottom[mask_with_pop], 
                            dx, dy, 
                            heights[mask_with_pop], 
                            color=housing_type_colors[housing_type_idx], 
                            alpha=0.8, 
                            edgecolor='none', 
                            linewidth=0, 
                            shade=True, 
                            zorder=10)

                z_bottom = z_bottom + heights
        
        z_min = z_bounds[idx, 0]
        z_max = z_bounds[idx, 1]
        ax.set_zlim(min(0, z_min), max(0, z_max))
        
        ax.set_xlabel('Longitude', labelpad=5)
        ax.set_ylabel('Latitude', labelpad=20)
        ax.set_zlabel('Nb of HHs', labelpad=20)
        ax.set_title(incgrp_label, fontsize=12, fontweight='bold')
        
        ax.tick_params(axis='x', pad=5, labelsize=8)
        ax.tick_params(axis='y', pad=15, labelsize=8)
        ax.tick_params(axis='z', pad=15, labelsize=8)
        
        ax.view_init(elev=elev, azim=270)
        ax.grid(True, alpha=0.3)

        if add_legend:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=housing_type_colors[i], 
                                    label=housing_type_labels[i]) 
                             for i in range(5)]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8, 
                     framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(path_output_plots + f'maps/{scenario_name}/map_spatial_pop_housing_type_distrib.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return

## ALSO FOR AGGREGATE POPULATION

def plot_one_income_group_3d_housing_types(elev, gdf, population_by_housing_type,
                                           scenario_name,
                                           z_bounds, housing_type_labels=None,
                                           add_legend=True, add_basemap=True, basemap_alpha=0.3):

    plt.ioff()
    
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        print(f"Converting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    fig = plt.figure(figsize=(20, 16))

    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values 
    y_coords = centroids.y.values 

    lon_range = x_coords.max() - x_coords.min()
    lat_range = y_coords.max() - y_coords.min()
    if lon_range < 1:
        scale_factor = 0.02
    elif lon_range < 10:
        scale_factor = 0.01
    else:
        scale_factor = 0.005
    
    dx = lon_range * scale_factor
    dy = lat_range * scale_factor

    housing_type_colors = plt.cm.Set3(np.linspace(0, 1, 5))

    if housing_type_labels is None:
        housing_type_labels = [f'Type {i+1}' for i in range(5)]
        
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    if add_basemap:
        add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)

    population_data = population_by_housing_type

    total_population = population_data.sum(axis=0)

    non_zero_mask = total_population > 0
    
    x_coords_filtered = x_coords[non_zero_mask]
    y_coords_filtered = y_coords[non_zero_mask]
    population_data_filtered = population_data[:, non_zero_mask]

    if len(x_coords_filtered) > 0:
        z_bottom = np.zeros_like(x_coords_filtered)

        for housing_type_idx in range(5):
            heights = population_data_filtered[housing_type_idx, :]

            mask_with_pop = heights > 0
            if mask_with_pop.any():
                ax.bar3d(x_coords_filtered[mask_with_pop], 
                        y_coords_filtered[mask_with_pop], 
                        z_bottom[mask_with_pop], 
                        dx, dy, 
                        heights[mask_with_pop], 
                        color=housing_type_colors[housing_type_idx], 
                        alpha=0.8, 
                        edgecolor='none', 
                        linewidth=0, 
                        shade=True, 
                        zorder=10)

            z_bottom = z_bottom + heights

    z_min = z_bounds[0]
    z_max = z_bounds[1]
    ax.set_zlim(min(0, z_min), max(0, z_max))

    ax.set_xlabel('Longitude', labelpad=5)
    ax.set_ylabel('Latitude', labelpad=20)
    ax.set_zlabel('Nb of HHs', labelpad=20)

    ax.tick_params(axis='x', pad=5, labelsize=8)
    ax.tick_params(axis='y', pad=15, labelsize=8)
    ax.tick_params(axis='z', pad=15, labelsize=8)

    ax.view_init(elev=elev, azim=270)
    ax.grid(True, alpha=0.3)

    if add_legend:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=housing_type_colors[i], 
                                label=housing_type_labels[i]) 
                         for i in range(5)]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=16, 
                 framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(path_output_plots + f'maps/{scenario_name}/map_spatial_pop_housing_type_distrib_agg.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return

## WE LOOP OVER SCENARIOS (IN A SIMPLE FASHION)

for scenario_name in scenario_names_map:
    plot_four_income_groups_3d_housing_types(
        elev=70,
        gdf=gdf,
        population_by_housing_type=population_array[scenario_name],  # shape (5, 4, 24014)
        incgrp_names=['poor', 'midpoor', 'midrich', 'rich'],
        incgrp_labels=['Low inc.', 'Med-low inc.', 'Med-high inc.', 'High inc.'],
        scenario_name=scenario_name,
        z_bounds=z_bounds,
        housing_type_labels=['FP', 'IB (basic)', 'IB (redev.)', 'IS', 'FS']
    )
    
for scenario_name in scenario_names_map:
    plot_one_income_group_3d_housing_types(
        elev=70,
        gdf=gdf,
        population_by_housing_type=np.nansum(population_array[scenario_name], 1),  # shape (5, 4, 24014)
        scenario_name=scenario_name,
        z_bounds=z_bounds_agg,
        housing_type_labels=['FP', 'IB (basic)', 'IB (redev.)', 'IS', 'FS']
    )

