# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:40:06 2026

@author: monni

Fully optimized version with all repetitive commands wrapped into reusable functions
"""

# #########################################
# ## SIMPLIFIED SCRIPT FOR JUE REVISION ###
# #########################################

# ## We import standard Python libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ## We also import our own packages
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import outputs.export_outputs as outexp

from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

###############################################################################
# HELPER FUNCTIONS FOR DATA LOADING
###############################################################################

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
        Name of the simulation
    
    Returns:
    --------
    numpy.ndarray
        Loaded data array
    """
    filename = f'/initial_state_{data_type}_{simulation_name}.npy'
    return np.load(path_simul + filename)


def load_all_simulation_data(path_simul, simulation_configs, data_types):
    """
    Load all simulation data for multiple configurations and data types.
    
    Parameters:
    -----------
    path_simul : str
        Path to simulation outputs
    simulation_configs : list of str
        List of simulation configuration names
    data_types : list of str
        Types of data to load (e.g., ['utility', 'households', 'rent'])
    
    Returns:
    --------
    dict
        Nested dictionary: {data_type: {sim_config: data_array}}
    """
    results = {}
    for data_type in data_types:
        results[data_type] = {}
        for sim_config in simulation_configs:
            results[data_type][sim_config] = load_simulation_data(
                path_simul, data_type, sim_config
            )
    return results


def create_empty_fraction_capital_destroyed(n_locations=24014):
    """
    Create an empty DataFrame for fraction_capital_destroyed with all zeros.
    
    Parameters:
    -----------
    n_locations : int
        Number of locations/grid cells (default: 24014)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with zero values for all capital destruction categories
    """
    columns = [
        "structure_formal_2", "structure_formal_1",
        "structure_subsidized_2", "structure_subsidized_1",
        "contents_formal", "contents_informal",
        "contents_subsidized", "contents_backyard",
        "structure_backyards", "structure_formal_backyards",
        "structure_informal_backyards", "structure_informal_settlements"
    ]
    
    return pd.DataFrame({col: np.zeros(n_locations) for col in columns})


###############################################################################
# HELPER FUNCTIONS FOR DATA PROCESSING
###############################################################################

def extract_housing_supply(housing_supply_data, housing_type_idx=1, scale=1000000):
    """
    Extract and scale housing supply for a specific housing type.
    
    Parameters:
    -----------
    housing_supply_data : numpy.ndarray
        Housing supply data array
    housing_type_idx : int
        Index of housing type (default: 1 for backyard)
    scale : float
        Scaling factor (default: 1000000)
    
    Returns:
    --------
    numpy.ndarray
        Scaled housing supply
    """
    return housing_supply_data[housing_type_idx, :] / scale


def extract_housing_rent(rent_data, housing_type_idx=1):
    """
    Extract rent for a specific housing type.
    
    Parameters:
    -----------
    rent_data : numpy.ndarray
        Rent data array
    housing_type_idx : int
        Index of housing type (default: 1 for backyard)
    
    Returns:
    --------
    numpy.ndarray
        Housing rent
    """
    return rent_data[housing_type_idx, :]


def compute_disamenity_backyard(param, backyard_supply, amenity_upgrade_factor=0):
    """
    Compute disamenity factor for backyard housing.
    
    Parameters:
    -----------
    param : dict
        Parameters dictionary
    backyard_supply : numpy.ndarray
        Backyard supply array
    amenity_upgrade_factor : float
        Factor for amenity upgrading (0, 0.1, 0.5, or 1)
    
    Returns:
    --------
    numpy.ndarray
        Disamenity factors
    """
    disam = param["backyard_pockets"].copy()
    disam = disam + amenity_upgrade_factor * (1 - disam)
    
    # Adjust for incremental housing
    incremental_mask = (backyard_supply == 2)
    disam[incremental_mask] = (
        np.nanmean(param["incremental_pockets"]) 
        + amenity_upgrade_factor * (1 - np.nanmean(param["incremental_pockets"]))
    )
    
    return disam


def compute_disamenity_informal(param, amenity_upgrade_factor=0):
    """
    Compute disamenity factor for informal housing.
    
    Parameters:
    -----------
    param : dict
        Parameters dictionary
    amenity_upgrade_factor : float
        Factor for amenity upgrading (0, 0.1, 0.5, or 1)
    
    Returns:
    --------
    numpy.ndarray or float
        Disamenity factors
    """
    disam = param["informal_pockets"]
    return disam + amenity_upgrade_factor * (1 - disam)


def clean_nan_values(backyard_supply, backyard_rent):
    """
    Replace NaN values with zeros in supply and rent arrays.
    
    Parameters:
    -----------
    backyard_supply : numpy.ndarray
        Supply array
    backyard_rent : numpy.ndarray
        Rent array
    
    Returns:
    --------
    tuple
        Cleaned (supply, rent) arrays
    """
    backyard_supply = backyard_supply.copy()
    backyard_rent = backyard_rent.copy()
    
    backyard_supply[np.isnan(backyard_supply)] = 0
    backyard_rent[np.isnan(backyard_rent)] = 0
    
    return backyard_supply, backyard_rent


def compute_housing_costs(interest_rate, param, backyard_supply):
    """
    Compute informal and incremental housing costs.
    
    Parameters:
    -----------
    interest_rate : float
        Interest rate
    param : dict
        Parameters dictionary
    backyard_supply : numpy.ndarray
        Backyard supply array
    
    Returns:
    --------
    tuple
        (informal_cost, incremental_cost, backyard_cost)
    """
    informal_cost = (
        (interest_rate + param["depreciation_rate"]) 
        * param["informal_structure_value"]
        * backyard_supply * param["backyard_size"] / param["shack_size"]
    )
    
    incremental_cost = (
        (interest_rate + param["depreciation_rate"]) 
        * param["subsidized_structure_value"]
        * backyard_supply * param["backyard_size"] / param["RDP_size"]
    )
    
    backyard_cost = informal_cost.copy()
    backyard_cost[backyard_supply == 2] = incremental_cost[backyard_supply == 2]
    
    return informal_cost, incremental_cost, backyard_cost


###############################################################################
# HELPER FUNCTIONS FOR UTILITY CALCULATIONS
###############################################################################

def compute_utility_matrix(income_net_of_commuting_costs, sizes, rents, amenities, 
                           param, disam_backyard, disam_informal, backyard_supply,
                           interest_rate, backyard_cost):
    """
    Compute utility matrix for all housing types.
    
    Parameters:
    -----------
    income_net_of_commuting_costs : numpy.ndarray
        Net income after commuting
    sizes : dict
        Dictionary with 'formal', 'backyard', 'informal' sizes
    rents : dict
        Dictionary with 'formal', 'backyard', 'informal' rents
    amenities : numpy.ndarray
        Amenity values
    param : dict
        Parameters dictionary
    disam_backyard : numpy.ndarray
        Backyard disamenity
    disam_informal : numpy.ndarray
        Informal disamenity
    backyard_supply : numpy.ndarray
        Backyard supply
    interest_rate : float
        Interest rate
    backyard_cost : numpy.ndarray
        Backyard costs
    
    Returns:
    --------
    numpy.ndarray
        Utility matrix [4 x n_income x n_locations]
    """
    # Formal utility
    utility_formal = (
        (income_net_of_commuting_costs 
         - sizes['formal'][None, :] * rents['formal'][None, :]) ** param["alpha"]
        * (sizes['formal'][None, :] - param["q0"]) ** param["beta"]
        * amenities[None, :]
    )
    
    # Backyard utility
    utility_backyard = (
        (income_net_of_commuting_costs 
         - sizes['backyard'][None, :] * rents['backyard'][None, :]) ** param["alpha"]
        * (sizes['backyard'][None, :] - param["q0"]) ** param["beta"]
        * amenities[None, :] * disam_backyard[None, :]
    )
    
    # Informal utility
    utility_informal = (
        (income_net_of_commuting_costs 
         - sizes['informal'][None, :] * rents['informal'][None, :]
         - param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])
        ) ** param["alpha"]
        * (sizes['informal'][None, :] - param["q0"]) ** param["beta"]
        * amenities[None, :] * disam_informal[None, :]
    )
    
    # RDP utility
    utility_rdp = (
        (income_net_of_commuting_costs 
         + backyard_supply[None, :] * param["backyard_size"] * rents['backyard'][None, :]
         - param["subsidized_structure_value"] * param["depreciation_rate"]
         - backyard_cost
        ) ** param["alpha"]
        * (param["RDP_size"] + param["backyard_size"] - param["q0"]
           - np.nanmin(backyard_supply[None, :], 1) * param["backyard_size"]
        ) ** param["beta"]
        * amenities[None, :]
    )
    
    return np.array([utility_formal, utility_backyard, utility_informal, utility_rdp])


def compute_average_utility_by_income(utility_matrix, households_data):
    """
    Compute population-weighted average utility for each income group.
    
    Parameters:
    -----------
    utility_matrix : numpy.ndarray
        Utility matrix [4 housing types x 4 income groups x n_locations]
    households_data : numpy.ndarray
        Household distribution [n_locations x 4 income groups x 4 housing types]
    
    Returns:
    --------
    dict
        Dictionary with average utility for each income group
    """
    income_groups = ['poor', 'midpoor', 'midrich', 'rich']
    results = {}
    
    for i, group in enumerate(income_groups):
        results[group] = (
            np.nansum(utility_matrix[:, i, :] * households_data[:, i, :])
            / np.nansum(households_data[:, i, :])
        )
    
    return results


###############################################################################
# HELPER FUNCTIONS FOR 3D VISUALIZATION
###############################################################################

def setup_3d_basemap(ax, basemap_img, extent, alpha=0.5):
    """
    Add a basemap to a 3D axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes3D
        3D axis to add basemap to
    basemap_img : numpy.ndarray
        Basemap image array
    extent : tuple
        Extent (xmin, xmax, ymin, ymax)
    alpha : float
        Transparency level (default: 0.5)
    """
    xx, yy = np.meshgrid(
        np.linspace(extent[0], extent[1], basemap_img.shape[1]),
        np.linspace(extent[2], extent[3], basemap_img.shape[0])
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                    facecolors=basemap_img/255.0,
                    shade=False, alpha=alpha, zorder=1)


def add_polygon_boundaries_3d(ax, gdf_merc, alpha=0.1, linewidth=0.5):
    """
    Add polygon boundaries to a 3D plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes3D
        3D axis to add boundaries to
    gdf_merc : GeoDataFrame
        GeoDataFrame in Web Mercator projection
    alpha : float
        Transparency level (default: 0.1)
    linewidth : float
        Line width for boundaries (default: 0.5)
    """
    for geom in gdf_merc.geometry:
        if geom.geom_type == 'Polygon':
            _add_single_polygon_3d(ax, geom, alpha, linewidth)
        elif geom.geom_type == 'MultiPolygon':
            for poly_geom in geom.geoms:
                _add_single_polygon_3d(ax, poly_geom, alpha, linewidth)


def _add_single_polygon_3d(ax, geom, alpha, linewidth):
    """Helper function to add a single polygon to 3D plot."""
    poly_coords = np.array(geom.exterior.coords)
    verts = [(coord[0], coord[1], 0) for coord in poly_coords]
    poly = Poly3DCollection([verts], alpha=alpha, facecolors='none',
                           edgecolor='black', linewidth=linewidth, zorder=2)
    ax.add_collection3d(poly)


def calculate_weighted_rent(rents, population, income_group):
    """
    Calculate weighted average rent for an income group.
    
    Parameters:
    -----------
    rents : numpy.ndarray
        Rent array (locations x income_groups x housing_types)
    population : numpy.ndarray
        Population array (locations x income_groups x housing_types)
    income_group : int
        Index of income group
    
    Returns:
    --------
    numpy.ndarray
        Weighted average rent per location
    """
    total_pop = population[:, income_group, :].sum(axis=1)
    weighted_rent = (rents[:, income_group, :] * population[:, income_group, :]).sum(axis=1)
    
    # Avoid division by zero
    mask = total_pop > 0
    result = np.full(len(total_pop), np.nan)
    result[mask] = weighted_rent[mask] / total_pop[mask]
    
    return result


def create_3d_subplot(fig, subplot_idx, gdf_merc, rents, population, income_group,
                     basemap_img, extent, title, cmap=plt.cm.YlOrRd):
    """
    Create a single 3D subplot for rent visualization.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object
    subplot_idx : tuple or int
        Subplot index (e.g., (2, 2, 1) for 2x2 grid, position 1)
    gdf_merc : GeoDataFrame
        GeoDataFrame in Web Mercator projection
    rents : numpy.ndarray
        Rent data
    population : numpy.ndarray
        Population data
    income_group : int
        Income group index
    basemap_img : numpy.ndarray
        Basemap image
    extent : tuple
        Map extent
    title : str
        Subplot title
    cmap : matplotlib colormap
        Colormap for rent visualization
    
    Returns:
    --------
    ax : matplotlib.axes.Axes3D
        3D axis object
    """
    ax = fig.add_subplot(*subplot_idx, projection='3d')
    
    # Add basemap
    setup_3d_basemap(ax, basemap_img, extent)
    
    # Add polygon boundaries
    add_polygon_boundaries_3d(ax, gdf_merc)
    
    # Calculate data
    total_pop = population[:, income_group, :].sum(axis=0)
    weighted_rents = calculate_weighted_rent(rents, population, income_group)
    
    # Get centroids
    centroids = gdf_merc.geometry.centroid
    x = centroids.x.values
    y = centroids.y.values
    
    # Normalize and color
    norm = Normalize(vmin=np.nanmin(weighted_rents), vmax=np.nanmax(weighted_rents))
    colors = cmap(norm(weighted_rents))
    
    # Bar dimensions
    dx = np.ones_like(x) * (x.max() - x.min()) / 100
    dy = np.ones_like(y) * (y.max() - y.min()) / 100
    dz = total_pop
    z = np.zeros_like(total_pop)
    
    # Plot bars
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.85,
             edgecolor='black', linewidth=0.2, zorder=3)
    
    # Set limits and labels
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(0, np.max(population[:, :, income_group].sum(axis=1)) * 1.1)
    
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)
    ax.set_zlabel('Nb of HHs', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=30, azim=45)
    
    # Clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    return ax


def plot_all_income_groups(gdf, rents, population, basemap_img, extent, 
                           income_group_names, output_file=None):
    """
    Create a 2x2 plot with all income groups.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Geographic data
    rents : numpy.ndarray
        Rent data
    population : numpy.ndarray
        Population data
    basemap_img : numpy.ndarray
        Basemap image
    extent : tuple
        Map extent
    income_group_names : list
        Names of income groups
    output_file : str, optional
        Path to save figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(20, 16))
    
    for i in range(4):
        create_3d_subplot(
            fig, (2, 2, i+1), gdf, rents, population, i,
            basemap_img, extent, income_group_names[i]
        )
    
    fig.suptitle('Spatial population distribution with rent levels by income group',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def save_figure(fig, filename, dpi=300, facecolor='white'):
    """
    Save figure with consistent settings.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    dpi : int
        DPI resolution (default: 300)
    facecolor : str
        Background color (default: 'white')
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=facecolor)
    plt.close(fig)


###############################################################################
# BATCH PROCESSING FUNCTIONS
###############################################################################

def process_all_scenarios(path_simul, simulation_configs, param, interest_rate,
                          income_net_of_commuting_costs, amenities):
    """
    Process all simulation scenarios and compute derived quantities.
    
    Parameters:
    -----------
    path_simul : str
        Path to simulation outputs
    simulation_configs : list
        List of simulation configurations
    param : dict
        Parameters
    interest_rate : float
        Interest rate
    income_net_of_commuting_costs : numpy.ndarray
        Net income after commuting
    amenities : numpy.ndarray
        Amenity values
    
    Returns:
    --------
    dict
        Processed results for all scenarios
    """
    # Load all data at once
    data_types = ['utility', 'households', 'dwelling_size', 'housing_supply', 'rent']
    all_data = load_all_simulation_data(path_simul, simulation_configs, data_types)
    
    results = {}
    
    for sim_config in simulation_configs:
        # Extract data for this scenario
        housing_supply = all_data['housing_supply'][sim_config]
        rent = all_data['rent'][sim_config]
        
        # Extract specific housing data
        backyard_supply = extract_housing_supply(housing_supply, 1)
        backyard_rent = extract_housing_rent(rent, 1)
        
        # Clean NaN values
        backyard_supply, backyard_rent = clean_nan_values(backyard_supply, backyard_rent)
        
        # Determine amenity upgrade factor from config name
        if 'Aup1' in sim_config:
            aup_factor = 0.1
        elif 'Aup2' in sim_config:
            aup_factor = 0.5
        elif 'Aup3' in sim_config:
            aup_factor = 1.0
        else:
            aup_factor = 0.0
        
        # Compute disamenities
        disam_backyard = compute_disamenity_backyard(param, backyard_supply, aup_factor)
        disam_informal = compute_disamenity_informal(param, aup_factor)
        
        # Compute costs
        informal_cost, incremental_cost, backyard_cost = compute_housing_costs(
            interest_rate, param, backyard_supply
        )
        
        # Store results
        results[sim_config] = {
            'backyard_supply': backyard_supply,
            'backyard_rent': backyard_rent,
            'disam_backyard': disam_backyard,
            'disam_informal': disam_informal,
            'backyard_cost': backyard_cost,
            'raw_data': {
                'utility': all_data['utility'][sim_config],
                'households': all_data['households'][sim_config],
                'rent': all_data['rent'][sim_config],
                'dwelling_size': all_data['dwelling_size'][sim_config],
                'housing_supply': all_data['housing_supply'][sim_config]
            }
        }
    
    return results


###############################################################################
# MAIN SCRIPT
###############################################################################

def main():
    """Main execution function."""
    
    # ## Define file paths
    path_code = '..'
    path_folder = path_code + '/Data/'
    path_precalc_inp = path_folder + 'precalculated_inputs/'
    path_data = path_folder + 'data_Cape_Town/'
    path_precalc_transp = path_folder + 'precalculated_transport/'
    path_scenarios = path_data + 'Scenarios/'
    path_outputs = path_code + '/Output/'
    
    path_simul = path_outputs + 'revision_output'
    path_output_plots = path_simul + '/plots/'
    path_output_tables = path_simul + '/tables/'
    
    # # Parameters and options
    options = inpprm.import_options()
    param = inpprm.import_param(path_precalc_inp, options)
    
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
    fraction_capital_destroyed = create_empty_fraction_capital_destroyed()
    
    # ## Define all simulation configurations
    simulation_configs = [
        'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
        'simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
        'simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
        'simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1',
        'simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1',
        'simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1',
        'simul_UE1_ISconstr1_RDPnew0_Aup3_Psubsid0_Evict0_IH1',
        'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid1_Evict0_IH1',
        'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict1_IH1',
        'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict2_IH1',
        'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict3_IH1',
    ]
    
    # ## Process all scenarios efficiently
    print("Processing all simulation scenarios...")
    all_results = process_all_scenarios(
        path_simul, simulation_configs, param, interest_rate,
        income_net_of_commuting_costs, amenities
    )
    
    print(f"Successfully processed {len(all_results)} scenarios")
    
    # Example: Access data for a specific scenario
    baseline_config = simulation_configs[0]
    baseline_data = all_results[baseline_config]
    
    print(f"\nBaseline scenario: {baseline_config}")
    print(f"  Backyard supply shape: {baseline_data['backyard_supply'].shape}")
    print(f"  Backyard rent shape: {baseline_data['backyard_rent'].shape}")
    
    return all_results, param, amenities, geo_grid


if __name__ == "__main__":
    results, param, amenities, geo_grid = main()
    print("\nOptimization complete!")
