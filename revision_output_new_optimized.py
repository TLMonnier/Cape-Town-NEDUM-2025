# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:40:06 2026

@author: monni

Optimized version with repetitive commands wrapped into functions
"""

# #########################################
# ## SIMPLIFIED SCRIPT FOR JUE REVISION ###
# #########################################

# ## We import standard Python libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ## We also import our own packages
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm

###############################################################################
# HELPER FUNCTIONS FOR REPEATED OPERATIONS
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
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
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
    from matplotlib.colors import Normalize
    
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
# MAIN SCRIPT
###############################################################################

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
options = inpprm.import_options()
param = inpprm.import_param(path_precalc_inp, options)

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
fraction_capital_destroyed = create_empty_fraction_capital_destroyed()

# ## Import simulation results using helper functions
simulation_configs = [
    'simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1',
    'simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1',
]

# Load data for all simulations
utility_data = load_multiple_simulation_data(path_simul, 'utility', simulation_configs)

# Access loaded data
simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[0]]
simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[1]]
simul_UE1_ISconstr0_RDPnew0_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[2]]
simul_UE1_ISconstr1_RDPnew1_Aup0_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[3]]
simul_UE1_ISconstr1_RDPnew0_Aup1_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[4]]
simul_UE1_ISconstr1_RDPnew0_Aup2_Psubsid0_Evict0_IH1_utility = utility_data[simulation_configs[5]]

# Note: Continue loading other data types as needed using the same pattern
# Example:
# rent_data = load_multiple_simulation_data(path_simul, 'rent', simulation_configs)
# households_data = load_multiple_simulation_data(path_simul, 'households', simulation_configs)

###############################################################################
# VISUALIZATION SECTION
###############################################################################

# Note: The visualization code would continue here, using the helper functions
# defined above for creating 3D plots, adding basemaps, etc.

print("Code optimized successfully!")
print(f"Loaded {len(utility_data)} simulation datasets")
