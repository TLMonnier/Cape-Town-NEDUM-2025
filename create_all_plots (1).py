# -*- coding: utf-8 -*-
"""
Comprehensive Plotting Script
Generates all figures from the revision analysis
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import contextily as ctx
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import outputs.export_outputs as outexp

###############################################################################
# SETUP AND DATA LOADING
###############################################################################

def setup_paths():
    """Setup all file paths."""
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
    
    return {
        'code': path_code,
        'folder': path_folder,
        'precalc_inp': path_precalc_inp,
        'data': path_data,
        'precalc_transp': path_precalc_transp,
        'scenarios': path_scenarios,
        'outputs': path_outputs,
        'simul': path_simul,
        'output_plots': path_output_plots,
        'output_tables': path_output_tables
    }


def load_all_data(paths):
    """Load all required data."""
    print("Loading data...")
    
    # Parameters and options
    options = inpprm.import_options()
    param = inpprm.import_param(paths['precalc_inp'], options)
    
    # Geographic data
    grid, center = inpdt.import_grid(paths['data'])
    amenities = inpdt.import_amenities(paths['precalc_inp'], options)
    geo_grid = gpd.read_file(paths['data'] + "grid_reference_500.shp")
    
    # Macro data
    (interest_rate, population, housing_type_data, total_RDP, backyard_data
     ) = inpdt.import_macro_data(param, paths['scenarios'], paths['folder'])
    
    # Income data
    income_class_by_housing_type = inpdt.import_hypothesis_housing_type()
    (mean_income, households_per_income_class, average_income, income_mult,
     income_baseline, households_per_income_and_housing
     ) = inpdt.import_income_classes_data(param, paths['data'])
    
    # Housing data
    (data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_baseline,
     grid_formal_density_HFA, threshold_income_distribution, income_distribution,
     cape_town_limits) = inpdt.import_households_data(paths['precalc_inp'])
    
    housing_types = pd.read_excel(paths['folder'] + 'housing_types_grid_sal.xlsx')
    housing_types[np.isnan(housing_types)] = 0
    
    # Land use projections
    (spline_RDP, spline_estimate_RDP, spline_land_RDP,
     spline_land_backyard, spline_land_informal, spline_land_constraints,
     number_properties_RDP) = (
         inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                               housing_type_data, paths['data'], paths['folder'])
         )
    
    coeff_land = inpdt.import_coeff_land(
        spline_land_constraints, spline_land_backyard, spline_land_informal,
        spline_land_RDP, param, options, 29)
    
    housing_limit = inpdt.import_housing_limit(grid, param)
    
    (param, minimum_housing_supply, agricultural_rent
     ) = inpprm.import_construction_parameters(
        param, grid, housing_types_sp, data_sp["dwelling_size"],
        mitchells_plain_grid_baseline, grid_formal_density_HFA, coeff_land,
        interest_rate, options
        )
    
    income_net_of_commuting_costs = np.load(
        paths['precalc_transp'] + 'GRID_incomeNetOfCommuting_0.npy')
    
    print("Data loaded successfully!")
    
    return {
        'param': param,
        'options': options,
        'grid': grid,
        'center': center,
        'amenities': amenities,
        'geo_grid': geo_grid,
        'interest_rate': interest_rate,
        'population': population,
        'housing_type_data': housing_type_data,
        'income_net_of_commuting_costs': income_net_of_commuting_costs,
        'mean_income': mean_income
    }


def load_simulation_results(paths, sim_configs):
    """Load simulation results for all configurations."""
    print("Loading simulation results...")
    
    results = {}
    for config in sim_configs:
        results[config] = {
            'utility': np.load(f"{paths['simul']}/initial_state_utility_{config}.npy"),
            'households': np.load(f"{paths['simul']}/initial_state_households_{config}.npy"),
            'dwelling_size': np.load(f"{paths['simul']}/initial_state_dwelling_size_{config}.npy"),
            'housing_supply': np.load(f"{paths['simul']}/initial_state_housing_supply_{config}.npy"),
            'rent': np.load(f"{paths['simul']}/initial_state_rent_{config}.npy")
        }
    
    print(f"Loaded {len(results)} simulation configurations")
    return results


###############################################################################
# PLOTTING FUNCTIONS
###############################################################################

def plot_housing_type_breakdown(scenario_data, paths, filename='htype_breakdown.png'):
    """
    Plot 1: Distribution of HHs across housing types by income group and scenario.
    Array order: [housing_type, income_group, location]
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['FP', 'IB (basic)', 'IB (redev.)', 'IS', 'FS']
    scenario_names = list(scenario_data.keys())
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    fig.suptitle('Distribution of HHs across housing types by income group and scenario', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for i, income_group in enumerate(income_groups):
        ax = axes[i]
        x = np.arange(len(scenario_names))
        width = 0.6
        bottom = np.zeros(len(scenario_names))
        
        for j, housing_type in enumerate(housing_types):
            # Extract data: sum over locations for each housing type and income group
            # Array order is [housing_type, income_group, location]
            values = [scenario_data[scenario][j, i, :].sum() for scenario in scenario_names]
            ax.bar(x, values, width, label=housing_type, bottom=bottom, color=colors[j])
            bottom += values
        
        ax.set_ylabel('Number of HHs', fontsize=11, fontweight='bold')
        ax.set_title(f'{income_group}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if i == 0:
            ax.legend(title='Housing types', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_rent_population_distribution(rents, population, geo_grid, paths, 
                                     filename='baseline_rent_pop_distrib.png'):
    """
    Plot 2: Rent and population distribution by income group.
    Array order: [housing_type, income_group, location]
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['Formal private', 'Backyard (basic)', 'Backyard (increm.)', 
                     'Informal settlement', 'Formal subsidized']
    n_housing_types = 5
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, income_group in enumerate(income_groups):
        ax = axes[idx]
        
        # Get data for this income group
        # Array order is [housing_type, income_group, location]
        pop_data = population[:, idx, :]  # [housing_types x locations]
        rent_data = rents[:, idx, :]  # [housing_types x locations]
        
        # Scatter plot
        colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
        
        for h_idx in range(n_housing_types):
            mask = pop_data[h_idx, :] > 0
            if mask.sum() > 0:
                ax.scatter(rent_data[h_idx, mask], pop_data[h_idx, mask],
                          alpha=0.5, s=30, c=[colors[h_idx]], 
                          label=housing_types[h_idx], edgecolors='none')
        
        ax.set_xlabel('Rent (ZAR/m²)', fontsize=10)
        ax.set_ylabel('Number of HHs', fontsize=10)
        ax.set_title(income_group, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_income_distribution(mean_income, population, geo_grid, paths,
                            filename='baseline_income_pop_distrib.png'):
    """
    Plot 3: Income distribution by income group.
    Array order: [housing_type, income_group, location]
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['Formal private', 'Backyard (basic)', 'Backyard (increm.)', 
                     'Informal settlement', 'Formal subsidized']
    n_housing_types = 5
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, income_group in enumerate(income_groups):
        ax = axes[idx]
        
        # Get population data: [housing_type, income_group, location]
        pop_data = population[:, idx, :]  # [housing_types x locations]
        income_value = mean_income[idx]
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
        
        for h_idx in range(n_housing_types):
            mask = pop_data[h_idx, :] > 0
            if mask.sum() > 0:
                # Create income array (constant for each income group)
                income_array = np.full(mask.sum(), income_value)
                ax.scatter(income_array, pop_data[h_idx, mask],
                          alpha=0.5, s=30, c=[colors[h_idx]], 
                          label=housing_types[h_idx], edgecolors='none')
        
        ax.set_xlabel('Income (ZAR/month)', fontsize=10)
        ax.set_ylabel('Number of HHs', fontsize=10)
        ax.set_title(income_group, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_dwelling_size_distribution(dwelling_sizes, population, geo_grid, paths,
                                   filename='baseline_dsize_pop_distrib.png'):
    """
    Plot 4: Dwelling size distribution by income group.
    Array order: [housing_type, income_group, location]
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['Formal private', 'Backyard (basic)', 'Backyard (increm.)', 
                     'Informal settlement', 'Formal subsidized']
    n_housing_types = 5
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, income_group in enumerate(income_groups):
        ax = axes[idx]
        
        # Array order is [housing_type, income_group, location]
        pop_data = population[:, idx, :]  # [housing_types x locations]
        dsize_data = dwelling_sizes  # [housing_types x locations]
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
        
        for h_idx in range(n_housing_types):
            mask = pop_data[h_idx, :] > 0
            if mask.sum() > 0:
                ax.scatter(dsize_data[h_idx, mask], pop_data[h_idx, mask],
                          alpha=0.5, s=30, c=[colors[h_idx]], 
                          label=housing_types[h_idx], edgecolors='none')
        
        ax.set_xlabel('Dwelling size (m²)', fontsize=10)
        ax.set_ylabel('Number of HHs', fontsize=10)
        ax.set_title(income_group, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_amenity_distribution(amenities, population, geo_grid, paths,
                             filename='baseline_amenity_pop_distrib.png'):
    """
    Plot 5: Amenity distribution by income group.
    Array order: [housing_type, income_group, location]
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['Formal private', 'Backyard (basic)', 'Backyard (increm.)', 
                     'Informal settlement', 'Formal subsidized']
    n_housing_types = 5
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, income_group in enumerate(income_groups):
        ax = axes[idx]
        
        # Array order is [housing_type, income_group, location]
        pop_data = population[:, idx, :]  # [housing_types x locations]
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_housing_types))
        
        for h_idx in range(n_housing_types):
            mask = pop_data[h_idx, :] > 0
            if mask.sum() > 0:
                ax.scatter(amenities[mask], pop_data[h_idx, mask],
                          alpha=0.5, s=30, c=[colors[h_idx]], 
                          label=housing_types[h_idx], edgecolors='none')
        
        ax.set_xlabel('Amenity index', fontsize=10)
        ax.set_ylabel('Number of HHs', fontsize=10)
        ax.set_title(income_group, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def calculate_weighted_rent(rents, population, income_group):
    """
    Calculate weighted average rent for an income group.
    Array order: [housing_type, income_group, location]
    """
    # Get data for this income group: [housing_types x locations]
    pop = population[:, income_group, :]
    rent = rents[:, income_group, :]
    
    # Sum across housing types
    total_pop = pop.sum(axis=0)  # [locations]
    weighted_rent = (rent * pop).sum(axis=0)  # [locations]
    
    mask = total_pop > 0
    result = np.full(len(total_pop), np.nan)
    result[mask] = weighted_rent[mask] / total_pop[mask]
    
    return result


def plot_3d_rent_map_single_income(gdf, rents, population, income_idx, 
                                  income_name, paths, basemap_source=None):
    """
    Plot 6: 3D rent map for a single income group.
    Array order: [housing_type, income_group, location]
    """
    print(f"Creating 3D map for {income_name}")
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get centroids
    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values
    y_coords = centroids.y.values
    
    # Calculate data: [housing_type, income_group, location]
    # Sum across housing types for total population
    total_pop = population[:, income_idx, :].sum(axis=0)  # [locations]
    weighted_rents = calculate_weighted_rent(rents, population, income_idx)
    
    # Filter non-zero population
    non_zero_mask = total_pop > 0
    x_filtered = x_coords[non_zero_mask]
    y_filtered = y_coords[non_zero_mask]
    heights_filtered = total_pop[non_zero_mask]
    rents_filtered = weighted_rents[non_zero_mask]
    
    # Bar dimensions
    lon_range = x_coords.max() - x_coords.min()
    lat_range = y_coords.max() - y_coords.min()
    scale_factor = 0.02 if lon_range < 1 else 0.01
    
    dx = lon_range * scale_factor
    dy = lat_range * scale_factor
    
    # Colors
    norm = Normalize(vmin=np.nanmin(rents_filtered), vmax=np.nanmax(rents_filtered))
    cmap = plt.cm.YlOrRd
    colors = cmap(norm(rents_filtered))
    
    # Plot
    ax.bar3d(x_filtered, y_filtered, np.zeros_like(heights_filtered), 
            dx, dy, heights_filtered, color=colors, alpha=0.8, 
            edgecolor='none')
    
    # Labels
    ax.set_xlabel('Longitude', labelpad=10)
    ax.set_ylabel('Latitude', labelpad=10)
    ax.set_zlabel('Nb of HHs', labelpad=10)
    ax.set_title(f'{income_name} - Population Distribution by Rent', 
                fontsize=14, fontweight='bold')
    ax.view_init(elev=30, azim=270)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    cbar.set_label('Weighted Average Rent (ZAR/m²)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    filename = f'3d_rent_map_{income_name.lower().replace(" ", "_")}.png'
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_all_3d_rent_maps(gdf, rents, population, paths):
    """Create 3D rent maps for all income groups."""
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    
    for idx, name in enumerate(income_groups):
        plot_3d_rent_map_single_income(gdf, rents, population, idx, name, paths)


def plot_scenario_comparison_rent(all_scenario_data, paths, 
                                  filename='scenario_comparison_rent.png'):
    """
    Plot rent distributions across multiple scenarios.
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['Formal private', 'Backyard (basic)', 'Backyard (increm.)', 
                     'Informal settlement', 'Formal subsidized']
    scenario_names = list(all_scenario_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Rent Distribution Across Scenarios', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for income_idx, income_group in enumerate(income_groups):
        ax = axes[income_idx]
        
        for scenario_idx, scenario_name in enumerate(scenario_names):
            rent_data = all_scenario_data[scenario_name]['rent']
            pop_data = all_scenario_data[scenario_name]['households']
            
            # Get data for this income group: [housing_types x locations]
            rent_income = rent_data[:, income_idx, :]
            pop_income = pop_data[:, income_idx, :]
            
            for h_idx in range(5):
                mask = pop_income[h_idx, :] > 0
                if mask.sum() > 0:
                    ax.scatter(rent_income[h_idx, mask], pop_income[h_idx, mask],
                              alpha=0.3, s=20, c=[colors[h_idx]], 
                              label=f'{housing_types[h_idx]}' if scenario_idx == 0 else '',
                              edgecolors='none', marker='o' if scenario_idx == 0 else 'x')
        
        ax.set_xlabel('Rent (ZAR/m²)', fontsize=10)
        ax.set_ylabel('Number of HHs', fontsize=10)
        ax.set_title(income_group, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if income_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_scenario_comparison_dwelling_size(all_scenario_data, paths,
                                          filename='scenario_comparison_dsize.png'):
    """
    Plot dwelling size distributions across multiple scenarios.
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['Formal private', 'Backyard (basic)', 'Backyard (increm.)', 
                     'Informal settlement', 'Formal subsidized']
    scenario_names = list(all_scenario_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Dwelling Size Distribution Across Scenarios', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for income_idx, income_group in enumerate(income_groups):
        ax = axes[income_idx]
        
        for scenario_idx, scenario_name in enumerate(scenario_names):
            dsize_data = all_scenario_data[scenario_name]['dwelling_size']
            pop_data = all_scenario_data[scenario_name]['households']
            
            # Get data for this income group: [housing_types x locations]
            pop_income = pop_data[:, income_idx, :]
            
            for h_idx in range(5):
                mask = pop_income[h_idx, :] > 0
                if mask.sum() > 0:
                    ax.scatter(dsize_data[h_idx, mask], pop_income[h_idx, mask],
                              alpha=0.3, s=20, c=[colors[h_idx]], 
                              label=f'{housing_types[h_idx]}' if scenario_idx == 0 else '',
                              edgecolors='none', marker='o' if scenario_idx == 0 else 'x')
        
        ax.set_xlabel('Dwelling Size (m²)', fontsize=10)
        ax.set_ylabel('Number of HHs', fontsize=10)
        ax.set_title(income_group, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if income_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_scenario_comparison_total_households(all_scenario_data, paths,
                                             filename='scenario_comparison_total_hh.png'):
    """
    Plot total households by housing type across scenarios.
    """
    print(f"Creating plot: {filename}")
    
    income_groups = ['Poor', 'Midpoor', 'Midrich', 'Rich']
    housing_types = ['FP', 'IB (basic)', 'IB (redev.)', 'IS', 'FS']
    scenario_names = list(all_scenario_data.keys())
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Total Households by Housing Type Across Scenarios', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    x = np.arange(len(scenario_names))
    width = 0.15
    
    for h_idx, housing_type in enumerate(housing_types):
        totals = []
        for scenario_name in scenario_names:
            pop_data = all_scenario_data[scenario_name]['households']
            # Sum across all income groups and locations for this housing type
            total = pop_data[h_idx, :, :].sum()
            totals.append(total)
        
        ax.bar(x + h_idx * width, totals, width, label=housing_type, color=colors[h_idx])
    
    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Households', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend(title='Housing Types', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(paths['output_plots'] + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


###############################################################################
# MAIN EXECUTION
###############################################################################

def main():
    """Main execution function."""
    print("=" * 80)
    print("COMPREHENSIVE PLOTTING SCRIPT")
    print("=" * 80)
    
    # Setup
    paths = setup_paths()
    data = load_all_data(paths)
    
    # Define simulation configurations
    sim_configs = [
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
    
    # Load simulation results
    sim_results = load_simulation_results(paths, sim_configs)
    
    # Prepare scenario data for housing type breakdown
    print("\nPreparing scenario data...")
    scenario_data = {}
    for config in sim_configs[:4]:  # First 4 scenarios for initial comparison
        # households array is [housing_type, income_group, location]
        scenario_data[config] = sim_results[config]['households']
    
    # Create plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80 + "\n")
    
    # Plot 1: Housing type breakdown (first 4 scenarios)
    plot_housing_type_breakdown(scenario_data, paths, 'htype_breakdown_base.png')
    
    # Plot housing type breakdown for upgrading scenarios
    scenario_data_upgrading = {}
    upgrading_configs = [sim_configs[0], sim_configs[4], sim_configs[5], sim_configs[6]]
    for config in upgrading_configs:
        scenario_data_upgrading[config] = sim_results[config]['households']
    plot_housing_type_breakdown(scenario_data_upgrading, paths, 'htype_breakdown_upgrading.png')
    
    # Plot housing type breakdown for eviction scenarios
    scenario_data_eviction = {}
    eviction_configs = [sim_configs[0], sim_configs[8], sim_configs[9], sim_configs[10]]
    for config in eviction_configs:
        scenario_data_eviction[config] = sim_results[config]['households']
    plot_housing_type_breakdown(scenario_data_eviction, paths, 'htype_breakdown_eviction.png')
    
    # Get baseline data
    baseline_config = sim_configs[0]
    baseline_households = sim_results[baseline_config]['households']
    baseline_rent = sim_results[baseline_config]['rent']
    baseline_dwelling_size = sim_results[baseline_config]['dwelling_size']
    
    # Plot 2-5: Baseline distribution plots
    print("\nGenerating baseline distribution plots...")
    plot_rent_population_distribution(baseline_rent, baseline_households, 
                                     data['geo_grid'], paths)
    plot_income_distribution(data['mean_income'], baseline_households, 
                           data['geo_grid'], paths)
    plot_dwelling_size_distribution(baseline_dwelling_size, baseline_households, 
                                  data['geo_grid'], paths)
    plot_amenity_distribution(data['amenities'], baseline_households, 
                            data['geo_grid'], paths)
    
    # Plot 6-9: Scenario comparison plots
    print("\nGenerating scenario comparison plots...")
    
    # Compare first 4 scenarios (baseline variations)
    scenario_comparison_data = {config: sim_results[config] for config in sim_configs[:4]}
    plot_scenario_comparison_rent(scenario_comparison_data, paths, 
                                 'scenario_comparison_rent_base.png')
    plot_scenario_comparison_dwelling_size(scenario_comparison_data, paths,
                                         'scenario_comparison_dsize_base.png')
    plot_scenario_comparison_total_households(scenario_comparison_data, paths,
                                            'scenario_comparison_total_hh_base.png')
    
    # Compare upgrading scenarios
    upgrading_comparison = {config: sim_results[config] for config in upgrading_configs}
    plot_scenario_comparison_rent(upgrading_comparison, paths,
                                 'scenario_comparison_rent_upgrading.png')
    plot_scenario_comparison_dwelling_size(upgrading_comparison, paths,
                                         'scenario_comparison_dsize_upgrading.png')
    plot_scenario_comparison_total_households(upgrading_comparison, paths,
                                            'scenario_comparison_total_hh_upgrading.png')
    
    # Compare eviction scenarios
    eviction_comparison = {config: sim_results[config] for config in eviction_configs}
    plot_scenario_comparison_rent(eviction_comparison, paths,
                                 'scenario_comparison_rent_eviction.png')
    plot_scenario_comparison_dwelling_size(eviction_comparison, paths,
                                         'scenario_comparison_dsize_eviction.png')
    plot_scenario_comparison_total_households(eviction_comparison, paths,
                                            'scenario_comparison_total_hh_eviction.png')
    
    # Plot 10-13: 3D rent maps
    print("\nGenerating 3D rent maps...")
    plot_all_3d_rent_maps(data['geo_grid'], baseline_rent, baseline_households, paths)
    
    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Total plots created: ~22")
    print(f"Plots saved to: {paths['output_plots']}")
    print("=" * 80)
    
    # Summary of generated plots
    print("\nGenerated plots:")
    print("  1. Housing type breakdowns (3 files)")
    print("  2. Baseline distributions (4 files)")
    print("  3. Scenario comparisons - base (3 files)")
    print("  4. Scenario comparisons - upgrading (3 files)")
    print("  5. Scenario comparisons - eviction (3 files)")
    print("  6. 3D rent maps (4 files)")
    print("  Total: 22 plot files")


if __name__ == "__main__":
    main()
