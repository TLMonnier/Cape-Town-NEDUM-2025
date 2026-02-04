# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:05:33 2026

@author: monni
"""

# %%

# TODO: Also housing types?

# FIGURES 10-11: 3D MAPS IN ABSOLUTE/RELATIVE VALUES (choose variable and scenario)

## FOOTNOTE: The plot shows the spatial distribution of the number of households per grid cell
## with corresponding variable levels for each income group, at baseline and for given scenario.
## [Choose variable and scenario for full description]
## (The aggregate distribution across income groups is also shown for reference.)
## The z axes and color scales are comparable across scenarios.

## FOOTNOTE bis: The plot shows the spatial distribution of the number of households per grid cell
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
                                add_colorbar=True, add_basemap=True, basemap_alpha=0.3,
                                negval=False, bounds=True):

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
        all_rents.extend(rent_data[rent_data != 0])

    for idx, (incgrp_name, incgrp_label) in enumerate(zip(incgrp_names, incgrp_labels)):
        
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d', computed_zorder=False)
        
        population_data = np.array(population_data_dict[incgrp_name])
        rent_data = np.array(rent_data_dict[incgrp_name])
        
        heights = population_data
        rents = rent_data
        
        z_min = z_bounds[idx, 0]
        z_max = z_bounds[idx, 1]
        ax.set_zlim(min(0, z_min), max(0, z_max))
        
        vmin = v_bounds[idx, 0]
        vmax = v_bounds[idx, 1]
        norm = Normalize(vmin=vmin, vmax=vmax)
     
        non_zero_mask = heights != 0
        x_coords_filtered = x_coords[non_zero_mask]
        y_coords_filtered = y_coords[non_zero_mask]
        heights_filtered = heights[non_zero_mask]
        rents_filtered = rents[non_zero_mask]
        
        if bounds==False:
            z_min = np.nanmin(heights_filtered)
            z_max = np.nanmax(heights_filtered)
            ax.set_zlim(min(0, z_min), max(0, z_max))
            
            # vmin = - max(abs(np.nanmin(rents_filtered)), abs(np.nanmax(rents_filtered)))
            # vmax = max(abs(np.nanmin(rents_filtered)), abs(np.nanmax(rents_filtered)))
            # vmin = np.nanmin(rents_filtered)
            # vmax = np.nanmax(rents_filtered)
            
            # norm = Normalize(vmin=vmin, vmax=vmax)
     
        # We do not deal with rent changes
        cmap = plt.cm.YlOrRd   
     
        if negval==False:
                        
            # cmap = plt.cm.YlOrRd 
            
            colors = cmap(norm(rents_filtered))
            colors[rents_filtered == 0] = [0.7, 0.7, 0.7, 0.8]
            
            if add_basemap:
                add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)
                
            ax.bar3d(x_coords_filtered, y_coords_filtered, np.zeros_like(heights_filtered), 
                    dx, dy, heights_filtered, color=colors, alpha=1, 
                    edgecolor='none', linewidth=0, shade=False, zorder=10)
            
        elif negval==True:
            positive_mask = heights > 0
            x_coords_pos = x_coords[positive_mask]
            y_coords_pos = y_coords[positive_mask]
            heights_pos = heights[positive_mask]
            rents_pos = rents[positive_mask]
            
            negative_mask = heights < 0
            x_coords_neg = x_coords[negative_mask]
            y_coords_neg = y_coords[negative_mask]
            heights_neg = heights[negative_mask]
            rents_neg = rents[negative_mask]
            
            # cmap = plt.cm.coolwarm
            
            colors_pos = cmap(norm(rents_pos))
            colors_pos[rents_pos == 0] = [0.7, 0.7, 0.7, 0.8]
            
            colors_neg = cmap(norm(rents_neg))
            colors_neg[rents_neg == 0] = [0.7, 0.7, 0.7, 0.8]

            ax.bar3d(x_coords_neg, y_coords_neg, np.zeros_like(heights_neg), 
                    dx, dy, heights_neg, color=colors_neg, alpha=1, 
                    edgecolor='none', linewidth=0, shade=False, zorder=-10)
        
            if add_basemap:
                add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)
                
            ax.bar3d(x_coords_pos, y_coords_pos, np.zeros_like(heights_pos), 
                    dx, dy, heights_pos, color=colors_pos, alpha=1, 
                    edgecolor='none', linewidth=0, shade=False, zorder=10)
        
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
    if negval==False:
        plt.savefig(path_output_plots + f'maps/{scenario_name}/map_spatial_pop_{xvar_name}_distrib.png', 
                    dpi=300, bbox_inches='tight')
    if negval==True:
        plt.savefig(path_output_plots + f'maps/{scenario_name}/changes/map_spatial_pop_{xvar_name}_distrib.png', 
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return

## ALSO FOR AGGREGATE POPULATION

def plot_single_income_group_3d(elev, gdf, population_data, rent_data,
                                xvar_name, xvar_label, scenario_name,
                                z_bounds_agg, v_bounds_agg,
                                add_colorbar=True, add_basemap=True, basemap_alpha=0.3,
                                negval=False, bounds=True):

    plt.ioff()
    
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS. Assuming EPSG:4326")
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        print(f"Converting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values 
    y_coords = centroids.y.values 
    
    heights = np.array(population_data)
    rents = np.array(rent_data)
    
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
    
    z_min = z_bounds_agg[0]
    z_max = z_bounds_agg[1]   
    ax.set_zlim(min(0, z_min), max(0, z_max))
    
    vmin = v_bounds_agg[0]
    vmax = v_bounds_agg[1]
    norm = Normalize(vmin=vmin, vmax=vmax)
  
    non_zero_mask = heights != 0
    x_coords_filtered = x_coords[non_zero_mask]
    y_coords_filtered = y_coords[non_zero_mask]
    heights_filtered = heights[non_zero_mask]
    rents_filtered = rents[non_zero_mask]
    
    if bounds==False:
        z_min = np.nanmin(heights_filtered)
        z_max = np.nanmax(heights_filtered)
        ax.set_zlim(min(0, z_min), max(0, z_max))
        
        # vmin = - max(abs(np.nanmin(rents_filtered)), abs(np.nanmax(rents_filtered)))
        # vmax = max(abs(np.nanmin(rents_filtered)), abs(np.nanmax(rents_filtered)))
        # vmin = np.nanmin(rents_filtered)
        # vmax = np.nanmax(rents_filtered)
        
        # norm = Normalize(vmin=vmin, vmax=vmax)
  
    cmap = plt.cm.YlOrRd  
  
    if negval==False:

        # cmap = plt.cm.YlOrRd        

        colors = cmap(norm(rents_filtered))
        colors[rents_filtered == 0] = [0.7, 0.7, 0.7, 0.8]
        
        if add_basemap:
            add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)
    
        ax.bar3d(x_coords_filtered, y_coords_filtered, np.zeros_like(heights_filtered), 
                dx, dy, heights_filtered, color=colors, alpha=1, 
                edgecolor='none', linewidth=0, shade=False, zorder=10)
        
    elif negval==True:
        
        positive_mask = heights > 0
        x_coords_pos = x_coords[positive_mask]
        y_coords_pos = y_coords[positive_mask]
        heights_pos = heights[positive_mask]
        rents_pos = rents[positive_mask]
        
        negative_mask = heights < 0
        x_coords_neg = x_coords[negative_mask]
        y_coords_neg = y_coords[negative_mask]
        heights_neg = heights[negative_mask]
        rents_neg = rents[negative_mask]

        # cmap = plt.cm.coolwarm

        colors_pos = cmap(norm(rents_pos))
        colors_pos[rents_pos == 0] = [0.7, 0.7, 0.7, 0.8]
        colors_neg = cmap(norm(rents_neg))
        colors_neg[rents_neg == 0] = [0.7, 0.7, 0.7, 0.8]
        
        ax.bar3d(x_coords_neg, y_coords_neg, np.zeros_like(heights_neg), 
                dx, dy, heights_neg, color=colors_neg, alpha=1, 
                edgecolor='none', linewidth=0, shade=False, zorder=-10)
        
        if add_basemap:
            add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)
    
        ax.bar3d(x_coords_pos, y_coords_pos, np.zeros_like(heights_pos), 
                dx, dy, heights_pos, color=colors_pos, alpha=1, 
                edgecolor='none', linewidth=0, shade=False, zorder=10)
    
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
    
    if negval==False:
        plt.savefig(path_output_plots + f'maps/{scenario_name}/map_spatial_pop_all_{xvar_name}_distrib.png', dpi=300, bbox_inches='tight')
    elif negval==True:
        plt.savefig(path_output_plots + f'maps/{scenario_name}/changes/map_spatial_pop_all_{xvar_name}_distrib.png', dpi=300, bbox_inches='tight')

    plt.close(fig)

    return

## WE LOOP OVER THE SCENARIOS

def process_scenario_variable_maps(elev, scenario_name, array_varname, 
                                   population_array, rent_array,
                                   array_label_dict_map,
                                   negval, bounds):
    
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
        elev, gdf, population_data_dict, rent_data_dict,
        array_varname, array_label_dict_map[array_varname],
        incgrp_names, incgrp_labels, scenario_name,
        z_bounds, v_bounds_dict[array_varname],
        add_colorbar=True, add_basemap=True, basemap_alpha=0.1,
        negval=negval, bounds=bounds)
    
    population_data = population_array[scenario_name].sum(axis=(0,1))
    rent_data = calculate_weighted_average_rent_allinc(
        population_array[scenario_name], rent_array[scenario_name][array_varname])
    
    plot_single_income_group_3d(elev, gdf, population_data, rent_data,
                                array_varname, array_label_dict_map[array_varname],
                                scenario_name,
                                z_bounds_agg, v_bounds_agg_dict[array_varname],
                                add_colorbar=True, add_basemap=True, basemap_alpha=0.1,
                                negval=negval, bounds=bounds)
    
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
    v_bounds[0, :] = [min(rent_data_dict['poor'][rent_data_dict['poor']!=0]),
                      max(rent_data_dict['poor'][rent_data_dict['poor']!=0])]
    v_bounds[1, :] = [min(rent_data_dict['midpoor'][rent_data_dict['midpoor']!=0]),
                      max(rent_data_dict['midpoor'][rent_data_dict['midpoor']!=0])]
    v_bounds[2, :] = [min(rent_data_dict['midrich'][rent_data_dict['midrich']!=0]),
                      max(rent_data_dict['midrich'][rent_data_dict['midrich']!=0])]
    v_bounds[3, :] = [min(rent_data_dict['rich'][rent_data_dict['rich']!=0]),
                      max(rent_data_dict['rich'][rent_data_dict['rich']!=0])]

    population_data = population_array[scenario_name].sum(axis=(0,1))
    rent_data = calculate_weighted_average_rent_allinc(
        population_array[scenario_name], rent_array[scenario_name][array_varname])
    
    z_bounds_agg = np.zeros(2)
    z_bounds_agg[0] = min(population_data)
    z_bounds_agg[1] = max(population_data)
    
    v_bounds_agg = np.zeros(2)
    v_bounds_agg[0] = min(rent_data[rent_data!=0])
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

population_array_changes = copy.copy(scenario_hhs_dict)
for key in population_array_changes.keys():
    population_array_changes[key] = population_array_changes[key]-base_scenario_hhs_dict['baseline']
    
rent_array_changes = copy.copy(scenarios_array_dict)
for key in rent_array_changes.keys():
    for array in array_varnames:
        rent_array_changes[key][array] = rent_array_changes[key][array]-base_scenario_array_dict['baseline'][array]

### THEN WE IMPORT THE PLOT VALUE BOUNDS (irrelevant for plots in changes)

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

# %%

### THEN WE LAUNCH PARALLEL COMPUTING

#### FIRST FOR ABSOLUTE VALUES

n_jobs = min(multiprocessing.cpu_count() - 1, len(scenario_names))

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_scenario_variable_maps)(
        70, scenario_name, array_varname, 
        population_array, rent_array,
        array_label_dict_map,
        negval=False, bounds=True)
    for scenario_name in scenario_names_map
    for array_varname in array_varnames
)

# %%

#### THEN FOR CHANGES

n_jobs = min(multiprocessing.cpu_count() - 1, len(scenario_names))

results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_scenario_variable_maps)(
        70, scenario_name, array_varname, 
        population_array_changes, scenarios_array_dict,
        array_label_dict_map,
        negval=True, bounds=False)
    for scenario_name in scenario_names
    for array_varname in array_varnames
)


###############################################################################

# %%%

# FIGURES 10-11bis: 3D MAPS IN RELATIVZ VALUES (choose variable and scenario)

## FOOTNOTE: The plot shows the spatial change in the number of households per grid cell
## from baseline to the given scenario, with corresponding variable level changes for each income group.
## [Choose variable and scenario for full description]
## (The aggregate distribution across income groups is also shown for reference.)
## The z axes and color scales are comparable across scenarios.




def plot_single_income_group_3d_negval(
        elev, gdf, population_data, rent_data,
        xvar_name, incgrp, title='Population Map',
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

    centroids = gdf.geometry.centroid
    x_coords = centroids.x.values 
    y_coords = centroids.y.values 
    
    heights = np.array(population_data)
    rents = np.array(rent_data)
  
    # TEST 
    non_zero_mask = heights != 0
    # non_zero_mask = heights > 0
    
    positive_mask = heights > 0
    negative_mask = heights < 0
    
    # x_coords_filtered = x_coords[non_zero_mask]
    # y_coords_filtered = y_coords[non_zero_mask]
    heights_filtered = heights[non_zero_mask]
    rents_filtered = rents[non_zero_mask]
    
    x_coords_filtered_pos = x_coords[positive_mask]
    y_coords_filtered_pos = y_coords[positive_mask]
    heights_filtered_pos = heights[positive_mask]
    rents_filtered_pos = rents[positive_mask]
    
    x_coords_filtered_neg = x_coords[negative_mask]
    y_coords_filtered_neg = y_coords[negative_mask]
    heights_filtered_neg = heights[negative_mask]
    rents_filtered_neg = rents[negative_mask]
    
    # x_coords_filtered_pos = np.copy(x_coords)
    # y_coords_filtered_pos = np.copy(y_coords)
    # heights_filtered_pos = np.copy(heights)
    # x_coords_filtered_pos[x_coords_filtered_pos<0] = 0
    # y_coords_filtered_pos[y_coords_filtered_pos<0] = 0
    # heights_filtered_pos[heights_filtered_pos<0] = 0
    
    # x_coords_filtered_neg = np.copy(x_coords)
    # y_coords_filtered_neg = np.copy(y_coords)
    # heights_filtered_neg = np.copy(heights)
    # x_coords_filtered_neg[x_coords_filtered_neg>0] = 0
    # y_coords_filtered_neg[y_coords_filtered_neg>0] = 0
    # heights_filtered_neg[heights_filtered_neg>0] = 0

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

    vmin = rents_filtered.min()
    vmax = rents_filtered.max()
    
    vmin_pos = rents_filtered_pos.min()
    vmax_pos = rents_filtered_pos.max()
    vmin_neg = rents_filtered_neg.min()
    vmax_neg = rents_filtered_neg.max()
    
    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(rents_filtered))
    
    norm_pos = Normalize(vmin=vmin_pos, vmax=vmax_pos)
    norm_neg = Normalize(vmin=vmin_neg, vmax=vmax_neg)
    
    colors_pos = cmap(norm_pos(rents_filtered_pos))
    colors_neg = cmap(norm_neg(rents_filtered_neg))

    colors[rents_filtered == 0] = [0.7, 0.7, 0.7, 0.8]
    colors_pos[rents_filtered_pos == 0] = [0.7, 0.7, 0.7, 0.8]
    colors_neg[rents_filtered_neg == 0] = [0.7, 0.7, 0.7, 0.8]

    # ax.bar3d(x_coords_filtered, y_coords_filtered, np.zeros_like(heights_filtered), 
    #         dx, dy, heights_filtered, color=colors, alpha=0.8, 
    #         edgecolor='none', linewidth=0, shade=True, zorder=10)
    ax.bar3d(x_coords_filtered_neg, y_coords_filtered_neg, np.zeros_like(heights_filtered_neg), 
            dx, dy, heights_filtered_neg, color=colors_neg, alpha=1, 
            edgecolor='none', linewidth=0, shade=True, zorder=-10)
    if add_basemap:
        add_basemap_to_3d(ax, gdf, alpha=basemap_alpha)
    ax.bar3d(x_coords_filtered_pos, y_coords_filtered_pos, np.zeros_like(heights_filtered_pos), 
            dx, dy, heights_filtered_pos, color=colors_pos, alpha=1, 
            edgecolor='none', linewidth=0, shade=True, zorder=10)
    
    # TEST
    z_min = np.min(heights_filtered)
    z_max = np.max(heights_filtered)
    ax.set_zlim(min(0, z_min), max(0, z_max)) # Ensure 0 is included, and limits cover range
    
    ax.set_xlabel('Longitude', labelpad=5)
    ax.set_ylabel('Latitude', labelpad=30)
    ax.set_zlabel('Nb of HHs', labelpad=30)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.tick_params(axis='x', pad=5)
    ax.tick_params(axis='y', pad=20)
    ax.tick_params(axis='z', pad=20)
    
    ax.view_init(elev=elev, azim=270)
    ax.grid(True, alpha=0.3)

    if add_colorbar and len(rents_filtered) > 0:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5, pad=0.01)
        cbar.set_label('Weighted avg annual rent (ZAR/m²)', rotation=270, labelpad=20)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.tight_layout()
    plt.savefig(path_output_plots + f'/baseline/maps/map_baseline_spatial_pop_{incgrp}_{xvar_name}_distrib.png', dpi=300, bbox_inches='tight')

    plt.close(fig)

    return

population_array = new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households
rent_array = new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_rent

# for income_idx in range(0,4):
for income_idx in [3]:
    new_population_array = new_simul_UE0_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households-new_simul_UE1_ISconstr1_RDPnew0_Aup0_Psubsid0_Evict0_IH1_households
    population_data = new_population_array[:, income_idx, :].sum(axis=0)
    rent_data = calculate_weighted_average_rent(population_array, rent_array, income_idx)
    fig = plot_single_income_group_3d_negval(
        20, gdf, population_data, rent_data,
        'rent_change', incgrp_names[income_idx],
        title=incgrp_labels[income_idx],
        add_basemap=True, basemap_alpha=0.2)
    
