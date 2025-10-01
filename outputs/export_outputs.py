# import scipy
import pandas as pd
import numpy as np
# import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import geopandas as gpd
import inspect
from scipy.interpolate import griddata


def retrieve_name(var, depth):
    """
    Return a string for the name of a variable.

    Parameters
    ----------
    var :
        Any variable
    depth : int
        Depth level of the function call (indicates how deep it should go to
        retrieve the original name of the variable): from 0 to 2

    Returns
    -------
    str
        Name of input variable

    """
    if depth == 0:
        (callers_local_vars
         ) = inspect.currentframe().f_back.f_back.f_locals.items()
    elif depth == 1:
        (callers_local_vars
         ) = inspect.currentframe().f_back.f_back.f_back.f_locals.items()
    else:
        (callers_local_vars
         ) = inspect.currentframe().f_back.f_locals.items()
    name = [var_name for var_name, var_val
            in callers_local_vars if var_val is var]
    return name[0]


def from_df_to_gdf(array, geo_grid):
    """
    Convert map array/series inputs into grid-level GeoDataFrames.

    Parameters
    ----------
    array : ndarray
        Any array with a grid geographic dimension
    geo_grid : GeoDataFrames
        Data frame with geometry for the analysis grid (24,014 points)

    Returns
    -------
    gdf : GeoDataFrame
        Data frame with geolocalized observations from input array

    """
    # Convert array or series to data frame
    df = pd.DataFrame(array)
    str_array = retrieve_name(array, depth=1)
    df = df.rename(columns={df.columns[0]: str_array})
    gdf = pd.merge(geo_grid, df, left_index=True, right_index=True)

    return gdf


def export_map(value, grid, geo_grid, path_plots, export_name, title,
               path_tables,
               ubnd, lbnd=0, cmap='Reds'):
    """
    Generate 2D heat maps of any spatial input.

    Parameters
    ----------
    value : ndarray
        Any one-dimensional array with values given at grid level
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    geo_grid : GeoDataFrames
        Data frame with geometry for the analysis grid (24,014 points)
    path_plots : str
        Path for saving output plots
    export_name : str
        Name given to saved output file
    title : str
        Title given tou output plot
    path_tables : str
        Path for saving output plots
    ubnd : float64
        Upper bound for plotted values
    lbnd : float64, optional
        Lower bound for plotted values. The default is 0.
    cmap : str, optional
        Type of choropleth map to be plotted (see pyplot options). The default
        is 'Reds'.

    Returns
    -------
    gdf : GeoDataFrame
        Data frame with geolocalized observations from input array

    """
    plt.figure(figsize=(10, 7))
    Map = plt.scatter(grid.x,
                      grid.y,
                      s=None,
                      c=value,
                      cmap=cmap,
                      marker='.')
    plt.colorbar(Map)
    plt.axis('off')
    plt.clim(lbnd, ubnd)
    plt.title(title)
    plt.savefig(path_plots + export_name)
    plt.close()

    if isinstance(value, np.ndarray):
        np.savetxt(path_tables + export_name + '.csv', value, delimiter=",")

    else:
        # value.to_csv(path_tables + str(value))
        value.to_csv(path_tables + export_name + '.csv')
        # gdf = from_df_to_gdf(value, geo_grid)
        # str_value = retrieve_name(value, depth=0)
        # gdf.to_file(path_tables + str_value + '.shp')
        # gdf.to_file(path_tables + export_name + '.shp')

    gdf = value
    print(export_name + ' done')

    return gdf


def discrete_map(value, grid, geo_grid, path_plots, export_name, title,
                 path_tables):
    """
    Generate 2D heat maps of any spatial input.

    Parameters
    ----------
    value : ndarray
        Any one-dimensional array with values given at grid level
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    geo_grid : GeoDataFrames
        Data frame with geometry for the analysis grid (24,014 points)
    path_plots : str
        Path for saving output plots
    export_name : str
        Name given to saved output file
    title : str
        Title given tou output plot
    path_tables : str
        Path for saving output plots
    ubnd : float64
        Upper bound for plotted values
    lbnd : float64, optional
        Lower bound for plotted values. The default is 0.
    cmap : str, optional
        Type of choropleth map to be plotted (see pyplot options). The default
        is 'Reds'.

    Returns
    -------
    gdf : GeoDataFrame
        Data frame with geolocalized observations from input array

    """
    colors = np.zeros(len(value))
    colors[(value > 0) & (value <= 10)] = 1
    colors[(value > 10) & (value <= 20)] = 2
    colors[(value > 20) & (value <= 50)] = 3
    colors[(value > 50) & (value <= 100)] = 4
    colors[(value > 100) & (value <= 200)] = 5
    colors[value > 200] = 6

    cmap = cm.get_cmap('viridis_r').copy()
    cmap.set_under('grey')

    plt.figure(figsize=(10, 7))
    Map = plt.scatter(grid.x,
                      grid.y,
                      s=None,
                      c=colors,
                      cmap=cmap,
                      vmin=1,
                      marker='.')
    #plt.colorbar(Map)
    plt.axis('off')
    #plt.clim(1,6)
    plt.title(title)
    plt.savefig(path_plots + export_name)
    plt.close()

    if isinstance(value, np.ndarray):
        np.savetxt(path_tables + export_name + '.csv', value, delimiter=",")

    else:
        # value.to_csv(path_tables + str(value))
        value.to_csv(path_tables + export_name + '.csv')
        # gdf = from_df_to_gdf(value, geo_grid)
        # str_value = retrieve_name(value, depth=0)
        # gdf.to_file(path_tables + str_value + '.shp')
        # gdf.to_file(path_tables + export_name + '.shp')

    gdf = value
    print(export_name + ' done')

    return gdf


def import_employment_geodata(households_per_income_class, param, path_data):
    """
    Import number of jobs per selected employment center.

    Parameters
    ----------
    households_per_income_class : ndarray(float64)
        Exogenous total number of households per income group (excluding people
        out of employment, for 4 groups)
    param : dict
        Dictionary of default parameters
    path_data : str
        Path towards data used in the model

    Returns
    -------
    jobsTable : DataFrame
        Number of jobs in each selected job center (185) per income group (4),
        with x and y coordinates
    selected_centers : ndarray(bool)
        Array of dummies for selecting employment centers above some number of
        jobs threshold (185), out of the total set of transport zones (1787)

    """
    # Number of jobs per Transport Zone (TZ)
    TAZ = pd.read_csv(path_data + 'TAZ_amp_2013_proj_centro2.csv')

    # Number of employees in each TZ for the 12 income classes
    # NB: we break income classes given by CoCT to stick better to equivalent
    # from census data, as we will reweight towards the end
    jobsCenters12Class = np.array(
        [np.zeros(len(TAZ.Ink1)), TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3,
         TAZ.Ink2/2, TAZ.Ink2/2, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3,
         TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3]
        )

    # We get geographic coordinates
    codeCentersInitial = TAZ.TZ2013
    xCoord = TAZ.X / 1000
    yCoord = TAZ.Y / 1000

    # We arbitrarily set the threshold at 2,500 jobs
    selectedCenters = (
        sum(jobsCenters12Class, 0) > param["job_center_threshold"])

    # Corrections where we don't have reliable transport data
    selectedCenters[xCoord > -10] = np.zeros(1, 'bool')
    selectedCenters[yCoord > -3719] = np.zeros(1, 'bool')
    selectedCenters[(xCoord > -20) & (yCoord > -3765)] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1010] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1012] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1394] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1499] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 4703] = np.zeros(1, 'bool')

    # Number of workers per group for the selected centers
    jobsCentersNgroup = np.zeros((len(xCoord), param["nb_of_income_classes"]))
    for j in range(0, param["nb_of_income_classes"]):
        jobsCentersNgroup[:, j] = np.sum(
            jobsCenters12Class[param["income_distribution"] == j + 1, :], 0)

    # jobsCentersNgroup = jobsCentersNgroup[selectedCenters, :]
    # Rescale (wrt census data) to keep the correct global income distribution
    # More specifically, this allows to go from individual to household level
    jobsCentersNGroupRescaled = (
        jobsCentersNgroup * households_per_income_class[None, :]
        / np.nansum(jobsCentersNgroup, 0)
        )

    jobsTable = pd.DataFrame()
    # jobsTable["code"] = codeCentersInitial
    jobsTable["x"] = xCoord
    jobsTable["y"] = yCoord
    jobsTable["poor_jobs"] = jobsCentersNGroupRescaled[:, 0]
    jobsTable["midpoor_jobs"] = jobsCentersNGroupRescaled[:, 1]
    jobsTable["midrich_jobs"] = jobsCentersNGroupRescaled[:, 2]
    jobsTable["rich_jobs"] = jobsCentersNGroupRescaled[:, 3]
    selected_centers = pd.DataFrame(selectedCenters)
    selected_centers = selected_centers.rename(
        columns={selected_centers.columns[0]: 'selected_centers'})

    return jobsTable, selected_centers


def valid_pop_housing_type(
        housing_type_1, housing_type_2,
        legend1, legend2, path_plots, path_tables):
    """
    Validation bar plot for number of simulated households per housing type.

    Parameters
    ----------
    housing_type_1 : ndarray(float64, ndim=2)
        Number of simulated households per grid cell in each housing type (4)
    housing_type_2 : ndarray(float64, ndim=2)
        Number of households from data per grid cell in each housing type (4)
    legend1 : str
        Legend for first array
    legend2 : str
        Legend for second array
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    data : DataFrame
        Validation table for number of simulated households per housing type

    """
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    # figure.tight_layout()
    data = pd.DataFrame(
        {legend1: np.nansum(housing_type_1, 1), legend2: housing_type_2},
        index=["Formal private", "Informal \n backyards",
               "Informal \n settlements", "Formal subsidized"])
    data.plot(kind="bar", ax=axis)
    axis.set_ylabel("Households", labelpad=15)
    axis.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    axis.tick_params(labelrotation=0)

    figure.savefig(path_plots + 'validation_pop_housing_type.png')
    plt.close(figure)

    data.to_csv(path_tables + 'validation_pop_housing_type.csv')
    print('validation_pop_housing_type done')

    return data


def valid_pop_htype_income(
        initial_state_households, households_per_income_and_housing,
        legend1, legend2, path_plots, path_tables):
    """
    Validation bar plot for nb of households across housing and income groups.

    Parameters
    ----------
    initial_state_households : ndarray(float64, ndim=3)
        Number of households per grid cell in each income group (4) and
        each housing type (4)
    households_per_income_and_housing : ndarray(float64, ndim=2)
        Exogenous number of households per income group (4, from poorest to
        richest) in each endogenous housing type (3: formal private,
    legend1 : str
        Legend for first array
    legend2 : str
        Legend for second array
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    data0 : DataFrame
        Validation table for simulated number of households across income
        groups in formal private housing
    data1 : DataFrame
        Validation table for simulated number of households across income
        groups in informal backyards
    data2 : DataFrame
        Validation table for simulated number of households across income
        groups in informal settlements

    """
    # We apply same reweighting as in equilibrium to match aggregates
    # from SAL data
    ratio = (np.nansum(initial_state_households)
             / np.nansum(households_per_income_and_housing))
    households_per_income_and_housing = (
        households_per_income_and_housing * ratio)

    # We take RDP out of validation data for formal private
    households_per_income_and_housing[0, 0] = np.max(
        households_per_income_and_housing[0, 0]
        - np.nansum(initial_state_households[3, :, :]),
        0)
    # NB: note that we do not plot RDP as we do not have its breakdown
    # across income groups (and we assumed only the poorest were eligible)

    data0 = pd.DataFrame(
        {legend1: np.nansum(initial_state_households[0, :, :], 1),
         legend2: households_per_income_and_housing[0, :]},
        index=["Poor", "Mid-poor", "Mid-rich", "Rich"])
    data1 = pd.DataFrame(
        {legend1: np.nansum(initial_state_households[1, :, :], 1),
         legend2: households_per_income_and_housing[1, :]},
        index=["Poor", "Mid-poor", "Mid-rich", "Rich"])
    data2 = pd.DataFrame(
        {legend1: np.nansum(initial_state_households[2, :, :], 1),
         legend2: households_per_income_and_housing[2, :]},
        index=["Poor", "Mid-poor", "Mid-rich", "Rich"])

    figure, axis = plt.subplots(3, 1, figsize=(10, 7))
    # figure.tight_layout()
    data0.plot(kind="bar", ax=axis[0])
    axis[0].set_title("Formal private")
    # axis[0].get_legend().remove()
    axis[0].set_xticks([])
    axis[0].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    data1.plot(kind="bar", ax=axis[1])
    axis[1].set_title("Informal backyards")
    # axis[1].get_legend().remove()
    axis[1].set_ylabel("Households", labelpad=15)
    axis[1].set_xticks([])
    axis[1].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    data2.plot(kind="bar", ax=axis[2])
    axis[2].set_title("Informal settlements")
    axis[2].tick_params(labelrotation=0)
    axis[2].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    figure.savefig(path_plots + 'validation_pop_housing_per_income.png')
    plt.close(figure)

    data0.to_csv(path_tables + 'validation_pop_formal_per_income.csv')
    data1.to_csv(path_tables + 'validation_pop_backyard_per_income.csv')
    data2.to_csv(path_tables + 'validation_pop_informal_per_income.csv')
    print('validation_pop_housing_per_income done')

    return data0, data1, data2


def validation_density(
        grid, initial_state_households_housing_types, housing_types,
        path_plots, path_tables):
    """
    Validation line plot for household density across space in 1D.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_households_housing_types : ndarray(float64, ndim=2)
        Number of households per grid cell in each housing type (4)
    housing_types : DataFrame
        Table yielding, for 4 different housing types (informal settlements,
        formal backyards, informal backyards, and formal private housing),
        the number of households in each grid cell (24,014), from SAL data.
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Validation table for household density across space in 1D.

    """
    # Note that formal data here includes RDP
    sum_housing_types = (housing_types.informal_grid
                         + housing_types.formal_grid
                         + housing_types.backyard_informal_grid)

    # Population density (per km²)
    xData = grid.dist
    yData = sum_housing_types / 0.25
    ySimul = np.nansum(
        initial_state_households_housing_types, 0) / 0.25

    df = pd.DataFrame(data=np.transpose(
        np.array([xData, yData, ySimul])), columns=["x", "yData", "ySimul"])
    df["round"] = round(df.x)
    new_df = df.groupby(['round']).mean()
    q1_df = df.groupby(['round']).quantile(0.25)
    q3_df = df.groupby(['round']).quantile(0.75)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.yData, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.ySimul, color="green", label="Simulation")
    # axes = plt.axes()
    ax.set_ylim([0, 1700])
    ax.set_xlim([0, 50])
    ax.fill_between(np.arange(
        max(df["round"] + 1)), q1_df.ySimul, q3_df.ySimul, color="lightgreen",
        label="Simul. interquart. range")
    ax.fill_between(np.arange(
        max(df["round"] + 1)), q1_df.yData, q3_df.yData, color="lightgrey",
        alpha=0.5, label="Data. interquart. range")
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Average households density (per km²)", labelpad=15)
    plt.tick_params(bottom=True, labelbottom=True)
    plt.tick_params(labelbottom=True)
    # plt.title("Population density")
    plt.savefig(path_plots + 'validation_density.png')
    plt.close()

    df.to_csv(path_tables + 'validation_density.csv')
    print('validation_density done')

    return df


def valid_pop_housing_types(
        grid, initial_state_households_housing_types, housing_types,
        path_plots, path_tables):
    """
    Validation line plot for nb of households per housing type across 1D-space.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_households_housing_types : ndarray(float64, ndim=2)
        Number of households per grid cell in each housing type (4)
    housing_types : DataFrame
        Table yielding, for 4 different housing types (informal settlements,
        formal backyards, informal backyards, and formal private housing),
        the number of households in each grid cell (24,014), from SAL data.
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Validation table for number of households (no density) per housing type
        across 1D-space.

    """
    # Housing types
    xData = grid.dist
    #  Here, we take RDP out of formal housing to plot formal private
    formal_data = (housing_types.formal_grid
                   - initial_state_households_housing_types[3, :])
    backyard_data = housing_types.backyard_informal_grid
    informal_data = housing_types.informal_grid
    formal_simul = initial_state_households_housing_types[0, :]
    informal_simul = initial_state_households_housing_types[2, :]
    backyard_simul = initial_state_households_housing_types[1, :]
    # Note that RDP population is not actually simulated but taken from data
    rdp_simul = initial_state_households_housing_types[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_data, backyard_data, informal_data, formal_simul,
             backyard_simul, informal_simul, rdp_simul]
            )),
        columns=["xData", "formal_data", "backyard_data", "informal_data",
                 "formal_simul", "backyard_simul", "informal_simul",
                 "rdp_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    # plt.figure(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_simul, color="green", label="Simulation")
    # axes = plt.axes()
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # plt.title("Formal")
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (formal private)", labelpad=15)
    plt.savefig(path_plots + 'validation_pop_formal.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # plt.title("Informal")
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal settlements)",
               labelpad=15)
    plt.legend()
    plt.tick_params(labelbottom=True)
    # plt.xticks(
    #     [10.5, 13, 16, 18, 24, 25, 27, 30, 37, 39, 46.5],
    #     ["Joe Slovo", "Hout Bay", "Du Noon", "Philippi", "Khayelitsa",
    #      "Wallacedene", "Khayelitsa", "Witsand", "Enkanini", "Pholile"],
    #     rotation='vertical')
    plt.savefig(path_plots + 'validation_pop_informal.png')
    plt.close()

    # print("1")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # plt.title("Backyard")
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal backyards)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_pop_backyard.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rdp_simul, color="black")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Total number of households (formal subsidized)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_pop_rdp.png')
    plt.close()

    df.to_csv(path_tables + 'validation_pop_per_housing.csv')
    print('validation_pop_per_housing done')

    return df


def simul_pop_housing_types(
        grid, initial_state_households_housing_types,
        path_plots, path_tables):
    """
    Line plot for number of households per housing type across 1D-space.

    This is used specifically for subsequent periods where no validation data
    is available.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_households_housing_types : ndarray(float64, ndim=2)
        Number of households per grid cell in each housing type (4)
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table for number of households (no density) per housing type across
        1D-space.

    """
    # Housing types
    xData = grid.dist
    formal_simul = initial_state_households_housing_types[0, :]
    informal_simul = initial_state_households_housing_types[2, :]
    backyard_simul = initial_state_households_housing_types[1, :]
    # Note that RDP population is not actually simulated but taken from data
    rdp_simul = initial_state_households_housing_types[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_simul,
             backyard_simul, informal_simul, rdp_simul]
            )),
        columns=["xData",
                 "formal_simul", "backyard_simul", "informal_simul",
                 "rdp_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    # plt.figure(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_simul, color="gold", label='Formal')
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_simul, color="red", label="Informal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_simul, color="darkorange", label="Backyard")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rdp_simul, color="maroon", label='Subsidized')
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households per housing type", labelpad=15)
    plt.savefig(path_plots + 'simulation_density_htype.png')
    plt.close()

    df.to_csv(path_tables + 'simulation_pop_per_housing.csv')
    print('simulation_pop_per_housing done')

    return df


def valid_pop_income_groups(
        grid, initial_state_household_centers, income_distribution_grid,
        path_plots, path_tables):
    """
    Validation line plot for nb of households per income group across 1D-space.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_household_centers : ndarray(float64, ndim=2)
        Number of households per grid cell in each income group (4)
        at baseline year (2011)
    income_distribution_grid : ndarray(uint16, ndim=2)
        Exogenous number of households in each grid cell (24,014) for each
        income group in the model (4)
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Validation table for nb of households per income group across 1D-space
        (no density)

    """
    # We apply same reweighting as in equilibrium to match aggregate
    # SAL data
    ratio = (np.nansum(initial_state_household_centers)
             / np.nansum(income_distribution_grid))
    income_distribution_grid = (
        income_distribution_grid * ratio)

    # Income groups
    xData = grid.dist
    poor_data = income_distribution_grid[0, :]
    midpoor_data = income_distribution_grid[1, :]
    midrich_data = income_distribution_grid[2, :]
    rich_data = income_distribution_grid[3, :]
    poor_simul = initial_state_household_centers[0, :]
    midpoor_simul = initial_state_household_centers[1, :]
    midrich_simul = initial_state_household_centers[2, :]
    rich_simul = initial_state_household_centers[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, poor_data, midpoor_data, midrich_data, rich_data,
             poor_simul, midpoor_simul, midrich_simul, rich_simul]
            )),
        columns=["xData", "poor_data", "midpoor_data", "midrich_data",
                 "rich_data", "poor_simul", "midpoor_simul", "midrich_simul",
                 "rich_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (poor)", labelpad=15)
    plt.savefig(path_plots + 'validation_pop_poor.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (mid-poor)", labelpad=15)
    plt.savefig(path_plots + 'validation_pop_midpoor.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (mid-rich)", labelpad=15)
    plt.savefig(path_plots + 'validation_pop_midrich.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (rich)", labelpad=15)
    plt.savefig(path_plots + 'validation_pop_rich.png')
    plt.close()

    df.to_csv(path_tables + 'validation_pop_per_income.csv')
    print('validation_pop_per_income done')

    return df


def simul_pop_income_groups(
        grid, initial_state_household_centers,
        path_plots, path_tables):
    """
    Line plot for number of households per income group across 1D-space.

    This is used specifically for subsequent periods when validation data is
    not available.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_household_centers : ndarray(float64, ndim=2)
        Number of households per grid cell in each income group (4)
        at baseline year (2011)
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table for number of households per income group across 1D-space (no
        density)

    """
    # Income groups
    xData = grid.dist
    poor_simul = initial_state_household_centers[0, :]
    midpoor_simul = initial_state_household_centers[1, :]
    midrich_simul = initial_state_household_centers[2, :]
    rich_simul = initial_state_household_centers[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData,
             poor_simul, midpoor_simul, midrich_simul, rich_simul]
            )),
        columns=["xData",
                 "poor_simul", "midpoor_simul", "midrich_simul",
                 "rich_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_simul, color="maroon", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_simul, color="red", label="Mid-poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_simul, color="darkorange", label="Mid-rich")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_simul, color="gold", label="Rich")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households per income group", labelpad=15)
    plt.savefig(path_plots + 'simulation_pop_incgroup.png')
    plt.close()

    df.to_csv(path_tables + 'simulation_pop_per_income.csv')
    print('simulation_pop_per_income done')

    return df


def simul_pop_htype_income(
        grid, initial_state_households, path_plots, path_tables):
    """
    Line plot per housing and income groups across 1D-space (no validation).

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_households : ndarray(float64, ndim=3)
        Number of households per grid cell in each income group (4) and
        each housing type (4)
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : Dataframe
        Validation table for number of households (no density) per housing and
        income groups across 1D-space

    """
    # Housing and income groups
    xData = grid.dist

    formal_poor_simul = initial_state_households[0, 0, :]
    formal_midpoor_simul = initial_state_households[0, 1, :]
    formal_midrich_simul = initial_state_households[0, 2, :]
    formal_rich_simul = initial_state_households[0, 3, :]

    backyard_poor_simul = initial_state_households[1, 0, :]
    backyard_midpoor_simul = initial_state_households[1, 1, :]

    informal_poor_simul = initial_state_households[2, 0, :]
    informal_midpoor_simul = initial_state_households[2, 1, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_poor_simul, formal_midpoor_simul,
             formal_midrich_simul, formal_rich_simul,
             backyard_poor_simul, backyard_midpoor_simul,
             informal_poor_simul, informal_midpoor_simul]
            )),
        columns=["xData", "formal_poor_simul", "formal_midpoor_simul",
                 "formal_midrich_simul", "formal_rich_simul",
                 "backyard_poor_simul", "backyard_midpoor_simul",
                 "informal_poor_simul", "informal_midpoor_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_midrich_simul, color="gold", label="Mid-rich")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_rich_simul, color="darkorange", label="Rich")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (formal private)", labelpad=15)
    plt.savefig(path_plots + 'simul_pop_FP_income.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal backyards)", labelpad=15)
    plt.savefig(path_plots + 'simul_pop_IB_income.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal settlements)",
               labelpad=15)
    plt.savefig(path_plots + 'simul_pop_IS_income.png')
    plt.close()

    df.to_csv(path_tables + 'simul_pop_per_housing_and_income.csv')
    print('simul_pop_per_housing_and_income done')

    return df


def plot_income_net_of_commuting_costs(
        grid, income_net_of_commuting_costs, path_plots, path_tables):
    """
    Line plot for average income net of commuting costs across 1D-space.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    income_net_of_commuting_costs : ndarray(float64, ndim=2)
        Expected annual income net of commuting costs (in rands, for
        one household), for each geographic unit, by income group (4)
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table for average income net of commuting costs across 1D-space

    """
    # Housing and income groups
    xData = grid.dist

    poor_simul = income_net_of_commuting_costs[0, :]
    midpoor_simul = income_net_of_commuting_costs[1, :]
    midrich_simul = income_net_of_commuting_costs[2, :]
    rich_simul = income_net_of_commuting_costs[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, poor_simul, midpoor_simul,
             midrich_simul, rich_simul]
            )),
        columns=["xData", "poor_simul", "midpoor_simul",
                 "midrich_simul", "rich_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_simul, color="gold", label="Mid-rich")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_simul, color="darkorange", label="Rich")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Average annual incomes net of commuting costs (rands, 2011)",
               labelpad=15)
    plt.savefig(path_plots + 'avg_income_net_of_commuting_1d.png')
    plt.close()

    df.to_csv(path_tables + 'avg_income_net_of_commuting_1d.csv')
    print('avg_income_net_of_commuting_1d done')

    return df


def plot_average_income(
        grid, average_income, path_plots, path_tables):
    """
    Line plot for average income across 1D-space.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    average_income : ndarray(float64)
        Average median income for each income group in the model (4)
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table for average income across 1D-space.

    """
    # Housing and income groups
    xData = grid.dist

    poor_simul = average_income[0, :]
    midpoor_simul = average_income[1, :]
    midrich_simul = average_income[2, :]
    rich_simul = average_income[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, poor_simul, midpoor_simul,
             midrich_simul, rich_simul]
            )),
        columns=["xData", "poor_simul", "midpoor_simul",
                 "midrich_simul", "rich_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_simul, color="gold", label="Mid-rich")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_simul, color="darkorange", label="Rich")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Estimated average incomes (rands, 2011)", labelpad=15)
    plt.savefig(path_plots + 'avg_income_1d.png')
    plt.close()

    df.to_csv(path_tables + 'avg_income_1d.csv')
    print('avg_income_1d done')

    return df


def valid_housing_supply(grid, initial_state_housing_supply, path_plots,
                         path_tables):
    """
    Line plot for housing supply per unit of available land across 1D-space.

    Breakdown is given per housing type.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_housing_supply : ndarray(float64, ndim=2)
        Housing supply per unit of available land (in m² per km²)
        for each housing type (4) in each grid cell
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table for housing supply per unit of available land across 1D-space

    """
    xData = grid.dist
    formal_simul = initial_state_housing_supply[0, :]
    backyard_simul = initial_state_housing_supply[1, :]
    informal_simul = initial_state_housing_supply[2, :]
    rdp_simul = initial_state_housing_supply[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_simul, backyard_simul, informal_simul,
             rdp_simul]
            )),
        columns=["xData", "formal_simul", "backyard_simul", "informal_simul",
                 "rdp_simul"]
        )
    df["round"] = round(df.xData)
    # NB: we take average instead of sum to be able to plot all housing types
    # in the same graph
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_simul, color="darkorange", label="Formal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_simul, color="gold", label="Backyard")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_simul, color="darkgreen", label="Informal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rdp_simul, color="darkblue", label="RDP")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Avg housing supply (in m² per km² of available land)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_housing_supply.png')
    plt.close()

    df.to_csv(path_tables + 'validation_housing_supply.csv')
    print('validation_housing_supply done')

    return df


def valid_housing_supply_noland(grid, initial_state_housing_supply, path_plots,
                                path_tables):
    """
    Line plot for housing supply (no land availability) across 1D-space.

    Breakdown is given per housing type.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_housing_supply : ndarray(float64, ndim=2)
        Housing supply per unit of available land (in m² per km²)
        for each housing type (4) in each grid cell
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table for housing supply (no land availability) across 1D-space

    """
    xData = grid.dist
    formal_simul = initial_state_housing_supply[0, :]
    backyard_simul = initial_state_housing_supply[1, :]
    informal_simul = initial_state_housing_supply[2, :]
    rdp_simul = initial_state_housing_supply[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_simul, backyard_simul, informal_simul,
             rdp_simul]
            )),
        columns=["xData", "formal_simul", "backyard_simul", "informal_simul",
                 "rdp_simul"]
        )
    df["round"] = round(df.xData)
    # NB: we take average instead of sum to be able to plot all housing types
    # in the same graph
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_simul, color="darkorange", label="Formal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_simul, color="gold", label="Backyard")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_simul, color="darkgreen", label="Informal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rdp_simul, color="darkblue", label="RDP")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total housing supply (in m²)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_housing_supply_noland.png')
    plt.close()

    df.to_csv(path_tables + 'validation_housing_supply_noland.csv')
    print('validation_housing_supply_noland done')

    return df


def simulation_housing_price(
        grid, initial_state_rent, interest_rate, param, center,
        housing_types_sp,
        path_plots, path_tables, land_price):
    """
    Line plot for housing/land prices/rents across 1D-space.

    Breakdown is given per housing type. This function is specifically used
    for subsequent periods when validation data is not available.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_rent : ndarray(float64, ndim=2)
        Average annual rent (in rands) per grid cell for each housing type (4)
    interest_rate : float64
        Interest rate for the overall economy, corresponding to an average
        over past years
    param : dict
        Dictionary of default parameters
    center : ndarray(float64)
        x and y coordinates of geographic centre of analysis grid
    housing_types_sp : DataFrame
        Table yielding, for each Small Place (1,046), the number of informal
        backyards, of informal settlements, and total dwelling units, as well
        as their (centroid) x and y coordinates
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots
    land_price : int
        Dummy set to 1 or 0, depending on whether we want to consider
        theoretical land price or annual housing rent

    Returns
    -------
    df : DataFrame
        Table for housing prices across 1D-space

    """
    sp_x = housing_types_sp["x_sp"]
    sp_y = housing_types_sp["y_sp"]
    xData = np.sqrt((sp_x - center[0]) ** 2 + (sp_y - center[1]) ** 2)

    # Note that in informal sectors, the annual housing rent is the same as
    # the annual land rent. The formula for underlying land price derived from
    # developer's decisions only makes sense in the formal private sector
    if land_price == 1:

        priceSimul = (
            ((initial_state_rent[0, :] * param["coeff_A"])
             / (interest_rate + param["depreciation_rate"]))
            ** (1 / param["coeff_a"])
            * param["coeff_a"]
            * param["coeff_b"] ** (param["coeff_b"] / param["coeff_a"])
            )

        priceSimulPricePoints_formal = griddata(
            np.transpose(np.array([grid.x, grid.y])),
            priceSimul,
            np.transpose(np.array([sp_x, sp_y]))
            )

        ySimulation = priceSimulPricePoints_formal
        df = pd.DataFrame(
            data=np.transpose(np.array([xData, ySimulation])),
            columns=["xData", "ySimulation"])
        df["round"] = round(df.xData)
        new_df = df.groupby(['round']).mean()
        which = ~np.isnan(new_df.ySimulation)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(new_df.xData[which], new_df.ySimulation[which],
                color="green", label="Formal")
        ax.set_ylim(0)
        ax.set_xlim([0, 50])
        ax.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel("Distance to the city center (km)", labelpad=15)
        plt.ylabel("Land price (R/m² of land)", labelpad=15)

        plt.legend()
        plt.tick_params(labelbottom=True)
        plt.tick_params(bottom=True, labelbottom=True)
        plt.savefig(path_plots + '/simulation_land_price.png')
        plt.close()

        df.to_csv(path_tables + 'simulation_land_price.csv')
        print('simulation_land_price done')

    elif land_price == 0:
        priceSimul = initial_state_rent

        priceSimulPricePoints_formal = griddata(
            np.transpose(np.array([grid.x, grid.y])),
            priceSimul[0, :],
            np.transpose(np.array([sp_x, sp_y]))
            )
        priceSimulPricePoints_informal = griddata(
            np.transpose(np.array([grid.x, grid.y])),
            priceSimul[1, :],
            np.transpose(np.array([sp_x, sp_y]))
            )
        priceSimulPricePoints_backyard = griddata(
            np.transpose(np.array([grid.x, grid.y])),
            priceSimul[2, :],
            np.transpose(np.array([sp_x, sp_y]))
            )

        ySimulation = priceSimulPricePoints_formal
        informalSimul = priceSimulPricePoints_informal
        backyardSimul = priceSimulPricePoints_backyard

        df = pd.DataFrame(
            data=np.transpose(np.array([xData, ySimulation, informalSimul,
                                        backyardSimul])),
            columns=["xData", "ySimulation", "informalSimul",
                     "backyardSimul"])
        df["round"] = round(df.xData)
        new_df = df.groupby(['round']).mean()

        which = ~np.isnan(new_df.ySimulation)
        which_informal = ~np.isnan(new_df.informalSimul)
        which_backyard = ~np.isnan(new_df.backyardSimul)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(new_df.xData[which], new_df.ySimulation[which],
                color="green", label="Formal")
        ax.plot(new_df.xData[which_informal],
                new_df.informalSimul[which_informal],
                color="red", label="Informal")
        ax.plot(new_df.xData[which_backyard],
                new_df.backyardSimul[which_backyard],
                color="blue", label="Backyard")
        ax.set_ylim(0)
        ax.set_xlim([0, 50])
        ax.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel("Distance to the city center (km)", labelpad=15)
        plt.ylabel("Annual housing rent (R/m² of housing)", labelpad=15)

        plt.legend()
        plt.tick_params(labelbottom=True)
        plt.tick_params(bottom=True, labelbottom=True)
        plt.savefig(path_plots + '/simulation_housing_rent.png')
        plt.close()

        df.to_csv(path_tables + 'simulation_housing_rent.csv')
        print('simulation_housing_rent done')

    return df


def valid_housing_price(
        grid, initial_state_rent, interest_rate, param,
        housing_types_sp, data_sp,
        path_plots, path_tables):
    """
    Line plot for housing prices across 1D-space.

    This focuses on validation for housing prices in the formal private sector.
    Note that we have missing values in the city centre and beyond 40km.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_rent : ndarray(float64, ndim=2)
        Average annual rent (in rands) per grid cell for each housing type (4)
    interest_rate : float64
        Interest rate for the overall economy, corresponding to an average
        over past years
    param : dict
        Dictionary of default parameters
    housing_types_sp : DataFrame
        Table yielding, for each Small Place (1,046), the number of informal
        backyards, of informal settlements, and total dwelling units, as well
        as their (centroid) x and y coordinates
    data_sp : DataFrame
        Table yielding, for each Small Place (1,046), the average dwelling size
        (in m²), the average land price and annual income level (in rands),
        the size of unconstrained area for construction (in m²), the total area
        (in km²), the distance to the city centre (in km), whether or not the
        location belongs to Mitchells Plain, and the SP code
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table for formal private housing prices across 1-D space (with
        validation data)

    """
    sp_x = housing_types_sp["x_sp"]
    sp_y = housing_types_sp["y_sp"]
    sp_price = data_sp["price"]
    priceSimul = initial_state_rent[0, :] / interest_rate
    # priceSimul = (
    #     ((initial_state_rent[0, :] * param["coeff_A"])
    #      / (interest_rate + param["depreciation_rate"]))
    #     ** (1 / param["coeff_a"])
    #     * param["coeff_a"]
    #     * param["coeff_b"] ** (param["coeff_b"] / param["coeff_a"])
    #     )

    xData = grid.dist
    yData = griddata(
        np.transpose(np.array([sp_x, sp_y])),
        sp_price,
        np.transpose(np.array([grid.x, grid.y]))
        )
    ySimulation = priceSimul

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, yData, ySimulation]
            )),
        columns=["xData", "yData", "ySimulation"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    new_df.to_csv(path_tables + 'validation_housing_price.csv')

    which_simul = ~np.isnan(new_df.ySimulation)
    which_data = ~np.isnan(new_df.yData)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(new_df.xData[which_data], new_df.yData[which_data],
            color="black", label="Data")
    ax.plot(new_df.xData[which_simul], new_df.ySimulation[which_simul],
            color="green", label="Simul")

    ax.set_ylim(0)
    ax.set_xlim([0, 40])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Average housing price/m² in formal private",
               labelpad=15)
    plt.savefig(path_plots + 'validation_housing_price.png')
    plt.close()

    return df


def valid_housing_demand(grid, center, initial_state_dwelling_size,
                         initial_state_households_housing_types,
                         housing_types_sp, data_sp,
                         path_plots, path_tables):
    """
    Line plot average dwelling size in formal private housing across 1D-space.

    Note that this is a validation plot using data at SP level.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    center : ndarray(float64)
        x and y coordinates of geographic centre of analysis grid
    initial_state_dwelling_size : ndarray(float64, ndim=2)
        Average dwelling size (in m²) per grid cell in each housing
        type (4)
    initial_state_households_housing_types : ndarray(float64, ndim=2)
        Number of households per grid cell in each housing type (4)
    housing_types_sp : DataFrame
        Table yielding, for each Small Place (1,046), the number of informal
        backyards, of informal settlements, and total dwelling units, as well
        as their (centroid) x and y coordinates
    data_sp : DataFrame
        Table yielding, for each Small Place (1,046), the average dwelling size
        (in m²), the average land price and annual income level (in rands),
        the size of unconstrained area for construction (in m²), the total area
        (in km²), the distance to the city centre (in km), whether or not the
        location belongs to Mitchells Plain, and the SP code
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table with simulated and validation dwelling sizes across 1-D space

    """
    sp_x = housing_types_sp["x_sp"]
    sp_y = housing_types_sp["y_sp"]
    sp_size = data_sp["dwelling_size"]

    # sizeSimulPoints = griddata(
    #     np.transpose(np.array([grid.x, grid.y])),
    #     initial_state_dwelling_size[0, :],
    #     np.transpose(np.array([sp_x, sp_y]))
    #     )
    sizeSimulPoints = initial_state_dwelling_size[0, :]

    # xData = np.sqrt((sp_x - center[0]) ** 2 + (sp_y - center[1]) ** 2)
    xData = grid.dist
    # yData = sp_size
    yData = griddata(
        np.transpose(np.array([sp_x, sp_y])),
        sp_size,
        np.transpose(np.array([grid.x, grid.y]))
        )
    # xSimulation = xData
    ySimulation = sizeSimulPoints

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, yData, ySimulation]
            )),
        columns=["xData", "yData", "ySimulation"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    new_df.to_csv(path_tables + 'validation_housing_demand.csv')

    which_simul = ~np.isnan(new_df.ySimulation)
    which_data = ~np.isnan(new_df.yData)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(new_df.xData[which_data], new_df.yData[which_data],
            color="black", label="Data")
    ax.plot(new_df.xData[which_simul], new_df.ySimulation[which_simul],
            color="green", label="Simul")

    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Average dwelling size in formal private (in m²)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_housing_demand.png')
    plt.close()

    return df


def simul_housing_demand(grid, center, initial_state_dwelling_size,
                         initial_state_households_housing_types,
                         path_plots, path_tables):
    """
    Line plot average dwelling size in formal private housing across 1D-space.

    This is not a validation plot function, and it is specifically used for
    subsequent periods when validation data is not available.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    center : ndarray(float64)
        x and y coordinates of geographic centre of analysis grid
    initial_state_dwelling_size : ndarray(float64, ndim=2)
        Average dwelling size (in m²) per grid cell in each housing
        type (4)
    initial_state_households_housing_types : ndarray(float64, ndim=2)
        Number of households per grid cell in each housing type (4)
    path_plots : str
        Path for saving output plots
    path_tables : str
        Path for saving output plots

    Returns
    -------
    df : DataFrame
        Table with simulated dwelling sizes across 1-D space

    """

    # sizeSimulPoints = griddata(
    #     np.transpose(np.array([grid.x, grid.y])),
    #     initial_state_dwelling_size[0, :],
    #     np.transpose(np.array([sp_x, sp_y]))
    #     )
    sizeSimulPoints = initial_state_dwelling_size[0, :]

    # xData = np.sqrt((sp_x - center[0]) ** 2 + (sp_y - center[1]) ** 2)
    xData = grid.dist
    # yData = sp_size
    # xSimulation = xData
    ySimulation = sizeSimulPoints

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, ySimulation]
            )),
        columns=["xData", "ySimulation"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    new_df.to_csv(path_tables + 'simulation_housing_demand.csv')

    which_simul = ~np.isnan(new_df.ySimulation)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(new_df.xData[which_simul], new_df.ySimulation[which_simul],
            color="green")

    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Average dwelling size in formal private (in m²)",
               labelpad=15)
    plt.savefig(path_plots + 'simulation_housing_demand.png')
    plt.close()

    return df
