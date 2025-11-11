# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import copy

import equilibrium.sub.compute_outputs as eqout


def compute_equilibrium(fraction_capital_destroyed, amenities, param,
                        housing_limit, population, households_per_income_class,
                        total_RDP, coeff_land, income_net_of_commuting_costs,
                        grid, options, agricultural_rent, interest_rate,
                        number_properties_RDP, average_income, mean_income,
                        income_class_by_housing_type, minimum_housing_supply,
                        construction_param, income_baseline):
    """
    Run the static equilibrium algorithm.

    This function runs the algorithm described in the technical documentation.
    It starts from arbitrary utility levels, and leverages optimality
    conditions on supply and demand to recover key output variables
    (detailed in equilibrium.sub.compute_outputs). Then, it updates utility
    levels to minimize the error between simulated and target number of
    households per income group. Note that the whole process abstracts from
    formal subsidised housing (fully exogenous in the model), that is added
    to final outputs at the end of the function.

    Parameters
    ----------
    fraction_capital_destroyed : DataFrame
        Data frame of expected fractions of capital destroyed, for housing
        structures and contents in different housing types, in each
        grid cell (24,014)
    amenities : ndarray(float64)
        Normalized amenity index (relative to the mean) for each grid cell
        (24,014)
    param : dict
        Dictionary of default parameters
    housing_limit : Series
        Maximum housing supply (in m² per km²) in each grid cell (24,014)
    population : int64
        Total number of households in the city (from Small-Area-Level data)
    households_per_income_class : ndarray(float64)
        Exogenous total number of households per income group (excluding people
        out of employment, for 4 groups)
    total_RDP : int
        Number of households living in formal subsidized housing (from SAL
        data)
    coeff_land : ndarray(float64, ndim=2)
        Updated land availability for each grid cell (24,014) and each
        housing type (4: formal private, informal backyards, informal
        settlements, formal subsidized)
    income_net_of_commuting_costs : ndarray(float64, ndim=2)
        Expected annual income net of commuting costs (in rands, for
        one household), for each geographic unit, by income group (4)
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    options : dict
        Dictionary of default options
    agricultural_rent : float64
        Annual housing rent below which it is not profitable for formal private
        developers to urbanize (agricultural) land: endogenously limits urban
        sprawl
    interest_rate : float64
        Real interest rate for the overall economy, corresponding to an average
        over past years
    number_properties_RDP : ndarray(float64)
        Number of formal subsidized dwellings per grid cell (24,014) at
        baseline year (2011)
    average_income : ndarray(float64)
        Average median income for each income group in the model (4)
    mean_income : float64
        Average median income across total population
    income_class_by_housing_type : DataFrame
        Set of dummies coding for housing market access (across 4 housing
        submarkets) for each income group (4, from poorest to richest)
    minimum_housing_supply : ndarray(float64)
        Minimum housing supply (in m²) for each grid cell (24,014), allowing
        for an ad hoc correction of low values in Mitchells Plain
    construction_param : ndarray(float64)
        (Calibrated) scale factor for the construction function of formal
        private developers
    income_baseline : DataFrame
        Table summarizing, for each income group in the data (12, including
        people out of employment), the number of households living in each
        endogenous housing type (3), their total number at baseline year (2011)
        in retrospect (2001), as well as the distribution of their average
        income (at baseline year)

    Returns
    -------
    initial_state_utility : ndarray(float64)
        Utility levels for each income group (4) at baseline year
        (2011)
    initial_state_error : ndarray(float64)
        Ratio (in %) of simulated number of households per income group over
        target population per income group at baseline year (2011)
    initial_state_simulated_jobs : ndarray(float64, ndim=2)
        Total number of households in each income group (4) in each
        endogenous housing type (3: formal private, informal backyards,
        informal settlements) at baseline year (2011)
    initial_state_households_housing_types : ndarray(float64, ndim=2)
        Number of households per grid cell in each housing type (4)
        at baseline year (2011)
    initial_state_household_centers : ndarray(float64, ndim=2)
        Number of households per grid cell in each income group (4)
        at baseline year (2011)
    initial_state_households : ndarray(float64, ndim=3)
        Number of households per grid cell in each income group (4) and
        each housing type (4) at baseline year (2011)
    initial_state_dwelling_size : ndarray(float64, ndim=2)
        Average dwelling size (in m²) per grid cell in each housing
        type (4) at baseline year (2011)
    initial_state_housing_supply : ndarray(float64, ndim=2)
        Housing supply per unit of available land (in m² per km²)
        for each housing type (4) in each grid cell at baseline year (2011)
    initial_state_rent : ndarray(float64, ndim=2)
        Average annual rent (in rands) per grid cell for each housing type
        (4) at baseline year (2011)
    initial_state_rent_matrix : ndarray(float64, ndim=3)
        Average annual willingness to pay (in rands) per grid cell
        for each income group (4) and each endogenous housing type (3)
        at baseline year (2011)
    initial_state_capital_land : ndarray(float64, ndim=2)
        Value (in rands) of the housing capital stock per unit of available
        land (in km²) for each endogenous housing type (3) per grid cell
        at baseline year (2011)
    initial_state_average_income : ndarray(float64)
        Not an output of the model per se : it is just the average median
        income for each income group in the model (4), that may change
        over time
    initial_state_limit_city : list
        Contains a ndarray(bool, ndim=3) of indicator dummies for having
        strictly more than one household per housing type and income group
        in each grid cell

    """
    # Adjust the population to include unemployed people, then take out RDP
    # by considering that they all belong to poorest income group

    # IRRELEVANT AS THIS WILL ONLY AFFECT SIZE OF THE ERROR
    
    # General reweighting using SAL data (no formal backyards)
    # if options["unempl_reweight"] == 0:
        
    # NB: not used in income calibration, is it a problem?
    ratio = population / sum(households_per_income_class)
    households_per_income_class = households_per_income_class * ratio

    # Alternative strategy: we attribute the unemployed population in
    # proportion with calibrated unemployment rates, without applying them
    # directly (as they are too noisy)
    # if options["unempl_reweight"] == 1:
    #     ratio = [2 / size for size in param["household_size"]]
    #     households_tot = households_per_income_class * ratio
    #     households_unempl = households_tot - households_per_income_class
    #     weights = households_unempl / sum(households_unempl)
    #     unempl_pop = income_baseline.Households_nb[0]
    #     unempl_attrib = [unempl_pop * w for w in weights]
    #     households_per_income_class = (
    #         households_per_income_class + unempl_attrib)
    #     ratio = population / sum(households_per_income_class)
    #     households_per_income_class = households_per_income_class * ratio
        # implicit_empl_rate = ((households_per_income_class - unempl_attrib)
        #                       / households_per_income_class)
        # 0.74/0.99/0.98/0.99

    # TEST
    # households_per_income_class = (
    #     households_per_income_class*population_growth_rate)

    #  Considering that all RDP belong to the poorest, we remove them from here
    households_per_income_class[0] = np.max(
        households_per_income_class[0] - total_RDP, 0)

    # Shorten the grid
    #  We select pixels with constructible shares for all housing types > 0.01
    #  and a positive maximum income net of commuting costs across classes
    #  NB: we will have no output in other pixels (numeric simplification)
    selected_pixels = (
        (np.sum(coeff_land, 0) > 0.01).squeeze()
        & (np.nanmax(income_net_of_commuting_costs, 0) > 0)
        )
    coeff_land_full = copy.deepcopy(coeff_land)
    coeff_land = coeff_land[:, selected_pixels]
    grid_temp = copy.deepcopy(grid)
    grid = grid.iloc[selected_pixels, :]
    housing_limit = housing_limit[selected_pixels]
    income_net_of_commuting_costs = income_net_of_commuting_costs[
        :, selected_pixels]
    minimum_housing_supply = minimum_housing_supply[selected_pixels]
    housing_in = copy.deepcopy(param["housing_in"][selected_pixels])
    amenities = amenities[selected_pixels]
    fraction_capital_destroyed = fraction_capital_destroyed.iloc[
        selected_pixels, :]
    param_pockets = param["informal_pockets"][selected_pixels]
    param_backyards_pockets = param["backyard_pockets"][selected_pixels]
    param_incremental_pockets = param["incremental_pockets"][selected_pixels]

    # Useful variables for the solver
    # (we only consider 3 types of housing in the solver)
    diff_utility = np.zeros((param["max_iter"], param["nb_of_income_classes"]))
    simulated_people_housing_types = np.zeros(
        (param["max_iter"], 3, len(grid.dist)))
    simulated_people = np.zeros((3, 4, len(grid.dist)))
    simulated_jobs = np.zeros(
        (param["max_iter"], 3, param["nb_of_income_classes"]))
    total_simulated_jobs = np.zeros(
        (param["max_iter"], param["nb_of_income_classes"]))
    rent_matrix = np.zeros((param["max_iter"], 3, len(grid.dist)))
    error_max_abs = np.zeros(param["max_iter"])
    error_max = np.zeros(param["max_iter"])
    error_mean = np.zeros(param["max_iter"])
    nb_error = np.zeros(param["max_iter"])
    error = np.zeros((param["max_iter"], param["nb_of_income_classes"]))
    housing_supply = np.empty((3, len(grid.dist)))
    dwelling_size = np.empty((3, len(grid.dist)))
    R_mat = np.empty((3, 4, len(grid.dist)))
    housing_supply[:] = np.nan
    dwelling_size[:] = np.nan
    R_mat[:] = np.nan

    # Initialisation solver
    utility = np.zeros((param["max_iter"], param["nb_of_income_classes"]))
    #  We take arbitrary utility levels, not too far from what we would expect,
    #  to make computation quicker

    # For the initial guess, we abstract from (dis)amenities and housing type and commuting costs
    # Then, we refine to ease computing given strong sorting
    
    # utility[0, :] = average_income**param["alpha"]*50**param["beta"]
    utility[0, :] = np.array([1200, 4800, 16000, 77000])
    # utility[0, :] = np.array([400, 900, 5400, 26000])
    
    index_iteration = 0
    #  We need to apply some convergence factor to our error terms to make them
    #  converge in our optimization: the formula comes from trial and error,
    #  but the intuition is that we put more weight on the error for households
    #  with high relative income as it will magnify the effect on rents, hence
    #  housing supply and resulting population distribution

    param["convergence_factor"] = (
        0.01 * (np.nanmean(average_income) / mean_income) ** 0.01
        )

    # Compute outputs solver (for each housing type, no RDP)
    #  Formal housing
    (simulated_jobs[index_iteration, 0, :], rent_matrix[index_iteration, 0, :],
     simulated_people_housing_types[index_iteration, 0, :],
     simulated_people[0, :, :], housing_supply[0, :], dwelling_size[0, :],
     R_mat[0, :, :]) = eqout.compute_outputs(
         'formal', utility[index_iteration, :], amenities, param,
         income_net_of_commuting_costs, fraction_capital_destroyed, grid,
         income_class_by_housing_type, options, housing_limit,
         agricultural_rent, interest_rate, coeff_land[0, :],
         minimum_housing_supply, construction_param, housing_in, param_pockets,
         param_backyards_pockets, param_incremental_pockets
         )
    #  Backyard housing
    (simulated_jobs[index_iteration, 1, :], rent_matrix[index_iteration, 1, :],
     simulated_people_housing_types[index_iteration, 1, :],
     simulated_people[1, :, :], housing_supply[1, :], dwelling_size[1, :],
     R_mat[1, :, :]) = eqout.compute_outputs(
         'backyard', utility[index_iteration, :], amenities, param,
         income_net_of_commuting_costs, fraction_capital_destroyed, grid,
         income_class_by_housing_type, options, housing_limit,
         agricultural_rent, interest_rate, coeff_land[1, :],
         minimum_housing_supply, construction_param, housing_in, param_pockets,
         param_backyards_pockets, param_incremental_pockets
         )
    #  Informal housing
    (simulated_jobs[index_iteration, 2, :], rent_matrix[index_iteration, 2, :],
     simulated_people_housing_types[index_iteration, 2, :],
     simulated_people[2, :, :], housing_supply[2, :], dwelling_size[2, :],
     R_mat[2, :, :]) = eqout.compute_outputs(
         'informal', utility[index_iteration, :], amenities, param,
         income_net_of_commuting_costs, fraction_capital_destroyed, grid,
         income_class_by_housing_type, options, housing_limit,
         agricultural_rent, interest_rate, coeff_land[2, :],
         minimum_housing_supply, construction_param, housing_in, param_pockets,
         param_backyards_pockets, param_incremental_pockets
         )

    # Compute error and adjust utility

    #  We first update the first iteration of output vector with the sum of
    #  simulated people per income group across housing types (no RDP)
    total_simulated_jobs[index_iteration, :] = np.sum(
        simulated_jobs[index_iteration, :, :], 0)

    #  diff_utility will be used to adjust the utility levels
    #  Note that optimization is made to stick to households_per_income_class,
    #  which does not include people living in RDP (as we take this as
    #  exogenous)

    #  We compare total population for each income group obtained from
    #  equilibrium condition (total_simulated_jobs) with target population
    #  allocation (households_per_income_class)

    #  We arbitrarily set a strictly positive minimum utility level at 10
    #  (as utility will be adjusted multiplicatively, we do not want to break
    #  the model with zero terms)

    diff_utility[index_iteration, :] = np.log(
        (total_simulated_jobs[index_iteration, :] + 10)
        / (households_per_income_class + 10))
    diff_utility[index_iteration, :] = (
        diff_utility[index_iteration, :] * param["convergence_factor"])
    (diff_utility[index_iteration, diff_utility[index_iteration, :] > 0]
     ) = (diff_utility[index_iteration, diff_utility[index_iteration, :] > 0]
          * 1.1)

    # Difference with reality
    error[index_iteration, :] = (total_simulated_jobs[index_iteration, :]
                                 / households_per_income_class - 1) * 100
    #  This is the parameter of interest for optimization
    error_max_abs[index_iteration] = np.nanmax(
        np.abs(total_simulated_jobs[index_iteration, :]
               / households_per_income_class - 1))
    #  Other parameters
    error_max[index_iteration] = -1
    error_mean[index_iteration] = np.nanmean(
        np.abs(total_simulated_jobs[index_iteration, :]
               / (households_per_income_class + 0.001) - 1))
    nb_error[index_iteration] = np.nansum(
        np.abs(total_simulated_jobs[index_iteration, :]
               / households_per_income_class - 1) > param["precision"])

    # Iteration (no RDP)
    # We use a progression bar
    with tqdm(total=param["max_iter"],
              desc="stops when error_max_abs <" + str(param["precision"])
              ) as pbar:
        while ((index_iteration < param["max_iter"] - 1)
               & (error_max_abs[index_iteration] > param["precision"])):

            # Adjust parameters to how close we are from the objective
            index_iteration = index_iteration + 1
            # When population is overestimated, we augment utility to reduce
            # population (cf. population constraint from standard model)
            utility[index_iteration, :] = np.exp(
                np.log(utility[index_iteration - 1, :])
                + diff_utility[index_iteration - 1, :])
            # This is a precaution as utility cannot be negative
            utility[index_iteration, utility[index_iteration, :] < 0] = 10

            # We reduce the convergence factor at each iteration in inverse
            # proportion with the estimation error not to overshoot target in
            # subsequent iterations

            # NB: we assume the minimum error is 100 not to break model with
            # zeros
            convergence_factor = (
                param["convergence_factor"] / (
                    1
                    + 0.5 * np.abs(
                        (total_simulated_jobs[index_iteration, :] + 100)
                        / (households_per_income_class + 100)
                        - 1)
                    )
                )

            # We also reduce it as time passes, as errors should become smaller
            # and smaller
            convergence_factor = (
                convergence_factor
                * (1 - 0.6 * index_iteration / param["max_iter"])
                )

            # Now, we do the same as in the initalization phase

            # Compute outputs solver

            #  Formal housing
            (simulated_jobs[index_iteration, 0, :],
             rent_matrix[index_iteration, 0, :],
             simulated_people_housing_types[index_iteration, 0, :],
             simulated_people[0, :, :],
             housing_supply[0, :],
             dwelling_size[0, :],
             R_mat[0, :, :]) = eqout.compute_outputs(
                 'formal', utility[index_iteration, :], amenities, param,
                 income_net_of_commuting_costs, fraction_capital_destroyed,
                 grid, income_class_by_housing_type, options, housing_limit,
                 agricultural_rent, interest_rate, coeff_land[0, :],
                 minimum_housing_supply, construction_param, housing_in,
                 param_pockets, param_backyards_pockets, param_incremental_pockets
                 )

            #  Backyard housing
            (simulated_jobs[index_iteration, 1, :],
             rent_matrix[index_iteration, 1, :],
             simulated_people_housing_types[index_iteration, 1, :],
             simulated_people[1, :, :],
             housing_supply[1, :],
             dwelling_size[1, :],
             R_mat[1, :, :]) = eqout.compute_outputs(
                 'backyard', utility[index_iteration, :], amenities, param,
                 income_net_of_commuting_costs, fraction_capital_destroyed,
                 grid, income_class_by_housing_type, options, housing_limit,
                 agricultural_rent, interest_rate, coeff_land[1, :],
                 minimum_housing_supply, construction_param, housing_in,
                 param_pockets, param_backyards_pockets, param_incremental_pockets
                 )

            #  Informal housing
            (simulated_jobs[index_iteration, 2, :],
             rent_matrix[index_iteration, 2, :],
             simulated_people_housing_types[index_iteration, 2, :],
             simulated_people[2, :, :],
             housing_supply[2, :],
             dwelling_size[2, :],
             R_mat[2, :, :]) = eqout.compute_outputs(
                 'informal', utility[index_iteration, :], amenities, param,
                 income_net_of_commuting_costs, fraction_capital_destroyed,
                 grid, income_class_by_housing_type, options, housing_limit,
                 agricultural_rent, interest_rate, coeff_land[2, :],
                 minimum_housing_supply, construction_param, housing_in,
                 param_pockets, param_backyards_pockets, param_incremental_pockets
                 )

            # Compute error and adjust utility
            total_simulated_jobs[index_iteration, :] = np.sum(
                simulated_jobs[index_iteration, :, :], 0)

            # diff_utility will be used to adjust the utility levels
            diff_utility[index_iteration, :] = np.log(
                (total_simulated_jobs[index_iteration, :] + 10)
                / (households_per_income_class + 10))
            diff_utility[index_iteration, :] = (
                diff_utility[index_iteration, :] * convergence_factor)
            diff_utility[
                index_iteration, diff_utility[index_iteration, :] > 0] = (
                    diff_utility[index_iteration,
                                 diff_utility[index_iteration, :] > 0] * 1.1
                    )

            # Variables to display
            error[index_iteration, :] = (
                total_simulated_jobs[index_iteration, :]
                / households_per_income_class - 1) #* 100
            error_max_abs[index_iteration] = np.max(np.abs(
                total_simulated_jobs[index_iteration,
                                     households_per_income_class != 0]
                / households_per_income_class - 1))
            m = np.argmax(np.abs(total_simulated_jobs[index_iteration, :]
                                 / households_per_income_class - 1))
            erreur_temp = (total_simulated_jobs[index_iteration, :]
                           / households_per_income_class - 1)
            error_max[index_iteration] = erreur_temp[m]

            error_mean[index_iteration] = np.mean(np.abs(
                total_simulated_jobs[index_iteration, :]
                / (households_per_income_class + 0.001) - 1))
            nb_error[index_iteration] = np.sum(np.abs(
                total_simulated_jobs[index_iteration, :]
                / households_per_income_class - 1) > param["precision"])
            pbar.set_postfix({'error_max_abs': error_max_abs[index_iteration]})
            pbar.update()

    
    # NOT CLEAR?

    # We plug back RDP houses in the output : let us define useful variables
    # first
    #  We correct output coming from data_RDP with more reliable estimations
    #  from SAL data (to include council housing)
    households_RDP = (number_properties_RDP * total_RDP
                      / sum(number_properties_RDP))
    #  Share of housing (no backyard) in RDP surface (with land in km²)
    construction_RDP = np.matlib.repmat(
        param["RDP_size"] / (param["RDP_size"] + param["backyard_size"]),
        1, len(grid_temp.dist))
    #  RDP dwelling size (in m²)
    dwelling_size_RDP = np.matlib.repmat(
        param["RDP_size"], 1, len(grid_temp.dist))

    # We fill the output matrix for each housing type
    simulated_people_with_RDP = np.zeros((4, 4, len(grid_temp.dist)))
    simulated_people_with_RDP[0, :, selected_pixels] = np.transpose(
        simulated_people[0, :, :, ])
    simulated_people_with_RDP[1, :, selected_pixels] = np.transpose(
        simulated_people[1, :, :, ])
    simulated_people_with_RDP[2, :, selected_pixels] = np.transpose(
        simulated_people[2, :, :, ])
    simulated_people_with_RDP[3, 0, :] = households_RDP

    # Outputs of the solver

    initial_state_error = error[index_iteration, :]
    
    #print(error_mean[index_iteration, :])

    #  Note that this does not contain RDP!!!
    initial_state_simulated_jobs = simulated_jobs[index_iteration, :, :]

    #  We sum across income groups (axis=1)
    initial_state_households_housing_types = np.sum(simulated_people_with_RDP,
                                                    1)
    #  We sum across housing types (axis=0)
    initial_state_household_centers = np.sum(simulated_people_with_RDP, 0)

    #  We keep both dimensions
    initial_state_households = simulated_people_with_RDP

    # Housing stock and dwelling size (fill with RDP)
    housing_supply_export = np.zeros((3, len(grid_temp.dist)))
    dwelling_size_export = np.zeros((3, len(grid_temp.dist)))
    housing_supply_export[:, selected_pixels] = housing_supply
    dwelling_size_export[:, selected_pixels] = copy.deepcopy(dwelling_size)
    dwelling_size_export[dwelling_size_export <= 0] = np.nan
    # NB: we multiply by construction_RDP because we want the housing supply
    # per unit of AVAILABLE land: RDP building is only a fraction of overall
    # surface (also accounts for backyarding)
    housing_supply_RDP = (
        construction_RDP * dwelling_size_RDP * households_RDP
        / (coeff_land_full[3, :] * 0.25)
        )
    housing_supply_RDP[np.isnan(housing_supply_RDP)] = 0
    dwelling_size_RDP = dwelling_size_RDP * (coeff_land_full[3, :] > 0)
    initial_state_dwelling_size = np.vstack(
        [dwelling_size_export, dwelling_size_RDP])
    # Note that RDP housing supply per unit of available land has nothing to do
    # with backyard housing supply per unit of available land
    initial_state_housing_supply = np.vstack(
        [housing_supply_export, housing_supply_RDP]
        )

    # Rents (HHs in RDP pay a rent of 0)
    rent_temp = copy.deepcopy(rent_matrix[index_iteration, :, :])
    rent_export = np.zeros((3, len(grid_temp.dist)))
    rent_export[:, selected_pixels] = copy.deepcopy(rent_temp)
    rent_export[:, selected_pixels == 0] = np.nan
    initial_state_rent = np.vstack(
        [rent_export, np.full(len(grid_temp.dist), np.nan)])
    rent_matrix_export = np.zeros(
        (3, len(average_income), len(grid_temp.dist)))
    rent_matrix_export[:, :, selected_pixels] = copy.deepcopy(R_mat)
    rent_matrix_export[:, :, selected_pixels == 0] = np.nan
    initial_state_rent_matrix = copy.deepcopy(rent_matrix_export)

    # Other outputs
    initial_state_utility = utility[index_iteration, :]
    # Housing capital value per unit of available land: see math appendix
    initial_state_capital_land = ((initial_state_housing_supply / 1000000
                                   / construction_param)
                                  ** (1 / param["coeff_b"]))
    #  NB: this is not an output of the model
    initial_state_average_income = copy.deepcopy(average_income)
    #  NB: this is not used in practice and is included for reference
    initial_state_limit_city = [initial_state_households > 1]

    return (initial_state_utility,
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
            initial_state_limit_city)
