# -*- coding: utf-8 -*-

import numpy as np
import equilibrium.sub.functions_solver as eqsol


def compute_outputs(housing_type,
                    utility,
                    amenities,
                    param,
                    income_net_of_commuting_costs,
                    fraction_capital_destroyed,
                    grid,
                    income_class_by_housing_type,
                    options,
                    housing_limit,
                    agricultural_rent,
                    interest_rate,
                    coeff_land,
                    minimum_housing_supply,
                    construction_param,
                    housing_in,
                    param_pockets,
                    param_backyards_pockets,
                    param_incremental_pockets):
    """
    Compute equilibrium outputs from theoretical formulas.

    From optimality conditions on supply and demand (see technical
    documentation for math formulas), this function computes, for a given
    housing type, the following outputs. First, the demanded dwelling size in
    each place per income group. Then, the bid-rent function / willingness to
    pay (per m² of housing) in each place per income group. By selecting the
    highest bid, we recover the final simulated dwweling size and market rent.
    From there, we also compute the housing supply per unit of available land,
    and the total number of households in each location, per income group.
    To do so, it leverages the equilibrium.sub.functions_solver module.

    Parameters
    ----------
    housing_type : str
        Endogenous housing type considered in the function: should be set to
        "formal", "backyard", or "informal"
    utility : ndarray(float64)
        Utility levels for each income group (4) considered in a given
        iteration
    amenities : ndarray(float64)
        Normalized amenity index (relative to the mean) for each grid cell
        (24,014)
    param : dict
        Dictionary of default parameters
    income_net_of_commuting_costs : ndarray(float64, ndim=2)
        Expected annual income net of commuting costs (in rands, for
        one household), for each geographic unit, by income group (4)
    fraction_capital_destroyed : DataFrame
        Data frame of expected fractions of capital destroyed, for housing
        structures and contents in different housing types, in each
        grid cell (24,014)
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    income_class_by_housing_type : DataFrame
        Set of dummies coding for housing market access (across 4 housing
        submarkets) for each income group (4, from poorest to richest)
    options : dict
        Dictionary of default options
    housing_limit : Series
        Maximum housing supply (in m² per km²) in each grid cell (24,014)
    agricultural_rent : float64
        Annual housing rent below which it is not profitable for formal private
        developers to urbanize (agricultural) land: endogenously limits urban
        sprawl
    interest_rate : float64
        Real interest rate for the overall economy, corresponding to an average
        over past years
    coeff_land : ndarray(float64, ndim=2)
        Updated land availability for each grid cell (24,014) and each
        housing type (4: formal private, informal backyards, informal
        settlements, formal subsidized)
    minimum_housing_supply : ndarray(float64)
        Minimum housing supply (in m²) for each grid cell (24,014), allowing
        for an ad hoc correction of low values in Mitchells Plain
    construction_param : ndarray(float64)
        (Calibrated) scale factor for the construction function of formal
        private developers
    housing_in : ndarray(float64)
        Theoretical minimum housing supply when formal private developers do
        not adjust (not used in practice), per grid cell (24,014)
    param_pockets : ndarray(float64)
        (Calibrated) disamenity index for living in an informal settlement,
        per grid cell (24,014)
    param_backyards_pockets : ndarray(float64)
        (Calibrated) disamenity index for living in an informal backyard,
        per grid cell (24,014)

    Returns
    -------
    job_simul : ndarray(float64)
        Simulated number of households per income group (4) for a given housing
        type, at a given iteration
    R : ndarray(float64)
        Simulated average annual rent (in rands/m²) for a given housing type,
        at a given iteration, for each selected pixel (4,043)
    people_init : ndarray(float64)
        Simulated number of households for a given housing type, at a given
        iteration, for each selected pixel (4,043)
    people_center : ndarray(float64, ndim=2)
        Simulated number of households for a given housing type, at a given
        iteration, for each selected pixel (4,043) and each income group (4)
    housing_supply : ndarray(float64)
        Simulated housing supply per unit of available land (in m² per km²)
        for a given housing type, at a given iteration, for each selected pixel
        (4,043)
    dwelling_size : ndarray(float64)
        Simulated average dwelling size (in m²) for a given housing type, at a
        given iteration, for each selected pixel (4,043)
    R_mat : ndarray(float64, ndim=2)
        Simulated willingness to pay / bid-rents (in rands/m²) for a given
        housing type, at a given iteration, for each selected pixel (4,043) and
        each income group (4)

    """
    # %% Dwelling size in selected pixels per (endogenous) housing type

    if housing_type == 'formal':

        dwelling_size = eqsol.compute_dwelling_size_formal(
            utility, amenities, param, income_net_of_commuting_costs,
            fraction_capital_destroyed)

        # Here, we introduce the minimum lot size
        dwelling_size = np.maximum(dwelling_size, param["mini_lot_size"])
        # And we make sure we do not consider cases where some income groups
        # would have no access to formal housing
        dwelling_size[income_class_by_housing_type.formal == 0, :] = np.nan

    elif housing_type == 'backyard':

        # Defined exogenously
        dwelling_size = param["shack_size"] * np.ones((4, len(grid.dist)))
        # As before
        dwelling_size[income_class_by_housing_type.backyard == 0, :] = np.nan

    elif housing_type == 'informal':

        # Defined exogenously
        dwelling_size = param["shack_size"] * np.ones((4, len(grid.dist)))
        # As before
        dwelling_size[income_class_by_housing_type.settlement == 0, :] = np.nan

    # %% Bid-rent functions in selected pixels per (endogenous) housing type

    if housing_type == 'formal':

        # See technical documentation for math formula
        R_mat = (param["beta"] * (income_net_of_commuting_costs)
                 / (dwelling_size - (param["alpha"] * param["q0"])))
        R_mat[income_net_of_commuting_costs < 0] = 0
        R_mat[income_class_by_housing_type.formal == 0, :] = 0
        
        # FOR UTILITY DECOMPOSITION: save individual variables?
        # utility_temp = (
        #     (income_net_of_commuting_costs - dwelling_size*R_mat)**param["alpha"]
        #     * (dwelling_size - param["q0"])**param["beta"]
        #     * amenities[None, :]
        #     )


    elif housing_type == 'backyard':

        # See technical documentation for math formula

        if options["actual_backyards"] == 1:
            R_mat = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs
                   - ((1 + np.array(
                       fraction_capital_destroyed.contents_backyard)[None, :]
                       * param["fraction_z_dwellings"])
                       * ((utility[:, None]
                           / (amenities[None, :]
                              * param_backyards_pockets[None, :]
                              * ((dwelling_size - param["q0"])
                                 ** param["beta"])))
                          ** (1 / param["alpha"])))
                   # - (param["informal_structure_value"]
                   #    * (interest_rate + param["depreciation_rate"]))
                   # - (np.array(
                   #     fraction_capital_destroyed.structure_backyards
                   #     )[None, :] * param["informal_structure_value"])
                   )
                )
            R_mat_nodisam = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs
                   - ((1 + np.array(
                       fraction_capital_destroyed.contents_backyard)[None, :]
                       * param["fraction_z_dwellings"])
                       * ((utility[:, None]
                           / (amenities[None, :]
                              # * param_backyards_pockets[None, :]**param["disam_reduc_fact"]
                              * param_incremental_pockets[None, :]
                              * ((dwelling_size - param["q0"])
                                 ** param["beta"])))
                          ** (1 / param["alpha"])))
                   # - (param["informal_structure_value"]
                   #    * (interest_rate + param["depreciation_rate"]))
                   # - (np.array(
                   #     fraction_capital_destroyed.structure_backyards
                   #     )[None, :] * param["informal_structure_value"])
                   )
                )




        elif options["actual_backyards"] == 0:
            R_mat = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs
                    - ((1 + np.array(
                        fraction_capital_destroyed.contents_backyard)[None, :]
                        * param["fraction_z_dwellings"])
                        * ((utility[:, None]
                            / (amenities[None, :]
                               * param_backyards_pockets[None, :]
                               * ((dwelling_size - param["q0"])
                                  ** param["beta"])))
                           ** (1 / param["alpha"])))
                    # - (param["informal_structure_value"]
                    #    * (interest_rate + param["depreciation_rate"]))
                    # - (np.array(
                    #     fraction_capital_destroyed.structure_informal_backyards
                    #     )[None, :] * param["informal_structure_value"])
                    )
                )
            R_mat_nodisam = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs
                    - ((1 + np.array(
                        fraction_capital_destroyed.contents_backyard)[None, :]
                        * param["fraction_z_dwellings"])
                        * ((utility[:, None]
                            / (amenities[None, :]
                               # * param_backyards_pockets[None, :]**param["disam_reduc_fact"]
                               * param_incremental_pockets[None, :]
                               * ((dwelling_size - param["q0"])
                                  ** param["beta"])))
                           ** (1 / param["alpha"])))
                    # - (param["informal_structure_value"]
                    #    * (interest_rate + param["depreciation_rate"]))
                    # - (np.array(
                    #     fraction_capital_destroyed.structure_informal_backyards
                    #     )[None, :] * param["informal_structure_value"])
                    )
                )

        R_mat[income_class_by_housing_type.backyard == 0, :] = 0
        R_mat_nodisam[income_class_by_housing_type.backyard == 0, :] = 0
        
        # We clean the results just in case
        R_mat_nodisam[R_mat_nodisam < 0] = 0
        R_mat_nodisam[np.isnan(R_mat_nodisam)] = 0
    
        # We select highest bidder (income group) in each location
        proba_nodisam = (R_mat_nodisam == np.nanmax(R_mat_nodisam, 0))
        # We correct the matrix if binding budget constraint
        # (and other precautions)
        limit_nodisam = ((income_net_of_commuting_costs > 0)
                 & (proba_nodisam > 0)
                 & (~np.isnan(income_net_of_commuting_costs))
                 & (R_mat_nodisam > 0))
        proba_nodisam = proba_nodisam * limit_nodisam
    
        # Yields directly the selected income group for each location
        which_group_nodisam = np.nanargmax(R_mat_nodisam, 0)
    
        # Then we recover rent and dwelling size associated with the selected
        # income group in each location
        R_nodisam = np.empty(len(which_group_nodisam))
        R_nodisam[:] = np.nan
        # dwelling_size_temp_nodisam = np.empty(len(which_group_nodisam))
        # dwelling_size_temp_nodisam[:] = np.nan
        for i in range(0, len(which_group_nodisam)):
            R_nodisam[i] = R_mat_nodisam[int(which_group_nodisam[i]), i]
        #     dwelling_size_temp_nodisam[i] = dwelling_size[int(which_group_nodisam[i]), i]
    
        # dwelling_size_nodisam = dwelling_size_temp_nodisam

    elif housing_type == 'informal':

        # See technical documentation for math formula

        R_mat = (
            (1 / param["shack_size"])
            * (income_net_of_commuting_costs
                - ((1 + np.array(fraction_capital_destroyed.contents_informal)[
                    None, :] * param["fraction_z_dwellings"])
                    * ((utility[:, None] / (amenities[None, :]
                                            * param_pockets[None, :]
                                            * ((dwelling_size - param["q0"])
                                               ** param["beta"])))
                       ** (1 / param["alpha"])))
                - (param["informal_structure_value"]
                   * (interest_rate + param["depreciation_rate"]))
                - (np.array(
                    fraction_capital_destroyed.structure_informal_settlements
                    )[None, :] * param["informal_structure_value"]))
            )

        R_mat[income_class_by_housing_type.settlement == 0, :] = 0

    # We clean the results just in case
    R_mat[R_mat < 0] = 0
    R_mat[np.isnan(R_mat)] = 0

    # We select highest bidder (income group) in each location
    proba = (R_mat == np.nanmax(R_mat, 0))
    # We correct the matrix if binding budget constraint
    # (and other precautions)
    limit = ((income_net_of_commuting_costs > 0)
             & (proba > 0)
             & (~np.isnan(income_net_of_commuting_costs))
             & (R_mat > 0))
    proba = proba * limit

    # Yields directly the selected income group for each location
    which_group = np.nanargmax(R_mat, 0)

    # Then we recover rent and dwelling size associated with the selected
    # income group in each location
    R = np.empty(len(which_group))
    R[:] = np.nan
    dwelling_size_temp = np.empty(len(which_group))
    dwelling_size_temp[:] = np.nan
    for i in range(0, len(which_group)):
        R[i] = R_mat[int(which_group[i]), i]
        dwelling_size_temp[i] = dwelling_size[int(which_group[i]), i]

    dwelling_size = dwelling_size_temp

    # %% Housing supply (per unit of available land)

    if housing_type == 'formal':
        housing_supply = eqsol.compute_housing_supply_formal(
            R, options, housing_limit, param, agricultural_rent, interest_rate,
            fraction_capital_destroyed, minimum_housing_supply,
            construction_param, housing_in, dwelling_size)
        housing_supply[R == 0] = 0
    elif housing_type == 'backyard':
        (housing_supply, R) = eqsol.compute_housing_supply_backyard(
            R, R_nodisam, param, income_net_of_commuting_costs,
            fraction_capital_destroyed, grid, income_class_by_housing_type,
            options, interest_rate)
        housing_supply[R == 0] = 0

        # DOUBLE CHECK (not sure if needed)
        R_mat[:, housing_supply == 2000000] = R_mat_nodisam[:, housing_supply == 2000000]
        # We select highest bidder (income group) in each location
        proba = (R_mat == np.nanmax(R_mat, 0))
        # We correct the matrix if binding budget constraint
        # (and other precautions)
        limit = ((income_net_of_commuting_costs > 0)
                 & (proba > 0)
                 & (~np.isnan(income_net_of_commuting_costs))
                 & (R_mat > 0))
        proba = proba * limit

    elif housing_type == 'informal':
        # We simply take a supply equal to the available constructible land,
        # hence ones when considering supply per land unit (informal
        # settlements are assumed not costly to build), then convert to m²
        housing_supply = 1000000 * np.ones(len(which_group))
        housing_supply[R == 0] = 0

    # %% Outputs

    # Yields population density in each selected pixel
    people_init = housing_supply / dwelling_size * (np.nansum(limit, 0) > 0)
    people_init[np.isnan(people_init)] = 0
    # Yields number of people per pixel, as 0.25 is the area of a pixel
    # (0.5*0.5 km) and coeff_land reduces it to inhabitable area
    people_init_land = people_init * coeff_land * 0.25

    # UPDATE PROBA?
    # We associate people in each selected pixel to the highest bidding income
    # group
    people_center = np.array(people_init_land)[None, :] * proba
    people_center[np.isnan(people_center)] = 0
    # Then we sum across pixels and get the number of people in each income
    # group for given housing type
    job_simul = np.nansum(people_center, 1)

    # We also put a floor equal to the agricultural rent for rents in the
    # formal private sector
    if housing_type == 'formal':
        R = np.maximum(R, agricultural_rent)

    return (job_simul, R, people_init, people_center, housing_supply,
            dwelling_size, R_mat)
