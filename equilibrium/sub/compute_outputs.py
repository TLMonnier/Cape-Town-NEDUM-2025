# -*- coding: utf-8 -*-

import numpy as np
import equilibrium.sub.functions_solver as eqsol


def compute_outputs(housing_type,
                    utility,
                    amenities,
                    param,
                    income_net_of_commuting_costs,
                    fraction_capital_destroyed,
                    fraction_capital_destroyed_protec,
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
                    param_incremental_pockets,
                    damages_table, damages_protec_table, interval_table_fathom):
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

    mask_self_protec = np.zeros(len(amenities))
    mask_self_protec[:] = np.nan

    if housing_type == 'formal':

        dwelling_size = eqsol.compute_dwelling_size_formal(
            utility, amenities, param, options, income_net_of_commuting_costs,
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
            if options["risk_misperc"] == 0:
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
            elif options["risk_misperc"] == 1:
                R_mat = (
                    (1 / param["shack_size"])
                    * (income_net_of_commuting_costs
                    - ((1 + np.array(
                        param["risk_internaliz"]*fraction_capital_destroyed.contents_backyard)[None, :]
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
                    #     param["risk_internaliz"]*fraction_capital_destroyed.structure_backyards
                    #     )[None, :] * param["informal_structure_value"])
                    )
                    )
                R_mat_nodisam = (
                    (1 / param["shack_size"])
                    * (income_net_of_commuting_costs
                    - ((1 + np.array(
                        param["risk_internaliz"]*fraction_capital_destroyed.contents_backyard)[None, :]
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
                    #     param["risk_internaliz"]*fraction_capital_destroyed.structure_backyards
                    #     )[None, :] * param["informal_structure_value"])
                    )
                    )

        elif options["actual_backyards"] == 0:
            if options["risk_misperc"] == 0:
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
            elif options["risk_misperc"] == 1:
                R_mat = (
                    (1 / param["shack_size"])
                    * (income_net_of_commuting_costs
                        - ((1 + np.array(
                            param["risk_internaliz"]*fraction_capital_destroyed.contents_backyard)[None, :]
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
                        #     param["risk_internaliz"]*fraction_capital_destroyed.structure_informal_backyards
                        #     )[None, :] * param["informal_structure_value"])
                        )
                    )
                R_mat_nodisam = (
                    (1 / param["shack_size"])
                    * (income_net_of_commuting_costs
                        - ((1 + np.array(
                            param["risk_internaliz"]*fraction_capital_destroyed.contents_backyard)[None, :]
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
                        #     param["risk_internaliz"]*fraction_capital_destroyed.structure_informal_backyards
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

        # NB: compute alternative rent under protection (just one mode for now)
        # Take care to dimensions and cleaning?

        if options["risk_misperc"] == 0:
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
            R_mat_protec = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs-param["sandbag_course_cost"]
                    - ((1 + np.array(fraction_capital_destroyed_protec.contents_informal)[
                        None, :] * param["fraction_z_dwellings"])
                        * ((utility[:, None] / (amenities[None, :]
                                                * param_pockets[None, :]
                                                * ((dwelling_size - param["q0"])
                                                ** param["beta"])))
                        ** (1 / param["alpha"])))
                    - (param["informal_structure_value"]
                    * (interest_rate + param["depreciation_rate"]))
                    - (np.array(
                        fraction_capital_destroyed_protec.structure_informal_settlements
                        )[None, :] * param["informal_structure_value"]))
                )

            # R_mat[R_mat_protec>R_mat] = R_mat_protec[R_mat_protec>R_mat]

            # NB: we do not define CRRA here??

            # # Take care to dimensions!
            # R_mat_CRRA = np.zeros((11, 4, len(amenities)))
            # R_mat_CRRA[:,:,:] = np.nan
            # R_mat_protec_CRRA = np.zeros((11, 4, len(amenities)))
            # R_mat_protec_CRRA[:,:,:] = np.nan

            # # NB: only in this case are some agents willing to pay more for protection

            # for i in range(11):
                # R_mat_CRRA[i,:,:] = (
                    # (1 / param["shack_size"])
                    # * (income_net_of_commuting_costs
                        # - ((1 + np.array(damages_table["contents_informal"])[
                            # i, None, :] * param["fraction_z_dwellings"])
                            # * ((utility[:, None] / (amenities[None, :]
                                                    # * param_pockets[None, :]
                                                    # * ((dwelling_size - param["q0"])
                                                    # ** param["beta"])))
                            # ** (1 / param["alpha"])))
                        # - (param["informal_structure_value"]
                        # * (interest_rate + param["depreciation_rate"]))
                        # - (np.array(
                            # damages_table["structure_informal_settlements"]
                            # )[i, None, :] * param["informal_structure_value"]))
                    # )

                # R_mat_protec_CRRA[i,:,:] = (
                    # (1 / param["shack_size"])
                    # * (income_net_of_commuting_costs-param["sandbag_course_cost"]
                        # - ((1 + np.array(damages_protec_table["contents_informal"])[
                            # i, None, :] * param["fraction_z_dwellings"])
                            # * ((utility[:, None] / (amenities[None, :]
                                                    # * param_pockets[None, :]
                                                    # * ((dwelling_size - param["q0"])
                                                    # ** param["beta"])))
                            # ** (1 / param["alpha"])))
                        # - (param["informal_structure_value"]
                        # * (interest_rate + param["depreciation_rate"]))
                        # - (np.array(
                            # damages_protec_table["structure_informal_settlements"]
                            # )[i, None, :] * param["informal_structure_value"]))
                    # )

        elif options["risk_misperc"] == 1:
            R_mat = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs
                    - ((1 + param["risk_internaliz"]*np.array(fraction_capital_destroyed.contents_informal)[
                        None, :] * param["fraction_z_dwellings"])
                        * ((utility[:, None] / (amenities[None, :]
                                                * param_pockets[None, :]
                                                * ((dwelling_size - param["q0"])
                                                ** param["beta"])))
                        ** (1 / param["alpha"])))
                    - (param["informal_structure_value"]
                    * (interest_rate + param["depreciation_rate"]))
                    - (param["risk_internaliz"]*np.array(
                        fraction_capital_destroyed.structure_informal_settlements
                        )[None, :] * param["informal_structure_value"]))
                )
            R_mat_protec = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs-param["sandbag_course_cost"]
                    - ((1 + param["risk_internaliz"]*np.array(fraction_capital_destroyed_protec.contents_informal)[
                        None, :] * param["fraction_z_dwellings"])
                        * ((utility[:, None] / (amenities[None, :]
                                                * param_pockets[None, :]
                                                * ((dwelling_size - param["q0"])
                                                ** param["beta"])))
                        ** (1 / param["alpha"])))
                    - (param["informal_structure_value"]
                    * (interest_rate + param["depreciation_rate"]))
                    - (param["risk_internaliz"]*np.array(
                        fraction_capital_destroyed_protec.structure_informal_settlements
                        )[None, :] * param["informal_structure_value"]))
                )

            # # Take care to dimensions!
            # R_mat_CRRA = np.zeros((11, 4, len(amenities)))
            # R_mat_CRRA[:,:,:] = np.nan
            # R_mat_protec_CRRA = np.zeros((11, 4, len(amenities)))
            # R_mat_protec_CRRA[:,:,:] = np.nan

            # for i in range(11):
                # R_mat_CRRA[i,:,:] = (
                    # (1 / param["shack_size"])
                    # * (income_net_of_commuting_costs
                        # - ((1 + param["risk_internaliz"]*np.array(damages_table["contents_informal"])[
                            # i, None, :] * param["fraction_z_dwellings"])
                            # * ((utility[:, None] / (amenities[None, :]
                                                    # * param_pockets[None, :]
                                                    # * ((dwelling_size - param["q0"])
                                                    # ** param["beta"])))
                            # ** (1 / param["alpha"])))
                        # - (param["informal_structure_value"]
                        # * (interest_rate + param["depreciation_rate"]))
                        # - (param["risk_internaliz"]*np.array(
                            # damages_table["structure_informal_settlements"]
                            # )[i, None, :] * param["informal_structure_value"]))
                    # )

                # R_mat_protec_CRRA[i,:,:] = (
                    # (1 / param["shack_size"])
                    # * (income_net_of_commuting_costs-param["sandbag_course_cost"]
                        # - ((1 + param["risk_internaliz"]*np.array(damages_protec_table["contents_informal"])[
                            # i, None, :] * param["fraction_z_dwellings"])
                            # * ((utility[:, None] / (amenities[None, :]
                                                    # * param_pockets[None, :]
                                                    # * ((dwelling_size - param["q0"])
                                                    # ** param["beta"])))
                            # ** (1 / param["alpha"])))
                        # - (param["informal_structure_value"]
                        # * (interest_rate + param["depreciation_rate"]))
                        # - (param["risk_internaliz"]*np.array(
                            # damages_protec_table["structure_informal_settlements"]
                            # )[i, None, :] * param["informal_structure_value"]))
                    # )

        R_mat[income_class_by_housing_type.settlement == 0, :] = 0
        R_mat_protec[income_class_by_housing_type.settlement == 0, :] = 0

        # R_mat_CRRA[:, income_class_by_housing_type.settlement == 0, :] = 0
        # R_mat_protec_CRRA[:, income_class_by_housing_type.settlement == 0, :] = 0

        # We clean the results before comparing (as done below)
        R_mat[R_mat < 0] = 0
        R_mat[np.isnan(R_mat)] = 0

        # We clean the results just in case
        R_mat_protec[R_mat_protec < 0] = 0
        R_mat_protec[np.isnan(R_mat_protec)] = 0

        # R_mat_CRRA[R_mat_CRRA < 0] = 0
        # R_mat_CRRA[np.isnan(R_mat_CRRA)] = 0

        # R_mat_protec_CRRA[R_mat_protec_CRRA < 0] = 0
        # R_mat_protec_CRRA[np.isnan(R_mat_protec_CRRA)] = 0
    
        # # We select highest bidder (income group) in each location
        # proba_protec = (R_mat_protec == np.nanmax(R_mat_protec, 0))
        # # We correct the matrix if binding budget constraint
        # # (and other precautions)
        # limit_protec = ((income_net_of_commuting_costs > 0)
        #          & (proba_protec > 0)
        #          & (~np.isnan(income_net_of_commuting_costs))
        #          & (R_mat_protec > 0))
        # proba_protec = proba_protec * limit_protec
    
        # Need to mark protection choice?
        if options["self_protec"]==1:

            # self_protec_mat = R_mat_protec>R_mat
            # # What is several groups would like to protect?
            # self_protec_mat = np.nansum(self_protec_mat, 0)
            # #Good enough?
            # which_group_mat = np.zeros(len(self_protec_mat))
            # which_group_mat[:] = np.nan
            # which_group_mat = np.nanargmax(self_protec_mat, 0)

            # R_mat_orig = np.copy(R_mat)
            # R_max_orig = np.nanmax(R_mat_orig, 0)
            # R_mat[R_mat_protec>R_mat] = R_mat_protec[R_mat_protec>R_mat]
            # R_max = np.nanmax(R_mat, 0)
            # mask_self_protec = (R_max>R_max_orig)

            # Protection is chosen wherever it raises the bid rent of an
            # income group: the surplus from protection is capitalized into
            # rents, as utility is pinned down at equilibrium level

            if options["risk_avers"] == 0:
                protec_mat = (R_mat_protec > R_mat)

            elif options["risk_avers"] == 1:
                # Same comparison on risk-averse bid rents, that are not
                # state-contingent: rents are paid before flood realization
                if options["risk_misperc"] == 0:
                    perceived_share = 1
                elif options["risk_misperc"] == 1:
                    perceived_share = param["risk_internaliz"]

                # Composite good consumption that yields target utility
                z_target = ((utility[:, None]
                             / (amenities[None, :] * param_pockets[None, :]
                                * ((dwelling_size - param["q0"])
                                   ** param["beta"])))
                            ** (1 / param["alpha"]))

                R_mat_CRRA = compute_bid_rent_informal_CRRA(
                    income_net_of_commuting_costs, z_target, 0,
                    perceived_share * np.array(
                        damages_table["contents_informal"]),
                    perceived_share * np.array(
                        damages_table["structure_informal_settlements"]),
                    interval_table_fathom, param, interest_rate)
                R_mat_protec_CRRA = compute_bid_rent_informal_CRRA(
                    income_net_of_commuting_costs, z_target,
                    param["sandbag_course_cost"],
                    perceived_share * np.array(
                        damages_protec_table["contents_informal"]),
                    perceived_share * np.array(
                        damages_protec_table["structure_informal_settlements"]),
                    interval_table_fathom, param, interest_rate)

                R_mat_CRRA[income_class_by_housing_type.settlement == 0, :] = 0
                R_mat_protec_CRRA[
                    income_class_by_housing_type.settlement == 0, :] = 0
                R_mat_CRRA[R_mat_CRRA < 0] = 0
                R_mat_CRRA[np.isnan(R_mat_CRRA)] = 0
                R_mat_protec_CRRA[R_mat_protec_CRRA < 0] = 0
                R_mat_protec_CRRA[np.isnan(R_mat_protec_CRRA)] = 0

                protec_mat = (R_mat_protec_CRRA > R_mat_CRRA)

            # R_mat_orig = np.copy(R_mat)
            # R_max_orig = np.nanmax(R_mat_orig, 0)
            # R_mat[protec_mat] = R_mat_protec[protec_mat]
            # R_max = np.nanmax(R_mat, 0)
            # mask_self_protec = (R_max>R_max_orig)

            # Equilibrium rents remain the (risk-neutral) expected-damage
            # ones: CRRA only drives the protection choice
            R_mat[protec_mat] = R_mat_protec[protec_mat]

            # Protection status of the highest bidder in each location
            # Really?
            which_group_protec = np.nanargmax(R_mat, 0)
            pixels = np.arange(R_mat.shape[1])
            mask_self_protec = (protec_mat[which_group_protec, pixels]
                                & (R_mat[which_group_protec, pixels] > 0))

        # FOR CRRA?

        # Yields directly the selected income group for each location
        # which_group = np.nanargmax(R_mat, 0)

        # which_group_CRRA = np.zeros((11,len(amenities)))
        # which_group_CRRA[:,:] = np.nan
        # which_group_protec_CRRA = np.zeros((11,len(amenities)))
        # which_group_protec_CRRA[:,:] = np.nan
        # for i in range(11):
            # which_group_CRRA[i,:] = np.nanargmax(R_mat_CRRA[i,:,:], 0)
            # which_group_protec_CRRA[i,:] = np.nanargmax(R_mat_protec_CRRA[i,:,:], 0)

        # # WE STICK TO BASELINE INCOME SORTING FOR PROTECTION COMPARISONS?
        # # BUT IS IT STILL DOMINANT IF CHOOSING TO PROTECT?

        # # Then we recover rent and dwelling size associated with the selected
        # # income group in each location
        # # R_protec = np.empty(len(which_group))
        # # R_protec[:] = np.nan
        # # # dwelling_size_temp_protec = np.empty(len(which_group_protec))
        # # # dwelling_size_temp_protec[:] = np.nan
        # # for i in range(0, len(which_group)):
        # #     R_protec[i] = R_mat_protec[int(which_group[i]), i]
        # # #   R_protec[i] = R_mat_protec[int(which_group_protec[i]), i]
        # # #     dwelling_size_temp_protec[i] = dwelling_size[int(which_group_protec[i]), i]
    
        # # R_CRRA[i,:] = np.empty(11, (len(which_group_CRRA[i,:])))
        # # R_CRRA[i,:] = np.nan
        # R_CRRA = np.zeros((11, len(which_group)))
        # R_CRRA[:,:] = np.nan
        # for i in range(11):
            # for j in range(0, len(which_group_CRRA[i,:])):
                # R_CRRA[i,j] = R_mat_CRRA[i, int(which_group_CRRA[i,j]), j]

        # R_protec_CRRA = np.zeros((11, len(which_group)))
        # R_protec_CRRA[:,:] = np.nan
        # for i in range(11):
            # for j in range(0, len(which_group_protec_CRRA[i,:])):
                # R_protec_CRRA[i,j] = R_mat_protec_CRRA[i, int(which_group_CRRA[i,j]), j]
                # # R_protec_CRRA[i,j] = R_mat_protec_CRRA[i, int(which_group_protec_CRRA[i,j]), j]

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

    #NEED TO REDEFINE EX POST?

    # Yields directly the selected income group for each location
    # When not already defined in informal
    which_group = np.nanargmax(R_mat, 0)

    # PROTECTION SHOULD NOT AFFECT INCOME SORTING!
    # FOR NOW, NEED TO CHECK EX POST

    # Then we recover rent and dwelling size associated with the selected
    # income group in each location 

    # Should be equal to R_max?
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

        # if options["self_protec"]==1:

            # # This is before deciding income group and protection choice?
            # # Income group should not change? What about soritng effects ex post?
            # # Only a pb for CRRA? Still need to check in final version

            # # dom_net_inc = np.empty(len(which_group))
            # # dom_net_inc[:] = np.nan
            # # for i in range(0, len(which_group)):
            # #     dom_net_inc[i] = income_net_of_commuting_costs[int(which_group[i]), i]

            # # dom_net_inc_protec = np.empty(len(which_group_protec))
            # # dom_net_inc_protec[:] = np.nan
            # # for i in range(0, len(which_group_protec)):
            # #     dom_net_inc_protec[i] = income_net_of_commuting_costs[int(which_group_protec[i]), i]

            # # if options["risk_avers"]==0:

            # #     # How to implement protection choice?
            # #     # Does everyone take it?

            # #     # What about risk misperception??
            # #     if options["risk_misperc"] == 0:
            # #         z = ((dom_net_inc
            # #             - (dwelling_size*R
            # #                 + param["informal_structure_value"]
            # #                 * (interest_rate + param["depreciation_rate"] + fraction_capital_destroyed.structure_informal_settlements)
            # #                 ))/(1+param["fraction_z_dwellings"]*fraction_capital_destroyed.contents_informal)
            # #                 )

            # #         z_protec = ((dom_net_inc_protec-param["sandbag_course_cost"]
            # #             - (dwelling_size*R_protec
            # #                 + param["informal_structure_value"]
            # #                 * (interest_rate + param["depreciation_rate"] + fraction_capital_destroyed_protec.structure_informal_settlements)
            # #                 ))/(1+param["fraction_z_dwellings"]*fraction_capital_destroyed_protec.contents_informal)
            # #                 )
            # #     elif options["risk_misperc"] == 1:
            # #         z = ((dom_net_inc
            # #             - (dwelling_size*R
            # #                 + param["informal_structure_value"]
            # #                 * (interest_rate + param["depreciation_rate"] + param["risk_internaliz"]*fraction_capital_destroyed.structure_informal_settlements)
            # #                 ))/(1+param["fraction_z_dwellings"]*param["risk_internaliz"]*fraction_capital_destroyed.contents_informal)
            # #                 )

            # #         z_protec = ((dom_net_inc_protec-param["sandbag_course_cost"]
            # #             - (dwelling_size*R_protec
            # #                 + param["informal_structure_value"]
            # #                 * (interest_rate + param["depreciation_rate"] + param["risk_internaliz"]*fraction_capital_destroyed_protec.structure_informal_settlements)
            # #                 ))/(1+param["fraction_z_dwellings"]*param["risk_internaliz"]*fraction_capital_destroyed_protec.contents_informal)
            # #                 )

            # #     R[z_protec>z] = R_protec[z_protec>z]

            # #     mask_self_protec = (z_protec > z)

            # if options["risk_avers"]==1:

                # # Define new rents and dom_net_inc before completing!!!

                # # z_noflood_noinsur = (
                # #     dom_net_inc
                # #     - (dwelling_size*R + param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))
                # #     )

                # # z_noflood_insur = (
                # #     dom_net_inc-param["sandbag_course_cost"]
                # #     - (dwelling_size*R + param["informal_structure_value"] * (interest_rate + param["depreciation_rate"]))
                # #     )

                # # damages_table, damages_protec_table, interval_table_fathom

                # # dom_net_inc_CRRA[i,j] = income_net_of_commuting_costs[which_group_CRRA[i,j], j]

                # # Here, in a few cases, income group 1 bids more than income group 0 when allowing for protection
                # # But would they decide to take the protection in utility terms??
                # # In fact, the richer income group underbids!!!

                # dom_net_inc_CRRA = np.take_along_axis(
                    # income_net_of_commuting_costs,
                    # np.asarray(which_group_CRRA).astype(int), axis=0)

                # dom_net_inc_protec_CRRA = np.take_along_axis(
                    # income_net_of_commuting_costs,
                    # np.asarray(which_group_protec_CRRA).astype(int), axis=0)

                # # Is strategy still working when dominant income group change?
                # if options["risk_misperc"] == 0:
                    # z_flood_noinsur_list = (
                        # (dom_net_inc_CRRA
                        # - (dwelling_size*R_CRRA
                            # + param["informal_structure_value"]
                            # * (interest_rate + param["depreciation_rate"] + damages_table["structure_informal_settlements"])
                            # ))/(1+param["fraction_z_dwellings"]*damages_table["contents_informal"])
                            # )
                    # z_flood_insur_list = (
                        # (dom_net_inc_protec_CRRA-param["sandbag_course_cost"]
                        # - (dwelling_size*R_protec_CRRA
                            # + param["informal_structure_value"]
                            # * (interest_rate + param["depreciation_rate"] + damages_protec_table["structure_informal_settlements"])
                            # ))/(1+param["fraction_z_dwellings"]*damages_protec_table["contents_informal"])
                            # )

                # elif options["risk_misperc"] == 1:
                    # z_flood_noinsur_list = (
                        # (dom_net_inc_CRRA
                        # - (dwelling_size*R_CRRA
                            # + param["informal_structure_value"]
                            # * (interest_rate + param["depreciation_rate"] + param["risk_internaliz"]*damages_table["structure_informal_settlements"])
                            # ))/(1+param["fraction_z_dwellings"]*param["risk_internaliz"]*damages_table["contents_informal"])
                            # )
                    # z_flood_insur_list = (
                        # (dom_net_inc_protec_CRRA-param["sandbag_course_cost"]
                        # - (dwelling_size*R_protec_CRRA
                            # + param["informal_structure_value"]
                            # * (interest_rate + param["depreciation_rate"] + param["risk_internaliz"]*damages_protec_table["structure_informal_settlements"])
                            # ))/(1+param["fraction_z_dwellings"]*param["risk_internaliz"]*damages_protec_table["contents_informal"])
                            # )

                # #NB: probabilities already sum to one and incorporate the zero (yearly) events

                # compar_noinsur = np.nansum(np.array(interval_table_fathom)[:,None]*z_flood_noinsur_list**(param["alpha"]*(1-param["CRRA"])), axis=0)
                # compar_insur = np.nansum(np.array(interval_table_fathom)[:,None]*z_flood_insur_list**(param["alpha"]*(1-param["CRRA"])), axis=0)

                # # What to do about this rent? Will it be used as such? Need to collapse, but how?
                # # Just take the already defined aggregate rents without CRRA breakdown?
                # # Still need to redefine?

                # # for i in range(11):
                # #     R_CRRA[i,:][compar_insur>compar_noinsur] = R_protec_CRRA[i,:][compar_insur>compar_noinsur]

                # # R[compar_insur>compar_noinsur] = R_protec[compar_insur>compar_noinsur]

                # mask_self_protec = (compar_insur>compar_noinsur)

                # # Need to be before income competition? Pb of definition with dominant income group...
                # R_max_protec = np.nanmax(R_mat_protec, 0)
                # R_max_orig[mask_self_protec] = R_max_protec[mask_self_protec]

                # # R_mat_orig = np.copy(R_mat)
                # # R_max_orig = np.nanmax(R_mat_orig, 0)
                # # R_mat[R_mat_protec>R_mat] = R_mat_protec[R_mat_protec>R_mat]
                # # R_max = np.nanmax(R_mat, 0)
                # # mask_self_protec = (R_max>R_max_orig)

                # # limit?


        if options["self_protec"]==0:
            mask_self_protec = np.zeros(len(which_group))
            mask_self_protec[:] = np.nan

        # With CRRA?

        # No need to compare full utilities: z is enough
        # NB: take care to dimensions

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
            dwelling_size, R_mat, mask_self_protec)

# R_mat_CRRA = compute_bid_rent_informal_CRRA(
#     income_net_of_commuting_costs, z_target, 0,
#     perceived_share * np.array(
#         damages_table["contents_informal"]),
#     perceived_share * np.array(
#         damages_table["structure_informal_settlements"]),
#     interval_table_fathom, param, interest_rate)

def compute_bid_rent_informal_CRRA(income_net_of_commuting_costs, z_target,
                                   protection_cost, damages_contents,
                                   damages_structure, proba_states, param,
                                   interest_rate, n_iter=60):
    """
    Compute risk-averse bid rents in informal settlements.

    The bid rent R is paid before flood realization, hence is not
    state-contingent. It solves, for each income group and grid cell:
        sum_i p_i * z_i(R)**(alpha*(1-CRRA)) = z_target**(alpha*(1-CRRA)),
    where z_i(R) is composite good consumption in flood state i:
        z_i(R) = (y - protection_cost - q*R - s*(r + d) - delta_s_i*s)
                 / (1 + f*delta_c_i)
    and z_target the (certain) consumption that yields the equilibrium
    utility level. Other utility components (amenities, dwelling size) do
    not vary across states and cancel out. The solution lies between the
    lowest and highest state-contingent bid rents, and is found by bisection.

    Parameters
    ----------
    income_net_of_commuting_costs : ndarray(float64, ndim=2)
        Expected annual income net of commuting costs, by income group (4)
        and grid cell
    z_target : ndarray(float64, ndim=2)
        Composite good consumption yielding target utility, by income group
        (4) and grid cell
    protection_cost : float64
        Annual cost of self-protection (0 without protection)
    damages_contents : ndarray(float64, ndim=2)
        (Perceived) fraction of contents destroyed, by flood state (11) and
        grid cell
    damages_structure : ndarray(float64, ndim=2)
        (Perceived) fraction of structures destroyed, by flood state (11) and
        grid cell
    proba_states : list
        Probability of each flood state (11), summing to one
    param : dict
        Dictionary of default parameters
    interest_rate : float64
        Real interest rate for the overall economy
    n_iter : int
        Number of bisection steps

    Returns
    -------
    R_mat_CRRA : ndarray(float64, ndim=2)
        Risk-averse bid rents (in rands/m²), by income group (4) and grid cell

    """
    expo = param["alpha"] * (1 - param["CRRA"])
    if expo == 0:
        raise ValueError("CRRA = 1 (log utility) is not handled")

    q = param["shack_size"]
    s = param["informal_structure_value"]
    proba_states = np.array(proba_states)[:, None, None]

    # Dimensions: flood states x income groups x grid cells
    denom = (1 + param["fraction_z_dwellings"] * damages_contents)[:, None, :]
    net_income = (
        (income_net_of_commuting_costs - protection_cost
         - s * (interest_rate + param["depreciation_rate"]))[None, :, :]
        - (damages_structure * s)[:, None, :])

    # State-contingent bid rents bracket the solution
    R_state = (net_income - denom * z_target[None, :, :]) / q
    R_low = np.min(R_state, 0)
    R_high = np.max(R_state, 0)
    target = z_target ** expo

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for _ in range(n_iter):
            R_mid = (R_low + R_high) / 2
            z = np.maximum((net_income - q * R_mid[None, :, :]) / denom, 0)
            expected = np.sum(proba_states * z ** expo, 0)
            # Rent is too low if expected utility exceeds target: for
            # CRRA > 1, expo < 0 and utility is decreasing in z**expo
            if expo > 0:
                too_low = (expected > target)
            else:
                too_low = (expected < target)
            R_low = np.where(too_low, R_mid, R_low)
            R_high = np.where(too_low, R_high, R_mid)

    return (R_low + R_high) / 2
