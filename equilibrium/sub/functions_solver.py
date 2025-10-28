# -*- coding: utf-8 -*-
"""



"""

import numpy as np
import copy
from scipy.interpolate import interp1d


def compute_dwelling_size_formal(utility, amenities, param,
                                 income_net_of_commuting_costs,
                                 fraction_capital_destroyed):
    """
    Return optimal dwelling size per income group for formal housing.

    This function leverages the explicit_qfunc() function to express
    dwelling size as an implicit function of observed values, coming from
    optimality conditions.

    Parameters
    ----------
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

    Returns
    -------
    dwelling_size : ndarray(float64, ndim=2)
        Simulated average dwelling size (in m²) for each selected pixel (4,043)
        and each income group (4)

    """
    # We reprocess income net of commuting costs not to break down equations
    # with negative values
    income_temp = copy.deepcopy(income_net_of_commuting_costs)
    income_temp[income_temp < 0] = np.nan

    # We are going to express dwelling size as an implicit function (coming
    # from optimality conditions) of observed variables. The corresponding
    # explicit function is given in explicit_qfunc(q, q_0, alpha), and the
    # observed part (corresponding to the left side of equation given in
    # technical documentation) is given below:
    left_side = (
        (np.array(utility)[:, None] / np.array(amenities)[None, :])
        * ((1 + (param["fraction_z_dwellings"]
                 * np.array(fraction_capital_destroyed.contents_formal)[
                     None, :])) ** (param["alpha"]))
        / ((param["alpha"] * income_temp) ** param["alpha"])
        )

    # We get a regression spline expressing dwelling size as an implicit
    # function of explicit_qfunc(q, q_0, alpha) for some arbitrarily chosen q
    # defined below:
    x = np.concatenate((
        [10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4),
         10 ** (-3), 10 ** (-2), 10 ** (-1)],
        np.arange(0.11, 0.15, 0.01),
        np.arange(0.15, 1.15, 0.05),
        np.arange(1.2, 3.1, 0.1),
        np.arange(3.5, 13.1, 0.25),
        np.arange(15, 60, 0.5),
        np.arange(60, 100, 2.5),
        np.arange(110, 210, 10),
        [250, 300, 500, 1000, 2000, 200000, 1000000, 10 ** 12]))

    f = interp1d(explicit_qfunc(x, param["q0"], param["alpha"]), x)

    # We define dwelling size as the image corresponding to observed values
    # from left_side, for each selected pixel and each income group
    dwelling_size = f(left_side)

    # We cap dwelling size to 10**12 (to avoid numerical difficulties with
    # infinite numbers)
    dwelling_size[dwelling_size > np.nanmax(x)] = np.nanmax(x)

    return dwelling_size


def explicit_qfunc(q, q_0, alpha):
    """
    Explicit function that will be inverted to recover optimal dwelling size.

    This function is used as part of compute_dwelling_size_formal().

    Parameters
    ----------
    q : ndarray(float64)
        Arbitrary values for dwelling size (in m²)
    q_0 : ndarray(float64)
        Parametric basic need in housing (in m²)
    alpha : float64
        (Calibrated) composite good elasticity in households' utility function

    Returns
    -------
    result : ndarray(float64)
        Theoretical values associated with observed variable left_side (see
        compute_dwelling_size_formal function) through optimality conditions,
        for arbitrary values of dwelling size

    """
    # Note that with above x definition, q-alpha*q_0 can be negative

    # Note that numpy returns null when trying to get the fractional power of a
    # negative number (which is fine, because we are not interested in such
    # values), hence we ignore the error
    np.seterr(divide='ignore', invalid='ignore')
    result = (
        (q - q_0)
        / ((q - (alpha * q_0)) ** alpha)
        )

    return result


def compute_housing_supply_formal(
        R, options, housing_limit, param, agricultural_rent, interest_rate,
        fraction_capital_destroyed, minimum_housing_supply, construction_param,
        housing_in, dwelling_size
        ):
    """
    Return optimal housing supply for formal private housing.

    This function leverages optimality conditions function to express
    housing supply as a function of rents.

    Parameters
    ----------
    R : ndarray(float64)
        Simulated average annual rent (in rands/m²) for a given housing type,
        for each selected pixel (4,043)
    options : dict
        Dictionary of default options
    housing_limit : Series
        Maximum housing supply (in m² per km²) in each grid cell (24,014)
    param : dict
        Dictionary of default parameters
    agricultural_rent : float64
        Annual housing rent below which it is not profitable for formal private
        developers to urbanize (agricultural) land: endogenously limits urban
        sprawl
    interest_rate : float64
        Real interest rate for the overall economy, corresponding to an average
        over past years
    fraction_capital_destroyed : DataFrame
        Data frame of expected fractions of capital destroyed, for housing
        structures and contents in different housing types, in each
        grid cell (24,014)
    minimum_housing_supply : ndarray(float64)
        Minimum housing supply (in m²) for each grid cell (24,014), allowing
        for an ad hoc correction of low values in Mitchells Plain
    construction_param : ndarray(float64)
        (Calibrated) scale factor for the construction function of formal
        private developers
    housing_in : ndarray(float64)
        Theoretical minimum housing supply when formal private developers do
        not adjust (not used in practice), per grid cell (24,014)
    dwelling_size : ndarray(float64)
        Simulated average dwelling size (in m²) for a given housing type, for
        each selected pixel (4,043)

    Returns
    -------
    housing_supply : ndarray(float64)
        Simulated housing supply per unit of available land (in m² per km²)
        for formal private housing, for each selected pixel (4,043)

    """
    if options["adjust_housing_supply"] == 1:
        # We consider two different damage functions above and below some
        # exogenous dwelling size threshold (proxies for the existence of
        # a second floor)
        capital_destroyed = np.ones(
            len(fraction_capital_destroyed.structure_formal_2))
        (capital_destroyed[dwelling_size > param["threshold"]]
         ) = fraction_capital_destroyed.structure_formal_2[
             dwelling_size > param["threshold"]
             ]
        (capital_destroyed[dwelling_size <= param["threshold"]]
         ) = fraction_capital_destroyed.structure_formal_1[
             dwelling_size <= param["threshold"]
             ]

        # See technical documentation for math formulas
        # NB: we convert values to supply in m² per km² of available land
        housing_supply = (
            1000000
            * (construction_param ** (1/param["coeff_a"]))
            * ((param["coeff_b"]
                / (interest_rate + param["depreciation_rate"]
                   + capital_destroyed))
               ** (param["coeff_b"]/param["coeff_a"]))
            * ((R) ** (param["coeff_b"]/param["coeff_a"]))
            )

        # Below the agricultural rent, no housing is built
        housing_supply[R < agricultural_rent] = 0

        housing_supply[np.isnan(housing_supply)] = 0
        housing_supply[housing_supply < 0] = 0
        housing_supply = np.minimum(housing_supply, housing_limit)

        # We also correct for a potential ad hoc minimum housing supply in
        # Mitchells_Plain
        housing_supply = np.maximum(
            housing_supply, minimum_housing_supply * 1000000)

    # Note that housing supply is just equal to a floor value when developers
    # do not adjust. In practice, this is only used in simulations for
    # subsequent years, and this value is set to the housing supply obtained
    # for the period before. We could, in theory, simulate an initial state
    # where developers do not adjust, although this makes no practical sense.
    else:
        housing_supply = housing_in

    return housing_supply


def compute_housing_supply_backyard(R, R_nodisam, param, income_net_of_commuting_costs,
                                    fraction_capital_destroyed, grid,
                                    income_class_by_housing_type, options, interest_rate):
    """
    Return optimal housing supply for informal backyards.

    This function leverages optimality conditions function to express
    housing supply as a function of rents.

    Parameters
    ----------
    R : ndarray(float64)
        Simulated average annual rent (in rands/m²) for a given housing type,
        for each selected pixel (4,043)
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

    Returns
    -------
    housing_supply : ndarray(float64)
        Simulated housing supply per unit of available land (in m² per km²)
        for informal backyards, for each selected pixel (4,043)

    """
    # Same as before
    capital_destroyed = np.ones(
        len(fraction_capital_destroyed.structure_formal_2))
    dwelling_size = param["RDP_size"] * np.ones((len(grid.dist)))
    # We consider two different damage functions above and below some
    # exogenous dwelling size threshold (proxies for the existence of
    # a second floor)
    # NB: in practice, as the size of a backyard "shack" is parametrically
    # fixed, this will always be considered as one floor
    capital_destroyed[dwelling_size > param["threshold"]
                      ] = fraction_capital_destroyed.structure_subsidized_2[
                          dwelling_size > param["threshold"]]
    capital_destroyed[dwelling_size <= param["threshold"]
                      ] = fraction_capital_destroyed.structure_subsidized_1[
                          dwelling_size <= param["threshold"]]

    # New computation due to structure costs!
               
    Z_IB = ((param["depreciation_rate"] + interest_rate)
            * (param["informal_structure_value"]/param["shack_size"]))
            
    mu = ((param["alpha"] *
         (param["RDP_size"] + param["backyard_size"] - param["q0"])
         / (param["backyard_size"]))
        - (param["beta"]
           * (income_net_of_commuting_costs[0, :]
              - (capital_destroyed * param["subsidized_structure_value"])
              - (param["depreciation_rate"] * param["subsidized_structure_value"]))
           / (param["backyard_size"] * (R-Z_IB)))
        )
    
    housing_supply = mu
    
    housing_supply = np.minimum(housing_supply, 1)
    housing_supply = np.maximum(housing_supply, 0)
    
    if options["incremental_housing"]==1:
    
        Z_IH = ((param["depreciation_rate"] + interest_rate)
                * (param["subsidized_structure_value"]/param["RDP_size"]))
        
        z_IB = (income_net_of_commuting_costs[0, :]
                - (param["depreciation_rate"] * param["subsidized_structure_value"])
                + (mu * param["backyard_size"] * (R-Z_IB)))
        z_IH = (income_net_of_commuting_costs[0, :]
                - (param["depreciation_rate"] * param["subsidized_structure_value"])
                + (2 * param["backyard_size"] * (R_nodisam-Z_IH)))
        
        U_IB = (z_IB**param["alpha"]
                * (param["RDP_size"] + (1-mu)*param["backyard_size"] - param["q0"])**param["beta"])
        U_IH = (z_IH**param["alpha"]
                * (param["RDP_size"] - param["q0"])**param["beta"])
        
        housing_supply = np.minimum(housing_supply, 1)
        housing_supply = np.maximum(housing_supply, 0)
        
        housing_supply[U_IH>U_IB] = 2
        R[U_IH>U_IB] = R_nodisam[U_IH>U_IB]
    
    # See technical documentation for math formulas
    # housing_supply = (
    #     (param["alpha"] *
    #      (param["RDP_size"] + param["backyard_size"] - param["q0"])
    #      / (param["backyard_size"]))
    #     - (param["beta"]
    #        * (income_net_of_commuting_costs[0, :]
    #           - (capital_destroyed * param["subsidized_structure_value"]))
    #        / (param["backyard_size"] * R))
    # )

    housing_supply[R == 0] = 0
    housing_supply = 1000000 * housing_supply

    return (housing_supply, R)
