# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import copy


def import_options():
    """
    Import default options.

    Import set of numerical values coding for options used in the model.
    We can group them as follows: structural assumptions regarding agents'
    behaviour, assumptions about different land uses, options about flood data
    used, about re-processing input data, about calibration process, about
    math correction relative to the original code, and about scenarios used
    for time-moving exogenous variables.

    Returns
    -------
    options : dict
        Dictionary of default options

    """
    # STRUCTURAL ASSUMPTIONS
    # Dummy for formal private developers adjusting housing supply to demand
    # (used to compute constrained vs. unconstrained equilibria in dynamic
    # simulations)
    options = {"adjust_housing_supply": 1}
    # Dummy for agents taking floods into account in their choices (to compare
    # the no-insurance vs. perfect risk-based insurance scenarios)
    options["agents_anticipate_floods"] = 0

    # LAND USE ASSUMPTIONS
    # Dummy for coding green belt (for urban edge scenarios)
    options["urban_edge"] = 1
    # Dummy for forbidding new informal housing construction (for eviction
    # scenarios)
    options["informal_land_constrained"] = 1

    # FLOOD DATA OPTIONS
    # Dummy for considering pluvial floods on top of fluvial floods
    # NB: FATHOM data is less reliable for pluvial than for fluvial flood risks
    options["pluvial"] = 1
    # Dummy for reducing pluvial risk for (better protected) formal structures
    # (ad hoc correction)
    options["correct_pluvial"] = 1
    # Dummy for working with defended (vs. undefended) fluvial flood maps
    # NB: FATHOM data is less reliable for defended than for undefended flood
    # risks (hence the need for custom flood maps to properly asses public
    # flood protection investments)
    options["defended"] = 0
    # Dummy for taking coastal floods into account (on top of fluvial floods)
    # NB: this comes from another data source (DELTARES) than the one used for
    # pluvial and fluvial flood risks (FATHOM)
    options["coastal"] = 1
    # Digital elevation model to be used with coastal flood data
    # (set to "MERITDEM" or "NASADEM")
    # NB: DEM used in fluvial/pluvial flood data is MERITDEM
    options["dem"] = "MERITDEM"
    # Dummy for taking sea-level rise into account in coastal flood data
    # NB: Deltares provides flood risk values under the IPCC AR5 assessment for
    # the RCP 8.5 climate change scenario, taken at year 2050 (pessimistic
    # projections). Dummy scenarios are used for other flood risks.
    options["climate_change"] = 1
    # Growth rate in flood risk probability for dummy climate change scenarios
    options["risk_increase"] = 2

    # REPROCESSING OPTIONS
    # Default is set at zero to save computing time
    # (data is simply loaded in the model)
    # Convert SAL (Small-Area-Level) data (on housing types) to grid level
    options["convert_sal_data"] = 0
    # Convert SP (Small Place) data on income groups to grid level
    options["convert_sp_data"] = 0
    # Compute income net of commuting costs for each pixel in each income
    # group based upon calibrated incomes for each job center in each income
    # group (for every period)
    options["compute_net_income"] = 0

    # MAIN CALIBRATION OPTIONS
    # Dummy for loading pre-calibrated parameters (from Pfeiffer et al.),
    # as opposed to newly calibrated paramaters
    options["load_precal_param"] = 0
    # Dummy for fitting informal housing disamenity parameter to grid level
    # NB: doing so is a matter of choice. It yields more accurate results on
    # spatial sorting, but also increases the risk of overfitting the model.
    # TAKES TIME
    options["location_based_calib"] = 1

    # TECHNICAL CALIBRATION OPTIONS
    # Dummy for defining dominant income group based on number of people
    # instead of median income at SP level
    options["correct_dominant_incgrp"] = 0
    # Dummy for substracting subsidized housing units to count of formal units
    # at SP level
    options["substract_RDP_from_formal"] = 0
    # Dummy for leaving Mitchells Plain out in calibration and to do ad hoc
    # correction of housing supply in the area
    options["correct_mitchells_plain"] = 0
    # Dummy for including more criteria in sample selection for calibration
    # (compared to Pfeiffer et al.)
    options["correct_selected_density"] = 1
    # Dummy for correcting the formula for the construction function scale
    # factor (compared to original code)
    options["correct_kappa"] = 1
    # Dummy for setting inflation base year at baseline year (2011)
    # (instead of 2012 in original code)
    options["correct_infla_base"] = 1
    # Dummy for taking round trips into account in estimated monetary
    # transport costs (compared to original code)
    options["correct_round_trip"] = 1
    # Dummy for taking unemployment into account in the formula for
    # the number of commuters
    options["correct_eq3"] = 0
    # Resolution of the scanning to be used in calibration
    # (for commuting gravity and utility function parameters)
    # NB: should be set to "rough", "normal", or "fine"
    options["scan_type"] = "fine"
    # Dummy for using GLM (instead of OLS) for the estimation of exogenous
    # amenity parameters (unstable)
    options["glm"] = 0
    # Dummy for using scipy solver to refine utility function parameter
    # estimates from scanning (only relevant when we allow all parameters to
    # vary, not just the amenity score)
    options["param_optim"] = 0
    # Dummy for adding numerical terms that prevent exponential overflow
    # when solving the commuting choice model, without changing results
    options["prevent_exp_overflow"] = 1

    # CODE CORRECTION OPTIONS
    # Dummy for taking formal backyards into account in backyard land use
    # coefficients and structural damages from floods (deprecated)
    options["actual_backyards"] = 0
    # Dummy for allocating no-income population to each income group based on
    # their respective unemployment rates, instead of applying a unique rate
    # (to be used when population data is given for active households only)
    options["unempl_reweight"] = 0
    # Dummy for correcting the formula for agricultural rent
    # (compared to original version of the code)
    options["correct_agri_rent"] = 1
    # Dummy for taking into account capital depreciation as a factor of
    # land price in profit function (has an impact on agricultural rent)
    options["deprec_land"] = 0

    # SCENARIO OPTIONS
    #  Code corresponds to low/medium/high
    options["inc_ineq_scenario"] = 2
    #  Code corresponds to low/medium/high/high_corrected
    options["pop_growth_scenario"] = 4
    #  NB: we do not add and option for interest rate: expected future value
    #  can just be plugged direcly into the scenario table.
    #  Same goes for inflation.
    #  However, price of fuel should be defined independently to be of interest
    #  We define dummy scenarios for the time being.
    #  Code corresponds to low/medium/high
    options["fuel_price_scenario"] = 1

    return options


def import_param(path_precalc_inp, options):
    """
    Import default parameters.

    Import set of numerical parameters used in the model.
    Some parameters are the output of a calibration process: it is the case
    of construction function parameters, incomes and associated gravity
    parameter, utility function parameters, and disamenity index for informal
    housing. Some other parameters are just defined ad hoc, based on existing
    empirical evidence.

    Parameters
    ----------
    path_precalc_inp : str
        Path for precalcuted input data (calibrated parameters)
    options : dict
        Dictionary of default options

    Returns
    -------
    param : dict
        Dictionary of default parameters

    """
    # Define baseline year
    param = {"baseline_year": 2011}

    # CALIBRATED PARAMETERS

    # Utility function parameters, as calibrated in Pfeiffer et al. (table C7)
    #  Elasticity of surplus housing
    if options["load_precal_param"] == 1:
        param["beta"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedUtility_beta.mat'
            )["calibratedUtility_beta"].squeeze()
    elif options["load_precal_param"] == 0:
        param["beta"] = np.load(
            path_precalc_inp + 'calibratedUtility_beta.npy')
    # param["beta"] = 0.25

    #  Note that, although this parameter value could be re-estimated, we
    #  choose to pin it down to its value in Pfeiffer et al. (0.25) by default.
    #  This is because this value corresponds to the benchmark estimated
    #  in Finlay and Williams (2022), table B.3: as this term is supposed
    #  to reflect households' real expenditure share on housing, its proper
    #  estimation is rendered difficult by biases coming from variation in
    #  local housing prices, notably (see Diamond and Gaubert, 2022,
    #  section 2.3.3 for more details).

    #  For sensitivity checks, we suggest an acceptable range of values going
    #  from 0.1 (weighted average from expenditure survey data) to 0.4
    #  (estimated in Avner et al., 2021, which is closely related to our
    #  framework). Those are also the bounds given by Rosenthal (2014) for
    #  renters vs. owner-occcupiers: remember that in our framework, all
    #  households are supposed to be renters, as the choice to rent or to buy
    #  is equivalent as long as housing prices reflect the capitalized value of
    #  an infinite flow of rents. The reality is, of course, more complex, and
    #  the choice of an intermediate value allows to capture that fact.

    # NB: For sensitivity checks, alternative input values of beta should be
    # tested directly in the calibration.calib_main_func module
    # (estim_util_func_param function) as the estimated value of the local
    # amenity score will depend on this parameter: we do not expect results
    # to vary significantly when keeping the benchmark (dis)amenity score
    # (imported in the data module), but the fit may slightly improve when
    # re-calibrating it.

    #  Composite good elasticity
    param["alpha"] = 1 - param["beta"]

    #  Basic need in housing
    if options["load_precal_param"] == 1:
        param["q0"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedUtility_q0.mat'
            )["calibratedUtility_q0"].squeeze()
    elif options["load_precal_param"] == 0:
        param["q0"] = np.load(
            path_precalc_inp + 'calibratedUtility_q0.npy')
    # param["q0"] = 4

    #  Again, we use estimate from Pfeiffer et al. as our default, but not for
    #  the same reasons. Indeed, there is a clear empirical counterpart to such
    #  parameter, which should be set equal to the first decile/percentile of
    #  the dwelling size distribution (including informal housing).
    #  As we do not observe such data, we set the value at 4 m² (typo in the
    #  working paper), which is a plausible value, as a placeholder before
    #  finding a more precise estimate.

    #  NB: this parameter sets the degree of non-homotheticity of housing
    #  demand, meaning by how much the real expenditure share that households
    #  spend on housing decrease with their income: the bigger, the stronger
    #  spatial sorting across income groups. This specification allows to
    #  micro-found heterogeneous housing demands as a single utility function.

    # Disamenity parameter for informal settlements and backyard shacks
    # NB: This is a (location-based) array of multiplicative terms between 0
    # and 1 that account for unobserved disamenity factors when living in
    # informal backyards or informal settlements (eviction probability, lack
    # of access to sanitation, etc.)

    if (options["location_based_calib"] == 0
       and options["load_precal_param"] == 1):
        disamenity_param = scipy.io.loadmat(
            path_precalc_inp + 'calibratedParamAmenities.mat'
            )["calibratedParamAmenities"].squeeze()
        param["informal_pockets"] = np.matlib.repmat(
            disamenity_param[1], 24014, 1).squeeze()
        param["backyard_pockets"] = np.matlib.repmat(
            disamenity_param[0], 24014, 1).squeeze()

    elif (options["location_based_calib"] == 0
          and options["load_precal_param"] == 0):
        param_amenity_settlement = np.load(
            path_precalc_inp + 'param_amenity_settlement.npy')
        param["informal_pockets"] = np.matlib.repmat(
            param_amenity_settlement, 24014, 1).squeeze()
        param_amenity_backyard = np.load(
            path_precalc_inp + 'param_amenity_backyard.npy')
        param["backyard_pockets"] = np.matlib.repmat(
            param_amenity_backyard, 24014, 1).squeeze()

    elif options["location_based_calib"] == 1:
        param["informal_pockets"] = np.load(
            path_precalc_inp + 'param_pockets.npy')
        param["backyard_pockets"] = np.load(
            path_precalc_inp + 'param_backyards.npy')

    # Housing production function parameters, as calibrated in Pfeiffer et al.
    # (table C7)
    #  Capital elasticity
    if options["load_precal_param"] == 1:
        param["coeff_b"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedHousing_b.mat')["coeff_b"].squeeze()
    elif options["load_precal_param"] == 0:
        param["coeff_b"] = np.load(
            path_precalc_inp + 'calibratedHousing_b.npy')
    #  Note that estimated value of 0.25 differs from standard values in the
    #  literature: Combes et al. (2021) find a value that is closer to 0.65.
    #  This captures the low capital-intensivity of the construction sector
    #  in Cape Town: more specifically, we may think of "formal" and "informal"
    #  regulations on building height, and high cost of capital relative to
    #  land (with small artisanal entrepreneurs, etc.).

    # Land elasticity
    param["coeff_a"] = 1 - param["coeff_b"]

    #  Scale parameter
    if options["load_precal_param"] == 1:
        param["coeff_A"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedHousing_kappa.mat'
            )["coeffKappa"].squeeze()
    elif options["load_precal_param"] == 0:
        param["coeff_A"] = np.load(
            path_precalc_inp + 'calibratedHousing_kappa.npy')
    #  Estimated default value of 0.04 does not have a straightforward
    #  interpretation: it is just a multiplicative constant up to which
    #  elasticity parameters are identified, that allows to fit the data.

    # Gravity parameter of the minimum Gumbel distribution (see Pfeiffer et
    # al.), as calibrated in appendix C3 (typo in original paper)
    if options["load_precal_param"] == 1:
        param["lambda"] = scipy.io.loadmat(path_precalc_inp + 'lambda.mat'
                                           )["lambdaKeep"].squeeze()
    elif options["load_precal_param"] == 0:
        param["lambda"] = np.load(path_precalc_inp + 'lambdaKeep.npy')
    # NB: this parameter is not identified separately from incomes. It governs
    # the variance of the Gumbel distribution underlying idiosyncratic
    # preferences for commuting choice. In more intuitive terms, the bigger
    # its value, the stronger the gravity in commuting choice, meaning the less
    # households will be willing to live far from their chosen employment
    # center (all things equal)

    # OTHER PARAMETERS

    # Number of jobs above which we retain transport zone (TAZ) as an
    # employment center (for calibration)
    param["job_center_threshold"] = 2500

    # Discount factors (typo in working paper)
    #  From Viguié et al. (2014)
    param["depreciation_rate"] = 0.025

    #  From World Development Indicator database (World Bank, 2016)
    #  NB: Note that this will not be used in practice as we will prefer
    #  interpolation from historical interest rates to smooth variability in
    #  annual values
    # param["interest_rate"] = 0.025

    # Housing parameters
    #  Size of an informal dwelling unit in m² (wrong value in Pfeiffer et al.)
    param["shack_size"] = 14
    #  Size of a social housing dwelling unit (m²), see table C6
    #  (Pfeiffer et al.)
    param["RDP_size"] = 40
    #  Size of backyards in RDP (m²), see table C6 (Pfeiffer et al.)
    #  NB: in theory, a backyard can therefore host up to 5 households
    param["backyard_size"] = 70
    #  Number of formal subsidized housing units built per year from 2011 until
    #  2020, from the CoCT's Housing Pipeline (Pfeiffer et al.)
    param["current_rate_public_housing"] = 5000
    #  Number of formal subsidized housing units built per year after 2020
    param["future_rate_public_housing"] = 1000
    #  Cost of inputs for building an informal dwelling unit (in rands)
    #  This is used to account for potential destructions from floods
    param["informal_structure_value"] = 3000

    #  Fraction of the composite good that is kept inside the house and that
    #  can possibly be destroyed by floods (food, furniture, etc.)
    #  Corresponds to average share of such goods in total household budget
    #  (excluding rent): comes from Quantec expenditure data
    #  (see HH-budget-floods.xlsx in Aux data directory)
    param["fraction_z_dwellings"] = 0.27
    #  NB: by default, we consider only spending on durable and semi-durable
    #  goods, as a fraction of compensation (excluding unearned income) net
    #  of commuting and housing costs. We invite the user to run sensitivity
    #  checks based on alternative assumptions.

    #  Value of a formal subsidized housing dwelling unit (in rands):
    #  again needed for flood damage estimation
    param["subsidized_structure_value"] = 127000

    # Max % of available land that can be built for housing (to take roads and
    # open space into account): benchmark value comes from Viguié et al., 2014
    # (table B1). More precisely, maximum fraction of available ground surface
    # devoted to building is 62% in Paris: we round this value to the upper
    # decile to account for the fact that Cape Town is more horizontally dense
    # (as are most cities in developing countries). This value will be used
    # in areas dedicated to formal private housing (formal subsidized housing
    # is exogenous).

    # For informal backyards, we set this value to 45% (of total backyard area)
    # This corresponds to 2.25 times the size of an informal "shack": we assume
    # that people that rent out part of their backyards are not willing to host
    # more than two households, with some space between the two.
    # For informal settlements, we set the parameter to roughly half the value
    # used for formal private housing, to reflect the interstitial nature of
    # most settlements.

    # Note that such selection does not exclude specific areas that would be
    # dedicated to commercial real estate from the analysis.
    # NB: we invite the user to run sensitivity checks based on alternative
    # assumptions

    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.4

    # Constraints on housing supply (in meters): a priori not binding
    # Radius (in km) defining the extent of the city centre
    param["historic_radius"] = 6
    # Note that we allow for higher construction in the centre vs. the rest of
    # the city
    param["limit_height_center"] = 80
    param["limit_height_out"] = 10

    # Agricultural land prices (in rands)
    #  Corresponds to the ninth decile in the sales data sets for 2011, when
    #  selecting only agricultural properties in rural areas (Pfeiffer et al.).
    #  There, it is assumed that "housing" prices equal land prices
    param["agricultural_price_baseline"] = 807.2
    #  Estimated the same way for 2001 (used for validation)
    param["agricultural_price_retrospect"] = 70.7

    # Year urban edge constraint kicks in (when option is used)
    param["year_urban_edge"] = 2011

    # Labor parameters
    #  Number of income classes set to 4 as in Pfeiffer et al. (see table A1)
    param["nb_of_income_classes"] = 4
    #  Equivalence between income classes in the data (12) and in the model
    #  (4), from poorest to richest. Note that we exclude people earning no
    #  income from the analysis (they will be recovered later)
    param["income_distribution"] = np.array(
        [0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4])

    #  Average number of employed workers per household of each income class
    #  (see appendix B1 in Pfeiffer et al.: corresponds to 2 times the
    #  annual employment rate): all representative households (made up of two
    #  working people) are assumed to work part of the year.
    #  Structural (as opposed to conjonctural) is therefore out of the scope of
    #  this model

    #  NB: we consider population in the labour force all along (no inactive
    #  people)
    #  NB: The exogenous employment rate is calibrated using educational
    #  attainment as a proxy for income level

    param["household_size"] = [1.14, 1.94, 1.92, 1.94]

    # Transportation cost parameters
    param["walking_speed"] = 4  # in km/h
    param["time_cost"] = 1  # equivalence in monetary terms

    # Parameters used in equilibrium.compute_equilibrium: iteration stops when
    # the error in the computed number of households per income bracket falls
    # below some precision level, or when the number of iterations reaches some
    # threshold (to limit processing time)
    param["max_iter"] = 1000
    param["precision"] = 0.001

    # Dynamic parameters
    #  Lag in housing building in years (from Viguié et al., 2014, table B1)
    param["time_invest_housing"] = 3
    #  Set the number of simulations per year
    param["iter_calc_lite"] = 1

    # Miscellaneous parameters

    # Size (in m²) above which we need to switch flood damage functions for
    # formal housing: corresponds to existence of a 2nd floor
    # NB: in the absence of relevant information, we set a default value
    # roughly equal to the median in validation data aggregated at the SP level
    param["threshold"] = 130

    # Make copies of parameters that may change over time (to be used in
    # simulations)
    param["informal_structure_value_ref"] = copy.deepcopy(
        param["informal_structure_value"])
    param["subsidized_structure_value_ref"] = copy.deepcopy(
        param["subsidized_structure_value"])

    return param


def import_construction_parameters(param, grid, housing_types_sp,
                                   dwelling_size_sp,
                                   mitchells_plain_grid_baseline,
                                   grid_formal_density_HFA, coeff_land,
                                   interest_rate, options):
    """
    Update default parameters with construction parameters.

    Import set of numerical construction-related parameters used in the model.
    They depend on pre-loaded data and are therefore imported as part of a
    separate function

    Parameters
    ----------
    param : dict
        Dictionary of default parameters
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    housing_types_sp : DataFrame
        Table yielding, for each Small Place (1,046), the number of
        informal backyards, informal settlements, and total number
        of dwelling units at baseline year (2011), as well as its
        x and y (centroid) coordinates
    dwelling_size_sp : Series
        Average dwelling size (in m²) in each Small Place (1,046)
        at baseline year (2011)
    mitchells_plain_grid_baseline : ndarray(uint8)
        Dummy coding for belonging to Mitchells Plain neighbourhood
        at the grid-cell (24,014) level
    grid_formal_density_HFA : ndarray(float64)
        Population density (per m²) in formal private housing at baseline year
        (2011) at the grid-cell (24,014) level
    coeff_land : ndarray(float64, ndim=2)
        Table yielding, for each grid cell (24,014), the percentage of land
        area available for construction in each housing type (4) respectively.
        In the order: formal private housing, informal backyards, informal
        settlements, formal subsidized housing.
    interest_rate : float64
        Real interest rate for the overall economy, corresponding to an average
        over past years
    options : dict
        Dictionary of default options

    Returns
    -------
    param : dict
        Updated dictionary of default parameters
    minimum_housing_supply : ndarray(float64)
        Minimum housing supply (in m²) for each grid cell (24,014), allowing
        for an ad hoc correction of low values in Mitchells Plain
    agricultural_rent : int
        Annual housing rent below which it is not profitable for formal private
        developers to urbanize (agricultural) land: endogenously limits urban
        sprawl

    """
    # We define housing supply per unit of land for simulations where
    # formal private developers do not adjust to demand
    param["housing_in"] = np.empty(len(grid_formal_density_HFA))
    param["housing_in"][:] = np.nan
    # Fill vector with population density in formal housing divided by the
    # share of built formal area (times 1.1) for areas with some formal housing
    # This gives the density in formal area instead of just the pixel area:
    # we therefore consider that developers always provide one surface unit of
    # housing per household per unit of land. In practice, this is not used.
    # NB: HFA = habitable floor area
    cond = coeff_land[0, :] != 0
    param["housing_in"][cond] = (
        grid_formal_density_HFA[cond] / coeff_land[0, :][cond] * 1.1)
    param["housing_in"][~np.isfinite(param["housing_in"])] = 0
    # Deal with formally non-built or non-inhabited areas
    param["housing_in"][
        (coeff_land[0, :] == 0) | np.isnan(param["housing_in"])
        ] = 0
    # Put a cap and a floor on values
    param["housing_in"][param["housing_in"] > 2 * (10**6)] = 2 * (10**6)
    param["housing_in"][param["housing_in"] < 0] = 0

    # In Mitchells Plain, housing supply is given exogenously (planning),
    # and only households of group 2 live there (coloured neighborhood).
    # We do the same as before: the idea is to have a min housing supply in
    # this zone whose formal density might be underestimated by the model.
    # Then, we may choose whether or not to apply this ad hoc correction.

    if options["correct_mitchells_plain"] == 0:
        param["minimum_housing_supply"] = np.zeros(len(grid.dist))
    elif options["correct_mitchells_plain"] == 1:
        # Original specification
        param["minimum_housing_supply"][mitchells_plain_grid_baseline] = (
            (grid_formal_density_HFA[mitchells_plain_grid_baseline]
             / coeff_land[0, :][mitchells_plain_grid_baseline]))
        param["minimum_housing_supply"][
            (coeff_land[0, :] < 0.1)
            | (np.isnan(param["minimum_housing_supply"]))
            ] = 0

    minimum_housing_supply = param["minimum_housing_supply"]

    # Let us define a minimum dwelling size for formal private housing.
    # We take minimum dwelling size of built areas where the share of informal
    # and backyard is smaller than 10% of the overall number of dwellings.
    # See Pfeiffer et al., section 4.2 (formal neighborhoods)
    param["mini_lot_size"] = np.nanmin(
        dwelling_size_sp[housing_types_sp.total_dwellings_SP_2011 != 0][
            (housing_types_sp.informal_SP_2011[
                housing_types_sp.total_dwellings_SP_2011 != 0]
                + housing_types_sp.backyard_SP_2011[
                    housing_types_sp.total_dwellings_SP_2011 != 0])
            / housing_types_sp.total_dwellings_SP_2011[
                housing_types_sp.total_dwellings_SP_2011 != 0]
            < 0.1
            ]
        )

    # We define agricultural (annual) rent to put a floor on formal private
    # housing market rents (and endogenously limit urban expansion).
    # Comes from zero profit condition for formal private developers: allows to
    # convert land prices into housing prices
    # (cf. Pfeiffer et al., footnote 16)

    # NB: although we could compute agricultural rent as the constant annual
    # coupon associated with the observed agricultural land price, this would
    # not capture the opportunity cost associated with developing the land,
    # but would just yield an equivalence for an agricultural use: we need to
    # refer to developers' profit function to determine a relevant value

    agricultural_rent = compute_agricultural_rent(
        param["agricultural_price_baseline"], param["coeff_A"], interest_rate,
        param, options
        )

    return param, minimum_housing_supply, agricultural_rent


def compute_agricultural_rent(rent, scale_fact, interest_rate, param, options):
    """
    Convert agricultural land price into theoretical annual housing rent.

    The conversion leverages the zero profit condition for formal private
    developers in equilibrium.

    Parameters
    ----------
    rent : float
        Parametric agricultural land price at baseline year (2011)
    scale_fact : float
        (Calibrated) scale factor for the construction function of
        formal private developers
    interest_rate : float
        Real interest rate for the overall economy, corresponding to an average
        over past years
    param : dict
        Dictionary of default parameters
    options : dict
        Dictionary of default options

    Returns
    -------
    agricultural_rent : float64
        Theoretical agricultural (annual) rent (corresponds to opportunity
        cost of non-urbanized land)

    """
    if options["correct_agri_rent"] == 1 and options["deprec_land"] == 1:
        agricultural_rent = (
            rent ** (param["coeff_a"])
            * (param["depreciation_rate"] + interest_rate)
            / (scale_fact * param["coeff_b"] ** param["coeff_b"]
                * param["coeff_a"] ** param["coeff_a"])
            )
    elif options["correct_agri_rent"] == 0 and options["deprec_land"] == 1:
        agricultural_rent = (
            rent ** (param["coeff_a"])
            * (param["depreciation_rate"] + interest_rate)
            / (scale_fact * param["coeff_b"] ** param["coeff_b"])
            )
    elif options["correct_agri_rent"] == 1 and options["deprec_land"] == 0:
        agricultural_rent = (
            rent ** param["coeff_a"]
            * interest_rate ** param["coeff_a"]
            * (param["depreciation_rate"] + interest_rate) ** param["coeff_b"]
            / (scale_fact * param["coeff_b"] ** param["coeff_b"]
                * param["coeff_a"] ** param["coeff_a"])
            )
    elif options["correct_agri_rent"] == 0 and options["deprec_land"] == 0:
        agricultural_rent = (
            rent ** param["coeff_a"]
            * interest_rate ** param["coeff_a"]
            * (param["depreciation_rate"] + interest_rate) ** param["coeff_b"]
            / (scale_fact * param["coeff_b"] ** param["coeff_b"])
            )

    return agricultural_rent
