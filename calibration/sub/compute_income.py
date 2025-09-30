# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import copy
import math


def import_transport_costs(grid, param, yearTraffic,
                           households_per_income_class,
                           spline_inflation, spline_fuel,
                           spline_population_income_distribution,
                           spline_income_distribution, path_precalc_inp,
                           path_precalc_transp, dim, options):
    """
    Compute monetary and time costs from commuting for some given year.

    This function leverages the CoCT's EMME/2 transport model for inputs.

    Parameters
    ----------
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city
    param : dict
        Dictionary of default parameters
    yearTraffic : int
        Year for which we want to run the function
    households_per_income_class : ndarray(float64)
        Exogenous total number of households per income group (excluding people
        out of employment, for 4 groups)
    spline_inflation : interp1d
        Linear interpolation for inflation rate (in base 100 relative to
        baseline year) over the years (baseline year set at 0)
    spline_fuel : interp1d
        Linear interpolation for fuel price (in rands per km)
        over the years (baseline year set at 0)
    spline_population_income_distribution : interp1d
        Linear interpolation for total population per income group in the data
        (12) over the years (baseline year set at 0)
    spline_income_distribution : interp1d
        Linear interpolation for median annual income (in rands) per income
        group in the data (12) over the years (baseline year set at 0)
    path_precalc_inp : str
        Path for precalcuted input data (calibrated parameters)
    path_precalc_transp : str
        Path for precalcuted transport inputs (intermediate outputs from
        commuting choice model)
    dim : str
        Geographic level of analysis at which we want to run the commuting
        choice model: should be set to "GRID" or "SP"
    options : dict
        Dictionary of default options

    Returns
    -------
    timeOutput : ndarray(float64, ndim=3)
        Duration (in min) of a round trip for each transport mode (5) between
        each selected geographic unit and each selected job center (185)
        centroids
    distanceOutput : ndarray(float64, ndim=3)
        Distance (in km) of a one-way trip for each transport mode (5) between
        each selected geographic unit and each selected job center (185)
        centroids. Note that distances actually do not change across transport
        modes.
    monetaryCost : ndarray(float64, ndim=3)
        Annual monetary cost (in rands) of a round trip for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids
    costTime : ndarray(float64, ndim=3)
        Daily share of working time spent commuting for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids. This will be multiplied by expected income to get the
        opportunity cost of time.

    """
    # STEP 1: IMPORT TRAVEL TIMES AND COSTS

    # Import travel times and distances
    transport_times = scipy.io.loadmat(path_precalc_inp
                                       + 'Transport_times_' + dim)

    # Public fares per km: see Pfeiffer et al. (appendix B2)
    # This is based on Roux (2013), table 4.15 : fares are regressed on
    # distance thresholds. The intercept defines the fixed component and the
    # slope defines the variable component (see Transport-costs.xlsx in Aux
    # data)

    # Note that we take into account the difference in inflation between the
    # year the data was collected and the year of our analysis (can be set to
    # other value than baseline)
    # NB: 40 in variable cost correspond to 20 working days per month, times
    # 2 for the round trip
    priceTrainPerKM = (
        0.164 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
                            )
    priceTrainFixedMonth = (
        4.48 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    # Note that bus and taxi had been interverted in Basile's code
    priceBusPerKM = (
        0.785 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceBusFixedMonth = (
        4.32 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    # priceTaxiPerKM = (
    #     0.785 * spline_inflation(2011 - param["baseline_year"])
    #     / spline_inflation(2013 - param["baseline_year"])
    #     )
    # priceTaxiFixedMonth = (
    #     4.32 * 40 * spline_inflation(2011 - param["baseline_year"])
    #     / spline_inflation(2013 - param["baseline_year"])
    #     )
    priceTaxiPerKM = (
        0.522 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceTaxiFixedMonth = (
        6.24 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    # priceBusPerKM = (
    #     0.522 * spline_inflation(2011 - param["baseline_year"])
    #     / spline_inflation(2013 - param["baseline_year"])
    #     )
    # priceBusFixedMonth = (
    #     6.24 * 40 * spline_inflation(2011 - param["baseline_year"])
    #     / spline_inflation(2013 - param["baseline_year"])
    #     )

    # Again, we correct for inflation with respect to year the function is run
    inflation = spline_inflation(yearTraffic)
    if options["correct_infla_base"] == 0:
        infla_base = spline_inflation(2012 - param["baseline_year"])
    elif options["correct_infla_base"] == 1:
        infla_base = spline_inflation(2011 - param["baseline_year"])
    priceTrainPerKM = priceTrainPerKM * inflation / infla_base
    priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_base
    priceTaxiPerKM = priceTaxiPerKM * inflation / infla_base
    priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_base
    priceBusPerKM = priceBusPerKM * inflation / infla_base
    priceBusFixedMonth = priceBusFixedMonth * inflation / infla_base
    priceFuelPerKM = spline_fuel(yearTraffic)

    # Private fixed costs, in rands (variable is already defined by
    # priceFuelPerKM)
    # See Pfeiffer et al. (appendix B2) : corresponds to monthly vehicle
    # depreciation in 2011
    priceFixedVehiculeMonth = 400
    priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_base

    # STEP 2: TRAVEL TIMES AND COSTS AS MATRIX

    # Parameters: see Pfeiffer et al. (appendix B2)
    # We assume 8 working hours per day and 20 working days per month
    # numberDaysPerYear = 235
    numberDaysPerYear = 240
    numberHourWorkedPerDay = 8

    # Time taken by each mode for a round trip (in sec)
    # Includes walking time to station and other features from EMME/2 model
    timeOutput = np.empty(
        (transport_times["durationTrain"].shape[0],
         transport_times["durationTrain"].shape[1], 5)
        )
    timeOutput[:] = np.nan

    # To get walking times, we take 2 times the distances by car (to get trips
    # in both directions) multiplied by 1.2 (sinusoity coefficient), divided
    # by the walking speed (in km/h), which we multiply by 60 to get minutes
    # NB: see Viguié et al. (2014), table B.1, for sinusoity estimate

    # if options["correct_round_trip"] == 1:
    #     # Note that we do have some negative durations: their number is small
    #     # so we just convert them to zero
    #     timeOutput[:, :, 0] = (transport_times["distanceCar"]
    #                            / param["walking_speed"] * 60 * 60 * 1.2 * 2)
    #     timeOutput[:, :, 0][np.isnan(transport_times["durationCar"])
    #                         ] = np.nan
    #     timeOutput[:, :, 1] = (
    #          copy.deepcopy(transport_times["durationTrain"])*2)
    #     timeOutput[:, :, 1][transport_times["durationTrain"] < 0] = 0
    #     timeOutput[:, :, 2] = copy.deepcopy(transport_times["durationCar"])*2
    #     timeOutput[:, :, 2][transport_times["durationCar"] < 0] = 0
    #     timeOutput[:, :, 3] = (
    #         copy.deepcopy(transport_times["durationMinibus"])*2)
    #     timeOutput[:, :, 3][transport_times["durationMinibus"] < 0] = 0
    #     timeOutput[:, :, 4] = copy.deepcopy(transport_times["durationBus"])*2
    #     timeOutput[:, :, 4][transport_times["durationBus"] < 0] = 0

    # Note that we do have some negative durations: their number is small,
    # so we just convert them to zero
    timeOutput[:, :, 0] = (transport_times["distanceCar"]
                           / param["walking_speed"] * 60 * 60 * 1.2 * 2)
    timeOutput[:, :, 0][np.isnan(transport_times["durationCar"])] = 0
    timeOutput[:, :, 1] = copy.deepcopy(transport_times["durationTrain"])
    timeOutput[:, :, 1][transport_times["durationTrain"] < 0] = 0
    timeOutput[:, :, 2] = copy.deepcopy(transport_times["durationCar"])
    timeOutput[:, :, 2][transport_times["durationCar"] < 0] = 0
    timeOutput[:, :, 3] = copy.deepcopy(transport_times["durationMinibus"])
    timeOutput[:, :, 3][transport_times["durationMinibus"] < 0] = 0
    timeOutput[:, :, 4] = copy.deepcopy(transport_times["durationBus"])
    timeOutput[:, :, 4][transport_times["durationBus"] < 0] = 0

    # Length (in km) using each mode
    if options["correct_round_trip"] == 1:
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:] = np.nan
        multiplierPrice[:, :, 0] = np.zeros((timeOutput[:, :, 0].shape))
        multiplierPrice[:, :, 1] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 2] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 3] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 4] = transport_times["distanceCar"] * 2
    elif options["correct_round_trip"] == 0:
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:] = np.nan
        multiplierPrice[:, :, 0] = np.zeros((timeOutput[:, :, 0].shape))
        multiplierPrice[:, :, 1] = transport_times["distanceCar"]
        multiplierPrice[:, :, 2] = transport_times["distanceCar"]
        multiplierPrice[:, :, 3] = transport_times["distanceCar"]
        multiplierPrice[:, :, 4] = transport_times["distanceCar"]

    # Convert the results to annual cost
    pricePerKM = np.empty(5)
    pricePerKM[:] = np.nan
    pricePerKM[0] = np.zeros(1)
    pricePerKM[1] = priceTrainPerKM*numberDaysPerYear
    pricePerKM[2] = priceFuelPerKM*numberDaysPerYear
    pricePerKM[3] = priceTaxiPerKM*numberDaysPerYear
    pricePerKM[4] = priceBusPerKM*numberDaysPerYear

    # Simple distances (useful for residence-workplace distance distribution)
    distanceOutput = np.empty((timeOutput.shape))
    distanceOutput[:] = np.nan
    distanceOutput[:, :, 0] = transport_times["distanceCar"]
    distanceOutput[:, :, 1] = transport_times["distanceCar"]
    distanceOutput[:, :, 2] = transport_times["distanceCar"]
    distanceOutput[:, :, 3] = transport_times["distanceCar"]
    distanceOutput[:, :, 4] = transport_times["distanceCar"]

    # STEP 3 : MONETARY AND TIME COSTS

    # Monetary price per year (for each employment center)
    # NB: 12 is for number of months in a year
    monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
    for index2 in range(0, 5):
        monetaryCost[:, :, index2] = (pricePerKM[index2]
                                      * multiplierPrice[:, :, index2])
    #  Train
    monetaryCost[:, :, 1] = monetaryCost[:, :, 1] + priceTrainFixedMonth * 12
    #  Private car
    monetaryCost[:, :, 2] = (monetaryCost[:, :, 2] + priceFixedVehiculeMonth
                             * 12)
    #  Minibus/taxi
    monetaryCost[:, :, 3] = monetaryCost[:, :, 3] + priceTaxiFixedMonth * 12
    #  Bus
    monetaryCost[:, :, 4] = monetaryCost[:, :, 4] + priceBusFixedMonth * 12

    # We assume that people not travelling a certain distance have an extra
    # high cost of doing so
    monetaryCost[np.isnan(monetaryCost)] = 10 ** 5

    # Daily share of working time spent commuting (all values in sec)
    # NB: The time cost parameter is a proportionality factor between time
    # and monetary values. By default, it is set as 1: the welfare loss from
    # time spent commuting directly translates into foregone wages.
    costTime = (timeOutput * param["time_cost"]
                / (60 * 60 * numberHourWorkedPerDay))

    # We assume that people not travelling a certain distance have an extra
    # high cost of doing so
    costTime[np.isnan(costTime)] = 1

    return timeOutput, distanceOutput, monetaryCost, costTime


def EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime,
                   job_centers, average_income, income_distribution,
                   list_lambda, options):
    """
    Estimate incomes per job center and number of commuters per distance bin.

    This function leverages the commutingSolve() function to iterate over
    income values until the target number of commuters for each job center is
    reached. This allows to compute incomes per job center (and income group)
    and to find the split of commuters across distance-from-CBD brackets, for
    several values of the gravity parameter (from commuting choice model).

    Parameters
    ----------
    param : dict
        Dictionary of default options
    timeOutput : ndarray(float64, ndim=3)
        Duration (in min) of a round trip for each transport mode (5) between
        each selected geographic unit and each selected job center (185)
        centroids
    distanceOutput : ndarray(float64, ndim=3)
        Distance (in km) of a one-way trip for each transport mode (5) between
        each selected geographic unit and each selected job center (185)
        centroids. Note that distances actually do not change across transport
        modes.
    monetaryCost : ndarray(float64, ndim=3)
        Annual monetary cost (in rands) of a round trip for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids
    costTime : ndarray(float64, ndim=3)
        Daily share of working time spent commuting for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids. This will be multiplied by expected income to get the
        opportunity cost of time.
    job_centers : ndarray(float64, ndim=2)
        Number of jobs in each selected job center (185) per income group (4).
        Remember that we rescale the number of individual jobs to reflect total
        household employment, as our income and population data are for
        households only: one job basically provides employment for two people.
        This simplification allows to model households as a single
        representative agent and to abstract from a two-body problem.
        Empirically, this holds on aggregate as households' position on the
        labor market is often determined by one household head.
    average_income : ndarray(float64)
        Average median income for each income group in the model (4)
    income_distribution : ndarray(uint16, ndim=2)
        Exogenous number of households in each Small Place (1,046) for each
        income group in the model (4)
    list_lambda : ndarray(float64)
        List of values over which to scan for the gravity parameter used in
        the commuting choice model
    options : dict
        Dictionary of default options

    Returns
    -------
    incomeCentersSave : ndarary(float64, ndim=3)
        Calibrated annual household income (in rands) for each income group (4)
        and each selected job center (185), for each scanned value of the
        gravity parameter
    distanceDistribution : ndarray(float64, ndim=2)
        Share of residence-workplace distances in each 5-km from CBD bracket
        for each scanned value of the gravity parameter
    scoreMatrix : ndarray(float64, ndim=2)
        Ratio of simulated over observed (rescaled) number of jobs for each
        income group (4) and each scanned value of the gravity parameter:
        defines an error metric for the quality of our calibration

    """
    # Setting time and space
    annualToHourly = 1 / (8*20*12)
    monetary_cost = monetaryCost * annualToHourly
    #  Corresponds to the brackets for which we have aggregate statistics on
    #  the number of commuters to fit our calibration
    bracketsDistance = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 200])

    #  We initialize output matrices
    incomeCentersSave = np.zeros((len(job_centers[:, 0]), 4, len(list_lambda)))
    distanceDistribution = np.zeros(
        (len(bracketsDistance) - 1, len(list_lambda)))
    scoreMatrix = np.zeros((len(list_lambda), param["nb_of_income_classes"]))

    # We begin simulations for different values of lambda

    for i in range(0, len(list_lambda)):

        param_lambda = list_lambda[i]

        print('Estimating for lambda = ', param_lambda)

        # We initialize output vectors for each lambda
        incomeCentersAll = -math.inf * np.ones((len(job_centers[:, 0]), 4))
        distanceDistributionGroup = np.zeros((len(bracketsDistance) - 1, 4))

        # We run separate simulations for each income group

        for j in range(0, param["nb_of_income_classes"]):

            # (Employed) household size varies with income group / transport
            # costs: we select the income group of interest
            householdSize = param["household_size"][j]
            # Same for "average" income (which needs to be adapted to hourly
            # from income data)
            averageIncomeGroup = average_income[j] * annualToHourly

            print('incomes for group ', j)

            # We consider job centers where selected income group represents
            # more than 1/4 of the number-of-jobs threshold (used to select
            # job centers among transport zones): it allows to avoid marginal
            # crossings between income classes and job centers, hence reduces
            # the number of equations to solve and makes optimization faster
            # (+ no corner cases)

            # NB: numeric solver (gradient descent) does not work when
            # function is not always differentiable (which is the case here as,
            # above/below some utility threshold, we have tipping effects),
            # hence we code our own solver to remain in the interior

            whichCenters = job_centers[:, j] > 600
            # whichCenters = (
            #     job_centers[:, j] > param["job_center_threshold"] / 4)
            popCenters = job_centers[whichCenters, j]

            # We reweight population in each income group per SP to make it
            # comparable with population in SELECTED job centers
            # Note that unemployed population is not included! This is
            # important with what follows.
            # Also note that SP data includes more areas than included in grid
            popResidence = (
                income_distribution[:, j]
                * sum(job_centers[whichCenters, j])
                / sum(income_distribution[:, j])
            )

            # Numeric parameters come from trial and error and do not change
            # results a priori: they just help convergence
            
            maxIter = 1000
            # TEST for crude scanning
            #maxIter = 100
            tolerance = 0.01
            if j == 0:
                factorConvergenge = (
                    0.01 / 100
                    * (np.log(param_lambda**10 / 10**10) + np.log(10**10))
                    )
            elif j == 1:
                factorConvergenge = (
                    0.002 / 100
                    * (np.log(param_lambda**10 / 10**10) + np.log(10**10))
                    )
            elif j == 2:
                factorConvergenge = (
                    0.001 / 100
                    * (np.log(param_lambda**10 / 10**10) + np.log(10**10))
                    )
            elif j == 3:
                factorConvergenge = (
                    0.0008 / 100
                    * (np.log(param_lambda**10 / 10**10) + np.log(10**10))
                    )

            iter = 0
            error = np.zeros((len(popCenters), maxIter))
            scoreIter = np.zeros(maxIter)
            errorMax = 1

            # Initializing the solver
            incomeCenters = np.zeros((sum(whichCenters), maxIter))
            incomeCenters[:, 0] = (
                averageIncomeGroup
                * (popCenters / np.nanmean(popCenters))
                ** (0.1)
            )
            # Initial error corresponds to the difference between observed
            # and simulated population working in each job center
            error[:, 0], _ = commutingSolve(
                incomeCenters[:, 0], averageIncomeGroup, popCenters,
                popResidence, monetary_cost, costTime, param_lambda,
                householdSize, whichCenters, bracketsDistance, distanceOutput,
                options)

            # Then we iterate by adding to each job center income the average
            # value of income per income group, weighted by the importance of
            # the error relative to the observed job center population: if we
            # underestimate the population, we increase the income
            while ((iter <= maxIter - 1) & (errorMax > tolerance)):

                incomeCenters[:, iter] = (
                    incomeCenters[:, max(iter-1, 0)]
                    + factorConvergenge
                    * averageIncomeGroup
                    * error[:, max(iter - 1, 0)]
                    / popCenters
                )

                # We also update the error term and store some values
                # before iterating over
                error[:, iter], _ = commutingSolve(
                    incomeCenters[:, iter], averageIncomeGroup, popCenters,
                    popResidence, monetary_cost, costTime, param_lambda,
                    householdSize, whichCenters, bracketsDistance,
                    distanceOutput, options)

                errorMax = np.nanmax(
                    np.abs(error[:, iter] / popCenters))
                scoreIter[iter] = np.nanmean(
                    np.abs(error[:, iter] / popCenters))

                iter = iter + 1

                # print(iter)
                # print("errorMax = " + str(errorMax))
                # print("errorMean = "
                #       + str(np.nanmean(np.abs(error[:, iter] / popCenters))))

            # At the end of the process, we keep the minimum score, and define
            # the corresponding best solution for some lambda and income group
            if (iter > maxIter - 1):
                scoreBest = np.amin(scoreIter)
                scoreMatrix[i, j] = scoreBest
                bestSolution = np.argmin(scoreIter)
                incomeCenters[:, iter-1] = incomeCenters[:, bestSolution]
                print(' - max iteration reached - mean error', scoreBest)

            # If we manage to have a maximum error that falls under the
            # tolerance threshold, we leave the loop and consider the solution
            # corresponding to the latest iteration
            else:
                scoreBest = scoreIter[iter-1]
                scoreMatrix[i, j] = scoreBest
                print(' - computed - max error', errorMax)
                # print(str(iter-1))

            # We also get (for the given income group) the number of commuters
            # for all job centers in given distance brackets
            _, distanceDistributionGroup[:, j] = commutingSolve(
                incomeCenters[:, iter-1], averageIncomeGroup, popCenters,
                popResidence, monetary_cost, costTime, param_lambda,
                householdSize, whichCenters, bracketsDistance, distanceOutput,
                options)

            # We rescale calibrated incomes to stick to overall income data
            # scale: remember that we only computed them for a subset of the
            # population. This allows income levels to be more representative
            # (although this does not change anything in relative terms)
            # NB: note that here, the way we define "average" income from the
            # data actually impact the rescaling
            incomeCentersRescaled = (
                incomeCenters[:, iter-1]
                * averageIncomeGroup
                / (np.nansum(incomeCenters[:, iter-1] * popCenters)
                    / np.nansum(popCenters))
            )

            # Then we update the output matrix for the given income group
            # (with lambda still fixed)
            incomeCentersAll[whichCenters, j] = incomeCentersRescaled

        # We can now loop over different values of lambda and store values back
        # in yearly format for incomes
        incomeCentersSave[:, :, i] = incomeCentersAll / annualToHourly

        # Likewise, for each value of lambda, we store the % of total commuters
        # for each distance bracket
        distanceDistribution[:, i] = (
            np.nansum(distanceDistributionGroup, 1)
            / np.nansum(distanceDistributionGroup)
        )

    return incomeCentersSave, distanceDistribution, scoreMatrix


def compute_ODflows(householdSize, monetaryCost, costTime, incomeCentersFull,
                    whichCenters, param_lambda, options):
    """
    Compute probability distribution and transport costs of commuting pairs.

    This function applies theoretical formulas from the commuting choice model,
    for a given income group and a given gravity parameter (see math appendix
    for more details).

    Parameters
    ----------
    householdSize : float
        Average number of employed workers per household for one income class
        (corresponds to 2 times the employment rate)
    monetaryCost : ndarray(float64, ndim=3)
        Hourly monetary cost (in rands) of a round trip for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids
    costTime : ndarray(float64, ndim=3)
        Daily share of working time spent commuting for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids. This will be multiplied by expected income to get the
        opportunity cost of time.
    incomeCentersFull : ndarray(float64)
        Hourly household income (in rands) for some income group and some
        iteration, per selected job center, rescaled to match income
        distribution across the overall population
    whichCenters : ndarray(bool)
        Dummy variable allowing the narrow selection of job centers used for
        income calibration for a given income group, among pre-selected job
        centers (185)
    param_lambda : float64
        Tested value for the gravity parameter used in the commuting choice
        model

    Returns
    -------
    transportCostModes : ndarray(float64, ndim=3)
        Expected (hourly) total commuting cost (in rands) for one agent,
        for all (narrowly) selected job centers, all SPs of residence (1046),
        and all transport modes (5)
    transportCost : ndarray(float64, ndim=2)
        Expected (hourly) total commuting cost (in rands) for one agent,
        for all (narrowly) selected job centers and all SPs of residence
        (1046): corresponds to min(transportCostModes) across transport modes.
    ODflows : ndarray(float64, ndim=2)
        Probability, for a given income group, to work in each (narrowly)
        selected job center for each potential SP of residence (1046)
    valueMax : ndarray(float64, ndim=2)
        Numerical parameter for each commuting pair that prevents logarithms
        and exponentials used in calculations from diverging towards infinity:
        it corresponds to the maximum argument for exponentials that appear in
        the formula for transportCost: this is not used as an input in other
        functions and is just included here for reference
    minIncome : ndarray(float64)
        Numerical parameter for each SP of residence that prevents logarithms
        and exponentials used in calculations from diverging towards infinity:
        it corresponds to minus the minimum argument for exponentials that
        appear in the formula for ODflows: this is not used as an input in
        other functions and is just included here for reference

    """
    # We first compute expected (hourly) total commuting cost per mode
    # Note that incomeCentersFull is already defined at the household level,
    # whereas monetaryCost is defined at the individual level: hence, we need
    # to multiply monetaryCost by householdsSize to get the value of the
    # monetary costs for all members of the households
    transportCostModes = (
        householdSize * monetaryCost[whichCenters, :, :]
        + (costTime[whichCenters, :, :] * incomeCentersFull[:, None, None])
    )

    # This prevents exponentials/logarithms from diverging towards infinity
    # This is neutral on the result and is set by trial and error
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    # We get expected commuting cost by minimizing transportCostModes across
    # transport modes. This yields the formula below (see log-sum
    # specification in math appendix)

    # Note that the valueMax term ensures some minimal value for the
    # exponential term, hence a value sufficiently bigger than zero in the
    # logarithm term. Then, its impact is cancelled out by substracting it
    # aside from the log-term.

    # Actually, this is not needed in this version of the model
    # if options["prevent_exp_overflow"] == 1:
    #     transportCost = (
    #         - 1 / param_lambda
    #         * np.log(
    #             np.nansum(np.exp(- param_lambda * transportCostModes
    #                               + valueMax[:, :, None]), 2))
    #         - valueMax
    #     )
    transportCost = (
        - 1 / param_lambda
        * np.log(
            np.nansum(np.exp(- param_lambda * transportCostModes), 2))
    )

    # Also for exponential/logarithm convergence
    minIncome = (np.nanmax(
        param_lambda * (incomeCentersFull[:, None] - transportCost), 0) - 700
        )

    # Then, we compute probability distribution of each commuting pair, from
    # discrete choice modelling.
    # Note that the minIncome term ensures that the values in the exponential
    # terms do not grow too big. Then, its impact is cancelled out as it
    # appears both in the numerator and the denominator of the fraction.
    # NB: here, we need to divide by the scale factor (typo in Pfeiffer et al.)

    if options["prevent_exp_overflow"] == 1:
        ODflows = (
            np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost)
                   - minIncome)
            / np.nansum(
                np.exp(param_lambda
                       * (incomeCentersFull[:, None] - transportCost)
                       - minIncome),
                0)[None, :]
        )
    elif options["prevent_exp_overflow"] == 0:
        ODflows = (
            np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost))
            / np.nansum(
                np.exp(param_lambda
                       * (incomeCentersFull[:, None] - transportCost)),
                0)[None, :]
        )

    return transportCostModes, transportCost, ODflows, valueMax, minIncome


def commutingSolve(incomeCentersTemp, averageIncomeGroup, popCenters,
                   popResidence, monetaryCost, costTime, param_lambda,
                   householdSize, whichCenters, bracketsDistance,
                   distanceOutput, options):
    """
    Compute error and distribution for job allocation simulated from incomes.

    This function leverages the compute_ODflows() function to compute a
    theoretical equivalent to number of jobs in each (narrowly) selected job
    center, and the distribution of those jobs across pre-defined
    distance-from-CBD brackets.

    Parameters
    ----------
    incomeCentersTemp : ndarray(float64)
        Hourly household income (in rands) for some income group and some
        iteration, per selected job center
    averageIncomeGroup : float64
        Average hourly median income (from data) for some income group
    popCenters : ndarray(float64)
        Number of jobs in each (narrowly) selected job center for some income
        group. Remember that we further restrict the set of selected job
        centers when calibrating incomes per income group for the sake of
        numerical simplicity. Also remember that we rescale the number of
        individual jobs to reflect total household employment, as our income
        and population data are for households only: one job basically provides
        employment for two people.
        This simplification allows to model households as a single
        representative agent and to abstract from a two-body problem.
        Empirically, this holds on aggregate as households' position on the
        labor market is often determined by one household head.
    popResidence : ndarray(float64)
        Number of households living in each SP (1046), per income group (4):
        comes from census data and does not include people not working
    monetaryCost : ndarray(float64, ndim=3)
        Hourly monetary cost (in rands) of a round trip for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids
    costTime : ndarray(float64, ndim=3)
        Daily share of working time spent commuting for each transport mode
        (5) between each selected geographic unit and each selected job center
        (185) centroids. This will be multiplied by expected income to get the
        opportunity cost of time.
    param_lambda : float64
        Tested value for the gravity parameter used in the commuting choice
        model
    householdSize : float
        Average number of employed workers per household for one income class
        (corresponds to 2 times the employment rate)
    whichCenters : ndarray(bool)
        Dummy variable allowing the narrow selection of job centers used for
        income calibration for a given income group, among pre-selected job
        centers (185)
    bracketsDistance : ndarray(int32)
        Array of floor values for distance brackets used to select the set of
        incomes + gravity parameter that best fit the distribution of
        residence-workplace distances according to those brackets
    distanceOutput : ndarray(float64, ndim=3)
        Distance (in km) of a one-way trip for each transport mode (5) between
        each selected geographic unit and each selected job center (185)
        centroids. Note that distances actually do not change across transport
        modes.
    options : dict
        Dictionary of default options

    Returns
    -------
    score : ndarray(float64)
        Difference between observed and simulated population working in each
        (narrowly) selected job center (for a given income group)
    nbCommuters : ndarray(float64)
        Number of commuters in each pre-defined distance bracket, for a given
        income group

    """
    # We redress the average income in each group per job center to match
    # income data, as resulting ODflows (see below) must be matched with
    # popResidence (defined for the overall population, and not a subset of
    # narrowly selected job centers)
    incomeCentersFull = (
        incomeCentersTemp
        * averageIncomeGroup
        / ((np.nansum(incomeCentersTemp * popCenters)
            / np.nansum(popCenters)))
    )

    # We are essentially interested in the ODflows matrix, that is the
    # probability, for a given income group, to work in each (narrowly)
    # selected job center for each potential SP of residence
    transportCostModes, transportCost, ODflows, *_ = compute_ODflows(
        householdSize, monetaryCost, costTime, incomeCentersFull,
        whichCenters, param_lambda, options)

    # We compare the true population distribution with its theoretical
    # equivalent computed from simulated income distribution: the closer
    # the score is to zero, the better (see math appendix)

    # NB: correction is not needed a priori, as income distribution data from
    # SP does not include people out of employment
    if options["correct_eq3"] == 1:
        score = (popCenters
                 - (householdSize / 2
                    * np.nansum(popResidence[None, :] * ODflows, 1))
                 )
    elif options["correct_eq3"] == 0:
        score = (popCenters
                 - np.nansum(popResidence[None, :] * ODflows, 1))

    # We also return the total number of commuters (in a given income group)
    # for (narrowly) selected job centers in predefined distance brackets
    nbCommuters = np.zeros(len(bracketsDistance) - 1)
    for k in range(0, len(bracketsDistance)-1):
        which = ((distanceOutput[whichCenters, :] > bracketsDistance[k])
                 & (distanceOutput[whichCenters, :] <= bracketsDistance[k + 1])
                 & (~np.isnan(distanceOutput[whichCenters, :])))
        if options["correct_eq3"] == 1:
            nbCommuters[k] = (
                np.nansum(which * ODflows * popResidence[None, :])
                * (householdSize / 2)
                )
        elif options["correct_eq3"] == 0:
            nbCommuters[k] = (
                np.nansum(which * ODflows * popResidence[None, :]))

    return score, nbCommuters
