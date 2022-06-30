# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import clone, BaseEstimator
from causal_generalization import draw_process, generate_dataset, steps_ahead, draw_var_process, generate_var_dataset, estimate_rademacher
import numpy
from typing import Dict, Tuple, Type, Sequence
import logging
import multiprocessing
from joblib import Parallel, delayed
import time
import pandas
import argparse
from pathlib import Path
from StatScorer import StatScorer


def do_grid_searches(estimators: Dict[str, Type[BaseEstimator]], param_grid: Dict[str, Dict[str, Sequence]],
                     dataset: Tuple[
                         numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                         numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray], params: numpy.ndarray) \
        -> Dict[str, Dict[str, float]]:
    """
    Worker function for multi processing.
    Calculates optimal regularization strength with respect to statistical error via grid search.
    :return dictionary. Statistical Error, Causal Error, the optimal reg strength and empirical Rademacher complexity.
        Each dict contains the estimators as keys and a scalar as value.
    """

    x_train, y_train, x_test, y_test, x_inter_train, y_inter_train, x_inter_test, y_inter_test, sample, y_train_omega\
        = dataset

    results = {}
    results['error_stat_lambda_stat'] = {}
    results['error_causal_lambda_stat'] = {}
    results['reg_stat'] = {}
    results['emp_rademacher'] = {}

    for e_name, e in estimators.items():
        if param_grid[e_name] is not None:
            ss = StatScorer(X_train=x_train, y_omega=y_train_omega, omega=omega)
            best_stat = GridSearchCV(estimator=e, param_grid=param_grid[e_name], scoring=ss,
                                     cv=TimeSeriesSplit(n_splits=5))
            best_stat = best_stat.fit(x_train, y_train)

            results['reg_stat'][e_name] = best_stat.best_params_['alpha']
        else:
            best_stat = clone(e)
            best_stat = best_stat.fit(x_train, y_train)
            best_causal = clone(e)
            best_causal = best_causal.fit(x_train, y_train)
            results['reg_stat'][e_name] = numpy.nan

        y_hat = steps_ahead(best_stat.predict, x_test, omega)
        y_inter_hat = steps_ahead(best_stat.predict, x_inter_test, omega)
        stat_err = mean_squared_error(y_hat, y_test)
        causal_err = mean_squared_error(y_inter_hat, y_inter_test)
        results['error_stat_lambda_stat'][e_name] = stat_err
        results['error_causal_lambda_stat'][e_name] = causal_err

        results['emp_rademacher'][e_name] = estimate_rademacher(x_train)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=0)
    parser.add_argument("--sigma_sq", help="Variance of noise", type=float, default=1)
    parser.add_argument("--var", help="If two dim VAR should be used", type=bool, default=False)
    parser.add_argument("--order", help="Process order", type=int, default=5)
    parser.add_argument("--runs", help="Number of independent repetitions. I.e. datasets drawn.", type=int, default=100)
    parser.add_argument("--train_samples", help="Number of training samples", type=int, default=100)
    parser.add_argument("--test_samples", help="Number of test samples", type=int, default=1000)
    parser.add_argument("--omega", help="How many steps to predict ahead.", type=int, default=1)
    parser.add_argument("--interv_all", help="Should intervention manipulate all relevant steps ore just one.",
                        type=bool, default=False)
    args = parser.parse_args()

    rand_seed = args.seed
    numpy.random.seed(rand_seed)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    if args.var and args.order != 1:
        logger.error('Currently only VARs of order 1 supported. You entered order {}.'.format(args.order))
        exit()
    sigma_sq = args.sigma_sq
    proc_order = args.order
    file_prefix = '../data/uai/misspec/' if args.var else '../data/uai/order{}/'.format(proc_order)
    Path(file_prefix).mkdir(parents=True, exist_ok=True)
    num_samples_train = args.train_samples
    num_samples_test = args.test_samples
    num_runs = args.runs
    omega = args.omega

    i = '_interv_all' if args.interv_all else ''
    file_suffix = '_' + str(num_runs) + '_' + str(num_samples_train) + '_' + str(omega) + '_' + str(rand_seed) + i \
                  + '.csv'
    start = time.time()

    # Init estimators and parameter grids
    estimators = {'OLS': LinearRegression(fit_intercept=False), 'Ridge': Ridge(fit_intercept=False),
                  'Lasso': Lasso(fit_intercept=False),
                  'ElasticNet': ElasticNet(fit_intercept=False)
                  }
    param_grid = {'OLS': None, 'Ridge': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]},
                  'Lasso': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]},
                  'ElasticNet': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
                                 'l1_ratio': numpy.arange(.1, 1, .1)}}
    # Draw process params and respective datasets for all parallel runs in advance to prevent race conditions
    datasets = []
    params = []
    for i in range(num_runs):
        logger.info('Generated Dataset ' + str(i))
        if args.var:
            real_proc_params = draw_var_process()
            params.append(real_proc_params.flatten())
            datasets.append(
                generate_var_dataset(real_proc_params, num_samples_train, num_samples_test, sigma_sq, omega))
        else:
            real_proc_params = draw_process(proc_order)
            params.append(real_proc_params)
            datasets.append(generate_dataset(real_proc_params, num_samples_train, num_samples_test, sigma_sq, omega,
                                             interv_all=args.interv_all))

    # Store process parameters in csv file
    if args.var:
        d = pandas.DataFrame(data=params, columns=['a' + str(i + 1) for i in range(4)])
    else:
        d = pandas.DataFrame(data=params, columns=['a' + str(i + 1) for i in range(proc_order)])
    d.to_csv(file_prefix + 'params' + file_suffix, index=False)

    # Do causal and statistical grid searches for all datasets in parallel
    results = Parallel(n_jobs=numpy.min([multiprocessing.cpu_count(), num_runs]), verbose=10)\
        (delayed(do_grid_searches)(estimators, param_grid, datasets[i], params[i]) for i in range(num_runs))
    end = time.time()
    logger.info('Time: ' + str(end - start))

    # Format results and put it into arrays
    result_matrix = {}
    for metric in results[0]:   # all runs return a dict with the same keys
        result_matrix[metric] = numpy.zeros((len(estimators.keys()), num_runs))
        for i, e in enumerate(estimators):
            result_matrix[metric][i, :] = [r[metric][e] for r in results]

    # Store results to csv files
    for metric in result_matrix:
        d = pandas.DataFrame(data=result_matrix[metric].T, columns=list(estimators.keys()))
        d.to_csv(file_prefix + metric + file_suffix, index=False)
