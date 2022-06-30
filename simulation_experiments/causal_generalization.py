# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

import numpy
from typing import Sequence, Tuple
from dataclasses import dataclass
from typing import Callable
from statsmodels.tsa.stattools import acf
from scipy.linalg import toeplitz

max_parameter = 2

def estimate_rademacher(X:numpy.ndarray, B=None):
    if B is None:
        B = numpy.linalg.norm(max_parameter*numpy.ones(X.shape[1]))  # Norm of the parameter vector with all maximum
    summed_norms = numpy.sum(numpy.linalg.norm(X, axis=0)**2)
    return B*numpy.sqrt(summed_norms) / X.shape[0]

def estimate_condition_number(ar_parameters: numpy.ndarray, measure='kappa'):

    sample = generate_sample(ar_parameters, sample_size=1000, sigma_sq=0.5)
    autocorrelation = acf(sample, fft=False, nlags=ar_parameters.shape[0])
    # print(autocorrelation)
    autocorrelation_matrix = toeplitz(autocorrelation)
    # print(autocorrelation_matrix)
    eigs, _ = numpy.linalg.eig(autocorrelation_matrix)
    if measure == 'kappa':
        kappa = numpy.max(numpy.abs(eigs)) / numpy.min(numpy.abs(eigs))
        return min(kappa, 1e8)
    elif measure == 'lambda_max':
        return numpy.max(numpy.abs(eigs))
    elif measure == 'lambda_min':
        return numpy.min(numpy.abs(eigs))
    else:
        logging.error('Estimate condition number can only return the measures "kappa", "lambda_max", "lambda_min". '
                      'You provided ' + str(measure))
        raise Exception

def estimate_condition_number_misspec(ar_parameters: numpy.ndarray, measure='kappa'):
    sample = generate_var_sample(ar_parameters, sample_size=1000, sigma_sq=0.5)[0, :]
    autocorrelation = acf(sample, fft=False, nlags=ar_parameters.shape[0])
    # print(autocorrelation)
    autocorrelation_matrix = toeplitz(autocorrelation)
    # print(autocorrelation_matrix)
    eigs, _ = numpy.linalg.eig(autocorrelation_matrix)
    if measure == 'kappa':
        kappa = numpy.max(numpy.abs(eigs)) / numpy.min(numpy.abs(eigs))
        return min(kappa, 1e8)
    elif measure == 'lambda_max':
        return numpy.max(numpy.abs(eigs))
    elif measure == 'lambda_min':
        return numpy.min(numpy.abs(eigs))
    else:
        logging.error('Estimate condition number can only return the measures "kappa", "lambda_max", "lambda_min". '
                      'You provided ' + str(measure))
        raise Exception

def steps_ahead(predictor:Callable, X:numpy.ndarray, omega:int) -> numpy.ndarray:
    X_i = X.copy()
    assert omega > 0
    for i in range(omega):
        y_i = predictor(X_i)
        if y_i.ndim < 2:
            y_i = numpy.expand_dims(y_i, axis=1)
        y_i_dim = y_i.shape[1]
        X_i[:, y_i_dim:] = X_i[:, 0:-y_i_dim]
        X_i[:, 0:y_i_dim] = y_i
    return y_i


def companion_matrix(ar_parameters: Sequence[float]) -> numpy.ndarray:
    """
    Build companion matrix for given parametervector
    :param ar_parameters: parameter vector of length process_order
    :return: process_order x process_order companion matrix
    """
    process_order = len(ar_parameters)
    matrix = numpy.eye(N=process_order, M=process_order, k=-1)
    matrix[0, :] = ar_parameters
    return matrix


def is_stationary(ar_parameters: Sequence[float]) -> bool:
    """
    Checks if given process is stationary using eigenvalue decomposition.
    :param ar_parameters: parameter vector of length process_order
    """
    comp_mat = companion_matrix(ar_parameters)
    eig_vals, _ = numpy.linalg.eig(comp_mat)
    spec_norm = numpy.max(numpy.abs(eig_vals))
    return spec_norm < 1


def is_var_stationary(ar_parameters: numpy.ndarray) -> bool:
    """
    Checks if given 2 dim VAR of lag 1 is stationary using eigenvalue decomposition.
    :param ar_parameters: parameter vector of size 2x2.
    """
    eig_vals, _ = numpy.linalg.eig(ar_parameters)
    spec_norm = numpy.max(numpy.abs(eig_vals))
    return spec_norm < 1

def draw_process(process_order: int) -> numpy.ndarray:
    """
    Draw uniformly random parameters for AR(p) process
    :param process_order: Order of process
    :return: parameter vector of length p
    """
    is_stat = False
    while not is_stat:
        ar_parameters = numpy.random.uniform(-max_parameter, max_parameter, size=process_order)  # draw random parameters of AR process
        if is_stationary(ar_parameters):  # return if stationary. Else draw new parameters
            is_stat = True
    return ar_parameters


def draw_var_process() -> numpy.ndarray:
    """
    Draw uniformly random parameters for a 2 dimensional VAR process with lag 1.
    """
    is_stat = False
    while not is_stat:
        ar_parameters = numpy.random.uniform(-max_parameter, max_parameter, size=(2, 2))  # draw random parameters of AR process
        if is_var_stationary(ar_parameters):  # return if stationary. Else draw new parameters
            is_stat = True
    return ar_parameters


def _spec_norm(ar_parameters: Sequence[float]) -> float:
    """
    Calculate spectral norm of companion matrix cossesponding to given AR parameters.
    Only used for debugging output.
    :param ar_parameters: parameter vector of length process_order
    """
    eig_val, eig_vec = numpy.linalg.eig(companion_matrix(ar_parameters))
    return numpy.max(numpy.abs(eig_val))


def _estimate_gamma1(ar_parameters: numpy.ndarray) -> float: # TODO remove?
    """
    Estimate autocorrelation of lag 1 by drawing a sample from the process and estimating it.
    I am not aware of closed from soultion for higher order processes.
    Only used for debugging output.
    :param ar_parameters: parameter vector of length process_order
    """
    dataset = generate_dataset(ar_parameters, 100, 0, 1, 1)
    x_train, y_train, x_test, y_test, x_inter_train, y_inter_train, x_inter_test, y_inter_test, sample, y_train_omega\
        = dataset
    cov = numpy.corrcoef(x_train[:-1], x_train[1:]) #TODO even correct?
    #print(x_train.shape)
    return cov[0, 1]

def estimate_gamma1_misspec(ar_parameters: numpy.ndarray) -> float:
    """
        Estimate autocorrelation of lag 1 by drawing a sample from the process and estimating it.
        I am not aware of closed from soultion for higher order processes.
        :param ar_parameters: 2x2 parameter matrix of var process
    """
    dataset = generate_misspec_dataset(ar_parameters, 100, 0, 1, 1)
    x_train, y_train, x_test, y_test, x_inter_train, y_inter_train, x_inter_test, y_inter_test, sample, y_train_omega\
        = dataset
    cov = numpy.corrcoef(x_train[:-1], x_train[1:])
    return cov[0, 1]

def estimate_correlation_between_dims(ar_parameters: numpy.ndarray) -> float:
    """
            Estimate correlation between two dimensions of 2-dim VAR process.
            I am not aware of closed from soultion.
            :param ar_parameters: 2x2 parameter matrix of var process
    """
    dataset = generate_var_dataset(ar_parameters, 1000, 0, 1, 1)
    x_train, y_train, x_test, y_test, x_inter_train, y_inter_train, x_inter_test, y_inter_test, sample, y_train_omega\
        = dataset
    cov = numpy.corrcoef(x_train[:-1, 1], x_train[1:, 0])
    return cov[0, 1]


def generate_dataset(real_proc_params: numpy.ndarray, num_samples_train: int, num_samples_test: int,
                     sigma_sq: float, omega: int = 1, interv_all : bool = False) -> Tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Generate design matrices and ground truth vectors for a sample from given AR process.
    Return design matrix for interventional and observational distribution as well as the sample vector.
    :param real_proc_params: parameter vector of length process_order
    :param num_samples_train: Number of samples in the training design matrix and labels
    :param num_samples_test: Number of samples in the test design matrix and labels
    :param sigma_sq: Variance of the generating process
    :param omega: How many timesteps to predict after x_{-1}, i.e. how many timesteps is the label vector shifted wrt to
     the design matrix
    :param interv_all: whether the intervention should independently set all p timesteps x_{t-1}, ..., x_{t-p}
     (if p is the order of the process or just x_{t-1}
    :return 7-tuple. Train design matrix, train lables, test design matrix, test lables, test design matrix with
    intervention, labels for test desing matrix with intervention, the whole sample as row vector
    """
    proc_order = real_proc_params.shape[0]
    sample = generate_sample(real_proc_params, num_samples_train + num_samples_test + omega - 1,
                             sigma_sq)  # sample plus initial p values

    sample_matrix = numpy.zeros((num_samples_train + num_samples_test, proc_order))
    for i in range(num_samples_train + num_samples_test):
        # Revert order of samples, so for the OLS estimators, a_1 is the parameter at index 0
        sample_matrix[i, :] = sample[i:i + proc_order][::-1]

    y_train = sample[proc_order:num_samples_train + proc_order]
    y_train_omega = sample[proc_order + omega - 1:num_samples_train + proc_order + omega - 1]
    x_train = sample_matrix[:num_samples_train, :]

    y_test = sample[proc_order + num_samples_train + omega - 1:]
    x_test = sample_matrix[num_samples_train:, :]

    interv = proc_order if interv_all else 1
    x_inter_test, y_inter_test = interv_from_observ_matrix(real_proc_params, x_test, sigma_sq, omega, intervene_num=interv)
    x_inter_train, y_inter_train = interv_from_observ_matrix(real_proc_params, x_train, sigma_sq, omega, intervene_num=interv)
    return x_train, y_train, x_test, y_test, x_inter_train, y_inter_train, x_inter_test, y_inter_test, sample, \
           y_train_omega


def generate_var_dataset(real_proc_params: numpy.ndarray, num_samples_train: int, num_samples_test: int,
                     sigma_sq: float, omega: int = 1) -> Tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Generate design matrices and ground truth vectors for a 2 dim VAR(1) process.
    Return design matrix for interventional and observational distribution as well as the sample vector.
    :param real_proc_params: parameter vector of length process_order
    :param num_samples_train: Number of samples in the training design matrix and labels
    :param num_samples_test: Number of samples in the test design matrix and labels
    :param sigma_sq: Variance of the generating process
    :param omega: How many timesteps to predict after x_{-1}, i.e. how many timesteps is the label vector shifted wrt to
     the design matrix
    :return 7-tuple. Train design matrix, train lables, test design matrix, test lables, test design matrix with
    intervention, labels for test desing matrix with intervention, the whole sample as row vector
    """

    sample = generate_var_sample(real_proc_params, num_samples_train + num_samples_test + omega - 1,
                             sigma_sq)  # sample plus initial p values

    sample_matrix = numpy.zeros((num_samples_train + num_samples_test, 2))

    y_train = sample[:, :num_samples_train].T
    y_train_omega = sample[:, omega:num_samples_train + omega].T
    y_test = sample[:, num_samples_train + omega:].T

    for i in range(num_samples_train + num_samples_test):
        sample_matrix[i, :] = sample[:, i].T

    x_train = sample_matrix[:num_samples_train, :]
    x_test = sample_matrix[num_samples_train:, :]

    # y_train is not shifted by omega, as we train to predict one step ahead
    # y_inter_train IS shifted, as we use it for hyperparameter seclection
    # Both y_inter_test and y_test are shifted.
    x_inter_test, y_inter_test = interv_from_observ_matrix_var(real_proc_params, x_test, sigma_sq, omega)
    x_inter_train, y_inter_train = interv_from_observ_matrix_var(real_proc_params, x_train, sigma_sq, omega)
    return x_train, y_train, x_test, y_test, x_inter_train, y_inter_train, x_inter_test, y_inter_test, sample,\
        y_train_omega

def generate_misspec_dataset(real_proc_params: numpy.ndarray, num_samples_train: int, num_samples_test: int,
                     sigma_sq: float, omega: int = 1) -> Tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Generate design matrices and ground truth vectors for a sample with a hidden confounder..
    Return design matrix for interventional and observational distribution as well as the sample vector.
    :param real_proc_params: parameter vector of length process_order
    :param num_samples_train: Number of samples in the training design matrix and labels
    :param num_samples_test: Number of samples in the test design matrix and labels
    :param sigma_sq: Variance of the generating process
    :param omega: How many timesteps to predict after x_{-1}, i.e. how many timesteps is the label vector shifted wrt to
     the design matrix
    :return 7-tuple. Train design matrix, train lables, test design matrix, test lables, test design matrix with
    intervention, labels for test desing matrix with intervention, the whole sample as row vector
    """
    x_train, y_train, x_test, y_test, x_inter_train, y_inter_train, x_inter_test, y_inter_test, sample, y_train_omega =\
        generate_var_dataset(real_proc_params, num_samples_train, num_samples_test, sigma_sq, omega)
    return x_train[:, 0], y_train[:, 0], x_test[:, 0], y_test[:, 0], x_inter_train[:, 0], y_inter_train[:, 0],\
           x_inter_test[:, 0], y_inter_test[:, 0], sample[:, 0], y_train_omega[:, 0]

def generate_sample(process_params: numpy.ndarray, sample_size: int, sigma_sq: float,
                    burnin: int = 100) -> numpy.ndarray:
    """
    Generate time series sample for given AR parameters.
    :param process_params: parameter vector of length process_order
    :param sample_size: length of the generated sample
    :param sigma_sq: Variance of the generating process
    :param burnin: Number of samples to generate and throw away at the beginning, to make result independent from
    initial values
    :return Sample array of length sample_size + proc_order
    """
    order = len(process_params)
    initial_values = numpy.random.uniform(-3, 3, size=order)
    # Allocate array that will contain burnin, sample and 'initial' values
    # (if burnin != 0 not really the initial values)
    sample = numpy.zeros(order + sample_size + burnin)
    sample[:order] = initial_values
    for i in range(sample_size + burnin):     # TODO also use steps_ahead?
        # Revert parameter vector, so a_1 is actually multiplied with x_{t-1} and not x_{t-p}
        x_i = numpy.dot(numpy.array(sample[i:i + order]), process_params[::-1]) + numpy.random.normal(0, sigma_sq)
        sample[i + order] = x_i
    return sample[burnin:]


def generate_var_sample(process_params: numpy.ndarray, sample_size: int, sigma_sq: float,
                    burnin: int = 100) -> numpy.ndarray:
    """
    Generate time series sample for given 2 dim VAR(1) parameters.
    :param process_params: parameter matrix of size 2x2
    :param sample_size: length of the generated sample
    :param sigma_sq: Variance of the generating process
    :param burnin: Number of samples to generate and throw away at the beginning, to make result independent from
    initial values
    :return Sample matrix of size 2x(sample_size + proc_order)
    """
    initial_values = numpy.random.uniform(-3, 3, size=(2,))
    # Allocate array that will contain burnin, sample and 'initial' values
    # (if burnin != 0 not really the initial values)
    sample = numpy.zeros(shape=(2, 1 + sample_size + burnin))
    sample[:, 0] = initial_values
    for i in range(1, sample_size + burnin):    # TODO also use steps_ahead?
        x_i = numpy.dot(process_params, sample[:, i]) + numpy.random.normal(0, sigma_sq, size=(2,))
        sample[:, i + 1] = x_i
    return sample[:, burnin:]


def interv_from_observ_matrix(process_params: numpy.ndarray, x_obs: numpy.ndarray, sigma_sq: float, omega: int = 1,
                              burnin: int = 100, intervene_num: int = 1) -> Tuple[
    numpy.ndarray, numpy.ndarray]:
    """
    Generate interventional sample from given parameters and observational sample.
    Right now, all interventions are made at time t-1.
    :param process_params: parameter vector of length process_order
    :param x_obs: Design matrix on which the intervention is made
    :param sigma_sq: Variance of the generating process
    :param omega: How many timesteps to predict after x_{-1}, i.e. how many timesteps is the label vector shifted wrt to
     the design matrix
    :param burnin: Number of samples to generate and throw away at the beginning, to make result independent from
    initial values
    :param intervene_num: Number of timesteps to intervene on. Interventions are made consecutively, starting at x_{t-1}
    :return Interventional design matrix and interventional label vector
    """
    proc_order = x_obs.shape[1]
    sample_size = x_obs.shape[0]
    x_int = x_obs.copy()
    for i in range(intervene_num):
        helper_sample = generate_sample(process_params, sample_size, sigma_sq, burnin)[proc_order:]
        helper_sample = numpy.random.permutation(helper_sample)  # TODO unnecessarily complex?
        x_int[:, i] = helper_sample.T

    y_int = steps_ahead(lambda x: numpy.dot(x, process_params) + numpy.random.normal(0, sigma_sq, sample_size),
                x_int, omega)
    return x_int, y_int


def interv_from_observ_matrix_var(process_params: numpy.ndarray, x_obs: numpy.ndarray, sigma_sq: float, omega: int = 1,
                              burnin: int = 100, intervene_num: int = 1) -> Tuple[
    numpy.ndarray, numpy.ndarray]:
    """
    Generate interventional sample from given parameters and observational sample.
    Right now, all interventions are made at time t-1.
    :param process_params: parameter vector of length process_order
    :param x_obs: Design matrix on which the intervention is made
    :param sigma_sq: Variance of the generating process
    :param omega: How many timesteps to predict after x_{-1}, i.e. how many timesteps is the label vector shifted wrt to
     the design matrix
    :param burnin: Number of samples to generate and throw away at the beginning, to make result independent from
    initial values
    :param intervene_num: Number of timesteps to intervene on. Interventions are made consecutively, starting at x_{t-1}
    :return Interventional design matrix and interventional label vector
    """

    sample_size = x_obs.shape[0]
    x_int = x_obs.copy()
    for i in range(intervene_num):
        helper_sample = generate_var_sample(process_params, sample_size, sigma_sq, burnin)[i, 1:]
        helper_sample = numpy.random.permutation(helper_sample)  # TODO unnecessarily complex?

        x_int[:, i] = helper_sample

    y_int = steps_ahead(lambda x: numpy.dot(process_params, x.T).T
                                  + numpy.random.normal(0, sigma_sq, (sample_size, 2)), x_int, omega)
    return x_int, y_int
