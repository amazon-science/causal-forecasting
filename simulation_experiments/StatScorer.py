# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from typing import Type
from causal_generalization import steps_ahead


class StatScorer:
    """
    Scores estimator by measuring the mse made on this observational data.

    A bit hacky to be able to train one step ahead, but evaluate omega steps ahead.
    """
    def __init__(self, X_train: numpy.ndarray, y_omega : numpy.ndarray, omega:int = 1):
        self.omega = omega
        self.row_lookup = {}    # lookup table to find interventional sample for observational sample
        for i in range(X_train.shape[0]):
            self.row_lookup[tuple(X_train[i, :])] = y_omega[i]

    def __call__(self, estimator: Type[BaseEstimator], X: numpy.ndarray, y: numpy.ndarray):
        y_omega = [self.row_lookup[tuple(X[i, :])] for i in range(X.shape[0])]

        y_hat = steps_ahead(estimator.predict, X, self.omega)
        return - mean_squared_error(y_omega, y_hat)
