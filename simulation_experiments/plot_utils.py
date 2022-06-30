# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas
import numpy
from causal_generalization import estimate_condition_number, estimate_condition_number_misspec

# adapted from https://joseph-long.com/writing/colorbars/
def colorbar(mappable, size='10%', label=r'Absolute Autocorrelation $|\gamma_1|$', tick_size=12, label_size=13):
    last_axes = pyplot.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, label=label)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.yaxis.label.set_size(label_size)
    cbar.solids.set(alpha=1)
    pyplot.sca(last_axes)
    return cbar

def get_autocorrelation(params, order, data_prefix, data_suffix, measure='kappa'):
    """
    Calculate autocorrelation for processes. Cache result in csv file.
    As of yet the script does not detect on its own if the data has changed, so g1.csv has to be reset manually if the
    experiments are changed.
    """
    try:  # Is there a cache file?
        g_frame = pandas.read_csv(data_prefix + 'g1' + data_suffix)
    except:
        g_frame = pandas.DataFrame()

    if measure in g_frame.keys(): # has autocorrelation been cached?
        return g_frame[measure].to_numpy()
    else:
        # Calculate true autocorrelation
        if len(params.keys()) == 2:
            logging.warning('Returns true autocorr for AR(2) process, regardless of "measure" argument!')
            g1 = numpy.abs(params.apply(lambda row: row['a1'] / (1 - row['a2']), axis=1))
            g_frame[measure] = g1
        elif order == 'misspec':
            g1 = numpy.abs(
                params.apply(lambda row: estimate_condition_number_misspec(row.to_numpy().reshape(2, 2), measure),
                             axis=1))
            g_frame[measure] = g1
        else:
            g1 = numpy.abs(params.apply(lambda row: estimate_condition_number(row.to_numpy(), measure), axis=1))
            g_frame[measure] = g1

        g_frame.to_csv(data_prefix + 'g1' + data_suffix, index=False)
    return g1.to_numpy()


def generalization_bound(autocorrelation, stat_err, num_samples, rademacher):
    """
    Estimate generalization bound from Theorem 1.
    """
    zeta = 1 + autocorrelation
    M = numpy.max(stat_err)
    delta = .5
    m = numpy.sqrt(num_samples)
    mu = numpy.sqrt(num_samples)
    delta_prime = delta - 2 * (mu - 1) * (0.1 ** m)
    bound = zeta * stat_err + zeta * rademacher \
            + 3 * zeta * M * numpy.sqrt((numpy.log(4 / delta_prime)) / (2 * mu))
    return bound.to_numpy()
