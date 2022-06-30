# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as pyplot
import pandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from plot_utils import get_autocorrelation


# adapted from https://joseph-long.com/writing/colorbars/
def colorbar(mappable, size='10%', label=r'Absolute Autocorrelation $|\gamma_1|$'):
    last_axes = pyplot.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, label=label)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.ax.yaxis.label.set_size(label_size)
    pyplot.sca(last_axes)
    return cbar


label_size = 13
tick_size = 12

seed = str(0)
num_runs = 10000
num_samples_train = 101
omega = 1

data_suffix = '_' + str(num_runs) + '_' + str(num_samples_train) + '_' + str(omega) + '_' + str(seed) + '.csv'
img_suffix = '_' + str(num_runs) + '_' + str(num_samples_train) + '_' + str(omega) + '_' + str(seed) + '_misspec' + \
             '.png'
img_prefix = '../img/aistats/misspec/'

order = 'misspec'
est_list = ['OLS', 'Ridge', 'Lasso', 'ElasticNet']

fig, ax = pyplot.subplots(1, len(est_list), figsize=(14, 3.2), sharex=True, sharey=True)
for est, axi in zip(est_list, ax):
    if order == 'misspec':
        data_prefix = '../data/aistats/misspec/'
    else:
        data_prefix = '../data/aistats/order{}/'.format(order)
    Path(img_prefix).mkdir(parents=True, exist_ok=True)

    params = pandas.read_csv(data_prefix + 'compare_params' + data_suffix)
    caus_err_lambda_stat = pandas.read_csv(data_prefix + 'compare_causal_stat' + data_suffix)
    stat_err_lambda_stat = pandas.read_csv(data_prefix + 'compare_stat_stat' + data_suffix)

    autocorr = get_autocorrelation(params[est], order, data_prefix, data_suffix)

    # Plot showing the scatter plot of the lambda_max vs quotient causal / stat error
    # ------------------
    # Prepare axis and labels
    axi.tick_params(labelsize=tick_size)
    axi.set_ylabel(est + r': Error quotient $\mathcal{G} / \mathcal{S}$', size=label_size)
    axi.set_xlabel(r'$\lambda_{max}$', size=label_size)
    axi.set_yscale('log')
    axi.set_xscale('linear')
    # Actual plot
    cmap = axi.scatter(autocorr, caus_err_lambda_stat[est] / stat_err_lambda_stat[est])

fig.tight_layout()
# Save and show
pyplot.savefig(img_prefix + 'correlation_vs_error' + img_suffix, bbox_inches='tight')
pyplot.show()
pyplot.close()
