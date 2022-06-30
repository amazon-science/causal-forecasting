# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy
import matplotlib.pyplot as pyplot
import pandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path


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
est_list = ['Ridge']
sample_list = [10, 101, 1000]
omega = 1
order = 5

img_prefix = '../img/aistats/order{}/'.format(order)
img_suffix = '_' + str(num_runs) + '_' + str(omega) + '_' + str(seed) + '.png'

fig, ax = pyplot.subplots(1, len(est_list), figsize=(7, 3.2), sharey=True)
if len(est_list) == 1:
    ax = [ax]
for est, axi in zip(est_list, ax):

    if order == 'misspec':
        data_prefix = '../data/aistats/misspec/'
    else:
        data_prefix = '../data/aistats/order{}/'.format(order)
    Path(img_prefix).mkdir(parents=True, exist_ok=True)

    cause_err = []
    stat_err = []
    for num_samples_train in sample_list:
        data_suffix = '_' + str(num_runs) + '_' + str(num_samples_train) + '_' + str(omega) + '_' + str(
            seed) + '.csv'
        caus_err_lambda_stat = pandas.read_csv(data_prefix + 'compare_causal_stat' + data_suffix)
        cause_err.append(caus_err_lambda_stat[est])
        stat_err_lambda_stat = pandas.read_csv(data_prefix + 'compare_stat_stat' + data_suffix)
        stat_err.append(stat_err_lambda_stat[est])
    cause_err = numpy.array(cause_err)
    stat_err = numpy.array(stat_err)

    # Violin plot showing the difference |G - S| at different sample sizes
    # ------------------
    # Prepare labels and axis
    axi.tick_params(labelsize=tick_size)
    axi.set_ylabel(est + r': Error Diff $|\mathcal{G} - \mathcal{S}|$', size=label_size)
    axi.set_xlabel(r'Size of Training Sample', size=label_size)
    axi.set_yscale('log')
    axi.set_xscale('linear')
    axi.set_xticks(numpy.arange(1, len(sample_list) + 1))
    axi.set_xticklabels([str(s) for s in sample_list])
    # Actual plot
    data = numpy.abs((cause_err - stat_err)).T
    print('Mean: ', numpy.mean(data, axis=0))
    print('Std: ', numpy.std(data, axis=0))
    cmap = axi.violinplot(data, showextrema=True, showmeans=False, showmedians=True, points=10000)
    # Plot Quartile
    quartile1, medians, quartile3 = numpy.percentile(data, [25, 50, 75], axis=0)
    axi.vlines(numpy.arange(1, len(sample_list)+1), quartile1, quartile3, color='k', linestyle='-', lw=5)

fig.tight_layout()
# Save and show
pyplot.savefig(img_prefix + 'sample_size_vs_error' + img_suffix, bbox_inches='tight')
pyplot.show()
pyplot.close()
