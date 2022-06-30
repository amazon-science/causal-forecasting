# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy
import matplotlib.pyplot as pyplot
import pandas
from plot_utils import get_autocorrelation, generalization_bound
from pathlib import Path

label_size = 13
tick_size = 12

seed = str(0)
num_runs = 10000
est = 'OLS'
interv_list = ['', '_interv_all']
order = 'misspec'
num_samples_train = 100
omega_list = [1, 5, 7]
img_prefix = '../img/uai/order{}/'.format(order)

fig, ax = pyplot.subplots(len(interv_list), len(omega_list), figsize=(4*len(omega_list) + 2, 3.2*len(interv_list)),
                          sharey=True, sharex=True)
if len(interv_list) < 2:
    ax = numpy.expand_dims(ax, axis=0)
if len(omega_list) < 2:
    ax = numpy.expand_dims(ax, axis=1)

for i, interv in enumerate(interv_list):
    for j, omega in enumerate(omega_list):
        axi = ax[i, j]
        data_suffix = '_' + str(num_runs) + '_' + str(num_samples_train) + '_' + str(omega) + '_' + str(
            seed) + interv + '.csv'
        img_suffix = '_' + str(num_runs) + '_' + str(num_samples_train) + '_' + str(omega) + '_' + str(seed) + '_all.png'

        if order == 'misspec':
            data_prefix = '../data/uai/misspec/'
        else:
            data_prefix = '../data/uai/order{}/'.format(order)
        Path(img_prefix).mkdir(parents=True, exist_ok=True)

        params = pandas.read_csv(data_prefix + 'params' + data_suffix)
        caus_err = pandas.read_csv(data_prefix + 'error_causal_lambda_stat' + data_suffix)
        stat_err = pandas.read_csv(data_prefix + 'error_stat_lambda_stat' + data_suffix)
        rademacher = pandas.read_csv(data_prefix + 'emp_rademacher' + data_suffix)

        autocorr = get_autocorrelation(params, order, data_prefix, data_suffix)
        bound = generalization_bound(autocorr, stat_err[est], num_samples_train, rademacher[est])
        # Bound is on causal error. We want to plot difference to stat err.
        bound = numpy.abs(bound - stat_err[est]).to_numpy()

        # Group points into 20 buckets according to kappa.
        # We only want to plot the maximum error difference per bucket
        error = pandas.DataFrame()
        error['diff'] = numpy.abs(stat_err[est] - caus_err[est])
        error['buckets'] = pandas.qcut(autocorr, 20)
        error = error.groupby('buckets')
        idx = error.idxmax()['diff']
        # Only plot points where error difference is maximal for each bucket
        bound = bound[idx]
        autocorr = autocorr[idx]
        error_max = error['diff'].max()
        error_mean = error['diff'].mean()
        lower, upper = [], []
        for _, group in error:
            l, u = numpy.quantile(group['diff'], [.5, .95])
            lower.append(l)
            upper.append(u)
        lower, upper = numpy.array(lower), numpy.array(upper)

        corr_idx = numpy.argsort(autocorr)  # Sort autocorr so we can use plot (instead of scatter) for bound

        # Plot showing the scatter plot of block-maxima of the difference between causal and statistical error and the
        # generalization bound from Theorem 1 against the condition number kappa.
        # ------------------
        # Prepare axis labels
        axi.tick_params(labelsize=tick_size)
        axi.set_ylabel(est + r': Difference in Errors', size=label_size)
        axi.set_xlabel(r'$\kappa$', size=label_size)
        axi.set_yscale('log')
        axi.set_xscale('log')

        # Actual plot
        axi.fill_between(autocorr[corr_idx], lower[corr_idx], upper[corr_idx],
                         label='90% Quantile of $|\mathcal{G} - \mathcal{S}|$', alpha=.5)
        axi.plot(autocorr[corr_idx], error_mean[corr_idx], '--', c='C0', label=r'Mean $|\mathcal{G} - \mathcal{S}|$')
        axi.plot(autocorr[corr_idx], error_max[corr_idx], c='C0', label=r'Maximal $|\mathcal{G} - \mathcal{S}|$')
        axi.plot(autocorr[corr_idx], bound[corr_idx], c='C1', label='Causal Error Bound')
        axi.legend(loc='upper left')

fig.tight_layout()

# Save and show
pyplot.savefig(img_prefix + 'correlation_vs_error_omega' + img_suffix, bbox_inches='tight')
pyplot.show()
pyplot.close()
