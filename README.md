
This is the code to reproduce the experiments for the paper 'Causal Forecasting: Generalization Bounds for Autoregressive Models'.

### Prerequisites
Python 3

### How to run
To install necessary dependencies call
```
pip3 install -r requirements.txt
```
#### Simulation Experiments
Run the simulation experiments with
```
cd simulation_experiments
python3 compare_estimators.py
```
You can also use different parameters, e.g.
```
python3 compare_estimators.py --seed=1
```
For a comprehensive list of parameters use the help flag.
The results will be saved in the `data` directory. To plot the results run
```
python3 correlation_vs_error_plot.py
python3 corr_vs_err_omega.py
python3 error_diff_hist.py
python3 error_stat_vs_causal.py
python3 sample_size_vs_error.py
python3 plot_corr_vs_err_misspec.py
```
Each file corresponds to a figure in the paper.
The resulting plots will also be stored in the `img` directory.

#### Real Data Experiments
Train the models remotly by using the script
```
cd gluonts
./scripts/launch_remote_jobs.sh
```
Then the plots from the paper can be created with the notebooks
```
gluonts/gluonts.ipynb
gluonts/remote-jobs-gluonts.ipynb
```

### License

This project is licensed under the Apache-2.0 License.
