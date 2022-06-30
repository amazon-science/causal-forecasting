# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn import metrics
import math
from pathlib import Path
import json
import os

import argparse
import logging

from gluonts.dataset import common as gluondata
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.dataset import common as gluondata
from gluonts.evaluation import Evaluator
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts import model
from gluonts.mx import Trainer
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.transformer import TransformerEstimator

from gluonts.transform import AdhocTransform


NUM_SAMPLES = 1000
WINDOW = 1

# for a given dataset cuts it off the last prediction_lengths steps and shuffles the
# window steps before that across the examples. 
def create_interventional_dataset(dataset, prediction_length, feature_name=None, window=1):
    # TODO: consider creating one interventional dataset and use the same dataset for evaluating
    #       different models.
    one_dim_target = dataset.process.trans[1].req_ndim == 1
    freq = dataset.process.trans[0].freq

    if feature_name is None:
        assert one_dim_target
        feature_name = "target"
    all_data = list(dataset)
    num_rows = len(dataset)
    
    shuffled_idxs_by_window = []
    for _ in range(window):
        row_idxs = np.array(range(num_rows))
        shuffled_idxs_by_window.append(row_idxs)
        np.random.shuffle(row_idxs)

    intervened_data = []
    for idx, row in enumerate(dataset):
        row = row.copy()
        # cut data off prediction length of dataset (for consistency with what input / output we use for the statistical error)
        row[feature_name] = row[feature_name][..., : -prediction_length]
        for window_idx in range(window):
            intervention = all_data[shuffled_idxs_by_window[window_idx][idx]][feature_name][..., -prediction_length - window + window_idx]
            row[feature_name][..., -window  + window_idx] = intervention
        intervened_data.append(row)
    return gluondata.ListDataset(intervened_data, freq, one_dim_target)


def create_interventional_same_row_dataset(dataset, prediction_length, feature_name=None, window=1):
    one_dim_target = dataset.process.trans[1].req_ndim == 1
    freq = dataset.process.trans[0].freq

    if feature_name is None:
        assert one_dim_target
        feature_name = "target"
    intervened_data = []
    for idx, row in enumerate(dataset):
        row = row.copy()
        # cut data off prediction length of dataset (for consistency with what input / output we use for the statistical error)
        row[feature_name] = row[feature_name][..., : -prediction_length]
        rnd_idx = np.random.choice(len(row[feature_name]) - window)
        intervention = row[feature_name][..., rnd_idx : rnd_idx + window]
        row[feature_name][..., -window : ] = intervention
        intervened_data.append(row)
    return gluondata.ListDataset(intervened_data, freq, one_dim_target)
   
    
    
def eval_rmse(forecast1, forecast2, omega=1):
    f1_at_omega = [f.mean[omega] for f in forecast1]
    f2_at_omega = [f.mean[omega] for f in forecast2]
    return math.sqrt(metrics.mean_squared_error(f1_at_omega, f2_at_omega))


def eval_predictor(predictor, dataset):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,  # test dataset
        predictor=predictor,  # predictor
        num_samples=NUM_SAMPLES,
    )
    forecast_it = list(forecast_it)
    ts_it = list(ts_it)
    evaluator = Evaluator(chunk_size=2)
    agg_metrics, _ = evaluator(ts_it, forecast_it, num_series=len(dataset.test))
    rmse = agg_metrics["RMSE"]
    return forecast_it, rmse, uncertainty_prediction(forecast_it)


def uncertainty_prediction(forecast):
    CI_width = []
    for entry in forecast:
        CI_width.append(np.mean(np.absolute(entry.quantile(0.9) - entry.quantile(0.1))))
    return np.mean(CI_width)

def compare(predictor1, predictor2, dataset, dataset_intervened1, dataset_intervened2):
    forecast_it1, rmse1, uncertainty1 = eval_predictor(predictor1, dataset)
    forecast_it2, rmse2, uncertainty2 = eval_predictor(predictor2, dataset)
    disagreement = eval_rmse(forecast_it1, forecast_it2)
    
    results = {
        "RMSE1": rmse1,
        "RMSE2": rmse2,
        "RMSE1vs2": disagreement,
        "80_CI_width_1": uncertainty1,
        "80_CI_width_2": uncertainty2,
    }
    
    for i, intervention in enumerate([dataset_intervened1, dataset_intervened2]):
        if intervention is None:
            continue
        forecast_intervened1 = list(predictor1.predict(intervention, num_samples=NUM_SAMPLES))
        forecast_intervened2 = list(predictor2.predict(intervention, num_samples=NUM_SAMPLES))
        intervened_disagreement = eval_rmse(forecast_intervened1, forecast_intervened2)
        uncertainty_intervened1 = uncertainty_prediction(forecast_intervened1)
        uncertainty_intervened2 = uncertainty_prediction(forecast_intervened2)
        results[f"interventional{i+1}_RMSE_pred1vs2"] = intervened_disagreement
        results[f"80_CI_width_intervened{i+1}_pred1"] = uncertainty_intervened1
        results[f"80_CI_width_intervened{i+1}_pred2"] = uncertainty_intervened2
    return {k: float(v) for (k,v) in results.items()}


def save_to_dir(predictor, directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        predictor.serialize(Path(directory))
    except:
        print("cannot save predictor")


def get_estimator(estimator_name, dataset):
    if estimator_name == "simple_ff":
        return SimpleFeedForwardEstimator(
                prediction_length=dataset.metadata.prediction_length,
                freq=dataset.metadata.freq,
                trainer=Trainer(
                    ctx="cpu",
                    epochs=1,
                    learning_rate=1e-3,
                    batch_size=32,
                ),
            )
    if estimator_name == "deep_ar":
        return DeepAREstimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
        )
    if estimator_name == "wavenet":
        return WaveNetEstimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
        )
    if estimator_name == "transformer":
        return TransformerEstimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
        )
    raise ValueError("unknown estimator " + estimator_name)



def find_file(root_path, file_name):
    for root, dirs, files in os.walk(root_path):
        if file_name in files:
            return os.path.join(root, file_name)
        

def load_data(path):
    return gluondata.load_datasets(
        metadata=path,
        train=os.path.join(path, "train"),
        test=os.path.join(path, "test")
    )


def train_and_eval(estimator_name, dataset_path, model_dir):
    dataset = load_data(dataset_path)
    predictor1 = get_estimator(estimator_name, dataset).train(dataset.train)
    predictor2 = get_estimator(estimator_name, dataset).train(dataset.train)
    save_to_dir(predictor1, os.path.join(model_dir, f"model1/"))
    save_to_dir(predictor2, os.path.join(model_dir, f"model2/"))
    print("saved")
    # 2. create interventional dataset
    dataset_intervened = create_interventional_dataset(
        dataset.test, predictor1.prediction_length, window=WINDOW)
    dataset_intervened2 = create_interventional_same_row_dataset(
        dataset.test, predictor1.prediction_length, feature_name=None, window=WINDOW)
    # 3. eval predictors
    print("eval")
    res = compare(predictor1, predictor2, dataset, dataset_intervened, dataset_intervened2)
    print(res)
    with open(os.path.join(model_dir, "results.json"), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--estimator_name", type=str, default="simple_ff")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_and_eval(
        args.estimator_name,
        args.train,
        args.model_dir,
    )
