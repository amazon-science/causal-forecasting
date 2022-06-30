# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import sys


module_path = "/home/ec2-user/SageMaker/code"
if module_path not in sys.path:
    sys.path.append(module_path)

import eval_model_dataset

from gluonts.dataset import common as gluondata


def test_interventional_dataset():
    test_data = np.arange(100).reshape([10, 10])
    freq = "1H"
    start = pd.Timestamp("01-01-2022", freq=freq)
    test_dataset = gluondata.ListDataset(
        [{'target': x, 'start': start} for x in test_data],
        freq=freq
    )
    prediction_length = 4
    for window in [1, 3]:
        interv_data = eval_model_dataset.create_interventional_dataset(
            test_dataset, prediction_length=prediction_length, feature_name=None, window=window)

        window_sums = [0.0 for _ in range(window)]
        for idx, (test_row, interv_row) in enumerate(zip(test_dataset, interv_data)):
            print("real")
            print(test_row)
            print("interv")
            print(interv_row)
            np.testing.assert_allclose(test_row['target'][:-prediction_length - window], interv_row['target'][: -window])

            for idx in range(window):
                window_sums[idx] += interv_row['target'][- window + idx]
                
        for idx in range(window):
            np.testing.assert_allclose(
                window_sums[idx] / 10.0, np.mean(test_data[:,-prediction_length - window + idx]),
                rtol=0.1)
