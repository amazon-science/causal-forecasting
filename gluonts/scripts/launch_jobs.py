# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import re
import sys
import time
    
from sagemaker.mxnet import MXNet
from sagemaker import get_execution_role

role = get_execution_role()

# TODO: consider taking those as args.
ESTIMATORS = ["deep_ar", "wavenet", "transformer"]
NUM_REPITITION = 5
DATASETS = [
        #"electricity",
        "traffic", #"m4_hourly"
                   ]


def launch_jobs():
    # make sure we are running this in the right directory.
    assert 'eval_model_dataset.py' in os.listdir("code/")

    for dataset in DATASETS:
        print(dataset)
        data_location = f"s3://causal-generalization/gluon-datasets/{dataset}"
        for iteration in range(NUM_REPITITION):
            print(f"run {iteration}")
            for estimator_name in ESTIMATORS:
                job_base = re.sub(r'[^a-zA-Z0-9]', '', f"{dataset}-{estimator_name}-run-{iteration}")
                print(f"model {estimator_name}")
                output_location = f"s3://causal-generalization/gluon-experiments/v3{dataset}/{estimator_name}/runid-{iteration}"
                estimator = MXNet(
                    entry_point="eval_model_dataset.py",
                    source_dir='code/',
                    role=role,
                    output_path=output_location,
                    base_job_name=job_base,
                    instance_count=1,
                    instance_type="ml.c4.4xlarge", # more memory needed for traffic dataset.
                    framework_version="1.7.0",
                    py_version="py3",
                    hyperparameters={"estimator_name": estimator_name},
                )

                estimator.fit({"train": data_location}, wait=False)
                time.sleep(10)  # avoid throttling error
            
            
            
            
if __name__ == "__main__":

    launch_jobs()
