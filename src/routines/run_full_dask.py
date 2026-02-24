"""
Dask-based script for running parallel topic decomposition on chromatin tracing data using topic3d.

This script:
1. Parses CLI arguments.
2. Loads raw or preprocessed proximity data.
3. Initializes the output directory structure, configuration and logging if indicated.
4. Prepares data for topic modeling by computing proximity vectors.
5. Optionally splits data into training and testing subsets.
6. Trains an LDA model and saves results.

An example run:

$ topic3d-run-dask \
    --data_file path/to/data/file.npy \
    --results_base_dir path/to/create/results/folder \
    --tag 'datasetname-tests' \
    --save_logs \
    --proximity_threshold 150 \
    --num_components 10,20,30 \
    --max_iter 500 \
    --train_data_proportion 0.75 \
    --num_jobs 10

This script is intended to be called from the CLI wrapper `topic3d-run-dask` and not directly by end users.

Dependencies:
- topic3d.workflow._parsers
- topic3d.workflow._dataops
- topic3d.workflow.training
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from copy import deepcopy
import time

from dask import config
from dask.distributed import Client, LocalCluster
from distributed import as_completed
# from concurrent.futures import ProcessPoolExecutor

from topic3d.workflow._parsers import parse_cli_arguments_full
from topic3d.workflow._dataops import load_data_file, prepare_proximity_data, split_proximity_data
from topic3d.workflow.training import initialize_full, decompose

config.set({'distributed.worker.daemon': False})


def print_text(text):
    print(str(text))

def run_decomposition_pipeline(args):

    print(f"Running the main pipeline")

    # Step 1: Load data using internal workflow API
    data = load_data_file(path_to_file=args.data_file)
    
    # Step 2: Convert Namespace to dict for supplying parameters as required by the API
    params = dict(vars(args))   # Generally params do not have all CLI arguments, this is just for easy scripting here

    # Step 3: Initialize run metadata and output directories
    args_full = initialize_full(params=params,
                           dir_results=args.results_base_dir,
                           path_to_data=args.data_file,
                           tag=getattr(args, "tag"),
                           save_logs=getattr(args, "save_logs"))
    
    # Step 4: Prepare proximity vectors from trace data and save
    data_prepped = prepare_proximity_data(data=data,
                                          params=params,
                                          dir_to_save=args_full["path_to_training"],
                                          uid=args_full.get("uid", None))

    # Step 5: Split into training and testing sets (if indicated in args)
    data_train, data_test = split_proximity_data(data=data_prepped,
                                                 train_data_proportion=params["train_data_proportion"],
                                                 uid=args_full.get("uid", None))

    # Step 7: Run LDA decomposition on the processed data
    _ = decompose(params=params,
                  data_train=data_train,
                  data_test=data_test,
                  dir_to_save=args_full["path_to_training"],
                  summary=args_full,
                  uid=args_full.get("uid", None))


# Function to run the decompositions in parallel
def main():

    # Parse command-line arguments
    args = parse_cli_arguments_full(parallelization=True)
    num_jobs_total = args.num_jobs * len(args.num_components)
    print(f"Total number of jobs: {num_jobs_total}")

    start_time = time.perf_counter()
    
    with LocalCluster(n_workers=num_jobs_total, threads_per_worker=1, processes=True) as cluster:
        
        with Client(cluster) as client:
            
            # time.sleep(1)
            # print(client.scheduler_info()["workers"])
            print(f"[DASK] Actual number of workers: {len(client.scheduler_info()["workers"])}")

            futures = []
            n_comps = [item for item in args.num_components for _ in range(args.num_jobs)]
            for index in range(num_jobs_total):
                # Modify args to reflect single job parameters
                args_filt = deepcopy(vars(args))
                args_filt["num_components"] = n_comps[index]
                del args_filt["num_jobs"]
                args_filt = argparse.Namespace(**args_filt)
                # Submitting jobs to run in parallel
                print(f"  [DASK] Spinning job {index:02} for {n_comps[index]} components")
                future = client.submit(
                    run_decomposition_pipeline,
                    args_filt
                )
                futures.append(future)
            print(client.run(lambda: "I'm alive!"))

            # # This will either return or raise; it won't silently hang
            # results = client.gather(futures)
            # for rs in results:
            #     if rs is not None:
            #         print(rs)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"[ERROR] Task failed: {e}")

            print("[DASK] Finished collecting results from all workers!")

    end_time = time.perf_counter()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()