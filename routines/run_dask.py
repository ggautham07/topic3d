#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
"""
Example run:

./launch.py --num_components 10,20 --root_dir ~/grey/rawData_2024/Experiment_96_Topic_Gautham/topic_decomposition --tag hafner_chr6_CTCF_0 --num_jobs 2

go to http://localhost:8787/ to see output

to connect to remote server you need to remap the localhost, eg:

ssh -L 8888:localhost:8787 marcnol@lifou

here you go to http://localhost:8888/ to see output

"""

import argparse
import os
from dask.distributed import Client, LocalCluster
from topic3d.workflow.training import initialize, decompose
from concurrent.futures import ProcessPoolExecutor
import uuid
import time

# test this later to see if performance improves!
from dask import config

#config.set({'distributed.worker.daemon': False})
config.set({'distributed.worker.daemon': True}) # need to test this in cines!

# Helper function to parse a single integer or a list of integers
def parse_num_components(value):
    try:
        # Check if it's a single integer
        if ',' in value:
            # If it contains commas, split and parse as a list of integers
            return [int(v) for v in value.split(',')]
        else:
            # Otherwise, parse it as a single integer and return it as a list
            return [int(value)]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for num_components: {value}")


# Function to parse command line arguments
def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Run decomposition processes in parallel.")

    # Add arguments based on your provided list
    parser.add_argument("--root_dir", type=str, required=True, help="Working folder")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Contact threshold")
    parser.add_argument("--hardness", type=float, required=True, help="Contact threshold hardness", default=1)
    parser.add_argument("--num_components", type=parse_num_components, default=[10],
                        help="Number of components (single integer or a list of integers, e.g., 10 or 10,20,30)")
    parser.add_argument("--max_iter", type=int, choices=range(10, 10001), metavar="[10-10000]",
                        default=2000, help="Number of iterations (default=2000)")
    parser.add_argument("--learning_method", type=str, choices=["batch", "online"], default="batch",
                        help="Learning method (batch/online, default=batch)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size while using online learning (default=32), ignored for batch learning")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of parallel jobs (default=1)")
    parser.add_argument("--train_data_proportion", type=float, metavar="[0.1-1]", default=1,
                        help="Training data proportion (default=1)")
    parser.add_argument("--normalization_method", type=str, choices=[None, "MMSF", "R(s)"], default=None,
                        help="Method to normalize pairwise distances - dataset needs to exist in the processed data")
    parser.add_argument("--imputed", action="store_true", required=False)
    # Parse arguments
    return parser.parse_args()

# Function to run the decompositions in parallel
def main():

    args = parse_arguments()


    start_time = time.time()

    data_dir = os.path.join(args.root_dir, "processed/data")

    # Ensure the data folder exists
    if not os.path.isdir(data_dir):
        print(f"Error: Data folder {data_dir} does not exist.")
        return

    file_to_process = os.path.join(data_dir, f"{args.tag}.h5")

    """
    # Start a Dask local cluster
    n_jobs = args.num_jobs * len(args.num_components)
    # cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1)
    cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1, processes=True)
    client = Client(cluster)
    print(f"Actual number of workers: {len(client.scheduler_info()['workers'])}")

    # Schedule parallel execution using Dask
    futures = []
    n_components = [item for item in args.num_components for _ in range(args.num_jobs)]
    for index in range(n_jobs):
        unique_id = uuid.uuid4()  # This will make each task unique
        print(f" Spinning job {index} for {n_components[index]} components")
        future = client.submit(
            run_decomposition,
            args.root_dir,
            args.tag,
            file_to_process,
            n_components[index],
            args.threshold,
            args.hardness,
            args.max_iter,
            args.learning_method,
            args.batch_size,
            1,  # num_jobs fixed to one if running dask to avoid LDA from creating other processes
            args.train_data_proportion,
            unique_id,
        )
        futures.append(future)

    # Collect results from Dask futures
    for future in futures:
        result = future.result()  # This will block until each computation is done
        print(result)

    print(f"> Finished collecting results from all workers!")
    client.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    """
    n_jobs = args.num_jobs * len(args.num_components)

    with LocalCluster(n_workers=n_jobs, threads_per_worker=1, processes=True) as cluster:
        with Client(cluster) as client:
            print(f"Actual number of workers: {len(client.scheduler_info()['workers'])}")

            futures = []
            n_components = [item for item in args.num_components for _ in range(args.num_jobs)]
            for index in range(n_jobs):
                unique_id = uuid.uuid4()
                print(f" Spinning job {index} for {n_components[index]} components")
                future = client.submit(
                    run_decomposition,
                    args.root_dir,
                    args.tag,
                    file_to_process,
                    n_components[index],
                    args.normalization_method,
                    args.imputed,
                    args.threshold,
                    args.hardness,
                    args.max_iter,
                    args.learning_method,
                    args.batch_size,
                    1,
                    args.train_data_proportion,
                    unique_id,
                )
                futures.append(future)

            # This will either return or raise; it won't silently hang
            results = client.gather(futures)
            for r in results:
                print(r)

            print("> Finished collecting results from all workers!")

    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()