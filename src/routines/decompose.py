"""
Main script for running topic decomposition on chromatin tracing data using topic3d.

This script:
1. Parses CLI arguments.
2. Loads proximity data for training.
3. Trains an LDA model and returns results.

An example run:

$ topic3d-decompose \
    --data_file path/to/data/file.npy \
    --num_components 10 \
    --max_iter 100 \

This script is intended to be called from the CLI wrapper `topic3d-decompose` and not directly by end users.

Dependencies:
- topic3d.workflow._parsers
- topic3d.workflow._dataops
- topic3d.workflow.training
"""


from topic3d.workflow._parsers import parse_cli_arguments_simple
from topic3d.workflow._dataops import load_data, load_model_file
from topic3d.workflow.training import initialize_simple, decompose


def main():
    """Execute the topic decomposition pipeline based on CLI arguments."""

    # Step 1: Parse command-line arguments
    args = parse_cli_arguments_simple()

    # Step 2: Convert Namespace to dict for supplying parameters as required by the API
    params = dict(vars(args))   # Generally params do not have all CLI arguments, this is just for easy scripting here

    # Step 3: Load data and using internal workflow API
    data_train = None if args.data_train_file is None else load_data(path_to_file=args.data_train_file)
    data_test = None if args.data_test_file is None else load_data(path_to_file=args.data_test_file)
    existing_model = None if args.model_file is None else load_model_file(args.model_file)

    # Step 4: Initialize run metadata and output directories if needed
    args_full = initialize_simple(params=params,
                                  dir_results=args.results_dir,
                                  path_to_data=[args.data_train_file, args.data_test_file],)

    # Step 5: Run LDA decomposition on the processed data
    _ = decompose(params=params,
                  data_train=data_train,
                  data_test=data_test,
                  model=existing_model,
                  dir_to_save=args_full["path_to_training"],
                  summary=args_full,
                  uid=args_full.get("uid", None))
    

if __name__ == "__main__":
    main()