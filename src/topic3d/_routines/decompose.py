"""
Main script for decomposing chromatin tracing data with a trained LDA model using topic3d.

This script:
1. Parses CLI arguments.
2. Loads raw FOF-CT data or preprocessed pairwise proximity data.
3. Initializes the output directory and loggers.
4. Decomposes the data.
5. Saves the results.

An example run:

$ topic3d decompose \
    --data_file path/to/data/file.npy \
    --results_dir path/to/results/folder \

This script is intended to be called from the CLI wrapper `topic3d decompose` and not directly by end users.

Dependencies:
- topic3d.workflow._parsers
- topic3d.workflow._dataops
- topic3d.workflow.core
"""


from topic3d.workflow._parsers import parse_cli_arguments_decompose
from topic3d.workflow._dataops import load_data_file, load_model_file
from topic3d.workflow.core import initialize_cli_run, decompose_data


def main():
    """Execute the topic decomposition pipeline based on CLI arguments."""

    # Step 1: Parse command-line arguments
    args = parse_cli_arguments_decompose()

    # Step 2: Convert Namespace to dict for supplying parameters as required by the API
    params = dict(vars(args))   # Generally params do not have all CLI arguments, this is just for easy scripting here

    # Step 3: Load data and using internal workflow API
    data = load_data_file(path_to_file=args.data_file)
    existing_model = load_model_file(args.model_file)

    # Step 4: Initialize run metadata and output directories if needed
    args_full = initialize_cli_run(params=params,
                                  dir_results=args.results_dir,
                                  path_to_data=args.data_file,)

    # Step 5: Run LDA decomposition on the processed data
    _, _ = decompose_data(data=data,
                          model=existing_model)
    
    # Step 6: Save results
    # TODO


if __name__ == "__main__":
    main()