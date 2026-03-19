"""
Main script for running topic decomposition on chromatin tracing data using topic3d.

This script:
1. Parses CLI arguments.
2. Loads raw or preprocessed proximity data.
3. Initializes the output directory structure, configuration and logging if indicated.
4. Prepares data for topic modeling by computing proximity vectors.
5. Optionally splits data into training and testing subsets.
6. Trains an LDA model and saves results.

An example run:

$ topic3d \
    --data_file path/to/data/file.npy \
    --results_base_dir path/to/create/results/folder \
    --tag 'datasetname-tests' \
    --save_logs \
    --proximity_threshold 150 \
    --num_components 15 \
    --max_iter 500 \
    --train_data_proportion 0.8

This script is intended to be called from the CLI wrapper `topic3d-run` and not directly by end users.

Dependencies:
- topic3d.workflow._parsers
- topic3d.workflow._dataops
- topic3d.workflow.training
"""

from os import path
from topic3d.workflow._parsers import parse_cli_arguments_runall, read_params_json
from topic3d.workflow._dataops import load_data_file
from topic3d.workflow.core import initialize, train_LDA_model, plot_learning_evaluation, decompose_data, save_results


def main(args_init):
    """Execute the topic decomposition pipeline based on CLI arguments."""

    # Step 1: Parse command-line arguments
    args = parse_cli_arguments_runall(args_init)

    # Step 2: Convert Namespace to dict for supplying parameters as required by the API
    params = dict(vars(args))   # Generally params do not have all CLI arguments, this is just for easy scripting here
    if args.params_file is not None:
        params.update(read_params_json(args.params_file))

    # Step 3: Load data using internal workflow API
    data_train = load_data_file(args.data_train_file)
    if args.data_test_file is None:
        data_test = None
    else:
        data_test = load_data_file(args.data_test_file)

    # Step 4: Initialize run metadata and output directories
    args_full = initialize(params=params,
                           dir_results=args.results_dir,
                           path_to_data=args.data_train_file,
                        #    tag=getattr(args, "tag"),
                        #    save_logs=getattr(args, "save_logs"))
                        )
    
    # # Step 5: Prepare proximity vectors from trace data and save
    # data_prepped = prepare_proximity_data(data=data,
    #                                       params=params,
    #                                       dir_to_save=args_full["path_to_training"],
    #                                       uid=args_full.get("uid", None))

    # # Step 6: Split into training and testing sets (if indicated in args)
    # data_train, data_test = split_proximity_data(data=data_prepped,
    #                                              train_data_proportion=params["train_data_proportion"],
    #                                              uid=args_full.get("uid", None))

    # Step 7: Run LDA decomposition on the processed data
    if args_full["evaluate_learning"]:
        model, perp_by_iter = train_LDA_model(params=params,
                                              data_train=data_train,
                                              data_test=data_test,
                                              summary=args_full,
                                              uid=args_full.get("uid", None))
        plot_learning_evaluation(perp_by_iter,
                                 path_to_save=path.join(args_full["path_to_training"], "perplexity_evolution.png"))
    else:
        model = train_LDA_model(params=params,
                                data_train=data_train,
                                data_test=data_test,
                                summary=args_full,
                                uid=args_full.get("uid", None))
    
    # evaluate_LDA_model()
    
    # Step 5: Run LDA decomposition on the processed data
    components, decomposition = decompose_data(data=data_train,
                                               model=model)
    
    save_results(model=model,
                 decomposition=decomposition,
                 dir_to_save=args_full["path_to_training"])

if __name__ == "__main__":
    main()