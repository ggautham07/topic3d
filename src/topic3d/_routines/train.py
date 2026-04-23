"""
Main script for training an LDA model on chromatin tracing data using topic3d.

This script:
1. Parses CLI arguments.
2. Loads raw FOF-CT data or preprocessed pairwise proximity data.
3. Initializes the output directory and loggers.
4. Trains an LDA model to capture CFMs.
5. Saves the trained model.

An example run:

$ topic3d train \
    --data_train_file path/to/data/file.npy \
    --results_dir path/to/results/folder \
    --num_components 15 \
    --max_iter 500 \
    --evaluate_learning

This script is intended to be called from the CLI wrapper `topic3d train` and not directly by end users.

Dependencies:
- topic3d.workflow._parsers
- topic3d.workflow._dataops
- topic3d.workflow.core
"""

from os import path
import logging
from topic3d.workflow._parsers import parse_cli_arguments_train, parse_params_json
from topic3d.workflow._dataops import load_data_file, prepare_proximity_data
from topic3d.workflow.core import initialize_cli_run, initialize_LDA_model, train_LDA_model, plot_learning_evaluation


def main(args_init):
    """Execute the topic decomposition pipeline based on CLI arguments."""

    # Step 1: Parse command-line arguments
    args = parse_cli_arguments_train(args_init)
    if not args.data_train_file.endswith(".csv"):
        args.save_proximity_data = False

    # Step 2: Convert Namespace to dict for supplying parameters as required by the API
    params = dict(vars(args))
    if args.params_file is not None:
        params.update(parse_params_json(args.params_file))

    # Step 3: Initialize run metadata and output directories
    args_full, logger = initialize_cli_run(params=params,
                                           dir_results=args.results_dir,
                                           path_to_data=args.data_train_file,)
    uid = args_full.get("uid", "no-uid")

    # Step 4: Load data using workflow API
    data_train_file_contents = load_data_file(args.data_train_file,
                                              uid=uid,
                                              logger=logger)
    if args.data_test_file is None:
        data_test_file_contents = None
    else:
        data_test_file_contents = load_data_file(args.data_test_file,
                                                 uid=uid,
                                                 logger=logger)

    proximity_params = {}
    for param in params.keys():
        if param.startswith("proximity_"):
            proximity_params[param.replace("proximity_", "")] = params[param]
    
    # Step 5: Process proximity data
    logger.info(f"[{uid}] Preparing/verifying training data...")
    data_train = prepare_proximity_data(data=data_train_file_contents,
                                        proximity_params=proximity_params,
                                        dir_to_save=args_full["path_to_training"] \
                                            if args.save_proximity_data else None,
                                        uid=uid,
                                        logger=logger)
    logger.info(f"[{uid}] Prepared/Verified training data successfully")
    if data_test_file_contents is None:
        data_test = None
    else:
        logger.info(f"[{uid}] Preparing/verifying testing data...")
        data_test = prepare_proximity_data(data=data_test_file_contents,
                                           proximity_params=proximity_params,
                                           uid=uid,
                                           logger=logger)
        logger.info(f"[{uid}] Prepared/Verified training data successfully")

    # Step 6: Initialize LDA model
    model = initialize_LDA_model(params=params,
                                 uid=uid,
                                 logger=logger)

    # Step 7: Train LDA model
    result_train = train_LDA_model(model=model,
                                   data_train=data_train,
                                   params=params,
                                   uid=uid,
                                   logger=logger)
    if args_full["evaluate_learning"]:
        model, perp_by_iter = result_train
        plot_learning_evaluation(perp_by_iter,
                                 path_to_save=path.join(args_full["path_to_training"], "perplexity_evolution.png"),
                                 uid=uid,
                                 logger=logger)
    else:
        model = result_train
    
    # Step 8: Evaluate LDA model using NPMI coherence
    # evaluate_LDA_model()

    # Step 9: save the trained LDA model and captured CFMs
    # TODO

if __name__ == "__main__":
    main()