from os import path
import argparse
import json

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
        raise argparse.ArgumentTypeError(f"Invalid value supplied for `num_components`: {value}")


# Main function for parsing arguments for topic3d-run and topic3d-run-dask
def parse_cli_arguments_legacy(parallelization=False):
    """
    Parse command-line arguments for the full topic decomposition pipeline using topic3d-full-*.

    Parameters
    ----------
    parallelization : bool, optional
        If True, include additional arguments for parallel processing (e.g., with Dask).
        Defaults to False.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments as a namespace.
    
    Notes
    -----
    - If `--imputed` is passed, the input file must contain precomputed imputed data.
    - If `--pwd_normalization` is passed, the label must already be defined in the input file.
    - When `parallelization=True`, `--num_components` is assumed to be set elsewhere (e.g., by a sweep).
    """
    
    parser = argparse.ArgumentParser(description="Perform topic decomposition of proximity data from chromatin tracing")

    # ========================
    # I/O PARAMETERS
    # ========================
    
    parser.add_argument("--data_file",
                        type=str,
                        required=True,
                        help="Path to the data. Must be an API-compatible HDF5 or Numpy file")

    parser.add_argument("--results_base_dir",
                        type=str,
                        required=True,
                        help="Directory in which to create folder(s) to store decomposition results")
    
    parser.add_argument("--tag",
                        type=str,
                        required=False,
                        help="Tag for the dataset. If not provided, file name is taken",
                        default=None)
    
    parser.add_argument("--save_logs",
                        action="store_true",
                        required=False,
                        help="Initialize and save logs in the results folder"
                        )
    
    # ========================
    # DATA PARAMETERS
    # ========================
    
    parser.add_argument("--proximity_threshold",
                        type=float,
                        required=True,
                        help="Proximity threshold in the data's measurement units")
    
    parser.add_argument("--proximity_hardness",
                        type=float,
                        required=False,
                        help="Sigmoid curve steepness for proximity thresholding (default=1.0). Values below 0.2 may be too low",
                        metavar="(0-1]",
                        default=1)
    
    parser.add_argument("--pwd_normalization",
                        type=str,
                        required=False,
                        help="Method to normalize pairwise distances. If indicated, label must already exist in data file",
                        choices=[None, "R(s)", "MMSF"],
                        default=None)
    
    parser.add_argument("--imputed",
                        action="store_true",
                        required=False,
                        help="Use imputed trace data. If indicated, label must already exist in data file"
                        )
    
    parser.add_argument("--min_efficiency",
                    type=int,
                    required=False,
                    help="Minimum percent of barcodes that must be detected in a trace to include for training (default=0)",
                    metavar="[0-100]",
                    default=0
                    )
    
    
    # ========================
    # MODEL PARAMETERS
    # ========================
    # TODO add argument to read LDA parameters from a JSON file in which case the arguments below are ignored

    if parallelization:
        parser.add_argument("--num_components",
                            type=parse_num_components,
                            required=True,
                            help="Number of components (single integer or a list of integers, e.g., 10 or 10,20,30)",
                            metavar="[2-100]")
    else:
        parser.add_argument("--num_components",
                            type=int,
                            required=True,
                            help="Number of components (single integer)",
                            metavar="[2-100]",
                            choices=range(2, 101))
    
    parser.add_argument("--max_iter",
                        type=int,
                        required=False,
                        help="Maximum number of training iterations (default=1000)",
                        metavar="[1-5000]",
                        choices=range(1, 5001),
                        default=1000)
    
    parser.add_argument("--learning_method",
                        type=str,
                        required=False,
                        help="Model learning method: 'batch' (default) or 'online'",
                        choices=["batch", "online"],
                        default="batch")
    
    parser.add_argument("--batch_size",
                        type=int,
                        required=False,
                        help="Batch size while using online learning, ignored for batch learning (default=32)",
                        default=64)
    
    parser.add_argument("--train_data_proportion",
                        type=float,
                        required=False,
                        help="Proportion of data to pick for training (remaining is testing data, default=1)",
                        metavar="[0.1-1]",
                        default=1)
    
    parser.add_argument("--random_state",
                        type=int,
                        required=False,
                        help="Set random seed for reproducible model training (default=None)",
                        default=None)
    
    if parallelization:
        parser.add_argument("--num_jobs",
                            type=int,
                            required=False,
                            help="Number of parallel jobs (default=1)",
                            default=1)

    # Parse and return arguments
    return parser.parse_args()


# Main function for parsing arguments for topic3d runall ...
def parse_cli_arguments_runall(args_init=None):
    """
    Parse command-line arguments for topic decomposition using topic3d runall.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments as a namespace.
    """
    
    parser = argparse.ArgumentParser(description="Decompose single-cell proximity data from chromatin tracing\
                                     experiments using Latent Dirichlet Allocation. CLI for routine topic3d runall")

    # ========================
    # FIRST PARSED PARAMETERS
    # ========================
    
    parser.add_argument("--data_train_file",
                        type=str,
                        required=True,
                        help="Path to the training data. Must be a Numpy file",
                        default=None)
    
    parser.add_argument("--data_test_file",
                        type=str,
                        required=False,
                        help="Path to the training data. Must be a Numpy file",
                        default=None)
    
    parser.add_argument("--model_file",
                        type=str,
                        required=False,
                        help="Path to an existing joblib model file. Must be compatible with Python3 version",
                        default=None)

    parser.add_argument("--results_dir",
                        type=str,
                        required=True,
                        help="Directory in which to store decomposition results",
                        default="")
    
    parser.add_argument("--params_file",
                         type=str,
                         required=False,
                         help="Path to parameters file in JSON format. If passed, parameter command-line arguments will be ignored.")
    
    args0, _ = parser.parse_known_args(args_init)

    if args0.params_file:

        return args0
    
    else:
        
        # ========================
        # REMAINING PARAMETERS
        # ========================

        parser.add_argument("--proximity_threshold",
                            type=int,
                            required=False,
                            help="Proximity threshold when FOF data is supplied")
        
        parser.add_argument("--proximity_hardness",
                            type=float,
                            required=False,
                            help="Proximity threshold hardness when FOF data is supplied. Hardness = 1 --> binary",
                            metavar=[0-1],
                            default=1)

        parser.add_argument("--num_CFMs",
                            type=int,
                            required=True,
                            help="Number of CFMs (single integer)",
                            metavar="[2-100]",
                            choices=range(2, 101))
        
        parser.add_argument("--max_iter",
                            type=int,
                            required=False,
                            help="Maximum number of training iterations (default=1000)",
                            metavar="[1-10,000]",
                            choices=range(1, 10001),
                            default=500)
        
        parser.add_argument("--random_state",
                            type=int,
                            required=False,
                            help="Set random seed for reproducible model training (default=None)",
                            default=None)
        
        parser.add_argument("--learning_method",
                            type=str,
                            required=False,
                            help="Model learning method: 'batch' (default) or 'online'",
                            choices=["batch", "online"],
                            default="batch")
        
        parser.add_argument("--batch_size",
                            type=int,
                            required=False,
                            help="Batch size while using online learning, ignored for batch learning (default=128)",
                            default=128)
        
        parser.add_argument("--evaluate_learning",
                            action="store_true",
                            required=False,
                            help="Evaluate the perplexity of the model during learning (increases training time)")
        
        parser.add_argument("--assess_model_quality",
                            action="store_true",
                            required=False,
                            help="Assess trained model quality using perplexity and NPMI coherence scores")

    # Parse and return arguments
    return parser.parse_args(args_init)


def read_params_json(fp):

    assert path.isfile(fp)

    return json.load(open(fp, mode="r"))