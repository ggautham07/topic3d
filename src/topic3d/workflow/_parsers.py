from os import path
import argparse
import json


# Helper function to parse a single integer or a list of integers
def parse_num_CFMs(input):
    try:
        # Check if it's a single integer
        if ',' in input:
            # If it contains commas, split and parse as a list of integers
            return [int(v) for v in input.split(',')]
        else:
            # Otherwise, parse it as a single integer and return it as a list
            return [int(input)]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value supplied for `num_components`: {input}")
    

def parse_params_json(fp):

    assert path.isfile(fp)

    return json.load(open(fp, mode="r"))


# Main function for parsing `topic3d train` arguments ...
def parse_cli_arguments_train(args_init=None):
    """
    Parse command-line arguments for `topic3d train`.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments as a namespace.
    """
    
    parser = argparse.ArgumentParser(description="Model chromatin tracing data using Latent Dirichlet Allocation.\n \
                                     This is the CLI for the model training routine.")

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
    
    parser.add_argument("--save_proximity_data",
                        action="store_true",
                        required=False,
                        help="Store proximity data in the results directory when FOF-CT data was provided as input")
    
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
                            help="Assess trained model quality using the NPMI coherence score")

    # Parse and return arguments
    return parser.parse_args(args_init)


# Main function for parsing `topic3d decompose` arguments ...
def parse_cli_arguments_decompose(args_init=None):
    """
    Parse command-line arguments for `topic3d decompose`.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments as a namespace.
    """
    
    parser = argparse.ArgumentParser(description="Model chromatin tracing data using Latent Dirichlet Allocation.\n \
                                     This is the CLI for the data decomposition routine.")

    # ========================
    # FIRST PARSED PARAMETERS
    # ========================
    
    parser.add_argument("--data_file",
                        type=str,
                        required=True,
                        help="Path to the training data. Must be a Numpy file",
                        default=None)
    
    parser.add_argument("--save_proximity_data",
                        action="store_true",
                        required=False,
                        help="Store proximity data in the results directory when FOF-CT data was provided as input")
    
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

    # Parse and return arguments
    return parser.parse_args(args_init)


# Main function for parsing arguments for topic3d runall ...
def parse_cli_arguments_runall(args_init=None):
    """
    Parse command-line arguments for `topic3d runall`.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments as a namespace.
    """
    
    parser = argparse.ArgumentParser(description="Model chromatin tracing data using Latent Dirichlet Allocation.\n \
                                     This is the CLI for the full pipeline.")

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

    parser.add_argument("--save_proximity_data",
                        action="store_true",
                        required=False,
                        help="Store proximity data in the results directory when FOF-CT data was provided as input")
    
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
                            help="Proximity threshold to use when FOF-CT data is supplied")
        
        parser.add_argument("--proximity_hardness",
                            type=float,
                            required=False,
                            help="Proximity hardness to use when FOF-CT data is supplied. Hardness = 1 --> binary",
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
                            help="Assess trained model quality using the NPMI coherence score")

    # Parse and return arguments
    return parser.parse_args(args_init)