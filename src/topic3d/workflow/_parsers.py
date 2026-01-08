import argparse

# Function to parse command line arguments
def parse_args():
    
    parser = argparse.ArgumentParser(description="Run decomposition processes in parallel.")

    # Add arguments based on your provided list
    parser.add_argument("--root_dir", type=str, required=True, help="Working folder")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Contact threshold")
    parser.add_argument("--hardness", type=float, required=True, help="Contact threshold hardness", default=1)
    parser.add_argument("--num_components", type=int,
                        help="Number of components (single +ve integer)")
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


def extract_params(args):
    return