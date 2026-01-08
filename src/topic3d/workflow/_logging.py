import sys
import logging

# Function to configure logging
def setup_log(fp):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(fp),
            logging.StreamHandler(sys.stdout)
        ]
    )