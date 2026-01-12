import sys
import logging

# Function to configure logging
def setup_log(fp):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(fp)],
        force=True
        )

def setup_log_simple():
    logging.basicConfig(
        level=logging.NOTSET,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
        )