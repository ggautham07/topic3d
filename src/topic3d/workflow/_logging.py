import sys
import logging
from io import StringIO

# Simple function to configure logging that will be saved
def setup_log_naive(fp):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(fp)],
        force=True
        )
    
# Usage not recommended since logs can overlap
def setup_log_nosave_naive():
    logging.basicConfig(
        level=logging.NOTSET,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
        )
    
# Recommended logger that will save logs
def setup_log(name: str, fp: str):
    assert isinstance(name, str)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_fp = logging.FileHandler(fp)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)
    logger.addHandler(handler_fp)
    return logger

# Recommended logger that will NOT save logs
def setup_log_nosave(name: str):
    assert isinstance(name, str)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def kill_logger(logger: logging.Logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
# Context manager to capture stdout from a function
# used primarily for extracting perplexity evaluated during LDA model training
class CaptureOutput(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout