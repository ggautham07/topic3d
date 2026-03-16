import sys
import logging
from io import StringIO

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
    
def setup_log_simple2(name: str):
    assert isinstance(name, str)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
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