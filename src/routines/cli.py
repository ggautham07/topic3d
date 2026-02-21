def main():
    print("topic3d: Topic decomposition of chromatin tracing data | Command line interface")
    print("Available commands:")
    print("  topic3d-decompose     Run LDA on proximity data with provided parameters")
    print("  topic3d-run           Run full pipeline including pre-processing on tracing data")
    print("  topic3d-run-dask      Run the above with distributed jobs via Dask")