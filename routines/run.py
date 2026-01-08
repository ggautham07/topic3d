from topic3d.workflow._parsers import parse_args, extract_params
from topic3d.workflow._dataops import load_data, prepare_proximity_data, split_proximity_data
from topic3d.workflow.training import initialize, decompose


def main():

    args = parse_args()

    params = extract_params(args)

    data = load_data(path_to_file=args.data_path, tag=args.tag)

    summary = initialize(params=params,
                         dir_results=args.results_dir,
                         path_to_data=args.data_path,
                         tag=args.tag,)
    
    data_prepped = prepare_proximity_data(data=data,
                                       params=params,
                                       dir_to_save=summary["path_to_training"],
                                       uid=summary.get("uid", None))

    data_train, data_test = split_proximity_data(data=data_prepped,
                                                 train_data_proportion=params["train_data_proportion"],
                                                 uid=summary.get("uid", None))


    _ = decompose(params=params,
                  data_train=data_train,
                  data_test=data_test,
                  dir_to_save=summary["path_to_training"],
                  summary=summary)
    

if __name__ == "__main__":
    main()