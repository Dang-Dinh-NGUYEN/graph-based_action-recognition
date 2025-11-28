import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Graph-based Action Recognition")
    parser.add_argument("--port", type=str, help="master port", default="9991")
    parser.add_argument("--autograd", action="store_true", help="enable autograd")

    mode_parser = parser.add_subparsers(dest="mode", required=True, help="select mode: process | train | eval")

    # --- Process Mode ---
    process_parser = mode_parser.add_parser("process", help="Process Data")

    process_parser.add_argument("--dataset", type=str, default="nturgb+d", help="path towards dataset: nturgb+d | nturgb+d_120 | pku_mmd | pku_mmd_v2 | mocap")
    process_parser.add_argument("--modality", default="joint", choices=["joint", "bone", "joint_bone"], help="modality: joint | bone | joint_bone")
    process_parser.add_argument("--benchmark", default="xsub", choices=["xsub", "xview", "xsetup", "multi_class", "multi_label"], help="benchmark: xsub | xview | xsetup")
    process_parser.add_argument("--part", default="train", choices=["train", "eval"], help="part: train | eval")
    process_parser.add_argument("--pre_transform", action="store_true", help="authorize pre-transformation")
    process_parser.add_argument("--force_reload", action="store_true", help="force processing")
    process_parser.add_argument("--summary", action="store_true", help="summary of preprocessed dataset")

    process_parser.set_defaults(mode="process")

    # --- Pre-train Mode ---
    pre_train_parser = mode_parser.add_parser("pre_train", help="Pre-train Model")

    pre_train_parser.add_argument("--config", required=True, type=str, help="path towards config file")
    pre_train_parser.add_argument("--from_scratch", action="store_true", help="train model from scratch")
    pre_train_parser.add_argument("--debug", action="store_true", help="enable debug mode")

    pre_train_parser.set_defaults(mode="pre_train")

    # --- Linear Mode ---
    linear_parser = mode_parser.add_parser("linear", help="Linear Model")

    linear_parser.add_argument("--config", required=True, type=str, help="path towards config file")
    linear_parser.add_argument("--from_scratch", action="store_true", help="train model from scratch")
    linear_parser.add_argument("--debug", action="store_true", help="enable debug mode")

    linear_parser.set_defaults(mode="linear")

    # --- Train Mode ---
    train_parser = mode_parser.add_parser("train", help="Train Model")

    train_parser.add_argument("--config", required=True, type=str, help="path towards config file")
    train_parser.add_argument("--from_scratch", action="store_true", help="train model from scratch")
    train_parser.add_argument("--debug", action="store_true", help="enable debug mode")
    # train_parser.add_argument("--summary", action="store_true", help="summary of model") 

    train_parser.set_defaults(mode="train")

    # --- Eval Mode ---
    eval_parser = mode_parser.add_parser("eval", help="Evaluate an ensemble of models")

    eval_parser.add_argument("--model_path", nargs='+', type=str, required=True, help="Paths to model files")
    eval_parser.add_argument("--model_config", nargs='+', type=str, required=True, help="path towards model's config")
    eval_parser.add_argument("--modalities", nargs='+', type=str, required=True, help="Modalities")
    eval_parser.add_argument("--metric", default="multiclass", choices=["multiclass", "multilabel"], help="Evaluation metric")
    eval_parser.add_argument("--alphas", nargs='+', type=float, required=True, help="Weight per model")
    eval_parser.add_argument("--display", action="store_true", help="Show predictions vs ground truth")

    eval_parser.set_defaults(mode="eval")

    return parser

