from data_generator import DATASET_REGISTRY

def process_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_REGISTRY[dataset_name]

    pre_transformer = config["pre_transform"] if args.pre_transform else None
    pre_filter = config["pre_filter"]

    print(f"Processing {args.dataset} dataset")
    print(f"Benchmark: {args.benchmark} - Modality: {args.modality} - Part: {args.part}")
    print(f"Pre_transform: {args.pre_transform}")

    dataset_class = config["class"]
    dataset = dataset_class(
        root=config["default_root"],
        pre_filter=pre_filter,
        pre_transform=pre_transformer,
        modality=args.modality,
        benchmark=args.benchmark,
        part=args.part,
        extended=config["extended"],
        force_reload=args.force_reload
    )
    
    if args.summary:
       dataset.print_summary()