import torch
from cli.parser import get_parser
from cli.config_handler import load_config
import torch.multiprocessing as mp
from cli.linear_eval_handler import linear_ddp
from cli.pre_train_handler import pre_train_ddp
import modules, data_generator, cli
from cli.eval_handler import *
from cli.process_handler import *
from cli.train_handler import *

# Set seeds
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Make CuDNN deterministic (may slow down training)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
def main():
    args = get_parser().parse_args()

    if hasattr(args, "config"):
        config = load_config(args.config)

        # Merge CLI args into config Namespace
        for key, value in vars(config).items():
            if value is not None:
                setattr(args, key, value)
        
    print(f"{args}\n")

    if args.mode == "process":
        process_data(args)

    elif args.mode == "train":
        world_size = args.world_size if hasattr(args, "world_size") else torch.cuda.device_count()
        print(f"Setting up training process on {world_size} GPUs \n")
        mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)

    elif args.mode == "pre_train":
        world_size = args.world_size if hasattr(args, "world_size") else torch.cuda.device_count()
        print(f"Setting up pre-training process on {world_size} GPUs \n")
        mp.spawn(pre_train_ddp, args=(world_size, args), nprocs=world_size, join=True)

    elif args.mode == "linear":
        world_size = args.world_size if hasattr(args, "world_size") else torch.cuda.device_count()
        print(f"Setting up linear eval protocol on {world_size} GPUs \n")
        mp.spawn(linear_ddp, args=(world_size, args), nprocs=world_size, join=True)
        
    elif args.mode == "eval":
        evaluate(args)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()