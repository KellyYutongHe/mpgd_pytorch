import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import random
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="wo",
        help="The name of the method (wo | ae | z | ...) (also the folder name of samples)",
    )
    parser.add_argument(
        "--ni",
        type=bool,
        default=True,
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10
    )
    parser.add_argument(
        "-s",
        "--sample_strategy",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--uncond",
        action='store_true'
    )
    parser.add_argument(
        "--rho_scale", type=float, default=0.1
    )
    parser.add_argument(
        "--prompt", type=str, default=""
    )
    parser.add_argument(
        "--stop", type=int, default=100
    )
    parser.add_argument(
        "--ref_path", type=str, default=None
    )
    parser.add_argument(
        "--scale_weight", type=float, default=None
    )
    parser.add_argument(
        "--rt", type=int, default=1
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        default="../SD_style/models/ldm/celeba256/model.ckpt",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--eta", type=float, default=1.0
    )
    parser.add_argument(
        "--repeat", type=int, default=1
    )
    parser.add_argument(
        "--repeat_start", type=int, default=700
    )
    parser.add_argument(
        "--repeat_end", type=int, default=400
    )
    parser.add_argument(
        "--ldm_start", type=int, default=800
    )
    parser.add_argument(
        "--ldm_end", type=int, default=500
    )
    
    

    args = parser.parse_args()
    seed_everything(args.seed)
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)
    
    args.method = args.image_folder
    args.model_type = "face"

    hyperparam = f"ddim{args.timesteps}_eta{args.eta}_rho{args.rho_scale}_seed{args.seed}_repeat{args.repeat}_{args.repeat_start}-{args.repeat_end}"
    if args.image_folder == "partial" or args.image_folder == "latent":
        hyperparam += f"_ldm{args.ldm_start}-{args.ldm_end}"
    if args.uncond:
        hyperparam = f"ddim{args.timesteps}_eta{args.eta}_seed{args.seed}_repeat{args.repeat}_{args.repeat_start}-{args.repeat_end}"
    args.image_folder = os.path.join(
        args.exp, args.sample_strategy, args.image_folder, hyperparam,
    )
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))

    return args


def main():
    args = parse_args_and_config()
    seed_everything(args.seed)
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args)
        runner.sample(args.sample_strategy)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
