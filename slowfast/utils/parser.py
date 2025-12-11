#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["configs/Kinetics/SLOWFAST_4x16_R50.yaml"],
        nargs="+",
    )
    parser.add_argument(
        "--opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def generate_parser(arg_metadata=None, description=None):
    """
    Generate an argparse.ArgumentParser instance configured with arguments 
    based on the given metadata.

    Parameters:
        arg_metadata (dict): 
            A dictionary where each key is an argument name and the corresponding 
            value is a dictionary of metadata that specifies how the argument should be parsed. 
            The metadata can include keys such as 'type', 'default', and 'help'.
            If a 'type' is not provided, it defaults to str.
        description (str, optional): 
            A description for the parser, which will be displayed in the help message.

    Returns:
        argparse.ArgumentParser:
             An ArgumentParser instance with arguments added according to the provided metadata.

    Notes:
        - For non-boolean arguments, the metadata is passed directly to parser.add_argument().
        - For boolean arguments if the default value is:
            * True, an argument with the prefix "--no-" is created, which when used sets the value to False.
            * False (or not provided), an argument with the prefix "--" is created, which when used sets the value to True.
        - The function copies each metadata dictionary to avoid modifying the original data.
        - An argument is marked as required if no default value is provided.
    """
    parser = argparse.ArgumentParser(description=description)
    for name, param in arg_metadata.items():
        param = dict(param)
        if "type" not in param:
            param["type"] = str
        param["help"] = param.get("help", "")
        param["required"] = "default" not in param
        if param["type"] is bool:
            param["default"] = param.get("default", False)
            if param["default"] is True:
                parser.add_argument(
                    f"--no-{name}", dest=name, action="store_false", help=param["help"]
                )
            else:
                parser.add_argument(
                    f"--{name}", dest=name, action="store_true", help=param["help"]
                )
        else:
            parser.add_argument(f"--{name}", **param)

    return parser
