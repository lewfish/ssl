import sys
import argparse
import os
import tempfile
from os.path import join, basename, isfile, splitext, isdir
from collections import OrderedDict

import torch
from rastervision.pipeline.file_system.utils import (
    sync_to_dir, make_dir, get_local_path)
from rastervision.aws_s3.s3_file_system import S3FileSystem

from vissl_experiments.utils import (
    execute, run_remote_if_needed, get_file, open_zip_file)


def extract_backbone(ckpt_path, out_path):
    print(f'Extracting backbone from {ckpt_path} to {out_path}...')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    state_dict = state_dict['classy_state_dict']['base_model']['model']['trunk']
    new_state_dict = OrderedDict()
    remove_prefix = '_feature_blocks.'
    for key, val in state_dict.items():
        new_key = key[len(remove_prefix):]
        new_state_dict[new_key] = val
    torch.save(new_state_dict, out_path)


def run_vissl(config, dataset_dir, output_dir, extra_args, pretrained_path=None):
    # Assume ImageNet format.
    train_dir = join(dataset_dir, 'train')
    val_dir = join(dataset_dir, 'val')
    cmd = [
        'python',
        '/opt/vissl/vissl/tools/run_distributed_engines.py',
        'hydra.verbose=true',
        f'config={config}',
        f'config.DATA.TRAIN.DATA_PATHS=["{train_dir}"]',
        f'config.DATA.TEST.DATA_PATHS=["{val_dir}"]',
        f'config.CHECKPOINT.DIR="{output_dir}"']
    if pretrained_path:
        cmd.append(f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE="{pretrained_path}"')
    cmd.extend(extra_args)
    execute(cmd)

def main(args, extra_args):
    make_dir(args.tmp_root)
    make_dir(args.cache_dir)

    with tempfile.TemporaryDirectory(dir=args.tmp_root) as tmp_dir:
        output_uri = get_local_path(args.output_uri, tmp_dir)
        pretrained_uri = (
            get_file(args.pretrained_uri, args.cache_dir)
            if args.pretrained_uri else None)
        dataset_uri = open_zip_file(args.dataset_uri, args.cache_dir)

        try:
            run_vissl(
                args.config, dataset_uri, output_uri, extra_args,
                pretrained_path=pretrained_uri)
            extract_backbone(
                join(output_uri, 'checkpoint.torch'),
                join(output_uri, 'backbone.torch'))
        finally:
            sync_to_dir(output_uri, args.output_uri)

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Run VISSL')
    parser.add_argument('--config', default='')
    parser.add_argument('--output-uri', default='')
    parser.add_argument('--dataset-uri', default='')
    parser.add_argument('--aws-batch', action='store_true')
    parser.add_argument('--pretrained-uri', type=str, default=None)
    parser.add_argument('--tmp-root', default='/opt/data/tmp/')
    parser.add_argument('--cache-dir', default='/opt/data/data-cache/')
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args, extra_args = parser.parse_known_args()
    s3_file_args = ['dataset_uri', 'pretrained_uri']
    run_remote_if_needed(args, extra_args, s3_file_args, 'vissl')
    main(args, extra_args)
