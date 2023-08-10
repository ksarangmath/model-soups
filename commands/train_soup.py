# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import submitit
import argparse
import logging
import pprint
import yaml
import os
import json

import sys
sys.path.append("/coc/pskynet4/ksarangmath3/lsr/model-soups/")
from finetune import main as soup_train

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser()

parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='/submitit/')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs to per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')
parser.add_argument(
    "--data_location",
    type=str,
    default=os.path.expanduser('/srv/share/datasets/ImageNet/'),
    help="The root directory for the datasets.",
)
parser.add_argument(
    "--root_path",
    type=str,
    default=os.path.expanduser('/srv/share/datasets/'),
    help="The root directory for the datasets.",
)
parser.add_argument(
    "--image_folder",
    type=str,
    default=os.path.expanduser('ImageNet/'),
    help="The root directory for the datasets.",
)
parser.add_argument(
    "--subset_file",
    type=str,
    default=os.path.expanduser('/srv/share4/ksarangmath3/lsr/robust-ssl/subsets/imagenet_subsets1/1imgs_class.txt'),
    help="The root directory for the datasets.",
)
parser.add_argument(
    "--pretrained_path",
    type=str,
    default=os.path.expanduser('/srv/share4/asingh866/msn/pretrained/msn/LPFT_subsets1_1imgs_class-lin-eval.pth.tar'),
    help="The root directory for the datasets.",
)
parser.add_argument(
    "--model_location",
    type=str,
    default=os.path.expanduser('/srv/share4/ksarangmath3/lsr/model-soups/soups/'),
    help="Where to download the models.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
)
parser.add_argument(
    "--workers",
    type=int,
    default=4,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=8,
)
parser.add_argument(
    "--warmup-length",
    type=int,
    default=500,
)
parser.add_argument(
    "--lr",
    type=float,
    default=2e-5,
)
parser.add_argument(
    "--wd",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--name",
    default='finetune_cp_msn',
    help='Filename for the checkpoints.'
)
parser.add_argument(
    "--timm_aug", action="store_true", default=False,
)
parser.add_argument(
    "--aug",
    type=str,
    default=None,
    help="randaug hparams",
)
parser.add_argument(
    "--mix",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--smoothing",
    type=float,
    default=0.0,
)

parser.add_argument(
    "--nb_classes",
    type=int,
    default=1000,
)

parser.add_argument(
    "--soup_num",
    type=int,
    default=0,
)

parser.add_argument(
    "--soup_size",
    type=int,
    default=72,
)

parser.add_argument(
    '--input_size', 
    default=224, 
    type=int,
    help='images input size'
)

parser.add_argument(
    '--num_iter', 
    default=None, 
    type=int,
    help='images input size'
)

parser.add_argument(
    "--forward-blocks",
    type=int,
    default=0,
)

parser.add_argument(
    "--eval_type",
    type=str,
    default='lineval',
    help="randaug hparams",
)

parser.add_argument(
    "--model_name",
    type=str,
    default='deit_base',
    help="randaug hparams",
)

def launch():

    yaml_params = None
    with open(args.fname, 'r') as y_file:
        yaml_params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(yaml_params)

    hparam_info = json.load(open('/srv/share4/ksarangmath3/lsr/model-soups/hparam_info.json'))

    hparam_info["model_0"] = {"epochs": 0}
    ft_param_list = []
    for i in range(0, args.soup_size):
        print(f"model_{i}")
        params = {}
        # print(args)
        for key in vars(args):
            params[key] = vars(args)[key]
        for key in hparam_info[f"model_{i}"]:
            params[key] = hparam_info[f"model_{i}"][key]
        for key in yaml_params:
            params[key] = yaml_params[key]

        params['soup_num'] = i
        
        ft_param_list.append(params)
    

    executor = submitit.AutoExecutor(folder=args.folder)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu='48G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=6,
        constraint='a40',
        gpus_per_node=args.tasks_per_node,
        slurm_array_parallelism=1
    )

    print(ft_param_list)

    jobs = executor.map_array(soup_train, ft_param_list)

    for job in jobs:
        print(job.job_id)


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
