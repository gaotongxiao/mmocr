# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmocr.utils import register_all_modules

recog_root = 'openmmlab:s3://openmmlab/datasets/ocr/recog/'
det_root = 'openmmlab:s3://openmmlab/datasets/ocr/det/'


def update_data_root(cfg: ConfigDict, dataloader_key: str, new_data_root: str):
    # TODO: It's just about to work. Need more check and test in the future.
    if not isinstance(cfg[dataloader_key], list):
        # Applicable to ConcatDataset only
        for i in range(len(cfg[dataloader_key].dataset.datasets)):
            data_root = cfg[dataloader_key].dataset.datasets[i].data_root
            if data_root[-1] == '/':
                data_root = data_root[:-1]
            data_root_base = osp.basename(data_root)
            if data_root_base not in ['det', 'rec']:
                # e.g. data/det/icdar2015
                cfg[dataloader_key].dataset.datasets[i].data_root = osp.join(
                    new_data_root, data_root_base)
            else:
                # e.g. data/rec
                cfg[dataloader_key].dataset.datasets[
                    i].data_root = new_data_root

    else:
        # Applicable to MultiEvalLoop where each dataloader takes one dataset
        for i in range(len(cfg[dataloader_key])):
            data_root = cfg[dataloader_key][i].dataset.data_root
            if data_root[-1] == '/':
                data_root = data_root[:-1]
            data_root_base = osp.basename(data_root)
            if data_root_base not in ['det', 'rec']:
                # e.g. data/det/icdar2015
                cfg[dataloader_key][i].dataset.data_root = osp.join(
                    new_data_root, data_root_base)
            else:
                # e.g. data/rec
                cfg[dataloader_key][i].dataset.data_root = new_data_root


def update_confg(config: str):
    cfg = Config.fromfile(config)

    # change file_client to petrel
    cfg.train_dataloader.dataset.pipeline[0].file_client_args = dict(
        backend='petrel')
    if not isinstance(cfg.val_dataloader, list):
        cfg.val_dataloader.dataset.pipeline[0].file_client_args = dict(
            backend='petrel')
    else:
        for i in range(len(cfg.val_dataloader)):
            cfg.val_dataloader[i].dataset.pipeline[0].file_client_args = dict(
                backend='petrel')
    if not isinstance(cfg.test_dataloader, list):
        cfg.test_dataloader.dataset.pipeline[0].file_client_args = dict(
            backend='petrel')
    else:
        for i in range(len(cfg.test_dataloader)):
            cfg.test_dataloader[i].dataset.pipeline[0].file_client_args = dict(
                backend='petrel')

    # change data_root to ceph
    if 'textdet' in config:
        update_data_root(cfg, 'train_dataloader', det_root)
        update_data_root(cfg, 'val_dataloader', det_root)
        update_data_root(cfg, 'test_dataloader', det_root)

    elif 'textrecog' in config:
        update_data_root(cfg, 'train_dataloader', recog_root)
        update_data_root(cfg, 'val_dataloader', recog_root)
        update_data_root(cfg, 'test_dataloader', recog_root)
        mmocr_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
        cfg.model.decoder.dictionary.dict_file = osp.join(
            mmocr_path, cfg.model.decoder.dictionary.dict_file)

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='The dir to save logs and models')
    parser.add_argument(
        '--resume', action='store_true', help='Whether to resume checkpoint.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='Enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='Whether to scale the learning rate automatically. It requires '
        '`auto_scale_lr` in config, and `base_batch_size` in `auto_scale_lr`')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    # cfg = Config.fromfile(args.config)
    cfg = update_confg(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.resume:
        cfg.resume = True

    if args.auto_scale_lr:
        if cfg.get('auto_scale_lr'):
            cfg.auto_scale_lr = True
        else:
            print_log(
                'auto_scale_lr does not exist in your config, '
                'please set `auto_scale_lr = dict(base_batch_size=xx)',
                logger='current',
                level=logging.WARNING)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
