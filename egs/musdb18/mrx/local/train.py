#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
import torch.nn as nn

from utils.utils import set_seed
from utils.augmentation import SequentialAugmentation, choose_augmentation
from dataset import AugmentationWaveTrainDataset, TrainDataLoader
from adhoc_dataset import WaveEvalDataset, EvalDataLoader
from adhoc_driver import AdhocTrainer
from models.mrx import MultiResolutionCrossNet
from criterion.sdr import NegSISDR

parser = argparse.ArgumentParser(description="Training of Multi-Resolution CrossNet (MRX)")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--config_path', type=str, default=None, help='Path to model configuration file')
parser.add_argument('--sample_rate', '-sr', type=int, default=44100, help='Sampling rate')
parser.add_argument('--duration', type=float, default=6, help='Duration')
parser.add_argument('--valid_duration', type=float, default=30, help='Max duration for validation')
parser.add_argument('--augmentation_path', type=str, default=None, help='Path to augmentation.yaml')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['sisdr'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 128')
parser.add_argument('--samples_per_epoch', type=int, default=64*100, help='Training samples in one epoch. Default: 6400.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_norbert', type=int, default=0, help='Use norbert.wiener for multichannel wiener filetering. 0: Not use norbert, 1: Use norbert (you have to install norbert)')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--num_workers', type=int, default=0, help='# of workers given to data loader for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    args.sources = args.sources.replace('[', '').replace(']', '').split(',')
    args.n_sources = len(args.sources)
    
    if args.samples_per_epoch <= 0:
        args.samples_per_epoch = None
    
    with open(args.augmentation_path) as f:
        config_augmentation = yaml.safe_load(f)
    
    augmentation = SequentialAugmentation()
    for name in config_augmentation['augmentation']:
        augmentation.append(choose_augmentation(name, **config_augmentation[name]))
    
    train_dataset = AugmentationWaveTrainDataset(
        args.musdb18_root,
        sample_rate=args.sample_rate, duration=args.duration, samples_per_epoch=args.samples_per_epoch,
        sources=args.sources, target=args.sources,
        include_valid=True,
        augmentation=augmentation
    )
    valid_dataset = WaveEvalDataset(args.musdb18_root, sample_rate=args.sample_rate, patch_duration=args.duration, max_duration=args.valid_duration, sources=args.sources, target=args.sources)
    
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    model = MultiResolutionCrossNet.build_from_config(config_path=args.config_path)

    print(model)
    print("# Parameters: {}".format(model.num_parameters), flush=True)
    
    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            print("Use CUDA")
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA")
        
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))
    
    # Criterion
    if args.criterion == 'sisdr':
        criterion = NegSISDR()
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    trainer = AdhocTrainer(model, loader, criterion, optimizer, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
