#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import WaveTrainDataset, WaveEvalDataset, TrainDataLoader, EvalDataLoader
from new_dataset import WaveTrainDataset as NewTrainDataset
from adhoc_driver import AdhocTrainer
# from models.dptnet import DPTNet
from models.custom.dpsarnn_net import DPSARNN_Net
from criterion.sdr import NegSISDR
from criterion.pit import PIT1d
from criterion.stft_loss import DEMUCSLoss, MagMSELoss, CombinePFPLoss, CombineSISNRLoss
from driver import MyNegSISNR

parser = argparse.ArgumentParser(description="Training of Conv-TasNet")

parser.add_argument('--train_wav_root', type=str, default=None, help='Path for training dataset ROOT directory')
parser.add_argument('--valid_wav_root', type=str, default=None, help='Path for validation dataset ROOT directory')
parser.add_argument('--train_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tr_mix')
parser.add_argument('--valid_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_cv_mix')
parser.add_argument('--sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--valid_duration', type=float, default=4, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--enc_basis', type=str, default='trainable', choices=['TENET','DCT','trainable','Fourier','trainableFourier'], help='Encoder type')
parser.add_argument('--dec_basis', type=str, default='trainable', choices=['TENET','DCT','trainable','Fourier','trainableFourier', 'pinv'], help='Decoder type')
parser.add_argument('--enc_nonlinear', type=str, default=None, help='Non-linear function of encoder')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--n_basis', '-N', type=int, default=64, help='# basis')
parser.add_argument('--kernel_size', '-L', type=int, default=2, help='Kernel size')
parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size//2')
parser.add_argument('--sep_bottleneck_channels', '-F', type=int, default=64, help='Bottleneck channels of separator')
parser.add_argument('--sep_hidden_channels', '-d_ff', type=int, default=128, help='Hidden channels of RNN in each direction')
parser.add_argument('--sep_chunk_size', '-K', type=int, default=100, help='Chunk size of separator')
parser.add_argument('--sep_hop_size', '-P', type=int, default=50, help='Hop size of separator')
parser.add_argument('--sep_num_blocks', '-B', type=int, default=6, help='# blocks of separator. Each block has B layers')
parser.add_argument('--sep_num_heads', type=int, default=4, help='Number of heads in multi-head attention')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--sep_norm', type=int, default=1, help='Normalization')
parser.add_argument('--sep_nonlinear', type=str, default='relu', help='Non-linear function of separator')
parser.add_argument('--sep_dropout', type=float, default=0, help='Dropout')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['demucs_mse','sisdr','pfp_sisnr','sisnr'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--k1', type=float, default=2e-1, help='Learning rate during warm up. Default: 2e-1')
parser.add_argument('--k2', type=float, default=4e-4, help='Learning rate after warm up. Default: 4e-4')
parser.add_argument('--noise_loss', default=False, action='store_true')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 4')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--worker', type=int, default=16, help='Random seed')
parser.add_argument('--new_dset', default=False, action='store_true')

def main(args):
    set_seed(args.seed)
    
    samples = int(args.sr * args.duration)
    overlap = samples//2
    max_samples = int(args.sr * args.valid_duration)
    
    if args.new_dset:
        train_dataset = NewTrainDataset(args.train_wav_root, args.train_list_path, samples=samples, overlap=overlap, n_sources=args.n_sources,noise_loss=args.noise_loss,use_h5py=True)
    else:
        train_dataset = WaveTrainDataset(args.train_wav_root, args.train_list_path, samples=samples, overlap=overlap, n_sources=args.n_sources,noise_loss=args.noise_loss,use_h5py=False)

    # train_dataset = WaveTrainDataset(args.train_wav_root, args.train_list_path, samples=samples, overlap=overlap, n_sources=args.n_sources, use_h5py=True)
    valid_dataset = WaveEvalDataset(args.valid_wav_root, args.valid_list_path, max_samples=max_samples, n_sources=args.n_sources)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    dl_workers = args.worker
    print('Dataloader workers:', dl_workers)
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, num_workers=dl_workers,shuffle=True)
    # loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size,num_workers=16, shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1,num_workers=8, shuffle=False)
    
    if not args.enc_nonlinear:
        args.enc_nonlinear = None
    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    model = DPSARNN_Net(
        args.n_basis, args.kernel_size, stride=args.stride,
        enc_basis=args.enc_basis, dec_basis=args.dec_basis, enc_nonlinear=args.enc_nonlinear, window_fn=args.window_fn,
        sep_hidden_channels=args.sep_hidden_channels, sep_bottleneck_channels=args.sep_bottleneck_channels,
        sep_chunk_size=args.sep_chunk_size, sep_hop_size=args.sep_hop_size, sep_num_blocks=args.sep_num_blocks,
        sep_num_heads=args.sep_num_heads, sep_norm=args.sep_norm, sep_nonlinear=args.sep_nonlinear, sep_dropout=args.sep_dropout,
        mask_nonlinear=args.mask_nonlinear,
        causal=args.causal,
        n_sources=args.n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

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
    # The learning rate is computed in AdhocTrainer. In fact, You don't have to specify learning rate in advance.
    warmup_steps = args.warmup_steps
    k = args.k1
    d_model = args.sep_bottleneck_channels
    init_lr = k * d_model ** (-0.5) * warmup_steps ** (-1.5)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))
    
    # Criterion
    if args.criterion == 'sisdr':
        criterion = NegSISDR()
    if args.criterion == 'sisnr':
        criterion = MyNegSISNR()
    elif args.criterion == 'pfp_sisnr':
        criterion = MyNegSISNR()
        criterion = CombinePFPLoss(criterion, 2000)
    elif args.criterion == 'demucs_mse':
        criterion = DEMUCSLoss('l2')
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    #pit_criterion = PIT1d(criterion, n_sources=args.n_sources)
    
    trainer = AdhocTrainer(model, loader, criterion, optimizer, args)
    trainer.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
