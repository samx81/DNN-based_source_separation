#!/usr/bin/env python
# -*- coding: utf-8 -*-

print()
import argparse
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import WaveTrainDataset, WaveEvalDataset, TrainDataLoader, EvalDataLoader
from new_dataset import WaveTrainDataset as NewTrainDataset
from adhoc_driver import AdhocTrainer
from models.galrnet import GALRNet
from criterion.sdr import NegSISDR, ThresholdedSNR
from criterion.stft_loss import DEMUCSLoss, MagMSELoss, CombinePFPLoss, CombineSISNRLoss
from criterion.distance import L2Loss
from driver import MyNegSISNR
from criterion.pit import PIT1d

parser = argparse.ArgumentParser(description="Training of Conv-TasNet")

parser.add_argument('--train_wav_root', type=str, default=None, help='Path for training dataset ROOT directory')
parser.add_argument('--valid_wav_root', type=str, default=None, help='Path for validation dataset ROOT directory')
parser.add_argument('--train_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tr_mix')
parser.add_argument('--valid_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_cv_mix')
parser.add_argument('--sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--conv', default=False, action='store_true')
parser.add_argument('--valid_duration', type=float, default=4, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--enc_basis', type=str, default='trainable', choices=['Deep_DCT', 'FiLM_DCT', 'DCT','TENET','TorchSTFT','DCCRN','DCTCN','trainable','Fourier','trainableFourier','trainableFourierTrainablePhase'], help='Encoder type')
parser.add_argument('--dec_basis', type=str, default='trainable', choices=['Deep_DCT','FiLM_DCT','DCT','TENET','TorchSTFT','DCCRN','DCTCN','trainable','Fourier','trainableFourier','trainableFourierTrainablePhase', 'pinv'], help='Decoder type')
parser.add_argument('--no-low-dim', dest='low_dim', action='store_false')
parser.add_argument('--noise_loss', default=False, action='store_true')
parser.add_argument('--local_att', default=False, action='store_true')
parser.set_defaults(low_dim=True)
parser.add_argument('--enc_nonlinear', type=str, default=None, help='Non-linear function of encoder')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--enc_onesided', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns kernel_size // 2 + 1 bins.')
parser.add_argument('--enc_return_complex', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns complex tensor, otherwise real tensor concatenated real and imaginary part in feature dimension.')
parser.add_argument('--n_bases', '-D', type=int, default=64, help='# bases')
parser.add_argument('--kernel_size', '-M', type=int, default=16, help='Kernel size')
parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size//2')
parser.add_argument('--sep_hidden_channels', '-H', type=int, default=128, help='Hidden channels of RNN in each direction')
parser.add_argument('--sep_chunk_size', '-K', type=int, default=100, help='Chunk size of separator')
parser.add_argument('--sep_hop_size', '-P', type=int, default=50, help='Hop size of separator')
parser.add_argument('--sep_down_chunk_size', '-Q', type=int, default=32, help='Downsampled chunk size of separator')
parser.add_argument('--sep_num_blocks', '-N', type=int, default=6, help='# blocks of separator. Each block has B layers')
parser.add_argument('--sep_num_heads', '-J', type=int, default=8, help='Number of heads in multi-head attention')
parser.add_argument('--sep_norm', type=int, default=1, help='Normalization')
parser.add_argument('--sep_dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['l2_sisnr','l2loss','pfp_sisnr','pfp_l2','pfp_thsnr','sisdr', 'this_sisdr', 'threshold_snr','demucs_l1','demucs_mse','mse'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 128')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--worker', type=int, default=16, help='Random seed')
parser.add_argument('--handcraft', type=int)
parser.add_argument('--intra_dropout', default=False, action='store_true')
parser.add_argument('--new_dset', default=False, action='store_true')
def main(args):
    set_seed(args.seed)
    
    samples = int(args.sr * args.duration)
    overlap = samples//2
    max_samples = int(args.sr * args.valid_duration)
    
    if args.new_dset:
        train_dataset = NewTrainDataset(args.train_wav_root, args.train_list_path, samples=samples, overlap=overlap, n_sources=args.n_sources,noise_loss=args.noise_loss,use_h5py=True)
    else:
        train_dataset = WaveTrainDataset(args.train_wav_root, args.train_list_path, samples=samples, overlap=overlap, n_sources=args.n_sources,noise_loss=args.noise_loss,use_h5py=True)
    
    valid_dataset = WaveEvalDataset(args.valid_wav_root, args.valid_list_path, max_samples=max_samples, n_sources=args.n_sources)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    dl_workers = args.worker
    print('Dataloader workers:', dl_workers)
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, num_workers=dl_workers,shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, num_workers=2, shuffle=False)
    if not args.enc_nonlinear:
        args.enc_nonlinear = None
    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None

    # print(f'uses low_dim :{args.low_dim}')
    model = GALRNet(
        args.n_bases, args.kernel_size, stride=args.stride, enc_basis=args.enc_basis, dec_basis=args.dec_basis, enc_nonlinear=args.enc_nonlinear, 
        window_fn=args.window_fn,enc_onesided=args.enc_onesided, enc_return_complex=args.enc_return_complex,
        sep_hidden_channels=args.sep_hidden_channels, 
        sep_chunk_size=args.sep_chunk_size, sep_hop_size=args.sep_hop_size, sep_down_chunk_size=args.sep_down_chunk_size, sep_num_blocks=args.sep_num_blocks,
        sep_num_heads=args.sep_num_heads, sep_norm=args.sep_norm, sep_dropout=args.sep_dropout,
        mask_nonlinear=args.mask_nonlinear,
        causal=args.causal, conv=args.conv,
        n_sources=args.n_sources, handcraft=args.handcraft,
        low_dimension=args.low_dim, local_att=args.local_att,intra_dropout=args.intra_dropout
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    # GALRNet returns waveform

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
        criterion = MyNegSISNR()
    elif args.criterion == 'this_sisdr':
        criterion = NegSISDR()
    elif args.criterion == 'threshold_snr':
        criterion = ThresholdedSNR()
    elif args.criterion == 'demucs_l1':
        criterion = DEMUCSLoss('l1')
    elif args.criterion == 'demucs_mse':
        criterion = DEMUCSLoss('l2')
    elif args.criterion == 'mse':
        criterion = MagMSELoss()
    elif args.criterion == 'l2loss':
        criterion = L2Loss()
    elif args.criterion == 'l2_sisnr':
        criterion = nn.MSELoss()
        criterion = CombineSISNRLoss(criterion, 0.7)
    elif args.criterion == 'pfp_sisnr':
        criterion = MyNegSISNR()
        criterion = CombinePFPLoss(criterion, 2000)
    elif args.criterion == 'pfp_thsnr':
        # criterion = MyNegSISNR()
        criterion = ThresholdedSNR()
        criterion = CombinePFPLoss(criterion, 1000)
    elif args.criterion == 'pfp_l2':
        # criterion = MyNegSISNR()
        criterion = DEMUCSLoss('l2')
        criterion = CombinePFPLoss(criterion, 1000)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    # pit_criterion = PIT1d(criterion, n_sources=args.n_sources)
    
    trainer = AdhocTrainer(model, loader, criterion, optimizer, args)
    trainer.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
