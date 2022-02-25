# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.complex import STFT
from criterion.pfp_loss import PerceptualLoss, PerceptualLoss_Hubert
from criterion.sdr import NegSISDR
from dct import sdct_torch

def stft(x, fft_size, hop_size, win_length, window, full=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    if full:
        return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1) ,x_stft
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

class STFTMSELoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTMSELoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.window = self.window.cuda()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        mag_loss = F.mse_loss(torch.log(y_mag), torch.log(x_mag))

        return mag_loss

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.window = self.window.cuda()
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss

class STDCTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STDCTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = torch.hann_window
        # self.window = self.window.cuda()
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = sdct_torch(x, self.fft_size, self.win_length, self.shift_size, window=self.window)
        y_mag = sdct_torch(y, self.fft_size, self.win_length, self.shift_size, window=self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss

class DEMUCSLoss(torch.nn.Module):
    """DEMUCS loss module."""

    def __init__(self, loss, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(DEMUCSLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()
        elif loss == 'l2':
            self.loss = nn.MSELoss()
        elif loss == 'huber':
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Invalid loss {loss}")

        self.stft_loss = STFTLoss(fft_size, shift_size, win_length, window)


    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T). Est
            y (Tensor): Groundtruth signal (B, T). Clean
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        loss = self.loss(y,x) # torch.loss -> clean, estimate
        sc_loss, mag_loss = self.stft_loss(x, y)

        loss += sc_loss + mag_loss
        return loss

class DCTDEMUCSLoss(torch.nn.Module):
    """DEMUCS loss module."""

    def __init__(self, loss, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(DCTDEMUCSLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()
        elif loss == 'l2':
            self.loss = nn.MSELoss()
        elif loss == 'huber':
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Invalid loss {loss}")

        self.stdct_loss = STDCTLoss(fft_size, shift_size, win_length, window)


    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T). Est
            y (Tensor): Groundtruth signal (B, T). Clean
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        loss = self.loss(y,x) # torch.loss -> clean, estimate
        sc_loss, mag_loss = self.stdct_loss(x, y)

        loss += sc_loss + mag_loss
        return loss

class MagMAELoss(torch.nn.Module):
    """DEMUCS loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(MagMAELoss, self).__init__()

        self.stft_loss = STFTLoss(fft_size, shift_size, win_length, window)


    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T). Est
            y (Tensor): Groundtruth signal (B, T). Clean
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        sc_loss, mag_loss = self.stft_loss(x, y)

        loss = mag_loss
        return loss
    
class FSNetLoss(torch.nn.Module):
    """DEMUCS loss module."""

    def __init__(self, alpha=0.4, fft_size=512, shift_size=100, win_length=500, window="hanning"):
        """Initialize STFT loss module."""
        super(FSNetLoss, self).__init__()

        self.time_loss = nn.MSELoss()
        self.fft_size, self.win_length, self.shift_size, self.window = fft_size, win_length, shift_size, window
        # self.complex_stft = STFT(fft_size, win_length, shift_size, window)
        self.window = torch.hann_window(self.fft_size).cuda() # should be win_length?
        self.spectral_loss = nn.L1Loss()
        
        self.alpha = alpha

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T). Est
            y (Tensor): Groundtruth signal (B, T). Clean
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        loss = self.alpha * self.time_loss(y,x) # torch.loss -> clean, estimate

        est_real, est_imag = torch.chunk(torch.stft(x, self.fft_size, 
                                            hop_length=self.shift_size, window=self.window),
                                        2, 2)
        clean_real, clean_imag = torch.chunk(torch.stft(y, self.fft_size, 
                                            hop_length=self.shift_size, window=self.window),
                                        2, 2)

        spectral_loss = self.spectral_loss(clean_real, est_real) + self.spectral_loss(clean_imag, est_imag)

        loss += (1 - self.alpha) * spectral_loss

        return loss

class CombinePFPLoss(torch.nn.Module):
    """DEMUCS loss module."""

    def __init__(self, loss, weight=2000, pfp_type='wav2vec'):
        """Initialize STFT loss module."""
        super(CombinePFPLoss, self).__init__()
        self.loss = loss
        self.weight = weight
        if pfp_type == 'wav2vec':
            self.pfp_loss = PerceptualLoss(model_type='wav2vec',loss_type="lp",
                PRETRAINED_MODEL_PATH='pretrain/wav2vec_large.pt').cuda()
        elif pfp_type == 'Hubert':
            self.pfp_loss = PerceptualLoss_Hubert()
        else:
            print('PFP Not implemented.')

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T). Est
            y (Tensor): Groundtruth signal (B, T). Clean
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        loss = self.loss(x, y)
        loss += self.weight * self.pfp_loss(x,y)


        return loss

class CombineSISNRLoss(torch.nn.Module):
    """DEMUCS loss module."""

    def __init__(self, loss, aux_weight=0.5):
        """Initialize STFT loss module."""
        super(CombineSISNRLoss, self).__init__()
        self.aux_loss = loss
        self.weight = aux_weight
        
        self.snr_loss = NegSISDR()

    def forward(self, x, y, latent_x=None, latent_y=None,):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T). Est
            y (Tensor): Groundtruth signal (B, T). Clean
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        loss = (1 - self.weight) * self.snr_loss(x, y)
        if latent_x:
            aux_loss =self.aux_loss(latent_x , latent_y)
        else:
            aux_loss = self.aux_loss(x , y)
        loss += self.weight * aux_loss


        return loss

class Simple_Loss(torch.nn.Module):
    """DEMUCS loss module."""

    def __init__(self, loss_fn, use_latent=False):
        """Initialize STFT loss module."""
        super(Simple_Loss, self).__init__()
        self.loss_fn = loss_fn
        self.use_latent = use_latent
        
        # self.snr_loss = NegSISDR()

    def forward(self, x, y, latent_x, latent_y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T). Est
            y (Tensor): Groundtruth signal (B, T). Clean
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        if self.use_latent:
            loss = self.loss_fn(latent_x, latent_y)
        else:
            loss = self.loss_fn(x, y)

        return loss

class STFTCombineDomain_Loss(torch.nn.Module):
    """DEMUCS loss module."""
        
    def __init__(self, loss_fn, complex_weight=0.3, fft_size=1024, shift_size=120, win_length=600, window="hann_window", consistency=False):
        """Initialize STFT loss module."""
        super(STFTMagDomain_Loss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.window = self.window.cuda()
        self.loss_fn = loss_fn
        self.complex_weight = complex_weight

    def forward(self, x, y, latent_x=None, latent_y=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag, x_stft = stft(x, self.fft_size, self.shift_size, self.win_length, self.window, full=True)
        y_mag, y_stft = stft(y, self.fft_size, self.shift_size, self.win_length, self.window, full=True)
        loss = (1 - self.complex_weight) * self.loss_fn(x_mag, y_mag)
        loss += self.complex_weight * self.loss_fn(x_stft, y_stft)

        return loss

class STFTMagDomain_Loss(torch.nn.Module):
    """DEMUCS loss module."""
        
    def __init__(self, loss_fn, aux_loss=None, weight=0.3, fft_size=1024, shift_size=120, win_length=600, window="hann_window", consistency=False):
        """Initialize STFT loss module."""
        super(STFTMagDomain_Loss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.window = self.window.cuda()
        self.loss_fn = loss_fn
        self.aux_loss = aux_loss
        self.weight = weight

    def forward(self, x, y, latent_x=None, latent_y=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)

        if self.aux_loss:
            loss = self.weight * self.aux_loss(x, y) + (1 - self.weight) * self.loss_fn(x_mag, y_mag)
        else:
            loss = self.loss_fn(x_mag, y_mag)

        return loss

class STDCTDomain_Loss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, loss_fn, fft_size=512, shift_size=128, win_length=512, window="hann_window"):
        """Initialize STFT loss module."""
        super(STDCTDomain_Loss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = torch.hann_window
        # self.window = self.window.cuda()
        self.loss_fn = loss_fn

    def forward(self, x, y, latent_x=None, latent_y=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = sdct_torch(x, self.fft_size, self.win_length, self.shift_size, window=self.window)
        y_mag = sdct_torch(y, self.fft_size, self.win_length, self.shift_size, window=self.window)
        loss = self.loss_fn(x_mag, y_mag)

        return loss
