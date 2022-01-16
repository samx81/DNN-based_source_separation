import torch
import torch.nn as nn
import torch.nn.functional as F

from models.galrnet import GALRNet

class TwoHead(nn.Module):
    def __init__(self, model_me, model_cs):
        super().__init__()
        
        # Network confguration
        self.model_me = model_me
        self.model_cs = model_cs
        
        self.decouple = Decouple()
        self.couple = Couple()
            
        # self.net = nn.Sequential(*net)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """

        # 可以研究看看到底 Time Domain 是不是 == STFT Mag
        # 根據老師的回應不是
        me_waveform = self.model_me(input)

        _, phase = self.decouple(input)
        me_mag, _ = self.decouple(me_waveform)

        coupled = self.couple(me_mag, phase)
        cs_waveform = self.model_cs(coupled)
        

        return output

class Decouple(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        # Network confguration
        super(Decouple, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.window = self.window.cuda()


    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x_stft = torch.stft(input, self.fft_size, self.shift_size, self.win_length, self.window)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        
        mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
        spec = torch.atan2(imag, real)
        return mag, spec

class Couple(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        # Network confguration
        super(Couple, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.window = self.window.cuda()


    def forward(self, mag, phase):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        x = torch.istft(torch.cat([real,imag], dim=1), self.fft_size, self.shift_size, self.win_length, self.window)

        return x

class MultiComplexFeatureGALRNet(GALRNet):
    # Encoder needs a glu
    def extract_latent(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_bases, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_bases = self.n_bases
        kernel_size, stride = self.kernel_size, self.stride
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?,1,?), but given {}".format(input.size())
        
        padding = (stride - (T-kernel_size)%stride)%stride
        padding_left = padding//2
        padding_right = padding - padding_left

        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input) # [bs, feature, timestep]
        mask = self.separator(w) # [bs, sources, feature, timestep]
        w = w.unsqueeze(dim=1) # [bs, feature, timestep] -> [bs, sources, feature, timestep]
        w_hat = w * mask # [bs, sources, feature, timestep]
        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_bases, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent