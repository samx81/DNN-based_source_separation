from re import X
from sys import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils_filterbank import choose_filterbank
from utils.utils_tasnet import choose_layer_norm
from models.gtu import GTU1d
from models.transform import Segment1d, OverlapAdd1d
from models.galr import GALR 
from models.galr_imp import GALR as GALR_GRU
from models.galr_imp import GALR_Res
from dccrn import DCCRN_Encoder,DCCRN_Decoder, DCTCN_Encoder, DCTCN_Decoder, TSTNN_Encoder, TSTNN_Decoder
from models.aligndenoise import TransformerDecoder, GALRDecoder

EPS=1e-12

class GALRNet(nn.Module):
    def __init__(
        self,
        n_basis, kernel_size, stride=None, enc_basis=None, dec_basis=None,
        sep_hidden_channels=128,
        sep_chunk_size=100, sep_hop_size=50, sep_down_chunk_size=None, sep_num_blocks=6,
        sep_num_heads=8, sep_norm=True, sep_dropout=0.1,
        mask_nonlinear='relu',
        causal=True,
        n_sources=2,
        low_dimension=True,
        eps=EPS,
        **kwargs
    ):
        super().__init__()
        
        if stride is None:
            stride = kernel_size // 2
        
        assert kernel_size % stride == 0, "kernel_size is expected divisible by stride"
        
        # Encoder-decoder
        self.n_basis = n_basis
        self.kernel_size, self.stride = kernel_size, stride
        self.enc_basis, self.dec_basis = enc_basis, dec_basis
        print(enc_basis, dec_basis)
        if enc_basis == 'trainable' and not dec_basis == 'pinv':    
            self.enc_nonlinear = kwargs['enc_nonlinear']
        else:
            self.enc_nonlinear = None

        # needed to implement 'Complex' 10/12    
        
        if enc_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase'] or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']:
            self.window_fn = kwargs['window_fn']
            self.enc_onesided, self.enc_return_complex = kwargs['enc_onesided'], kwargs['enc_return_complex']
        else:
            self.window_fn = None
            self.enc_onesided, self.enc_return_complex = None, None
        
        
        print(f'window_fn:{self.window_fn}')
        # Separator configuration
        self.sep_hidden_channels = sep_hidden_channels
        self.sep_chunk_size, self.sep_hop_size, self.sep_down_chunk_size = sep_chunk_size, sep_hop_size, sep_down_chunk_size
        self.sep_num_blocks = sep_num_blocks
        self.sep_num_heads = sep_num_heads
        self.sep_norm = sep_norm
        self.sep_dropout = sep_dropout
        self.low_dimension = low_dimension
        
        self.causal = causal
        self.sep_norm = sep_norm
        self.mask_nonlinear = mask_nonlinear
        
        
        self.n_sources = n_sources
        self.eps = eps
        
        # Network configuration
        if 'DCCRN' in [enc_basis, dec_basis]:
            encoder, decoder = DCCRN_Encoder(kernel_num=[32,64,64,64], kernel_size=3), DCCRN_Decoder(kernel_num=[32,64,64,64], kernel_size=3)
        elif 'DCTCN' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            # encoder, decoder = DCTCN_Encoder(causal=causal), DCTCN_Decoder()
            decoder = TSTNN_Decoder(n_basis, 2, kernel_size, stride=stride, **kwargs)
            encoder = TSTNN_Encoder(2, n_basis, kernel_size, stride=stride, **kwargs)
        else:
            encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        random_mask = kwargs.get('random_mask', None)
        self.local_att = kwargs.get('local_att', None)
        print(kwargs)

        self.encoder = encoder
        self.separator = Separator(
            n_basis, hidden_channels=sep_hidden_channels,
            chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
            num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
            low_dimension=low_dimension,
            causal=causal,
            n_sources=n_sources,
            eps=eps, random_mask=random_mask,local_att=self.local_att
        )
        self.decoder = decoder
        
        self.num_parameters = self._get_num_parameters()
        
        # Load custom child code
        self.load()
        
    def load(self):
        pass


    def forward(self, input, placehold=None):
        output, latent, output_denoise, latent_denoise = self.extract_latent(input)
        
        return output, latent, output_denoise, latent_denoise
        
    def extract_latent(self, input):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_basis, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
        kernel_size, stride = self.kernel_size, self.stride
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())
        
        padding = (stride - (T - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left
        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)
        # print(w.shape)
        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        else:
            # mask = self.separator(w)
            mask, mask_denoise = self.separator(w)

            w = w.unsqueeze(dim=1)
            w_hat = w * mask
            if mask_denoise is None:
                w_hat_denoise_lst = None
            else:
                w_hat_denoise_lst = []
                for i in mask_denoise:
                    w_hat_denoise_lst.append(w * i)

        latent = mask
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        # output = x_hat
        output = F.pad(x_hat, (-padding_left, -padding_right))

        if mask_denoise is not None:
            output_denoise = []
            for w_hat_denoise in w_hat_denoise_lst:
                latent_denoise = w_hat_denoise
                w_hat_denoise = w_hat_denoise.view(batch_size*n_sources, n_basis, -1)
                x_hat_denoise = self.decoder(w_hat_denoise)
                x_hat_denoise = x_hat_denoise.view(batch_size, n_sources, -1)
                # output_denoise = x_hat_denoise
                output_denoise.append(F.pad(x_hat_denoise, (-padding_left, -padding_right)))
        else:
            output_denoise, latent_denoise = None, None
        return output, latent, output_denoise, latent_denoise
    
    def get_config(self):
        config = {
            'n_basis': self.n_basis,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_basis': self.enc_basis,
            'dec_basis': self.dec_basis,
            'enc_nonlinear': self.enc_nonlinear,
            'window_fn': self.window_fn,
            'enc_onesided': self.enc_onesided,
            'enc_return_complex': self.enc_return_complex,
            'sep_hidden_channels': self.sep_hidden_channels,
            'sep_chunk_size': self.sep_chunk_size,
            'sep_hop_size': self.sep_hop_size,
            'sep_down_chunk_size': self.sep_down_chunk_size,
            'sep_num_blocks': self.sep_num_blocks,
            'sep_num_heads': self.sep_num_heads,
            'sep_norm': self.sep_norm,
            'sep_dropout': self.sep_dropout,
            'low_dimension': self.low_dimension,
            'mask_nonlinear': self.mask_nonlinear,
            'causal': self.causal,
            'n_sources': self.n_sources,
            'eps': self.eps,
            'local_att': self.local_att
        }
    
        return config

    @classmethod
    def build_model(cls, model_path):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)

        n_basis = config.get('n_basis') or config['n_bases']
        kernel_size, stride = config['kernel_size'], config['stride']
        enc_basis, dec_basis = config.get('enc_bases') or config['enc_basis'], config.get('dec_bases') or config['dec_basis']
        print(enc_basis, dec_basis)
        enc_nonlinear = config['enc_nonlinear']
        enc_onesided, enc_return_complex = config.get('enc_onesided') or None, config.get('enc_return_complex') or None
        window_fn = config['window_fn']
        
        sep_hidden_channels = config['sep_hidden_channels']
        sep_chunk_size, sep_hop_size = config['sep_chunk_size'], config['sep_hop_size']
        sep_down_chunk_size, sep_num_blocks = config['sep_down_chunk_size'], config['sep_num_blocks']
        
        sep_norm = config['sep_norm']
        sep_dropout, sep_num_heads = config['sep_dropout'], config['sep_num_heads']
        mask_nonlinear = config['mask_nonlinear']

        causal = config['causal']
        n_sources = config['n_sources']
        low_dimension = config['low_dimension']
        
        eps = config['eps']
        local_att = config.get('local_att', False)
        
        model = cls(
            n_basis, kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear, 
            enc_onesided=enc_onesided, enc_return_complex=enc_return_complex,
            window_fn=window_fn,sep_hidden_channels=sep_hidden_channels, 
            sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size, sep_down_chunk_size=sep_down_chunk_size, sep_num_blocks=sep_num_blocks,
            sep_num_heads=sep_num_heads, sep_norm=sep_norm, sep_dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources,
            low_dimension=low_dimension,
            eps=eps, local_att=local_att
        )
        
        return model
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class GALRNet_Res(GALRNet):
    def __init__(
        self,*args, **kwargs
    ):
        super().__init__(*args,**kwargs)
        n_basis = args[0]
        sep_hidden_channels = kwargs['sep_hidden_channels']
        sep_chunk_size = kwargs['sep_chunk_size']
        sep_hop_size = kwargs['sep_hop_size']
        sep_down_chunk_size = kwargs['sep_down_chunk_size']
        sep_num_blocks = kwargs['sep_num_blocks']
        sep_num_heads = kwargs['sep_num_heads']
        sep_norm = kwargs['sep_norm']
        sep_dropout = kwargs['sep_dropout']
        mask_nonlinear = kwargs['mask_nonlinear']
        low_dimension = kwargs['low_dimension']
        causal = kwargs['causal']
        n_sources = kwargs['n_sources']
        eps = kwargs.get('eps', EPS)

        self.separator = Separator_Res(
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps
            )
    
    def diffusion4(self, x, var, mask, noise_level=None):
        batch_size = x.shape[0]
        #Uniform index
        idx = torch.randint(0, len(self.betas), size=(batch_size,))
        lb = self.alphas_bar[idx + 1]
        ub = self.alphas_bar[idx]
        if not noise_level:
            noise_level = torch.rand(size=(batch_size,)) * (ub - lb) + lb
            noise_level = noise_level.unsqueeze(-1).unsqueeze(-1).to(device=x.device)
        noisy_x = torch.sqrt(noise_level) * x + torch.sqrt(1.0 - noise_level) * torch.randn_like(x) * var
        return x * mask + noisy_x * (1 - mask)

    def forward(self, input, source=None):
        output, latent, output_denoise, latent_denoise = self.extract_latent(input, source)
        
        return output, latent, output_denoise, latent_denoise
        
    def extract_latent(self, input, source=None):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_basis, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
        kernel_size, stride = self.kernel_size, self.stride
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())

        padding = (stride - (T - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left
        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)

        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        else:
            # mask = self.separator(w)
            mask, mask_denoise = self.separator(w)

            w = w.unsqueeze(dim=1)
            w_hat = w * mask
            if mask_denoise is None:
                w_hat_denoise_lst = None
            else:
                w_hat_denoise_lst = []
                for i in mask_denoise:
                    w_hat_denoise_lst.append(w * i)

        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        # output = x_hat
        output = F.pad(x_hat, (-padding_left, -padding_right))

        if mask_denoise is not None:
            if source is not None:
                source = source.unsqueeze(dim=1)
                source = F.pad(source, (padding_left, padding_right))
                o = self.encoder(source)
                # print(w.shape)
                oracle_mask = w / o
            output_denoise = []
            for w_hat_denoise in w_hat_denoise_lst:
                latent_denoise = w_hat_denoise

                # # produce sample
                # var = torch.max
                # mask = 

                w_hat_denoise = w_hat_denoise.view(batch_size*n_sources, n_basis, -1)
                x_hat_denoise = self.decoder(w_hat_denoise)
                x_hat_denoise = x_hat_denoise.view(batch_size, n_sources, -1)
                # output_denoise = x_hat_denoise
                output_denoise.append(F.pad(x_hat_denoise, (-padding_left, -padding_right)))
        else:
            output_denoise, latent_denoise = None, None
        return output, latent, output_denoise, latent_denoise

class Mask_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
    
    def forward(self, x, y):
        loss = self.l1(y, x) + self.l2(y, x)
        return loss

class GALRNet_Denoise(GALRNet):
    def __init__(
        self,*args, **kwargs
    ):
        super().__init__(*args,**kwargs)
        n_basis = args[0]
        sep_hidden_channels = kwargs['sep_hidden_channels']
        sep_chunk_size = kwargs['sep_chunk_size']
        sep_hop_size = kwargs['sep_hop_size']
        sep_down_chunk_size = kwargs['sep_down_chunk_size']
        sep_num_blocks = kwargs['sep_num_blocks']
        sep_num_heads = kwargs['sep_num_heads']
        sep_norm = kwargs['sep_norm']
        sep_dropout = kwargs['sep_dropout']
        mask_nonlinear = kwargs['mask_nonlinear']
        low_dimension = kwargs['low_dimension']
        causal = kwargs['causal']
        n_sources = kwargs['n_sources']
        eps = kwargs.get('eps', EPS)

        self.separator = Separator_Denoise(
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps
            )
        self.denoiser = GALRDecoder(
                n_basis, sep_hidden_channels,
                chunk_size=sep_chunk_size, down_chunk_size=sep_down_chunk_size,
                num_blocks=4, num_heads=sep_num_heads//2,
                norm=sep_norm, dropout=sep_dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
            )

        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        self.mask_comp = nn.Linear(n_basis, n_basis)
        self.mask_comp_norm = choose_layer_norm('gLN', n_basis, causal=causal, eps=eps)

        self.betas = np.linspace(0.08, 0.2, 50)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.tensor(np.concatenate(([1], np.cumprod(self.alphas))), dtype=torch.float32)

        if self.mask_nonlinear == 'relu':
            self.mask_nonlinear_fn = nn.ReLU()
        elif self.mask_nonlinear == 'sigmoid':
            self.mask_nonlinear_fn = nn.Sigmoid()
        elif self.mask_nonlinear == 'softmax':
            self.mask_nonlinear_fn = nn.Softmax(dim=1)
        elif self.mask_nonlinear == 'tanh':
            self.mask_nonlinear_fn = nn.Tanh()

        self.mask_loss = Mask_Loss()

    def diffusion4(self, x, gt, var, mask, noise_level=None):
        batch_size = x.shape[0]
        #Uniform index
        idx = torch.randint(0, len(self.betas), size=(batch_size,))
        lb = self.alphas_bar[idx + 1]
        ub = self.alphas_bar[idx]
        if not noise_level:
            noise_level = torch.rand(size=(batch_size,)) * (ub - lb) + lb
            noise_level = noise_level.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device=x.device)
        noisy_x = torch.sqrt(noise_level) * gt + torch.sqrt(1.0 - noise_level) * torch.randn_like(x) * var
        return x * mask + noisy_x * (1 - mask)

    def forward(self, input, source=None):
        output, latent, output_denoise, latent_denoise = self.extract_latent(input, source)
        
        return output, latent, output_denoise, latent_denoise
        
    def extract_latent(self, input, source=None):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_basis, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
        kernel_size, stride = self.kernel_size, self.stride
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())

        padding = (stride - (T - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left
        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)

        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        else:
            mask = self.separator(w)

            w = w.unsqueeze(dim=1)
            w_hat = w * mask

        batch_size, n_sources, num_features, n_frames = mask.size()

        if source is not None:
            with torch.no_grad():
                source = source.unsqueeze(dim=1)
                source = F.pad(source, (padding_left, padding_right))
                o = self.encoder(source)
                o = o.unsqueeze(dim=1)

            mix = w + EPS
            oracle_mask = o / mix

            mean = (0.3 * oracle_mask + 0.7 * mask) / 2
            var = 0.3* oracle_mask - mean
            # var = oracle_mask + (mask - oracle_mask) * 0.7

            keep_mask = torch.abs(oracle_mask - mask).le(0.01).float()

            # print(torch.sum(keep_mask), torch.mean(torch.abs(oracle_mask - mask)), flush=True)

            noise_mask = self.mask_nonlinear_fn(self.diffusion4(mask, mean, torch.abs(var), keep_mask))
        else:
            noise_mask = mask

        x = mask.view(batch_size*n_sources, num_features, n_frames)
        noise_mask = noise_mask.view(batch_size*n_sources, num_features, n_frames)

        denoise_padding = (self.sep_hop_size-(n_frames-self.sep_chunk_size)%self.sep_hop_size)%self.sep_hop_size
        denoise_padding_left = denoise_padding//2
        denoise_padding_right = denoise_padding - denoise_padding_left
        
        denoise_x = F.pad(noise_mask, (denoise_padding_left, denoise_padding_right))

        x = F.pad(x, (denoise_padding_left, denoise_padding_right))
        x = self.separator.segment1d(x)

        denoise_x = self.separator.segment1d(denoise_x) # -> (batch_size, C, S, chunk_size)
        
        denoise_x = self.separator.norm2d(denoise_x)
        
        denoise_x = self.denoiser(denoise_x, x) # -> 傳入 encoder x?
        denoise_x = self.separator.overlap_add1d(denoise_x)
        denoise_x = F.pad(denoise_x, (-denoise_padding_left, -denoise_padding_right))

        denoise_x = denoise_x.view(batch_size, n_sources, num_features, n_frames)

        denoise_x += mask
        # denoise_x = self.mask_comp(torch.cat([denoise_x.transpose(-1,-2), mask.transpose(-1,-2)], dim=-1))
        # denoise_x = denoise_x.transpose(-1,-2)

        # TODO: Remove this debug 
        if source is not None:
            # latent_denoise = self.mask_loss(torch.clamp(oracle_mask, -1, 1), torch.clamp(denoise_x, -1, 1)) * 0.7
            # latent_denoise += self.mask_loss(torch.clamp(oracle_mask, -1, 1), torch.clamp(mask, -1, 1)) * 0.3
            latent_denoise = self.mask_loss(oracle_mask, denoise_x) * 0.7
            latent_denoise += self.mask_loss(oracle_mask, mask) * 0.3
            latent_denoise = torch.clamp(latent_denoise, 0, 10)
            # latent_denoise = None
            # latent_denoise = output_denoise
        else:
            output_denoise, latent_denoise = None, None

        w_hat_denoise = w * denoise_x
        # latent_denoise = w_hat_denoise

        w_hat_denoise = w_hat_denoise.view(batch_size*n_sources, n_basis, -1)
        x_hat_denoise = self.decoder(w_hat_denoise)
        x_hat_denoise = x_hat_denoise.view(batch_size, n_sources, -1)

        output_denoise = F.pad(x_hat_denoise, (-padding_left, -padding_right))
        output_denoise = [output_denoise]

        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)
        x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        # output = x_hat
        output = F.pad(x_hat, (-padding_left, -padding_right))

        with torch.no_grad():
            x = self.denoiser(x, x) # -> 傳入 encoder x?
            x = self.separator.overlap_add1d(x)
            x = F.pad(x, (-denoise_padding_left, -denoise_padding_right))
            x = x.view(batch_size, n_sources, num_features, n_frames)
            x += mask
            latent = w * x

            latent = latent.view(batch_size*n_sources, n_basis, -1)
            latent = self.decoder(latent)
            latent = latent.view(batch_size, n_sources, -1)
            latent = F.pad(latent, (-padding_left, -padding_right))

        return output, latent, output_denoise, latent_denoise

class GALRNet_Denoise2(GALRNet):
    def __init__(
        self,*args, **kwargs
    ):
        super().__init__(*args,**kwargs)
        n_basis = args[0]
        sep_hidden_channels = kwargs['sep_hidden_channels']
        sep_chunk_size = kwargs['sep_chunk_size']
        sep_hop_size = kwargs['sep_hop_size']
        sep_down_chunk_size = kwargs['sep_down_chunk_size']
        sep_num_blocks = kwargs['sep_num_blocks']
        sep_num_heads = kwargs['sep_num_heads']
        sep_norm = kwargs['sep_norm']
        sep_dropout = kwargs['sep_dropout']
        mask_nonlinear = kwargs['mask_nonlinear']
        low_dimension = kwargs['low_dimension']
        causal = kwargs['causal']
        n_sources = kwargs['n_sources']
        eps = kwargs.get('eps', EPS)

        self.separator = Separator_Denoise(
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps
            )
        self.denoiser = Denoiser(
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, 
                num_blocks=4, num_heads=sep_num_heads // 2, 
                norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal, n_sources=n_sources,
                eps=eps
            )

        self.mask_loss = Mask_Loss()

    def forward(self, input, source=None):
        output, latent, output_denoise, latent_denoise = self.extract_latent(input, source)
        
        return output, latent, output_denoise, latent_denoise
        
    def extract_latent(self, input, source=None):
        """
        Args:
            input (batch_size, 1, T)
        Returns:
            output (batch_size, n_sources, T)
            latent (batch_size, n_sources, n_basis, T'), where T' = (T-K)//S+1
        """
        n_sources = self.n_sources
        n_basis = self.n_basis
        kernel_size, stride = self.kernel_size, self.stride
        
        batch_size, C_in, T = input.size()
        
        assert C_in == 1, "input.size() is expected (?, 1, ?), but given {}".format(input.size())

        padding = (stride - (T - kernel_size) % stride) % stride
        padding_left = padding // 2
        padding_right = padding - padding_left
        input = F.pad(input, (padding_left, padding_right))
        w = self.encoder(input)

        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        else:
            mask = self.separator(w)

            w = w.unsqueeze(dim=1)
            w_hat = w * mask # (batchsize, n_source, feature, time)

        if source is not None:
            with torch.no_grad():
                source = source.unsqueeze(dim=1)
                source = F.pad(source, (padding_left, padding_right))
                o = self.encoder(source)
                o = o.unsqueeze(dim=1)
                mix = w + EPS
                oracle_mask = o / mix
                oracle_mask = F.relu(oracle_mask)
            
            denoise_x, post = self.denoiser(w_hat, w_hat, o)
        else:
            denoise_x, post = self.denoiser(w_hat, w_hat)
        #    denoise_x = mask
            # (bs, n_source, feat, time)

        # TODO: Remove this debug 
        if source is not None:
            #latent_denoise = self.mask_loss(post.unsqueeze(dim=1), mask)
            # latent_denoise = self.mask_loss(oracle_mask, denoise_x) * 0.7
            # latent_denoise += self.mask_loss(oracle_mask, mask) * 0.3
            # latent_denoise = torch.clamp(latent_denoise, 0, 10)
             latent_denoise = None
            # latent_denoise = output_denoise
        else:
            output_denoise, latent_denoise = None, None

        w_hat_denoise = w * denoise_x

        w_hat_denoise = w_hat_denoise.view(batch_size*n_sources, n_basis, -1)
        x_hat_denoise = self.decoder(w_hat_denoise)
        x_hat_denoise = x_hat_denoise.view(batch_size, n_sources, -1)

        output_denoise = F.pad(x_hat_denoise, (-padding_left, -padding_right))
        output_denoise = [output_denoise]

        # w_hat_denoise = w * (denoise_x * 0.3 + mask * 0.7) 

        # w_hat_denoise = w_hat_denoise.view(batch_size*n_sources, n_basis, -1)
        # x_hat_denoise = self.decoder(w_hat_denoise)
        # x_hat_denoise = x_hat_denoise.view(batch_size, n_sources, -1)

        # output_denoise = F.pad(x_hat_denoise, (-padding_left, -padding_right))
        # output_denoise = [output_denoise]

        latent = w_hat
        x_hat = w_hat.view(batch_size*n_sources, n_basis, -1)
        x_hat = self.decoder(x_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        # output = x_hat
        output = F.pad(x_hat, (-padding_left, -padding_right))

        with torch.no_grad():
            no_diffusion, post = self.denoiser(w_hat, w_hat) # -> 傳入 encoder x?
            # no_diffusion = mask
            latent = w * no_diffusion

            latent = latent.view(batch_size*n_sources, n_basis, -1)
            latent = self.decoder(latent)
            latent = latent.view(batch_size, n_sources, -1)
            latent = F.pad(latent, (-padding_left, -padding_right))

            raw_diff, post = self.denoiser(w_hat, w_hat, diff=True)
            raw_diff = w * raw_diff
            raw_diff = raw_diff.view(batch_size*n_sources, n_basis, -1)
            raw_diff = self.decoder(raw_diff)
            raw_diff = raw_diff.view(batch_size, n_sources, -1)
            raw_diff = F.pad(raw_diff, (-padding_left, -padding_right))
            output_denoise.insert(0, raw_diff)


        return output, latent, output_denoise, latent_denoise


class GALRNet_Res_NoDeno(GALRNet):
    def __init__(
        self,*args, **kwargs
    ):
        super().__init__(*args,**kwargs)
        n_basis = args[0]
        sep_hidden_channels = kwargs['sep_hidden_channels']
        sep_chunk_size = kwargs['sep_chunk_size']
        sep_hop_size = kwargs['sep_hop_size']
        sep_down_chunk_size = kwargs['sep_down_chunk_size']
        sep_num_blocks = kwargs['sep_num_blocks']
        sep_num_heads = kwargs['sep_num_heads']
        sep_norm = kwargs['sep_norm']
        sep_dropout = kwargs['sep_dropout']
        mask_nonlinear = kwargs['mask_nonlinear']
        low_dimension = kwargs['low_dimension']
        causal = kwargs['causal']
        n_sources = kwargs['n_sources']
        eps = kwargs.get('eps', EPS)

        self.separator = Separator_Res_NoDeno(
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps
            )

class PosteriorConverter(nn.Module):
    def __init__(self, in_feat, out_feat, causal=True, dropout=0.1, mask_nonlinear='softmax', eps=EPS):
        super().__init__()

        self.conv1 = nn.Conv2d(in_feat, out_feat, 1)
        self.conv2 = nn.Conv2d(out_feat, out_feat, 1)
        norm_name = 'cLN' if causal else 'gLN'
        self.norm = choose_layer_norm(norm_name, out_feat, causal=causal, eps=eps)
        self.fc = nn.Linear(out_feat, out_feat)

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        elif mask_nonlinear == 'tanh':
            self.mask_nonlinear = nn.Tanh()
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
    
    def forward(self, input):
        
        x = self.conv1(input)
        x = self.conv2(x)

        if self.dropout:
            x = self.dropout1d(x)
            
        x = self.norm(x)
        x = self.fc(x.transpose(1,-1)).transpose(1,-1)

        x = self.mask_nonlinear(x)

        return x


class Denoiser(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        random_mask=False,
        local_att=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        self.decoder = GALRDecoder(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
            )

        self.betas = np.linspace(0.08, 0.2, 50)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.tensor(np.concatenate(([1], np.cumprod(self.alphas))), dtype=torch.float32)

        self.posterior = PosteriorConverter(num_features, num_features * 4, causal=causal, eps=eps, mask_nonlinear='sigmoid')
        self.fc = nn.Linear(num_features * 4, num_features)

        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)
        
        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        elif mask_nonlinear == 'tanh':
            self.mask_nonlinear = nn.Tanh()
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
            
    def diffusion4(self, x, var, mask, noise_level=None):
        batch_size = x.shape[0]
        chunks = x.shape[2]
        #Uniform index
        idx = torch.randint(0, len(self.betas), size=(batch_size, chunks,))
        lb = self.alphas_bar[idx + 1]
        ub = self.alphas_bar[idx]
        if not noise_level:
            noise_level = torch.rand(size=(batch_size, chunks,)) * (ub - lb) + lb
            noise_level = noise_level.unsqueeze(1).unsqueeze(-1).to(device=x.device)
        noisy_x = torch.sqrt(noise_level) * x + torch.sqrt(1.0 - noise_level) * torch.randn_like(x) * var
        # return noisy_x
        return x * mask + noisy_x * (1 - mask)

    def forward(self, x, src, source=None, diff=False):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, n_sources, num_features, n_frames = x.size()
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left

        src = src.view(batch_size*n_sources, num_features, n_frames)     
        src = F.pad(src, (padding_left, padding_right))
        src = self.segment1d(src) # -> (batch_size, C, S, chunk_size)
        src = self.norm2d(src)

        src = self.posterior(src)
        src = self.fc(src.transpose(1,-1)).transpose(1,-1)
        # src = src.squeeze()

        x = x.view(batch_size*n_sources, num_features, n_frames)
        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)

        if source is not None:
            # mean = (source + x) / 2
            # var = source - mean
            source = source.view(batch_size*n_sources, num_features, n_frames)
            source = F.pad(source, (padding_left, padding_right))
            source = self.segment1d(source) # -> (batch_size, C, S, chunk_size)
            source = self.norm2d(source)

            source_post = self.posterior(source)
            x           = self.posterior(x)


            # 這裡可以再調整，如讓 freq bin 被收縮後，若相近就保留
            keep_mask = torch.abs(source_post - x).le(0.01).float()

            var = torch.max(source_post * 0.3, x)
 
            noisy_x = self.diffusion4(x, var, keep_mask)

            # x = self.fc(x.transpose(-1,-2)).transpose(-1,-2)
            noisy_x = self.fc(noisy_x.transpose(1,-1)).transpose(1,-1)
            
            # x = x * keep_mask + noisy_x * (1 - keep_mask)
            x = noisy_x
        else:
            x = self.posterior(x)
            if diff:
                x = self.diffusion4(x, x, torch.zeros_like(x))
            x = self.fc(x.transpose(1,-1)).transpose(1,-1)
            # x = x.squeeze()

        x_post = self.decoder(x, src)
        x_post = self.overlap_add1d(x_post)
        x_post = F.pad(x_post, (-padding_left, -padding_right))

        x_post = self.prelu(x_post) # -> (batch_size, C, n_frames)
        x_post = x_post.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x_post = self.gtu(x_post) # -> (batch_size*n_sources, num_features, n_frames)
        x_post = self.mask_nonlinear(x_post) # -> (batch_size*n_sources, num_features, n_frames)
        output = x_post.view(batch_size, n_sources, num_features, n_frames)

        raw_post = self.overlap_add1d(x)
        raw_post = F.pad(raw_post, (-padding_left, -padding_right))
        
        return output, raw_post

class Separator(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        random_mask=False,
        local_att=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )
        else:
            self.galr = GALR(
                num_features, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )
        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(num_features, n_sources*num_features, kernel_size=1, stride=1)
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)
        
        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        elif mask_nonlinear == 'tanh':
            self.mask_nonlinear = nn.Tanh()
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
            
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, n_frames = input.size()
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left
        
        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.galr(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

class Separator_Res_NoDeno(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        random_mask=False,
        local_att=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR_Res(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )
        else:
            self.galr = GALR_Res(
                num_features, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )

        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(num_features, n_sources*num_features, kernel_size=1, stride=1)
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)
        
        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        elif mask_nonlinear == 'tanh':
            self.mask_nonlinear = nn.Tanh()
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
            
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, n_frames = input.size()
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.galr(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)

        output = x.view(batch_size, n_sources, num_features, n_frames)

        return output, None

class Separator_Denoise(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        random_mask=False,
        local_att=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )
        else:
            self.galr = GALR(
                num_features, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )
        # self.denoiser = TransformerDecoder(
        #     num_features, hidden_channels,
        #     num_blocks=2, num_heads=num_heads // 2, # Temp params
        #     norm=norm, nonlinear=mask_nonlinear, dropout=dropout,
        #     causal=causal, eps=eps
        # )
        

        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(num_features, n_sources*num_features, kernel_size=1, stride=1)
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)
        
        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        elif mask_nonlinear == 'tanh':
            self.mask_nonlinear = nn.Tanh()
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
            
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, n_frames = input.size()
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.galr(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)

        output = x.view(batch_size, n_sources, num_features, n_frames)

        return output

class Separator_Res(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        random_mask=False,
        local_att=False
    ):
        super().__init__()

        self.iteration = 3
        print(self.iteration)
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )
            self.denoiser = GALRDecoder(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=2, num_heads=num_heads//2,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
            )
        else:
            self.galr = GALR(
                num_features, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                random_mask=random_mask,local_att=local_att
            )
            self.denoiser = GALRDecoder(
                num_features, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=2, num_heads=num_heads//2,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
            )
        # self.denoiser = TransformerDecoder(
        #     num_features, hidden_channels,
        #     num_blocks=2, num_heads=num_heads // 2, # Temp params
        #     norm=norm, nonlinear=mask_nonlinear, dropout=dropout,
        #     causal=causal, eps=eps
        # )
        

        self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
        self.prelu = nn.PReLU()
        self.map = nn.Conv1d(num_features, n_sources*num_features, kernel_size=1, stride=1)
        self.gtu = GTU1d(num_features, num_features, kernel_size=1, stride=1)
        
        if mask_nonlinear == 'relu':
            self.mask_nonlinear = nn.ReLU()
        elif mask_nonlinear == 'sigmoid':
            self.mask_nonlinear = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear = nn.Softmax(dim=1)
        elif mask_nonlinear == 'tanh':
            self.mask_nonlinear = nn.Tanh()
        else:
            raise ValueError("Cannot support {}".format(mask_nonlinear))
            
    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, n_sources, num_features, n_frames)
        """
        num_features, n_sources = self.num_features, self.n_sources
        chunk_size, hop_size = self.chunk_size, self.hop_size
        batch_size, num_features, n_frames = input.size()
        
        padding = (hop_size-(n_frames-chunk_size)%hop_size)%hop_size
        padding_left = padding//2
        padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size)
        x = self.norm2d(x)
        x = self.galr(x)
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)

        output = x.view(batch_size, n_sources, num_features, n_frames)

        # TODO: 取得 MASK，Encoder 出來的東西相除

        denoise_x = F.pad(x, (padding_left, padding_right))

        x = F.pad(x, (padding_left, padding_right))
        x = self.segment1d(x)
        
        intermid_lst = []
        # Denoiser
        
        # moving out, and remove norm2d, 11/06 1415
        denoise_x = self.segment1d(denoise_x) # -> (batch_size, C, S, chunk_size)
        
        for _ in range(self.iteration):
            denoise_x = self.norm2d(denoise_x)
            denoise_x = self.denoiser(denoise_x, x) # -> 傳入 encoder x?
            denoise_output = self.overlap_add1d(denoise_x)
            denoise_output = F.pad(denoise_output, (-padding_left, -padding_right))
            
            denoise_output = denoise_output.view(batch_size, n_sources, num_features, n_frames)
            intermid_lst.append(denoise_output)
        # intermid_lst.append(output)
        return output, intermid_lst


def _test_separator():
    batch_size, n_frames = 2, 5
    M, H = 16, 32 # H is the number of channels for each direction
    K, P, Q = 3, 2, 2
    N = 3
    
    sep_norm = True
    mask_nonlinear = 'sigmoid'
    low_dimension = True
    
    causal = True
    n_sources = 2
    
    input = torch.randn((batch_size, M, n_frames), dtype=torch.float)
    
    separator = Separator(
        M, hidden_channels=H,
        chunk_size=K, hop_size=P, down_chunk_size=Q,
        num_blocks=N,
        norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        causal=causal,
        n_sources=n_sources
    )
    print(separator)

    output = separator(input)
    print(input.size(), output.size())

def _test_galrnet():
    batch_size = 2
    C, T = 1, 128
    K, P, Q = 3, 2, 2

    # Encoder & decoder
    M, D = 8, 16
    
    # Separator
    H = 32 # for each direction
    N, J = 4, 4
    sep_norm = True

    low_dimension=True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    print("-"*10, "Trainable Basis & Non causal", "-"*10)
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = 'relu'
    
    causal = False
    mask_nonlinear = 'sigmoid'
    n_sources = 2
    
    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        causal=causal,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    print()
    
    print("-"*10, "Fourier Basis & Causal", "-"*10)
    enc_basis, dec_basis = 'Fourier', 'Fourier'
    window_fn = 'hamming'
    
    causal = True
    mask_nonlinear = 'softmax'
    n_sources = 3
    
    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, window_fn=window_fn,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        causal=causal,
        n_sources=n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
    
def _test_galrnet_paper():
    batch_size = 2
    K, P, Q = 100, 50, 32
    
    # Encoder & decoder
    C, T = 1, 1024
    M, D = 16, 64
    
    # Separator
    H = 128 # for each direction
    N = 6
    J = 8
    sep_norm = True

    low_dimension=True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = None
    
    causal = False
    mask_nonlinear = 'relu'
    n_sources = 2
    
    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        n_sources=n_sources,
        causal=causal
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())
   
def _test_big_galrnet_paper():
    batch_size = 2
    K, P, Q = 100, 50, 32
    
    # Encoder & decoder
    C, T = 1, 1024
    M, D = 16, 128
    
    # Separator
    H = 128 # for each direction
    N = 6
    J = 8
    sep_norm = True

    low_dimension=True
    
    input = torch.randn((batch_size, C, T), dtype=torch.float)
    
    enc_basis, dec_basis = 'trainable', 'trainable'
    enc_nonlinear = None
    
    causal = False
    mask_nonlinear = 'relu'
    n_sources = 2
    
    model = GALRNet(
        D, kernel_size=M, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear,
        sep_hidden_channels=H,
        sep_chunk_size=K, sep_hop_size=P, sep_down_chunk_size=Q,
        sep_num_blocks=N, sep_num_heads=J,
        sep_norm=sep_norm, mask_nonlinear=mask_nonlinear,
        low_dimension=low_dimension,
        n_sources=n_sources,
        causal=causal
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Separator", "="*10)
    _test_separator()
    print()
    
    print("="*10, "GALRNet", "="*10)
    _test_galrnet()
    print()

    print("="*10, "GALRNet (same configuration in paper)", "="*10)
    _test_galrnet_paper()
    print()

    print("="*10, "Bigger GALRNet (same configuration in paper)", "="*10)
    _test_big_galrnet_paper()
    print()
