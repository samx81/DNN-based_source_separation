from re import X
from sys import flags
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.filterbank import choose_filterbank
from utils.tasnet import choose_layer_norm
from models.gtu import GTU1d
from models.transform import Segment1d, OverlapAdd1d
from models.galr import GALR
from models.custom.galr_dcn import GALR as GALR_DCN
from dccrn import DCCRN_Encoder,DCCRN_Decoder, DCTCN_Encoder, \
    DCTCN_Decoder,Naked_Encoder, Naked_Decoder, FiLM_Encoder, \
    Deep_Encoder, Deep_Decoder
from dct import CosineDecoder, CosineEncoder
import dense_dilated

EPS = 1e-12

class GALRNet_SO(nn.Module):
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
        if 'TorchSTFT' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = Naked_Encoder(feat_type='fft',causal=causal), Naked_Decoder(feat_type='fft')
        elif 'DCT' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = Naked_Encoder(feat_type='dct',causal=causal), Naked_Decoder(feat_type='dct')
        elif 'DCT_Learn' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = CosineEncoder(n_basis, kernel_size, stride, trainable=True), CosineDecoder(n_basis, kernel_size, stride, trainable=True)
        elif 'in_dct_learn' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = CosineEncoder(n_basis, kernel_size, stride, trainable=True, center=True), Naked_Decoder(feat_type='dct')
        elif 'TENET' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = Naked_Encoder(feat_type='TENET',causal=causal), Naked_Decoder(feat_type='TENET')
        else:
            encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        self.conv = conv = kwargs.get('conv', None)
        self.handcraft = kwargs.get('handcraft', None)
        self.local_att = kwargs.get('local_att', None)
        self.intra_dropout = kwargs.get('intra_dropout', None)
        print(kwargs)

        self.encoder = encoder
        if self.handcraft is 1 or self.handcraft == True:# and enc_basis in ['TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            self.separator = Separator_HC( # Separator_HC Separator_NoSegment Separator_NoSegment_HC
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv, local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        else:
            self.separator = Separator(
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv,local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        self.decoder = decoder
        
        self.num_parameters = self._get_num_parameters()
        
        # Load custom child code
        self.load()
        
    def load(self):
        pass

    def spliceout_fn(self, spec_noisy, spec_clean,interval_num, max_interval):
        # interval_num = N, max_interval = T
        spec_shape = spec_noisy[0].shape # tau
        spec_len = spec_shape[-1]
        for i in range(spec_noisy.shape[0]):
            mask = torch.ones(spec_shape, dtype=bool)
            for j in range(interval_num):
                remove_length = torch.randint(max_interval, size=(1,))
                start = torch.randint(int(spec_len - remove_length) , size=(1,))
                mask[:,start : start + remove_length] = False
            timestep_left = torch.count_nonzero(mask[0].int())
            noisy = spec_noisy[i]
            noisy = noisy[mask].view(-1, timestep_left)

            clean = spec_clean[i]
            clean = clean[mask].view(-1, timestep_left)

            padding = (spec_len - noisy.shape[-1])
            padding_r = padding//2 if (padding//2) *2 == padding else padding//2 +1
            noisy = F.pad(noisy, (padding//2, padding_r))
            spec_noisy[i] = noisy

            clean = F.pad(clean, (padding//2, padding_r))
            spec_clean[i] = clean
        return spec_noisy, spec_clean
        # noisy, clean = spec_noisy[mask], spec_clean[mask]
        # return noisy, clean


    def forward(self, input, sources=None):
        if sources is not None:
            output, latent, s = self.extract_latent(input, sources)
        
            return output, latent, s
        else:
            output, latent= self.extract_latent(input)
            return output, latent
        
    def extract_latent(self, input, sources=None):
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
        ## TODO: 改這邊的 kernel size 讓輸出有對準
        if self.enc_basis in ['DCCRN', 'DCTCN', 'TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            padding = (100 - (T - 400) % 100) % 100
        else:
            padding = (stride - (T - kernel_size) % stride) % stride

        padding_left = padding // 2
        padding_right = padding - padding_left
        input = F.pad(input, (padding_left, padding_right))
    
        w = self.encoder(input)
        if sources is not None: 
            with torch.no_grad():
                sources = F.pad(sources, (padding_left, padding_right))
                s = self.encoder(sources)

        # print(w.shape)
        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        else:
            if sources is not None: 
                x, s = self.spliceout_fn(w, s, 24, 2)
            else:
                x = w
            mask = self.separator(x)
            # Negative mask loss
            mask[:,1] = torch.ones(mask[:,0].size()).to(mask.get_device()) - mask[:,0]

            w = w.unsqueeze(dim=1)
            w_hat = w * mask

        # latent = w
        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)

        x_hat = self.decoder(w_hat)
        if sources is not None: 
            with torch.no_grad():
                s_modify = self.decoder(s)
        
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = F.pad(x_hat, (-padding_left, -padding_right))
        if sources is None: 
            return output, latent
        s_modify = F.pad(s_modify, (-padding_left, -padding_right))
        
        return output, latent, s_modify.unsqueeze(1)
    
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
            'conv': self.conv,
            'handcraft': self.handcraft,
            'local_att': self.local_att,
            'intra_dropout': self.intra_dropout
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
        handcraft = config.get('handcraft', False)
        intra_dropout = config.get('intra_dropout', False)
        local_att = config.get('local_att', False)
        
        model = cls(
            n_basis, kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear, 
            enc_onesided=enc_onesided, enc_return_complex=enc_return_complex,
            window_fn=window_fn,sep_hidden_channels=sep_hidden_channels, 
            sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size, sep_down_chunk_size=sep_down_chunk_size, sep_num_blocks=sep_num_blocks,
            sep_num_heads=sep_num_heads, sep_norm=sep_norm, sep_dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources, handcraft=handcraft,
            low_dimension=low_dimension,
            eps=eps, local_att=local_att,intra_dropout=intra_dropout
        )
        
        return model
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters


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
            # encoder, decoder = DCCRN_Encoder(kernel_num=[4, 8, 16, 32, 32, 32], kernel_size=5), DCCRN_Decoder(kernel_num=[4, 8, 16, 32, 32, 32], kernel_size=5)
            # self.n_basis = n_basis = 128
            encoder, decoder = DCCRN_Encoder(kernel_num=[8, 16, 32, 64, 64], kernel_size=5), DCCRN_Decoder(kernel_num=[8, 16, 32, 64, 64], kernel_size=5)
            self.n_basis = n_basis = 512
            self.sep_hidden_channels = sep_hidden_channels = n_basis // 2
        elif 'DCTCN' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = DCTCN_Encoder(causal=causal), DCTCN_Decoder()
        elif 'TorchSTFT' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = Naked_Encoder(feat_type='fft',causal=causal), Naked_Decoder(feat_type='fft')
        elif 'DCT' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = Naked_Encoder(feat_type='dct',causal=causal), Naked_Decoder(feat_type='dct')
        elif 'DCT_Learn' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = CosineEncoder(n_basis, kernel_size, stride, trainable=True), CosineDecoder(n_basis, kernel_size, stride, trainable=True)
        elif 'in_dct_learn' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = CosineEncoder(n_basis, kernel_size, stride, trainable=True, center=True), Naked_Decoder(feat_type='dct')
        elif 'FiLM_DCT' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = FiLM_Encoder(), Naked_Decoder(feat_type='dct')
        elif 'Deep_DCT' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = Deep_Encoder(feat_type='dct',causal=causal), Deep_Decoder(feat_type='dct')
        elif 'TENET' in [enc_basis, dec_basis]:
            # self.n_basis = n_basis = 8192
            encoder, decoder = Naked_Encoder(feat_type='TENET',causal=causal), Naked_Decoder(feat_type='TENET')
        else:
            encoder, decoder = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        
        self.conv = conv = kwargs.get('conv', None)
        self.handcraft = kwargs.get('handcraft', None)
        self.local_att = kwargs.get('local_att', None)
        self.intra_dropout = kwargs.get('intra_dropout', None)
        print(kwargs)

        self.encoder = encoder
        if self.handcraft is 1 or self.handcraft == True:# and enc_basis in ['TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            self.separator = Separator_HC( # Separator_HC Separator_NoSegment Separator_NoSegment_HC
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv, local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        elif self.handcraft == 2 :# and enc_basis in ['TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            self.separator = Separator_NoSegment_HC( # Separator_HC Separator_NoSegment Separator_NoSegment_HC
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=False,  # low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv, local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        elif self.handcraft == 3 :# and enc_basis in ['TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            self.separator = Separator_NoSegment_Deep( # Separator_HC Separator_NoSegment Separator_NoSegment_HC
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=False,  # low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv, local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        elif self.handcraft == 4 :# and enc_basis in ['TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            self.separator = Separator_NoSegment_Deep( # Separator_HC Separator_NoSegment Separator_NoSegment_HC
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=False,  # low_dimension,
                causal=causal,
                n_sources=n_sources, bottleneck=True,
                eps=eps, conv=conv, local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        elif self.handcraft == 5 :# and enc_basis in ['TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            self.separator = Separator_HC_Inv( # Separator_HC Separator_NoSegment Separator_NoSegment_HC
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=False,  # low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv, local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        elif self.handcraft == 6 :# and enc_basis in ['TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            self.separator = Separator_HC_DCN( # Separator_HC Separator_NoSegment Separator_NoSegment_HC
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=False,  # low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv, local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        else:
            self.separator = Separator(
                n_basis, hidden_channels=sep_hidden_channels,
                chunk_size=sep_chunk_size, hop_size=sep_hop_size, down_chunk_size=sep_down_chunk_size, num_blocks=sep_num_blocks,
                num_heads=sep_num_heads, norm=sep_norm, dropout=sep_dropout, mask_nonlinear=mask_nonlinear,
                low_dimension=low_dimension,
                causal=causal,
                n_sources=n_sources,
                eps=eps, conv=conv,local_att=self.local_att, intra_dropout=self.intra_dropout
            )
        self.decoder = decoder
        
        self.num_parameters = self._get_num_parameters()
        
        # Load custom child code
        self.load()
        
    def load(self):
        pass

    def forward(self, input):
        output, latent = self.extract_latent(input)
        
        return output, latent
        
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
        ## TODO: 改這邊的 kernel size 讓輸出有對準
        if self.enc_basis in ['DCCRN', 'DCTCN', 'TorchSTFT', 'TENET', 'DCT', 'FiLM_DCT']:
            padding = (100 - (T - 400) % 100) % 100
        else:
            padding = (stride - (T - kernel_size) % stride) % stride

        padding_left = padding // 2
        padding_right = padding - padding_left
        input = F.pad(input, (padding_left, padding_right))
        if self.enc_basis == 'FiLM_DCT':
            w, dct = self.encoder(input)
        else:
            w = self.encoder(input)

        # print(w.shape)
        if torch.is_complex(w):
            amplitude, phase = torch.abs(w), torch.angle(w)
            mask = self.separator(amplitude)
            amplitude, phase = amplitude.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            w_hat = amplitude * mask * torch.exp(1j * phase)
        elif self.enc_basis == 'FiLM_DCT':
            mask = self.separator(w)
            dct = dct.unsqueeze(dim=1)
            w_hat = dct * mask
        else:
            mask = self.separator(w)
            
            # Negative mask loss
            mask[:,1] = torch.ones(mask[:,0].size()).to(mask.get_device()) - mask[:,0]

            w = w.unsqueeze(dim=1)
            w_hat = w * mask

        # latent = w
        latent = w_hat
        w_hat = w_hat.view(batch_size*n_sources, n_basis, -1)


        if 'DCTCN' == self.dec_basis:
            x_hat = self.decoder(w, mask)
        else:
            x_hat = self.decoder(w_hat)
        x_hat = x_hat.view(batch_size, n_sources, -1)
        output = x_hat
        output = F.pad(x_hat, (-padding_left, -padding_right))
        
        return output, latent
    
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
            'conv': self.conv,
            'handcraft': self.handcraft,
            'local_att': self.local_att,
            'intra_dropout': self.intra_dropout
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
        handcraft = config.get('handcraft', False)
        intra_dropout = config.get('intra_dropout', False)
        local_att = config.get('local_att', False)
        
        model = cls(
            n_basis, kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, enc_nonlinear=enc_nonlinear, 
            enc_onesided=enc_onesided, enc_return_complex=enc_return_complex,
            window_fn=window_fn,sep_hidden_channels=sep_hidden_channels, 
            sep_chunk_size=sep_chunk_size, sep_hop_size=sep_hop_size, sep_down_chunk_size=sep_down_chunk_size, sep_num_blocks=sep_num_blocks,
            sep_num_heads=sep_num_heads, sep_norm=sep_norm, sep_dropout=sep_dropout,
            mask_nonlinear=mask_nonlinear,
            causal=causal,
            n_sources=n_sources, handcraft=handcraft,
            low_dimension=low_dimension,
            eps=eps, local_att=local_att,intra_dropout=intra_dropout
        )
        
        return model
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

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
        conv=False,
        local_att=False, intra_dropout=False
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
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        else:
            self.galr = GALR(
                num_features, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
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
        x = self.segment1d(x) # -> (batch_size, C, S, chunk_size), 
                              # C == channel == freq, S == segments 
        # print(x.shape)
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

class Separator_NoSegment_Deep(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        conv=False, bottleneck=False,
        local_att=False, intra_dropout=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        # self.segment1d = Segment1d(chunk_size, hop_size)
        
        self.bottleneck = bottleneck
        if self.bottleneck:
            self.masknet_feature = hidden_channels // 2
            self.bn_conv2d = nn.Conv2d(num_features, self.masknet_feature, 1,1)
            self.bn_iconv2d = nn.Conv2d(self.masknet_feature, num_features, 1,1)
        else:
            self.masknet_feature = num_features

        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, self.masknet_feature, causal=causal, eps=eps)

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                self.masknet_feature, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        else:
            self.galr = GALR(
                self.masknet_feature, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        # self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
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

        self.conv2d = nn.Conv2d(1, 8, 1,1)
        self.in_dilated = dense_dilated.DenseBlock(512, 8)

        self.iconv2d = nn.Conv2d(8, 1, 1, 1)
        self.out_dilated = dense_dilated.DenseBlock(512, 8)

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
        
        x = x.unsqueeze(1) # (batch_size, 1, freq, time)
        x = self.conv2d(x).transpose(-1,-2) # (bs, 8 , freq, time), (bs, 8, time, freq)
        x = self.in_dilated(x).permute(0, 3, 1, 2) # (bs, freq, channel, time)

        if self.bottleneck:
            x = self.bn_conv2d(x)
        # x = self.segment1d(x) # -> (batch_size, C, S, chunk_size), 
                              # C == channel == freq, S == segments 
        # print(x.shape)
        x = self.norm2d(x)
        x = self.galr(x)

        if self.bottleneck:
            x = self.bn_iconv2d(x)

        x = x.permute(0,2,3,1)
        x = self.out_dilated(x)
        x = self.iconv2d(x).squeeze(1).transpose(-1, -2)
        # x = x.squeeze(2)
        # x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

class Separator_NoSegment(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        conv=False,
        local_att=False, intra_dropout=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        # self.segment1d = Segment1d(chunk_size, hop_size)
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
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        else:
            self.galr = GALR(
                num_features, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        # self.overlap_add1d = OverlapAdd1d(chunk_size, hop_size)
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
        x = x.unsqueeze(2)
        # x = self.segment1d(x) # -> (batch_size, C, S, chunk_size), 
                              # C == channel == freq, S == segments 
        # print(x.shape)
        x = self.norm2d(x)
        x = self.galr(x)
        x = x.squeeze(2)
        # x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames)
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

class Separator_NoSegment_HC(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        conv=False,
        local_att=False, intra_dropout=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.conv2d = nn.Conv2d(num_features, hidden_channels // 2, 1,1)
        self.iconv2d = nn.Conv2d(hidden_channels // 2, num_features, 1,1) 
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        # intra_dropout = True

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                hidden_channels // 2, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        else:
            self.galr = GALR(
                hidden_channels // 2, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
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
        elif mask_nonlinear == 'prelu':
            self.mask_nonlinear = nn.PReLU()
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
        x = x.unsqueeze(2)
        x = self.prelu(x) # New
        x = self.norm2d(x)
        x = self.conv2d(x)# New
        x = self.prelu(x) # New
        x = self.galr(x)

        x = self.prelu(x) # New
        x = self.iconv2d(x)# New 
        
        x = x.squeeze(2)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames), where C = num_features
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

from involution import Involution2d
class Separator_HC_Inv(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        conv=False,
        local_att=False, intra_dropout=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        # TODO: 可以改 kernel size, 因為切塊後並不一定有相關 (1, 7)
        self.inv2d: nn.Module = Involution2d(in_channels=num_features, out_channels=hidden_channels // 2, kernel_size=(1,7), padding=(0,3))
        # self.conv2d = nn.Conv2d(num_features, hidden_channels // 2, 1,1)
       # self.iinv2d: nn.Module = Involution2d(in_channels=hidden_channels // 2, out_channels=num_features, kernel_size=(1,7), padding=(0,3))

        self.iconv2d = nn.Conv2d(hidden_channels // 2, num_features, 1,1) 
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        # intra_dropout = True

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                hidden_channels // 2, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        else:
            self.galr = GALR(
                hidden_channels // 2, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
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
        elif mask_nonlinear == 'prelu':
            self.mask_nonlinear = nn.PReLU()
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
        x = self.prelu(x) # New
        x = self.norm2d(x)
        x = self.inv2d(x)# New
        x = self.prelu(x) # New
        x = self.galr(x)

        x = self.prelu(x) # New
        x = self.iconv2d(x)
        #x = self.iinv2d(x)# New 
        
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames), where C = num_features
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output


# Hand Craft Feature
class Separator_HC(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        conv=False,
        local_att=False, intra_dropout=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.conv2d = nn.Conv2d(num_features, hidden_channels // 2, 1,1)
        self.iconv2d = nn.Conv2d(hidden_channels // 2, num_features, 1,1) 
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        # intra_dropout = True

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR(
                hidden_channels // 2, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        else:
            self.galr = GALR(
                hidden_channels // 2, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
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
        elif mask_nonlinear == 'prelu':
            self.mask_nonlinear = nn.PReLU()
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
        x = self.prelu(x) # New
        x = self.norm2d(x)
        x = self.conv2d(x)# New
        x = self.prelu(x) # New
        x = self.galr(x)

        x = self.prelu(x) # New
        x = self.iconv2d(x)# New 
        
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames), where C = num_features
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

class Separator_HC_DCN(nn.Module):
    def __init__(
        self,
        num_features, hidden_channels=128,
        chunk_size=100, hop_size=50, down_chunk_size=None, num_blocks=6, num_heads=4,
        norm=True, dropout=0.1, mask_nonlinear='relu',
        low_dimension=True,
        causal=True,
        n_sources=2,
        eps=EPS,
        conv=False,
        local_att=False, intra_dropout=False
    ):
        super().__init__()
        
        self.num_features, self.n_sources = num_features, n_sources
        self.chunk_size, self.hop_size = chunk_size, hop_size
        
        self.segment1d = Segment1d(chunk_size, hop_size)
        self.conv2d = nn.Conv2d(num_features, hidden_channels // 2, 1,1)
        self.iconv2d = nn.Conv2d(hidden_channels // 2, num_features, 1,1) 
        norm_name = 'cLN' if causal else 'gLN'
        self.norm2d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)

        # intra_dropout = True

        if low_dimension:
            # If low-dimension representation, latent_dim and chunk_size are required
            if down_chunk_size is None:
                raise ValueError("Specify down_chunk_size")
            self.galr = GALR_DCN(
                hidden_channels // 2, hidden_channels,
                chunk_size=chunk_size, down_chunk_size=down_chunk_size,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
            )
        else:
            self.galr = GALR_DCN(
                hidden_channels // 2, hidden_channels,
                num_blocks=num_blocks, num_heads=num_heads,
                norm=norm, dropout=dropout,
                low_dimension=low_dimension,
                causal=causal,
                eps=eps,
                conv=conv,local_att=local_att, intra_dropout=intra_dropout
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
        elif mask_nonlinear == 'prelu':
            self.mask_nonlinear = nn.PReLU()
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
        x = self.prelu(x) # New
        x = self.norm2d(x)
        x = self.conv2d(x)# New
        x = self.prelu(x) # New
        x, _ = self.galr(x)

        x = self.prelu(x) # New
        x = self.iconv2d(x)# New 
        
        x = self.overlap_add1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        x = self.prelu(x) # -> (batch_size, C, n_frames), where C = num_features
        x = self.map(x) # -> (batch_size, n_sources*C, n_frames)
        x = x.view(batch_size*n_sources, num_features, n_frames) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.gtu(x) # -> (batch_size*n_sources, num_features, n_frames)
        x = self.mask_nonlinear(x) # -> (batch_size*n_sources, num_features, n_frames)
        output = x.view(batch_size, n_sources, num_features, n_frames)
        
        return output

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
