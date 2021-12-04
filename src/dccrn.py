import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT 
from tenet_stft import STFT, ISTFT
from complexnn import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm
from models.tcn import TemporalConvNet
from utils.utils_tasnet import choose_layer_norm
import dct

EPS = 1e-12

class SPConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose1d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels * r, kernel_size=kernel_size, stride=1)
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, W))
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, -1))
        return out

class TSTNN_Encoder(nn.Module):
    def __init__(self, in_channels, n_basis, kernel_size=16, stride=8, **kwargs):
        super().__init__()
        
        self.kernel_size, self.stride = kernel_size, stride

        self.stft, _ = choose_filterbank(n_basis, kernel_size=kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)
        # self.pad1 = nn.ConstantPad1d((1, 1,  0), value=0.)
        # 1 > 64
        self.conv1d = nn.Conv1d(in_channels, n_basis, kernel_size=1, stride=1, bias=False)

        self.inp_norm = nn.GroupNorm(1,n_basis)
        self.inp_prelu = nn.PReLU()

        self.enc_dense1 = TemporalConvNet(
            n_basis, hidden_channels=n_basis * 2, skip_channels=n_basis, kernel_size=3, num_blocks=1, num_layers=2,
            dilated=True, separable=True, causal=False, nonlinear='prelu', norm=True
        )

        # halve
        self.enc_conv1 = nn.Conv1d(in_channels=n_basis, out_channels=n_basis, kernel_size=kernel_size, stride=stride, bias=False) 
        self.enc_norm1 = nn.GroupNorm(1,n_basis)
        self.enc_prelu1 = nn.PReLU()
    
    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.conv1d(x)))  # [b, 64, num_frames, frame_size]
        out = self.enc_dense1(out)   # [b, 64, num_frames, frame_size]
        x1 = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, num_frames, 256] # why pad?
        return x1

class TSTNN_Decoder(nn.Module):
    def __init__(self, n_basis, in_channels, kernel_size=16, stride=8, **kwargs):
        super().__init__()
        
        self.kernel_size, self.stride = kernel_size, stride
        # 1 > 64

        self.dec_dense1 = TemporalConvNet(
            n_basis, hidden_channels=n_basis * 2, skip_channels=n_basis, kernel_size=3, num_blocks=1, num_layers=2,
            dilated=True, separable=True, causal=False, nonlinear='prelu', norm=True
        )

        # halve
        # self.dec_conv1 = SPConvTranspose1d(in_channels=n_basis, out_channels=n_basis, kernel_size=kernel_size, r=stride)
        self.dec_conv1 =  nn.ConvTranspose1d(in_channels=n_basis, out_channels=n_basis, kernel_size=kernel_size, stride=stride, bias=False) 
        self.dec_norm1 = nn.GroupNorm(1, n_basis)
        self.dec_prelu1 = nn.PReLU()
        self.out_conv = nn.Conv1d(n_basis, in_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(out)))
        out = self.out_conv(out)
        return out

class DCCRN_Encoder(nn.Module):
    def __init__(self, 
                win_len=400,
                win_inc=100, 
                fft_len=512,
                win_type='hanning',
                masking_mode='E',
                use_cbn = False,
                kernel_size=5,
                kernel_num=[32,64,128,256,256, 256]):

        super(DCCRN_Encoder, self).__init__() 

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        self.kernel_size = kernel_size
        self.kernel_num = [2]+kernel_num 

        fix=True
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.encoder = nn.ModuleList()

        for idx in range(len(self.kernel_num)-1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx], self.kernel_num[idx+1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1), padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx+1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )
    def forward(self, inputs, lens=None, ref=None, valid=False):
        # when inference, only one utt
        if inputs.dim() == 1:
            inputs = torch.unsqueeze(inputs, 0)
        specs = self.stft(inputs)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real,imag],1)
        cspecs = cspecs[:,:,1:]

        out = cspecs
        encoder_out = []
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)
        
        batch_size, channels, dims, lengths = out.size()
        out = out.view(batch_size, channels * dims, lengths)
        return out

class DCCRN_Decoder(nn.Module):
    def __init__(self, 
                win_len=400,
                win_inc=100, 
                fft_len=512,
                win_type='hanning',
                masking_mode='E',
                use_cbn = False,
                kernel_size=5,
                kernel_num=[32,64,128,256,256, 256]):

        super(DCCRN_Decoder, self).__init__() 

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        self.kernel_size = kernel_size
        self.kernel_num = [2]+kernel_num 

        fix=True
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num)-1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                    ComplexConvTranspose2d(
                        self.kernel_num[idx], self.kernel_num[idx-1],
                        kernel_size =(self.kernel_size, 2),
                        stride=(2, 1), padding=(2,0), output_padding=(1,0)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx-1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx-1]),
                    nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(ComplexConvTranspose2d(
                        self.kernel_num[idx],self.kernel_num[idx-1],
                        kernel_size =(self.kernel_size, 2),
                        stride=(2, 1), padding=(2,0), output_padding=(1,0)
                    ),)
                )

    def forward(self, inputs):
        # when inference, only one utt
        out = inputs
        batch_size, cxd, lengths = out.size()
        channel = self.kernel_num[-1]
        dim = cxd // channel
        out = out.view(batch_size, channel, dim, lengths )
        for idx in range(len(self.decoder)):
            out = self.decoder[idx](out)
            out = out[...,1:]
        
        out = F.pad(out, [0,0,1,0])   
        out = out.view(batch_size, -1, lengths)
        # out = torch.cat([real, imag], 1)
        
        out_wav = self.istft(out)
        return out_wav

class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), 
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), 
                    nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), 
                    nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class DCTCN_Encoder(nn.Module):
    def __init__(self, 
                win_len=400, win_inc=100, fft_len=512, win_type='hanning',
                bottleneck_channels=2, hidden_channels=128, skip_channels=64, kernel_size=3, num_blocks=1, num_layers=4,
                dilated=True, separable=True, causal=True, nonlinear='prelu', norm=True, eps=EPS):

        super(DCTCN_Encoder, self).__init__() 

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        fix=True
        # self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'real', fix=fix)

        norm_name = 'cLN' if causal else 'gLN'
        
        self.channels = 4
        self.bottleneck_conv2d = nn.Conv2d(2, self.channels, kernel_size=(1, 1))
        self.bottleneck_norm2d = nn.LayerNorm(256)
        self.prelu = nn.PReLU(self.channels)

        self.encoder = DenseBlock(256, depth=4, in_channels=self.channels)

        # self.hd2d = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        # self.norm2d = choose_layer_norm(norm_name, 64, causal=causal, eps=eps)

    def forward(self, inputs):
        # when inference, only one utt
        if inputs.dim() == 1:
            inputs = torch.unsqueeze(inputs, 0)
        specs = self.stft(inputs)
        # real = specs[:,:self.fft_len//2+1]
        # imag = specs[:,self.fft_len//2+1:]

        real = specs[0]
        imag = specs[1]
        
        # spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        # spec_mags = spec_mags
        # spec_phase = torch.atan2(imag, real)
        # spec_phase = spec_phase
        cspecs = torch.stack([real,imag],1)
        cspecs = cspecs[:,:,1:]
        cspecs = cspecs.permute(0,1,3,2) # [B, 2, num_frames, num_bins]


        # batch_size, lengths = cspecs.size()[0], cspecs.size()[-1]
        # cspecs = cspecs.reshape(batch_size, self.fft_len, lengths)
        out = cspecs

        out = self.bottleneck_conv2d(out)
        out = self.bottleneck_norm2d(out)
        out = self.prelu(out)

        out = self.encoder(out)
        # out = self.hd2d(out)
        # out = self.norm2d(out)
        # out = self.prelu(out)
        # batch_size, channels, dims, lengths = out.size()

        # out = out.permute(0, 1, 2, 3)
        out = out.permute(0, 1, 3, 2)
        batch_size, channels, dims, lengths = out.size()
        out = out.contiguous().view(batch_size, channels * dims, lengths)
        return out

class DCTCN_Decoder(nn.Module):
    def __init__(self, 
                win_len=400,
                win_inc=100, 
                fft_len=512,
                win_type='hanning',
                masking_mode='E',
                use_cbn = False,
                kernel_size=5,):

        super(DCTCN_Decoder, self).__init__() 

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        self.kernel_size = kernel_size

        fix=True
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.channels = 4
        self.decoder = DenseBlock(256, depth= 4, in_channels=self.channels)
        self.out_conv = nn.Conv2d(in_channels=self.channels, out_channels=2, kernel_size=(1, 1))

    def forward(self, inputs, mask, masking_mode='C'):
        # when inference, only one utt
        batch_size, S, channelsxdims, lengths = mask.size()
        inputs = inputs.view(-1, self.channels, channelsxdims//self.channels, lengths).permute(0,1,3,2)
        mask = mask.view(-1, self.channels, channelsxdims//self.channels, lengths).permute(0,1,3,2)

        mask = self.decoder(mask)
        mask = self.out_conv(mask)

        real = inputs[:,0].view(batch_size, -1, lengths, channelsxdims//self.channels)
        imag = inputs[:,1].view(batch_size, -1, lengths, channelsxdims//self.channels)

        mask_real = mask[:,0].view(batch_size, -1, lengths, channelsxdims//self.channels)
        mask_imag = mask[:,1].view(batch_size, -1, lengths, channelsxdims//self.channels)

        if masking_mode == 'E' :
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2( imag_phase, real_phase )
            #mask_mags = torch.clamp_(mask_mags,0,100) 
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase)
        elif masking_mode == 'C':
            real,imag = real*mask_real-imag*mask_imag, real*mask_imag+imag*mask_real
        elif masking_mode == 'R':
            real, imag = real*mask_real, imag*mask_imag

        real = torch.cat(
            (torch.zeros(
                (real.size()[0], real.size()[1],real.size()[2], 1)
                ).to(device='cuda'), real),
             dim=-1)
        # batch_size, S, length, dims
        imag = torch.cat((torch.zeros((imag.size()[0], imag.size()[1],imag.size()[2], 1)).to(device='cuda'), imag), -1)
        out = torch.cat([real,imag],-1) #  batch_size, S, length, (dim+1) *2
        out = out.view(batch_size*S, lengths, -1).permute(0,2,1)
        
        # batchsize, dims, length
        out_wav = self.istft(out)
        out_wav = out_wav.view(batch_size, S, -1)
        return out_wav

class Naked_Encoder(nn.Module):
    def __init__(self, win_len=400, win_inc=100, fft_len=512, 
                        win_type='hanning',feat_type='fft', causal=True, eps=EPS):

        super(Naked_Encoder, self).__init__() 

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.feat_type = feat_type

        if feat_type == 'fft':
            
            self.transform = torch.stft
            norm_name = 'cLN' if causal else 'gLN'

            self.bottleneck_conv1d = nn.Conv1d(fft_len + 2, fft_len, kernel_size=1)
            self.norm2d = choose_layer_norm(norm_name, fft_len, causal=causal, eps=eps)
            self.prelu = nn.PReLU(fft_len)
        elif feat_type == 'dct':

            self.transform = dct.sdct_torch
            self.prelu = nn.PReLU(fft_len)
        elif feat_type == 'TENET':

            self.transform = STFT(fft_len, win_len, win_inc, win_type=win_type)
        
        # TODO: let window tensor can be auto convert to fft length
        if win_type == 'hanning':
            if feat_type == 'dct':
                self.window = torch.hann_window
            else:
                self.window = torch.hann_window(win_len).cuda()

    def forward(self, inputs):
        # when inference, only one utt
        if inputs.dim() == 1:
            inputs = torch.unsqueeze(inputs, 0)
        elif inputs.dim() == 3:
            inputs = inputs.view(-1, inputs.shape[-1])

        if self.feat_type == 'TENET':
            out = self.transform(inputs)
        elif self.feat_type == 'dct':
            out = self.transform(inputs, self.fft_len, self.win_len, self.win_inc, window=self.window)
            out = self.prelu(out)
        else:
            # onesided
            specs = self.transform(inputs, self.fft_len, self.win_inc, self.win_len, self.window)
        
            # => (batch_size, fft // 2 + 1, timestep, 2)

            out = specs.transpose(-1, -2) # => (batch_size, (fft // 2 + 1), 2, timestep)
            bs, _,_, T = out.size()
            out = specs.contiguous().view(bs, -1, T) # => (batch_size, fft + 2 , timestep)
            out = self.bottleneck_conv1d(out) # => (batch_size, fft, timestep)
            out = self.norm2d(out) # => (batch_size, fft, timestep)
            out = self.prelu(out)
        # batch_size, channels, dims, lengths = out.size()

        return out

class Naked_Decoder(nn.Module):
    def __init__(self, 
                win_len=400,
                win_inc=100, 
                fft_len=512,
                win_type='hanning',
                feat_type='fft'):

        super(Naked_Decoder, self).__init__() 

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.feat_type = feat_type 
        # self.transform = ISTFT(fft_len, win_len, win_inc, win_type=win_type)
        if feat_type == 'fft':
            self.inv_transform = torch.istft
            self.prelu = nn.PReLU(fft_len + 2)
            self.bottleneck_conv1d = nn.Conv1d(fft_len, fft_len + 2, kernel_size=1)
            # self.norm2d = choose_layer_norm(norm_name, fft_len, causal=causal, eps=eps)
        elif feat_type == 'dct':
            self.inv_transform = dct.isdct_torch
        elif feat_type == 'TENET':
            # pass
            self.inv_transform = ISTFT(fft_len, win_len, win_inc, win_type=win_type)

        # TODO: let window tensor can be auto convert to fft length
        if win_type == 'hanning':
            if feat_type == 'dct':
                self.window = torch.hann_window
            else:
                self.window = torch.hann_window(win_len).cuda()

    def forward(self, inputs):
        # when inference, only one utt
        # if inputs.dim() == 4:
        if self.feat_type == 'TENET':
            out_wav = self.inv_transform(inputs)
        elif self.feat_type == 'dct':
            out_wav = self.inv_transform(inputs, window_length=self.win_len, frame_step=self.win_inc, window=self.window)
        else:
            out = self.bottleneck_conv1d(inputs)
            out = self.prelu(out)
            bsxS, feat, time = out.shape
            out = out.view(bsxS, feat//2, 2, time)
            out = out.transpose(-1,-2)

            # batchsize, dims, length
            
            out_wav = self.inv_transform(out, self.fft_len, self.win_inc, self.win_len, self.window)
        # out_wav = out_wav.view(batch_size, S, -1)
        return out_wav