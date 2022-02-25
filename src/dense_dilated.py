import torch
import torch.nn as nn

# https://github.com/samleoqh/DDCM-Semantic-Segmentation-PyTorch/blob/master/models/modules/ddcm_block.py
# Dense Dilated Convolutions Merging
class DDCM_Block(nn.Module):
    def __init__(self, in_dim, out_dim, rates, kernel=3, bias=False, extend_dim=False):
        super(DDCM_Block, self).__init__()
        self.features = []
        self.num = len(rates)
        self.in_dim = in_dim
        self.out_dim = out_dim

        if self.num > 0:
            if extend_dim:
                self.out_dim = out_dim * self.num
            for idx, rate in enumerate(rates):
                self.features.append(nn.Sequential(
                    nn.Conv2d(self.in_dim + idx * out_dim,
                              out_dim,
                              kernel_size=kernel, dilation=rate,
                              padding=rate * (kernel - 1) // 2, bias=bias),
                    nn.PReLU(),
                    nn.BatchNorm2d(out_dim))
                )

            self.features = nn.ModuleList(self.features)

        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(self.in_dim + out_dim * self.num,
                      self.out_dim, kernel_size=1, bias=bias),
            nn.PReLU(),
            nn.BatchNorm2d(self.out_dim),
        )

    def forward(self, x):
        for f in self.features:
            x = torch.cat([f(x), x], 1)
        x = self.conv1x1_out(x)
        return x

# https://github.com/key2miao/TSTNN/blob/master/new_model.py

# Returns a tensor which keep same shape as input
class DenseBlock(nn.Module):
    def __init__(self, feature_dim, channel=64, depth=5):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.channel = channel
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            # dilation grows acroading to depth 
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.channel * (i + 1), self.channel, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(feature_dim))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.channel))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class DilatedConv(nn.Module):
    def __init__(self, feature_dim, ratio, channel=64):
        super(DilatedConv, self).__init__()
        twidth = 2
        kernel_size = (twidth, 3)
        
        dil = 2 ** ratio
        pad_length = twidth + (dil - 1) * (twidth - 1) - 1
        self.pad = nn.ConstantPad2d((1, 1, pad_length, 0), value=0.)
        self.conv = nn.Conv2d(channel * (ratio + 1), channel, kernel_size=kernel_size, dilation=(dil, 1))
        self.norm = nn.LayerNorm(feature_dim)
        self.prelu = nn.PReLU(channel)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.prelu(out)
        return out

class DenseBlock_New(nn.Module):
    def __init__(self, feature_dim, channel=64, depth=4):
        super(DenseBlock_New, self).__init__()
        net = [DilatedConv(feature_dim, i, channel=channel) for i in range(depth)]

        self.net = nn.ModuleList(net)
            
    def forward(self, x):
        skip = x
        for i in self.net:
            out = i(skip)
            skip = torch.cat([out, skip], dim=1)
        return out