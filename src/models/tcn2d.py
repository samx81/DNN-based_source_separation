import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_layer_norm

EPS = 1e-12

"""
    Temporal Convolutional Network
    See "Temporal Convolutional Networks for Action Segmentation and Detection"
    https://arxiv.org/abs/1611.05267
"""

class TemporalConvNet2d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=(2,3), num_blocks=3, num_layers=5, dilated=True, separable=False, causal=True, nonlinear=None, norm=True, eps=EPS):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        net = []
        
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                net.append(ConvBlock2d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, dual_head=False, eps=eps))
            else:
                net.append(ConvBlock2d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, dual_head=True, eps=eps))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        num_blocks = self.num_blocks
        
        x = input
        skip_connection = 0
        
        for idx in range(num_blocks):
            print(idx)
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip
            print(skip_connection.shape)

        output = skip_connection
        
        return output

class ConvBlock2d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=(2,3), num_layers=5, dilated=True, separable=False, causal=True, nonlinear=None, norm=True, dual_head=False, eps=EPS):
        super().__init__()
        
        self.num_layers = num_layers
        print(num_layers)
        
        net = []
        
        for idx in range(num_layers):
            if dilated:
                dilation = (2**idx, 1)
                stride = 1
            else:
                dilation = (1 , 1)
                # stride = (1,2 )
                stride = 1
            if not dual_head and idx == num_layers - 1:
                net.append(ResidualBlock2d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, dual_head=False, eps=eps))
            else:
                net.append(ResidualBlock2d(num_features, hidden_channels=hidden_channels * (idx+1) , skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, dual_head=True, eps=eps))
            
        self.net = nn.Sequential(*net)

    def forward(self, input):
        num_layers = self.num_layers
        
        x = input
        skip_connection = 0
        
        for idx in range(num_layers):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        return x, skip_connection

# nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,dilation=(dil, 1)))

class ResidualBlock2d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=(2,3), stride=(1,1), dilation=(1,1), separable=False, causal=True, nonlinear=None, norm=True, dual_head=True, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm = norm
        self.dual_head = dual_head
        
        self.bottleneck_conv2d = nn.Conv2d(num_features, hidden_channels, kernel_size=1, stride=1)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear2d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            # self.norm2d = choose_layer_norm(norm_name, hidden_channels, causal=causal, eps=eps)
            self.norm2d = nn.GroupNorm(1, hidden_channels)
        if separable:
            self.separable_conv2d = DepthwiseSeparableConv2d(hidden_channels, num_features, skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, dual_head=dual_head, eps=eps)
        else:
            if dual_head:
                self.output_conv2d = nn.Conv2d(hidden_channels, num_features, kernel_size=kernel_size, dilation=dilation)
            self.skip_conv2d = nn.Conv2d(hidden_channels, skip_channels, kernel_size=kernel_size, dilation=dilation)
            
    def forward(self, input):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        separable, causal = self.separable, self.causal
        dual_head = self.dual_head
        
        _, _, T_original, feat = input.size()
        
        residual = input
        input = self.bottleneck_conv2d(input)
        print(input)
        if nonlinear:
            x = self.nonlinear2d(input)
        if norm:
            x = self.norm2d(input)
        
        padding = (T_original - 1) * stride - T_original + (kernel_size[0] - 1) * dilation[0] + 1
        
        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding//2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))
        
        if separable:
            output, skip = self.separable_conv2d(x) # output may be None
        else:
            if dual_head:
                output = self.output_conv2d(x)
            else:
                output = None
            skip = self.skip_conv2d(x)
        
        if output is not None:
            print(output.shape)
            output = output + residual
            
        return output, skip

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels=256, skip_channels=256, kernel_size=(1,3), stride=(1,2), dilation=1, causal=True, nonlinear=None, norm=True, dual_head=True, eps=EPS):
        super().__init__()
        
        self.dual_head = dual_head
        self.norm = norm
        self.eps = eps
        
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear2d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm2d = choose_layer_norm(norm_name, in_channels, causal=causal, eps=eps)

        if dual_head:
            self.output_pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
        self.skip_pointwise_conv2d = nn.Conv2d(in_channels, skip_channels, kernel_size=1, stride=1)
        
    def forward(self, input):
        nonlinear, norm = self.nonlinear, self.norm
        dual_head = self.dual_head
        
        x = self.depthwise_conv2d(input)
        
        if nonlinear:
            x = self.nonlinear2d(x)
        if norm:
            x = self.norm2d(x)
        if dual_head:
            output = self.output_pointwise_conv2d(x)
        else:
            output = None
        skip = self.skip_pointwise_conv2d(x)
        
        return output, skip

def _test_tcn():
    batch_size = 4
    T = 128
    in_channels, out_channels, skip_channels = 16, 16, 32
    kernel_size, stride = 3, 1
    num_blocks = 3
    num_layers = 4
    dilated, separable = True, False
    causal = True
    nonlinear = 'prelu'
    norm = True
    dual_head = False
    
    input = torch.randn((batch_size, in_channels, T), dtype=torch.float)
    
    model = TemporalConvNet(in_channels, hidden_channels=out_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm)
    
    print(model)
    output = model(input)
    
    print(input.size(), output.size())

if __name__ == '__main__':
    _test_tcn()
