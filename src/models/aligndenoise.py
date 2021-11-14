import torch
import torch.nn as nn

from utils.utils_tasnet import choose_layer_norm
from models.galr import LocallyRecurrentBlock, GloballyAttentiveBlockBase, LayerNormAlongChannel

EPS=1e-12

class GALRDecoder(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, num_heads=8, norm=True, dropout=0.1, low_dimension=True, causal=False, eps=EPS, **kwargs):
        super().__init__()
        
        # Network confguration
        net = []
        
        for _ in range(num_blocks):
            net.append(GALRBlock(num_features, hidden_channels, name=0, num_heads=num_heads, norm=norm, dropout=dropout, low_dimension=low_dimension, causal=causal, eps=eps, **kwargs))
        
        # self.net = nn.Sequential(*net)
        self.net = nn.ModuleList(net)

    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = input
        for n in self.net:
            x = n(x, src)

        return x

class GALRBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, name=None, random_mask=False, num_heads=8, causal=False, norm=True, dropout=0.1, low_dimension=True, eps=EPS, **kwargs):
        super().__init__()
        
        self.intra_chunk_block = LocallyRecurrentBlock(num_features, hidden_channels=hidden_channels, norm=norm, eps=eps)

        self.intra_chunk_att = LocallyAttentiveBlock(num_features, num_heads=num_heads, causal=causal, norm=norm, dropout=dropout, eps=eps)

        if low_dimension:
            chunk_size = kwargs['chunk_size']
            down_chunk_size = kwargs['down_chunk_size']
            self.inter_chunk_block = LowDimensionGloballyAttentiveBlock(num_features, name=name, masking=random_mask, chunk_size=chunk_size, down_chunk_size=down_chunk_size, num_heads=num_heads, causal=causal, norm=norm, dropout=dropout, eps=eps)
        else:
            self.inter_chunk_block = GloballyAttentiveBlock(num_features,name=name, masking=random_mask, num_heads=num_heads, causal=causal, norm=norm, dropout=dropout, eps=eps)
        
    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """

        x = self.intra_chunk_block(input)
        
        x = self.intra_chunk_att(x, src)

        output = self.inter_chunk_block(x, src)

        return output

class GloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, name=None, masking=False, num_heads=8, causal=False, norm=True, dropout=0.1, eps=EPS):
        super().__init__()

        self.norm = norm
        self.masking = masking
        self.name = name 

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)
        self.src_multihead_attn = nn.MultiheadAttention(num_features, num_heads)


        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        
        if self.norm:
            norm_name = 'cLN' if causal else 'gLM'
            self.norm2d_out = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        
    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        batch_size, num_features, S, K = input.size()

        if self.masking:
            rand_num = torch.randint(S, (torch.randint(int(S*0.3),(1,)),)).cuda()
            x = input.clone()
            x = x.index_fill_(2, rand_num, 0)
        else:
            x = input

        if self.norm:
            x = self.norm2d_in(x) # -> (batch_size, num_features, S, K)
        else:
            x = x

        encoding = self.positional_encoding(length=S*K, dimension=num_features).permute(1,0).view(num_features, S, K).to(x.device)
        x = x + encoding # -> (batch_size, num_features, S, K)

        x = x.permute(2,0,3,1).contiguous() # -> (S, batch_size, K, num_features)
        x = x.view(S, batch_size*K, num_features) # -> (S, batch_size*K, num_features)

        block_src = src.permute(2, 0, 3, 1).contiguous() # -> (K, batch_size, S, num_features)
        block_src = block_src.view(S, batch_size*K, num_features) # -> (K, batch_size*S, num_features)

        residual = x # (S, batch_size*K, num_features)

        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x, _ = self.src_multihead_attn(x, block_src, block_src) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual # -> (S, batch_size*K, num_features)
        x = x.view(S, batch_size, K, num_features)
        x = x.permute(1,3,0,2).contiguous() # -> (batch_size, num_features, S, K)

        if self.norm:
            x = self.norm2d_out(x) # -> (batch_size, num_features, S, K)
        x = x + input

        # torch.save([input, x_masked, x_after_normed, x_with_encoding, x_after_att, x_addback], 'x_global.pt')
        output = x.view(batch_size, num_features, S, K)

        return output

class LowDimensionGloballyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, name=None,  masking=False, chunk_size=100, down_chunk_size=32, num_heads=8, causal=False, norm=True, dropout=0.1, eps=EPS):
        super().__init__()

        self.down_chunk_size = down_chunk_size
        self.norm = norm
        self.name = name 

        self.masking = masking

        self.fc_map = nn.Linear(chunk_size, down_chunk_size)

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)
        self.src_multihead_attn = nn.MultiheadAttention(num_features, num_heads)

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        
        if self.norm:
            norm_name = 'cLN' if causal else 'gLN'
            self.norm2d_out = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        
        self.fc_inv = nn.Linear(down_chunk_size, chunk_size)

    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        Q = self.down_chunk_size
        batch_size, num_features, S, K = input.size()

        if self.masking:
            rand_num = torch.randint(S, (torch.randint(int(S*0.3),(1,)),)).cuda()
            x = input.clone()
            x = x.index_fill_(2, rand_num, 0)
        else:
            x = input

        x = self.fc_map(x) # (batch_size, num_features, S, K) -> (batch_size, num_features, S, Q)
        src = self.fc_map(src) # (batch_size, num_features, S, K) -> (batch_size, num_features, S, Q)


        if self.norm:
            x = self.norm2d_in(x) # -> (batch_size, num_features, S, Q)
            src = self.norm2d_in(src)
        
        encoding = self.positional_encoding(length=S*Q, dimension=num_features).permute(1,0).view(num_features, S, Q).to(x.device)
        x = x + encoding # -> (batch_size, num_features, S, Q)

        x = x.permute(2,0,3,1).contiguous() # -> (S, batch_size, Q, num_features)
        x = x.view(S, batch_size*Q, num_features) # -> (S, batch_size*Q, num_features)

        block_src = src.permute(2, 0, 3, 1).contiguous() # -> (K, batch_size, S, num_features)
        block_src = block_src.view(S, batch_size*Q, num_features) # -> (K, batch_size*S, num_features)

        residual = x # (S, batch_size*Q, num_features)

        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x, _ = self.src_multihead_attn(x, block_src, block_src) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual # -> (S, batch_size*Q, num_features)
        x = x.view(S, batch_size, Q, num_features)
        x = x.permute(1,3,0,2).contiguous() # -> (batch_size, num_features, S, Q)

        if self.norm:
            x = self.norm2d_out(x) # -> (batch_size, num_features, S, Q)
        
        x = self.fc_inv(x) # (batch_size, num_features, S, Q) -> (batch_size, num_features, S, K)        
        x = x + input
        
        output = x.view(batch_size, num_features, S, K)

        return output

class LocallyAttentiveBlock(GloballyAttentiveBlockBase):
    def __init__(self, num_features, masking=False, num_heads=8, causal=False, norm=True, dropout=0.1, eps=EPS):
        super().__init__()

        self.norm = norm
        self.masking = masking

        if self.norm:
            self.norm2d_in = LayerNormAlongChannel(num_features, eps=eps)

        self.multihead_attn = nn.MultiheadAttention(num_features, num_heads)
        self.src_multihead_attn = nn.MultiheadAttention(num_features, num_heads)

        if dropout is not None:
            self.dropout = True
            self.dropout1d = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        
        if self.norm:
            norm_name = 'cLN' if causal else 'gLM'
            self.norm2d_out = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
        
    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, K): K is chunk size
        Returns:
            output (batch_size, num_features, S, K)
        """
        batch_size, num_features, S, K = input.size()
        self.masking = False
        if self.masking:
            rand_num = torch.randint(input.shape[0], (int(input.shape[0]*0.3),)).cuda()
            x = input.clone()
            x = x.index_fill_(0, rand_num, 0)
        else:
            x = input

        if self.norm:
            x = self.norm2d_in(x) # -> (batch_size, num_features, S, K)
        else:
            x = x
        encoding = self.positional_encoding(length=S*K, dimension=num_features).permute(1,0).view(num_features, S, K).to(x.device)
        x = x + encoding # -> (batch_size, num_features, S, K)
        # x = x.permute(2,0,3,1).contiguous() # -> (S, batch_size, K, num_features)
        x = x.permute(3,0,2,1).contiguous() # -> (K, batch_size, S, num_features)

        x = x.view(K, batch_size*S, num_features) # -> (K, batch_size*S, num_features)

        block_src = src.permute(3,0,2,1).contiguous() # -> (K, batch_size, S, num_features)
        block_src = block_src.view(K, batch_size*S, num_features) # -> (K, batch_size*S, num_features)

        residual = x # (K, batch_size*S, num_features)

        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x, _ = self.src_multihead_attn(x, block_src, block_src) # (T_tgt, batch_size, num_features), (batch_size, T_tgt, T_src), where T_tgt = T_src = T

        if self.dropout:
            x = self.dropout1d(x)
        x = x + residual # -> (K, batch_size*S, num_features)
        x = x.view(K, batch_size, S, num_features)
        x = x.permute(1,3,2,0).contiguous() # -> (batch_size, num_features, S, K)

        if self.norm:
            x = self.norm2d_out(x) # -> (batch_size, num_features, S, K)
        x = x + input
        # torch.save([input, x_masked, x_after_normed, x_with_encoding, x_after_att, x_addback], 'x_global.pt')
        output = x.view(batch_size, num_features, S, K)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()
        
        # Network confguration
        net = []
        
        for _ in range(num_blocks):
            net.append(DPTransformerDecoderBlock(num_features, hidden_channels, num_heads=num_heads, norm=norm, nonlinear=nonlinear, dropout=dropout, causal=causal, eps=eps))
        
        # self.net = nn.Sequential(*net)
        self.net = nn.ModuleList(net)

    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = input
        for n in self.net:
            x, src = n(x, src)

        return x

class DPTransformerDecoderBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()
        
        self.intra_decoder_block = Decoder(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=False,
            eps=eps
        )
        self.inter_decoder_block = InterDecoder(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=False,
            eps=eps
        )

    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        output, src = self.intra_decoder_block(input, src)
        output, src = self.inter_decoder_block(input, src)
        
        return output, src


class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()
        
        self.decoder_block = Decoder(
            num_features, hidden_channels, num_heads=num_heads,
            norm=norm, nonlinear=nonlinear, dropout=dropout,
            causal=False,
            eps=eps
        )

    def forward(self, input, src):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        output, src = self.decoder_block(input, src)
        
        return output, src

class InterDecoder(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()
        
        self.self_multihead_attn = SelfMultiheadAttentionBlock(num_features, num_heads, norm=norm, dropout=dropout, causal=causal, eps=eps)
        self.src_multihead_attn = SrcMultiheadAttentionBlock(num_features, num_heads, norm=norm, dropout=dropout, causal=causal, eps=eps)
        self.subnet = FeedForwardBlock(num_features, hidden_channels, norm=norm, nonlinear=nonlinear, causal=causal, eps=eps)

    def positional_encoding(self, length: int, dimension: int, base=10000):
        """
        Args:
            length <int>: 
            dimension <int>: 
        Returns:
            output (length, dimension): positional encording
        """
        assert dimension%2 == 0, "dimension is expected even number but given odd number."

        position = torch.arange(length) # (length,)
        position = position.unsqueeze(dim=1) # (length, 1)
        index = torch.arange(dimension//2) / dimension # (dimension//2,)
        index = index.unsqueeze(dim=0) # (1, dimension//2)
        indices = position / base**index
        output = torch.cat([torch.sin(indices), torch.cos(indices)], dim=1)
        
        return output
    
    def forward(self, input, src):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        batch_size, num_features, S, K = input.size() # k = chunk_size (timestep)
        x = input
        encoding = self.positional_encoding(length=S*K, dimension=num_features).permute(1,0).view(num_features, S, K).to(x.device)
        # print(input.get_device(), encoding.get_device())
        x = x + encoding # -> (batch_size, num_features, S, K)
        x = x.permute(2, 0, 3, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (S, batch_size, chunk_size, num_features)
        x = x.view(S, batch_size*K, num_features) # (S, batch_size, chunk_size, num_features) -> (S, batch_size*chunk_size, num_features)

        block_src = src.permute(2, 0, 3, 1).contiguous() # -> (K, batch_size, S, num_features)
        block_src = block_src.view(S, batch_size*K, num_features) # -> (K, batch_size*S, num_features)

        x = self.self_multihead_attn(x)
        x = self.src_multihead_attn(x, block_src)
        output = self.subnet(x)
        
        output = output.view(S, batch_size, K, num_features)
        output = output.permute(1, 3, 0, 2).contiguous() # -> (batch_size, num_features, S, K)
        
        return output, src


class Decoder(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, norm=True, nonlinear='relu', dropout=0, causal=False, eps=EPS):
        super().__init__()
        
        self.self_multihead_attn = SelfMultiheadAttentionBlock(num_features, num_heads, norm=norm, dropout=dropout, causal=causal, eps=eps)
        self.src_multihead_attn = SrcMultiheadAttentionBlock(num_features, num_heads, norm=norm, dropout=dropout, causal=causal, eps=eps)
        self.subnet = FeedForwardBlock(num_features, hidden_channels, norm=norm, nonlinear=nonlinear, causal=causal, eps=eps)

    def positional_encoding(self, length: int, dimension: int, base=10000):
        """
        Args:
            length <int>: 
            dimension <int>: 
        Returns:
            output (length, dimension): positional encording
        """
        assert dimension%2 == 0, "dimension is expected even number but given odd number."

        position = torch.arange(length) # (length,)
        position = position.unsqueeze(dim=1) # (length, 1)
        index = torch.arange(dimension//2) / dimension # (dimension//2,)
        index = index.unsqueeze(dim=0) # (1, dimension//2)
        indices = position / base**index
        output = torch.cat([torch.sin(indices), torch.cos(indices)], dim=1)
        
        return output
    
    def forward(self, input, src):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        batch_size, num_features, S, K = input.size() # k = chunk_size (timestep)
        x = input
        encoding = self.positional_encoding(length=S*K, dimension=num_features).permute(1,0).view(num_features, S, K).to(x.device)
        # print(input.get_device(), encoding.get_device())
        x = x + encoding # -> (batch_size, num_features, S, K)
        x = x.permute(3,0,2,1).contiguous() # -> (K, batch_size, S, num_features)
        x = x.view(K, batch_size*S, num_features) # -> (K, batch_size*S, num_features)

        block_src = src.permute(3,0,2,1).contiguous() # -> (K, batch_size, S, num_features)
        block_src = block_src.view(K, batch_size*S, num_features) # -> (K, batch_size*S, num_features)

        x = self.self_multihead_attn(x)
        x = self.src_multihead_attn(x, block_src)
        output = self.subnet(x)
        
        output = output.view(K, batch_size, S, num_features)
        output = output.permute(1,3,2,0).contiguous() # -> (batch_size, num_features, S, K)
        
        return output, src

class SrcMultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, norm=True, dropout=0, causal=False, eps=EPS):
        super().__init__()

        if dropout == 0:
            self.dropout = False
        else:
            self.dropout = True
        
        self.norm = norm

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        if self.dropout:
            self.dropout1d = nn.Dropout(p=dropout)
        if self.norm:
            norm_name = 'cLN' if causal else 'gLM'
            self.norm1d = choose_layer_norm(norm_name, embed_dim, causal=causal, eps=eps)
    
    def forward(self, input, src):
        """
        Args:
            input (T, batch_size, embed_dim)
        Returns:
            output (T, batch_size, embed_dim)
        """
        x = input # (T, batch_size, embed_dim)

        residual = x
        x, _ = self.multihead_attn(x, src, src) # (T_tgt, batch_size, embed_dim), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x = x + residual

        if self.dropout:
            x = self.dropout1d(x)
        
        if self.norm:
            x = x.permute(1, 2, 0) # (batch_size, embed_dim, T)
            x = self.norm1d(x) # (batch_size, embed_dim, T)
            x = x.permute(2, 0, 1).contiguous() # (batch_size, embed_dim, T) -> (T, batch_size, embed_dim)
        
        output = x

        return output

class SelfMultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, norm=True, dropout=0, causal=False, eps=EPS):
        super().__init__()

        if dropout == 0:
            self.dropout = False
        else:
            self.dropout = True
        
        self.norm = norm

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        if self.dropout:
            self.dropout1d = nn.Dropout(p=dropout)
        if self.norm:
            norm_name = 'cLN' if causal else 'gLM'
            self.norm1d = choose_layer_norm(norm_name, embed_dim, causal=causal, eps=eps)
    
    def forward(self, input):
        """
        Args:
            input (T, batch_size, embed_dim)
        Returns:
            output (T, batch_size, embed_dim)
        """
        x = input # (T, batch_size, embed_dim)

        residual = x
        x, _ = self.multihead_attn(x, x, x) # (T_tgt, batch_size, embed_dim), (batch_size, T_tgt, T_src), where T_tgt = T_src = T
        x = x + residual

        if self.dropout:
            x = self.dropout1d(x)
        
        if self.norm:
            x = x.permute(1, 2, 0) # (batch_size, embed_dim, T)
            x = self.norm1d(x) # (batch_size, embed_dim, T)
            x = x.permute(2, 0, 1).contiguous() # (batch_size, embed_dim, T) -> (T, batch_size, embed_dim)
        
        output = x

        return output

class FeedForwardBlock(nn.Module):
    def __init__(self, num_features, hidden_channels, norm=True, nonlinear='relu', causal=False, eps=EPS):
        super().__init__()

        if causal:
            bidirectional = False
            num_directions = 1 # uni-direction
        else:
            bidirectional = True
            num_directions = 2 # bi-direction
        
        self.norm = norm

        self.rnn = nn.LSTM(num_features, hidden_channels, batch_first=False, bidirectional=bidirectional)
        if nonlinear == 'relu':
            self.nonlinear1d = nn.ReLU()
        else:
            raise ValueError("Not support nonlinear function {}.".format(nonlinear))
        self.fc = nn.Linear(num_directions*hidden_channels, num_features)

        if self.norm:
            norm_name = 'cLN' if causal else 'gLM'
            self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps)
    
    def forward(self, input):
        """
        Args:
            input (T, batch_size, num_features)
        Returns:
            output (T, batch_size, num_features)
        """
        x = input # (T, batch_size, num_features)

        self.rnn.flatten_parameters()

        residual = x
        x, (_, _) = self.rnn(x) # (T, batch_size, num_features) -> (T, batch_size, num_directions*hidden_channels)
        x = self.nonlinear1d(x) # -> (T, batch_size, num_directions*hidden_channels)
        x = self.fc(x) # (T, batch_size, num_directions*hidden_channels) -> (T, batch_size, num_features)
        x = x + residual
        
        if self.norm:
            x = x.permute(1, 2, 0) # (T, batch_size, num_features) -> (batch_size, num_features, T)
            x = self.norm1d(x) # (batch_size, num_features, T)
            x = x.permute(2, 0, 1).contiguous() # (batch_size, num_features, T) -> (T, batch_size, num_features)

        output = x

        return output

def _test_multihead_attn_block():
    batch_size = 2
    T = 10
    embed_dim = 8
    num_heads = 4
    input = torch.randn((T, batch_size, embed_dim), dtype=torch.float)

    print('-'*10, "Non causal & No dropout", '-'*10)
    causal = False
    dropout = 0
    
    model = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('-'*10, "Causal & Dropout (p=0.3)", '-'*10)
    causal = True
    dropout = 0.3
    
    model = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_feedforward_block():
    batch_size = 2
    T = 10
    num_features, hidden_channels = 12, 10

    input = torch.randn((T, batch_size, num_features), dtype=torch.float)

    print('-'*10, "Causal", '-'*10)
    causal = True
    nonlinear = 'relu'
    
    model = FeedForwardBlock(num_features, hidden_channels, nonlinear=nonlinear, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_improved_transformer():
    batch_size = 2
    T = 10
    num_features, hidden_channels = 12, 10
    num_heads = 4

    input = torch.randn((T, batch_size, num_features), dtype=torch.float)

    print('-'*10, "Non causal", '-'*10)
    causal = False
    
    model = ImprovedTransformer(num_features, hidden_channels, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_transformer_block():
    batch_size = 2
    num_features, hidden_channels = 12, 8
    S, chunk_size = 10, 5 # global length and local length
    num_heads = 3
    input = torch.randn((batch_size, num_features, S, chunk_size), dtype=torch.float)

    print('-'*10, "transformer block for intra chunk", '-'*10)
    model = IntraChunkTransformer(num_features, hidden_channels=hidden_channels, num_heads=num_heads)
    print(model)

    output = model(input)
    print(input.size(), output.size())
    print()

    print('-'*10, "transformer block for inter chunk", '-'*10)
    causal = True
    model = InterChunkTransformer(num_features, hidden_channels=hidden_channels, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

def _test_dptransformer():
    batch_size = 2
    num_features, hidden_channels = 12, 8
    S, chunk_size = 10, 5 # global length and local length
    num_blocks = 6
    num_heads = 3
    input = torch.randn((batch_size, num_features, S, chunk_size), dtype=torch.float)
    causal = True

    model = DualPathTransformer(num_features, hidden_channels, num_blocks=num_blocks, num_heads=num_heads, causal=causal)
    print(model)

    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print('='*10, "Multihead attention block", '='*10)
    _test_multihead_attn_block()
    print()

    print('='*10, "feed-forward block", '='*10)
    _test_feedforward_block()
    print()

    print('='*10, "improved transformer", '='*10)
    _test_improved_transformer()
    print()
    
    print('='*10, "transformer block", '='*10)
    _test_transformer_block()
    print()

    print('='*10, "Dual path transformer network", '='*10)
    _test_dptransformer()
    print()