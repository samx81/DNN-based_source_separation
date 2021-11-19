import torch
import torch.nn as nn

from algorithm.clustering import KMeans

EPS = 1e-12

class DANet(nn.Module):
    def __init__(self, n_bins, embed_dim=20, hidden_channels=300, num_blocks=4, dropout=0, causal=False, mask_nonlinear='sigmoid', iter_clustering=10, take_log=True, take_db=False, eps=EPS):
        super().__init__()
        
        self.n_bins = n_bins
        self.hidden_channels, self.embed_dim = hidden_channels, embed_dim
        self.num_blocks = num_blocks

        self.dropout = dropout
        self.causal = causal

        if causal:
            num_directions = 1
            bidirectional = False
        else:
            num_directions = 2
            bidirectional = True
        
        self.mask_nonlinear = mask_nonlinear
        
        self.rnn = nn.LSTM(n_bins, hidden_channels, num_layers=num_blocks, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(num_directions*hidden_channels, n_bins*embed_dim)
        
        if mask_nonlinear == 'sigmoid':
            self.mask_nonlinear2d = nn.Sigmoid()
        elif mask_nonlinear == 'softmax':
            self.mask_nonlinear2d = nn.Softmax(dim=1)
        else:
            raise NotImplementedError("")
        
        self.iter_clustering = iter_clustering
        self.take_log, self.take_db = take_log, take_db
        self.eps = eps

        if self.take_log and self.take_db:
            raise ValueError("Either take_log or take_db should be False.")
    
    def forward(self, input, assignment=None, threshold_weight=None, n_sources=None, iter_clustering=None):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
            assignment <torch.Tensor>: Speaker assignment during training. Tensor shape is (batch_size, n_sources, n_bins, n_frames).
            threshold_weight <torch.Tensor> or <float>: (batch_size, 1, n_bins, n_frames)
        Returns:
            output (batch_size, n_sources, n_bins, n_frames)
        """
        output, _, _ = self.extract_latent(input, assignment, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=iter_clustering)
        
        return output
    
    def extract_latent(self, input, assignment=None, threshold_weight=None, n_sources=None, iter_clustering=None):
        """
        Args:
            input <torch.Tensor>: Amplitude with shape of (batch_size, 1, n_bins, n_frames).
            assignment <torch.Tensor>: Speaker assignment during training. Tensor shape is (batch_size, n_sources, n_bins, n_frames).
            threshold_weight <torch.Tensor> or <float>: (batch_size, 1, n_bins, n_frames)
        Returns:
            output <torch.Tensor>: (batch_size, n_sources, n_bins, n_frames)
            latent <torch.Tensor>: (batch_size, n_bins * n_frames, embed_dim)
            attractor <torch.Tensor>: (batch_size, n_sources, embed_dim)
        """
        if iter_clustering is None:
            iter_clustering = self.iter_clustering
        
        if n_sources is not None:
            if assignment is not None and n_sources != assignment.size(1):
                raise ValueError("n_sources is different from assignment.size(1)")
        else:
            if assignment is None:
                raise ValueError("Specify assignment, given None!")
            n_sources = assignment.size(1)
        
        embed_dim = self.embed_dim
        eps = self.eps
        
        batch_size, _, n_bins, n_frames = input.size()

        self.rnn.flatten_parameters()

        if self.take_log:
            x = torch.log(input + eps)
        elif self.take_db:
            x = 20 * torch.log10(input + eps)
        else:
            x = input
        
        x = x.squeeze(dim=1).permute(0, 2, 1).contiguous() # (batch_size, n_frames, n_bins)
        x, _ = self.rnn(x) # (batch_size, n_frames, n_bins)
        x = self.fc(x) # (batch_size, n_frames, embed_dim * n_bins)
        x = x.view(batch_size, n_frames, embed_dim, n_bins)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, embed_dim, n_bins, n_frames)
        latent = x.view(batch_size, embed_dim, n_bins * n_frames)
        latent = latent.permute(0, 2, 1).contiguous() # (batch_size, n_bins * n_frames, embed_dim)
        
        if assignment is None:
            if self.training:
                raise ValueError("assignment is required.")
            
            if threshold_weight is not None:
                assert batch_size == 1, "KMeans is expected same number of samples among all batches, so if threshold_weight is given, batch_size should be 1."
                
                flatten_latent = latent.view(batch_size * n_bins * n_frames, embed_dim) # (batch_size * n_bins * n_frames, embed_dim)
                flatten_threshold_weight = threshold_weight.view(-1) # (batch_size * n_bins * n_frames)
                nonzero_indices, = torch.nonzero(flatten_threshold_weight, as_tuple=True) # (n_nonzeros,)
                latent_nonzero = flatten_latent[nonzero_indices] # (n_nonzeros, embed_dim)
                latent_nonzero = latent_nonzero.view(batch_size, -1, embed_dim) # (batch_size, n_nonzeros, embed_dim)
            
            kmeans = KMeans(latent, K=n_sources)
            _, attractor = kmeans(iteration=iter_clustering) # (batch_size, n_bins * n_frames), (batch_size, n_sources, embed_dim)
        else:
            threshold_weight = threshold_weight.view(batch_size, 1, n_bins * n_frames)
            assignment = assignment.view(batch_size, n_sources, n_bins * n_frames) # (batch_size, n_sources, n_bins * n_frames)
            assignment = threshold_weight * assignment
            attractor = torch.bmm(assignment, latent) / (assignment.sum(dim=2, keepdim=True) + eps) # (batch_size, n_sources, embed_dim)
        
        similarity = torch.bmm(attractor, latent.permute(0, 2, 1)) # (batch_size, n_sources, n_bins * n_frames)
        similarity = similarity.view(batch_size, n_sources, n_bins, n_frames)
        mask = self.mask_nonlinear2d(similarity) # (batch_size, n_sources, n_bins, n_frames)
        output = mask * input

        return output, latent, attractor
    
    def get_config(self):
        config = {
            'n_bins': self.n_bins,
            'embed_dim': self.embed_dim,
            'hidden_channels': self.hidden_channels,
            'num_blocks': self.num_blocks,
            'dropout': self.dropout,
            'causal': self.causal,
            'mask_nonlinear': self.mask_nonlinear,
            'iter_clustering': self.iter_clustering,
            'take_log': self.take_log, 'take_db': self.take_db,
            'eps': self.eps
        }
        
        return config
    
    @classmethod
    def build_model(cls, model_path, load_state_dict=False):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        n_bins = config['n_bins']
        embed_dim = config['embed_dim']
        hidden_channels = config['hidden_channels']
        num_blocks = config['num_blocks']
        dropout = config['dropout']
        
        causal = config['causal']
        mask_nonlinear = config['mask_nonlinear']
        iter_clustering = config['iter_clustering']
        take_log, take_db = config['take_log'], config['take_db']
        
        eps = config['eps']
        
        model = cls(
            n_bins, embed_dim=embed_dim, hidden_channels=hidden_channels,
            num_blocks=num_blocks, dropout=dropout, causal=causal, mask_nonlinear=mask_nonlinear,
            iter_clustering=iter_clustering,
            take_log=take_log, take_db=take_db,
            eps=eps
        )

        if load_state_dict:
            model.load_state_dict(config['state_dict'])
        
        return model
    
    @property
    def num_parameters(self):
        _num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()
                
        return _num_parameters

def _test_danet():
    batch_size = 2
    K = 10
    
    H = 32
    B = 4
    
    n_bins, n_frames = 4, 128
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    assignment = compute_ideal_binary_mask(sources, source_dim=1)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)
    
    model = DANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, assignment, threshold_weight=threshold_weight)
    
    print(input.size(), output.size())

def _test_danet_paper():
    batch_size = 2
    K = 20
    
    H = 300
    B = 4
    
    n_bins, n_frames = 129, 256
    n_sources = 2
    causal = False
    mask_nonlinear = 'sigmoid'
    
    sources = torch.randn((batch_size, n_sources, n_bins, n_frames), dtype=torch.float)
    input = sources.sum(dim=1, keepdim=True)
    assignment = compute_ideal_binary_mask(sources, source_dim=1)
    threshold_weight = torch.randint(0, 2, (batch_size, 1, n_bins, n_frames), dtype=torch.float)
    
    model = DANet(n_bins, embed_dim=K, hidden_channels=H, num_blocks=B, causal=causal, mask_nonlinear=mask_nonlinear)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    output = model(input, assignment, threshold_weight=threshold_weight)
    
    print(input.size(), output.size())
        
if __name__ == '__main__':
    from algorithm.frequency_mask import compute_ideal_binary_mask

    torch.manual_seed(111)

    print("="*10, "DANet", "="*10)
    _test_danet()
    print()

    print("="*10, "DANet (same configuration in paper)", "="*10)
    _test_danet_paper()
    print()