import torch
import torch.nn as nn
import lightning
import itertools

from .siren import SirenNet

class KernelNet(nn.Module):
    def __init__(self, dim, neighbor_range, exclude_corner=False, hidden_dim=256, num_layers=3, w0_initial=30.):
        super().__init__()
        
        self.dim = dim
        self.edge_length = 2*neighbor_range+1
        self.neighbor_range = neighbor_range
        self.exclude_corner = exclude_corner
        if not exclude_corner:
            self.output_dim = int((2*neighbor_range+1)**dim)
        else:
            self.output_dim = int((2*neighbor_range+1)**dim - 2**dim)
        
        self.kernel_net = torch.nn.Sequential(
            SirenNet(
                dim_in = dim,
                dim_hidden = hidden_dim,
                dim_out = hidden_dim,
                num_layers = num_layers,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU()
            ),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.output_dim),
            torch.nn.Softmax(dim=-1)
        )
        
        self.get_kernel_mask()
    
    def get_kernel_mask(self, ):
        # Create a list of ranges for each dimension
        ranges = [range(-self.neighbor_range, self.neighbor_range + 1)] * self.dim

        # Generate all possible offset combinations
        offset_combinations = torch.tensor(list(itertools.product(*ranges)))
        
        if self.exclude_corner:
            dist_neighbors = offset_combinations.float().norm(dim=-1)
            valid_neighbors = dist_neighbors < dist_neighbors.max()
        else:
            valid_neighbors = torch.ones(offset_combinations.shape[0], dtype=bool)
        self.register_buffer('valid_neighbors', valid_neighbors)
        
    @property
    def kernel_mask_flat(self, ):
        return self.valid_neighbors
    
    @property
    def kernel_mask(self, ):
        return self.valid_neighbors.reshape((self.edge_length,) * self.dim)
    
    def forward(self, x):
        return self.kernel_net(x)
    