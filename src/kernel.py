import torch
import torch.nn as nn
#import lightning
import pytorch_lightning as lightning
import itertools

from .siren import SirenNet

class KernelNet(nn.Module):
    def __init__(self, 
                 dim, 
                 neighbor_range, 
                 exclude_corner=False, 
                 hidden_dim=256, 
                 num_layers=3, 
                 scale_factor_initial=300.):
        super().__init__()
        
        self.dim = dim
        self.edge_length = 2*neighbor_range+1
        self.neighbor_range = neighbor_range
        self.exclude_corner = exclude_corner
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if not exclude_corner:
            self.output_dim = int((2*neighbor_range+1)**dim)
        else:
            self.output_dim = int((2*neighbor_range+1)**dim - 2**dim)
        
        self.register_parameter('scale_factor', nn.Parameter(torch.tensor(scale_factor_initial)))
        
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
        
        self.sf_net = torch.nn.Sequential(
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
            torch.nn.ReLU()
        )
        
        # self.kernel_net = torch.nn.Sequential(
        #     SirenNet(
        #         dim_in = dim,
        #         dim_hidden = hidden_dim,
        #         dim_out = hidden_dim,
        #         num_layers = num_layers,
        #         w0_initial = 30.,
        #         final_activation = torch.nn.GELU()
        #     ),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(hidden_dim, self.output_dim),
        #     torch.nn.ReLU()
        # )
        
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
        # return self.scale_factor * self.kernel_net(x)
        return self.sf_net(x) * self.kernel_net(x)
    

class L_Kernel(lightning.LightningModule):
    def __init__(self, 
                 forward_model, 
                 model_config={
                     'dim': 4,
                     'neighbor_range': 1,
                     'exclude_corner': True,
                     'hidden_dim': 256,
                     'num_layers': 3,
                     'scale_factor_initial': 300.,
                 },
                 loss_bkg_mag_weight=5e-2
                ):
        super().__init__()
        self.save_hyperparameters('model_config', 'loss_bkg_mag_weight')
        
        self.model_config = model_config
        self.loss_bkg_mag_weight = loss_bkg_mag_weight
        
        self.kernel_net = KernelNet(**model_config)
        self.bkgd_net = SirenNet(
                dim_in = self.kernel_net.dim,
                dim_hidden = self.kernel_net.hidden_dim,
                dim_out = 1,
                num_layers = self.kernel_net.num_layers,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU()
        )
        self.forward_model = forward_model
        
    def forward(self, x):
        return self.kernel_net(x)
    
    def compute_metrics_on_batch(self, batch):
        kappa = self.forward(
            batch['center_pts'].to(self.dtype).to(self.device))
        neighb_data = self.forward_model.forward_qw(
            batch['neighb_pts'].to(self.dtype).to(self.device)) 
        s_sig = torch.einsum(
            'ij, ij -> i', 
            kappa, neighb_data[:,self.kernel_net.kernel_mask_flat]
        ).unsqueeze(-1)
        s_bkg = self.bkgd_net(batch['center_pts'].to(self.dtype).to(self.device))
        #s_pred = s_sig
        s_pred = s_sig + s_bkg
        s_target = batch['center_data']
        loss_reconst = torch.nn.functional.mse_loss(s_pred.cpu(), s_target.cpu())
        loss_bkg_mag = self.loss_bkg_mag_weight * s_bkg.pow(2).mean()
 
        #loss_bkg_mag = loss_bkg_mag * 0
        # loss = loss_reconst + loss_bkg_mag
        return loss_reconst, loss_bkg_mag
    
    def training_step(self, batch, batch_idx):
        loss_reconst, loss_bkg_mag = self.compute_metrics_on_batch(batch)
        # Print the norm of the parameters for debugging
        # kernel_norm = sum(p.norm().item() for p in self.kernel_net.parameters())
        # forward_model_norm = sum(p.norm().item() for p in self.forward_model.parameters())
        # print(f"Batch {batch_idx}: kernel norm: {kernel_norm:.4f}, forward_model norm: {forward_model_norm:.4f}")
        
        loss = loss_reconst + loss_bkg_mag
        # self.log('kernel_scale_factor', self.kernel_net.scale_factor.item(), prog_bar=True)
        self.log('train_reconst', loss_reconst.item(), prog_bar=True)
        self.log('train_bkg_mag', loss_bkg_mag.item(), prog_bar=True)
        self.log('train_loss', loss.item(), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_reconst, loss_bkg_mag = self.compute_metrics_on_batch(batch)
        loss = loss_reconst + loss_bkg_mag
        self.log('val_reconst', loss_reconst.item(), prog_bar=True)
        self.log('val_bkg_mag', loss_bkg_mag.item(), prog_bar=True)
        self.log('val_loss', loss.item(), prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        # optimizer = torch.optim.AdamW(
        #     [
        #         {'params': self.kernel_net.kernel_net.parameters(), 'lr': 5e-4},
        #         {'params': self.kernel_net.scale_factor, 'lr': 1e-1},
        #         {'params': self.bkgd_net.parameters(), 'lr': 5e-4},
        #     ], 
        #     lr=5e-4
        # )
        return optimizer