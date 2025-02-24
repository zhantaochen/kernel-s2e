import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader
from src.data_utils import get_neighbors, func_index_tensor
from src.sqw import SpecNeuralRepr
from src.qpath import linspace_2D_equidistant
from src.siren import SirenNet
from src.dataset import NeighborDataset
from src.kernel import KernelNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Set default tensor type
torch.set_default_dtype(torch.float32)

def load_data(filepath: str):
    """Loads dataset from a given file path."""
    return torch.load(filepath, weights_only=False)

def preprocess_data(data_dict):
    """Prepares data structures for training."""
    hklw_grid = torch.vstack([
        _.unsqueeze(0) for _ in torch.meshgrid(*[v for v in data_dict['grid'].values()], indexing='ij')
    ]).permute(1, 2, 3, 4, 0)
    mask = data_dict['S'] > 1e-3
    eroded_mask = binary_erosion(mask, iterations=2)
    interior_points = np.transpose(np.nonzero(eroded_mask))
    return hklw_grid, interior_points

def load_trained_model(checkpoint_path: str):
    """Loads a pre-trained spectral neural representation model."""
    return SpecNeuralRepr.load_from_checkpoint(checkpoint_path)

def initialize_kernel_network(device: torch.device, dtype: torch.dtype):
    """Initializes the kernel network with SIREN layers."""
    kernel_net = torch.nn.Sequential(
        SirenNet(dim_in=4, dim_hidden=256, dim_out=256, num_layers=3, w0_initial=30., final_activation=torch.nn.ReLU()),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 5**4),
        torch.nn.Softmax(dim=-1)
    )
    return kernel_net.to(device).to(dtype)

class L_Kernel(pl.LightningModule):
    """Kernel learning module using PyTorch Lightning."""
    def __init__(self, forward_model, dim=3, neighbor_range=1, exclude_corner=True):
        super().__init__()
        self.save_hyperparameters()
        self.kernel_net = KernelNet(dim=dim, neighbor_range=neighbor_range, exclude_corner=exclude_corner)
        self.forward_model = forward_model
        
    def forward(self, x):
        return self.kernel_net(x)
    
    def compute_metrics_on_batch(self, batch):
        kappa = self.forward(batch['center_pts'].to(self.dtype).to(self.device))
        neighb_data = self.forward_model.forward_qw(batch['neighb_pts'].to(self.dtype).to(self.device))
        s_pred = torch.einsum('ij, ij -> i', kappa, neighb_data[:, self.kernel_net.kernel_mask_flat]).unsqueeze(-1)
        s_target = batch['center_data']
        return torch.nn.functional.mse_loss(s_pred.cpu(), s_target.cpu())
    
    def training_step(self, batch, batch_idx):
        # Compute the metrics (loss)
        loss = self.compute_metrics_on_batch(batch)
        # Log the loss for checkpointing and monitoring
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_model(model, dataloader, max_epochs=10,limit_train_batches=None,checkpoint_dir=None):
    """Trains the kernel model using PyTorch Lightning."""
    # Define checkpoint callback
    callbacks = []
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,  # Folder to save checkpoints
            filename="kernel_model-{epoch:02d}-{val_loss:.4f}",  # Naming pattern
            save_top_k=3,  # Keep only the best 3 models
            monitor="train_loss",  # Save best based on validation loss
            mode="min"
        )
        callbacks.append(checkpoint_callback)  # Add callback to list
    
    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        limit_train_batches=limit_train_batches,
        callbacks=callbacks,  # <- Pass callbacks here
        default_root_dir=checkpoint_dir,  # <- Ensure logs & checkpoints go here
        devices=1
    )

    trainer.fit(model, dataloader)

def main():
    data_path = '/pscratch/sd/y/yn754/data/S2e/summarized_neutron_data_w_bkg_260meV_ML.pt'
    model_checkpoint = '/pscratch/sd/y/yn754/data/S2e/input/version_14896845/checkpoints/epoch=7160-step=343728.ckpt'
    
    # Load and preprocess data
    data_dict = load_data(data_path)
    hklw_grid, interior_points = preprocess_data(data_dict)
    
    # Load model, if none
    #model_sqw = load_trained_model(model_checkpoint)
    model_sqw = SpecNeuralRepr()
    model_sqw.params = torch.tensor([29, 1.7])
    
    # Initialize kernel network
    kernel_net = initialize_kernel_network(model_sqw.device, model_sqw.dtype)
    
    # Prepare dataset and dataloader
    dataset = NeighborDataset(hklw_grid, data_dict['S'], neighbor_range=2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Train model
    kernel_model = L_Kernel(forward_model=model_sqw, dim=4, neighbor_range=2, exclude_corner=True).to(model_sqw.device)
    train_model(kernel_model, dataloader,max_epochs=5000, limit_train_batches = 300000,checkpoint_dir=None)
    
if __name__ == "__main__":
    main()
