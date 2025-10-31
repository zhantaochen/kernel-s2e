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
from src.kernel import KernelNet, L_Kernel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import os

# Set default tensor type
torch.set_default_dtype(torch.float32)

def load_data(filepath: str):
    """Loads dataset from a given file path."""
    return torch.load(filepath, weights_only=False)

def preprocess_data(data_dict, roi=None):
    """Prepares data structures for training."""
    hklw_grid = torch.vstack([
        _.unsqueeze(0) for _ in torch.meshgrid(*[v for v in data_dict['grid'].values()], indexing='ij')
    ]).permute(1, 2, 3, 4, 0)
    mask = data_dict['S'] > 1e-3
    if roi is not None:
        mask = data_dict['S'][roi] > 1e-3
        hklw_grid = hklw_grid[roi]
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


def train_model(model, dataloader, max_epochs=10,limit_train_batches=None, checkpoint_dir=None,save_best = False, wandb_logger = None):
    """Trains the kernel model using PyTorch Lightning."""
    # Define checkpoint callback
    callbacks = []
    if save_best:
        checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,  # Folder to save checkpoints
        filename="kernel_model-{epoch:02d}-{val_loss:.4f}",  # Naming pattern
        save_top_k=1,  # Keep only the best 3 models
        save_last=True,
        monitor="train_loss",  # Save best based on validation loss
        mode="min"
        )
        callbacks.append(checkpoint_callback)  # Add callback to list

    trainer = pl.Trainer(
    max_epochs=max_epochs, 
    limit_train_batches=limit_train_batches,
    callbacks=callbacks,  # <- Pass callbacks here
    default_root_dir=checkpoint_dir,  # <- Ensure logs & checkpoints go here
    #devices=1,
    #devices=[0],
    logger=wandb_logger,
    )

    trainer.fit(model, dataloader)

def main():
    data_path = '../data/summarized_neutron_data_w_bkg_260meV_synthetic_J32.00_Jp-2.60.pt' #path to synthetic data
    SpecNeuralRepr_ckpt = '../checkpoints/epoch=7160-step=343728.ckpt' #checkpoint for forward model

    # Load and preprocess data
    data_dict = load_data(data_path)
    hklw_grid, interior_points = preprocess_data(data_dict)

    # Load model, if none
    model_sqw = SpecNeuralRepr.load_from_checkpoint(SpecNeuralRepr_ckpt,map_location=torch.device('cpu'))
    model_sqw.params = torch.tensor([32, -2.6]) #specify J and J_p

    #Configurations
    config = {'model_config': {'dim': 4,
                         'neighbor_range': 4,
                         'exclude_corner': True,
                         'hidden_dim': 256,
                         'num_layers': 3,
                         'scale_factor_initial': 300.
                         }, #model config for Siren
            'loss_bkg_mag_weight' : 5e-2, #background magnitude regularization lambda
            'loss_bkg_TV_weight':0, #TV regularization
            'model_sqw_params':model_sqw.params,
                 }

    # Format subdirectory name based on config
    config_tag = f"synthetic_nr{config['model_config']['neighbor_range']}_bkw{config['loss_bkg_mag_weight']:.0e}_tvw{config['loss_bkg_TV_weight']:.0e}"
    checkpoint_dir = f"../checkpoints/{config_tag}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize WandB
    wandb_logger = WandbLogger(
        project="Spectral_Neural_Representation",  # Change this to your project name
        name=f"synthetic_nr_{config['model_config']['neighbor_range']}_bkgw_{config['loss_bkg_mag_weight']}_tvw{config['loss_bkg_TV_weight']}",
        save_dir=checkpoint_dir,  # Save logs to checkpoint directory
        log_model = True,
    )
    wandb_logger.log_hyperparams(config)
 
    # Prepare dataset and dataloader
    dataset = NeighborDataset(hklw_grid, data_dict['S'], neighbor_range= config['model_config']['neighbor_range'])
    dataloader = DataLoader(dataset, batch_size= 3000, shuffle=True,num_workers=32)
    
    # Train model
    kernel_model = L_Kernel(forward_model=model_sqw, forward_model_params = model_sqw.params, model_config = config['model_config'],loss_bkg_mag_weight = config['loss_bkg_mag_weight'],loss_bkg_TV_weight = config['loss_bkg_TV_weight'])
    train_model(kernel_model, dataloader,max_epochs=30, limit_train_batches = None,checkpoint_dir=checkpoint_dir,save_best=True,wandb_logger=wandb_logger)
    wandb.finish()
     
if __name__ == "__main__":
     main()
