import torch
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import itertools

from src.sqw import SpecNeuralRepr
from src.dataset import NeighborDataset
from src.kernel import L_Kernel

# Load data
data_dict = torch.load('data/summarized_neutron_data_w_bkg_260meV_ML.pt')

# Prepare grid
hklw_grid = torch.vstack([_.unsqueeze(0) for _ in torch.meshgrid(*[v for k, v in data_dict['grid'].items()], indexing='ij')]).permute(1, 2, 3, 4, 0)

# Load pre-trained model
model_sqw = SpecNeuralRepr.load_from_checkpoint('version_14896845/checkpoints/epoch=7160-step=343728.ckpt')

# Define base configuration
base_config = {
    'dim': 4,
    'exclude_corner': True,
    'hidden_dim': 256,
    'num_layers': 3,
    'scale_factor_initial': 'none',
}

# Define ranges for neighbor_range and loss_bkg_mag_weight
neighbor_ranges = [3,4,5]
#neighbor_ranges = [5]
loss_bkg_mag_weights = [5e-6]

# Generate all combinations
configurations = list(itertools.product(neighbor_ranges, loss_bkg_mag_weights))

# Function to run training
def run_training(neighbor_range, loss_bkg_mag_weight):
    model_config = base_config.copy()
    model_config['neighbor_range'] = neighbor_range
    
    # Prepare dataset and dataloader
    dataset = NeighborDataset(hklw_grid, data_dict['S'], neighbor_range=neighbor_range)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=32)
    
    model_sqw.params = torch.tensor([29.0, 1.68])
    L_model = L_Kernel(forward_model=model_sqw, model_config=model_config, loss_bkg_mag_weight=loss_bkg_mag_weight)
    
    # Set logger to a common directory
    logger = TensorBoardLogger("/pscratch/sd/e/edmundxu/kernel-s2e/lightning_logs", name="training_runs")
    
    # Set checkpoint callback to save all checkpoints in the same directory
    checkpoint_callback = ModelCheckpoint(
        dirpath="/pscratch/sd/e/edmundxu/kernel-s2e/lightning_logs/training_runs/checkpoints",
        every_n_train_steps=10, save_last=True, save_top_k=1, monitor="train_loss",
        filename=f"nr_{neighbor_range}_lbmw_{loss_bkg_mag_weight}-{{epoch}}-{{step}}"
    )
    
    torch.set_float32_matmul_precision('high')
    
    trainer = lightning.Trainer(
        max_epochs=20, accelerator='gpu', logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=1, devices=1, sync_batchnorm=True,
        enable_checkpointing=True, default_root_dir='./'
    )
    
    trainer.fit(L_model, dataloader)

# Run training for each configuration
for neighbor_range, loss_bkg_mag_weight in configurations:
    print(f"Training with neighbor_range={neighbor_range}, loss_bkg_mag_weight={loss_bkg_mag_weight}")
    run_training(neighbor_range, loss_bkg_mag_weight)
