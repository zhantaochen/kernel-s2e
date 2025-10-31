from src.dataset import NeighborDataset
from torch.utils.data import DataLoader
from src.kernel import L_Kernel
from src.sqw import SpecNeuralRepr
from kernel_learning import *
from src.data_utils import get_neighbors
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import argparse

torch.set_default_dtype(torch.float32)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### load data and forward model##########
#data_path = '../summarized_neutron_data_w_bkg_260meV_ML.pt' #experimental data
data_path = '../summarized_neutron_data_w_bkg_260meV_synthetic_J32.00_Jp-2.60.pt' #synthetic data
data_dict = torch.load(data_path,weights_only=False)
print(data_dict.keys())

### forward model########################
best_ckpt = '../checkpoints/epoch=7160-step=343728.ckpt' #checkpoint for forward model
model_sqw0 = SpecNeuralRepr.load_from_checkpoint(best_ckpt,map_location=torch.device('cpu')).to(device) #specify forward model
#model_sqw0.params = torch.tensor([29, 1.7]) #experimental data
model_sqw0.params = torch.tensor([32,-2.6]) #synthetic data

#########################################

### load from new checkpoint ##########
ckpt_path = '../checkpoints/MSE_nr3_bkw5e-04_tvw0e+00/kernel_model-epoch=27-val_loss=0.0000.ckpt' #specify model checkpoint

model = L_Kernel.load_from_checkpoint(ckpt_path,forward_model = model_sqw0,map_location=torch.device('cpu')).to(device)
model_sqw = model.forward_model 
model_sqw.params

config = torch.load(ckpt_path,map_location=torch.device('cpu'))['hyper_parameters']
model_config = torch.load(ckpt_path,map_location=torch.device('cpu'))['hyper_parameters']['model_config']
model.load_state_dict(torch.load(ckpt_path,map_location=torch.device('cpu'))['state_dict'])

plot_path = f'../plots/MSE_nr_{config["model_config"]["neighbor_range"]}_bkw{config["loss_bkg_mag_weight"]}_tvw{config['loss_bkg_TV_weight']:.0e}/' #synthetic
os.makedirs(plot_path, exist_ok=True)
print('writing to ',plot_path)

hklw_grid = torch.vstack([_.unsqueeze(0) for _ in torch.meshgrid(*[v for k, v in data_dict['grid'].items()], indexing='ij')]).permute(1, 2, 3, 4, 0)
dataset = NeighborDataset(hklw_grid, data_dict['S'], neighbor_range=model_config['neighbor_range'])
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

batch = next(iter(dataloader))
center_pts = batch['center_pts']

from src.qpath import linspace_2D_equidistant

kpts = torch.from_numpy(linspace_2D_equidistant([[0,0],[0.5,0],[0.5,0.5],[0,0]], 100))
wpts = data_dict['grid']['w_grid']
qw_coords = []
for _l in data_dict['grid']['l_grid']:
    _qw_coords = torch.cat([
        kpts.unsqueeze(0).expand(wpts.shape[0], -1, -1),
        _l * torch.ones(wpts.shape[0], kpts.shape[0], 1),
        wpts.view(wpts.shape[0], -1).unsqueeze(1).expand(-1, kpts.shape[0], -1)
       ], dim=2)
    qw_coords.append(_qw_coords)
    
_l = data_dict['grid']['l_grid'][0]

params = config['forward_model_params']
# x_input = torch.zeros(wpts.shape[0], kpts.shape[0], 6)
# x_input[...,:4] = torch.cat([
#     kpts.unsqueeze(0).expand(wpts.shape[0], -1, -1),
#     _l * torch.ones(wpts.shape[0], kpts.shape[0], 1),
#     wpts.view(wpts.shape[0], -1).unsqueeze(1).expand(-1, kpts.shape[0], -1)
#    ], dim=2)
# x_input[...,4:] = params


center_pts = torch.vstack([_.unsqueeze(0) for _ in qw_coords])

s_sig = torch.zeros(center_pts.view(-1, 4).shape[:-1]+(1,))
s_bkg = torch.zeros_like(s_sig)

from src.experiment import NeutronExperiment


experiment_config = {
    "q_grid": tuple([data_dict['grid'][_grid] for _grid in ['h_grid', 'k_grid', 'l_grid']]),
    "w_grid": data_dict['grid']['w_grid'],
    "S_grid": data_dict['S'],
    "S_scale_factor": 1.
}

experiment = NeutronExperiment(**experiment_config)
# experiment.prepare_experiment(hklw_grid)

batches = np.array_split(np.arange(center_pts.view(-1,4).shape[0]), 10)

for i, batch_idx in tqdm(enumerate(batches), total=len(batches)):
    
    _center_pts = center_pts.view(-1,4)[batch_idx]
    _mask = experiment.get_mask_on_coords(_center_pts.cpu().numpy())
    
    sample_pts, _ = get_neighbors(_center_pts, dim=model_config['dim'], neighbor_range=model_config['neighbor_range'], exclude_corner=model_config['exclude_corner'], deltas=dataset.deltas)
    
    s_sample_list = []
    #s_sample = model_sqw.forward_qw(sample_pts)
    for i in range(0, len(sample_pts), 2**10):
        batch = sample_pts[i : i + 2**10]  # Send batch to CUDA
        with torch.no_grad():  # Disable gradient computation if not training
            batch_output = model_sqw.forward_qw(batch)
        s_sample_list.append(batch_output)  # Move result to CPU to free CUDA memory
    s_sample = torch.cat(s_sample_list, dim=0)  # Concatenate all batch results
    
    with torch.no_grad():
        kappa = model.kernel_net(_center_pts.to(model.dtype).to(model.device))
        _s_bkg = model.bkgd_net(_center_pts.to(model.dtype).to(model.device)).cpu()
    _s_sig = torch.einsum('ij, ij -> i', kappa, s_sample[:,model.kernel_net.kernel_mask_flat]).unsqueeze(-1)
    s_sig[batch_idx] = _s_sig.cpu() * torch.from_numpy(_mask).unsqueeze(-1)
    s_bkg[batch_idx] = _s_bkg * torch.from_numpy(_mask).unsqueeze(-1)


S_exp = 0.
mask_exp = 0.
for _qw_coords in qw_coords:
    S_exp += experiment.get_measurements_on_coords(_qw_coords)
    mask_exp += experiment.get_mask_on_coords(_qw_coords)

#save data
s_bkg_2d = s_bkg.reshape(center_pts.shape[:-1]).detach().cpu()
s_sig_2d = s_sig.reshape(center_pts.shape[:-1]).detach().cpu()
s_sim_2d = model_sqw.forward_qw(center_pts.to(model_sqw.device)).cpu()
s_exp_2d = S_exp

#
import matplotlib.pyplot as plt

# Choose perceptually uniform colormap suitable for scattering data
cmap = "viridis"

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

# Panel A: Background component
im0 = axes[0, 0].imshow(
    s_bkg.reshape(center_pts.shape[:-1]).detach().cpu().sum(0),
    origin='lower', aspect='auto', vmax=20, cmap=cmap
)
axes[0, 0].set_title("A. Estimated Background Component", fontsize=12)
axes[0, 0].set_xlabel("h (r.l.u.)", fontsize=10)
axes[0, 0].set_ylabel("k (r.l.u.)", fontsize=10)
cbar0 = fig.colorbar(im0, ax=axes[0, 0])
cbar0.set_label("Intensity (a.u.)", fontsize=10)

# Panel B: Signal component
im1 = axes[0, 1].imshow(
    s_sig.reshape(center_pts.shape[:-1]).sum(0),
    origin='lower', aspect='auto', cmap=cmap
)
axes[0, 1].set_title(r"B. Estimated Convolved Signal Component $S_{sim} * \kappa$", fontsize=12)
axes[0, 1].set_xlabel("h (r.l.u.)", fontsize=10)
axes[0, 1].set_ylabel("k (r.l.u.)", fontsize=10)
cbar1 = fig.colorbar(im1, ax=axes[0, 1])
cbar1.set_label("Intensity (a.u.)", fontsize=10)

# Panel C: Forward model output
im2 = axes[1, 0].imshow(
    model_sqw.forward_qw(center_pts.to(model_sqw.device)).cpu().sum(0),
    origin='lower', aspect='auto', cmap=cmap
)
axes[1, 0].set_title("C. Forward Model Output $S_{sim}$", fontsize=12)
axes[1, 0].set_xlabel("h (r.l.u.)", fontsize=10)
axes[1, 0].set_ylabel("k (r.l.u.)", fontsize=10)
cbar2 = fig.colorbar(im2, ax=axes[1, 0])
cbar2.set_label("Intensity (a.u.)", fontsize=10)

# Panel D: Experimental data
im3 = axes[1, 1].imshow(S_exp, origin='lower', aspect='auto', vmax=300, cmap=cmap)
axes[1, 1].set_title("D. Experimental Measurement $S_{expt}$", fontsize=12)
axes[1, 1].set_xlabel("h (r.l.u.)", fontsize=10)
axes[1, 1].set_ylabel("k (r.l.u.)", fontsize=10)
cbar3 = fig.colorbar(im3, ax=axes[1, 1])
cbar3.set_label("Intensity (a.u.)", fontsize=10)

# Optional: Main title
# fig.suptitle("Scattering Signal Decomposition in Reciprocal Space", fontsize=14)

# # Save in publication-ready format
plt.savefig(plot_path + "scattering_decomposition.png", dpi=300, bbox_inches='tight')
plt.savefig(plot_path + "scattering_decomposition.pdf", bbox_inches='tight')
plt.show()




S_mask = data_dict['S'] > 1e-3

batches = np.array_split(torch.where(S_mask.reshape(-1))[0], 2500)

center_pts_full_grid = hklw_grid.reshape(-1,4)
s_sig_full_grid = torch.zeros((data_dict['S'].numel(), 1,))
s_bkg_full_grid = torch.zeros_like(s_sig_full_grid)

batches = np.array_split(torch.where(S_mask.reshape(-1))[0], 2000)

for i, batch_idx in tqdm(enumerate(batches), total=len(batches)):
    
    _center_pts = center_pts_full_grid[batch_idx]
    _mask = experiment.get_mask_on_coords(_center_pts.cpu().numpy())
    
    sample_pts, _ = get_neighbors(_center_pts, dim=model_config['dim'], neighbor_range=model_config['neighbor_range'], exclude_corner=model_config['exclude_corner'], deltas=dataset.deltas)
    #s_sample = model_sqw.forward_qw(sample_pts)
    s_sample_list = []
    #s_sample = model_sqw.forward_qw(sample_pts)
    for i in range(0, len(sample_pts), 2**10):
        batch = sample_pts[i : i + 2**10]  # Send batch to CUDA
        with torch.no_grad():  # Disable gradient computation if not training
            batch_output = model_sqw.forward_qw(batch)
        s_sample_list.append(batch_output)  # Move result to CPU to free CUDA memory
    s_sample = torch.cat(s_sample_list, dim=0)  # Concatenate all batch results
    
    with torch.no_grad():
        kappa = model.kernel_net(_center_pts.to(model.dtype).to(model.device))
        _s_bkg = model.bkgd_net(_center_pts.to(model.dtype).to(model.device)).cpu()
    _s_sig = torch.einsum('ij, ij -> i', kappa, s_sample[:,model.kernel_net.kernel_mask_flat]).unsqueeze(-1)
    s_sig_full_grid[batch_idx] = _s_sig.cpu() * torch.from_numpy(_mask).unsqueeze(-1)
    s_bkg_full_grid[batch_idx] = _s_bkg * torch.from_numpy(_mask).unsqueeze(-1)


    # Assumed shape: (N, D) where N is total points, D is dimension (e.g., 5 for hklwq)


batch_size = 1024  # adjust based on GPU memory
num_points = center_pts_full_grid.shape[0]

device = model_sqw.device
outputs = []

# Run inference in batches
for i in tqdm(range(0, num_points, batch_size)):
    batch = center_pts_full_grid[i:i+batch_size].to(device)
    with torch.no_grad():
        out = model_sqw.forward_qw(batch).cpu()
    outputs.append(out)

# Concatenate and reshape if needed
outputs_full = torch.cat(outputs, dim=0)  # shape: (N, ...)

# np.save(f's_sig_full_grid_nr{config['model_config']['neighbor_range']}_bkw{config['loss_bkg_mag_weight']}_tvw{config['loss_bkg_TV_weight']}.npy', s_sig_full_grid)
# np.save(f's_bkg_full_grid_nr{config['model_config']['neighbor_range']}_bkw{config['loss_bkg_mag_weight']}_tvw{config['loss_bkg_TV_weight']}.npy', s_bkg_full_grid)

# Load the data dictionary if not already loaded
data_dict = load_data(data_path)
S_expt = data_dict['S']  # This is your experimental scattering data

# For visualization, you may want to reshape it:
# (Assuming that S_expt has been flattened or requires reshaping to match your grid)
s_exp_full_2d = S_expt.reshape(hklw_grid.shape[:-1])
s_sim_full_2d = outputs_full.reshape(hklw_grid.shape[:-1])
s_bkg_full_2d = s_bkg_full_grid.reshape(hklw_grid.shape[:-1])
s_sig_full_2d = s_sig_full_grid.reshape(hklw_grid.shape[:-1])

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Prepare the 2D arrays by reshaping and summing over the last two axes.
S_expt_plot = s_exp_full_2d.sum(-1).sum(-1)
S_sim_plot  = outputs_full.reshape(hklw_grid.shape[:-1]).sum(-1).sum(-1)
S_sig_plot  = s_sig_full_grid.reshape(hklw_grid.shape[:-1]).sum(-1).sum(-1)
S_bkg_plot  = s_bkg_full_grid.reshape(hklw_grid.shape[:-1]).sum(-1).sum(-1)
recon_plot  = (s_bkg_full_grid + s_sig_full_grid).reshape(hklw_grid.shape[:-1]).sum(-1).sum(-1)

# Create a figure with a 2-row x 3-column GridSpec.
# The first column (spanning both rows) will host the large experimental plot.
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[1.5, 1, 1], wspace=0.4, hspace=0.4)

# Big left panel: Experimental Scattering Data (spanning both rows)
ax_main = fig.add_subplot(gs[:, 0])
im_main = ax_main.imshow(S_expt_plot, origin='lower', aspect='auto', cmap='viridis')
ax_main.set_title("Experimental Scattering Data $S_{expt}$", fontsize=14)
ax_main.set_xlabel("h (r.l.u.)", fontsize=12)
ax_main.set_ylabel("k (r.l.u.)", fontsize=12)
cbar_main = plt.colorbar(im_main, ax=ax_main, fraction=0.046, pad=0.04)
cbar_main.set_label("Intensity (a.u.)", fontsize=12)

# Compute the aspect ratio from the image data (height / width)
aspect_ratio = S_expt_plot.shape[0] / S_expt_plot.shape[1]
# Try to set the axis box aspect to preserve the original image proportions
try:
    ax_main.set_box_aspect(aspect_ratio)
except Exception:
    ax_main.set_aspect(aspect_ratio)
# Center the axis content within its allocated grid cell
ax_main.set_anchor('C')

# Top-middle panel: Simulated Scattering Data S_sim
ax1 = fig.add_subplot(gs[0, 1])
im1 = ax1.imshow(S_sim_plot, origin='lower', aspect='auto', cmap='viridis')
ax1.set_title("Simulated Scattering Data $S_{sim}$", fontsize=10)
ax1.set_xlabel("h (r.l.u.)", fontsize=10)
ax1.set_ylabel("k (r.l.u.)", fontsize=10)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Intensity (a.u.)", fontsize=10)

# Top-right panel: Estimated Convolved Signal Component
ax2 = fig.add_subplot(gs[0, 2])
im2 = ax2.imshow(S_sig_plot, origin='lower', aspect='auto', cmap='viridis')
ax2.set_title(r"Estimated Convolved Signal $K_\phi * S_{sim}$", fontsize=10)
ax2.set_xlabel("h (r.l.u.)", fontsize=10)
ax2.set_ylabel("k (r.l.u.)", fontsize=10)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Intensity (a.u.)", fontsize=10)

# Bottom-middle panel: Estimated Background Component
ax3 = fig.add_subplot(gs[1, 1])
im3 = ax3.imshow(S_bkg_plot, origin='lower', aspect='auto', cmap='viridis')
ax3.set_title(r"Estimated Background $B_{\theta}$", fontsize=10)
ax3.set_xlabel("h (r.l.u.)", fontsize=10)
ax3.set_ylabel("k (r.l.u.)", fontsize=10)
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label("Intensity (a.u.)", fontsize=10)

# Bottom-right panel: Reconstructed $S_{expt}$ = s_bkg + s_sig
ax4 = fig.add_subplot(gs[1, 2])
im4 = ax4.imshow(recon_plot, origin='lower', aspect='auto', cmap='viridis')
ax4.set_title(r"Reconstrion  $B_\theta + K_\phi * S_{sim}$", fontsize=10)
ax4.set_xlabel("h (r.l.u.)", fontsize=10)
ax4.set_ylabel("k (r.l.u.)", fontsize=10)
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cbar4.set_label("Intensity (a.u.)", fontsize=10)

# Save the composite figure using filenames based on your configuration.
filename = plot_path + "reconstruction"
plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{filename}.pdf', bbox_inches='tight')

plt.show()

np.savez_compressed(
    os.path.join(plot_path, "data_arrays.npz"),
    s_bkg_2d = s_bkg_2d,
    s_sig_2d = s_sig_2d,
    s_sim_2d = s_sim_2d,
    s_exp_2d = s_exp_2d,
    s_exp_full_2d = s_exp_full_2d,
    s_sim_full_2d = s_sim_full_2d,
    s_bkg_full_2d = s_bkg_full_2d,
    s_sig_full_2d = s_sig_full_2d,
)

