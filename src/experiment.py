import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def convert_to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().clone().numpy()
    elif isinstance(data, list):
        data = np.asarray(data)
    return data

def convert_to_torch(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.copy())
    elif isinstance(data, list):
        data = torch.from_numpy(np.ndarray(data))
    return data

class NeutronExperiment:

    def __init__(self, q_grid, w_grid, S_grid, S_scale_factor=1., mask_threshold=1e-3):
        """
        q_grid: tuple of (h_grid, k_grid, l_grid), each of shape (num_qi) for i = h,k,l
        w_grid: array of shape (num_w,)
        S_grid: array of shape (num_h, num_k, num_l, num_w)
        """
        self.h_grid = convert_to_torch(q_grid[0])
        self.k_grid = convert_to_torch(q_grid[1])
        self.l_grid = convert_to_torch(q_grid[2])
        self.w_grid = convert_to_torch(w_grid)

        self.S_scale_factor = S_scale_factor

        self.S_func = RegularGridInterpolator(
            [convert_to_numpy(_) for _ in [q_grid[0], q_grid[1], q_grid[2], w_grid]],
            self.S_scale_factor * convert_to_numpy(S_grid),
            bounds_error=False, fill_value=0, method='linear'
        )
        
        self.mask_func = RegularGridInterpolator(
            [convert_to_numpy(_) for _ in [q_grid[0], q_grid[1], q_grid[2], w_grid]],
            (convert_to_numpy(S_grid) > mask_threshold).astype(np.float32),
            bounds_error=False, fill_value=0, method='linear'
        )
    
    def prepare_experiment(self, coords):
        self.Sqw = torch.from_numpy(self.get_measurements_on_coords(coords))
        self.Sqw = self.Sqw.clamp_min(0.0)
    
    def get_measurements_by_mask(self, mask):
        S_out = self.Sqw[mask]
        return S_out
    
    def get_measurements_on_coords(self, coords):
        S_out = self.S_func(coords)
        return S_out
    
    def get_mask_on_coords(self, coords):
        S_out = self.mask_func(coords)
        return (S_out > 0.5).astype(np.float32)
    