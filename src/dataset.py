import numpy as np

from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion

from .data_utils import get_neighbors, func_index_tensor, ensure_array, ensure_tensor


class NeighborDataset(Dataset):
    def __init__(self, grid, data, neighbor_range, data_mask_threshold=1e-3):
        
        self.deltas = (grid[tuple(1 for _ in range(grid.ndim - 1)) + (slice(None),)] - 
                       grid[tuple(0 for _ in range(grid.ndim - 1)) + (slice(None),)]).detach().cpu()
        self.grid = grid
        if data.ndim == grid.ndim-1:
            self.data = data.unsqueeze(-1)
        else:
            self.data = data
        self.neighbor_range = neighbor_range
        self.data_mask = data > data_mask_threshold
        
        _interior_idx = binary_erosion(
            ensure_array(self.data_mask), iterations=neighbor_range)
        _interior_idx = np.transpose(np.nonzero(_interior_idx))
        self.interior_idx = ensure_tensor(_interior_idx)
        del _interior_idx
        
    def __len__(self):
        return self.interior_idx.shape[0]

    def __getitem__(self, idx):
        center_idx = self.interior_idx[idx]
        center_pts = func_index_tensor(self.grid, center_idx)
        
        neighb_idx, neighb_mask = get_neighbors(center_idx, dim=self.grid.shape[-1], 
                                   neighbor_range=self.neighbor_range)
        neighb_pts = func_index_tensor(self.grid, neighb_idx)
        out_dict = {
            'center_pts': center_pts.squeeze(0),
            'neighb_pts': neighb_pts.squeeze(0),
            'neighb_mask': neighb_mask,
            'center_data': func_index_tensor(self.data, center_idx).squeeze(0),
        }
        return out_dict