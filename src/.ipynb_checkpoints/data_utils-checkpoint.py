import numpy as np
import torch
import itertools

def ensure_array(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")
    
def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        return torch.tensor(x)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def get_neighbors(index, dim, neighbor_range, return_flat=True, exclude_corner=False):
    """    
    Get the indices of all neighbors around a given index or batch of indices.
    Drafted by Claude 3 Sonnet LLM on Mar 11, 2024.

    Args:
        index (torch.Tensor): Index or batch of indices.
        dim (int): Dimension of the indices.
        neighbor_range (int): Range of neighbors to include (1 for immediate neighbors, 2 for two steps away, etc.).

    Returns:
        torch.Tensor: Tensor of indices of all neighbors, including the input index(es).
    """
    # Convert the input index to a tensor if it's not already
    index = torch.tensor(index) if not torch.is_tensor(index) else index
    
    # Flatten the input index tensor
    flat_index = index.reshape(-1, dim)

    # Create a list of ranges for each dimension
    ranges = [range(-neighbor_range, neighbor_range + 1)] * dim

    # Generate all possible offset combinations
    offset_combinations = torch.tensor(list(itertools.product(*ranges)))

    # Broadcast the flattened input index tensor and add all offset combinations
    neighbor_indices = flat_index[:, None, :] + offset_combinations[None, :, :]
    
    if exclude_corner:
        dist_neighbors = offset_combinations.float().norm(dim=-1)
        valid_neighbors = dist_neighbors < dist_neighbors.max()
    else:
        valid_neighbors = torch.ones(offset_combinations.shape[0], device=index.device, dtype=bool)

    if return_flat:
        return neighbor_indices, valid_neighbors
    else:
        return neighbor_indices.reshape(-1, *((2*neighbor_range+1,) * dim), dim), valid_neighbors.reshape(*((2*neighbor_range+1,) * dim))
    

def func_index_tensor(data_tensor, index_tensor):
    """
    Index data_tensor based on index_tensor.

    Args:
        data_tensor (torch.Tensor): Tensor of shape (a1, a2, ..., ad, D).
        index_tensor (torch.Tensor): Tensor of shape (..., d), reshaped to (N, d), 
                        where each element of the second dimension corresponds
                        to each of the first 4 dimensions of data_tensor.

    Returns:
        torch.Tensor: Tensor of shape (N, D) containing the indexed values from data_tensor.
    """
    if index_tensor.ndim == 1:
        index_tensor = index_tensor[None, :]
    assert index_tensor.shape[-1] == data_tensor.ndim-1, \
        "The second dimension of index_tensor must match data_tensor.ndim-1."

    # Convert index_tensor to a compatible indexing format
    index_shape = index_tensor.shape[:-1]
    index_tensor = index_tensor.reshape(-1, index_tensor.shape[-1]).long()
    
    # Index tensor A using the created indices
    indexed_values = data_tensor[[idx for idx in index_tensor.T]]

    return indexed_values.reshape(*index_shape, data_tensor.shape[-1])