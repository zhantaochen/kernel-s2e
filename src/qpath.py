import numpy as np


# Function created by OpenAI's GPT-4 model
def linspace_2D_equidistant(points, N, return_indices=False):
    points = np.asarray(points, dtype=float)  # Ensure points are float type
    N_pt = len(points)

    if N_pt < 2:
        raise ValueError("At least two points are required.")

    # Calculate the total length of the path
    segment_lengths = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    total_length = np.sum(segment_lengths)

    # Preallocate the array size
    result = np.zeros((N, 2), dtype=float)  # Ensure result is float type

    # Determine how many points are needed for each segment
    N_segment = np.round(segment_lengths / total_length * (N-1)).astype(int)

    # Handle rounding error by adjusting the last segment
    N_segment[-1] = N - np.sum(N_segment[:-1]) - 1

    # Generate points for each segment
    idx = 0
    critical_point_indices = [0]  # Start with the first critical point
    for i in range(N_pt - 1):
        x_vals = np.linspace(points[i, 0], points[i + 1, 0], N_segment[i]+1)[:-1]
        y_vals = np.linspace(points[i, 1], points[i + 1, 1], N_segment[i]+1)[:-1]

        result[idx:idx+N_segment[i], :] = np.stack((x_vals, y_vals), axis=-1)
        idx += N_segment[i]
        critical_point_indices.append(idx)

    # Include the last point of the last segment
    result[-1] = points[-1]
    if return_indices:
        return result, critical_point_indices
    else:
        return result