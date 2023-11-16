import numpy as np
import xarray as xr


def generate_grid(p_min, p_max, n_p=5):
    if n_p == 1:
        return np.array([p_min])

    assert p_min <= p_max

    delta = p_max - p_min
    if delta == 0:
        return [p_min]
    if n_p > delta + 1:
        n_p = delta + 1
    step = int(delta / (n_p - 1))
    last = step * (n_p - 1)
    deltas = np.arange(0, last + 1, step, dtype=int)
    remainder = delta - last
    adjust = [0] * (n_p - remainder) + list(range(1, remainder + 1))
    deltas += adjust
    return deltas + p_min


def open_dataset(path):
    ds = xr.open_dataset(path, cache=False, mask_and_scale=False)
    ds.load()
    ds.close()
    return ds


def get_new_range(previous_range, p_min, p_max, n_p=5):
    if p_min == p_max:
        grid = np.array(previous_range)
        idx = np.argwhere(grid == p_min).item()
        p_min = grid[max(0, idx - 1)]
        p_max = grid[min(idx + 1, len(grid) - 1)]

    return generate_grid(p_min, p_max, n_p=n_p)
