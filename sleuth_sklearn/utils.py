import numpy as np
import xarray as xr

def generate_grid(p_min, p_max, n_p=5):
    assert p_min <= p_max
    delta = p_max - p_min
    if delta == 0:
        return [p_min]
    if n_p > delta + 1:
        n_p = delta + 1
    step = int(delta/(n_p - 1))
    last = step*(n_p - 1)
    deltas = np.arange(0, last + 1, step, dtype=int)
    remainder = delta - last
    adjust = [0]*(n_p - remainder) + list(range(1, remainder + 1))
    deltas += adjust
    return deltas + p_min


def open_dataset(path):
    ds = xr.open_dataset(
        path,
        cache = False,
        mask_and_scale = False
    )
    ds.load()
    ds.close()
    return ds


def get_new_params(param_grid, df):
    top = df.sort_values("rank_test_score").head(3)
    param_grid_new = {}
    for name in ["diffusion", "breed", "spread", "slope", "road"]:
        col = top[f"param_coef_{name}"]

        c_min = col.min()
        c_max = col.max()
        if c_min == c_max:
            grid = param_grid[f"coef_{name}"]
            grid = np.array(grid)
            idx = np.argwhere(grid == c_min).item()
            c_min = grid[max(0, idx - 1)]
            c_max = grid[min(idx + 1, len(grid) - 1)]

        param_grid_new[f"coef_{name}"] = generate_grid(c_min, c_max)
    return param_grid_new
