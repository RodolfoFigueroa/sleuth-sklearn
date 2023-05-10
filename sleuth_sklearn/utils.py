import numpy as np

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
