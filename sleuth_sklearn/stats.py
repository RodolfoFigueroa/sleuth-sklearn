import numpy as np

from dataclasses import dataclass, fields, field
from scipy.ndimage import convolve, label, center_of_mass


@dataclass
class StatsVal:
    sng: float = 0.0
    sdc: float = 0.0
    og: float = 0.0
    rt: float = 0.0
    num_growth_pix: float = 0.0
    diffusion: float = 0.0
    spread: float = 0.0
    breed: float = 0.0
    slope_resistance: float = 0.0
    road_gravity: float = 0.0
    edges: float = 0.0
    clusters: float = 0.0
    pop: float = 0.0
    xmean: float = 0.0
    ymean: float = 0.0
    slope: float = 0.0
    rad: float = 0.0
    mean_cluster_size: float = 0.0
    percent_urban: float = 0.0
    growth_rate: float = 0.0

    def reset(self):
        for f in fields(self):
            setattr(self, f.name, 0.0)


@dataclass
class Record:
    year: int
    is_calibration: bool
    diffusion: int
    breed: int
    spread: int
    slope: int
    road: int
    monte_carlo: int = 0
    this_year: StatsVal = field(default_factory=StatsVal)
    average: StatsVal = field(default_factory=StatsVal)
    std: StatsVal = field(default_factory=StatsVal)

    def update_mean_std(self):
        # Update the mean and sum of squares using
        # Welford's algorithm

        for f in fields(self.this_year):
            value = getattr(self.this_year, f.name)
            prev_mean = getattr(self.average, f.name)
            prev_sum = getattr(self.std, f.name)

            new_mean = prev_mean + (value - prev_mean) / self.monte_carlo
            new_sum = prev_sum + (value - prev_mean) * (value - new_mean)

            setattr(self.average, f.name, new_mean)
            setattr(self.std, f.name, new_sum)

    def compute_std(self):
        for f in fields(self.this_year):
            sum_sq = getattr(self.std, f.name)
            setattr(self.std, f.name, np.sqrt(sum_sq / self.monte_carlo))


@dataclass
class UrbAttempt:
    successes: int = 0
    z_failure: int = 0
    delta_failure: int = 0
    slope_failure: int = 0
    excluded_failure: int = 0

    def reset(self):
        for f in fields(self):
            setattr(self, f.name, 0)


def compute_stats(urban, slope):
    # Assuming binarized urban raster (0/1)
    area = urban.sum()
    # orginal sleuth code discounts roads and excluded pixels
    # and include roads pixels as urban, which seems weird
    # anyhow, since excluded and roads are fixed, this just rescales
    percent_urban = area / np.prod(urban.size) * 100

    # number of pixels on urban edge
    edges = count_edges(urban)

    # Get a labeled array, by default considers von neumann neighbors
    _, nclusters = label(urban)
    assert nclusters > 0

    mean_cluster_size = area / nclusters

    avg_slope = (slope * urban).mean()

    # Centroid of urban pixels
    ymean, xmean = center_of_mass(urban)

    # radius of circle of area equal to urban area
    rad = np.sqrt(area/np.pi)

    # Returns a dict of statistics
    # Seems pop and area are the same in orginal SLEUTH code
    return (
        edges,
        nclusters,
        area,
        xmean,
        ymean,
        avg_slope,
        rad,
        mean_cluster_size,
        percent_urban
    )


def count_edges(urban):
    # Peform a convolution to search for edges
    # Orignal SLEUTH code searches in the Von Neuman
    # neighborhood for empty cells. This is akin to perform
    # a convolution with the Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Scipy's ndimage.convolve is faster than signal.convolve
    # for 2D images
    # signal.convolve is more general and handles ndim arrays
    # TODO: splicitly pass output array to save memory
    conv = convolve(urban, kernel, mode='constant', output=int)

    edges = (conv < 0).sum()

    # Alterantive: loop only over urbanized pixels. Urbanize pixel
    # coordinates may be stored in a set, which allows for fast
    # lookup, insertion and deletion. But the convolution operator may
    # be adapted for GPU computation.

    return edges
