import numpy as np
import sleuth_sklearn.labeling as sl

from numba import njit
from sleuth_sklearn.indices import J


@njit
def compute_stats(urban, slope):
    # Assuming binarized urban raster (0/1)
    area = urban.sum()
    # orginal sleuth code discounts roads and excluded pixels
    # and include roads pixels as urban, which seems weird
    # anyhow, since excluded and roads are fixed, this just rescales
    percent_urban = area / urban.size * 100

    # number of pixels on urban edge
    edges = count_edges(urban)

    # Get a labeled array, by default considers von neumann neighbors
    _, nclusters = sl.hoshen_kopelman(urban > 0)
    assert nclusters > 0

    mean_cluster_size = area / nclusters

    avg_slope = (slope * urban).mean()

    # Centroid of urban pixels
    nzi, nzj = urban.nonzero()
    ymean, xmean = nzi.mean(), nzj.mean()

    # radius of circle of area equal to urban area
    rad = np.sqrt(area / np.pi)

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
        percent_urban,
    )


@njit
def count_edges(arr):
    s = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            a = -4 * arr[i, j]
            b = arr[i - 1, j] if i - 1 >= 0 else 0
            c = arr[i + 1, j] if i + 1 < arr.shape[0] else 0
            d = arr[i, j - 1] if j - 1 >= 0 else 0
            e = arr[i, j + 1] if j + 1 < arr.shape[1] else 0
            tot = a + b + c + d + e
            s += tot < 0
    return s


@njit
def evaluate_records(
    records,
    years,
    calibration_stats,
):
    # Calculate calibration statistics
    # The modified coefficients are extracted
    # from the last oberseved year average record.

    # The optimal SLEUTH metric is the product of:
    # compare, pop, edges, clusters, slope, x_mean, and y_mean
    # Extract mean records for urban years
    # sim_means = [record[I.AVERAGE] for record, year in zip(records, sim_years) if year in self.years_]
    sim_idx = np.array([year in years for year in range(years[0], years[-1] + 1)])
    sim_means = records[sim_idx]

    assert sim_means.shape == calibration_stats.shape

    # Compare: ratio of final urbanizations at last control years
    final_pop_sim = sim_means[-1][J.POP]
    final_pop_urb = calibration_stats[-1][J.POP]
    compare = min(final_pop_sim, final_pop_urb) / max(final_pop_sim, final_pop_urb)

    # Find regression coefficients, ignore seed year
    # osm_metrics_names = ["pop", "edges", "clusters", "slope", "xmean", "ymean"]
    osm_metrics_idx = [J.POP, J.EDGES, J.CLUSTERS, J.SLOPE, J.XMEAN, J.YMEAN]
    osm_metrics = []
    for idx in osm_metrics_idx:
        # simulation skips seed year
        sim_vals = [s[idx] for s in sim_means[1:]]
        urb_vals = [s[idx] for s in calibration_stats[1:]]
        # r, _ = pearsonr(sim_vals, urb_vals)
        cor_mat = np.corrcoef(sim_vals, urb_vals)
        r = cor_mat[0, 1]
        r = r**2
        osm_metrics.append(r)
    osm_metrics = np.array(osm_metrics)

    # Optimal metric
    osm = np.prod(osm_metrics) * compare
    return osm
