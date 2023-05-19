import itertools

import numpy as np
import sleuth_sklearn.spread as sp
import sleuth_sklearn.stats as st
import sleuth_sklearn.utils as su

from numba import jit, njit, prange
from sklearn.base import BaseEstimator
from sleuth_sklearn.indices import I, J


@njit
def evaluate_records(
    records,
    *,
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
    sim_means = records[sim_idx, I.AVERAGE, :]

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


@njit(parallel=True)
def fill_montecarlo_grid(
    X0,
    *,
    nyears,
    nrows,
    ncols,
    n_iters,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    coef_diffusion,
    coef_breed,
    coef_spread,
    coef_slope,
    coef_road,
    crit_slope,
    prng,
):
    grid_MC = np.zeros((nyears, nrows, ncols), dtype=np.uintc)
    records = np.zeros((nyears, 3, 20), dtype=np.float_)
    for i in prange(n_iters):
        grid_MC += sp.grow(
            X0,
            nyears=nyears,
            grid_slope=grid_slope,
            grid_excluded=grid_excluded,
            grid_roads=grid_roads,
            grid_roads_dist=grid_roads_dist,
            grid_roads_i=grid_roads_i,
            grid_roads_j=grid_roads_j,
            coef_diffusion=coef_diffusion,
            coef_breed=coef_breed,
            coef_spread=coef_spread,
            coef_slope=coef_slope,
            coef_road=coef_road,
            crit_slope=crit_slope,
            prng=prng,
            records=records,
            num_iters=i + 1,
        )
    return grid_MC, records


@jit
def evaluate_combinations(
    X0,
    combs,
    *,
    years,
    nyears,
    nrows,
    ncols,
    n_iters,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    crit_slope,
    prng,
    calibration_stats,
):
    out = np.zeros(len(combs))

    for i in range(len(combs)):
        c_diffusion, c_breed, c_spread, c_slope, c_road = combs[i]
        index = (c_diffusion, c_breed, c_spread, c_slope, c_road)
        print(index)

        grid_MC, records = fill_montecarlo_grid(
            X0,
            nyears=nyears,
            nrows=nrows,
            ncols=ncols,
            n_iters=n_iters,
            grid_slope=grid_slope,
            grid_excluded=grid_excluded,
            grid_roads=grid_roads,
            grid_roads_dist=grid_roads_dist,
            grid_roads_i=grid_roads_i,
            grid_roads_j=grid_roads_j,
            coef_diffusion=c_diffusion,
            coef_breed=c_breed,
            coef_spread=c_spread,
            coef_slope=c_slope,
            coef_road=c_road,
            crit_slope=crit_slope,
            prng=prng,
        )

        out[i] = evaluate_records(
            records, years=years, calibration_stats=calibration_stats
        )
    return out


class SLEUTH(BaseEstimator):
    def __init__(
        self,
        *,
        n_iters=10,
        n_refinement_iters=3,
        n_refinement_splits=5,
        n_refinement_winners=3,
        coef_range_diffusion=(1, 100),
        coef_range_breed=(1, 100),
        coef_range_spread=(1, 100),
        coef_range_slope=(1, 100),
        coef_range_road=(1, 100),
        grid_slope=np.empty(0),
        grid_excluded=np.empty(0),
        grid_roads=np.empty(0),
        grid_roads_dist=np.empty(0),
        grid_roads_i=np.empty(0),
        grid_roads_j=np.empty(0),
        crit_slope=50,
        random_state=None,
    ):
        self.n_iters = n_iters

        self.n_refinement_iters = n_refinement_iters
        self.n_refinement_splits = n_refinement_splits
        self.n_refinement_winners = n_refinement_winners

        self.coef_range_diffusion = coef_range_diffusion
        self.coef_range_breed = coef_range_breed
        self.coef_range_spread = coef_range_spread
        self.coef_range_slope = coef_range_slope
        self.coef_range_road = coef_range_road

        self.grid_slope = grid_slope
        self.grid_excluded = grid_excluded
        self.grid_roads = grid_roads
        self.grid_roads_dist = grid_roads_dist
        self.grid_roads_i = grid_roads_i
        self.grid_roads_j = grid_roads_j

        self.crit_slope = crit_slope

        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError

        # Set PRNG
        self.random_state_ = np.random.Generator(np.random.SFC64(self.random_state))

        self.years_ = y

        self.calibration_stats_ = np.zeros((len(y), 20))

        # TODO: Vectorize this
        for i, year in enumerate(y):
            stats_val = self.calibration_stats_[i]
            (
                stats_val[J.EDGES],
                stats_val[J.CLUSTERS],
                stats_val[J.POP],
                stats_val[J.XMEAN],
                stats_val[J.YMEAN],
                stats_val[J.SLOPE],
                stats_val[J.RAD],
                stats_val[J.MEAN_CLUSTER_SIZE],
                stats_val[J.PERCENT_URBAN],
            ) = st.compute_stats(X[i], self.grid_slope)

        nyears = y[-1] - y[0] + 1
        _, nrows, ncols = X.shape

        param_grid = dict(
            diffusion=su.generate_grid(
                *self.coef_range_diffusion, self.n_refinement_splits
            ),
            breed=su.generate_grid(*self.coef_range_breed, self.n_refinement_splits),
            spread=su.generate_grid(*self.coef_range_spread, self.n_refinement_splits),
            slope=su.generate_grid(*self.coef_range_slope, self.n_refinement_splits),
            road=su.generate_grid(*self.coef_range_slope, self.n_refinement_splits),
        )

        self.param_grids_ = []

        for refinement_iter in range(self.n_refinement_iters):
            # Prevent searching over the same grid
            if refinement_iter > 0:
                for grid in self.param_grids_:
                    if grid == param_grid:
                        break

            self.param_grids_.append(param_grid)
            self.osm_ = {}

            combs = list(itertools.product(*param_grid.values()))
            osm = evaluate_combinations(
                X[0],
                combs,
                nyears=nyears,
                nrows=nrows,
                ncols=ncols,
                n_iters=self.n_iters,
                grid_slope=self.grid_slope,
                grid_excluded=self.grid_excluded,
                grid_roads=self.grid_roads,
                grid_roads_dist=self.grid_roads_dist,
                grid_roads_i=self.grid_roads_i,
                grid_roads_j=self.grid_roads_j,
                crit_slope=self.crit_slope,
                prng=self.random_state_,
                years=self.years_,
                calibration_stats=self.calibration_stats_,
            )

            scores_sorted = list(
                sorted(self.osm_.items(), key=lambda x: x[1], reverse=True)
            )
            top_params = np.array(
                [x[0] for x in scores_sorted[: self.n_refinement_winners]]
            )

            new_param_grid = {}
            for i, field in enumerate(
                ["diffusion", "breed", "spread", "slope", "road"]
            ):
                c_min = top_params[i].min()
                c_max = top_params[i].max()
                new_param_grid[field] = su.get_new_range(
                    param_grid[field], c_min, c_max, self.n_refinement_splits
                )

            param_grid = new_param_grid

        final_params = max(self.osm_, key=self.osm_.get)
        self.coef_diffusion_ = final_params[0]
        self.coef_breed_ = final_params[1]
        self.coef_spread_ = final_params[2]
        self.coef_slope_ = final_params[3]
        self.coef_road_ = final_params[4]

        return self

    def predict(self, X, nyears):
        # Create monte carlo grid to accumulate probability of urbanization
        # one grid per simulated year
        self.random_state_ = np.random.Generator(np.random.SFC64(self.random_state))

        nrows, ncols = X.shape
        grid_MC = np.zeros((nyears, nrows, ncols))

        # Perform simulation from start to end year
        for i in range(self.n_iters):
            self.grow(
                grid_MC,
                X,
                coef_diffusion=self.coef_diffusion_,
                coef_breed=self.coef_breed_,
                coef_spread=self.coef_spread_,
                coef_slope=self.coef_slope_,
                coef_road=self.coef_road_,
            )

        grid_MC /= self.n_iters
        return grid_MC

    def _more_tags(self):
        return {"X_types": ["3darray"]}
