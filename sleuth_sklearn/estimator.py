import itertools

import numpy as np
import sleuth_sklearn.spread as sp
import sleuth_sklearn.stats as st
import sleuth_sklearn.utils as su

from numba import njit, types
from sklearn.base import BaseEstimator
from sleuth_sklearn.indices import J


@njit(
    types.f8[:](
        types.b1[:, :],
        types.i4[:, :],
        types.i4[:],
        types.i8,
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.NumPyRandomGeneratorType("prng"),
        types.f8[:, :],
    ),
)
def evaluate_combinations(
    X0,
    combs,
    years,
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
    nyears = years[-1] - years[0] + 1
    out = np.zeros(len(combs), dtype=np.float64)

    for i in range(len(combs)):
        c_diffusion, c_breed, c_spread, c_slope, c_road = combs[i]
        index = (c_diffusion, c_breed, c_spread, c_slope, c_road)
        print(index)

        records = sp.fill_montecarlo_grid(
            X0,
            nyears=nyears,
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

        out[i] = st.evaluate_records(
            records, years=years, calibration_stats=calibration_stats
        )
    return out


@njit(types.f8[:, :](types.b1[:, :, :], types.i4[:], types.i4[:, :]))
def calculate_initial_stats(X, y, grid_slope):
    calibration_stats = np.zeros((len(y), J.TOTAL_SIZE), dtype=np.float64)

    idx_arr = np.array(
        [
            J.EDGES,
            J.CLUSTERS,
            J.POP,
            J.XMEAN,
            J.YMEAN,
            J.SLOPE,
            J.RAD,
            J.MEAN_CLUSTER_SIZE,
            J.PERCENT_URBAN,
        ]
    )
    # TODO: Vectorize this
    for i in range(len(y)):
        calibration_stats[i, idx_arr] = st.compute_stats(X[i], grid_slope)

    return calibration_stats


class SLEUTH(BaseEstimator):
    def __init__(
        self,
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
        X = np.array(X, dtype=bool)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError

        # Set initial params
        self.random_state_ = np.random.Generator(np.random.SFC64(self.random_state))
        self.years_ = y
        self.calibration_stats_ = calculate_initial_stats(X, y, self.grid_slope)

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
            for grid in self.param_grids_:
                if grid == param_grid:
                    break

            self.param_grids_.append(param_grid)
            self.osm_ = {}

            combs = np.array(
                list(itertools.product(*param_grid.values())), dtype=np.int32
            )
            osm = evaluate_combinations(
                X[0],
                combs,
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

            for key, value in zip(combs, osm):
                self.osm_[tuple(key)] = value

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
                c_min = top_params[:, i].min()
                c_max = top_params[:, i].max()
                new_param_grid[field] = su.get_new_range(
                    param_grid[field], c_min, c_max, self.n_refinement_splits
                )

            self.param_grids_.append(new_param_grid)
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
