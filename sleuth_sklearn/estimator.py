import itertools
import multiprocessing
import numba

import numpy as np
import pandas as pd
import sleuth_sklearn.spread as sp
import sleuth_sklearn.stats as st
import sleuth_sklearn.utils as su

from numba import njit, typed, types
from pathlib import Path
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
        types.ListType(types.NumPyRandomGeneratorType("prng")),
        types.f8[:, :],
    ),
    cache=True,
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
    prngs,
    calibration_stats,
):
    nyears = years[-1] - years[0] + 1
    out = np.zeros(len(combs), dtype=np.float64)

    for i in range(len(combs)):
        c_diffusion, c_breed, c_spread, c_slope, c_road = combs[i]
        index = (c_diffusion, c_breed, c_spread, c_slope, c_road)
        print(index)

        _, records = sp.fill_montecarlo_grid(
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
            prngs=prngs,
        )

        res = st.evaluate_records(records, years, calibration_stats)
        print(res)
        out[i] = res
    return out


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
        n_jobs=-1
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
        self.n_jobs = n_jobs


    def fit(self, X, y, out_dir=None):
        if out_dir is not None:
            out_dir = Path(out_dir)

        X = np.array(X, dtype=bool)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y don't have the same number of years.")
        
        if self.n_jobs == -1:
            self.n_threads_ = multiprocessing.cpu_count()
        else:
            self.n_threads_ = self.n_jobs

        numba.set_num_threads(self.n_threads_)

        # Set RNG
        self.seed_sequence_calibration_ = np.random.SeedSequence(self.random_state + 1)
        self.prngs_calibration_ = typed.List(
            [
                np.random.Generator(np.random.SFC64(x))
                for x in self.seed_sequence_calibration_.generate_state(self.n_iters)
            ]
        )

        # Set initial params
        self.years_ = y
        self.calibration_stats_ = self.calculate_initial_stats(X, y)

        current_grid = [
            su.generate_grid(*self.coef_range_diffusion, self.n_refinement_splits),
            su.generate_grid(*self.coef_range_breed, self.n_refinement_splits),
            su.generate_grid(*self.coef_range_spread, self.n_refinement_splits),
            su.generate_grid(*self.coef_range_slope, self.n_refinement_splits),
            su.generate_grid(*self.coef_range_road, self.n_refinement_splits),
        ]
        

        # self.param_grids_ = np.zeros(
        #     (self.n_refinement_iters, 5, self.n_refinement_splits)
        # )
        self.osm_ = {}

        for refinement_iter in range(self.n_refinement_iters):
            # # Prevent searching over the same grid
            # for grid in self.param_grids_:
            #     if (grid == current_grid).all():
            #         break

            # self.param_grids_[refinement_iter] = current_grid

            combs = np.array(list(itertools.product(*current_grid)), dtype=np.int32)

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
                years=self.years_,
                calibration_stats=self.calibration_stats_,
                prngs=self.prngs_calibration_,
            )

            for key, value in zip(combs, osm):
                self.osm_[tuple(key)] = value

            scores_sorted = list(
                sorted(self.osm_.items(), key=lambda x: x[1], reverse=True)
            )
            top_params = np.array(
                [x[0] for x in scores_sorted[:self.n_refinement_winners]]
            )

            new_param_grid = [None] * 5
            for i in range(5):
                c_min = top_params[:, i].min()
                c_max = top_params[:, i].max()
                new_param_grid[i] = su.get_new_range(
                    current_grid[i], c_min, c_max, self.n_refinement_splits
                )

            current_grid = new_param_grid

            if out_dir is not None:
                temp_df = pd.DataFrame(self.osm_.items(), columns=["params", "osm"])
                temp_df.to_csv(out_dir / f"stage_{refinement_iter}.csv", index=False)

        final_params = max(self.osm_, key=self.osm_.get)
        self.coef_diffusion_ = final_params[0]
        self.coef_breed_ = final_params[1]
        self.coef_spread_ = final_params[2]
        self.coef_slope_ = final_params[3]
        self.coef_road_ = final_params[4]

        return self

    def predict(self, X, nyears):
        self.seed_sequence_prediction_ = np.random.SeedSequence(self.random_state - 1)
        self.prngs_prediction_ = typed.List(
            [
                np.random.Generator(np.random.SFC64(x))
                for x in self.seed_sequence_prediction_.generate_state(self.n_iters)
            ]
        )

        # Perform simulation from start to end year
        grid_MC, records = sp.fill_montecarlo_grid(
            X0=X,
            nyears=nyears,
            n_iters=self.n_iters,
            grid_slope=self.grid_slope,
            grid_excluded=self.grid_excluded,
            grid_roads=self.grid_roads,
            grid_roads_dist=self.grid_roads_dist,
            grid_roads_i=self.grid_roads_i,
            grid_roads_j=self.grid_roads_j,
            coef_diffusion=self.coef_diffusion_,
            coef_breed=self.coef_breed_,
            coef_spread=self.coef_spread_,
            coef_slope=self.coef_slope_,
            coef_road=self.coef_road_,
            crit_slope=self.crit_slope,
            prngs=self.prngs_prediction_
        )
        return grid_MC, records


    def calculate_initial_stats(self, X, y):
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
            calibration_stats[i, idx_arr] = st.compute_stats(X[i], self.grid_slope)

        return calibration_stats


    def _more_tags(self):
        return {"X_types": ["3darray"]}
