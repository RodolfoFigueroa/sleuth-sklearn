import itertools
import multiprocessing
import numba

import numpy as np
import pandas as pd
import sleuth_sklearn.spread as sp
import sleuth_sklearn.stats as st
import sleuth_sklearn.utils as su
import xarray as xr

from numba import njit, typed, types
from pathlib import Path
from sklearn.base import BaseEstimator
from sleuth_sklearn.indices import J


def reshape_nc(arr, current_grid, all_years):
    reshaped = arr.reshape(*[len(g) for g in current_grid], len(all_years), J.TOTAL_SIZE)
    out = xr.DataArray(
        data=reshaped,
        dims=["diffusion", "breed", "spread", "slope", "road", "year", "stat"],
        coords=dict(
            diffusion=current_grid[0],
            breed=current_grid[1],
            spread=current_grid[2],
            slope=current_grid[3],
            road=current_grid[4],
            year=all_years,
            stat=J.NAMES
        )
    )
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
        n_jobs=-1,
        log_dir=None,
        verbose=0
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
        
        self.log_dir = log_dir
        self.verbose = verbose


    def fit(self, X, y):
        if self.verbose > 0:
            if self.log_dir is None:
                raise IOError("verbose > 0 but log_dir wasn't set in constructor.")
            else:
                self.log_dir_path_ = Path(self.log_dir)
                self.log_dir_path_.mkdir(exist_ok=True, parents=True)


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
        if self.random_state is not None:
            self.seed_sequence_calibration_ = np.random.SeedSequence(self.random_state + 1)
        else:
            self.seed_sequence_calibration_ = np.random.SeedSequence()
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

            nyears = self.years_[-1] - self.years_[0] + 1
            osm = np.empty(len(combs), dtype=np.float64)
            records_mean = np.empty((len(combs), nyears, J.TOTAL_SIZE), dtype=np.float64)
            records_std = np.empty((len(combs), nyears, J.TOTAL_SIZE), dtype=np.float64)

            for i in range(len(combs)):
                c_diffusion, c_breed, c_spread, c_slope, c_road = combs[i]
                index = (c_diffusion, c_breed, c_spread, c_slope, c_road)
                print(index)

                if self.verbose <= 1:
                    _, records_mean_partial, records_std_partial = sp.fill_montecarlo_grid(
                        X[0],
                        nyears=nyears,
                        n_iters=self.n_iters,
                        grid_slope=self.grid_slope,
                        grid_excluded=self.grid_excluded,
                        grid_roads=self.grid_roads,
                        grid_roads_dist=self.grid_roads_dist,
                        grid_roads_i=self.grid_roads_i,
                        grid_roads_j=self.grid_roads_j,
                        coef_diffusion=c_diffusion,
                        coef_breed=c_breed,
                        coef_spread=c_spread,
                        coef_slope=c_slope,
                        coef_road=c_road,
                        crit_slope=self.crit_slope,
                        prngs=self.prngs_calibration_,
                    )
                else:
                    _, records_mean_partial, records_std_partial = sp.fill_montecarlo_grid_io(
                        X[0],
                        nyears=nyears,
                        n_iters=self.n_iters,
                        grid_slope=self.grid_slope,
                        grid_excluded=self.grid_excluded,
                        grid_roads=self.grid_roads,
                        grid_roads_dist=self.grid_roads_dist,
                        grid_roads_i=self.grid_roads_i,
                        grid_roads_j=self.grid_roads_j,
                        coef_diffusion=c_diffusion,
                        coef_breed=c_breed,
                        coef_spread=c_spread,
                        coef_slope=c_slope,
                        coef_road=c_road,
                        crit_slope=self.crit_slope,
                        prngs=self.prngs_calibration_,
                        log_dir=self.log_dir_path_
                    )


                res = st.evaluate_records(records_mean_partial, self.years_, self.calibration_stats_)
                print(res)

                osm[i] = res
                records_mean[i] = records_mean_partial
                records_std[i] = records_std_partial

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

            #= Save results =#
            param_arr = np.array(list(self.osm_.keys()))

            if self.verbose > 0:
                out_df = pd.DataFrame(
                    zip(*param_arr.T, self.osm_.values()),
                    columns=["diffusion", "breed", "spread", "slope", "road", "osm"]
                )
                out_df.to_csv(self.log_dir_path_ / f"stage_{refinement_iter}.csv", index=False)
                
                start_year = self.years_[0]
                end_year = self.years_[-1]
                all_years = list(range(start_year, end_year + 1))

                means_reshaped = reshape_nc(records_mean, current_grid, all_years)
                stds_reshaped = reshape_nc(records_std, current_grid, all_years)

                means_reshaped.to_netcdf(self.log_dir_path / f"means_{refinement_iter}.nc")
                stds_reshaped.to_netcdf(self.log_dir_path / f"stds_{refinement_iter}.nc")


            #= Update param grid =#
            current_grid = new_param_grid

        final_params = max(self.osm_, key=self.osm_.get)
        self.coef_diffusion_ = final_params[0]
        self.coef_breed_ = final_params[1]
        self.coef_spread_ = final_params[2]
        self.coef_slope_ = final_params[3]
        self.coef_road_ = final_params[4]

        return self

    def predict(self, X, nyears):
        if self.random_state is not None:
            self.seed_sequence_prediction_ = np.random.SeedSequence(self.random_state - 1)
        else:
            self.seed_sequence_prediction_ = np.random.SeedSequence()
        self.prngs_prediction_ = typed.List(
            [
                np.random.Generator(np.random.SFC64(x))
                for x in self.seed_sequence_prediction_.generate_state(self.n_iters)
            ]
        )

        # Perform simulation from start to end year
        grid_MC, records_mean, records_std = sp.fill_montecarlo_grid(
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
        return grid_MC, records_mean, records_std


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
