import itertools

import numpy as np
import sleuth_sklearn.spread as sp
import sleuth_sklearn.stats as st
import sleuth_sklearn.utils as su

from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sleuth_sklearn.stats import Record, StatsVal


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
        random_state=None
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

        self.calibration_stats_ = [StatsVal() for year in y]
        for i, year in enumerate(y):
            stats_val = self.calibration_stats_[i]
            (
                stats_val.edges,
                stats_val.clusters,
                stats_val.pop,
                stats_val.xmean,
                stats_val.ymean,
                stats_val.slope,
                stats_val.rad,
                stats_val.mean_cluster_size,
                stats_val.percent_urban,
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

            combs = itertools.product(*param_grid.values())
            for c_diffusion, c_breed, c_spread, c_slope, c_road in combs:
                index = (c_diffusion, c_breed, c_spread, c_slope, c_road)

                # If score was already calculated
                if index in self.osm_:
                    continue

                grid_MC = np.zeros((nyears, nrows, ncols))

                records = []
                for year in y:
                    print(year)
                    record = Record(
                        year=year,
                        is_calibration=True,
                        diffusion=c_diffusion,
                        breed=c_breed,
                        spread=c_spread,
                        slope=c_slope,
                        road=c_road,
                    )
                    records.append(record)

                for i in range(self.n_iters):
                    self.grow(
                        grid_MC,
                        X[0],
                        coef_diffusion=c_diffusion,
                        coef_breed=c_breed,
                        coef_spread=c_spread,
                        coef_slope=c_slope,
                        coef_road=c_road,
                        records=records,
                    )

                for record in records:
                    record.compute_std()

                self.osm_[index] = self._evaluate_records(records)

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

    def _evaluate_records(self, records):
        # Calculate calibration statistics
        # The modified coefficients are extracted
        # from the last oberseved year average record.

        # The optimal SLEUTH metric is the product of:
        # compare, pop, edges, clusters, slope, x_mean, and y_mean
        # Extract mean records for urban years
        sim_means = [record.average for record in records]

        assert len(sim_means) == len(self.calibration_stats_)

        # Compare: ratio of final urbanizations at last control years
        final_pop_sim = sim_means[-1].pop
        final_pop_urb = self.calibration_stats_[-1].pop
        compare = min(final_pop_sim, final_pop_urb) / max(final_pop_sim, final_pop_urb)

        # Find regression coefficients, ignore seed year
        osm_metrics_names = ["pop", "edges", "clusters", "slope", "xmean", "ymean"]
        osm_metrics = []
        for metric in osm_metrics_names:
            # simulation skips seed year
            sim_vals = [getattr(s, metric) for s in sim_means[1:]]
            urb_vals = [getattr(s, metric) for s in self.calibration_stats_[1:]]
            r, _ = pearsonr(sim_vals, urb_vals)
            r = r**2
            osm_metrics.append(r)

        # Optimal metric
        osm = np.prod(osm_metrics) * compare
        return osm

    def grow(
        self,
        grid_MC,
        seed_grid,
        *,
        coef_diffusion,
        coef_breed,
        coef_spread,
        coef_slope,
        coef_road,
        records=None
    ):
        nyears = grid_MC.shape[0]

        # Initialize Z grid to the seed (first urban grid)
        # TODO: Zero grid instead of creating new one.
        # grd_Z = urban_grid[0].copy()
        grd_Z = seed_grid.copy()

        # Precalculate/reset slope weighs
        # This can change due to self-modification during growth.
        sweights = 1 - sp.slope_weight(self.grid_slope, coef_slope, self.crit_slope)

        for i in range(nyears):
            # Apply CA rules for current year
            sng, sdc, og, rt, num_growth_pix = sp.spread(
                grd_Z,
                self.grid_slope,
                self.grid_excluded,
                self.grid_roads,
                self.grid_roads_dist,
                self.grid_roads_i,
                self.grid_roads_j,
                coef_diffusion,
                coef_breed,
                coef_spread,
                coef_slope,
                coef_road,
                self.crit_slope,
                self.random_state_,
                sweights,
            )

            if records is not None:
                record = records[i]
                record.monte_carlo += 1

                # Send stats to current year (ints)
                record.this_year.sng = sng
                record.this_year.sdc = sdc
                record.this_year.og = og
                record.this_year.rt = rt
                record.this_year.num_growth_pix = num_growth_pix

                # Store coefficients
                record.this_year.diffusion = coef_diffusion
                record.this_year.spread = coef_spread
                record.this_year.breed = coef_breed
                record.this_year.slope_resistance = coef_slope
                record.this_year.road_gravity = coef_road

                # Compute stats
                (
                    record.this_year.edges,
                    record.this_year.clusters,
                    record.this_year.pop,
                    record.this_year.xmean,
                    record.this_year.ymean,
                    record.this_year.slope,
                    record.this_year.rad,
                    record.this_year.mean_cluster_size,
                    record.this_year.percent_urban,
                ) = sp.compute_stats(grd_Z, self.grid_slope)

                # Growth
                record.this_year.growth_rate = (
                    100.0 * num_growth_pix / record.this_year.pop
                )

                # Update mean and sum of squares
                record.update_mean_std()

            # Accumulate MC samples
            # TODO: avoid indexing making sure Z grid is at most 1.
            grid_MC[i][grd_Z > 0] += 1

    def _more_tags(self):
        return {"X_types": ["3darray"]}
