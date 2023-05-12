import numpy as np
import sleuth_sklearn.spread as sp
import sleuth_sklearn.stats as st

from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sleuth_sklearn.stats import Record, StatsVal


class SLEUTH(BaseEstimator):
    def __init__(
        self,
        *,
        total_mc=10,
        grid_slope=np.empty(0),
        grid_excluded=np.empty(0),
        grid_roads=np.empty(0),
        grid_roads_dist=np.empty(0),
        grid_roads_i=np.empty(0),
        grid_roads_j=np.empty(0),
        coef_diffusion=0,
        coef_breed=0,
        coef_spread=0,
        coef_slope=0,
        coef_road=0,
        crit_slope=0,
        random_state=None
    ):
        self.coef_diffusion = coef_diffusion
        self.coef_breed = coef_breed
        self.coef_spread = coef_spread
        self.coef_slope = coef_slope
        self.coef_road = coef_road

        self.grid_slope = grid_slope
        self.grid_excluded = grid_excluded
        self.grid_roads = grid_roads
        self.grid_roads_dist = grid_roads_dist
        self.grid_roads_i = grid_roads_i
        self.grid_roads_j = grid_roads_j

        self.crit_slope = crit_slope

        self.total_mc = total_mc
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError

        # Set PRNG
        self.random_state_ = np.random.Generator(np.random.SFC64(self.random_state))

        nyears = y[-1] - y[0] + 1
        _, nrows, ncols = X.shape
        grid_MC = np.zeros((nyears, nrows, ncols))

        self.records_ = []
        for year in range(y[0], y[-1] + 1):
            record = Record(
                year=year,
                is_calibration=year in y,
                diffusion=self.coef_diffusion,
                breed=self.coef_breed,
                spread=self.coef_spread,
                slope=self.coef_slope,
                road=self.coef_road,
            )
            self.records_.append(record)

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

        for i in range(self.total_mc):
            self.grow(grid_MC, X[0], write_records=True)

        for record in self.records_:
            record.compute_std()

        return self

    def predict(self, X, nyears):
        # Create monte carlo grid to accumulate probability of urbanization
        # one grid per simulated year
        self.random_state_ = np.random.Generator(np.random.SFC64(self.random_state))

        nrows, ncols = X.shape
        grid_MC = np.zeros((nyears, nrows, ncols))

        # Perform simulation from start to end year
        for i in range(self.total_mc):
            self.grow(grid_MC, X, write_records=False)

        grid_MC /= self.total_mc
        return grid_MC

    def score(self, X, y):
        # Calculate calibration statistics
        # The modified coefficients are extracted
        # from the last oberseved year average record.

        # The optimal SLEUTH metric is the product of:
        # compare, pop, edges, clusters, slope, x_mean, and y_mean
        # Extract mean records for urban years
        sim_means = [
            record.average for record in self.records_ if record.is_calibration
        ]
        sim_means = sim_means[1:]

        # Compare: ratio of final urbanizations at last control years
        final_pop_sim = sim_means[-1].pop
        final_pop_urb = self.calibration_stats_[-1].pop
        compare = min(final_pop_sim, final_pop_urb) / max(final_pop_sim, final_pop_urb)

        # Find regression coefficients, ignore seed year
        osm_metrics_names = ["pop", "edges", "clusters", "slope", "xmean", "ymean"]
        osm_metrics = []
        for metric in osm_metrics_names:
            # simulation skips seed year
            sim_vals = [getattr(s, metric) for s in sim_means]
            urb_vals = [getattr(s, metric) for s in self.calibration_stats_[1:]]
            r, _ = pearsonr(sim_vals, urb_vals)
            r = r**2
            osm_metrics.append(r)

        # Optimal metric
        osm = np.prod(osm_metrics) * compare
        return osm

    def grow(self, grid_MC, seed_grid, write_records=False):
        nyears = grid_MC.shape[0]

        # Initialize Z grid to the seed (first urban grid)
        # TODO: Zero grid instead of creating new one.
        # grd_Z = urban_grid[0].copy()
        grd_Z = seed_grid.copy()

        # Precalculate/reset slope weighs
        # This can change due to self-modification during growth.
        sweights = 1 - sp.slope_weight(
            self.grid_slope, self.coef_slope, self.crit_slope
        )

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
                self.coef_diffusion,
                self.coef_breed,
                self.coef_spread,
                self.coef_slope,
                self.coef_road,
                self.crit_slope,
                self.random_state_,
                sweights,
            )

            if write_records:
                record = self.records_[i]
                record.monte_carlo += 1

                # Send stats to current year (ints)
                record.this_year.sng = sng
                record.this_year.sdc = sdc
                record.this_year.og = og
                record.this_year.rt = rt
                record.this_year.num_growth_pix = num_growth_pix

                # Store coefficients
                record.this_year.diffusion = self.coef_diffusion
                record.this_year.spread = self.coef_spread
                record.this_year.breed = self.coef_breed
                record.this_year.slope_resistance = self.coef_slope
                record.this_year.road_gravity = self.coef_road

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
