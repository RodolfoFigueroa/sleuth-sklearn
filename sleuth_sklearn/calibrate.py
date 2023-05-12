import argparse
import os

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sleuth_sklearn.estimator import SLEUTH
from sleuth_sklearn.utils import open_dataset, get_new_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="Input file with all grids.",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--stage",
        help="Initial stage. If greater than one, the previous stage files must exist.",
        type=int,
        choices=[1, 2, 3],
        required=False,
        default=1,
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="The directory where to store the generated stage files, or look for them if --stage is greater than 1.",
        type=str,
        required=False,
        default="./",
    )
    parser.add_argument(
        "-i",
        "--iters",
        help="Number of Monte Carlo iterations to run.",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        help="Seed to use for the PRNG.",
        type=int,
        required=False,
        default=42,
    )
    parser.add_argument(
        "-c",
        "--critical-slope",
        help="The critical slope coefficient.",
        type=float,
        required=False,
        default=50,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="The level of verbosity to use for GridSearchCV.",
        type=int,
        required=False,
        default=0,
    )

    args = parser.parse_args()

    stage_dir = Path(args.directory)
    ds = open_dataset(args.filename)

    wanted_years = [year for year in ds.year.values if year >= 2000]
    urban_grids = ds.sel(year=wanted_years)["urban"].values

    param_grid_base = dict(
        coef_diffusion=[1, 25, 50, 75, 100],
        coef_breed=[1, 25, 50, 75, 100],
        coef_spread=[1, 25, 50, 75, 100],
        coef_slope=[1, 25, 50, 75, 100],
        coef_road=[1, 25, 50, 75, 100],
    )

    if args.stage == 1:
        df = None
        stages = [1, 2, 3]
        param_grid = param_grid_base
    else:
        path = stage_dir / f"./stage_{args.stage - 1}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError("Stage file not found.")
        df = pd.read_csv(path)

        if args.stage == 2:
            stages = [2, 3]
            param_grid = get_new_params(param_grid_base, df)

    for stage in stages:
        print(param_grid)

        model = SLEUTH(
            total_mc=args.iters,
            grid_slope=ds["slope"].values,
            grid_excluded=ds["excluded"].values,
            grid_roads=ds["roads"].values,
            grid_roads_i=ds["road_i"].values,
            grid_roads_j=ds["road_j"].values,
            grid_roads_dist=ds["dist"].values,
            random_state=args.random_seed,
            crit_slope=args.critical_slope,
        )

        gs = GridSearchCV(
            estimator=model,
            cv=[(slice(None), slice(None))],
            param_grid=param_grid,
            n_jobs=-1,
            verbose=args.verbose,
        )

        res = gs.fit(urban_grids, wanted_years)
        df = pd.DataFrame(res.cv_results_)
        df.to_csv(stage_dir / f"./stage_{stage}.csv", index=False)

        param_grid = get_new_params(param_grid, df)
