import argparse
import toml

import pandas as pd

from pathlib import Path
from sleuth_sklearn.estimator import SLEUTH
from sleuth_sklearn.utils import open_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "PATH",
        help="The path to the config file.",
        type=str
    )
    
    args = parser.parse_args()

    config_path = Path(args.PATH)
    with open(config_path, "r") as f:
        config = toml.load(f)

    ds = open_dataset(config["paths"]["data"])

    if "random_seed" in config["montecarlo"]:
        random_seed = config["montecarlo"]["random_seed"]
    else:
        random_seed = None
    
    model = SLEUTH(
        n_iters=config["montecarlo"]["iterations"],
        n_refinement_iters=config["refinement"]["iterations"],
        n_refinement_splits=config["refinement"]["splits"],
        n_refinement_winners=config["refinement"]["winners"],
        coef_range_breed=config["coefficients"]["breed"],
        coef_range_diffusion=config["coefficients"]["diffusion"],
        coef_range_road=config["coefficients"]["road"],
        coef_range_slope=config["coefficients"]["slope"],
        coef_range_spread=config["coefficients"]["spread"],
        grid_excluded=ds["excluded"].values,
        grid_roads=ds["roads"].values,
        grid_roads_dist=ds["dist"].values,
        grid_roads_i=ds["road_i"].values,
        grid_roads_j=ds["road_j"].values,
        grid_slope=ds["slope"].values,
        crit_slope=config["misc"]["critical_slope"],
        random_state=random_seed,
        n_jobs=config["multiprocessing"]["threads"]
    )

    if "start_year" in config["calibration"]:
        start_year = config["calibration"]["start_year"]
        
        if start_year not in ds["year"].values:
            raise Exception(f"Year {start_year} not in available observations.")
            
        wanted_years = [year for year in ds["year"].values if year >= start_year]
        
        X = ds["urban"].sel(year=wanted_years).values
        y = wanted_years
    else:
        X = ds["urban"].values
        y = ds["year"].values

    if "stage_dir" in config["calibration"]:
        out_dir = config["calibration"]["stage_dir"]
    else:
        out_dir = None

    model.fit(X, y, out_dir=out_dir)

    df = pd.DataFrame(model.osm_.items(), columns=["params", "osm"])
    
    df["diffusion"] = df["params"].str[0]
    df["breed"] = df["params"].str[1]
    df["spread"] = df["params"].str[2]
    df["slope"] = df["params"].str[3]
    df["road"] = df["params"].str[4]

    df = df[["diffusion", "breed", "spread", "slope", "road", "osm"]]

    df.to_csv(config["calibration"]["out_path"], index=False)