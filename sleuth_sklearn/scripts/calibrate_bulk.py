import toml

from argparse import ArgumentParser
from pathlib import Path
from sleuth_sklearn.estimator import SLEUTH

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "PATH",
        help="The path to the configuration file.",
        type=str
    )
    args = parser.parse_args()
    
    with open(args.path, "r") as f:
        config = toml.load(f)

    model = SLEUTH(
        n_iters=config["montecarlo"]["iterations"],
        n_refinement_iters=config["refinement"]["iterations"],
        n_refinement_winners=config["refinement"]["winners"],
        n_refinement_splits=config["refinement"]["splits"],
        coef_range_diffusion=config["coefficients"]["diffusion"],
        coef_range_breed=config["coefficients"]["breed"],
        coef_range_spread=config["coefficients"]["spread"],
        coef_range_slope=config["coefficients"]["slope"],
        coef_range_road=config["coefficients"]["road"],
        
    )