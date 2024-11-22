import argparse
from clustering.optimize import BHO_Clustering

def main():
    """
    Main function to handle command-line arguments and execute the BHO_Clustering optimization.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Optimize clustering hyperparameters using Optuna.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save results and plots.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials (default: 100).")
    parser.add_argument("--save_plots", type=bool, default=False, help="Save optimization plots (default: False).")
    
    # Parse arguments
    args = parser.parse_args()

    # Instantiate and run the BHO_Clustering optimizer
    optimizer = BHO_Clustering(
        config_path=args.config_path,
        output_path=args.output_path,
        n_trials=args.n_trials,
        save_plots=args.save_plots
    )

    # Run optimization and save results
    optimizer.optimize()
    optimizer.save_results()

if __name__ == "__main__":
    main()
