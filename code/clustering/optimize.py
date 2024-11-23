#!/usr/bin/env python3

import igraph as ig
import numpy as np
import optuna
import pandas as pd
import os
import time
import random
from typing import (
    Union, 
    Tuple, 
    Dict, 
    Any
)
import yaml 
import argparse

import platform
import psutil

from metrics import Metrics  
from algorithms import Algorithms
from utils import (
    setup_logger, 
    network_to_igraph_format
)

class BHO_Clustering:
    def __init__(self, 
                 config_path: Union[str, os.PathLike],
                 network_csv: Union[str, os.PathLike], 
                 output_path: Union[str, os.PathLike], 
                 study_name: str = "community_detection_optimization",
                 n_trials: int = 100
                 ) -> None:
        """
        Initialize the BHO_Clustering class.

        :param network_csv: Path to the CSV file representing the network.
        :param output_path: Path to save the study results and plots.
        :param study_name: Name of the Optuna study.
        :param n_trials: Number of optimization trials.
        :param save_plots: Boolean indicating whether to save optimization plots.
        """
        # Load configuration and initialize variables
        config = self._load_config(config_path=config_path)
        self.graph: ig.Graph = network_to_igraph_format(network_csv=network_csv)
        self.selected_algorithm: str = config["algorithm"]
        self.hyperparameters = {  # Map of hyperparameter names to their configs
            (param["name"], param["type"]): param for param in config.get("parameters", [])
        }
        self.output_path: Union[str, os.PathLike] = output_path
        self.study_name: str = study_name
        self.n_trials: int = n_trials
        self.study: optuna.Study = None  # Will hold the Optuna study object after optimization

        # Setup logger
        os.makedirs('./logs', exist_ok=True)
        self.logger = setup_logger(
            name="Bayesian_Hyperparameter_Optimization_Clustering_Networks",
            log_file="logs/bho_optimization.log"
        )

        # Log study details
        self.logger.info(f"Study '{self.study_name}' initialized with the following parameters:")
        self.logger.info(f"  Config Path: {config_path}")
        self.logger.info(f"  Network CSV: {network_csv}")
        self.logger.info(f"  Output Path: {output_path}")
        self.logger.info(f"  Algorithm: {self.selected_algorithm}")
        self.logger.info(f"  Number of Trials: {self.n_trials}")

        # Log hyperparameter details
        self.logger.info("Hyperparameter Configuration:")
        for (param_name, param_type), param_config in self.hyperparameters.items():
            self.logger.info(f"  {param_name} ({param_type}): {param_config}")

        # Log execution environment details
        self._log_system_info()

    def _log_system_info(self):
        """
        Logs system information where the script is being executed.
        """
        self.logger.info("Execution Environment:")
        self.logger.info(f"  OS: {platform.system()} {platform.release()} ({platform.version()})")
        self.logger.info(f"  Python Version: {platform.python_version()}")
        self.logger.info(f"  Processor: {platform.processor()}")
        self.logger.info(f"  CPU Cores: {psutil.cpu_count(logical=True)}")
        self.logger.info(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        self.logger.info(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the YAML configuration file.

        :param config_path: Path to the YAML configuration file.
        :return: A dictionary containing the algorithm and parameters.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            self.logger.critical("Error loading hyperparameter configuration. Check example at code/clustering/configs.")

    def _extract_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Extracts hyperparameters from the search space using the Optuna trial object.

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial object used for suggesting hyperparameters.

        Returns
        -------
        Dict[str, Any]
            A dictionary of hyperparameters suggested by the Optuna trial.

        Raises
        ------
        ValueError
            If an unknown function type is encountered while suggesting hyperparameters.

        Notes
        -----
        This method uses the Optuna trial object to suggest hyperparameter values based
        on the configured search space. It supports different types of distributions
        (e.g., 'loguniform', 'uniform', 'int', 'categorical') and logs the extraction process.
        """
        final_hyperparam: Dict[str, Any] = {}

        for (hyperparam, func_type), param_config in self.hyperparameters.items():
            low = param_config.get("min") or param_config.get("low")
            high = param_config.get("max") or param_config.get("high")
            if func_type == "log_float":
                final_hyperparam[hyperparam] = trial.suggest_float(hyperparam, low, high, log=True)
            if func_type == "float":
                final_hyperparam[hyperparam] = trial.suggest_float(hyperparam, low, high, log=False)
            elif func_type == "uniform":
                final_hyperparam[hyperparam] = trial.suggest_float(hyperparam, low, high)
            elif func_type == "int":
                final_hyperparam[hyperparam] = trial.suggest_int(hyperparam, int(low), int(high))
            elif func_type == "categorical":
                choices = param_config["choices"]
                final_hyperparam[hyperparam] = trial.suggest_categorical(hyperparam, choices)
            else:
                self.logger.critical(f"Unknown function type {func_type} for hyperparameter {hyperparam}. Choose from log_float | float | uniform | int | categorical.")
                raise ValueError(f"Unknown function type {func_type} for hyperparameter {hyperparam}")
        
        return final_hyperparam

    def _train(self, 
               trial: optuna.Trial
               ) -> Tuple[float, float]:
        """
        Train the model and compute the metrics for optimization.

        :param trial: An Optuna trial object.
        :return: A tuple of (modularity_score, fes_score).
        """
        # Set a seed for reproducibility
        seed = trial.number  # Use trial number as the seed
        np.random.seed(seed)
        random.seed(seed)
        trial.set_user_attr('seed', seed)

        # Record the start time
        start_time = time.time()

        # Extract hyperparameters
        hyperparams = self._extract_hyperparameters(trial)

        # Run the selected clustering algorithm
        if self.selected_algorithm == "multilevel":
            clusters = Algorithms.multilevel_clustering(graph=self.graph, **hyperparams)
        elif self.selected_algorithm == "walktrap":
            clusters = self.graph.community_walktrap(**hyperparams).as_clustering()
        elif self.selected_algorithm == "leiden":
            clusters = self.graph.community_leiden(**hyperparams)
        else:
            raise ValueError(f"Unsupported algorithm: {self.selected_algorithm}")

        # Convert clusters to list of lists
        cluster_list = [cluster for cluster in clusters]

        # Compute modularity
        modularity_score = Metrics.modularity(graph=self.graph, clusters=cluster_list, logger=self.logger)

        # Compute functional enrichment score
        fes_score = Metrics.functional_enrichment_score(graph=self.graph, clusters=cluster_list, logger=self.logger)

        # Calculate execution time
        execution_time = time.time() - start_time
        trial.set_user_attr('execution_time', execution_time)

        self.logger.info(f"Training for trial {trial.number} successfully finished in {execution_time :.2f} seconds.")

        return modularity_score, fes_score

    def optimize(self) -> None:
        """
        Optimize the give clustering algorithm with TPE Sampler for both metrics. 

        References
        --------------------------------
        (1) OZAKI, Yoshihiko, et al. 
            Multiobjective tree-structured Parzen estimator. 
            Journal of Artificial Intelligence Research, 2022, vol. 73, p. 1209-1250.
        """
        # Create pruner and sampler
        pruner = optuna.pruners.MedianPruner()
        sampler = optuna.samplers.TPESampler()

        # Set up storage (database) for saving results
        storage: str = f'sqlite:///{self.output_path}/{self.study_name}.db'

        # Create the study
        self.study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=self.study_name,
            load_if_exists=True
        )

        # Execute the optimization
        self.study.optimize(self._train, n_trials=self.n_trials)

    def save_results(self) -> None:
        """
        Save the optimization results to a CSV file.
        """
        if self.study is None:
            raise ValueError("No study found. Please run optimize() first.")

        # Create a DataFrame from the study's trials
        df: pd.DataFrame = self.study.trials_dataframe(attrs=('number', 'values', 'params', 'state', 'user_attrs'))

        # Save the DataFrame to a CSV file
        csv_path: Union[str, os.PathLike] = os.path.join(self.output_path, 'optuna_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

def main():
    """
    Main function to handle command-line arguments and execute the BHO_Clustering optimization.

    Example:
    python code/clustering/optimize.py --config_path code/clustering/configs/multilevel.yaml --network_csv code/data/network.tsv --study_name try --output_path results --n_trials 2 --save_plots True
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Optimize clustering hyperparameters using Optuna.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--network_csv", type=str, required=True, help="Path to the CSV network file.")
    parser.add_argument("--study_name", type=str, required=True, help="Name of the study.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save results and plots.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials (default: 100).")

    # Parse arguments
    args = parser.parse_args()

    # Instantiate and run the BHO_Clustering optimizer
    optimizer = BHO_Clustering(
        config_path=args.config_path,
        network_csv=args.network_csv,
        output_path=args.output_path,
        study_name=args.study_name,
        n_trials=args.n_trials,
    )

    # Run optimization and save results
    optimizer.optimize()
    optimizer.save_results()

if __name__ == "__main__":
    main()