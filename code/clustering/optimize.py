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
from utils.misc import (
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
        Parameters
        ----------
        config_path : Union[str, os.PathLike]
            Path to the YAML configuration file containing the clustering algorithm and hyperparameter details.
        network_csv : Union[str, os.PathLike]
            Path to the CSV file representing the protein-protein interaction network.
        output_path : Union[str, os.PathLike]
            Directory to save the results and plots of the optimization process.
        study_name : str, optional
            Name of the Optuna study (default is "community_detection_optimization").
        n_trials : int, optional
            Number of optimization trials to run (default is 100).

        Raises
        ------
        FileNotFoundError
            If the network CSV file or configuration YAML file does not exist.

        Notes
        -----
        Initializes and configures the Bayesian Hyperparameter Optimization for clustering
        algorithms. Logs study details and the execution environment.
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

        Notes
        -----
        This method gathers and logs details about the operating system, processor,
        memory, and Python version. Useful for reproducibility and debugging.
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
        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the algorithm and hyperparameter details.

        Raises
        ------
        FileNotFoundError
            If the YAML file cannot be found.
        yaml.YAMLError
            If the YAML file cannot be parsed.
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
        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial object used for suggesting hyperparameters.

        Returns
        -------
        Tuple[float, float]
            A tuple containing two metrics: modularity score and functional enrichment score.

        Raises
        ------
        ValueError
            If the selected algorithm is not supported.

        Notes
        -----
        This method performs clustering using the specified algorithm and evaluates
        the results using modularity and functional enrichment scores.
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
            clusters = Algorithms.multilevel_clustering(graph=self.graph, logger=self.logger, **hyperparams)
        elif self.selected_algorithm == "walktrap":
            clusters = Algorithms.walktrap_clustering(graph=self.graph, logger=self.logger, **hyperparams)
        elif self.selected_algorithm == "leiden":
            clusters = Algorithms.leiden_clustering(graph=self.graph, logger=self.logger, **hyperparams)
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
        Runs the optimization process for the selected clustering algorithm.

        Notes
        -----
        This method uses the Tree-structured Parzen Estimator (TPE) for sampling and the
        Hyperband pruning strategy for efficient optimization of multiple objectives.

        References
        ----------
        (1) OZAKI, Yoshihiko, et al. 
            Multiobjective tree-structured Parzen estimator. 
            Journal of Artificial Intelligence Research, 2022, vol. 73, p. 1209-1250.
        (2) LI, Lisha, et al. 
            Hyperband: A novel bandit-based approach to hyperparameter optimization. 
            Journal of Machine Learning Research, 2018, vol. 18, no 185, p. 1-52.
        """

        # Dynamically allocates resources (trials) to promising candidates based on intermediate results.
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=5, 
            max_resource=100,
            reduction_factor=3 
        )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=20, # Determines the number of random trials before the TPE model is built
            n_ei_candidates=64, # Number of candidates taken into consideration when calculating expected improvement
            multivariate=True, # Sampling hyperparameters considering correlations between them
            seed=42  # For reproducibility
        )


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
        Saves the optimization results to a CSV file.

        Notes
        -----
        The results include trial numbers, parameter values, objective values, and
        other trial attributes. This method ensures reproducibility by storing all
        relevant data.

        Raises
        ------
        ValueError
            If the `optimize` method has not been executed and no study exists.
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
    Handles command-line arguments and executes the BHO_Clustering optimization process.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    network_csv : str
        Path to the CSV file representing the protein-protein interaction network.
    study_name : str
        Name of the Optuna study.
    output_path : str
        Directory to save the results and plots of the optimization process.
    n_trials : int, optional
        Number of optimization trials to run (default is 100).

    Notes
    -----
    This function is the entry point for the script and handles argument parsing
    and execution.

    Example
    -------
    python code/clustering/optimize.py \
        --config_path code/clustering/configs/multilevel.yaml \
        --network_csv code/data/network.tsv \
        --study_name multilevel_optimization \
        --output_path results \
        --n_trials 100
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
        n_trials=args.n_trials
    )

    # Run optimization and save results
    optimizer.optimize()
    optimizer.save_results()

if __name__ == "__main__":
    main()