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
    Any, 
    List
)
import yaml 

from metrics import Metrics  
from algorithms import Algorithms
from utils import (
    setup_logger, 
    network_to_igraph_format
)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    :param config_path: Path to the YAML configuration file.
    :return: A dictionary containing the algorithm and parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class BHO_Clustering:
    def __init__(self, 
                 config_path: Union[str, os.PathLike],
                 network_csv: Union[str, os.PathLike], 
                 output_path:  Union[str, os.PathLike], 
                 study_name: int = "community_detection_optimization",
                 n_trials: int = 100, 
                 storage_name: int = "optuna_study",
                 save_plots: bool = False
                 ) -> None:
        """
        Initialize the BHO_Clustering class.

        :param network_csv: Path to the CSV file representing the network.
        :param output_path: Path to save the study results and plots.
        :param n_trials: Number of optimization trials.
        :param save_plots: Boolean indicating whether to save optimization plots.
        """
        config = load_config(config_path=config_path)
        self.graph: ig.Graph = network_to_igraph_format(network_csv=network_csv)
        self.selected_algorithm: str = config["algorithm"]
        self.hyperparameters = {  # Map of hyperparameter names to their configs
            (param["name"], param["type"]): param for param in config.get("parameters", [])
        }
        self.output_path: Union[str, os.PathLike] = output_path
        self.study_name: int = study_name
        self.n_trials: int = n_trials
        self.save_plots: bool = save_plots
        self.storage_name: int = storage_name
        self.study: optuna.Study = None  # Will hold the Optuna study object after optimization

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
            if func_type == "loguniform":
                final_hyperparam[hyperparam] = trial.suggest_float(hyperparam, low, high, log=True)
            elif func_type == "uniform":
                final_hyperparam[hyperparam] = trial.suggest_float(hyperparam, low, high)
            elif func_type == "int":
                final_hyperparam[hyperparam] = trial.suggest_int(hyperparam, int(low), int(high))
            elif func_type == "categorical":
                choices = param_config["choices"]
                final_hyperparam[hyperparam] = trial.suggest_categorical(hyperparam, choices)
            else:
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
        hyperparams = self.extract_hyperparameters(trial)

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
        modularity_score = Metrics.modularity(self.graph, cluster_list)

        # Compute functional enrichment score
        # TODO: Gonzalo must adjust the functional_enrichment_score function to accept cluster_list 
        fes_score = Metrics.functional_enrichment_score(cluster_list)

        # Calculate execution time
        execution_time = time.time() - start_time
        trial.set_user_attr('execution_time', execution_time)

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
        storage: str = f'sqlite:///{self.output_path}/optuna_results.db'

        # Create the study
        self.study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name='community_detection_optimization',
            load_if_exists=True
        )

        # Execute the optimization
        self.study.optimize(self._train, n_trials=self.n_trials)

        # Save plots if requested
        if self.save_plots:
            self._save_plots()

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

    def _save_plots(self):
        """
        Generate and save optimization plots.
        """
        import optuna.visualization as vis

        # Pareto front plot
        pareto_front_fig = vis.plot_pareto_front(
            self.study,
            target_names=['Modularity', 'FES']
        )
        pareto_front_fig.write_image(f'{self.output_path}/pareto_front.png')

        # Parameter importance plot
        param_importance_fig = vis.plot_param_importances(self.study)
        param_importance_fig.write_image(f'{self.output_path}/param_importances.png')

        print("Plots saved to output path.")
