import igraph as ig
import numpy as np
import optuna
import pandas as pd
import os
import time
import random
from typing import Union, Tuple, Dict

from metrics import Metrics  
from utils import setup_logger, network_to_igraph_format
from algorithms import Algorithms

class BHO_Clustering:
    def __init__(self, 
                 network_csv: Union[str, os.PathLike], 
                 output_path:  Union[str, os.PathLike], 
                 n_trials: int = 100, 
                 save_plots: bool = False
                 ) -> None:
        """
        Initialize the BHO_Clustering class.

        :param network_csv: Path to the CSV file representing the network.
        :param output_path: Path to save the study results and plots.
        :param n_trials: Number of optimization trials.
        :param save_plots: Boolean indicating whether to save optimization plots.
        """
        self.graph: ig.Graph = network_to_igraph_format(network_csv=network_csv)
        self.output_path: Union[str, os.PathLike] = output_path
        self.n_trials: int = n_trials
        self.save_plots: bool = save_plots
        self.study: optuna.Study = None  # Will hold the Optuna study object after optimization

    def optimize(self) -> None:
        """
        Set up the Optuna study and execute the optimization process.
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

        # Sample hyperparameters
        resolution = trial.suggest_float('resolution', low=0.5, high=2.0)

        # Run the community detection algorithm
        clusters = Algorithms.clustering(
            graph=self.graph,
            resolution=resolution
        )

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
