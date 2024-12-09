#!/usr/bin/env python3

import logging
import os
import pandas as pd
import gseapy as gp
from typing import Any, List, Dict, Union
import igraph

VERBOSE: bool = os.environ.get("VERBOSE", "0") == "1"


class FunctionalAnalysis:
    """
    The FunctionalAnalysis class provides methods for performing functional enrichment analysis on gene clusters.

    This class uses a graph structure (igraph.Graph) to store clusters and genes as nodes and integrates with the Enrichr 
    service for enrichment analysis. It supports saving the analysis results to a CSV file and filtering the results based 
    on various criteria such as adjusted p-value, Combined Score, or overlap percentage.

    Key Features:
    - Perform functional enrichment analysis for clusters defined in a JSON structure.
    - Save enrichment analysis results to a specified CSV file.
    - Filter enrichment results based on adjusted p-value, Combined Score, or overlap percentage.
    - Verbose logging support for detailed execution tracking.

    Methods:
    - __init__(graph): Initializes the class with a graph containing the clusters.
    - perform_analysis(clustering_data, output_file, algorithm): Performs enrichment analysis on clusters and saves results.
    - filter_results(input_file, output_file, p_value_threshold, combined_score_min, overlap_percentage_min): Filters and saves results.
    """
    
    def __init__(self, graph: igraph.Graph, logger: logging.Logger):
        """
        Initializes the class with a graph containing the clusters.

        :param graph: Graph object that already includes the clusters and genes as nodes.
        """
        self.logger: logging.Logger = logger
        if not isinstance(graph, igraph.Graph):
            self.logger.error("The 'graph' argument must be an object of type igraph.Graph.")
            raise
        self.graph = graph

    @staticmethod
    def _perform_enrichment_analysis(self, genes: List[str]) -> pd.DataFrame:
        """
        Performs a functional enrichment analysis for a list of genes.

        :param genes: List of genes for which to perform the enrichment analysis.
        :return: DataFrame with the results of the functional analysis.
        """

        try:
            if not genes:
                self.logger.error("The genes list is empty.")
                raise

            # Functional analysis using Enrichr
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets="GO_Biological_Process_2021",  # Functional database
                organism="Human",                        # Organism
                outdir=None,                             # Do not save results to disk.
                no_plot=True,                            # Do not generate plots.
            )

            # Return results if there is data
            if enr.results.empty:
                return (
                    pd.DataFrame()
                )  # Return an empty DataFrame if there are no results.

            return enr.results

        except Exception as e:
            self.logger.error(f"Error in _perform_enrichment_analysis for genes {genes}: {e}")
            return pd.DataFrame()

    def perform_analysis(
        self, clustering_data: Dict[str, Any], output_file: str, algorithm: str = None
    ):
        """
        Performs functional analysis for clusters extracted from a JSON and saves the results to a CSV file.

        :param clustering_data: Dictionary with the clustering structure.
        :param output_file: Path to the CSV file where the results will be saved.
        :param algorithm: Name of the algorithm for which to perform the analysis. (Optional)
        """

        try:
            if not isinstance(clustering_data, dict):
                self.logger.error("The 'clustering_data' argument must be a dictionary.")
                raise

            all_results = []

            # Filter by algorithm if specified
            if algorithm:
                if algorithm not in clustering_data:
                    self.logger.error(f"The algorithm '{algorithm}' is not present in the clustering data.")
                    raise
                clustering_data = {algorithm: clustering_data[algorithm]}

            for key, clusters in clustering_data.items():
                for cluster_id, cluster_info in clusters.items():
                    genes = cluster_info.get("Genes", [])

                    if VERBOSE:
                        print(f"Performing functional analysis for {key} - {cluster_id}...")
                    enrichment_results = self._perform_enrichment_analysis(genes)

                    if enrichment_results.empty:
                        if VERBOSE:
                            print(
                                f"No enrichment results found for {key} - {cluster_id}."
                            )
                        continue

                    enrichment_results["Algorithm"] = key
                    enrichment_results["Cluster"] = cluster_id
                    all_results.append(enrichment_results)

            if all_results:
                final_results = pd.concat(all_results, ignore_index=True).drop(
                    columns=["Old P-value", "Old Adjusted P-value"], errors="ignore"
                )
                final_results.to_csv(output_file, index=False)
                if VERBOSE:
                    print(f"Results saved in {output_file}.")
            else:
                if VERBOSE:
                    print("No significant results found for the clusters.")

        except Exception as e:
            self.logger.error(f"Error in perform_analysis: {e}")

    def filter_results(
        self,
        input_file: str,
        output_file: str,
        p_value_threshold: float = None,
        combined_score_min: float = None,
        overlap_percentage_min: float = None,
    ):
        """
        Filters the functional analysis results based on adjusted p-value, Combined Score, or Overlap percentage
        (at least one filtering criterion must be provided).

        :param input_file: Path to the CSV file with the functional analysis results.
        :param output_file: Path to the CSV file where the filtered results will be saved.
        :param p_value_threshold: Maximum threshold for the adjusted p-value. (Optional)
        :param combined_score_min: Minimum threshold for the Combined Score. (Optional)
        :param overlap_percentage_min: Minimum threshold for the Overlap percentage. (Optional)
        """

        try:
            # Load the input data
            data = pd.read_csv(input_file)

            # Validate that at least one filter criterion is provided
            if (
                p_value_threshold is None
                and combined_score_min is None
                and overlap_percentage_min is None
            ):
                self.logger.error("You must provide at least one filtering criterion (p_value_threshold, combined_score_min, or overlap_percentage_min).")
                raise

            # Calculate the Overlap Percentage if needed
            if overlap_percentage_min is not None:
                data["Overlap_Percentage"] = data["Overlap"].apply(
                    lambda x: int(x.split("/")[0]) / int(x.split("/")[1])
                )

            # Convert 'Genes' column to list format
            data["Genes"] = data["Genes"].apply(lambda x: str(x).split(";"))

            # Apply filters
            if p_value_threshold is not None:
                data = data[data["Adjusted P-value"] <= p_value_threshold]

            if combined_score_min is not None:
                data = data[data["Combined Score"] >= combined_score_min]

            if overlap_percentage_min is not None:
                data = data[data["Overlap_Percentage"] >= overlap_percentage_min]

            # Save the filtered results
            data.to_csv(output_file, index=False)
            if VERBOSE:
                print(f"Filtered results saved in {output_file}.")

        except Exception as e:
            self.logger.error(f"Error filtering results: {e}")
