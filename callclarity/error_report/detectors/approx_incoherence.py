from typing import List
from ._abstract_detector import AbstractDetector

from loguru import logger

import pandas as pd
import numpy as np

from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

class ApproximateIncoherenceDetector(AbstractDetector):

    def __init__(self, labels, texts: List[str] = [], embeddings:List[List[float]] = [], max_embedding_distance: float = 0.08):
        
        self.labels = labels
        self.texts = texts # just for compatibility, not used
        self.embeddings = embeddings

        assert embeddings is not [], "Embeddings must be provided"

        self.max_embedding_distance = max_embedding_distance

        self.input_df = pd.DataFrame(
            data = {'embedding': self.embeddings,'label': self.labels}
        )
        self.output_df = pd.DataFrame()

        self.nn_graph = None

    def _get_clusters(self):
        # get radius neighbors graph
        nn_graph = radius_neighbors_graph(
            self.embeddings,
            radius = self.max_embedding_distance, 
            mode = "distance", metric = "cosine", 
            include_self = True,
            n_jobs = -1
            
            )
        
        # apply 1 if connected by distance < max_embedding_distance, otherwise 0
        coo_graph = nn_graph.tocoo()
        mask = (coo_graph.data > 0) & (coo_graph.data <= self.max_embedding_distance)
        filtered_nn_graph = csr_matrix((coo_graph.data[mask], (coo_graph.row[mask], coo_graph.col[mask])), shape=nn_graph.shape)

        # get the clusters
        n_components, clusters = connected_components(filtered_nn_graph)

        # store the computed nn_graph
        self.nn_graph = nn_graph

        return n_components, clusters
        

    def get_errors(self):

        def _get_row_severity(row):
            is_error = row["is_approx_incoherence"]

            if is_error:
                # collecting inputs
                current_label = row["label"]
                current_label_freq = row["approx_incoherence_cat_counts"][current_label]
                cluster_size = row["approx_incoherence_cluster_size"]

                # calculating severity
                severity = 1 - (current_label_freq / cluster_size)
            else:
                severity = np.nan
            return severity

        def _get_row_confidence(row):
            is_error = row["is_approx_incoherence"]

            if is_error:
                # collecting inputs
                top_freq = list(row["approx_incoherence_cat_counts"].values())[0]
                cluster_size = row["approx_incoherence_cluster_size"]
                confidence = top_freq / cluster_size
            
            else:
                confidence = np.nan
            return confidence
        
        df = self.input_df.copy()

        # Get clusters
        logger.info(" └ Calculating clusters") 
        n_components, clusters = self._get_clusters()
        df["_cluster"] = clusters

        logger.info(f"   └ {n_components:,} clusters were found.") 

        # Getting the approx conflicts
        logger.info(f"Calculating 'approx_incoherence' conflicts")
        df_approx_conflicts = (
            df
            .groupby("_cluster")
            .agg(
                approx_incoherence_cat_counts = ("label", lambda s: s.value_counts().to_dict()),
                approx_incoherence_n_cats = ("label", lambda s: s.nunique()),
                approx_incoherence_cluster_size = ("label", lambda s: s.shape[0]),
            )
            .reset_index()
        
        )

        logger.info(" └ Getting cluster indices") 
        df_approx_conflicts = df_approx_conflicts.sort_values("approx_incoherence_cluster_size", ascending=False)
        df_approx_conflicts["approx_incoherence_cluster_index"] = range(df_approx_conflicts.shape[0])

        logger.info(" └ Getting error recommendations")
        # calculate the recommended category by checking the most frequent within that cluster
        df_approx_conflicts["approx_incoherence_recommendation"] = df_approx_conflicts["approx_incoherence_cat_counts"].apply(lambda x: list(x.items())[0][0])

        logger.info(" └ Getting incoherence top frequency")
        df_approx_conflicts["approx_incoherence_top_frequency"] = df_approx_conflicts["approx_incoherence_cat_counts"].apply(lambda x: list(x.items())[0][1])

        # flag error and subset for merge
        df_approx_conflicts = df_approx_conflicts[[
            "_cluster", "approx_incoherence_recommendation", "approx_incoherence_cluster_index", "approx_incoherence_cluster_size",
            "approx_incoherence_top_frequency", "approx_incoherence_cat_counts",
            ]]
        
        logger.info(" └ Getting error flags")
        # merge back to original dataframe to get the flags of error and recommendations
        df = df.merge(
            df_approx_conflicts,
            on = ["_cluster"],
            how = "left",
            validate = "many_to_one"
            )

        # drop _cluster column
        df = df.drop(columns = ["_cluster"])

        # create cluster_id column
        df["cluster_id"] = df["approx_incoherence_cluster_index"]
        
        df["is_approx_incoherence"] = np.where(
          (df["approx_incoherence_recommendation"].notnull()) & (df["label"] !=  df["approx_incoherence_recommendation"]),
          1,
          0,
        )

        df["approx_incoherence_cluster_index"] = np.where(
          df["is_approx_incoherence"] == 1,
          df["approx_incoherence_cluster_index"],
          None,
        )        

        df["approx_incoherence_recommendation"] = np.where(
            df["is_approx_incoherence"] == 1,
            df["approx_incoherence_recommendation"],
            None,
        )

        # calculate total errors within each cluster
        df["approx_incoherence_cluster_candidates"] = np.where(
            df["approx_incoherence_cluster_index"].notnull(),
            df.groupby("approx_incoherence_cluster_index")["is_approx_incoherence"].transform("sum"),
            np.nan
        )

        # calculate error severity
        df["approx_incoherence_severity"] = np.where(
            df["is_approx_incoherence"] == 1, 
            df.apply(lambda row: _get_row_severity(row), axis = 1), 
            np.nan
            )
        
        # calculate recommendation confidence
        df["approx_incoherence_recommendation_confidence"] = np.where(
            df["is_approx_incoherence"] == 1, 
            df.apply(lambda row: _get_row_confidence(row), axis = 1), 
            np.nan            
        )

        # return list of errors flags and recommendations
        error_flags = df["is_approx_incoherence"].tolist()
        error_severity = df["approx_incoherence_severity"].tolist()
        error_recs = df["approx_incoherence_recommendation"].tolist()
        error_recs_confidences = df["approx_incoherence_recommendation_confidence"].tolist()

        error_metadata = [
            {
                "approx_incoherence_cluster_index": index, 
                "approx_incoherence_cluster_size": size, 
                "approx_incoherence_cluster_candidates": error,
                }
            for index, size, error in
            zip(
                df["approx_incoherence_cluster_index"].tolist(),
                df["approx_incoherence_cluster_size"].tolist(),
                df["approx_incoherence_cluster_candidates"].tolist(),
            )
        ]   

        # store the final dataframe
        self.output_df = pd.DataFrame(
            data = {
                "cluster_id":df["cluster_id"].tolist(),
                "is_approx_incoherence": error_flags,
                "approx_incoherence_severity": error_severity,
                "approx_incoherence_recommendation": error_recs,
                "approx_incoherence_recommendation_confidence": error_recs_confidences,
                "approx_incoherence_metadata": error_metadata,
            }
        )
         
        return error_flags, error_severity, error_recs, error_recs_confidences, error_metadata
