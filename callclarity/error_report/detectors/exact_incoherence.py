from typing import List

from ._abstract_detector import AbstractDetector

from loguru import logger

import pandas as pd
import numpy as np

class ExactIncoherenceDetector(AbstractDetector):

    def __init__(self, labels, texts: List[str] = [], embeddings:List[List[float]] = []):
        
        self.labels = labels
        self.texts = texts
        self.embeddings = embeddings # just for compatibility, not used

        assert texts is not [], "Texts must be provided"

        self.input_df = pd.DataFrame(
            data = {'text': self.texts,'label': self.labels}
        )
        self.output_df = pd.DataFrame()

    def get_errors(self):

        def _get_row_severity(row):
            is_error = row["is_exact_incoherence"]

            if is_error:
                # collecting inputs
                current_label = row["label"]
                current_label_freq = row["exact_incoherence_cat_counts"][current_label]
                cluster_size = row["exact_incoherence_cluster_size"]
                
                # calculating severity
                severity = 1 - (current_label_freq / cluster_size)
            else:
                severity = np.nan
            return severity
        
        def _get_row_confidence(row):
            is_error = row["is_exact_incoherence"]

            if is_error:
                # collecting inputs
                top_freq = list(row["exact_incoherence_cat_counts"].values())[0]
                cluster_size = row["exact_incoherence_cluster_size"]
                confidence = top_freq / cluster_size
            
            else:
                confidence = np.nan
            return confidence
        
        df = self.input_df.copy()

        # Getting the exact conflicts
        logger.info(f"Calculating 'exact_incoherence' conflicts")
        df_exact_conflicts = (
            df
            .groupby("text")
            .agg(
                exact_incoherence_cat_counts = ("label", lambda s: s.value_counts().to_dict()),
                exact_incoherence_n_cats = ("label", lambda s: s.nunique()),
                exact_incoherence_cluster_size = ("label", lambda s: s.shape[0]),
            )
            .reset_index()
        
        )
        logger.info(" └ Filtering conflicted cases")
        # filter only the cases that had conflicts
        df_exact_conflicts = df_exact_conflicts[df_exact_conflicts["exact_incoherence_n_cats"] > 1]

        logger.info(" └ Getting cluster indices") 
        df_exact_conflicts = df_exact_conflicts.sort_values("exact_incoherence_cluster_size", ascending=False)
        df_exact_conflicts["exact_incoherence_cluster_index"] = range(df_exact_conflicts.shape[0])

        logger.info(" └ Getting error recommendations")
        # calculate the recommended category by checking the most frequent within that cluster
        df_exact_conflicts["exact_incoherence_recommendation"] = df_exact_conflicts["exact_incoherence_cat_counts"].apply(lambda x: list(x.items())[0][0])

        logger.info(" └ Getting incoherence top frequency")
        df_exact_conflicts["exact_incoherence_top_frequency"] = df_exact_conflicts["exact_incoherence_cat_counts"].apply(lambda x: list(x.items())[0][1])

        # flag error and subset for merge
        df_exact_conflicts = df_exact_conflicts[
            [
                "text", "exact_incoherence_recommendation", "exact_incoherence_cluster_index", "exact_incoherence_cluster_size", 
                "exact_incoherence_top_frequency", "exact_incoherence_cat_counts"]
            ]
        
        logger.info(" └ Getting error flags")
        # merge back to original dataframe to get the flags of error and recommendations
        df = df.merge(
            df_exact_conflicts,
            on = ["text"],
            how = "left",
            validate = "many_to_one"
            )
        
        df["is_exact_incoherence"] = np.where(
          (df["exact_incoherence_recommendation"].notnull()) & (df["label"] !=  df["exact_incoherence_recommendation"]),
          1,
          0,
        )
        df["exact_incoherence_recommendation"] = np.where(
            df["is_exact_incoherence"] == 1,
            df["exact_incoherence_recommendation"],
            None,
        )

        # calculate total errors within each cluster
        df["exact_incoherence_cluster_candidates"] = np.where(
            df["exact_incoherence_cluster_index"].notnull(),
            df.groupby("exact_incoherence_cluster_index")["is_exact_incoherence"].transform("sum"),
            np.nan
        )

        # calculate error severity
        df["exact_incoherence_severity"] = np.where(
            df["is_exact_incoherence"] == 1, 
            df.apply(lambda row: _get_row_severity(row), axis = 1), 
            np.nan
            )

        # calculate recommendation confidence
        df["exact_incoherence_recommendation_confidence"] = np.where(
            df["is_exact_incoherence"] == 1, 
            df.apply(lambda row: _get_row_confidence(row), axis = 1), 
            np.nan            
        )

        # return list of errors flags and recommendations
        error_flags = df["is_exact_incoherence"].tolist()
        error_severity = df["exact_incoherence_severity"].tolist()
        error_recs = df["exact_incoherence_recommendation"].tolist()
        error_recs_confidences = df["exact_incoherence_recommendation_confidence"].tolist()

        error_metadata = [
            {
                "exact_incoherence_cluster_index": index, 
                "exact_incoherence_cluster_size": size, 
                "exact_incoherence_cluster_candidates": error,
                }
            for index, size, error in
            zip(
                df["exact_incoherence_cluster_index"].tolist(),
                df["exact_incoherence_cluster_size"].tolist(),
                df["exact_incoherence_cluster_candidates"].tolist(),
            )
        ]   

        # store the final dataframe
        self.output_df = pd.DataFrame(
            data = {
                "is_exact_incoherence": error_flags,
                "exact_incoherence_severity": error_severity,
                "exact_incoherence_recommendation": error_recs,
                "exact_incoherence_recommendation_confidence": error_recs_confidences,
                "exact_incoherence_metadata": error_metadata,
            }
        )
         
        return error_flags, error_severity, error_recs, error_recs_confidences, error_metadata
