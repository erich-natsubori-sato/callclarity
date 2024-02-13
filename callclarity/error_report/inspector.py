from typing import List, Dict, Any

import pandas as pd
import numpy as np
from loguru import logger

from callclarity.error_report.embeddings.embedder import TextEmbedder
from callclarity.error_report.detectors.approx_incoherence import ApproximateIncoherenceDetector
from callclarity.error_report.detectors.exact_incoherence import ExactIncoherenceDetector
from callclarity.error_report.detectors.model_incoherence import ModelIncoherenceDetector
from callclarity.error_report.sorters.semantic_sorter import SemanticSorter

# get as input
    # all the the detector classes
    # all the input kwargs
    # all the output columns

# TODO: add severity for exact incoherence severity and approx incoherence severity (It is equal the number of percentage)
# TODO: add consolidated severity 
# TODO: deduplicate before creating model incoherence, then broadcast to every row with the same description, to make sure the same severity is assigned
# TODO: maybe leave the semantic order in the model incoherence priority levels

class IncoherenceInspector:

    def __init__(
            self, 
            ids:List[Any], texts: List[str], labels: List[str], embeddings: List[List[float]] = [],
            detector_classes = [ExactIncoherenceDetector, ApproximateIncoherenceDetector, ModelIncoherenceDetector],
            detectors_list_kwargs: list[Dict[str, Any]] = [{},{},{}],
            ):

        self.ids = list(ids)
        self.texts = list(texts)
        self.labels = list(labels)
        self.embeddings = embeddings if embeddings != [] else self._init_embeddings()

        # detectors params
        self.detector_classes = detector_classes
        self.detectors_list_kwargs = detectors_list_kwargs
        self.detectors = []

        self.input_df = self._init_dataframe()
        self.output_df = self.input_df.copy()

    def _init_embeddings(self):
        logger.info("Embedding texts")
        embeddings = TextEmbedder(self.texts).embed().tolist()
        
        return embeddings

    def _init_dataframe(self):
        df = pd.DataFrame(
            data = {
                'id': self.ids,
                'text': self.texts,
                'label': self.labels,
                'embedding': self.embeddings,
                }
            )
        
        return df

    def get_errors(self):

        df = self.output_df

        for detector_class, detector_kwargs in zip(self.detector_classes, self.detectors_list_kwargs):
            # instantiate detector
            detector_obj = detector_class(
                labels = self.labels,
                texts = self.texts,
                embeddings = self.embeddings,
                **detector_kwargs
                )
            
            # detect 
            detector_obj.get_errors()

            # add error columns
            df = pd.concat(
                [df, detector_obj.output_df],
                axis = 1
                )
            
            # reassign detector
            self.detectors.append(detector_obj)
        
        self.output_df = df
        
        return self.output_df
    
    def _get_row_priority(self, row):
        
        # get priority
        p0_condition = row["is_exact_incoherence"]
        p1_condition = (
            (not row["is_exact_incoherence"])
            and (row["is_approx_incoherence"])
        )
        p2_condition = (
            (not row["is_exact_incoherence"])
            and (not row["is_approx_incoherence"])
            and (row["is_model_incoherence"] and row["model_incoherence_severity"] > 0.9)
        )
        p3_condition = (
            (not row["is_exact_incoherence"])
            and (not row["is_approx_incoherence"])
            and (row["is_model_incoherence"] and row["model_incoherence_severity"] <= 0.9)
        )

        priority = None
        if p0_condition:
            priority = 0
        elif p1_condition:
            priority = 1
        elif p2_condition:
            priority = 2
        elif p3_condition:
            priority = 3

        return priority

    def assign_priority(self):

        df = self.output_df

        # get row-level priority
        df["priority"] = df.apply(
            lambda row: self._get_row_priority(row),
            axis = 1
            )
        
        # get cluster-level priority
        df["_approx_incoherence_cluster_index"] = df["approx_incoherence_metadata"].str["approx_incoherence_cluster_index"]
        df["cluster_priority"] = (
            df
            .groupby("cluster_id")
            ["priority"].transform("min")
        )

        self.output_df = df

        return self.output_df
    
    def assign_incoherence(self):
        
        df = self.output_df

        # get row-level incoherence
        df["is_incoherent"] = df["priority"].notnull().astype("int")

        # get cluster-level priority
        df["cluster_incoherence_count"] = (
            df
            .groupby("cluster_id")
            ["is_incoherent"].transform("sum")
        )

        # get approx cluster size
        df["cluster_total_count"] = df["approx_incoherence_metadata"].str["approx_incoherence_cluster_size"]

        self.output_df = df
        
        return self.output_df
    
    def assign_severity(self):

        df = self.output_df

        # assign label
        df["incoherence_severity"] = np.select(
            condlist = [
                (df["priority"] == 0),
                (df["priority"] == 1),
                ((df["priority"] == 2) | (df["priority"] == 3)),
            ],
            choicelist= [
                df["exact_incoherence_severity"],
                df["approx_incoherence_severity"],
                df["model_incoherence_severity"],
            ],
            default = None,
        )

        self.output_df = df

        return self.output_df

    def assign_recommendation(self):

        df = self.output_df

        # assign label
        df["label_recommended"] = np.select(
            condlist = [
                (df["priority"] == 0),
                (df["priority"] == 1),
                ((df["priority"] == 2) | (df["priority"] == 3)),
            ],
            choicelist= [
                df["exact_incoherence_recommendation"],
                df["approx_incoherence_recommendation"],
                df["model_incoherence_recommendation"],
            ],
            default = None,
        )
        
        # assign probability
        df["recommendation_confidence"] = np.select(
            condlist = [
                (df["priority"] == 0),
                (df["priority"] == 1),
                ((df["priority"] == 2) | (df["priority"] == 3)),
            ],
            choicelist= [
                df["exact_incoherence_recommendation_confidence"],
                df["approx_incoherence_recommendation_confidence"],
                df["model_incoherence_recommendation_confidence"],
            ],
            default = None,
        )

        self.output_df = df
        
        return self.output_df
    
    def add_suggestion_col(self):
        self.output_df["label_suggested"] = None

    def sort_rows(self):
        
        df = self.output_df
        
        # semantically sort
        semantic_sorter = SemanticSorter(self.embeddings, embedding_distance_step=0.01)
        sorted_indices = semantic_sorter.get_sort_indices()
        df = df.iloc[sorted_indices]

        # sort by cluster priority and number of errors
        df = df.sort_values(
            ["cluster_priority","cluster_incoherence_count", "cluster_id","priority"],
              ascending = [True, False, True ,True]
              )

        self.output_df = df

        return self.output_df
    
    def sort_cols(self):

        df = self.output_df

        sorted_cols = [
            "cluster_id", "cluster_priority",
            "id", "priority",
            "incoherence_severity",
            "text", "label",
            "label_recommended",
            "recommendation_confidence",
            "label_suggested",
            "is_incoherent",
            "exact_incoherence_severity","exact_incoherence_recommendation", "exact_incoherence_recommendation_confidence",
            "approx_incoherence_severity","approx_incoherence_recommendation", "approx_incoherence_recommendation_confidence",
            "model_incoherence_severity","model_incoherence_recommendation", "model_incoherence_recommendation_confidence",
            "is_exact_incoherence","is_approx_incoherence", "is_model_incoherence", 
            "exact_incoherence_metadata", "approx_incoherence_metadata","model_incoherence_metadata",
            "cluster_incoherence_count", "cluster_total_count",
        ]

        dropped_cols = set(df.columns).difference(set(sorted_cols))
        if len(dropped_cols) > 0:
            logger.warning(f"Columns {dropped_cols} will be dropped as they are not in the sorted columns list")

        df = df[sorted_cols]

        self.output_df = df

        return self.output_df

    def inspect(self):
        
        self.get_errors()
        self.assign_priority()
        self.assign_incoherence()
        self.assign_severity()
        self.assign_recommendation()
        self.add_suggestion_col()
        self.sort_rows()
        self.sort_cols()

        return self.output_df
