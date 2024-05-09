from typing import List
from ._abstract_detector import AbstractDetector

from loguru import logger

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from cleanlab.classification import CleanLearning

class ModelIncoherenceDetector(AbstractDetector):

    def __init__(self, labels, texts: List[str] = [], embeddings:List[List[float]] = []):
        
        self.labels = labels
        self.texts = texts # just for compatibility, not used
        self.embeddings = embeddings

        assert embeddings is not [], "Embeddings must be provided"

        self.input_df = pd.DataFrame(
            data = {'embedding': self.embeddings,'label': self.labels}
        )
        self.output_df = pd.DataFrame()

        self.label_encoder = None
        self.model = None

    def _get_clean_learning_issues(self):
        # get radius neighbors graph

        model = LogisticRegression(max_iter=400, solver = 'saga', n_jobs = -1)
        cl = CleanLearning(model, cv_n_folds=5)
        encoder = LabelEncoder().fit(self.labels)

        # fit clean learning model
        cl = cl.fit(
            X = np.array(self.embeddings), 
            labels = encoder.transform(self.labels).tolist(),
            )

        # find clean learning issues
        cl_issues = cl.get_label_issues()
        cl_issues["predicted_label"] = encoder.inverse_transform(
            cl_issues["predicted_label"]
            )
        
        self.label_encoder = encoder
        self.model = cl

        return cl_issues
        

    def get_errors(self):
        
        df = self.input_df.copy()

        # Get clusters
        logger.info(" └ Training models") 
        cl_issues = self._get_clean_learning_issues()
        
        n_cl_issues = cl_issues["is_label_issue"].sum()

        logger.info(f"   └ {n_cl_issues:,} clean learning issues were found.")

        # Getting the model conflicts
        logger.info(f"Calculating 'model_incoherence' conflicts")
        cl_issues["is_model_incoherence"] = cl_issues["is_label_issue"].astype("int").tolist()
        
        # Getting the recommendation
        cl_issues["model_incoherence_recommendation"] = np.where(
            cl_issues["is_label_issue"],
            cl_issues["predicted_label"],
            None
        ).tolist()        

        # Getting the severity level
        cl_issues["model_incoherence_severity"] = np.where(
            cl_issues["is_label_issue"],
            1 - cl_issues["label_quality"],
            None
        ).tolist()

        # calculate recommendation confidence
        cl_issues["model_incoherence_recommendation_confidence"] = np.where(
            cl_issues["is_label_issue"],
            self.model.predict_proba(np.array(self.embeddings)).max(axis = 1),
            None
        ).tolist()
        
        logger.info(" └ Getting error flags")
        # return list of errors flags and recommendations
        error_flags = cl_issues["is_model_incoherence"].tolist()
        error_severity = cl_issues["model_incoherence_severity"].tolist()
        error_recs = cl_issues["model_incoherence_recommendation"].tolist()
        error_recs_confidences = cl_issues["model_incoherence_recommendation_confidence"].tolist()

        error_metadata = [{} for _ in range(cl_issues.shape[0])]   

        # store the final dataframe
        self.output_df = pd.DataFrame(
            data = {
                "is_model_incoherence": error_flags,
                "model_incoherence_severity":error_severity,
                "model_incoherence_recommendation": error_recs,
                "model_incoherence_recommendation_confidence": error_recs_confidences,
                "model_incoherence_metadata": error_metadata,
            }
        )
         
        return error_flags, error_severity, error_recs, error_recs_confidences, error_metadata
