import os
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from cred_card_proj.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from cred_card_proj.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred, pred_proba=None):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        if pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(actual, pred_proba)

        return metrics

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_fraud = model.predict(test_x)
            predicted_proba = (
                model.predict_proba(test_x)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            scores = self.eval_metrics(test_y, predicted_fraud, predicted_proba)

            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            for metric_name, metric_value in scores.items():
                mlflow.log_metric(metric_name, metric_value)

            # Full classification report as text artifact — useful for
            # per-class breakdown (fraud vs non-fraud) beyond the scalar metrics
            report_text = classification_report(test_y, predicted_fraud)
            report_path = Path(self.config.classification_report_file_name)
            report_path.write_text(report_text)
            mlflow.log_artifact(str(report_path))

            # Confusion matrix as artifact too — very useful for fraud detection
            # to see false negatives (missed fraud) vs false positives
            cm = confusion_matrix(test_y, predicted_fraud)
            cm_path = Path(self.config.confusion_matrix)
            save_json(path=cm_path, data={"confusion_matrix": cm.tolist()})
            mlflow.log_artifact(str(cm_path))

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.xgboost.log_model(
                    model, "model", registered_model_name="XGBoostModel"
                )
            else:
                mlflow.xgboost.log_model(model, "model")
