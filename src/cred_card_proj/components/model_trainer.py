import pandas as pd
import os
from cred_card_proj import logger
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
from cred_card_proj.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        sm = SMOTE(random_state=42)
        train_x_res, train_y_res = sm.fit_resample(train_x, train_y)

        xgb_model = XGBClassifier(eval_metric=self.config.eval_metric, random_state=42)
        xgb_model.fit(train_x_res, train_y_res)

        joblib.dump(
            xgb_model, os.path.join(self.config.root_dir, self.config.model_name)
        )
