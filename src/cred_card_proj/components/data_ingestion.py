import os
from dotenv import load_dotenv

load_dotenv()

from cred_card_proj import logger
from cred_card_proj.utils.common import get_size
from cred_card_proj.entity.config_entity import DataIngestionConfig
from pathlib import Path

import kagglehub #read env before importing kagglehub


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            logger.info(
                f"Dataset downloaded successfully from Kaggle!\nStored in {Path(path).as_posix()}"
            )
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )
