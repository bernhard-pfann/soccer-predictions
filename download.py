# Kaggle links:
# "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017"
# "https://www.kaggle.com/datasets/cashncarry/fifaworldranking"


import os

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


def init_kaggle_api(data_dir: str = "data") -> KaggleApi:
    """
    Initialize and authenticate the Kaggle API client.

    :param data_dir: Local directory where datasets will be stored.
    :return: Authenticated Kaggle API instance.
    """
    load_dotenv(".env")

    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

    os.makedirs(data_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    return api


def download_kaggle_data(data_dir: str = "data") -> None:
    """
    Download required Kaggle datasets/files.

    - Downloads only `results.csv` from the football results dataset
    - Downloads the full FIFA rankings dataset

    :param data_dir: Local directory where datasets will be stored.
    """
    api = init_kaggle_api(data_dir)
    api.dataset_download_file(
        dataset="martj42/international-football-results-from-1872-to-2017",
        file_name="results.csv",
        path=data_dir,
    )

    api.dataset_download_files(dataset="cashncarry/fifaworldranking", path=data_dir, unzip=True)
    print("Downloads completed.")
