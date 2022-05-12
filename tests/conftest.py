import pytest
import pandas as pd
from src.clean_dataframe import DataframeCleaner
from src.create_train_val_split import PatientSplitter
from src.config import Config


@pytest.fixture()
def config():
    return Config()

@pytest.fixture()
def clean_df(config):
    df = pd.read_excel(config.label_file)
    cleaner = DataframeCleaner(df)
    cleaner.clean_dataframe()
    cleaner.remove_mixed_stones()
    df = cleaner.df

    return df


@pytest.fixture()
def patient_splitter(clean_df, config):
    patient_splitter = PatientSplitter(
        config.dataset_path, clean_df, train_percent=config.train_percent, seed=config.seed)
    return patient_splitter


@pytest.fixture()
def fastai_df(patient_splitter, config):
    return patient_splitter.create_fastai_df()

