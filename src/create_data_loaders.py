import pandas as pd
from fastai.vision.data import ImageDataLoaders

from clean_dataframe import DataframeCleaner
from config import Config
from create_train_val_split import PatientSplitter

config = Config()


def create_data_loaders(config, crossfold_index):
    df = pd.read_excel(config.label_file)
    cleaner = DataframeCleaner(df)
    cleaner.clean_dataframe()
    cleaner.remove_mixed_stones()
    df = cleaner.df

    # randomly select 25% of patients
    patient_splitter = PatientSplitter(config.dataset_path, df, train_percent=config.train_percent, seed=config.seed)
    fastai_df = patient_splitter.create_fastai_df(crossfold_index=crossfold_index)

    dls = ImageDataLoaders.from_df(fastai_df, config.dataset_path, num_workers=0,
                                   bs=config.batch_size, valid_col='is_valid')
    return dls










