from pathlib import Path


class Config:
    def __init__(self, seed=1234, train_percent=0.75, batch_size=256):
        self.root = Path('D:\\Research projects\\stoneclassification\\')
        self.dataset_path = self.root / 'dataset' / 'frames_preprocessed'
        self.label_file = self.root / 'dataset' / 'Sleutelbestand_VASC_in_vivo_anoniem_28_02_2022.xlsx'
        
        self.trained_model_dir = self.root / 'classification' / 'src' / 'trained_models'

        self.n_splits = 4
        self.seed = seed
        self.train_percent = train_percent
        self.batch_size = batch_size
