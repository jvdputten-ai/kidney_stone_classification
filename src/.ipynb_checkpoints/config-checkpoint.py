from pathlib import Path


class Config:
    def __init__(self):
        self.root = Path('D:\\Research projects\\stoneclassification\\')
        self.dataset_path = self.root / 'dataset' / 'frames_preprocessed'
        self.label_file = self.root / 'dataset' / 'Sleutelbestand_VASC_in_vivo_anoniem_28_02_2022.xlsx'

        self.seed = 1234
        self.train_percent = 0.75
        self.batch_size = 256
