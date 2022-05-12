import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.helper_functions import convert_label


class PatientSplitter:
    def __init__(self, root, df, train_percent, seed=1234, n_splits=4):
        self.root = root
        self.video_df = df
        self.image_df = self.create_image_df()
        self.train_percent = train_percent
        self.seed = seed
        self.n_splits = 4
        self.cv = StratifiedGroupKFold(n_splits=n_splits, random_state=self.seed, shuffle=True)
        self.patient_nrs = self.get_patient_nrs()

    def get_num_patients(self):
        return len(self.get_patient_nrs())

    def get_patient_nrs(self):
        return list(set(self.video_df['# patient']))

    def create_fastai_df(self, crossfold_index=0):
        if crossfold_index > (self.n_splits-1):
            raise ValueError(f'crossfold_index is larger than number of crossfolds (n_crossfolds = {self.n_splits})')

        fastai_df = self.image_df
        is_valid = pd.Series([True]*len(self.image_df))

        for cv_idx, (train_idx, val_idx) in enumerate(self.cv.split(self.image_df['name'], self.image_df['labels'], self.image_df['video_name'])):
            if cv_idx == crossfold_index:
                is_valid[train_idx] = False
                is_valid[val_idx] = True
                fastai_df['is_valid'] = is_valid
                return fastai_df
            else:
                continue

    def create_image_df(self):
        name = []
        labels = []
        video_name = []

        for index, row in self.video_df.iterrows():  # loop through each video

            video_path = self.root / row['Video']
            images = os.listdir(video_path)  # get list of individual frames
            for image in images:
                name.append(row['Video'] + '/' + image)
                labels.append(convert_label(row['nCa/Ca/M']))  # convert to label to class name
                video_name.append(row['Video'])

        image_df = pd.DataFrame({'name': name, 'labels': labels, 'video_name': video_name})
        return image_df












