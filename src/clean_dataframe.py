import math
import pandas as pd
pd.options.mode.chained_assignment = None


class DataframeCleaner:
    def __init__(self, df):
        self.df = df

    def fill_out_patient_label(self):

        series = self.df['# patient']
        current_nr = 1
        for idx, item in enumerate(series):
            if not math.isnan(item):
                current_nr = int(item)
            else:
                series.iloc[idx] = current_nr
        series = series.astype('int32')
        self.df['# patient'] = series

    def convert_video_name_to_string(self):
        self.df['Video'] = self.df['Video'].apply(lambda x: str(x).zfill(4))

    def clean_dataframe(self):
        self.df = self.df[['Video', '# patient', 'nCa/Ca/M']]
        self.fill_out_patient_label()
        self.convert_video_name_to_string()

    def remove_mixed_stones(self):
        self.df = self.df[self.df['nCa/Ca/M'] != 2]
