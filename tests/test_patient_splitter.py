
from src.create_train_val_split import PatientSplitter


class TestPatientSplitter:

    def test_no_overlap_fastai_df(self, fastai_df):
        # GIVEN: a split patient object
        # WHEN: it is converted to a fastai_df input
        # THEN: there is no overlap in the valid and train set
        df_train = fastai_df[fastai_df['is_valid'] == False]
        df_val = fastai_df[fastai_df['is_valid'] == True]

        im_train = set(df_train['name'])
        im_val = set(df_val['name'])
        assert not (im_train & im_val)

    def test_two_categories(self, fastai_df):
        # GIVEN: a fastai_df
        # THEN: there are only two categories ['calcium' 'non-calcium']
        labels = set(fastai_df['labels'])
        assert len(labels) == 2
        assert 'calcium' in labels
        assert 'non-calcium' in labels

    def test_same_seed_same_split(self, clean_df, config):
        # GIVEN: 2 patient_splitter with same random speed
        splitter1 = PatientSplitter(config.dataset_path, clean_df,
                                    train_percent=config.train_percent, seed=1, n_splits=config.n_splits)
        splitter2 = PatientSplitter(config.dataset_path, clean_df,
                                    train_percent=config.train_percent, seed=1, n_splits=config.n_splits)

        for i in range(config.n_splits):
            # WHEN: fastai_df is created
            fastai_df_1 = splitter1.create_fastai_df(crossfold_index=i)
            fastai_df_2 = splitter2.create_fastai_df(crossfold_index=i)
            # THEN: The split is the same
            assert fastai_df_1['is_valid'].equals(fastai_df_2['is_valid'])

    def test_different_seed_different_split(self, clean_df, config):
        # GIVEN: 2 patient_splitter with same random speed
        splitter1 = PatientSplitter(config.dataset_path, clean_df,
                                    train_percent=config.train_percent, seed=1, n_splits=config.n_splits)
        splitter2 = PatientSplitter(config.dataset_path, clean_df,
                                    train_percent=config.train_percent, seed=2, n_splits=config.n_splits)

        for i in range(config.n_splits):
            # WHEN: fastai_df is created
            fastai_df_1 = splitter1.create_fastai_df(crossfold_index=i)
            fastai_df_2 = splitter2.create_fastai_df(crossfold_index=i)
            # THEN: The split is the same
            assert not fastai_df_1['is_valid'].equals(fastai_df_2['is_valid'])

