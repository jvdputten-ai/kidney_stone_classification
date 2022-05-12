from fastai.vision.all import *

from config import Config
from create_data_loaders import create_data_loaders
from helper_functions import dataset_info

import matplotlib.pyplot as plt


if __name__ == '__main__':
    config = Config()
    dls = create_data_loaders(config)
    dataset_info(dls)
    learn = cnn_learner(dls, resnet18, metrics=accuracy).to_fp16()
    learn.fine_tune(5, 5e-3)

    interp = Interpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.show()

