

def convert_label(label):
    if isinstance(label, int):
        if label == 0:
            return 'non-calcium'
        if label == 1:
            return 'calcium'
        if label == 2:
            return 'mixed'

    if isinstance(label, str):
        if label == 'non-calcium':
            return 0
        if label == 'calcium':
            return 1
        if label == 'mixed':
            return 2


def dataset_info(dls):
    print(f'Training images: {len(dls.train_ds)}')
    print(f'Validation images: {len(dls.valid_ds)}')
