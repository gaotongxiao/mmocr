mjsynth_textrecog_data_root = 'data/mjsynth'

mjsynth_textrecog_train = dict(
    type='OCRDataset',
    data_root=mjsynth_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)
