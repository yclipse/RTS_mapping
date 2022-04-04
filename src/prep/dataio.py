import random


def split(dst, train_split=0.8, val_split=0.1, test_split=0.1):
    '''train/val/test split '''

    random.Random(4).shuffle(dst)  # seeded random shuffle to keep the split unchanged
    dsize = len(dst)
    train = dst[:round(train_split*dsize)]
    val = dst[round(train_split*dsize):round((train_split+val_split)*dsize)]
    test = dst[round((1-test_split)*dsize):]
    return train, val, test


def read_files(dir, split_data=True, **kwargs):
    '''read filenames from directory'''

    data = sorted(glob(os.path.join(dir, '*')))
    if split_data:
        data = split(data, **kwargs)
    return data
