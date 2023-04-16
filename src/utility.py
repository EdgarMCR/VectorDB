import os
import re
import math
import pickle
import gzip
import lzma
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import faiss

import src.constants as co


def persist_to_file(file_name):
    def decorator(original_func):

        def new_func():
            cache = load_from_pickle(file_name)

            if cache is None:
                cache = original_func()
                save_to_pickle(file_name, cache)
            return cache

        return new_func

    return decorator


def save_to_pickle(path: Path, obj: object, compress: bool = False):
    if compress:
        with lzma.open(path, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


def load_from_pickle(path: Path, compress: bool = False) -> object:
    try:
        if compress:
            with lzma.open(path, 'rb') as f:
                obj = pickle.load(f)

        else:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
    except Exception as ex:  # General exception because I'm not sure what exceptions are thrown by the lzma module
        logging.error("Decompression of `{}` failed with `{}`.".format(path, str(ex)))
        obj = None
    return obj


@persist_to_file(co.TMP_FOLDER / 'load_data.pkl')
def load_data() -> pd.DataFrame:
    """ Load data from CSV """
    with gzip.open(co.DATA_PATH) as f:
        df = pd.read_csv(f)
    assert len(df.columns) > 4, 'Unexpected format of CSV'

    df2 = df.iloc[:, 1:3]
    df2['vec'] = df.iloc[:, 3:].values.tolist()
    df2['vec'] = df2['vec'].apply(np.array)
    assert isinstance(df2['vec'][0], np.ndarray)

    return df2


def save_data_without_vectors():
    df = load_data()
    df = df.drop('vec', axis=1)
    df.to_pickle(co.META_DATA_PATH)


def load_data_without_vectors():
    df = pd.read_pickle(co.META_DATA_PATH)
    return df


def pickle_into_several_parts(folder: Path, name: str, obj: object, parts: int):
    if isinstance(folder, str):
        folder = Path(folder)

    content = pickle.dumps(obj)
    leng = len(content)
    plen = int(math.floor(leng/parts))
    for ii in range(parts):
        if ii == parts -1:
            part = content[ii * plen:]
        else:
            part = content[ii * plen: (ii + 1) * plen]

        file_name = name + '.part%02d.pkl.xz' % ii
        path = folder / file_name
        with lzma.open(path, 'wb') as f:
            f.write(part)


def pickle_from_several_parts(folder: Path, name: str):
    if isinstance(folder, str):
        folder = Path(folder)

    pattern = name + r'\.part\d{2}.pkl.xz'
    files = sorted([x for x in os.listdir(folder) if re.match(pattern, str(x))])

    content = b''
    for file in files:
        with lzma.open(folder / file, 'rb') as f:
            content += f.read()

    return pickle.loads(content)


def get_vectors(df: pd.DataFrame) -> np.ndarray:
    """ Numpy is not behaving as I expect when calling df['vec'].values.
    For development speed, I will use a computationally expensive loop here.
    """
    leng = len(df)
    vecs = np.zeros((leng, 512))
    for ii, row in df.iterrows():
        vecs[ii, :] = row['vec']

    return vecs


def look_up_vector(index, vec: np.ndarray, k: int, df: pd.DataFrame):
    D, I = index.search(vec, k)  # search
    print(f"{I=}")
    for index_ in I[0]:
        print(f"{index_=}, {df['Name'].iloc[index_]=}, {df['Confidence'].iloc[index_]=}")


def load_vector(index: int):
    """ Load vector from original dataset by index.
    Used for easy of development and work-around for cases when `index.reconstruct` errors.
    For production system, figure our the bug or save vectors in a DB. This would double as a handy store for training
    data.

    This stackoverflow question suggests that it is because we use pip instead of conda to install. However,
    I have to use pip as I am deploying to Azure Function
    https://stackoverflow.com/questions/70624600/faiss-how-to-retrieve-vector-by-id-from-python
    """
    if index < 0 or 70300 < index:
        logging.error("Index `{}` is out of range".format(index))
        return None

    sp = co.DATA_FOLDER / 'vectors' / '{}.npy'.format(index)
    vec = np.load(sp)
    return vec

