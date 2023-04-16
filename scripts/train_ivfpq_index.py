import time
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import faiss

import src.constants as co
import src.utility as util
import src.match_vector as mv


def train(vecs):
    d = 512
    nlist = 50  # Number of inverted lists (number of partitions or cells).

    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    index.train(vecs)

    return index


def train_and_save_model():
    df = util.load_data()
    vecs = util.get_vectors(df)
    index = train(vecs)

    folder, name = co.DATA_FOLDER,  'IVFPQ_index_{}'.format(dt.now().strftime("%Y%m%dT%H%M%S"))
    util.pickle_into_several_parts(folder, name, index, 8)


def main():
    train_and_save_model()



if __name__ == "__main__":
    ss = dt.now().strftime('%H:%M:%S'); start_time = time.time()
    main()
    es = dt.now().strftime('%H:%M:%S'); l = ''.join(['-'] * 37); elapsed = time.time() - start_time
    print("\n%s\n-------- %s - %s --------\n--------- % 9.3f seconds ---------\n%s" % (l, ss, es, elapsed, l))