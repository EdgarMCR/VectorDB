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
    d = 512              # Dimension (length) of vectors.
    M = 32               # Number of connections that would be made for each new vertex during HNSW construction.
    efConstruction = 16  # Sets the number of nearest neighbours in insertion layer during construction
    efSearch = 10        # Number of NN during search

    # Create the index.
    index = faiss.IndexHNSWFlat(d, M)

    index.hnsw.efConstruction = efConstruction
    index.add(vecs)  # build the index
    index.hnsw.efSearch = efSearch

    return index


def train_and_save_model():
    df = util.load_data()
    vecs = util.get_vectors(df)
    index = train(vecs)

    folder, name = co.DATA_FOLDER,  'hsmw_index_{}'.format(dt.now().strftime("%Y%m%dT%H%M%S"))
    util.pickle_into_several_parts(folder, name, index, 8)


def main():
    # train_and_save_model()

    df = util.load_data()
    vecs = util.get_vectors(df)
    print(f"{np.min(vecs)=}, {np.max(vecs)=}")
    flattened = np.ravel(vecs)
    print(f"{np.min(flattened)=}, {np.max(flattened)=}")
    vec1 = np.reshape(vecs[0], (1, -1))
    vec1[0, 0] = 0.06
    # and now we can search
    results = mv.get_all_matches_within_distance(vec1, 0.3)
    keys = [mv.NAME, mv.CONF, mv.DIST, mv.IND]
    for rslt in results:
        for k in keys:
            print(f"{k=}, {rslt[k]=}", end=', ')
        print('')
    print(f"{len(results)=}")


if __name__ == "__main__":
    ss = dt.now().strftime('%H:%M:%S'); start_time = time.time()
    main()
    es = dt.now().strftime('%H:%M:%S'); l = ''.join(['-'] * 37); elapsed = time.time() - start_time
    print("\n%s\n-------- %s - %s --------\n--------- % 9.3f seconds ---------\n%s" % (l, ss, es, elapsed, l))