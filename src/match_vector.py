import time
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime as dt
from typing import List, Dict

import numpy as np
import faiss

import src.constants as co
import src.utility as util

MODEL, DF = None, None
VEC, NAME, CONF, DIST, IND = 'vector', 'name', 'confidence', 'distance', 'index'


def maybe_load_model():
    global MODEL, DF
    if MODEL is None:
        MODEL = util.pickle_from_several_parts(co.DATA_FOLDER, co.MODEL_NAME)
        DF = util.load_from_pickle(co.DATA_FOLDER / 'df_wo_vec.pkl')
    return MODEL, DF


def get_matching_vectors(vector: np.ndarray, k: int, with_vector: bool = False):
    """ Return the k closest vectors """
    vector = np.array(vector).astype(np.float32)

    index, df = maybe_load_model()
    distances, indices = index.search(vector, k)
    matches = []
    for index_ in range(len(indices[0])):
        ii = indices[0][index_]

        result = {
            NAME: df['Name'].iloc[ii],
            CONF: df['Confidence'].iloc[ii],
            DIST: distances[0][index_],  # We should be able to reconstruct the vectors using `reconstruct`
            IND: ii
        }
        if with_vector:
            vec = util.load_vector(ii)
            result[VEC] = vec

        matches.append(result)
    return matches


def get_all_matches_within_distance(vector: list, max_distance: float, with_vector: bool = False) -> dict:
    """ We want to identify all matches within distance `max_distance`.

    Here we use an exhaustive search but stop if we exceed 128 results for performance reasons.
    """
    vector = np.array(vector).reshape((1, 512))
    ks = [4, 16, 128]
    for k in ks:
        results = get_matching_vectors(vector, k, with_vector)
        filtered = [x for x in results if x[DIST] < max_distance]
        if len(filtered) < k:
            # Stop searching once we have identified a vector that is further away
            break
    filtered = convert_result_to_python_types(filtered)
    return filtered


def convert_result_to_python_types(results: List[Dict]) -> List[Dict]:
    new_results = []
    for rslt in results:
        new_rslt = {
            NAME: str(rslt[NAME]),
            CONF: float(rslt[CONF]),
            DIST: float(rslt[DIST]),
            IND: int(rslt[IND])
        }
        if VEC in rslt:
            new_rslt[VEC] = rslt[VEC].tolist()
        new_results.append(new_rslt)

    return new_results
