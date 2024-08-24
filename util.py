import hnsw
import numpy as np
import pandas as pd
import pickle
from math import prod

from iris_integration import (
    MAX_ROT,
    convert_to_irisint,
    int_distance,
    rotated_tpl,
    IrisTemplate,
    irisint_make_query,
)


### High level operations

def find_iris(idx, tpl, ef, threshold_dist):
    for rot_tpl in irisint_make_query(tpl):
        nearest_dist, nearest_id = idx.search(rot_tpl, 1, ef=ef)[0]
        if nearest_dist < threshold_dist:
            return nearest_id
    return None

def insert_iris(idx, tpl):
    irisint = convert_to_irisint(tpl)
    tpl_id = idx.insert(irisint)
    return tpl_id


### Serialization

def extract_state(idx):
    return (
        idx.M,
        idx.efConstruction,
        idx.m_L,
        idx.vectors,
        idx.entry_point,
        idx.layers,
    )

def load_from_state(state):
    M, ef_construction, m_L, vectors, entry_point, layers = state
    idx = hnsw.HNSW(
        M=M,
        efConstruction=ef_construction,
        m_L=m_L,
        distance_func=int_distance,
    )
    idx.vectors = vectors
    idx.entry_point = entry_point
    idx.layers = layers
    return idx

def dump_hnsw_index(idx, filename):
    data = extract_state(idx)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_hnsw_index(filename):
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    return load_from_state(state)
