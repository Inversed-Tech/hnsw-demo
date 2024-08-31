from random import randint
import pandas as pd
import dill

import hnsw
from iris.io.dataclasses import IrisTemplate

from iris_integration import (
    DIM,
    iris_with_noise,
    irisint_make_query as make_query,
    irisint_query_to_vector as query_to_vector,
    irisint_distance as distance,
    convert_from_irisint,
)

## HNSW Demo

n_elements = 1
m_L = 0.30

# db1k64m64c.txt
M = 64
efConstruction = 64


K = 5
noise_level = 0.30
threshold = 0.36
efSearch = 128

    
def past_stats():
    return []


def make_db():
    return hnsw.HNSW(
        M=M,
        efConstruction=efConstruction,
        m_L=m_L,
        distance_func=distance,
        query_to_vector_func=query_to_vector,
    )

db = make_db()

## DB extracting from the file
file = 'db1k64m64c.txt'
with open(file, 'rb') as f:
    db = dill.load(f)

print("Database is extracted from the file...")

## search each item in DB

database = []
for _ in range(len(db.vectors)):
    target_idx = randint(0, len(db.vectors)-1)
    target = db.vectors[target_idx] ## (iris, mask)
    database.append((target_idx, target))

df_database = pd.DataFrame(database, columns=["ID", "Template"])


# for _ in range(10):
n_found = 0 
n_threshold = 0
for i in range(len(db.vectors)):
    _cand = df_database.iloc[i]
    cand = convert_from_irisint(_cand.Template, DIM)
    _noisy = iris_with_noise(cand, noise_level=noise_level)
    query = make_query(_noisy)
    db.reset_stats()
    res = db.search(query, K=K, ef=efSearch)[0]
    search_stats = db.get_stats()
    _found = pd.DataFrame([res], columns=["Distance", "ID"])
    found = _cand.ID in _found.ID.values     
    if found == True:
        n_found += 1
    # print(res[0])     
    if 0.0 <= res[0] < threshold:
        n_threshold += 1 

    # print(f"Time     : {search_stats['duration_sec']} seconds")      

print(f"Found    : {n_found} out of {db.get_stats()['db_size']} elements")

print(f"Threshold: {n_threshold} out of {db.get_stats()['db_size']} elements")

