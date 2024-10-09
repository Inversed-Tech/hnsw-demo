import numpy as np
from numpy import log as ln
import math
import pandas as pd

import pickle

import hnsw

from iris_integration import (
    # Generate test templates.
    DIM,
    MAX_ROT,
    iris_random,
    iris_with_noise,
    irisint_make_query as make_query,
    irisint_query_to_vector as query_to_vector,
    irisint_distance as distance,
)

## HNSW Demo

_n_insertions = [10, 100, 500, 1000, 5000, 10000]
_efSearch = [8, 16, 32, 64, 128, 256, 512]


n_elements = 100000 
K = 5
m_L = 0.30
M = 128
efConstruction = 512



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

## Database constrcution
db = make_db()
_params = db.get_params()
_params["Current Size"] = db.get_stats()["db_size"]
# print("Database current size  : " + str(_params))

_construction = []
db.reset_stats()
for _ in range(int(n_elements)):
    _tpl_construct = iris_random()
    _query_construct = make_query(_tpl_construct)
    _id_construct = db.insert(_query_construct)
    _construction.append((_id_construct, _tpl_construct))
 
_construct_stats = db.get_stats()
past_stats().append(_construct_stats)
df_construction = pd.DataFrame(_construction, columns=["ID", "Template"])
pd.DataFrame(past_stats())
print("DB construction is done!")

print("Time             : " + str(search_stats['duration_sec']) + " seconds")
print("M                : " + str(M))
print("efConstruction   : " + str(efConstruction))
print("m_L              : " + str(m_L))
print("K                : " + str(K))
print("Inital DB size   : " + str(n_elements))


## Expriments are starting... 

for efSearch in _efSearch:
    for n_insertions in _n_insertions:

        ## Insertion
        _insertions = []
        db.reset_stats()
        for _ in range(int(n_insertions)):
            _tpl = iris_random()
            _query = make_query(_tpl)
            _id = db.insert(_query)
            _insertions.append((_id, _tpl))

        _insert_stats = db.get_stats()
        past_stats().append(_insert_stats)
        df_insertions = pd.DataFrame(_insertions, columns=["ID", "Template"])
        pd.DataFrame(past_stats())


        ## Search
        noise_level = 0.30

        target = df_insertions.iloc[0]
        noisy_tpl = iris_with_noise(target.Template, noise_level=noise_level)

        db.reset_stats()
        query = make_query(noisy_tpl)
        res = db.search(query, K, ef=efSearch)
        search_stats = db.get_stats()
        df_found = pd.DataFrame(res, columns=["Distance", "ID"]) 
        df_found.index.name = "Rank"
        ## print("result: " + str(res))

        print(f"Searching for vector `ID {target.ID}`, with `{int(noise_level*100)}%` noise.")
        print(f"`Top {K}` Nearest Neighbors:")
        print(df_found)
        found = target.ID in df_found.ID.values
        if found == True:
            print("Found")
        else:
            print("Not found")  

        print("efSearch         : " + str(efSearch))
        print("Last DB size     : " + str(search_stats['db_size']))
        print("Query size       : " + str(n_insertions))
        print("Layer size       : " + str(search_stats['n_layers']))
        print("Search number    : " + str(search_stats['n_searches']))
        print("Distance number  : " + str(search_stats['n_distances']))
        print("Comparison number: " + str(search_stats['n_comparisons']))
        print("Time             : " + str(search_stats['duration_sec']) + " seconds")
        print("\n\n\n\n")
    print("\n\n\n\n")          

print("All experiments are done!")