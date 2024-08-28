from __future__ import print_function
import sys
# print(sys.argv, len(sys.argv))

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

## DataBase construction for HNSW Demo

##################### a name for csv file "test.csv" in line 72
##################### Decide DB construction via hnsw
n_elements = 1      # the number of elements (n_elements) loaded to DB
M = 64              # M value
efConstruction = 32 # efConstruction
m_L = 0.30

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
## print("Database current size: " + str(_params))

_construction = []
db.reset_stats()
for _ in range(int(n_elements)):
    _tpl_construct = iris_random()
    _query_construct = make_query(_tpl_construct)
    _id_construct = db.insert(_query_construct)
    _construction.append((_id_construct, _tpl_construct))
    ## print(_tpl_construct)
## print(_construction)
 
construct_stats = db.get_stats()
past_stats().append(construct_stats)
df_construction = pd.DataFrame(_construction, columns=["ID", "Template"])
pd.DataFrame(past_stats())
# print(df_construction)
print("DB construction is done!")

## Write DB to a csv file
df_construction.to_csv('test.csv', index=False)  
print("DB construction is written to a csv file!")

print("Inital DB size   : " + str(db.get_stats()["db_size"]))
print("M                : " + str(M))
print("efConstruction   : " + str(efConstruction))
print("m_L              : " + str(m_L))
print("Time             : " + str(construct_stats['duration_sec']) + " seconds")