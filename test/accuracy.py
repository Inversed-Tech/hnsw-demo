import numpy as np
from numpy import log as ln
import math
import pandas as pd

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


## Calculate mL
def calculate_mL(M):
    if M <= 0:
        raise ValueError("M must be greater than 0 but also greater than 1")
    m_l = 1 / math.log(M)
    return round(m_l, 2)


n_elements = 100

M = 128
# if M <= 0:
#     raise ValueError("M must be greater than 0")

efConstruction = 128
# if efConstruction <= 0:
#     raise ValueError("efConstruction must be greater than 0")


m_L = 0.30  ## calculate_mL(M)

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


## Insertion
n_insertions = 10
# if n_insertions < 0:
#     raise ValueError("n_insertions must be greater than or equal to 0")

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
efSearch = 128
# if efSearch <= 0:
#     raise ValueError("efSearch must be greater than 0")

K = 5
# if K <= 0:
#     raise ValueError("K must be greater than 0")

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


# print("Inital size  		     : " + str(n_elements))
# print("Query size     		 : " + str(n_insertions))
# print("Database size  		 : " + str(search_stats['db_size']))
# print("M              		 : " + str(M))
# print("efConstruction 		 : " + str(efConstruction))
# print("efSearch			 	 : " + str(efSearch))
# print("m_L            		 : " + str(m_L))

print("Time : " + str(search_stats['duration_sec']) + " seconds")


## Recall calculation
db_size = len(df_construction.index)
## print(db_size)

target_db = df_construction.iloc[0]
data_db = make_query(target_db.Template)
vec_db = query_to_vector(data_db)
# #print(vec_db)


data_q_size = len(df_insertions.index)
## print(data_q_size)

target_q = df_insertions.iloc[0]
noisy_tpl_q = iris_with_noise(target_q.Template, noise_level=noise_level)
data_q = make_query(noisy_tpl_q)



## Exhaustive search using distance
dist_arr = []
def knn(data, queries, k):
	_data_q = []
	for list in data_q:
		_data_q.append(list)
		## print(_data_q)
		_distance = distance(_data_q, vec_db)
		## print(_distance)
		dist_arr.append(_distance)
		_data_q.remove(list)
	## print(dist_arr)    
	nearest_neighbors = np.argsort(dist_arr)[:k]
	return nearest_neighbors


nearest_neighbors = knn(vec_db, data_q, K)
## print(nearest_neighbors)


## Print the results
for i, neighbors in enumerate(nearest_neighbors):
    print(f"Query {i} Nearest Neighbors: {dist_arr[neighbors]}")

      
## Measure recall

# ## Get ID values from res
_res_insertion = [x[1] for x in res]
# ## print(_res_insertion)
_index = [[i] for i in _res_insertion]
## print(_index)

if nearest_neighbors[0] == _index[0]:
    print("Correct!")
else:
    print("Not correct!")  



## Get Distance values from res
_res_insertion = [x[0] for x in res]
label = [[i] for i in _res_insertion]
correct_label = [[i] for i in dist_arr]

correct = 0
for i in range(K):
    if label == correct_label:
        correct += 1
        break

# print("recall is :", float(correct)/(K*n_insertions))




