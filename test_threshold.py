from random import randint
import pandas as pd

import hnsw

from iris_integration import (
    # Generate test templates.
    DIM,
    iris_random,
    iris_with_noise,
    irisint_make_query as make_query,
    irisint_query_to_vector as query_to_vector,
    irisint_distance as distance,
    convert_from_irisint,
)


## HNSW Demo

n_elements = 100
K = 5
m_L = 0.30
M = 128
efConstruction = 128

efSearch = 64
n_insertions = 10
noise_level = 0.30
threshold = 0.36


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
pd.DataFrame([_params])

# ## DBconstrcution
_construction = []
db.reset_stats()
for _ in range(int(n_elements)):
    _tpl_construct = iris_random()
    _query_construct = make_query(_tpl_construct)
    _id_construct = db.insert(_query_construct)
    _construction.append((_id_construct, _tpl_construct))
    ## print(_tpl_construct)
## print(_construction)
 
_construct_stats = db.get_stats()
past_stats().append(_construct_stats)
df_construction = pd.DataFrame(_construction, columns=["ID", "Template"])
pd.DataFrame(past_stats())
# print(df_construction)
print("DB construction is done!")

print("DB construction  : " + str(_construct_stats['duration_sec']) + " seconds")
# print("Inital DB size   : " + str(_construct_stats["db_size"]))
# print("M                : " + str(M))
# print("efConstruction   : " + str(efConstruction))
# print("m_L              : " + str(m_L))
# print("K                : " + str(K))

## Expriments are starting... 
n_count = 0
for number in range(100):
    ## Insertion
    _insertions = []
    db.reset_stats()
    for _ in range(int(n_insertions)):
        _tpl = iris_random()
        _query = make_query(_tpl)
        _id = db.insert(_query)
        _insertions.append((_id, _tpl))

    insert_stats = db.get_stats()
    past_stats().append(insert_stats)
    df_insertions = pd.DataFrame(_insertions, columns=["ID", "Template"])
    # pd.DataFrame(past_stats())
    # print("Time             : " + str(insert_stats['duration_sec']) + " seconds")


    ## Search
    target = df_insertions.iloc[0]
    noisy_tpl = iris_with_noise(target.Template, noise_level=noise_level)
    query = make_query(noisy_tpl)

    nearest_dist, nearest_id = db.search(query, K=K, ef=efSearch)[0]
    if nearest_dist < threshold:
        n_count += 1

    search_stats = db.get_stats()
    # print("Time             : " + str(search_stats['duration_sec']) + " seconds")

print("Last DB size     : " + str(db.get_stats()["db_size"])) 
print("match: " + str(n_count))
# print("nonmatch: " + str(noncount))
         
# print("All experiments are done!")


## search each item in DB
n_threshold = 0 
for _ in range(len(db.vectors)):
    target_idx = randint(0, len(db.vectors)-1)
    target = convert_from_irisint(db.vectors[target_idx], DIM)
    noisy = iris_with_noise(target, noise_level=noise_level)
    query = make_query(noisy)
    nearest_dist, nearest_id = db.search(query, K=K, ef=efSearch)[0]
    if nearest_dist < threshold:
        n_threshold += 1

print(f"Threshold: {n_threshold} out of {db.get_stats()['db_size']} elements")