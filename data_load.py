import dill
import pandas as pd

import hnsw

from iris_integration import (
    DIM,
    iris_with_noise,
    irisint_make_query as make_query,
    irisint_query_to_vector as query_to_vector,
    irisint_distance as distance,
    convert_from_irisint,
    iris_random,
)


##################### Data loading from a file
file = 'data100.txt'

## HNSW Demo
m_L = 0.30
K = 5
noise_level = 0.30

## In order to have the same data insertion 
## use np.random.seed(1) in the line 18 of iris_integration.py
M = 32                     # M value
efConstruction = 128       # efConstruction value
efSearch = 64              # efSearch value

threshold = 0.36        # when noise_level=0.30, threshold could be ~0.48 

n_insertions = 0

    
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


_data = []
## Data extracting from the file
with open(file, 'rb') as f:
    _data = dill.load(f)

print("\nDatabase is extracted from the file...\n")
## print(_data)

db = make_db()

db.reset_stats()
_construction = []
for i in range(len(_data)):
    _query_construct = make_query(_data[i])
    _id_construct = db.insert(_query_construct)
    _construction.append((_id_construct, _data[i]))

construct_stats = db.get_stats()
past_stats().append(construct_stats)
df_construction = pd.DataFrame(_construction, columns=["ID", "Template"])
pd.DataFrame(past_stats())
# print(df_construction)


## insertion
if n_insertions > 0:
    _insertions = []
    for _ in range(int(n_insertions)):
        ## TODO: change line 18 if you want to construct a DB
        _tpl = iris_random()
        _query = make_query(_tpl)
        db.reset_stats()
        _id = db.insert(_query)
        _insertions.append((_id, _tpl))

        insert_stats = db.get_stats()
        past_stats().append(insert_stats)
        # print(f"Time     : {insert_stats['duration_sec']} seconds") 

    df_insertions = pd.DataFrame(_insertions, columns=["ID", "Template"])
    pd.DataFrame(past_stats())
    
    # print(f"DB construction  : {insert_stats['duration_sec']} seconds")
    # print(f"Last DB size     : {db.get_stats()['db_size']}") 
    print("\nInsertion is done...\n")
else:
    print(f"\nThere is {n_insertions} insertions...\n")



## search each item in DB

## Loding DB to DataFrame
database = []
for i in range(len(db.vectors)):
    target_idx = i
    target = db.vectors[target_idx] ## (iris, mask)
    database.append((target_idx, target))

df_database = pd.DataFrame(database, columns=["ID", "Template"])


for number in range(10):
    print(f"Search {number+1} is started...")


    n_found = 0 
    n_threshold = 0

    for i in range(len(db.vectors)):
        _cand = df_database.iloc[i]
        cand = convert_from_irisint(_cand.Template, DIM)
        _noisy = iris_with_noise(cand, noise_level=noise_level)
        db.reset_stats()
        query = make_query(_noisy)
        res = db.search(query, K=5, ef=efSearch)
        df_found = pd.DataFrame(res, columns=["Distance", "ID"])
        found = _cand.ID in df_found.ID.values     
        if found == True:
            n_found += 1

        search_stats = db.get_stats()
        past_stats().append(search_stats)         
        # print(f"Time     : {search_stats['duration_sec']} seconds") 
  
        if res[0][0] < threshold: # 0.0 < res[0][0] < threshold
            n_threshold += 1 
        # else:
        #     print(f"{res[0][0]}") 

        ## nearest_dist, nearest_id = db.search(query, K=5, ef=efSearch)[0]  
        ## if res[0][0] == nearest_dist: #0.0 < res[0] < threshold
        ##     print(f"{res[0][0]} and {nearest_dist}")  
        ## # else:
        ## #     print(f"{res[0][0]}") 
    
    print(f"Found    : {n_found} out of {db.get_stats()['db_size']} elements")

    print(f"Threshold: {n_threshold} out of {db.get_stats()['db_size']} elements\n")
    
 

