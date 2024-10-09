import pandas as pd

import hnsw

from iris_integration import (
    iris_random,
    iris_with_noise,
    irisint_make_query as make_query,
    irisint_query_to_vector as query_to_vector,
    irisint_distance as distance,
)

## HNSW Demo
K = 5
m_L = 0.30

n_elements = 100      # the number of elts for constructing DB
M = 128               # M value
efConstruction = 128  # efConstruction


n_insertions = 10     # the number of elts to insert
efSearch = 128        # efSearch value
noise_level = 0.30    # iris codes can be noisy

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

print("Inital DB size   : " + str(_construct_stats["db_size"]))
# print("M                : " + str(M))
# print("efConstruction   : " + str(efConstruction))
# print("m_L              : " + str(m_L))
# print("K                : " + str(K))
print("DB construction  : " + str(_construct_stats['duration_sec']) + " seconds\n")

## Expriments are starting... 
count = 0
noncount = 0

if n_insertions > 0:
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
    pd.DataFrame(past_stats())

print("Insertion is done!")
print(f"DB size after insertions    : {insert_stats['db_size']}") 
print(f"Insertion time              : {insert_stats['duration_sec']} seconds\n")



 ## Search for one elt in DB
target = df_insertions.iloc[0] # change "0" of the target elt in DB to search
noisy_tpl = iris_with_noise(target.Template, noise_level=noise_level)

db.reset_stats()
query = make_query(noisy_tpl)
res = db.search(query, K, ef=efSearch)
search_stats = db.get_stats()
df_found = pd.DataFrame(res, columns=["Distance", "ID"]) 
## print("result: " + str(res))

# print(f"Searching for vector `ID {target.ID}`, with `{int(noise_level*100)}%` noise.")
# print(f"`Top {K}` Nearest Neighbors:")
# print(df_found)
found = target.ID in df_found.ID.values
if found == True:
    count += 1
    # print("Found")
else:
    noncount += 1
    # print("Not found")

print("Search is done!")
# print("Last DB size     : " + str(db.get_stats()["db_size"]))
print("Search time      : " + str(search_stats['duration_sec']) + " seconds")      
# print("efSearch         : " + str(efSearch))
# print("Query size       : " + str(n_insertions))
# print("Layer size       : " + str(search_stats['n_layers']))
# print("Search number    : " + str(search_stats['n_searches']))
# print("Distance number  : " + str(search_stats['n_distances']))
# print("Comparison number: " + str(search_stats['n_comparisons']))
print(f"\nMatch result for iris code {target.ID}")
print("match   : " + str(count))
print("nonmatch: " + str(noncount))
