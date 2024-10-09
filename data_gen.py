import dill

from iris_integration import (
    iris_random,

)

## In order to generate the same data 
## use np.random.seed(0) in the line 18 of iris_integration.py
##################### Decide DB construction via hnsw
n_elements = 100              # the number of elements (n_elements) loaded to DB
file = 'data100.txt'          # file name


_construction = []
for _ in range(int(n_elements)):
    _tpl_construct = iris_random()
    _construction.append(_tpl_construct)

 
print("Data generation is done!")

## writing the data to the file
with open(file, 'wb') as f:
    dill.dump(_construction, f)

print("DB construction is written to a txt file!")
