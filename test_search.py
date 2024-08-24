import sys
import hnsw

from random import randint

from util import (
    load_hnsw_index,
    find_iris,
)

from iris_integration import (
    iris_with_noise,
    int_distance,
    convert_from_irisint,
)

IRIS_DIM = (2, 32, 200)


def print_help_message():
    print(f"""\
Usage:
  {sys.argv[0]} input_file noise ef_search repetitions

  Load a previously generated HNSW index from file and attempt to find some
  number of queries generated from random entries in the index with added noise.

Arguments:
  input_file   Filename in the current directory to retrieve index from
  noise        Likelihood to flip each bit of an iris code, between 0 and 1
  ef_search    Exploration factor used for graph search
  repetitions  Number of random searches to conduct in the index""")


def main():
    if len(sys.argv) != 5 or (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print_help_message()
        exit(0)

    input_file = str(sys.argv[1])
    noise = float(sys.argv[2])
    ef_search = int(sys.argv[3])
    repetitions = int(sys.argv[4])

    print(f"Reading index from file...")
    hnsw_index = load_hnsw_index(input_file)

    print(f"Searching for {repetitions} randomized index entries...")
    n_found = 0
    for _ in range(repetitions):
        target_idx = randint(0, len(hnsw_index.vectors)-1)
        target = convert_from_irisint(hnsw_index.vectors[target_idx], IRIS_DIM)
        noisy = iris_with_noise(target, noise_level=0.30)
        result = find_iris(hnsw_index, noisy, 32, 0.36)
        if result is not None:
            n_found += 1
    print(f"Found {n_found} of {repetitions} randomized index entries")

    print("Done!")


if __name__=="__main__":
    main()
    exit(0)
