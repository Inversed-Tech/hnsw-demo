import sys
import hnsw

from util import (
    dump_hnsw_index,
    insert_iris,
)

from iris_integration import (
    iris_random,
    iris_with_noise,
    int_distance,
)

IRIS_DIM = (2, 32, 200)


def print_help_message():
    print(f"""\
Usage:
  {sys.argv[0]} M ef_construction m_L index_size output_file

  Generate an HNSW index with specified parameters and size and serialize to
  an output file using the Python pickle module.  Reminder: deserializing
  of pickle data can result in arbitrary code execution, so ensure that any
  pickle data comes from a trusted source.

Arguments:
  M                Connectivity value of HNSW graph construction
  ef_construction  Exploration factor used for graph insertions
  m_L              Constant defining the size of successive layers, has a
                   recommended value of 1/ln(M)
  index_size       Number of random entries to insert into index
  output_file      Filename in the current directory to serialize index to""")


def main():
    if len(sys.argv) != 6 or (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print_help_message()
        exit(0)

    M = int(sys.argv[1])
    ef_construction = int(sys.argv[2])
    m_L = float(sys.argv[3])
    index_size = int(sys.argv[4])
    output_file = str(sys.argv[5])

    print(f"Generating index with {index_size} entries...")
    hnsw_index = hnsw.HNSW(
        M=M,
        efConstruction=ef_construction,
        m_L=m_L,
        distance_func=int_distance,
    )

    print("Inserted entries:")
    for i in range(index_size):
        _tpl = iris_random(IRIS_DIM)
        _id = insert_iris(hnsw_index, _tpl)
        if i % 100 == 99:
            print(f"  {i+1}")

    print(f"Writing index to file '{output_file}'")
    dump_hnsw_index(hnsw_index, output_file)

    print("Done!")


if __name__=="__main__":
    main()
    exit(0)
