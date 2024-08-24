import numpy as np

import iris
from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.utils import hamming_distance

FAST = False # FAST = True
if FAST:
    # 1_600 bits.
    MAX_ROT = 4
    DIM = (2, 32 // 2, 200 // 4)
else:
    # 12_800 bits.
    MAX_ROT = 15
    DIM = (2, 32, 200)

# iris_code_version is added to IrisTemplate
def iris_random(dim=DIM) -> IrisTemplate:
    return IrisTemplate(
        iris_codes=[
            np.random.randint(0, 2, dim[1:], dtype=np.bool_) for _ in range(dim[0])
        ],
        mask_codes=[np.ones(dim[1:], dtype=np.bool_) for _ in range(dim[0])], iris_code_version= "v3.0"
    )

# iris_code_version is added to be aligned with 
# the latest version of open-iris (v1.1.1)
def iris_with_noise(tpl: IrisTemplate, noise_level=0.25) -> IrisTemplate:
    iris_codes = []
    for iris_code in tpl.iris_codes:
        noise = np.random.uniform(0, 1, iris_code.shape) < noise_level
        noisy = iris_code ^ noise
        iris_codes.append(noisy)
    return IrisTemplate(iris_codes=iris_codes, mask_codes=tpl.mask_codes, iris_code_version= "v3.0")


# --- Reference variant `iris` using open-iris functions. ---
# Very slow.


# A template is a query.
def iris_make_query(tpl: IrisTemplate) -> IrisTemplate:
    return tpl


# Roll x to all rotations, compare to y, and return the minimum distance.
def iris_distance(query: IrisTemplate, vector: IrisTemplate) -> float:
    return hamming_distance(query, vector, MAX_ROT)[0]


# Store the template, that is the query itself.
def iris_query_to_vector(query: IrisTemplate) -> IrisTemplate:
    return query


# --- Variant `irisnp` with pre-computed rotations in Numpy packed bits. ---
# Not great because numpy does not have a bit_count function.


# Precompute all rotations as a query.
def irisnp_make_query(tpl: IrisTemplate) -> list[tuple[np.ndarray, np.ndarray]]:
    query_all_rotations = [
        _tpl_to_numpy(_rotated(tpl, rot)) for rot in range(-MAX_ROT, MAX_ROT + 1)
    ]
    return query_all_rotations


# Distance between each precomputed rotation and a vector.
def irisnp_distance(
    query: list[tuple[np.ndarray, np.ndarray]], vector: tuple[np.ndarray, np.ndarray]
) -> float:
    # Unpack the vector from irisnp_query_to_vector.
    y_code, y_mask = (np.unpackbits(vector[0]), np.unpackbits(vector[1]))
    return min(_np_distance(x_code, x_mask, y_code, y_mask) for x_code, x_mask in query)


# Store the no-rotation vector from irisnp_make_query. Pack the bits to save memory.
def irisnp_query_to_vector(
    query: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    code, mask = query[MAX_ROT]
    return (np.packbits(code), np.packbits(mask))

# iris_code_version is added to IrisTemplate
def _rotated(tpl: IrisTemplate, rot: int) -> IrisTemplate:
    return IrisTemplate(
        iris_codes=[np.roll(iris_code, rot, axis=1) for iris_code in tpl.iris_codes],
        mask_codes=[np.roll(mask_code, rot, axis=1) for mask_code in tpl.mask_codes],
        iris_code_version= "v3.0"
    )


def _tpl_to_numpy(tpl: IrisTemplate) -> tuple[np.ndarray, np.ndarray]:
    code = np.concatenate(tpl.iris_codes).flatten()
    mask = np.concatenate(tpl.mask_codes).flatten()
    return (code, mask)


def _np_distance(
    x_code: np.ndarray,
    x_mask: np.ndarray,
    y_code: np.ndarray,
    y_mask: np.ndarray,
) -> float:
    mask = x_mask & y_mask
    diff = (x_code ^ y_code) & mask
    return float(np.sum(diff)) / float(np.sum(mask))


# --- Variant `irisint` with pre-computed rotations in Python big int ---
# The fastest by far.
# This may use CPU POPCNT instructions. It works on 30 bits limbs.


# Precompute all rotations as a query.
def irisint_make_query(tpl: IrisTemplate) -> list[tuple[int, int]]:
    all_rotations = [_rotated(tpl, rot) for rot in range(-MAX_ROT, MAX_ROT + 1)]
    query = [_tpl_to_bigint(x) for x in all_rotations]
    return query


# Distance between each precomputed rotation and a vector.
def irisint_distance(query: list[tuple[int, int]], y: tuple[int, int]) -> float:
    return min(_int_distance(x, y) for x in query)


# Store the no-rotation vector.
def irisint_query_to_vector(query: list[tuple[int, int]]) -> tuple[int, int]:
    return query[MAX_ROT]


def _tpl_to_bigint(tpl: IrisTemplate) -> tuple[int, int]:
    code = _np_to_bigint(np.concatenate(tpl.iris_codes))
    mask = _np_to_bigint(np.concatenate(tpl.mask_codes))
    return (code, mask)


def _np_to_bigint(a: np.ndarray) -> int:
    assert a.dtype == np.bool_
    x = 0
    a_bytes = np.packbits(a, axis=None)
    for b in a_bytes:
        x = (x << 8) | int(b)
    return x


def _int_distance(x: tuple[int, int], y: tuple[int, int]):
    (x_code, x_mask) = x
    (y_code, y_mask) = y

    mask = x_mask & y_mask
    diff = (x_code ^ y_code) & mask

    return diff.bit_count() / mask.bit_count()


# --- Test of the implementations. ---
def iris_test():
    tpl = iris_random()

    # Storage formats of the template.
    vector = iris_query_to_vector(iris_make_query(tpl))
    vector_np = irisnp_query_to_vector(irisnp_make_query(tpl))
    vector_int = irisint_query_to_vector(irisint_make_query(tpl))

    # Exact matches produce zero distance.
    assert iris_distance(iris_make_query(tpl), vector) == 0.0
    assert irisnp_distance(irisnp_make_query(tpl), vector_np) == 0.0
    assert irisint_distance(irisint_make_query(tpl), vector_int) == 0.0

    # Noisy query produces the same non-zero distance for all implementations.
    noisy_tpl = iris_with_noise(tpl)

    query = iris_make_query(noisy_tpl)
    query_np = irisnp_make_query(noisy_tpl)
    query_int = irisint_make_query(noisy_tpl)

    dist = iris_distance(query, vector)
    dist_np = irisnp_distance(query_np, vector_np)
    dist_int = irisint_distance(query_int, vector_int)

    assert 0.0 < dist < 0.5
    assert dist == dist_np == dist_int

    print("Test iris distance implementations: ✅")
