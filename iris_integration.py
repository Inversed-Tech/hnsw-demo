import streamlit as st
import numpy as np

import iris
from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.utils import hamming_distance

FAST = True
if FAST:
    # 1_600 bits.
    MAX_ROT = 4
    DIM = (2, 32 // 2, 200 // 4)
else:
    # 12_800 bits.
    MAX_ROT = 15
    DIM = (2, 32, 200)


def iris_random(dim=DIM) -> IrisTemplate:
    return IrisTemplate(
        iris_codes=[
            np.random.randint(0, 2, DIM[1:], dtype=np.bool_) for _ in range(DIM[0])
        ],
        mask_codes=[np.ones(DIM[1:], dtype=np.bool_) for _ in range(DIM[0])],
    )


def iris_with_noise(tpl: IrisTemplate, noise_level=0.25) -> IrisTemplate:
    iris_codes = []
    for iris_code in tpl.iris_codes:
        noise = np.random.uniform(0, 1, iris_code.shape) < noise_level
        noisy = iris_code ^ noise
        iris_codes.append(noisy)
    return IrisTemplate(iris_codes=iris_codes, mask_codes=tpl.mask_codes)


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


# Precompute all rotations as a query.
def irisnp_make_query(tpl: IrisTemplate) -> list[tuple[np.ndarray, np.ndarray]]:
    all_rotations = [_rotated(tpl, rot) for rot in range(-MAX_ROT, MAX_ROT + 1)]
    query = [_tpl_to_numpy(x) for x in all_rotations]
    return query


# Distance between each precomputed rotation and a vector.
def irisnp_distance(
    query: list[tuple[np.ndarray, np.ndarray]], vector: tuple[np.ndarray, np.ndarray]
) -> float:
    return min(_np_distance(x, vector) for x in query)


# Store the no-rotation vector.
def irisnp_query_to_vector(
    query: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    return query[MAX_ROT]


def _rotated(tpl: IrisTemplate, rot: int) -> IrisTemplate:
    return IrisTemplate(
        iris_codes=[np.roll(iris_code, rot, axis=1) for iris_code in tpl.iris_codes],
        mask_codes=[np.roll(mask_code, rot, axis=1) for mask_code in tpl.mask_codes],
    )


def _tpl_to_numpy(tpl: IrisTemplate) -> tuple[np.ndarray, np.ndarray]:
    code = np.packbits(np.concatenate(tpl.iris_codes))
    mask = np.packbits(np.concatenate(tpl.mask_codes))
    return (code, mask)


def _np_distance(
    x: tuple[np.ndarray, np.ndarray], y: tuple[np.ndarray, np.ndarray]
) -> float:
    # TODO: try bitwise_count in Numpy 2.0.

    x_code, x_mask = (np.unpackbits(x[0]), np.unpackbits(x[1]))
    y_code, y_mask = (np.unpackbits(y[0]), np.unpackbits(y[1]))

    mask = x_mask & y_mask
    diff = (x_code ^ y_code) & mask

    return np.sum(diff) / np.sum(mask, dtype=float)


# --- Variant `irisint` with pre-computed rotations in Python big int ---
# The fastest.


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
