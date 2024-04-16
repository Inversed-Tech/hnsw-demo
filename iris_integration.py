import streamlit as st
import numpy as np

import iris
from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.utils import hamming_distance


def iris_random(dim=(32, 200)):
    iris_codes = [
        np.random.randint(0, 2, dim, dtype=np.bool_),
        np.random.randint(0, 2, dim, dtype=np.bool_),
    ]
    mask_codes = [
        np.ones(dim, dtype=np.bool_),
        np.ones(dim, dtype=np.bool_),
    ]
    return IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes)


def iris_distance(x: IrisTemplate, y: IrisTemplate) -> float:
    return hamming_distance(x, y, 15)[0]


# Query and stored Vectors are the same here.
def iris_query_to_vector(x: IrisTemplate) -> IrisTemplate:
    return x


def iris_with_noise(base: IrisTemplate, level=0.25):
    iris_codes = []
    for iris_code in base.iris_codes:
        noise = np.random.uniform(0, 1, iris_code.shape) < level
        noisy = iris_code ^ noise
        iris_codes.append(noisy)
    return IrisTemplate(iris_codes=iris_codes, mask_codes=base.mask_codes)
