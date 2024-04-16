import streamlit as st
import numpy as np

import iris
from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.utils import hamming_distance


def random_iris(dim=(32, 200)):
    iris_codes = [
        np.random.randint(0, 2, dim, dtype=np.bool_),
        np.random.randint(0, 2, dim, dtype=np.bool_),
    ]
    mask_codes = [
        np.ones(dim, dtype=np.bool_),
        np.ones(dim, dtype=np.bool_),
    ]
    return IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes)


def distance_iris(x, y):
    return hamming_distance(x, y, 15)[0]
