import json
import numpy as np
import pandas as pd
from typing import Any
from iris_integration import _np_to_bigint
from hnsw import HNSW

def parse_string_to_dict(input_string: str) -> Any:
    """
    Parses a JSON string into a Python dict (requires to fit to the Rust copy-out).

    Args:
        input_string (str): The string to be parsed.

    Returns:
        Any: Parsed dict if successful, None if parsing fails.
    """
    try:
        parsed_dict = json.loads(input_string)
        return parsed_dict
    except json.JSONDecodeError as e:
        print(f"Error parsing string: {e}")
        return None


def copy_in(db: HNSW, vectors: pd.DataFrame, links: pd.DataFrame, entries: pd.DataFrame) -> None:
    """
    Copies data into the HNSW database by updating vectors, layers, insertions, and entry point.

    Args:
        db (HNSW): The HNSW object to be updated.
        vectors (pd.DataFrame): DataFrame containing vector data.
        links (pd.DataFrame): DataFrame containing links data.
        entries (pd.DataFrame): DataFrame containing entry point data.
    """
    _update_vectors(db, vectors)
    _update_layers(db, links)
    _update_n_insertions(db, len(vectors))
    _update_entry_point(db, entries)


def _update_entry_point(db: HNSW, entries: pd.DataFrame) -> None:
    """
    Updates the entry point in the HNSW database.

    Args:
        db (HNSW): The HNSW object where the entry point is updated.
        entries (pd.DataFrame): DataFrame with the entry points.
    """
    db.entry_point[:] = entries['id'].values.tolist()


def _update_n_insertions(db: HNSW, num_insertions: int) -> None:
    """
    Updates the number of insertions in the HNSW database.

    Args:
        db (HNSW): The HNSW object where the number of insertions is updated.
        num_insertions (int): Number of insertions according to the imported data
    """
    db.n_insertions = num_insertions


def _update_layers(db: HNSW, links: pd.DataFrame) -> None:
    """
    Updates the layers in the HNSW database.

    Args:
        db (HNSW): The HNSW object where layers are updated.
        links (pd.DataFrame): DataFrame containing all links across all layers.
    """
    def process_links(df: DataFrame) -> dict:
        def process_row(row: pd.Series) -> list:
            return [(item[1], item[0]) for item in row['queue']]
        
        df['processed_queue'] = df['links'].apply(lambda x: process_row(x))
        return pd.Series(df['processed_queue'].values, index=df['source_ref']).to_dict()

    db.layers = links.groupby('layer').apply(process_links).sort_index(ascending=True).tolist()


def _update_vectors(db: HNSW, vectors: pd.DataFrame) -> None:
    """
    Updates the vectors in the HNSW database.

    Args:
        db (HNSW): The HNSW object where vectors are updated.
        vectors (pd.DataFrame): DataFrame containing vector data.
    """
    def process_vectors(data_dict: Any) -> tuple:
        data = np.array(data_dict['data']['data'])
        bi = np.where(data == -1, 1, 0) # Following methodology defined in Rust
        mi = np.where(data != 0, 1, 0) # Following methodology defined in Rust
        return (_np_to_bigint(bi.astype(np.bool_)), _np_to_bigint(mi.astype(np.bool_)))

    vectors_sorted = vectors.sort_values(by='id')
    processed_points = vectors_sorted['point'].apply(process_vectors)
    db.vectors = processed_points.tolist()
