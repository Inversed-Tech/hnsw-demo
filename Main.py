import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

import hnsw
from iris_integration import (
    # Generate test templates.
    DIM,
    MAX_ROT,
    iris_random,
    iris_with_noise,
    # Test implementation correctness.
    # iris_test,
    # Choose an implementation: `iris_`, `irisnp_`, `irisint_`.
    irisint_make_query as make_query,
    irisint_query_to_vector as query_to_vector,
    irisint_distance as distance,
)

"# HNSW Demo"

# with st.expander("**📊 Database Parameters**", expanded=False):
with st.sidebar:
    "## 📊 Database Parameters"

    f"**Dimension: `{DIM}`**"
    f"**Rotations: `± {MAX_ROT}`**"

    M = int(
        st.number_input(
            "**M** \n- Number of neighbors in the graph. *E.g. 16 - 256.*",
            2,
            value=128,
            step=8,
        )
    )
    efConstruction = int(
        st.number_input(
            "**efConstruction** \n- Breadth of search during insertion. *E.g. 16 - 256.*",
            1,
            value=128,
            step=8,
        )
    )
    m_L = st.number_input(
        "**m_L** \n- Factor for the number of layers. *E.g. 0.3 or 1/ln(M).*",
        0.1,
        1.0,
        value=0.3,
        step=0.1,
    )

    @st.cache_resource
    def make_db():
        return hnsw.HNSW(
            M=M,
            efConstruction=efConstruction,
            m_L=m_L,
            distance_func=distance,
            query_to_vector_func=query_to_vector,
        )

    @st.cache_resource
    def past_stats():
        return []

    if st.button("Reset Database", type="primary"):
        make_db.clear()  # type: ignore
        past_stats.clear()  # type: ignore

    f"**Current DB:**"
    db = make_db()
    _params = db.get_params()
    _params["Current Size"] = db.get_stats()["db_size"]
    st.dataframe(pd.DataFrame([_params]).T)


sta, stb = st.columns([6, 6], gap="large")

with sta:
    "## 🧩 Insertion"

    n_insertions = st.number_input("Insert Vectors", 1, value=10, step=100)

    _insertions = []
    db.reset_stats()
    for _ in range(int(n_insertions)):
        _tpl = iris_random()
        _query = make_query(_tpl)
        _id = db.insert(_query)
        _insertions.append((_id, _tpl))
    _insert_stats = db.get_stats()
    past_stats().append(_insert_stats)
    df_insertions = pd.DataFrame(_insertions, columns=["ID", "Template"])

    f"Auto-Inserted `{len(df_insertions)}` more vectors. Stats of the last few runs:"
    st.dataframe(pd.DataFrame(past_stats())[-3:].T)

    st.button("Insert More")


with stb:
    "## 🔎 Search"
    efSearch = int(
        st.number_input(
            "**efSearch:** Breadth of search. Minimum `K` for top-K results. *E.g. 16 - 256.*",
            1,
            value=128,
            step=8,
        )
    )
    K = 5
    noise_level = 0.30

    target = df_insertions.iloc[0]
    noisy_tpl = iris_with_noise(target.Template, noise_level=noise_level)

    db.reset_stats()
    query = make_query(noisy_tpl)
    res = db.search(query, K, ef=efSearch)
    search_stats = db.get_stats()
    df_found = pd.DataFrame(res, columns=["Distance", "ID"])
    df_found.index.name = "Rank"

    f"Searching for vector `ID {target.ID}`, with `{int(noise_level*100)}%` noise."
    f"`Top {K}` Nearest Neighbors:"
    st.dataframe(df_found)
    found = target.ID in df_found.ID.values
    st.write("✅ Found!" if found else f"❌ Not Found!")


"### 📈 Insertion Stats"
df_stats = pd.DataFrame(past_stats())
# DB size is reported after the insertions so we adjust it to the middle of the insertion run.
df_stats["db_size_during_insertions"] = (
    df_stats["db_size"] - df_stats["n_insertions"] // 2
)
df_stats["n_distances_per_insertion"] = (
    df_stats["n_distances"] / df_stats["n_insertions"]
)
df_stats["n_comparisons_per_insertion"] = (
    df_stats["n_comparisons"] / df_stats["n_insertions"]
)
st.line_chart(
    df_stats,
    x="db_size_during_insertions",
    y=["n_distances_per_insertion", "n_comparisons_per_insertion"],
)
