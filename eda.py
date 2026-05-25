import pandas as pd
import numpy as np


def show_summary(df, st):
    """Simple statistical summary."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        st.dataframe(desc, use_container_width=True)
    else:
        st.info("No numeric columns found for statistical summary.")