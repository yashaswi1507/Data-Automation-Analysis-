import pandas as pd
import numpy as np


def show_summary(df, st):
    """Simple statistical summary."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric  = [c for c in df.columns if c not in numeric_cols]

    # Explain excluded columns
    if non_numeric:
        st.caption(
            f"ℹ️ Showing {len(numeric_cols)} numeric column(s). "
            f"Text/category columns ({', '.join(non_numeric[:4])}{'...' if len(non_numeric)>4 else ''}) "
            f"are excluded — they have no averages or ranges."
        )

    if not numeric_cols:
        st.info("No numeric columns found for statistical summary.")
        return

    # Unit hints
    unit_hints = {}
    for c in numeric_cols:
        cl = c.lower()
        if any(k in cl for k in ["price","salary","revenue","cost","income","amount","fee","wage","pay"]):
            unit_hints[c] = "💰 currency"
        elif any(k in cl for k in ["pct","percent","rate","ratio"]):
            unit_hints[c] = "📊 percentage"
    if unit_hints:
        st.caption("💡 Detected: " + " | ".join([f"**{c}** → {v}" for c,v in unit_hints.items()]))

    # Describe with plain English labels
    desc = df[numeric_cols].describe().round(2)
    desc.index = desc.index.map({
        "count": "Count (non-missing)",
        "mean":  "Mean (average)",
        "std":   "Std Dev (spread)",
        "min":   "Min",
        "25%":   "25th Percentile",
        "50%":   "50th Percentile (median)",
        "75%":   "75th Percentile",
        "max":   "Max",
    })
    st.dataframe(desc, use_container_width=True)

    with st.expander("❓ What do these terms mean?", expanded=False):
        st.markdown("""
| Term | Meaning |
|------|---------|
| **Count** | Non-missing rows |
| **Mean** | Average value |
| **Std Dev** | How spread out the values are — small = consistent, large = varies a lot |
| **Min / Max** | Smallest and largest value |
| **25th Percentile** | 25% of values fall below this |
| **50th Percentile** | Middle value — half above, half below (Median) |
| **75th Percentile** | 75% of values fall below this |
        """)

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.divider()
        st.subheader("⚠️ Missing Values")
        miss_df = missing.reset_index()
        miss_df.columns = ["Column", "Missing Count"]
        miss_df["Missing %"] = (miss_df["Missing Count"] / len(df) * 100).round(1).astype(str) + "%"
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values!")
