import pandas as pd
import numpy as np
import re


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION ALIASES
# Every word a user might type → internal op name
# ─────────────────────────────────────────────────────────────────────────────

OP_ALIASES = {
    # mean
    "average": "mean", "avg": "mean", "mean": "mean",

    # sum
    "total": "sum", "sum": "sum", "overall": "sum", "combined": "sum",

    # count
    "count": "count", "how many": "count", "number of": "count",
    "frequency": "count", "occurrences": "count",

    # max
    "maximum": "max", "max": "max", "highest": "max",
    "largest": "max", "most": "max", "top": "max",
    "biggest": "max", "best": "max",

    # min
    "minimum": "min", "min": "min", "lowest": "min",
    "smallest": "min", "least": "min", "bottom": "min",
    "worst": "min", "fewest": "min",

    # median
    "median": "median", "middle": "median", "mid": "median",

    # std
    "std": "std", "standard deviation": "std", "deviation": "std",
    "variance": "std",

    # unique
    "unique": "unique", "distinct": "unique", "different": "unique",
}

# Words to strip before column matching
STOP_WORDS = {
    "what", "is", "the", "of", "in", "for", "by", "and", "or",
    "a", "an", "show", "me", "give", "find", "get", "tell",
    "per", "each", "every", "all", "column", "field", "value",
    "values", "data", "dataset", "table", "rows",
}


# ─────────────────────────────────────────────────────────────────────────────
# FUZZY COLUMN FINDER
# Finds the best matching column from user query text.
# Works even if user types partial name, different case, or drops spaces.
# ─────────────────────────────────────────────────────────────────────────────

def _find_column(text, cols):
    """
    Returns (original_col_name, score) or (None, 0).
    Tries exact → contains → word overlap → partial token match.
    """
    text_clean = text.lower().strip()

    best_col   = None
    best_score = 0

    for col in cols:
        col_lower = col.lower().strip()
        col_words = set(re.split(r"[\s_\-/]+", col_lower))

        # 1. Exact match
        if col_lower == text_clean:
            return col, 100

        # 2. Column name fully contained in query text
        if col_lower in text_clean:
            score = 90 - (len(text_clean) - len(col_lower))
            if score > best_score:
                best_score = score
                best_col   = col

        # 3. Query contains all words of the column name
        text_words = set(re.split(r"\s+", text_clean))
        overlap    = col_words & text_words
        if overlap and len(overlap) == len(col_words):
            score = 80
            if score > best_score:
                best_score = score
                best_col   = col

        # 4. Partial word overlap (at least one content word matches)
        content_col_words = col_words - STOP_WORDS
        content_txt_words = text_words - STOP_WORDS
        shared = content_col_words & content_txt_words
        if shared:
            score = 50 + len(shared) * 10
            if score > best_score:
                best_score = score
                best_col   = col

    return best_col, best_score


# ─────────────────────────────────────────────────────────────────────────────
# PARSE QUERY
# ─────────────────────────────────────────────────────────────────────────────

def parse_query(query, df):
    """
    Returns (op, target_col, group_col).
    All three can be None if not detected.
    """
    original_query = query.strip()
    q              = original_query.lower()
    cols           = df.columns.tolist()

    # ── Step 1: Detect operation ─────────────────────────────────
    op = None

    # Check multi-word aliases first (longest match wins)
    for alias in sorted(OP_ALIASES, key=len, reverse=True):
        if alias in q:
            op = OP_ALIASES[alias]
            break

    # ── Step 2: Split on "by" / "per" / "for each" / "grouped by" ─
    group_col  = None
    target_col = None

    # Patterns that introduce a grouping dimension
    by_patterns = [
        r"\bby\b", r"\bper\b", r"\bfor each\b",
        r"\bgrouped by\b", r"\bacross\b", r"\bbreakdown by\b",
    ]

    split_idx  = None
    split_text = None

    for pat in by_patterns:
        m = re.search(pat, q)
        if m:
            split_idx  = m.start()
            split_text = q[m.end():].strip()
            break

    if split_text is not None:
        left_text  = q[:split_idx]
        right_text = split_text

        target_col, _ = _find_column(left_text, cols)
        group_col,  _ = _find_column(right_text, cols)

        # If target not found on left, search full query minus group
        if target_col is None:
            remaining = q.replace(group_col.lower() if group_col else "", "")
            target_col, _ = _find_column(remaining, cols)
    else:
        # No grouping — find single target column from whole query
        target_col, _ = _find_column(q, cols)

    # ── Step 3: Fallback — try every col and pick best match ─────
    if target_col is None:
        best_col   = None
        best_score = 0
        for col in cols:
            _, score = _find_column(col.lower(), cols)
            col_lower = col.lower()
            if col_lower in q:
                score = 90
            if score > best_score:
                best_score = score
                best_col   = col
        target_col = best_col

    # ── Step 4: Default op if still None ─────────────────────────
    # Infer from column type
    if op is None and target_col is not None:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            op = "mean"   # sensible default for numeric
        else:
            op = "count"  # sensible default for categorical

    return op, target_col, group_col


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE QUERY
# ─────────────────────────────────────────────────────────────────────────────

def execute_query(df, op, target, group):
    """
    Runs the query and returns a result (scalar or Series).
    Returns a dict with 'result', 'error', 'query_desc' keys.
    """
    if op is None and target is None:
        return {
            "result": None,
            "error":  "Could not understand the query. Try: 'average salary by department' or 'total sales by region'.",
            "query_desc": "",
        }

    if target is None:
        return {
            "result": None,
            "error":  "Could not identify which column to analyse. Please mention the column name in your query.",
            "query_desc": "",
        }

    # Map internal op → pandas method
    OP_MAP = {
        "mean":   lambda s: s.mean(),
        "sum":    lambda s: s.sum(),
        "max":    lambda s: s.max(),
        "min":    lambda s: s.min(),
        "median": lambda s: s.median(),
        "count":  lambda s: s.count(),
        "std":    lambda s: s.std(),
        "unique": lambda s: s.nunique(),
    }

    if op not in OP_MAP:
        return {
            "result": None,
            "error":  f"Unknown operation '{op}'. Supported: average, total, count, max, min, median, std, unique.",
            "query_desc": "",
        }

    fn = OP_MAP[op]

    try:
        if group:
            result      = df.groupby(group)[target].agg(op)
            query_desc  = f"{op.upper()} of '{target}' grouped by '{group}'"
        else:
            series      = df[target]
            result      = fn(series)
            query_desc  = f"{op.upper()} of '{target}'"

        return {
            "result":     result,
            "error":      None,
            "query_desc": query_desc,
        }

    except Exception as e:
        return {
            "result": None,
            "error":  f"Query failed: {str(e)}",
            "query_desc": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE INSIGHT
# ─────────────────────────────────────────────────────────────────────────────

def generate_query_insight(result_dict, target, group):
    """
    Generates a human-readable insight string from the query result.
    Returns a list of insight strings.
    """
    result = result_dict.get("result")
    error  = result_dict.get("error")

    if error or result is None:
        return []

    insights = []

    # ── Grouped result (Series) ───────────────────────────────────
    if isinstance(result, pd.Series) and len(result) > 0:
        result_sorted = result.sort_values(ascending=False)

        top_name  = result_sorted.index[0]
        top_val   = result_sorted.iloc[0]
        bot_name  = result_sorted.index[-1]
        bot_val   = result_sorted.iloc[-1]

        insights.append(
            f"🥇 Highest '{target}': **{top_name}** with {_fmt(top_val)}"
        )
        insights.append(
            f"🔻 Lowest '{target}': **{bot_name}** with {_fmt(bot_val)}"
        )

        if len(result_sorted) > 2:
            spread = top_val - bot_val
            insights.append(
                f"📊 Range across {len(result_sorted)} groups: {_fmt(spread)}"
            )

        # Check if one group dominates (>50% of total)
        try:
            total = result_sorted.sum()
            if total != 0:
                top_pct = (top_val / total) * 100
                if top_pct > 50:
                    insights.append(
                        f"⚠️ '{top_name}' accounts for {top_pct:.1f}% of total — significantly dominant."
                    )
        except Exception:
            pass

    # ── Scalar result ─────────────────────────────────────────────
    else:
        try:
            val = float(result)
            insights.append(f"📌 Result: **{_fmt(val)}**")
        except Exception:
            insights.append(f"📌 Result: **{result}**")

    return insights


def _fmt(val):
    """Format number nicely — comma sep, max 2 decimal places."""
    try:
        f = float(val)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except Exception:
        return str(val)