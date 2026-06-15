"""
Query Engine — Improved NLP
Better intent detection, handles complex queries,
synonyms, typos, Hindi-English mixed queries.
"""
import pandas as pd
import numpy as np
import re


# ─────────────────────────────────────────────────────────────
# OPERATION ALIASES — expanded with common variations
# ─────────────────────────────────────────────────────────────

OP_ALIASES = {
    # mean
    "average": "mean", "avg": "mean", "mean": "mean",
    "arithmetic mean": "mean", "typical": "mean",

    # sum
    "total": "sum", "sum": "sum", "overall": "sum",
    "combined": "sum", "aggregate": "sum", "add up": "sum",
    "cumulative": "sum", "grand total": "sum",

    # count
    "count": "count", "how many": "count", "number of": "count",
    "frequency": "count", "occurrences": "count", "instances": "count",
    "records": "count", "entries": "count", "rows": "count",
    "how much": "count",

    # max
    "maximum": "max", "max": "max", "highest": "max",
    "largest": "max", "most": "max", "top": "max",
    "biggest": "max", "best": "max", "peak": "max",
    "greatest": "max", "upper": "max", "ceiling": "max",

    # min
    "minimum": "min", "min": "min", "lowest": "min",
    "smallest": "min", "least": "min", "bottom": "min",
    "worst": "min", "fewest": "min", "floor": "min",
    "weakest": "min",

    # median
    "median": "median", "middle": "median", "mid": "median",
    "midpoint": "median", "50th percentile": "median",

    # std
    "std": "std", "standard deviation": "std", "deviation": "std",
    "variance": "std", "spread": "std", "variability": "std",

    # unique
    "unique": "unique", "distinct": "unique", "different": "unique",
    "how many types": "unique", "categories": "unique",
    "variety": "unique",
}

STOP_WORDS = {
    "what", "is", "the", "of", "in", "for", "by", "and", "or",
    "a", "an", "show", "me", "give", "find", "get", "tell",
    "per", "each", "every", "all", "column", "field", "value",
    "values", "data", "dataset", "table", "rows", "please",
    "can", "you", "could", "would", "should", "will", "do",
    "does", "did", "want", "need", "like", "have", "has",
    "across", "within", "among", "between", "from", "to", "with",
}

# ─────────────────────────────────────────────────────────────
# COLUMN MATCHING — fuzzy + semantic
# ─────────────────────────────────────────────────────────────

# Common synonyms for column name matching
COL_SYNONYMS = {
    "salary":     ["pay", "wage", "income", "compensation", "earnings", "ctc", "remuneration"],
    "revenue":    ["sales", "income", "turnover", "earnings", "proceeds"],
    "profit":     ["gain", "margin", "surplus", "net income", "earnings"],
    "age":        ["years", "old", "yrs"],
    "gender":     ["sex", "male", "female"],
    "department": ["dept", "division", "team", "unit", "group"],
    "score":      ["grade", "marks", "rating", "points", "result"],
    "date":       ["time", "when", "period", "day", "month", "year"],
    "name":       ["person", "employee", "customer", "client", "user"],
    "city":       ["location", "place", "town", "region", "area"],
    "price":      ["cost", "amount", "fee", "charge", "rate", "value"],
    "quantity":   ["qty", "amount", "count", "units", "volume", "number"],
    "category":   ["type", "kind", "class", "group", "segment"],
    "status":     ["state", "condition", "stage", "level"],
}


def _levenshtein(s1, s2):
    """Simple edit distance for typo tolerance."""
    if abs(len(s1) - len(s2)) > 3:
        return 99
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]


def _find_column(text, cols, df=None):
    """
    Multi-strategy column finder:
    1. Exact match
    2. Column name in text (substring)
    3. All words of column match in text
    4. Synonym matching
    5. Partial word overlap
    6. Typo tolerance (Levenshtein)
    Returns (col_name, score) or (None, 0)
    """
    text_clean = text.lower().strip()
    text_words = set(re.split(r"[\s_\-/,\.]+", text_clean)) - STOP_WORDS

    best_col   = None
    best_score = 0

    for col in cols:
        col_lower = col.lower().strip()
        col_words = set(re.split(r"[\s_\-/]+", col_lower)) - STOP_WORDS
        score     = 0

        # 1. Exact match
        if col_lower == text_clean:
            return col, 100

        # 2. Full column name in text
        if col_lower in text_clean:
            score = max(score, 95 - max(0, len(text_clean) - len(col_lower)) // 2)

        # 3. All column words in text
        if col_words and col_words.issubset(text_words):
            score = max(score, 85)

        # 4. Synonym check
        for base_word, synonyms in COL_SYNONYMS.items():
            # If column name matches a base word
            col_matches_base = any(w in col_lower for w in [base_word] + synonyms)
            # If text contains a synonym
            text_has_synonym = any(syn in text_clean for syn in synonyms + [base_word])
            if col_matches_base and text_has_synonym:
                score = max(score, 75)

        # 5. Partial word overlap
        overlap = col_words & text_words
        if overlap:
            ratio = len(overlap) / max(len(col_words), 1)
            score = max(score, int(40 + ratio * 40))

        # 6. Typo tolerance — check each word
        for cw in col_words:
            for tw in text_words:
                if len(cw) >= 4 and _levenshtein(cw, tw) <= 1:
                    score = max(score, 60)

        if score > best_score:
            best_score = score
            best_col   = col

    return best_col, best_score


# ─────────────────────────────────────────────────────────────
# INTENT DETECTION — what does user actually want?
# ─────────────────────────────────────────────────────────────

def _detect_intent(query):
    """
    Extended intent detection beyond simple op aliases.
    Returns op string or None.
    """
    q = query.lower()

    # Check all aliases (longest match first)
    for alias in sorted(OP_ALIASES, key=len, reverse=True):
        if alias in q:
            return OP_ALIASES[alias]

    # Contextual patterns
    patterns = [
        (r"\bwhat.*(highest|most|top|maximum|best)\b",       "max"),
        (r"\bwhat.*(lowest|least|minimum|worst|bottom)\b",   "min"),
        (r"\bwhat.*(average|typical|normally|usually)\b",    "mean"),
        (r"\bhow many\b",                                    "count"),
        (r"\blist.*(unique|different|distinct)\b",           "unique"),
        (r"\btotal\b",                                       "sum"),
        (r"\bsum\b",                                         "sum"),
        (r"\bbreakdown\b",                                   "count"),
        (r"\bdistribution\b",                                "count"),
    ]

    for pattern, op in patterns:
        if re.search(pattern, q):
            return op

    return None


# ─────────────────────────────────────────────────────────────
# QUERY PREPROCESSOR — clean and normalize
# ─────────────────────────────────────────────────────────────

def _preprocess_query(query):
    """Clean and normalize query text."""
    q = query.lower().strip()

    # Remove punctuation except hyphens/underscores
    q = re.sub(r"[?!.,;:\"'(){}\[\]]", " ", q)

    # Normalize whitespace
    q = re.sub(r"\s+", " ", q).strip()

    # Common shorthand expansions
    replacements = {
        r"\bdept\b":    "department",
        r"\bqty\b":     "quantity",
        r"\bno\b\.":    "number",
        r"\bpct\b":     "percent",
        r"\bvs\b":      "versus",
        r"\bw/\b":      "with",
        r"\bexcl\b":    "excluding",
        r"\bincl\b":    "including",
    }
    for pattern, replacement in replacements.items():
        q = re.sub(pattern, replacement, q)

    return q


# ─────────────────────────────────────────────────────────────
# PARSE QUERY
# ─────────────────────────────────────────────────────────────

BY_PATTERNS = [
    r"\bgrouped by\b", r"\bbreakdown by\b", r"\bsplit by\b",
    r"\bby each\b",    r"\bfor each\b",     r"\bper\b",
    r"\bby\b",         r"\bacross\b",
]


def parse_query(query, df):
    """
    Returns (op, target_col, group_col).
    Improved NLP — handles complex queries, typos, synonyms.
    """
    q    = _preprocess_query(query)
    cols = df.columns.tolist()

    # ── 1. Detect operation ───────────────────────────────────
    op = _detect_intent(q)

    # ── 2. Split on grouping keywords ────────────────────────
    group_col  = None
    target_col = None
    split_pos  = None
    split_end  = None

    for pat in BY_PATTERNS:
        m = re.search(pat, q)
        if m:
            split_pos = m.start()
            split_end = m.end()
            break

    if split_pos is not None:
        left_text  = q[:split_pos].strip()
        right_text = q[split_end:].strip()

        target_col, t_score = _find_column(left_text, cols, df)
        group_col,  g_score = _find_column(right_text, cols, df)

        # If target not on left, search full query minus group text
        if target_col is None or t_score < 30:
            remaining = q.replace(group_col.lower() if group_col else "", "")
            target_col, _ = _find_column(remaining, cols, df)

        # Sanity: target and group should not be the same
        if target_col == group_col:
            group_col = None

    else:
        # No grouping — find single target
        target_col, _ = _find_column(q, cols, df)

    # ── 3. Fallback — scan all columns ───────────────────────
    if target_col is None:
        best_col   = None
        best_score = 0
        for col in cols:
            col_lower = col.lower()
            if col_lower in q:
                _, score = _find_column(col_lower, cols, df)
                score = max(score, 90)
            else:
                _, score = _find_column(col_lower, cols, df)
            if score > best_score:
                best_score = score
                best_col   = col
        if best_score >= 30:
            target_col = best_col

    # ── 4. Default op from column type ───────────────────────
    if op is None and target_col is not None:
        try:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                op = "mean"
            else:
                op = "count"
        except Exception:
            op = "count"

    return op, target_col, group_col


# ─────────────────────────────────────────────────────────────
# EXECUTE QUERY
# ─────────────────────────────────────────────────────────────

OP_MAP = {
    "mean":   "mean",
    "sum":    "sum",
    "max":    "max",
    "min":    "min",
    "median": "median",
    "count":  "count",
    "std":    "std",
    "unique": "nunique",
}


def execute_query(df, op, target, group):
    """Execute and return result dict."""
    if df.empty:
        return {"result": None, "error": "Dataset is empty.", "query_desc": ""}

    if op is None and target is None:
        return {
            "result":     None,
            "error":      "Could not understand the query. Try: 'average salary by department' or 'total sales by region'.",
            "query_desc": "",
        }

    if target is None:
        available = ", ".join([f"'{c}'" for c in df.columns[:8]])
        return {"result": None, "error": f"Column not found. Available: {available}", "query_desc": ""}

    if op not in OP_MAP:
        return {
            "result":     None,
            "error":      f"Unknown operation '{op}'. Supported: average, total, count, max, min, median, std, unique.",
            "query_desc": "",
        }

    if target not in df.columns:
        available = ", ".join([f"'{c}'" for c in df.columns[:8]])
        return {"result": None, "error": f"Column '{target}' not found. Available: {available}", "query_desc": ""}

    if group and group not in df.columns:
        return {"result": None, "error": f"Group column '{group}' not found.", "query_desc": ""}

    try:
        pandas_op = OP_MAP[op]
        op_label  = op.upper()

        if group:
            result     = df.groupby(group)[target].agg(pandas_op)
            query_desc = f"{op_label} of '{target}' by '{group}'"
        else:
            result     = getattr(df[target], pandas_op)()
            query_desc = f"{op_label} of '{target}'"

        return {"result": result, "error": None, "query_desc": query_desc}

    except Exception as e:
        return {"result": None, "error": f"Query failed: {str(e)}", "query_desc": ""}


# ─────────────────────────────────────────────────────────────
# GENERATE INSIGHT
# ─────────────────────────────────────────────────────────────

def generate_query_insight(result_dict, target, group):
    """Human-readable insight from query result."""
    result = result_dict.get("result")
    error  = result_dict.get("error")

    if error or result is None:
        return []

    insights = []

    if isinstance(result, pd.Series) and len(result) > 0:
        sorted_r = result.sort_values(ascending=False)
        top_name = sorted_r.index[0]
        top_val  = sorted_r.iloc[0]
        bot_name = sorted_r.index[-1]
        bot_val  = sorted_r.iloc[-1]

        insights.append(f"🥇 Highest **{target}**: **{top_name}** → {_fmt(top_val)}")
        insights.append(f"🔻 Lowest **{target}**: **{bot_name}** → {_fmt(bot_val)}")

        if len(sorted_r) > 2:
            insights.append(f"📊 Spread across {len(sorted_r)} groups: {_fmt(top_val - bot_val)}")

        try:
            total = sorted_r.sum()
            if total > 0:
                top_pct = (top_val / total) * 100
                if top_pct > 50:
                    insights.append(
                        f"⚠️ **{top_name}** accounts for {top_pct:.1f}% of total — dominates significantly."
                    )
                elif top_pct > 30:
                    insights.append(f"📌 **{top_name}** is the largest contributor at {top_pct:.1f}%.")
        except Exception:
            pass

    else:
        try:
            val = float(result)
            insights.append(f"📌 Result: **{_fmt(val)}**")
        except Exception:
            insights.append(f"📌 Result: **{result}**")

    return insights


def _fmt(val):
    try:
        f = float(val)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except Exception:
        return str(val)
