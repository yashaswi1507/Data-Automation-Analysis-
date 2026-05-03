def parse_query(query, df):
    query = query.lower()
    cols = df.columns.tolist()
    cols_lower = [c.lower() for c in cols]

    if "average" in query or "mean" in query:
        op = "mean"
    elif "max" in query:
        op = "max"
    elif "min" in query:
        op = "min"
    else:
        op = None

    target = None
    group = None

    if "by" in query:
        left = query.split("by")[0]
        right = query.split("by")[1]

        for col in cols_lower:
            if col in left:
                target = col

        for col in cols_lower:
            if col in right:
                group = col
    else:
        for col in cols_lower:
            if col in query:
                target = col

    return op, target, group


def execute_query(df, op, target, group):
    if op is None or target is None:
        return "Invalid query"

    real_target = df.columns[[c.lower() for c in df.columns].index(target)]

    if group:
        real_group = df.columns[[c.lower() for c in df.columns].index(group)]
        return df.groupby(real_group)[real_target].agg(op)

    return getattr(df[real_target], op)()


def generate_query_insight(result, target, group):
    if group is not None and hasattr(result, "idxmax"):
        return f"Highest {target} is in {result.idxmax()}"
    return ""