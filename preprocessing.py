import pandas as pd
import numpy as np
import re


class DataPreprocessor:
    """
    Universal cleaning pipeline.
    Works on ANY dataset — student, sales, HR, hospital, IoT, etc.
    Never assumes column names. Every decision is based on actual data.
    """

    def __init__(self, df, outlier_option, missing_option,
                 dataset_profile=None, column_profiles=None):
        self.df             = df.copy()
        self.outlier_option = outlier_option   # "No Action" | "Remove Outliers" | "Cap Outliers"
        self.missing_option = missing_option   # "Mean" | "Median" | "Mode" | "Drop Rows"
        self.dataset_profile  = dataset_profile
        self.column_profiles  = column_profiles or {}
        self.report = []

    # ─────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────

    def _profile(self, col):
        """Return the profiler's dict for this column (or empty dict)."""
        return self.column_profiles.get(col, {})

    def _detected_type(self, col):
        return self._profile(col).get("detected_type", "")

    def _strategy(self, col):
        return self._profile(col).get("cleaning_strategy", "")

    # ─────────────────────────────────────────────────────────────
    # STEP 1 ─ STANDARDIZE RAW TEXT
    # Strip whitespace, unify null-like strings → NaN
    # Works on every dataset regardless of domain.
    # ─────────────────────────────────────────────────────────────

    NULL_STRINGS = {
        "?", "", " ", "na", "n/a", "null", "none",
        "nan", "--", "-", "nil", "missing", "unknown", "not available",
        "#n/a", "#null!", "n.a.", "n.a", "nd", "not applicable",
    }

    def _standardize(self):
        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = self.df[col].astype(str).str.strip()

        # Case-insensitive null replacement
        def _nullify(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower()
            return np.nan if s in self.NULL_STRINGS else val

        self.df = self.df.map(_nullify)
        # Also catch purely whitespace cells
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # ─────────────────────────────────────────────────────────────
    # STEP 2 ─ SPLIT STRUCTURED CODE COLUMNS
    # e.g. "ORD-001" → col_Type="ORD", col_Number=1
    #      "EMP_042" → col_Type="EMP", col_Number=42
    # Detected purely from the DATA, not the column name.
    # ─────────────────────────────────────────────────────────────

    _SEP_RE  = re.compile(r"^([A-Za-z0-9]+)([-_/|\\])([A-Za-z0-9]+)$")

    def _split_structured(self):
        to_drop = []

        for col in list(self.df.columns):
            if self._detected_type(col) != "structured_code":
                continue

            sample = self.df[col].dropna().astype(str).str.strip().head(50).tolist()
            if not sample:
                continue

            # Find which separator is dominant in this column's data
            sep_counts = {}
            for v in sample:
                m = self._SEP_RE.match(v)
                if m:
                    sep = m.group(2)
                    sep_counts[sep] = sep_counts.get(sep, 0) + 1

            if not sep_counts:
                continue

            dominant_sep = max(sep_counts, key=sep_counts.get)
            hit_rate = sep_counts[dominant_sep] / len(sample)

            if hit_rate < 0.60:
                continue

            try:
                split_data = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.split(re.escape(dominant_sep), n=1, expand=True)
                )

                if split_data.shape[1] < 2:
                    continue

                left  = split_data[0].str.strip().replace("nan", np.nan)
                right = split_data[1].str.strip().replace("nan", np.nan)

                right_num = pd.to_numeric(right, errors="coerce")
                left_num  = pd.to_numeric(left,  errors="coerce")

                if right_num.notna().mean() > 0.60:
                    # Normal: ALPHA-123
                    self.df[f"{col}_Type"]   = left
                    self.df[f"{col}_Number"] = right_num
                elif left_num.notna().mean() > 0.60:
                    # Reverse: 123-ALPHA
                    self.df[f"{col}_Type"]   = right
                    self.df[f"{col}_Number"] = left_num
                else:
                    # Both text: just split into Part1 / Part2
                    self.df[f"{col}_Part1"] = left
                    self.df[f"{col}_Part2"] = right

                to_drop.append(col)
                self.report.append(f"✔ Split '{col}' → 2 new columns")

            except Exception as e:
                self.report.append(f"⚠ Could not split '{col}': {e}")

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)

    # ─────────────────────────────────────────────────────────────
    # STEP 3 ─ SAFE NUMERIC CONVERSION
    # For columns that look numeric but are stored as strings.
    # Never converts names, categories, addresses, emails, phones.
    # Uses the actual data to decide — not the column name.
    # ─────────────────────────────────────────────────────────────

    # Types that must NEVER be force-converted to numeric
    _SKIP_CONVERT = {"name", "category", "text", "address",
                     "email", "phone", "date", "structured_code"}

    def _convert_numerics(self):
        for col in self.df.columns:
            if self._detected_type(col) in self._SKIP_CONVERT:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            try:
                # Remove currency/thousand separators before trying
                cleaned = (
                    self.df[col]
                    .astype(str)
                    .str.replace(r"[,\$€£₹%\s]", "", regex=True)
                )
                converted = pd.to_numeric(cleaned, errors="coerce")
                # Only convert if 90%+ values are valid numbers
                if converted.notna().mean() >= 0.90:
                    self.df[col] = converted
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────
    # STEP 4 ─ HANDLE MISSING VALUES  (type-aware, data-driven)
    #
    # Decision tree per column:
    #   skip types       → leave as-is (names, emails, phones, addresses, text)
    #   numeric          → mean / median / mode (user's choice)
    #   category         → mode fill (most frequent value)
    #   date             → forward-fill → backward-fill
    #   structured parts → fill numeric part with median, text part with mode
    #   Drop Rows        → global option, applied before any fill
    # ─────────────────────────────────────────────────────────────

    _SKIP_FILL = {"name", "text", "address", "email", "phone"}

    def _handle_missing(self):
        for col in self.df.columns:
            missing = int(self.df[col].isnull().sum())
            if missing == 0:
                continue

            col_type = self._detected_type(col)

            # Never touch these
            if col_type in self._SKIP_FILL:
                continue

            # ── Drop Rows (global user preference) ──────
            if self.missing_option == "Drop Rows":
                before = len(self.df)
                self.df.dropna(subset=[col], inplace=True)
                dropped = before - len(self.df)
                self.report.append(f"✔ Dropped {dropped} rows with missing '{col}'")
                continue

            # ── Numeric column ───────────────────────────
            if pd.api.types.is_numeric_dtype(self.df[col]) or col_type == "numeric":
                fill_val = self._numeric_fill(col)
                self.df[col] = self.df[col].fillna(fill_val)
                self.report.append(
                    f"✔ '{col}' ({missing} missing) → filled with "
                    f"{self.missing_option.lower()} = {round(float(fill_val), 4)}"
                )

            # ── Date column ──────────────────────────────
            elif col_type == "date":
                self.df[col] = (
                    self.df[col]
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )
                self.report.append(f"✔ '{col}' ({missing} missing) → forward/back filled (date)")

            # ── Category or anything else fillable ───────
            else:
                mode_vals = self.df[col].mode()
                fill_val  = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                self.df[col] = self.df[col].fillna(fill_val)
                self.report.append(
                    f"✔ '{col}' ({missing} missing) → filled with mode = '{fill_val}'"
                )

    def _numeric_fill(self, col):
        """Return the fill value based on user's missing_option."""
        s = self.df[col]
        if self.missing_option == "Mean":
            return s.mean()
        elif self.missing_option == "Mode":
            m = s.mode()
            return m[0] if len(m) > 0 else s.median()
        else:  # Median (default)
            return s.median()

    # ─────────────────────────────────────────────────────────────
    # STEP 5 ─ REMOVE DUPLICATES
    # ─────────────────────────────────────────────────────────────

    def _remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = before - len(self.df)
        self.report.append(f"✔ Removed {removed} duplicate row(s)")

    # ─────────────────────────────────────────────────────────────
    # STEP 6 ─ HANDLE OUTLIERS
    # Only on truly numeric columns.
    # Skips IDs, codes, dates, categories.
    # Uses IQR method — works for any distribution.
    # ─────────────────────────────────────────────────────────────

    _SKIP_OUTLIER = {"identifier", "structured_code", "date",
                     "category", "name", "text", "address", "email", "phone"}

    def _handle_outliers(self):
        if self.outlier_option == "No Action":
            return

        for col in self.df.select_dtypes(include="number").columns:
            if self._detected_type(col) in self._SKIP_OUTLIER:
                continue

            try:
                Q1    = self.df[col].quantile(0.25)
                Q3    = self.df[col].quantile(0.75)
                IQR   = Q3 - Q1
                if IQR == 0:
                    continue   # constant column — skip
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                n_out = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())

                if self.outlier_option == "Remove Outliers":
                    self.df = self.df[
                        (self.df[col] >= lower) & (self.df[col] <= upper)
                    ]
                    self.report.append(
                        f"✔ '{col}' → removed {n_out} outlier row(s)"
                    )
                elif self.outlier_option == "Cap Outliers":
                    self.df[col] = self.df[col].clip(lower, upper)
                    self.report.append(
                        f"✔ '{col}' → capped {n_out} outlier(s) "
                        f"[{round(lower,2)}, {round(upper,2)}]"
                    )

            except Exception as e:
                self.report.append(f"⚠ Outlier skipped for '{col}': {e}")

    # ─────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────

    def process(self):
        missing_before = int(self.df.isnull().sum().sum())
        rows_before    = len(self.df)

        self.report.append(f"📋 Rows: {rows_before}  |  Columns: {self.df.shape[1]}")
        self.report.append(f"📋 Missing values before: {missing_before}")
        self.report.append("─" * 45)

        self._standardize()
        self._split_structured()
        self._convert_numerics()
        self._handle_missing()
        self._remove_duplicates()
        self._handle_outliers()

        missing_after = int(self.df.isnull().sum().sum())
        self.report.append("─" * 45)
        self.report.append(f"✅ Done  |  Rows: {len(self.df)}  |  Columns: {self.df.shape[1]}")
        self.report.append(f"✅ Missing values after: {missing_after}")

        return self.df, self.report