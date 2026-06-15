import pandas as pd
import numpy as np
import re


class DataPreprocessor:
    """
    Universal cleaning pipeline.
    Works on ANY dataset — student, sales, HR, hospital, IoT, etc.
    Every fill decision is based on the column's own data personality,
    not on column names and not blindly on user's global choice.
    """

    def __init__(self, df, outlier_option, missing_option,
                 dataset_profile=None, column_profiles=None):
        self.df              = df.copy()
        self.outlier_option  = outlier_option   # "No Action" | "Remove Outliers" | "Cap Outliers"
        self.missing_option  = missing_option   # user's sidebar choice — used as a hint, not law
        self.dataset_profile = dataset_profile
        self.column_profiles = column_profiles or {}
        self.report          = []

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _profile(self, col):
        return self.column_profiles.get(col, {})

    def _detected_type(self, col):
        return self._profile(col).get("detected_type", "")

    # ─────────────────────────────────────────────────────────────
    # COLUMN PERSONALITY ENGINE
    # Decides the BEST fill strategy for any column purely from data.
    # Column names are never read.
    # ─────────────────────────────────────────────────────────────

    def _personality(self, col):
        """
        Returns a dict describing this column's data personality:
          fill_method : "unknown" | "mode" | "median" | "mean" | "ffill" | "drop"
          fill_reason : human-readable explanation for the report
        """
        col_type = self._detected_type(col)
        s        = self.df[col].dropna()

        # ── Types that should be left as-is or marked Unknown ────
        # Names, emails, phones, addresses, free text —
        # filling these with a statistical value makes no sense.
        if col_type in {"name", "email", "phone", "address", "text"}:
            return {
                "fill_method": "unknown",
                "fill_reason": f"non-fillable type ({col_type}) → 'Unknown'"
            }

        # ── Date columns → forward fill (time continuity) ────────
        if col_type == "date":
            return {
                "fill_method": "ffill",
                "fill_reason": "date column → forward/back fill"
            }

        # ── Non-numeric (category / object) ──────────────────────
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            n_unique     = s.nunique()
            total_values = len(s)

            # Binary or very low cardinality (2-4 unique values like gender, yes/no, blood group)
            # Mode is risky — we don't know which value the missing rows should be.
            # Mark as Unknown so we don't introduce false information.
            if n_unique <= 4:
                return {
                    "fill_method": "unknown",
                    "fill_reason": f"low-cardinality category ({n_unique} unique) → 'Unknown' (safer than mode)"
                }

            # Higher cardinality categories (country, department, product_type, etc.)
            # Mode is reasonable — the most common value is a valid guess.
            return {
                "fill_method": "mode",
                "fill_reason": f"categorical column ({n_unique} unique values) → mode"
            }

        # ── From here: column is numeric ─────────────────────────

        if len(s) == 0:
            return {"fill_method": "unknown", "fill_reason": "all values missing"}

        n_unique     = s.nunique()
        total_values = len(s)
        skewness     = float(s.skew()) if total_values >= 3 else 0.0
        value_range  = float(s.max() - s.min()) if total_values > 0 else 0.0

        # ── Is this column "discrete-like"? ──────────────────────
        # We don't rely on dtype (float64 vs int64) because missing values
        # force pandas to store integers as float64 (e.g. age 22.0, 35.0).
        # Instead, measure THREE things from the data itself:
        #
        #   1. granularity  — what fraction of values are whole numbers?
        #                     age: 22.0, 35.0 → ~98% whole → high
        #                     height: 165.3, 172.8 → ~5% whole → low
        #
        #   2. density      — unique values relative to the range they cover
        #                     age 18-80: 88 unique over range 62 → 1.4 per unit → dense
        #                     salary 30k-500k: 400 unique over 470k range → 0.0008 → sparse
        #
        #   3. range bound  — is the value range small enough to be a human scale?
        #                     age/score/year/rating → bounded (range < 200)
        #                     salary/price/revenue → unbounded (range >> 200)
        #
        # A column is "discrete-like" when granularity is high AND
        # (density is high OR range is bounded).
        # This catches: age, score, rating, year, num_children, experience_years
        # even when stored as float64 with decimals like 0.92 (infant age).

        whole_number_frac = (s == s.round()).mean()   # 0.0 to 1.0
        granularity_high  = whole_number_frac >= 0.80 # 80%+ values are whole numbers

        density           = n_unique / value_range if value_range > 0 else 0
        density_high      = density >= 0.30           # at least 1 unique per ~3 units

        range_bounded     = value_range <= 300        # covers age(120), score(100),
                                                      # year(current-1900~125), rating(5)

        is_discrete = granularity_high and (density_high or range_bounded)

        if is_discrete:
            return {
                "fill_method": "mode",
                "fill_reason": (
                    f"discrete numeric — {whole_number_frac*100:.0f}% whole numbers, "
                    f"range={value_range:.0f}, density={density:.2f} → mode"
                )
            }

        # ── Heavily skewed continuous (salary, price, revenue) ───
        # Skew > 1.0 means a long tail / outliers → median is safer than mean.
        if abs(skewness) > 1.0:
            return {
                "fill_method": "median",
                "fill_reason": f"skewed distribution (skew={skewness:.2f}) → median"
            }

        # ── Normally distributed continuous (height, weight, temp) ─
        return {
            "fill_method": "mean",
            "fill_reason": f"continuous symmetric (skew={skewness:.2f}) → mean"
        }

    # ─────────────────────────────────────────────────────────────
    # FILL EXECUTOR
    # Applies the personality decision to actually fill the column.
    # ─────────────────────────────────────────────────────────────

    def _fill_column(self, col, label_prefix="✔"):
        """
        Fills missing values in `col` according to its data personality.
        `label_prefix` lets split sub-columns use a different indent.
        """
        missing = int(self.df[col].isnull().sum())
        if missing == 0:
            return

        # If user chose "Drop Rows" — override everything
        if self.missing_option == "Drop Rows":
            before = len(self.df)
            self.df.dropna(subset=[col], inplace=True)
            self.report.append(
                f"{label_prefix} '{col}' ({missing} missing) → dropped rows"
            )
            return

        p      = self._personality(col)
        method = p["fill_method"]
        reason = p["fill_reason"]

        # ── Manual Override ──────────────────────────────
        # If user selected a specific method (not "Auto"),
        # use it for numeric columns only.
        # Non-numeric, date, skip-types still follow personality.
        if self.missing_option not in ("Auto", "Drop Rows"):
            if pd.api.types.is_numeric_dtype(self.df[col]) or self._detected_type(col) == "numeric":
                override_map = {
                    "Mean":   "mean",
                    "Median": "median",
                    "Mode":   "mode",
                }
                overridden = override_map.get(self.missing_option)
                if overridden:
                    method = overridden
                    reason = f"manual override → {self.missing_option.lower()}"

        if method == "unknown":
            self.df[col] = self.df[col].fillna("Unknown")
            self.report.append(
                f"{label_prefix} '{col}' ({missing} missing) → 'Unknown'  [{reason}]"
            )

        elif method == "mode":
            mode_vals = self.df[col].mode()
            fill_val  = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
            self.df[col] = self.df[col].fillna(fill_val)
            self.report.append(
                f"{label_prefix} '{col}' ({missing} missing) → mode='{fill_val}'  [{reason}]"
            )

        elif method == "median":
            fill_val = float(self.df[col].median())
            self.df[col] = self.df[col].fillna(fill_val)
            self.report.append(
                f"{label_prefix} '{col}' ({missing} missing) → median={round(fill_val,4)}  [{reason}]"
            )

        elif method == "mean":
            fill_val = float(self.df[col].mean())
            self.df[col] = self.df[col].fillna(fill_val)
            self.report.append(
                f"{label_prefix} '{col}' ({missing} missing) → mean={round(fill_val,4)}  [{reason}]"
            )

        elif method == "ffill":
            self.df[col] = self.df[col].ffill().bfill()
            self.report.append(
                f"{label_prefix} '{col}' ({missing} missing) → forward/back filled  [{reason}]"
            )

    # ─────────────────────────────────────────────────────────────
    # STEP 1 — STANDARDIZE RAW TEXT
    # ─────────────────────────────────────────────────────────────

    NULL_STRINGS = {
        "?", "", " ", "na", "n/a", "null", "none", "nan",
        "--", "-", "nil", "missing", "unknown", "not available",
        "#n/a", "#null!", "n.a.", "n.a", "nd", "not applicable",
    }

    def _standardize(self):
        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = self.df[col].astype(str).str.strip()

        def _nullify(val):
            if pd.isna(val):
                return np.nan
            return np.nan if str(val).strip().lower() in self.NULL_STRINGS else val

        self.df = self.df.map(_nullify)
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # ─────────────────────────────────────────────────────────────
    # STEP 2 — SPLIT STRUCTURED CODE COLUMNS
    # Detected from data (separator pattern), never from column name.
    # After split, each new column is filled by its own personality.
    # ─────────────────────────────────────────────────────────────

    _SEP_RE = re.compile(r"^([A-Za-z0-9]+)([-_/|\\])([A-Za-z0-9]+)$")

    def _split_structured(self):
        to_drop = []

        for col in list(self.df.columns):
            if self._detected_type(col) != "structured_code":
                continue

            sample = self.df[col].dropna().astype(str).str.strip().head(50).tolist()
            if not sample:
                continue

            sep_counts = {}
            for v in sample:
                m = self._SEP_RE.match(v)
                if m:
                    sep = m.group(2)
                    sep_counts[sep] = sep_counts.get(sep, 0) + 1

            if not sep_counts:
                continue

            dominant_sep = max(sep_counts, key=sep_counts.get)
            if sep_counts[dominant_sep] / len(sample) < 0.60:
                continue

            try:
                split_data = (
                    self.df[col]
                    .astype(str).str.strip()
                    .str.split(re.escape(dominant_sep), n=1, expand=True)
                )
                if split_data.shape[1] < 2:
                    continue

                left  = split_data[0].str.strip().replace("nan", np.nan)
                right = split_data[1].str.strip().replace("nan", np.nan)

                right_num = pd.to_numeric(right, errors="coerce")
                left_num  = pd.to_numeric(left,  errors="coerce")

                if right_num.notna().mean() > 0.60:
                    self.df[f"{col}_Type"]   = left
                    self.df[f"{col}_Number"] = right_num
                    new_cols = [f"{col}_Type", f"{col}_Number"]

                elif left_num.notna().mean() > 0.60:
                    self.df[f"{col}_Type"]   = right
                    self.df[f"{col}_Number"] = left_num
                    new_cols = [f"{col}_Type", f"{col}_Number"]

                else:
                    self.df[f"{col}_Part1"] = left
                    self.df[f"{col}_Part2"] = right
                    new_cols = [f"{col}_Part1", f"{col}_Part2"]

                self.report.append(f"✔ Split '{col}' → {new_cols}")

                # Fill each new column by its own personality
                for nc in new_cols:
                    self._fill_column(nc, label_prefix="  ↳")

                to_drop.append(col)

            except Exception as e:
                self.report.append(f"⚠ Could not split '{col}': {e}")

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)

    # ─────────────────────────────────────────────────────────────
    # STEP 3 — SAFE NUMERIC CONVERSION
    # ─────────────────────────────────────────────────────────────

    _SKIP_CONVERT = {"name", "category", "text", "address",
                     "email", "phone", "date", "structured_code"}

    def _convert_numerics(self):
        for col in self.df.columns:
            if self._detected_type(col) in self._SKIP_CONVERT:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            try:
                cleaned   = self.df[col].astype(str).str.replace(r"[,\$€£₹%\s]", "", regex=True)
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().mean() >= 0.90:
                    self.df[col] = converted
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────
    # STEP 4 — HANDLE MISSING VALUES
    # Each column filled by its own personality — not a global rule.
    # ─────────────────────────────────────────────────────────────

    def _handle_missing(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
            self._fill_column(col)

    # ─────────────────────────────────────────────────────────────
    # STEP 5 — REMOVE DUPLICATES
    # ─────────────────────────────────────────────────────────────

    def _remove_duplicates(self):
        before  = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = before - len(self.df)
        if removed > 0:
            self.report.append(f"✔ Removed {removed} duplicate row(s)")
        # Don't show message if no duplicates — avoids confusion

    # ─────────────────────────────────────────────────────────────
    # STEP 6 — HANDLE OUTLIERS (numeric only, personality-aware)
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
                    continue
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                n_out = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())

                if self.outlier_option == "Remove Outliers":
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                    self.report.append(f"✔ '{col}' → removed {n_out} outlier row(s)")

                elif self.outlier_option == "Cap Outliers":
                    self.df[col] = self.df[col].clip(lower, upper)
                    self.report.append(
                        f"✔ '{col}' → capped {n_out} outlier(s) [{round(lower,2)}, {round(upper,2)}]"
                    )
            except Exception as e:
                self.report.append(f"⚠ Outlier skipped for '{col}': {e}")

    # ─────────────────────────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────────────────────────

    def process(self):
        # Edge case guards
        if self.df.empty:
            self.report.append("⚠️ Empty dataset — nothing to process.")
            return self.df, self.report
        if self.df.shape[1] == 0:
            self.report.append("⚠️ No columns found.")
            return self.df, self.report

        missing_before = int(self.df.isnull().sum().sum())
        self.report.append(f"📋 Rows: {len(self.df)}  |  Columns: {self.df.shape[1]}")
        self.report.append(f"📋 Missing before: {missing_before}")
        self.report.append("─" * 50)

        try: self._standardize()
        except Exception as e: self.report.append(f"⚠️ Standardize skipped: {e}")

        try: self._split_structured()
        except Exception as e: self.report.append(f"⚠️ Split skipped: {e}")

        try: self._convert_numerics()
        except Exception as e: self.report.append(f"⚠️ Numeric convert skipped: {e}")

        try: self._handle_missing()
        except Exception as e: self.report.append(f"⚠️ Missing fill skipped: {e}")

        try: self._remove_duplicates()
        except Exception as e: self.report.append(f"⚠️ Dedup skipped: {e}")

        try: self._handle_outliers()
        except Exception as e: self.report.append(f"⚠️ Outlier handling skipped: {e}")

        missing_after = int(self.df.isnull().sum().sum())
        self.report.append("─" * 50)
        self.report.append(f"✅ Done  |  Rows: {len(self.df)}  |  Columns: {self.df.shape[1]}")
        self.report.append(f"✅ Missing after: {missing_after}")
        return self.df, self.report
