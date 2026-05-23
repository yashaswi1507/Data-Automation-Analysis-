import pandas as pd
import numpy as np
import re


class DataPreprocessor:

    def __init__(
        self,
        df,
        outlier_option,
        missing_option,
        dataset_profile=None,
        column_profiles=None
    ):

        self.df = df.copy()

        self.outlier_option = outlier_option
        self.missing_option = missing_option

        self.dataset_profile = dataset_profile
        self.column_profiles = column_profiles or {}

        self.report = []

    # =====================================================
    # MAIN PROCESS
    # =====================================================

    def process(self):

        # =====================================================
        # CLEAN OBJECT COLUMNS
        # =====================================================

        object_cols = self.df.select_dtypes(
            include=['object']
        ).columns

        for col in object_cols:

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
            )

        # =====================================================
        # REPLACE WEIRD VALUES
        # =====================================================

        self.df.replace(
            [
                "?",
                "",
                " ",
                "NA",
                "N/A",
                "null",
                "NULL",
                "--",
                "nan",
                "NaN",
                "None",
                "none"
            ],
            np.nan,
            inplace=True
        )

        self.df.replace(
            r'^\s*$',
            np.nan,
            regex=True,
            inplace=True
        )

        # =====================================================
        # STORE MISSING INFO
        # =====================================================

        self.original_null_counts = (
            self.df.isnull().sum()
        )

        self.total_missing_before = (
            self.original_null_counts.sum()
        )

        # =====================================================
        # SAFE FEATURE ENGINEERING
        # =====================================================

        columns_to_drop = []

        for col in object_cols:

            # ---------------------------------------------
            # SKIP NAME / TEXT / ADDRESS TYPE COLUMNS
            # ---------------------------------------------

            profile = self.column_profiles.get(col, {})

            detected_type = profile.get(
                "detected_type",
                ""
            )

            if detected_type in [
                "name",
                "text",
                "address"
            ]:
                continue

            temp_col = (
                self.df[col]
                .astype(str)
                .str.strip()
            )

            sample_values = (
                temp_col
                .dropna()
                .head(20)
                .tolist()
            )

            if len(sample_values) == 0:
                continue

            separators = [
                "-",
                "_",
                "|"
            ]

            found_separator = None

            for sep in separators:

                match_count = sum([
                    sep in str(v)
                    for v in sample_values
                ])

                if match_count >= len(sample_values) * 0.8:

                    found_separator = sep
                    break

            if found_separator is None:
                continue

            # ---------------------------------------------
            # SPLIT ONLY STRUCTURED CODE COLUMNS
            # ---------------------------------------------

            if detected_type == "structured_code":

                try:

                    split_data = temp_col.str.split(
                        found_separator,
                        expand=True
                    )

                    if split_data.shape[1] == 2:

                        self.df[
                            f"{col}_Type"
                        ] = split_data[0]

                        self.df[
                            f"{col}_Number"
                        ] = pd.to_numeric(
                            split_data[1],
                            errors="coerce"
                        )

                        columns_to_drop.append(col)

                        self.report.append(
                            f"✔ Split structured column '{col}'"
                        )

                except:
                    pass

        # =====================================================
        # DROP OLD STRUCTURED COLS
        # =====================================================

        if len(columns_to_drop) > 0:

            self.df.drop(
                columns=columns_to_drop,
                inplace=True
            )

        # =====================================================
        # SAFE NUMERIC CONVERSION
        # =====================================================

        for col in self.df.columns:

            # skip already object important cols

            profile = self.column_profiles.get(col, {})

            detected_type = profile.get(
                "detected_type",
                ""
            )

            if detected_type in [
                "name",
                "category",
                "text",
                "address",
                "identifier"
            ]:
                continue

            try:

                converted = pd.to_numeric(
                    self.df[col],
                    errors='coerce'
                )

                ratio = (
                    converted.notna().mean()
                )

                if ratio > 0.90:

                    self.df[col] = converted

            except:
                pass

        # =====================================================
        # REPORT
        # =====================================================

        self.report.append(
            f"Rows: {len(self.df)}"
        )

        self.report.append(
            f"Columns: {self.df.shape[1]}"
        )

        self.report.append(
            f"Missing Before Cleaning: {self.total_missing_before}"
        )

        # =====================================================
        # HANDLE MISSING VALUES
        # =====================================================

        for col in self.df.columns:

            missing = (
                self.df[col]
                .isnull()
                .sum()
            )

            if missing == 0:
                continue

            # ---------------------------------------------
            # NUMERIC
            # ---------------------------------------------

            if pd.api.types.is_numeric_dtype(
                self.df[col]
            ):

                if self.missing_option == "Mean":

                    fill_value = (
                        self.df[col].mean()
                    )

                elif self.missing_option == "Median":

                    fill_value = (
                        self.df[col].median()
                    )

                elif self.missing_option == "Mode":

                    fill_value = (
                        self.df[col]
                        .mode()[0]
                    )

                elif self.missing_option == "Drop Rows":

                    self.df.dropna(inplace=True)
                    continue

                self.df[col] = (
                    self.df[col]
                    .fillna(fill_value)
                )

            # ---------------------------------------------
            # CATEGORICAL
            # ---------------------------------------------

            else:

                mode_value = (
                    self.df[col]
                    .mode()
                )

                if len(mode_value) > 0:

                    self.df[col] = (
                        self.df[col]
                        .fillna(mode_value[0])
                    )

                else:

                    self.df[col] = (
                        self.df[col]
                        .fillna("Unknown")
                    )

            self.report.append(
                f"✔ Missing handled in '{col}'"
            )

        # =====================================================
        # REMOVE DUPLICATES
        # =====================================================

        before = len(self.df)

        self.df.drop_duplicates(
            inplace=True
        )

        removed = before - len(self.df)

        self.report.append(
            f"Removed duplicates: {removed}"
        )

        # =====================================================
        # OUTLIERS
        # =====================================================

        numeric_cols = self.df.select_dtypes(
            include='number'
        ).columns

        for col in numeric_cols:

            try:

                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)

                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                if self.outlier_option == "Remove Outliers":

                    self.df = self.df[
                        (
                            self.df[col] >= lower
                        )
                        &
                        (
                            self.df[col] <= upper
                        )
                    ]

                elif self.outlier_option == "Cap Outliers":

                    self.df[col] = (
                        self.df[col]
                        .clip(lower, upper)
                    )

            except:
                pass

        # =====================================================
        # FINAL
        # =====================================================

        self.report.append(
            "✔ Data Cleaning Completed"
        )

        return self.df, self.report