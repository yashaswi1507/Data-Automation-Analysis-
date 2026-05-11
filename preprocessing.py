import pandas as pd
import numpy as np


class DataPreprocessor:

    def __init__(
        self,
        df,
        outlier_option="No Action",
        missing_option="Median"
    ):

        self.df = df.copy()
        self.report = []

        self.outlier_option = outlier_option
        self.missing_option = missing_option

    def process(self):

        # =====================================================
        # CLEAN OBJECT COLUMNS
        # =====================================================

        for col in self.df.select_dtypes(include='object').columns:

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
            )

        # =====================================================
        # CONVERT WEIRD VALUES TO NaN
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
                "NaN"
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
        # FEATURE ENGINEERING FIRST
        # =====================================================

        columns_to_drop = []

        for col in self.df.columns:

            if self.df[col].dtype == "object":

                temp_col = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                )

                # Ignore NaN strings
                temp_col = temp_col.replace(
                    ["nan", "None"],
                    np.nan
                )

                contains_letters = temp_col.str.contains(
                r'[A-Za-z]',
                regex=True,
                na=False
                )

                contains_numbers = temp_col.str.contains(
                r'\d',
                regex=True,
                na=False
                )

            # If column has BOTH letters and numbers anywhere
                has_letters = contains_letters.any()
                has_numbers = contains_numbers.any()

                if has_letters and has_numbers:

                    # TEXT PART
                    self.df[f"{col}_Type"] = (

                        temp_col
                        .str.extract(
                            r'([A-Za-z./]+)',
                            expand=False
                        )

                    )

                    self.df[f"{col}_Type"] = (
                        self.df[f"{col}_Type"]
                        .fillna("NUM")
                    )

                    # NUMBER PART
                    self.df[f"{col}_Number"] = pd.to_numeric(

                        temp_col
                        .str.extract(
                            r'(\d+)',
                            expand=False
                        ),

                        errors='coerce'
                    )

                    columns_to_drop.append(col)

                    self.report.append(
                        f"✔ Split mixed column '{col}' into Type and Number"
                    )

                # =================================================
                # HANDLE MULTI VALUE TEXT
                # =================================================

                elif (
                    temp_col
                    .str.contains(" ", na=False)
                    .mean() > 0.3
                ):

                    self.df[col] = (
                        temp_col
                        .str.split()
                        .str[0]
                    )

                    self.report.append(
                        f"✔ Simplified multi-value column '{col}'"
                    )

        # =====================================================
        # DROP ORIGINAL MIXED COLUMNS
        # =====================================================

        if columns_to_drop:

            self.df.drop(
                columns=columns_to_drop,
                inplace=True
            )

        # =====================================================
        # CONVERT MOSTLY NUMERIC COLUMNS
        # =====================================================

        for col in self.df.columns:

            try:

                numeric_check = pd.to_numeric(
                    self.df[col],
                    errors='coerce'
                )

                ratio = numeric_check.notna().mean()

                # Convert only if mostly numeric
                if ratio > 0.8:

                    self.df[col] = numeric_check

            except:

                pass

        # =====================================================
        # BEFORE CLEANING REPORT
        # =====================================================

        total_rows = len(self.df)

        duplicate_count = self.df.duplicated().sum()

        null_counts = self.df.isnull().sum()

        self.report.append(f"Total Rows: {total_rows}")

        self.report.append(
            f"Duplicate Rows: {duplicate_count}"
        )

        for col in null_counts.index:

            if null_counts[col] > 0:

                self.report.append(
                    f"Missing in '{col}': {null_counts[col]}"
                )

        # =====================================================
        # HANDLE MISSING VALUES
        # =====================================================

        for col in self.df.columns:

            missing = self.df[col].isnull().sum()

            if missing > 0:

                # ================= NUMERIC =================

                if pd.api.types.is_numeric_dtype(
                    self.df[col]
                ):

                    if self.missing_option == "Mean":

                        fill_value = self.df[col].mean()
                        method_used = "Mean"

                    elif self.missing_option == "Median":

                        fill_value = self.df[col].median()
                        method_used = "Median"

                    elif self.missing_option == "Mode":

                        fill_value = self.df[col].mode()[0]
                        method_used = "Mode"

                    elif self.missing_option == "Drop Rows":

                        self.df.dropna(inplace=True)
                        method_used = "Drop Rows"

                    if self.missing_option != "Drop Rows":

                        self.df[col] = self.df[col].fillna(
                            fill_value
                        )

                # ================= CATEGORICAL =================

                else:

                    mode_value = self.df[col].mode()

                    if len(mode_value) > 0:

                        self.df[col] = self.df[col].fillna(
                            mode_value[0]
                        )

                    else:

                        self.df[col] = self.df[col].fillna(
                            "Unknown"
                        )

                    method_used = "Mode"

                self.report.append(
                    f"✔ Handled missing values in '{col}' using {method_used}"
                )

        # =====================================================
        # REMOVE DUPLICATES
        # =====================================================

        before_duplicates = len(self.df)

        self.df.drop_duplicates(inplace=True)

        removed_duplicates = (
            before_duplicates - len(self.df)
        )

        self.report.append(
            f"Removed duplicate rows: {removed_duplicates}"
        )

        # =====================================================
        # OUTLIER HANDLING
        # =====================================================

        outlier_count = 0

        numeric_cols = self.df.select_dtypes(
            include='number'
        ).columns

        for col in numeric_cols:

            Q1 = self.df[col].quantile(0.25)

            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            outliers = (

                (
                    (self.df[col] < lower)
                    |
                    (self.df[col] > upper)

                ).sum()

            )

            if outliers > 0:

                outlier_count += outliers

                self.report.append(
                    f"Outliers in '{col}': {outliers}"
                )

                # ================= REMOVE =================

                if (
                    self.outlier_option
                    == "Remove Outliers"
                ):

                    self.df = self.df[

                        (self.df[col] >= lower)

                        &

                        (self.df[col] <= upper)

                    ]

                    self.report.append(
                        f"Removed outliers from '{col}'"
                    )

                # ================= CAP =================

                elif (
                    self.outlier_option
                    == "Cap Outliers"
                ):

                    self.df[col] = self.df[col].clip(
                        lower,
                        upper
                    )

                    self.report.append(
                        f"Capped outliers in '{col}'"
                    )

        # =====================================================
        # FINAL REPORT
        # =====================================================

        self.report.append(
            "Data Cleaning Completed"
        )

        if outlier_count == 0:

            self.report.append(
                "No significant outliers detected"
            )

        return self.df, self.report