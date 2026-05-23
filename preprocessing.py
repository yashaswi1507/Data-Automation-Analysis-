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

        self.column_profiles = (column_profiles or {})

        self.report = []

    def process(self):

        # CLEAN OBJECT COLUMNS

        for col in self.df.select_dtypes(include='object').columns:

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
            )

        # CONVERT WEIRD VALUES TO NaN

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

        self.original_null_counts = self.df.isnull().sum()

        self.total_missing_before = (
            self.original_null_counts.sum()
        )

        columns_to_drop = []

        # FEATURE ENGINEERING

        for col in self.df.columns:

            if pd.api.types.is_string_dtype(self.df[col]):

                temp_col = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                )

                temp_col = temp_col.replace(
                    ["nan", "None"],
                    np.nan
                )

                sample_values = (
                    temp_col
                    .dropna()
                    .head(20)
                    .tolist()
                )

                if len(sample_values) == 0:
                    continue

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

                has_letters = contains_letters.any()
                has_numbers = contains_numbers.any()

                separators = [
                    "-",
                    "_",
                    "|",
                    "/"
                ]

                found_separator = None

                for sep in separators:

                    matches = [
                        sep in str(val)
                        for val in sample_values
                    ]

                    if (
                        sum(matches)
                        >= len(sample_values) * 0.8
                    ):

                        found_separator = sep
                        break

                valid_split_pattern = False

                if found_separator is not None:

                    split_lengths = []

                    for val in sample_values:

                        parts = str(val).split(found_separator)

                        split_lengths.append(
                            len(parts)
                        )

                    if len(set(split_lengths)) == 1:

                        valid_split_pattern = True

                date_like_pattern = any(

                    re.search(
                        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
                        str(val)
                    )

                    for val in sample_values
                )

                time_like_pattern = any(

                    re.search(
                        r'^\d{1,2}:\d{2}',
                        str(val)
                    )

                    for val in sample_values
                )

                if (

                    has_letters
                    and
                    has_numbers
                    and
                    valid_split_pattern
                    and
                    not date_like_pattern
                    and
                    not time_like_pattern

                ):

                    try:

                        split_data = (

                            temp_col
                            .str.split(
                                found_separator,
                                expand=True
                            )

                        )

                        if split_data.shape[1] == 2:

                            self.df[
                                f"{col}_Type"
                            ] = (
                                split_data[0]
                                .str.strip()
                            )

                            self.df[
                                f"{col}_Number"
                            ] = pd.to_numeric(
                                split_data[1],
                                errors='coerce'
                            )

                            columns_to_drop.append(col)

                            self.report.append(
                                f"✔ Split mixed column '{col}'"
                            )

                    except:
                        pass

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
                        f"✔ Simplified '{col}'"
                    )

        # DROP MIXED COLUMNS

        if columns_to_drop:

            self.df.drop(
                columns=columns_to_drop,
                inplace=True
            )

        # CONVERT MOSTLY NUMERIC COLUMNS

        for col in self.df.columns:

            try:

                numeric_check = pd.to_numeric(
                    self.df[col],
                    errors='coerce'
                )

                ratio = numeric_check.notna().mean()

                if ratio > 0.8:

                    self.df[col] = numeric_check

            except:
                pass

        # REPORT

        total_rows = len(self.df)

        duplicate_count = self.df.duplicated().sum()

        self.report.append(
            f"Total Rows: {total_rows}"
        )

        self.report.append(
            f"Duplicate Rows: {duplicate_count}"
        )

        self.report.append(
            f"Missing Values Before Cleaning: {self.total_missing_before}"
        )

        # HANDLE MISSING VALUES

        for col in self.df.columns:

            missing = self.df[col].isnull().sum()

            if missing > 0:

                if pd.api.types.is_numeric_dtype(
                    self.df[col]
                ):

                    if self.missing_option == "Mean":

                        fill_value = self.df[col].mean()

                    elif self.missing_option == "Median":

                        fill_value = self.df[col].median()

                    elif self.missing_option == "Mode":

                        fill_value = self.df[col].mode()[0]

                    elif self.missing_option == "Drop Rows":

                        self.df.dropna(inplace=True)
                        continue

                    self.df[col] = self.df[col].fillna(
                        fill_value
                    )

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

                self.report.append(
                    f"✔ Cleaned missing values in '{col}'"
                )

        # REMOVE DUPLICATES

        before_duplicates = len(self.df)

        self.df.drop_duplicates(inplace=True)

        removed_duplicates = (
            before_duplicates - len(self.df)
        )

        self.report.append(
            f"Removed duplicate rows: {removed_duplicates}"
        )

        # OUTLIERS

        numeric_cols = self.df.select_dtypes(
            include='number'
        ).columns

        for col in numeric_cols:

            Q1 = self.df[col].quantile(0.25)

            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            if self.outlier_option == "Remove Outliers":

                self.df = self.df[
                    (self.df[col] >= lower)
                    &
                    (self.df[col] <= upper)
                ]

            elif self.outlier_option == "Cap Outliers":

                self.df[col] = self.df[col].clip(
                    lower,
                    upper
                )

        self.report.append(
            "Data Cleaning Completed"
        )

        return self.df, self.report