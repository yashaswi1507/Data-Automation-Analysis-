import pandas as pd
import numpy as np
import re


class DatasetProfiler:

    def __init__(self, df):

        self.df = df.copy()

        self.profile_report = {}

    # =====================================================
    # MAIN FUNCTION
    # =====================================================

    def profile_dataset(self):

        for col in self.df.columns:

            profile = self.profile_column(col)

            self.profile_report[col] = profile

        return self.profile_report

    # =====================================================
    # PROFILE SINGLE COLUMN
    # =====================================================

    def profile_column(self, col):

        series = self.df[col]

        profile = {}

        # BASIC INFO

        profile["dtype"] = str(series.dtype)

        profile["missing_count"] = (
            series.isnull().sum()
        )

        profile["missing_percent"] = round(

            (
                series.isnull().sum()
                /
                len(series)
            ) * 100,

            2
        )

        profile["unique_count"] = (
            series.nunique()
        )

        profile["unique_percent"] = round(

            (
                series.nunique()
                /
                len(series)
            ) * 100,

            2
        )

        # SAMPLE VALUES

        sample_values = (

            series.dropna()
            .astype(str)
            .head(20)
            .tolist()

        )

        profile["sample_values"] = sample_values

        # DETECT COLUMN TYPE

        detected_type = self.detect_column_type(
            series,
            sample_values
        )

        profile["detected_type"] = detected_type

        # CLEANING STRATEGY

        cleaning_strategy = (
            self.detect_cleaning_strategy(
                detected_type
            )
        )

        profile["cleaning_strategy"] = (
            cleaning_strategy
        )

        return profile

    # =====================================================
    # DETECT COLUMN TYPE
    # =====================================================

    def detect_column_type(
        self,
        series,
        sample_values
    ):

        # NUMERIC

        numeric_check = pd.to_numeric(
            series,
            errors="coerce"
        )

        numeric_ratio = (
            numeric_check.notna().mean()
        )

        if numeric_ratio > 0.85:

            return "numeric"

        # DATE

        try:

            date_check = pd.to_datetime(
                series,
                errors="coerce"
            )

            date_ratio = (
                date_check.notna().mean()
            )

            if date_ratio > 0.8:

                return "date"

        except:
            pass

        # IDENTIFIER

        unique_ratio = (
            series.nunique()
            /
            len(series)
        )

        if unique_ratio > 0.9:

            avg_length = np.mean([

                len(str(v))

                for v in sample_values

            ])

            if avg_length < 25:

                return "identifier"

        # ADDRESS

        address_keywords = [

            "street",
            "road",
            "sector",
            "block",
            "lane",
            "avenue",
            "city",
            "flat"
        ]

        address_score = 0

        for val in sample_values:

            val_lower = str(val).lower()

            if any(

                word in val_lower

                for word in address_keywords

            ):

                address_score += 1

        if address_score >= 3:

            return "address"

        # NAME

        name_pattern_count = 0

        for val in sample_values:

            if re.fullmatch(

                r"[A-Za-z ]+",

                str(val)

            ):

                name_pattern_count += 1

        if name_pattern_count >= len(sample_values) * 0.7:

            return "name"

        # STRUCTURED CODE

        structured_count = 0

        separators = [
            "-",
            "_",
            "/",
            "|"
        ]

        for val in sample_values:

            val = str(val)

            has_letter = bool(
                re.search(r"[A-Za-z]", val)
            )

            has_number = bool(
                re.search(r"\d", val)
            )

            has_separator = any(
                sep in val
                for sep in separators
            )

            if (
                has_letter
                and
                has_number
                and
                has_separator
            ):

                structured_count += 1

        if structured_count >= len(sample_values) * 0.7:

            return "structured_code"

        # CATEGORY

        if unique_ratio < 0.2:

            return "category"

        # TEXT

        return "text"

    # =====================================================
    # CLEANING STRATEGY
    # =====================================================

    def detect_cleaning_strategy(
        self,
        detected_type
    ):

        strategies = {

            "numeric": "median",

            "category": "mode",

            "identifier": "unknown",

            "structured_code": "preserve",

            "address": "unknown",

            "name": "unknown",

            "date": "forward_fill",

            "text": "unknown"
        }

        return strategies.get(
            detected_type,
            "unknown"
        )

