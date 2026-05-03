import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.report = []

    def process(self):

        # ================= BEFORE CLEANING =================
        total_rows = len(self.df)
        duplicate_count = self.df.duplicated().sum()
        null_counts = self.df.isnull().sum()

        self.report.append(f"Total Rows: {total_rows}")
        self.report.append(f"Duplicate Rows: {duplicate_count}")

        for col in null_counts.index:
            if null_counts[col] > 0:
                self.report.append(f"Missing in '{col}': {null_counts[col]}")

        # ================= HANDLE MISSING =================
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # ================= REMOVE DUPLICATES =================
        self.df.drop_duplicates(inplace=True)

        # ================= OUTLIER DETECTION =================
        outlier_count = 0

        for col in self.df.select_dtypes(include='number').columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if outliers > 0:
                self.report.append(f"Outliers in '{col}': {outliers}")
                outlier_count += outliers

        # ================= AFTER CLEANING =================
        self.report.append("Data Cleaning Completed")
        self.report.append("Missing values handled")
        self.report.append("Duplicates removed")

        if outlier_count == 0:
            self.report.append("No significant outliers detected")

        return self.df, self.report