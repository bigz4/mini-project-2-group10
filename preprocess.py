"""
preprocess.py
Reusable data loading + preprocessing builders for Mini Project II (Loan Approval Prediction).

Design goals:
- Leakage-safe: all transforms are inside sklearn/imblearn Pipelines.
- Colab-free compatible: lightweight defaults.
- Supports A-upgrade: missingness indicators + categorical-aware oversampling (SMOTENC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


TARGET_COL = "Loan_Status"


@dataclass(frozen=True)
class DatasetSpec:
    target_col: str = TARGET_COL
    positive_label: str = "Y"
    negative_label: str = "N"


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV path."""
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")
    return df


def split_X_y(df: pd.DataFrame, spec: DatasetSpec = DatasetSpec()) -> Tuple[pd.DataFrame, pd.Series]:
    """Split into X and y (binary 0/1)."""
    y = df[spec.target_col].map({spec.positive_label: 1, spec.negative_label: 0})
    if y.isna().any():
        bad = df.loc[y.isna(), spec.target_col].unique().tolist()
        raise ValueError(f"Unexpected target values encountered: {bad}")
    X = df.drop(columns=[spec.target_col])
    return X, y


def add_missingness_indicators(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add *_missing indicator columns for every feature with missing values."""
    X2 = X.copy()
    indicators: List[str] = []
    for col in X2.columns:
        if X2[col].isna().any():
            ind_col = f"{col}_missing"
            X2[ind_col] = X2[col].isna().astype(int)
            indicators.append(ind_col)
    return X2, indicators


def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical features based on dtype."""
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric, categorical


def build_preprocessor_onehot(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Preprocessor for linear models: numeric median+scale; categorical mode+onehot."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_preprocessor_smotenc(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[ColumnTransformer, List[int]]:
    """Preprocessor for SMOTENC: numeric median+scale; categorical mode+ordinal."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    n_num = len(numeric_features)
    n_cat = len(categorical_features)
    cat_indices = list(range(n_num, n_num + n_cat))
    return preprocessor, cat_indices


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
