"""
Shared preprocessing classes for the US Used Vehicle Prices project.

All custom sklearn transformers live here so that:
  1. Notebooks 02, 03a, 03b, 03c, and 04 can all import from the same place.
  2. joblib.load() can resolve class references correctly across notebooks.

Usage (in any notebook):
    from preprocessing import (
        OdometerBinnedYearImputer,
        YearBinnedOdometerImputer,
        OdometerBinnedConditionImputer,
        ModelBinnedManufacturerImputer,
        CascadingCategoricalImputer,
        MeanTargetEncoder,
    )
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# Imputers
# ============================================================

BIN_SIZE = 3000


class OdometerBinnedYearImputer(BaseEstimator, TransformerMixin):

    def __init__(self, bin_size=BIN_SIZE):
        self.bin_size = bin_size

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=["year", "odometer"])
        odo_bin = (X["odometer"] // self.bin_size) * self.bin_size
        self.mapping_ = X.groupby(odo_bin)["year"].median().to_dict()
        self.fallback_ = X["year"].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        mask = X["year"].isna()
        odo_bin = (X["odometer"] // self.bin_size) * self.bin_size
        X.loc[mask, "year"] = odo_bin[mask].map(self.mapping_).fillna(self.fallback_)
        return X


class YearBinnedOdometerImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=["year", "odometer"])
        self.mapping_ = X.groupby("year")["odometer"].median().to_dict()
        self.fallback_ = X["odometer"].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        mask = X["odometer"].isna()
        X.loc[mask, "odometer"] = X.loc[mask, "year"].map(self.mapping_).fillna(self.fallback_)
        return X


class OdometerBinnedConditionImputer(BaseEstimator, TransformerMixin):

    def __init__(self, bin_edges=None, bin_labels=None, fallback="good"):
        self.bin_edges  = bin_edges  or [0, 50_000, 100_000, 150_000, 200_000, np.inf]
        self.bin_labels = bin_labels or ["0-50K", "50K-100K", "100K-150K", "150K-200K", "200K+"]
        self.fallback   = fallback

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=["odometer", "condition"]).copy()
        X["_cond_bin"] = pd.cut(X["odometer"], bins=self.bin_edges, labels=self.bin_labels, right=False)
        valid = X.dropna(subset=["condition", "_cond_bin"])
        self.mapping_ = (
            valid.groupby(["_cond_bin", "condition"], observed=True)
            .size()
            .reset_index(name="_n")
            .sort_values("_n", ascending=False)
            .drop_duplicates("_cond_bin")
            .set_index("_cond_bin")["condition"]
            .to_dict()
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        mask = X["condition"].isna()
        cond_bin = pd.cut(X["odometer"], bins=self.bin_edges, labels=self.bin_labels, right=False)
        X.loc[mask, "condition"] = cond_bin[mask].map(self.mapping_).fillna(self.fallback)
        return X


class ModelBinnedManufacturerImputer(BaseEstimator, TransformerMixin):

    def __init__(self, fallback="unknown"):
        self.fallback = fallback

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=["model", "manufacturer"])
        valid = X.dropna(subset=["model", "manufacturer"])
        self.mapping_ = (
            valid.groupby(["model", "manufacturer"])
            .size()
            .reset_index(name="_n")
            .sort_values("_n", ascending=False)
            .drop_duplicates("model")
            .set_index("model")["manufacturer"]
            .to_dict()
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        mask = X["manufacturer"].isna()
        X.loc[mask, "manufacturer"] = (
            X.loc[mask, "model"].map(self.mapping_).fillna(self.fallback)
        )
        return X


class CascadingCategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, target_cols=None, fallback="unknown"):
        self.target_cols = target_cols or ["fuel", "transmission", "drive", "type",
                                           "title_status", "paint_color", "state"]
        self.fallback = fallback

    def _vectorized_mode(self, df, group_col, value_col):
        valid = df.dropna(subset=[group_col, value_col])
        return (
            valid.groupby([group_col, value_col])
            .size()
            .reset_index(name="_n")
            .sort_values("_n", ascending=False)
            .drop_duplicates(group_col)
            .set_index(group_col)[value_col]
            .to_dict()
        )

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.mode_by_model_ = {}
        self.mode_by_mfr_   = {}
        self.mode_overall_  = {}

        for col in self.target_cols:
            self.mode_by_model_[col] = self._vectorized_mode(X, "model", col)
            self.mode_by_mfr_[col]   = self._vectorized_mode(X, "manufacturer", col)
            mode = X[col].mode()
            self.mode_overall_[col]  = mode.iloc[0] if not mode.empty else self.fallback

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.target_cols:
            mask = X[col].isna()
            if not mask.any():
                continue

            imputed = pd.Series(self.mode_overall_[col], index=X.index)

            mfr_match = X["manufacturer"].map(self.mode_by_mfr_[col])
            imputed = imputed.where(mfr_match.isna(), mfr_match)

            model_match = X["model"].map(self.mode_by_model_[col])
            imputed = imputed.where(model_match.isna(), model_match)

            X.loc[mask, col] = imputed[mask]

        return X


# ============================================================
# Encoders
# ============================================================

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces each category with the mean target value learned from training data.
    Operates on all columns passed to it by ColumnTransformer.
    Unseen categories fall back to the global mean.
    Used instead of sklearn's TargetEncoder which allocates a dense
    (n_samples x n_categories) indicator matrix — infeasible for high-cardinality
    columns like 'model' with 24K+ unique values.
    """

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(y_arr.mean())
        self.encodings_ = {}
        for col in X.columns:
            tmp = pd.DataFrame({"cat": X[col].values, "target": y_arr})
            self.encodings_[col] = tmp.groupby("cat")["target"].mean().to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.encodings_.get(col, {})).fillna(self.global_mean_)
        # Return numpy array so sklearn's set_output(transform="pandas")
        # can wrap it correctly using get_feature_names_out().
        return X.values

    def get_feature_names_out(self, input_features=None):
        """Return feature names so ColumnTransformer can propagate them."""
        if input_features is not None:
            return np.array(input_features)
        return np.array(list(self.encodings_.keys()))
