from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class PgxPipeline:
    def __init__(self, target_col, data_frame) -> None:
        self.df = data_frame
        self.target_col = target_col
        self.num_cols = ["MW", "Rrup", "VS30", target_col]
        self.log_cols = ["Rrup", "VS30", target_col]
        self.cat_cols = ["FaultType"]

    def log_func(self, X):
        new_df = X.copy()
        new_df[self.log_cols] = np.log10(new_df[self.log_cols])

        return new_df

    def get_pipeline(self):
        ct = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), self.num_cols),
                ("encoder", OneHotEncoder(drop="first"), self.cat_cols),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            [("log10", FunctionTransformer(self.log_func)), ("col_trans", ct)]
        )
        pipeline.fit(self.df)

        return pipeline


class SpectralAccelerationPipeline:
    def __init__(self, data_frame) -> None:
        self.df = data_frame
        self.num_cols = ["Mw", "Rrup", "VS30", "Period", "SA"]
        self.log_cols = ["Rrup", "VS30", "SA"]
        self.cat_cols = ["FaultType"]

    def log_func(self, X):
        new_df = X.copy()
        new_df[self.log_cols] = np.log10(new_df[self.log_cols])

        return new_df

    def get_pipeline(self):
        ct = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), self.num_cols),
                ("encoder", OneHotEncoder(drop="first"), self.cat_cols),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            [("log10", FunctionTransformer(self.log_func)), ("col_trans", ct)]
        )
        pipeline.fit(self.df)

        return pipeline


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_index=3):
        self.column_index = column_index

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.delete(X, self.column_index, axis=1)


class KerasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.model.predict(X)


def get_pgx_pred(pipeline, Mw, Rrup, VS30, fault_type, target_col):
    df = pd.DataFrame(
        [[Mw, fault_type, Rrup, VS30, 1]],
        columns=["MW", "FaultType", "Rrup", "VS30", target_col],
    )
    pred_scaled = pipeline.predict(df)
    scaler = (
        pipeline.named_steps["preprocess"].named_steps["col_trans"].transformers_[0][1]
    )
    pred = scaler.inverse_transform([[1, 1, 1, pred_scaled[0]]])[-1][-1]
    return 10**pred


def get_sa_pred(process_pipeline, tf_model, Mw, Rrup, VS30, fault_type, periods):
    dataset = [[Mw, fault_type, Rrup, VS30, period, 1] for period in periods]
    df = pd.DataFrame(
        dataset,
        columns=["MW", "FaultType", "Rrup", "VS30", "Period", "SA"],
    )
    df_scaled = process_pipeline.transform(df)
    pred_scaled = tf_model.predict(df_scaled).flatten()

    scaler = (
        process_pipeline.named_steps["preprocess"]
        .named_steps["col_trans"]
        .transformers_[0][1]
    )
    pred = scaler.inverse_transform([[1, 1, 1, 1, sa] for sa in pred_scaled])[:, -1]

    return 10**pred
