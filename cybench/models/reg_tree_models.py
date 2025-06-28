from collections.abc import Iterable
from cybench.models.model import BaseModel
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle
import numpy as np
import logging
import pandas as pd

from cybench.datasets.dataset import Dataset
from cybench.util.data import data_to_pandas
from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, SOIL_PROPERTIES, TIME_SERIES_INPUTS, TIME_SERIES_AGGREGATIONS, METEO_INDICATORS

from sklearn.tree import DecisionTreeRegressor
import os

from cybench.util.features import (
    unpack_time_series,
    design_features,
)

class RegressionTree(BaseModel): 
    def __init__(self):
        self._train_df = None
        self._logger = logging.getLogger(__name__)
        self.model = None
        self._feature_cols = None

    def _design_features(self, crop: str, dataset: Iterable):
        soil_df = data_to_pandas(dataset, data_cols=[KEY_LOC] + SOIL_PROPERTIES)
        soil_df = soil_df.drop_duplicates()

        dfs_x = {"soil": soil_df}
        for x, ts_cols in TIME_SERIES_INPUTS.items():
            df_ts = data_to_pandas(
                dataset, data_cols=[KEY_LOC, KEY_YEAR] + [KEY_DATES] + ts_cols
            )
            df_ts = unpack_time_series(df_ts, ts_cols)
            # fill in NAs
            df_ts = df_ts.astype({k: "float" for k in ts_cols})
            df_ts = (
                df_ts.set_index([KEY_LOC, KEY_YEAR, "date"])
                .sort_index()
                .interpolate(method="linear")
            )
            dfs_x[x] = df_ts.reset_index()

        features = design_features(crop, dfs_x)
        return features
    
    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        train_features = self._design_features("maize", dataset)
        train_labels = data_to_pandas(
            dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
        )
        self._feature_cols = [
            ft for ft in train_features.columns if ft not in [KEY_LOC, KEY_YEAR]
        ]
        #print("feature columns", self._feature_cols)
        train_data = train_features.merge(train_labels, on=[KEY_LOC, KEY_YEAR])
        #print("td", train_data)
        X = train_data[self._feature_cols].values
        # print("len of X", len(X))
        # print("type of X", type(X))
        #print("X[0]", X[0])
        y = train_data[KEY_TARGET].values
        #print("y train", y)
        self.model = DecisionTreeRegressor(**fit_params, max_depth=5)
        self.model.fit(X, y)
        return self, {}
    
    
    def predict_items(self, dataset: Dataset, **predict_params):
        test_features = self._design_features("maize", dataset)
        test_labels = data_to_pandas(
            dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
        )
        ft_cols = list(test_features.columns)[len([KEY_LOC, KEY_YEAR]) :]
        missing_features = [ft for ft in self._feature_cols if ft not in ft_cols]
        for ft in missing_features:
            test_features[ft] = 0.0

        test_features = test_features[[KEY_LOC, KEY_YEAR] + self._feature_cols]
        test_data = test_features.merge(test_labels, on=[KEY_LOC, KEY_YEAR])
        X_test = test_data[self._feature_cols].values
        ret = self.model.predict(X_test)
        return ret, {}
    
    def save(self, model_name):
        """Save model, e.g. using pickle.

        Args:
          model_name: Filename that will be used to save the model.
        """
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    def load(cls, model_name):
        """Deserialize a saved model.

        Args:
          model_name: Filename that was used to save the model.

        Returns:
          The deserialized model.
        """
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model
  
        
        


























# run_name = <run_name>
# dataset_name = "maize_US"
# result = run_benchmark(run_name=run_name, 
#                        model_name="my_model",
#                        model_constructor=MyModel,
#                        model_init_kwargs: <int args>,
#                        model_fit_kwargs: <fit params>,
#                        dataset_name=dataset_name)

# metrics = ["normalized_rmse", "mape", "r2"]
# df_metrics = result["df_metrics"].reset_index()
# print(df_metrics.groupby("model").agg({ m : "mean" for m in metrics }))
