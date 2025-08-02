# src/models/linear_regression_model.py

from sklearn.linear_model import LinearRegression
import numpy as np
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self, vehicle_name):
        super().__init__('LinearRegression', vehicle_name)

    def _get_features_and_target(self, data, model_type):
        base_features = ['speed', 'acceleration', 'ext_temp']
        rolling_features = [col for col in data.columns if 'roll' in col]
        features_to_use = base_features + rolling_features
        X = data[features_to_use]
        
        if model_type == 'hybrid':
            y = data['target'] - data['physics_prediction']
        else: # ml_only
            y = data['target']
            
        return X, y

    def find_best_params(self, train_data, storage_path):
        # Linear Regression은 튜닝할 하이퍼파라미터가 없으므로 빈 dict를 반환합니다.
        # storage_path 인자는 다른 모델과의 호환성을 위해 유지합니다.
        print("LinearRegression does not require hyperparameter tuning.")
        return {}

    def train(self, train_data, params, model_type='hybrid'):
        X_train, y_train = self._get_features_and_target(train_data, model_type)
        self.model = LinearRegression(**params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, test_data, model_type='hybrid'):
        X_test, _ = self._get_features_and_target(test_data, model_type)
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        
        predictions = self.model.predict(X_test)
        
        if model_type == 'hybrid':
            final_predictions = predictions + test_data['physics_prediction'].values
            return final_predictions
        else: # ml_only
            return predictions