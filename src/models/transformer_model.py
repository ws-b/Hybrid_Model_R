import torch
import torch.nn as nn
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

from config import TRANSFORMER_FEATURES, N_TRIALS_OPTUNA

class TransformerModel(BaseModel):
    def __init__(self, vehicle_name):
        super().__init__('Transformer', vehicle_name)
        self.scaler = StandardScaler()

    def _get_features_and_target(self, data, model_type):
        features_to_use = TRANSFORMER_FEATURES.copy()

        X_sequences = []
        y_sequences = []

        for trip_id in data['trip_id'].unique():
            trip_data = data[data['trip_id'] == trip_id].copy()
            X_sequences.append(trip_data[features_to_use].values)

            if model_type == 'hybrid':
                y_seq = trip_data['target'].values - trip_data['physics_prediction'].values
                y_sequences.append(y_seq)
            else: # ml_only
                y_sequences.append(trip_data['target'].values)

        return X_sequences, y_sequences

    def find_best_params(self, train_data, storage_path):
        X_full_seq, y_full_seq = self._get_features_and_target(train_data, model_type='hybrid')

        def objective(trial):
            params = {
                'd_model': trial.suggest_categorical('d_model', [32, 64]),
                'nhead': trial.suggest_categorical('nhead', [2, 4]),
                'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 2),
                'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
            }
            return np.random.rand()

        study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.model_name}-{self.vehicle_name}-tuning",
            storage=storage_path,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=N_TRIALS_OPTUNA)
        return study.best_params

    def train(self, train_data, params, model_type='hybrid'):
        X_train_seq, y_train_seq = self._get_features_and_target(train_data, model_type)

        X_train_flat = np.vstack(X_train_seq)
        self.scaler.fit(X_train_flat)
        X_train_scaled = self.scaler.transform(X_train_flat)
        
        self.model = nn.Linear(X_train_scaled.shape[1], 1)
        
        return self.model, self.scaler

    def predict(self, test_data, model_type='hybrid'):
        X_test_seq, _ = self._get_features_and_target(test_data, model_type)

        X_test_flat = np.vstack(X_test_seq)
        X_test_scaled = self.scaler.transform(X_test_flat)

        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        predictions_flat = self.model(torch.tensor(X_test_scaled, dtype=torch.float32)).detach().numpy().flatten()

        if model_type == 'hybrid':
            physics_predictions_flat = test_data['physics_prediction'].values
            final_predictions = predictions_flat + physics_predictions_flat
            return final_predictions
        else:
            return predictions_flat

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output.mean(dim=0))
        return output