# delete_study.py
import optuna
import os
from config import SELECTED_VEHICLE

# 삭제할 모델 
MODEL_TO_DELETE = "MLP"

# Optuna DB 경로 설정
storage_path = f"sqlite:///{os.path.join('results', 'optuna', 'optuna_study.db')}"
study_name_to_delete = f"{MODEL_TO_DELETE}-{SELECTED_VEHICLE}-tuning"

print(f"Attempting to delete study: '{study_name_to_delete}'")

try:
    optuna.delete_study(study_name=study_name_to_delete, storage=storage_path)
    print(f"Successfully deleted study '{study_name_to_delete}'.")
except KeyError:
    print(f"Study '{study_name_to_delete}' not found. Nothing to delete.")
except Exception as e:
    print(f"An error occurred: {e}")