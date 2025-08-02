# process_results.py

import os
import json
import pandas as pd
from config import SELECTED_VEHICLE, SAMPLING_SIZES, MODELS_TO_RUN

def process_results():
    """
    Reads all log files, aggregates the results, and prints a final summary table
    with 'Physics_Based' model as the first column.
    """
    log_dir = os.path.join('results', 'logs', SELECTED_VEHICLE)
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    all_results = []
    # 한 번만 추가하기 위한 플래그
    physics_added_for_iteration = set()

    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            log_path = os.path.join(log_dir, filename)
            with open(log_path, 'r') as f:
                data = json.load(f)
                
                model_name = filename.split('_size_')[0]
                iteration_key = (data['sample_size'], data['iteration'])
                
                # 물리 모델은 각 iteration 당 한 번만 추가
                if iteration_key not in physics_added_for_iteration:
                    all_results.append({
                        'sample_size': data['sample_size'],
                        'model': "Physics_Based",
                        'rmse': data['results'].get('physics_rmse_on_sample')
                    })
                    physics_added_for_iteration.add(iteration_key)

                # ML 모델 결과 추가
                all_results.append({
                    'sample_size': data['sample_size'],
                    'model': f"Hybrid_{model_name}",
                    'rmse': data['results'].get('hybrid_rmse')
                })
                all_results.append({
                    'sample_size': data['sample_size'],
                    'model': f"OnlyML_{model_name}",
                    'rmse': data['results'].get('ml_only_rmse')
                })

    if not all_results:
        print("No result files found to process.")
        return

    df = pd.DataFrame(all_results)
    df.dropna(subset=['rmse'], inplace=True)

    summary = df.groupby(['sample_size', 'model'])['rmse'].agg(['mean', 'std']).reset_index()
    
    pivot_mean = summary.pivot(index='sample_size', columns='model', values='mean')
    pivot_std = summary.pivot(index='sample_size', columns='model', values='std')

    # 1. 현재 컬럼 목록을 가져옵니다.
    all_model_columns = sorted(pivot_mean.columns.tolist())
    
    # 2. 원하는 순서로 재구성합니다.
    ordered_columns = []
    if 'Physics_Based' in all_model_columns:
        ordered_columns.append('Physics_Based')
        all_model_columns.remove('Physics_Based')
    
    # Hybrid 모델들을 먼저 추가 (정렬된 순서로)
    hybrid_cols = sorted([col for col in all_model_columns if 'Hybrid' in col])
    ordered_columns.extend(hybrid_cols)

    # OnlyML 모델들을 추가 (정렬된 순서로)
    onlyml_cols = sorted([col for col in all_model_columns if 'OnlyML' in col])
    ordered_columns.extend(onlyml_cols)
    
    # 3. 데이터프레임의 컬럼 순서를 재적용합니다.
    pivot_mean = pivot_mean.reindex(columns=ordered_columns)
    pivot_std = pivot_std.reindex(columns=ordered_columns)
    # -------------------------------

    # 최종 결과 출력
    print("\n" + "="*80)
    print(f"Final Aggregated Results for: {SELECTED_VEHICLE}")
    print("="*80)
    
    print("\n--- Average RMSE ---")
    print(pivot_mean.to_string(float_format="%.2f"))
    
    print("\n\n--- Standard Deviation of RMSE ---")
    print(pivot_std.to_string(float_format="%.2f"))
    print("\n" + "="*80)


if __name__ == "__main__":
    process_results()