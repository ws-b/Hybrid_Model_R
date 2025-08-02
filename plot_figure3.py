# plot_final_figure.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import SELECTED_VEHICLE, SAMPLING_SIZES, MODELS_TO_RUN

# ===================================================================
# 그래프에서 제외할 모델 목록을 여기에 기입하세요.
# 예: ['Hybrid_Transformer', 'OnlyML_Transformer', 'OnlyML_LinearRegression']
MODELS_TO_EXCLUDE = ['Hybrid_Transformer', 'OnlyML_Transformer', 'Hybrid_RandomForest', 'OnlyML_RandomForest']
# ===================================================================

def create_final_plot():
    """
    Loads all experiment logs, processes the data, and generates the final
    normalized RMSE plot, similar to Figure 3 in the paper.
    """
    log_dir = os.path.join('results', 'logs', SELECTED_VEHICLE)
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    # 1. Load all result files into a DataFrame
    all_results = []
    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            log_path = os.path.join(log_dir, filename)
            with open(log_path, 'r') as f:
                data = json.load(f)
                
                model_name = filename.split('_size_')[0]
                
                all_results.append({
                    'sample_size': data['sample_size'],
                    'model_name': model_name,
                    'hybrid_rmse': data['results'].get('hybrid_rmse'),
                    'ml_only_rmse': data['results'].get('ml_only_rmse'),
                    'physics_rmse': data['results'].get('physics_rmse_on_sample')
                })

    if not all_results:
        print("No result files found to process.")
        return

    df = pd.DataFrame(all_results)
    df.dropna(inplace=True)

    # 2. Calculate Mean and Standard Deviation for each sample size
    summary_data = {}
    for size in sorted(df['sample_size'].unique()):
        size_df = df[df['sample_size'] == size]
        summary_data[size] = {
            'physics_mean': size_df['physics_rmse'].mean(),
            'physics_std': size_df['physics_rmse'].std(),
        }
        for model in MODELS_TO_RUN:
            model_df = size_df[size_df['model_name'] == model]
            summary_data[size][f'hybrid_{model}_mean'] = model_df['hybrid_rmse'].mean()
            summary_data[size][f'hybrid_{model}_std'] = model_df['hybrid_rmse'].std()
            summary_data[size][f'ml_only_{model}_mean'] = model_df['ml_only_rmse'].mean()
            summary_data[size][f'ml_only_{model}_std'] = model_df['ml_only_rmse'].std()

    # 3. Generate plots: one with error bars and one without
    generate_plot(summary_data, use_errorbars=True)
    generate_plot(summary_data, use_errorbars=False)

def generate_plot(summary_data, use_errorbars=True):
    """Generates and saves the normalized RMSE plot."""
    
    sizes = sorted(summary_data.keys())
    
    # --- Prepare data for plotting ---
    plot_data = {}
    model_keys = ['Physics_Based'] + [f"Hybrid_{m}" for m in MODELS_TO_RUN] + [f"OnlyML_{m}" for m in MODELS_TO_RUN]
    
    for key in model_keys:
        plot_data[f"{key}_mean"] = []
        plot_data[f"{key}_std"] = []

    for size in sizes:
        data = summary_data[size]
        phys_mean = data['physics_mean'] if data['physics_mean'] > 0 else 1.0
        
        for model in MODELS_TO_RUN:
            h_key = f"hybrid_{model}"
            plot_data[f"Hybrid_{model}_mean"].append(data[f"{h_key}_mean"] / phys_mean)
            plot_data[f"Hybrid_{model}_std"].append(data[f"{h_key}_std"] / phys_mean)
            ml_key = f"ml_only_{model}"
            plot_data[f"OnlyML_{model}_mean"].append(data[f"{ml_key}_mean"] / phys_mean)
            plot_data[f"OnlyML_{model}_std"].append(data[f"{ml_key}_std"] / phys_mean)
        
        plot_data["Physics_Based_mean"].append(1.0)
        plot_data["Physics_Based_std"].append(0.0)

    # --- Plotting ---
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6.5))

    color_palette = [
        '#0073c2', '#efc000', '#cd534c', '#20854e', '#925e9f',
        '#e18727', '#4dbbd5', '#ee4c97', '#7e6148'
    ]
    
    model_styles = {
        'XGBoost': {'marker': 'o'}, 'RandomForest': {'marker': 's'},
        'MLP': {'marker': 'D'}, 'Transformer': {'marker': '^'},
        'LinearRegression': {'marker': 'p'},
    }
    
    ax.plot(sizes, plot_data["Physics_Based_mean"], label='Physics-Based', linestyle='--', color='#747678', linewidth=2)
    
    color_index = 0
    for model_name in MODELS_TO_RUN:
        color = color_palette[color_index % len(color_palette)]
        marker = model_styles.get(model_name, {}).get('marker', 'x')
        
        # Hybrid 모델
        h_key = f"Hybrid_{model_name}"
        if h_key not in MODELS_TO_EXCLUDE: # [수정] 제외 목록에 없으면 그리기
            h_mean = plot_data.get(f"{h_key}_mean")
            h_std = plot_data.get(f"{h_key}_std")
            if h_mean and not pd.isna(h_mean).all():
                if use_errorbars:
                    ax.errorbar(sizes, h_mean, yerr=h_std, label=f'Hybrid ({model_name})', marker=marker, color=color, capsize=3, linewidth=1.5, markersize=7)
                else:
                    ax.plot(sizes, h_mean, label=f'Hybrid ({model_name})', marker=marker, color=color, linewidth=1.5, markersize=7)

        # Only ML 모델
        ml_key = f"OnlyML_{model_name}"
        if ml_key not in MODELS_TO_EXCLUDE: # [수정] 제외 목록에 없으면 그리기
            ml_mean = plot_data.get(f"{ml_key}_mean")
            ml_std = plot_data.get(f"{ml_key}_std")
            if ml_mean and not pd.isna(ml_mean).all():
                if use_errorbars:
                    ax.errorbar(sizes, ml_mean, yerr=ml_std, label=f'Only ML ({model_name})', marker=marker, color=color, capsize=3, linewidth=1.5, markersize=7, mfc='white')
                else:
                    ax.plot(sizes, ml_mean, label=f'Only ML ({model_name})', marker=marker, color=color, linewidth=1.5, markersize=7, mfc='white')
        
        color_index += 1

    ax.set_xlabel('Number of Trips', fontsize=12)
    ax.set_ylabel('Normalized RMSE', fontsize=12)
    ax.set_title(f'Normalized RMSE vs. Number of Trips for {SELECTED_VEHICLE}', fontsize=14, weight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(False)
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.minorticks_off()
    ax.set_facecolor('#f7f7f7')

    plt.tight_layout()

    # --- Save Figure ---
    output_dir = os.path.join('results', 'plots', SELECTED_VEHICLE)
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = "with_errorbars" if use_errorbars else "no_errorbars"
    save_filename = f"{SELECTED_VEHICLE}_final_rmse_plot_{suffix}.png"
    save_path = os.path.join(output_dir, save_filename)
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    create_final_plot()