"""
COM748 Task 1.6 — Per-client SHAP divergence analysis.

For each FL client, compute mean(|SHAP|) from the client's XGBoost component
on a sample of its test set. Report top-10 features per client plus a
side-by-side comparison. The narrative: the federated ensemble aggregates
globally but each client preserves client-specific feature signal.

Outputs:
  paper_v2_outputs/per_client_shap_top10.csv        # long format
  paper_v2_outputs/per_client_shap_comparison.csv   # ranks side-by-side
  paper_v2_outputs/per_client_shap_bars.png         # 3-panel bar chart
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated')
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper_v2_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_N = 2000  # SHAP evaluation sample per client
SEED = 42
TOP_K = 10

CLIENTS = ['ulb', 'baf', 'synthetic']

print('=' * 72)
print('Task 1.6 — Per-client SHAP (top-10 mean |SHAP|)')
print('=' * 72)

long_rows = []
per_client_full = {}

for name in CLIENTS:
    print(f'\n[{name}]')
    X = pd.read_csv(os.path.join(DATA_DIR, f'{name}_X_test.csv'))
    xgb = joblib.load(os.path.join(MODELS_DIR, f'fl_{name}_xgb.joblib'))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(SAMPLE_N, len(X)), replace=False)
    X_sample = X.iloc[idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)

    df = pd.DataFrame({
        'Feature': X.columns,
        'MeanAbsSHAP': mean_abs,
    }).sort_values('MeanAbsSHAP', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    df['Dataset'] = name.upper() if name != 'synthetic' else 'Synthetic'
    per_client_full[name] = df

    top10 = df.head(TOP_K)
    print(top10[['Rank', 'Feature', 'MeanAbsSHAP']].to_string(index=False))
    for _, r in top10.iterrows():
        long_rows.append({
            'Dataset': r['Dataset'],
            'Rank': int(r['Rank']),
            'Feature': r['Feature'],
            'MeanAbsSHAP': round(float(r['MeanAbsSHAP']), 6),
        })

long_df = pd.DataFrame(long_rows)
long_csv = os.path.join(OUTPUT_DIR, 'per_client_shap_top10.csv')
long_df.to_csv(long_csv, index=False)
print(f'\nWrote {long_csv}')

# Side-by-side comparison: rank 1..10 per client, feature name cell
comp = pd.DataFrame({
    'Rank': range(1, TOP_K + 1),
    'ULB':       per_client_full['ulb'].head(TOP_K)['Feature'].values,
    'BAF':       per_client_full['baf'].head(TOP_K)['Feature'].values,
    'Synthetic': per_client_full['synthetic'].head(TOP_K)['Feature'].values,
})
comp_csv = os.path.join(OUTPUT_DIR, 'per_client_shap_comparison.csv')
comp.to_csv(comp_csv, index=False)
print(f'Wrote {comp_csv}')
print('\nSide-by-side top-10:')
print(comp.to_string(index=False))

# 3-panel bar chart
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
for ax, name in zip(axes, CLIENTS):
    d = per_client_full[name].head(TOP_K).iloc[::-1]  # reverse for horizontal bar top-down
    ax.barh(d['Feature'], d['MeanAbsSHAP'], color='steelblue')
    ax.set_title(name.upper() if name != 'synthetic' else 'Synthetic', fontsize=11)
    ax.set_xlabel('Mean |SHAP|')
    ax.grid(alpha=0.3, axis='x')
fig.suptitle('Per-client SHAP top-10 features (FL XGBoost component)', fontsize=12)
fig.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'per_client_shap_bars.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'Wrote {fig_path}')

# Overlap analysis — how many of each pair's top-10 coincide?
u = set(comp['ULB']); b = set(comp['BAF']); s = set(comp['Synthetic'])
print(f'\nTop-10 overlap: ULB∩BAF={len(u & b)}, ULB∩Synthetic={len(u & s)}, '
      f'BAF∩Synthetic={len(b & s)}, all3={len(u & b & s)}')