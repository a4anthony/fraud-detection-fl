"""
Task 5 — Per-client SHAP at category level.

Computes mean(|SHAP|) per feature on the existing FL XGBoost component
(reproducing run_shap_per_client.py methodology — TreeExplainer on a 2000-row
sample of each client's test set, seed 42), then aggregates by category
according to docs/feature_categories.md.

Outputs:
  results-tables/per_client_shap_categorised.csv
  figures/per_client_shap_categories.png

Honest disclosure: ULB has 28 PCA-anonymised features that all fall in one
category (pca_anonymised), so its category total for that bucket sums 28
contributions while other categories sum 1-2. The figure shows two views:
  (a) raw category sums (left panel) — pca_anonymised dominates ULB
  (b) shares within each client (right panel) — comparable across clients
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results-tables')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
SAMPLE_N = 2000

CLIENTS = ['ulb', 'baf', 'synthetic']

# Category mapping per docs/feature_categories.md
ULB_CAT = {'Time': 'temporal', 'Amount': 'amount'}
ULB_CAT.update({f'V{i}': 'pca_anonymised' for i in range(1, 29)})

BAF_CAT = {
    'income': 'identity', 'name_email_similarity': 'identity',
    'prev_address_months_count': 'account',
    'current_address_months_count': 'account',
    'customer_age': 'identity', 'days_since_request': 'temporal',
    'intended_balcon_amount': 'amount', 'payment_type': 'merchant',
    'zip_count_4w': 'velocity', 'velocity_6h': 'velocity',
    'velocity_24h': 'velocity', 'velocity_4w': 'velocity',
    'bank_branch_count_8w': 'velocity',
    'date_of_birth_distinct_emails_4w': 'identity',
    'employment_status': 'identity', 'credit_risk_score': 'risk_score',
    'email_is_free': 'identity', 'housing_status': 'identity',
    'phone_home_valid': 'identity', 'phone_mobile_valid': 'identity',
    'bank_months_count': 'account', 'has_other_cards': 'account',
    'proposed_credit_limit': 'account', 'foreign_request': 'session',
    'source': 'session', 'session_length_in_minutes': 'session',
    'device_os': 'device', 'keep_alive_session': 'session',
    'device_distinct_emails_8w': 'device', 'device_fraud_count': 'device',
    'month': 'temporal',
}

SYN_CAT = {
    'amount': 'amount', 'transaction_type': 'merchant',
    'merchant_category': 'merchant', 'location': 'location',
    'device_used': 'device', 'time_since_last_transaction': 'temporal',
    'spending_deviation_score': 'amount', 'velocity_score': 'velocity',
    'geo_anomaly_score': 'location', 'payment_channel': 'merchant',
}

CATEGORY_MAPS = {'ulb': ULB_CAT, 'baf': BAF_CAT, 'synthetic': SYN_CAT}

CATEGORY_ORDER = [
    'velocity', 'temporal', 'amount', 'identity', 'device', 'merchant',
    'account', 'session', 'location', 'risk_score', 'pca_anonymised', 'other',
]
CATEGORY_COLOR = {
    'velocity':       '#1f77b4',
    'temporal':       '#ff7f0e',
    'amount':         '#2ca02c',
    'identity':       '#d62728',
    'device':         '#9467bd',
    'merchant':       '#8c564b',
    'account':        '#e377c2',
    'session':        '#7f7f7f',
    'location':       '#bcbd22',
    'risk_score':     '#17becf',
    'pca_anonymised': '#aec7e8',
    'other':          '#cccccc',
}


def main():
    print('=' * 72)
    print('Task 5 — Per-client SHAP categorisation')
    print('=' * 72)

    cat_long_rows = []
    feature_long_rows = []
    per_client_cat_totals = {}

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

        cat_map = CATEGORY_MAPS[name]
        unmapped = [c for c in X.columns if c not in cat_map]
        if unmapped:
            print(f'  WARN unmapped features (will be "other"): {unmapped}')

        for col, mabs in zip(X.columns, mean_abs):
            cat = cat_map.get(col, 'other')
            feature_long_rows.append(dict(
                Client=name.upper() if name != 'synthetic' else 'Synthetic',
                Feature=col, Category=cat,
                MeanAbsSHAP=round(float(mabs), 6),
            ))

        cat_totals = (
            pd.DataFrame({'cat': [cat_map.get(c, 'other') for c in X.columns],
                          'mabs': mean_abs})
            .groupby('cat')['mabs'].sum()
        )
        per_client_cat_totals[name] = cat_totals
        for cat, total in cat_totals.items():
            cat_long_rows.append(dict(
                Client=name.upper() if name != 'synthetic' else 'Synthetic',
                Category=cat,
                SumMeanAbsSHAP=round(float(total), 6),
                ShareWithinClient=round(float(total) / float(cat_totals.sum()), 4),
                NumFeaturesInCategory=int(
                    sum(1 for c in X.columns if cat_map.get(c, 'other') == cat)
                ),
            ))
        print(cat_totals.sort_values(ascending=False).round(4).to_string())

    # Save long-format CSV (for the writing-up step)
    cat_df = pd.DataFrame(cat_long_rows).sort_values(['Client', 'Category'])
    cat_csv = os.path.join(RESULTS_DIR, 'per_client_shap_categorised.csv')
    cat_df.to_csv(cat_csv, index=False)
    print(f'\nWrote {cat_csv}')

    feat_df = pd.DataFrame(feature_long_rows).sort_values(
        ['Client', 'MeanAbsSHAP'], ascending=[True, False])
    feat_csv = os.path.join(RESULTS_DIR, 'per_client_shap_features_full.csv')
    feat_df.to_csv(feat_csv, index=False)
    print(f'Wrote {feat_csv}')

    # Build the figure: two panels side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)

    cats_used = [c for c in CATEGORY_ORDER
                 if any(c in per_client_cat_totals[k].index for k in CLIENTS)]
    n_cat = len(cats_used)
    client_labels = [n.upper() if n != 'synthetic' else 'Synthetic'
                     for n in CLIENTS]
    x = np.arange(len(client_labels))
    width = 0.7 / n_cat

    # Panel A — raw sums (each category gets its own grouped bar)
    for i, cat in enumerate(cats_used):
        vals = [float(per_client_cat_totals[n].get(cat, 0.0)) for n in CLIENTS]
        ax1.bar(x + (i - n_cat / 2 + 0.5) * width, vals, width,
                color=CATEGORY_COLOR.get(cat, '#999999'),
                edgecolor='black', linewidth=0.3, label=cat)
    ax1.set_xticks(x)
    ax1.set_xticklabels(client_labels, fontsize=10)
    ax1.set_ylabel('Sum of mean |SHAP| within category', fontsize=10)
    ax1.set_title('A. Raw category totals\n(ULB pca_anonymised dominates by construction)',
                  fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Panel B — share within each client (stacked bars)
    bottoms = np.zeros(len(CLIENTS))
    for cat in cats_used:
        vals = []
        for n in CLIENTS:
            tot = float(per_client_cat_totals[n].sum())
            v = float(per_client_cat_totals[n].get(cat, 0.0))
            vals.append(v / tot if tot > 0 else 0.0)
        ax2.bar(x, vals, bottom=bottoms,
                color=CATEGORY_COLOR.get(cat, '#999999'),
                edgecolor='black', linewidth=0.3, label=cat)
        bottoms = bottoms + np.array(vals)
    ax2.set_xticks(x)
    ax2.set_xticklabels(client_labels, fontsize=10)
    ax2.set_ylabel('Share of total mean |SHAP| within client', fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_title('B. Within-client shares\n(comparable across clients)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
               framealpha=0.9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    fig.suptitle('Per-client SHAP by feature category — FL XGBoost component',
                 fontsize=12, y=1.00)
    fig.text(0.5, -0.01,
             'TreeSHAP on 2,000-row sample of each client test set (seed 42); '
             'see docs/feature_categories.md for taxonomy.',
             ha='center', fontsize=8.5, color='dimgrey', style='italic')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'per_client_shap_categories.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Wrote {fig_path}')


if __name__ == '__main__':
    main()