"""
Build figures/fraudx_ai_matched_comparison.png from results-tables/fraudx_ai_matched.csv.

By default produces the 3-bar Task 1 figure:
    federated_smote  |  centralised_no_smote_w_2_0  |  centralised_no_smote_w_580 (sensitivity)
                     |  + fraudx_ai_reported as a horizontal reference line

Run with --include-federated-cw to include the Task 2 federated class-weighted
result as a fourth bar (after Task 2's CSV exists).

300 DPI, navy + gold palette consistent with existing project figures, every
bar annotated with its exact AUPRC value, sensitivity bar visually distinct
(hatched + lighter colour) so the headline match is unambiguous.
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results-tables')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

NAVY = '#1f3a5f'
NAVY_LIGHT = '#5d7ba0'
GOLD = '#c9a227'
GOLD_LIGHT = '#e3c876'
GREY_REF = '#7a7a7a'


def load_auprc_at_05(csv_path, configuration):
    df = pd.read_csv(csv_path)
    row = df[(df['Configuration'] == configuration) & (df['Threshold'] == '0.5')]
    if row.empty:
        raise KeyError(f"No row for {configuration} @ 0.5 in {csv_path}")
    return float(row.iloc[0]['AUPRC'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--include-federated-cw', action='store_true',
                   help='Add Task 2 federated class-weighted bar (requires '
                        'results-tables/federated_class_weighted.csv to exist)')
    p.add_argument('--out', default=os.path.join(FIG_DIR,
                                                 'fraudx_ai_matched_comparison.png'))
    args = p.parse_args()

    matched_csv = os.path.join(RESULTS_DIR, 'fraudx_ai_matched.csv')
    matched = pd.read_csv(matched_csv)

    # Pull AUPRCs at threshold 0.5 (FraudX AI's reference operating point).
    # All Task 1/Task 2 rows are evaluated on the 50% eval slice of the test
    # set (val/eval split with seed 42), matching the convention in
    # retrain_weighted.py / weighted_vs_unweighted.csv. We anchor with
    # federated_smote_eval_half (NOT federated_smote_full_test) so all bars
    # are on the same eval set — apples to apples.
    fed_smote = float(matched[(matched['Configuration'] == 'federated_smote_eval_half')
                              & (matched['Threshold'] == '0.5')].iloc[0]['AUPRC'])
    matched_w2 = float(matched[(matched['Configuration'] == 'centralised_no_smote_w_2_0')
                               & (matched['Threshold'] == '0.5')].iloc[0]['AUPRC'])
    sens_w580 = float(matched[(matched['Configuration'] == 'centralised_no_smote_w_580')
                              & (matched['Threshold'] == '0.5')].iloc[0]['AUPRC'])
    fraudx_ref = float(matched[matched['Configuration'] == 'fraudx_ai_reported'].iloc[0]['AUPRC'])

    bars = [
        dict(label='Federated SMOTE\n(this work, baseline)',
             value=fed_smote, color=NAVY, hatch=None,
             source='weighted_vs_unweighted.csv (eval half)'),
        dict(label='Centralised, no SMOTE\nw = {0:1, 1:2}  (matched)',
             value=matched_w2, color=GOLD, hatch=None,
             source='fraudx_ai_matched.csv (eval half)'),
        dict(label='Centralised, no SMOTE\nw = natural ratio (sensitivity)',
             value=sens_w580, color=GOLD_LIGHT, hatch='////',
             source='fraudx_ai_matched.csv (eval half)'),
    ]

    if args.include_federated_cw:
        fcw_csv = os.path.join(RESULTS_DIR, 'federated_class_weighted.csv')
        if not os.path.exists(fcw_csv):
            raise FileNotFoundError(
                f'--include-federated-cw was passed but {fcw_csv} does not exist.')
        fcw_auprc = load_auprc_at_05(fcw_csv, 'federated_class_weighted_ulb')
        # Insert before the sensitivity bar so the narrative goes:
        # baseline → matched → federated CW → sensitivity
        bars.insert(2, dict(
            label='Federated, no SMOTE\nclass-weighted (this work)',
            value=fcw_auprc, color=NAVY_LIGHT, hatch=None,
            source='federated_class_weighted.csv',
        ))

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    xs = list(range(len(bars)))
    rects = ax.bar(
        xs,
        [b['value'] for b in bars],
        color=[b['color'] for b in bars],
        edgecolor='black',
        linewidth=0.8,
        hatch=[b['hatch'] for b in bars],
    )

    # FraudX AI reference line
    ax.axhline(fraudx_ref, color=GREY_REF, linestyle='--', linewidth=1.4,
               zorder=0)
    ax.text(len(bars) - 0.5, fraudx_ref + 0.005,
            f'FraudX AI reported = {fraudx_ref:.4f}\n'
            f'(Baisholan et al. 2025, Table 2; full test set)',
            ha='right', va='bottom', fontsize=9, color=GREY_REF, style='italic')

    # Annotate each bar with exact value
    for rect, b in zip(rects, bars):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.008,
                f'{b["value"]:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels([b['label'] for b in bars], fontsize=10)
    ax.set_ylabel('AUPRC (ULB eval half, n=28,481, t = 0.5)', fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    title = ('FraudX AI matched comparison — ULB eval half (n=28,481, '
             '49 frauds, 0.172%)')
    if args.include_federated_cw:
        title += '\nincl. federated class-weighted variant (Task 2)'
    ax.set_title(title, fontsize=12, pad=12)

    fig.text(
        0.5, 0.005,
        'Headline match: gold solid bar (w = {0:1, 1:2}) reproduces FraudX AI §4.1 class-weight setup. '
        'Hatched gold bar is sensitivity at natural neg/pos ratio (~577); not a match.',
        ha='center', fontsize=8.5, color='dimgrey', style='italic',
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()