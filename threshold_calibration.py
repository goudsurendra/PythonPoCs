"""
Threshold Backtesting & Calibration Engine
===========================================
Reads transactions.csv and thresholds.csv from the same directory,
backtests each monitoring rule across a range of thresholds, and
reports how alert volume responds to threshold changes.

Outputs:
  - Console calibration report
  - calibration_output/threshold_sensitivity.png
  - calibration_output/alert_distribution.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'calibration_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds are tested at these multiples of the current configured value
THRESHOLD_MULTIPLIERS = [0.50, 0.75, 1.00, 1.25, 1.50]


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data():
    df_tx = pd.read_csv(os.path.join(BASE_DIR, 'transactions.csv'))
    df_tx['amount'] = pd.to_numeric(df_tx['amount'], errors='coerce')

    df_rules = pd.read_csv(os.path.join(BASE_DIR, 'thresholds.csv'))
    df_rules['threshold'] = pd.to_numeric(df_rules['threshold'], errors='coerce')

    return df_tx, df_rules


# ---------------------------------------------------------------------------
# 2. RULE LOGIC REGISTRY
#
# Each entry maps a rule-name (from thresholds.csv) to a callable:
#   fn(df_tx, threshold) -> set of flagged account_key values
#
# Add new rules here as the thresholds.csv grows.
# ---------------------------------------------------------------------------

def _rule_large_single_transaction(df_tx, threshold):
    """Flag accounts with any single transaction >= threshold (all types)."""
    flagged = df_tx.loc[df_tx['amount'] >= threshold, 'account_key'].unique()
    return set(flagged)


def _rule_excessive_card_payments(df_tx, threshold):
    """Flag accounts whose total card debits >= threshold."""
    card_debits = df_tx[
        (df_tx['transaction_type'].str.lower() == 'card') &
        (df_tx['debit_or_credit'].str.lower() == 'debit')
    ]
    agg = card_debits.groupby('account_key')['amount'].sum()
    return set(agg[agg >= threshold].index)


RULE_REGISTRY = {
    'Monitoring of big payments': _rule_large_single_transaction,
    'excessive card payments':    _rule_excessive_card_payments,
}


# ---------------------------------------------------------------------------
# 3. CALIBRATION ENGINE
# ---------------------------------------------------------------------------

def _threshold_range(current):
    return [round(current * m) for m in THRESHOLD_MULTIPLIERS]


def _backtest_rule(df_tx, rule_name, current_threshold):
    if rule_name not in RULE_REGISTRY:
        raise ValueError(
            f"No logic registered for rule '{rule_name}'. "
            f"Add it to RULE_REGISTRY in threshold_calibration.py."
        )
    fn = RULE_REGISTRY[rule_name]
    n_accounts = df_tx['account_key'].nunique()
    results = []
    for threshold in _threshold_range(current_threshold):
        flagged = fn(df_tx, threshold)
        results.append({
            'threshold':        threshold,
            'alerts':           len(flagged),
            'alert_rate':       len(flagged) / n_accounts * 100 if n_accounts else 0.0,
            'flagged_accounts': flagged,
        })
    return results


def run_calibration(df_tx, df_rules):
    all_results = {}
    for _, row in df_rules.iterrows():
        rule_name = row['rule-name']
        all_results[rule_name] = {
            'segment': row['customer segment'],
            'current': row['threshold'],
            'results': _backtest_rule(df_tx, rule_name, row['threshold']),
        }
    return all_results


# ---------------------------------------------------------------------------
# 4. CONSOLE REPORT
# ---------------------------------------------------------------------------

def print_report(df_tx, all_results):
    sep = '=' * 66
    print(f"\n{sep}")
    print("  THRESHOLD CALIBRATION REPORT")
    print(f"  Accounts     : {df_tx['account_key'].nunique()}")
    print(f"  Transactions : {len(df_tx):,}")
    print(sep)

    for rule_name, data in all_results.items():
        current        = data['current']
        segment        = data['segment']
        results        = data['results']
        current_alerts = next(r['alerts'] for r in results if r['threshold'] == current)

        print(f"\n  Rule     : {rule_name}")
        print(f"  Segment  : {segment}")
        print(f"  Current threshold : ${current:,.0f}\n")
        print(f"  {'Threshold':<18} {'Alerts':>8}  {'Alert Rate':>11}  {'vs Current':>11}")
        print(f"  {'─' * 54}")

        for r in results:
            is_current = r['threshold'] == current
            delta      = r['alerts'] - current_alerts
            delta_str  = '─' if is_current else (f"+{delta}" if delta > 0 else str(delta))
            marker     = '  ◄ current' if is_current else ''
            print(
                f"  ${r['threshold']:>14,.0f}"
                f"  {r['alerts']:>6}"
                f"  {r['alert_rate']:>9.1f}%"
                f"  {delta_str:>11}"
                f"{marker}"
            )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------------------

def plot_sensitivity(all_results):
    n_rules = len(all_results)
    fig, axes = plt.subplots(1, n_rules, figsize=(7 * n_rules, 5))
    if n_rules == 1:
        axes = [axes]
    fig.suptitle('Threshold Sensitivity Analysis', fontsize=13, fontweight='bold')

    for ax, (rule_name, data) in zip(axes, all_results.items()):
        current    = data['current']
        results    = data['results']
        thresholds = [r['threshold']  for r in results]
        alerts     = [r['alerts']     for r in results]
        rates      = [r['alert_rate'] for r in results]

        ax.plot(thresholds, alerts, 'b-o', lw=2, ms=6, label='Alert Count')
        ax2 = ax.twinx()
        ax2.plot(thresholds, rates, 'g--s', lw=1.5, ms=5, alpha=0.8, label='Alert Rate %')
        ax2.set_ylabel('Alert Rate (%)', color='green', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='green')

        ax.axvline(x=current, color='red', lw=2, ls=':', label=f'Current (${current:,.0f})')
        ax.set_title(rule_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Threshold ($)', fontsize=9)
        ax.set_ylabel('Alert Count', color='blue', fontsize=9)
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"${t:,.0f}" for t in thresholds], rotation=15, fontsize=8)
        ax.grid(True, alpha=0.3)

        lines  = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='upper right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'threshold_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Sensitivity chart saved  → {path}")
    plt.close()


def plot_alert_distribution(df_tx, all_results):
    all_accounts = sorted(df_tx['account_key'].unique())
    rule_labels  = list(all_results.keys())

    # Flagged sets at current thresholds
    rule_hits = {
        name: next(
            r['flagged_accounts']
            for r in data['results']
            if r['threshold'] == data['current']
        )
        for name, data in all_results.items()
    }

    hit_counts = {acct: sum(acct in rule_hits[r] for r in rule_labels) for acct in all_accounts}
    dist       = pd.Series(hit_counts).value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Alert Distribution at Current Thresholds', fontsize=12, fontweight='bold')

    colours = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax1.bar(dist.index, dist.values, color=colours[:len(dist)])
    ax1.set_xlabel('Rules Triggered per Account', fontsize=10)
    ax1.set_ylabel('Number of Accounts', fontsize=10)
    ax1.set_title('Rule Hit Distribution', fontsize=11, fontweight='bold')
    ax1.set_xticks(dist.index)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, dist.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(val), ha='center', fontsize=10, fontweight='bold')

    matrix = np.array([
        [1 if acct in rule_hits[r] else 0 for r in rule_labels]
        for acct in all_accounts
    ])
    im = ax2.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(rule_labels)))
    ax2.set_xticklabels([r.replace(' ', '\n') for r in rule_labels], fontsize=9)
    ax2.set_yticks(range(len(all_accounts)))
    ax2.set_yticklabels(all_accounts, fontsize=8)
    ax2.set_title('Account × Rule Flag Matrix\n(red = flagged)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'alert_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Alert distribution chart → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    print("\nLoading data...")
    df_tx, df_rules = load_data()
    print(f"  {len(df_tx):,} transactions loaded  |  {df_tx['account_key'].nunique()} accounts")
    print(f"  {len(df_rules)} rules loaded from thresholds.csv")

    print("\nRunning calibration...")
    all_results = run_calibration(df_tx, df_rules)

    print_report(df_tx, all_results)

    print("Saving charts...")
    plot_sensitivity(all_results)
    plot_alert_distribution(df_tx, all_results)

    print("\nDone.\n")


if __name__ == '__main__':
    main()
