"""
Threshold Backtesting & Calibration Engine
===========================================
Generates synthetic transaction data and backtests transaction monitoring
rules across a range of thresholds to measure alert volume and sensitivity.

Outputs:
  - Console calibration report
  - calibration_output/threshold_sensitivity.png
  - calibration_output/alert_distribution.png
  - sample_transactions.xlsx  (synthetic data for inspection)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------------

TRANSACTION_TYPES = {
    'ATM Withdrawal':      ('debit',  500,    3_000),
    'Wire Transfer Out':   ('debit',  2_000,  50_000),
    'POS Purchase':        ('debit',  20,     2_000),
    'Online Transfer Out': ('debit',  500,    15_000),
    'Loan Payment':        ('debit',  1_000,  5_000),
    'Payroll Credit':      ('credit', 2_000,  8_000),
    'Wire Transfer In':    ('credit', 5_000,  100_000),
    'Check Deposit':       ('credit', 500,    20_000),
    'Online Transfer In':  ('credit', 200,    10_000),
}


def generate_transactions(n_accounts=30, days=180, seed=42):
    np.random.seed(seed)
    start_date = datetime(2024, 1, 1)
    account_numbers = [f'ACC{1000 + i}' for i in range(n_accounts)]
    suspicious_accounts = set(account_numbers[:5])
    records = []

    for acct in account_numbers:
        is_suspicious = acct in suspicious_accounts
        n_tx = np.random.randint(150, 400) if is_suspicious else np.random.randint(50, 200)

        for _ in range(n_tx):
            tx_date = start_date + timedelta(days=int(np.random.randint(0, days)))
            desc = np.random.choice(list(TRANSACTION_TYPES.keys()))
            tx_type, lo, hi = TRANSACTION_TYPES[desc]

            if is_suspicious and desc == 'Wire Transfer Out':
                # Structuring: keep amounts just below the $10k reporting threshold
                amount = round(np.random.uniform(8_500, 9_900), 2)
            elif is_suspicious:
                amount = round(np.random.uniform(lo, hi * 2), 2)
            else:
                amount = round(np.random.uniform(lo, hi), 2)

            records.append({
                'account_number': acct,
                'date':           tx_date.date(),
                'amount':         amount,
                'transaction_type': tx_type,
                'description':    desc,
            })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['account_number', 'date']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. RULE DEFINITIONS
# ---------------------------------------------------------------------------

RULES = {
    'large_single_debit': {
        'label':             'Large Single Debit',
        'description':       'Any single debit transaction >= threshold',
        'thresholds':        [5_000, 7_500, 10_000, 15_000, 20_000],
        'current_threshold': 10_000,
        'unit':              '$',
    },
    'high_monthly_debit': {
        'label':             'High Monthly Debit Aggregate',
        'description':       'Total debits in a calendar month >= threshold',
        'thresholds':        [10_000, 20_000, 30_000, 50_000, 75_000],
        'current_threshold': 30_000,
        'unit':              '$',
    },
    'high_velocity': {
        'label':             'High Transaction Velocity',
        'description':       'Transaction count in any rolling 7-day window >= threshold',
        'thresholds':        [10, 15, 20, 25, 30],
        'current_threshold': 20,
        'unit':              'transactions',
    },
    'structuring': {
        'label':             'Structuring Indicator',
        'description':       'N+ debit transactions between $8,000-$9,999 within any 72-hour window',
        'thresholds':        [2, 3, 4, 5],
        'current_threshold': 3,
        'unit':              'transactions',
    },
}


# ---------------------------------------------------------------------------
# 3. RULE FUNCTIONS  (each returns a set of flagged account numbers)
# ---------------------------------------------------------------------------

def apply_large_single_debit(df, threshold):
    debits = df[df['transaction_type'] == 'debit']
    return set(debits.loc[debits['amount'] >= threshold, 'account_number'].unique())


def apply_high_monthly_debit(df, threshold):
    debits = df[df['transaction_type'] == 'debit'].copy()
    debits['month'] = debits['date'].dt.to_period('M')
    monthly = debits.groupby(['account_number', 'month'])['amount'].sum()
    return set(monthly[monthly >= threshold].index.get_level_values('account_number').unique())


def apply_high_velocity(df, threshold):
    flagged = set()
    for acct, grp in df.groupby('account_number'):
        dates = grp['date'].sort_values().values
        for d in dates:
            window_end = d + np.timedelta64(7, 'D')
            if ((dates >= d) & (dates <= window_end)).sum() >= threshold:
                flagged.add(acct)
                break
    return flagged


def apply_structuring(df, threshold):
    flagged = set()
    near_threshold = df[
        (df['transaction_type'] == 'debit') &
        (df['amount'] >= 8_000) &
        (df['amount'] < 10_000)
    ].copy()
    for acct, grp in near_threshold.groupby('account_number'):
        dates = grp['date'].sort_values().values
        for d in dates:
            window_end = d + np.timedelta64(3, 'D')
            if ((dates >= d) & (dates <= window_end)).sum() >= threshold:
                flagged.add(acct)
                break
    return flagged


RULE_FUNCTIONS = {
    'large_single_debit': apply_large_single_debit,
    'high_monthly_debit': apply_high_monthly_debit,
    'high_velocity':      apply_high_velocity,
    'structuring':        apply_structuring,
}


# ---------------------------------------------------------------------------
# 4. CALIBRATION ENGINE
# ---------------------------------------------------------------------------

def backtest_rule(df, rule_name, rule_config):
    fn = RULE_FUNCTIONS[rule_name]
    n_accounts = df['account_number'].nunique()
    results = []
    for threshold in rule_config['thresholds']:
        flagged = fn(df, threshold)
        results.append({
            'threshold':        threshold,
            'alerts':           len(flagged),
            'alert_rate':       len(flagged) / n_accounts * 100,
            'flagged_accounts': flagged,
        })
    return results


def run_calibration(df):
    return {name: backtest_rule(df, name, cfg) for name, cfg in RULES.items()}


# ---------------------------------------------------------------------------
# 5. CONSOLE REPORT
# ---------------------------------------------------------------------------

def fmt_threshold(value, unit):
    return f"${value:,}" if unit == '$' else f"{value:,} {unit}"


def print_calibration_report(df, all_results):
    n_accounts = df['account_number'].nunique()
    sep = '=' * 66

    print(f"\n{sep}")
    print("  THRESHOLD CALIBRATION REPORT")
    print(f"  Accounts : {n_accounts}   |   Transactions : {len(df):,}")
    print(f"  Period   : {df['date'].min().date()}  to  {df['date'].max().date()}")
    print(sep)

    for rule_name, results in all_results.items():
        cfg     = RULES[rule_name]
        current = cfg['current_threshold']
        current_alerts = next(r['alerts'] for r in results if r['threshold'] == current)

        print(f"\n  Rule  : {cfg['label']}")
        print(f"  Desc  : {cfg['description']}")
        print(f"  Current threshold : {fmt_threshold(current, cfg['unit'])}\n")
        print(f"  {'Threshold':<22} {'Alerts':>8}  {'Alert Rate':>11}  {'vs Current':>11}")
        print(f"  {'─' * 58}")

        for r in results:
            is_current = r['threshold'] == current
            delta      = r['alerts'] - current_alerts
            delta_str  = '─' if is_current else (f"+{delta}" if delta > 0 else str(delta))
            marker     = '  ◄ current' if is_current else ''
            print(
                f"  {fmt_threshold(r['threshold'], cfg['unit']):<22}"
                f"  {r['alerts']:>6}"
                f"  {r['alert_rate']:>9.1f}%"
                f"  {delta_str:>11}"
                f"{marker}"
            )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# 6. PLOTS
# ---------------------------------------------------------------------------

def plot_sensitivity_curves(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.01)

    for ax, (rule_name, results) in zip(axes.flatten(), all_results.items()):
        cfg        = RULES[rule_name]
        thresholds = [r['threshold']  for r in results]
        alerts     = [r['alerts']     for r in results]
        rates      = [r['alert_rate'] for r in results]
        current    = cfg['current_threshold']

        ax.plot(thresholds, alerts, 'b-o', lw=2, ms=6, label='Alert Count')
        ax2 = ax.twinx()
        ax2.plot(thresholds, rates, 'g--s', lw=1.5, ms=5, alpha=0.8, label='Alert Rate %')
        ax2.set_ylabel('Alert Rate (%)', color='green', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='green')

        ax.axvline(x=current, color='red', lw=2, ls=':', label=f'Current ({fmt_threshold(current, cfg["unit"])})')
        ax.set_title(cfg['label'], fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Threshold ({cfg["unit"]})', fontsize=9)
        ax.set_ylabel('Alerts', color='blue', fontsize=9)
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_xticks(thresholds)
        ax.set_xticklabels(
            [fmt_threshold(t, cfg['unit']) for t in thresholds],
            rotation=15, fontsize=8
        )
        ax.grid(True, alpha=0.3)

        lines  = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='upper right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'threshold_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Chart saved → {path}")
    plt.close()


def plot_alert_distribution(df, all_results):
    # Count how many rules each account triggers at current thresholds
    all_accounts = sorted(df['account_number'].unique())
    rule_hits    = {}

    for rule_name, results in all_results.items():
        current = RULES[rule_name]['current_threshold']
        flagged = next(r['flagged_accounts'] for r in results if r['threshold'] == current)
        rule_hits[RULES[rule_name]['label']] = flagged

    hit_counts = {acct: sum(acct in f for f in rule_hits.values()) for acct in all_accounts}
    dist        = pd.Series(hit_counts).value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Alert Distribution at Current Thresholds', fontsize=13, fontweight='bold')

    colours = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    bars = ax1.bar(dist.index, dist.values, color=colours[:len(dist)])
    ax1.set_xlabel('Number of Rules Triggered per Account', fontsize=10)
    ax1.set_ylabel('Number of Accounts', fontsize=10)
    ax1.set_title('Rule Hit Distribution', fontsize=11, fontweight='bold')
    ax1.set_xticks(dist.index)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, dist.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(val), ha='center', fontsize=10, fontweight='bold')

    rule_labels = list(rule_hits.keys())
    matrix = np.array([[1 if acct in rule_hits[r] else 0
                        for r in rule_labels]
                       for acct in all_accounts])

    im = ax2.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(rule_labels)))
    ax2.set_xticklabels([r.replace(' ', '\n') for r in rule_labels], fontsize=8)
    ax2.set_yticks(range(len(all_accounts)))
    ax2.set_yticklabels(all_accounts, fontsize=7)
    ax2.set_title('Account × Rule Flag Matrix\n(red = flagged)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'alert_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Chart saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    print("\nGenerating synthetic transaction data...")
    df = generate_transactions(n_accounts=30, days=180)
    print(f"  {len(df):,} transactions  |  {df['account_number'].nunique()} accounts")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_transactions.xlsx')
    df.to_excel(data_path, index=False)
    print(f"  Sample data saved → {data_path}")

    print("\nRunning calibration across all rules and thresholds...")
    all_results = run_calibration(df)

    print_calibration_report(df, all_results)

    print("Saving charts...")
    plot_sensitivity_curves(all_results)
    plot_alert_distribution(df, all_results)

    print("\nDone.\n")


if __name__ == '__main__':
    main()
