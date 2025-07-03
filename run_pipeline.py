#!/usr/bin/env python3
import argparse
import subprocess
import glob
import os
import pandas as pd

def find_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return sorted(files)[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full FX pipeline with optional fixed features')
    parser.add_argument('--symbol', required=True, help='Currency pair symbol, e.g. USDJPY')
    parser.add_argument('--timeframe', required=True, help='Timeframe, e.g. M5')
    parser.add_argument('--bars', type=int, default=100000, help='Number of bars')
    parser.add_argument('--tp', type=int, default=140, help='Take-profit pips')
    parser.add_argument('--sl', type=int, default=10, help='Stop-loss pips')
    parser.add_argument('--sample_frac', type=float, default=1.0, help='Sampling fraction for feature selection')
    parser.add_argument('--fixed_features', nargs='+', help='List of features to use; if set, skip auto-selection')
    parser.add_argument('--n_workers', type=int, help='Parallel workers for SHAP')
    parser.add_argument('--window_size', type=int, help='Chunk size for SHAP')
    args = parser.parse_args()

    base = '/content/drive/MyDrive/project_10k_to_1m_data'
    raw_dir = os.path.join(base, 'raw', args.symbol, args.timeframe)
    proc_dir = os.path.join(base, 'processed', args.symbol, args.timeframe)
    models_dir = os.path.join(base, 'processed', 'models')
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1) Raw CSV
    raw_pattern = os.path.join(raw_dir, f"{args.symbol}_{args.timeframe}_{args.bars}_*.csv")
    raw_csv = find_latest_file(raw_pattern)
    print(f"[INFO] Using raw CSV: {raw_csv}")

    # 2) Feature generation
    feat_csv = os.path.join(proc_dir, f"feat_{args.symbol}_{args.timeframe}_{args.bars}.csv")
    cmd = [
        'python3', 'src/data/feature_gen.py',
        '--csv', raw_csv,
        '--out', feat_csv
    ]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 3) Label generation
    labeled_csv = os.path.join(proc_dir, f"labeled_{args.symbol}_{args.timeframe}_{args.bars}.csv")
    cmd = [
        'python3', 'src/data/label_gen.py',
        '--file', feat_csv,
        '--tp', str(args.tp),
        '--sl', str(args.sl),
        '--exclude_before_release', 'True',
        '--release_exclude_window_mins', '30',
        '--out', labeled_csv
    ]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 4) Feature selection or fixed
    if args.fixed_features:
        selected = args.fixed_features
        selfeat_csv = labeled_csv  # use labeled CSV for model training, selecting columns later
        print(f"[INFO] Using fixed features: {selected}")
    else:
        selfeat_csv = os.path.join(proc_dir, f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}.csv")
        cmd = [
            'python3', 'src/data/auto_feature_selection_v7_fastfallback.py',
            '--csv', labeled_csv,
            '--out_dir', proc_dir,
            '--out', selfeat_csv,
            '--sample_frac', str(args.sample_frac)
        ]
        if args.n_workers:
            cmd += ['--n_workers', str(args.n_workers)]
        if args.window_size:
            cmd += ['--window_size', str(args.window_size)]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        # parse selected features from header
        df_sel = pd.read_csv(selfeat_csv, nrows=0)
        selected = [c for c in df_sel.columns if c not in ['time','label','future_return']]
        print(f"[INFO] Auto-selected features: {selected}")

    # 5) Baseline training
    baseline_cmd = [
        'python3', 'src/models/train_baseline.py',
        '--csv', selfeat_csv,
        '--features'
    ] + selected
    print(f"[RUN] {' '.join(baseline_cmd)}")
    subprocess.run(baseline_cmd, check=True)

    # 6) XGBoost training
    train_cmd = [
        'python3', 'src/models/train_model.py',
        '--symbol', args.symbol,
        '--timeframe', args.timeframe,
        '--bars', str(args.bars),
        '--features'
    ] + selected
    print(f"[RUN] {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    # 7) Backtest
    backtest_cmd = [
        'python3', 'src/models/backtest.py',
        '--symbol', args.symbol,
        '--timeframe', args.timeframe,
        '--bars', str(args.bars),
        '--features'
    ] + selected
    print(f"[RUN] {' '.join(backtest_cmd)}")
    subprocess.run(backtest_cmd, check=True)

    print('\n--- All done! ---')
