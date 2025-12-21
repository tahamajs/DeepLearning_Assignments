"""
Aggregate experiment result CSVs across multiple runs and produce summary CSV and LaTeX tables.

Usage:
    python aggregate_results.py --pattern "codes/**/results/run*_metrics.csv" --out_dir report/appendices/aggregates

The script expects CSV files with the same metric columns (e.g., accuracy, macro_f1, BLEU1...)
Each CSV should contain one row per run with a `run_id` column.

Outputs:
 - aggregate_summary.csv: mean/std/sem and 95% CI per metric
 - aggregate_table.tex: LaTeX table snippet ready to \input into the report
"""
import argparse
import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_csvs(pattern):
    paths = glob.glob(pattern, recursive=True)
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df['_source'] = p
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No CSVs found for pattern: {pattern}")
    return pd.concat(dfs, ignore_index=True)


def summarize(df, groupby_cols=None, metrics=None):
    if groupby_cols is None:
        groupby_cols = []
    if metrics is None:
        metrics = [c for c in df.columns if c not in ['run_id','_source'] + list(groupby_cols)]
    results = []
    groups = [()] if not groupby_cols else df.groupby(groupby_cols)
    if groupby_cols:
        iter_groups = groups
    else:
        iter_groups = [(None, df)]

    for key, grp in iter_groups:
        row = {}
        if groupby_cols:
            if isinstance(key, tuple):
                for kcol, kval in zip(groupby_cols, key):
                    row[kcol] = kval
            else:
                row[groupby_cols[0]] = key
        for m in metrics:
            vals = grp[m].dropna().values
            n = len(vals)
            mean = np.mean(vals) if n>0 else np.nan
            std = np.std(vals, ddof=1) if n>1 else np.nan
            sem = std/np.sqrt(n) if n>1 else np.nan
            # 95% CI (t-distribution)
            ci_low, ci_high = (np.nan, np.nan)
            if n>1:
                t = stats.t.ppf(0.975, n-1)
                ci_low = mean - t*sem
                ci_high = mean + t*sem
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_sem"] = sem
            row[f"{m}_ci_low"] = ci_low
            row[f"{m}_ci_high"] = ci_high
        results.append(row)
    return pd.DataFrame(results)


def write_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote CSV: {path}")


def format_latex_table(summary_df, metrics, caption, label, out_path):
    # Build columns for mean (± std)
    cols = ['Name']
    for m in metrics:
        cols.append(m)
    lines = []
    header = ' & '.join(['Method'] + [f"{m}" for m in metrics]) + " \\\n"
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append('\\begin{tabular}{l' + 'c'*len(metrics) + '}')
    lines.append('\\toprule')
    lines.append(header)
    lines.append('\\midrule')
    for _, r in summary_df.iterrows():
        name = r.get('method', r.get('_source', ''))
        cells = [name]
        for m in metrics:
            mean = r.get(f"{m}_mean", np.nan)
            std = r.get(f"{m}_std", np.nan)
            if np.isnan(mean):
                cells.append('--')
            else:
                cells.append(f"{mean:.3f} $\pm$ {std:.3f}")
        lines.append(' & '.join(cells) + ' \\\n')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote LaTeX table to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='codes/**/results/run*_metrics.csv')
    parser.add_argument('--out_dir', type=str, default='report/appendices/aggregates')
    parser.add_argument('--groupby', type=str, nargs='*', default=[])
    args = parser.parse_args()

    df = load_csvs(args.pattern)
    # if 'method' missing, derive method name from parent folder
    if 'method' not in df.columns:
        df['method'] = df['_source'].apply(lambda p: Path(p).parents[1].name if len(Path(p).parents)>1 else Path(p).parent.name)
    metrics = [c for c in df.columns if c not in ['run_id','_source','method'] + args.groupby]
    summary = summarize(df, groupby_cols=args.groupby, metrics=metrics)
    out_csv = os.path.join(args.out_dir, 'aggregate_summary.csv')
    write_csv(summary, out_csv)
    # add method column if absent
    if 'method' in df.columns:
        # aggregate by method
        summary_by_method = summary
        if 'method' not in summary_by_method.columns:
            summary_by_method['method'] = 'method'
        tex_out = os.path.join(args.out_dir, 'aggregate_table.tex')
        format_latex_table(summary_by_method, metrics, caption='Aggregated metrics across runs', label='tab:aggregates', out_path=tex_out)

if __name__ == '__main__':
    main()
