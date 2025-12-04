#!/usr/bin/env python3
"""Export feature-selection artifacts to CSV and HTML summary files.

Reads from `scripts/output/shap_ranking.txt`, `scripts/output/boruta_selected.txt`,
and `scripts/output/correlated_pairs.csv` and writes:
- `scripts/output/feature_selection_summary.csv`
- `scripts/output/feature_selection_report.html`

Usage: python export_feature_selection.py [--csv] [--html]
"""
import argparse
import os
import pandas as pd
from datetime import datetime, timezone


OUT_DIR = os.path.join('scripts', 'output')


def read_shap(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['feature', 'mean_abs_shap'])
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(path)
    # normalize column names
    cols = [c.lower() for c in df.columns]
    if 'feature' in cols and 'mean_abs_shap' in cols:
        df = df.rename(columns={df.columns[cols.index('feature')]: 'feature',
                                df.columns[cols.index('mean_abs_shap')]: 'mean_abs_shap'})
    else:
        df = df.iloc[:, :2]
        df.columns = ['feature', 'mean_abs_shap']
    df['mean_abs_shap'] = pd.to_numeric(df['mean_abs_shap'], errors='coerce')
    return df


def read_boruta(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return set(lines)


def read_correlated(path):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)
    # Heuristic: assume first two columns are feature pairs
    if df.shape[1] >= 2:
        a = df.columns[0]
        b = df.columns[1]
        mapping = {}
        for _, row in df.iterrows():
            f1 = str(row[a])
            f2 = str(row[b])
            mapping.setdefault(f1, set()).add(f2)
            mapping.setdefault(f2, set()).add(f1)
        return mapping
    return {}


def generate_summary(shap_df, boruta_set, corr_map):
    features = list(pd.unique(shap_df['feature'].fillna('')))
    # include boruta-only features
    for f in boruta_set:
        if f not in features:
            features.append(f)

    rows = []
    for feat in features:
        shap_val = shap_df.loc[shap_df['feature'] == feat, 'mean_abs_shap']
        shap_val = float(shap_val.iloc[0]) if len(shap_val) > 0 else float('nan')
        bor = feat in boruta_set
        corr_with = sorted(list(corr_map.get(feat, [])))
        rows.append({
            'Feature': feat,
            'mean_abs_shap': shap_val,
            'boruta_selected': bor,
            'correlated_with_count': len(corr_with),
            'correlated_with': ';'.join(corr_with)
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(by='mean_abs_shap', ascending=False, na_position='last')
    return out


def write_csv(df, out_path):
    df.to_csv(out_path, index=False)


def write_html(df, out_path):
    # basic HTML template
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
    html = [
        '<!doctype html>',
        '<html lang="en">',
        '<head><meta charset="utf-8"><title>Feature Selection Report</title></head>',
        '<body>',
        f'<h1>Feature Selection Report</h1>',
        f'<p>Generated: {now} (UTC)</p>',
        '<h2>Top SHAP Features</h2>',
        df[['Feature', 'mean_abs_shap']].head(50).to_html(index=False, classes='table table-striped'),
        '<h2>Boruta Selection + Correlation Summary</h2>',
        df[['Feature', 'boruta_selected', 'correlated_with_count', 'correlated_with']].head(200).to_html(index=False),
        '<hr>',
        '<p>Files:</p>',
        '<ul>',
        f'<li><a href="shap_ranking.txt">SHAP ranking (raw)</a></li>',
        f'<li><a href="boruta_selected.txt">Boruta selected (raw)</a></li>',
        f'<li><a href="correlated_pairs.csv">Correlated pairs (raw)</a></li>',
        '</ul>',
        '</body></html>'
    ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-html', action='store_true', help='Do not write HTML report')
    parser.add_argument('--no-csv', action='store_true', help='Do not write CSV summary')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    shp_path = os.path.join(OUT_DIR, 'shap_ranking.txt')
    boruta_path = os.path.join(OUT_DIR, 'boruta_selected.txt')
    corr_path = os.path.join(OUT_DIR, 'correlated_pairs.csv')

    shap_df = read_shap(shp_path)
    boruta_set = read_boruta(boruta_path)
    corr_map = read_correlated(corr_path)

    summary = generate_summary(shap_df, boruta_set, corr_map)

    csv_out = os.path.join(OUT_DIR, 'feature_selection_summary.csv')
    html_out = os.path.join(OUT_DIR, 'feature_selection_report.html')

    if not args.no_csv:
        write_csv(summary, csv_out)
        print('Wrote CSV summary to', csv_out)
    if not args.no_html:
        write_html(summary, html_out)
        print('Wrote HTML report to', html_out)


if __name__ == '__main__':
    main()
