#!/usr/bin/env python3
"""Create recommended feature list by combining Boruta + SHAP and pruning correlated groups.

Algorithm:
- Read `scripts/output/boruta_selected.txt` as conservative set.
- Read `scripts/output/shap_ranking.txt` for SHAP mean-abs importances.
- Read `scripts/output/correlated_pairs.csv` for very-high correlation pairs (abs_corr > 0.95).
- Build correlated groups (connected components). For each group that intersects the Boruta set,
  keep the feature with highest SHAP mean_abs_shap (or Boruta order as fallback).
- Produce final list sorted by SHAP importance and cap at `--max-features` (default 30).
- Write `scripts/output/recommended_features.txt` with one feature per line.

This script is safe to run repeatedly and tolerant to missing optional files.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import networkx as nx

OUT = Path(__file__).resolve().parents[0] / 'output'
OUT.mkdir(parents=True, exist_ok=True)

BORUTA = OUT / 'boruta_selected.txt'
SHAP = OUT / 'shap_ranking.txt'
CORR = OUT / 'correlated_pairs.csv'
OUT_FILE = OUT / 'recommended_features.txt'


def read_boruta(path: Path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def read_shap(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=['feature', 'mean_abs_shap'])
    df = pd.read_csv(path)
    # tolerate different column names
    if 'feature' not in df.columns:
        # try first column as feature
        df = df.rename(columns={df.columns[0]: 'feature', df.columns[1]: 'mean_abs_shap'})
    return df[['feature', 'mean_abs_shap']].drop_duplicates()


def read_corr(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=['feature_a', 'feature_b', 'abs_corr'])
    df = pd.read_csv(path)
    # ensure columns
    cols = df.columns.tolist()
    if len(cols) >= 3:
        # try to normalize
        df = df.rename(columns={cols[0]: 'feature_a', cols[1]: 'feature_b', cols[2]: 'abs_corr'})
    return df[['feature_a', 'feature_b', 'abs_corr']]


def build_groups_from_pairs(pairs_df, threshold=0.95):
    # Build graph of edges where abs_corr > threshold
    G = nx.Graph()
    for _, row in pairs_df.iterrows():
        try:
            a = row['feature_a']
            b = row['feature_b']
            corr = float(row['abs_corr'])
        except Exception:
            continue
        if pd.isna(a) or pd.isna(b):
            continue
        if corr > threshold:
            G.add_edge(a, b, weight=corr)
    # Add isolated nodes? Not necessary
    groups = list(nx.connected_components(G))
    return groups


def choose_from_group(group, shap_df, boruta_list):
    # group: set of features
    # prefer features present in boruta_list; else pick highest SHAP overall
    candidates = list(group)
    shap_map = dict(zip(shap_df['feature'], shap_df['mean_abs_shap']))
    # restrict to boruta if possible
    boruta_in_group = [f for f in candidates if f in boruta_list]
    if boruta_in_group:
        # pick highest SHAP among these
        ranked = sorted(boruta_in_group, key=lambda f: shap_map.get(f, 0.0), reverse=True)
        return ranked[0]
    # otherwise, pick highest SHAP in full group
    ranked_all = sorted(candidates, key=lambda f: shap_map.get(f, 0.0), reverse=True)
    return ranked_all[0]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-features', type=int, default=30, help='Maximum number of recommended features')
    parser.add_argument('--corr-threshold', type=float, default=0.95, help='Correlation threshold to consider features redundant')
    args = parser.parse_args(argv)

    boruta_list = read_boruta(BORUTA)
    shap_df = read_shap(SHAP)
    corr_df = read_corr(CORR)

    if not boruta_list:
        print('No Boruta file found or empty; falling back to top SHAP features')
    else:
        print(f'Read {len(boruta_list)} Boruta-selected features')

    if not shap_df.empty:
        print(f'Read SHAP ranking with {len(shap_df)} features')
    else:
        print('No SHAP ranking available')

    if not corr_df.empty:
        print(f'Read correlation pairs: {len(corr_df)}')
    else:
        print('No correlation pairs available')

    # If Boruta present, start from that set; otherwise use top SHAP features
    if boruta_list:
        base_set = [f for f in boruta_list]
    elif not shap_df.empty:
        base_set = shap_df['feature'].tolist()
    else:
        print('No input features available to recommend. Exiting.')
        return 2

    # Build correlated groups
    groups = build_groups_from_pairs(corr_df, threshold=args.corr_threshold)
    # Map each feature to its group id (for features not in any group, group id is None)
    feature_to_group = {}
    for gid, g in enumerate(groups):
        for f in g:
            feature_to_group[f] = gid

    selected = set()
    used_groups = set()
    shap_map = dict(zip(shap_df['feature'], shap_df['mean_abs_shap']))

    # First pass: for each group that contains any base_set features, pick representative
    for gid, group in enumerate(groups):
        group_features = set(group)
        inter = group_features.intersection(set(base_set))
        if not inter:
            continue
        pick = choose_from_group(inter if inter else group_features, shap_df, base_set)
        selected.add(pick)
        used_groups.add(gid)

    # Add any base_set features that were not part of a high-corr group
    for f in base_set:
        if f in feature_to_group:
            continue
        selected.add(f)

    # If selected smaller than cap, fill up from SHAP-ranked features excluding ones already selected
    if len(selected) < args.max_features and not shap_df.empty:
        for f in shap_df['feature'].tolist():
            if f in selected:
                continue
            selected.add(f)
            if len(selected) >= args.max_features:
                break

    # Final ordering: by SHAP importance (descending), fallback to Boruta order
    def score(f):
        if f in shap_map:
            return shap_map[f]
        try:
            return float('-inf')
        except Exception:
            return float('-inf')

    final = sorted(list(selected), key=lambda x: score(x), reverse=True)
    final = final[: args.max_features]

    with OUT_FILE.open('w', encoding='utf-8') as f:
        for feat in final:
            f.write(feat + '\n')

    print(f'Wrote {len(final)} recommended features to {OUT_FILE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
