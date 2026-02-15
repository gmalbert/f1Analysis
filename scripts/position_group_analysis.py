"""
Position group analysis utilities

Produces:
 - scripts/output/mae_by_season.csv
 - scripts/output/mae_trends.png
 - scripts/output/ci_by_driver_track.csv
 - scripts/output/heatmap_driver_by_circuit.png
 - scripts/output/heatmap_constructor_by_circuit.png
 - scripts/output/position_group_analysis_report.html

The script is defensive about column names and will try several common names
found in the workspace prediction files.
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import json
import json_helpers


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data_files'
OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_prediction_files():
    return sorted(DATA_DIR.glob('predictions_*.csv'))


def load_predictions(paths):
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(p)
        df['_source_file'] = p.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def guess_columns(df):
    preds = ['PredictedFinalPosition','predictedPosition','predicted_position','prediction','pred','y_pred','predicted']
    actuals = ['Rank','resultsFinalPositionNumber','final_position','actual_position','resultPosition','position','fin_pos']
    driver_cols = ['resultsDriverName','DriverId','driverId','driver_id','driver','Driver','driverName','ResultsDriver']
    constructor_cols = ['constructorName','constructorId','ConstructorId','constructor_id','constructor','Constructor']
    circuit_cols = ['grandPrixName','circuitName','shortCircuitName','grandPrix','circuit']

    def find(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    return {
        'pred': find(preds),
        'actual': find(actuals),
        'driver': find(driver_cols),
        'constructor': find(constructor_cols),
        'circuit': find(circuit_cols),
    }


def extract_year_from_filename(name):
    m = re.search(r'(19|20)\d{2}', name)
    if m:
        return int(m.group(0))
    return None


def compute_mae_by_season(df):
    if 'season' in df.columns:
        df['season'] = df['season'].astype(int)
    else:
        df['_year'] = df['_source_file'].map(lambda x: extract_year_from_filename(x) or 0)
        df['season'] = df['_year']

    # Compute MAE per season using .groupby(...)['residual'].agg(...) to avoid
    # the FutureWarning caused by apply operating on grouping columns.
    mae = df.groupby('season')['residual'].agg(lambda x: np.mean(np.abs(x)) if len(x) > 0 else np.nan)
    mae = mae.reset_index()
    mae.columns = ['season', 'mae']
    return mae.sort_values('season')


def make_mae_plot(mae_df, outpath):
    plt.figure(figsize=(8,4))
    sns.lineplot(data=mae_df, x='season', y='mae', marker='o')
    plt.title('MAE by Season')
    plt.ylabel('MAE (positions)')
    plt.xlabel('Season')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def compute_confidence_intervals(df, driver_col, circuit_col):
    group_cols = []
    if driver_col:
        group_cols.append(driver_col)
    if circuit_col:
        group_cols.append(circuit_col)
    if not group_cols:
        # fallback to overall
        percentiles = np.percentile(df['residual'].dropna().values, [2.5,97.5])
        return pd.DataFrame([{'group':'overall','p2_5':percentiles[0],'p97_5':percentiles[1]}])

    # Use apply so we can return a pair of percentiles per group (array-like)
    def _pct(x):
        arr = x.dropna().values
        if len(arr) >= 5:
            return np.percentile(arr, [2.5, 97.5])
        return np.array([np.nan, np.nan])

    ag = df.groupby(group_cols)['residual'].apply(_pct)
    rows = []
    for idx, val in ag.items():
        low, high = (val if isinstance(val, (list,tuple,np.ndarray)) else [np.nan,np.nan])
        row = {}
        if isinstance(idx, tuple):
            for k,v in zip(group_cols, idx):
                row[k] = v
        else:
            row[group_cols[0]] = idx
        row.update({'p2_5': low, 'p97_5': high})
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap_for_pivot(pivot, outpath, title):
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot, cmap='viridis', linewidths=0.3, linecolor='white', cbar_kws={'label':'avg abs error'})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    files = find_prediction_files()
    if not files:
        print('No prediction files found in', DATA_DIR)
        return

    df = load_predictions(files)
    if df.empty:
        print('No data loaded from prediction files')
        return

    cols = guess_columns(df)
    pred_col = cols['pred']
    actual_col = cols['actual']
    driver_col = cols['driver']
    constructor_col = cols['constructor']
    circuit_col = cols['circuit']

    if pred_col is None or actual_col is None:
        print('Could not locate predicted or actual columns. Found:', cols)
        return

    # coerce numeric
    df['predicted'] = pd.to_numeric(df[pred_col], errors='coerce')
    df['actual'] = pd.to_numeric(df[actual_col], errors='coerce')
    df = df.dropna(subset=['predicted','actual'])
    df['residual'] = df['predicted'] - df['actual']
    df['abs_error'] = df['residual'].abs()

    # MAE by season
    mae = compute_mae_by_season(df)
    mae_out = OUT_DIR / 'mae_by_season.csv'
    mae.to_csv(mae_out, index=False)
    make_mae_plot(mae, OUT_DIR / 'mae_trends.png')
    print('Wrote', mae_out)
    # Flag whether we have more than one season — used to decide whether
    # to include the MAE-by-season section in the HTML report (hide when
    # there's only a single season of data).
    has_multiple_seasons = int(mae['season'].nunique()) > 1

    # Confidence intervals by driver+track
    ci = compute_confidence_intervals(df, driver_col, circuit_col)
    ci_out = OUT_DIR / 'confid_int_by_driver_track.csv'
    ci.to_csv(ci_out, index=False)
    print('Wrote', ci_out)

    # Also produce driver-only and constructor-only confidence intervals
    if driver_col:
        ci_driver = compute_confidence_intervals(df, driver_col, None)
        ci_driver_out = OUT_DIR / 'confid_int_by_driver.csv'
        ci_driver.to_csv(ci_driver_out, index=False)
        print('Wrote', ci_driver_out)

    if constructor_col:
        ci_constructor = compute_confidence_intervals(df, None, constructor_col)
        ci_constructor_out = OUT_DIR / 'confid_int_by_constructor.csv'
        ci_constructor.to_csv(ci_constructor_out, index=False)
        print('Wrote', ci_constructor_out)

    # Heatmaps: driver x circuit (avg abs error)
    # choose sensible driver and circuit labels
    driver_label = driver_col or 'Driver'
    circuit_label = circuit_col or 'Circuit'
    # For readability, pick top drivers by race count
    if driver_col and circuit_col:
        counts = df[driver_col].value_counts()
        top_drivers = counts.head(20).index.tolist()
        small = df[df[driver_col].isin(top_drivers)]
        pivot = small.pivot_table(index=driver_col, columns=circuit_col, values='abs_error', aggfunc='mean')
        heatmap_for_pivot(pivot.fillna(np.nan), OUT_DIR / 'heatmap_driver_by_circuit.png', 'Avg abs error: Driver x Circuit')
        print('Wrote heatmap_driver_by_circuit.png')

    # Constructor heatmap
    if constructor_col and circuit_col:
        counts_c = df[constructor_col].value_counts()
        top_cons = counts_c.head(20).index.tolist()
        smallc = df[df[constructor_col].isin(top_cons)]
        pivotc = smallc.pivot_table(index=constructor_col, columns=circuit_col, values='abs_error', aggfunc='mean')
        heatmap_for_pivot(pivotc.fillna(np.nan), OUT_DIR / 'heatmap_constructor_by_circuit.png', 'Avg abs error: Constructor x Circuit')
        print('Wrote heatmap_constructor_by_circuit.png')

    # Simple HTML report
    report_path = OUT_DIR / 'position_group_analysis_report.html'
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write('<html><head><meta charset="utf-8"><title>Position Group Analysis Report</title></head><body>')
        fh.write('<h1>Position Group Analysis</h1>')
        # Use local time (with system timezone) and format as a readable timestamp
        fh.write(f'<p>Generated: {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")}</p>')
        # Only include the MAE by season section when multiple seasons are present
        if has_multiple_seasons:
            fh.write('<h2>MAE by Season</h2>')
            fh.write('<img src="mae_trends.png" style="max-width:100%;height:auto;"/>')
        fh.write('<h2>Confidence intervals (driver+circuit)</h2>')
        # Explanatory notes for the heatmaps to help interpret values
        fh.write('<h2>Notes: How to read the heatmaps</h2>')
        fh.write('<p>The heatmaps show the <strong>average absolute error</strong> (in finishing positions) aggregated for the intersection of the row (driver or constructor) and the column (circuit).</p>')
        fh.write('<ul>')
        fh.write('<li><strong>Color scale</strong>: darker/warmer colors indicate larger average absolute error.</li>')
        fh.write('<li><strong>Rows</strong>: drivers (top 20 by race count) or constructors.</li>')
        fh.write('<li><strong>Columns</strong>: circuits (circuit names).</li>')
        fh.write('<li><strong>Missing cells</strong>: blank or neutral color means insufficient data (no races for that pair).</li>')
        fh.write('<li><strong>Sample size</strong>: per-driver/circuit averages use all available races; confidence intervals are empirical percentiles computed only when a group has at least 5 residuals.</li>')
        fh.write('</ul>')
        if (OUT_DIR / 'heatmap_driver_by_circuit.png').exists():
            fh.write('<h2>Driver x Circuit heatmap</h2>')
            fh.write('<img src="heatmap_driver_by_circuit.png" style="max-width:100%;height:auto;"/>')
        if (OUT_DIR / 'heatmap_constructor_by_circuit.png').exists():
            fh.write('<h2>Constructor x Circuit heatmap</h2>')
            fh.write('<img src="heatmap_constructor_by_circuit.png" style="max-width:100%;height:auto;"/>')
        fh.write('</body></html>')
    print('Wrote', report_path)
    # Write a small metadata JSON to make UI parsing robust (timestamp, counts, flags)
    metadata = {}
    metadata['generated_at'] = datetime.now().astimezone().isoformat()
    try:
        metadata['total_residuals'] = int(len(df))
    except Exception:
        metadata['total_residuals'] = None
    try:
        metadata['seasons'] = mae['season'].astype(int).tolist() if not mae.empty else []
    except Exception:
        metadata['seasons'] = []
    metadata['has_multiple_seasons'] = bool(has_multiple_seasons)
    metadata['mae_rows'] = int(len(mae)) if mae is not None else 0
    metadata['ci_driver_track_rows'] = int(len(ci)) if ci is not None else 0
    metadata['ci_driver_rows'] = int(len(ci_driver)) if 'ci_driver' in locals() and ci_driver is not None else 0
    metadata['ci_constructor_rows'] = int(len(ci_constructor)) if 'ci_constructor' in locals() and ci_constructor is not None else 0
    metadata['heatmap_driver_created'] = (OUT_DIR / 'heatmap_driver_by_circuit.png').exists()
    metadata['heatmap_constructor_created'] = (OUT_DIR / 'heatmap_constructor_by_circuit.png').exists()
    # list of primary output files (use strings)
    files = [str(mae_out), str(ci_out)]
    if 'ci_driver_out' in locals():
        files.append(str(ci_driver_out))
    if 'ci_constructor_out' in locals():
        files.append(str(ci_constructor_out))
    if (OUT_DIR / 'heatmap_driver_by_circuit.png').exists():
        files.append(str(OUT_DIR / 'heatmap_driver_by_circuit.png'))
    if (OUT_DIR / 'heatmap_constructor_by_circuit.png').exists():
        files.append(str(OUT_DIR / 'heatmap_constructor_by_circuit.png'))
    files.append(str(report_path))
    metadata['files'] = files

    meta_path = OUT_DIR / 'position_group_analysis_metadata.json'
    try:
        json_helpers.safe_dump(metadata, meta_path, indent=2)
        print('Wrote', meta_path)
    except Exception as e:
        print('Failed to write metadata:', e)


if __name__ == '__main__':
    main()
