import pandas as pd
import sys

shp_path = 'scripts/output/shap_ranking.txt'
print('Reading:', shp_path)
try:
    try:
        df = pd.read_csv(shp_path, sep=None, engine='python')
        print('read_csv with sep=None succeeded')
    except Exception as e:
        print('read_csv sep=None failed:', e)
        df = pd.read_csv(shp_path)
        print('read_csv fallback succeeded')
    print('DF type:', type(df))
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist()[:10])
    print('Head:')
    print(df.head(5).to_string(index=False))
except Exception as e:
    print('Top-level exception reading shap:', e)
    sys.exit(2)

# Try parsing using fallback text parser
try:
    rows = []
    def _is_number(x):
        try:
            float(x)
            return True
        except Exception:
            return False
    with open(shp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if ',' in s:
                parts = [p.strip() for p in s.split(',') if p.strip()]
            elif '\t' in s:
                parts = [p.strip() for p in s.split('\t') if p.strip()]
            elif ' - ' in s:
                parts = [p.strip() for p in s.split(' - ') if p.strip()]
            elif ':' in s and s.count(':') == 1:
                parts = [p.strip() for p in s.split(':') if p.strip()]
            else:
                parts = s.split()
            if len(parts) >= 2 and _is_number(parts[-1]):
                feature = ' '.join(parts[:-1]).strip()
                val = parts[-1]
            elif len(parts) >= 2:
                feature = ' '.join(parts[:-1]).strip()
                val = parts[-1]
            else:
                feature = s
                val = ''
            rows.append({'Feature': feature, 'SHAP': val})
    df2 = pd.DataFrame(rows)
    df2['SHAP'] = pd.to_numeric(df2['SHAP'], errors='coerce')
    print('\nFallback parsed rows:', len(df2))
    print(df2.head(5).to_string(index=False))
except Exception as e:
    print('Fallback parser exception:', e)
    sys.exit(3)

print('\nDone')
