"""Compare real Streamlit widget selections with the React view protocol."""
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0, str(root))
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(root / 'fastapi_react/backend'))
from app.services.presentation import clean, render_view

app = AppTest.from_file(root / 'raceAnalysis.py', default_timeout=180).run()
app.checkbox(key='filter_results_main').check().run()
values = {'filter_results_main': True}
results = []

def walk(nodes):
    for node in nodes:
        yield node
        yield from walk(node.get('children', []))

def compare(name):
    if app.exception:
        raise RuntimeError([str(e.value) for e in app.exception])
    node = next(n for n in walk(render_view(1, values)['nodes']) if n['type'] == 'table')
    reference = next(t for t in app.tabs if t.label == '📊 Data Explorer').dataframe[0]
    frame = reference.value.copy()
    frame.columns = frame.columns.astype(str)
    keys = [c['key'] for c in node['columns']]
    config = json.loads(reference.proto.columns or '{}')
    result = {
        'scenario': name,
        'rows': len(node['rows']),
        'columns': len(keys),
        'values_match': clean(frame[keys].to_numpy()) == node['rows'],
        'formats_match': all(c.get('format') == config.get(c['key'], {}).get('type_config', {}).get('format') for c in node['columns']),
        'labels_match': all(c['label'] == config.get(c['key'], {}).get('label', c['key']) for c in node['columns']),
    }
    results.append(result)
    print(json.dumps(result), flush=True)

app.checkbox(key='checkbox_filter_DNF').check().run()
values['checkbox_filter_DNF'] = True
compare('Boolean: DNF')
app.checkbox(key='checkbox_filter_DNF').uncheck()
app.selectbox(key='filter_constructorName').select('Ferrari').run()
values.update(checkbox_filter_DNF=False, filter_constructorName='Ferrari')
compare('Category: Ferrari')
app.slider(key='range_filter_grandPrixYear').set_value((2020, 2024)).run()
values['range_filter_grandPrixYear'] = [2020, 2024]
compare('Category and numeric year range')
from datetime import date
app.slider(key='range_filter_short_date').set_value((date(2021, 1, 1), date(2023, 12, 31))).run()
values['range_filter_short_date'] = ['2021-01-01', '2023-12-31']
compare('Category, numeric and date range')

report = {'checks': results, 'failures': [r for r in results if not all(r[k] for k in ('values_match', 'formats_match', 'labels_match'))]}
(Path(__file__).parent / 'reference-filter-parity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
if report['failures']:
    sys.exit(1)
