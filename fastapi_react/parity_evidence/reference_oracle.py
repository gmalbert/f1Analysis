"""Read the reference app's actual table values without running its server."""
import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
import json

root=Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(root)
sys.path.insert(0,str(root))
app=AppTest.from_file(root/'raceAnalysis.py',default_timeout=180).run()
print('Exceptions:',[str(e.value) for e in app.exception])
print('Tabs:',[(t.label,len(t.dataframe)) for t in app.tabs])
print('Tables:',[(len(t.value),len(t.value.columns),list(t.value.columns)[:5]) for t in app.dataframe])
print('First table config:',app.dataframe[0].proto.columns)
sys.path.insert(0,str(root/'fastapi_react/backend'))
from app.services.presentation import render_view,clean

def tables(nodes):
    for node in nodes:
        if node['type']=='table':yield node
        yield from tables(node.get('children',[]))

labels=['📊 Data Explorer','📈 Analytics & Visualizations','🏎️ Schedule','🏁 Next Race','🤖 Predictive Models','💾 Data & Debug','📐 Betting Research']
results=[]
scenarios=[(page,{}) for page in range(1,8)]+[(5,{'_tabs:📊 Model Performance':i}) for i in range(1,7)]
for page,values in scenarios:
    label=labels[page-1]
    reference=next(t for t in app.tabs if t.label==label)
    view=render_view(page,values)
    for number,node in enumerate(tables(view['nodes'])):
        keys=[c['key'] for c in node['columns']]
        candidates=[t for t in reference.dataframe if len(t.value)==len(node['rows']) and all(k in t.value.columns.astype(str) for k in keys)]
        matching=[]
        for table in candidates:
            frame=table.value.copy(); frame.columns=frame.columns.astype(str)
            if clean(frame[keys].to_numpy())==node['rows']: matching.append(table)
        result={'page':page,'table':number,'rows':len(node['rows']),'columns':len(keys),'values_match':bool(matching)}
        if matching:
            config=json.loads(matching[0].proto.columns or '{}')
            result['labels_match']=all(c['label']==config.get(c['key'],{}).get('label',c['key']) for c in node['columns'])
            original=matching[0].value.columns.astype(str).tolist()
            order=list(matching[0].proto.column_order)
            expected=order if order else original
            expected=[k for k in expected if k in original and not config.get(k,{}).get('hidden',False)]
            result['column_order_match']=keys==expected
            result['index_visibility_match']=node['hide_index']==config.get('_index',{}).get('hidden',False)
        results.append(result)
report={'table_checks':results,'failures':[r for r in results if not all(r.get(k,False) for k in ['values_match','labels_match','column_order_match','index_visibility_match'])]}
(Path(__file__).parent/'reference-table-parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
if report['failures']:sys.exit(1)

# A selected model must change the actual estimator and predictions, including
# ensemble classes that were pickled under the former application module.
models=['XGBoost','LightGBM','CatBoost','Ensemble (XGBoost + LightGBM + CatBoost)','Position Group','Track-Weighted Ensemble']
for model in models:
    next(w for w in app.selectbox if w.label=='Select Model Type').select(model).run()
    if app.exception:raise RuntimeError([e.value for e in app.exception])
    reference=next(t for t in app.tabs if t.label==labels[4])
    for number,node in enumerate(tables(render_view(5,{'Select Model Type':model})['nodes'])):
        keys=[c['key'] for c in node['columns']]
        matching=False
        for table in reference.dataframe:
            frame=table.value.copy();frame.columns=frame.columns.astype(str)
            if len(frame)==len(node['rows']) and all(k in frame for k in keys) and clean(frame[keys].to_numpy())==node['rows']:
                matching=True;break
        result={'model':model,'table':number,'values_match':matching}
        report['table_checks'].append(result)
        if not matching:report['failures'].append(result)
    print('Compared model:',model,flush=True)

(Path(__file__).parent/'reference-table-parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if report['failures']:sys.exit(1)
