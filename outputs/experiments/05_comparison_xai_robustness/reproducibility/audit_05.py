from pathlib import Path
import json, hashlib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
ROOT=Path(r'C:\Scripts\industrial-defect-intelligence')
RUN=ROOT/'outputs/experiments/05_comparison_xai_robustness'
checks=[]
def check(name,condition):
    checks.append({'check':name,'pass':bool(condition)})
    if not condition: raise AssertionError(name)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
    pr,re,f,_=precision_recall_fscore_support(y,p,labels=np.arange(6),average='macro',zero_division=0)
    return np.array([accuracy_score(y,p),pr,re,f])
classes=json.loads((ROOT/'outputs/metrics/split_lock.json').read_text())['classes']
nb=nbformat.read(ROOT/'05_comparison_xai_robustness.ipynb',as_version=4)
nbformat.validate(nb)
before=nbformat.read(RUN/'reproducibility/notebook_before_execution.ipynb',as_version=4)
check('Executed analysis code unchanged after test release',[c.source for c in nb.cells if c.cell_type=='code']==[c.source for c in before.cells if c.cell_type=='code'])
check('All code cells executed without errors',all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code'))
check('No em dashes in notebook source',all('\u2014' not in c.source for c in nb.cells))
check('Exact installation command',r'venv\Scripts\python.exe -m pip install numpy pandas matplotlib pillow scikit-learn scikit-image joblib torch==2.14.0 torchvision==0.29.0 captum==0.9.0' in nb.cells[0].source)
table=pd.read_csv(RUN/'tables/clean_metrics.csv').set_index('model')
with np.load(RUN/'cache/clean_predictions.npz',allow_pickle=False) as s:
    y=s['truth']; predictions={'HOG-SVM':s['svm_pred'],'CNN':s['cnn_pred']}
    manifest=pd.read_csv(ROOT/'outputs/metrics/data_split_manifest.csv')
    expected=manifest.loc[manifest.split.eq('test')].sort_values('path')
    check('Exactly the locked test rows in cache',len(y)==360 and np.array_equal(s['paths'],expected.path.to_numpy()))
    for model,p in predictions.items():
        check(model+' independent clean metric calculation',np.allclose(metric(y,p),table.loc[model,['accuracy','precision_macro','recall_macro','f1_macro']].to_numpy(dtype=float),atol=1e-12))
        cm=pd.read_csv(RUN/'tables'/f'{model.lower()}_confusion_counts.csv',index_col=0)
        check(model+' confusion counts',np.array_equal(cm.to_numpy(),confusion_matrix(y,p,labels=np.arange(6))))
        pr,re,f,support=precision_recall_fscore_support(y,p,labels=np.arange(6),zero_division=0)
        pc=pd.read_csv(RUN/'tables/per_class_metrics.csv').set_index(['model','class']).loc[model].reindex(classes)
        check(model+' per-class metrics',np.allclose(pc[['precision','recall','f1-score','support']].to_numpy(),np.array([pr,re,f,support]).T,atol=1e-12))
    robust=pd.read_csv(RUN/'tables/robustness_metrics.csv')
    for condition,group in robust.groupby('condition'):
        with np.load(RUN/'cache'/f'stress_{condition}.npz') as z:
            for r in group.itertuples():
                p=z['svm_pred' if r.model=='HOG-SVM' else 'cnn_pred']
                check(condition+' '+r.model+' saved metric',np.allclose(metric(y,p),[r.accuracy,r.precision_macro,r.recall_macro,r.f1_macro],atol=1e-12))
                check(condition+' '+r.model+' deltas',np.allclose([r.delta_accuracy,r.delta_macro_f1],[r.accuracy-table.loc[r.model,'accuracy'],r.f1_macro-table.loc[r.model,'f1_macro']],atol=1e-12))
    properties=pd.read_csv(RUN/'tables/test_image_properties.csv')
    subgroups=pd.read_csv(RUN/'tables/diagnostic_subgroup_metrics.csv')
    for r in subgroups.itertuples():
        idx=np.flatnonzero(properties[r.property+'_bin'].eq(r.bin))
        check(r.model+' '+r.property+' '+r.bin+' subgroup',len(idx)==r.n and np.allclose(metric(y[idx],predictions[r.model][idx]),[r.accuracy,r.precision_macro,r.recall_macro,r.f1_macro],atol=1e-12))
inventory=json.loads((RUN/'reproducibility/source_inventory.json').read_text())
check('Original sources remain unchanged',all(sha(ROOT/v['path'])==v['sha256'] for v in inventory.values()))
receipt=json.loads((RUN/'cache/clean_receipt.json').read_text())
check('Single clean evaluation integrity',receipt['main_evaluation_count']==1 and sha(RUN/'cache/clean_predictions.npz')==receipt['cache_sha256'])
check('Acceptance checks passed',pd.read_csv(RUN/'reports/acceptance_checks.csv').status.eq('PASS').all())
check('HTML exported',(RUN/'reports/05_comparison_xai_robustness.html').stat().st_size>10000)
(RUN/'reports/independent_validation.json').write_text(json.dumps({'status':'PASS','checks':checks,'scope':'Recalculated from saved predictions; no model inference or fitting'},indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','checks':len(checks),'clean_metrics':table.to_dict(orient='index')},indent=2))
