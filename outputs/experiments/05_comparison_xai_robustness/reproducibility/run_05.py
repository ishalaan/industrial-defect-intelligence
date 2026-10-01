from pathlib import Path
import sys, json, time, traceback, shutil, hashlib, os
import nbformat
from nbclient import NotebookClient
ROOT=Path(r'C:\Scripts\industrial-defect-intelligence')
RUN=ROOT/'outputs/experiments/05_comparison_xai_robustness'
REP=RUN/'reproducibility'
REP.mkdir(parents=True,exist_ok=True)
kernel=REP/'jupyter/kernels/stage05'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Stage 05 project Python','language':'python'}),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(REP/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
path=ROOT/'05_comparison_xai_robustness.ipynb'
nb=nbformat.read(path,as_version=4)
snapshot=REP/'notebook_before_execution.ipynb'
if not snapshot.exists(): shutil.copy2(path,snapshot)
for f in ['build_05.py','run_05.py']:
    source=Path(__file__).parent/f
    if source.exists(): shutil.copy2(source,REP/f)
client=NotebookClient(nb,timeout=1800,kernel_name='stage05',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
log=REP/'execution_log.jsonl'
try:
    with client.setup_kernel():
        count=0
        for i,cell in enumerate(nb.cells):
            if cell.cell_type!='code': continue
            count+=1; started=time.time()
            print(f'Executing code cell {count} (notebook index {i})',flush=True)
            try:
                client.execute_cell(cell,i,execution_count=count)
            finally:
                nbformat.write(nb,path)
            with log.open('a',encoding='utf-8') as h:
                h.write(json.dumps({'cell_index':i,'execution_count':count,'seconds':time.time()-started,'status':'PASS'})+'\n')
            for out in cell.get('outputs',[]):
                if out.output_type=='stream': print(out.text[-2200:],flush=True)
    nbformat.validate(nb)
    shutil.copy2(path,RUN/'05_comparison_xai_robustness.executed.ipynb')
    from nbconvert import HTMLExporter
    body,_=HTMLExporter().from_notebook_node(nb)
    (RUN/'reports/05_comparison_xai_robustness.html').write_text(body,encoding='utf-8')
    inventory=[]
    for p in sorted(RUN.rglob('*')):
        if p.is_file() and p.name!='output_inventory_sha256.json':
            inventory.append({'path':p.relative_to(RUN).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (REP/'output_inventory_sha256.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print('COMPLETE: executed notebook, HTML and output hash inventory saved',flush=True)
except BaseException:
    error=traceback.format_exc()
    (REP/'execution_error.txt').write_text(error,encoding='utf-8')
    print(error,flush=True)
    raise
