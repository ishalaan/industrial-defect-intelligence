from pathlib import Path
import nbformat, shutil
root=Path(r'C:\Scripts\industrial-defect-intelligence')
run=root/'outputs/experiments/05_comparison_xai_robustness'
assert not (run/'cache/clean_evaluation_started.json').exists()
p=root/'05_comparison_xai_robustness.ipynb'
shutil.copy2(p,run/'reproducibility/preflight_attempt_1.ipynb')
nb=nbformat.read(p,as_version=4)
for c in nb.cells:
    c.source=c.source.replace("check('No SVM probability fit',svm.named_steps['svm'].probability is False)","check('No SVM probability prediction method',not hasattr(svm,'predict_proba'))")
    if c.cell_type=='code':
        c.outputs=[]; c.execution_count=None
nbformat.write(nb,p)
shutil.copy2(p,run/'reproducibility/notebook_before_execution.ipynb')
shutil.copy2(run/'reproducibility/execution_error.txt',run/'reproducibility/preflight_attempt_1_error.txt')
