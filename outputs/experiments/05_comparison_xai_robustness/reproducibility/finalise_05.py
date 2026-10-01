from pathlib import Path
import hashlib, json, shutil
import nbformat
import pandas as pd
from nbconvert import HTMLExporter

ROOT=Path(r'C:\Scripts\industrial-defect-intelligence')
RUN=ROOT/'outputs/experiments/05_comparison_xai_robustness'
OUT=Path(r'C:\Users\iesha\Documents\Codex\2026-09-27\referenced-chatgpt-conversation-this-is-an\outputs')
OUT.mkdir(exist_ok=True)
p=ROOT/'05_comparison_xai_robustness.ipynb'
nb=nbformat.read(p,as_version=4)
assert json.loads((RUN/'reports/independent_validation.json').read_text())['status']=='PASS'
original_code=[c.source for c in nb.cells if c.cell_type=='code']
clean=pd.read_csv(RUN/'tables/clean_metrics.csv').set_index('model')
pc=pd.read_csv(RUN/'tables/per_class_metrics.csv').set_index(['model','class'])
robust=pd.read_csv(RUN/'tables/robustness_metrics.csv').set_index(['condition','model'])
counts=pd.read_csv(RUN/'tables/diagnostic_subgroup_class_counts.csv').set_index(['property','bin'])
sg=pd.read_csv(RUN/'tables/diagnostic_subgroup_metrics.csv').set_index(['property','bin','model'])
assert counts.loc[('sharpness','low'),'n']==121
assert counts.loc[('sharpness','low'),['crazing','patches','rolled_in_scale']].eq(0).all()
assert len(pd.read_csv(RUN/'tables/cnn_high_confidence_errors.csv'))==0
notes={
7:f'''### Reading the clean comparison critically

The aggregate gain is not uniform across defect categories. CNN pitted-surface recall is {pc.loc[('CNN','pitted_surface'),'recall']:.4f}, compared with {pc.loc[('HOG-SVM','pitted_surface'),'recall']:.4f} for HOG-SVM. The CNN confusion matrix records 10 pitted-surface images predicted as inclusion. Conversely, patches recall is {pc.loc[('CNN','patches'),'recall']:.4f} for CNN versus {pc.loc[('HOG-SVM','patches'),'recall']:.4f} for HOG-SVM. An application where confusing pitting with inclusion is particularly costly could judge this trade-off differently from macro F1. No cost weighting is invented or selected from these test results.''',
11:'''### Interpretation of the displayed cases

In `crazing_100.jpg`, the predicted and true targets coincide and the maps are identical, as expected. Stronger responses occur in several regions of the textured image; the map does not trace a validated defect boundary. For `crazing_19.jpg`, the incorrect patches target has a broad positive response, while the crazing target is more concentrated on the right. This difference is target-dependent evidence, not proof that a particular feature caused the error.

For `patches_14.jpg`, the true patches map includes the prominent central dark region, while the predicted crazing map places relatively greater weight around it. The image is nevertheless misclassified. A visually plausible true-class map therefore does not imply that the corresponding class wins the model's decision. In `scratches_69.jpg`, both targets have broad responses and different emphasis near the vertical structure. The highest-softmax CNN error has a score of approximately 0.865, below the predeclared 0.90 threshold. There are no errors meeting that threshold in this sample; this does not establish calibration or justify a production confidence threshold.

Three of the five selected images are crazing examples because class-prefixed filenames affect the first-path rule. The rule avoids discretionary visual cherry-picking but sacrifices class coverage. These cases cannot characterise explanations across all six classes. HOG panels show gradient structure only; their appearance cannot explain the SVM's individual decisions.''',
13:f'''### Class composition explains an apparent subgroup collapse

The low-sharpness group has 121 images: 60 inclusion, 20 pitted surface and 41 scratches, with no crazing, patches or rolled-in-scale examples. Its fixed-six-class macro F1 is {sg.loc[('sharpness','low','CNN'),'f1_macro']:.4f} for CNN and {sg.loc[('sharpness','low','HOG-SVM'),'f1_macro']:.4f} for HOG-SVM, while accuracy is {sg.loc[('sharpness','low','CNN'),'accuracy']:.4f} and {sg.loc[('sharpness','low','HOG-SVM'),'accuracy']:.4f}, respectively. The three absent classes contribute zero to the declared macro calculation. Consequently, these low macro scores must not be presented as evidence of a comparable overall failure rate or as a causal effect of blur. Sharpness is strongly entangled with defect class in this collection. The class-specific recall table and subgroup sizes provide the more useful basis for targeted review.''',
15:f'''### The clean ranking does not hold under every stress

Under radius-0.5 blur, HOG-SVM macro F1 is {robust.loc[('blur_radius_0.5','HOG-SVM'),'f1_macro']:.4f}, exceeding CNN's {robust.loc[('blur_radius_0.5','CNN'),'f1_macro']:.4f}. Under contrast factor 0.8, HOG-SVM remains at {robust.loc[('contrast_0.8','HOG-SVM'),'f1_macro']:.4f}, while CNN falls to {robust.loc[('contrast_0.8','CNN'),'f1_macro']:.4f}. Stability under uniform contrast reduction is consistent with the normalised HOG representation, but this observation does not isolate the responsible mechanism.

CNN retains higher absolute macro F1 under the two brightness shifts and Gaussian noise. However, under noise its drop from clean performance is larger: {robust.loc[('noise_sigma_5','CNN'),'delta_macro_f1']:+.4f}, versus {robust.loc[('noise_sigma_5','HOG-SVM'),'delta_macro_f1']:+.4f} for HOG-SVM. Neither model is robust to that declared noise condition in the sense of maintaining its clean performance. A numerically small pixel perturbation is not necessarily mild for model performance. These findings make camera focus, noise and contrast relevant priorities for a future factory-specific evaluation; they do not warrant retrofitting preprocessing using this test set.'''
}
new=[]
for i,c in enumerate(nb.cells):
    if i==20: continue
    if i==18: new.append(nb.cells[20])
    new.append(c)
    if i in notes:
        new.append(nbformat.v4.new_markdown_cell(notes[i],metadata={'stage05_post_execution_interpretation':True}))
nb.cells=new
assert [c.source for c in nb.cells if c.cell_type=='code']==original_code
assert all('\u2014' not in c.source for c in nb.cells)
nbformat.validate(nb); nbformat.write(nb,p)
shutil.copy2(p,RUN/'05_comparison_xai_robustness.executed.ipynb')
body,_=HTMLExporter().from_notebook_node(nb)
(RUN/'reports/05_comparison_xai_robustness.html').write_text(body,encoding='utf-8')
parts=[]
for c in nb.cells:
    if c.cell_type=='markdown': parts.append(c.source)
    else:
        for output in c.get('outputs',[]):
            data=output.get('data',{})
            if 'text/markdown' in data: parts.append(data['text/markdown'])
(RUN/'reports/complete_commentary.md').write_text('\n\n'.join(parts),encoding='utf-8')
(RUN/'reports/post_execution_review.json').write_text(json.dumps({'status':'PASS','changes':'Added result-specific and visually inspected commentary only; moved reproducibility notes before final conclusion','code_cells_unchanged':True,'models_rerun':False,'figures_visually_reviewed':['clean_confusion_matrices.png','gradcam_and_hog_cases.png','robustness_deltas.png'],'initial_preflight_failure':'SVM deprecated probability sentinel; corrected before any test decoding; failed attempt retained in reproducibility'},indent=2),encoding='utf-8')
for name in ['audit_05.py','finalise_05.py','fix_preflight.py']:
    shutil.copy2(Path(__file__).parent/name,RUN/'reproducibility'/name)
# User-facing copies for this task. The authoritative runnable notebook stays in the project root.
shutil.copy2(p,OUT/'05_comparison_xai_robustness.ipynb')
shutil.copy2(RUN/'reports/05_comparison_xai_robustness.html',OUT/'05_comparison_xai_robustness.html')
shutil.copy2(RUN/'reports/complete_commentary.md',OUT/'05_complete_commentary.md')
print('Final commentary reconciled with saved results; code and predictions unchanged.')
