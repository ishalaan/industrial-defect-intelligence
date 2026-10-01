from pathlib import Path
import json, hashlib, shutil, zipfile
ROOT=Path(r'C:\Scripts\industrial-defect-intelligence')
RUN=ROOT/'outputs/experiments/05_comparison_xai_robustness'
OUT=Path(r'C:\Users\iesha\Documents\Codex\2026-09-27\referenced-chatgpt-conversation-this-is-an\outputs')
assert json.loads((RUN/'reports/independent_validation.json').read_text())['status']=='PASS'
review_path=RUN/'reports/post_execution_review.json'
review=json.loads(review_path.read_text())
review['figures_visually_reviewed']+=['diagnostic_subgroups.png','stress_examples.png']
review_path.write_text(json.dumps(review,indent=2),encoding='utf-8')
(RUN/'README.md').write_text('''# Stage 05 final evidence

Status: PASS. Both frozen models were evaluated on the same 360 locked test images. No model was retrained, calibrated or retuned. The clean receipt records one main evaluation. All five fixed stress conditions, attribution, subgroup and timing analyses completed.

The authoritative runnable notebook is ../../.. / 05_comparison_xai_robustness.ipynb, in the project root. The executed archive is 05_comparison_xai_robustness.executed.ipynb. Read reports/05_comparison_xai_robustness.html for the complete rendered notebook, reports/complete_commentary.md for the full narrative, and tables/ for numerical evidence.

reports/independent_validation.json records independent recomputation from saved predictions, without running either model. reports/post_execution_review.json records figure and narrative review. reproducibility/output_inventory_sha256.json records output checksums. Source artefact hashes and the predeclared protocol were verified unchanged at completion.

The first preflight attempt stopped because scikit-learn represents the unused SVM probability option with a deprecation sentinel. The check was corrected after verifying there is no probability-prediction method. This happened before any test decoding. That unsuccessful preflight and its error are preserved in reproducibility; they do not represent a second test evaluation. The successful execution follows in execution_log.jsonl.

Do not delete cache/clean_evaluation_started.json, cache/clean_receipt.json or cache/clean_predictions.npz to repeat evaluation. A rerun reuses integrity-checked clean predictions. Future model changes require independent evaluation data.

The output archive does not include the dataset, frozen models or virtual environment. Those remain unchanged in the original project. The HTML can be read independently; notebook execution requires the existing project root and verified upstream artefacts.
'''.replace('../../.. / 05_comparison_xai_robustness.ipynb','../../../05_comparison_xai_robustness.ipynb'),encoding='utf-8')
for name in ['audit_05.py','package_05.py']:
    shutil.copy2(Path(__file__).parent/name,RUN/'reproducibility'/name)
inventory=[]
for p in sorted(RUN.rglob('*')):
    if p.is_file() and p.name!='output_inventory_sha256.json':
        inventory.append({'path':p.relative_to(RUN).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(RUN/'reproducibility/output_inventory_sha256.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
for r in inventory:
    assert hashlib.sha256((RUN/r['path']).read_bytes()).hexdigest()==r['sha256']
with zipfile.ZipFile(OUT/'05_evidence_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RUN.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(RUN))
print(json.dumps({'status':'PASS','output_files':len(inventory)+1,'bundle_bytes':(OUT/'05_evidence_bundle.zip').stat().st_size,'validated_hashes':len(inventory)},indent=2))
