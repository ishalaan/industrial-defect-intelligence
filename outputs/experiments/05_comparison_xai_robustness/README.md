# Stage 05 final evidence

Status: PASS. Both frozen models were evaluated on the same 360 locked test images. No model was retrained, calibrated or retuned. The clean receipt records one main evaluation. All five fixed stress conditions, attribution, subgroup and timing analyses completed.

The authoritative runnable notebook is ../../../05_comparison_xai_robustness.ipynb, in the project root. The executed archive is 05_comparison_xai_robustness.executed.ipynb. Read reports/05_comparison_xai_robustness.html for the complete rendered notebook, reports/complete_commentary.md for the full narrative, and tables/ for numerical evidence.

reports/independent_validation.json records independent recomputation from saved predictions, without running either model. reports/post_execution_review.json records figure and narrative review. reproducibility/output_inventory_sha256.json records output checksums. Source artefact hashes and the predeclared protocol were verified unchanged at completion.

The first preflight attempt stopped because scikit-learn represents the unused SVM probability option with a deprecation sentinel. The check was corrected after verifying there is no probability-prediction method. This happened before any test decoding. That unsuccessful preflight and its error are preserved in reproducibility; they do not represent a second test evaluation. The successful execution follows in execution_log.jsonl.

Do not delete cache/clean_evaluation_started.json, cache/clean_receipt.json or cache/clean_predictions.npz to repeat evaluation. A rerun reuses integrity-checked clean predictions. Future model changes require independent evaluation data.

The output archive does not include the dataset, frozen models or virtual environment. Those remain unchanged in the original project. The HTML can be read independently; notebook execution requires the existing project root and verified upstream artefacts.
