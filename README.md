# Industrial Defect Intelligence

MSc comparative computer vision project: six-class NEU-CLS defect classification using HOG–SVM and a compact CNN. There is no normal-steel class.

## Current milestone

`02_data_preparation_and_split.ipynb` corrects filename labels and validates the raw collection before publishing the locked split. `02a_development_validation_protocol.ipynb` prepares five shared development folds without training models.

Run Jupyter from this project folder and select the project `venv` kernel. Restart the kernel and run all cells in order. Notebook 01 is the historical audit: its duplicate failure is resolved by notebook 02's manifest exclusion, not by editing raw images or rewriting the historical result.

Install for notebook 02:
```bat
venv\Scripts\python.exe -m pip install numpy pandas matplotlib pillow scikit-learn ipykernel
```
Install for notebook 02a:
```bat
venv\Scripts\python.exe -m pip install numpy pandas scikit-learn ipykernel
```

The existing `requirements.txt` is retained. Tested package versions are recorded in `outputs/reports/02_environment.json` and `requirements-data-preparation.txt`.

## Data and outputs

- `data/NEU-CLS/`: original files, read only by these notebooks.
- `outputs/metrics/data_split_manifest.csv`: all 1,800 image rows, including exclusions.
- `outputs/metrics/split_lock.json`: split policy and checksum; preserve with the manifest.
- `outputs/reports/02_split_status.json`: latest split run status; downstream work must require PASS and a matching checksum.
- `outputs/reports/02_split_checks.csv`: critical validation results.
- `outputs/figures/02_split_class_distribution.png`: development/test class counts.
- `outputs/metrics/development_cv_folds.csv`: shared five-fold development assignments.
- `outputs/metrics/development_cv_lock.json`: fold lock linked to the split.
- `outputs/reports/before_label_fix/`: historical failed notebook and invalid outputs. Never use these as modelling inputs.

The seed is 42. The split contains 1,439 development and 360 test images (60 test images per class). `patches_101.jpg` is retained; the identical `patches_105.jpg` is excluded. Reruns reuse the lock and fail on unexpected changes. Do not delete locks just to get a run to pass.

## Remaining milestones

1. Completed: `03_classical_hog_svm.ipynb`, bounded development-only classical model selection.
2. Completed: `04_deep_learning_cnn.ipynb`, compact CPU CNN under the same five-fold validation protocol.
3. `05_comparison_xai_robustness.ipynb`: frozen-model final test comparison, errors, XAI, subgroup diagnostics, robustness and efficiency.
4. Consolidate into `industrial_defect_intelligence.ipynb`, with all core code in the notebook and an empty-folder reproduction check.

The final submission is not yet complete. It must create directories, obtain the unchanged dataset from the user's HTTPS mirror, verify the archive checksum, and regenerate its own manifests/folds/outputs. No dependency on helper scripts, previously executed notebooks or existing output CSVs is permitted in that final notebook. The mirror URL and verified original archive checksum are still pending.

See `METHODOLOGY.md` for decisions and limitations to retain in the presentation. The preparation milestone trained no models; the classical baseline is now complete (see below).


## Classical baseline completed

`03_classical_hog_svm.ipynb` has been executed using only development data. It verifies the split/fold acceptance records, extracts fixed HOG features and compares six SVM settings on the five shared folds. Every fold fits its own scaler. It saves all candidate/fold metrics, out-of-fold predictions, per-class results, figures and the selected model refitted on development data.

Each execution writes a separate `outputs/experiments/03_hog_svm_<run_id>/` directory. `outputs/reports/03_classical_latest.json` points to the latest successful run. The result is model-selection evidence, not final test performance. The test set remains locked.

Install for notebook 03:
```bat
venv\Scripts\python.exe -m pip install numpy pandas matplotlib pillow scikit-image scikit-learn joblib ipykernel
```
Tested versions are in `requirements-classical.txt`; the notebook records its environment and complete protocol per run. Notebook 04 has now completed the same five-fold development protocol; see the CNN milestone below.


## CNN milestone completed

`04_deep_learning_cnn.ipynb` is executed and validated with all five development folds. It saves fold histories/checkpoints, metrics, development diagnostics, final development refit, environment details, timing and access safeguards. The model uses greyscale 200x200 inputs, fold-training-only normalisation and validation macro F1 checkpoint selection. The test set remains unopened.

Install from the project root:
```bat
venv\Scripts\python.exe -m pip install torch==2.14.0 torchvision==0.29.0 numpy pandas matplotlib pillow scikit-learn
```
Tested dependencies are recorded in `requirements-cnn.txt`. Captum is optional for the synthetic compatibility check and will be needed for later Grad-CAM; it is already installed in the current environment.

`outputs/reports/04_cnn_latest.json` points to the latest successful run in `outputs/experiments/04_cnn_<run_id>/`. Each run includes `protocol.json`, `fold_metrics.csv`, fold histories, `cv_summary.csv`, confusion matrices, learning curves, `checkpoints/`, `image_access_log.csv`, `acceptance_checks.csv` and `result.json`. This validated run also includes `critical_interpretation.md`. Partial runs remain marked IN_PROGRESS or FAIL and are not published as successful results.

To repeat the experiment, restart the kernel and run all cells once. The five-fold cells and final development refit take several minutes on CPU. Do not run them concurrently. Every run writes a separate directory. Do not choose the best seed run after viewing performance. A failed/interrupted run must be rerun from the start; checkpoints are inference artefacts rather than automatic training-resume files.

The next milestone is to freeze the final comparison/XAI/subgroup/robustness plan before opening the test set. No test evaluation has been performed in notebook 04.
