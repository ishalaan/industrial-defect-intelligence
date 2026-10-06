# Industrial Defect Intelligence

## Purpose and overview

Industrial Defect Intelligence compares classical machine learning and deep learning for six-class steel surface defect classification. It investigates whether a compact convolutional neural network (CNN) improves on handcrafted Histogram of Oriented Gradients (HOG) features with a Support Vector Machine (SVM), and whether any performance gain is maintained under image-condition variation and practical computational constraints.

The project uses NEU-CLS: 1,800 greyscale 200 x 200 images across crazing, inclusion, patches, pitted surface, rolled-in scale and scratches. **There is no normal or defect-free class.** The task classifies an already selected defect image; it does not determine whether arbitrary steel is defective or locate defects within a production image.

The complete implementation is [`industrial_defect_intelligence.ipynb`](industrial_defect_intelligence.ipynb). Its workflow is:

1. Obtain the dataset if absent, verify the downloaded archive, and audit the image files.
2. Derive labels from filenames and exclude one redundant exact duplicate while preserving the raw images.
3. Create a reproducible stratified development/test split and five shared development folds.
4. Select a HOG-SVM configuration and train a compact PyTorch CNN using development data only.
5. Compare 25- and 40-epoch CNN limits under a predeclared sensitivity rule.
6. Refit the selected models on development data and freeze their specifications before test evaluation.
7. Compare clean test performance, class confusions, Grad-CAM explanations, image-condition subgroups, robustness and computational costs.
8. Save the models, predictions, figures, environment details, checksums and acceptance records in a new run directory.

## Installation and dependencies

The recorded environment was **Windows 11, Python 3.14.0 and CPU-only PyTorch 2.14.0**, with torchvision 0.29.0 and Captum 0.9.0. CUDA is not required. Training uses up to four CPU threads. The macOS and Linux commands below follow the same environment setup, but those platforms have not been verified in this project's recorded execution evidence. Package availability and numerical behaviour may differ across platforms.

Internet access is required to install dependencies and to download the dataset when it is absent. No API key, hosted model account, database or `.env` configuration is required by the complete notebook.

### Get the source

Open a terminal in the folder where you want to save the project. On Windows, use **Command Prompt (CMD)**; if PowerShell is open, type `cmd` first.

```text
git clone git@github.com:ishalaan/industrial-defect-intelligence.git
```

The SSH command requires a key linked to a GitHub account with repository access. If SSH is not configured, use HTTPS instead:

```text
git clone https://github.com/ishalaan/industrial-defect-intelligence.git
```

If using the submitted notebook, a source archive or an existing checkout, skip cloning. Put the notebook in its own project folder. Replace `path/to/industrial-defect-intelligence/` below with that folder's actual location. The complete notebook can run without the development notebooks or pre-existing experiment outputs.

### Windows: Command Prompt (CMD)

```bat
cd /d "path/to/industrial-defect-intelligence/"
python -m venv venv
venv\Scripts\activate.bat
```

If Windows recognises `py` instead of `python`, use `py -m venv venv`. After activation, use `python` as shown below. No PowerShell execution-policy change is needed.

### macOS: Terminal

```sh
cd "path/to/industrial-defect-intelligence/"
python3 -m venv venv
source venv/bin/activate
```

### Linux: Terminal (Bash)

```sh
cd "path/to/industrial-defect-intelligence/"
python3 -m venv venv
source venv/bin/activate
```

If Linux reports that `venv` is unavailable, install the venv package for your Python version through your distribution's package manager, then retry.

### Install the notebook dependencies

With the environment activated, run:

```text
python -m pip install numpy pandas matplotlib pillow scikit-learn scikit-image joblib torch==2.14.0 torchvision==0.29.0 captum==0.9.0 tqdm scipy ipython ipykernel nbformat nbclient jupyterlab ipywidgets
```

`ipywidgets` supplies optional notebook progress-bar support. Its absence can cause an IProgress warning without invalidating training. Torchvision is retained for consistency with the recorded environment; the custom CNN does not use pretrained torchvision models.

For the existing Windows project, the same command can be invoked explicitly through its environment:

```bat
venv\Scripts\python.exe -m pip install numpy pandas matplotlib pillow scikit-learn scikit-image joblib torch==2.14.0 torchvision==0.29.0 captum==0.9.0 tqdm scipy ipython ipykernel nbformat nbclient jupyterlab ipywidgets
```

The repository's `requirements.txt` is the pinned environment snapshot, including indirect and Windows-specific packages such as `pywinpty`. On a compatible Windows setup it can be installed with `python -m pip install -r requirements.txt`. Use the direct dependency command above for a notebook-only submission or another platform, and retain the newly recorded versions when comparing results. The smaller `requirements-data-preparation.txt`, `requirements-classical.txt` and `requirements-cnn.txt` document individual development stages.

Create the environment once. On subsequent visits, activate it and launch Jupyter; there is no need to recreate it or reinstall packages routinely.

## Dataset and configuration

The complete notebook uses the [Figshare NEU-CLS version-1 distribution](https://doi.org/10.6084/m9.figshare.28903550.v1), with original Northeastern University provenance credited separately. The notebook records the distribution's CC BY 4.0 licence and attribution. Its direct download endpoint is [the NEU-CLS archive](https://ndownloader.figshare.com/files/54094775).

If image files are already present under `data/NEU-CLS/`, they are audited in place. Otherwise, the notebook downloads the archive, checks its recorded byte count and published MD5, validates archive paths and extracts the data. A partial or unexpected dataset fails the audit rather than being silently mixed with a new download. SHA-256 hashes record local image and model identity. Matching the published distribution checksum is not proof that its packaging is identical to a separately hosted university archive.

The experimental configuration is visible near the top of the notebook:

- **Seed:** 42 for partitioning and the declared seeded training procedures.
- **Duplicate policy:** retain the first sorted path for exact image content and exclude the redundant copy from modelling. The audited pair is `patches_101.jpg` and `patches_105.jpg`.
- **Usable pool:** 1,799 images; development 1,439; test 360; excluded duplicate 1.
- **Validation:** five stratified development folds shared by both model families.
- **Classical search:** linear and RBF SVMs with C in {0.1, 1, 10}; gamma `scale` for RBF.
- **CNN:** 6,142 trainable parameters, greyscale input, three convolutional blocks, adaptive average pooling, dropout 0.25, AdamW, learning rate 0.001 and batch size 32.
- **Stopping and selection:** validation macro F1, patience five, earliest exact tie, and a controlled 25-versus-40 epoch-cap comparison. The final refit uses the ceiling of the median selected fold epoch.

Do not change settings in response to test results while presenting the exercise as the original experiment. A protocol change is a new development experiment and must be documented.

## Run the complete notebook

From the project folder in the activated environment:

```text
python -m jupyter lab
```

Then:

1. Open `industrial_defect_intelligence.ipynb`.
2. Select the Python kernel belonging to the project environment.
3. Confirm that the notebook's working directory is the intended project folder. Paths are based on `Path.cwd()`, not automatically on the notebook's filename.
4. Choose **Restart Kernel and Run All Cells** and allow the run to finish.
5. Review the final acceptance table and `PASS` result.

The environment table records the Python executable. If the kernel is unclear, `import sys; print(sys.executable)` should point inside the project `venv`.

**This notebook retrains models.** It runs 30 SVM candidate/fold fits, two five-fold CNN experiments and final development refits before evaluating the frozen models. Allow several minutes or longer depending on CPU load and hardware. Progress is printed during training. Start with a fresh kernel: the raw-file access guard is process-scoped and a previous run's guard remains active until that kernel is restarted.

Each execution creates `outputs/fresh_<timestamp>/`. Models and results are saved there rather than replacing an earlier run. Saving or autosaving the notebook itself updates its displayed outputs, so use a separate notebook copy and working folder if preserving an executed presentation master. Timings will vary; seeded training improves repeatability but does not guarantee identical scores or model bytes across software and hardware changes.

### Presentation verification without retraining

`industrial_defect_intelligence_demo.ipynb` is a separate verification copy. It is bound to the original local experiment at `outputs/fresh_20260927T194627_610401Z/` and currently uses the project path `C:\Scripts\industrial-defect-intelligence`. It therefore requires that saved evidence and is **not** the standalone notebook for a fresh reproduction on another machine.

Open the demo with the project kernel and choose **Restart Kernel and Run All Cells**. Its code checks frozen model and evidence-file hashes, recalculates metrics from saved prediction labels, and displays original figures and timing tables. It does not train models, load model objects, generate fresh predictions, open raw test images or write experiment files. Jupyter can still save the demo notebook's displayed outputs.

The original markdown commentary is retained unchanged by request; it describes the complete experiment. The first code-cell output identifies the actual demonstration as saved-evidence verification. Describe it in the recording as verification of previously completed training, not live execution of the entire training workflow. If required evidence is absent or changed, the demo fails rather than retraining automatically.

## Validation and recorded results

The recorded complete run is [`outputs/fresh_20260927T194627_610401Z/`](outputs/fresh_20260927T194627_610401Z/), completed on 27 September 2026. Its [`status.json`](outputs/fresh_20260927T194627_610401Z/status.json) reports PASS and its [`acceptance_checks.csv`](outputs/fresh_20260927T194627_610401Z/acceptance_checks.csv) records **15 passing end-to-end acceptance checks**. These are notebook checks, not a standalone pytest suite.

The checks cover counts, exact-content separation, shared folds, all SVM candidates, both CNN cap experiments, development-only access before the freeze, denial of premature test access, prediction/metric validity, and unchanged model, protocol, split and freeze artefacts.

### Development results

Values below are five-fold means with sample standard deviations. They are model-selection estimates rather than independent final evaluation.

| Model | Accuracy, mean +/- SD | Macro F1, mean +/- SD |
| --- | ---: | ---: |
| HOG-SVM | 0.8888 +/- 0.0121 | 0.8871 +/- 0.0125 |
| Compact CNN | 0.9375 +/- 0.0288 | 0.9377 +/- 0.0284 |

The selected SVM is RBF with C = 10 and gamma `scale`. The CNN sensitivity experiment retained the 25-epoch cap: extending to 40 did not improve the selected fold scores in this recorded run. The final CNN was refitted for 20 epochs. See [`development_comparison.csv`](outputs/fresh_20260927T194627_610401Z/development_comparison.csv) and [`sensitivity_decision.json`](outputs/fresh_20260927T194627_610401Z/sensitivity_decision.json).

### Frozen-model test results

Both models were evaluated on the same 360 reserved images after their specifications were frozen.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| HOG-SVM | 0.9028 | 0.9055 | 0.9028 | 0.9023 |
| Compact CNN | 0.9556 | 0.9611 | 0.9556 | 0.9555 |

These values come from [`test_results.csv`](outputs/fresh_20260927T194627_610401Z/test_results.csv); per-image labels are retained in [`test_predictions.csv`](outputs/fresh_20260927T194627_610401Z/test_predictions.csv).

Higher clean accuracy did not imply uniformly greater robustness. Under the declared blur stress, macro F1 fell to 0.7659 for HOG-SVM and 0.5547 for the CNN. Under the noise stress it fell to 0.5640 and 0.5402 respectively. These are controlled synthetic perturbations, not measurements of production-line reliability. See [`robustness.csv`](outputs/fresh_20260927T194627_610401Z/robustness.csv).

The demo's code was separately validated through **184 read-only checks** when prepared. It verifies the saved evidence and recalculated metrics; it is not an independent replication of training or a new test sample.

### Saved evidence

| File within the recorded run | Purpose |
| --- | --- |
| `protocol.json`, `environment.json`, `acquisition.json` | Declared settings, software/hardware and data acquisition record. |
| `manifest.csv`, `shared_folds.csv`, `duplicate_decisions.csv` | Image roles, development folds and duplicate handling. |
| `svm_candidate_folds.csv`, `svm_selection.csv` | All classical candidate/fold results and selection summary. |
| `cnn25_folds.csv`, `cnn40_folds.csv`, `history_cap*_fold*.csv` | CNN fold metrics and epoch histories for both caps. |
| `freeze.json`, `final_svm.joblib`, `final_cnn.pt` | Frozen specification, model hashes and final fitted models. |
| `test_predictions.csv`, `test_results.csv`, `*_per_class.csv` | Clean predictions and aggregate/per-class metrics. |
| `*_confusion.png`, `selected_cnn_learning_curves.png` | Confusion analysis and recorded learning curves. |
| `explanations.png`, `gradcam_registry.csv` | Saved Grad-CAM/HOG visualisations and selected cases. |
| `subgroup_metrics.csv`, `subgroup_composition.csv` | Image-condition diagnostics and their class composition. |
| `robustness.csv`, `predictions_*.csv` | Stress-test scores and saved perturbed-image predictions. |
| `latency.csv`, `throughput.csv`, `efficiency.csv`, `timing_scope.json` | Recorded computational measurements and their scope. |
| `raw_access_log.csv`, `denied_access_log.csv`, `acceptance_checks.csv`, `result.json` | Execution safeguards and completion evidence. |

## Project structure

| Path | Purpose |
| --- | --- |
| `industrial_defect_intelligence.ipynb` | Complete standalone experimental implementation; retrains and generates a new run. |
| `industrial_defect_intelligence_demo.ipynb` | Local verification demonstration using the fixed saved experiment. |
| `01_dataset_audit_and_eda.ipynb` | Original image integrity audit and exploratory analysis. |
| `02_data_preparation_and_split.ipynb` | Corrected labels, reproducible duplicate handling and saved locked split. |
| `02a_development_validation_protocol.ipynb` | Shared five-fold development assignments. |
| `03_classical_hog_svm.ipynb` | Classical development experiments. |
| `04_deep_learning_cnn.ipynb` | Compact CNN development and refit. |
| `04a_cnn_epoch_sensitivity.ipynb` | Development-only epoch-cap sensitivity experiment. |
| `05_comparison_xai_robustness.ipynb` | Staged frozen-model evaluation, explanation and robustness analysis. |
| `data/NEU-CLS/` | Original local dataset; not required in advance for the complete notebook. |
| `outputs/fresh_<timestamp>/` | Independent outputs from each complete notebook run. |
| `outputs/experiments/`, `outputs/metrics/`, `outputs/reports/`, `outputs/figures/` | Evidence and artefacts from staged development. |
| `requirements.txt`, `requirements-*.txt` | Environment snapshot and stage-specific dependency records. |
| `METHODOLOGY.md` | Development decisions and critical methodological notes. |
| `jupyter-run.bat` | Convenience launcher for the local Windows workflow. |
| `venv/` | Local Python environment; recreate rather than copy between machines. |

The staged notebooks document development history. Their earlier audit failures remain historical evidence; duplicate handling and label corrections were implemented in subsequent stages. They need not be executed before the complete notebook.

## Key design decisions

- **Data integrity before fitting:** filename labels and exact-content checks prevent generic folder names and redundant copies from silently contaminating evaluation. Exclusions are recorded rather than deleting raw images.
- **Shared validation:** the same development folds permit a paired descriptive comparison. SVM scaling and CNN normalisation are fitted within the current training subset. BatchNorm statistics update only during training.
- **Bounded model selection:** six SVM configurations and one compact CNN design keep the experiment inspectable. The CNN cap check changes the epoch limit under explicit rules, without test-guided tuning.
- **Separation of selection and evaluation:** freeze the selected models, preprocessing, subgroup cut points and stress severities before opening the reserved test images. Reuse clean predictions for later clean diagnostics.
- **Explainability with boundaries:** Grad-CAM supplies spatial attribution for CNN predictions (Selvaraju et al., 2017). HOG plots show the representation, not an explanation of the SVM's individual decision.
- **Robustness beyond clean scores:** brightness, contrast, blur and noise stresses test sensitivity under a declared project-specific protocol, informed by the broader corruption-testing literature (Hendrycks and Dietterich, 2019).
- **Traceability:** versioned run directories, explicit environment records, saved predictions and model checksums support verification. Repeated execution is not an independent new evaluation dataset.

## Limitations and responsible use

- **Representativeness:** the benchmark does not establish transfer across factories, cameras, steel grades or acquisition conditions. Image-level splitting cannot establish production-group independence when such identifiers are unavailable.
- **Statistical interpretation:** CV and epoch selection reuse development validation evidence. Fold SD is descriptive, not a confidence interval, and reflects both partition and initialisation effects. Search budgets differ between model families. No formal significance claim is made.
- **Integrity and provenance:** exact duplicate removal does not detect all near-duplicates. Earlier full-collection EDA limits claims of prospective blindness. Local provenance and repackaging remain distinct from matching a distribution checksum.
- **Subgroups and explanations:** image-condition groups can be small and class-confounded. They are not demographic fairness categories. Heatmaps do not establish causality, label correctness or safe deployment.
- **Operational use:** no normal class, severity labels, cost model or validated acceptance threshold is available. Do not use this benchmark result to justify autonomous acceptance or rejection of steel. Production assessment requires independent representative validation, human oversight, escalation, monitoring and accountability.
- **Resource measurements:** latency and throughput depend on workload, machine and timing scope. Synthetic stress scores and desktop timings are not production service-level guarantees or energy measurements.
- **Professional practice:** attribute the original researchers and distribution, respect applicable licences, protect any future proprietary production images, retain experiment records and disclose assistance in accordance with institutional requirements.

## External technology acknowledgements

The implementation uses **PyTorch** for the custom CNN; **scikit-learn** for SVMs, preprocessing, stratified splitting and metrics; **scikit-image** for HOG; **Captum** for Grad-CAM; **NumPy** and **pandas** for arrays and tabular analysis; **Pillow** for image decoding; **SciPy** for image-condition measurements and perturbations; **Matplotlib** for figures; **joblib** for SVM persistence; and **tqdm** for progress reporting. **JupyterLab, IPython, ipykernel, nbformat and nbclient** support notebook execution and display. **Git and GitHub** support source version control. Dataset and software licences remain separate from the project's own implementation.

## Academic references

The following references are also identified in the complete notebook; consult its reference list for the additional distribution-shift, steel-inspection and implementation sources.

Cao, W. (2025) *NEU-CLS*, version 1. Figshare. Available at: <https://doi.org/10.6084/m9.figshare.28903550.v1> (Accessed: 6 October 2026). Distribution licence: <https://creativecommons.org/licenses/by/4.0/> (Accessed: 6 October 2026).

Song, K. and Yan, Y. (2013) 'A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects', *Applied Surface Science*, 285, pp. 858-864. Dataset description and citation: <https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/> (Accessed: 6 October 2026).

Dalal, N. and Triggs, B. (2005) 'Histograms of oriented gradients for human detection', *IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 1, pp. 886-893. Available at: <https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf> (Accessed: 6 October 2026).

Cortes, C. and Vapnik, V. (1995) 'Support-vector networks', *Machine Learning*, 20, pp. 273-297. Available at: <https://doi.org/10.1007/BF00994018> (Accessed: 6 October 2026).

Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) 'Grad-CAM: Visual explanations from deep networks via gradient-based localization', *IEEE International Conference on Computer Vision*. Available at: <https://arxiv.org/abs/1610.02391> (Accessed: 6 October 2026).

Hendrycks, D. and Dietterich, T. (2019) 'Benchmarking neural network robustness to common corruptions and perturbations', *International Conference on Learning Representations*. Available at: <https://arxiv.org/abs/1903.12261> (Accessed: 6 October 2026).
