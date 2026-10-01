# Methodological decision record

Research question: To what extent does end-to-end representation learning improve multiclass steel-surface defect classification over handcrafted HOG features, and are any performance gains maintained under validation, image-condition variation and practical deployment constraints?

## Confirmed protocol from the earlier project conversation

- Computer Vision track; NEU-CLS six-class classification; HOG–SVM versus a compact PyTorch CNN.
- Stratified 80/20 development/test split, seed 42; five-fold stratified CV within development for both models.
- Required reporting: accuracy, macro precision/recall/F1, per-class results, confusion matrices and fold variability.
- Explainability: CNN Grad-CAM (planned Captum); HOG visualisation explains representation, not necessarily the SVM decision.
- Diagnostic subgroups: defect classes and brightness/contrast/sharpness conditions; thresholds learned from development only. This is not demographic fairness analysis.
- Synthetic robustness: declared brightness, contrast, blur and noise changes; freeze severities before final evaluation.
- Efficiency, ethics and deployment discussion; no deployed application is required. Final submission must be one self-contained notebook. Map final submission coverage to the original assignment brief, which is available in the wider assignment context.

## Observations, risks and decisions

1. Image folders are generic `images` under nested `train` and `valid` containers. Parent-folder labels made all 1,800 records one spurious class. Parse the filename prefix, validate the numeric suffix and exact six-label vocabulary, and reject conflicting folder evidence. Publish outputs only after critical checks pass.
2. One byte- and pixel-identical Patches pair could leak across partitions. Retain the lexicographically first relative path (`patches_101.jpg`), exclude `patches_105.jpg` in the manifest and preserve every raw file. Pool: 1,799; development: 1,439; test: 360. Raw class balance is 300 each; modelling patches count is 299.
3. Both original source containers are pooled before creating the declared experiment. They do not define the new development/test split. Annotation files are not classifier inputs.
4. RGB storage contains greyscale pixels. Explicit one-channel conversion will occur in memory during modelling.
5. A fixed seed alone is insufficient to prevent silent resplitting after data/library changes. Persist manifest and checksum lock; verify and reuse assignments. Store environment versions. Shared CV folds are separately locked to the split checksum.
6. Audit EDA already inspected the complete collection, including eventual test images. Disclose that the test partition was reserved after descriptive EDA. Do not present it as prospectively untouched external validation. Make all subsequent model choices from development data only.
7. Exact duplicate checks do not establish independence of near-duplicates or common acquisition groups. Without acquisition identifiers, image-level stratification cannot demonstrate cross-factory generalisation.
8. The retained NEU-CLS.zip matches the Figshare version-1 file size and published MD5. All 3,602 archive files match the corresponding local files by SHA-256, and all 1,800 image hashes match the frozen modelling manifest. The locally computed archive SHA-256 and source metadata are recorded in data/NEU-CLS/Dataset-Information.txt. Figshare does not supply SHA-256 in the retrieved metadata; equivalence to the separately hosted original Northeastern University archive and the intervening repackaging history remain unverified.

## Submission and presentation evidence still needed

- Use the Figshare download URL and retained-archive SHA-256 recorded in data/NEU-CLS/Dataset-Information.txt for the final notebook download and extraction procedure. Safe extraction and empty-folder reproduction remain consolidation work; do not rerun the modelling notebooks during this documentation audit.
- Model-selection records and frozen settings; CNN early-stopping/epoch policy; fold-level metrics, timings and seeds.
- Final held-out comparison after both models are frozen. Do not retune after viewing test results.
- Explainability limitations: a plausible heatmap is not proof of causal feature use; HOG feature plots are not prediction attribution.
- Benchmark results cannot establish defect-versus-normal detection, other defect classes, other factories/cameras or production readiness. Keep human review in deployment discussion.
- Cite the dataset source and technical references in the final notebook/presentation; distinguish measured results from planned experiments. No fabricated results.


## Classical milestone: measured development results

Run: `03_hog_svm_20260923T231558_826155Z`. Six SVM candidates (linear/RBF, C 0.1/1/10; gamma=scale), five shared folds. Fixed native-resolution HOG (9 orientations, 16x16 cells, 2x2 blocks, L2-Hys), 4,356 features. HOG omits incomplete border cells. This is an explicit baseline design, not a tuned descriptor. StandardScaler is fitted within each fold. Macro F1 selects the model; exact ties follow the declared candidate order.

Selected: `{'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}`. Mean fold accuracy 0.888804; macro precision 0.891109; macro recall 0.888697; macro F1 0.887114 (fold SD 0.012463). These are selection CV scores, not nested-CV estimates or final test results. Fold SD is descriptive, not a confidence interval. Pooled OOF diagnostics reuse the folds used for configuration selection.

Selected pipeline refitted on all 1,439 development images and saved. Model serialization predictions were checked. Test images opened: 0. Preparation artifacts and development file hashes remained unchanged. No final test evaluation has occurred. Model comparison and final inference timing remain pending the CNN milestone.


# CNN development findings

Validated run: `04_cnn_20260925T221442_538853Z`. All five predefined development folds completed on CPU. No locked test image was opened or evaluated.

The compact CNN has 6,142 trainable parameters. Mean development accuracy was 0.9375; macro precision 0.9413; macro recall 0.9375; macro F1 0.9377, with fold SD 0.0284. The macro F1 difference relative to the recorded HOG-SVM development result was +0.0506. These are descriptive model-selection estimates. Both procedures used validation for selection, with different search budgets; the difference is not an independent final test result or a formal significance claim.

The weakest pooled validation recall was for pitted surface (0.8833). The largest directed confusion was scratches predicted as inclusion (23 images). This identifies a diagnostic focus, not a verified cause of error. Texture similarity, acquisition conditions and model capacity need separate investigation; the confusion matrix cannot distinguish them.

Selected fold epochs were [20, 15, 8, 23, 23]; completed epochs were [25, 20, 13, 25, 25]. 3 of five folds reached the epoch cap. The mean training-minus-validation macro F1 gap at selected epochs was -0.0072. The curve shapes and variation matter more than treating a single gap as proof of overfitting. A fold reaching the cap may have continued improving; a patience stop only describes this declared stopping rule. We preserved the declared budget rather than retuning after seeing results.

The final model was refitted on all 1,439 development images for 20 epochs, following the predeclared ceiling-of-median rule. Five fold checkpoints and this final checkpoint are preserved. The final file is 32.1 KiB. Five-fold wall time was 6.57 minutes, including diagnostics/checkpointing; final refit took 0.94 minutes. Warm single-image forward time was 1.402 ms on this recorded CPU/thread setup. This excludes decoding and normalisation and must not be presented as end-to-end deployment latency.

Captum LayerGradCam passed a finite-output synthetic compatibility test. This establishes computational compatibility only; it does not validate an explanation or show causal reliance on defect structure. The first strided convolution and pooling reduce resolution, and global averaging discards spatial arrangement. These choices trade capacity and localisation for a small, bounded CPU experiment.

Remaining limitations include validation-epoch selection optimism, one seed per fold, overlapping fold training sets, full-collection EDA before the split, unresolved acquisition-group/near-duplicate independence, and unverified equivalence to the original university archive. Fold SD reflects both partition and initialisation variation; these sources are not separately estimated. There is no normal-steel class. The locked test set remains reserved for the later frozen comparison.
