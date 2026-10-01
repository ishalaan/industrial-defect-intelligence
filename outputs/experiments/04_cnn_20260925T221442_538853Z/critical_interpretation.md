# CNN development findings

Validated run: `04_cnn_20260925T221442_538853Z`. All five predefined development folds completed on CPU. No locked test image was opened or evaluated.

The compact CNN has 6,142 trainable parameters. Mean development accuracy was 0.9375; macro precision 0.9413; macro recall 0.9375; macro F1 0.9377, with fold SD 0.0284. The macro F1 difference relative to the recorded HOG-SVM development result was +0.0506. These are descriptive model-selection estimates. Both procedures used validation for selection, with different search budgets; the difference is not an independent final test result or a formal significance claim.

The weakest pooled validation recall was for pitted surface (0.8833). The largest directed confusion was scratches predicted as inclusion (23 images). This identifies a diagnostic focus, not a verified cause of error. Texture similarity, acquisition conditions and model capacity need separate investigation; the confusion matrix cannot distinguish them.

Selected fold epochs were [20, 15, 8, 23, 23]; completed epochs were [25, 20, 13, 25, 25]. 3 of five folds reached the epoch cap. The mean training-minus-validation macro F1 gap at selected epochs was -0.0072. The curve shapes and variation matter more than treating a single gap as proof of overfitting. A fold reaching the cap may have continued improving; a patience stop only describes this declared stopping rule. We preserved the declared budget rather than retuning after seeing results.

The final model was refitted on all 1,439 development images for 20 epochs, following the predeclared ceiling-of-median rule. Five fold checkpoints and this final checkpoint are preserved. The final file is 32.1 KiB. Five-fold wall time was 6.57 minutes, including diagnostics/checkpointing; final refit took 0.94 minutes. Warm single-image forward time was 1.402 ms on this recorded CPU/thread setup. This excludes decoding and normalisation and must not be presented as end-to-end deployment latency.

Captum LayerGradCam passed a finite-output synthetic compatibility test. This establishes computational compatibility only; it does not validate an explanation or show causal reliance on defect structure. The first strided convolution and pooling reduce resolution, and global averaging discards spatial arrangement. These choices trade capacity and localisation for a small, bounded CPU experiment.

Remaining limitations include validation-epoch selection optimism, one seed per fold, overlapping fold training sets, full-collection EDA before the split, unresolved acquisition-group/near-duplicate independence, and unverified local archive provenance. Fold SD reflects both partition and initialisation variation; these sources are not separately estimated. There is no normal-steel class. The locked test set remains reserved for the later frozen comparison.
