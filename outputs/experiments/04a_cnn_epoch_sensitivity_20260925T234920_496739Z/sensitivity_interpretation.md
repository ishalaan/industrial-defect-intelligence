# CNN epoch-cap sensitivity: measured interpretation

Baseline: 04_cnn_20260925T222923_890724Z. Sensitivity: 04a_cnn_epoch_sensitivity_20260925T234920_496739Z.

**Decision: retain 25; negligible improvement with natural validation plateaus.**

Mean macro F1 changed by -0.000000. 0 folds met the predeclared material-improvement-plus-truncation criterion; 5/5 stopped naturally before 40.

| Metric | 25 mean ± SD | 40 mean ± SD | Mean paired delta |
|---|---:|---:|---:|
| accuracy | 0.937464 ± 0.028826 | 0.937464 ± 0.028826 | +0.000000 |
| precision_macro | 0.941295 ± 0.026107 | 0.941295 ± 0.026107 | +0.000000 |
| recall_macro | 0.937500 ± 0.028842 | 0.937500 ± 0.028842 | +0.000000 |
| f1_macro | 0.937747 ± 0.028422 | 0.937747 ± 0.028422 | -0.000000 |

| Fold | Best epoch 25 / 40 | Epochs run 40 | Patience triggered | Hit 40 | Macro F1 40 | Delta F1 |
|---|---:|---:|---|---|---:|---:|
| 1 | 20 / 20 | 25 | True | False | 0.965152 | +0.000000 |
| 2 | 15 / 15 | 20 | True | False | 0.927876 | -0.000000 |
| 3 | 8 / 8 | 13 | True | False | 0.893120 | +0.000000 |
| 4 | 23 / 23 | 28 | True | False | 0.954774 | -0.000000 |
| 5 | 23 / 23 | 28 | True | False | 0.947815 | +0.000000 |

Historical CV wall time: 712.73s versus 703.23s; difference -9.50s. Measured computation after epoch 25: 39.89s across 6 extra epochs. Historical timing differences are confounded by machine load.

Common-prefix reproduction: True. Environment match: True. Zero test images opened, zero test predictions inspected, and no test evaluation. The audit hook permitted only development dataset reads; the synthetic prohibited request was rejected before access. The manifest supplied boundary metadata only. Evidence is scoped to this execution, not other processes or earlier EDA.

A patience stop indicates a local plateau under the fixed macro-F1 rule, not global convergence. Validation loss was recorded throughout and did not determine checkpoint selection. The same validation folds influence checkpoint and budget selection; these are descriptive development results with selection optimism. Fold SD is not an uncertainty interval. One seed per fold does not separate seed and partition variation. No significance or equivalence claim is made. No additional caps, hyperparameter changes or final test access followed this result.

The final budget specification preserves the original ceiling-of-median refit rule. No refit was performed here. All original and new results are retained, including adverse outcomes.

## Review of the measured trajectories

All 108 shared epochs reproduced within 1.12e-16, consistent with CSV floating-point round-trip differences. Displayed negative zero deltas are numerical round-off, not measured deterioration. Selected epochs remained [20, 15, 8, 23, 23]; stopping epochs were [25, 20, 13, 28, 28].

Fold 4: validation loss was 0.184140 at epoch 25 and 0.169792 at epoch 28; validation macro F1 was 0.944479 and 0.951457, respectively. Neither fold surpassed its epoch-23 selected macro F1 during the extension.

Fold 5: validation loss was 0.195320 at epoch 25 and 0.203196 at epoch 28; validation macro F1 was 0.944408 and 0.924919, respectively. Neither fold surpassed its epoch-23 selected macro F1 during the extension.

The loss curves do not establish a flat loss plateau: cross-entropy can improve without changing the class decisions that determine macro F1. The adequacy conclusion is therefore specific to the declared macro-F1 checkpoint and patience policy. It does not show that all optimisation had ceased, that later improvements were impossible, or that the CNN had reached its full potential. Keeping early stopping unchanged is essential to the controlled question.

The original run discussed in prior narrative had a different timestamp and runtime. This check uses the latest successful saved pointer, frozen before training, whose original CV runtime was 712.73 seconds. The new runtime was 703.23 seconds despite six extra epochs. This illustrates why historical wall-time subtraction cannot isolate extension cost; the within-run post-25 computation was 39.89 seconds, excluding checkpoint I/O. No faster-training claim follows.

The retained maximum cap is 25; the unchanged ceiling-of-median rule still gives 20 epochs for final development refitting. The existing final development checkpoint was not replaced. All 1,511 acceptance checks passed; all 14 code cells completed without error.
