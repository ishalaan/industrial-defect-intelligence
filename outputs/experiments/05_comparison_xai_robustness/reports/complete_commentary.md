# 05. Frozen model comparison, explainability and robustness

```text
venv\Scripts\python.exe -m pip install numpy pandas matplotlib pillow scikit-learn scikit-image joblib torch==2.14.0 torchvision==0.29.0 captum==0.9.0
```

Run from the project root using its existing virtual environment. The command above is the requested installation command, not an instruction to change an already working environment. Actual versions are recorded below. No model is fitted, calibrated or retuned in this notebook.

**Research question:** To what extent does learned representation improve six-class steel-surface defect classification over fixed HOG features, and how does that comparison change under image-condition stress and practical computation constraints?

This is the authorised final evaluation of the frozen models from notebooks 03 and 04, with the epoch decision verified against 04a. Both models receive the same 360 locked test images. Development contains 1,439 images and one duplicate remains excluded. Results apply to classification among six known defect categories. There is no normal or defect-free class.

**Evaluation boundary.** All modelling choices are frozen. Thresholds for diagnostic subgroups come from development images only. Perturbations, example-selection rules and reporting conventions are declared before test decoding. Reruns reuse integrity-checked saved clean predictions; they do not initiate another main evaluation. Stress tests, attribution and timing are separate analyses of the frozen models. Nothing observed here authorises model selection. A future changed model would require a new evaluation plan and independent data.

**Evidence limitations inherited from the project.** Descriptive EDA included the full collection before the split. This is therefore a model-selection holdout, not prospectively untouched external validation. Image-level splitting and exact duplicate removal do not establish independence of near-duplicates or acquisition groups. Local archive provenance remains unverified. The original assignment brief is absent; coverage follows the recorded methodology and the twelve substantive analysis requirements supplied for this stage.

## 1. Verify the evidence chain before test access

The latest pointers are checked against their result hashes. Model files are located by matching the recorded digest, rather than by assuming a checkpoint name. The split, shared folds, preparation status, model configuration, class order, normalisation and epoch decision must all agree. SHA-256 proves consistency with these local records, not external provenance. Earlier development records must explicitly report no test evaluation.

## 2. Predeclare analysis and check inference compatibility

Brightness is mean intensity, contrast is population intensity standard deviation, and sharpness is the variance of the four-neighbour Laplacian on interior pixels, all calculated in the original 0 to 255 scale as in EDA. Sharpness also responds to noise and texture. Each property uses global development tertiles: low is at or below the first threshold, middle is above the first and at or below the second, and high is above the second. These are diagnostic subgroups, not demographic fairness groups. Class composition is reported because it can confound apparent image-condition effects.

Stress severities are deliberately bounded engineering probes, without a claim that they reproduce a particular factory: Gaussian blur radius 0.5 pixels, brightness shifts of plus and minus 10/255, contrast factor 0.8 about each image mean, and Gaussian noise with standard deviation 5/255 and seed 20260927. Each is applied separately, clipped to [0,1], with the same resulting array supplied to both models. No severity is selected using performance. There is one noise realisation per image, so stochastic robustness uncertainty is not estimated.

Grad-CAM selection is the first path-sorted example in each joint correctness category, supplemented by the highest-softmax CNN error and the first CNN-correct example, with duplicate paths removed. This produces at most six illustrative cases and includes correct and incorrect CNN predictions whenever they exist. High-confidence means CNN maximum softmax at least 0.90, fixed before evaluation. This is a score threshold, not a calibrated reliability guarantee.

## 3. One-time clean evaluation on identical images

The clean evaluation creates an exclusive start marker and saves both models' outputs in one atomic cache. Once committed, subsequent runs verify and reuse it. If execution stops after the start marker but before committing the cache, the notebook deliberately stops for investigation rather than silently repeating the main evaluation. Decoding for later diagnostics is logged separately from model prediction. Metrics use the fixed six-class vocabulary and zero division equals zero.

The frozen CNN records accuracy **0.9556** and macro F1 **0.9555**, compared with **0.9028** and **0.9023** for HOG-SVM. The observed CNN minus SVM macro F1 difference is **+0.0532**. Each test class has 60 images, so macro recall and accuracy coincide here. Precision and F1 still respond differently to the allocation of false positives. These are results for this locked sample and these frozen implementations, not a universal comparison of handcrafted and learned representations.

For HOG-SVM, the lowest per-class recall is patches at 0.8000; ties are resolved alphabetically for this summary. The largest directed confusion is scratches predicted as inclusion (9 images; first class-order tie). A confusion identifies where review is needed, but it cannot establish whether texture ambiguity, labelling or acquisition conditions caused the error.

For CNN, the lowest per-class recall is pitted surface at 0.8167; ties are resolved alphabetically for this summary. The largest directed confusion is pitted surface predicted as inclusion (10 images; first class-order tie). A confusion identifies where review is needed, but it cannot establish whether texture ambiguity, labelling or acquisition conditions caused the error.

### Reading the clean comparison critically

The aggregate gain is not uniform across defect categories. CNN pitted-surface recall is 0.8167, compared with 0.8500 for HOG-SVM. The CNN confusion matrix records 10 pitted-surface images predicted as inclusion. Conversely, patches recall is 0.9833 for CNN versus 0.8000 for HOG-SVM. An application where confusing pitting with inclusion is particularly costly could judge this trade-off differently from macro F1. No cost weighting is invented or selected from these test results.

## 4. Paired disagreements and uncertainty

The two predictions are paired by image. Correctness overlap is more informative than two isolated accuracy values: it distinguishes shared failures from errors that differ between representations. The high-confidence error table uses the CNN's uncalibrated softmax score. The SVM was frozen with probability estimation disabled; its decision scores are not substituted for probabilities and no calibration is fitted after test access. See the [SVC documentation](https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html).

A predeclared paired stratified bootstrap resamples image indices within each true class, applying each resample to both models. Percentile intervals describe uncertainty conditional on this sample, class balance and frozen fitted models. Unknown acquisition dependence, model-selection variability and domain shift are not captured. The intervals are exploratory, without multiplicity adjustment or a universal superiority claim.

HOG-SVM alone is correct on **10** images, CNN alone on **29**, and both are wrong on **6**. Both are correct on **315**. There are **0 CNN errors with maximum softmax at least 0.90**. Such errors matter for automation bias: a confident display can still accompany a wrong class. Error complementarity motivates future investigation but is not evidence for an ensemble selected on this test set.

The image-level paired bootstrap 95% percentile interval for the CNN minus SVM macro F1 difference is **[+0.0198, +0.0865]**. Interpret this alongside the observed paired error counts. It does not include uncertainty from factories, production batches, repeated training or class prevalence.

## 5. CNN Grad-CAM and HOG representation visualisation

[Captum LayerGradCam](https://captum.ai/api/layer.html) attributes a selected output to a convolutional layer. Here the target is a class **logit**, the layer is `conv3`, and `relu_attributions=True` keeps positive contributions. Maps are 25 by 25 before bilinear upsampling to 200 by 200. Predicted-class and true-class targets are shown side by side, including when they coincide. Each panel is independently scaled to [0,1] for display; raw maps and extrema are saved, and colour intensity cannot be compared between panels. An all-zero positive map remains zero.

The selection rule deliberately spans error types rather than choosing visually persuasive maps. It is illustrative and outcome-conditioned, not representative sampling. No localisation masks, expert annotation, randomisation checks or causal intervention validate these explanations. Upsampling creates no new spatial evidence, and a plausible hotspot is not proof that the network recognises a physical defect. The original [Grad-CAM paper](https://arxiv.org/abs/1610.02391) describes the method; the limits of this implementation remain material.

HOG panels show local gradient structure used by the descriptor. They are **representation visualisations, not prediction attribution** for the nonlinear SVM. Incomplete border cells are omitted by the frozen descriptor.

The selection rule yields **5 unique images**, with **2 CNN-correct** and **3 CNN-incorrect** cases. Of the 10 target maps, **0** have no positive attribution. Predicted and true targets contextualise the same image; visual differences suggest questions for expert review rather than establishing why the model failed. Spatially broad responses can reflect distributed texture evidence or background sensitivity, which this analysis cannot separate.

### Interpretation of the displayed cases

In `crazing_100.jpg`, the predicted and true targets coincide and the maps are identical, as expected. Stronger responses occur in several regions of the textured image; the map does not trace a validated defect boundary. For `crazing_19.jpg`, the incorrect patches target has a broad positive response, while the crazing target is more concentrated on the right. This difference is target-dependent evidence, not proof that a particular feature caused the error.

For `patches_14.jpg`, the true patches map includes the prominent central dark region, while the predicted crazing map places relatively greater weight around it. The image is nevertheless misclassified. A visually plausible true-class map therefore does not imply that the corresponding class wins the model's decision. In `scratches_69.jpg`, both targets have broad responses and different emphasis near the vertical structure. The highest-softmax CNN error has a score of approximately 0.865, below the predeclared 0.90 threshold. There are no errors meeting that threshold in this sample; this does not establish calibration or justify a production confidence threshold.

Three of the five selected images are crazing examples because class-prefixed filenames affect the first-path rule. The rule avoids discretionary visual cherry-picking but sacrifices class coverage. These cases cannot characterise explanations across all six classes. HOG panels show gradient structure only; their appearance cannot explain the SVM's individual decisions.

## 6. Diagnostic subgroup analysis

The thresholds are fixed from development and applied unchanged to clean test properties. Reported macro F1 always averages the same six classes, including zero for undefined class scores, and the class counts accompany every subgroup. Small or missing classes can depress this metric and make comparisons unstable. Within-class recalls help distinguish composition effects from condition effects but do not remove confounding. Multiple overlapping exploratory comparisons are not causal tests or demographic fairness evidence.

The lowest observed diagnostic subgroup macro F1 for HOG-SVM is **0.4162** in **sharpness, low** (n=121). This identifies an audit priority, not a causal effect of that property. The accompanying class counts and within-class recalls are necessary context; brightness, contrast and sharpness can describe the defect itself as well as the imaging setup.

The lowest observed diagnostic subgroup macro F1 for CNN is **0.4335** in **sharpness, low** (n=121). This identifies an audit priority, not a causal effect of that property. The accompanying class counts and within-class recalls are necessary context; brightness, contrast and sharpness can describe the defect itself as well as the imaging setup.

### Class composition explains an apparent subgroup collapse

The low-sharpness group has 121 images: 60 inclusion, 20 pitted surface and 41 scratches, with no crazing, patches or rolled-in-scale examples. Its fixed-six-class macro F1 is 0.4335 for CNN and 0.4162 for HOG-SVM, while accuracy is 0.9091 and 0.8512, respectively. The three absent classes contribute zero to the declared macro calculation. Consequently, these low macro scores must not be presented as evidence of a comparable overall failure rate or as a causal effect of blur. Sharpness is strongly entangled with defect class in this collection. The class-specific recall table and subgroup sizes provide the more useful basis for targeted review.

## 7. Controlled robustness stress tests

Each predeclared condition transforms all 360 test images and preserves their labels. Models, scaler and CNN normalisation remain fixed. Delta means perturbed minus clean performance, with values on a 0 to 1 scale; multiplying by 100 gives percentage points. Negative values indicate degradation. Perturbed predictions and input array hashes are saved for every condition. These synthetic perturbations are stress tests, **not proof of performance under real factory shift**. No combined perturbations, severity sweep or new training is performed.

For HOG-SVM, the largest macro F1 degradation among the five declared stresses is **noise_sigma_5**: macro F1 **0.5168**, delta **-0.3855**, accuracy **0.5694**, delta **-0.3333**. These magnitudes depend on the chosen severities and one sample. An improvement under a perturbation would be descriptive, not permission to adopt that transform after test inspection.

For CNN, the largest macro F1 degradation among the five declared stresses is **noise_sigma_5**: macro F1 **0.5442**, delta **-0.4113**, accuracy **0.6167**, delta **-0.3389**. These magnitudes depend on the chosen severities and one sample. An improvement under a perturbation would be descriptive, not permission to adopt that transform after test inspection.

### The clean ranking does not hold under every stress

Under radius-0.5 blur, HOG-SVM macro F1 is 0.8668, exceeding CNN's 0.7936. Under contrast factor 0.8, HOG-SVM remains at 0.9023, while CNN falls to 0.8911. Stability under uniform contrast reduction is consistent with the normalised HOG representation, but this observation does not isolate the responsible mechanism.

CNN retains higher absolute macro F1 under the two brightness shifts and Gaussian noise. However, under noise its drop from clean performance is larger: -0.4113, versus -0.3855 for HOG-SVM. Neither model is robust to that declared noise condition in the sense of maintaining its clean performance. A numerically small pixel perturbation is not necessarily mild for model performance. These findings make camera focus, noise and contrast relevant priorities for a future factory-specific evaluation; they do not warrant retrofitting preprocessing using this test set.

## 8. Computation and deployment evidence

Both methods are benchmarked on this same CPU with four numerical-library threads where supported. Timing uses the first 32 development images, avoiding additional main test evaluation. Batch sizes 1 and 32 use three warm-ups and ten timed repetitions. Stages distinguish warm file read/decode, preprocessing, model-only inference and file-to-prediction end-to-end latency. CNN preprocessing includes scaling and frozen normalisation; SVM preprocessing includes scaling and HOG, with its fitted StandardScaler inside model inference. Each stage is measured independently, so its median need not sum to the end-to-end median. Throughput is batch size divided by median time, not sustained production throughput. The raw samples, library thread settings and hardware description are retained.

Warm filesystem caching, operating-system scheduling and sequential benchmarking limit precision. Timings exclude camera acquisition, queueing, network transfer, process startup and service orchestration. File size is serialisation size, not working memory or installed runtime size. No edge device, cloud API, GPU, energy use or production service is benchmarked.

HOG-SVM: median warm file-to-prediction latency is **29.352 ms** for one image, with **13.389 ms** independently measured preprocessing. Batch-32 end-to-end throughput is **39.7 images/s**. These local CPU measurements include preprocessing and decoding, but not an industrial acquisition pipeline.

CNN: median warm file-to-prediction latency is **7.036 ms** for one image, with **0.205 ms** independently measured preprocessing. Batch-32 end-to-end throughput is **350.7 images/s**. These local CPU measurements include preprocessing and decoding, but not an industrial acquisition pipeline.

The frozen SVM file occupies **24,503,366 bytes** and uses **4,356 HOG features**; the CNN checkpoint occupies **32,865 bytes** and has **6,142 parameters**. The CNN checkpoint is smaller, but its framework and runtime footprint are not measured. Serialisation formats differ, so the ratio is a storage observation rather than a general memory-efficiency result. Edge deployment would need target-device timing, memory and thermal measurements; a cloud or inference API would add transport latency, availability, access control and commercial-data governance requirements.

## Reproducibility and reading order

All stage-05 outputs live in `outputs/experiments/05_comparison_xai_robustness/`. Read `reports/predeclared_protocol.json`, `reports/result.json`, `tables/clean_metrics.csv`, `tables/per_class_metrics.csv`, `tables/robustness_metrics.csv` and `reports/evidence_interpretation.md` first. Tables, figures, raw attribution maps, prediction caches, timing samples, environment versions, source hashes and image-access logs are retained. The executed notebook and HTML report are archived alongside the execution log after the notebook completes.

The clean receipt guards the one-time main evaluation. Do not delete it or the start marker to obtain a fresh run. If code or upstream sources change, the protocol checks stop reuse for review. Saved predictions permit metric verification without reopening model selection. Installation versions are recorded for reproducibility; checkpoint and source integrity must still be verified on any other machine.

Technical references: [Captum LayerGradCam](https://captum.ai/api/layer.html), [Grad-CAM original paper](https://arxiv.org/abs/1610.02391), [scikit-learn SVC](https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html). Dataset provenance and the original assignment brief remain outstanding for final submission consolidation, as recorded in `METHODOLOGY.md`.

## 9. Ethics, monitoring and deployment implications

**Scope and consequences.** Every image in this benchmark is already assigned to a defect category. Neither model can establish that steel is defect-free, and accuracy here cannot be converted into defect detection sensitivity, false reject rates or safety assurance. Misclassification could send material to an unsuitable review or treatment route, increase scrap and rework, or delay recognition of a costly defect. The study does not quantify these operational costs or connect labels to engineering acceptance limits. A production system would require normal material, unknown defects and ambiguous examples, with domain experts defining the relevant consequences and acceptance criteria.

**Representativeness.** A balanced six-class archive does not reproduce factory prevalence or demonstrate robustness across cameras, steel grades, production lines, lighting or factories. Prior full-collection EDA, unverified archive provenance and unknown acquisition groups further constrain generalisation. The present subgroup results identify audit priorities; the synthetic stresses probe selected image transformations. Neither resolves external validity. A later prospective evaluation should be separated by time and acquisition group, include multiple sites and retain factory metadata where lawful and appropriate.

**Human oversight and automation bias.** Confident errors and shared model failures make an unreviewed automatic disposition rule unjustified. A pilot should assist trained inspectors, preserve the image and model version, allow correction and escalation, and make uncertainty and scope visible. A Grad-CAM overlay is not an assurance badge. This notebook does not derive a rejection threshold, select an ensemble or optimise a cost policy from the test set. Any such policy needs development or newly collected validation data and an independently assessed operating point.

**Privacy and commercial sensitivity.** Steel crops may contain little personal information, but factory captures and metadata can reveal process settings, product identifiers, timestamps, client specifications or proprietary defects. Incidental people or identifiers in a wider acquisition stream require minimisation. Control access, retention and export; preserve traceability without unnecessary identifying data. Local inference may reduce transfer requirements but does not eliminate security duties. Cloud or API inference needs agreed data handling, encryption, access controls, service continuity and an assessment of network latency and commercial confidentiality. No legal compliance or safety certification is claimed here.

**Monitoring and change control.** During a supervised pilot, monitor camera settings, image-property distributions, missing/corrupt inputs, class prediction frequencies, review disagreement and delayed expert-labelled per-class errors. Input drift alone does not prove performance decline, while stable input summaries do not guarantee correctness. Define escalation rules with domain stakeholders, retain an audit trail, and review performance across equipment, shifts and grades. Retraining requires versioned data, documented label review, development-only model selection and a fresh independent evaluation. Retire this test set as a model-selection resource now that results are visible.

## 10. Evidence-based comparative conclusion

**Observed facts.** On the identical 360-image locked test set, CNN macro F1 is 0.9555 and HOG-SVM macro F1 is 0.9023, a CNN minus SVM difference of +0.0532. CNN accuracy is 0.9556 and SVM accuracy is 0.9028. CNN has higher absolute macro F1 in 3 of the five predeclared stress conditions. Absolute stressed performance and loss relative to each model's clean baseline answer different questions and must be read together. The paired errors, per-class scores, diagnostic subgroups and timing tables qualify the headline comparison.

**Limitations.** This is one frozen compact CNN and one fixed HOG descriptor with a selected RBF SVM, using different development search procedures. It does not isolate architecture from optimisation or representation choices. Image-level bootstrap uncertainty excludes unknown acquisition dependence and external shift. Selected attribution examples are not validated explanations; uncalibrated scores are not correctness probabilities. No normal class, independent factory cohort, deployment service or operational harm model is evaluated.

**Deployment implication.** The measured comparison informs which frozen implementation merits further supervised assessment under the target factory's constraints. It does not establish universal superiority or production readiness. Preserve both models and this evaluation as evidence; use new, representative and independently assessed data for any subsequent model or operating-policy changes.