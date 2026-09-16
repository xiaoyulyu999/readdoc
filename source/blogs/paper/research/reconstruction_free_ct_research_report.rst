================================================================================
Explainable Reconstruction-Free Deep Learning for Diagnosis from CT Projection Data
================================================================================

:Author: Xiaoyu Lyu
:Document type: PhD-level research report and proposal
:Version: 1.0
:Date: 16 September 2026
:Keywords: computed tomography, sinogram, projection-domain learning,
           reconstruction-free diagnosis, explainable artificial intelligence,
           domain shift, intracranial haemorrhage

.. contents:: Table of Contents
   :depth: 3
   :local:

Executive Summary
=================

Conventional computed-tomography (CT) artificial intelligence follows a long
pipeline: X-ray measurements are reconstructed into human-readable images and
the images are then analysed by a second algorithm. This project investigates a
different proposition: whether a clinically useful and explainable diagnostic
decision can be produced directly from the projection measurements (commonly
represented as sinograms), without requiring image reconstruction in the
diagnostic branch.

The feasibility of this idea is supported by SinoNet, which demonstrated body-
region recognition and intracranial-haemorrhage (ICH) detection directly in
sinogram space [Lee2019]_, and by a later automated sinogram-based ICH system
[Sindhura2024]_. However, the evidence base remains narrow. Existing studies do
not establish whether performance survives changes in scanner manufacturer,
geometry, protocol and dose; whether projection-space explanations faithfully
identify the anatomical evidence used by a model; or whether any performance or
latency advantage persists under a rigorous patient-level comparison with an
image-domain system.

This report proposes a paired, multi-domain experimental study. A
physics-informed projection-domain network will be compared with an
image-domain baseline using the same patients, targets and data partitions. Its
projection-space attribution will be transformed into image coordinates by a
geometry-aware adjoint/backprojection operator. Explanation localisation,
faithfulness, stability and clinical usefulness will be evaluated separately;
a visually plausible heatmap will not be accepted as evidence of a faithful
explanation. External or manufacturer-held-out testing, calibration analysis,
decision-curve analysis and paired confidence intervals are built into the
design.

The expected contribution is not a claim that reconstructed CT images should be
removed from clinical care. Rather, it is a defensible dual-pathway model:

::

   CT projection measurements
           |-- diagnostic branch --> early AI prediction + calibrated uncertainty
           `-- reconstruction branch --> conventional CT images for human review

The diagnostic branch may support rapid triage, low-dose acquisition and
machine-oriented sensing, while the reconstruction branch preserves the image
needed by radiologists. The proposed PhD contribution is a validated framework
for determining when direct measurement-domain inference is accurate,
explainable and transportable enough to be clinically meaningful.

1. Background and Rationale
===========================

1.1 The conventional CT-to-AI pipeline
--------------------------------------

Let :math:`x` denote the unknown attenuation image, :math:`A_\theta` the CT
forward operator determined by scanner geometry :math:`\theta`, and :math:`y`
the measured projection data. A simplified measurement model is

.. math::

   y \sim \operatorname{Poisson}(I_0 \exp(-A_\theta x)) + \epsilon,

where :math:`I_0` is incident photon intensity and :math:`\epsilon` represents
electronic and modelling noise. After calibration and logarithmic
transformation, the line-integral data are commonly written as

.. math::

   p = A_\theta x + n.

The standard pipeline first estimates an image
:math:`\hat{x}=R_\theta(p)` by filtered backprojection (FBP), iterative
reconstruction or a learned reconstruction method, and then predicts
:math:`\hat{z}=f_\phi(\hat{x})`. Reconstruction is indispensable for human
interpretation, but it is not self-evident that :math:`R_\theta(p)` is the best
representation for every machine task. Reconstruction may suppress, distort or
re-weight information; it also introduces algorithm- and kernel-specific
variation.

The proposed diagnostic route learns

.. math::

   \hat{z}=g_\psi(p,\theta),

where scanner geometry is supplied explicitly or encoded through a
geometry-normalisation layer. This is *reconstruction-free diagnosis*, not
reconstruction-free clinical care.

1.2 Why projection-domain inference matters
-------------------------------------------

The approach has five plausible advantages:

* **Earlier triage:** prediction can begin before a full image series is
  reconstructed and transmitted.
* **Dose efficiency:** a task-specific model may remain useful with fewer views
  or lower photon counts, as suggested by sparse-view results in [Lee2019]_.
* **Information preservation:** the model accesses measurements before a
  reconstruction operator and display pipeline impose their priors.
* **Machine-oriented acquisition:** view selection and dose allocation could be
  optimised for a clinical task rather than only for visual image quality.
* **Scientific insight:** paired projection- and image-domain experiments can
  reveal which diagnostic signals survive or are altered by reconstruction.

These are hypotheses, not established clinical benefits. Latency must include
pre-processing and data transfer; dose claims require explicit simulation or
prospective acquisition; and any diagnostic advantage must be shown on held-out
patients and scanners.

1.3 Feasibility of the data
---------------------------

The Low-Dose CT Image and Projection Data collection contains 299 CT exams of
the head, chest and abdomen: 150 from Siemens systems and 149 from GE systems.
Routine-dose and simulated low-dose projections, reconstructed images and
pathology information are provided [Moen2021]_ [TCIA2020]_. A vendor-neutral
DICOM-CT-PD format and its validation were published earlier [Chen2015]_. This
resource makes a manufacturer-held-out experiment technically possible,
although label completeness and task-specific sample size must be audited
before a final disease target is fixed.

LoDoPaB-CT offers a large standardized low-dose parallel-beam benchmark
[Leuschner2021]_. It is useful for representation pretraining and controlled
noise experiments, but it is derived from reconstructed images and simulated
forward projection; it must not be represented as a substitute for native
clinical raw data.

2. Critical Review of the Literature
=====================================

2.1 Direct interpretation of CT projections
-------------------------------------------

Lee et al. introduced SinoNet and evaluated direct sinogram-space learning for
body-region identification and ICH detection [Lee2019]_. The study is important
because it established technical feasibility and reported favourable behaviour
under sparse sampling. Its scientific limitation is not that it failed, but
that feasibility on selected tasks does not establish transportability,
calibration or clinical utility.

Sindhura et al. subsequently proposed an automated sinogram-based deep-learning
pipeline for ICH detection and subtype classification [Sindhura2024]_. It
expanded the target from binary detection toward multi-label classification and
included activation-map analysis. However, an activation map in projection
coordinates is not inherently meaningful to a radiologist. A high-intensity
region in a sinogram can combine contributions from many anatomical locations,
so anatomical attribution requires an explicit mapping and validation strategy.

These papers establish a starting point, not a completed field. The published
evidence is concentrated on head CT and ICH, with limited independent validation
and limited study of scanner shift. There is no widely accepted benchmark for
the faithfulness of projection-domain explanations.

2.2 Learned reconstruction is related but not equivalent
---------------------------------------------------------

Learned reconstruction methods such as FBPConvNet [Jin2017]_, Learned
Primal-Dual [Adler2018]_ and AUTOMAP [Zhu2018]_ map measurements toward an image.
They show how data consistency and imaging physics can be integrated with deep
networks. Reviews of learned inverse problems also warn that impressive
in-distribution image metrics do not guarantee stability or diagnostic value
[Arridge2019]_ [Wang2020]_.

This project differs in its primary endpoint. A reconstruction system is
optimised to estimate :math:`x`; a reconstruction-free system is optimised for a
clinical target :math:`z`. Nevertheless, reconstruction research provides
essential architecture ideas: differentiable projectors, adjoint operators,
data-consistency constraints and geometry-aware representations.

2.3 Explainability in a non-human representation
-------------------------------------------------

Grad-CAM [Selvaraju2017]_ and Integrated Gradients [Sundararajan2017]_ are
widely used attribution methods. Their outputs are often visually persuasive,
but saliency is not automatically faithful. Sanity checks have shown that some
maps may be weakly dependent on trained parameters or data [Adebayo2018]_. In
clinical AI, explanation can create unwarranted confidence when it is evaluated
only as a picture [Ghassemi2021]_. Medical-imaging studies likewise show that
localisation quality and model performance are separate properties
[Arun2021]_.

Projection-domain XAI adds an inverse problem. If :math:`S(p)` is a
projection-space attribution map, a first anatomical visualisation is

.. math::

   H_x = \mathcal{N}\left(A_\theta^T(W \odot S(p))\right),

where :math:`A_\theta^T` is the geometry-matched adjoint/backprojection,
:math:`W` is an optional reliability or filtering weight, and
:math:`\mathcal{N}` normalises the resulting heatmap. This operation visualises
where attributed rays intersect, but it does **not** prove causal localisation.
It must be tested using projection-consistent perturbations, annotated lesions,
parameter randomisation and repeat acquisitions or controlled transforms.

2.4 Reliability and reporting
------------------------------

Discrimination alone is insufficient. Neural-network probabilities are often
miscalibrated [Guo2017]_; paired AUC comparisons should respect correlation
[DeLong1988]_; and imaging-AI reporting should state data provenance, patient-
level partitioning, missing labels and external testing [Mongan2020]_. The
proposed study therefore treats calibration, uncertainty, subgroup performance
and transparent reporting as primary design requirements.

3. Research Gap and Original Contribution
==========================================

The central gap is not simply “few people have classified sinograms.” The more
important unresolved question is:

   **Can a direct CT projection-domain model remain diagnostically accurate and
   calibrated across acquisition domains while producing explanations that are
   both faithful to the model and anatomically meaningful to clinicians?**

The proposed original contributions are:

#. a geometry-aware reconstruction-free diagnostic architecture evaluated on
   native clinical projection data;
#. a strictly paired comparison with image-domain and joint-domain baselines;
#. an anatomical explanation method that maps projection attribution through
   the scanner geometry rather than resizing a heatmap;
#. a validation protocol separating explanation localisation, faithfulness,
   stability and human usefulness;
#. manufacturer-, protocol- and dose-shift experiments with calibrated
   uncertainty; and
#. an open, reproducible benchmark specification and reporting checklist for
   reconstruction-free CT diagnosis.

4. Aim, Research Questions and Hypotheses
=========================================

4.1 Aim
-------

To develop and rigorously evaluate an explainable, geometry-aware deep-learning
framework for diagnosis directly from CT projection data, and to determine the
conditions under which it provides a clinically meaningful advantage over
reconstructed-image inference.

4.2 Research questions
----------------------

**RQ1.** Does projection-domain inference achieve non-inferior discrimination
and sensitivity to an image-domain model trained and tested on the same
patients?

**RQ2.** How does performance change under reduced dose, sparse views, altered
geometry and manufacturer shift?

**RQ3.** Can projection-space attribution be mapped to anatomically valid
locations, and is it more faithful or stable than image-domain saliency?

**RQ4.** Does a dual-domain or multi-task model improve robustness without
removing the latency or dose advantages of direct inference?

**RQ5.** Can calibrated uncertainty identify unsafe out-of-domain cases and
support a “defer to radiologist/reconstruction” policy?

4.3 Pre-specified hypotheses
----------------------------

* **H1 (non-inferiority):** the lower bound of the paired 95% confidence
  interval for :math:`AUC_{proj}-AUC_{img}` exceeds a clinically justified
  non-inferiority margin :math:`-\delta`.
* **H2 (sparse-view robustness):** projection-domain performance degrades less
  than image-domain performance when angular views are reduced.
* **H3 (explanation validity):** geometry-aware backprojected attribution has
  better lesion localisation and deletion/insertion faithfulness than naive
  sinogram heatmaps and chance controls.
* **H4 (domain shift):** explicit geometry conditioning and domain
  randomisation reduce the manufacturer-held-out performance and calibration
  gap.
* **H5 (selective prediction):** uncertainty-based deferral improves sensitivity
  and calibration among retained cases as coverage decreases.

The non-inferiority margin and minimum clinically acceptable sensitivity must be
set with radiological input before accessing the final test set.

5. Study Design
===============

5.1 Clinical task and staged scope
----------------------------------

The preferred first task is acute ICH triage in non-contrast head CT because it
is time-critical and has precedent in projection-domain research. Final task
selection is conditional on the native labels in the chosen dataset.

The work should be staged:

* **Stage A — controlled feasibility:** synthetic phantoms and forward-projected
  public CT images; verify geometry, perturbations and attribution mapping.
* **Stage B — native clinical projections:** train and internally validate on
  patient projection data.
* **Stage C — transportability:** hold out a manufacturer, scanner family,
  centre or time period.
* **Stage D — reader-centred evaluation:** assess whether anatomical
  explanations improve error detection or triage decisions.

5.2 Data sources
----------------

Primary candidate
^^^^^^^^^^^^^^^^^

**TCIA Low-Dose CT Image and Projection Data** [TCIA2020]_ [Moen2021]_

* native vendor-derived projection data in an open format;
* paired routine-dose and simulated lower-dose data;
* paired reconstructed images;
* GE and Siemens examinations;
* head, chest and abdominal protocols with pathology information.

Secondary sources
^^^^^^^^^^^^^^^^^

* **LoDoPaB-CT** for scalable representation pretraining and controlled
  low-dose experiments [Leuschner2021]_.
* **RSNA ICH data** for image-domain pretraining or label-model development,
  with explicit acknowledgement that native projections are unavailable
  [Flanders2020]_.
* digital phantoms with known lesion masks for tests in which exact anatomical
  ground truth is required.

5.3 Eligibility and unit of analysis
------------------------------------

Inclusion criteria should specify the examination type, available projection
geometry, interpretable reconstruction, diagnostic label and patient identity.
Exclusions should be recorded with reason codes. The primary unit of analysis
is the examination; slice- or view-level predictions are aggregated within an
examination. All data from a patient must remain in one partition. If multiple
series exist, the hierarchy must be retained in both sampling and confidence
intervals.

5.4 Data audit before model development
---------------------------------------

Before training, produce a locked data sheet covering:

* patient and examination counts rather than file counts;
* prevalence and subtype distribution;
* scanner manufacturer/model, geometry, kernel, dose and protocol;
* label source, label uncertainty and missingness;
* duplicate or near-duplicate series;
* demographic variables and subgroup coverage;
* corrupt projections, truncation, metal and motion artefacts; and
* the mapping between raw projections, reconstructed series and annotations.

A small dataset must not be made artificially large by random view- or
slice-level splitting.

5.5 Partition strategy
----------------------

Use three distinct levels:

#. **development split:** grouped, stratified patient-level training and
   validation;
#. **locked internal test:** untouched until the analysis plan is frozen; and
#. **external/domain test:** manufacturer-, scanner-, site- or temporal-held-out
   data.

Nested grouped cross-validation is appropriate for model selection if the
number of examinations is limited. The final confidence interval must be based
on independent patients, using cluster bootstrap when examinations or series
are repeated.

6. Proposed Models
==================

6.1 Baselines
-------------

The comparison must include:

**B0 — clinical/statistical baseline**
   prevalence-only and simple acquisition-feature models to expose leakage.

**B1 — image-domain baseline**
   FBP or supplied reconstruction followed by a 2-D/2.5-D CNN or transformer.

**B2 — projection-domain baseline**
   a reproducible SinoNet-like convolutional model.

**B3 — proposed projection model**
   geometry-aware encoder with view and detector positional information.

**B4 — dual-domain model**
   projection features fused with reconstructed-image features.

**B5 — multi-task model**
   direct diagnosis plus an auxiliary reconstruction, anatomy or consistency
   objective used during training; diagnostic inference need not generate a
   full image.

6.2 Geometry-aware projection encoder
-------------------------------------

For projection tensor :math:`p \in \mathbb{R}^{V\times D\times C}` with views
:math:`V`, detector elements :math:`D` and channels/slices :math:`C`, the model
will include:

* detector-local convolutions for short-range patterns;
* view-axis attention or sequence modelling for angular dependence;
* explicit angular and detector-coordinate embeddings;
* geometry metadata (source-to-detector distance, pitch, detector spacing and
  view angle) embedded as conditioning variables;
* masked training for missing or sparsely sampled views; and
* examination-level attention pooling with regularisation against single-view
  shortcuts.

An illustrative objective is

.. math::

   \mathcal{L} = \mathcal{L}_{cls}
   + \lambda_{con}\mathcal{L}_{consistency}
   + \lambda_{dom}\mathcal{L}_{domain}
   + \lambda_{cal}\mathcal{L}_{calibration}
   + \lambda_{aux}\mathcal{L}_{aux}.

Here, consistency compares predictions under physically valid view subsampling
and noise transformations; domain loss discourages reliance on manufacturer
identity; and the auxiliary term can predict anatomy or a low-resolution
reconstruction. Each term must be ablated.

6.3 Preventing shortcut learning
--------------------------------

Projection data contain acquisition signatures that may correlate with labels.
Controls should therefore include:

* prediction of scanner/manufacturer from learned embeddings;
* balanced sampling or inverse-probability weighting across domains;
* metadata-only baselines;
* adversarial or invariant feature learning;
* evaluation after removal of borders, tags and constant detector regions;
* protocol-stratified performance; and
* label permutation and negative-control tasks.

7. Explainability Framework
===========================

7.1 Explanation generation
--------------------------

At least two method families should be used:

* a gradient-based method (Integrated Gradients or input-times-gradient); and
* a perturbation-based method using angular wedges, detector bands or
  forward-projected anatomical masks.

Grad-CAM may be included for comparison but should not be the sole method.
Attribution is calculated in calibrated line-integral coordinates, not on a
display-normalised screenshot.

7.2 Geometry-aware anatomical mapping
-------------------------------------

Projection attribution is mapped to image space with the same geometry used by
the scan. Alternatives to compare are:

#. unfiltered adjoint backprojection;
#. filtered backprojection of signed attribution;
#. positive/negative attribution mapped separately;
#. ray-intersection accumulation normalised by exposure; and
#. optimisation of an image-space evidence map whose forward projection best
   matches the measured attribution.

Because backprojection smears evidence along rays, explanation resolution and
uncertainty must be reported. The map is an *anatomical rendering of model
evidence*, not a reconstructed lesion.

7.3 Four separate validation dimensions
-----------------------------------------

**Localisation**
   Dice/IoU where appropriate, point-to-mask distance, energy within lesion,
   pointing-game accuracy and free-response localisation. Threshold-free
   measures should accompany thresholded overlap.

**Faithfulness**
   deletion/insertion curves, comprehensiveness, sufficiency and prediction
   change after projection-consistent lesion removal/insertion. Masking arbitrary
   sinogram pixels is physically invalid and should be avoided.

**Stability**
   similarity under repeat reconstruction, noise resampling, small geometry
   perturbations, view subsampling and random initialisation.

**Human usefulness**
   blinded reader study measuring error detection, confidence calibration,
   localisation time and decision time with no explanation, naive sinogram
   explanation and anatomically mapped explanation.

7.4 Required sanity checks
--------------------------

* progressive model-parameter randomisation;
* label randomisation;
* random and centre-prior heatmap controls;
* comparison with lesion prevalence maps;
* explanation generated before versus after calibration;
* signed attribution inspection; and
* pre-defined failure examples, including metal and truncation artefacts.

8. Experimental Programme
=========================

8.1 Experiment 1: controlled proof of mechanism
------------------------------------------------

Use phantoms and forward-projected images with known lesions. Establish that the
model learns disease signal rather than geometry identifiers, and validate that
backprojected attribution recovers the known lesion region better than chance.

8.2 Experiment 2: paired diagnostic comparison
------------------------------------------------

Train B1--B5 on identical patient partitions. Compare examination-level AUC,
AUPRC, sensitivity at a fixed specificity, specificity at a fixed sensitivity,
log loss and Brier score. Pre-select the operating point on validation data.

8.3 Experiment 3: dose and view reduction
------------------------------------------

Evaluate routine dose and multiple lower-dose/view conditions. Noise simulation
must follow a documented measurement model. Plot performance, calibration and
latency against estimated dose or view count. Distinguish simulated from real
low-dose evidence.

8.4 Experiment 4: domain shift
------------------------------

Train on one manufacturer and test on another, then reverse the direction if
sample size permits. Repeat for protocol and dose strata. Compare standard
training, geometry conditioning, domain randomisation and invariant learning.

8.5 Experiment 5: explanation validation
------------------------------------------

Evaluate localisation, faithfulness and stability using a locked set with
lesion annotations. Do not select explanation parameters on the test set.

8.6 Experiment 6: selective prediction and workflow
----------------------------------------------------

Use deep ensembles or another validated uncertainty method. Report
risk--coverage curves and performance after deferral. Measure end-to-end
latency from available projection data to prediction, including decoding,
transfer and preprocessing; GPU kernel time alone is not clinically meaningful.

9. Statistical Analysis Plan
============================

9.1 Primary endpoint
--------------------

The primary endpoint is examination-level AUC for projection- versus
image-domain inference on the locked test set. The paired difference will be
estimated using DeLong's method [DeLong1988]_ or patient-level bootstrap. If the
study is framed as non-inferiority, :math:`\delta`, alpha and the decision rule
will be registered before test evaluation.

9.2 Secondary endpoints
-----------------------

* AUPRC because disease prevalence may be low;
* sensitivity and specificity with exact or bootstrap confidence intervals;
* calibration intercept/slope, Brier score and expected calibration error;
* decision-curve net benefit across clinically relevant thresholds;
* explanation localisation, faithfulness and stability;
* end-to-end latency, memory and computational cost; and
* subgroup and domain performance with interaction tests.

9.3 Sample-size principle
-------------------------

Sample size should be based on the number of positive and negative independent
examinations required to estimate the paired primary endpoint with the desired
precision or non-inferiority power. It must not be calculated from the number of
views or slices. If the available positive count is inadequate, the study
should be explicitly described as feasibility research and should emphasise
confidence intervals over thresholded significance.

9.4 Multiplicity and reproducibility
------------------------------------

One primary hypothesis is designated. Secondary comparisons will use false-
discovery-rate control or be labelled exploratory. Seeds, partitions,
preprocessing, geometry conversion, environment files and model checkpoints
will be versioned. Results will be reported across several seeds rather than
from the best run.

10. Ablation and Stress Tests
=============================

The minimum ablation matrix includes:

* geometry metadata on/off;
* view attention versus convolution only;
* diagnostic loss alone versus auxiliary physics/anatomy objectives;
* domain randomisation on/off;
* native projections versus simulated projections;
* routine dose versus reduced dose;
* attribution method and anatomical mapping method; and
* full model versus metadata-only and shuffled-label controls.

Stress tests include detector dropout, angle jitter, altered photon noise,
truncated fields of view, metal, motion, missing views and out-of-distribution
scanner settings. Robustness claims require clinically plausible perturbation
ranges.

11. Ethics, Governance and Clinical Safety
===========================================

The public datasets are de-identified, but the data-use terms and required
acknowledgements remain binding. Any local clinical extension requires ethical
approval, data-protection review and a data-management plan. Patient data should
be stored and processed under institutional controls with minimum necessary
access.

An ICH triage system is a high-risk medical AI use case. Under the EU AI Act and
medical-device framework, eventual deployment would require risk management,
traceability, human oversight, performance monitoring and quality-management
processes. A research prototype must be clearly separated from a diagnostic
device.

Key safety principles are:

* the model assists prioritisation and does not replace radiological review;
* low confidence or out-of-domain cases are deferred;
* the conventional reconstruction pathway remains available;
* failure modes are documented by scanner and subgroup;
* explanation is never presented as proof that the prediction is correct; and
* prospective impact must be tested before clinical adoption.

12. Expected Outcomes and Interpretation
========================================

Three scientifically valuable outcomes are possible.

**Outcome A — projection model is non-inferior and more robust at low dose.**
   This supports further work on early triage and task-oriented acquisition.

**Outcome B — in-domain performance is strong but cross-scanner performance
falls.**
   The main contribution becomes a precise characterisation of geometry and
manufacturer shift plus mitigation methods.

**Outcome C — image-domain inference remains superior.**
   This still answers an important question. Analysis may reveal that
   reconstruction supplies a beneficial inductive bias or that present data are
   insufficient for direct learning.

The project is therefore falsifiable: success is not defined as forcing the
projection model to win, but as producing reliable evidence about when direct
measurement-domain inference works and why.

13. Limitations and Risk Mitigation
===================================

.. list-table:: Principal risks and mitigations
   :header-rows: 1
   :widths: 26 34 40

   * - Risk
     - Consequence
     - Mitigation
   * - Limited labelled native projections
     - Overfitting and wide confidence intervals
     - Narrow the first target; self-supervised pretraining; report feasibility
       status; seek institutional collaboration.
   * - Manufacturer-specific formats
     - Non-portable preprocessing
     - Use DICOM-CT-PD; retain geometry metadata; validate forward/adjoint pairs.
   * - Simulated rather than native projections
     - Unrealistic performance
     - Separate simulated and native analyses; reserve native data for final
       validation.
   * - Confounding by protocol
     - Shortcut learning
     - Metadata baselines, stratification, held-out domains and adversarial tests.
   * - Attractive but unfaithful XAI
     - False clinician trust
     - Sanity checks, causal perturbations and independent localisation metrics.
   * - No external labels or lesion masks
     - Weak clinical conclusions
     - Expert re-annotation of a locked subset with inter-reader agreement.
   * - Compute and I/O cost
     - Infeasible 3-D experiments
     - Patch/view streaming, mixed precision, staged 2-D/2.5-D development and
       pre-specified resource budgets.

14. Work Packages and Timeline
==============================

.. list-table:: Indicative 36-month plan
   :header-rows: 1
   :widths: 12 23 45 20

   * - Months
     - Work package
     - Activities
     - Deliverable
   * - 1--4
     - WP1: Protocol and data audit
     - Systematic review, dataset agreements, geometry validation, analysis plan
     - Registered protocol and data sheet
   * - 5--9
     - WP2: Reproducible baselines
     - FBP/image baseline, SinoNet replication, patient-level partitions
     - Benchmark report
   * - 10--16
     - WP3: Geometry-aware model
     - Architecture, self-supervised pretraining, ablations
     - Methods paper
   * - 17--22
     - WP4: Explainability
     - Backprojected attribution, sanity checks, localisation and faithfulness
     - XAI paper and toolkit
   * - 23--27
     - WP5: Transportability
     - Manufacturer/dose/protocol shift, calibration, selective prediction
     - External-validation paper
   * - 28--31
     - WP6: Human evaluation
     - Reader-centred explanation and workflow study, subject to approvals
     - Clinical evaluation report
   * - 32--36
     - WP7: Synthesis
     - Thesis integration, release package, limitations and future work
     - Thesis and reproducibility archive

15. Proposed Thesis Structure
=============================

#. Introduction and clinical motivation
#. CT measurement physics and reconstruction
#. Systematic review of projection-domain diagnostic AI
#. Data standardisation and reproducible baselines
#. Geometry-aware reconstruction-free diagnosis
#. Anatomically mapped and validated projection-domain explanations
#. Domain shift, calibration and selective prediction
#. Clinical workflow evaluation
#. General discussion, limitations and translation roadmap

16. Publication Strategy
========================

The research naturally supports four manuscripts:

#. systematic or scoping review of reconstruction-free diagnostic imaging;
#. paired projection- versus image-domain benchmark;
#. geometry-aware XAI with faithfulness and localisation validation; and
#. cross-manufacturer robustness, calibration and selective prediction.

Target venues may include *Medical Image Analysis*, *IEEE Transactions on
Medical Imaging*, *Medical Physics*, *Radiology: Artificial Intelligence* and
MICCAI. Venue selection should follow the final contribution rather than be
fixed before results are known.

17. Reproducibility Checklist
=============================

* publish inclusion/exclusion flow and patient counts;
* release patient-grouped split identifiers where permitted;
* document every geometry conversion and unit;
* numerically test forward/adjoint consistency;
* publish preprocessing and dose-simulation code;
* report all model-selection criteria and seeds;
* lock test data and operating points;
* include calibration and uncertainty, not only AUC;
* report negative results and subgroup/domain failures;
* validate XAI with sanity checks and causal perturbations; and
* follow CLAIM reporting guidance [Mongan2020]_.

18. Conclusion
==============

Direct diagnosis from CT projection data is technically credible and clinically
interesting, but it has not yet earned broad clinical trust. The decisive PhD
problem is therefore not to build another sinogram classifier. It is to create
and test a system whose performance survives acquisition shift, whose
uncertainty supports safe deferral, and whose explanation can be mapped to
anatomy and shown to be faithful.

The most defensible deployment concept is a dual pathway: direct AI analysis for
early triage alongside conventional reconstruction for radiological review. A
carefully paired and externally tested study can determine whether this design
offers genuine diagnostic, dose or workflow value. Even a negative comparison
would be meaningful if it identifies the information, inductive bias or data
scale that reconstruction contributes.

References
==========

.. [Lee2019] Lee, H., Huang, C., Yune, S., Tajmir, S. H., Kim, M. and Do, S.
   (2019). Machine Friendly Machine Learning: Interpretation of Computed
   Tomography Without Image Reconstruction. *Scientific Reports*, 9, 15540.
   https://doi.org/10.1038/s41598-019-51779-5

.. [Sindhura2024] Sindhura, C., Al Fahim, M., Yalavarthy, P. K. and Gorthi, S.
   (2024). Fully automated sinogram-based deep learning model for detection and
   classification of intracranial hemorrhage. *Medical Physics*, 51,
   1944--1956. https://doi.org/10.1002/mp.16714

.. [Moen2021] Moen, T. R., Chen, B., Holmes, D. R. III, Duan, X., Yu, Z., Yu,
   L., Leng, S., Fletcher, J. G. and McCollough, C. H. (2021). Low-dose CT image
   and projection dataset. *Medical Physics*, 48(2), 902--911.
   https://doi.org/10.1002/mp.14594

.. [TCIA2020] McCollough, C. et al. (2020). Low Dose CT Image and Projection
   Data (LDCT-and-Projection-data), Version 6. The Cancer Imaging Archive.
   https://doi.org/10.7937/9NPB-2637

.. [Chen2015] Chen, B. et al. (2015). Technical Note: Development and
   validation of an open data format for CT projection data. *Medical Physics*,
   42(12). https://doi.org/10.1118/1.4935406

.. [Leuschner2021] Leuschner, J., Schmidt, M., Baguer, D. O. and Maass, P.
   (2021). LoDoPaB-CT, a benchmark dataset for low-dose computed tomography
   reconstruction. *Scientific Data*, 8, 109.
   https://doi.org/10.1038/s41597-021-00893-z

.. [Jin2017] Jin, K. H., McCann, M. T., Froustey, E. and Unser, M. (2017).
   Deep Convolutional Neural Network for Inverse Problems in Imaging. *IEEE
   Transactions on Image Processing*, 26(9), 4509--4522.
   https://doi.org/10.1109/TIP.2017.2713099

.. [Adler2018] Adler, J. and Öktem, O. (2018). Learned Primal-Dual
   Reconstruction. *IEEE Transactions on Medical Imaging*, 37(6), 1322--1332.
   https://doi.org/10.1109/TMI.2018.2799231

.. [Zhu2018] Zhu, B. et al. (2018). Image reconstruction by domain-transform
   manifold learning. *Nature*, 555, 487--492.
   https://doi.org/10.1038/nature25988

.. [Arridge2019] Arridge, S., Maass, P., Öktem, O. and Schönlieb, C.-B. (2019).
   Solving inverse problems using data-driven models. *Acta Numerica*, 28,
   1--174. https://doi.org/10.1017/S0962492919000059

.. [Wang2020] Wang, G., Ye, J. C. and De Man, B. (2020). Deep learning for
   tomographic image reconstruction. *Nature Machine Intelligence*, 2,
   737--748. https://doi.org/10.1038/s42256-020-00273-z

.. [Selvaraju2017] Selvaraju, R. R. et al. (2017). Grad-CAM: Visual
   Explanations from Deep Networks via Gradient-Based Localization. *IEEE
   International Conference on Computer Vision*, 618--626.
   https://doi.org/10.1109/ICCV.2017.74

.. [Sundararajan2017] Sundararajan, M., Taly, A. and Yan, Q. (2017). Axiomatic
   Attribution for Deep Networks. *Proceedings of ICML*, 3319--3328.
   https://proceedings.mlr.press/v70/sundararajan17a.html

.. [Adebayo2018] Adebayo, J. et al. (2018). Sanity Checks for Saliency Maps.
   *Advances in Neural Information Processing Systems*, 31.
   https://arxiv.org/abs/1810.03292

.. [Ghassemi2021] Ghassemi, M., Oakden-Rayner, L. and Beam, A. L. (2021). The
   false hope of current approaches to explainable artificial intelligence in
   health care. *The Lancet Digital Health*, 3(11), e745--e750.
   https://doi.org/10.1016/S2589-7500(21)00208-9

.. [Arun2021] Arun, N. et al. (2021). Assessing the Trustworthiness of Saliency
   Maps for Localizing Abnormalities in Medical Imaging. *Radiology: Artificial
   Intelligence*, 3(6), e200267. https://doi.org/10.1148/ryai.2021200267

.. [Guo2017] Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On
   Calibration of Modern Neural Networks. *Proceedings of ICML*, 1321--1330.
   https://proceedings.mlr.press/v70/guo17a.html

.. [DeLong1988] DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988).
   Comparing the areas under two or more correlated receiver operating
   characteristic curves: a nonparametric approach. *Biometrics*, 44(3),
   837--845. https://doi.org/10.2307/2531595

.. [Mongan2020] Mongan, J., Moy, L. and Kahn, C. E. Jr. (2020). Checklist for
   Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and
   Reviewers. *Radiology: Artificial Intelligence*, 2(2), e200029.
   https://doi.org/10.1148/ryai.2020200029

.. [Flanders2020] Flanders, A. E. et al. (2020). Construction of a Machine
   Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage
   Challenge. *Radiology: Artificial Intelligence*, 2(3), e190211.
   https://doi.org/10.1148/ryai.2020190211

Online Resources
================

* TCIA collection page: https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/
* Mayo CT Clinical Innovation Center resources:
  https://www.mayo.edu/research/centers-programs/ct-clinical-innovation-center/resources
* EU AI Act overview:
  https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai

