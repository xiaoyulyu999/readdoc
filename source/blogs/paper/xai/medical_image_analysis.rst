Explainable AI in Deep Learning-Based Medical Image Analysis
============================================================

Paper Information
-----------------

:Title: Explainable Artificial Intelligence (XAI) in Deep Learning-Based Medical Image Analysis
:Authors: Bas H.M. van der Velden, Hugo J. Kuijf, Kenneth G.A. Gilhuijs, Max A. Viergever
:Journal: Medical Image Analysis
:Volume: 79
:Year: 2022
:Article: 102470
:Topic: Explainable AI / Medical Image Analysis
:Type: Survey
:Status: Reading
:PDF: `Google Drive <https://drive.google.com/file/d/1DzPekiL5HvMvvRk3au9uih4oiXmKtPfS/view?usp=sharing>`_
:DOI: `10.1016/j.media.2022.102470 <https://doi.org/10.1016/j.media.2022.102470>`_


Purpose of Reading
------------------

This paper provides a systematic overview of Explainable Artificial
Intelligence (XAI) methods used in deep learning-based medical image
analysis.

The main purposes of reading this paper are:

* Understand the fundamental taxonomy of XAI.
* Understand how XAI methods are applied to medical images.
* Compare visual, textual, and example-based explanations.
* Understand the strengths and limitations of common XAI techniques.
* Learn how XAI explanations should be evaluated.
* Identify research gaps relevant to explainable chest X-ray classification.
* Build a theoretical foundation for future postgraduate and PhD research.


Key Concepts
------------

The paper organizes XAI methods along three fundamental dimensions:

1. Model-based vs. Post-hoc
2. Model-specific vs. Model-agnostic
3. Global vs. Local

These dimensions answer three different questions:

* **When is explainability introduced?**
* **Which models can the explanation method work with?**
* **What is the scope of the explanation?**


Reading Notes
-------------

2. XAI Framework
~~~~~~~~~~~~~~~~


2.1 Model-based vs. Post-hoc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model-based Explanation
"""""""""""""""""""""""

Model-based explanation makes interpretability part of the model itself.

Traditional interpretable models include methods such as linear models
and sparse models, where the relationship between input features and
predictions can be inspected relatively easily.

For deep neural networks, intrinsic interpretability is considerably
more difficult because the model may contain thousands or millions of
parameters.

Key idea::

    Interpretability is part of the model design.

Advantages:

* Explanation is directly related to the model.
* Potentially avoids explaining a separate black-box model.

Limitations:

* Difficult to achieve with complex deep neural networks.
* Interpretability constraints may restrict model design.


Post-hoc Explanation
""""""""""""""""""""

Post-hoc explanation is performed after a model has already been trained.

Instead of changing the original model, an explanation method attempts
to understand why the trained model produced a particular prediction.

Examples include:

* Grad-CAM
* Saliency maps
* SHAP
* LIME
* Occlusion sensitivity

Key idea::

    Train first -> Explain afterwards

This is particularly important in deep learning because an existing
high-performance neural network can be analysed without redesigning
the entire model.


2.2 Model-specific vs. Model-agnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model-specific Explanation
"""""""""""""""""""""""""""

Model-specific methods depend on the internal structure of a particular
type of model.

For example, some explanation techniques use convolutional feature maps
or gradients inside a CNN.

Examples:

* CAM
* Grad-CAM
* Guided backpropagation

Advantages:

* Can exploit internal model information.
* Often computationally efficient.

Limitations:

* Cannot necessarily be transferred to a different model architecture.


Model-agnostic Explanation
""""""""""""""""""""""""""

Model-agnostic methods do not require detailed knowledge of the internal
architecture of the model.

They mainly investigate the relationship between:

    Input -> Model -> Output

A common strategy is to modify or perturb the input and observe how the
prediction changes.

Examples:

* LIME
* Occlusion-based methods
* SHAP in its general formulation

Advantages:

* Can potentially be applied to different model architectures.
* Useful when the internal model is inaccessible.

Limitations:

* Perturbation can be computationally expensive.
* Artificial perturbations may not represent realistic medical images.


2.3 Global vs. Local
^^^^^^^^^^^^^^^^^^^^


Global Explanation
""""""""""""""""""

Global explanations attempt to understand the overall behaviour of a
model across a dataset.

Typical question::

    What has the model learned in general?

Examples include:

* Dataset-level feature importance
* Analysis of learned representations
* Visualization of learned filters

Global explanation is useful for understanding systematic behaviour,
bias, and general patterns learned by the model.


Local Explanation
"""""""""""""""""

Local explanations explain an individual prediction.

Typical question::

    Why did the model make this prediction for this patient?

For medical imaging, a saliency map highlighting the region responsible
for predicting a disease is an example of local explanation.

For chest X-ray classification::


    Chest X-ray
        |
        v
    Neural Network
        |
        +----> Disease prediction
        |
        +----> Local explanation / heatmap


3. XAI in Medical Image Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper divides XAI methods in medical imaging into three broad groups:

1. Visual explanation
2. Textual explanation
3. Example-based explanation


3.1 Visual Explanation
^^^^^^^^^^^^^^^^^^^^^^

Visual explanation is the most widely used form of XAI in medical image
analysis.

The objective is usually to identify image regions that contributed to
a model prediction.

Typical output::

    Medical Image
          |
          v
         CNN
          |
          v
      Prediction
          |
          v
     Saliency Map


Backpropagation-based Methods
"""""""""""""""""""""""""""""

These methods use information propagated through the neural network to
identify important regions.

Important techniques include:

* Backpropagation
* Guided backpropagation
* CAM
* Grad-CAM
* Layer-wise Relevance Propagation (LRP)
* Deep SHAP


CAM
"""

Class Activation Mapping (CAM) generates localization maps from CNN
feature maps.

A major limitation is that the architecture generally requires global
average pooling.


Grad-CAM
""""""""

Grad-CAM generalizes the idea of CAM by using gradients flowing into
convolutional feature maps.

Conceptually::

    CNN prediction
         |
         v
    Calculate gradients
         |
         v
    Weight feature maps
         |
         v
    Generate heatmap

Grad-CAM is particularly attractive for medical imaging because it can
provide a visual explanation of which anatomical regions contributed
to a prediction.


Perturbation-based Methods
""""""""""""""""""""""""""

Instead of using internal gradients, perturbation-based approaches modify
parts of the input and observe how the prediction changes.

Important examples include:

* Occlusion sensitivity
* LIME
* Meaningful perturbation
* Prediction difference analysis

General idea::

    Original Image -> Prediction

    Modify region
          |
          v
    New prediction

          |
          v

    Difference = importance of region


LIME
""""

LIME approximates the behaviour of a complex model locally using a
simpler interpretable model.

For images, regions or superpixels can be perturbed to determine which
parts of the image influence the prediction.


Medical Imaging Problem
"""""""""""""""""""""""

Perturbation requires particular care in medical imaging.

Simply replacing part of an image with noise or a constant value can
create medically unrealistic images.

Therefore, meaningful perturbations should ideally preserve realistic
anatomical structure.


3.2 Textual Explanation
^^^^^^^^^^^^^^^^^^^^^^^

Textual explanation describes model decisions using human-readable
concepts or language.

Examples include:

* Image captioning
* Radiology report generation
* Image captioning with visual explanation
* Testing with Concept Activation Vectors (TCAV)

For medical imaging, a system could potentially produce:

.. code-block:: R

    Chest X-ray
         |
         v
       Model
         |
         +----> Disease prediction
         |
         +----> Visual localization
         |
         +----> Textual explanation

For example::

    "Opacity is present in the lower right lung."

This type of explanation may be closer to the way clinicians communicate
than a heatmap alone.


TCAV
""""

Testing with Concept Activation Vectors (TCAV) explains neural network
behaviour using human-understandable concepts.

Instead of asking which pixels are important, TCAV can investigate
whether concepts meaningful to humans influence the prediction.

This represents an important shift from:

    pixel-level explanation

towards:

    concept-level explanation

.. note::

   Shen et al. (2019) used what they called a hierarchical seman-
   tic CNN to predict malignancy of lung nodules on CT. They clas-
   sified five textual descriptions of image characteristics represen-
   tative of lung nodule malignancy that are typically assessed by a

   radiologist. The task of finding textual descriptions was combined
   with the main task of classifying lung nodule malignancy. Although
   their hierarchical semantic CNN did not significantly outperform a

   normal CNN in predicting nodule malignancy, the method did pro-
   vide human-interpretable characteristics of the nodules.

XAI 的目标不一定是提高 accuracy，而是让模型的预测过程更容易被人理解。

普通 CNN：

.. code-block:: R

   CT nodule
    ↓
   CNN
    ↓
   Malignant / Benign

而 hierarchical semantic CNN 更像：

.. code-block:: R

                  CT Nodule
                     ↓
                    CNN
                     ↓
          Learned Representation
              ↙              ↘
             ↓                ↓
      Semantic features      Malignancy
                ↓             prediction
        characteristic 1
        characteristic 2
        characteristic 3
        characteristic 4
        characteristic 5

也就是说，它不仅回答：

“Is this nodule malignant?”

还试图告诉医生：

“What characteristics does this nodule have?”

.. important::

   “the method did provide human-interpretable characteristics of the nodules.”

这些 characteristics 是放射科医生本身会评估的影像学特征，所以模型输出和医生熟悉的医学概念之间建立了联系。

3.3 Example-based Explanation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example-based explanations explain a prediction by presenting similar
examples.

This resembles human reasoning in medicine.

A clinician may reason::

    "This case resembles previous patients with the same condition."

Typical approaches include:

* Triplet networks
* Influence functions
* Prototype-based models
* Latent-space similarity

General idea::

    New patient
        |
        v
    Learned representation
        |
        v
    Similar historical cases
        |
        v
    Explanation


4. Comparing XAI Methods
~~~~~~~~~~~~~~~~~~~~~~~~

The paper highlights that explanation methods should not only be compared
by visual appearance.

Important practical considerations include:

* Explanation quality
* Robustness
* Computational cost
* Need for parameter tuning
* Model dependency
* Open-source availability

Backpropagation-based methods are generally computationally cheaper than
perturbation-based methods because perturbation methods require repeated
model predictions.


5. Evaluation of XAI
~~~~~~~~~~~~~~~~~~~~

One of the most important lessons from this paper is that producing an
explanation is not sufficient.

The explanation itself must be evaluated.

Three broad evaluation strategies are relevant.


Application-grounded Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real domain experts evaluate explanations in a real or realistic task.

For medical AI, this could involve radiologists assessing whether an
explanation improves clinical decision making.

This provides strong clinical relevance but can be expensive and
difficult to conduct.


Human-grounded Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Human participants perform simplified evaluation tasks.

This can reduce the cost compared with using highly trained medical
experts, but the evaluation becomes an approximation of real clinical
use.


Functionally-grounded Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The explanation is evaluated using quantitative proxies rather than
direct human evaluation.

For visual explanations, a possible approach is to compare a generated
heatmap with expert-annotated disease regions.

Possible metrics include::

    Explanation map
          vs.
    Expert annotation

This allows repeated quantitative evaluation, although expert
annotations themselves can be expensive to obtain.


Critical Review
---------------

Strengths
~~~~~~~~~

**1. Clear taxonomy**

The three-dimensional framework provides a useful conceptual structure
for understanding XAI:

* model-based / post-hoc
* model-specific / model-agnostic
* global / local

This makes a complicated research area easier to analyse systematically.


**2. Medical-imaging focus**

The paper does not simply summarize general computer-vision XAI methods.
It discusses their application and adaptation to medical imaging.


**3. Broad coverage**

The survey covers a large body of research across different anatomical
locations and imaging modalities.


**4. Goes beyond saliency maps**

The paper includes:

* visual explanation
* textual explanation
* example-based explanation
* evaluation
* criticism of XAI
* future research opportunities

This provides a broader understanding of explainability than simply
using Grad-CAM.


**5. Strong foundation for postgraduate research**

The paper provides useful terminology and taxonomy for structuring a
literature review and comparing different XAI methods.


Limitations
~~~~~~~~~~~

**1. Literature coverage has a time boundary**

The survey includes papers only up to October 2020.

Therefore, it should be treated as a foundation rather than a complete
description of the current XAI landscape.


**2. No single XAI framework is universally accepted**

The proposed framework is derived from existing XAI taxonomies, but
alternative frameworks exist.


**3. XAI evaluation remains difficult**

A visually convincing heatmap is not necessarily a faithful explanation
of model behaviour.

Different explanation methods may also produce different explanations
for the same prediction.


**4. Clinical usefulness is not guaranteed**

An explanation useful to a machine-learning researcher may not provide
the information required by a radiologist or another clinician.

Clinical explanations may need to incorporate additional information
such as patient history, treatment and expected outcomes.


**5. Robustness remains an open issue**

Research comparing explanation techniques has produced conflicting
results.

Therefore, explanation reliability should itself be investigated rather
than assumed.


Research Gaps
-------------

The paper suggests several important research directions.


1. Rigorous Evaluation of Explanations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many studies generate explanations but do not rigorously evaluate them.

A stronger research question is therefore not simply::

    Can Grad-CAM generate a heatmap?

but::

    Is the Grad-CAM explanation reliable, stable and clinically meaningful?


2. Explanation Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~

Different models can achieve similar predictive performance while
producing different explanations.

This motivates research into::

    explanation consistency
    explanation stability
    explanation robustness


3. Domain Knowledge
~~~~~~~~~~~~~~~~~~~

Future medical XAI should increasingly incorporate medical domain
knowledge.

This requires collaboration between:

* AI researchers
* medical imaging researchers
* clinicians
* domain experts


4. Bias and Causality
~~~~~~~~~~~~~~~~~~~~~

A model may obtain high predictive performance because of dataset bias
rather than true disease-related features.

XAI can potentially help identify these shortcuts.

A future direction is therefore to connect:

    Explainability + Bias Detection + Causal Reasoning


5. Appropriate Sample Size
~~~~~~~~~~~~~~~~~~~~~~~~~~

There is no established consensus about the minimum sample size required
for different XAI techniques in medical imaging.

This represents an important methodological research problem.


6. Multi-modal Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Future systems may benefit from combining several explanation forms:

    Visual explanation
            +
    Textual explanation
            +
    Example-based explanation
            |
            v
    More comprehensive explanation


Relevance to My Research
------------------------

This paper provides a theoretical foundation for research on explainable
deep learning for chest X-ray classification.

A simple project could be::

    Chest X-ray
         |
         v
    CNN / Transformer
         |
         v
    Disease Classification
         |
         v
      Grad-CAM

However, this would mainly demonstrate an existing XAI technique.

A stronger postgraduate research design would investigate the quality
of explanations rather than merely generate them.


Potential Research Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A possible research pipeline is::

                  Chest X-ray
                       |
             +---------+---------+
             |                   |
             v                   v
            CNN              Transformer
         DenseNet             ViT / Swin
             |                   |
             +---------+---------+
                       |
                       v
                Disease Prediction
                       |
             +---------+---------+
             |         |         |
             v         v         v
          Grad-CAM    SHAP    Attention
             |         |         |
             +---------+---------+
                       |
                       v
              Explanation Evaluation
                       |
          +------------+------------+
          |            |            |
          v            v            v
      Stability    Localization   Confidence
          |            |            |
          +------------+------------+
                       |
                       v
             Clinically meaningful XAI


Possible Research Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RQ1**

How consistent are visual explanations generated by different deep
learning architectures for multi-label chest X-ray classification?


**RQ2**

Do CNN-based and Transformer-based models focus on similar anatomical
regions when predicting the same pathology?


**RQ3**

How stable are XAI explanations when the input image is subjected to
small clinically irrelevant perturbations?


**RQ4**

Is model confidence associated with explanation consistency?


**RQ5**

Do explanation methods reliably localize clinically relevant regions
rather than dataset-specific artefacts?


From Master's Project to PhD
----------------------------

A possible progression of this research direction is::

    Stage 1 -- Master's
    ------------------
    Chest X-ray classification
            +
    CNN / Transformer comparison
            +
    Grad-CAM / attention visualization


                    |
                    v


    Stage 2 -- Strong Dissertation
    ------------------------------
    Explanation evaluation
            +
    Stability / consistency
            +
    Disease-specific analysis
            +
    Confidence analysis


                    |
                    v


    Stage 3 -- PhD Research
    -----------------------
    Robust and clinically meaningful XAI
            +
    Bias / shortcut detection
            +
    Causal explanations
            +
    Multi-modal explanations
            +
    Clinical expert evaluation
            +
    External dataset validation


Take-away
---------

The central lesson from this paper is:

    Generating an explanation is not the same as validating an explanation.

For postgraduate research, knowing how to apply Grad-CAM is useful.

For PhD-level research, the more important questions are:

* Is the explanation faithful?
* Is it stable?
* Is it reproducible?
* Is it clinically meaningful?
* Does it generalize across datasets and models?
* Can it expose bias or shortcut learning?