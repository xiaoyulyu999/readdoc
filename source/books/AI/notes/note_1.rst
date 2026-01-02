How Cross-Entropy Loss Works in Logistic Regression
===================================================

This document explains, step by step, how each training sample
:math:`(x_i, y_i)` affects the computation of logistic regression
and the resulting cross-entropy loss. Each step highlights the role of
the input features :math:`x_i` and the label :math:`y_i`.

--------------------------------------------------

Step 1 — Compute Model Prediction from :math:`x_i`
--------------------------------------------------

The logistic regression model computes:

.. math::

   \hat y_i = \sigma(w^T x_i + b)

where:

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}

**Role of :math:`x_i`:**

* :math:`x_i` is the feature vector representing the input sample.
* The linear combination :math:`w^T x_i + b` computes a raw score for the
  sample based on its features.
* The sigmoid function converts this raw score into a probability
  :math:`\hat y_i` for the positive class.

**Intuition:**
:math:`x_i` determines how confident the model is that this sample belongs
to the positive class. Different features in :math:`x_i` increase or decrease
the predicted probability.

**Role of :math:`y_i`:**
At this step, :math:`y_i` is not yet used. This is purely the model’s
prediction based on :math:`x_i`.

--------------------------------------------------

Step 2 — Compute Probability of Observed Label :math:`y_i`
--------------------------------------------------

Using the predicted probability :math:`\hat y_i`, we compute the likelihood
of the actual label :math:`y_i` with the Bernoulli formula:

.. math::

   P(y_i \mid x_i) = \hat y_i^{y_i} (1 - \hat y_i)^{1 - y_i}

**Role of :math:`y_i`:**

* :math:`y_i` acts as a selector:
  - If :math:`y_i = 1`, the probability is :math:`\hat y_i`.
  - If :math:`y_i = 0`, the probability is :math:`1 - \hat y_i`.
* This ensures that we only “care” about the probability assigned to the
  correct class.

**Role of :math:`x_i`:**

* :math:`x_i` indirectly affects this probability because it determined
  :math:`\hat y_i` in Step 1.
* If :math:`x_i` is far from typical positive examples, :math:`\hat y_i`
  will be low, reducing the likelihood for a positive label.

**Intuition:**
The model is now “scored” based on how well its prediction
(:math:`\hat y_i`) matches the true label (:math:`y_i`). Each sample
contributes a probability based on its features and label.

--------------------------------------------------

Step 3 — Compute Likelihood for All Samples
--------------------------------------------------

Assuming samples are independent:

.. math::

   L(w,b) = \prod_{i=1}^{n} P(y_i \mid x_i)

**Role of :math:`y_i`:**

* Each label :math:`y_i` selects the correct term in the product for its
  sample.
* Misclassified samples drastically reduce the total likelihood.

**Role of :math:`x_i`:**

* :math:`x_i` affects how large :math:`P(y_i \mid x_i)` is.
* Samples with features that are hard to classify contribute more strongly
  to the likelihood penalty if the model predicts incorrectly.

**Intuition:**
The model’s total “score” is determined by how well it predicted the
correct label for each sample, considering the features of each sample.

--------------------------------------------------

Step 4 — Take Log to Get Log-Likelihood
--------------------------------------------------

.. math::

   \ln L = \sum_{i=1}^{n} \Big[y_i \ln \hat y_i + (1 - y_i) \ln (1 - \hat y_i)\Big]

**Purpose of log:**

* Converts the product of probabilities into a sum for numerical stability.
* Each sample’s contribution is now additive, making optimization easier.

**Role of :math:`y_i`:**

* :math:`y_i` still selects which term contributes to the sum.
* If :math:`y_i = 1`, the term is :math:`\ln \hat y_i`.
* If :math:`y_i = 0`, the term is :math:`\ln (1 - \hat y_i)`.

**Role of :math:`x_i`:**

* :math:`x_i` influences :math:`\hat y_i`, which in turn determines
  the log-probability contribution.

**Intuition:**
Each sample adds a “reward” (high log-probability) or “penalty” (low
log-probability) depending on how well its features led to the correct
prediction.

--------------------------------------------------

Step 5 — Negative Log-Likelihood = Cross-Entropy Loss
--------------------------------------------------

The negative log-likelihood (what we actually minimize) is:

.. math::

   \ell_i = - \Big[y_i \ln \hat y_i + (1 - y_i) \ln (1 - \hat y_i)\Big]

**Role of :math:`y_i`:**

* Determines which side of the loss is active:
  - :math:`y_i = 1` → penalizes the model if :math:`\hat y_i` is small.
  - :math:`y_i = 0` → penalizes the model if :math:`\hat y_i` is large.

**Role of :math:`x_i`:**

* :math:`x_i` affects :math:`\hat y_i` via the model.
* Hard-to-classify samples (features that do not strongly indicate the
  true label) will produce higher loss, pushing the model to adjust weights.

**Intuition:**
The model is penalized more when it assigns low probability to the true
label. Each sample’s features determine how easily it can be correctly
classified, and each label determines which prediction counts as “correct.”

--------------------------------------------------

Final Insight
-------------

Training logistic regression is equivalent to maximizing the likelihood
of the observed data under a Bernoulli model:

.. math::

   \arg\max L(w,b) = \arg\min (-\ln L(w,b))

**Summary of roles:**

* :math:'x_i' → determines the predicted probability :math:`\hat y_i`.
* :math:`y_i` → selects which probability counts as “correct” in the
  likelihood.
* The loss penalizes the model when :math:`x_i` leads to a prediction that
  does not match :math:`y_i`.

This explains why cross-entropy loss directly measures the model’s
performance in probabilistic terms for each input-label pair.
