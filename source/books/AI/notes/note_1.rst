How Cross-Entropy Loss Works in Logistic Regression
===================================================

This document explains, step by step, how each training sample
:math:`(x_i, y_i)` is transformed into a probabilistic penalty that drives
model learning.

--------------------------------------------------

Step 1 — Mapping Input :math:`x_i` to a Probability
--------------------------------------------------

For each sample :math:`x_i`, logistic regression computes:

.. math::

   \hat y_i = \sigma(w^T x_i + b)

where:

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}

**Effect on :math:`x_i`:**

* The feature vector :math:`x_i` is converted into a scalar score.
* The sigmoid function maps this score into a probability
  :math:`\hat y_i \in (0,1)`.

So :math:`\hat y_i` represents the model’s estimated probability that
:math:`x_i` belongs to the positive class.

--------------------------------------------------

Step 2 — Using :math:`y_i` to Select the Correct Probability
--------------------------------------------------

The conditional probability of the observed label :math:`y_i` is modeled by
a Bernoulli distribution:

.. math::

   P(y_i \mid x_i) = \hat y_i^{y_i}(1-\hat y_i)^{(1-y_i)}

**Effect on :math:`y_i`:**

* If :math:`y_i = 1`, the expression becomes :math:`P = \hat y_i`.
* If :math:`y_i = 0`, the expression becomes :math:`P = 1-\hat y_i`.

Thus, :math:`y_i` acts as a switch that selects the probability assigned to
the true class.

--------------------------------------------------

Step 3 — Likelihood of the Whole Dataset
--------------------------------------------------

Assuming samples are independent, the total likelihood is:

.. math::

   L(w,b) = \prod_{i=1}^{n} P(y_i \mid x_i)

**Effect:**

* Each sample contributes multiplicatively to the model score.
* A single badly predicted sample can significantly reduce the total
  likelihood.

--------------------------------------------------

Step 4 — Log-Likelihood
--------------------------------------------------

Taking the logarithm transforms the product into a sum:

.. math::

   \ln L = \sum_{i=1}^{n}
   \Big[y_i \ln \hat y_i + (1-y_i)\ln(1-\hat y_i)\Big]

**Effect:**

* Each sample now contributes an additive score.
* Good predictions increase the total score, while poor predictions decrease
  it.

--------------------------------------------------

Step 5 — Negative Log-Likelihood (Loss Function)
--------------------------------------------------

To convert maximization into a minimization problem, we define the loss:

.. math::

   \ell_i =
   -\Big[y_i \ln \hat y_i + (1-y_i)\ln(1-\hat y_i)\Big]

**Effect on each sample:**

* If the true label is 1 and :math:`\hat y_i \to 0`, the loss tends to infinity.
* If the true label is 0 and :math:`\hat y_i \to 1`, the loss tends to infinity.
* If :math:`\hat y_i` is close to the true label, the loss approaches zero.

Therefore, the loss penalizes the model according to how little probability
it assigns to the true class.

--------------------------------------------------

Final Insight
-------------

Training logistic regression is equivalent to maximizing the likelihood of
the observed data under a Bernoulli model:

.. math::

   \arg\max L(w,b) = \arg\min (-\ln L(w,b))

This shows that cross-entropy loss is simply the negative log-likelihood of a
probabilistic model.
