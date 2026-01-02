.. warning::
   Something seems easy,but mess up easily

Logistic Regression, Likelihood and Cross-Entropy
================================================


Step 1 — Logistic regression model
----------------------------------

For each data point :math:`x_i`:

.. math::

   \hat y_i = P(y_i=1|x_i) = \sigma(w^T x_i + b)

where:

.. math::

   \sigma(z) = \frac{1}{1 + e^{-z}}

--------------------------------------------------

Step 2 — Probability model (Bernoulli)
--------------------------------------

Since :math:`y_i \in \{0,1\}`, we model the output using a Bernoulli distribution:

.. math::

   P(y_i|x_i) = \hat y_i^{y_i} (1 - \hat y_i)^{(1 - y_i)}

--------------------------------------------------

Step 3 — Likelihood of the whole dataset
----------------------------------------

.. math::

   L(w,b) = \prod_{i=1}^{n} \hat y_i^{y_i} (1 - \hat y_i)^{(1 - y_i)}

--------------------------------------------------

Step 4 — Log-likelihood
-----------------------

.. math::

   \ln L = \sum_{i=1}^{n}
           \Big[y_i \ln \hat y_i +
           (1 - y_i)\ln(1 - \hat y_i)\Big]

--------------------------------------------------

Step 5 — Negative log-likelihood (Loss)
---------------------------------------

.. math::

   -\ln L = -\sum_{i=1}^{n}
            \Big[y_i \ln \hat y_i +
            (1 - y_i)\ln(1 - \hat y_i)\Big]

This is the loss function used in logistic regression.

--------------------------------------------------

Step 6 — Why this is Cross-Entropy
----------------------------------

The general cross-entropy between a true distribution :math:`p` and model distribution :math:`q` is:

.. math::

   H(p, q) = -\sum_x p(x)\ln q(x)

For each training sample, the true distribution is:

.. math::

   p(y) =
   \begin{cases}
   1 & \text{if } y = y_i \\
   0 & \text{otherwise}
   \end{cases}

Substituting this into the cross-entropy formula:

.. math::

   H(p, q) =
   -\big[y_i\ln \hat y_i +
   (1-y_i)\ln(1-\hat y_i)\big]

which matches the negative log-likelihood exactly.

--------------------------------------------------

Final Insight
-------------

Minimizing cross-entropy in logistic regression is mathematically equivalent to
maximizing the likelihood of the Bernoulli probabilistic model.

Therefore:

.. math::

   \arg\max L(w,b) = \arg\min (-\ln L(w,b))
