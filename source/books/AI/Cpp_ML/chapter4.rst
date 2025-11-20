Machine Learning Algorithms - Clustering
========================================

**Clustering** is an unssupervised ML method that’s used for splitting the original dataset of objects into groups classified by properties.

Measuring distance in clustering
--------------------------------

Step 1: Normalize feature values.
Normalization ensures that each feature has the same impact in a distance measure calculation.

Here are some popular ones that are used for **numerical properties**:

.. admonition:: Euclidean distance

   .. math::

      \delta(x, \bar{x}) = \sqrt{\sum_{i}^{n}(x_i - \bar{x}_i)^2}

   This is a geometric distance in the multidimensional space.

.. admonition:: Squared Euclidean distance

   .. math::

      d^2(x, \bar{x}) = \sum_{i=1}^{n} (x_i - \bar{x}_i)^2

   Squared Euclidean distance has the same properties as Euclidean distance but assigns greater significance (weight) to the distant values than to closer ones.


.. admonition:: Manhattan distance

   .. math::

      d_{\text{Manhattan}}(x, \bar{x}) = \sum_{i=1}^{n} |x_i - \bar{x}_i|

   Average difference by coordinates. In most cases, its value gives the same clustering results as Euclidean distance. However, it reduces the significance (weight) of the distant values (outliers).





