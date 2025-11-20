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


.. admonition:: Manhattan distance (also known as L1 distance or taxicab distance)

   .. math::

      d_{\text{Manhattan}}(x, \bar{x}) = \sum_{i=1}^{n} |x_i - \bar{x}_i|

   Average difference by coordinates. In most cases, its value gives the same clustering results as Euclidean distance. However, it reduces the significance (weight) of the distant values (outliers).

.. admonition:: Chebyshev distance (also known as L∞ distance or maximum metric)

   .. math::
      d_{\text{Chebyshev}}(x, \bar{x}) = \max_{i=1}^{n} |x_i - \bar{x}_i|

   Chebyshev distance can be useful when we need to classify two objects as different when they differ only by one of the coordinates.

.. admonition:: Image shows the different distances:

   .. images:: 4_1.jpg

   Here, we can see that Manhattan distance is the sum of the distances in both dimensions, like walking along city blocks. Euclidean distance is just the length of a straight line. Chebyshev distance is a more flexible alternative to Manhattan distance because diagonal moves are also taken into account.




