Measuring Performance and Selecting Models
==========================================

Performance metrics for ML models
---------------------------------

1. Regression Metrics
~~~~~~~~~~~~~~~~~~~~~
.. admonition:: **MSE**

   **Mean Squared Error (MSE)** is a widely used metric for regression algorithms to estimate their quality.
   It represents the average of the squared differences between the predicted and the actual (ground truth) values.

   The formula for MSE is given by:

   .. math::

      \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2

   Where:

   - :math:`N` is the total number of predictions (or data points)
   - :math:`y_i` is the ground truth (actual) value for the *i*-th item
   - :math:`\hat{y}_i` is the predicted value for the *i*-th item

   MSE is always non-negative, and a smaller MSE indicates better model performance.
