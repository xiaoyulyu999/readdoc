Measuring Performance and Selecting Models
==========================================

Performance metrics for ML models
---------------------------------

1. Regression Metrics
~~~~~~~~~~~~~~~~~~~~~
.. admonition:: **MSE** - quality

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

.. admonition:: **RMSE** - performance

   **Root Mean Squared Error (RMSE)** is a commonly used metric for evaluating regression models.
   It is the square root of the Mean Squared Error (MSE) and provides an estimate of the
   average magnitude of prediction errors, expressed in the same units as the target variable.

   The formula for RMSE is given by:

   .. math::

      \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}

   Where:

   - :math:`N` is the total number of predictions (or data points)
   - :math:`y_i` is the ground truth (actual) value for the *i*-th item
   - :math:`\hat{y}_i` is the predicted value for the *i*-th item

   RMSE penalizes larger errors more heavily than smaller ones due to the squaring operation.
   A lower RMSE value indicates a better fit between the model predictions and the actual values.

.. admonition::**MAE** - quality

   Mean Absolute Error (MAE) is a commonly used metric for evaluating the performance of regression models.
   It measures the average magnitude of the errors between predicted and actual values, without considering their direction.
   In other words, it represents how far the predictions are from the true values on average. Which can be problematic in some cases. For example, if a model consistently underestimates or overestimates the true value, the MAE will still give a low score, even though the model may not be performing well. But this metric is more robust for outliers than RMSE.

   The formula for MAE is given by:

   .. math::

      \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|

   Where:

   - :math:`N` is the total number of predictions (or data points)
   - :math:`y_i` is the ground truth (actual) value for the *i*-th item
   - :math:`\hat{y}_i` is the predicted value for the *i*-th item

   MAE is simple to interpret since it gives the average absolute difference between predictions and actual observations.
   A smaller MAE indicates a better model fit.
