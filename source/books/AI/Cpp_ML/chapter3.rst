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

.. admonition:: **MAE** - quality

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


.. admonition:: **R-squared**

   R-squared (:math:`R^2`) measures how well a regression model explains the variability of the
   dependent variable. It compares the variance captured by the model with the total variance
   present in the data.

   It is defined in terms of two quantities:

   1. **Total Sum of Squares (SStot)** — measures the total variance in the data:

      .. math::

         SS_{tot} = \sum_i (y_i - \bar{y})^2

   2. **Residual Sum of Squares (SSres)** — measures the variance that is not explained by the model:

      .. math::

         SS_{res} = \sum_i (y_i - \hat{y}_i)^2

   The coefficient of determination is then calculated as:

   .. math::

      R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

   Equivalently, this can be expressed as:

   .. math::

      R^2 = 1 -
      \frac{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
           {\frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2}

   Where:

   - :math:`y_i` is the ground truth (actual) value for the *i*-th item
   - :math:`\hat{y}_i` is the predicted value for the *i*-th item
   - :math:`\bar{y}` is the mean of all ground truth values
   - :math:`n` is the total number of data points

   **Interpretation:**

   - :math:`R^2 = 1` → perfect prediction
   - :math:`R^2 = 0` → model predicts no better than the mean
   - :math:`R^2 < 0` → model performs worse than predicting the mean

.. admonition:: **Adjusted R-squared**

   Adjusted R-squared (:math:`\bar{R}^2`) is a modified version of the coefficient of determination (:math:`R^2`)
   that adjusts for the number of independent variables (predictors) in the model.
   It accounts for the possibility that simply adding more variables can artificially inflate the value of :math:`R^2`,
   even if those variables do not actually improve the model’s predictive power.

   The formula for Adjusted R-squared is given by:

   .. math::

      \bar{R}^2 = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}

   Where:

   - :math:`R^2` is the coefficient of determination
   - :math:`n` is the number of observations (data points)
   - :math:`p` is the number of independent variables (predictors)

   **Interpretation:**

   - Adjusted R-squared increases only if the new predictor improves the model
     more than would be expected by chance.
   - If an added predictor does not contribute useful information,
     Adjusted R-squared will decrease.
   - A higher Adjusted R-squared indicates a model that fits the data better,
     while penalizing unnecessary model complexity.

   In general, Adjusted R-squared provides a more reliable measure than :math:`R^2` when comparing models with different numbers of predictors.

.. admonition:: **Classification metrics**

   Classification metrics are used to evaluate the performance of classification algorithms —
   models that assign input data to discrete categories (e.g., spam vs. not spam, positive vs. negative sentiment).

   These metrics quantify how well the predicted class labels match the true labels.


   A confusion matrix is a table that summarizes the performance of a classification model by comparing
   predicted and actual class labels.

   .. math::

      \begin{bmatrix}
      TP & FP \\
      FN & TN
      \end{bmatrix}

   Where:

   - **TP (True Positive):** correctly predicted positive instances
   - **TN (True Negative):** correctly predicted negative instances
   - **FP (False Positive):** negative instances incorrectly predicted as positive
   - **FN (False Negative):** positive instances incorrectly predicted as negative

   From this matrix, we can derive several key metrics.

   **Accuracy**
   ------------

   Accuracy measures the overall correctness of the model:

   .. math::

      \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

   It represents the proportion of correctly classified samples among all samples.

   **Precision**
   -------------

   Precision measures the proportion of correctly predicted positive instances
   among all instances predicted as positive:

   .. math::

      \text{Precision} = \frac{TP}{TP + FP}

   High precision indicates that the model produces few false positives.

   **Recall** (Sensitivity or True Positive Rate)
   ----------------------------------------------

   Recall measures the proportion of actual positives that were correctly identified:

   .. math::

      \text{Recall} = \frac{TP}{TP + FN}

   High recall indicates that the model successfully detects most positive instances.

   **F1-Score**
   ------------

   The F1-score is the harmonic mean of Precision and Recall, balancing both metrics:

   .. math::

      F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}

   It is particularly useful when the dataset is imbalanced, as it accounts for both false positives and false negatives.

   **Macro**, **Micro**, and **Weighted Averages**
   -----------------------------------------------

   For multi-class classification problems:

   - **Macro-average:** computes the metric independently for each class and takes the average (treats all classes equally).
   - **Micro-average:** aggregates the contributions of all classes to compute the metric globally.
   - **Weighted-average:** like macro-average but weights each class by its support (number of true instances).
