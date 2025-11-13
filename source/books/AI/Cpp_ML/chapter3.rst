Measuring Performance and Selecting Models
==========================================

Performance metrics for ML models
---------------------------------

Regression Metrics
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

.. admonition:: **Accuracy**

   Accuracy measures the overall correctness of the model:

   .. math::

      \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

   It represents the proportion of correctly classified samples among all samples.
   In general, this metric is not very useful because it doesn’t show us the real picture in terms of cases with an odd number of classes. Let’s consider a spam classification task and assume we have 10 spam letters and 100 non-spam letters. Our algorithm predicted 90 of them correctly as non-spam and classified only 5 spam letters correctly. In this case, the accuracy = 86.4 However, if the algorithm predicts all letters as non-spam, then its accuracy should be 90.9. This is showing that our model doesn't work because it is unable to predict all spam letters, but the accuracy value is good enough.

.. admonition:: **Precision**, **Recall**

   Precision measures the proportion of correctly predicted positive instances
   among all instances predicted as positive:

   .. math::

      \text{Precision} = \frac{TP}{TP + FP}

   High precision indicates that the model produces few false positives.

   **Recall** (Sensitivity or True Positive Rate)

   Recall measures the proportion of actual positives that were correctly identified:

   .. math::

      \text{Recall} = \frac{TP}{TP + FN}

   High recall indicates that the model successfully detects most positive instances.

.. admonition:: **F-Score (F-Measure)**

   The F-score, also known as the F-measure, is a classification metric that combines **Precision**
   and **Recall** into a single value. It represents the harmonic mean of Precision and Recall,
   providing a balance between the two.

   The general formula for the F-score is:

   .. math::

      F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}
                                     {(\beta^2 \times \text{Precision}) + \text{Recall}}

   Where:

   - :math:`\text{Precision}` = :math:`\frac{TP}{TP + FP}`
   - :math:`\text{Recall}` = :math:`\frac{TP}{TP + FN}`
   - :math:`\beta` is a weighting factor that determines the importance of Recall relative to Precision

   **Special Cases**

   - **F1-Score:** when :math:`\beta = 1`, Precision and Recall are equally weighted:

     .. math::

        F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}
                          {\text{Precision} + \text{Recall}}

   - **F0.5-Score:** gives more weight to **Precision** than Recall.
   - **F2-Score:** gives more weight to **Recall** than Precision.

   Interpretation
   --------------

   - A **higher F-score** indicates better model performance in terms of balancing Precision and Recall.
   - F-score is especially useful when dealing with **imbalanced datasets**, where relying solely on Accuracy can be misleading.

   Comparison with Other Metrics
   -----------------------------

   - **Precision** alone ignores false negatives.
   - **Recall** alone ignores false positives.
   - **F-score** provides a single number that reflects both kinds of errors, making it suitable for tasks like information retrieval or medical diagnosis.



   **Macro**, **Micro**, and **Weighted Averages**

   For multi-class classification problems:

   - **Macro-average:** computes the metric independently for each class and takes the average (treats all classes equally).
   - **Micro-average:** aggregates the contributions of all classes to compute the metric globally.
   - **Weighted-average:** like macro-average but weights each class by its support (number of true instances).

.. admonition:: **AUC-ROC** (Under the Receiver Operating Characteristic Curve)

   The **Receiver Operating Characteristic (ROC)** curve and its associated **Area Under the Curve (AUC)**
   are important metrics for evaluating the performance of classification models, especially
   binary classifiers.

   ROC Curve
   ----------

   The ROC curve plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)**
   at various classification thresholds.

   .. math::

      \text{TPR} = \frac{TP}{TP + FN}
      \qquad
      \text{FPR} = \frac{FP}{FP + TN}

   Where:

   - :math:`TP` — True Positives
   - :math:`TN` — True Negatives
   - :math:`FP` — False Positives
   - :math:`FN` — False Negatives

   The ROC curve shows the trade-off between sensitivity (recall) and specificity as the threshold varies.

   Area Under the Curve (AUC)
   --------------------------

   The **AUC** represents the area under the ROC curve, providing a single scalar value
   that summarizes the model’s ability to distinguish between positive and negative classes.

   .. math::

      \text{AUC} = \int_0^1 TPR(FPR) \, d(FPR)

   **Interpretation:**

   - :math:`AUC = 1.0` → perfect classification
   - :math:`AUC = 0.5` → random guessing
   - :math:`AUC < 0.5` → model performs worse than random guessing

   In practice, a higher AUC value indicates a better performing classifier across different threshold settings.

   Advantages
   ----------

   - AUC–ROC is **threshold-independent**, meaning it evaluates model performance across all classification thresholds.
   - Useful when the dataset is **imbalanced**, since it considers both the true positive rate and false positive rate.

   Limitations
   -----------

   - AUC–ROC can be **overly optimistic** when classes are highly imbalanced.
   - For such cases, the **Precision–Recall (PR) curve** and **AUC–PR** metric may provide a more informative view.

.. admonition:: **Log-Loss**

   Log-loss, also known as *cross-entropy loss*, is a performance metric
   commonly used in binary and multi-class classification problems. It
   measures the uncertainty of your predictions based on how far they
   are from the true labels. Lower log-loss indicates a better predictive
   model.

   Binary Log-Loss Function
   ------------------------

   For a binary classification problem, the log-loss for a single sample is
   defined as:

   .. math::

       L(y, p) = - \left[ y \log(p) + (1 - y) \log(1 - p) \right]

   Where:

   * ``y`` is the true label (0 or 1)
   * ``p`` is the predicted probability that ``y = 1``

   Average Log-Loss
   ----------------

   For ``N`` samples, the overall log-loss is:

   .. math::

       \text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N}
       \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]

   Notes
   -----

   * Log-loss heavily penalizes confident but wrong predictions.
   * A perfect model achieves a log-loss of ``0``.
   * It works best when predictions are expressed as probabilities,
     not hard class labels.

Understanding bias and variance characteristics
-----------------------------------------------

Bias
~~~~

Bias is a prediction characteristic that tells us about the distance between model predictions and ground truth values. Usually, we use the term high bias or underfitting to say that model prediction is too far from the ground truth values, which means that the model generalization ability is weak.

.. admonition:: Regression model predictions with the polynomial degree equal to 1

   .. image:: 3_3.jpg

   This graph shows the original values, the values used for validation, and a line that represents the polynomial regression model output. iIn this case, the polynomial degree is equal to 1. We can see that the predictied values do not describe the original data at all, so we can say that this model has a high bias. Also, we can plot validation metrics for each training cycle to get more information about the training process and the model's behavior.

Variance
~~~~~~~~

Variance is a prediction characteristic that tells us about the variability of model predictions; in other words, how big the range of output values can be. Usually, we use the term high variance or overfitting in the case when a model tries to incorporate many training samples very precisely. In such a case, the model cannot provide a good approximation for new data but has excellent performance on the training data.

