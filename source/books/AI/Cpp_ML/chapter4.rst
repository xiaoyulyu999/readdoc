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

   .. image:: images/4_1.jpg

   Here, we can see that Manhattan distance is the sum of the distances in both dimensions, like walking along city blocks. Euclidean distance is just the length of a straight line. Chebyshev distance is a more flexible alternative to Manhattan distance because diagonal moves are also taken into account.

Types of clustering algorithms
------------------------------

partition-based, spectral, hierarchical, density-based, and model-based.
The partition-based group of clustering algorithms can be logically divided into distance-based methods and ones based on graph theory.

.. warning::

   We can split cluster analysis into the following phases:
   - Selecting objects
   - Determining the set of object properties that we will use for the metric.
   - Normalizing property values.
   - Calculating the metric
   - Identifying distinct groups of objects based on metric values.

After analyzing clustering results, some correction may be required for the selected metric of the chosen algorithm.

Partition-based clustering algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   - combine objects into groups
   - usually require either the number of desired clusters or a threshold that regulates the number of output clusters to be specified explicitly.
   - The choice of a similarity measure can significantly affect the quality and accuracy of the clusters produced, potentially leading to misinterpretations of data patterns and insights.

Distance-based clustering algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
k-means and k-medoids algorithms.

Take the k input parameter and divide the data space into k clusters so that the similarity between objects in one cluster is maximal.

They take the k input parameter and divide the data space into k clusters so that the similarity between objects in one cluster is maximal. Also, they minimize the similarity between objects of different clusters. The similarity value is calculated as the distance from the object to the cluster center. The main difference between these methods lies in the way the cluster center is defined.

- **k-means algorithm**, the similarity is proportional to the distance to the cluster center of mass. The cluster center of mass is the average value of cluster objects’ coordinates in the data space.
   - steps:
      First, we select k random objects and define each of them as a cluster prototype that represents the cluster’s center of mass. Then, the remaining objects are attached to the cluster with greater similarity. After that, the center of mass of each cluster is recalculated. For each obtained partition, a particular evaluation function is calculated, the values of which at each step form a converging series. This process continues until the specified series converges to its limit value.

   - Pro:
      The k-means method works well when clusters are compact clouds that are significantly separated from each other. It’s useful for processing large amounts of data.
   - Con:
      but It isn’t applicable for detecting clusters of non-convex shapes or clusters with very different sizes. Moreover, the method is susceptible to noise and isolated points since even a small number of such points can significantly affect how the center mass of the cluster is calculated.


- **k-medoids**

in contrast to the k-means algorithm, uses one of the cluster objects (known as the representative object) as the center of the cluster. As in the k-means method, k representative objects are selected at random. Each of the remaining objects is combined into a cluster with the nearest representative object. Then, each representative object is replaced iteratively with an arbitrary unrepresentative object from the data space. The replacement process continues until the quality of the resulting clusters improves. **The clustering quality is determined by the sum of deviations between objects and the representative object of the corresponding cluster**, which the method tries to minimize. Thus, the iterations continue until the representative object in each of the clusters becomes the medoid.

The medoid is the object closest to the center of the cluster. The algorithm is poorly scalable for processing large amounts of data, but this problem is solved by the Clustering Large Applications based on RANdomized Search (CLARANS) algorithm, which complements the k-medoids method. CLARANS attempts to address scalability issues by using a randomized search technique to find good solutions more efficiently. Such an approach makes it possible to quickly converge on a good solution without exhaustively searching all possible combinations of medoids. For multidimensional clustering, the Projected Clustering (PROCLUS) algorithm can be used.

Graph theory-based clustering algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph vertices correspond to objects, and the edge weights are equal to the distance between vertices.

The advantages of graph clustering algorithms are their excellent visibility, relative ease of implementation, and their ability to make various improvements based on geometrical considerations. The main graph theory concepts used for clustering are selecting connected components, constructing a minimum spanning tree, and multilayer graph clustering.

Recognizing and Finding Spanning Trees in Graph Theory_

.. _Recognizing and Finding Spanning Trees in Graph Theory: https://www.youtube.com/watch?v=b233VKD6udo

Spectral clustering algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Stanford University — Watch on YouTube <https://www.youtube.com/watch?v=uxsDKhZHDcc>`_


