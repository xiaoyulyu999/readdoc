Efron (1979): Bootstrap Methods — 全文知识点总结
================================================

论文信息
--------

**论文题目：** *Bootstrap Methods: Another Look at the Jackknife*

**作者：** Bradley Efron

**期刊：** *The Annals of Statistics*

**年份：** 1979

**卷期：** Vol. 7, No. 1

**页码：** 1–26

核心问题
--------

这篇论文要解决的根本问题是：

给定来自未知总体分布 :math:`F` 的一个随机样本

.. math::

   X=(X_1,X_2,\ldots,X_n)\sim F

实际观察到的数据为

.. math::

   x=(x_1,x_2,\ldots,x_n)

如果我们关心某个统计量或随机量

.. math::

   R(X,F)

如何只根据当前观察到的样本 :math:`x`，估计 :math:`R(X,F)` 的
**sampling distribution（抽样分布）**？

也就是：

.. code-block:: text

   Observed sample x
          ↓
   Estimate sampling distribution

抽样分布决定了很多统计推断量，例如：

* bias
* variance
* standard error
* confidence interval
* hypothesis testing

Parameter、Estimator 与 Sampling Distribution
---------------------------------------------

设总体真正的 parameter 为：

.. math::

   \theta(F)

例如总体均值：

.. math::

   \theta(F)=\mu

Estimator 写作：

.. math::

   t(X)

例如：

.. math::

   t(X)=\bar X

于是估计误差可以写成：

.. math::

   R(X,F)=t(X)-\theta(F)

例如：

.. math::

   R=\bar X-\mu

如果知道 :math:`R` 的抽样分布，就可以进一步研究 estimator 的：

* bias
* variability
* standard error
* confidence interval

Bootstrap 并不只适用于 :math:`t(X)-\theta(F)`，原则上可以处理更一般的
:math:`R(X,F)`。

Bootstrap 的核心思想：Empirical Distribution
--------------------------------------------

真实总体分布：

.. math::

   F

未知。

Bootstrap 用观察到的数据构造经验分布：

.. math::

   \hat F

定义为：

.. math::

   \hat F=
   \frac{1}{n}\sum_{i=1}^{n}\delta_{x_i}

也就是说，在每一个观测值 :math:`x_i` 上放置概率质量：

.. math::

   \frac{1}{n}

因此：

.. math::

   P_{\hat F}(X=x_i)=\frac1n

例如样本：

.. code-block:: text

   2, 5, 8, 10

经验分布为：

.. code-block:: text

   2    probability = 0.25
   5    probability = 0.25
   8    probability = 0.25
   10   probability = 0.25

最核心的思想是：

.. math::

   F\text{ unknown}
   \Rightarrow
   \hat F\text{ approximates }F

Bootstrap Sample
----------------

从经验分布 :math:`\hat F` 中重新抽取一个大小仍然为 :math:`n` 的样本：

.. math::

   X^*=(X_1^*,X_2^*,\ldots,X_n^*)

其中：

.. math::

   X_i^*\overset{iid}{\sim}\hat F

这称为：

**bootstrap sample**

最重要的特点是：

**sampling with replacement（有放回抽样）**

例如原始样本：

.. code-block:: text

   A B C D

一个 bootstrap sample 可以是：

.. code-block:: text

   A A C D

也可以是：

.. code-block:: text

   B B B A

所以：

* 某些 observation 可以重复出现；
* 某些 observation 可以完全不出现。

Bootstrap 与 Jackknife 的区别
-----------------------------

Bootstrap：

.. math::

   n\text{ observations, sampling with replacement}

Ordinary Jackknife：

.. math::

   n-1\text{ observations, sampling without replacement}

Bootstrap Algorithm
-------------------

Step 1：建立经验分布
~~~~~~~~~~~~~~~~~~~~

根据：

.. math::

   x_1,\ldots,x_n

建立：

.. math::

   \hat F

Step 2：生成 Bootstrap Sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从 :math:`\hat F` 有放回抽取 :math:`n` 个 observation：

.. math::

   X^*\sim \hat F

Step 3：计算 Bootstrap Statistic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

计算：

.. math::

   R^*=R(X^*,\hat F)

Step 4：构造 Bootstrap Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

研究：

.. math::

   \mathcal L^*(R^*)

并用它近似：

.. math::

   \mathcal L_F(R)

整个流程：

.. code-block:: text

   F
   ↓
   X
   ↓
   F-hat
   ↓
   X*
   ↓
   R*
   ↓
   Bootstrap distribution

为什么 Bootstrap 有效
---------------------

现实世界：

.. math::

   X\sim F

Bootstrap 世界：

.. math::

   X^*\sim\hat F

如果：

.. math::

   \hat F\approx F

那么希望：

.. math::

   \mathcal L_{\hat F}(R^*)
   \approx
   \mathcal L_F(R)

也就是：

.. math::

   F\approx\hat F
   \Rightarrow
   R(X,F)\approx R(X^*,\hat F)

Bootstrap Distribution
----------------------

现实中我们想知道：

.. math::

   R(X,F)

的 distribution。

Bootstrap 研究：

.. math::

   R^*=R(X^*,\hat F)

条件于当前观察到的数据 :math:`x`。

常见符号：

.. math::

   E^*,\qquad Var^*,\qquad P^*

表示在 bootstrap sampling mechanism 下计算 expectation、variance 和 probability，
同时把 original observed sample 固定。

三种计算 Bootstrap Distribution 的方法
---------------------------------------

Method 1：Direct Theoretical Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

直接利用概率论推导 bootstrap distribution。

优点：

* exact
* 不需要 simulation

缺点：

* 对复杂 statistic 往往难以解析计算。

Method 2：Monte Carlo Bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这是现代最常使用的 bootstrap。

生成：

.. math::

   X^{*1},X^{*2},\ldots,X^{*B}

计算：

.. math::

   R^{*1},R^{*2},\ldots,R^{*B}

然后用这些 bootstrap replications 的经验分布近似真正的 bootstrap distribution。

流程：

.. code-block:: text

   Original sample
        ↓
   Resample 1 → statistic*
   Resample 2 → statistic*
   Resample 3 → statistic*
        ...
   Resample B → statistic*
        ↓
   Bootstrap distribution

Bootstrap mean：

.. math::

   \bar T^*
   =
   \frac1B
   \sum_{b=1}^{B}T^{*b}

Bootstrap variance：

.. math::

   \widehat{Var}_{boot}
   =
   \frac1{B-1}
   \sum_{b=1}^{B}
   (T^{*b}-\bar T^*)^2

Bootstrap standard error：

.. math::

   SE_{boot}
   =
   \sqrt{\widehat{Var}_{boot}}

Method 3：Taylor Expansion
~~~~~~~~~~~~~~~~~~~~~~~~~

对 bootstrap statistic 在经验分布附近做 Taylor expansion：

.. math::

   R(P^*)
   \approx
   R(P_0)
   +
   \nabla R(P_0)(P^*-P_0)
   +
   \text{quadratic term}

这个展开可以用来近似 bootstrap mean 和 variance。

论文的重要结论之一是：

.. math::

   \boxed{\text{Jackknife is approximately a Taylor/linear approximation to Bootstrap}}

在这篇论文的框架里：

**infinitesimal jackknife 本质上对应 delta method。**

Bootstrap 与 Jackknife 的关系
-----------------------------

传统 Jackknife：

每次删除一个 observation：

.. math::

   x_{(-1)},x_{(-2)},\ldots,x_{(-n)}

重新计算：

.. math::

   t_{(-1)},t_{(-2)},\ldots,t_{(-n)}

再利用这些值估计：

* bias
* variance

Efron 通过对 bootstrap distribution 做 Taylor expansion 发现：

* first-order term 给出近似 variance；
* second-order term 与 bias correction 有关；
* 结果与 infinitesimal jackknife 和 ordinary jackknife 密切对应。

因此：

.. math::

   \boxed{
   \text{Jackknife}
   \approx
   \text{local linear version of Bootstrap}
   }

为什么 Bootstrap 有时比 Jackknife 更好
-------------------------------------

Jackknife 每次只删除一个 observation，因此对 empirical distribution 的扰动很小。

大致尺度：

.. math::

   O(1/n)

而 bootstrap sampling fluctuation 的典型尺度是：

.. math::

   O(n^{-1/2})

因此 bootstrap 更接近真实 sampling variation 的尺度。

对于不够 smooth 的 statistic，例如 median，
ordinary jackknife 的局部线性近似可能失败。

Bernoulli Parameter 示例
------------------------

假设：

.. math::

   X_i\in\{0,1\}

总体 parameter：

.. math::

   \theta=P(X=1)

Estimator：

.. math::

   \hat\theta=\bar X

经验分布下：

.. math::

   P_{\hat F}(X^*=1)=\bar x

所以：

.. math::

   n\bar X^*
   \sim
   Binomial(n,\bar x)

于是：

.. math::

   E^*(\bar X^*-\bar x)=0

以及：

.. math::

   Var^*(\bar X^*)
   =
   \frac{\bar x(1-\bar x)}{n}

Bootstrap 在这个简单问题中自然恢复了经典 Bernoulli variance estimator。

Sample Variance 示例
--------------------

如果：

.. math::

   \theta(F)=Var_F(X)

并使用 sample variance 作为 estimator，
bootstrap 可以用于估计 sample variance 的 sampling variability。

在这个 regular statistic 中，bootstrap variance approximation 与传统 jackknife
variance estimate 非常接近。

Sample Median
-------------

Section 3 研究：

.. math::

   \theta(F)=Median(F)

Estimator：

.. math::

   t(X)=X_{(m)}

即 sample median。

Median 是一个不够 smooth 的 statistic。

Ordinary jackknife 对 sample median variance 的估计甚至可能：

**not asymptotically consistent**

而 bootstrap 可以得到正确的 asymptotic variance。

对于连续 density :math:`f`：

.. math::

   Var(\hat m)
   \approx
   \frac{1}{4n f(\theta)^2}

这个例子是 bootstrap 比 ordinary jackknife 更一般的重要证明之一。

Bootstrap Median 与 Multinomial Counts
--------------------------------------

假设：

.. math::

   n=2m-1

bootstrap sample 中每个 observation 被抽中的次数：

.. math::

   (N_1^*,\ldots,N_n^*)

满足：

.. math::

   Multinomial
   \left(
   n;
   \frac1n,\ldots,\frac1n
   \right)

利用累计 counts，可以推导 bootstrap median 落在哪个 observed order statistic 上。

这是 Method 1 的典型例子。

Symmetrized Bootstrap
---------------------

如果已知总体 :math:`F` 是 symmetric 的，
可以把这一结构信息加入 bootstrap。

不再只用普通 empirical distribution：

.. math::

   \hat F

而构造关于 sample median 对称的：

.. math::

   \hat F_{\mathrm{SYM}}

这说明：

**Bootstrap 不一定必须完全 nonparametric。**

如果有可靠 structural assumptions，可以将其纳入 resampling mechanism。

Smoothed Bootstrap
------------------

普通 empirical distribution 是离散的。

对于连续总体，可以给 resampled observation 加一点随机扰动：

.. math::

   X_i^*
   =
   x_{I_i}
   +
   \text{small noise}

形成：

**smoothed bootstrap**

概念上：

.. code-block:: text

   Observed sample point
          +
   small random perturbation

论文比较了：

* ordinary bootstrap
* symmetrized bootstrap
* smoothed bootstrap

结果显示，在研究的 median 小样本例子中，
最简单的 ordinary bootstrap 已经表现得相当好。

因此：

.. math::

   \boxed{
   \text{More complexity does not automatically mean better inference}
   }

Two-Sample Bootstrap
--------------------

假设：

.. math::

   X_1,\ldots,X_m\sim F

以及：

.. math::

   Y_1,\ldots,Y_n\sim G

分别建立：

.. math::

   \hat F,\qquad\hat G

然后：

.. math::

   X_i^*\sim\hat F

.. math::

   Y_j^*\sim\hat G

再计算：

.. math::

   R((X^*,Y^*),(\hat F,\hat G))

因此 bootstrap 可以自然扩展到 two-sample problems。

Discriminant Analysis 与 Classification Error
---------------------------------------------

论文 Section 4 研究 linear discriminant analysis。

训练 classifier 后，training error 通常：

.. math::

   \text{underestimates true error}

这就是今天 Machine Learning 中常说的：

**optimistic training error**

Bootstrap 可以通过：

1. resample training data；
2. 重新训练 classifier；
3. 重新计算 error；
4. 估计 apparent error 与 true error 之间的差异；

来估计 classification optimism。

Bootstrap 与 Cross-Validation
-----------------------------

论文在特定 discriminant-analysis simulation 中比较：

* bootstrap
* leave-one-out cross-validation

结果表明：

* cross-validation 对期望误差的估计接近 unbiased；
* 但其 estimator variance 明显更大；
* 在该实验中，大约是 bootstrap estimator variance 的三倍。

注意：

这并不意味着 bootstrap 永远优于 cross-validation。

它只说明：

**在论文研究的特定 setting 中，bootstrap estimator 更稳定。**

Bias 与 Variance
----------------

论文的 discriminant example 还强调：

有时：

.. math::

   \text{variance problem}
   >
   \text{bias problem}

也就是说，过度追求 bias correction 可能没有实际意义，
如果 estimator 自身的 trial-to-trial variability 更大。

评估 estimator 时应同时考虑：

.. math::

   Bias^2 + Variance

Bootstrap Replications 数量
---------------------------

增加 bootstrap replications：

.. math::

   B

主要减少：

**Monte Carlo error**

但不能消除：

**original sample uncertainty**

因此：

.. math::

   B=1,000,000
   \neq
   \text{having more real observations}

当 :math:`B` 已经足够大后，再增加 replications 的收益会逐渐变小。

Ratio Estimation
----------------

论文研究：

.. math::

   \theta(F)
   =
   \frac{E(Y)}{E(Z)}

Estimator：

.. math::

   t(X)
   =
   \frac{\bar Y}{\bar Z}

这是一个 nonlinear statistic。

Method 3 可以通过 Taylor expansion 近似：

* bias
* variance

如果担心 Taylor approximation 不够好，则可以直接使用：

**Method 2：Monte Carlo Bootstrap**

Wilcoxon Statistic
------------------

Two-sample parameter：

.. math::

   \theta(F,G)
   =
   P(X<Y)

Estimator：

.. math::

   \hat\theta
   =
   \frac1{mn}
   \sum_i\sum_j I(X_i<Y_j)

Bootstrap 可以直接推导其 variance，
并与经典 Wilcoxon variance approximation 对应。

Unbalanced Jackknife
--------------------

Two-sample 问题中，ordinary leave-one-out 并不唯一。

例如到底应该：

* leave one :math:`X_i` out；
* leave one :math:`Y_j` out；
* leave one pair out。

Bootstrap/Taylor framework 帮助说明如何正确处理这类
unbalanced situations。

Regression Bootstrap
--------------------

一般 regression model：

.. math::

   X_i
   =
   g_i(\beta)+\epsilon_i

其中：

.. math::

   \epsilon_i\sim F

未知。

先拟合得到：

.. math::

   \hat\beta

Residual：

.. math::

   \hat\epsilon_i
   =
   x_i-g_i(\hat\beta)

构造 residual empirical distribution：

.. math::

   \hat F_\epsilon

然后生成 bootstrap data：

.. math::

   X_i^*
   =
   g_i(\hat\beta)
   +
   \epsilon_i^*

其中：

.. math::

   \epsilon_i^*
   \sim
   \hat F_\epsilon

每一个 bootstrap dataset 重新拟合：

.. math::

   \hat\beta^*

重复得到：

.. math::

   \hat\beta^{*1},
   \hat\beta^{*2},
   \ldots,
   \hat\beta^{*B}

就可以近似：

.. math::

   \hat\beta

的 sampling distribution。

这就是：

**residual bootstrap**

Linear Regression
-----------------

在线性模型：

.. math::

   X=C\beta+\epsilon

OLS estimator：

.. math::

   \hat\beta
   =
   (C'C)^{-1}C'X

经典 covariance：

.. math::

   Cov(\hat\beta)
   =
   \sigma^2(C'C)^{-1}

Residual bootstrap 得到：

.. math::

   Cov^*(\hat\beta^*)
   \approx
   \hat\sigma^2(C'C)^{-1}

说明：

.. math::

   \boxed{
   \text{Bootstrap reproduces standard theory in regular cases}
   }

Resampling 必须符合 Data-Generating Mechanism
-------------------------------------------

Regression 部分非常重要的思想是：

不能机械地对所有问题都简单 resample rows。

例如 residual bootstrap 会：

* 保持 design points 固定；
* 只重新抽 residuals。

这保留了 regression model 的结构。

因此：

.. important::

   Resampling scheme 必须反映 underlying data-generating mechanism。

这是现代 bootstrap 使用中最重要的原则之一。

Transformation
--------------

如果：

.. math::

   \phi=g(\theta)

并且：

.. math::

   s=g(t)

那么 bootstrap realization 也可以直接 transform。

如果 :math:`g` 是 monotonic，
bootstrap quantiles 可以自然对应到 transformed scale。

Pearson Correlation 与 Fisher Transformation
--------------------------------------------

对于 correlation coefficient：

.. math::

   r

Fisher transformation：

.. math::

   z
   =
   \tanh^{-1}(r)

也就是：

.. math::

   z
   =
   \frac12
   \ln
   \frac{1+r}{1-r}

相比直接研究：

.. math::

   \hat r-r

transformed quantity 往往更接近：

**pivotal quantity**

Pivotal Quantity
----------------

如果 statistic：

.. math::

   T(X,\theta)

的 distribution 不依赖未知 parameter :math:`\theta`，
或者近似不依赖，则称为：

**pivotal quantity**

经典例子：

.. math::

   \frac{\bar X-\mu}{S/\sqrt n}

在 normal sampling 下：

.. math::

   \sim t_{n-1}

Bootstrap 中，如果研究的 quantity 更接近 pivotal，
bootstrap approximation 往往更可靠。

Bootstrap 与 Confidence Interval 的限制
---------------------------------------

拥有 bootstrap distribution 并不意味着可以机械地构造可靠 confidence interval。

尤其当：

.. math::

   \theta^*-\hat\theta

和：

.. math::

   \hat\theta-\theta

的 distributions 之间不能通过一个稳定的 pivotal relationship 联系时，
简单的 interval transformation 可能出问题。

这也是后来进一步发展出：

* percentile bootstrap
* basic bootstrap
* bootstrap-t
* BCa interval

等方法的原因。

Frequency Statement 与 Likelihood Statement
-------------------------------------------

Bootstrap 和 Jackknife 给的是：

**approximate frequency statements**

而不是：

**approximate likelihood statements**

也就是说，Bootstrap 主要回答：

“如果重复类似 sampling experiment，statistic 会怎样变化？”

而不是直接回答：

“观察到当前 data 后，parameter 自身有多大 probability 落在某个区间？”

因此 bootstrap 仍属于 frequentist inference framework。

Studentized Statistic
---------------------

可以研究：

.. math::

   R
   =
   \frac{
   t(X)-\widehat{Bias}(t)-\theta
   }{
   \sqrt{\widehat{Var}(t)}
   }

而不是直接假设它一定服从：

.. math::

   t_{n-1}

可以直接 bootstrap 这个 studentized statistic：

.. math::

   R^{*1},R^{*2},\ldots,R^{*B}

再用其 empirical distribution 做 inference。

这与后来 bootstrap-t 思想密切相关。

Bootstrap 的渐近理论
--------------------

若 sample space 为：

.. math::

   \mathcal X
   =
   \{1,\ldots,L\}

真实 distribution 可表示为：

.. math::

   f=(f_1,\ldots,f_L)

empirical distribution：

.. math::

   \hat f

真实 sampling：

.. math::

   \hat f
   \sim
   Multinomial(n,f)/n

Bootstrap sampling：

.. math::

   \hat f^*
   \mid
   \hat f
   \sim
   Multinomial(n,\hat f)/n

由于：

.. math::

   \hat f\to f

所以 bootstrap fluctuation：

.. math::

   \hat f^*-\hat f

可以近似真实 sampling fluctuation：

.. math::

   \hat f-f

在 regularity conditions 下：

.. math::

   \sqrt n(\hat f-f)
   \Rightarrow
   N(0,\Sigma)

以及：

.. math::

   \sqrt n(\hat f^*-\hat f)
   \Rightarrow
   N(0,\Sigma)

所以 Bootstrap consistency 的核心直觉可以记为：

.. code-block:: text

   Empirical distribution consistency
              +
             CLT
              +
      Functional smoothness

Smoothness
----------

如果 statistic 可以写成：

.. math::

   T(F)

并且对 distribution 的小变化比较平滑，
Taylor/delta approximation 通常工作良好。

对于像 median 这样的 non-smooth statistic，
ordinary jackknife 可能失败，而 bootstrap 理论需要更谨慎处理。

Subsampling
-----------

Subsampling 通常：

**without replacement**

Bootstrap：

**with replacement**

并且 bootstrap sample size 通常仍然为：

.. math::

   n

论文指出，两类方法在一定条件下都可能得到正确的 asymptotic distribution。

Grouped Jackknife
-----------------

Ordinary jackknife：

.. code-block:: text

   delete 1 observation

Grouped jackknife：

.. code-block:: text

   delete g observations

当 :math:`g` 较大时，它对 empirical distribution 的扰动更明显。

对于 median，grouped jackknife 能得到正确的 asymptotic variance，
而 ordinary jackknife 可能失败。

Parametric Bootstrap
--------------------

Nonparametric bootstrap 使用：

.. math::

   \hat F_{\mathrm{empirical}}

如果假设：

.. math::

   F=F_\theta

属于某个 parametric family，
可以先估计：

.. math::

   \hat\theta

再使用：

.. math::

   F_{\hat\theta}

生成 bootstrap sample：

.. math::

   X_i^*
   \sim
   F_{\hat\theta}

这就是：

**parametric bootstrap**

Nonparametric 与 Parametric Bootstrap
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 方法
     - Resampling Distribution
     - 特点
   * - Nonparametric Bootstrap
     - :math:`\hat F_{\mathrm{empirical}}`
     - assumptions 少，但不利用具体 distribution model
   * - Parametric Bootstrap
     - :math:`F_{\hat\theta}`
     - model 正确时可能更 efficient，但受 model misspecification 影响

Parametric Bootstrap 与 Fisher Information
-------------------------------------------

对于 one-parameter maximum likelihood estimator，
parametric bootstrap 的 Taylor approximation 可以恢复经典结果：

.. math::

   Var(\hat\theta)
   \approx
   \frac1{I(\theta)}

其中：

.. math::

   I(\theta)

为 Fisher information。

这说明 Bootstrap 并不是和 classical statistics 对立，
而是可以统一很多传统结果。

Bootstrap 的统一视角
--------------------

整篇论文可以压缩为：

.. code-block:: text

   Unknown population F
          ↓
   Observed sample X
          ↓
   Estimate F using F-hat
          ↓
   Simulate sampling again
          ↓
          X*
          ↓
   Recalculate statistic
          ↓
          T*
          ↓
   Approximate sampling distribution

然后从 bootstrap distribution 可以估计：

.. code-block:: text

   Bias
   Variance
   Standard error
   Sampling uncertainty
   Error rate
   Test-statistic distribution
   Confidence-related quantities

最重要的 Bootstrap 公式
-----------------------

Empirical distribution：

.. math::

   \hat F
   =
   \frac1n
   \sum_{i=1}^{n}
   \delta_{x_i}

Bootstrap sample：

.. math::

   X_1^*,\ldots,X_n^*
   \overset{iid}{\sim}
   \hat F

Bootstrap statistic：

.. math::

   T^*=t(X^*)

Bootstrap distribution：

.. math::

   \mathcal L^*(T^*\mid X)

近似：

.. math::

   \mathcal L_F(T)

在合适条件下：

.. math::

   \mathcal L^*(T^*-\hat\theta)
   \approx
   \mathcal L_F(T-\theta)

Bootstrap bias estimate：

.. math::

   \widehat{Bias}_{boot}
   =
   E^*(T^*)-\hat\theta

Monte Carlo approximation：

.. math::

   \widehat{Bias}_{boot}
   =
   \frac1B
   \sum_{b=1}^{B}T^{*b}
   -
   \hat\theta

Bootstrap variance：

.. math::

   \widehat{Var}_{boot}
   =
   \frac1{B-1}
   \sum_{b=1}^{B}
   (T^{*b}-\bar T^*)^2

Bootstrap standard error：

.. math::

   SE_{boot}
   =
   \sqrt{\widehat{Var}_{boot}}

Bootstrap、Jackknife、Subsampling 与 Cross-Validation
---------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 20 35

   * - Method
     - Resampling
     - Sample Size
     - Main Purpose
   * - Bootstrap
     - With replacement
     - Usually :math:`n`
     - Sampling distribution, SE, bias, CI
   * - Jackknife
     - Leave-one-out
     - :math:`n-1`
     - Bias and variance approximation
   * - Subsampling
     - Without replacement
     - Usually :math:`m<n`
     - Sampling distribution
   * - Cross-validation
     - Train/validation partition
     - Variable
     - Prediction/generalisation error

最关键关系：

.. math::

   \boxed{
   \text{Jackknife}
   \approx
   \text{an approximation to Bootstrap}
   }

与 Machine Learning 的关系
--------------------------

对于一个模型 metric，例如：

.. math::

   Accuracy=0.86

真正的问题不仅是：

“Accuracy 是多少？”

还应该问：

“这个 0.86 有多稳定？”

可以：

.. code-block:: text

   Dataset
      ↓
   Bootstrap resample
      ↓
   Train / evaluate model
      ↓
   Metric*
      ↓
   Repeat B times
      ↓
   Distribution of metric

可以用于研究：

* Accuracy uncertainty
* AUC uncertainty
* F1 uncertainty
* Sensitivity uncertainty
* coefficient uncertainty
* feature-importance stability

Bootstrap 与 Bagging
--------------------

Bootstrap：

.. code-block:: text

   bootstrap datasets
          ↓
   estimate uncertainty

Bagging：

.. code-block:: text

   bootstrap datasets
          ↓
   train many models
          ↓
   aggregate predictions

因此：

.. math::

   \boxed{
   \text{Bootstrap = resampling principle}
   }

而：

.. math::

   \boxed{
   \text{Bagging = bootstrap + aggregation}
   }

Random Forest 中的 bootstrap sampling 就继承了这一思想。

Bootstrap 不是万能的
--------------------

Bootstrap 成功依赖多个条件：

#. :math:`\hat F` 是否能合理近似 :math:`F`；
#. statistic 是否 sufficiently regular；
#. resampling mechanism 是否符合 data-generating mechanism；
#. statistic 是否接近 pivotal；
#. sample size 是否足够；
#. confidence interval construction 是否合理；
#. dependence structure 是否得到保留。

因此不能简单认为：

.. code-block:: text

   I resampled the dataset 1000 times,
   therefore the inference must be correct.

这是错误的理解。

论文没有完整解决的问题
----------------------

这篇 1979 年论文并没有声称建立了 Bootstrap 的所有现代理论。

它主要通过：

* theoretical examples
* Taylor approximation
* finite-support asymptotics
* Monte Carlo simulations

展示 bootstrap 的可行性。

后来继续发展的内容包括：

* bootstrap consistency theory
* higher-order accuracy
* percentile bootstrap
* bootstrap-t
* BCa confidence interval
* block bootstrap
* cluster bootstrap
* wild bootstrap

这些属于后续发展，而不是这篇 1979 原论文全部完成的内容。

论文知识地图
------------

.. code-block:: text

   Efron (1979)
   Bootstrap Methods
   │
   ├── 1 Introduction
   │     ├── Jackknife limitations
   │     └── Bootstrap as broader framework
   │
   ├── 2 Bootstrap Methods
   │     ├── empirical distribution F-hat
   │     ├── sampling with replacement
   │     ├── bootstrap distribution
   │     ├── Method 1: exact
   │     ├── Method 2: Monte Carlo
   │     └── Method 3: Taylor expansion
   │
   ├── 3 Median
   │     ├── ordinary bootstrap
   │     ├── jackknife failure
   │     ├── symmetrized bootstrap
   │     └── smoothed bootstrap
   │
   ├── 4 Discriminant Analysis
   │     ├── classification error
   │     ├── optimism / bias
   │     ├── bootstrap error estimate
   │     └── comparison with CV
   │
   ├── 5 Relation to Jackknife
   │     ├── multinomial weights
   │     ├── Taylor expansion
   │     ├── infinitesimal jackknife
   │     ├── delta method
   │     └── ratio estimator
   │
   ├── 6 Wilcoxon
   │     ├── two-sample bootstrap
   │     ├── variance
   │     └── unbalanced jackknife
   │
   ├── 7 Regression
   │     ├── residual bootstrap
   │     ├── regression covariance
   │     └── preserve model structure
   │
   └── 8 Remarks
         ├── Monte Carlo implementation
         ├── transformations
         ├── pivotal quantities
         ├── confidence inference caveat
         ├── studentization
         ├── asymptotic validity
         ├── subsampling
         ├── grouped jackknife
         └── parametric bootstrap

最需要记住的 15 个知识点
-----------------------

#. Bootstrap 是一种 resampling method。
#. 使用 original sample 构建 empirical distribution :math:`\hat F`。
#. Bootstrap sample 是大小通常为 :math:`n` 的有放回抽样。
#. 重复 resampling 得到 :math:`T^{*1},\ldots,T^{*B}`。
#. 这些值组成 bootstrap distribution。
#. Bootstrap distribution 用来近似 sampling distribution。
#. Bootstrap 可以估计 bias、variance、standard error 和 uncertainty。
#. Bootstrap 不会产生新的真实 information。
#. Monte Carlo Bootstrap 是现代最常用版本。
#. Jackknife 可以看成 Bootstrap 的 Taylor/linear approximation。
#. Jackknife 对 non-smooth statistic，例如 median，可能失败。
#. Bootstrap 可以扩展到 one-sample、two-sample、classification 和 regression。
#. Regression resampling 必须尊重 model/data-generating structure。
#. Pivotal quantities 与 transformation 会影响 inference quality。
#. Bootstrap 并非万能，resampling design 和 assumptions 非常重要。

一句话总结
----------

.. important::

   如果真实实验无法被反复进行，就利用 observed data 构造一个替代总体，
   然后在计算机中反复模拟 sampling process，从而估计 statistic 的 sampling uncertainty。

最核心的理论关系：

.. math::

   \boxed{
   \text{Bootstrap}
   \supset
   \text{Jackknife as an approximation}
   }
"""

path = Path("/mnt/data/efron_bootstrap_1979_summary.rst")
path.write_text(rst.strip() + "\n", encoding="utf-8")

print(f"Created: {path}")
print(f"Lines: {len(rst.splitlines())}")
