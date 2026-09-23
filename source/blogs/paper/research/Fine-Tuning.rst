================================================================================================
Research Proposal: Parameter-Efficient Adaptation and XAI Diagnostics for Inverse Protein Design
================================================================================================

.. meta::
   :description: Research proposal draft on LoRA adaptation, catastrophic forgetting mitigation, and XAI diagnostics for ProteinMPNN.
   :keywords: ProteinMPNN, LoRA, PEFT, Catastrophic Forgetting, CKA, Captum, ESMFold

.. contents:: Table of Contents
   :depth: 2
   :local:

Executive Summary
=================

本研究计划旨在解决基于深度学习的逆向蛋白质设计模型在特定下游生物物理性质适配时的结构退化问题。作为反向折叠领域的代表性工作，**ProteinMPNN (Dauparas et al., Science, 2022)** 具备强大的零样本（Zero-shot）3D 几何表征与骨架折叠能力[cite: 2, 3]。然而，当面向极端热稳定性或宿主表达偏好进行任务微调时，传统的全参数微调极易破坏预训练模型对空间拓扑的记忆，引发灾难性遗忘。

本项目提出采用**低秩自适应（Low-Rank Adaptation, LoRA）**对 ProteinMPNN 实施参数高效微调（PEFT），在完全冻结 3D 结构编码器的前提下，仅微调解码器前馈层极小比例（< 0.2%）的参数。同时，引入**中心核对齐（CKA）**与**积分梯度（Captum Integrated Gradients）**构建双重可解释性（XAI）特征诊断流水线，并结合 **ESMFold** 回测系统，建立“参数轻量注入 - 隐层表征对齐 - 特征归因诊断 - 结构自洽验证”的完整闭环评估体系。

.. figure:: _static/framework_overview.png
   :align: center
   :alt: Methodological Framework Overview

   Figure 1: End-to-End Architectural Pipeline for LoRA Adaptation, XAI Diagnostics, and ESMFold Validation.

Research Objectives & Core Deliverables
=======================================

本项目计划达成以下四个维度的具体目标：

1. **Parameter-Efficient Adaptation Engine**：构建非侵入式 LoRA 注入模块，实现仅更新 < 0.2% 参数即可使序列具备目标下游性质。
2. **Data-Leakage Free Pipeline**：建立严格的时间截断与同源去冗余数据集划分规范，保证测试样本对预训练模型完全未见。
3. **Multi-Scale XAI Diagnostic Protocol**：利用 CKA 与 Captum 形成宏观矩阵相似度与微观原子归因的双尺度抗遗忘验证探针。
4. **Self-Consistent Foldability Verification**：依托 ESMFold 计算 pLDDT 与 RMSD 指标，实现在硅（In Silico）结构折叠验证闭环。

Data Curation & Homology Separation Protocol
=============================================

为规避数据泄露（Data Leakage）并确保评价体系的绝对客观，下游微调与测试数据集需执行严格的生物信息学过滤：

* **Temporal Split**：仅采用 2021 年之后在 PDB 数据库释放的全新蛋白质三维结构（Post-2021 PDB releases），与预训练切片完全隔离。
* **Sequence Identity De-redundancy**：使用 MMseqs2 进行全局序列聚类，剔除与 ProteinMPNN 原生训练集相似度大于 30% 的所有同源链（Homology threshold < 30%）。
* **Data Serialization**：将三维原子坐标统一处理为包含 ``N, CA, C, O`` 四原子点云的定长 ``.jsonl`` 结构字典。

.. code-block:: text

   # Data Protocol Specification
   Raw PDB Files (Post-2021) ──► MMseqs2 (Cluster < 30% Identity) ──► JSONL Data Serialization

Experimental Methodology & Architecture Design
==============================================

1. Model Configuration & Controlled Baselines
---------------------------------------------

实验设计包含三组严格受控的对比模型：

.. list-table:: Experimental Comparison Matrix
   :widths: 25 20 25 30
   :header-rows: 1

   * - Model Designation
     - Trainable Parameters
     - Adaptation Strategy
     - Scientific Role
   * - **Zero-shot Baseline**
     - 0 (0.00%)
     - Completely Frozen
     - Performance Benchmark[cite: 2]
   * - **Full Fine-Tuning**
     - ~16.2M (100.0%)
     - Global Backpropagation
     - Catastrophic Forgetting Negative Control
   * - **LoRA Adaptation (Ours)**
     - ~15K (< 0.20%)
     - Decoder FFN Injection
     - Proposed Mitigation Strategy

2. LoRA Injection Implementation
--------------------------------

在 ``protein_mpnn_utils.py`` 的解码器前馈层 ``PositionWiseFeedForward`` 中注入低秩增量矩阵：

.. code-block:: python

   import math
   import torch
   import torch.nn as nn

   class LoRALinear(nn.Module):
       """
       LoRA Adapter for Feed-Forward Networks in ProteinMPNN Decoder.
       Decomposes weight updates: delta_W = (B * A) * (alpha / r)
       """
       def __init__(self, original_linear: nn.Linear, r: int = 4, lora_alpha: float = 16.0, lora_dropout: float = 0.1):
           super().__init__()
           self.original_linear = original_linear

           # Freeze base pre-trained weights
           self.original_linear.weight.requires_grad = False
           if self.original_linear.bias is not None:
               self.original_linear.bias.requires_grad = False

           self.r = r
           self.scaling = lora_alpha / r

           # Low-rank learnable parameter matrices
           self.lora_A = nn.Parameter(torch.zeros(r, original_linear.in_features))
           self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, r))
           self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()

           # Weight initialization: Kaiming uniform for A, zero initialization for B
           nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
           nn.init.zeros_(self.lora_B)

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           base_output = self.original_linear(x)
           lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
           return base_output + lora_output

Multi-Scale XAI Diagnostic Framework
====================================

本项目从宏观表示矩阵相似度与微观残基特征敏感度两个层级，对微调后的模型进行“病理切片式”诊断：

1. Macro-Level Representation Analysis (Layer-Wise CKA)
-------------------------------------------------------

计算冻结基准模型与微调模型各层隐藏特征激活向量 :math:`X` 与 :math:`Y` 之间的中心核对齐得分（CKA）：

.. math::

   \text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}

* **评价指标**：CKA 取值范围为 :math:`[0, 1]`。若 LoRA 模型的各层 CKA 得分持续稳定在 :math:`> 0.85`，则定量证明内部三维拓扑特征表征未发生结构性溃退。

2. Micro-Level Feature Attribution (Captum Integrated Gradients)
----------------------------------------------------------------

采用 PyTorch 官方可解释性框架 **Captum**，针对输入的三维骨架原子坐标张量 :math:`X \in \mathbb{R}^{B \times L \times 4 \times 3}` 计算逐残基积分梯度：

.. math::

   \text{Attr}_i(x) = (x_i - x_i') \times \int_{0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} \, d\alpha

* **评价指标**：分析高贡献度残基与目标位点之间的三维空间欧氏距离分布，验证微调模型是否保留了对空间物理微环境的敏感性。

Structural Foldability Evaluation (ESMFold Pipeline)
====================================================

利用正向折叠工具 **ESMFold** 对模型设计的氨基酸序列进行自洽性结构回测：

.. code-block:: text

   Input 3D Backbone ──► ProteinMPNN Sampling ──► Designed Sequences ──► ESMFold Structure Prediction
                                                                                   │
                                                                                   ▼
   Evaluation Metrics: pLDDT (Confidence) & Backbone RMSD (Alignment) ◄────────────┘

* **Predicted Local Distance Difference Test (pLDDT)**：评估序列折叠的局部二级结构规整度（目标阈值：``pLDDT > 80``）。
* **Root-Mean-Square Deviation (RMSD)**：将预测结构原子与真实天然骨架进行三维刚体叠合（Superimposition），评估折叠准确度（目标阈值：``RMSD < 2.0 Å``）。

Work Breakdown Structure & Research Roadmap
===========================================

.. list-table:: Project Execution Milestones
   :widths: 15 35 30 20
   :header-rows: 1

   * - Phase
     - Milestone Description
     - Core Toolchain
     - Status
   * - **Phase 1**
     - Source Code Audit & LoRA Prototype Injection
     - PyTorch, ``protein_mpnn_utils.py``
     - Completed
   * - **Phase 2**
     - Benchmark Dataset Curation (< 30% Identity)
     - MMseqs2, PDB REST API
     - In Progress
   * - **Phase 3**
     - Adaptation Training & Ablation Verification
     - CUDA, LoRA, CosineAnnealingLR
     - Pending
   * - **Phase 4**
     - Dual-Scale XAI Diagnostic Profiling
     - Captum (IG), Centered Kernel Alignment (CKA)
     - Pending
   * - **Phase 5**
     - ESMFold Back-folding & Metric Analytics
     - ESMFold, PyMOL, BioPython
     - Pending

References
==========

.. [Dauparas2022] Dauparas, J., et al. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49-56[cite: 3].
.. [Hu2021] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
.. [Kornblith2019] Kornblith, S., et al. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*.
.. [Sundararajan2017] Sundararajan, M., et al. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*.