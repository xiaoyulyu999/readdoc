============================================
ProteinMPNN 源码架构解析：前置数据与文件解析
============================================

本章节针对 ``protein_mpnn_utils.py`` 的底层源码实现展开逐行深度解析[cite: 7]。主要涵盖基础依赖引入、FASTA 序列解析器、交叉熵指标量化评估以及底层 PDB 坐标哈希抽取字典。

.. contents:: 目录导航
   :depth: 2
   :local:

模块 1：环境依赖与全局导入 (Lines 1–15)
=======================================

该模块导入了 Python 系统底层、数值矩阵运算以及 PyTorch 深度图学习所需的核心库[cite: 7]。

.. code-block:: python

   # ==========================================
   # Part 1: Dependency Imports & Environment
   # ==========================================
   from __future__ import print_function
   import json, time, os, sys, glob
   import shutil
   import numpy as np
   import torch
   from torch import optim
   from torch.utils.data import DataLoader
   from torch.utils.data.dataset import random_split, Subset

   import copy
   import torch.nn as nn
   import torch.nn.functional as F
   import random
   import itertools

   # Reference origin: Adopted from Ingraham et al., NeurIPS 2019 (Generative Models for Graph-Based Protein Design)

**核心机制拆解**：

* **系统与数据管线支持**：``json`` 负责解析 ``.jsonl`` 格式的蛋白质结构预处理数据；``DataLoader`` 与 ``Subset`` 提供可扩展的批次分发与数据集划分能力[cite: 7]。
* **张量计算与图网络核心**：基于 ``torch``、``torch.nn`` 与 ``F`` 构建三维图神经网络（GNN）与因果自回归解码器[cite: 7]。
* **算法来源溯源**：该实现的图特征嵌入思想脱胎于 NeurIPS 2019 经典蛋白质设计架构[cite: 7]。

模块 2：FASTA 序列文件解析器 (Lines 17–33)
==========================================

``parse_fasta`` 函数负责从标准的 FASTA 生物序列文本中读取蛋白质 ID 并重组为连续字符数组[cite: 7]。

.. code-block:: python

   # ==========================================
   # Part 2: FASTA Sequence Parsing Utility
   # ==========================================
   def parse_fasta(filename, limit=-1, omit=[]):
       header = []       # Header buffer: Stores protein identifiers (lines starting with '>')
       sequence = []     # Sequence buffer: Stores multi-line amino acid strings
       lines = open(filename, "r")
       for line in lines:
           line = line.rstrip()           # Strip trailing whitespace and newlines
           if line[0] == ">":            # Check for FASTA record header identifier
               if len(header) == limit:  # Terminate early if maximum sequence count is reached
                   break
               header.append(line[1:])   # Remove leading '>' character and record header
               sequence.append([])       # Initialize an empty segment list for the current entry
           else:
               if omit:                  # Filter out user-specified unwanted characters/residues
                   line = [item for item in line if item not in omit]
                   line = ''.join(line)
               line = ''.join(line)
               sequence[-1].append(line) # Append current sequence line chunk to active record
       lines.close()
       sequence = [''.join(seq) for seq in sequence] # Concatenate multi-line segments into full sequences
       return np.array(header), np.array(sequence)   # Return paired NumPy string arrays

**实现逻辑说明**：

1. **FASTA 格式识别**：通过判断每行首字符 ``line[0] == ">"`` 区分蛋白质元数据头部与具体的氨基酸序列片段[cite: 7]。
2. **截断机制与字符过滤**：通过 ``limit`` 参数控制批量测试时的数据吞吐量上限，借助 ``omit`` 参数过滤掉非天然或非法字符[cite: 7]。
3. **序列无损重组**：将由于行宽限制而换行断开的序列分片拼接恢复为完整长字符串[cite: 7]。

模块 3：损失函数与离散标签解码 (Lines 35–48)
============================================

本模块实现了模型对数似然度量与离散 Token 到天然氨基酸单字母的还原映射[cite: 7]。

.. code-block:: python

   # ==========================================
   # Part 3: Evaluation Metrics & Sequence Mapping
   # ==========================================
   def _scores(S, log_probs, mask):
       """
       Computes per-residue negative log-likelihood (NLL) scores under valid sequence mask.
       Args:
           S: Ground truth sequence indices [B, L]
           log_probs: Log probability distributions over 21 amino acids [B, L, 21]
           mask: Binary tensor indicating valid non-padded residue positions [B, L]
       """
       criterion = torch.nn.NLLLoss(reduction='none') # Element-wise loss preservation
       loss = criterion(
           log_probs.contiguous().view(-1, log_probs.size(-1)), # Flattened logits: [B*L, 21]
           S.contiguous().view(-1)                              # Flattened ground truth: [B*L]
       ).view(S.size())                                         # Reshaped back to [B, L]
       scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1) # Normalized average per sequence
       return scores

   def _S_to_seq(S, mask):
       """
       Decodes discrete integer token indices back into a 1-letter amino acid string.
       Maps index 0-20 to natural amino acids, with 'X' denoting unknown/masked positions.
       """
       alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
       seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
       return seq

**技术要点**：

* **``_scores`` 的掩码交叉熵**：采用 ``NLLLoss(reduction='none')`` 保持独立误差，通过逐元素乘以 ``mask`` 将填充的无意义 Pad 位点屏蔽剔除，并在样本有效长度上求平均[cite: 7]。
* **``_S_to_seq`` 词表转换**：依据 ``alphabet = 'ACDEFGHIKLMNPQRSTVWYX'`` 字典，将模型输出的高维离散索引映射回 20 种天然氨基酸字母及占位符 ``X``[cite: 7]。

模块 4：底层 PDB 坐标抽取与哈希映射 (Lines 50–70)
=================================================

``parse_PDB_biounits`` 是读取底层原子坐标、构建图节点输入特征的最前端接口[cite: 7]。

.. code-block:: python

   # ==========================================
   # Part 4: Coordinate Parsing from Raw PDB Text
   # ==========================================
   def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
       '''
       Extracts 3D Cartesian coordinates and primary sequences from raw PDB text lines.
       Args:
           x: Path to local .pdb file
           atoms: Subset of backbone atoms to parse (default: N, CA, C)
           chain: Specific chain ID to filter, or None for all chains
       Returns:
           xyz: Coordinate tensor [L, len(atoms), 3]
           sequence: Extracted amino acid sequence
       '''
       alpha_1 = list("ARNDCQEGHILKMFPSTWYV-") # 1-letter amino acid vocabulary with gap '-'
       states = len(alpha_1)
       alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                  'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP'] # 3-letter standard mapping

       # Bi-directional mapping hash tables
       aa_1_N = {a:n for n,a in enumerate(alpha_1)}
       aa_3_N = {a:n for n,a in enumerate(alpha_3)}
       aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
       aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
       aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

**技术要点**：

* 构建了包含天然氨基酸单字母、三字母缩写与整数标号之间的双向查找表（Hash Tables），为后续从 PDB 文本中正则匹配主链原子（:math:`N, C_\alpha, C, O`）奠定底层对齐基础[cite: 7]。