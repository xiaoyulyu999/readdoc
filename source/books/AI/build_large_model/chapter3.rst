CHATPER 3
=========

3.1 The problem with modeling long sequence
-------------------------------------------

The reasons for using attention mechanisms in neural networks

.. image:: c3/1.png
   :alt: Description of the image
   :width: 600px

We will implement 4 different variants of attention machanism, These different attention variants build on each other. The goal is to arrive at a compact and efficient implementation fo multi-head attention that we can plug into the LLM.

.. image:: c3/2.png
   :alt: Description of the image
   :width: 600px
it’s not possible to merely translate word by word. Instead, the translation process requires contextual understanding and grammatical alignment.

.. image:: c3/3.png
   :alt: Description of the image
   :width: 600px

To address this problem, it is common to use a deep neural network with two submodules, an encoder and a decoder.

.. image:: c3/4.png
   :alt: Description of the image
   :width: 600px

Before the advent of transformer models, encoder–decoder RNNs were a popular choice for machine translation. The encoder takes a sequence of tokens from the source language as input, where a hidden state (an intermediate neural network layer) of the encoder encodes a compressed representation of the entire input sequence. Then, the decoder uses its current hidden state to begin the translation, token by token.

The big limitation of encoder–decoder RNNs is that the RNN can’t directly access earlier hidden states from the encoder during the decoding phase. Consequently, it relies solely on the current hidden state, which encapsulates all relevant information. This can lead to a loss of context, especially in complex sentences where dependencies might span long distances.

3.2 Cpaturing data dependencies with attention mechanisms
---------------------------------------------------------
.. image:: c3/5.png
   :alt: Description of the image
   :width: 600px

attention mechanism, the text-generating decoder part of the network can access all input tokens selectively. This means that some input tokens are more important than others for generating a given output token. The importance is determined by the attention weights

only three years later, researchers found that RNN architectures are not required for building deep neural networks for natural language processing and proposed the original transformer architecture.

Self-attention is a mechanism that allows each position in the input sequence to consider the relevancy of, or “attend to,” all other positions in the same sequence when computing the representation of a sequence. Self-attention is a key component of contemporary LLMs based on the transformer architecture, such as the GPT series.

.. image:: c3/6.png
   :alt: Description of the image
   :width: 600px

Self-attention is a mechanism in transformers used to compute more efficient input representations by allowing each position in a sequence to interact with and weight the importance of all other positions within the same sequence.

3.3 Attending to different parts of the input with self-attention.
------------------------------------------------------------------
The “self” refers to the mechanism’s ability to compute attention weights by relating different positions within a single input sequence.

3.3.1 A simple self-attention mechanism without trainable weights.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: c3/7.png
   :alt: Description of the image
   :width: 600px

The goal of self-attention is to compute a context vector for each input element that combines information from all other input elements. In this example, we compute the context vector z(2). The importance or contribution of each input element for computing z(2) is determined by the attention weights a21 to a2T. When computing z(2), the attention weights are calculated with respect to input element x(2) and all other inputs.

.. tip::
   Context vectors play a crucial role in self-attention. Their purpose is to create enriched representations of each element in an input sequence (like a sentence) by incorporating information from all other elements in the sequence.

.. code-block:: python

   import torch
   inputs = torch.tensor(
     [[0.43, 0.15, 0.89], # Your     (x^1)
      [0.55, 0.87, 0.66], # journey  (x^2)
      [0.57, 0.85, 0.64], # starts   (x^3)
      [0.22, 0.58, 0.33], # with     (x^4)
      [0.77, 0.25, 0.10], # one      (x^5)
      [0.05, 0.80, 0.55]] # step     (x^6)
   )

.. image:: c3/8.png
   :alt: Description of the image
   :width: 600px

.. admonition:: 🔰 1. The first step: Compute the intermediate values w (Attention scores)

   Why we not use the token embedding?
   0.87 is truncated to 0.8. In this truncated version, the embeddings of the words “journey” and “starts” may appear similar by random chance.

.. code-block:: python

   # calculate the intermediate attention scores between the query token and each input token.
   # We determine these scores by computing the dot product of the query, x(2), with every other input token:

   second_word_query_token = inputs[1] # the second word's token
   attention_score_to_second_word = torch.empty(inputs.shape[0]) # same shape with the first dim [6] of inputs [6, 3] but with uninitialized data
   for num_index, matrix in enumerate(inputs):
     attention_score_to_second_word[num_index] = torch.dot(matrix, second_word_query_token)

   attention_score_to_second_word

   #tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

.. tip::

   the dot product is a measure of similarity because it quantifies how closely two vectors are aligned: a higher dot product indicates a greater degree of alignment or similarity between the vectors. In the context of self-attention mechanisms, the dot product determines the extent to which each element in a sequence focuses on, or “attends to,” any other element: the higher the dot product, the higher the similarity and attention score between two elements.

.. admonition:: 🔰 2. The second step: Nomalize the weights.

   This normalization is a convvention that is useful for interpretation and maintaining training stability in an LLM.

.. image:: c3/9.png
   :alt: Description of the image
   :width: 600px

.. code-block:: python

   sum_of_weights = attention_score_to_second_word.sum()
   normalize_weights = attention_score_to_second_word / sum_of_weights
   normalize_weights

   # tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

.. tip::

   The results are bit difference.

   - tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
   - tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
   It becauses the Softmax takes exponation of each value. Softmax = exp(tensor_i) / sum (exp (tensor)) Make difference more obviously.

.. admonition:: 🔰 3. Calculating the context vector z(2)

   by **multiplying** the embedded input tokens x(i), with the corresponding weights and then **summing** the resulting vecetors.

.. image:: c3/10.png
   :alt: Description of the image
   :width: 600px

.. code-block:: python

   # Context vector for input 2
   # the context vector should have same shape with the query's shape.
   #从inputs encoded 变成 context vector, 目的是通过dim不变的情况下,让所有的vector之间通过weights产生连接.
   query = inputs[1]

   context_vector_v2 = torch.zeros(query.shape)
   for idx, matrix in enumerate(inputs):
     context_vector_v2 += matrix * softmax_result[idx]

   context_vector_v2

   # tensor([0.4419, 0.6515, 0.5683])

3.3.2 Computing attention weights for all input tokens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: c3/11.png
   :alt: Description of the image
   :width: 600px

.. tip::

   COMPUTE ATTENTION SCORES。=> matrix @ matrix.T

.. admonition:: 🔰 3-steps:

   - Token embedding for each of them.
   - Weight
   - Context Vector

.. image:: c3/12.png

.. code-block:: python

   # Dot product for each 1-1 2-1 3-1 ...2-1 2-2 3-2...6-6

   len_words = inputs.shape[0]
   attention_score_matrix = torch.empty(len_words, len_words)
   for row_index, row_value in enumerate(inputs):
     for col_index, cal_value in enumerate(inputs):
       attention_score_matrix[row_index, col_index] = torch.dot(row_value, cal_value)

   attention_score_matrix

   '''
   tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
   '''

   # Faster process for Dot product
   # matrix @ matrix.T
   attention_score_matrix = inputs @ inputs.T
   attention_score_matrix

   '''
   tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
   '''

   normalize_weights = torch.softmax(attention_score_matrix, dim=-1)
   normalize_weights

   '''
   tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
   '''

3.4 Implementing self-attention with trainable weights
------------------------------------------------------

.. image:: c3/13.png

3.4.1 Computing the attention weights step by step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: c3/14.png

.. code-block:: python

   inputs

   '''
   tensor([[0.4300, 0.1500, 0.8900],
           [0.5500, 0.8700, 0.6600],
           [0.5700, 0.8500, 0.6400],
           [0.2200, 0.5800, 0.3300],
           [0.7700, 0.2500, 0.1000],
           [0.0500, 0.8000, 0.5500]])

.. admonition:: 🔰 Trainable weight matrices: Wq, Wk, and Wv.

   In the first step of the self-attention mechanism with trainable weight matrices, we compute query (q), key (k), and value (v) vectors for input elements x. Similar to previous sections, we designate the second input, x(2), as the query input. The query vector q(2) is obtained via matrix multiplication between the input x(2) and the weight matrix Wq. Similarly, we obtain the key and value vectors via matrix multiplication involving the weight matrices Wk and Wv.

   Example, We start by computing only one context vector, z2, for illustration purposes. Just like the image.

.. admonition:: 🔰 Step - 1

   Compute the query, key and value vectors

.. code-block:: python


   '''
   the input and output dimensions are usually the same,
   but to better follow the computation, we’ll use different。
   '''
   x_2 = inputs[1]
   dim_in = inputs.shape[1]
   dim_out = 2

   '''
   Initialize Wq, Wk, Wv
   …'''
   Compute the query, key and value vectors, as showing in the image.
   '''
   query_2 = x_2 @ W_query
   key_2 = x_2 @ W_key
   value_2 = x_2 @ W_value

   key_2, W_query.shape

   # (tensor([0.4433, 1.1419]), torch.Size([3, 2]))

   keys = inputs @ W_key
   values = inputs @ W_value

   keys, values

   '''
   (tensor([[0.3669, 0.7646],
         [0.4433, 1.1419],
         [0.4361, 1.1156],
         [0.2408, 0.6706],
         [0.1827, 0.3292],
         [0.3275, 0.9642]]),
    tensor([[0.1855, 0.8812],
            [0.3951, 1.0037],
            [0.3879, 0.9831],
            [0.2393, 0.5493],
            [0.1492, 0.3346],
            [0.3221, 0.7863]]))
   '''

.. tip::

   Weight parameters vs. attention weights weight parameters are the fundamental, learned coefficients that define the network’s connections, while attention weights are dynamic, context-specific values.

   - Weight Parameters = model’s permanent knowledge
   - Attention Weights = model’s temporary focus on a specific input

.. admonition:: 🔰 Step - 2

   Computing the attention scores.

.. image:: c3/15.png

.. code-block:: python

   '''
   Attention score for x2
   '''
   w2 = query_2.dot(key_2)

   '''
   Generalize this computation to all attention scores via matrix mulitiplication:
   1*3 @ 3*6 = 1*6
   '''
   attention_score_2 = query_2 @ keys.T
   attention_score_2

   '''
   Go from attention scores to the attention Weights.
   Scale the attention scores by dividing them by the square root of the embedding dimension of the keys
   '''
   d_k = keys.shape[-1] # 2
   attention_weights_2 = torch.softmax(attention_score_2 / d_k ** 0.5, dim = -1)
   w2, attention_score_2, attention_weights_2

   '''
   (tensor(1.8524),
    tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]),
    tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]))
   '''

.. image:: c3/15.png

.. tip::

   *The rationale behind scaled-dot product attention*

   The reason for the normalization by the embedding dimension size is to improve the training performance by avoiding small gradients. For instance, when scaling up the embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large dot products can result in very small gradients during backpropagation due to the softmax function applied to them. As dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero. These small gradients can drastically slow down learning or cause training to stagnate.

   The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.

.. admonition:: 🔰 Final step

   Compute the context vectors.

.. image:: c3/16.png

.. code-block:: python

   context_vector = attention_weights_2 @ values
   context_vector, attention_weights_2.shape, values.shape

   #(tensor([0.3061, 0.8210]), torch.Size([6]), torch.Size([6, 2]))

.. danger::
   REMEMBER THIS IS ONLY FOR A SINGLE CONTEXT VECTOR z_2

.. admonition:: 🔰 Why Query, Key and Value?

   The terms “key,” “query,” and “value” in the context of attention mechanisms are borrowed from the domain of information retrieval and databases, where similar concepts are used to store, search, and retrieve information.

   - A query is analogous to a search query in a database. It represents the current item (e.g., a word or token in a sentence) the model focuses on or tries to understand. The query is used to probe the other parts of the input sequence to determine how much attention to pay to them.

   - The key is like a database key used for indexing and searching. In the attention mechanism, each item in the input sequence (e.g., each word in a sentence) has an associated key. These keys are used to match the query.

   - The value in this context is similar to the value in a key-value pair in a database. It represents the actual content or representation of the input items. Once the model determines which keys (and thus which parts of the input) are most relevant to the query (the current focus item), it retrieves the corresponding values.

3.4.2 Implementing a compact self-attention Python class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   class SelfAttention_v1(nn.Module):
     def __init__(self, dim_in, dim_out) -> None:
       super().__init__()
       self.W_query = nn.Parameter(torch.rand(dim_in, dim_out))
       self.W_key = nn.Parameter(torch.rand(dim_in, dim_out))
       self.W_value = nn.Parameter(torch.rand(dim_in, dim_out))

     def forward(self, query_focus):
       keys = query_focus @ self.W_key
       queries = query_focus @ self.W_query
       values = query_focus @ self.W_value
       attention_scores = queries @ keys.T
       attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim = -1)
       context_vec = attention_weights @ values
       return context_vec

   '''
   Let's check
   '''
   dim_in, dim_out
   torch.manual_seed(123)
   v1_module = SelfAttention_v1(dim_in, dim_out)
   v1_module(inputs)

   '''
   tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
   '''
Steps:

- Input vectors -> Input Matrix X [tiktoken, window slider, Token embedding]
- Wq, Wk, Wv. query_x @ [torch.nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=False)] ->  Q, K, V
- attention_score = Q@ K.T
   -> attention_weight = torch.softmax(attention_score / keys.shape[-1] ** 0.5, dim=-1)

   -> context_vector = attention_weight @ V

.. admonition:: A self-attention class using Pytorch linear layers:

   .. code-block:: python

      class SelfAttention_v2(nn.Module):
        def __init__(self, d_in, d_out, qkv_bias=False):
          super().__init__()
          self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
          self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
          self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x):
          keys = self.w_key(x)
          queries = self.w_query(x)
          values = self.w_value(x)
          attention_scores = queries @ keys.T
          attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
          context_vecter = attention_weights @ values
          return context_vecter

      # class SelfAttention_v1(nn.Module):
      #   def __init__(self, dim_in, dim_out) -> None:
      #     super().__init__()
      #     self.W_query = nn.Parameter(torch.rand(dim_in, dim_out))
      #     self.W_key = nn.Parameter(torch.rand(dim_in, dim_out))
      #     self.W_value = nn.Parameter(torch.rand(dim_in, dim_out))

      #   def forward(self, query_focus):
      #     keys = query_focus @ self.W_key
      #     queries = query_focus @ self.W_query
      #     values = query_focus @ self.W_value
      #     attention_scores = queries @ keys.T
      #     attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim = -1)
      #     context_vec = attention_weights @ values
      #     return context_vec

      torch.manual_seed(789)
      sa_v2 = SelfAttention_v2(dim_in, dim_out)
      out_put = sa_v2(inputs)
      out_put

      '''
      tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
      '''

.. admonition:: 🔰 Compare SelfAttention_v1 and v2:

   Note that nn.Linear in SelfAttention_v2 uses a different weight initialization scheme as nn.Parameter(torch.rand(d_in, d_out)) used in SelfAttention_v1, which causes both mechanisms to produce different results. To check that both implementations, SelfAttention_v1 and SelfAttention_v2, are otherwise similar, we can transfer the weight matrices from a SelfAttention_v2 object to a SelfAttention_v1, such that both objects then produce the same results.

.. code-block:: python

   '''
   our task is to correctly assign the weights from an instance of SelfAttention_v2
   to an instance of SelfAttention_v1. To do this, you need to understand the
   relationship between the weights in both versions. (Hint: nn.Linear stores the
   weight matrix in a transposed form.) After the assignment, you should observe
   that both instances produce the same outputs.
   '''

   def transfer_weights(sa_v2, sa_v1):
       # Transfer the weights from sa_v2 to sa_v1
       with torch.no_grad(): # tell the model this part not for backpropagation
           sa_v1.W_query.copy_(sa_v2.w_query.weight.T)
           sa_v1.W_key.copy_(sa_v2.w_key.weight.T)
           sa_v1.W_value.copy_(sa_v2.w_value.weight.T)
   # Initialize both modules
   sa_v2 = SelfAttention_v2(dim_in, dim_out)
   sa_v1 = SelfAttention_v1(dim_in, dim_out)

   # Transfer weights
   transfer_weights(sa_v2, sa_v1)

   # Test on some input
   out1 = sa_v1(inputs)
   out2 = sa_v2(inputs)

   # Check closeness
   print(torch.allclose(out1, out2, atol=1e-6))  # Should print: True

.. tip::

   Attention(Q,K,V)=softmax(Q⋅K.T/√dk)⋅V


