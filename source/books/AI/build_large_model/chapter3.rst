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

.. admonition:: 1. The first step: Compute the intermediate values w (Attention scores)

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

.. admonition:: 2. The second step: Nomalize the weights.

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

.. admonition:: 3. Calculating the context vector z(2)

   by **multiplying** the embedded input tokens x(i), with the corresponding weights and then **summing** the resulting vecetors.

.. image:: c3/10.png
   :alt: Description of the image
   :width: 600px