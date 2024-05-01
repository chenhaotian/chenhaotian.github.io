---
title: "Breaking Down Transformers: The Core of Attention as Low-Rank Approximation"
header:
  overlay_image: /assets/post_images/attention-low-rank.jpg
  overlay_filter: 0.5
categories:
  - Models
tag:
  - transformer
  - attention layer
  - low-rank approximation
  - cosine similarity
  - reconstruction error
---

>  Why attention? How does it work?

These are likely the first questions that come to mind when someone first learns about Transformers. While there's a wealth of literature examining the mechanisms of the attention layer, the **low-rank approximation** perspective has been touched upon but not extensively explored. In this article, we'll delve into this perspective in hopes of providing a clearer understanding of the underlying mechanisms of the Transformer. By viewing attention through the lens of low-rank approximation, we aim to uncover insights that could lead to potential optimizations and innovations.



## Reconstruction Error: Where It All Began

Generative models have popped up in a ton of cool applications, each one trying to uncover deeper connections among real-world elements. Nearly all these models focus on one main goal during training: keep the reconstruction error as low as possible. Essentially, they aim to minimize the mismatch between the real entity and its generated counterpart. Here's what that looks like in <a name="examples">practice</a>:

+ **Machine Translation**: Imagine you want to translate a text, say passage $A$, from one language to another to get passage $B$. If the model spits out passage $B'$ based on $A$, the loss function typically involves a special kind of distance measurement between $B$ and $B'$. This distance acts as a way to quantify how off the mark $B'$ is from being a perfect reconstruction of $B$.

+ **Text to Image Generation**: Following the same thread, when you're generating images from text, the loss function usually boils down to how accurately the generated (reconstructed) image mirrors the intended real image. The smaller the discrepancy, the better the model is at "imagining" the right visuals from the text input.

+ **Text to Text Generation**: Similarly, one may expect the generated answer given a question well represent the actual answer from the training dataset.

In the quest to minimize reconstruction error, a model must typically nail two key things to be successful. The first is to **capture relations**, where it needs a method to grasp the complex relationships between real-world entities. Then it should be able to **reconstruction**, that is to craft new entities from an input, drawing on the relationships it has learned between that input and other entities.

>  Then how does the attention layer manage to meet the two key points?

To simplify things, let's zero in on the **self-attention layer**. It **captures relationships through cosine similarities**, which effectively measure how closely different elements are related. Then, it **reconstructs the output through kernel smoothing**, which in essence is weighted average of other entities, weights defined upon the learned cosine similarities.

Below, we'll dive into how the self-attention layer pinpoints relationships using cosine similarities. The kernel smoothing part is pretty straightforward and doesn’t need much detail—it flows naturally from what we discuss. First up, let’s explore low-rank approximation, setting the stage for our exploration of cosine similarities in attention layers.



## Low-Rank Approximation, Why and How?

>  Actually, putting the backstory aside,  low-rank approximation tells us that **any real matrix can be seen as a asymmetric cosine similarity matrix**.

Instead of trotting out the same old textbook definition, I want to dive into what really makes low-rank approximation tick, especially when it comes to its role in deep neural network applications. 

As we've outlined above, our goal is to find a mechanism that can encode the relationships between a set of entities, assuming these entities are already represented as vectors of the same dimension $d$, and there are $n$ such entities. For simplicity let's use $X_{n\times d}$ to denote the set where each row is an entity. We expect this mechanism to

1. <a name="req1">Captures relationships using cosine similarities</a>, since the entities are already encoded into $d$ dimensional vectors, using cosine similarity is a natural choice,
2. <a name="req2">Allow for asymmetric similarities</a>, because, for example when the entities are words, we may expect the word "apple" to be more similar to "fruit" since every apple is a fruit, while "fruit" may be less similar to "apple" because there can be many other kind of fruits,
3. <a name="req3">Seamlessly integrate into a neural network trained with backpropagation</a>, and
4. <a name="req4">Guarantee scalability</a>

With these objectives in focus, let's revisit the concept of low-rank approximation. We'll then detail how it fulfills these specific requirements.

A practical approach to encoding the cosine similarities for a matrix $ X_{n \times d} $ is by using an $ n \times n $ matrix, denoted as $ S_{n \times n} $. In this matrix, each element $ s_{ij} $ represents the cosine similarity between entity $i$ and $j$. We allow for $ s_{ij} \neq s_{ji} $, meaning that $ S $ can be asymmetric, aligning with [requirement 2](#req2). To efficiently handle this, we aim to decompose $ S $ into two lower-rank matrices $U_{n \times k}$ and $V_{n\times k}$. where $k$ represents the rank of the approximation. This can be expressed as:


$$
\begin{align}
S \approx UV^T \label{lowrank}
\end{align}
$$


Equation $\eqref{lowrank}$ demonstrates that the matrix $ S $ is constructed using the cosine similarities between the "row space" coordinates in $ U $ and the "column space" coordinates in $ V $. **This construction assumes that the same entity $x_i$, as the $i$-th row from $X$ exists in two different spaces, represented by coordinates $ u_i $ and $ v_i $ respectively, where $ u_i $ and $ v_i $ are the $ i $-th rows of $ U $ and $ V $.** This approach fulfills [requirement 1](#req1) by ensuring that relationships are measured using these similarities.

$U$ and $V$ can also be learned through gradient descent. For example in the simplest case where $S$ is known beforehand, we would then like to minimize the following reconstruction error:


$$
\begin{align}
\{U^*,V^*\} = \text{argmin}_{U,V} \frac{1}{2}\sum_{ij}(S - UV^T)^2 \label{simpleloss}
\end{align}
$$


The gradients are given by:
$$
\begin{align}
\frac{\partial J}{\partial U} = (UV^T - S)V\\
\frac{\partial J}{\partial V} = (UV^T - S)^TU
\end{align}
$$
Then, $U$ and $V$ are updated as follows:

$$
\begin{align}
U \leftarrow U - \alpha \frac{\partial J}{\partial U}\\
V \leftarrow V - \alpha \frac{\partial J}{\partial V}
\end{align}
$$
where $\alpha$ is the gradient descent learning rate.

In real-world deep neural network applications, we don't directly access the actual similarity matrix $S$, and the loss functions aren't always defined using the $ L2 $ norm as shown in $\eqref{simpleloss}$. However, as highlighted in the [examples](#examples), all the loss functions essentially boil down to reconstruction errors between the model's output and some target output, the target output can simply be some local content without impacting the overall differentiability. Furthermore, due to the constraints imposed by backpropagation, the loss functions must be differentiable with respect to the model parameters, whether they involve the $ L2 $ norm or not. Hence one can embed $\eqref{lowrank}$ directly in a proven nonlinear function, such as softmax or ReLU, and let it serve as a layer in a deep neural network and guarantees local optimum learned for $U$ and $V$. Thereby fulfilling [requriement 3](#req3).

Up until now, the first three requirements have been addressed using the standard low-rank approximation method, $\eqref{lowrank}$. To meet requirement 4, we introduce an "extended" low-rank approximation approach. It's important to note that treating $ U $ and $ V $ as learnable parameters would limit the scalability of the neural network and restrict its applicability beyond the original training data, since $ U $ and $ V $ are specific to the data in $ X $. To ensure the low-rank approximation is effective for any input, we modify the standard method by having it learn a transformation of $ X $ instead of directly learning the factorization. The "extended" low-rank approximation is defined as follows:


$$
\begin{align}
S &\approx U_{X}V_{X}^T \label{lowrank2} \\
\text{Where } & U_X = XW_{d\times k}^{U}\\
&V_X = XW_{d\times k}^{(V)}
\end{align}
$$


In applications, $k$ is selected in advance and typically satisfies $k\le d$. $\eqref{lowrank2}$ is the same as $\eqref{lowrank}$ except that the learnable parameters are $U$ and $V$ in $\eqref{lowrank}$, while in $\eqref{lowrank2}$ they become $W_{(U)}$ and $W_{(V)}$. With $\eqref{lowrank2}$, we are able to reuse $W_{(U)}$ and $W_{(V)}$ to get the low-rank approximation of a different data set, say $Y_{m\times d}$, without retrain the model. The decompositions are simply $U_Y = Y W^{(U)}$ and $V_Y = Y W^{(V)}$.  Thus [requirement 4](req4) is fulfilled.

In conclusion, we can see that the expression
$$
\begin{align}
U_{X}V_{X}^T = XW^{U} (XW^{(V)})^T, \label{mp}
\end{align}
$$
which consists of the input $X$ and parameters $W^{(U)}$ and $W^{(V)}$, can be used as a layer in a neural network directly or within a proven nonlinear transformation. Such that this layer can capture the possibly asymmetric cosine similarities between entities in $X$, supports gradient update on it's learnable parameters and is scalable over different size of input data (as listed in requirements [1](#req1) through [4](#req4)). By now, you can probably see where this is leading—yes, this formulation is exactly how the matrix product part of the attention layer is defined.



## Attention Layer and It's Hidden Assumptions

Attention layer, despite its somewhat misleading name, **is based on the assumption that there exists an unknown and asymmetric similarity matrix that captures the relationships between input entities**, the input entities are represented as a token matrix $X_{n \times d}$ where each row corresponds to a token of an entity. The essence of the attention layer is defined by the following equations:


$$
\begin{align}
Y &= \text{softmax}(U_XV_X^T)X \label{attention} \\
&U_X = XW^{(U)} \\
&V_X = XW^{(V)} \\
\end{align}
$$


It's worth noting that the right-hand side of $\eqref{attention}$ typically includes an additional parameter $W^{(Z)}$ and scaling factor $\sqrt{d_k}$ in the literature, expressed as $Y = \text{softmax}(\frac{UV^T}{\sqrt{d_k}})XW^{(Z)}$. I've omitted $W^{(Z)}$ and $\sqrt{d_k}$ here to focus on the core concepts.

Since the attention layer operates under the same assumptions and requirements discussed in the context of low-rank approximation. The significance of $U_XV_X^T$ in $\eqref{attention}$ is pretty straightforward, as detailed underneath $\eqref{mp}$.

The softmax operation applied to the low-rank approximation can be viewed as a kernel function that measures the similarity between vectors. The output of the softmax function gives a set of weights that are used to compute a weighted sum of the values. This is in essense a kernel smoothing process because:

+ **Each output token is a smooth combination of other tokens**, where the weights are non-negative and sum to one due to the softmax.
+ **The low-rank approximation mechanism adapts the weights based on the input itself**, meaning that it smooth the data according to the approximated cosine similarities among the input entities.



Thus concluded our discussion. As a recap, we discussed how the self-attention layer in Transformers encapsulates relationships between entities using cosine similarities and enhances these computations through a framework of low-rank approximations. This approach not only efficiently captures essential interactions within the input tokens but also ensures the model remains computationally manageable.



## Citation

For attribution in academic contexts, please cite this work as:

> Chen, H. (2024). Breaking down transformers: the core of attention as Low-Rank approximation. *Probabilistic Nexus*. https://chenhaotian.github.io/models/attention-as-low-rank-approximation/

BibTeX citation:

```latex
@misc{chen2024attentionlowrank,
	author = {Chen, Haotian},
	title = {Breaking down transformers: the core of attention as Low-Rank approximation},
	year = {2024},
	url = {https://chenhaotian.github.io/models/attention-as-low-rank-approximation/},
}
```









