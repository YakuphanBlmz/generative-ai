# A Deep Dive into the Transformer Architecture

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Genesis of the Transformer](#2-the-genesis-of-the-transformer)
- [3. Core Components of the Transformer Architecture](#3-core-components-of-the-transformer-architecture)
  - [3.1. Encoder-Decoder Structure](#31-encoder-decoder-structure)
  - [3.2. Self-Attention Mechanism](#32-self-attention-mechanism)
  - [3.3. Multi-Head Attention](#33-multi-head-attention)
  - [3.4. Positional Encoding](#34-positional-encoding)
  - [3.5. Position-wise Feed-Forward Networks](#35-position-wise-feed-forward-networks)
  - [3.6. Residual Connections and Layer Normalization](#36-residual-connections-and-layer-normalization)
- [4. Training and Inference](#4-training-and-inference)
- [5. Advantages and Impact](#5-advantages-and-impact)
- [6. Code Example: Scaled Dot-Product Attention](#6-code-example-scaled-dot-product-attention)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

The **Transformer architecture**, introduced in the seminal 2017 paper "Attention Is All You Need" by Vaswani et al., has revolutionized the field of **Natural Language Processing (NLP)** and, more broadly, **Generative Artificial Intelligence (AI)**. Prior to its advent, recurrent neural networks (RNNs) and convolutional neural networks (CNNs), particularly Long Short-Term Memory (LSTM) networks, were the dominant paradigms for sequence-to-sequence tasks. While effective, these models often struggled with capturing long-range dependencies efficiently and were inherently sequential, limiting parallelization during training. The Transformer architecture elegantly addresses these limitations by entirely forsaking recurrence and convolutions, relying instead solely on a mechanism called **self-attention**. This innovative approach has paved the way for the development of highly powerful and scalable models, including foundational **Large Language Models (LLMs)** like GPT, BERT, and T5, significantly advancing the capabilities of generative AI systems across diverse applications from machine translation to text generation and beyond. This document delves into the intricate components and operational principles that define the Transformer architecture, elucidating its profound impact on modern AI.

<a name="2-the-genesis-of-the-transformer"></a>
## 2. The Genesis of the Transformer

Before the Transformer, sequence processing models, such as **Recurrent Neural Networks (RNNs)** and their variants like **LSTMs** and **Gated Recurrent Units (GRUs)**, processed input sequences token by token. This sequential processing created a bottleneck, as computations for one token largely depended on the preceding ones. This inherent sequentiality limited parallelization and, consequently, the speed of training, especially for very long sequences. Furthermore, while LSTMs aimed to mitigate the **vanishing gradient problem**, they still struggled with maintaining long-range dependencies across extremely extended sequences, often suffering from information decay over time.

**Convolutional Neural Networks (CNNs)** offered some parallelization advantages by processing fixed-size local contexts, but capturing global dependencies required stacking many layers, increasing computational depth and potentially diluting local context. The **attention mechanism**, first introduced in the context of neural machine translation with RNNs, partially alleviated these issues by allowing the model to focus on relevant parts of the input sequence when generating each output token. However, this attention was still applied on top of a recurrent structure.

The "Attention Is All You Need" paper proposed a radical departure: eliminate recurrence and convolution entirely and build a model solely based on attention mechanisms. This design decision was motivated by the desire to achieve greater parallelization, handle longer dependencies more effectively, and simplify the overall architecture. The Transformer's ability to compute attention for all parts of the sequence simultaneously, coupled with its fixed-depth structure, enabled unprecedented gains in training efficiency and model capacity, marking a pivotal moment in the history of neural networks for sequence modeling.

<a name="3-core-components-of-the-transformer-architecture"></a>
## 3. Core Components of the Transformer Architecture

The Transformer architecture is characterized by several interconnected components, each playing a crucial role in its ability to process sequences effectively. At its highest level, it follows an **encoder-decoder structure**, although encoder-only (e.g., BERT) and decoder-only (e.g., GPT) variations are also widely used.

<a name="31-encoder-decoder-structure"></a>
### 3.1. Encoder-Decoder Structure

The original Transformer model consists of a stack of **encoders** and a stack of **decoders**.
*   **Encoder:** The encoder stack processes the input sequence (e.g., source sentence in machine translation) and transforms it into a continuous representation, or a set of contextualized embeddings. Each encoder layer contains a multi-head self-attention mechanism and a position-wise feed-forward network. The output of the top encoder is fed as **keys (K)** and **values (V)** to the "encoder-decoder attention" layers in the decoder.
*   **Decoder:** The decoder stack generates the output sequence (e.g., target sentence). Each decoder layer typically includes three sub-layers: a masked multi-head self-attention mechanism (to prevent attending to future tokens), an encoder-decoder multi-head attention mechanism (to attend over the encoder's output), and a position-wise feed-forward network.

<a name="32-self-attention-mechanism"></a>
### 3.2. Self-Attention Mechanism

The cornerstone of the Transformer is the **self-attention mechanism**. This mechanism allows the model to weigh the importance of different words in the input sequence (or parts of the output sequence) when processing a particular word. For each token in the input, self-attention computes an output by aggregating information from all other tokens, weighted by their relevance. This enables the model to capture context and dependencies regardless of the distance between words in the sequence.

The core of self-attention is the **scaled dot-product attention**. It takes three main inputs:
*   **Queries (Q):** A matrix derived from the current token (or position) for which we want to compute an attention-weighted representation.
*   **Keys (K):** A matrix derived from all tokens in the sequence, used to match against queries.
*   **Values (V):** A matrix derived from all tokens, whose weighted sum forms the output.

The attention function computes the output as a weighted sum of the values, where the weight assigned to each value is determined by the dot-product similarity between the query and the corresponding key, followed by a **softmax** function to normalize the weights. The scaling factor, $d_k$, (square root of the dimension of keys) is used to prevent the dot products from becoming too large, which can push the softmax function into regions with extremely small gradients.

Mathematically, scaled dot-product attention is defined as:
$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

<a name="33-multi-head-attention"></a>
### 3.3. Multi-Head Attention

While single self-attention is powerful, **Multi-Head Attention** enhances the model's ability to focus on different parts of the input/output sequence simultaneously and to capture a richer set of relationships. Instead of performing a single attention function, Multi-Head Attention linearly projects the Queries, Keys, and Values $h$ times with different, learned linear projections. For each of these $h$ "heads," it then performs the scaled dot-product attention independently. The outputs from these $h$ attention heads are then concatenated and once again linearly projected to produce the final output. This allows the model to learn different notions of "relevance" or "attention" at different positions and integrate this diverse information.

<a name="34-positional-encoding"></a>
### 3.4. Positional Encoding

Since the Transformer completely foregoes recurrence and convolution, it inherently lacks any mechanism to understand the **order** or **position** of tokens in a sequence. To address this, **Positional Encoding** is added to the input embeddings at the bottom of the encoder and decoder stacks. These positional encodings are vectors that carry information about the relative or absolute position of each token. The original paper used fixed sinusoidal functions for this purpose, allowing the model to generalize to longer sequence lengths than those encountered during training. The positional encoding vector for a given position is simply added to the word embedding vector, creating a combined representation that encodes both the token's identity and its position.

<a name="35-position-wise-feed-forward-networks"></a>
### 3.5. Position-wise Feed-Forward Networks

Each layer in both the encoder and decoder contains a **position-wise feed-forward network (FFN)**. This is a simple, fully connected feed-forward network that is applied independently and identically to each position in the sequence. It consists of two linear transformations with a ReLU activation in between: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$. The parameters $(W_1, b_1, W_2, b_2)$ are shared across all positions but are different for each layer. These FFNs are crucial for allowing the model to process the contextual information derived from the attention mechanisms, adding non-linearity and increasing the model's expressive power.

<a name="36-residual-connections-and-layer-normalization"></a>
### 3.6. Residual Connections and Layer Normalization

For stability and to facilitate training of deep networks, each sub-layer (self-attention, encoder-decoder attention, FFN) in the Transformer architecture employs a **residual connection** (also known as a skip connection) followed by **layer normalization**. That is, the output of each sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$, where $\text{Sublayer}(x)$ is the function implemented by the sub-layer itself (e.g., self-attention). Residual connections help mitigate the vanishing gradient problem and allow gradients to flow more easily through the network, while layer normalization stabilizes the activations across different features and positions, further aiding training convergence.

<a name="4-training-and-inference"></a>
## 4. Training and Inference

**Training** a Transformer model typically involves vast amounts of data, often comprising billions of text tokens. The model is trained to minimize a loss function, commonly **cross-entropy loss**, by predicting the next token in a sequence or generating a target sequence given an input. For tasks like machine translation, the model is trained end-to-end on parallel corpora. For **Large Language Models (LLMs)**, training often occurs in two main phases:
1.  **Pre-training:** A self-supervised task, such as masked language modeling (like BERT) or causal language modeling (like GPT), is performed on a massive, diverse dataset to learn general language representations.
2.  **Fine-tuning:** The pre-trained model is then adapted to specific downstream tasks (e.g., sentiment analysis, question answering) using smaller, task-specific labeled datasets.

**Inference** with a Transformer decoder often involves an **auto-regressive** process. For text generation, the model starts with a special `<START>` token, predicts the next word, then feeds both the `<START>` token and the predicted word back into the model to predict the subsequent word, and so on, until an `<END>` token is generated or a maximum sequence length is reached. Various decoding strategies, such as **greedy decoding**, **beam search**, or **top-k/nucleus sampling**, are employed to enhance the quality and diversity of the generated output.

<a name="5-advantages-and-impact"></a>
## 5. Advantages and Impact

The Transformer architecture's shift from recurrent mechanisms to an attention-only paradigm brought forth several significant advantages:

*   **Parallelization:** The most prominent advantage is the ability to process all tokens in a sequence simultaneously. This parallelism drastically reduces training time on modern hardware (GPUs/TPUs) compared to sequential RNNs, enabling the training of much larger models on bigger datasets.
*   **Long-Range Dependencies:** Self-attention directly models relationships between any two tokens in a sequence, regardless of their distance. This overcomes the vanishing gradient and information decay issues that plagued RNNs over long sequences, allowing Transformers to capture very long-range contextual dependencies effectively.
*   **Interpretability:** While deep learning models are often black boxes, the attention weights within Transformers can offer some insights into which parts of the input are most influential for a given prediction, providing a degree of **interpretability**.
*   **Transfer Learning:** The pre-training and fine-tuning paradigm, pioneered with Transformer models like BERT and GPT, has become standard in NLP. Pre-trained Transformers can be fine-tuned with relatively small task-specific datasets, achieving state-of-the-art results across a wide array of downstream tasks, demonstrating powerful **transfer learning** capabilities.

The impact of the Transformer architecture on AI has been profound. It has not only become the de facto standard for NLP tasks, powering virtually all state-of-the-art **Large Language Models (LLMs)** like GPT-3, GPT-4, Llama, and PaLM, but has also extended its influence to other domains. Notably, **Vision Transformers (ViTs)** have successfully adapted the attention mechanism for image processing, demonstrating competitive performance with traditional Convolutional Neural Networks (CNNs) in computer vision. Its core principles are now being explored in various modalities, including speech, time series, and multi-modal AI, solidifying its position as one of the most significant architectural innovations in modern deep learning.

<a name="6-code-example-scaled-dot-product-attention"></a>
## 6. Code Example: Scaled Dot-Product Attention

Below is a minimalist Python code snippet demonstrating the core computation of **scaled dot-product attention** using NumPy. This illustrates how queries, keys, and values interact to produce attention weights and the final output.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes scaled dot-product attention.

    Args:
        Q (np.array): Query matrix. Shape (batch_size, num_queries, d_k)
        K (np.array): Key matrix. Shape (batch_size, num_keys, d_k)
        V (np.array): Value matrix. Shape (batch_size, num_keys, d_v)
        mask (np.array, optional): Optional mask to hide certain connections.
                                  Shape (batch_size, num_queries, num_keys)

    Returns:
        np.array: Attention-weighted output. Shape (batch_size, num_queries, d_v)
        np.array: Attention weights. Shape (batch_size, num_queries, num_keys)
    """
    d_k = Q.shape[-1]  # Dimension of keys and queries
    
    # 1. Compute dot product of Q and K^T
    # (batch_size, num_queries, d_k) @ (batch_size, d_k, num_keys)
    # -> (batch_size, num_queries, num_keys)
    scores = np.matmul(Q, K.swapaxes(-2, -1))
    
    # 2. Scale the scores
    scaled_scores = scores / np.sqrt(d_k)
    
    # 3. Apply optional mask (e.g., for padding or preventing future leakage)
    if mask is not None:
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores) # -1e9 effectively becomes 0 after softmax
    
    # 4. Apply softmax to get attention weights
    # Across the last dimension (num_keys), for each query
    attention_weights = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # 5. Multiply weights with Values
    # (batch_size, num_queries, num_keys) @ (batch_size, num_keys, d_v)
    # -> (batch_size, num_queries, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# Example Usage:
# Let's assume a batch size of 1, 3 queries (e.g., current words),
# 5 keys/values (e.g., all words in a sentence),
# d_k = 4 (embedding dimension) and d_v = 4 (value dimension)
batch_size = 1
num_queries = 3
num_keys = 5
d_k = 4
d_v = 4 # Often d_k == d_v

# Randomly initialize Q, K, V matrices
Q_example = np.random.rand(batch_size, num_queries, d_k)
K_example = np.random.rand(batch_size, num_keys, d_k)
V_example = np.random.rand(batch_size, num_keys, d_v)

# Compute attention
output_example, weights_example = scaled_dot_product_attention(Q_example, K_example, V_example)

print("Output shape:", output_example.shape) # Expected: (1, 3, 4)
print("Attention weights shape:", weights_example.shape) # Expected: (1, 3, 5)

# Example with a mask: Masking the last two keys for all queries
mask_example = np.ones((batch_size, num_queries, num_keys))
mask_example[:, :, -2:] = 0 # Set last two keys to be ignored

output_masked, weights_masked = scaled_dot_product_attention(Q_example, K_example, V_example, mask=mask_example)

print("\nOutput shape (with mask):", output_masked.shape)
print("Attention weights (with mask, last two columns should be near zero):\n", weights_masked[0])

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion

The Transformer architecture represents a monumental leap in the field of deep learning, particularly for sequence modeling. By entirely abandoning sequential recurrence and embracing the **self-attention mechanism**, it unlocked unprecedented levels of parallelism, enabling the creation of models with billions of parameters. This architectural innovation has directly led to the rise of sophisticated **Large Language Models (LLMs)** that power a wide array of generative AI applications, from highly accurate machine translation and sophisticated chatbots to complex code generation and creative content creation. The Transformer's ability to effectively model long-range dependencies, combined with its efficient training capabilities, has cemented its status as the foundational architecture for modern AI. As research continues to explore its variants and applications across modalities, the Transformer's influence is set to expand even further, driving the next wave of advancements in artificial intelligence.

---
<br>

<a name="türkçe-içerik"></a>
## Transformer Mimarisine Derin Bir Bakış

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Transformer'ın Doğuşu](#2-transformerın-doğuşu)
- [3. Transformer Mimarisinin Temel Bileşenleri](#3-transformer-mimarisinin-temel-bileşenleri)
  - [3.1. Kodlayıcı-Kod Çözücü Yapısı](#31-kodlayıcı-kod-çözücü-yapısı)
  - [3.2. Öz-Dikkat Mekanizması](#32-öz-dikkat-mekanizması)
  - [3.3. Çok Başlı Dikkat](#33-çok-başlı-dikkat)
  - [3.4. Konumsal Kodlama](#34-konumsal-kodlama)
  - [3.5. Konum Tabanlı İleri Beslemeli Ağlar](#35-konum-tabanlı-ileri-beslemeli-ağlar)
  - [3.6. Artık Bağlantılar ve Katman Normalizasyonu](#36-artık-bağlantılar-ve-katman-normalizasyonu)
- [4. Eğitim ve Çıkarım](#4-eğitim-ve-çıkarım)
- [5. Avantajlar ve Etki](#5-avantajlar-ve-etki)
- [6. Kod Örneği: Ölçeklendirilmiş Nokta Çarpımı Dikkat](#6-kod-örneği-ölçeklendirilmiş-nokta-çarpımı-dikkat)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Vaswani ve arkadaşları tarafından 2017'de yayımlanan "Attention Is All You Need" adlı çığır açan makalede tanıtılan **Transformer mimarisi**, **Doğal Dil İşleme (NLP)** ve daha geniş anlamda **Üretken Yapay Zeka (AI)** alanlarında devrim yaratmıştır. Ortaya çıkışından önce, tekrarlayan sinir ağları (RNN'ler) ve evrişimli sinir ağları (CNN'ler), özellikle Uzun Kısa Süreli Bellek (LSTM) ağları, diziden diziye görevler için baskın paradigmalardı. Bu modeller etkili olmakla birlikte, uzun menzilli bağımlılıkları verimli bir şekilde yakalamakta ve doğaları gereği sıralı olmaları nedeniyle eğitim sırasında paralelleştirmeyi sınırlamakta zorlanıyorlardı. Transformer mimarisi, yineleme ve evrişimleri tamamen terk ederek, yalnızca **öz-dikkat** adı verilen bir mekanizmaya dayanarak bu sınırlamaları zarif bir şekilde ele almaktadır. Bu yenilikçi yaklaşım, GPT, BERT ve T5 gibi temel **Büyük Dil Modelleri (LLM'ler)** dahil olmak üzere son derece güçlü ve ölçeklenebilir modellerin geliştirilmesinin önünü açmış, makine çevirisinden metin üretimine ve ötesine kadar çeşitli uygulamalarda üretken yapay zeka sistemlerinin yeteneklerini önemli ölçüde geliştirmiştir. Bu belge, Transformer mimarisini tanımlayan karmaşık bileşenleri ve çalışma prensiplerini inceleyerek modern yapay zeka üzerindeki derin etkisini açıklığa kavuşturmaktadır.

<a name="2-transformerın-doğuşu"></a>
## 2. Transformer'ın Doğuşu

Transformer'dan önce, **Tekrarlayan Sinir Ağları (RNN'ler)** ve LSTM'ler ile Gated Recurrent Unit (GRU) gibi varyantları gibi dizi işleme modelleri, giriş dizilerini belirteç belirteç (token by token) işliyordu. Bu sıralı işleme, bir belirteç için yapılan hesaplamaların büyük ölçüde önceki belirteçlere bağlı olması nedeniyle bir darboğaz oluşturuyordu. Bu doğal sıralılık, paralelleştirmeyi ve dolayısıyla eğitim hızını sınırlıyordu, özellikle çok uzun diziler için. Dahası, LSTM'ler **kaybolan gradyan problemini** hafifletmeyi amaçlasalar da, aşırı uzun diziler boyunca uzun menzilli bağımlılıkları sürdürmekte hala zorlanıyor, genellikle zamanla bilgi kaybı yaşıyorlardı.

**Evrişimli Sinir Ağları (CNN'ler)**, sabit boyutlu yerel bağlamları işleyerek bazı paralelleştirme avantajları sunuyordu, ancak küresel bağımlılıkları yakalamak, birçok katmanın üst üste istiflenmesini gerektiriyordu, bu da hesaplama derinliğini artırıyor ve potansiyel olarak yerel bağlamı seyrediliyordu. İlk olarak RNN'lerle birlikte nöral makine çevirisi bağlamında tanıtılan **dikkat mekanizması**, her çıktı belirtecini üretirken modelin giriş dizisinin ilgili kısımlarına odaklanmasına izin vererek bu sorunları kısmen hafifletti. Ancak bu dikkat, hala tekrarlayan bir yapının üzerinde uygulanıyordu.

"Attention Is All You Need" makalesi radikal bir değişiklik önerdi: tekrarlama ve evrişimi tamamen ortadan kaldırmak ve yalnızca dikkat mekanizmalarına dayalı bir model oluşturmak. Bu tasarım kararı, daha fazla paralelleştirme, uzun bağımlılıkları daha etkili bir şekilde ele alma ve genel mimariyi basitleştirme arzusundan kaynaklanıyordu. Transformer'ın dizinin tüm kısımları için aynı anda dikkat hesaplama yeteneği, sabit derinlikli yapısıyla birleştiğinde, eğitim verimliliğinde ve model kapasitesinde benzeri görülmemiş kazanımlar sağlayarak, dizi modelleme için sinir ağları tarihinde önemli bir dönüm noktası oldu.

<a name="3-transformer-mimarisinin-temel-bileşenleri"></a>
## 3. Transformer Mimarisinin Temel Bileşenleri

Transformer mimarisi, her biri dizileri etkili bir şekilde işleme yeteneğinde kritik bir rol oynayan, birbiriyle bağlantılı çeşitli bileşenlerle karakterize edilir. En üst düzeyde, bir **kodlayıcı-kod çözücü yapısını** takip eder, ancak yalnızca kodlayıcılı (örn. BERT) ve yalnızca kod çözücülü (örn. GPT) varyasyonlar da yaygın olarak kullanılmaktadır.

<a name="31-kodlayıcı-kod-çözücü-yapısı"></a>
### 3.1. Kodlayıcı-Kod Çözücü Yapısı

Orijinal Transformer modeli, bir dizi **kodlayıcı** ve bir dizi **kod çözücüden** oluşur.
*   **Kodlayıcı:** Kodlayıcı yığını, giriş dizisini (örn. makine çevirisinde kaynak cümle) işler ve sürekli bir gösterime veya bağlamsallaştırılmış gömülü bir diziye dönüştürür. Her kodlayıcı katmanı, çok başlı öz-dikkat mekanizması ve konum tabanlı bir ileri beslemeli ağ içerir. En üstteki kodlayıcının çıktısı, kod çözücüdeki "kodlayıcı-kod çözücü dikkat" katmanlarına **anahtarlar (K)** ve **değerler (V)** olarak beslenir.
*   **Kod Çözücü:** Kod çözücü yığını, çıktı dizisini (örn. hedef cümle) üretir. Her kod çözücü katmanı tipik olarak üç alt katman içerir: maskeli çok başlı öz-dikkat mekanizması (gelecekteki belirteçlere dikkat etmeyi önlemek için), bir kodlayıcı-kod çözücü çok başlı dikkat mekanizması (kodlayıcının çıktısı üzerinde dikkat etmek için) ve konum tabanlı bir ileri beslemeli ağ.

<a name="32-öz-dikkat-mekanizması"></a>
### 3.2. Öz-Dikkat Mekanizması

Transformer'ın temel taşı, **öz-dikkat mekanizmasıdır**. Bu mekanizma, modelin belirli bir kelimeyi işlerken giriş dizisindeki (veya çıktı dizisinin parçalarındaki) farklı kelimelerin önemini ağırlıklandırmasına olanak tanır. Girişteki her belirteç için öz-dikkat, diğer tüm belirteçlerden gelen bilgiyi, alaka düzeylerine göre ağırlıklandırarak bir çıktı hesaplar. Bu, modelin kelimeler arasındaki mesafeye bakılmaksızın bağlamı ve bağımlılıkları yakalamasını sağlar.

Öz-dikkat mekanizmasının çekirdeği, **ölçeklendirilmiş nokta çarpımı dikkattir**. Üç ana girdi alır:
*   **Sorgular (Q):** Dikkat ağırlıklı bir temsilini hesaplamak istediğimiz geçerli belirteçten (veya konumdan) türetilen bir matris.
*   **Anahtarlar (K):** Sorgularla eşleştirmek için kullanılan, dizideki tüm belirteçlerden türetilen bir matris.
*   **Değerler (V):** Ağırlıklı toplamı çıktıyı oluşturan, tüm belirteçlerden türetilen bir matris.

Dikkat fonksiyonu, değerlerin ağırlıklı bir toplamı olarak çıktıyı hesaplar; burada her değere atanan ağırlık, sorgu ile ilgili anahtar arasındaki nokta çarpımı benzerliği ile belirlenir ve ardından ağırlıkları normalleştirmek için bir **softmax** fonksiyonu uygulanır. $d_k$ (anahtarların boyutunun karekökü) ölçeklendirme faktörü, nokta çarpımlarının çok büyük olmasını önlemek için kullanılır, bu da softmax fonksiyonunu son derece küçük gradyanlara sahip bölgelere itebilir.

Matematiksel olarak, ölçeklendirilmiş nokta çarpımı dikkat şu şekilde tanımlanır:
$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

<a name="33-çok-başlı-dikkat"></a>
### 3.3. Çok Başlı Dikkat

Tek başına öz-dikkat güçlü olsa da, **Çok Başlı Dikkat**, modelin giriş/çıkış dizisinin farklı kısımlarına aynı anda odaklanma ve daha zengin bir ilişki kümesi yakalama yeteneğini geliştirir. Tek bir dikkat fonksiyonu uygulamak yerine, Çok Başlı Dikkat, Sorguları, Anahtarları ve Değerleri $h$ kez farklı, öğrenilmiş doğrusal projeksiyonlarla doğrusal olarak yansıtır. Bu $h$ "başlığın" her biri için, daha sonra ölçeklendirilmiş nokta çarpımı dikkatini bağımsız olarak gerçekleştirir. Bu $h$ dikkat başlığından gelen çıktılar daha sonra birleştirilir ve nihai çıktıyı üretmek için bir kez daha doğrusal olarak yansıtılır. Bu, modelin farklı konumlarda farklı "alaka düzeyi" veya "dikkat" kavramlarını öğrenmesine ve bu çeşitli bilgileri entegre etmesine olanak tanır.

<a name="34-konumsal-kodlama"></a>
### 3.4. Konumsal Kodlama

Transformer, yineleme ve evrişimi tamamen terk ettiği için, dizideki belirteçlerin **sırasını** veya **konumunu** anlama mekanizmasından yoksundur. Bunu ele almak için, kodlayıcı ve kod çözücü yığınlarının altına giriş gömülü değerlerine **Konumsal Kodlama** eklenir. Bu konumsal kodlamalar, her belirtecin göreceli veya mutlak konumu hakkında bilgi taşıyan vektörlerdir. Orijinal makale bu amaçla sabit sinüzoidal fonksiyonlar kullanmış, bu da modelin eğitim sırasında karşılaşılanlardan daha uzun dizi uzunluklarına genelleşmesine olanak sağlamıştır. Belirli bir konum için konumsal kodlama vektörü, kelime gömülü vektörüne basitçe eklenir ve belirtecin kimliğini ve konumunu kodlayan birleşik bir temsil oluşturur.

<a name="35-konum-tabanlı-ileri-beslemeli-ağlar"></a>
### 3.5. Konum Tabanlı İleri Beslemeli Ağlar

Hem kodlayıcı hem de kod çözücüdeki her katman, bir **konum tabanlı ileri beslemeli ağ (FFN)** içerir. Bu, dizideki her konuma bağımsız ve aynı şekilde uygulanan basit, tamamen bağlı bir ileri beslemeli ağdır. Arasında bir ReLU aktivasyonu bulunan iki doğrusal dönüşümden oluşur: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$. Parametreler $(W_1, b_1, W_2, b_2)$ tüm konumlar arasında paylaşılır ancak her katman için farklıdır. Bu FFN'ler, modelin dikkat mekanizmalarından türetilen bağlamsal bilgileri işlemesine izin vermek, doğrusal olmayanlık eklemek ve modelin ifade gücünü artırmak için çok önemlidir.

<a name="36-artık-bağlantılar-ve-katman-normalizasyonu"></a>
### 3.6. Artık Bağlantılar ve Katman Normalizasyonu

Derin ağların kararlılığı ve eğitimini kolaylaştırmak için, Transformer mimarisindeki her alt katman (öz-dikkat, kodlayıcı-kod çözücü dikkat, FFN) bir **artık bağlantı** (atlama bağlantısı olarak da bilinir) ve ardından **katman normalizasyonu** kullanır. Yani, her alt katmanın çıktısı $\text{LayerNorm}(x + \text{Sublayer}(x))$ şeklindedir; burada $\text{Sublayer}(x)$, alt katmanın kendisi tarafından uygulanan fonksiyondur (örn. öz-dikkat). Artık bağlantılar, kaybolan gradyan problemini hafifletmeye yardımcı olur ve gradyanların ağ boyunca daha kolay akmasını sağlar, katman normalizasyonu ise farklı özellikler ve konumlar arasındaki aktivasyonları stabilize ederek eğitim yakınsamasına daha fazla yardımcı olur.

<a name="4-eğitim-ve-çıkarım"></a>
## 4. Eğitim ve Çıkarım

Bir Transformer modelini **eğitmek** tipik olarak, genellikle milyarlarca metin belirteci içeren büyük miktarda veri gerektirir. Model, bir dizideki bir sonraki belirteci tahmin ederek veya bir giriş verildiğinde bir hedef dizi üreterek, genellikle **çapraz entropi kaybı** olan bir kayıp fonksiyonunu minimize etmek için eğitilir. Makine çevirisi gibi görevler için model, paralel korpuslar üzerinde uçtan uca eğitilir. **Büyük Dil Modelleri (LLM'ler)** için eğitim genellikle iki ana aşamada gerçekleşir:
1.  **Ön eğitim:** Genel dil temsillerini öğrenmek için maskeli dil modellemesi (BERT gibi) veya nedensel dil modellemesi (GPT gibi) gibi kendi kendine denetimli bir görev, büyük ve çeşitli bir veri kümesi üzerinde gerçekleştirilir.
2.  **İnce ayar:** Önceden eğitilmiş model daha sonra daha küçük, göreve özel etiketli veri kümeleri kullanılarak belirli alt görevlere (örn. duygu analizi, soru yanıtlama) uyarlanır.

Bir Transformer kod çözücü ile **çıkarım**, genellikle **otoregresif** bir süreç içerir. Metin üretimi için model, özel bir `<START>` belirteci ile başlar, bir sonraki kelimeyi tahmin eder, ardından hem `<START>` belirtecini hem de tahmin edilen kelimeyi bir sonraki kelimeyi tahmin etmek için modele geri besler ve bu böylece bir `<END>` belirteci üretilene veya maksimum dizi uzunluğuna ulaşılana kadar devam eder. Üretilen çıktının kalitesini ve çeşitliliğini artırmak için **açgözlü kod çözme**, **ışın arama** veya **top-k/çekirdek örnekleme** gibi çeşitli kod çözme stratejileri kullanılır.

<a name="5-avantajlar-ve-etki"></a>
## 5. Avantajlar ve Etki

Transformer mimarisinin tekrarlayan mekanizmalardan yalnızca dikkat paradigmasına geçişi, birkaç önemli avantajı beraberinde getirmiştir:

*   **Paralelleştirme:** En belirgin avantaj, bir dizideki tüm belirteçleri eşzamanlı olarak işleyebilme yeteneğidir. Bu paralellik, sıralı RNN'lere kıyasla modern donanımlarda (GPU'lar/TPU'lar) eğitim süresini önemli ölçüde azaltır ve çok daha büyük modellerin daha büyük veri kümeleri üzerinde eğitilmesini mümkün kılar.
*   **Uzun Menzilli Bağımlılıklar:** Öz-dikkat, bir dizideki herhangi iki belirteç arasındaki ilişkileri mesafelerine bakılmaksızın doğrudan modeller. Bu, RNN'leri uzun dizilerde rahatsız eden kaybolan gradyan ve bilgi bozulması sorunlarının üstesinden gelerek Transformer'ların çok uzun menzilli bağlamsal bağımlılıkları etkili bir şekilde yakalamasına olanak tanır.
*   **Yorumlanabilirlik:** Derin öğrenme modelleri genellikle kara kutu olsa da, Transformer'lar içindeki dikkat ağırlıkları, belirli bir tahmin için girdinin hangi kısımlarının en etkili olduğuna dair bazı içgörüler sunarak bir dereceye kadar **yorumlanabilirlik** sağlar.
*   **Transfer Öğrenimi:** BERT ve GPT gibi Transformer modelleriyle öncülük edilen ön eğitim ve ince ayar paradigması, NLP'de standart haline gelmiştir. Önceden eğitilmiş Transformer'lar, nispeten küçük göreve özel veri kümeleriyle ince ayar yapılabilir, çok çeşitli alt görevlerde son teknoloji sonuçlar elde ederek güçlü **transfer öğrenme** yetenekleri sergilemişlerdir.

Transformer mimarisinin yapay zeka üzerindeki etkisi derin olmuştur. Sadece NLP görevleri için fiili standart haline gelmekle kalmamış, GPT-3, GPT-4, Llama ve PaLM gibi neredeyse tüm son teknoloji **Büyük Dil Modellerini (LLM'ler)** desteklemekle kalmamış, aynı zamanda diğer alanlara da etkisini genişletmiştir. Özellikle, **Vision Transformer'lar (ViT'ler)**, dikkat mekanizmasını görüntü işleme için başarıyla uyarlayarak, bilgisayar görüşünde geleneksel Evrişimli Sinir Ağları (CNN'lerle) rekabetçi performans göstermiştir. Temel prensipleri artık konuşma, zaman serileri ve çok modlu yapay zeka dahil olmak üzere çeşitli modalitelerde keşfedilmekte ve modern derin öğrenmedeki en önemli mimari yeniliklerden biri olarak konumunu sağlamlaştırmaktadır.

<a name="6-kod-örneği-ölçeklendirilmiş-nokta-çarpımı-dikkat"></a>
## 6. Kod Örneği: Ölçeklendirilmiş Nokta Çarpımı Dikkat

Aşağıda, NumPy kullanarak **ölçeklendirilmiş nokta çarpımı dikkat** çekirdek hesaplamasını gösteren minimalist bir Python kod parçacığı bulunmaktadır. Bu, sorguların, anahtarların ve değerlerin dikkat ağırlıklarını ve nihai çıktıyı üretmek için nasıl etkileşime girdiğini göstermektedir.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Ölçeklendirilmiş nokta çarpımı dikkatini hesaplar.

    Argümanlar:
        Q (np.array): Sorgu matrisi. Şekil (batch_size, num_queries, d_k)
        K (np.array): Anahtar matrisi. Şekil (batch_size, num_keys, d_k)
        V (np.array): Değer matrisi. Şekil (batch_size, num_keys, d_v)
        mask (np.array, isteğe bağlı): Belirli bağlantıları gizlemek için isteğe bağlı maske.
                                       Şekil (batch_size, num_queries, num_keys)

    Döndürür:
        np.array: Dikkat ağırlıklı çıktı. Şekil (batch_size, num_queries, d_v)
        np.array: Dikkat ağırlıkları. Şekil (batch_size, num_queries, num_keys)
    """
    d_k = Q.shape[-1]  # Anahtarların ve sorguların boyutu
    
    # 1. Q ve K^T'nin nokta çarpımını hesapla
    # (batch_size, num_queries, d_k) @ (batch_size, d_k, num_keys)
    # -> (batch_size, num_queries, num_keys)
    scores = np.matmul(Q, K.swapaxes(-2, -1))
    
    # 2. Skorları ölçekle
    scaled_scores = scores / np.sqrt(d_k)
    
    # 3. İsteğe bağlı maskeyi uygula (örn. doldurma veya gelecekteki sızıntıyı önlemek için)
    if mask is not None:
        # Masked = 0 olan yerleri -1e9 ile değiştir, softmax sonrası 0'a yakın olur
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores) 
    
    # 4. Dikkat ağırlıklarını almak için softmax uygula
    # Her sorgu için son boyut (num_keys) boyunca
    attention_weights = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # 5. Ağırlıkları Değerler ile çarp
    # (batch_size, num_queries, num_keys) @ (batch_size, num_keys, d_v)
    # -> (batch_size, num_queries, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# Örnek Kullanım:
# 1'lik bir parti boyutu, 3 sorgu (örn. mevcut kelimeler),
# 5 anahtar/değer (örn. bir cümledeki tüm kelimeler),
# d_k = 4 (gömme boyutu) ve d_v = 4 (değer boyutu) varsayalım.
batch_size = 1
num_queries = 3
num_keys = 5
d_k = 4
d_v = 4 # Genellikle d_k == d_v

# Q, K, V matrislerini rastgele başlat
Q_ornek = np.random.rand(batch_size, num_queries, d_k)
K_ornek = np.random.rand(batch_size, num_keys, d_k)
V_ornek = np.random.rand(batch_size, num_keys, d_v)

# Dikkat hesapla
output_ornek, weights_ornek = scaled_dot_product_attention(Q_ornek, K_ornek, V_ornek)

print("Çıktı şekli:", output_ornek.shape) # Beklenen: (1, 3, 4)
print("Dikkat ağırlıkları şekli:", weights_ornek.shape) # Beklenen: (1, 3, 5)

# Maskeli örnek: Tüm sorgular için son iki anahtarı maskeleme
mask_ornek = np.ones((batch_size, num_queries, num_keys))
mask_ornek[:, :, -2:] = 0 # Son iki anahtarı yoksay olarak ayarla

output_masked, weights_masked = scaled_dot_product_attention(Q_ornek, K_ornek, V_ornek, mask=mask_ornek)

print("\nÇıktı şekli (maske ile):", output_masked.shape)
print("Dikkat ağırlıkları (maske ile, son iki sütun sıfıra yakın olmalı):\n", weights_masked[0])

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç

Transformer mimarisi, derin öğrenme alanında, özellikle dizi modellemesi için anıtsal bir sıçramayı temsil etmektedir. Sıralı yinelemeyi tamamen terk edip **öz-dikkat mekanizmasını** benimseyerek, milyarlarca parametreye sahip modellerin oluşturulmasını sağlayan benzeri görülmemiş düzeyde paralellik açığa çıkarmıştır. Bu mimari yenilik, yüksek doğruluklu makine çevirisinden gelişmiş sohbet botlarına, karmaşık kod üretiminden yaratıcı içerik oluşturmaya kadar geniş bir üretken yapay zeka uygulamaları yelpazesini güçlendiren sofistike **Büyük Dil Modellerinin (LLM'ler)** yükselişine doğrudan yol açmıştır. Transformer'ın uzun menzilli bağımlılıkları etkili bir şekilde modelleme yeteneği, verimli eğitim yetenekleriyle birleştiğinde, modern yapay zeka için temel mimari statüsünü sağlamlaştırmıştır. Araştırmalar, çeşitli modalitelerdeki varyantlarını ve uygulamalarını keşfetmeye devam ettikçe, Transformer'ın etkisi daha da genişleyerek yapay zekadaki bir sonraki ilerleme dalgasını yönlendirecektir.
