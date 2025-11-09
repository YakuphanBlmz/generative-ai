# A Deep Dive into the Transformer Architecture

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Genesis of Transformers and the "Attention Is All You Need" Paper](#2-the-genesis-of-transformers-and-the-attention-is-all-you-need-paper)
- [3. Core Components of the Transformer Architecture](#3-core-components-of-the-transformer-architecture)
  - [3.1. Encoder-Decoder Architecture](#31-encoder-decoder-architecture)
  - [3.2. Positional Encoding](#32-positional-encoding)
  - [3.3. Multi-Head Self-Attention Mechanism](#33-multi-head-self-attention-mechanism)
  - [3.4. Feed-Forward Networks](#34-feed-forward-networks)
  - [3.5. Residual Connections and Layer Normalization](#35-residual-connections-and-layer-normalization)
- [4. Training and Inference](#4-training-and-inference)
- [5. Impact, Applications, and Limitations](#5-impact-applications-and-limitations)
- [6. Code Example: Illustrating Scaled Dot-Product Attention](#6-code-example-illustrating-scaled-dot-product-attention)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
### 1. Introduction

The **Transformer architecture**, introduced by Vaswani et al. in their seminal 2017 paper "Attention Is All You Need," revolutionized the field of **Natural Language Processing (NLP)** and subsequently, the broader domain of **Generative AI**. Prior to the Transformer, **recurrent neural networks (RNNs)** and their variants like **Long Short-Term Memory (LSTM)** networks were the dominant models for sequential data processing, particularly for tasks such as machine translation, text summarization, and speech recognition. While effective, these models suffered from inherent limitations, including sequential processing, which hindered **parallelization** and made capturing long-range dependencies challenging due to vanishing or exploding gradients.

The Transformer architecture elegantly addresses these issues by largely discarding recurrence and convolutions in favor of a mechanism called **self-attention**. This allows the model to weigh the importance of different parts of the input sequence when processing each element, effectively capturing dependencies regardless of their distance. This paradigm shift not only enabled unprecedented levels of **parallel computing** during training but also significantly improved model performance across a wide array of tasks. The Transformer has since become the foundational architecture for state-of-the-art **Large Language Models (LLMs)** like BERT, GPT, T5, and many others, underpinning much of the recent progress in **Generative AI**. This document provides a deep dive into the intricate components and operational principles of the Transformer architecture, exploring its design, mechanisms, and profound impact.

<a name="2-the-genesis-of-transformers-and-the-attention-is-all-you-need-paper"></a>
### 2. The Genesis of Transformers and the "Attention Is All You Need" Paper

The concept of **attention mechanisms** was not new to the "Attention Is All You Need" paper. Earlier sequence-to-sequence models, often based on encoder-decoder RNNs, had incorporated attention to selectively focus on relevant parts of the input sequence during decoding. However, these models still relied heavily on the sequential nature of RNNs for encoding the input. The breakthrough of the Transformer was to propose a model that relied *solely* on attention mechanisms, doing away with both recurrence and convolution entirely.

The primary motivation behind this radical architectural choice was to overcome the inherent limitations of RNNs:
1.  **Lack of Parallelization:** RNNs process tokens one by one, making them slow to train on large datasets due to their inability to fully exploit modern parallel computing hardware (GPUs).
2.  **Difficulty with Long-Range Dependencies:** While LSTMs and GRUs partially mitigated this, maintaining information over very long sequences remained a challenge due to the fixed-size hidden state and the gradient problems.

By replacing recurrence with **self-attention**, the Transformer allowed each token in the input sequence to simultaneously attend to all other tokens, computing interactions in parallel. This enabled the model to directly capture relationships between distant words, regardless of their position in the sequence, and drastically accelerated training times, making it feasible to train much larger models on vast amounts of data. This fundamental shift paved the way for the era of pre-trained language models that now dominate **Generative AI**.

<a name="3-core-components-of-the-transformer-architecture"></a>
### 3. Core Components of the Transformer Architecture

The Transformer is a **sequence-to-sequence** model, typically composed of an **encoder** stack and a **decoder** stack. Both stacks consist of identical layers, with each layer incorporating several sub-layers.

<a name="31-encoder-decoder-architecture"></a>
#### 3.1. Encoder-Decoder Architecture

The **encoder** is responsible for processing the input sequence and generating a contextualized representation. It consists of `N` identical layers stacked on top of each other. Each encoder layer has two main sub-layers: a **Multi-Head Self-Attention mechanism** and a **position-wise fully connected Feed-Forward Network**.

The **decoder** is responsible for generating the output sequence, one token at a time, given the encoder's output and the previously generated tokens. It also consists of `N` identical layers. Each decoder layer has three sub-layers: a **Masked Multi-Head Self-Attention mechanism** (to prevent attending to future tokens during training), a **Multi-Head Attention mechanism** that attends to the encoder's output, and a **position-wise fully connected Feed-Forward Network**.

Both encoder and decoder inputs first pass through an **embedding layer** to convert input tokens into continuous vector representations. These embeddings are then combined with **positional encodings**.

<a name="32-positional-encoding"></a>
#### 3.2. Positional Encoding

Since the Transformer architecture largely eschews recurrence and convolution, it lacks an inherent notion of word order. To inject information about the relative or absolute position of tokens in the sequence, **positional encodings** are added to the input embeddings. These encodings are typically sinusoidal functions of different frequencies, allowing the model to learn to attend to relative positions. The intuition is that sine and cosine functions can represent relative positions because `sin(pos + k)` can be expressed as a linear function of `sin(pos)` and `cos(pos)`. This makes it easier for the model to generalize to longer sequences than it saw during training.

Mathematically, the positional encoding for a token at position `pos` and dimension `i` (where `i` is an even index) is given by:
`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
where `d_model` is the dimensionality of the embeddings.

<a name="33-multi-head-self-attention-mechanism"></a>
#### 3.3. Multi-Head Self-Attention Mechanism

At the heart of the Transformer lies the **attention mechanism**, specifically **Scaled Dot-Product Attention**, which enables the model to weigh the importance of different input tokens relative to each other. For each token, the attention mechanism computes three vectors: a **Query (Q)**, a **Key (K)**, and a **Value (V)**. These are derived by linearly transforming the input embeddings (or the output of the previous layer) with learned weight matrices.

The **Scaled Dot-Product Attention** function computes the attention scores as follows:
`Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V`
where `d_k` is the dimension of the keys, used for scaling to prevent very large dot products from pushing the softmax into regions with extremely small gradients. The `softmax` function then converts these scores into probabilities, indicating how much attention each output token should pay to each input token.

**Multi-Head Attention** extends this by allowing the model to jointly attend to information from different representation subspaces at different positions. Instead of performing a single attention function with `d_model` dimensional keys, values, and queries, the input is linearly projected `h` times with different, learned linear projections to `d_k`, `d_k`, and `d_v` dimensions, respectively. For each of these `h` projected versions, an attention function is computed in parallel. The resulting `h` attention outputs are then concatenated and once again linearly projected to produce the final values, allowing the model to learn diverse relationships.

In the encoder, this is **Self-Attention** because `Q`, `K`, and `V` all come from the same previous layer output. In the decoder, there are two attention sub-layers:
1.  **Masked Self-Attention**: `Q`, `K`, `V` come from the decoder's previous output, but attention is masked to prevent attending to future positions.
2.  **Encoder-Decoder Attention**: `Q` comes from the decoder's previous output, while `K` and `V` come from the output of the encoder. This allows the decoder to focus on relevant parts of the input sequence.

<a name="34-feed-forward-networks"></a>
#### 3.4. Feed-Forward Networks

Each encoder and decoder layer contains a **position-wise fully connected feed-forward network**. This network is applied independently and identically to each position. It consists of two linear transformations with a **ReLU activation** in between:
`FFN(x) = max(0, x * W1 + b1) * W2 + b2`
This simple network adds non-linearity and allows the model to process the information aggregated by the attention mechanism further.

<a name="35-residual-connections-and-layer-normalization"></a>
#### 3.5. Residual Connections and Layer Normalization

For each sub-layer (self-attention, encoder-decoder attention, and feed-forward network), the Transformer architecture employs a **residual connection** around it, followed by **layer normalization**. That is, the output of each sub-layer is `LayerNorm(x + Sublayer(x))`, where `Sublayer(x)` is the function implemented by the sub-layer itself.

**Residual connections**, originally from ResNet, help alleviate the vanishing gradient problem in deep networks by allowing gradients to flow directly through the identity mapping. This enables the training of much deeper models.
**Layer Normalization** normalizes the inputs across the features for each sample independently. This helps stabilize training and reduces training time by making the optimization landscape smoother.

<a name="4-training-and-inference"></a>
### 4. Training and Inference

**Training:** The Transformer is typically trained using **teacher forcing** for sequence-to-sequence tasks. During training, the decoder is provided with the correct previous token from the target sequence (shifted right) as input for predicting the next token. This ensures that the model learns to generate the correct sequence given the input context. The objective function is usually **cross-entropy loss**, optimizing the model to predict the next token accurately. Optimizers like **Adam** with a custom learning rate scheduler (warm-up followed by decay) are commonly used.

**Inference:** During inference (e.g., machine translation or text generation), the decoder must generate tokens one by one. The process starts with a special `<start>` token. The decoder predicts the next token, which is then fed back into the decoder's input for the next step, along with the encoder's output. This iterative process continues until an `<end>` token is generated or a maximum sequence length is reached. **Beam search** is often employed instead of greedy decoding to explore multiple potential output sequences and find a more optimal one.

<a name="5-impact-applications-and-limitations"></a>
### 5. Impact, Applications, and Limitations

The **Transformer** has profoundly impacted **Generative AI** and **NLP**. Its ability to model long-range dependencies efficiently and its amenability to parallel computation led to unprecedented advancements.

**Key Impacts:**
*   **Foundation for LLMs:** Transformers are the backbone of virtually all modern **Large Language Models (LLMs)** like Google's BERT and T5, OpenAI's GPT series, Meta's LLaMA, and many others.
*   **Transfer Learning:** The architecture enabled the pre-training paradigm, where a model is trained on a massive unlabeled text corpus (e.g., internet text) and then fine-tuned on smaller, task-specific datasets, leading to significant performance gains.
*   **Multimodality:** Transformers have extended beyond NLP to computer vision (Vision Transformers), speech processing, and even multimodal models that combine different types of data.

**Applications:**
*   **Machine Translation:** State-of-the-art results in translating between languages.
*   **Text Summarization:** Generating concise summaries of longer texts.
*   **Question Answering:** Understanding and answering questions based on given contexts.
*   **Text Generation:** Creating human-like text for various purposes, from creative writing to code generation.
*   **Sentiment Analysis, Named Entity Recognition, and more.**

**Limitations:**
*   **Quadratic Complexity:** The standard self-attention mechanism has a quadratic computational and memory complexity with respect to the sequence length (`O(N^2)`), making it challenging to process very long sequences (e.g., entire books) without modifications.
*   **Lack of Inductive Bias:** Unlike convolutional networks which have an inductive bias for local features, or RNNs for sequential order, the Transformer's "blank slate" nature requires vast amounts of data to learn these relationships. Positional encodings attempt to mitigate this, but it's still a factor.
*   **High Computational Cost:** While parallelizable, training and running large Transformer models, especially LLMs, demand significant computational resources (GPUs/TPUs) and energy.

Researchers are actively exploring ways to address these limitations, such as **sparse attention mechanisms**, **recurrent transformers**, and other architectural optimizations to improve efficiency and handle longer contexts.

<a name="6-code-example-illustrating-scaled-dot-product-attention"></a>
### 6. Code Example: Illustrating Scaled Dot-Product Attention

Here's a simplified Python code snippet demonstrating the core mathematical operation of **Scaled Dot-Product Attention**. It does not include multi-head or masking for brevity but showcases the `Q * K^T` operation, scaling, and softmax.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention.
    Args:
        query (torch.Tensor): Query tensor (batch_size, num_heads, seq_len_q, d_k).
        key (torch.Tensor): Key tensor (batch_size, num_heads, seq_len_k, d_k).
        value (torch.Tensor): Value tensor (batch_size, num_heads, seq_len_v, d_v).
        mask (torch.Tensor, optional): Mask tensor for preventing attention to certain positions.
    Returns:
        torch.Tensor: Output of the attention mechanism.
        torch.Tensor: Attention weights (attention scores).
    """
    d_k = query.size(-1) # Get the dimension of the keys
    
    # Calculate attention scores: Query * Key^T
    # (batch_size, num_heads, seq_len_q, d_k) @ (batch_size, num_heads, d_k, seq_len_k)
    # -> (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale the scores
    scores = scores / (d_k ** 0.5)
    
    # Apply mask if provided (e.g., for masked self-attention in decoder)
    if mask is not None:
        # Fill masked positions with a very large negative number
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights (probabilities)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Multiply weights with Value tensor
    # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_v, d_v)
    # -> (batch_size, num_heads, seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage (simplified, without batch or multi-head dimensions)
# Imagine d_model = 512, d_k = 64, d_v = 64, seq_len = 10
seq_len = 10
d_k = 64
d_v = 64

# Create dummy Query, Key, Value tensors
# For self-attention, Q, K, V come from the same source
dummy_input = torch.randn(seq_len, d_k) # Simplified single-head, single-batch
query = dummy_input
key = dummy_input
value = dummy_input

# Let's say we want to process 3 different query positions.
# Let's reshape them to simulate (seq_len_q, d_k), (seq_len_k, d_k), (seq_len_v, d_v)
query = torch.randn(3, d_k) # Example: 3 query vectors
key = torch.randn(seq_len, d_k) # All 10 input keys
value = torch.randn(seq_len, d_v) # All 10 input values

attention_output, weights = scaled_dot_product_attention(query.unsqueeze(0).unsqueeze(0), 
                                                         key.unsqueeze(0).unsqueeze(0), 
                                                         value.unsqueeze(0).unsqueeze(0))

print("Attention Output Shape:", attention_output.squeeze(0).squeeze(0).shape)
print("Attention Weights Shape:", weights.squeeze(0).squeeze(0).shape)
# Each of the 3 query positions now has an output vector of size d_v,
# and attention weights showing how much it focused on each of the 10 input positions.


(End of code example section)
```

<a name="7-conclusion"></a>
### 7. Conclusion

The **Transformer architecture** stands as a monumental achievement in the field of **Generative AI** and **deep learning**. By skillfully leveraging **self-attention mechanisms** and novel architectural designs like **positional encodings**, **residual connections**, and **layer normalization**, it effectively addresses the limitations of prior sequential models. Its ability to process information in parallel and capture complex, long-range dependencies has not only propelled **Natural Language Processing** to new heights but has also opened doors for advancements in computer vision and multimodal learning. While challenges related to computational complexity for extremely long sequences persist, ongoing research continues to refine and extend the Transformer's capabilities. The enduring influence of the "Attention Is All You Need" paper and its architectural innovations underscores the Transformer's role as a cornerstone of modern AI, shaping the landscape of intelligent systems for the foreseeable future.

---
<br>

<a name="türkçe-içerik"></a>
## Transformatör Mimarisine Derin Bir Bakış

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Transformatörlerin Doğuşu ve "Attention Is All You Need" Makalesi](#2-transformatörlerin-doğuşu-ve-attention-is-all-you-need-makalesi)
- [3. Transformatör Mimarisinin Temel Bileşenleri](#3-transformatör-mimarisinin-temel-bileşenleri)
  - [3.1. Kodlayıcı-Çözücü Mimarisi](#31-kodlayıcı-çözücü-mimarisi)
  - [3.2. Konumsal Kodlama](#32-konumsal-kodlama)
  - [3.3. Çok-Başlı Öz-Dikkat Mekanizması](#33-çok-başlı-öz-dikkat-mekanizması)
  - [3.4. İleri Beslemeli Ağlar](#34-ileri-beslemeli-ağlar)
  - [3.5. Kalıntı Bağlantılar ve Katman Normalizasyonu](#35-kalıntı-bağlantılar-ve-katman-normalizasyonu)
- [4. Eğitim ve Çıkarım](#4-eğitim-ve-çıkarım)
- [5. Etki, Uygulamalar ve Sınırlamalar](#5-etki-uygulamalar-ve-sınırlamalar)
- [6. Kod Örneği: Ölçekli Nokta Çarpımı Dikkatini Gösteren Örnek](#6-kod-örneği-ölçekli-nokta-çarpımı-dikkatini-gösteren-örnek)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
### 1. Giriş

Vaswani ve arkadaşları tarafından 2017'de yayınlanan "Attention Is All You Need" adlı çığır açan makaleyle tanıtılan **Transformatör mimarisi**, **Doğal Dil İşleme (NLP)** ve ardından daha geniş bir alan olan **Üretken Yapay Zeka** alanında devrim yarattı. Transformatör'den önce, ardışık veri işleme için, özellikle makine çevirisi, metin özetleme ve konuşma tanıma gibi görevler için **tekrarlayan sinir ağları (TSA'lar)** ve **Uzun Kısa Süreli Bellek (UKSA)** ağları gibi varyantları baskın modellerdi. Etkili olsalar da, bu modeller, **paralelleştirmeyi** engelleyen ardışık işleme ve kaybolan veya patlayan gradyanlar nedeniyle uzun menzilli bağımlılıkları yakalamakta zorlanma gibi doğuştan gelen sınırlamalara sahipti.

Transformatör mimarisi, yineleme ve evrişimleri büyük ölçüde terk ederek **öz-dikkat** adı verilen bir mekanizma lehine bu sorunları zarif bir şekilde ele alır. Bu, modelin her bir öğeyi işlerken giriş dizisinin farklı kısımlarının önemini tartmasına olanak tanır ve böylece mesafelerine bakılmaksızın bağımlılıkları etkili bir şekilde yakalar. Bu paradigma değişimi, eğitim sırasında benzeri görülmemiş düzeyde **paralel hesaplamayı** mümkün kılmakla kalmadı, aynı zamanda çok çeşitli görevlerde model performansını önemli ölçüde artırdı. Transformatör, o zamandan beri BERT, GPT, T5 gibi son teknoloji **Büyük Dil Modelleri (BDM'ler)** ve diğer birçok modelin temel mimarisi haline geldi ve **Üretken Yapay Zeka'daki** son gelişmelerin çoğunun temelini oluşturdu. Bu belge, Transformatör mimarisinin karmaşık bileşenlerine ve operasyonel prensiplerine derinlemesine bir bakış sunarak tasarımını, mekanizmalarını ve derin etkisini incelemektedir.

<a name="2-transformatörlerin-doğuşu-ve-attention-is-all-you-need-makalesi"></a>
### 2. Transformatörlerin Doğuşu ve "Attention Is All You Need" Makalesi

**Dikkat mekanizmaları** kavramı, "Attention Is All You Need" makalesine yeni değildi. Daha önceki, genellikle kodlayıcı-çözücü TSA'larına dayalı sıradan-sıraya modelleri, kod çözme sırasında giriş dizisinin ilgili kısımlarına seçici olarak odaklanmak için dikkat mekanizmasını zaten kullanmışlardı. Ancak, bu modeller hala girişi kodlamak için TSA'ların ardışık doğasına büyük ölçüde güveniyordu. Transformatörün çığır açan yönü, hem yinelemeden hem de evrişimden tamamen vazgeçerek yalnızca dikkat mekanizmalarına dayanan bir model önermesidir.

Bu radikal mimari seçimin ana motivasyonu, TSA'ların doğuştan gelen sınırlamalarını aşmaktı:
1.  **Paralelleştirme Eksikliği:** TSA'lar belirteçleri tek tek işler, bu da modern paralel hesaplama donanımlarından (GPU'lar) tam olarak yararlanamadıkları için büyük veri kümeleri üzerinde eğitimi yavaşlatır.
2.  **Uzun Menzilli Bağımlılıklarla Zorluk:** LSTN'ler ve GRU'lar bunu kısmen hafifletse de, sabit boyutlu gizli durum ve gradyan sorunları nedeniyle çok uzun diziler boyunca bilgiyi sürdürmek zorlu bir görev olarak kalmıştır.

**Öz-dikkat** ile yinelemeyi değiştirerek, Transformatör, giriş dizisindeki her bir belirtecin aynı anda diğer tüm belirteçlere dikkat etmesine izin verdi ve etkileşimleri paralel olarak hesapladı. Bu, modelin dizideki konumlarından bağımsız olarak uzak kelimeler arasındaki ilişkileri doğrudan yakalamasına olanak sağladı ve eğitim sürelerini büyük ölçüde hızlandırarak çok daha büyük modelleri geniş veri kümeleri üzerinde eğitmeyi mümkün kıldı. Bu temel değişim, şu anda **Üretken Yapay Zeka'ya** hakim olan önceden eğitilmiş dil modelleri çağının yolunu açtı.

<a name="3-transformatör-mimarisinin-temel-bileşenleri"></a>
### 3. Transformatör Mimarisinin Temel Bileşenleri

Transformatör, tipik olarak bir **kodlayıcı** yığını ve bir **çözücü** yığınından oluşan bir **sıradan-sıraya** modelidir. Her iki yığın da aynı katmanlardan oluşur ve her katman birkaç alt katman içerir.

<a name="31-kodlayıcı-çözücü-mimarisi"></a>
#### 3.1. Kodlayıcı-Çözücü Mimarisi

**Kodlayıcı**, giriş dizisini işlemden geçirmekten ve bağlamsal bir temsil oluşturmaktan sorumludur. Üst üste yığılmış `N` adet aynı katmandan oluşur. Her kodlayıcı katmanı iki ana alt katmana sahiptir: bir **Çok-Başlı Öz-Dikkat mekanizması** ve bir **konum-tabanlı tam bağlı İleri Beslemeli Ağ**.

**Çözücü**, kodlayıcının çıktısı ve daha önce üretilen belirteçler göz önüne alındığında, çıktı dizisini birer birer üretmekten sorumludur. Ayrıca `N` adet aynı katmandan oluşur. Her çözücü katmanı üç alt katmana sahiptir: bir **Maskelenmiş Çok-Başlı Öz-Dikkat mekanizması** (eğitim sırasında gelecekteki belirteçlere dikkat etmeyi önlemek için), kodlayıcının çıktısına dikkat eden bir **Çok-Başlı Dikkat mekanizması** ve bir **konum-tabanlı tam bağlı İleri Beslemeli Ağ**.

Hem kodlayıcı hem de çözücü girişleri, önce giriş belirteçlerini sürekli vektör temsillerine dönüştürmek için bir **gömülü temsil katmanından** geçer. Bu gömülü temsiller daha sonra **konumsal kodlamalarla** birleştirilir.

<a name="32-konumsal-kodlama"></a>
#### 3.2. Konumsal Kodlama

Transformatör mimarisi yineleme ve evrişimi büyük ölçüde reddettiği için, kelime sırası hakkında doğal bir anlayışa sahip değildir. Belirteçlerin dizideki göreceli veya mutlak konumu hakkında bilgi eklemek için, giriş gömülü temsillerine **konumsal kodlamalar** eklenir. Bu kodlamalar genellikle farklı frekanslara sahip sinüzoidal fonksiyonlardır ve modelin göreceli konumlara dikkat etmeyi öğrenmesini sağlar. Sezgi şudur ki, sinüs ve kosinüs fonksiyonları göreceli konumları temsil edebilir çünkü `sin(pos + k)`, `sin(pos)` ve `cos(pos)`'un doğrusal bir fonksiyonu olarak ifade edilebilir. Bu, modelin eğitim sırasında gördüğünden daha uzun dizilere genelleme yapmasını kolaylaştırır.

Matematiksel olarak, `pos` konumundaki ve `i` boyutundaki bir belirteç için konumsal kodlama (burada `i` çift bir dizindir) şu şekilde verilir:
`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
`PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
burada `d_model`, gömülü temsillerin boyutluluğudur.

<a name="33-çok-başlı-öz-dikkat-mekanizması"></a>
#### 3.3. Çok-Başlı Öz-Dikkat Mekanizması

Transformatörün kalbinde, modelin farklı giriş belirteçlerinin birbirlerine göre önemini tartmasını sağlayan **dikkat mekanizması**, özellikle de **Ölçekli Nokta Çarpımı Dikkat** bulunur. Her bir belirteç için dikkat mekanizması üç vektör hesaplar: bir **Sorgu (S)**, bir **Anahtar (A)** ve bir **Değer (D)**. Bunlar, giriş gömülü temsillerinin (veya önceki katmanın çıktısının) öğrenilmiş ağırlık matrisleri ile doğrusal olarak dönüştürülmesiyle elde edilir.

**Ölçekli Nokta Çarpımı Dikkat** fonksiyonu, dikkat skorlarını şu şekilde hesaplar:
`Dikkat(S, A, D) = softmax( (S * A^T) / sqrt(d_k) ) * D`
burada `d_k`, anahtarların boyutudur ve çok büyük nokta çarpımlarının softmax'ı aşırı küçük gradyanlara sahip bölgelere itmesini önlemek için ölçeklendirme için kullanılır. `softmax` fonksiyonu daha sonra bu skorları, her bir çıktı belirtecinin her bir giriş belirtecine ne kadar dikkat etmesi gerektiğini gösteren olasılıklara dönüştürür.

**Çok-Başlı Dikkat**, modelin farklı konumlardaki farklı temsil alt uzaylarından gelen bilgilere aynı anda dikkat etmesini sağlayarak bunu genişletir. `d_model` boyutlu anahtarlar, değerler ve sorgularla tek bir dikkat fonksiyonu gerçekleştirmek yerine, giriş `h` kez `d_k`, `d_k` ve `d_v` boyutlarına farklı, öğrenilmiş doğrusal projeksiyonlarla doğrusal olarak yansıtılır. Bu `h` yansıtılmış versiyonun her biri için paralel olarak bir dikkat fonksiyonu hesaplanır. Elde edilen `h` dikkat çıktısı daha sonra birleştirilir ve modelin farklı ilişkileri öğrenmesini sağlayacak şekilde tekrar doğrusal olarak yansıtılır.

Kodlayıcıda bu **Öz-Dikkat**'tir, çünkü `S`, `A` ve `D` hepsi aynı önceki katman çıktısından gelir. Çözücüde iki dikkat alt katmanı vardır:
1.  **Maskelenmiş Öz-Dikkat**: `S`, `A`, `D` çözücünün önceki çıktısından gelir, ancak gelecekteki konumlara dikkat etmeyi önlemek için dikkat maskelenir.
2.  **Kodlayıcı-Çözücü Dikkat**: `S` çözücünün önceki çıktısından gelirken, `A` ve `D` kodlayıcının çıktısından gelir. Bu, çözücünün giriş dizisinin ilgili kısımlarına odaklanmasını sağlar.

<a name="34-ileri-beslemeli-ağlar"></a>
#### 3.4. İleri Beslemeli Ağlar

Her kodlayıcı ve çözücü katmanı, **konum-tabanlı tam bağlı ileri beslemeli bir ağ** içerir. Bu ağ, her konuma bağımsız ve özdeş bir şekilde uygulanır. Arasında bir **ReLU aktivasyonu** bulunan iki doğrusal dönüşümden oluşur:
`İBA(x) = max(0, x * W1 + b1) * W2 + b2`
Bu basit ağ, doğrusal olmayanlık ekler ve modelin dikkat mekanizması tarafından bir araya getirilen bilgiyi daha fazla işlemesini sağlar.

<a name="35-kalıntı-bağlantılar-ve-katman-normalizasyonu"></a>
#### 3.5. Kalıntı Bağlantılar ve Katman Normalizasyonu

Her alt katman (öz-dikkat, kodlayıcı-çözücü dikkat ve ileri beslemeli ağ) için, Transformatör mimarisi etrafına bir **kalıntı bağlantı** ve ardından **katman normalizasyonu** kullanır. Yani, her alt katmanın çıktısı `KatmanNorm(x + AltKatman(x))` şeklindedir, burada `AltKatman(x)` alt katmanın kendisi tarafından uygulanan fonksiyondur.

Aslen ResNet'ten gelen **kalıntı bağlantılar**, gradyanların doğrudan kimlik eşlemesi aracılığıyla akmasına izin vererek derin ağlardaki kaybolan gradyan sorununu hafifletmeye yardımcı olur. Bu, çok daha derin modellerin eğitilmesini mümkün kılar.
**Katman Normalizasyonu**, her örnek için özellikleri bağımsız olarak normalleştirir. Bu, optimizasyon alanını daha düzgün hale getirerek eğitimi stabilize etmeye ve eğitim süresini azaltmaya yardımcı olur.

<a name="4-eğitim-ve-çıkarım"></a>
### 4. Eğitim ve Çıkarım

**Eğitim:** Transformatör, sıradan-sıraya görevleri için tipik olarak **öğretmen zorlaması** kullanılarak eğitilir. Eğitim sırasında, çözücüye bir sonraki belirteci tahmin etmek için hedef diziden doğru önceki belirteç (sağa kaydırılmış) giriş olarak sağlanır. Bu, modelin giriş bağlamı göz önüne alındığında doğru diziyi üretmeyi öğrenmesini sağlar. Amaç fonksiyonu genellikle, bir sonraki belirteci doğru bir şekilde tahmin etmek için modeli optimize eden **çapraz-entropi kaybıdır**. Özel bir öğrenme oranı zamanlayıcısı (ısınma ve ardından bozunma) ile **Adam** gibi optimize ediciler yaygın olarak kullanılır.

**Çıkarım:** Çıkarım sırasında (örn. makine çevirisi veya metin üretimi), çözücü belirteçleri birer birer üretmelidir. Süreç özel bir `<başlangıç>` belirteci ile başlar. Çözücü bir sonraki belirteci tahmin eder ve bu belirteç, kodlayıcının çıktısıyla birlikte bir sonraki adım için çözücünün girişine geri beslenir. Bu yinelemeli süreç, bir `<bitiş>` belirteci üretilene veya maksimum dizi uzunluğuna ulaşılana kadar devam eder. Daha optimal bir çıktı dizisi bulmak için birden fazla potansiyel çıktı dizisini keşfetmek amacıyla açgözlü kod çözme yerine genellikle **hüzme arama** kullanılır.

<a name="5-etki-uygulamalar-ve-sınırlamalar"></a>
### 5. Etki, Uygulamalar ve Sınırlamalar

**Transformatör**, **Üretken Yapay Zeka** ve **NLP** alanlarını derinden etkilemiştir. Uzun menzilli bağımlılıkları verimli bir şekilde modelleme yeteneği ve paralel hesaplamaya yatkınlığı, benzeri görülmemiş ilerlemelere yol açmıştır.

**Temel Etkiler:**
*   **BDM'ler için Temel:** Transformatörler, Google'ın BERT ve T5'i, OpenAI'nin GPT serisi, Meta'nın LLaMA'sı ve diğerleri gibi hemen hemen tüm modern **Büyük Dil Modellerinin (BDM'ler)** omurgasını oluşturur.
*   **Transfer Öğrenimi:** Mimari, bir modelin büyük, etiketsiz bir metin kümesi (örn. internet metni) üzerinde eğitildiği ve daha sonra daha küçük, göreve özel veri kümeleri üzerinde ince ayar yapıldığı ön eğitim paradigmasını mümkün kıldı ve önemli performans artışlarına yol açtı.
*   **Çok Modluluk:** Transformatörler, NLP'nin ötesine bilgisayar görüşü (Vision Transformers), konuşma işleme ve hatta farklı veri türlerini birleştiren çok modlu modellere kadar genişlemiştir.

**Uygulamalar:**
*   **Makine Çevirisi:** Diller arasında çeviri yapmada son teknoloji sonuçlar.
*   **Metin Özetleme:** Daha uzun metinlerin kısa özetlerini oluşturma.
*   **Soru Cevaplama:** Verilen bağlamlara dayanarak soruları anlama ve yanıtlama.
*   **Metin Üretimi:** Yaratıcı yazımdan kod üretimine kadar çeşitli amaçlar için insan benzeri metin oluşturma.
*   **Duygu Analizi, Adlandırılmış Varlık Tanıma ve daha fazlası.**

**Sınırlamalar:**
*   **Karesel Karmaşıklık:** Standart öz-dikkat mekanizması, dizi uzunluğuna göre karesel bir hesaplama ve bellek karmaşıklığına (`O(N^2)`) sahiptir, bu da çok uzun dizileri (örn. tüm kitaplar) değişiklikler olmadan işlemeyi zorlaştırır.
*   **İndüktif Önyargı Eksikliği:** Yerel özellikler için indüktif önyargısı olan evrişimli ağların veya ardışık sıra için TSA'ların aksine, Transformatörün "boş sayfa" doğası, bu ilişkileri öğrenmek için çok büyük miktarda veri gerektirir. Konumsal kodlamalar bunu hafifletmeye çalışır, ancak yine de bir faktördür.
*   **Yüksek Hesaplama Maliyeti:** Paralelleştirilebilir olsa da, büyük Transformatör modellerinin, özellikle BDM'lerin eğitimi ve çalıştırılması önemli hesaplama kaynakları (GPU'lar/TPU'lar) ve enerji gerektirir.

Araştırmacılar, verimliliği artırmak ve daha uzun bağlamları ele almak için **seyrek dikkat mekanizmaları**, **tekrarlayan transformatörler** ve diğer mimari optimizasyonlar gibi bu sınırlamaları ele almanın yollarını aktif olarak araştırmaktadır.

<a name="6-kod-örneği-ölçekli-nokta-çarpımı-dikkatini-gösteren-örnek"></a>
### 6. Kod Örneği: Ölçekli Nokta Çarpımı Dikkatini Gösteren Örnek

İşte **Ölçekli Nokta Çarpımı Dikkatinin** temel matematiksel işlemini gösteren basitleştirilmiş bir Python kod parçacığı. Kısalık açısından çok-başlı veya maskeleme içermemekle birlikte, `S * A^T` işlemini, ölçeklendirmeyi ve softmax'ı gösterir.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Ölçekli nokta çarpımı dikkatini hesaplar.
    Argümanlar:
        query (torch.Tensor): Sorgu tensörü (batch_size, num_heads, seq_len_q, d_k).
        key (torch.Tensor): Anahtar tensörü (batch_size, num_heads, seq_len_k, d_k).
        value (torch.Tensor): Değer tensörü (batch_size, num_heads, seq_len_v, d_v).
        mask (torch.Tensor, opsiyonel): Belirli pozisyonlara dikkati engellemek için maske tensörü.
    Döndürür:
        torch.Tensor: Dikkat mekanizmasının çıktısı.
        torch.Tensor: Dikkat ağırlıkları (dikkat skorları).
    """
    d_k = query.size(-1) # Anahtarların boyutunu al
    
    # Dikkat skorlarını hesapla: Sorgu * Anahtar^T
    # (batch_size, num_heads, seq_len_q, d_k) @ (batch_size, num_heads, d_k, seq_len_k)
    # -> (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Skorları ölçekle
    scores = scores / (d_k ** 0.5)
    
    # Sağlanmışsa maskeyi uygula (örn. çözücüdeki maskelenmiş öz-dikkat için)
    if mask is not None:
        # Maskelenmiş pozisyonları çok büyük negatif bir sayıyla doldur
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Dikkat ağırlıklarını (olasılıkları) almak için softmax uygula
    attention_weights = F.softmax(scores, dim=-1)
    
    # Ağırlıkları Değer tensörü ile çarp
    # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_v, d_v)
    # -> (batch_size, num_heads, seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Örnek kullanım (basitleştirilmiş, yığın veya çok-başlı boyutlar olmadan)
# d_model = 512, d_k = 64, d_v = 64, dizi_uzunluğu = 10 varsayalım
seq_len = 10
d_k = 64
d_v = 64

# Sahte Sorgu, Anahtar, Değer tensörleri oluştur
# Öz-dikkat için, S, A, D aynı kaynaktan gelir
dummy_input = torch.randn(seq_len, d_k) # Basitleştirilmiş tek-başlı, tek-yığın
query = dummy_input
key = dummy_input
value = dummy_input

# 3 farklı sorgu konumunu işlemek istediğimizi varsayalım.
# Bunları (seq_len_q, d_k), (seq_len_k, d_k), (seq_len_v, d_v) simüle etmek için yeniden şekillendirelim
query = torch.randn(3, d_k) # Örnek: 3 sorgu vektörü
key = torch.randn(seq_len, d_k) # Tüm 10 giriş anahtarı
value = torch.randn(seq_len, d_v) # Tüm 10 giriş değeri

attention_output, weights = scaled_dot_product_attention(query.unsqueeze(0).unsqueeze(0), 
                                                         key.unsqueeze(0).unsqueeze(0), 
                                                         value.unsqueeze(0).unsqueeze(0))

print("Dikkat Çıkışı Şekli:", attention_output.squeeze(0).squeeze(0).shape)
print("Dikkat Ağırlıkları Şekli:", weights.squeeze(0).squeeze(0).shape)
# Her 3 sorgu konumunun artık d_v boyutunda bir çıktı vektörü var
# ve her 10 giriş konumuna ne kadar odaklandığını gösteren dikkat ağırlıkları var.

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
### 7. Sonuç

**Transformatör mimarisi**, **Üretken Yapay Zeka** ve **derin öğrenme** alanında anıtsal bir başarı olarak durmaktadır. **Öz-dikkat mekanizmalarını** ve **konumsal kodlamalar**, **kalıntı bağlantılar** ve **katman normalizasyonu** gibi yeni mimari tasarımları ustaca kullanarak, önceki ardışık modellerin sınırlamalarını etkili bir şekilde ele almaktadır. Bilgiyi paralel olarak işleme ve karmaşık, uzun menzilli bağımlılıkları yakalama yeteneği, yalnızca **Doğal Dil İşleme'yi** yeni zirvelere taşımakla kalmamış, aynı zamanda bilgisayar görüşü ve çok modlu öğrenmedeki ilerlemelerin de kapısını aralamıştır. Son derece uzun diziler için hesaplama karmaşıklığı ile ilgili zorluklar devam etse de, devam eden araştırmalar Transformatörün yeteneklerini geliştirmeye ve genişletmeye devam etmektedir. "Attention Is All You Need" makalesinin ve mimari yeniliklerinin kalıcı etkisi, Transformatörün modern Yapay Zeka'nın temel taşı olarak rolünü ve öngörülebilir gelecek için akıllı sistemlerin manzarasını şekillendirme gücünü vurgulamaktadır.