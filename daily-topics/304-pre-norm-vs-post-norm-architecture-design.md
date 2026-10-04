# Pre-Norm vs. Post-Norm Architecture Design

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Transformer Architecture: A Brief Overview](#2-transformer-architecture-brief-overview)
- [3. Normalization Techniques in Neural Networks](#3-normalization-techniques)
    - [3.1. Layer Normalization](#3-1-layer-normalization)
- [4. Post-Normalization Architecture (Post-Norm)](#4-post-normalization)
    - [4.1. Characteristics and Implementation](#4-1-characteristics-implementation-post)
    - [4.2. Advantages and Disadvantages](#4-2-advantages-disadvantages-post)
- [5. Pre-Normalization Architecture (Pre-Norm)](#5-pre-normalization)
    - [5.1. Characteristics and Implementation](#5-1-characteristics-implementation-pre)
    - [5.2. Advantages and Disadvantages](#5-2-advantages-disadvantages-pre)
- [6. Empirical Evidence and Best Practices](#6-empirical-evidence)
- [7. Code Example](#7-code-example)
- [8. Conclusion](#8-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The **Transformer architecture**, introduced by Vaswani et al. (2017) in "Attention Is All You Need," has revolutionized the field of Generative AI, becoming the backbone for state-of-the-art models in Natural Language Processing (NLP) and increasingly in computer vision and other domains. A critical component enabling the effective training and scaling of these deep neural networks is **normalization**. Within the Transformer block, specifically in its feed-forward and multi-head attention sub-layers, the placement of **Layer Normalization (LayerNorm)** has evolved, leading to two primary architectural paradigms: **Post-Normalization (Post-Norm)** and **Pre-Normalization (Pre-Norm)**.

This document delves into the fundamental differences between Post-Norm and Pre-Norm architectures, exploring their distinct implementations, theoretical underpinnings, and practical implications for model training stability and performance. Understanding these architectural nuances is crucial for researchers and practitioners aiming to design, optimize, and scale large-scale generative models effectively.

<a name="2-transformer-architecture-brief-overview"></a>
## 2. Transformer Architecture: A Brief Overview
At its core, a Transformer block consists of two main sub-layers: a **Multi-Head Self-Attention** mechanism and a position-wise **Feed-Forward Network (FFN)**. Both sub-layers are typically followed by a **residual connection** and a **normalization layer**. The residual connection, also known as a skip connection, helps alleviate the vanishing gradient problem in deep networks by allowing gradients to flow directly through the identity mapping. The normalization layer, predominantly **Layer Normalization**, stabilizes activations, enabling faster and more stable training.

The output of each sub-layer can be generally described as `SubLayer(x) + x`, where `x` is the input to the sub-layer. The placement of LayerNorm relative to `SubLayer(x)` and `x` defines the Pre-Norm and Post-Norm variants.

<a name="3-normalization-techniques"></a>
## 3. Normalization Techniques in Neural Networks
Normalization layers are a cornerstone of deep learning, addressing issues like internal covariate shift, unstable gradients, and sensitivity to initialization. Various normalization techniques exist, such as Batch Normalization, Layer Normalization, Instance Normalization, and Group Normalization. For Transformer models, **Layer Normalization** is almost exclusively preferred due to its independence from batch size and its effectiveness in recurrent and self-attention networks where feature dimensions vary across time steps or positions.

<a name="3-1-layer-normalization"></a>
### 3.1. Layer Normalization
**Layer Normalization (LayerNorm)** normalizes the inputs across the features of an individual training example, rather than across a batch of examples. For a given input `x` with `D` features, LayerNorm computes the mean `μ` and variance `σ^2` across these `D` features for each sample independently. The normalized output `y` is then computed as:

`y = γ * ((x - μ) / √(σ^2 + ε)) + β`

where `γ` and `β` are learnable affine transformation parameters that scale and shift the normalized output, allowing the network to retain representational power. `ε` is a small constant added for numerical stability. This per-sample, per-feature normalization makes LayerNorm particularly suitable for variable-length sequences and contexts where batch statistics are unreliable or vary significantly.

<a name="4-post-normalization"></a>
## 4. Post-Normalization Architecture (Post-Norm)
The original Transformer architecture, as proposed by Vaswani et al. (2017), employs a **Post-Normalization (Post-Norm)** design.

<a name="4-1-characteristics-implementation-post"></a>
### 4.1. Characteristics and Implementation
In a Post-Norm architecture, the Layer Normalization operation is applied *after* the residual connection and *after* the sub-layer computation (Multi-Head Attention or Feed-Forward Network). The general structure within a Transformer block for a sub-layer can be represented as:

`y = x + SubLayer(x)`
`output = LayerNorm(y)`

This means the input `x` first goes through the `SubLayer(x)`, then its output is added to the original `x` (residual connection), and *then* the combined result is normalized.

The complete sub-layer operation would look like:
1. `attention_output = MultiHeadAttention(query, key, value)`
2. `attention_residual_added = attention_output + x`
3. `attention_norm = LayerNorm(attention_residual_added)`
4. `ffn_output = FeedForwardNetwork(attention_norm)`
5. `ffn_residual_added = ffn_output + attention_norm`
6. `final_output = LayerNorm(ffn_residual_added)`

<a name="4-2-advantages-disadvantages-post"></a>
### 4.2. Advantages and Disadvantages
**Advantages:**
*   **Original Formulation:** It's the architecture used in the original Transformer paper, hence well-established.
*   **Performance on Simpler Tasks:** For smaller models or simpler tasks, Post-Norm can sometimes achieve slightly better final performance after convergence due to the normalization layer directly influencing the subsequent layer's input.

**Disadvantages:**
*   **Training Instability:** The primary drawback of Post-Norm is its **training instability**, especially with deeper models or when using higher learning rates. Because LayerNorm is applied *after* the residual connection, the scale of the activations can grow significantly large before normalization, particularly in early training stages or if the sub-layers produce large outputs. This can lead to very large gradients and make the model difficult to train, often requiring careful learning rate scheduling (e.g., warm-up) and smaller initial learning rates.
*   **Sensitivity to Initialization:** It is more sensitive to weight initialization schemes.

<a name="5-pre-normalization"></a>
## 5. Pre-Normalization Architecture (Pre-Norm)
To address the training instability of Post-Norm, the **Pre-Normalization (Pre-Norm)** architecture was proposed, notably popularized by models like GPT-2 and widely adopted in subsequent large language models.

<a name="5-1-characteristics-implementation-pre"></a>
### 5.1. Characteristics and Implementation
In a Pre-Norm architecture, the Layer Normalization operation is applied *before* the sub-layer computation (Multi-Head Attention or Feed-Forward Network) and *before* the residual connection. The general structure within a Transformer block for a sub-layer can be represented as:

`normalized_x = LayerNorm(x)`
`output = x + SubLayer(normalized_x)`

This means the input `x` is first normalized, then the normalized input `normalized_x` is passed to the `SubLayer`, and finally, the output of `SubLayer` is added to the *original* `x` (residual connection).

The complete sub-layer operation would look like:
1. `attention_input_norm = LayerNorm(x)`
2. `attention_output = MultiHeadAttention(attention_input_norm, attention_input_norm, attention_input_norm)`
3. `attention_residual_added = attention_output + x`
4. `ffn_input_norm = LayerNorm(attention_residual_added)`
5. `ffn_output = FeedForwardNetwork(ffn_input_norm)`
6. `final_output = ffn_output + attention_residual_added`

<a name="5-2-advantages-disadvantages-pre"></a>
### 5.2. Advantages and Disadvantages
**Advantages:**
*   **Improved Training Stability:** This is the most significant advantage. By normalizing the input to each sub-layer, Pre-Norm ensures that the activations passed to the attention and FFN layers are always within a stable range, regardless of the depth of the network or the learning rate. This prevents activation explosion and gradient instability, allowing for much deeper models, faster convergence, and the use of higher learning rates without extensive warm-up.
*   **Easier Optimization:** The stable gradient flow makes the optimization landscape smoother and easier to navigate, leading to more robust training.
*   **Common in Modern LLMs:** It is the de-facto standard for training large-scale generative models like GPT-3, T5, LLaMA, and their successors due to its superior training characteristics.

**Disadvantages:**
*   **Performance Gap:** In some very specific cases or very shallow networks, Post-Norm *might* achieve a slightly better final validation loss or accuracy, though this difference is often negligible and outweighed by Pre-Norm's stability.
*   **Initial Optimization Challenge:** Some research suggests that Pre-Norm might initially optimize slower than Post-Norm, as the gradients through the residual path are effectively attenuated by the normalization. However, this is generally compensated by its overall stability and ability to handle larger learning rates.

<a name="6-empirical-evidence"></a>
## 6. Empirical Evidence and Best Practices
The shift from Post-Norm to Pre-Norm in Transformer architectures is largely driven by empirical observations in scaling large models. Researchers found that as Transformer models grow in depth and width, Post-Norm models become increasingly challenging to train, often failing to converge or suffering from numerical instability. The paper "On Layer Normalization in the Transformer Architecture" by Xiong et al. (2020) provides a detailed analysis, demonstrating that Pre-Norm significantly improves gradient flow and allows for much deeper networks to be trained successfully.

Modern **Large Language Models (LLMs)** predominantly employ the Pre-Norm configuration. This design choice is fundamental to their ability to scale to hundreds of billions of parameters while maintaining stable and efficient training. While initial training steps for Pre-Norm might seem to progress slower in terms of immediate loss reduction compared to a Post-Norm model with a carefully tuned warm-up, its long-term stability and ability to use aggressive learning schedules typically lead to better overall performance and faster wall-clock training time for very large models.

<a name="7-code-example"></a>
## 7. Code Example
Here’s a simplified Python (PyTorch-like) representation of a single Transformer sub-layer showing the difference between Pre-Norm and Post-Norm.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # Simplified attention: just linear projections and aggregation
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        # In a real MHA, this would involve splitting heads, scaled dot-product attention, concatenation
        output = (q + k + v) / 3 # Placeholder for actual attention logic
        return self.linear_out(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderBlockPostNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention Sub-layer (Post-Norm)
        # Output of attention is added to input, THEN normalized.
        attn_output = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x) # LayerNorm AFTER residual connection

        # Feed-Forward Sub-layer (Post-Norm)
        # Output of FFN is added to previous output, THEN normalized.
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x) # LayerNorm AFTER residual connection
        return x

class TransformerEncoderBlockPreNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention Sub-layer (Pre-Norm)
        # Input is normalized, THEN passed to attention. Output is added to original input.
        norm_x = self.norm1(x) # LayerNorm BEFORE sub-layer
        attn_output = self.attention(norm_x, norm_x, norm_x)
        x = x + self.dropout(attn_output) # Residual connection with original 'x'

        # Feed-Forward Sub-layer (Pre-Norm)
        # Input is normalized, THEN passed to FFN. Output is added to previous residual.
        norm_x = self.norm2(x) # LayerNorm BEFORE sub-layer
        ffn_output = self.feed_forward(norm_x)
        x = x + self.dropout(ffn_output) # Residual connection with previous 'x'
        return x

# Example usage:
d_model = 512 # Embedding dimension
num_heads = 8
d_ff = 2048 # Feed-forward dimension
batch_size = 4
seq_len = 10

dummy_input = torch.randn(batch_size, seq_len, d_model)

# Post-Norm Block
post_norm_block = TransformerEncoderBlockPostNorm(d_model, num_heads, d_ff)
post_norm_output = post_norm_block(dummy_input)
print(f"Post-Norm Output shape: {post_norm_output.shape}")

# Pre-Norm Block
pre_norm_block = TransformerEncoderBlockPreNorm(d_model, num_heads, d_ff)
pre_norm_output = pre_norm_block(dummy_input)
print(f"Pre-Norm Output shape: {pre_norm_output.shape}")

(End of code example section)
```

<a name="8-conclusion"></a>
## 8. Conclusion
The choice between Post-Normalization and Pre-Normalization architecture in Transformer models represents a significant design decision with profound implications for training dynamics and scalability. While Post-Norm was the original design, its susceptibility to training instability and gradient issues, especially in deeper models, led to the widespread adoption of Pre-Norm.

The Pre-Norm architecture, by applying Layer Normalization *before* each sub-layer and the residual connection, effectively stabilizes activations and gradients throughout the network. This property is paramount for training the massive, deep generative AI models that define the current state-of-the-art. Modern research and practical implementations almost universally favor Pre-Norm for its robustness, enabling researchers to push the boundaries of model size and complexity with greater confidence. Understanding this distinction is not merely an academic exercise but a practical necessity for anyone building or working with contemporary Transformer-based generative models.

---
<br>

<a name="türkçe-içerik"></a>
## Önişlemeli (Pre-Norm) ve Sonişlemeli (Post-Norm) Mimari Tasarımı

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Transformer Mimarisine Kısa Bir Bakış](#2-transformer-mimarisine-kisa-bir-bakis)
- [3. Sinir Ağlarında Normalizasyon Teknikleri](#3-sinir-aglarinda-normalizasyon-teknikleri)
    - [3.1. Katman Normalizasyonu](#3-1-katman-normalizasyonu)
- [4. Sonişlemeli Normalizasyon Mimarisi (Post-Norm)](#4-sonislemeli-normalizasyon)
    - [4.1. Özellikleri ve Uygulaması](#4-1-ozellikleri-uygulamasi-post)
    - [4.2. Avantajları ve Dezavantajları](#4-2-avantajlari-dezavantajlari-post)
- [5. Önişlemeli Normalizasyon Mimarisi (Pre-Norm)](#5-onislemeli-normalizasyon)
    - [5.1. Özellikleri ve Uygulaması](#5-1-ozellikleri-uygulamasi-pre)
    - [5.2. Avantajları ve Dezavantajları](#5-2-avantajlari-dezavantajlari-pre)
- [6. Ampirik Kanıtlar ve En İyi Uygulamalar](#6-ampirik-kanitlar)
- [7. Kod Örneği](#7-kod-ornegi)
- [8. Sonuç](#8-sonuc)

<a name="1-giriş"></a>
## 1. Giriş
Vaswani ve arkadaşları tarafından 2017'de "Attention Is All You Need" adlı makalede tanıtılan **Transformer mimarisi**, Üretken Yapay Zeka (Generative AI) alanında devrim yaratmış ve Doğal Dil İşleme (NLP) ile giderek artan bir şekilde bilgisayar görüşü ve diğer alanlardaki en gelişmiş modellerin temelini oluşturmuştur. Bu derin sinir ağlarının etkili bir şekilde eğitilmesini ve ölçeklenmesini sağlayan kritik bir bileşen **normalizasyondur**. Özellikle Transformer bloğunda, ileri beslemeli ve çok başlı dikkat alt katmanlarında, **Katman Normalizasyonunun (LayerNorm)** yerleşimi evrilerek iki ana mimari paradigmaya yol açmıştır: **Sonişlemeli Normalizasyon (Post-Normalization veya Post-Norm)** ve **Önişlemeli Normalizasyon (Pre-Normalization veya Pre-Norm)**.

Bu belge, Post-Norm ve Pre-Norm mimarileri arasındaki temel farkları, bunların ayrı uygulamalarını, teorik temellerini ve model eğitim kararlılığı ile performansına yönelik pratik etkilerini incelemektedir. Bu mimari nüansları anlamak, büyük ölçekli üretken modelleri etkin bir şekilde tasarlamayı, optimize etmeyi ve ölçeklendirmeyi amaçlayan araştırmacılar ve uygulayıcılar için kritik öneme sahiptir.

<a name="2-transformer-mimarisine-kisa-bir-bakis"></a>
## 2. Transformer Mimarisine Kısa Bir Bakış
Temelde, bir Transformer bloğu iki ana alt katmandan oluşur: bir **Çok Başlı Öz Dikkat (Multi-Head Self-Attention)** mekanizması ve konum bazlı bir **İleri Beslemeli Ağ (Feed-Forward Network - FFN)**. Her iki alt katman da tipik olarak bir **kalan bağlantı (residual connection)** ve bir **normalizasyon katmanı** ile takip edilir. Atlamalı bağlantı olarak da bilinen kalan bağlantı, gradyanların doğrudan kimlik eşlemesi yoluyla akmasına izin vererek derin ağlardaki kaybolan gradyan sorununu hafifletmeye yardımcı olur. Ağırlıklı olarak **Katman Normalizasyonu** olan normalizasyon katmanı, aktivasyonları stabilize ederek daha hızlı ve daha kararlı eğitime olanak tanır.

Her alt katmanın çıktısı genellikle `SubLayer(x) + x` olarak tanımlanabilir, burada `x` alt katmana verilen girdidir. LayerNorm'un `SubLayer(x)` ve `x`'e göre yerleşimi, Pre-Norm ve Post-Norm varyantlarını tanımlar.

<a name="3-sinir-aglarinda-normalizasyon-teknikleri"></a>
## 3. Sinir Ağlarında Normalizasyon Teknikleri
Normalizasyon katmanları, derin öğrenmenin temel taşlarından biridir ve iç kovaryat kayması, dengesiz gradyanlar ve başlatmaya karşı hassasiyet gibi sorunları ele alır. Toplu Normalizasyon (Batch Normalization), Katman Normalizasyonu (Layer Normalization), Örnek Normalizasyonu (Instance Normalization) ve Grup Normalizasyonu (Group Normalization) gibi çeşitli normalizasyon teknikleri mevcuttur. Transformer modelleri için, toplu boyuttan bağımsızlığı ve özyinelemeli ve öz dikkat ağlarında özellik boyutlarının zaman adımları veya konumlar arasında değiştiği durumlardaki etkinliği nedeniyle neredeyse yalnızca **Katman Normalizasyonu** tercih edilir.

<a name="3-1-katman-normalizasyonu"></a>
### 3.1. Katman Normalizasyonu
**Katman Normalizasyonu (LayerNorm)**, girdileri tek bir eğitim örneğinin özellikleri üzerinde normalleştirir, örneklerin bir toplu işi üzerinde değil. `D` özelliğe sahip belirli bir `x` girdisi için LayerNorm, her örnek için bağımsız olarak bu `D` özellikler üzerindeki ortalama `μ` ve varyans `σ^2` değerlerini hesaplar. Normalleştirilmiş çıktı `y` daha sonra şu şekilde hesaplanır:

`y = γ * ((x - μ) / √(σ^2 + ε)) + β`

burada `γ` ve `β`, öğrenilebilir afin dönüşüm parametreleri olup, normalize edilmiş çıktıyı ölçeklendirir ve kaydırır, böylece ağın temsil gücünü korumasını sağlar. `ε`, sayısal kararlılık için eklenen küçük bir sabittir. Bu örnek başına, özellik başına normalizasyon, LayerNorm'u özellikle değişken uzunluklu diziler ve toplu istatistiklerin güvenilmez olduğu veya önemli ölçüde değiştiği bağlamlar için uygun hale getirir.

<a name="4-sonislemeli-normalizasyon"></a>
## 4. Sonişlemeli Normalizasyon Mimarisi (Post-Norm)
Vaswani ve arkadaşları (2017) tarafından önerilen orijinal Transformer mimarisi, bir **Sonişlemeli Normalizasyon (Post-Norm)** tasarımını kullanır.

<a name="4-1-ozellikleri-uygulamasi-post"></a>
### 4.1. Özellikleri ve Uygulaması
Post-Norm mimarisinde, Katman Normalizasyon işlemi, kalan bağlantıdan *sonra* ve alt katman hesaplamasından (Çok Başlı Dikkat veya İleri Beslemeli Ağ) *sonra* uygulanır. Bir Transformer bloğu içindeki bir alt katman için genel yapı şu şekilde temsil edilebilir:

`y = x + SubLayer(x)`
`output = LayerNorm(y)`

Bu, `x` girdisinin önce `SubLayer(x)`'ten geçtiği, ardından çıktısının orijinal `x`'e (kalan bağlantı) eklendiği ve *sonra* birleşik sonucun normalize edildiği anlamına gelir.

Tam alt katman işlemi şu şekilde görünür:
1. `attention_output = MultiHeadAttention(query, key, value)`
2. `attention_residual_added = attention_output + x`
3. `attention_norm = LayerNorm(attention_residual_added)`
4. `ffn_output = FeedForwardNetwork(attention_norm)`
5. `ffn_residual_added = ffn_output + attention_norm`
6. `final_output = LayerNorm(ffn_residual_added)`

<a name="4-2-avantajlari-dezavantajlari-post"></a>
### 4.2. Avantajları ve Dezavantajları
**Avantajları:**
*   **Orijinal Formülasyon:** Orijinal Transformer makalesinde kullanılan mimaridir, dolayısıyla iyi kurulmuştur.
*   **Daha Basit Görevlerde Performans:** Daha küçük modeller veya daha basit görevler için Post-Norm, yakınsamadan sonra bazen biraz daha iyi nihai performans elde edebilir, çünkü normalizasyon katmanı doğrudan sonraki katmanın girdisini etkiler.

**Dezavantajları:**
*   **Eğitim Kararsızlığı:** Post-Norm'un birincil dezavantajı, özellikle daha derin modellerde veya daha yüksek öğrenme oranları kullanıldığında **eğitim kararsızlığıdır**. LayerNorm, kalan bağlantıdan *sonra* uygulandığı için, aktivasyonların ölçeği normalizasyondan önce önemli ölçüde büyüyebilir, özellikle erken eğitim aşamalarında veya alt katmanlar büyük çıktılar üretiyorsa. Bu durum, çok büyük gradyanlara yol açabilir ve modeli eğitmeyi zorlaştırabilir, genellikle dikkatli öğrenme oranı zamanlaması (örneğin, ısınma) ve daha küçük başlangıç öğrenme oranları gerektirir.
*   **Başlatmaya Duyarlılık:** Ağırlık başlatma şemalarına karşı daha hassastır.

<a name="5-onislemeli-normalizasyon"></a>
## 5. Önişlemeli Normalizasyon Mimarisi (Pre-Norm)
Post-Norm'un eğitim kararsızlığını gidermek için, özellikle GPT-2 gibi modellerle popüler hale gelen ve sonraki büyük dil modellerinde yaygın olarak benimsenen **Önişlemeli Normalizasyon (Pre-Norm)** mimarisi önerilmiştir.

<a name="5-1-ozellikleri-uygulamasi-pre"></a>
### 5.1. Özellikleri ve Uygulaması
Pre-Norm mimarisinde, Katman Normalizasyon işlemi, alt katman hesaplamasından (Çok Başlı Dikkat veya İleri Beslemeli Ağ) *önce* ve kalan bağlantıdan *önce* uygulanır. Bir Transformer bloğu içindeki bir alt katman için genel yapı şu şekilde temsil edilebilir:

`normalized_x = LayerNorm(x)`
`output = x + SubLayer(normalized_x)`

Bu, `x` girdisinin önce normalize edildiği, ardından normalize edilmiş girdi `normalized_x`'in `SubLayer`'a verildiği ve son olarak `SubLayer` çıktısının *orijinal* `x`'e (kalan bağlantı) eklendiği anlamına gelir.

Tam alt katman işlemi şu şekilde görünür:
1. `attention_input_norm = LayerNorm(x)`
2. `attention_output = MultiHeadAttention(attention_input_norm, attention_input_norm, attention_input_norm)`
3. `attention_residual_added = attention_output + x`
4. `ffn_input_norm = LayerNorm(attention_residual_added)`
5. `ffn_output = FeedForwardNetwork(ffn_input_norm)`
6. `final_output = ffn_output + attention_residual_added`

<a name="5-2-avantajlari-dezavantajlari-pre"></a>
### 5.2. Avantajları ve Dezavantajları
**Avantajları:**
*   **Geliştirilmiş Eğitim Kararlılığı:** Bu, en önemli avantajıdır. Her alt katmanın girdisini normalize ederek, Pre-Norm, ağın derinliğinden veya öğrenme oranından bağımsız olarak dikkat ve FFN katmanlarına iletilen aktivasyonların her zaman kararlı bir aralıkta kalmasını sağlar. Bu, aktivasyon patlamasını ve gradyan kararsızlığını önler, çok daha derin modellerin, daha hızlı yakınsamanın ve kapsamlı ısınma olmaksızın daha yüksek öğrenme oranlarının kullanılmasına olanak tanır.
*   **Daha Kolay Optimizasyon:** Kararlı gradyan akışı, optimizasyon manzarasını daha pürüzsüz ve gezinmesi daha kolay hale getirir, bu da daha sağlam bir eğitime yol açar.
*   **Modern LLM'lerde Yaygın:** GPT-3, T5, LLaMA gibi büyük ölçekli üretken modelleri ve onların ardıllarını eğitmek için üstün eğitim özellikleri nedeniyle fiili standarttır.

**Dezavantajları:**
*   **Performans Farkı:** Bazı çok özel durumlarda veya çok sığ ağlarda, Post-Norm *biraz* daha iyi nihai doğrulama kaybı veya doğruluğu elde edebilir, ancak bu fark genellikle önemsizdir ve Pre-Norm'un kararlılığı tarafından gölgede bırakılır.
*   **Başlangıç Optimizasyon Zorluğu:** Bazı araştırmalar, Pre-Norm'un, kalan yol boyunca gradyanların normalizasyon tarafından etkili bir şekilde zayıflatılması nedeniyle başlangıçta Post-Norm'dan daha yavaş optimize olabileceğini öne sürmektedir. Ancak, bu durum genellikle genel kararlılığı ve daha büyük öğrenme oranlarını işleme yeteneği ile telafi edilir.

<a name="6-ampirik-kanitlar"></a>
## 6. Ampirik Kanıtlar ve En İyi Uygulamalar
Transformer mimarilerinde Post-Norm'dan Pre-Norm'a geçiş, büyük modelleri ölçeklendirmedeki ampirik gözlemlerle büyük ölçüde yönlendirilmiştir. Araştırmacılar, Transformer modelleri derinlik ve genişlik açısından büyüdükçe, Post-Norm modellerinin eğitilmesinin giderek daha zor hale geldiğini, genellikle yakınsayamadığını veya sayısal kararsızlıktan muzdarip olduğunu bulmuşlardır. Xiong ve arkadaşları (2020) tarafından yazılan "On Layer Normalization in the Transformer Architecture" adlı makale, Pre-Norm'un gradyan akışını önemli ölçüde iyileştirdiğini ve çok daha derin ağların başarılı bir şekilde eğitilmesine izin verdiğini gösteren ayrıntılı bir analiz sunmaktadır.

Modern **Büyük Dil Modelleri (LLM'ler)** ağırlıklı olarak Pre-Norm konfigürasyonunu kullanır. Bu tasarım seçimi, yüz milyarlarca parametreye ölçeklenebilme yeteneklerinin temelidir ve bunu yaparken kararlı ve verimli eğitimi sürdürür. Pre-Norm'un başlangıç eğitim adımları, dikkatle ayarlanmış bir ısınma ile Post-Norm modeline kıyasla anlık kayıp azaltma açısından daha yavaş ilerliyor gibi görünse de, uzun vadeli kararlılığı ve agresif öğrenme programlarını kullanma yeteneği, genellikle çok büyük modeller için genel olarak daha iyi performans ve daha hızlı gerçek zamanlı eğitim süresi sağlar.

<a name="7-kod-ornegi"></a>
## 7. Kod Örneği
İşte Pre-Norm ve Post-Norm arasındaki farkı gösteren basitleştirilmiş bir Python (PyTorch benzeri) tek bir Transformer alt katmanı temsili.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # Basitleştirilmiş dikkat: sadece doğrusal projeksiyonlar ve toplama
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        # Gerçek bir MHA'da, bu, başlıkları bölmeyi, ölçekli nokta çarpımı dikkatini, birleştirmeyi içerirdi.
        output = (q + k + v) / 3 # Gerçek dikkat mantığı için yer tutucu
        return self.linear_out(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderBlockPostNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Çok Başlı Dikkat Alt Katmanı (Post-Norm)
        # Dikkat çıktısı girdiye eklenir, SONRA normalize edilir.
        attn_output = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x) # Kalan bağlantıdan SONRA LayerNorm

        # İleri Beslemeli Alt Katman (Post-Norm)
        # FFN çıktısı önceki çıktıya eklenir, SONRA normalize edilir.
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x) # Kalan bağlantıdan SONRA LayerNorm
        return x

class TransformerEncoderBlockPreNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Çok Başlı Dikkat Alt Katmanı (Pre-Norm)
        # Girdi normalize edilir, SONRA dikkate iletilir. Çıktı orijinal girdiye eklenir.
        norm_x = self.norm1(x) # Alt katmandan ÖNCE LayerNorm
        attn_output = self.attention(norm_x, norm_x, norm_x)
        x = x + self.dropout(attn_output) # Orijinal 'x' ile kalan bağlantı

        # İleri Beslemeli Alt Katman (Pre-Norm)
        # Girdi normalize edilir, SONRA FFN'ye iletilir. Çıktı önceki kalana eklenir.
        norm_x = self.norm2(x) # Alt katmandan ÖNCE LayerNorm
        ffn_output = self.feed_forward(norm_x)
        x = x + self.dropout(ffn_output) # Önceki 'x' ile kalan bağlantı
        return x

# Örnek kullanım:
d_model = 512 # Gömme boyutu
num_heads = 8
d_ff = 2048 # İleri besleme boyutu
batch_size = 4
seq_len = 10

dummy_input = torch.randn(batch_size, seq_len, d_model)

# Post-Norm Bloğu
post_norm_block = TransformerEncoderBlockPostNorm(d_model, num_heads, d_ff)
post_norm_output = post_norm_block(dummy_input)
print(f"Post-Norm Çıktı şekli: {post_norm_output.shape}")

# Pre-Norm Bloğu
pre_norm_block = TransformerEncoderBlockPreNorm(d_model, num_heads, d_ff)
pre_norm_output = pre_norm_block(dummy_input)
print(f"Pre-Norm Çıktı şekli: {pre_norm_output.shape}")

(Kod örneği bölümünün sonu)
```

<a name="8-sonuc"></a>
## 8. Sonuç
Transformer modellerinde Sonişlemeli Normalizasyon ve Önişlemeli Normalizasyon mimarisi arasındaki seçim, eğitim dinamikleri ve ölçeklenebilirlik için derin etkileri olan önemli bir tasarım kararıdır. Post-Norm orijinal tasarım olmasına rağmen, özellikle daha derin modellerde eğitim kararsızlığına ve gradyan sorunlarına yatkınlığı, Pre-Norm'un yaygın olarak benimsenmesine yol açmıştır.

Pre-Norm mimarisi, her alt katmandan ve kalan bağlantıdan *önce* Katman Normalizasyonu uygulayarak, ağ boyunca aktivasyonları ve gradyanları etkin bir şekilde stabilize eder. Bu özellik, güncel son teknoloji ürünleri tanımlayan devasa, derin üretken yapay zeka modellerini eğitmek için çok önemlidir. Modern araştırmalar ve pratik uygulamalar, sağlamlığı nedeniyle neredeyse evrensel olarak Pre-Norm'u tercih etmektedir, bu da araştırmacıların model boyutu ve karmaşıklığı sınırlarını daha fazla güvenle zorlamasına olanak tanır. Bu ayrımı anlamak sadece akademik bir alıştırma değil, çağdaş Transformer tabanlı üretken modellerle çalışan veya bunları oluşturan herkes için pratik bir zorunluluktur.