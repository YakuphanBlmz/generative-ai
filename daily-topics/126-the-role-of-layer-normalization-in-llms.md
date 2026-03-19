# The Role of Layer Normalization in LLMs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenge of Training Deep Networks and Why Layer Normalization Emerged](#2-the-challenge-of-training-deep-networks-and-why-layer-normalization-emerged)
- [3. How Layer Normalization Works](#3-how-layer-normalization-works)
- [4. Layer Normalization in Large Language Models (LLMs)](#4-layer-normalization-in-large-language-models-llms)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction
The advent of **Large Language Models (LLMs)** has revolutionized the field of natural language processing, enabling unprecedented capabilities in understanding, generating, and translating human language. Architectures such as the **Transformer** have been pivotal in this revolution, primarily due to their ability to process long-range dependencies efficiently. A critical, yet often understated, component enabling the stable and effective training of these immensely deep neural networks is **Layer Normalization (LN)**. Introduced by Ba, Kiros, and Hinton in 2016, Layer Normalization addresses several challenges inherent in training deep models, particularly those with recurrent or sequential structures, which are ubiquitous in LLMs. This document delves into the fundamental principles of Layer Normalization, its operational mechanics, and its indispensable role in the stability, convergence, and performance of modern LLMs.

### 2. The Challenge of Training Deep Networks and Why Layer Normalization Emerged
Deep neural networks, by their very nature, are susceptible to issues that can severely impede training. Two prominent problems are **vanishing gradients** and **exploding gradients**, where the gradients propagated back through many layers become either infinitesimally small or excessively large, preventing effective weight updates. Another critical challenge is **internal covariate shift**, a phenomenon where the distribution of activations in a network changes during training due to the continuous adjustment of parameters in preceding layers. This shift forces subsequent layers to constantly adapt to new input distributions, slowing down the training process and requiring smaller learning rates.

**Batch Normalization (BN)**, introduced earlier, offered a robust solution to internal covariate shift by normalizing activations across the batch dimension. However, BN presents limitations when applied to recurrent neural networks (RNNs) and Transformers, which often deal with variable-length sequences and require normalization independent of batch size:
*   **Variable Batch Statistics:** In sequence models, batch statistics can be unreliable or undefined for sequences that are shorter than the batch size or when processing single samples during inference.
*   **Small Batch Sizes:** BN's effectiveness diminishes significantly with small batch sizes, leading to noisy estimates of mean and variance, which can destabilize training.
*   **Inference Challenges:** During inference, BN typically uses moving averages of batch statistics computed during training, which might not generalize well to unseen data distributions or vary significantly from sample to sample.

Layer Normalization emerged as an elegant alternative, specifically designed to address these shortcomings by normalizing inputs across the **feature dimension** rather than the batch dimension. This makes LN's operation entirely independent of the batch size and the specific inputs in a given batch, making it highly suitable for sequence models and large-scale distributed training scenarios characteristic of LLMs.

### 3. How Layer Normalization Works
Unlike Batch Normalization, which computes normalization statistics (mean and variance) across the batch and spatial dimensions for each feature, Layer Normalization computes these statistics independently for each training example within each layer. For a given input feature vector $x \in \mathbb{R}^D$ (where $D$ is the dimensionality of the hidden state or features in a layer), Layer Normalization calculates the mean $\mu$ and variance $\sigma^2$ across all $D$ features for that single sample.

The operational steps are as follows:
1.  **Calculate Mean:** For each sample $x_i$ in a batch, and for each layer, the mean of the activations is computed across all features $k=1 \dots D$:
    $\mu_i = \frac{1}{D} \sum_{k=1}^{D} x_{i,k}$
2.  **Calculate Variance:** Similarly, the variance is computed across all features for that same sample:
    $\sigma_i^2 = \frac{1}{D} \sum_{k=1}^{D} (x_{i,k} - \mu_i)^2$
3.  **Normalize:** Each feature $x_{i,k}$ is then normalized using these sample-specific statistics:
    $\hat{x}_{i,k} = \frac{x_{i,k} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$
    where $\epsilon$ is a small constant added for numerical stability, preventing division by zero.
4.  **Scale and Shift:** Finally, the normalized activations are scaled by a learnable gain parameter $\gamma$ and shifted by a learnable bias parameter $\beta$:
    $y_{i,k} = \gamma \hat{x}_{i,k} + \beta$
    These parameters, $\gamma$ and $\beta$, are unique to each layer and allow the network to retain representational power, effectively undoing the normalization if it proves detrimental to the learning process. By learning these affine transformation parameters, the network can adaptively scale and shift the normalized outputs, ensuring that the optimal activation range is maintained.

This approach ensures that the normalization process is consistent regardless of the batch size, making it particularly effective for sequences where the input length can vary and for architectures like Transformers where each position in a sequence can be thought of as an independent "feature" within the context of a layer's processing.

### 4. Layer Normalization in Large Language Models (LLMs)
The Transformer architecture, which underpins most modern LLMs (e.g., BERT, GPT-series, T5), extensively utilizes Layer Normalization. Within a Transformer block, Layer Normalization is typically applied at two primary locations:
1.  **Before the Multi-Head Attention mechanism:** Normalizing the input to the attention layer.
2.  **Before the Feed-Forward Network (FFN):** Normalizing the input to the FFN, after the attention output has been added.

The exact placement of LN within the Transformer block can vary, leading to different architectural variants:
*   **Post-Normalization (Post-LN):** The original Transformer architecture applied LN *after* the residual connection and the subsequent layer (attention or FFN). While simpler to implement, post-LN models can be harder to train due to larger variance in gradients, especially in very deep networks.
*   **Pre-Normalization (Pre-LN):** More recent and successful Transformer variants, such as those used in GPT-2/3 and T5, often apply LN *before* the self-attention and FFN sub-layers, with the residual connection bypassing the LN. This "pre-LN" setup typically leads to more stable training, faster convergence, and allows for the training of much deeper models without vanishing gradients, as the gradients can flow more directly through the residual connections.

The benefits of Layer Normalization in LLMs are manifold:
*   **Stabilized Training:** By standardizing the input distributions to each sub-layer, LN significantly reduces internal covariate shift, leading to more stable gradients and allowing for higher learning rates. This stability is crucial for training models with billions of parameters.
*   **Faster Convergence:** Stable gradients and reduced shift enable the optimization process to converge more rapidly to a good solution.
*   **Robustness to Batch Size:** As LN operates independently on each sample, it is immune to the issues associated with small or variable batch sizes, which is a common scenario in LLM training and inference.
*   **Improved Generalization:** By promoting more stable gradient flow and preventing saturation of activation functions, LN can contribute to better generalization capabilities of the model.

While Layer Normalization is dominant, advanced variants such as **RMSNorm** (Root Mean Square Normalization) have also been explored. RMSNorm simplifies LN by omitting the subtraction of the mean, normalizing only by the root mean square, which can offer computational efficiency and comparable performance in certain contexts, particularly in very deep models where the mean is often close to zero.

In summary, Layer Normalization is not merely an auxiliary component but a foundational element that underpins the successful training and performance of modern LLMs. Its ability to stabilize gradients and allow for efficient training of extremely deep networks has been critical in pushing the boundaries of what is achievable in natural language understanding and generation.

### 5. Code Example
This example demonstrates a simplified Layer Normalization operation on a 2D tensor representing a single token's hidden state, typical for a hidden layer in an LLM.

```python
import torch

def layer_norm_manual(x, gamma, beta, epsilon=1e-5):
    """
    Manually implements Layer Normalization for a 2D tensor (single example, multiple features).
    Args:
        x (torch.Tensor): Input tensor of shape (sequence_length, hidden_size).
                          For simplicity, we consider a single token's hidden state (1, hidden_size).
        gamma (torch.Tensor): Learnable scaling parameter.
        beta (torch.Tensor): Learnable shifting parameter.
        epsilon (float): Small constant for numerical stability.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    # Calculate mean and variance across the 'hidden_size' dimension (last dimension)
    # This corresponds to normalizing features for a single example (token)
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased=False for population variance

    # Normalize
    x_normalized = (x - mean) / torch.sqrt(variance + epsilon)

    # Scale and shift
    output = gamma * x_normalized + beta
    return output

# Example usage:
# Assume a hidden state for a single token in an LLM layer
hidden_size = 768
# Create a dummy tensor representing the hidden state of one token
# Shape (1, hidden_size) for single token, or (seq_len, hidden_size) for multiple tokens
single_token_hidden_state = torch.randn(1, hidden_size)

# Learnable parameters (gamma and beta)
# These would typically be initialized to ones for gamma and zeros for beta
# and updated during training.
gamma_param = torch.ones(1, hidden_size)
beta_param = torch.zeros(1, hidden_size)

# Apply manual Layer Normalization
normalized_state_manual = layer_norm_manual(single_token_hidden_state, gamma_param, beta_param)

print("Original Hidden State (first 5 features):\n", single_token_hidden_state[:, :5])
print("Manually Normalized State (first 5 features):\n", normalized_state_manual[:, :5])
print("Mean of Manual Normalized State (should be close to 0):\n", normalized_state_manual.mean(dim=-1))
print("Variance of Manual Normalized State (should be close to 1):\n", normalized_state_manual.var(dim=-1, unbiased=False))

# Verify with PyTorch's built-in LayerNorm (if using a batch dimension, LN applies per-sample)
# For PyTorch's LayerNorm, the 'normalized_shape' specifies the last D dimensions to normalize.
# Here, it's the 'hidden_size'.
layer_norm_pytorch = torch.nn.LayerNorm(hidden_size, eps=1e-5)
# Ensure gamma and beta are set to our manual parameters for comparison
with torch.no_grad():
    layer_norm_pytorch.weight.copy_(gamma_param[0]) # weight corresponds to gamma
    layer_norm_pytorch.bias.copy_(beta_param[0])    # bias corresponds to beta

normalized_state_pytorch = layer_norm_pytorch(single_token_hidden_state)
print("\nPyTorch LayerNorm State (first 5 features):\n", normalized_state_pytorch[:, :5])

(End of code example section)
```

### 6. Conclusion
Layer Normalization stands as an indispensable innovation in the development and proliferation of Large Language Models. By providing a mechanism to stabilize activations and gradients within deep networks, independent of batch size, it directly addresses critical challenges that plagued earlier normalization techniques in the context of sequence processing. Its strategic placement within the Transformer architecture, particularly in pre-normalization configurations, has been key to enabling the training of models with billions of parameters, facilitating faster convergence and enhanced overall performance. The continuous evolution of normalization techniques, building upon the principles of Layer Normalization, underscores its foundational significance in the ongoing progress of artificial intelligence, particularly in the domain of deep learning for natural language processing. As LLMs continue to grow in complexity and scale, the principles exemplified by Layer Normalization will remain paramount for their successful development and deployment.

---
<br>

<a name="türkçe-içerik"></a>
## LLM'lerde Katman Normalizasyonunun Rolü

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Derin Ağları Eğitme Zorluğu ve Katman Normalizasyonunun Ortaya Çıkışı](#2-derin-ağları-eğitme-zorluğu-ve-katman-normalizasyonunun-ortaya-çıkışı)
- [3. Katman Normalizasyonu Nasıl Çalışır?](#3-katman-normalizasyonu-nasıl-çalışır)
- [4. Büyük Dil Modellerinde (LLM'ler) Katman Normalizasyonu](#4-büyük-dil-modellerinde-llmler-katman-normalizasyonu)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

### 1. Giriş
**Büyük Dil Modellerinin (LLM'ler)** yükselişi, doğal dil işleme alanında devrim yaratarak insan dilini anlama, üretme ve çevirme konusunda eşi benzeri görülmemiş yetenekler sağlamıştır. Özellikle **Transformer** gibi mimariler, uzun menzilli bağımlılıkları etkili bir şekilde işleme yetenekleri sayesinde bu devrimde kilit rol oynamıştır. Bu son derece derin sinir ağlarının istikrarlı ve etkili bir şekilde eğitilmesini sağlayan kritik ancak genellikle göz ardı edilen bir bileşen ise **Katman Normalizasyonu (LN)**'dur. Ba, Kiros ve Hinton tarafından 2016'da tanıtılan Katman Normalizasyonu, derin modellerin, özellikle de LLM'lerde yaygın olan tekrarlayan veya sıralı yapılara sahip olanların eğitiminde karşılaşılan birçok zorluğu ele almaktadır. Bu belge, Katman Normalizasyonunun temel prensiplerini, operasyonel mekaniklerini ve modern LLM'lerin istikrarı, yakınsaması ve performansı üzerindeki vazgeçilmez rolünü ayrıntılı olarak incelemektedir.

### 2. Derin Ağları Eğitme Zorluğu ve Katman Normalizasyonunun Ortaya Çıkışı
Derin sinir ağları, doğaları gereği eğitimi ciddi şekilde engelleyebilecek sorunlara karşı hassastır. Öne çıkan iki sorun, çok sayıda katmandan geriye doğru yayılan gradyanların ya sonsuz derecede küçülmesine ya da aşırı derecede büyümesine neden olan **kaybolan gradyanlar** ve **patlayan gradyanlardır**, bu da etkili ağırlık güncellemelerini engeller. Diğer bir kritik zorluk ise **iç kovaryat kayması (internal covariate shift)**'dir; bu olgu, bir ağdaki aktivasyonların dağılımının, önceki katmanlardaki parametrelerin sürekli ayarlanması nedeniyle eğitim sırasında değişmesidir. Bu kayma, sonraki katmanları sürekli olarak yeni girdi dağılımlarına uyum sağlamaya zorlayarak eğitim sürecini yavaşlatır ve daha küçük öğrenme oranları gerektirir.

Daha önce tanıtılan **Topluluk Normalizasyonu (Batch Normalization - BN)**, aktivasyonları topluluk (batch) boyutunda normalleştirerek iç kovaryat kaymasına sağlam bir çözüm sunmuştur. Ancak BN, genellikle değişken uzunlukta dizilerle uğraşan ve topluluk boyutundan bağımsız normalizasyon gerektiren yinelemeli sinir ağlarına (RNN'ler) ve Transformer'lara uygulandığında sınırlamalar ortaya koymaktadır:
*   **Değişken Topluluk İstatistikleri:** Dizi modellerinde, topluluk istatistikleri, dizi uzunlukları topluluk boyutundan kısa olduğunda veya çıkarım sırasında tek örnekler işlenirken güvenilmez veya tanımsız olabilir.
*   **Küçük Topluluk Boyutları:** BN'nin etkinliği, küçük topluluk boyutlarıyla önemli ölçüde azalır, bu da ortalama ve varyansın gürültülü tahminlerine yol açar ve eğitimi istikrarsızlaştırabilir.
*   **Çıkarım Zorlukları:** Çıkarım sırasında, BN tipik olarak eğitim sırasında hesaplanan topluluk istatistiklerinin hareketli ortalamalarını kullanır, bu da yeni, daha önce görülmemiş veri dağılımlarına iyi genellenmeyebilir veya örnekten örneğe önemli ölçüde değişebilir.

Katman Normalizasyonu, bu eksiklikleri gidermek için özel olarak tasarlanmış zarif bir alternatif olarak ortaya çıkmıştır; girdileri topluluk boyutu yerine **özellik boyutu** boyunca normalleştirir. Bu, LN'nin çalışmasını topluluk boyutundan ve belirli bir topluluktaki belirli girdilerden tamamen bağımsız hale getirir, bu da onu LLM'lerin karakteristiği olan dizi modelleri ve büyük ölçekli dağıtık eğitim senaryoları için son derece uygun kılar.

### 3. Katman Normalizasyonu Nasıl Çalışır?
Topluluk Normalizasyonunun aksine, Katman Normalizasyonu, her özellik için topluluk ve uzamsal boyutlarda normalizasyon istatistiklerini (ortalama ve varyans) hesaplarken, Katman Normalizasyonu bu istatistikleri her katmanda her eğitim örneği için bağımsız olarak hesaplar. Belirli bir girdi özellik vektörü $x \in \mathbb{R}^D$ (burada $D$, bir katmandaki gizli durumun veya özelliklerin boyutluluğudur) için Katman Normalizasyonu, bu tek örnek için tüm $D$ özellik boyunca ortalama $\mu$ ve varyans $\sigma^2$ değerlerini hesaplar.

Operasyonel adımlar aşağıdaki gibidir:
1.  **Ortalama Hesaplama:** Bir topluluktaki her $x_i$ örneği için ve her katman için, aktivasyonların ortalaması tüm özellikler $k=1 \dots D$ boyunca hesaplanır:
    $\mu_i = \frac{1}{D} \sum_{k=1}^{D} x_{i,k}$
2.  **Varyans Hesaplama:** Benzer şekilde, varyans aynı örnek için tüm özellikler boyunca hesaplanır:
    $\sigma_i^2 = \frac{1}{D} \sum_{k=1}^{D} (x_{i,k} - \mu_i)^2$
3.  **Normalleştirme:** Her özellik $x_{i,k}$ daha sonra bu örneğe özgü istatistikler kullanılarak normalleştirilir:
    $\hat{x}_{i,k} = \frac{x_{i,k} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$
    burada $\epsilon$, sayısal kararlılık için eklenen küçük bir sabittir ve sıfıra bölmeyi önler.
4.  **Ölçekleme ve Kaydırma:** Son olarak, normalleştirilmiş aktivasyonlar, öğrenilebilir bir kazanç parametresi $\gamma$ ile ölçeklenir ve öğrenilebilir bir sapma parametresi $\beta$ ile kaydırılır:
    $y_{i,k} = \gamma \hat{x}_{i,k} + \beta$
    Bu parametreler, $\gamma$ ve $\beta$, her katmana özgüdür ve ağın temsil gücünü korumasını sağlar, böylece normalizasyonun öğrenme süreci için zararlı olduğu ortaya çıkarsa normalleştirmeyi etkili bir şekilde geri alabilir. Bu afin dönüşüm parametrelerini öğrenerek, ağ, normalleştirilmiş çıktıları uyarlanabilir bir şekilde ölçekleyebilir ve kaydırabilir, böylece optimal aktivasyon aralığının korunmasını sağlar.

Bu yaklaşım, normalizasyon sürecinin topluluk boyutundan bağımsız olarak tutarlı olmasını sağlar, bu da girdi uzunluğunun değişebileceği diziler ve bir dizideki her pozisyonun bir katmanın işlenmesi bağlamında bağımsız bir "özellik" olarak düşünülebileceği Transformer gibi mimariler için özellikle etkilidir.

### 4. Büyük Dil Modellerinde (LLM'ler) Katman Normalizasyonu
Çoğu modern LLM'nin (örneğin BERT, GPT serisi, T5) temelini oluşturan Transformer mimarisi, Katman Normalizasyonunu yoğun bir şekilde kullanır. Bir Transformer bloğu içinde, Katman Normalizasyonu tipik olarak iki ana yerde uygulanır:
1.  **Çok Başlı Dikkat (Multi-Head Attention) mekanizmasından önce:** Dikkat katmanına girdinin normalleştirilmesi.
2.  **İleri Beslemeli Ağı (Feed-Forward Network - FFN) öncesi:** Dikkat çıktısı eklendikten sonra, FFN'ye girdinin normalleştirilmesi.

LN'nin Transformer bloğu içindeki kesin yerleşimi farklı mimari varyantlara yol açabilir:
*   **Sonra Normalizasyon (Post-Normalization - Post-LN):** Orijinal Transformer mimarisi, LN'yi artık bağlantısından ve sonraki katmandan (dikkat veya FFN) *sonra* uyguladı. Uygulaması daha basit olsa da, özellikle çok derin ağlarda daha büyük gradyan varyansı nedeniyle Post-LN modellerini eğitmek daha zor olabilir.
*   **Ön Normalizasyon (Pre-Normalization - Pre-LN):** GPT-2/3 ve T5'te kullanılanlar gibi daha yeni ve başarılı Transformer varyantları, genellikle LN'yi kendini dikkat (self-attention) ve FFN alt katmanlarından *önce* uygular, artık bağlantı LN'yi atlar. Bu "Pre-LN" kurulumu genellikle daha istikrarlı eğitime, daha hızlı yakınsamaya yol açar ve gradyanların artık bağlantılar aracılığıyla daha doğrudan akabilmesi nedeniyle gradyanların kaybolması olmadan çok daha derin modellerin eğitilmesine olanak tanır.

LLM'lerde Katman Normalizasyonunun faydaları çok çeşitlidir:
*   **İstikrarlı Eğitim:** Her alt katmanın girdi dağılımlarını standartlaştırarak, LN iç kovaryat kaymasını önemli ölçüde azaltır, bu da daha istikrarlı gradyanlara ve daha yüksek öğrenme oranlarına izin verir. Bu istikrar, milyarlarca parametreye sahip modelleri eğitmek için çok önemlidir.
*   **Daha Hızlı Yakınsama:** İstikrarlı gradyanlar ve azaltılmış kayma, optimizasyon sürecinin iyi bir çözüme daha hızlı yakınsamasını sağlar.
*   **Topluluk Boyutuna Karşı Sağlamlık:** LN her örnek üzerinde bağımsız olarak çalıştığı için, LLM eğitimi ve çıkarımında yaygın bir senaryo olan küçük veya değişken topluluk boyutlarıyla ilişkili sorunlara karşı bağışıktır.
*   **Gelişmiş Genelleme:** Daha istikrarlı gradyan akışını teşvik ederek ve aktivasyon fonksiyonlarının doygunluğunu önleyerek, LN modelin daha iyi genelleme yeteneklerine katkıda bulunabilir.

Katman Normalizasyonu baskın olsa da, **RMSNorm** (Karekök Ortalama Normalizasyonu) gibi gelişmiş varyantlar da araştırılmıştır. RMSNorm, ortalamanın çıkarılmasını atlayarak, sadece karekök ortalama ile normalleştirerek LN'yi basitleştirir, bu da belirli bağlamlarda, özellikle ortalamanın genellikle sıfıra yakın olduğu çok derin modellerde hesaplama verimliliği ve karşılaştırılabilir performans sunabilir.

Özetle, Katman Normalizasyonu sadece yardımcı bir bileşen değil, modern LLM'lerin başarılı eğitimini ve performansını destekleyen temel bir unsurdur. Gradyanları stabilize etme ve topluluk boyutundan bağımsız olarak son derece derin ağların verimli bir şekilde eğitilmesine olanak tanıma yeteneği, doğal dil anlama ve üretmede ulaşılabilir olanın sınırlarını zorlamada kritik olmuştur.

### 5. Kod Örneği
Bu örnek, bir LLM'deki gizli bir katman için tipik olan, tek bir jetonun gizli durumunu temsil eden 2 boyutlu bir tensör üzerinde basitleştirilmiş bir Katman Normalizasyon işlemini göstermektedir.

```python
import torch

def layer_norm_manual(x, gamma, beta, epsilon=1e-5):
    """
    Tek bir örnek, birden çok özellik için 2D tensör üzerinde Katman Normalizasyonunu manuel olarak uygular.
    Argümanlar:
        x (torch.Tensor): (sequence_length, hidden_size) şeklinde girdi tensörü.
                          Basitlik için, tek bir jetonun gizli durumunu (1, hidden_size) ele alıyoruz.
        gamma (torch.Tensor): Öğrenilebilir ölçekleme parametresi.
        beta (torch.Tensor): Öğrenilebilir kaydırma parametresi.
        epsilon (float): Sayısal kararlılık için küçük sabit.
    Dönüş:
        torch.Tensor: Normalleştirilmiş tensör.
    """
    # 'hidden_size' boyutu (son boyut) boyunca ortalama ve varyansı hesapla
    # Bu, tek bir örnek (jeton) için özellikleri normalleştirmeye karşılık gelir
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False) # populasyon varyansı için unbiased=False

    # Normalleştir
    x_normalized = (x - mean) / torch.sqrt(variance + epsilon)

    # Ölçekle ve kaydır
    output = gamma * x_normalized + beta
    return output

# Örnek kullanım:
# Bir LLM katmanında tek bir jetonun gizli durumunu varsayalım
hidden_size = 768
# Tek bir jetonun gizli durumunu temsil eden sahte bir tensör oluştur
# Tek jeton için şekil (1, hidden_size) veya birden çok jeton için (seq_len, hidden_size)
single_token_hidden_state = torch.randn(1, hidden_size)

# Öğrenilebilir parametreler (gamma ve beta)
# Bunlar tipik olarak gamma için bire, beta için sıfıra başlatılır
# ve eğitim sırasında güncellenir.
gamma_param = torch.ones(1, hidden_size)
beta_param = torch.zeros(1, hidden_size)

# Manuel Katman Normalizasyonunu uygula
normalized_state_manual = layer_norm_manual(single_token_hidden_state, gamma_param, beta_param)

print("Orijinal Gizli Durum (ilk 5 özellik):\n", single_token_hidden_state[:, :5])
print("Manuel Normalleştirilmiş Durum (ilk 5 özellik):\n", normalized_state_manual[:, :5])
print("Manuel Normalleştirilmiş Durumun Ortalaması (0'a yakın olmalı):\n", normalized_state_manual.mean(dim=-1))
print("Manuel Normalleştirilmiş Durumun Varyansı (1'e yakın olmalı):\n", normalized_state_manual.var(dim=-1, unbiased=False))

# PyTorch'un yerleşik LayerNorm'u ile doğrula (topluluk boyutu kullanılıyorsa, LN örnek başına uygulanır)
# PyTorch'un LayerNorm'u için, 'normalized_shape' normalleştirilecek son D boyutunu belirtir.
# Burada 'hidden_size'dır.
layer_norm_pytorch = torch.nn.LayerNorm(hidden_size, eps=1e-5)
# Karşılaştırma için gamma ve beta'nın manuel parametrelerimize ayarlandığından emin ol
with torch.no_grad():
    layer_norm_pytorch.weight.copy_(gamma_param[0]) # weight gamma'ya karşılık gelir
    layer_norm_pytorch.bias.copy_(beta_param[0])    # bias beta'ya karşılık gelir

normalized_state_pytorch = layer_norm_pytorch(single_token_hidden_state)
print("\nPyTorch LayerNorm Durumu (ilk 5 özellik):\n", normalized_state_pytorch[:, :5])

(Kod örneği bölümünün sonu)
```

### 6. Sonuç
Katman Normalizasyonu, Büyük Dil Modellerinin geliştirilmesi ve yaygınlaşmasında vazgeçilmez bir yenilik olarak öne çıkmaktadır. Derin ağlar içinde aktivasyonları ve gradyanları topluluk boyutundan bağımsız olarak stabilize etme mekanizması sağlayarak, dizi işleme bağlamında önceki normalizasyon tekniklerini rahatsız eden kritik zorlukları doğrudan ele almaktadır. Transformer mimarisi içindeki stratejik yerleşimi, özellikle ön-normalizasyon konfigürasyonlarında, milyarlarca parametreye sahip modellerin eğitimini mümkün kılarak, daha hızlı yakınsama ve genel performansın artırılması için anahtar olmuştur. Katman Normalizasyonu prensipleri üzerine inşa edilen normalizasyon tekniklerinin sürekli evrimi, yapay zekanın, özellikle doğal dil işleme için derin öğrenme alanındaki devam eden ilerlemesinde temel önemini vurgulamaktadır. LLM'ler karmaşıklık ve ölçek olarak büyümeye devam ettikçe, Katman Normalizasyonunun örneklediği prensipler, başarılı geliştirme ve dağıtımları için çok önemli olmaya devam edecektir.