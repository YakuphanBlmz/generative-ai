# RMSNorm: Root Mean Square Layer Normalization

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Need for Normalization in Neural Networks](#2-the-need-for-normalization-in-neural-networks)
- [3. Understanding RMSNorm](#3-understanding-rmsnorm)
  - [3.1. Mathematical Formulation](#31-mathematical-formulation)
  - [3.2. Advantages of RMSNorm](#32-advantages-of-rmsnorm)
  - [3.3. Comparison with LayerNorm](#33-comparison-with-layernorm)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

---

### 1. Introduction
**RMSNorm**, or **Root Mean Square Layer Normalization**, represents a streamlined yet highly effective normalization technique specifically designed for deep neural networks. Introduced as an alternative to the widely adopted **Layer Normalization (LayerNorm)**, RMSNorm aims to enhance model training stability and improve performance, particularly within architectures that heavily rely on attention mechanisms, such as **Transformers**. Its primary distinction lies in its computational simplicity: it normalizes activations by scaling them based on their root mean square, eschewing the mean subtraction step characteristic of LayerNorm. This simplification translates into reduced computational overhead while often maintaining, or even surpassing, the performance benefits of its more complex counterparts in certain contexts. As deep learning models continue to grow in size and complexity, especially in the realm of Generative AI, efficient and stable training methods like RMSNorm become increasingly critical for practical deployment and research.

### 2. The Need for Normalization in Neural Networks
Deep neural networks, while powerful, are notoriously difficult to train, especially as their depth increases. Several challenges commonly arise during the training process:

*   **Internal Covariate Shift**: This phenomenon refers to the change in the distribution of network activations due to the constant updates of the parameters in preceding layers. As parameters change, the inputs to subsequent layers also change, forcing these layers to continuously adapt to new input distributions. This slows down training and requires lower learning rates.
*   **Vanishing and Exploding Gradients**: In very deep networks, gradients can become extremely small (**vanishing gradients**) or extremely large (**exploding gradients**) as they propagate backward through many layers. Vanishing gradients hinder effective learning in early layers, while exploding gradients lead to unstable updates and divergence.
*   **Sensitivity to Initialization**: Without proper normalization, neural networks can be highly sensitive to the initial values of their weights, often requiring careful tuning to avoid training instabilities.

**Normalization layers** address these issues by standardizing the inputs to activation functions, ensuring that the distributions of activations remain stable across training iterations. This standardization helps in:
*   Stabilizing gradient flow, mitigating vanishing and exploding gradients.
*   Allowing for higher learning rates, accelerating convergence.
*   Reducing the dependency on careful weight initialization.
*   Regularizing the model, potentially improving generalization.

Techniques like **Batch Normalization (BatchNorm)** and **Layer Normalization (LayerNorm)** have become foundational components in modern deep learning architectures. RMSNorm emerges as an evolution, seeking to provide similar benefits with greater efficiency.

### 3. Understanding RMSNorm
**RMSNorm** operates on the principle of re-scaling the input features based on their **Root Mean Square (RMS)**. Unlike other normalization techniques, it deliberately omits the mean-centering step, focusing solely on maintaining a consistent scale for the activations. This design choice is motivated by the observation that in many neural network architectures, especially **Transformers**, the absolute magnitude of activations often carries more relevant information than their exact mean.

#### 3.1. Mathematical Formulation
For an input tensor `x` (representing the activations of a layer), RMSNorm computes the normalized output `y` as follows:

$$y = \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}} \cdot g$$

Where:
*   `x`: The input tensor to be normalized. In the context of Layer Normalization, this would typically be an activation vector for a single data point across its features.
*   `N`: The number of elements (features) over which the normalization is performed.
*   $\sum_{i=1}^{N} x_i^2$: The sum of squares of all elements in `x`.
*   $\frac{1}{N} \sum_{i=1}^{N} x_i^2$: The **mean square** of the elements in `x`.
*   $\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}$: The **Root Mean Square (RMS)** of `x`, with a small constant $\epsilon$ (epsilon) added for numerical stability to prevent division by zero, particularly when the mean square is very close to zero.
*   `g`: A learnable **gain** or scaling parameter (often a vector, one per feature) that allows the network to restore the original scale or learn an optimal scale for each feature. This parameter is typically initialized to ones.

The core idea is to divide each element of the input `x` by its RMS, effectively normalizing its magnitude without shifting its mean. The learnable gain `g` then permits the model to adjust the scale of the normalized output, enhancing its representational capacity.

#### 3.2. Advantages of RMSNorm
RMSNorm offers several compelling advantages that make it an attractive option for various deep learning applications, particularly in the realm of large-scale models and Transformers:

*   **Computational Efficiency**: By eliminating the mean subtraction step, RMSNorm requires fewer floating-point operations (FLOPs) compared to LayerNorm. This computational simplicity translates into faster training and inference times, which is a significant benefit for resource-intensive models.
*   **Reduced Memory Footprint**: The simpler mathematical operations can also lead to a marginally reduced memory footprint, though this advantage is often context-dependent and hardware-specific.
*   **Training Stability**: Despite its simplicity, RMSNorm effectively contributes to stabilizing the training process by controlling the magnitude of activations. This helps in mitigating issues like vanishing or exploding gradients and allows for more aggressive learning rates.
*   **Performance in Transformers**: RMSNorm has demonstrated strong performance in **Transformer** architectures. In these models, where attention mechanisms process tokens based on their magnitudes, maintaining a consistent scale without necessarily centering the mean can be beneficial. Some research suggests that mean-centering might even remove useful signal in certain scenarios within Transformers.
*   **Simplicity and Interpretability**: The straightforward nature of RMSNorm makes it easier to understand and implement, contributing to faster experimentation cycles.

#### 3.3. Comparison with LayerNorm
While both RMSNorm and LayerNorm are designed to normalize activations within a layer independently for each sample, their underlying mechanisms and implications differ:

| Feature                   | Layer Normalization (LayerNorm)                                                                 | RMSNorm (Root Mean Square Layer Normalization)                                                                      |
| :------------------------ | :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Mathematical Operation** | Centers the activations by subtracting the mean, then scales by the standard deviation.           | Scales the activations by dividing them by their Root Mean Square (RMS). No mean subtraction.                       |
| **Formula**               | $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$                            | $y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \cdot g$                                                                    |
| **Learnable Parameters**  | Two parameters: $\gamma$ (gain/scale) and $\beta$ (bias/shift), both initialized to 1 and 0 resp. | Typically one parameter: $g$ (gain/scale), initialized to 1. (Some implementations might include a bias.)            |
| **Computational Cost**    | Higher, due to calculating both mean and standard deviation.                                    | Lower, as it only calculates the mean square (and then RMS), avoiding mean calculation.                              |
| **Impact on Activations** | Forces activations to have an approximate mean of 0 and standard deviation of 1.                  | Ensures activations have a consistent scale (RMS of 1, before scaling by `g`), but does not enforce a zero mean.     |
| **Suitability**           | Widely used across various neural networks; robust in many contexts.                             | Particularly effective in Transformer architectures and large language models where efficiency is paramount.          |
| **Signal Preservation**   | Might remove useful information encoded in the mean of activations in certain scenarios.         | Preserves the mean of activations, which can be beneficial if the mean carries important information.                 |

In essence, LayerNorm aims for a more rigid standardization to a canonical distribution, whereas RMSNorm prioritizes maintaining a consistent magnitude with greater computational efficiency. The choice between them often depends on the specific architecture, the characteristics of the data, and the computational budget. For highly efficient models like those often found in modern Generative AI, RMSNorm presents a compelling case due to its balance of performance and resource optimization.

### 4. Code Example

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm).
    Normalizes input by its root mean square, without mean centering.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter 'g' for scaling
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate RMS: sqrt(mean(x^2))
        # Keepdim ensures the dimension is retained for broadcasting
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Apply RMS normalization and then multiply by the learnable weight 'g'
        return self._norm(x) * self.weight

# Example usage:
if __name__ == '__main__':
    batch_size = 4
    sequence_length = 10
    embedding_dim = 768

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor mean (first element): {input_tensor[0,0,:].mean():.4f}")
    print(f"Input tensor RMS (first element): {torch.sqrt(input_tensor[0,0,:].pow(2).mean()):.4f}\n")

    # Initialize RMSNorm layer
    rms_norm_layer = RMSNorm(embedding_dim)

    # Apply RMSNorm
    output_tensor = rms_norm_layer(input_tensor)
    print(f"Output tensor shape: {output_tensor.shape}")
    print(f"Output tensor mean (first element): {output_tensor[0,0,:].mean():.4f}")
    print(f"Output tensor RMS (first element): {torch.sqrt(output_tensor[0,0,:].pow(2).mean()):.4f}")

    # Verify that RMS of normalized tensor is approximately 1 (before applying 'weight')
    normalized_x = input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + rms_norm_layer.eps)
    print(f"RMS of normalized_x (before weight, first element): {torch.sqrt(normalized_x[0,0,:].pow(2).mean()):.4f}")


(End of code example section)
```

### 5. Conclusion
**RMSNorm** stands out as an elegant and computationally efficient normalization strategy that has proven its worth in the rapidly evolving landscape of deep learning, particularly within large-scale **Transformer** architectures prevalent in **Generative AI**. By simplifying the normalization process to just scaling by the **Root Mean Square (RMS)** and omitting the mean-centering step, RMSNorm offers a powerful balance between training stability and computational cost. Its ability to accelerate convergence and reduce resource consumption makes it an increasingly popular choice for developing and deploying massive models where every computational saving counts. As the demand for larger and more capable generative models grows, techniques like RMSNorm, which provide robust performance with minimal overhead, will continue to play a pivotal role in pushing the boundaries of what AI can achieve. Its ongoing adoption underscores a broader trend in deep learning research: the pursuit of simpler, more efficient, and equally effective methods for training increasingly complex neural networks.

### 6. References
*   Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *Advances in Neural Information Processing Systems, 32*. [Paper Link (arXiv)](https://arxiv.org/abs/1907.09292)
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems, 30*. [Paper Link (arXiv)](https://arxiv.org/abs/1706.03762)

---
<br>

<a name="türkçe-içerik"></a>
## RMSNorm: Kök Ortalama Kare Katman Normalizasyonu

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Sinir Ağlarında Normalizasyon İhtiyacı](#2-sinir-ağlarında-normalizasyon-ihtiyacı)
- [3. RMSNorm'u Anlamak](#3-rmsnormu-anlamak)
  - [3.1. Matematiksel Formülasyon](#31-matematiksel-formülasyon)
  - [3.2. RMSNorm'un Avantajları](#32-rmsnormun-avantajları)
  - [3.3. LayerNorm ile Karşılaştırma](#33-layernorm-ile-karşılaştırma)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Kaynaklar](#6-kaynaklar)

---

### 1. Giriş
**RMSNorm** veya **Kök Ortalama Kare Katman Normalizasyonu**, derin sinir ağları için özel olarak tasarlanmış, basitleştirilmiş ancak son derece etkili bir normalizasyon tekniğidir. Yaygın olarak benimsenen **Katman Normalizasyonu (LayerNorm)**'na bir alternatif olarak sunulan RMSNorm, özellikle **Transformer** gibi dikkat mekanizmalarına yoğun bir şekilde dayanan mimarilerde model eğitiminin kararlılığını artırmayı ve performansı iyileştirmeyi amaçlamaktadır. Temel farklılığı, hesaplama basitliğidir: ortalama çıkarma adımını atlayarak, aktivasyonları kök ortalama karelerine göre ölçeklendirerek normalleştirir. Bu basitleştirme, hesaplama yükünün azalmasına yol açarken, belirli bağlamlarda daha karmaşık muadillerinin performans faydalarını sürdürmekte, hatta bazen aşmaktadır. Üretken Yapay Zeka (Generative AI) alanında derin öğrenme modellerinin boyut ve karmaşıklık açısından büyümesi devam ederken, RMSNorm gibi verimli ve kararlı eğitim yöntemleri, pratik uygulama ve araştırma için giderek daha kritik hale gelmektedir.

### 2. Sinir Ağlarında Normalizasyon İhtiyacı
Derin sinir ağları, güçlü olmalarına rağmen, özellikle derinlikleri arttıkça eğitilmesi zor yapılar olarak bilinir. Eğitim süreci boyunca yaygın olarak birkaç zorluk ortaya çıkar:

*   **İç Ortak Değişim Kayması (Internal Covariate Shift)**: Bu olgu, önceki katmanlardaki parametrelerin sürekli güncellenmesi nedeniyle ağ aktivasyonlarının dağılımındaki değişimi ifade eder. Parametreler değiştikçe, sonraki katmanlara gelen girdiler de değişir ve bu da bu katmanları sürekli olarak yeni girdi dağılımlarına uyum sağlamaya zorlar. Bu durum, eğitimi yavaşlatır ve daha düşük öğrenme oranları gerektirir.
*   **Gradyanların Yok Olması ve Patlaması (Vanishing and Exploding Gradients)**: Çok derin ağlarda, gradyanlar birçok katmandan geriye doğru yayılırken aşırı derecede küçük (**yok olan gradyanlar**) veya aşırı derecede büyük (**patlayan gradyanlar**) hale gelebilir. Yok olan gradyanlar, erken katmanlarda etkili öğrenmeyi engellerken, patlayan gradyanlar kararsız güncellemelere ve ayrışmaya yol açar.
*   **Başlatmaya Duyarlılık**: Uygun normalizasyon olmadan, sinir ağları ağırlıklarının başlangıç değerlerine karşı oldukça hassas olabilir ve eğitim kararsızlıklarını önlemek için genellikle dikkatli ayarlamalar gerektirir.

**Normalizasyon katmanları**, aktivasyon fonksiyonlarına gelen girdileri standartlaştırarak, aktivasyon dağılımlarının eğitim yinelemeleri boyunca kararlı kalmasını sağlayarak bu sorunları giderir. Bu standartlaştırma şunlara yardımcı olur:
*   Gradyan akışını stabilize etmek, yok olan ve patlayan gradyanları hafifletmek.
*   Daha yüksek öğrenme oranlarına izin vermek, yakınsamayı hızlandırmak.
*   Dikkatli ağırlık başlatmaya olan bağımlılığı azaltmak.
*   Modeli düzenlemek, potansiyel olarak genelleştirmeyi iyileştirmek.

**Toplu Normalizasyon (Batch Normalization)** ve **Katman Normalizasyonu (Layer Normalization)** gibi teknikler, modern derin öğrenme mimarilerinde temel bileşenler haline gelmiştir. RMSNorm, benzer faydaları daha fazla verimlilikle sağlamayı amaçlayan bir evrim olarak ortaya çıkmaktadır.

### 3. RMSNorm'u Anlamak
**RMSNorm**, girdi özelliklerini **Kök Ortalama Kare (Root Mean Square - RMS)** değerlerine göre yeniden ölçeklendirme prensibiyle çalışır. Diğer normalizasyon tekniklerinden farklı olarak, ortalama çıkarma adımını kasıtlı olarak atlar ve yalnızca aktivasyonlar için tutarlı bir ölçek korumaya odaklanır. Bu tasarım tercihi, birçok sinir ağı mimarisinde, özellikle **Transformer**'larda, aktivasyonların mutlak büyüklüğünün genellikle tam ortalamalarından daha fazla ilgili bilgi taşıdığı gözleminden kaynaklanmaktadır.

#### 3.1. Matematiksel Formülasyon
Bir girdi tensörü `x` (bir katmanın aktivasyonlarını temsil eden) için RMSNorm, normalize edilmiş çıktı `y`'yi aşağıdaki gibi hesaplar:

$$y = \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}} \cdot g$$

Burada:
*   `x`: Normalleştirilecek girdi tensörü. Katman Normalizasyonu bağlamında, bu genellikle tek bir veri noktası için özellikler boyunca bir aktivasyon vektörü olacaktır.
*   `N`: Normalizasyonun yapıldığı eleman (özellik) sayısı.
*   $\sum_{i=1}^{N} x_i^2$: `x` içindeki tüm elemanların karelerinin toplamı.
*   $\frac{1}{N} \sum_{i=1}^{N} x_i^2$: `x` içindeki elemanların **ortalama karesi**.
*   $\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}$: `x`'in **Kök Ortalama Karesi (RMS)**. Özellikle ortalama kare sıfıra çok yakın olduğunda, sıfıra bölmeyi önlemek için küçük bir $\epsilon$ (epsilon) sabiti eklenir.
*   `g`: Ağın orijinal ölçeği geri yüklemesine veya her özellik için optimal bir ölçek öğrenmesine izin veren, öğrenilebilir bir **kazanç** veya ölçekleme parametresi (genellikle özellik başına bir tane olmak üzere bir vektör). Bu parametre tipik olarak birler ile başlatılır.

Temel fikir, girdi `x`'in her elemanını kendi RMS'sine bölmek, böylece ortalamasını değiştirmeden büyüklüğünü etkin bir şekilde normalleştirmektir. Öğrenilebilir kazanç `g`, modelin normalize edilmiş çıktının ölçeğini ayarlamasına izin vererek temsil yeteneğini artırır.

#### 3.2. RMSNorm'un Avantajları
RMSNorm, özellikle büyük ölçekli modeller ve Transformer'lar alanında, çeşitli derin öğrenme uygulamaları için çekici bir seçenek haline getiren birkaç ikna edici avantaj sunar:

*   **Hesaplama Verimliliği**: Ortalama çıkarma adımını ortadan kaldırarak, RMSNorm, LayerNorm'a kıyasla daha az kayan nokta işlemine (FLOP) ihtiyaç duyar. Bu hesaplama basitliği, kaynak yoğun modeller için önemli bir fayda olan daha hızlı eğitim ve çıkarım sürelerine dönüşür.
*   **Azaltılmış Bellek Ayak İzi**: Daha basit matematiksel işlemler, marjinal olarak azaltılmış bir bellek ayak izine de yol açabilir, ancak bu avantaj genellikle bağlama ve donanıma özgüdür.
*   **Eğitim Kararlılığı**: Basitliğine rağmen, RMSNorm aktivasyonların büyüklüğünü kontrol ederek eğitim sürecini stabilize etmeye etkili bir şekilde katkıda bulunur. Bu, gradyanların yok olması veya patlaması gibi sorunları hafifletmeye yardımcı olur ve daha agresif öğrenme oranlarına izin verir.
*   **Transformer'larda Performans**: RMSNorm, **Transformer** mimarilerinde güçlü bir performans sergilemiştir. Dikkat mekanizmalarının token'ları büyüklüklerine göre işlediği bu modellerde, ortalamayı mutlaka merkezlemeye gerek kalmadan tutarlı bir ölçek sağlamak faydalı olabilir. Bazı araştırmalar, Transformer'lar içindeki belirli senaryolarda ortalama merkezlemenin faydalı sinyali bile kaldırabileceğini öne sürmektedir.
*   **Basitlik ve Yorumlanabilirlik**: RMSNorm'un basit yapısı, anlaşılmasını ve uygulanmasını kolaylaştırır, bu da daha hızlı deney döngülerine katkıda bulunur.

#### 3.3. LayerNorm ile Karşılaştırma
Hem RMSNorm hem de LayerNorm, bir katmandaki aktivasyonları her örnek için bağımsız olarak normalleştirmek üzere tasarlanmış olsa da, temel mekanizmaları ve etkileri farklılık gösterir:

| Özellik                   | Katman Normalizasyonu (LayerNorm)                                                                 | RMSNorm (Kök Ortalama Kare Katman Normalizasyonu)                                                                      |
| :------------------------ | :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Matematiksel İşlem**    | Ortalamayı çıkararak aktivasyonları ortalar, ardından standart sapmaya göre ölçeklendirir.         | Aktivasyonları Kök Ortalama Kare (RMS) değerlerine bölerek ölçeklendirir. Ortalama çıkarma yoktur.                       |
| **Formül**                | $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$                            | $y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \cdot g$                                                                    |
| **Öğrenilebilir Parametreler**  | İki parametre: $\gamma$ (kazanç/ölçek) ve $\beta$ (önyargı/kaydırma), sırasıyla 1 ve 0 ile başlatılır. | Genellikle tek parametre: $g$ (kazanç/ölçek), 1 ile başlatılır. (Bazı uygulamalar bir önyargı içerebilir.)            |
| **Hesaplama Maliyeti**    | Hem ortalama hem de standart sapma hesaplandığı için daha yüksektir.                           | Yalnızca ortalama kare (ve ardından RMS) hesaplandığı ve ortalama hesaplamasından kaçınıldığı için daha düşüktür.      |
| **Aktivasyonlar Üzerindeki Etki** | Aktivasyonları yaklaşık 0 ortalama ve 1 standart sapmaya sahip olmaya zorlar.                     | Aktivasyonların tutarlı bir ölçeğe (g ile ölçeklendirmeden önce RMS'si 1) sahip olmasını sağlar, ancak sıfır ortalama uygulamaz. |
| **Uygunluk**              | Çeşitli sinir ağlarında yaygın olarak kullanılır; birçok bağlamda sağlamdır.                     | Özellikle verimliliğin çok önemli olduğu Transformer mimarilerinde ve büyük dil modellerinde etkilidir.          |
| **Sinyal Koruması**       | Belirli senaryolarda aktivasyonların ortalamasında kodlanmış yararlı bilgileri kaldırabilir.         | Aktivasyonların ortalamasını korur, bu da ortalamanın önemli bilgi taşıması durumunda faydalı olabilir.                 |

Esasen, LayerNorm daha katı bir standartlaştırmayı kanonik bir dağılıma hedeflerken, RMSNorm daha fazla hesaplama verimliliği ile tutarlı bir büyüklüğü korumayı önceliklendirir. Aralarındaki seçim genellikle belirli mimariye, verilerin özelliklerine ve hesaplama bütçesine bağlıdır. Modern Üretken Yapay Zeka'da sıkça bulunan yüksek verimli modeller için, RMSNorm performans ve kaynak optimizasyonu dengesi nedeniyle ikna edici bir durum sunmaktadır.

### 4. Kod Örneği

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Kök Ortalama Kare Katman Normalizasyonunu (RMSNorm) uygular.
    Girdiyi, ortalama merkezleme yapmadan, kök ortalama karesine göre normalleştirir.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Ölçeklendirme için öğrenilebilir kazanç parametresi 'g'
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMS'yi hesapla: sqrt(ortalama(x^2))
        # Keepdim, yayınlama (broadcasting) için boyutun korunmasını sağlar
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # RMS normalizasyonunu uygula ve ardından öğrenilebilir ağırlık 'g' ile çarp
        return self._norm(x) * self.weight

# Örnek kullanım:
if __name__ == '__main__':
    batch_size = 4
    sequence_length = 10
    embedding_dim = 768

    # Sahte bir girdi tensörü oluştur
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)
    print(f"Girdi tensörü boyutu: {input_tensor.shape}")
    print(f"Girdi tensörü ortalaması (ilk eleman): {input_tensor[0,0,:].mean():.4f}")
    print(f"Girdi tensörü RMS'si (ilk eleman): {torch.sqrt(input_tensor[0,0,:].pow(2).mean()):.4f}\n")

    # RMSNorm katmanını başlat
    rms_norm_layer = RMSNorm(embedding_dim)

    # RMSNorm'u uygula
    output_tensor = rms_norm_layer(input_tensor)
    print(f"Çıktı tensörü boyutu: {output_tensor.shape}")
    print(f"Çıktı tensörü ortalaması (ilk eleman): {output_tensor[0,0,:].mean():.4f}")
    print(f"Çıktı tensörü RMS'si (ilk eleman): {torch.sqrt(output_tensor[0,0,:].pow(2).mean()):.4f}")

    # Normalize edilmiş tensörün RMS'sinin yaklaşık 1 olduğunu doğrula ('weight' uygulanmadan önce)
    normalized_x = input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + rms_norm_layer.eps)
    print(f"normalized_x'in RMS'si (ağırlık öncesi, ilk eleman): {torch.sqrt(normalized_x[0,0,:].pow(2).mean()):.4f}")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
**RMSNorm**, derin öğrenmenin hızla gelişen manzarasında, özellikle **Üretken Yapay Zeka**'da yaygın olan büyük ölçekli **Transformer** mimarilerinde değerini kanıtlamış zarif ve hesaplama açısından verimli bir normalizasyon stratejisi olarak öne çıkmaktadır. Normalizasyon sürecini sadece **Kök Ortalama Kare (RMS)** ile ölçeklendirmeye indirgeyerek ve ortalama merkezleme adımını atlayarak, RMSNorm, eğitim kararlılığı ve hesaplama maliyeti arasında güçlü bir denge sunar. Yakınsamayı hızlandırma ve kaynak tüketimini azaltma yeteneği, her hesaplama tasarrufunun önemli olduğu devasa modelleri geliştirmek ve dağıtmak için onu giderek daha popüler bir seçim haline getirmektedir. Daha büyük ve daha yetenekli üretken modeller için talep arttıkça, minimum yük ile sağlam performans sağlayan RMSNorm gibi teknikler, yapay zekanın neler başarabileceğinin sınırlarını zorlamada merkezi bir rol oynamaya devam edecektir. Sürekli benimsenmesi, derin öğrenme araştırmalarındaki daha geniş bir eğilimin altını çizmektedir: giderek daha karmaşık sinir ağlarını eğitmek için daha basit, daha verimli ve eşit derecede etkili yöntemler arayışı.

### 6. Kaynaklar
*   Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *Advances in Neural Information Processing Systems, 32*. [Makale Bağlantısı (arXiv)](https://arxiv.org/abs/1907.09292)
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems, 30*. [Makale Bağlantısı (arXiv)](https://arxiv.org/abs/1706.03762)







