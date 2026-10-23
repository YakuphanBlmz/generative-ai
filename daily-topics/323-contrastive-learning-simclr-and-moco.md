# Contrastive Learning: SimCLR and MoCo

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background on Self-Supervised Learning and Contrastive Learning Principles](#2-background)
- [3. SimCLR: A Simple Framework for Contrastive Learning](#3-simclr)
- [4. MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](#4-moco)
- [5. Comparative Analysis: SimCLR vs. MoCo](#5-comparative-analysis)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

The remarkable success of deep learning models, particularly in computer vision, has largely been attributed to the availability of vast amounts of labeled data. However, annotating large datasets is a costly and time-consuming endeavor. This challenge has propelled the research into **self-supervised learning (SSL)**, a paradigm where models learn meaningful representations from unlabeled data by solving automatically generated "pretext tasks." Among the various approaches in SSL, **contrastive learning** has emerged as a particularly powerful and effective method for learning high-quality visual representations.

Contrastive learning operates on the principle of bringing "positive pairs" (different views of the same data instance) closer together in an embedding space while pushing "negative pairs" (different data instances) farther apart. This document delves into two seminal works that have significantly advanced the field of contrastive learning: **SimCLR** (A Simple Framework for Contrastive Learning of Visual Representations) and **MoCo** (Momentum Contrast for Unsupervised Visual Representation Learning). We will explore their core mechanisms, architectural designs, and fundamental contributions to self-supervised representation learning.

<a name="2-background"></a>
## 2. Background on Self-Supervised Learning and Contrastive Learning Principles

### 2.1. Self-Supervised Learning (SSL)

**Self-supervised learning** aims to learn useful feature representations from unlabeled data. Instead of relying on human-annotated labels, SSL algorithms create their own labels or "pretext tasks" from the data itself. For instance, a model might be trained to predict missing patches of an image, colorize grayscale images, or predict the relative position of image patches. The underlying assumption is that by solving these pretext tasks, the model is forced to learn semantically rich and transferable features. These learned features can then be fine-tuned for downstream tasks with limited labeled data, often achieving performance comparable to, or even surpassing, models trained purely with supervision.

### 2.2. Core Principles of Contrastive Learning

**Contrastive learning** is a specific and highly effective form of SSL. Its central idea revolves around defining and optimizing a **contrastive loss function**. The goal of this loss is to ensure that:
1.  **Positive pairs**, which are different augmented views of the same original data instance, are embedded close to each other in a latent space.
2.  **Negative pairs**, which are distinct data instances (or their augmentations), are pushed away from each other in the latent space.

A typical contrastive learning framework involves:
*   An **encoder network** (e.g., a ResNet) that maps raw input data (e.g., an image) into a lower-dimensional latent representation.
*   A **projection head** (often a small MLP) applied after the encoder, which maps the encoded representations into a space where the contrastive loss is applied.
*   A set of **data augmentation** techniques to generate diverse views of the input data.
*   A **contrastive loss function**, most commonly the **InfoNCE (Noise-Contrastive Estimation) loss**, which quantifies the similarity between positive pairs relative to negative pairs.

The InfoNCE loss for a query representation $q$ and a positive key representation $k^+$ relative to a set of negative key representations $\{k_1^-, k_2^-, \dots, k_N^-\}$ is typically formulated as:

$$
L_{InfoNCE} = - \log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\exp(\text{sim}(q, k^+) / \tau) + \sum_{i=1}^N \exp(\text{sim}(q, k_i^-) / \tau)}
$$

where $\text{sim}(\cdot, \cdot)$ is a similarity function (e.g., cosine similarity) and $\tau$ is a **temperature parameter** that scales the logits, controlling the sharpness of the distribution. A good contrastive model learns an embedding space where the similarity between positive pairs is maximized, and the similarity between negative pairs is minimized.

<a name="3-simclr"></a>
## 3. SimCLR: A Simple Framework for Contrastive Learning

SimCLR, introduced by Chen et al. (2020), demonstrated that a simple framework could achieve state-of-the-art results in self-supervised learning without specialized architectures or memory banks. Its elegance lies in its simplicity and effectiveness.

### 3.1. SimCLR Architecture and Mechanism

The core components of SimCLR are:
1.  **Data Augmentation:** For each image $x$ in a batch, two different augmented views, $x_i$ and $x_j$, are generated by applying a sequence of random transformations (e.g., random cropping, color jittering, Gaussian blur, random grayscale). These two views form a **positive pair**.
2.  **Base Encoder:** A neural network $f(\cdot)$ (typically a ResNet) extracts representations $h_i = f(x_i)$ and $h_j = f(x_j)$ from the augmented views.
3.  **Projection Head:** A small MLP $g(\cdot)$ (e.g., a two-layer MLP with ReLU activation in the hidden layer) maps the representations to a lower-dimensional space where the contrastive loss is applied: $z_i = g(h_i)$ and $z_j = g(h_j)$. This projection head is crucial for improved performance, as it allows the encoder to learn richer representations without being constrained by the dimensionality of the contrastive loss space.
4.  **Contrastive Loss Function:** The **NT-Xent (normalized temperature-scaled cross-entropy) loss**, which is a variant of InfoNCE, is applied. For a given positive pair $(z_i, z_j)$, all other $2(N-1)$ augmented samples within the same batch (where $N$ is the batch size) are treated as **negative samples**.

The loss for a positive pair $(z_i, z_j)$ is:

$$
L_{i,j} = - \log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

The total loss for a batch is the sum of $L_{i,j}$ and $L_{j,i}$ over all positive pairs.

### 3.2. Key Findings and Contributions

SimCLR highlighted several critical factors for effective contrastive learning:
*   **Stronger Data Augmentation:** The combination of various transformations, especially random cropping and color distortions, was shown to be vital.
*   **Projection Head:** The non-linear projection head $g(\cdot)$ improved the quality of learned representations.
*   **Large Batch Sizes:** SimCLR relies heavily on large batch sizes (e.g., 4096 or 8192) to generate a sufficient number of negative samples within each batch. This is because all other samples in the batch are implicitly treated as negatives.
*   **Temperature Parameter:** The temperature $\tau$ in the InfoNCE loss was found to be crucial for learning effective representations.

SimCLR demonstrated that with careful design choices, high-quality self-supervised representations could be learned using a conceptually straightforward approach, albeit at the cost of significant computational resources due to its large batch size requirement.

<a name="4-moco"></a>
## 4. MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

While SimCLR showed the power of in-batch negatives, its reliance on very large batch sizes (which translates to immense GPU memory and computation) became a bottleneck. **MoCo**, proposed by He et al. (2020), addressed this limitation by introducing a **dynamic dictionary** with a **momentum-updated encoder** to provide a large and consistent set of negative samples.

### 4.1. MoCo Architecture and Mechanism

MoCo reimagines contrastive learning as a dictionary look-up task. The goal is to train an encoder that can perform dictionary look-ups effectively.
1.  **Two Encoders:** MoCo employs two encoders:
    *   **Query Encoder** ($f_q$): This is the network whose parameters $\theta_q$ are being trained via backpropagation. It encodes the query samples.
    *   **Key Encoder** ($f_k$): This network, with parameters $\theta_k$, encodes the key samples (which populate the dictionary). Crucially, $f_k$ is a momentum-updated version of $f_q$.
2.  **Momentum Update:** The parameters of the key encoder are updated as a moving average of the query encoder's parameters:
    $$
    \theta_k \leftarrow m \theta_k + (1-m) \theta_q
    $$
    where $m \in [0, 1)$ is the **momentum coefficient** (typically set high, e.g., 0.999). This slow update ensures that the key representations in the dictionary remain consistent over time, but still gradually evolve.
3.  **Memory Bank (Queue):** Instead of using only in-batch negatives, MoCo maintains a **queue** of encoded key representations from previous mini-batches. This queue acts as a dynamic dictionary, providing a large pool of negative samples. When a new mini-batch arrives, its encoded keys are added to the queue, and the oldest keys are removed. This mechanism allows for a dictionary size that is decoupled from the mini-batch size.
4.  **Contrastive Loss:** Similar to SimCLR, the InfoNCE loss is used. A query $q$ is contrasted with a positive key $k^+$ (from the same input, encoded by $f_k$) and a large set of negative keys $\{k_i^-\}$ sampled from the memory bank.

### 4.2. Key Findings and Contributions

MoCo's innovations led to several benefits:
*   **Decoupled Dictionary Size:** It allows for an arbitrarily large dictionary of negative samples, independent of the mini-batch size. This is a significant advantage over SimCLR, enabling the use of standard, smaller batch sizes (e.g., 256) while still having access to thousands of negatives.
*   **Consistent Negative Samples:** The momentum update mechanism for the key encoder ensures that the keys in the dictionary are relatively consistent, preventing them from changing too rapidly, which would make the contrastive learning task unstable.
*   **State-of-the-art Performance:** MoCo achieved strong performance on various benchmarks, demonstrating that large, diverse negative sample sets (achieved through the memory queue) are more critical than large batch sizes themselves.

MoCo provided an elegant solution to the large batch size problem, making contrastive learning more accessible and computationally efficient, thus accelerating its adoption in practical applications.

<a name="5-comparative-analysis"></a>
## 5. Comparative Analysis: SimCLR vs. MoCo

Both SimCLR and MoCo are foundational works in contrastive learning, yet they approach the problem of obtaining effective negative samples differently.

| Feature                 | SimCLR                                                              | MoCo                                                                       |
| :---------------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------- |
| **Negative Samples**    | All other samples in the current large mini-batch.                  | Dynamically maintained queue (memory bank) of representations from past mini-batches. |
| **Batch Size Reliance** | **High reliance** on very large batch sizes (e.g., 4096-8192) to generate sufficient negatives. | **Low reliance** on mini-batch size; effective with standard batch sizes (e.g., 256). |
| **Key Encoder Update**  | No separate key encoder; both views are processed by the same encoder and updated via backpropagation. | Separate key encoder ($f_k$) with parameters $\theta_k$ updated via **momentum moving average** of query encoder's parameters $\theta_q$. |
| **Encoder Architecture**| A single encoder $f$ and a projection head $g$.                    | A query encoder $f_q$, a key encoder $f_k$, and a projection head $g$.       |
| **Computational Cost**  | High GPU memory and computation due to large batch sizes.           | More computationally efficient due to smaller batch sizes and momentum update. |
| **Dictionary Properties**| Dictionary of negatives is formed on-the-fly and varies with each batch. | Dictionary of negatives is large, consistent, and slowly evolving.        |
| **Performance**         | Achieved SOTA at its time, but limited by hardware for scaling.     | Achieved SOTA, more scalable and robust to batch size.                     |

**Similarities:**
*   Both utilize **strong data augmentation** to create positive pairs.
*   Both employ an **InfoNCE-like contrastive loss** (NT-Xent for SimCLR).
*   Both use a **projection head** after the encoder to map representations to a latent space for loss computation.
*   Both aim to learn **high-quality visual representations** in a self-supervised manner.

In essence, SimCLR provided a baseline that highlighted the importance of data augmentation, projection heads, and a sufficient number of negatives. MoCo then improved upon this by introducing a more efficient mechanism for sourcing negative samples, making contrastive learning more practical and scalable. These works collectively paved the way for more sophisticated and powerful self-supervised learning techniques in computer vision and beyond.

<a name="6-code-example"></a>
## 6. Code Example

Below is a simplified Python code snippet demonstrating the core concept of a contrastive loss function (InfoNCE). This example focuses on the loss calculation, assuming `query`, `positive_key`, and `negative_keys` are already extracted feature vectors.

```python
import torch
import torch.nn.functional as F

def info_nce_loss(query_features, positive_key_features, negative_key_features, temperature=0.07):
    """
    Computes the InfoNCE loss for a single query.

    Args:
        query_features (torch.Tensor): Feature vector of the query (e.g., z_i). Shape: [feature_dim]
        positive_key_features (torch.Tensor): Feature vector of the positive key (e.g., z_j). Shape: [feature_dim]
        negative_key_features (torch.Tensor): Feature vectors of negative keys. Shape: [num_negatives, feature_dim]
        temperature (float): Scaling factor for logits.

    Returns:
        torch.Tensor: Scalar InfoNCE loss.
    """
    # Normalize features to unit length for cosine similarity
    query_features = F.normalize(query_features, dim=0)
    positive_key_features = F.normalize(positive_key_features, dim=0)
    negative_key_features = F.normalize(negative_key_features, dim=1)

    # Compute similarity between query and positive key
    # (dot product of normalized vectors is cosine similarity)
    positive_logits = torch.dot(query_features, positive_key_features) / temperature

    # Compute similarity between query and all negative keys
    negative_logits = torch.matmul(query_features, negative_key_features.T) / temperature

    # Concatenate positive and negative logits
    # The first element corresponds to the positive sample
    logits = torch.cat([positive_logits.unsqueeze(0), negative_logits], dim=0)

    # Create labels: 0 for the positive sample (at index 0)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    # Compute cross-entropy loss (which is equivalent to InfoNCE here)
    loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))

    return loss

# Example Usage:
# Assuming feature vectors are 128-dimensional
feature_dim = 128
num_negatives = 100

# Generate dummy features
query = torch.randn(feature_dim)
positive_key = torch.randn(feature_dim)
negative_keys = torch.randn(num_negatives, feature_dim)

# Calculate loss
loss_value = info_nce_loss(query, positive_key, negative_keys)
print(f"Calculated InfoNCE Loss: {loss_value.item():.4f}")

# SimCLR/MoCo would apply this in a batch-wise manner, often
# calculating all-to-all similarities within a batch or with a memory bank.

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion

Contrastive learning has revolutionized self-supervised learning, providing a powerful paradigm for extracting meaningful representations from unlabeled data. SimCLR demonstrated the efficacy of strong data augmentation, projection heads, and large batch sizes with simple architectures, pushing the boundaries of what was achievable in SSL. MoCo, in turn, addressed the practical limitations of large batch sizes by introducing a momentum-updated key encoder and a dynamic memory queue, making contrastive learning more scalable and efficient.

Together, SimCLR and MoCo represent pivotal advancements that have significantly contributed to the understanding and application of self-supervised learning. Their methodologies have not only achieved impressive performance on various vision tasks but have also inspired a new wave of research in representation learning, moving deep learning models closer to autonomous feature extraction without the immense burden of human annotation. The principles established by these works continue to form the bedrock of many advanced self-supervised techniques used today in diverse domains from computer vision to natural language processing.

---
<br>

<a name="türkçe-içerik"></a>
## Karşılaştırmalı Öğrenme: SimCLR ve MoCo

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Öz-Denetimli Öğrenme ve Karşılaştırmalı Öğrenme İlkeleri Üzerine Arka Plan](#2-arka-plan)
- [3. SimCLR: Karşılaştırmalı Öğrenme için Basit Bir Çerçeve](#3-simclr-tr)
- [4. MoCo: Denetimsiz Görsel Temsil Öğrenimi için Momentum Kontrastı](#4-moco-tr)
- [5. Karşılaştırmalı Analiz: SimCLR ve MoCo](#5-karşılaştırmalı-analiz)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Derin öğrenme modellerinin, özellikle bilgisayar görüşü alanındaki dikkat çekici başarısı, büyük ölçüde geniş etiketli veri kümelerinin mevcudiyetine bağlanmıştır. Ancak, büyük veri kümelerini etiketlemek maliyetli ve zaman alıcı bir iştir. Bu zorluk, modellerin etiketlenmemiş verilerden otomatik olarak oluşturulan "gerekçe görevleri" çözerek anlamlı temsiller öğrendiği bir paradigmayı olan **öz-denetimli öğrenme (SSL)** araştırmalarını hızlandırmıştır. SSL'deki çeşitli yaklaşımlar arasında, **karşılaştırmalı öğrenme**, yüksek kaliteli görsel temsiller öğrenmek için özellikle güçlü ve etkili bir yöntem olarak ortaya çıkmıştır.

Karşılaştırmalı öğrenme, "pozitif çiftleri" (aynı veri örneğinin farklı görünümleri) bir gömme uzayında birbirine yaklaştırırken, "negatif çiftleri" (farklı veri örnekleri) birbirinden uzaklaştırma prensibiyle çalışır. Bu belge, karşılaştırmalı öğrenme alanını önemli ölçüde ilerleten iki seminal çalışmayı ele almaktadır: **SimCLR** (Görsel Temsillerin Karşılaştırmalı Öğrenimi için Basit Bir Çerçeve) ve **MoCo** (Denetimsiz Görsel Temsil Öğrenimi için Momentum Kontrastı). Çekirdek mekanizmalarını, mimari tasarımlarını ve öz-denetimli temsil öğrenimine yaptıkları temel katkıları inceleyeceğiz.

<a name="2-arka-plan"></a>
## 2. Öz-Denetimli Öğrenme ve Karşılaştırmalı Öğrenme İlkeleri Üzerine Arka Plan

### 2.1. Öz-Denetimli Öğrenme (SSL)

**Öz-denetimli öğrenme**, etiketlenmemiş verilerden faydalı özellik temsilleri öğrenmeyi amaçlar. İnsan tarafından etiketlenmiş verilere güvenmek yerine, SSL algoritmaları verinin kendisinden kendi etiketlerini veya "gerekçe görevlerini" oluşturur. Örneğin, bir model bir görüntünün eksik yamalarını tahmin etmek, gri tonlamalı görüntüleri renklendirmek veya görüntü yamalarının göreceli konumunu tahmin etmek için eğitilebilir. Temel varsayım, bu gerekçe görevlerini çözerek modelin semantik olarak zengin ve aktarılabilir özellikler öğrenmeye zorlanmasıdır. Öğrenilen bu özellikler, sınırlı etiketli verilerle alt görevler için ince ayar yapılabilir ve genellikle yalnızca denetimle eğitilmiş modellere kıyasla benzer veya hatta daha iyi performans elde edebilir.

### 2.2. Karşılaştırmalı Öğrenmenin Temel İlkeleri

**Karşılaştırmalı öğrenme**, SSL'nin spesifik ve oldukça etkili bir biçimidir. Temel fikri, bir **karşılaştırmalı kayıp fonksiyonu** tanımlamak ve optimize etmek etrafında döner. Bu kaybın amacı şunları sağlamaktır:
1.  Aynı orijinal veri örneğinin farklı artırılmış görünümleri olan **pozitif çiftler**, bir gizli uzayda birbirine yakın gömülür.
2.  Farklı veri örnekleri (veya onların artırımları) olan **negatif çiftler**, gizli uzayda birbirinden uzaklaştırılır.

Tipik bir karşılaştırmalı öğrenme çerçevesi şunları içerir:
*   Ham giriş verilerini (örn. bir görüntü) daha düşük boyutlu bir gizli temsile eşleyen bir **kodlayıcı ağ** (örn. bir ResNet).
*   Kodlayıcıdan sonra uygulanan, kodlanmış temsilleri karşılaştırmalı kaybın uygulandığı bir uzaya eşleyen bir **projeksiyon başlığı** (genellikle küçük bir MLP).
*   Giriş verisinin çeşitli görünümlerini oluşturmak için bir dizi **veri artırma** tekniği.
*   En yaygın olarak **InfoNCE (Gürültü Kontrastlı Tahmin) kaybı** olan, pozitif çiftler arasındaki benzerliği negatif çiftlere göre ölçen bir **karşılaştırmalı kayıp fonksiyonu**.

Bir sorgu temsili $q$ ve pozitif bir anahtar temsili $k^+$ ile negatif anahtar temsilleri kümesi $\{k_1^-, k_2^-, \dots, k_N^-\}$ için InfoNCE kaybı tipik olarak şu şekilde formüle edilir:

$$
L_{InfoNCE} = - \log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\exp(\text{sim}(q, k^+) / \tau) + \sum_{i=1}^N \exp(\text{sim}(q, k_i^-) / \tau)}
$$

Burada $\text{sim}(\cdot, \cdot)$ bir benzerlik fonksiyonudur (örn. kosinüs benzerliği) ve $\tau$, dağılımın keskinliğini kontrol eden, logitleri ölçekleyen bir **sıcaklık parametresidir**. İyi bir karşılaştırmalı model, pozitif çiftler arasındaki benzerliğin maksimize edildiği ve negatif çiftler arasındaki benzerliğin minimize edildiği bir gömme uzayı öğrenir.

<a name="3-simclr-tr"></a>
## 3. SimCLR: Karşılaştırmalı Öğrenme için Basit Bir Çerçeve

Chen ve ark. (2020) tarafından tanıtılan SimCLR, basit bir çerçevenin özel mimariler veya bellek bankaları olmadan öz-denetimli öğrenmede en son teknoloji sonuçlar elde edebileceğini göstermiştir. Zarafeti sadeliğinde ve etkinliğinde yatar.

### 3.1. SimCLR Mimarisi ve Mekanizması

SimCLR'nin temel bileşenleri şunlardır:
1.  **Veri Artırma:** Bir partideki her görüntü $x$ için, bir dizi rastgele dönüşüm (örn. rastgele kırpma, renk titremesi, Gauss bulanıklığı, rastgele gri tonlama) uygulanarak iki farklı artırılmış görünüm, $x_i$ ve $x_j$ oluşturulur. Bu iki görünüm bir **pozitif çift** oluşturur.
2.  **Temel Kodlayıcı:** Bir sinir ağı $f(\cdot)$ (tipik olarak bir ResNet), artırılmış görünümlerden temsilleri $h_i = f(x_i)$ ve $h_j = f(x_j)$ çıkarır.
3.  **Projeksiyon Başlığı:** Küçük bir MLP $g(\cdot)$ (örn. gizli katmanda ReLU aktivasyonlu iki katmanlı bir MLP), temsilleri karşılaştırmalı kaybın uygulandığı daha düşük boyutlu bir uzaya eşler: $z_i = g(h_i)$ ve $z_j = g(h_j)$. Bu projeksiyon başlığı, kodlayıcının karşılaştırmalı kayıp uzayının boyutsallığı ile sınırlı kalmadan daha zengin temsiller öğrenmesini sağladığı için gelişmiş performans için çok önemlidir.
4.  **Karşılaştırmalı Kayıp Fonksiyonu:** InfoNCE'nin bir varyantı olan **NT-Xent (normalize edilmiş sıcaklık ölçekli çapraz entropi) kaybı** uygulanır. Belirli bir pozitif çift $(z_i, z_j)$ için, aynı partideki diğer $2(N-1)$ artırılmış örnek (burada $N$ parti boyutudur) **negatif örnekler** olarak kabul edilir.

Bir pozitif çift $(z_i, z_j)$ için kayıp:

$$
L_{i,j} = - \log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Bir parti için toplam kayıp, tüm pozitif çiftler üzerindeki $L_{i,j}$ ve $L_{j,i}$ toplamıdır.

### 3.2. Temel Bulgular ve Katkılar

SimCLR, etkili karşılaştırmalı öğrenme için birkaç kritik faktörü vurgulamıştır:
*   **Daha Güçlü Veri Artırma:** Çeşitli dönüşümlerin, özellikle rastgele kırpma ve renk bozulmalarının kombinasyonunun hayati olduğu gösterilmiştir.
*   **Projeksiyon Başlığı:** Doğrusal olmayan projeksiyon başlığı $g(\cdot)$, öğrenilen temsillerin kalitesini artırmıştır.
*   **Büyük Parti Boyutları:** SimCLR, her partide yeterli sayıda negatif örnek oluşturmak için büyük parti boyutlarına (örn. 4096 veya 8192) büyük ölçüde güvenmektedir. Bunun nedeni, partideki diğer tüm örneklerin örtük olarak negatif olarak kabul edilmesidir.
*   **Sıcaklık Parametresi:** InfoNCE kaybındaki sıcaklık $\tau$'nun, etkili temsiller öğrenmek için çok önemli olduğu bulunmuştur.

SimCLR, dikkatli tasarım seçimleriyle, kavramsal olarak basit bir yaklaşımla, ancak büyük parti boyutu gereksinimi nedeniyle önemli hesaplama kaynakları pahasına yüksek kaliteli öz-denetimli temsillerin öğrenilebileceğini göstermiştir.

<a name="4-moco-tr"></a>
## 4. MoCo: Denetimsiz Görsel Temsil Öğrenimi için Momentum Kontrastı

SimCLR, parti içi negatiflerin gücünü göstermiş olsa da, çok büyük parti boyutlarına bağımlılığı (muazzam GPU belleği ve hesaplama anlamına gelir) bir darboğaz haline gelmiştir. He ve ark. (2020) tarafından önerilen **MoCo**, **momentumla güncellenen bir kodlayıcı**ya sahip **dinamik bir sözlük** sunarak bu sınırlamayı ele almıştır ve böylece geniş ve tutarlı bir negatif örnek kümesi sağlamıştır.

### 4.1. MoCo Mimarisi ve Mekanizması

MoCo, karşılaştırmalı öğrenmeyi bir sözlük arama görevi olarak yeniden tasarlar. Amaç, sözlük aramalarını etkili bir şekilde yapabilen bir kodlayıcı eğitmektir.
1.  **İki Kodlayıcı:** MoCo iki kodlayıcı kullanır:
    *   **Sorgu Kodlayıcısı** ($f_q$): Parametreleri $\theta_q$ geri yayılım yoluyla eğitilen ağdır. Sorgu örneklerini kodlar.
    *   **Anahtar Kodlayıcısı** ($f_k$): $\theta_k$ parametrelerine sahip bu ağ, anahtar örneklerini (sözlüğü dolduran) kodlar. Önemlisi, $f_k$, $f_q$'nun momentumla güncellenmiş bir versiyonudur.
2.  **Momentum Güncellemesi:** Anahtar kodlayıcının parametreleri, sorgu kodlayıcının parametrelerinin hareketli ortalaması olarak güncellenir:
    $$
    \theta_k \leftarrow m \theta_k + (1-m) \theta_q
    $$
    Burada $m \in [0, 1)$ **momentum katsayısıdır** (tipik olarak yüksek, örn. 0.999 olarak ayarlanır). Bu yavaş güncelleme, sözlükteki anahtar temsillerinin zaman içinde tutarlı kalmasını, ancak yine de yavaş yavaş gelişmesini sağlar.
3.  **Bellek Bankası (Kuyruk):** MoCo, sadece parti içi negatifler yerine, önceki mini-partilerden kodlanmış anahtar temsillerinin bir **kuyruğunu** tutar. Bu kuyruk, dinamik bir sözlük görevi görür ve geniş bir negatif örnek havuzu sağlar. Yeni bir mini-parti geldiğinde, kodlanmış anahtarları kuyruğa eklenir ve en eski anahtarlar kaldırılır. Bu mekanizma, sözlük boyutunun mini-parti boyutundan bağımsız olmasını sağlar.
4.  **Karşılaştırmalı Kayıp:** SimCLR'ye benzer şekilde, InfoNCE kaybı kullanılır. Bir sorgu $q$, pozitif bir anahtar $k^+$ (aynı girdiden, $f_k$ tarafından kodlanmış) ve bellek bankasından örneklenen geniş bir negatif anahtar kümesi $\{k_i^-\}$ ile karşılaştırılır.

### 4.2. Temel Bulgular ve Katkılar

MoCo'nun yenilikleri çeşitli faydalar sağlamıştır:
*   **Ayrışık Sözlük Boyutu:** Mini-parti boyutundan bağımsız olarak isteğe bağlı olarak büyük bir negatif örnek sözlüğüne izin verir. Bu, SimCLR'ye göre önemli bir avantajdır ve standart, daha küçük parti boyutlarının (örn. 256) kullanılmasına izin verirken, binlerce negatife erişim sağlar.
*   **Tutarlı Negatif Örnekler:** Anahtar kodlayıcı için momentum güncelleme mekanizması, sözlükteki anahtarların nispeten tutarlı kalmasını sağlar, karşılaştırmalı öğrenme görevini kararsız hale getirecek kadar hızlı değişmelerini önler.
*   **En Son Teknoloji Performansı:** MoCo, çeşitli kıyaslamalarda güçlü performans göstererek, büyük, çeşitli negatif örnek kümelerinin (bellek kuyruğu aracılığıyla elde edilen) büyük parti boyutlarının kendilerinden daha kritik olduğunu kanıtlamıştır.

MoCo, büyük parti boyutu sorununa zarif bir çözüm sunarak karşılaştırmalı öğrenmeyi daha erişilebilir ve hesaplama açısından daha verimli hale getirmiş, böylece pratik uygulamalarda benimsenmesini hızlandırmıştır.

<a name="5-karşılaştırmalı-analiz"></a>
## 5. Karşılaştırmalı Analiz: SimCLR ve MoCo

Hem SimCLR hem de MoCo, karşılaştırmalı öğrenmede temel çalışmalardır, ancak etkili negatif örnekler elde etme sorununa farklı şekillerde yaklaşırlar.

| Özellik                 | SimCLR                                                              | MoCo                                                                       |
| :---------------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------- |
| **Negatif Örnekler**    | Mevcut büyük mini-partideki diğer tüm örnekler.                  | Geçmiş mini-partilerden gelen temsillerin dinamik olarak tutulan kuyruğu (bellek bankası). |
| **Parti Boyutuna Bağımlılık** | Yeterli negatif oluşturmak için çok büyük parti boyutlarına (örn. 4096-8192) **yüksek bağımlılık**. | Mini-parti boyutuna **düşük bağımlılık**; standart parti boyutları (örn. 256) ile etkilidir. |
| **Anahtar Kodlayıcı Güncellemesi**  | Ayrı bir anahtar kodlayıcı yok; her iki görünüm de aynı kodlayıcı tarafından işlenir ve geri yayılım yoluyla güncellenir. | Parametreleri $\theta_k$ olan ayrı anahtar kodlayıcı ($f_k$), sorgu kodlayıcının parametreleri $\theta_q$'nun **momentum hareketli ortalaması** yoluyla güncellenir. |
| **Kodlayıcı Mimarisi**| Tek bir kodlayıcı $f$ ve bir projeksiyon başlığı $g$.                    | Bir sorgu kodlayıcısı $f_q$, bir anahtar kodlayıcısı $f_k$ ve bir projeksiyon başlığı $g$.       |
| **Hesaplama Maliyeti**  | Büyük parti boyutları nedeniyle yüksek GPU belleği ve hesaplama.           | Daha küçük parti boyutları ve momentum güncellemesi nedeniyle hesaplama açısından daha verimli. |
| **Sözlük Özellikleri**| Negatifler sözlüğü anında oluşturulur ve her partide değişir. | Negatifler sözlüğü büyüktür, tutarlıdır ve yavaşça gelişir.        |
| **Performans**         | Zamanında SOTA'ya ulaştı, ancak ölçekleme için donanım tarafından sınırlıydı.     | SOTA'ya ulaştı, daha ölçeklenebilir ve parti boyutuna karşı daha sağlam.                     |

**Benzerlikler:**
*   Her ikisi de pozitif çiftler oluşturmak için **güçlü veri artırma** kullanır.
*   Her ikisi de **InfoNCE benzeri bir karşılaştırmalı kayıp** (SimCLR için NT-Xent) kullanır.
*   Her ikisi de temsilleri kayıp hesaplaması için gizli bir uzaya eşlemek üzere kodlayıcıdan sonra bir **projeksiyon başlığı** kullanır.
*   Her ikisi de öz-denetimli bir şekilde **yüksek kaliteli görsel temsiller** öğrenmeyi amaçlar.

Esasen, SimCLR veri artırma, projeksiyon başlıkları ve yeterli sayıda negatifin önemini vurgulayan bir temel sağlamıştır. MoCo ise, negatif örnekleri elde etmek için daha verimli bir mekanizma sunarak, karşılaştırmalı öğrenmeyi daha pratik ve ölçeklenebilir hale getirerek bunu geliştirmiştir. Bu çalışmalar, bilgisayar görüşü ve ötesinde daha sofistike ve güçlü öz-denetimli öğrenme teknikleri için yolu birlikte açmıştır.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği

Aşağıda, karşılaştırmalı kayıp fonksiyonunun (InfoNCE) temel konseptini gösteren basitleştirilmiş bir Python kod parçacığı bulunmaktadır. Bu örnek, `query`, `positive_key` ve `negative_keys`'in zaten çıkarılmış özellik vektörleri olduğu varsayılarak kayıp hesaplamasına odaklanmaktadır.

```python
import torch
import torch.nn.functional as F

def info_nce_loss(query_features, positive_key_features, negative_key_features, temperature=0.07):
    """
    Tek bir sorgu için InfoNCE kaybını hesaplar.

    Argümanlar:
        query_features (torch.Tensor): Sorgunun özellik vektörü (örn. z_i). Şekil: [feature_dim]
        positive_key_features (torch.Tensor): Pozitif anahtarın özellik vektörü (örn. z_j). Şekil: [feature_dim]
        negative_key_features (torch.Tensor): Negatif anahtarların özellik vektörleri. Şekil: [num_negatives, feature_dim]
        temperature (float): Logitler için ölçeklendirme faktörü.

    Döndürür:
        torch.Tensor: Skaler InfoNCE kaybı.
    """
    # Kosinüs benzerliği için özellikleri birim uzunluğa normalize et
    query_features = F.normalize(query_features, dim=0)
    positive_key_features = F.normalize(positive_key_features, dim=0)
    negative_key_features = F.normalize(negative_key_features, dim=1)

    # Sorgu ve pozitif anahtar arasındaki benzerliği hesapla
    # (normalize edilmiş vektörlerin nokta çarpımı kosinüs benzerliğidir)
    positive_logits = torch.dot(query_features, positive_key_features) / temperature

    # Sorgu ve tüm negatif anahtarlar arasındaki benzerliği hesapla
    negative_logits = torch.matmul(query_features, negative_key_features.T) / temperature

    # Pozitif ve negatif logitleri birleştir
    # İlk eleman pozitif örneğe karşılık gelir
    logits = torch.cat([positive_logits.unsqueeze(0), negative_logits], dim=0)

    # Etiketleri oluştur: Pozitif örnek için 0 (0. dizinde)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    # Çapraz entropi kaybını hesapla (burada InfoNCE'ye eşdeğerdir)
    loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))

    return loss

# Örnek Kullanım:
# Özellik vektörlerinin 128 boyutlu olduğu varsayılıyor
feature_dim = 128
num_negatives = 100

# Sahte özellikler oluştur
query = torch.randn(feature_dim)
positive_key = torch.randn(feature_dim)
negative_keys = torch.randn(num_negatives, feature_dim)

# Kaybı hesapla
loss_value = info_nce_loss(query, positive_key, negative_keys)
print(f"Hesaplanan InfoNCE Kaybı: {loss_value.item():.4f}")

# SimCLR/MoCo, bunu parti bazında uygulayarak, genellikle bir parti içinde
# veya bir bellek bankasıyla tümden-tüme benzerlikleri hesaplar.

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç

Karşılaştırmalı öğrenme, öz-denetimli öğrenmeyi devrim niteliğinde değiştirmiş, etiketlenmemiş verilerden anlamlı temsiller çıkarmak için güçlü bir paradigma sağlamıştır. SimCLR, güçlü veri artırma, projeksiyon başlıkları ve büyük parti boyutlarının basit mimarilerle etkinliğini göstererek, SSL'de elde edilebilir olanın sınırlarını zorlamıştır. MoCo ise, momentumla güncellenen bir anahtar kodlayıcı ve dinamik bir bellek kuyruğu tanıtarak büyük parti boyutlarının pratik sınırlamalarını ele almış, karşılaştırmalı öğrenmeyi daha ölçeklenebilir ve verimli hale getirmiştir.

SimCLR ve MoCo birlikte, öz-denetimli öğrenmenin anlaşılmasına ve uygulanmasına önemli ölçüde katkıda bulunan temel ilerlemeleri temsil etmektedir. Metodolojileri, çeşitli görüntü işleme görevlerinde etkileyici performans sergilemekle kalmamış, aynı zamanda temsil öğreniminde yeni bir araştırma dalgasına ilham vererek, derin öğrenme modellerini insan etiketlemesinin büyük yükü olmadan otonom özellik çıkarma hedefine yaklaştırmıştır. Bu çalışmalarla oluşturulan ilkeler, bilgisayar görüşünden doğal dil işlemeye kadar çeşitli alanlarda bugün kullanılan birçok gelişmiş öz-denetimli tekniğin temelini oluşturmaya devam etmektedir.
