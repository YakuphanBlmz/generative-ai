# DoRA: Weight-Decomposed Low-Rank Adaptation

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: LoRA and Parameter-Efficient Fine-Tuning (PEFT)](#2-background-lora-and-parameter-efficient-fine-tuning-peft)
- [3. DoRA: Weight-Decomposed Low-Rank Adaptation](#3-dora-weight-decomposed-low-rank-adaptation)
    - [3.1. The Intuition Behind DoRA](#31-the-intuition-behind-dora)
    - [3.2. Weight Decomposition into Magnitude and Direction](#32-weight-decomposition-into-magnitude-and-direction)
    - [3.3. Low-Rank Adaptation of the Directional Component](#33-low-rank-adaptation-of-the-directional-component)
    - [3.4. Reconstructing the Adapted Weight Matrix](#34-reconstructing-the-adapted-weight-matrix)
- [4. Advantages and Implications](#4-advantages-and-implications)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The rapid advancements in large-scale pre-trained models, particularly **Large Language Models (LLMs)** and **Vision Transformers (ViTs)**, have revolutionized numerous domains. However, fine-tuning these colossal models for downstream tasks presents significant challenges, primarily due to their immense number of parameters, which can reach hundreds of billions. Full fine-tuning demands substantial computational resources, including vast amounts of memory and processing power, making it inaccessible for many researchers and practitioners. This has spurred the development of **Parameter-Efficient Fine-Tuning (PEFT)** methods, which aim to adapt pre-trained models to new tasks by updating only a small subset of parameters while freezing the majority.

Among the various PEFT techniques, **Low-Rank Adaptation (LoRA)** has emerged as a highly effective and widely adopted approach. LoRA introduces small, learnable low-rank matrices into the model's architecture, which are added to the original weight matrices during fine-tuning. While LoRA has demonstrated remarkable success in mitigating the computational burden and achieving competitive performance, its updates can sometimes inadvertently impact the overall magnitude of the weight matrices, potentially limiting its full expressive power.

**DoRA (Weight-Decomposed Low-Rank Adaptation)** is a novel PEFT method that builds upon LoRA by introducing a critical insight: separating the magnitude and directional components of pre-trained weights. By decomposing the weight matrices into these two components and applying LoRA updates exclusively to the directional part, DoRA aims to achieve more stable and effective fine-tuning. This document delves into the theoretical underpinnings, architectural details, and practical implications of DoRA, providing a comprehensive overview of its mechanism and advantages.

## 2. Background: LoRA and Parameter-Efficient Fine-Tuning (PEFT)
To fully appreciate DoRA, it is essential to understand the context of **Parameter-Efficient Fine-Tuning (PEFT)** and the mechanism of **LoRA**.

**Parameter-Efficient Fine-Tuning (PEFT)** encompasses a family of techniques designed to adapt large pre-trained models to specific downstream tasks without fine-tuning all their parameters. The core idea is to significantly reduce the number of trainable parameters, thereby decreasing memory consumption, accelerating training, and mitigating the risk of catastrophic forgetting. Common PEFT strategies include:
*   **Adapter-based methods:** Injecting small, task-specific neural modules (adapters) between layers of the frozen pre-trained model.
*   **Prefix-Tuning and Prompt-Tuning:** Learning a small set of continuous task-specific vectors (prefixes or prompts) that are prepended to the input embeddings.
*   **Low-Rank Adaptation (LoRA):** The focus of our discussion, which modifies the pre-trained weight matrices directly.

**Low-Rank Adaptation (LoRA)**, proposed by Hu et al. (2021), operates by freezing the original pre-trained weight matrices `W_0 ∈ R^(d_out × d_in)` and injecting learnable low-rank decomposition matrices into the model. For any given weight matrix `W_0`, LoRA adds an update `ΔW` defined as the product of two smaller matrices, `A ∈ R^(d_in × r)` and `B ∈ R^(r × d_out)`, where `r` is the **rank** and `r << min(d_in, d_out)`. The adapted weight matrix `W'` is then computed as `W' = W_0 + B * A`. Crucially, only `A` and `B` are trainable, while `W_0` remains fixed. This dramatically reduces the number of trainable parameters. For example, if `W_0` is `1000x1000`, `A` is `1000x4`, and `B` is `4x1000`, the trainable parameters are `4000 + 4000 = 8000` instead of `1,000,000`.

LoRA's effectiveness stems from the **low-rank hypothesis**, which suggests that the updates needed for adaptation to new tasks often reside in a low-dimensional subspace. By injecting low-rank updates, LoRA efficiently captures task-specific knowledge. Despite its successes, LoRA's updates directly influence the combined magnitude and direction of the original weight matrix. This unified approach can sometimes lead to suboptimal performance, as the directional changes might inadvertently alter the magnitude in ways that are not ideal for fine-tuning, especially when `r` is chosen to be very small. DoRA addresses this limitation by decoupling these two aspects.

## 3. DoRA: Weight-Decomposed Low-Rank Adaptation
**DoRA (Weight-Decomposed Low-Rank Adaptation)** is an advanced PEFT method that refines LoRA by introducing a fundamental decomposition of pre-trained weight matrices. Its core innovation lies in separating the **magnitude** and **directional** components of weights and applying LoRA updates specifically to the directional part. This separation allows for more precise control over how weight matrices are updated, leading to enhanced performance and stability during fine-tuning.

### 3.1. The Intuition Behind DoRA
The primary intuition behind DoRA is that the impact of a weight matrix `W` can be broadly categorized into two aspects: its **magnitude** (how strong its connections are) and its **direction** (what specific features or patterns it focuses on). In LoRA, the low-rank update `ΔW` affects both these aspects simultaneously. However, changes to the magnitude of weights can have a significant and often sensitive impact on the network's behavior, affecting gradient flow and activation distributions.

DoRA postulates that for effective fine-tuning, it might be more beneficial to adapt the *direction* of the weight vectors while carefully controlling or preserving their *magnitudes*. By decoupling these components, DoRA can ensure that LoRA-style updates primarily refine the relational and feature-specific aspects of the weights without disrupting the learned scales, which are often crucial for the model's stability and performance.

### 3.2. Weight Decomposition into Magnitude and Direction
For any given pre-trained weight matrix `W_0 ∈ R^(d_out × d_in)`, DoRA first decomposes each row vector (or column vector, depending on convention) into its magnitude and a corresponding unit vector for its direction.
Specifically, for each row `w_i` of `W_0`:
*   The **magnitude** `m_i` is the Euclidean norm of the vector: `m_i = ||w_i||_2`.
*   The **directional vector** `v_i` is the unit vector: `v_i = w_i / ||w_i||_2`.

This decomposition can be generalized for the entire matrix `W_0` into a scalar magnitude vector `m ∈ R^(d_out)` (where each `m_i` is an element) and a directional matrix `V_0 ∈ R^(d_out × d_in)` composed of unit row vectors `v_i`.
Thus, the original weight matrix `W_0` can be expressed as:
`W_0 = diag(m) * V_0`
where `diag(m)` is a diagonal matrix with magnitudes `m_i` on its diagonal.

### 3.3. Low-Rank Adaptation of the Directional Component
Instead of applying the LoRA update `B * A` directly to `W_0`, DoRA applies it *only* to the **directional matrix** `V_0`. This means that during fine-tuning, a new directional matrix `V'` is learned as:
`V' = V_0 + B * A`
where `B ∈ R^(d_out × r)` and `A ∈ R^(r × d_in)` are the trainable low-rank matrices, similar to LoRA. Note that `V_0` here refers to the *original* normalized directional component of `W_0`.

After applying the low-rank update, the rows of `V'` are typically no longer unit vectors. To maintain the directional property and ensure proper scaling, `V'` is then **re-normalized** such that each of its row vectors becomes a unit vector. Let `v'_i` be the `i`-th row of `V'`. The re-normalized directional vector `v''_i` is:
`v''_i = v'_i / ||v'_i||_2`
This results in a new, adapted, and re-normalized directional matrix `V''`.

### 3.4. Reconstructing the Adapted Weight Matrix
With the updated and re-normalized directional matrix `V''` in hand, DoRA reconstructs the final adapted weight matrix `W_DoRA` by combining `V''` with the *original* magnitudes `m`.
`W_DoRA = diag(m) * V''`

This is a crucial step. By using the *original magnitudes*, DoRA explicitly preserves the scale information initially encoded in `W_0` while allowing the directional component `V` to be adaptively modified by the LoRA layers. This mechanism ensures that the magnitude changes are decoupled from the directional changes, preventing the fine-tuning process from disrupting beneficial scaling properties. Some variants might allow for a small, learnable scalar multiplier for `m` as well, but the primary approach emphasizes keeping `m` fixed.

## 4. Advantages and Implications
DoRA offers several significant advantages over traditional LoRA and other PEFT methods:

1.  **Improved Performance and Stability:** By explicitly decoupling magnitude and direction, DoRA ensures that low-rank updates primarily refine the semantic direction of the weights without inadvertently altering their learned scale. This can lead to more stable training and better generalization performance, particularly for tasks where precise magnitude relationships are critical.
2.  **Enhanced Expressiveness:** LoRA's updates, especially with very low ranks, might struggle to capture complex changes when they must simultaneously account for both magnitude and direction. DoRA allows the low-rank matrices to focus purely on directional shifts, potentially leading to a more expressive adaptation with the same low rank `r`.
3.  **Robustness to Rank Selection:** Empirical studies suggest that DoRA can achieve strong performance across a wider range of low-rank values, potentially being less sensitive to the specific choice of `r` compared to LoRA. This makes hyperparameter tuning slightly more forgiving.
4.  **Preservation of Pre-trained Knowledge:** By fixing the magnitude component `m` from the pre-trained weights, DoRA helps preserve a crucial aspect of the original model's learned representation. This can prevent catastrophic forgetting and ensure that fine-tuning builds effectively upon the pre-trained knowledge.
5.  **Compatibility with Existing LoRA Implementations:** DoRA can often be implemented as a wrapper or a slight modification around existing LoRA modules, making it relatively straightforward to integrate into current PEFT frameworks. The additional computational overhead during forward/backward passes is minimal, involving mostly element-wise operations and normalization.

However, it's also important to consider potential implications:
*   **Increased Complexity:** While conceptually elegant, DoRA introduces a slightly more complex mathematical structure and implementation compared to plain LoRA, requiring explicit decomposition and re-composition steps.
*   **Magnitude Fixation:** While beneficial in many cases, fixing the magnitude can be a limitation if a downstream task genuinely requires significant shifts in weight magnitudes. However, the existing literature suggests that directional changes are often more critical.

Overall, DoRA represents an important step forward in parameter-efficient fine-tuning, offering a principled way to enhance the effectiveness of low-rank adaptation by recognizing the distinct roles of weight magnitude and direction.

## 5. Code Example
This conceptual Python snippet illustrates how one might *think* about decomposing a weight matrix and applying an update to its directional component, followed by reconstruction. This is a simplified representation, not a full DoRA implementation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Original pre-trained weight (frozen)
        # In a real scenario, this would be loaded from a pre-trained model
        self.W_0 = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA A and B matrices (learnable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank)) # Initialize B to zero for identity at start

        # Decompose W_0 into magnitude and direction
        # Each row of W_0 is considered a vector
        magnitudes = torch.norm(self.W_0, p=2, dim=1, keepdim=True) # (out_features, 1)
        self.m = nn.Parameter(magnitudes, requires_grad=False) # Store magnitude (frozen)

        # Directional component V_0 (unit vectors for each row)
        # Avoid division by zero for rows with zero magnitude
        eps = 1e-6
        self.V_0 = nn.Parameter(self.W_0 / (magnitudes + eps), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Apply LoRA update to the directional component V_0
        # The update ΔV = B * A
        delta_V = self.lora_B @ self.lora_A

        # Updated directional component V' = V_0 + ΔV
        V_prime = self.V_0 + delta_V

        # 2. Re-normalize V' to get V'' (each row as a unit vector)
        magnitudes_V_prime = torch.norm(V_prime, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        eps = 1e-6
        V_double_prime = V_prime / (magnitudes_V_prime + eps)

        # 3. Reconstruct the adapted weight matrix W_DoRA = diag(m) * V''
        # Element-wise multiplication of each magnitude with its corresponding row in V''
        W_DoRA = self.m * V_double_prime

        # Apply the adapted weight matrix
        return F.linear(x, W_DoRA)

# Example usage
input_dim = 768
output_dim = 3072
lora_rank = 8

# Create a DoRA-enhanced linear layer
dora_linear_layer = DoRALinear(input_dim, output_dim, lora_rank)

# Simulate some input
dummy_input = torch.randn(1, input_dim) # Batch size 1

# Forward pass
output = dora_linear_layer(dummy_input)
print(f"Output shape: {output.shape}")

# Total trainable parameters for DoRA_Linear:
# lora_A: rank * in_features
# lora_B: out_features * rank
trainable_params = sum(p.numel() for p in dora_linear_layer.parameters() if p.requires_grad)
print(f"Trainable parameters in DoRALinear: {trainable_params}")

# For comparison, original W_0 parameters:
original_params = dora_linear_layer.W_0.numel()
print(f"Original W_0 parameters: {original_params}")

# Percentage of trainable parameters
print(f"Percentage of trainable parameters: {(trainable_params / original_params) * 100:.2f}%")

(End of code example section)
```

## 6. Conclusion
DoRA (Weight-Decomposed Low-Rank Adaptation) represents a significant advancement in the field of Parameter-Efficient Fine-Tuning (PEFT), building upon the foundational success of LoRA. By introducing a principled decomposition of pre-trained weights into separate magnitude and directional components, DoRA enables a more nuanced and effective adaptation strategy. It allows LoRA-style updates to focus primarily on refining the *direction* of weight vectors, which is often more critical for learning new task-specific features, while preserving the *magnitude* information that contributes to the model's stability and learned scale.

This approach leads to improved performance, enhanced training stability, and greater robustness across various downstream tasks and model architectures, particularly in the context of large language models and vision transformers. DoRA's ability to maintain high performance with a minimal number of trainable parameters underscores its value in democratizing access to and accelerating research with state-of-the-art foundation models. As PEFT methods continue to evolve, DoRA stands out as a powerful technique that refines our understanding of how to efficiently adapt massive neural networks to specific applications, paving the way for more performant and accessible AI systems.

---
<br>

<a name="türkçe-içerik"></a>
## DoRA: Ağırlık-Ayrıştırılmış Düşük-Rank Adaptasyonu

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: LoRA ve Parametre-Verimli İnce Ayar (PEFT)](#2-arka-plan-lora-ve-parametre-verimli-ince-ayar-peft)
- [3. DoRA: Ağırlık-Ayrıştırılmış Düşük-Rank Adaptasyonu](#3-dora-ağırlık-ayrıştırılmış-düşük-rank-adaptasyonu)
    - [3.1. DoRA'nın Sezgisel Temeli](#31-doranın-sezgisel-temeli)
    - [3.2. Ağırlığın Büyüklük ve Yön Olarak Ayrıştırılması](#32-ağırlığın-büyüklük-ve-yön-olarak-ayrıştırılması)
    - [3.3. Yön Bileşeninin Düşük-Rank Adaptasyonu](#33-yön-bileşeninin-düşük-rank-adaptasyonu)
    - [3.4. Adapte Edilmiş Ağırlık Matrisinin Yeniden İnşası](#34-adapte-edilmiş-ağırlık-matrisinin-yeniden-inşası)
- [4. Avantajlar ve Etkileri](#4-avantajlar-ve-etkileri)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
**Büyük Dil Modelleri (LLM'ler)** ve **Görsel Dönüştürücüler (ViT'ler)** başta olmak üzere, büyük ölçekli önceden eğitilmiş modellerdeki hızlı gelişmeler birçok alanı dönüştürdü. Ancak, bu devasa modelleri belirli görevler için ince ayarlamak, ağırlıklı olarak yüz milyarlarca parametreye ulaşabilen sayıları nedeniyle önemli zorluklar sunmaktadır. Tam ince ayar, büyük miktarda bellek ve işlem gücü dahil olmak üzere önemli hesaplama kaynakları gerektirir ve bu da birçok araştırmacı ve uygulayıcı için erişilemez hale getirir. Bu durum, önceden eğitilmiş modelleri, parametrelerin yalnızca küçük bir alt kümesini güncelleyerek ve çoğunluğunu dondurarak yeni görevlere adapte etmeyi amaçlayan **Parametre-Verimli İnce Ayar (PEFT)** yöntemlerinin geliştirilmesini teşvik etmiştir.

Çeşitli PEFT teknikleri arasında, **Düşük-Rank Adaptasyonu (LoRA)**, oldukça etkili ve yaygın olarak benimsenen bir yaklaşım olarak ortaya çıkmıştır. LoRA, modelin mimarisine küçük, öğrenilebilir düşük-rank matrisler ekler ve bunlar ince ayar sırasında orijinal ağırlık matrislerine eklenir. LoRA, hesaplama yükünü azaltmada ve rekabetçi performans elde etmede dikkat çekici başarı göstermiş olsa da, güncellemeleri bazen ağırlık matrislerinin genel büyüklüğünü istemeden etkileyebilir, bu da tam ifade gücünü sınırlayabilir.

**DoRA (Ağırlık-Ayrıştırılmış Düşük-Rank Adaptasyonu)**, LoRA'yı temel alan yeni bir PEFT yöntemidir ve önceden eğitilmiş ağırlıkların büyüklük ve yön bileşenlerini ayırma kritik bir anlayışını getirir. Ağırlık matrislerini bu iki bileşene ayırarak ve LoRA güncellemelerini yalnızca yönsel kısma uygulayarak, DoRA daha istikrarlı ve etkili ince ayar elde etmeyi hedefler. Bu belge, DoRA'nın teorik temellerini, mimari detaylarını ve pratik çıkarımlarını inceleyerek mekanizması ve avantajları hakkında kapsamlı bir genel bakış sunmaktadır.

## 2. Arka Plan: LoRA ve Parametre-Verimli İnce Ayar (PEFT)
DoRA'yı tam olarak takdir etmek için, **Parametre-Verimli İnce Ayar (PEFT)** bağlamını ve **LoRA**'nın mekanizmasını anlamak esastır.

**Parametre-Verimli İnce Ayar (PEFT)**, büyük önceden eğitilmiş modelleri tüm parametrelerini ince ayarlamadan belirli görevlere adapte etmek için tasarlanmış bir teknik ailesini kapsar. Temel fikir, eğitilebilir parametre sayısını önemli ölçüde azaltmak ve böylece bellek tüketimini azaltmak, eğitimi hızlandırmak ve felaket unutma riskini azaltmaktır. Yaygın PEFT stratejileri şunları içerir:
*   **Adaptör tabanlı yöntemler:** Donmuş önceden eğitilmiş modelin katmanları arasına küçük, göreve özel nöral modüller (adaptörler) enjekte etmek.
*   **Önek Ayarı (Prefix-Tuning) ve İstem Ayarı (Prompt-Tuning):** Girdi gömmelerine önden eklenen küçük, sürekli göreve özel vektörler (önekler veya istemler) öğrenmek.
*   **Düşük-Rank Adaptasyonu (LoRA):** Doğrudan önceden eğitilmiş ağırlık matrislerini değiştiren, tartışmamızın odak noktası.

Hu ve diğerleri (2021) tarafından önerilen **Düşük-Rank Adaptasyonu (LoRA)**, orijinal önceden eğitilmiş `W_0 ∈ R^(d_out × d_in)` ağırlık matrislerini dondurarak ve modele öğrenilebilir düşük-rank ayrıştırma matrisleri enjekte ederek çalışır. Herhangi bir `W_0` ağırlık matrisi için, LoRA, iki küçük matrisin, `A ∈ R^(d_in × r)` ve `B ∈ R^(r × d_out)`'un ürünü olarak tanımlanan bir güncelleme `ΔW` ekler; burada `r` **rank**'tır ve `r << min(d_in, d_out)`. Adapte edilmiş ağırlık matrisi `W'` daha sonra `W' = W_0 + B * A` olarak hesaplanır. Önemli olarak, sadece `A` ve `B` eğitilebilirdir, `W_0` ise sabit kalır. Bu, eğitilebilir parametre sayısını önemli ölçüde azaltır. Örneğin, `W_0` `1000x1000` ise, `A` `1000x4` ve `B` `4x1000` ise, eğitilebilir parametreler `1,000,000` yerine `4000 + 4000 = 8000`'dir.

LoRA'nın etkinliği, yeni görevlere adaptasyon için gereken güncellemelerin genellikle düşük boyutlu bir alt uzayda yattığını öne süren **düşük-rank hipotezinden** kaynaklanmaktadır. Düşük-rank güncellemeler enjekte ederek, LoRA göreve özgü bilgileri verimli bir şekilde yakalar. Başarılarına rağmen, LoRA'nın güncellemeleri orijinal ağırlık matrisinin birleşik büyüklüğünü ve yönünü doğrudan etkiler. Bu birleşik yaklaşım bazen suboptimal performansa yol açabilir, çünkü yönsel değişiklikler büyüklüğü ince ayar için ideal olmayan şekillerde istemeden değiştirebilir, özellikle `r` çok küçük seçildiğinde. DoRA, bu iki yönü ayırarak bu sınırlamayı ele alır.

## 3. DoRA: Ağırlık-Ayrıştırılmış Düşük-Rank Adaptasyonu
**DoRA (Ağırlık-Ayrıştırılmış Düşük-Rank Adaptasyonu)**, önceden eğitilmiş ağırlık matrislerinin temel bir ayrışmasını getirerek LoRA'yı rafine eden gelişmiş bir PEFT yöntemidir. Temel yeniliği, ağırlıkların **büyüklük** ve **yönsel** bileşenlerini ayırması ve LoRA güncellemelerini özellikle yönsel kısma uygulamasıdır. Bu ayrım, ağırlık matrislerinin nasıl güncellendiği üzerinde daha hassas kontrol sağlayarak ince ayar sırasında artırılmış performans ve istikrara yol açar.

### 3.1. DoRA'nın Sezgisel Temeli
DoRA'nın arkasındaki temel sezgi, bir ağırlık matrisinin `W` etkisinin kabaca iki yönlü olarak kategorize edilebileceğidir: **büyüklüğü** (bağlantılarının ne kadar güçlü olduğu) ve **yönü** (hangi belirli özelliklere veya kalıplara odaklandığı). LoRA'da, düşük-rank güncellemesi `ΔW` bu iki yönü eşzamanlı olarak etkiler. Ancak, ağırlıkların büyüklüğündeki değişiklikler, ağın davranışını önemli ve genellikle hassas bir şekilde etkileyebilir, gradyan akışını ve aktivasyon dağılımlarını etkiler.

DoRA, etkili ince ayar için, ağırlık vektörlerinin *yönünü* adapte etmenin, büyüklüklerini dikkatlice kontrol ederken veya korurken daha faydalı olabileceğini varsayar. Bu bileşenleri ayırarak, DoRA, LoRA tarzı güncellemelerin, modelin istikrarı ve performansı için genellikle çok önemli olan öğrenilmiş ölçekleri bozmadan, ağırlıkların ilişkisel ve özelliğe özgü yönlerini öncelikle rafine etmesini sağlayabilir.

### 3.2. Ağırlığın Büyüklük ve Yön Olarak Ayrıştırılması
Herhangi bir önceden eğitilmiş `W_0 ∈ R^(d_out × d_in)` ağırlık matrisi için, DoRA önce her satır vektörünü (veya sütun vektörünü, konvansiyona bağlı olarak) büyüklüğüne ve yönü için karşılık gelen birim vektöre ayırır.
Özellikle, `W_0`'ın her `w_i` satırı için:
*   **Büyüklük** `m_i`, vektörün Öklid normudur: `m_i = ||w_i||_2`.
*   **Yönel vektör** `v_i`, birim vektördür: `v_i = w_i / ||w_i||_2`.

Bu ayrışma, tüm `W_0` matrisi için, bir skaler büyüklük vektörü `m ∈ R^(d_out)` (burada her `m_i` bir elemandır) ve birim satır vektörlerden `v_i` oluşan bir yönel matris `V_0 ∈ R^(d_out × d_in)` olarak genellenebilir.
Böylece, orijinal ağırlık matrisi `W_0` şu şekilde ifade edilebilir:
`W_0 = diag(m) * V_0`
burada `diag(m)`, köşegeninde `m_i` büyüklükleri olan bir köşegen matristir.

### 3.3. Yön Bileşeninin Düşük-Rank Adaptasyonu
LoRA güncellemesini `B * A` doğrudan `W_0`'a uygulamak yerine, DoRA bunu *sadece* **yönel matris** `V_0`'a uygular. Bu, ince ayar sırasında yeni bir yönel matris `V'`'nin şu şekilde öğrenildiği anlamına gelir:
`V' = V_0 + B * A`
burada `B ∈ R^(d_out × r)` ve `A ∈ R^(r × d_in)` eğitilebilir düşük-rank matrislerdir, LoRA'ya benzer şekilde. Burada `V_0`, `W_0`'ın *orijinal* normalleştirilmiş yönel bileşenini ifade eder.

Düşük-rank güncellemesini uyguladıktan sonra, `V'`'nin satırları genellikle artık birim vektörler değildir. Yönel özelliğini korumak ve uygun ölçeklendirmeyi sağlamak için, `V'` daha sonra **yeniden normalleştirilir**, böylece her bir satır vektörü birim vektör haline gelir. `V'`'nin `i`-inci satırı `v'_i` olsun. Yeniden normalleştirilmiş yönel vektör `v''_i` şudur:
`v''_i = v'_i / ||v'_i||_2`
Bu, yeni, adapte edilmiş ve yeniden normalleştirilmiş bir yönel matris `V''` ile sonuçlanır.

### 3.4. Adapte Edilmiş Ağırlık Matrisinin Yeniden İnşası
Güncellenmiş ve yeniden normalleştirilmiş yönel matris `V''` eldeki mevcutken, DoRA, `V''`'yi *orijinal* `m` büyüklükleri ile birleştirerek nihai adapte edilmiş ağırlık matrisi `W_DoRA`'yı yeniden inşa eder.
`W_DoRA = diag(m) * V''`

Bu çok önemli bir adımdır. *Orijinal büyüklükleri* kullanarak, DoRA, `W_0`'da başlangıçta kodlanmış ölçek bilgilerini açıkça korurken, yönel bileşen `V`'nin LoRA katmanları tarafından uyarlanabilir bir şekilde değiştirilmesine izin verir. Bu mekanizma, büyüklük değişikliklerinin yönel değişikliklerinden ayrılmasını sağlayarak, ince ayar sürecinin faydalı ölçekleme özelliklerini bozmasını engeller. Bazı varyantlar, `m` için küçük, öğrenilebilir bir skaler çarpan da sağlayabilir, ancak birincil yaklaşım `m`'yi sabit tutmayı vurgular.

## 4. Avantajlar ve Etkileri
DoRA, geleneksel LoRA ve diğer PEFT yöntemlerine göre çeşitli önemli avantajlar sunar:

1.  **Geliştirilmiş Performans ve Kararlılık:** DoRA, büyüklüğü ve yönü açıkça ayırarak, düşük-rank güncellemelerinin, ağırlıkların öğrenilmiş ölçeğini istemeden değiştirmeden, ağırlıkların anlamsal yönünü öncelikle rafine etmesini sağlar. Bu, özellikle hassas büyüklük ilişkilerinin kritik olduğu görevler için daha kararlı eğitime ve daha iyi genelleme performansına yol açabilir.
2.  **Artırılmış İfade Gücü:** LoRA'nın güncellemeleri, özellikle çok düşük ranklarla, hem büyüklüğü hem de yönü aynı anda hesaba katmak zorunda kaldıklarında karmaşık değişiklikleri yakalamakta zorlanabilir. DoRA, düşük-rank matrislerin tamamen yönsel kaymalara odaklanmasına izin vererek, aynı düşük `r` rankı ile daha etkileyici bir adaptasyona yol açabilir.
3.  **Rank Seçimine Karşı Sağlamlık:** Deneysel çalışmalar, DoRA'nın daha geniş bir düşük-rank değeri aralığında güçlü performans elde edebileceğini ve potansiyel olarak LoRA'ya kıyasla `r`'nin belirli seçimine daha az duyarlı olduğunu göstermektedir. Bu, hiperparametre ayarını biraz daha affedici hale getirir.
4.  **Önceden Eğitilmiş Bilginin Korunması:** DoRA, önceden eğitilmiş ağırlıklardan `m` büyüklük bileşenini sabitleyerek, orijinal modelin öğrenilmiş temsilinin kritik bir yönünü korumaya yardımcı olur. Bu, felaket unutmayı önleyebilir ve ince ayarın önceden eğitilmiş bilgi üzerine etkili bir şekilde inşa edilmesini sağlayabilir.
5.  **Mevcut LoRA Uygulamalarıyla Uyumluluk:** DoRA, genellikle mevcut LoRA modüllerinin etrafında bir sarmalayıcı veya hafif bir değişiklik olarak uygulanabilir, bu da onu mevcut PEFT çerçevelerine entegre etmeyi nispeten kolaylaştırır. İleri/geri geçişler sırasındaki ek hesaplama yükü, çoğunlukla eleman bazında işlemler ve normalizasyon içerdiğinden minimaldir.

Ancak, potansiyel etkileri de göz önünde bulundurmak önemlidir:
*   **Artan Karmaşıklık:** Kavramsal olarak zarif olsa da, DoRA, düz LoRA'ya kıyasla biraz daha karmaşık bir matematiksel yapı ve uygulama getirir, açık ayrıştırma ve yeniden birleştirme adımları gerektirir.
*   **Büyüklük Sabitleme:** Birçok durumda faydalı olsa da, büyüklüğü sabitlemek, eğer bir sonraki görev ağırlık büyüklüklerinde gerçekten önemli kaymalar gerektiriyorsa bir sınırlama olabilir. Ancak, mevcut literatür, yönsel değişikliklerin genellikle daha kritik olduğunu öne sürmektedir.

Genel olarak DoRA, ağırlık büyüklüğü ve yönünün farklı rollerini tanıyarak düşük-rank adaptasyonunun etkinliğini artırmanın ilkeli bir yolunu sunarak, parametre-verimli ince ayar alanında önemli bir ilerlemeyi temsil etmektedir.

## 5. Kod Örneği
Bu kavramsal Python kodu, bir ağırlık matrisinin nasıl ayrıştırılabileceğini ve yön bileşenine bir güncellemenin nasıl uygulanabileceğini, ardından yeniden inşa etmeyi *nasıl düşünebileceğimizi* göstermektedir. Bu, tam bir DoRA uygulamasının basitleştirilmiş bir temsilidir.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Orijinal önceden eğitilmiş ağırlık (dondurulmuş)
        # Gerçek bir senaryoda, bu önceden eğitilmiş bir modelden yüklenirdi
        self.W_0 = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA A ve B matrisleri (öğrenilebilir)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank)) # Başlangıçta B'yi sıfıra ayarla

        # W_0'ı büyüklük ve yöne ayır
        # W_0'ın her satırı bir vektör olarak kabul edilir
        magnitudes = torch.norm(self.W_0, p=2, dim=1, keepdim=True) # (out_features, 1)
        self.m = nn.Parameter(magnitudes, requires_grad=False) # Büyüklüğü sakla (dondurulmuş)

        # Yön bileşeni V_0 (her satır için birim vektörler)
        # Sıfır büyüklüğe sahip satırlar için sıfıra bölmeyi önle
        eps = 1e-6
        self.V_0 = nn.Parameter(self.W_0 / (magnitudes + eps), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. LoRA güncellemesini yönel bileşen V_0'a uygula
        # Güncelleme ΔV = B * A
        delta_V = self.lora_B @ self.lora_A

        # Güncellenmiş yönel bileşen V' = V_0 + ΔV
        V_prime = self.V_0 + delta_V

        # 2. V'yi V'' elde etmek için yeniden normalleştir (her satır bir birim vektör olarak)
        magnitudes_V_prime = torch.norm(V_prime, p=2, dim=1, keepdim=True)
        # Sıfıra bölmeyi önle
        eps = 1e-6
        V_double_prime = V_prime / (magnitudes_V_prime + eps)

        # 3. Adapte edilmiş ağırlık matrisi W_DoRA = diag(m) * V''yi yeniden inşa et
        # Her büyüklüğün V''deki karşılık gelen satırıyla eleman bazında çarpımı
        W_DoRA = self.m * V_double_prime

        # Adapte edilmiş ağırlık matrisini uygula
        return F.linear(x, W_DoRA)

# Örnek kullanım
input_dim = 768
output_dim = 3072
lora_rank = 8

# DoRA ile geliştirilmiş bir doğrusal katman oluştur
dora_linear_layer = DoRALinear(input_dim, output_dim, lora_rank)

# Bazı girdileri simüle et
dummy_input = torch.randn(1, input_dim) # Batch boyutu 1

# İleri besleme
output = dora_linear_layer(dummy_input)
print(f"Çıktı şekli: {output.shape}")

# DoRALinear için toplam eğitilebilir parametreler:
# lora_A: rank * in_features
# lora_B: out_features * rank
trainable_params = sum(p.numel() for p in dora_linear_layer.parameters() if p.requires_grad)
print(f"DoRALinear'daki eğitilebilir parametreler: {trainable_params}")

# Karşılaştırma için, orijinal W_0 parametreleri:
original_params = dora_linear_layer.W_0.numel()
print(f"Orijinal W_0 parametreleri: {original_params}")

# Eğitilebilir parametrelerin yüzdesi
print(f"Eğitilebilir parametrelerin yüzdesi: {(trainable_params / original_params) * 100:.2f}%")

(Kod örneği bölümünün sonu)
```

## 6. Sonuç
DoRA (Ağırlık-Ayrıştırılmış Düşük-Rank Adaptasyonu), LoRA'nın temel başarısını temel alan, Parametre-Verimli İnce Ayar (PEFT) alanında önemli bir ilerlemeyi temsil etmektedir. Önceden eğitilmiş ağırlıkları ayrı büyüklük ve yön bileşenlerine ilkeli bir ayrıştırma getirerek, DoRA daha incelikli ve etkili bir adaptasyon stratejisi sağlar. Bu, LoRA tarzı güncellemelerin öncelikle ağırlık vektörlerinin *yönünü* rafine etmeye odaklanmasını sağlar; bu, yeni göreve özgü özelliklerin öğrenilmesi için genellikle daha kritiktir, aynı zamanda modelin kararlılığına ve öğrenilmiş ölçeğine katkıda bulunan *büyüklük* bilgisini korur.

Bu yaklaşım, özellikle büyük dil modelleri ve görsel dönüştürücüler bağlamında, çeşitli alt akış görevleri ve model mimarileri genelinde iyileştirilmiş performansa, artırılmış eğitim kararlılığına ve daha fazla sağlamlığa yol açar. DoRA'nın minimum sayıda eğitilebilir parametre ile yüksek performansı sürdürme yeteneği, son teknoloji temel modellerle araştırmaya erişimi demokratikleştirmede ve hızlandırmadaki değerini vurgulamaktadır. PEFT yöntemleri gelişmeye devam ettikçe, DoRA, büyük sinir ağlarını belirli uygulamalara verimli bir şekilde nasıl adapte edeceğimiz konusundaki anlayışımızı rafine eden güçlü bir teknik olarak öne çıkmakta ve daha performanslı ve erişilebilir yapay zeka sistemleri için zemin hazırlamaktadır.



