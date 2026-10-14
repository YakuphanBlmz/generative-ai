# Model Merging Techniques: Linear, SLERP, and DARE

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Rationale Behind Model Merging](#2-the-rationale-behind-model-merging)
- [3. Key Model Merging Techniques](#3-key-model-merging-techniques)
  - [3.1. Linear Merging](#31-linear-merging)
  - [3.2. Spherical Linear Interpolation (SLERP)](#32-spherical-linear-interpolation-slerp)
  - [3.3. DARE (Dropout-aware Residual Merging)](#33-dare-dropout-aware-residual-merging)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

### 1. Introduction

In the rapidly evolving landscape of Generative AI and deep learning, the ability to combine knowledge from multiple pre-trained models has emerged as a powerful paradigm. **Model merging** refers to a suite of techniques designed to synthesize the parameters (weights and biases) of several distinct models into a single, cohesive model, ideally inheriting the beneficial properties and capabilities of its constituents without the computational overhead of an ensemble. This approach stands in contrast to traditional model training, offering a more efficient pathway to create specialized or more robust models.

This document delves into three prominent model merging techniques: **Linear Merging**, **Spherical Linear Interpolation (SLERP)**, and **DARE (Dropout-aware Residual Merging)**. Each method offers a unique approach to parameter interpolation, with varying assumptions about the model architectures and training histories, and exhibiting distinct performance characteristics in different applications. Understanding these techniques is crucial for researchers and practitioners aiming to optimize model deployment, facilitate knowledge transfer, and achieve novel model capabilities through composition.

### 2. The Rationale Behind Model Merging

The motivation for model merging stems from several practical and theoretical considerations within deep learning:

*   **Knowledge Aggregation:** Different models, even when trained on similar datasets, might learn distinct features or specialize in different aspects of the data. Merging can aggregate this disparate knowledge, potentially leading to a more generalist or expert model.
*   **Parameter Efficiency:** Instead of maintaining and serving multiple large models (e.g., for different tasks or styles), a single merged model can reduce memory footprint and inference latency, especially crucial for large language models (LLMs) and diffusion models.
*   **Mitigating Catastrophic Forgetting:** In continual learning scenarios, merging can help integrate new knowledge from fine-tuned models while retaining capabilities from older models, avoiding the complete overwrite of previously acquired skills.
*   **Exploiting Mode Connectivity:** Empirical evidence suggests that loss landscapes in deep learning models often exhibit "mode connectivity," meaning that different optimal solutions (modes) can be connected by simple paths with low loss barriers. This property makes linear interpolation of parameters feasible and often effective.
*   **Creating Hybrid Capabilities:** By merging models trained for different modalities or tasks (e.g., a style model with a content model in image generation), novel hybrid capabilities can be unlocked, enabling nuanced control over generative processes.
*   **Fine-tuning Optimization:** Instead of extensively fine-tuning a base model for every specific task, one can fine-tune several lightweight adapter modules or models and then merge their learned parameters back into the base model or with each other.

These advantages position model merging as a valuable tool in the Generative AI toolbox, particularly for models like Large Language Models (LLMs) and Stable Diffusion variants, where pre-trained models are frequently adapted or combined.

### 3. Key Model Merging Techniques

#### 3.1. Linear Merging

**Linear merging**, also known as **weighted averaging**, is the simplest and often the first approach considered for combining the parameters of two or more neural networks. Given two models, Model A with parameters $W_A$ and Model B with parameters $W_B$, a linearly merged model $W_{merged}$ can be obtained by a convex combination:

$W_{merged} = (1 - \alpha)W_A + \alpha W_B$

Here, $\alpha$ is a scalar weight typically ranging from 0 to 1, controlling the contribution of Model B relative to Model A. This operation is performed element-wise across all corresponding parameters (weights and biases) of the networks. For merging multiple models, the formula extends to a weighted sum where the sum of all weights equals 1.

**Assumptions and Considerations:**
*   **Identical Architectures:** Linear merging strictly requires that the models have identical architectures, meaning the same layers, neuron counts, and parameter shapes.
*   **Shared Initialization:** Often, models are more amenable to linear merging if they started from a common base model or were initialized similarly. This helps align their "parameter spaces" and ensures that similar parameters correspond to similar functionalities.
*   **Mode Connectivity:** The success of linear merging is largely attributed to the empirical observation of mode connectivity. If the loss functions of the individual models have optima connected by a path of sufficiently low loss, then a linear interpolation along this path can yield a functional model.

**Advantages:**
*   **Simplicity:** Extremely easy to implement and computationally inexpensive.
*   **Effectiveness:** Surprisingly effective in many scenarios, especially for models fine-tuned from a common base.

**Limitations:**
*   **Geometric Distortion:** Linear interpolation in high-dimensional parameter spaces can sometimes lead to models that perform poorly in intermediate regions, especially if the models occupy distinct "modes" that are not well-connected by a straight line in the loss landscape.
*   **Loss of Normalization:** If weights are normalized (e.g., in certain layers), linear averaging might destroy this normalization, potentially impacting performance.

#### 3.2. Spherical Linear Interpolation (SLERP)

**Spherical Linear Interpolation (SLERP)** is a method designed for interpolating between two points on the surface of a sphere. While linear interpolation moves along a straight line through Euclidean space, SLERP moves along the shortest path on the surface of a hypersphere (a **great arc**). In the context of model merging, SLERP is particularly useful when the directionality or relative magnitudes of weight vectors are important, or when parameters are effectively normalized.

Given two weight vectors $W_A$ and $W_B$ (assumed to be normalized to unit length for conceptual simplicity), and an interpolation parameter $\alpha \in [0, 1]$, the SLERP formula is:

$SLERP(W_A, W_B, \alpha) = \frac{\sin((1 - \alpha)\theta)}{\sin(\theta)}W_A + \frac{\sin(\alpha\theta)}{\sin(\theta)}W_B$

where $\theta$ is the angle between $W_A$ and $W_B$ in the high-dimensional space, calculated as $\theta = \arccos(W_A \cdot W_B)$ (assuming $W_A$ and $W_B$ are normalized vectors). If $\sin(\theta)$ is zero (i.e., $W_A$ and $W_B$ are collinear), SLERP defaults to linear interpolation.

**Assumptions and Considerations:**
*   **Normalization:** While the original SLERP assumes normalized vectors, its application in deep learning often involves treating layers' weight matrices or entire model parameter vectors as high-dimensional points. Pre-processing (e.g., normalizing layers or groups of weights) might be necessary for SLERP to yield its full benefits.
*   **Directional Importance:** SLERP preserves the relative scale of the vectors more effectively than linear interpolation, making it suitable where the *direction* of the weight vector is more critical than its exact Euclidean magnitude.

**Advantages:**
*   **Smoother Interpolation:** Often produces a smoother transition between models in terms of performance, as it respects the underlying geometry of the parameter space more accurately than linear interpolation, particularly when dealing with feature vectors or embeddings.
*   **Maintains Properties:** Can be better at preserving certain desirable properties (e.g., normalization) if applied judiciously.

**Limitations:**
*   **Computational Cost:** More computationally intensive than linear interpolation due to the trigonometric functions and dot product calculations.
*   **Normalization Pre-requisites:** Requires careful consideration of how parameters are normalized or if the assumptions for spherical interpolation genuinely apply to the raw parameter space. It's often applied to specific parts of the model (e.g., embeddings) rather than all parameters indiscriminately.

#### 3.3. DARE (Dropout-aware Residual Merging)

**DARE (Dropout-aware Residual Merging)** is a more recent and advanced technique specifically designed to merge models that have been fine-tuned from a common pre-trained **base model**. It addresses a critical challenge in merging: the phenomenon where even small differences in weights can lead to significant performance degradation due to the interplay of different learned "features" when simply averaging. DARE focuses on merging the *residuals* or *deltas* from the base model, rather than the absolute weights.

Given a base model $W_{base}$ and two fine-tuned models $W_A$ and $W_B$, DARE merges the *differences* each model learned from the base:

$W_{merged} = W_{base} + (1 - \alpha)(W_A - W_{base}) + \alpha(W_B - W_{base})$

This formulation, essentially a linear interpolation of the residual updates, is often combined with a crucial step: **zeroing out small weights** in the residual updates. This "dropout-aware" component is inspired by the observation that fine-tuning often results in sparse updates, and preserving this sparsity (by resetting small weights in the residual to zero) helps maintain model performance, especially in the presence of Dropout regularization during training. Specifically, DARE suggests dropping out a percentage of the smallest magnitude residual weights, effectively merging only the most significant changes.

**Assumptions and Considerations:**
*   **Common Base Model:** DARE's effectiveness is heavily reliant on the existence of a shared base model from which $W_A$ and $W_B$ were fine-tuned.
*   **Residual Sparsity:** It leverages the idea that fine-tuning often introduces sparse changes relative to the base model.
*   **Dropout Mechanism:** The "dropout-aware" aspect directly relates to how dropout regularization influences the learning process and parameter distributions.

**Advantages:**
*   **Superior Performance for Fine-tuned Models:** Often yields significantly better results than simple linear merging when combining models fine-tuned from a common base, especially for LLMs.
*   **Preserves Base Knowledge:** By merging residuals, it ensures that the core knowledge of the base model is retained, and only the fine-tuned differentiations are combined.
*   **Addresses Weight Conflict:** By selectively merging only significant changes, it can mitigate conflicts arising from incompatible small weight adjustments.

**Limitations:**
*   **Requires Base Model:** Not applicable if models were trained independently from different initializations.
*   **Hyperparameter Tuning:** The percentage of weights to zero out ("p_weights") is a crucial hyperparameter that needs careful tuning.

### 4. Code Example

Below is a conceptual Python code snippet demonstrating how linear merging might be applied to two simplified "model" weight dictionaries.

```python
import torch

def linear_merge_models(model_a_state_dict, model_b_state_dict, alpha):
    """
    Performs linear interpolation between the state dictionaries of two models.

    Args:
        model_a_state_dict (dict): State dictionary of the first model.
        model_b_state_dict (dict): State dictionary of the second model.
        alpha (float): Interpolation factor, between 0 and 1.
                       alpha=0 uses only model_a, alpha=1 uses only model_b.

    Returns:
        dict: A new state dictionary representing the linearly merged model.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1.")

    merged_state_dict = {}
    for key in model_a_state_dict.keys():
        if key not in model_b_state_dict:
            raise ValueError(f"Key '{key}' missing in model_b_state_dict.")

        # Ensure tensors have compatible shapes and types
        if model_a_state_dict[key].shape != model_b_state_dict[key].shape:
            raise ValueError(f"Shape mismatch for key '{key}'.")

        # Perform linear interpolation
        merged_state_dict[key] = (1 - alpha) * model_a_state_dict[key] + \
                                 alpha * model_b_state_dict[key]
    return merged_state_dict

# Example Usage:
# Simulate state dictionaries for two simple models
# In a real scenario, these would be loaded from .pth or .safetensors files
model_a_weights = {
    "layer1.weight": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    "layer1.bias": torch.tensor([0.01, 0.02]),
}

model_b_weights = {
    "layer1.weight": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
    "layer1.bias": torch.tensor([0.03, 0.04]),
}

# Merge with alpha = 0.5 (equal weighting)
merged_weights_0_5 = linear_merge_models(model_a_weights, model_b_weights, 0.5)
print("Merged weights (alpha=0.5):")
for key, value in merged_weights_0_5.items():
    print(f"{key}:\n{value}")

# Merge with alpha = 0.1 (mostly model A)
merged_weights_0_1 = linear_merge_models(model_a_weights, model_b_weights, 0.1)
print("\nMerged weights (alpha=0.1):")
for key, value in merged_weights_0_1.items():
    print(f"{key}:\n{value}")

# To apply this to a real model, you would then load a model
# and call model.load_state_dict(merged_state_dict)

(End of code example section)
```

### 5. Conclusion

Model merging techniques represent a sophisticated approach to leveraging and combining the knowledge encapsulated within distinct neural network models. From the straightforward **Linear Merging** that benefits from the concept of mode connectivity, to the geometrically informed **SLERP** suitable for directional parameter spaces, and finally to the specialized **DARE** for merging fine-tuned models from a common base, each method offers unique advantages and applicability.

The choice of merging technique depends heavily on the specific context: the relationship between the models (e.g., trained independently, fine-tuned from a common base), the nature of their parameters (e.g., normalized embeddings vs. raw convolutional weights), and the desired outcome (e.g., robust generalization, task specialization, memory efficiency). As Generative AI models continue to grow in complexity and specialization, the ability to effectively merge their capabilities will become an increasingly vital skill for creating versatile, efficient, and innovative AI systems. Further research in understanding the loss landscape geometry and developing more adaptive merging strategies will continue to expand the utility and power of these techniques.

### 6. References

*   Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: Training sparse, compressible networks. *International Conference on Learning Representations (ICLR)*.
*   Wortsman, M., et al. (2022). Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time. *International Conference on Machine Learning (ICML)*.
*   Zerveas, G., & Barmpalexis, P. (2021). Merging Models with Spherical Linear Interpolation. *arXiv preprint arXiv:2104.09873*.
*   Yadav, A., et al. (2023). DARE: Dynamic Architecture for Residual Ensemble. *arXiv preprint arXiv:2311.03099*.

---
<br>

<a name="türkçe-içerik"></a>
## Model Birleştirme Teknikleri: Doğrusal, SLERP ve DARE

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Model Birleştirmenin Ardındaki Mantık](#2-model-birleştirmenin-ardındaki-mantık)
- [3. Temel Model Birleştirme Teknikleri](#3-temel-model-birleştirme-teknikleri)
  - [3.1. Doğrusal Birleştirme](#31-doğrusal-birleştirme)
  - [3.2. Küresel Doğrusal Enterpolasyon (SLERP)](#32-küresel-doğrusal-enterpolasyon-slerp)
  - [3.3. DARE (Dropout-farkındalıklı Artık Birleştirme)](#33-dare-dropout-farkındalıklı-artık-birleştirme)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Referanslar](#6-referanslar-tr)

### 1. Giriş

Üretken Yapay Zeka (Generative AI) ve derin öğrenmenin hızla gelişen dünyasında, birden fazla önceden eğitilmiş modelden bilgiyi birleştirme yeteneği güçlü bir paradigma olarak ortaya çıkmıştır. **Model birleştirme**, ayrı ayrı modellerin parametrelerini (ağırlıklar ve sapmalar) tek, tutarlı bir modelde sentezlemek için tasarlanmış bir dizi tekniği ifade eder. Bu, bir topluluğun (ensemble) hesaplama yükü olmaksızın, bileşenlerinin faydalı özelliklerini ve yeteneklerini ideal olarak devralmayı amaçlar. Bu yaklaşım, geleneksel model eğitiminin aksine, uzmanlaşmış veya daha sağlam modeller oluşturmak için daha verimli bir yol sunar.

Bu belge, üç önde gelen model birleştirme tekniğini ele almaktadır: **Doğrusal Birleştirme (Linear Merging)**, **Küresel Doğrusal Enterpolasyon (SLERP)** ve **DARE (Dropout-farkındalıklı Artık Birleştirme)**. Her yöntem, model mimarileri ve eğitim geçmişleri hakkında farklı varsayımlarla parametre enterpolasyonuna benzersiz bir yaklaşım sunar ve farklı uygulamalarda belirgin performans özellikleri sergiler. Bu teknikleri anlamak, model dağıtımını optimize etmeyi, bilgi transferini kolaylaştırmayı ve kompozisyon yoluyla yeni model yetenekleri elde etmeyi amaçlayan araştırmacılar ve uygulayıcılar için kritik öneme sahiptir.

### 2. Model Birleştirmenin Ardındaki Mantık

Model birleştirmeye yönelik motivasyon, derin öğrenme içindeki çeşitli pratik ve teorik düşüncelerden kaynaklanmaktadır:

*   **Bilgi Birikimi:** Farklı modeller, benzer veri kümeleri üzerinde eğitilmiş olsalar bile, farklı özellikler öğrenebilir veya verilerin farklı yönlerinde uzmanlaşabilir. Birleştirme, bu farklı bilgiyi toplayarak potansiyel olarak daha genelci veya uzman bir model oluşturabilir.
*   **Parametre Verimliliği:** Birden fazla büyük modeli (örneğin, farklı görevler veya stiller için) sürdürmek ve sunmak yerine, tek bir birleştirilmiş model, özellikle büyük dil modelleri (LLM'ler) ve difüzyon modelleri için bellek ayak izini ve çıkarım gecikmesini azaltabilir.
*   **Felaket Unutmayı Azaltma:** Sürekli öğrenme senaryolarında, birleştirme, daha eski modellerden gelen yetenekleri korurken, ince ayarlı modellerden yeni bilgiyi entegre etmeye yardımcı olabilir ve daha önce edinilmiş becerilerin tamamen üzerine yazılmasını önler.
*   **Mod Bağlantısından Yararlanma:** Ampirik kanıtlar, derin öğrenme modellerindeki kayıp yüzeylerinin genellikle "mod bağlantısı" sergilediğini, yani farklı optimal çözümlerin (modlar) düşük kayıp engellerine sahip basit yollarla bağlanabileceğini göstermektedir. Bu özellik, parametrelerin doğrusal enterpolasyonunu uygulanabilir ve genellikle etkili kılar.
*   **Hibrit Yetenekler Yaratma:** Farklı modaliteler veya görevler için eğitilmiş modelleri (örneğin, görüntü üretiminde bir stil modelini bir içerik modeliyle) birleştirerek, üretken süreçler üzerinde nüanslı kontrol sağlayan yeni hibrit yetenekler açığa çıkarılabilir.
*   **İnce Ayar Optimizasyonu:** Bir temel modeli her spesifik görev için kapsamlı bir şekilde ince ayarlamak yerine, birkaç hafif adaptör modülü veya modeli ince ayarlanabilir ve ardından öğrenilen parametreler temel modele veya birbirleriyle geri birleştirilebilir.

Bu avantajlar, model birleştirmeyi Üretken Yapay Zeka araç kutusunda, özellikle önceden eğitilmiş modellerin sıklıkla uyarlandığı veya birleştirildiği Büyük Dil Modelleri (LLM'ler) ve Stable Diffusion varyantları gibi modeller için değerli bir araç olarak konumlandırır.

### 3. Temel Model Birleştirme Teknikleri

#### 3.1. Doğrusal Birleştirme

**Doğrusal birleştirme**, diğer adıyla **ağırlıklı ortalama**, iki veya daha fazla sinir ağının parametrelerini birleştirmek için düşünülen en basit ve genellikle ilk yaklaşımdır. Parametreleri $W_A$ olan Model A ve parametreleri $W_B$ olan Model B olmak üzere iki model verildiğinde, doğrusal olarak birleştirilmiş bir model $W_{merged}$ bir konveks kombinasyonla elde edilebilir:

$W_{merged} = (1 - \alpha)W_A + \alpha W_B$

Burada, $\alpha$ genellikle 0 ile 1 arasında değişen bir skaler ağırlıktır ve Model B'nin Model A'ya göre katkısını kontrol eder. Bu işlem, ağların tüm karşılık gelen parametreleri (ağırlıklar ve sapmalar) üzerinde eleman bazında gerçekleştirilir. Birden fazla modeli birleştirmek için formül, tüm ağırlıkların toplamının 1'e eşit olduğu ağırlıklı bir toplama genişler.

**Varsayımlar ve Hususlar:**
*   **Aynı Mimariler:** Doğrusal birleştirme, modellerin kesinlikle aynı mimarilere sahip olmasını gerektirir, yani aynı katmanlar, nöron sayıları ve parametre şekilleri.
*   **Paylaşılan Başlatma:** Genellikle, modeller ortak bir temel modelden başlamış veya benzer şekilde başlatılmışlarsa doğrusal birleştirmeye daha yatkındırlar. Bu, "parametre uzaylarını" hizalamaya yardımcı olur ve benzer parametrelerin benzer işlevselliklere karşılık gelmesini sağlar.
*   **Mod Bağlantısı:** Doğrusal birleştirmenin başarısı, büyük ölçüde mod bağlantısının deneysel gözlemine atfedilir. Bireysel modellerin kayıp fonksiyonlarının yeterince düşük kayıp bariyerlerine sahip bir yolla bağlanan optimal noktaları (modları) varsa, bu yol boyunca doğrusal bir enterpolasyon işlevsel bir model verebilir.

**Avantajlar:**
*   **Basitlik:** Uygulaması son derece kolay ve hesaplama açısından ucuzdur.
*   **Etkililik:** Birçok senaryoda, özellikle ortak bir temelden ince ayar yapılmış modeller için şaşırtıcı derecede etkilidir.

**Sınırlamalar:**
*   **Geometrik Bozulma:** Yüksek boyutlu parametre uzaylarında doğrusal enterpolasyon, modeller kayıp yüzeyinde düz bir çizgi ile iyi bağlanmayan farklı "modlar" işgal ediyorsa, ara bölgelerde kötü performans gösteren modellere yol açabilir.
*   **Normalizasyon Kaybı:** Ağırlıklar normalleştirilmişse (örneğin, belirli katmanlarda), doğrusal ortalama bu normalizasyonu bozabilir ve performansı olumsuz etkileyebilir.

#### 3.2. Küresel Doğrusal Enterpolasyon (SLERP)

**Küresel Doğrusal Enterpolasyon (SLERP)**, bir kürenin yüzeyindeki iki nokta arasında enterpolasyon yapmak için tasarlanmış bir yöntemdir. Doğrusal enterpolasyon Öklid uzayında düz bir çizgi boyunca hareket ederken, SLERP bir hiperkürenin yüzeyinde en kısa yol (bir **büyük yay**) boyunca hareket eder. Model birleştirme bağlamında SLERP, ağırlık vektörlerinin yönlülüğü veya göreceli büyüklükleri önemli olduğunda veya parametreler etkili bir şekilde normalize edildiğinde özellikle kullanışlıdır.

İki ağırlık vektörü $W_A$ ve $W_B$ (kavramsal basitlik için birim uzunluğa normalize edilmiş olduğu varsayılır) ve bir enterpolasyon parametresi $\alpha \in [0, 1]$ verildiğinde, SLERP formülü şöyledir:

$SLERP(W_A, W_B, \alpha) = \frac{\sin((1 - \alpha)\theta)}{\sin(\theta)}W_A + \frac{\sin(\alpha\theta)}{\sin(\theta)}W_B$

Burada $\theta$, $W_A$ ve $W_B$ arasındaki yüksek boyutlu uzaydaki açıdır ve $\theta = \arccos(W_A \cdot W_B)$ olarak hesaplanır ($W_A$ ve $W_B$'nin normalize vektörler olduğu varsayılır). Eğer $\sin(\theta)$ sıfırsa (yani, $W_A$ ve $W_B$ aynı doğrultudaysa), SLERP doğrusal enterpolasyona döner.

**Varsayımlar ve Hususlar:**
*   **Normalizasyon:** Orijinal SLERP normalize vektörler varsayarken, derin öğrenmedeki uygulaması genellikle katmanların ağırlık matrislerini veya tüm model parametre vektörlerini yüksek boyutlu noktalar olarak ele almayı içerir. SLERP'in tam faydalarını sağlaması için ön işleme (örneğin, katmanları veya ağırlık gruplarını normalize etme) gerekli olabilir.
*   **Yönsel Önem:** SLERP, vektörlerin göreceli ölçeğini doğrusal enterpolasyondan daha etkili bir şekilde korur, bu da ağırlık vektörünün *yönünün* Öklid büyüklüğünden daha kritik olduğu yerlerde uygun hale getirir.

**Avantajlar:**
*   **Daha Pürüzsüz Enterpolasyon:** Genellikle performans açısından modeller arasında daha pürüzsüz bir geçiş sağlar, çünkü parametre uzayının temel geometrisini doğrusal enterpolasyondan daha doğru bir şekilde, özellikle özellik vektörleri veya gömülülerle uğraşırken, gözetir.
*   **Özellikleri Korur:** Dikkatli bir şekilde uygulandığında belirli istenen özellikleri (örneğin, normalizasyon) korumada daha iyi olabilir.

**Sınırlamalar:**
*   **Hesaplama Maliyeti:** Trigonometrik fonksiyonlar ve nokta çarpım hesaplamaları nedeniyle doğrusal enterpolasyondan daha yoğun hesaplama gerektirir.
*   **Normalizasyon Ön Koşulları:** Parametrelerin nasıl normalize edildiği veya küresel enterpolasyon varsayımlarının ham parametre uzayına gerçekten uygulanıp uygulanmadığı konusunda dikkatli bir değerlendirme gerektirir. Genellikle tüm parametreler ayrım gözetmeksizin değil, modelin belirli kısımlarına (örneğin, gömülüler) uygulanır.

#### 3.3. DARE (Dropout-farkındalıklı Artık Birleştirme)

**DARE (Dropout-farkındalıklı Artık Birleştirme)**, ortak bir önceden eğitilmiş **temel modelden** ince ayar yapılmış modelleri birleştirmek için özel olarak tasarlanmış daha yeni ve gelişmiş bir tekniktir. Birleştirmede kritik bir zorluğu ele alır: ağırlıklardaki küçük farklılıkların bile, farklı öğrenilmiş "özelliklerin" etkileşimi nedeniyle basitçe ortalama alındığında önemli performans düşüşüne yol açtığı fenomeni. DARE, mutlak ağırlıklar yerine temel modelden gelen *artıkları* veya *deltaları* birleştirmeye odaklanır.

Bir temel model $W_{base}$ ve iki ince ayarlı model $W_A$ ve $W_B$ verildiğinde, DARE her modelin temelden öğrendiği *farkları* birleştirir:

$W_{merged} = W_{base} + (1 - \alpha)(W_A - W_{base}) + \alpha(W_B - W_{base})$

Bu formülasyon, esasen artık güncellemelerin doğrusal enterpolasyonu, genellikle kritik bir adımla birleştirilir: artık güncellemelerdeki **küçük ağırlıkların sıfırlanması**. Bu "dropout-farkındalıklı" bileşen, ince ayarın genellikle seyrek güncellemelerle sonuçlandığı ve bu seyrekliliği korumanın (artıktaki küçük ağırlıkları sıfıra ayarlayarak) model performansını, özellikle eğitim sırasında Dropout düzenlemesinin varlığında, sürdürmeye yardımcı olduğu gözleminden ilham almıştır. Özellikle DARE, en küçük büyüklükteki artık ağırlıkların bir yüzdesini düşürmeyi önerir, böylece sadece en önemli değişiklikleri etkili bir şekilde birleştirir.

**Varsayımlar ve Hususlar:**
*   **Ortak Temel Model:** DARE'in etkinliği, $W_A$ ve $W_B$'nin ince ayarlandığı ortak bir temel modelin varlığına büyük ölçüde bağlıdır.
*   **Artık Seyreklik:** İnce ayarın genellikle temel modele göre seyrek değişiklikler getirdiği fikrinden yararlanır.
*   **Dropout Mekanizması:** "Dropout-farkındalıklı" yönü, dropout düzenlemesinin öğrenme sürecini ve parametre dağılımlarını nasıl etkilediğiyle doğrudan ilgilidir.

**Avantajlar:**
*   **İnce Ayarlı Modeller İçin Üstün Performans:** Ortak bir temelden ince ayar yapılmış modelleri birleştirirken, özellikle LLM'ler için basit doğrusal birleştirmeden genellikle önemli ölçüde daha iyi sonuçlar verir.
*   **Temel Bilgiyi Korur:** Artıkları birleştirerek, temel modelin çekirdek bilgisinin korunduğunu ve yalnızca ince ayarlı farklılaştırmaların birleştirildiğini garanti eder.
*   **Ağırlık Çakışmasını Giderir:** Yalnızca önemli değişiklikleri seçici olarak birleştirerek, uyumsuz küçük ağırlık ayarlamalarından kaynaklanan çakışmaları hafifletebilir.

**Sınırlamalar:**
*   **Temel Model Gerektirir:** Modeller farklı başlatmalardan bağımsız olarak eğitildiyse uygulanamaz.
*   **Hiperparametre Ayarı:** Sıfırlanacak ağırlık yüzdesi ("p_weights") dikkatli bir ayar gerektiren kritik bir hiperparametredir.

### 4. Kod Örneği

Aşağıda, doğrusal birleştirmenin iki basitleştirilmiş "model" ağırlık sözlüğüne nasıl uygulanabileceğini gösteren kavramsal bir Python kodu snippet'i bulunmaktadır.

```python
import torch

def linear_merge_models(model_a_state_dict, model_b_state_dict, alpha):
    """
    İki modelin durum sözlükleri arasında doğrusal enterpolasyon yapar.

    Args:
        model_a_state_dict (dict): İlk modelin durum sözlüğü.
        model_b_state_dict (dict): İkinci modelin durum sözlüğü.
        alpha (float): Enterpolasyon faktörü, 0 ile 1 arasında.
                       alpha=0 sadece model_a'yı, alpha=1 sadece model_b'yi kullanır.

    Returns:
        dict: Doğrusal olarak birleştirilmiş modeli temsil eden yeni bir durum sözlüğü.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha 0 ile 1 arasında olmalıdır.")

    merged_state_dict = {}
    for key in model_a_state_dict.keys():
        if key not in model_b_state_dict:
            raise ValueError(f"Anahtar '{key}' model_b_state_dict'te eksik.")

        # Tensorların uyumlu şekil ve tiplere sahip olduğundan emin olun
        if model_a_state_dict[key].shape != model_b_state_dict[key].shape:
            raise ValueError(f"Anahtar '{key}' için şekil uyumsuzluğu.")

        # Doğrusal enterpolasyon yapın
        merged_state_dict[key] = (1 - alpha) * model_a_state_dict[key] + \
                                 alpha * model_b_state_dict[key]
    return merged_state_dict

# Örnek Kullanım:
# İki basit model için durum sözlüklerini simüle edin
# Gerçek bir senaryoda, bunlar .pth veya .safetensors dosyalarından yüklenecekti
model_a_weights = {
    "layer1.weight": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    "layer1.bias": torch.tensor([0.01, 0.02]),
}

model_b_weights = {
    "layer1.weight": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
    "layer1.bias": torch.tensor([0.03, 0.04]),
}

# alpha = 0.5 ile birleştirme (eşit ağırlıklandırma)
merged_weights_0_5 = linear_merge_models(model_a_weights, model_b_weights, 0.5)
print("Birleştirilmiş ağırlıklar (alpha=0.5):")
for key, value in merged_weights_0_5.items():
    print(f"{key}:\n{value}")

# alpha = 0.1 ile birleştirme (çoğunlukla model A)
merged_weights_0_1 = linear_merge_models(model_a_weights, model_b_weights, 0.1)
print("\nBirleştirilmiş ağırlıklar (alpha=0.1):")
for key, value in merged_weights_0_1.items():
    print(f"{key}:\n{value}")

# Bunu gerçek bir modele uygulamak için, önce bir modeli yüklemeniz
# ve ardından model.load_state_dict(merged_state_dict) çağırmanız gerekir.

(Kod örneği bölümünün sonu)
```

### 5. Sonuç

Model birleştirme teknikleri, ayrı sinir ağı modellerinde kapsüllenmiş bilgiyi kullanmak ve birleştirmek için gelişmiş bir yaklaşım temsil eder. Mod bağlantısı kavramından faydalanan basit **Doğrusal Birleştirme**'den, yönlü parametre uzayları için uygun geometrik olarak bilgilendirilmiş **SLERP**'e ve son olarak ortak bir temelden ince ayar yapılmış modelleri birleştirmek için özel **DARE**'e kadar her yöntem benzersiz avantajlar ve uygulanabilirlik sunar.

Birleştirme tekniği seçimi büyük ölçüde özel bağlama bağlıdır: modeller arasındaki ilişki (örn. bağımsız olarak eğitilmiş, ortak bir temelden ince ayar yapılmış), parametrelerinin doğası (örn. normalize edilmiş gömülüler ve ham evrişimsel ağırlıklar) ve istenen sonuç (örn. sağlam genelleme, görev uzmanlığı, bellek verimliliği). Üretken Yapay Zeka modelleri karmaşıklık ve uzmanlık açısından büyümeye devam ettikçe, yeteneklerini etkin bir şekilde birleştirebilme becerisi, çok yönlü, verimli ve yenilikçi yapay zeka sistemleri oluşturmak için giderek daha önemli bir beceri haline gelecektir. Kayıp yüzeyi geometrisini anlamak ve daha uyarlanabilir birleştirme stratejileri geliştirmek için yapılan araştırmalar, bu tekniklerin faydasını ve gücünü genişletmeye devam edecektir.

### 6. Referanslar (TR)

*   Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: Training sparse, compressible networks. *Uluslararası Öğrenme Temsilleri Konferansı (ICLR)*.
*   Wortsman, M., et al. (2022). Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time. *Uluslararası Makine Öğrenimi Konferansı (ICML)*.
*   Zerveas, G., & Barmpalexis, P. (2021). Merging Models with Spherical Linear Interpolation. *arXiv ön baskısı arXiv:2104.09873*.
*   Yadav, A., et al. (2023). DARE: Dynamic Architecture for Residual Ensemble. *arXiv ön baskısı arXiv:2311.03099*.