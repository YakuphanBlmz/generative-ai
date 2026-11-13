# Full Fine-Tuning vs. PEFT: Trade-offs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Full Fine-Tuning](#2-full-fine-tuning)
  - [2.1. Description](#21-description)
  - [2.2. Advantages](#22-advantages)
  - [2.3. Disadvantages](#23-disadvantages)
- [3. Parameter-Efficient Fine-Tuning (PEFT)](#3-parameter-efficient-fine-tuning-peft)
  - [3.1. Description](#31-description)
  - [3.2. Common PEFT Techniques](#32-common-peft-techniques)
  - [3.3. Advantages](#33-advantages)
  - [3.4. Disadvantages](#34-disadvantages)
- [4. Comparative Analysis: Trade-offs](#4-comparative-analysis-trade-offs)
  - [4.1. Performance Potential](#41-performance-potential)
  - [4.2. Computational Resources](#42-computational-resources)
  - [4.3. Storage Requirements](#43-storage-requirements)
  - [4.4. Deployment Complexity](#44-deployment-complexity)
  - [4.5. Catastrophic Forgetting](#45-catastrophic-forgetting)
  - [4.6. Implementation and Development Complexity](#46-implementation-and-development-complexity)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
### 1. Introduction
The advent of **Large Language Models (LLMs)** has revolutionized the field of Artificial Intelligence, demonstrating unprecedented capabilities across a wide array of natural language understanding and generation tasks. However, these foundational models are typically pre-trained on vast, general datasets and may not perform optimally on specific downstream tasks or specialized domains without further adaptation. This process of adapting a pre-trained model to a new task or dataset is commonly referred to as **fine-tuning**.

Two primary paradigms have emerged for fine-tuning LLMs: **Full Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)**. While full fine-tuning involves updating all or a significant portion of the model's parameters, PEFT methods aim to achieve comparable performance by modifying only a small fraction of the total parameters, often by introducing new, smaller parameter sets or by selectively updating existing ones. The choice between these two approaches presents a critical decision point for practitioners and researchers, necessitating a thorough understanding of their respective trade-offs concerning performance, computational overhead, storage, and deployment efficiency. This document provides a comprehensive analysis of these trade-offs, offering insights into when to employ each strategy.

<a name="2-full-fine-tuning"></a>
### 2. Full Fine-Tuning

<a name="21-description"></a>
#### 2.1. Description
**Full fine-tuning**, also known as standard fine-tuning, involves taking a pre-trained LLM and updating **all** of its parameters using a new, task-specific dataset. During this process, the entire neural network architecture, including its vast number of layers and weights, is subjected to further training with a supervised learning objective relevant to the target task (e.g., text classification, summarization, question answering). This approach leverages the extensive knowledge encoded in the pre-trained weights as an initialization point, allowing the model to adapt its internal representations and output layers to the nuances of the new data.

<a name="22-advantages"></a>
#### 2.2. Advantages
1.  **Maximal Performance Potential:** Full fine-tuning generally offers the highest potential for performance on the target task. By allowing all parameters to adjust, the model can deeply ingrain new patterns and adapt its entire knowledge base to the specific data distribution, often leading to superior accuracy, fluency, and task-specific understanding, especially for tasks that significantly diverge from the pre-training distribution.
2.  **Comprehensive Adaptation:** This method enables the model to undergo a complete transformation, potentially learning new features and relationships across all layers. This is particularly beneficial when the target task requires a fundamental shift in the model's understanding or generation capabilities.
3.  **Simplicity of Concept:** Conceptually, full fine-tuning is straightforward: load the model, prepare the data, and train. Many existing frameworks and tutorials are optimized for this traditional approach.

<a name="23-disadvantages"></a>
#### 2.3. Disadvantages
1.  **High Computational Cost:** Full fine-tuning is extremely resource-intensive. It requires significant **GPU memory** to load and process all model parameters and gradients, often necessitating high-end hardware. Training time can also be prohibitively long, even with powerful accelerators.
2.  **Large Storage Requirements:** Each fine-tuned model requires storing the entire set of model weights, which can amount to hundreds of gigabytes or even terabytes for very large LLMs. This becomes a significant challenge when managing multiple fine-tuned versions for different tasks.
3.  **Risk of Catastrophic Forgetting:** When training on a new dataset, especially if it's small or very different from the pre-training data, the model can "forget" some of its previously acquired general knowledge. This phenomenon, known as **catastrophic forgetting**, can degrade performance on broader tasks while improving specific task performance.
4.  **Deployment Challenges:** Deploying multiple fully fine-tuned models can be cumbersome and resource-intensive, as each requires its own dedicated computational resources and memory footprint. Serving many distinct models simultaneously for different user requests becomes economically unfeasible for many organizations.
5.  **Data Requirements:** To effectively prevent catastrophic forgetting and ensure robust adaptation, full fine-tuning often benefits from larger, high-quality task-specific datasets, which can be expensive and time-consuming to curate.

<a name="3-parameter-efficient-fine-tuning-peft"></a>
### 3. Parameter-Efficient Fine-Tuning (PEFT)

<a name="31-description"></a>
#### 3.1. Description
**Parameter-Efficient Fine-Tuning (PEFT)** encompasses a family of techniques designed to adapt pre-trained LLMs to downstream tasks by modifying only a small subset of the model's parameters, or by introducing a minimal number of new, trainable parameters. The core idea is to keep the vast majority of the original pre-trained model parameters frozen, thereby drastically reducing the computational and storage overhead associated with fine-tuning. Despite only updating a fraction of the parameters, PEFT methods have consistently demonstrated the ability to achieve performance comparable to, or very close to, full fine-tuning across a wide range of tasks.

<a name="32-common-peft-techniques"></a>
#### 3.2. Common PEFT Techniques
Several popular PEFT methods exist, each with its own approach to parameter efficiency:
*   **LoRA (Low-Rank Adaptation):** This method injects small, trainable rank decomposition matrices into each layer of the pre-trained model. Instead of fine-tuning the original weight matrices, only these much smaller rank decomposition matrices are optimized. When deployed, these matrices are added to the frozen pre-trained weights.
*   **Prefix-Tuning:** It involves prepending a small, trainable sequence of continuous **"prefix"** vectors to the input embeddings at each layer of the transformer model. The pre-trained model weights remain frozen, and only these prefix vectors are updated.
*   **Prompt-Tuning:** Similar to prefix-tuning but simpler, this technique learns a soft prompt (a sequence of trainable tokens) that is prepended to the input text, allowing the model to condition its output without modifying its internal weights.
*   **Adapter-based Methods:** These methods insert small, task-specific neural network modules (adapters) between layers of the pre-trained model. Only the parameters within these adapter modules are trained, while the main model weights are frozen.

<a name="33-advantages"></a>
#### 3.3. Advantages
1.  **Significantly Reduced Computational Cost:** PEFT drastically cuts down on GPU memory usage and training time. Since fewer parameters are updated, the memory required for gradients is minimal, making it feasible to fine-tune large models on consumer-grade GPUs or with limited resources.
2.  **Minimal Storage Footprint:** PEFT methods store only the small adapter weights or parameter deltas, not the entire model. For example, a LoRA adapter might be a few megabytes, compared to hundreds of gigabytes for the full model. This dramatically eases storage and version control.
3.  **Mitigation of Catastrophic Forgetting:** By keeping the majority of the pre-trained weights frozen, PEFT methods are inherently less prone to catastrophic forgetting. The core knowledge of the LLM remains intact, while adaptation is handled by the newly introduced, task-specific parameters.
4.  **Efficient Deployment and Multi-task Serving:** PEFT enables a "base model + adapter" architecture. A single large pre-trained model can be loaded into memory once, and different small adapter modules can be swapped in and out quickly to serve various tasks. This greatly improves resource utilization and scalability for multi-task deployments.
5.  **Faster Experimentation:** The reduced training time and resource requirements allow for quicker iteration and experimentation with different hyperparameters, datasets, and PEFT configurations.

<a name="33-disadvantages"></a>
#### 3.4. Disadvantages
1.  **Potential Performance Ceiling:** While often achieving comparable results, PEFT methods might not always reach the absolute peak performance of full fine-tuning for highly complex tasks or when the target domain is extremely divergent from the pre-training data. The constrained parameter space limits the extent of adaptation.
2.  **Increased Implementation Complexity (initially):** Implementing PEFT methods requires understanding specific techniques (LoRA, Prefix-Tuning, etc.) and often relies on specialized libraries (e.g., Hugging Face's PEFT library). This can introduce an initial learning curve compared to standard fine-tuning.
3.  **Hyperparameter Tuning:** PEFT methods introduce new hyperparameters (e.g., LoRA rank, alpha, adapter placement) that need careful tuning to optimize performance, adding another layer of complexity to the optimization process.

<a name="4-comparative-analysis-trade-offs"></a>
### 4. Comparative Analysis: Trade-offs

The decision between full fine-tuning and PEFT hinges on a careful evaluation of several key trade-offs:

<a name="41-performance-potential"></a>
#### 4.1. Performance Potential
*   **Full Fine-Tuning:** Theoretically offers the highest performance ceiling as it allows for the most extensive adaptation. Ideal when maximum possible accuracy is paramount and resource constraints are not a major concern.
*   **PEFT:** Consistently achieves near state-of-the-art performance, often within a few percentage points of full fine-tuning. For most practical applications, the slight performance gap is an acceptable trade-off for the significant resource savings.

<a name="42-computational Resources"></a>
#### 4.2. Computational Resources
*   **Full Fine-Tuning:** Demands substantial GPU memory, VRAM, and processing power. Typically requires multiple high-end GPUs (e.g., A100s, H100s) for large models.
*   **PEFT:** Dramatically reduces memory and computational requirements. Can often be run on single consumer-grade GPUs (e.g., RTX 3090, 4090) or even CPU for smaller models or specific methods. This makes advanced LLM fine-tuning accessible to a much broader audience.

<a name="43-storage-requirements"></a>
#### 4.3. Storage Requirements
*   **Full Fine-Tuning:** Requires storing the entire model checkpoint, which can be hundreds of gigabytes. Managing multiple versions can quickly deplete storage.
*   **PEFT:** Stores only the small, trainable adapter weights or delta parameters (typically in the order of megabytes). This enables efficient storage, versioning, and sharing of task-specific adaptations.

<a name="44-deployment-complexity"></a>
#### 4.4. Deployment Complexity
*   **Full Fine-Tuning:** Each fine-tuned model is a standalone entity, requiring its full loading and memory footprint. Serving many distinct tasks simultaneously becomes expensive.
*   **PEFT:** Facilitates modular deployment. A single base LLM can be loaded once, and different lightweight adapter modules can be dynamically swapped in for various tasks, leading to much more efficient resource utilization and easier multi-task inference.

<a name="45-catastrophic-forgetting"></a>
#### 4.5. Catastrophic Forgetting
*   **Full Fine-Tuning:** More susceptible to forgetting general knowledge if fine-tuned on small or highly specialized datasets, especially without careful regularization or combined training strategies.
*   **PEFT:** Inherently less prone to catastrophic forgetting because the bulk of the pre-trained model weights remains frozen, preserving its foundational knowledge.

<a name="46-implementation-and-development-complexity"></a>
#### 4.6. Implementation and Development Complexity
*   **Full Fine-Tuning:** Conceptually simpler; widely supported by standard training loops and frameworks.
*   **PEFT:** Requires understanding specific techniques and potentially using specialized libraries (e.g., Hugging Face PEFT). This can add an initial learning curve, but once understood, it streamlines development due to faster iteration cycles.

<a name="5-code-example"></a>
### 5. Code Example
This example illustrates the conceptual difference between loading a full model (for full fine-tuning) and loading a base model with a PEFT adapter (LoRA).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. Example: Loading a base model for potential Full Fine-Tuning ---
# This would load all model parameters.
print("Loading a full model for potential full fine-tuning...")
model_name = "mistralai/Mistral-7B-v0.1"
try:
    # In a full fine-tuning scenario, you'd load the model normally.
    # For large models, this will require significant VRAM.
    full_model = AutoModelForCausalLM.from_pretrained(model_name)
    # The tokenizer is common to both approaches
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Full model loaded. Total parameters: {full_model.num_parameters()}")
    # To perform full fine-tuning, you would then train `full_model`
    # without any specific PEFT wrappers.
except Exception as e:
    print(f"Could not load full model (requires high VRAM): {e}")
    print("Proceeding with PEFT example which is more resource-efficient.")


# --- 2. Example: Loading a model with LoRA for Parameter-Efficient Fine-Tuning ---
print("\nLoading a base model and applying LoRA for PEFT...")
# Load the base model in 4-bit precision for memory efficiency
# This is common in PEFT to further reduce memory footprint
base_model_peft = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True, # Significantly reduces VRAM usage
    device_map="auto"
)
tokenizer_peft = AutoTokenizer.from_pretrained(model_name)

# Prepare model for k-bit training (required for 4-bit)
base_model_peft = prepare_model_for_kbit_training(base_model_peft)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA attention dimension
    lora_alpha=16, # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Modules to apply LoRA to
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Whether to train bias parameters
    task_type="CAUSAL_LM", # Task type
)

# Apply LoRA to the base model
peft_model = get_peft_model(base_model_peft, lora_config)

print(f"PEFT model (LoRA) loaded. Trainable parameters: {peft_model.print_trainable_parameters()}")
# The output `trainable params: 4,194,304 || all params: 7,241,730,176 || trainable%: 0.05791696409405625`
# clearly shows that only a tiny fraction of parameters are trainable.

# To train with PEFT, you would then train `peft_model`.
# After training, you would save only the PEFT adapter weights:
# peft_model.save_pretrained("my_lora_adapter")

(End of code example section)
```

<a name="6-conclusion"></a>
### 6. Conclusion
The choice between full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) is a strategic one, dictated by a combination of available computational resources, performance objectives, storage considerations, and deployment strategies.

**Full fine-tuning** remains the gold standard for achieving the absolute highest performance, particularly when the target task demands extensive adaptation and computational resources are abundant. However, its significant demands on memory, processing power, and storage, coupled with the risk of catastrophic forgetting, make it an often impractical choice for many.

**PEFT methods**, conversely, offer a compelling alternative. By focusing on adapting only a small fraction of a model's parameters, they dramatically reduce resource requirements, minimize storage footprint, mitigate catastrophic forgetting, and streamline deployment in multi-task scenarios. While they might occasionally exhibit a marginal performance gap compared to full fine-tuning, this trade-off is often negligible for most practical applications, making PEFT the more **accessible, scalable, and environmentally friendly** option for adapting modern LLMs.

In an era where LLMs continue to grow in size and complexity, PEFT techniques are becoming indispensable tools, democratizing access to powerful AI models and enabling more efficient and sustainable development of specialized AI applications. Practitioners are increasingly adopting PEFT as the default strategy, reserving full fine-tuning only for niche cases where absolute peak performance outweighs all other considerations.

---
<br>

<a name="türkçe-içerik"></a>
## Tam İnce Ayar ve PEFT: Takaslar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Tam İnce Ayar](#2-tam-ince-ayar)
  - [2.1. Tanım](#21-tanım)
  - [2.2. Avantajlar](#22-avantajlar)
  - [2.3. Dezavantajlar](#23-dezavantajlar)
- [3. Parametre Verimli İnce Ayar (PEFT)](#3-parametre-verimli-İnce-ayar-peft)
  - [3.1. Tanım](#31-tanım)
  - [3.2. Yaygın PEFT Teknikleri](#32-yaygın-peft-teknikleri)
  - [3.3. Avantajlar](#33-avantajlar)
  - [3.4. Dezavantajlar](#34-dezavantajlar)
- [4. Karşılaştırmalı Analiz: Takaslar](#4-karşılaştırmalı-analiz-takaslar)
  - [4.1. Performans Potansiyeli](#41-performans-potansiyeli)
  - [4.2. Hesaplama Kaynakları](#42-hesaplama-kaynakları)
  - [4.3. Depolama Gereksinimleri](#43-depolama-gereksinimleri)
  - [4.4. Dağıtım Karmaşıklığı](#44-dağıtım-karmaşıklığı)
  - [4.5. Felaket Unutkanlığı](#45-felaket-unutkanlığı)
  - [4.6. Uygulama ve Geliştirme Karmaşıklığı](#46-uygulama-ve-geliştirme-karmaşıklığı)
- [5. Kod Örneği](#5-kod-Örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
### 1. Giriş
**Büyük Dil Modellerinin (LLM'ler)** ortaya çıkışı, Doğal Dil Anlama ve Üretimi alanlarında benzeri görülmemiş yetenekler sergileyerek Yapay Zeka alanında devrim yarattı. Ancak, bu temel modeller genellikle geniş, genel veri kümeleri üzerinde önceden eğitilmiştir ve özel alt görevlerde veya uzmanlık alanlarında ek adaptasyon olmadan en uygun performansı gösteremeyebilir. Önceden eğitilmiş bir modelin yeni bir göreve veya veri kümesine uyarlanması sürecine yaygın olarak **ince ayar (fine-tuning)** denir.

LLM'leri ince ayarlamak için iki ana paradigma ortaya çıkmıştır: **Tam İnce Ayar (Full Fine-Tuning)** ve **Parametre Verimli İnce Ayar (PEFT)**. Tam ince ayar, modelin tüm veya önemli bir kısmının parametrelerini güncellemeyi içerirken, PEFT yöntemleri, toplam parametrelerin yalnızca küçük bir kısmını değiştirerek veya yeni, daha küçük parametre kümeleri ekleyerek ya da mevcut olanları seçici olarak güncelleyerek benzer performansı elde etmeyi amaçlar. Bu iki yaklaşım arasındaki seçim, uygulayıcılar ve araştırmacılar için kritik bir karar noktası sunar ve performans, hesaplama yükü, depolama ve dağıtım verimliliği ile ilgili karşılıklı ödünleşimlerin kapsamlı bir şekilde anlaşılmasını gerektirir. Bu belge, bu ödünleşimlerin kapsamlı bir analizini sunarak, her stratejinin ne zaman kullanılması gerektiğine dair içgörüler sağlar.

<a name="2-tam-ince-ayar"></a>
### 2. Tam İnce Ayar

<a name="21-tanım"></a>
#### 2.1. Tanım
Standart ince ayar olarak da bilinen **tam ince ayar**, önceden eğitilmiş bir LLM'yi alıp, tüm parametrelerini yeni, göreve özel bir veri kümesi kullanarak güncellemeyi içerir. Bu süreçte, çok sayıda katmanı ve ağırlığı dahil olmak üzere tüm sinir ağı mimarisi, hedef görevle (örneğin, metin sınıflandırma, özetleme, soru yanıtlama) ilgili denetimli bir öğrenme hedefiyle daha fazla eğitime tabi tutulur. Bu yaklaşım, önceden eğitilmiş ağırlıklarda kodlanmış kapsamlı bilgiyi bir başlangıç noktası olarak kullanarak, modelin dahili temsillerini ve çıktı katmanlarını yeni verilerin nüanslarına uyarlamasını sağlar.

<a name="22-avantajlar"></a>
#### 2.2. Avantajlar
1.  **Maksimum Performans Potansiyeli:** Tam ince ayar genellikle hedef görevde en yüksek performans potansiyelini sunar. Tüm parametrelerin ayarlanmasına izin vererek, model yeni desenleri derinlemesine işleyebilir ve tüm bilgi tabanını belirli veri dağıtımına uyarlayabilir; bu da özellikle ön eğitim dağılımından önemli ölçüde sapan görevler için üstün doğruluk, akıcılık ve göreve özel anlayışa yol açar.
2.  **Kapsamlı Adaptasyon:** Bu yöntem, modelin tamamen dönüşmesini sağlayarak tüm katmanlarda yeni özellikler ve ilişkiler öğrenme potansiyeli sunar. Bu, hedef görevin modelin anlayışında veya üretim yeteneklerinde temel bir değişiklik gerektirdiği durumlarda özellikle faydalıdır.
3.  **Kavramsal Basitlik:** Kavramsal olarak, tam ince ayar basittir: modeli yükle, veriyi hazırla ve eğit. Birçok mevcut çerçeve ve eğitim materyali bu geleneksel yaklaşım için optimize edilmiştir.

<a name="23-dezavantajlar"></a>
#### 2.3. Dezavantajlar
1.  **Yüksek Hesaplama Maliyeti:** Tam ince ayar aşırı derecede kaynak yoğundur. Tüm model parametrelerini ve gradyanlarını yüklemek ve işlemek için önemli **GPU belleği** gerektirir ve genellikle yüksek performanslı donanım gerektirir. Eğitim süresi, güçlü hızlandırıcılarla bile aşırı uzun olabilir.
2.  **Büyük Depolama Gereksinimleri:** Her ince ayarlı model, tüm model ağırlık setini saklamayı gerektirir; bu da çok büyük LLM'ler için yüzlerce gigabayt veya hatta terabaytları bulabilir. Bu, farklı görevler için birden fazla ince ayarlı sürümü yönetirken önemli bir zorluk haline gelir.
3.  **Felaket Unutkanlığı Riski:** Yeni bir veri kümesi üzerinde eğitim yaparken, özellikle küçükse veya ön eğitim verilerinden çok farklıysa, model daha önce edindiği bazı genel bilgileri "unutabilir". **Felaket unutkanlığı** olarak bilinen bu fenomen, belirli görev performansını iyileştirirken daha geniş görevlerde performansı düşürebilir.
4.  **Dağıtım Zorlukları:** Birden fazla tam ince ayarlı modeli dağıtmak hantal ve kaynak yoğun olabilir, zira her biri kendi özel hesaplama kaynaklarını ve bellek alanını gerektirir. Farklı kullanıcı istekleri için aynı anda birçok farklı modeli sunmak, birçok kuruluş için ekonomik olarak imkansız hale gelir.
5.  **Veri Gereksinimleri:** Felaket unutkanlığını etkili bir şekilde önlemek ve sağlam adaptasyon sağlamak için, tam ince ayar genellikle daha büyük, yüksek kaliteli göreve özel veri kümelerinden yararlanır; bunların kürasyonu pahalı ve zaman alıcı olabilir.

<a name="3-parametre-verimli-İnce-ayar-peft"></a>
### 3. Parametre Verimli İnce Ayar (PEFT)

<a name="31-tanım"></a>
#### 3.1. Tanım
**Parametre Verimli İnce Ayar (PEFT)**, önceden eğitilmiş LLM'leri alt görevlere uyarlamak için tasarlanmış bir teknik ailesini kapsar. Bu, modelin parametrelerinin yalnızca küçük bir alt kümesini değiştirerek veya minimal sayıda yeni, eğitilebilir parametreler ekleyerek yapılır. Temel fikir, önceden eğitilmiş orijinal model parametrelerinin büyük çoğunluğunu dondurulmuş tutmak ve böylece ince ayarla ilişkili hesaplama ve depolama yükünü dramatik bir şekilde azaltmaktır. Parametrelerin sadece bir kısmını güncellemesine rağmen, PEFT yöntemleri, geniş bir görev yelpazesinde tam ince ayara benzer veya çok yakın performans elde etme yeteneğini tutarlı bir şekilde göstermiştir.

<a name="32-yaygın-peft-teknikleri"></a>
#### 3.2. Yaygın PEFT Teknikleri
Birçok popüler PEFT yöntemi mevcuttur ve her birinin parametre verimliliğine kendi yaklaşımı vardır:
*   **LoRA (Low-Rank Adaptation):** Bu yöntem, önceden eğitilmiş modelin her katmanına küçük, eğitilebilir düşük ranklı ayrıştırma matrisleri enjekte eder. Orijinal ağırlık matrislerini ince ayarlamak yerine, yalnızca bu çok daha küçük ranklı ayrıştırma matrisleri optimize edilir. Dağıtıldığında, bu matrisler dondurulmuş önceden eğitilmiş ağırlıklara eklenir.
*   **Prefix-Tuning:** Transformer modelinin her katmanındaki girdi gömülü temsillerine küçük, eğitilebilir sürekli **"ön ek" (prefix)** vektör dizisi eklemeyi içerir. Önceden eğitilmiş model ağırlıkları dondurulmuş kalır ve yalnızca bu ön ek vektörleri güncellenir.
*   **Prompt-Tuning:** Prefix-tuning'e benzer ancak daha basittir; bu teknik, girdi metnine eklenen yumuşak bir prompt (eğitilebilir belirteçlerden oluşan bir dizi) öğrenir ve modelin dahili ağırlıklarını değiştirmeden çıktısını koşullandırmasını sağlar.
*   **Adaptör Tabanlı Yöntemler:** Bu yöntemler, önceden eğitilmiş modelin katmanları arasına küçük, göreve özel sinir ağı modülleri (adaptörler) yerleştirir. Ana model ağırlıkları dondurulmuş kalırken, yalnızca bu adaptör modüllerindeki parametreler eğitilir.

<a name="33-avantajlar"></a>
#### 3.3. Avantajlar
1.  **Önemli Ölçüde Azaltılmış Hesaplama Maliyeti:** PEFT, GPU bellek kullanımını ve eğitim süresini dramatik bir şekilde azaltır. Daha az parametre güncellendiği için, gradyanlar için gereken bellek minimaldir; bu da büyük modelleri tüketici sınıfı GPU'larda veya sınırlı kaynaklarla ince ayarlamayı mümkün kılar.
2.  **Minimum Depolama Alanı:** PEFT yöntemleri yalnızca küçük adaptör ağırlıklarını veya parametre deltalarını saklar, tüm modeli değil. Örneğin, bir LoRA adaptörü, tam model için yüzlerce gigabaytın aksine, birkaç megabayt olabilir. Bu, depolama ve sürüm kontrolünü büyük ölçüde kolaylaştırır.
3.  **Felaket Unutkanlığının Azaltılması:** Önceden eğitilmiş ağırlıkların çoğunu dondurulmuş tutarak, PEFT yöntemleri doğal olarak felaket unutkanlığına daha az eğilimlidir. LLM'nin temel bilgisi bozulmadan kalırken, adaptasyon yeni eklenen, göreve özel parametreler tarafından yönetilir.
4.  **Verimli Dağıtım ve Çoklu Görev Sunumu:** PEFT, "temel model + adaptör" mimarisi sağlar. Tek bir büyük önceden eğitilmiş model belleğe bir kez yüklenebilir ve farklı küçük adaptör modülleri çeşitli görevler için hızlı bir şekilde değiştirilebilir. Bu, çoklu görev dağıtımları için kaynak kullanımını ve ölçeklenebilirliği büyük ölçüde iyileştirir.
5.  **Daha Hızlı Deneme:** Azalan eğitim süresi ve kaynak gereksinimleri, farklı hiperparametreler, veri kümeleri ve PEFT yapılandırmalarıyla daha hızlı yineleme ve deneme yapılmasına olanak tanır.

<a name="34-dezavantajlar"></a>
#### 3.4. Dezavantajlar
1.  **Potansiyel Performans Tavanı:** Genellikle karşılaştırılabilir sonuçlar elde etse de, PEFT yöntemleri, çok karmaşık görevler veya hedef alanın ön eğitim verilerinden son derece farklı olduğu durumlarda tam ince ayarın mutlak zirve performansına her zaman ulaşamayabilir. Kısıtlı parametre alanı, adaptasyonun kapsamını sınırlar.
2.  **Artan Uygulama Karmaşıklığı (başlangıçta):** PEFT yöntemlerini uygulamak, belirli teknikleri (LoRA, Prefix-Tuning vb.) anlamayı gerektirir ve genellikle özel kitaplıklara (örneğin, Hugging Face'in PEFT kütüphanesi) dayanır. Bu, standart ince ayara kıyasla başlangıçta bir öğrenme eğrisi oluşturabilir.
3.  **Hiperparametre Ayarı:** PEFT yöntemleri, performansı optimize etmek için dikkatli ayar gerektiren yeni hiperparametreler (örneğin, LoRA rankı, alfa, adaptör yerleşimi) getirir; bu da optimizasyon sürecine başka bir karmaşıklık katmanı ekler.

<a name="4-karşılaştırmalı-analiz-takaslar"></a>
### 4. Karşılaştırmalı Analiz: Takaslar

Tam ince ayar ve PEFT arasındaki karar, çeşitli önemli takasların dikkatli bir şekilde değerlendirilmesine bağlıdır:

<a name="41-performans-potansiyeli"></a>
#### 4.1. Performans Potansiyeli
*   **Tam İnce Ayar:** En kapsamlı adaptasyona izin verdiği için teorik olarak en yüksek performans tavanını sunar. Maksimum olası doğruluğun öncelikli olduğu ve kaynak kısıtlamalarının büyük bir sorun olmadığı durumlarda idealdir.
*   **PEFT:** Genellikle tam ince ayara yakın, genellikle birkaç yüzde puanlık bir farkla, son teknolojiye yakın performans elde eder. Çoğu pratik uygulama için, küçük performans farkı, önemli kaynak tasarrufu için kabul edilebilir bir takastır.

<a name="42-hesaplama-kaynakları"></a>
#### 4.2. Hesaplama Kaynakları
*   **Tam İnce Ayar:** Önemli GPU belleği, VRAM ve işlem gücü gerektirir. Genellikle büyük modeller için birden fazla üst düzey GPU (örneğin, A100'ler, H100'ler) gerektirir.
*   **PEFT:** Bellek ve hesaplama gereksinimlerini dramatik bir şekilde azaltır. Genellikle tek bir tüketici sınıfı GPU'da (örneğin, RTX 3090, 4090) veya daha küçük modeller veya belirli yöntemler için CPU'da bile çalıştırılabilir. Bu, gelişmiş LLM ince ayarını çok daha geniş bir kitleye erişilebilir kılar.

<a name="43-depolama-gereksinimleri"></a>
#### 4.3. Depolama Gereksinimleri
*   **Tam İnce Ayar:** Tüm model kontrol noktasını saklamayı gerektirir, bu da yüzlerce gigabayt olabilir. Birden fazla sürümü yönetmek depolama alanını hızla tüketebilir.
*   **PEFT:** Yalnızca küçük, eğitilebilir adaptör ağırlıklarını veya delta parametrelerini saklar (genellikle megabaytlar mertebesinde). Bu, göreve özel adaptasyonların verimli bir şekilde depolanmasını, sürümlemesini ve paylaşılmasını sağlar.

<a name="44-dağıtım-karmaşıklığı"></a>
#### 4.4. Dağıtım Karmaşıklığı
*   **Tam İnce Ayar:** Her ince ayarlı model, tam yükleme ve bellek alanı gerektiren bağımsız bir varlıktır. Birçok farklı görevi aynı anda sunmak pahalı hale gelir.
*   **PEFT:** Modüler dağıtımı kolaylaştırır. Tek bir temel LLM bir kez yüklenebilir ve farklı hafif adaptör modülleri çeşitli görevler için dinamik olarak değiştirilebilir, bu da çok daha verimli kaynak kullanımı ve daha kolay çoklu görev çıkarımına yol açar.

<a name="45-felaket-unutkanlığı"></a>
#### 4.5. Felaket Unutkanlığı
*   **Tam İnce Ayar:** Küçük veya yüksek derecede uzmanlaşmış veri kümeleri üzerinde ince ayar yapıldığında, özellikle dikkatli düzenlileştirme veya birleşik eğitim stratejileri olmadan genel bilgileri unutmaya daha yatkındır.
*   **PEFT:** Önceden eğitilmiş model ağırlıklarının büyük çoğunluğu dondurulmuş kaldığı için, felaket unutkanlığına doğal olarak daha az eğilimlidir ve temel bilgisini korur.

<a name="46-uygulama-ve-geliştirme-karmaşıklığı"></a>
#### 4.6. Uygulama ve Geliştirme Karmaşıklığı
*   **Tam İnce Ayar:** Kavramsal olarak daha basittir; standart eğitim döngüleri ve çerçeveleri tarafından yaygın olarak desteklenir.
*   **PEFT:** Belirli teknikleri anlamayı ve potansiyel olarak özel kitaplıkları (örneğin, Hugging Face PEFT) kullanmayı gerektirir. Bu, başlangıçta bir öğrenme eğrisi ekleyebilir, ancak anlaşıldığında, daha hızlı yineleme döngüleri nedeniyle geliştirmeyi kolaylaştırır.

<a name="5-kod-Örneği"></a>
### 5. Kod Örneği
Bu örnek, tam bir modelin (tam ince ayar için) yüklenmesi ile bir temel modelin PEFT adaptörü (LoRA) ile yüklenmesi arasındaki kavramsal farkı göstermektedir.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. Örnek: Potansiyel Tam İnce Ayar için bir temel modelin yüklenmesi ---
# Bu, tüm model parametrelerini yükleyecektir.
print("Potansiyel tam ince ayar için tam model yükleniyor...")
model_name = "mistralai/Mistral-7B-v0.1"
try:
    # Tam ince ayar senaryosunda, modeli normal şekilde yüklersiniz.
    # Büyük modeller için bu, önemli VRAM gerektirecektir.
    full_model = AutoModelForCausalLM.from_pretrained(model_name)
    # Tokenizer her iki yaklaşım için de ortaktır
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tam model yüklendi. Toplam parametreler: {full_model.num_parameters()}")
    # Tam ince ayar yapmak için, `full_model`'i
    # herhangi bir özel PEFT sarmalayıcı olmadan eğitmeniz gerekir.
except Exception as e:
    print(f"Tam model yüklenemedi (yüksek VRAM gerektirir): {e}")
    print("Daha kaynak verimli olan PEFT örneği ile devam ediliyor.")


# --- 2. Örnek: Parametre Verimli İnce Ayar için LoRA ile bir modelin yüklenmesi ---
print("\nTemel model yükleniyor ve PEFT için LoRA uygulanıyor...")
# Bellek verimliliği için temel modeli 4-bit hassasiyetle yükleyin
# Bu, PEFT'te bellek ayak izini daha da azaltmak için yaygındır
base_model_peft = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True, # VRAM kullanımını önemli ölçüde azaltır
    device_map="auto"
)
tokenizer_peft = AutoTokenizer.from_pretrained(model_name)

# Modeli k-bit eğitim için hazırlayın (4-bit için gereklidir)
base_model_peft = prepare_model_for_kbit_training(base_model_peft)

# LoRA yapılandırmasını tanımlayın
lora_config = LoraConfig(
    r=8,  # LoRA dikkat boyutu
    lora_alpha=16, # LoRA ölçeklendirme için alfa parametresi
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # LoRA'nın uygulanacağı modüller
    lora_dropout=0.05, # LoRA katmanları için dropout olasılığı
    bias="none", # Bias parametrelerinin eğitilip eğitilmeyeceği
    task_type="CAUSAL_LM", # Görev türü
)

# LoRA'yı temel modele uygulayın
peft_model = get_peft_model(base_model_peft, lora_config)

print(f"PEFT modeli (LoRA) yüklendi. Eğitilebilir parametreler: {peft_model.print_trainable_parameters()}")
# Çıktı `trainable params: 4,194,304 || all params: 7,241,730,176 || trainable%: 0.05791696409405625`
# parametrelerin sadece küçük bir kısmının eğitilebilir olduğunu açıkça göstermektedir.

# PEFT ile eğitim yapmak için, `peft_model`'i eğitmeniz gerekir.
# Eğitimden sonra sadece PEFT adaptör ağırlıklarını kaydedersiniz:
# peft_model.save_pretrained("my_lora_adapter")

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
### 6. Sonuç
Tam ince ayar ve Parametre Verimli İnce Ayar (PEFT) arasındaki seçim, mevcut hesaplama kaynakları, performans hedefleri, depolama konuları ve dağıtım stratejilerinin bir kombinasyonu tarafından belirlenen stratejik bir karardır.

**Tam ince ayar**, özellikle hedef görev kapsamlı adaptasyon gerektirdiğinde ve hesaplama kaynakları bol olduğunda, mutlak en yüksek performansı elde etmek için altın standart olmaya devam etmektedir. Ancak, bellek, işlem gücü ve depolama üzerindeki önemli talepleri, felaket unutkanlığı riskiyle birleştiğinde, birçok durum için genellikle pratik olmayan bir seçenek haline gelmektedir.

**PEFT yöntemleri** ise cazip bir alternatif sunar. Bir modelin parametrelerinin yalnızca küçük bir kısmını uyarlamaya odaklanarak, kaynak gereksinimlerini dramatik bir şekilde azaltır, depolama alanını en aza indirir, felaket unutkanlığını azaltır ve çok görevli senaryolarda dağıtımı kolaylaştırır. Tam ince ayara kıyasla ara sıra marjinal bir performans farkı gösterebilseler de, bu takas çoğu pratik uygulama için genellikle ihmal edilebilir düzeydedir; bu da PEFT'i modern LLM'leri uyarlamak için daha **erişilebilir, ölçeklenebilir ve çevre dostu** bir seçenek haline getirir.

LLM'lerin boyut ve karmaşıklık açısından büyümeye devam ettiği bir çağda, PEFT teknikleri vazgeçilmez araçlar haline gelmekte, güçlü yapay zeka modellerine erişimi demokratikleştirmekte ve uzmanlaşmış yapay zeka uygulamalarının daha verimli ve sürdürülebilir gelişimini sağlamaktadır. Uygulayıcılar, tam ince ayarı yalnızca mutlak zirve performansının diğer tüm hususları geride bıraktığı niş durumlar için saklayarak, PEFT'i varsayılan strateji olarak giderek daha fazla benimsemektedir.

