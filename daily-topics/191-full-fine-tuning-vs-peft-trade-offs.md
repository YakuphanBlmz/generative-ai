# Full Fine-Tuning vs. PEFT: Trade-offs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Full Fine-Tuning](#2-full-fine-tuning)
  - [2.1 Mechanism and Approach](#21-mechanism-and-approach)
  - [2.2 Advantages](#22-advantages)
  - [2.3 Disadvantages](#23-disadvantages)
- [3. Parameter-Efficient Fine-Tuning (PEFT)](#3-parameter-efficient-fine-tuning-peft)
  - [3.1 Mechanism and Approach](#31-mechanism-and-approach)
  - [3.2 Key PEFT Techniques](#32-key-peft-techniques)
  - [3.3 Advantages](#33-advantages)
  - [3.4 Disadvantages](#34-disadvantages)
- [4. Key Trade-offs: Full Fine-Tuning vs. PEFT](#4-key-trade-offs-full-fine-tuning-vs-peft)
  - [4.1 Computational Resources (Memory & Processing Power)](#41-computational-resources-memory--processing-power)
  - [4.2 Storage Overhead](#42-storage-overhead)
  - [4.3 Performance Ceiling and Task Specificity](#43-performance-ceiling-and-task-specificity)
  - [4.4 Training Speed and Iteration Time](#44-training-speed-and-iteration-time)
  - [4.5 Implementation Complexity and Ease of Use](#45-implementation-complexity-and-ease-of-use)
  - [4.6 Generalization and Catastrophic Forgetting](#46-generalization-and-catastrophic-forgetting)
  - [4.7 Scalability and Multi-Task Adaptation](#47-scalability-and-multi-task-adaptation)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction

The advent of **Large Language Models (LLMs)** and other large-scale pre-trained models has revolutionized various domains, from natural language processing to computer vision. These models, often trained on vast and diverse datasets, possess remarkable general-purpose capabilities. However, to effectively deploy them for specific downstream tasks or domain-specific applications, they typically require **fine-tuning**. Fine-tuning adapts a pre-trained model to a new dataset, aligning its behavior with the specific requirements of the target task.

Historically, the standard approach has been **Full Fine-Tuning**, which involves updating all, or almost all, parameters of the pre-trained model. While effective, this method incurs significant computational costs and resource demands. In response to these challenges, a new paradigm known as **Parameter-Efficient Fine-Tuning (PEFT)** has emerged. PEFT techniques aim to achieve comparable performance to full fine-tuning while significantly reducing the number of trainable parameters, thereby decreasing computational overhead and storage requirements.

This document provides a comprehensive analysis of the trade-offs between Full Fine-Tuning and PEFT methods. We will delve into their respective mechanisms, advantages, disadvantages, and critically compare them across various dimensions such as resource consumption, performance, scalability, and implementation complexity. Understanding these trade-offs is crucial for making informed decisions when adapting large pre-trained models to specific applications.

### 2. Full Fine-Tuning

Full fine-tuning, often considered the traditional approach, involves retraining the entire pre-trained model on a new, task-specific dataset.

#### 2.1 Mechanism and Approach

In **Full Fine-Tuning**, the pre-trained model's architecture remains fixed, but all of its weights (parameters) are unfrozen and updated during the training process using a small learning rate. The model is trained on a labeled dataset specific to the target task (e.g., sentiment analysis, text summarization, image classification on a new domain). The **loss function** is computed based on the model's predictions on this new dataset, and **gradient descent** (or its variants like Adam) is used to adjust all parameters to minimize this loss. The process effectively shifts the model's learned representations from its general pre-training distribution to the specific distribution of the fine-tuning task.

#### 2.2 Advantages

*   **Potentially Higher Performance Ceiling:** By allowing all parameters to adapt, full fine-tuning has the potential to achieve the highest possible performance on the target task, especially when the target task significantly deviates from the pre-training data distribution or requires highly specialized knowledge. The entire capacity of the large model is leveraged for the specific task.
*   **Maximum Adaptability:** The model has maximum flexibility to learn new features and modify existing ones, as every part of its architecture is open to change. This can be critical for tasks that are very different from those encountered during pre-training.
*   **Conceptual Simplicity:** For many practitioners, the concept of simply continuing training with a new dataset is straightforward, even if the execution can be resource-intensive.

#### 2.3 Disadvantages

*   **High Computational Cost:** Full fine-tuning requires substantial computational resources, primarily **GPU memory (VRAM)**, to store gradients for all parameters and perform backpropagation across the entire model. This can make it inaccessible for individuals or organizations with limited hardware.
*   **Long Training Times:** Updating billions of parameters takes a considerable amount of time, even with powerful GPUs, leading to slower iteration cycles during development.
*   **Large Storage Requirements:** Each fine-tuned version of the model requires saving a complete copy of all its parameters, which can be hundreds of gigabytes for the largest LLMs. This becomes prohibitive when fine-tuning for multiple tasks or experimenting with different configurations.
*   **Risk of Catastrophic Forgetting:** Over-tuning on a small, specific dataset can cause the model to forget much of the general knowledge and capabilities it acquired during pre-training. This phenomenon, known as **catastrophic forgetting**, leads to a degradation of performance on tasks outside the specific fine-tuning objective.
*   **Data Intensive:** While less data than pre-training, full fine-tuning still benefits significantly from larger fine-tuning datasets, which can be costly to acquire and label.

### 3. Parameter-Efficient Fine-Tuning (PEFT)

**Parameter-Efficient Fine-Tuning (PEFT)** is a collection of techniques designed to adapt large pre-trained models to downstream tasks by updating only a small subset of the model's parameters, or by introducing a small number of new, trainable parameters.

#### 3.1 Mechanism and Approach

Instead of retraining the entire model, PEFT methods strategically modify specific parts of the architecture or inject small, trainable modules. The vast majority of the pre-trained model's weights remain frozen and unchanged. This drastically reduces the number of parameters that need to be updated during backpropagation, leading to significant savings in computational resources, training time, and storage. The core idea is to leverage the robust general knowledge encoded in the frozen pre-trained weights, while efficiently adapting specific capabilities for the new task.

#### 3.2 Key PEFT Techniques

Several prominent PEFT techniques have emerged, each with its own approach:

*   **LoRA (Low-Rank Adaptation):** Perhaps the most widely adopted PEFT method. LoRA injects small, trainable rank-decomposition matrices into each layer of the original model. Instead of directly fine-tuning the full weight matrices of a pre-trained language model, LoRA represents the update to the weight matrices with a low-rank decomposition. This means only these much smaller matrices are trained, while the original pre-trained weights remain frozen.
*   **QLoRA (Quantized LoRA):** An extension of LoRA that further reduces memory usage by quantizing the pre-trained model to 4-bit precision during training, while still performing LoRA updates. This allows fine-tuning even larger models on consumer-grade GPUs.
*   **Prompt Tuning/Soft Prompts:** Instead of modifying the model's weights, prompt tuning appends a set of trainable "soft prompts" (continuous vectors) to the input embeddings. The pre-trained model's weights are frozen, and only these soft prompt vectors are updated to guide the model's behavior for the specific task.
*   **Prefix Tuning:** Similar to prompt tuning, but it adds a small, trainable prefix of continuous task-specific vectors to every layer of the transformer model.
*   **Adapter Modules:** These techniques insert small, shallow neural network modules (adapters) between layers of the pre-trained model. Only the parameters of these adapter modules are fine-tuned, while the base model remains frozen.
*   **IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations):** Modifies the pre-trained model by learning a set of trainable vectors that scale the key, value, and feed-forward activations within the transformer layers.

#### 3.3 Advantages

*   **Significantly Reduced Computational Resources:** PEFT methods drastically cut down on **GPU memory** requirements and processing power needed for training, making it feasible to fine-tune large models on more accessible hardware.
*   **Faster Training Times:** With fewer parameters to update, training converges much quicker, leading to faster experimentation and development cycles.
*   **Minimal Storage Overhead:** Only the small, task-specific adapter weights (e.g., LoRA weights, prompt vectors) need to be stored, not a full copy of the entire model. This is a massive advantage for managing multiple fine-tuned models.
*   **Mitigates Catastrophic Forgetting:** By keeping the majority of the pre-trained model frozen, PEFT techniques are much less prone to forgetting the general knowledge acquired during pre-training, preserving the model's broader capabilities.
*   **Scalability for Multi-Task Learning:** A single large pre-trained model can serve as a base for numerous tasks, each with its own small, efficient adapter. This allows for efficient deployment and management of a diverse set of specialized models.
*   **Data Efficiency:** Often requires less task-specific data compared to full fine-tuning to achieve strong performance, as it leverages the extensive knowledge of the frozen base model.

#### 3.4 Disadvantages

*   **Potentially Lower Performance Ceiling:** While often achieving performance very close to full fine-tuning, PEFT methods might not always reach the absolute peak performance, especially for highly novel or complex tasks where extensive modification of the base model's representations is truly necessary.
*   **Hyperparameter Tuning Complexity:** PEFT introduces new hyperparameters specific to the chosen technique (e.g., LoRA rank `r`, alpha values, adapter sizes). Optimal performance often requires careful tuning of these PEFT-specific parameters, which can add a layer of complexity.
*   **Integration Overhead:** Integrating PEFT libraries (like `peft` from Hugging Face) into existing training pipelines can introduce some initial setup complexity compared to a straightforward full fine-tuning script.
*   **Architectural Constraints:** Some PEFT methods are more suited for certain model architectures (e.g., Transformers) than others, although the field is rapidly evolving.

### 4. Key Trade-offs: Full Fine-Tuning vs. PEFT

The choice between Full Fine-Tuning and PEFT hinges on a careful evaluation of several critical trade-offs.

#### 4.1 Computational Resources (Memory & Processing Power)

*   **Full Fine-Tuning:** Extremely resource-intensive. Requires very high-end GPUs with ample VRAM (e.g., 40GB+ for large LLMs) to handle the entire model and its gradients. Training can take days or weeks even with powerful hardware.
*   **PEFT:** Highly resource-efficient. Can reduce VRAM requirements by 3-10x or more (e.g., fine-tuning a 7B parameter model on a 16GB GPU with QLoRA). Training times are significantly shorter, often completing in hours or even minutes for smaller datasets. This is the primary driver for PEFT's popularity.

#### 4.2 Storage Overhead

*   **Full Fine-Tuning:** Each fine-tuned model checkpoint is a complete copy of the original model, often weighing tens or hundreds of gigabytes. Managing multiple such models can quickly consume vast amounts of storage.
*   **PEFT:** Only the small adapter weights are saved. For LoRA, these "LoRA adapters" are typically in the order of tens or hundreds of megabytes, making storage and deployment incredibly efficient. A single base model can be loaded, and different adapters can be swapped in or merged on the fly for various tasks.

#### 4.3 Performance Ceiling and Task Specificity

*   **Full Fine-Tuning:** Theoretically offers the highest performance ceiling because all parameters are optimized for the specific task. It can drastically change the model's behavior and learn highly specialized representations, which might be necessary for extremely complex or out-of-distribution tasks.
*   **PEFT:** Often achieves performance very close to full fine-tuning, typically within a small margin. For many practical applications, the performance difference is negligible, especially given the resource savings. However, there might be niche cases where the constrained adaptation of PEFT is insufficient to reach optimal performance, or where the task requires fundamental re-learning of core representations.

#### 4.4 Training Speed and Iteration Time

*   **Full Fine-Tuning:** Slower training due to the large number of parameters and gradients. This leads to longer iteration cycles, making experimentation and hyperparameter tuning more time-consuming.
*   **PEFT:** Significantly faster training due to the drastically reduced number of trainable parameters. This enables rapid experimentation, quicker iteration on hyperparameters, and faster deployment of updated models.

#### 4.5 Implementation Complexity and Ease of Use

*   **Full Fine-Tuning:** Conceptually straightforward in frameworks like Hugging Face Transformers: load model, load dataset, train. However, memory management and distributed training for very large models add practical complexity.
*   **PEFT:** Requires understanding specific PEFT techniques and integrating libraries like Hugging Face `peft`. While these libraries simplify the process, there's an initial learning curve and the need to tune PEFT-specific hyperparameters. Once set up, it can be very straightforward.

#### 4.6 Generalization and Catastrophic Forgetting

*   **Full Fine-Tuning:** More susceptible to **catastrophic forgetting**, especially when fine-tuned on small datasets. The model can lose its general capabilities learned during pre-training, making it brittle for tasks beyond the specific fine-tuning objective.
*   **PEFT:** Less prone to catastrophic forgetting because the majority of the pre-trained model's parameters remain frozen, preserving its foundational knowledge and general capabilities. This often leads to better generalization across related tasks and robustness.

#### 4.7 Scalability and Multi-Task Adaptation

*   **Full Fine-Tuning:** Poor scalability for multi-task scenarios. Each task requires a separate, full copy of the fine-tuned model, leading to immense storage and deployment challenges.
*   **PEFT:** Excellent scalability. A single frozen base model can be loaded, and multiple lightweight PEFT adapters (one per task) can be loaded and swapped as needed. This allows for efficiently serving many specialized tasks from a single model instance, dramatically reducing memory footprint during inference for multi-task serving.

### 5. Code Example

The following Python code snippet illustrates how to set up a model for Parameter-Efficient Fine-Tuning using LoRA with the Hugging Face `transformers` and `peft` libraries.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Load a pre-trained model (e.g., Llama-2-7b)
# For larger models, consider 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "meta-llama/Llama-2-7b-hf" # Replace with your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, # Apply 4-bit quantization
    device_map="auto"
)

# 2. Prepare model for k-bit training (important for QLoRA)
model = prepare_model_for_kbit_training(model)

# 3. Configure LoRA
# r: LoRA attention dimension (rank)
# lora_alpha: LoRA scaling factor
# target_modules: Modules to apply LoRA to (e.g., linear layers in attention)
# lora_dropout: Dropout probability for LoRA layers
# bias: Whether to train bias parameters
# task_type: Type of task (e.g., CAUSAL_LM for text generation)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Get the PEFT model
# This wraps the base model, only making LoRA parameters trainable
model = get_peft_model(model, lora_config)

# Print trainable parameters to see the significant reduction
model.print_trainable_parameters()

# Example: The model is now ready for training with a standard Trainer or custom loop.
# Only a tiny fraction of the original model's parameters are now trainable.
# The `trainable_parameters` count will be much smaller than the `all_param` count.


(End of code example section)
```

### 6. Conclusion

The choice between Full Fine-Tuning and Parameter-Efficient Fine-Tuning (PEFT) is a critical decision in the application of large pre-trained models. Full Fine-Tuning offers the highest potential for task-specific performance and complete adaptation, but at a steep cost in computational resources, storage, and training time. It is best suited for scenarios where maximum performance is paramount, resources are abundant, and the target task is fundamentally different from the pre-training objective.

Conversely, PEFT methods provide an elegant solution to the challenges posed by the increasing size of modern foundation models. By strategically updating only a small fraction of parameters, PEFT significantly reduces resource consumption, accelerates training, and minimizes storage overhead, all while largely preserving the model's general knowledge and often achieving performance very close to full fine-tuning. PEFT is particularly advantageous in environments with limited hardware, when managing multiple task-specific models, or when rapid experimentation and deployment are priorities.

In most contemporary applications, especially with the proliferation of multi-billion parameter models, PEFT techniques like LoRA and QLoRA have become the de facto standard. They offer an optimal balance between performance and efficiency, democratizing access to powerful fine-tuning capabilities that were previously restricted by extreme computational demands. While full fine-tuning remains a viable option for specific research or highly specialized production needs, PEFT represents a more scalable, accessible, and often sufficient approach for adapting large models to diverse downstream tasks.

---
<br>

<a name="türkçe-içerik"></a>
## Tam İnce Ayar ve PEFT: Takaslar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Tam İnce Ayar](#2-tam-ince-ayar)
  - [2.1 Mekanizma ve Yaklaşım](#21-mekanizma-ve-yaklaşım)
  - [2.2 Avantajlar](#22-avantajlar)
  - [2.3 Dezavantajlar](#23-dezavantajlar)
- [3. Parametre Verimli İnce Ayar (PEFT)](#3-parametre-verimli-ince-ayar-peft)
  - [3.1 Mekanizma ve Yaklaşım](#31-mekanizma-ve-yaklaşım)
  - [3.2 Başlıca PEFT Teknikleri](#32-başlıca-peft-teknikleri)
  - [3.3 Avantajlar](#33-avantajlar)
  - [3.4 Dezavantajlar](#34-dezavantajlar)
- [4. Temel Takaslar: Tam İnce Ayar ve PEFT](#4-temel-takaslar-tam-ince-ayar-ve-peft)
  - [4.1 Hesaplama Kaynakları (Bellek ve İşlem Gücü)](#41-hesaplama-kaynakları-bellek-ve-işlem-gücü)
  - [4.2 Depolama Alanı Maliyeti](#42-depolama-alanı-maliyeti)
  - [4.3 Performans Tavanı ve Görev Özgünlüğü](#43-performans-tavanı-ve-görev-özgünlüğü)
  - [4.4 Eğitim Hızı ve Yineleme Süresi](#44-eğitim-hızı-ve-yineleme-süresi)
  - [4.5 Uygulama Karmaşıklığı ve Kullanım Kolaylığı](#45-uygulama-karmaşıklığı-ve-kullanım-kolaylığı)
  - [4.6 Genelleme ve Felaket Unutma](#46-genelleme-ve-felaket-unutma)
  - [4.7 Ölçeklenebilirlik ve Çoklu Görev Adaptasyonu](#47-ölçeklenebilirlik-ve-çoklu-görev-adaptasyonu)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

### 1. Giriş

**Büyük Dil Modellerinin (BDM'ler)** ve diğer büyük ölçekli önceden eğitilmiş modellerin ortaya çıkışı, doğal dil işlemeden bilgisayar görüşüne kadar çeşitli alanlarda devrim yaratmıştır. Geniş ve çeşitli veri kümeleri üzerinde eğitilen bu modeller, dikkate değer genel amaçlı yeteneklere sahiptir. Ancak, bunları belirli alt görevler veya alana özgü uygulamalar için etkili bir şekilde kullanmak için genellikle **ince ayar** gereklidir. İnce ayar, önceden eğitilmiş bir modeli yeni bir veri kümesine uyarlayarak, davranışını hedef görevin özel gereksinimleriyle hizalar.

Tarihsel olarak, standart yaklaşım, önceden eğitilmiş modelin tüm veya neredeyse tüm parametrelerini güncellemeyi içeren **Tam İnce Ayar** olmuştur. Bu yöntem etkili olsa da, önemli hesaplama maliyetleri ve kaynak talepleri getirir. Bu zorluklara yanıt olarak, **Parametre Verimli İnce Ayar (PEFT)** olarak bilinen yeni bir paradigma ortaya çıkmıştır. PEFT teknikleri, tam ince ayara benzer performans elde etmeyi hedeflerken, eğitilebilir parametre sayısını önemli ölçüde azaltır, böylece hesaplama yükünü ve depolama gereksinimlerini düşürür.

Bu belge, Tam İnce Ayar ve PEFT yöntemleri arasındaki takasların kapsamlı bir analizini sunmaktadır. Her birinin mekanizmalarını, avantajlarını, dezavantajlarını derinlemesine inceleyecek ve bunları kaynak tüketimi, performans, ölçeklenebilirlik ve uygulama karmaşıklığı gibi çeşitli boyutlarda kritik bir şekilde karşılaştıracağız. Bu takasları anlamak, büyük önceden eğitilmiş modelleri belirli uygulamalara uyarlarken bilinçli kararlar vermek için çok önemlidir.

### 2. Tam İnce Ayar

Genellikle geleneksel yaklaşım olarak kabul edilen tam ince ayar, önceden eğitilmiş modelin tamamının yeni, göreve özgü bir veri kümesi üzerinde yeniden eğitilmesini içerir.

#### 2.1 Mekanizma ve Yaklaşım

**Tam İnce Ayarda**, önceden eğitilmiş modelin mimarisi sabit kalır, ancak tüm ağırlıkları (parametreleri) serbest bırakılır ve küçük bir öğrenme oranı kullanılarak eğitim süreci boyunca güncellenir. Model, hedef göreve özgü (örneğin, duygu analizi, metin özetleme, yeni bir alanda görüntü sınıflandırma) etiketli bir veri kümesi üzerinde eğitilir. **Kayıp fonksiyonu**, modelin bu yeni veri kümesi üzerindeki tahminlerine göre hesaplanır ve bu kaybı en aza indirmek için tüm parametreleri ayarlamak üzere **gradyan inişi** (veya Adam gibi varyantları) kullanılır. Bu süreç, modelin öğrenilmiş temsillerini genel ön eğitim dağılımından ince ayar görevinin belirli dağılımına etkili bir şekilde kaydırır.

#### 2.2 Avantajlar

*   **Potansiyel Olarak Daha Yüksek Performans Tavanı:** Tüm parametrelerin adapte olmasına izin vererek, tam ince ayar, özellikle hedef görev ön eğitim verisi dağılımından önemli ölçüde saptığında veya yüksek düzeyde özel bilgi gerektirdiğinde, hedef görevde mümkün olan en yüksek performansı elde etme potansiyeline sahiptir. Büyük modelin tüm kapasitesi belirli görev için kullanılır.
*   **Maksimum Uyarlanabilirlik:** Model, mimarisinin her bölümü değişime açık olduğundan, yeni özellikler öğrenmek ve mevcut olanları değiştirmek için maksimum esnekliğe sahiptir. Bu, ön eğitim sırasında karşılaşılanlardan çok farklı görevler için kritik olabilir.
*   **Kavramsal Basitlik:** Birçok uygulayıcı için, yeni bir veri kümesiyle eğitime devam etme kavramı, yürütme kaynak yoğun olsa bile, basittir.

#### 2.3 Dezavantajlar

*   **Yüksek Hesaplama Maliyeti:** Tam ince ayar, tüm parametreler için gradyanları depolamak ve tüm model boyunca geri yayılımı gerçekleştirmek için önemli hesaplama kaynakları, özellikle **GPU belleği (VRAM)** gerektirir. Bu, sınırlı donanıma sahip bireyler veya kuruluşlar için erişilemez hale getirebilir.
*   **Uzun Eğitim Süreleri:** Milyarlarca parametreyi güncellemek, güçlü GPU'larla bile önemli miktarda zaman alır ve geliştirme sırasında daha yavaş yineleme döngülerine yol açar.
*   **Büyük Depolama Gereksinimleri:** Modelin her ince ayarlı sürümü, tüm parametrelerinin tam bir kopyasını kaydetmeyi gerektirir; bu, en büyük BDM'ler için yüzlerce gigabayt olabilir. Bu, birden çok görev için ince ayar yaparken veya farklı yapılandırmalarla denemeler yaparken kısıtlayıcı hale gelir.
*   **Felaket Unutma Riski:** Küçük, belirli bir veri kümesi üzerinde aşırı ince ayar yapmak, modelin ön eğitim sırasında edindiği genel bilginin ve yeteneklerinin çoğunu unutmasına neden olabilir. **Felaket unutma** olarak bilinen bu fenomen, belirli ince ayar hedefi dışındaki görevlerde performansın düşmesine yol açar.
*   **Veri Yoğun:** Ön eğitime göre daha az veri gerektirse de, tam ince ayar hala daha büyük ince ayar veri kümelerinden önemli ölçüde fayda sağlar ve bunların elde edilmesi ve etiketlenmesi maliyetli olabilir.

### 3. Parametre Verimli İnce Ayar (PEFT)

**Parametre Verimli İnce Ayar (PEFT)**, büyük önceden eğitilmiş modelleri alt görevlere uyarlamak için tasarlanmış bir teknikler koleksiyonudur; bu, modelin parametrelerinin yalnızca küçük bir alt kümesini güncelleyerek veya az sayıda yeni, eğitilebilir parametre ekleyerek yapılır.

#### 3.1 Mekanizma ve Yaklaşım

Modelin tamamını yeniden eğitmek yerine, PEFT yöntemleri mimarinin belirli kısımlarını stratejik olarak değiştirir veya küçük, eğitilebilir modüller enjekte eder. Önceden eğitilmiş modelin ağırlıklarının büyük çoğunluğu dondurulmuş ve değişmeden kalır. Bu, geri yayılım sırasında güncellenmesi gereken parametre sayısını önemli ölçüde azaltır, bu da hesaplama kaynaklarında, eğitim süresinde ve depolamada önemli tasarruflar sağlar. Temel fikir, dondurulmuş önceden eğitilmiş ağırlıklarda kodlanmış sağlam genel bilgiyi kullanırken, yeni görev için belirli yetenekleri verimli bir şekilde uyarlamaktır.

#### 3.2 Başlıca PEFT Teknikleri

Birkaç önemli PEFT tekniği ortaya çıkmıştır ve her birinin kendi yaklaşımı vardır:

*   **LoRA (Low-Rank Adaptation):** Belki de en yaygın olarak benimsenen PEFT yöntemidir. LoRA, orijinal modelin her katmanına küçük, eğitilebilir rank-ayrışım matrisleri enjekte eder. Önceden eğitilmiş bir dil modelinin tam ağırlık matrislerini doğrudan ince ayar yapmak yerine, LoRA ağırlık matrislerindeki güncellemeyi düşük ranklı bir ayrışımla temsil eder. Bu, yalnızca bu çok daha küçük matrislerin eğitilmesi anlamına gelirken, orijinal önceden eğitilmiş ağırlıklar dondurulmuş kalır.
*   **QLoRA (Quantized LoRA):** LoRA'nın bir uzantısı olup, eğitim sırasında önceden eğitilmiş modeli 4-bit hassasiyete nicemleyerek bellek kullanımını daha da azaltır ve LoRA güncellemelerini gerçekleştirmeye devam eder. Bu, daha büyük modelleri bile tüketici sınıfı GPU'larda ince ayar yapmayı mümkün kılar.
*   **Prompt Tuning/Soft Prompts:** Modelin ağırlıklarını değiştirmek yerine, prompt tuning, girdi gömülü katmanlarına bir dizi eğitilebilir "yumuşak prompt" (sürekli vektörler) ekler. Önceden eğitilmiş modelin ağırlıkları dondurulur ve modelin belirli görev için davranışını yönlendirmek üzere yalnızca bu yumuşak prompt vektörleri güncellenir.
*   **Prefix Tuning:** Prompt tuning'e benzer, ancak transformer modelinin her katmanına küçük, eğitilebilir, göreve özgü sürekli vektörlerden oluşan bir önek ekler.
*   **Adapter Modülleri:** Bu teknikler, önceden eğitilmiş modelin katmanları arasına küçük, sığ sinir ağı modülleri (adapterler) ekler. Yalnızca bu adapter modüllerinin parametreleri ince ayar yapılırken, temel model dondurulmuş kalır.
*   **IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations):** Transformer katmanları içindeki anahtar, değer ve ileri besleme aktivasyonlarını ölçeklendiren bir dizi eğitilebilir vektör öğrenerek önceden eğitilmiş modeli değiştirir.

#### 3.3 Avantajlar

*   **Önemli Ölçüde Azaltılmış Hesaplama Kaynakları:** PEFT yöntemleri, **GPU belleği** gereksinimlerini ve eğitim için gereken işlem gücünü önemli ölçüde azaltarak, büyük modelleri daha erişilebilir donanımlarda ince ayar yapmayı mümkün kılar.
*   **Daha Hızlı Eğitim Süreleri:** Güncellenecek daha az parametre ile eğitim çok daha hızlı yakınsar, bu da daha hızlı deneme ve geliştirme döngülerine yol açar.
*   **Minimum Depolama Alanı Maliyeti:** Yalnızca küçük, göreve özgü adapter ağırlıklarının (örneğin, LoRA ağırlıkları, prompt vektörleri) depolanması gerekir, modelin tamamının tam bir kopyası değil. Bu, birden çok ince ayarlı modeli yönetmek için büyük bir avantajdır.
*   **Felaket Unutmayı Azaltır:** Önceden eğitilmiş modelin çoğunluğunu dondurulmuş tutarak, PEFT teknikleri, ön eğitim sırasında edinilen genel bilgiyi unutmaya çok daha az eğilimlidir ve modelin daha geniş yeteneklerini korur.
*   **Çoklu Görev Öğrenimi için Ölçeklenebilirlik:** Tek bir büyük önceden eğitilmiş model, her biri kendi küçük, verimli adapterine sahip sayısız görev için temel olarak hizmet edebilir. Bu, çeşitli özel modellerin verimli bir şekilde dağıtılmasına ve yönetilmesine olanak tanır.
*   **Veri Verimliliği:** Geniş temel modelin kapsamlı bilgisini kullandığı için, güçlü performans elde etmek için genellikle tam ince ayara kıyasla daha az göreve özgü veri gerektirir.

#### 3.4 Dezavantajlar

*   **Potansiyel Olarak Daha Düşük Performans Tavanı:** Tam ince ayara çok yakın performans elde etse de, PEFT yöntemleri, özellikle temel modelin temsillerinin kapsamlı bir şekilde değiştirilmesinin gerçekten gerekli olduğu çok yeni veya karmaşık görevler için mutlak en yüksek performansı her zaman yakalayamayabilir.
*   **Hiperparametre Ayar Karmaşıklığı:** PEFT, seçilen tekniğe özgü yeni hiperparametreler (örneğin, LoRA rank `r`, alfa değerleri, adapter boyutları) sunar. Optimal performans genellikle bu PEFT'ye özgü parametrelerin dikkatli ayarlanmasını gerektirir, bu da bir karmaşıklık katmanı ekleyebilir.
*   **Entegrasyon Yükü:** PEFT kütüphanelerini (Hugging Face'den `peft` gibi) mevcut eğitim pipeline'larına entegre etmek, doğrudan tam ince ayar betiğine kıyasla bazı başlangıç kurulum karmaşıklığı getirebilir.
*   **Mimari Kısıtlamalar:** Bazı PEFT yöntemleri, diğerlerinden ziyade belirli model mimarileri (örneğin, Transformer'lar) için daha uygundur, ancak alan hızla gelişmektedir.

### 4. Temel Takaslar: Tam İnce Ayar ve PEFT

Tam İnce Ayar ve PEFT arasındaki seçim, birkaç kritik takasın dikkatli bir şekilde değerlendirilmesine bağlıdır.

#### 4.1 Hesaplama Kaynakları (Bellek ve İşlem Gücü)

*   **Tam İnce Ayar:** Aşırı kaynak yoğun. Tüm modeli ve gradyanlarını işlemek için geniş VRAM'a sahip çok yüksek seviyeli GPU'lar (örneğin, büyük BDM'ler için 40GB+) gerektirir. Güçlü donanımla bile eğitim günler veya haftalar sürebilir.
*   **PEFT:** Yüksek derecede kaynak verimli. VRAM gereksinimlerini 3-10 kat veya daha fazla azaltabilir (örneğin, QLoRA ile 16GB GPU üzerinde 7B parametreli bir modeli ince ayar yapmak). Eğitim süreleri önemli ölçüde daha kısadır, genellikle daha küçük veri kümeleri için saatler veya hatta dakikalar içinde tamamlanır. Bu, PEFT'in popülaritesinin temel nedenidir.

#### 4.2 Depolama Alanı Maliyeti

*   **Tam İnce Ayar:** Her ince ayarlı model kontrol noktası, orijinal modelin tam bir kopyasıdır ve genellikle onlarca veya yüzlerce gigabayt ağırlığındadır. Birden fazla böyle modeli yönetmek hızla büyük miktarda depolama alanı tüketebilir.
*   **PEFT:** Yalnızca küçük adapter ağırlıkları kaydedilir. LoRA için, bu "LoRA adapterleri" genellikle onlarca veya yüzlerce megabayt düzeyindedir, bu da depolama ve dağıtımı inanılmaz derecede verimli hale getirir. Tek bir temel model yüklenebilir ve çeşitli görevler için farklı adapterler anında değiştirilebilir veya birleştirilebilir.

#### 4.3 Performans Tavanı ve Görev Özgünlüğü

*   **Tam İnce Ayar:** Teorik olarak en yüksek performans tavanını sunar çünkü tüm parametreler belirli görev için optimize edilmiştir. Modelin davranışını önemli ölçüde değiştirebilir ve oldukça özel temsiller öğrenebilir, bu da son derece karmaşık veya dağılım dışı görevler için gerekli olabilir.
*   **PEFT:** Tam ince ayara çok yakın performans elde eder, genellikle küçük bir marj içinde. Birçok pratik uygulama için, kaynak tasarrufu göz önüne alındığında performans farkı ihmal edilebilir düzeydedir. Ancak, PEFT'in kısıtlı adaptasyonunun optimal performansa ulaşmak için yetersiz olduğu veya görevin temel temsillerin temelden yeniden öğrenilmesini gerektirdiği niş durumlar olabilir.

#### 4.4 Eğitim Hızı ve Yineleme Süresi

*   **Tam İnce Ayar:** Çok sayıda parametre ve gradyan nedeniyle daha yavaş eğitim. Bu, daha uzun yineleme döngülerine yol açar, denemeyi ve hiperparametre ayarını daha zaman alıcı hale getirir.
*   **PEFT:** Dramatik bir şekilde azaltılmış eğitilebilir parametre sayısı nedeniyle önemli ölçüde daha hızlı eğitim. Bu, hızlı deneme, hiperparametreler üzerinde daha hızlı yineleme ve güncellenmiş modellerin daha hızlı dağıtımını sağlar.

#### 4.5 Uygulama Karmaşıklığı ve Kullanım Kolaylığı

*   **Tam İnce Ayar:** Hugging Face Transformers gibi çerçevelerde kavramsal olarak basittir: modeli yükle, veri kümesini yükle, eğit. Ancak, çok büyük modeller için bellek yönetimi ve dağıtılmış eğitim pratik karmaşıklık ekler.
*   **PEFT:** Belirli PEFT tekniklerini anlamayı ve Hugging Face `peft` gibi kütüphaneleri entegre etmeyi gerektirir. Bu kütüphaneler süreci basitleştirse de, başlangıçta bir öğrenme eğrisi ve PEFT'ye özgü hiperparametreleri ayarlama ihtiyacı vardır. Kurulduktan sonra, kullanımı çok basit olabilir.

#### 4.6 Genelleme ve Felaket Unutma

*   **Tam İnce Ayar:** Özellikle küçük veri kümeleri üzerinde ince ayar yapıldığında, **felaket unutmaya** daha yatkındır. Model, ön eğitim sırasında öğrendiği genel yeteneklerini kaybedebilir, bu da belirli ince ayar hedefinin ötesindeki görevler için kırılgan hale getirir.
*   **PEFT:** Felaket unutmaya daha az eğilimlidir çünkü önceden eğitilmiş modelin parametrelerinin çoğunluğu dondurulmuş kalır, temel bilgisini ve genel yeteneklerini korur. Bu genellikle ilgili görevler arasında daha iyi genelleme ve sağlamlık sağlar.

#### 4.7 Ölçeklenebilirlik ve Çoklu Görev Adaptasyonu

*   **Tam İnce Ayar:** Çoklu görev senaryoları için zayıf ölçeklenebilirlik. Her görev, ince ayarlı modelin ayrı, tam bir kopyasını gerektirir, bu da muazzam depolama ve dağıtım zorluklarına yol açar.
*   **PEFT:** Mükemmel ölçeklenebilirlik. Tek bir dondurulmuş temel model yüklenebilir ve birden çok hafif PEFT adapteri (görev başına bir tane) gerektiğinde yüklenebilir ve değiştirilebilir. Bu, tek bir model örneğinden birçok özel görevin verimli bir şekilde sunulmasına olanak tanır ve çoklu görev sunumu sırasında çıkarım için bellek ayak izini önemli ölçüde azaltır.

### 5. Kod Örneği

Aşağıdaki Python kod parçacığı, Hugging Face `transformers` ve `peft` kütüphanelerini kullanarak LoRA ile Parametre Verimli İnce Ayar için bir modelin nasıl kurulacağını göstermektedir.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Önceden eğitilmiş bir model yükleyin (örneğin, Llama-2-7b)
# Daha büyük modeller için, bellek verimliliği için 4-bit nicemlemeyi düşünün
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "meta-llama/Llama-2-7b-hf" # İstediğiniz modelle değiştirin
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, # 4-bit nicemlemeyi uygulayın
    device_map="auto"
)

# 2. Modeli k-bit eğitim için hazırlayın (QLoRA için önemlidir)
model = prepare_model_for_kbit_training(model)

# 3. LoRA'yı yapılandırın
# r: LoRA dikkat boyutu (rank)
# lora_alpha: LoRA ölçekleme faktörü
# target_modules: LoRA'yı uygulayacağınız modüller (örneğin, dikkat katmanlarındaki doğrusal katmanlar)
# lora_dropout: LoRA katmanları için Dropout olasılığı
# bias: Bias parametrelerini eğitip eğitmemek
# task_type: Görev türü (örneğin, metin üretimi için CAUSAL_LM)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. PEFT modelini alın
# Bu, temel modeli sarar ve yalnızca LoRA parametrelerini eğitilebilir hale getirir
model = get_peft_model(model, lora_config)

# Önemli ölçüdeki azalmayı görmek için eğitilebilir parametreleri yazdırın
model.print_trainable_parameters()

# Örnek: Model artık standart bir Trainer veya özel döngü ile eğitime hazır.
# Orijinal modelin parametrelerinin yalnızca küçük bir kısmı artık eğitilebilir.
# `trainable_parameters` sayısı, `all_param` sayısından çok daha küçük olacaktır.

(Kod örneği bölümünün sonu)
```

### 6. Sonuç

Tam İnce Ayar ve Parametre Verimli İnce Ayar (PEFT) arasındaki seçim, büyük önceden eğitilmiş modellerin uygulanmasında kritik bir karardır. Tam İnce Ayar, göreve özgü performans ve tam adaptasyon için en yüksek potansiyeli sunar, ancak hesaplama kaynakları, depolama ve eğitim süresi açısından yüksek bir maliyetle. Maksimum performansın çok önemli olduğu, kaynakların bol olduğu ve hedef görevin ön eğitim hedefinden temelden farklı olduğu senaryolar için en uygundur.

Tersine, PEFT yöntemleri, modern temel modellerin artan boyutunun getirdiği zorluklara zarif bir çözüm sunar. Parametrelerin yalnızca küçük bir kısmını stratejik olarak güncelleyerek, PEFT kaynak tüketimini önemli ölçüde azaltır, eğitimi hızlandırır ve depolama alanını en aza indirir, aynı zamanda modelin genel bilgisini büyük ölçüde korur ve çoğu zaman tam ince ayara çok yakın performans elde eder. PEFT, özellikle sınırlı donanıma sahip ortamlarda, birden fazla göreve özgü modeli yönetirken veya hızlı deneme ve dağıtımın öncelikli olduğu durumlarda avantajlıdır.

Günümüzdeki çoğu uygulamada, özellikle milyarlarca parametreli modellerin çoğalmasıyla, LoRA ve QLoRA gibi PEFT teknikleri fiili standart haline gelmiştir. Performans ve verimlilik arasında optimal bir denge sunarak, daha önce aşırı hesaplama talepleriyle kısıtlanan güçlü ince ayar yeteneklerine erişimi demokratikleştirirler. Tam ince ayar, belirli araştırma veya yüksek düzeyde uzmanlaşmış üretim ihtiyaçları için geçerli bir seçenek olmaya devam etse de, PEFT, büyük modelleri çeşitli alt görevlere uyarlamak için daha ölçeklenebilir, erişilebilir ve genellikle yeterli bir yaklaşımı temsil etmektedir.

