# Hugging Face PEFT Library Deep Dive

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenge of Fine-tuning Large Language Models](#2-the-challenge-of-fine-tuning-large-language-models)
- [3. Understanding Parameter-Efficient Fine-Tuning (PEFT) Methodologies](#3-understanding-parameter-efficient-fine-tuning-peft-methodologies)
    - [3.1. LoRA (Low-Rank Adaptation)](#31-lora-low-rank-adaptation)
    - [3.2. Prefix-Tuning](#32-prefix-tuning)
    - [3.3. P-Tuning/P-Tuning v2](#33-p-tuningp-tuning-v2)
    - [3.4. Prompt-Tuning](#34-prompt-tuning)
    - [3.5. QLoRA](#35-qlora)
- [4. The Hugging Face PEFT Library](#4-the-hugging-face-peft-library)
    - [4.1. Architecture and Design Principles](#41-architecture-and-design-principles)
    - [4.2. Key Classes and Functions](#42-key-classes-and-functions)
    - [4.3. Workflow](#43-workflow)
- [5. Advantages and Use Cases](#5-advantages-and-use-cases)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

## 1. Introduction

The rapid evolution of Generative AI, particularly in the domain of Large Language Models (LLMs), has brought forth models with billions of parameters, capable of performing a wide array of complex tasks. While these models, such as GPT-3, Llama, and Falcon, exhibit remarkable generalisation capabilities, adapting them to specific downstream tasks or proprietary datasets often necessitates a process called **fine-tuning**. Traditionally, fine-tuning involves updating all or a significant portion of the model's parameters, which can be prohibitively expensive in terms of computational resources (GPU memory, training time) and storage for each task-specific variant. This challenge becomes particularly acute in industrial applications where numerous specialised models are required.

The **Hugging Face PEFT (Parameter-Efficient Fine-Tuning) library** emerges as a critical innovation addressing this dilemma. It provides a lightweight and efficient approach to adapt pre-trained LLMs to new tasks without the need to retrain the entire model or store full copies of the fine-tuned models. By selectively updating a small subset of parameters or introducing a minimal number of new, trainable parameters, PEFT methodologies can achieve performance comparable to full fine-tuning while drastically reducing resource consumption. This document delves deep into the Hugging Face PEFT library, exploring its underlying principles, supported methodologies, architectural design, and practical applications, providing a comprehensive understanding of its role in democratising and scaling LLM adaptation.

## 2. The Challenge of Fine-tuning Large Language Models

Fine-tuning large pre-trained models presents several significant obstacles that impede their widespread and cost-effective deployment:

*   **Computational Cost:** Updating billions of parameters requires substantial computational power, primarily high-end GPUs with vast amounts of VRAM. A single full fine-tuning run can take days or weeks on expensive hardware, making iterative experimentation and rapid deployment challenging.
*   **Memory Footprint:** Loading an entire LLM, its optimizer states, and gradients into GPU memory for fine-tuning often exceeds the capacity of consumer-grade or even many professional GPUs. This necessitates techniques like gradient accumulation or ZeRO (Zero Redundancy Optimizer), but these only mitigate the issue to an extent.
*   **Storage Requirements:** Each fine-tuned variant of an LLM typically requires storing a complete copy of the model weights, which can be hundreds of gigabytes per model. For an organisation deploying dozens or hundreds of specialised models, this leads to an explosion in storage costs and management complexity.
*   **Catastrophic Forgetting:** When fine-tuning a model on a new, specific dataset, there's a risk of **catastrophic forgetting**, where the model loses its previously acquired general knowledge and capabilities in favour of optimising for the new task. Parameter-efficient methods can sometimes mitigate this by preserving the core pre-trained weights.
*   **Environmental Impact:** The energy consumption associated with training and fine-tuning large models contributes significantly to their carbon footprint, raising environmental concerns.

These challenges highlight the imperative for more efficient adaptation strategies, paving the way for the development and adoption of PEFT methodologies.

## 3. Understanding Parameter-Efficient Fine-Tuning (PEFT) Methodologies

The PEFT library consolidates various parameter-efficient fine-tuning techniques, offering a unified API for their implementation. These methods primarily focus on either adding a few new trainable parameters to the existing model or creating small, trainable adapter modules that influence the model's behaviour.

### 3.1. LoRA (Low-Rank Adaptation)

**LoRA** is arguably the most widely adopted and effective PEFT technique. Its core idea is to freeze the pre-trained model weights and inject trainable rank-decomposition matrices into each layer of the Transformer architecture. Specifically, for a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA introduces two low-rank matrices, $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$, where $r \ll \min(d, k)$. The update to the weights during fine-tuning is then represented as $W_0 + \Delta W$, where $\Delta W = BA$. Only the matrices $A$ and $B$ are trained, significantly reducing the number of trainable parameters.

This approach offers several advantages:
*   **Reduced Trainable Parameters:** Typically reduces the number of trainable parameters by orders of magnitude compared to full fine-tuning.
*   **No Inference Latency:** During inference, the adapted weights ($W_0 + BA$) can be merged back into the original $W_0$, incurring no additional inference latency.
*   **Memory Efficiency:** Drastically lowers GPU memory requirements during training.

### 3.2. Prefix-Tuning

**Prefix-Tuning** (Li & Liang, 2021) involves adding a small, continuous, task-specific "prefix" to the input sequence of a large language model. These prefixes are trainable vectors that are prepended to the keys and values in the attention layers of the Transformer. The pre-trained model's original parameters remain frozen. By optimising these prefix vectors, the model's behaviour can be guided towards specific tasks without altering its core knowledge. The prefix acts as a 'soft prompt' that influences the model's internal representations.

### 3.3. P-Tuning/P-Tuning v2

**P-Tuning** (Liu et al., 2021) extends the concept of prompt engineering by optimising continuous prompt embeddings instead of discrete tokens. Instead of manually crafting prompts, P-tuning uses a small neural network (e.g., an MLP) to generate these continuous prompt embeddings, which are then fed into the input of the pre-trained model. This allows for more expressive and fine-grained control over the model's behaviour.

**P-Tuning v2** (Liu et al., 2022) further refines this by applying the prompt tokens to *all* layers of the Transformer, not just the input layer. This multi-layer application, combined with other improvements, makes P-Tuning v2 more stable and performant, often achieving competitive results with full fine-tuning while still being parameter-efficient. It resembles prefix-tuning in its multi-layer adaptation but with continuous embeddings generated via a prompt encoder.

### 3.4. Prompt-Tuning

**Prompt-Tuning** (Lester et al., 2021) is a simpler variant where only a sequence of task-specific "soft prompts" (continuous embeddings) are learned and prepended to the input embeddings. Unlike P-Tuning, it doesn't typically involve a prompt encoder network; the prompt embeddings are directly optimised. It's often considered the simplest form of PEFT that operates purely at the input layer.

### 3.5. QLoRA

**QLoRA** (Dettmers et al., 2023) is an advanced technique that combines LoRA with **quantization**. It allows fine-tuning a LLM with 4-bit quantized base models while still achieving full 16-bit fine-tuning performance. QLoRA introduces several innovations:
*   **4-bit NormalFloat (NF4):** A new data type that is theoretically optimal for normally distributed data.
*   **Double Quantization:** Quantizing the quantization constants themselves to save even more memory.
*   **Paged Optimizers:** Using NVIDIA's unified memory to prevent out-of-memory errors during gradient spikes.

By using QLoRA, models with tens of billions of parameters can be fine-tuned on a single GPU (e.g., Llama 2 70B on a single 80GB A100), making large-scale LLM adaptation accessible to a much broader audience.

## 4. The Hugging Face PEFT Library

The Hugging Face PEFT library is built on top of the popular `transformers` library, providing a seamless and intuitive experience for applying parameter-efficient fine-tuning methods. Its design focuses on modularity, ease of use, and integration.

### 4.1. Architecture and Design Principles

The library's architecture is designed around abstracting the complexities of different PEFT methods into a unified interface. Key design principles include:
*   **Unified API:** A single set of classes and functions (`PeftModel`, `PeftConfig`, `get_peft_model`) allows users to experiment with various PEFT techniques without needing to rewrite substantial portions of their code.
*   **Modularity:** Each PEFT method (LoRA, Prefix-Tuning, etc.) is implemented as a distinct configuration class and a corresponding adapter module, making it easy to add new methods or extend existing ones.
*   **Integration with Transformers:** PEFT models can be seamlessly integrated with `transformers.Trainer`, making it straightforward to apply these techniques within existing training pipelines. The `PeftModel` wraps a `transformers` model, allowing it to function as a regular `torch.nn.Module`.
*   **Resource Efficiency:** The core objective is to minimise memory footprint and computational overhead during fine-tuning.

### 4.2. Key Classes and Functions

*   `**PeftConfig**`: This is the base class for all PEFT method-specific configuration objects (e.g., `LoraConfig`, `PrefixTuningConfig`). It defines the parameters relevant to a particular PEFT method, such as `r`, `lora_alpha`, `target_modules` for LoRA, or `num_virtual_tokens` for Prefix-Tuning.
*   `**PeftModel**`: This class wraps a `transformers` model and injects the PEFT layers/modules according to the provided `PeftConfig`. It inherits from `torch.nn.Module` and behaves like a regular PyTorch model, but with only the PEFT parameters being trainable by default.
*   `**get_peft_model(model, peft_config)**`: This crucial function takes a `transformers` model and a `PeftConfig` object, then returns a `PeftModel` instance. It handles the injection of PEFT modules into the base model.
*   `**get_peft_model_state_dict(model)**`: Extracts the state dictionary containing only the trainable PEFT parameters.
*   `**set_peft_model_state_dict(model, peft_model_state_dict)**`: Loads PEFT parameters into a `PeftModel`.

### 4.3. Workflow

The typical workflow for using the Hugging Face PEFT library involves a few straightforward steps:

1.  **Load a base pre-trained model:** Use `AutoModelForCausalLM` or `AutoModelForSequenceClassification` from `transformers` to load the desired LLM.
2.  **Define a `PeftConfig`:** Instantiate a configuration object for your chosen PEFT method (e.g., `LoraConfig`), specifying its parameters.
3.  **Create a `PeftModel`:** Pass the base model and the `PeftConfig` to `get_peft_model()` to obtain a PEFT-wrapped model.
4.  **Prepare data and Trainer:** Set up your dataset, tokenizer, and `transformers.TrainingArguments` as usual.
5.  **Train the model:** Pass the `PeftModel` to `transformers.Trainer` and start training. Only the PEFT parameters will be updated.
6.  **Save/Load PEFT adapters:** Use `model.save_pretrained(path)` to save only the PEFT adapter weights. These can be loaded back into a base model later using `PeftModel.from_pretrained(base_model, path)`.
7.  **Merge adapters (optional):** For methods like LoRA, the adapter weights can be merged into the base model's weights using `model.merge_and_unload()` for deployment without the PEFT library dependency.

## 5. Advantages and Use Cases

The Hugging Face PEFT library offers a multitude of advantages and opens up new possibilities for working with large models:

*   **Significant Resource Reduction:** The primary benefit is the dramatic reduction in GPU memory consumption and training time, making fine-tuning accessible on less powerful hardware and speeding up development cycles.
*   **Reduced Storage Footprint:** Only the small adapter weights need to be saved per task, rather than the entire model, leading to substantial storage savings. A single base model can support numerous task-specific adapters.
*   **Mitigation of Catastrophic Forgetting:** By keeping the original LLM weights frozen, PEFT methods often help preserve the general knowledge encoded in the base model, reducing the risk of catastrophic forgetting.
*   **Flexibility and Experimentation:** The unified API makes it easy to experiment with different PEFT methods and their hyperparameters to find the optimal strategy for a given task.
*   **Improved Scalability:** Organisations can fine-tune hundreds of domain-specific models from a single base LLM, allowing for scalable customisation.
*   **Wide Applicability:** PEFT techniques are effective across a broad range of NLP tasks, including text classification, sentiment analysis, named entity recognition, summarisation, and generative tasks like dialogue generation and creative writing.

Typical use cases include:
*   **Domain Adaptation:** Fine-tuning an LLM for a specific industry (e.g., legal, medical, finance) using a small domain-specific dataset.
*   **Task Specialisation:** Adapting a general-purpose LLM to perform a very specific task, such as extracting particular entities from text or answering questions about a niche knowledge base.
*   **Personalisation:** Creating personalised versions of generative models for individual users or groups.
*   **Edge Device Deployment:** While still resource-intensive, PEFT-tuned models are smaller and potentially more amenable to deployment on devices with limited resources than fully fine-tuned models.

## 6. Code Example

The following short example demonstrates how to apply LoRA using the Hugging Face PEFT library for fine-tuning a pre-trained language model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load a pre-trained base model and tokenizer
model_name = "facebook/opt-125m" # A small model for demonstration
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token if not already set (common for GPT-like models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA attention dimension
    lora_alpha=16,  # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "v_proj"], # Target query and value projection layers
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Type of bias to be trained, "none" for no bias
    task_type=TaskType.CAUSAL_LM # Task type for the model
)

# 3. Get the PEFT model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# For a real scenario, you would now prepare a dataset and pass it to Trainer.
# Example: Dummy data for illustration
# from datasets import Dataset
# raw_texts = ["This is a test sentence.", "Another sentence for fine-tuning."]
# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, max_length=128)
# dummy_dataset = Dataset.from_dict({"text": raw_texts}).map(tokenize_function, batched=True)
#
# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-4,
#     per_device_train_batch_size=2,
#     num_train_epochs=1,
#     save_steps=100,
#     logging_steps=10,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dummy_dataset,
#     tokenizer=tokenizer,
# )
#
# trainer.train()
#
# # Save only the PEFT adapter weights
# model.save_pretrained("./lora_adapter")

(End of code example section)
```

## 7. Conclusion

The Hugging Face PEFT library represents a pivotal advancement in the field of Generative AI, specifically addressing the formidable challenges associated with fine-tuning large language models. By providing robust implementations of various parameter-efficient fine-tuning methodologies like LoRA, Prefix-Tuning, and QLoRA, the library enables researchers and practitioners to adapt multi-billion-parameter models to specific tasks and domains with significantly reduced computational resources, memory footprint, and storage requirements.

Its seamless integration with the `transformers` ecosystem ensures a familiar and efficient workflow, democratising access to advanced LLM customisation. The ability to achieve performance comparable to full fine-tuning with orders of magnitude fewer trainable parameters fundamentally changes the landscape of LLM deployment, making it more sustainable, scalable, and accessible. As LLMs continue to grow in size and complexity, the role of libraries like PEFT will become increasingly critical, empowering a wider community to harness the full potential of generative AI for diverse and impactful applications.

---
<br>

<a name="türkçe-içerik"></a>
## Hugging Face PEFT Kütüphanesine Derinlemesine Bakış

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Büyük Dil Modellerinde İnce Ayarın Zorlukları](#2-büyük-dil-modellerinde-ince-ayarın-zorlukları)
- [3. Parametre Verimli İnce Ayar (PEFT) Metodolojilerini Anlamak](#3-parametre-verimli-i̇nce-ayar-peft-metodolojilerini-anlamak)
    - [3.1. LoRA (Düşük Ranklı Adaptasyon)](#31-lora-düşük-ranklı-adaptasyon)
    - [3.2. Prefix-Tuning (Ön Ek Ayarı)](#32-prefix-tuning-ön-ek-ayarı)
    - [3.3. P-Tuning/P-Tuning v2](#33-p-tuningp-tuning-v2)
    - [3.4. Prompt-Tuning (İstem Ayarı)](#34-prompt-tuning-i̇stem-ayarı)
    - [3.5. QLoRA](#35-qlora)
- [4. Hugging Face PEFT Kütüphanesi](#4-hugging-face-peft-kütüphanesi)
    - [4.1. Mimari ve Tasarım İlkeleri](#41-mimari-ve-tasarım-i̇lkeleri)
    - [4.2. Temel Sınıflar ve Fonksiyonlar](#42-temel-sınıflar-ve-fonksiyonlar)
    - [4.3. İş Akışı](#43-i̇ş-akışı)
- [5. Avantajlar ve Kullanım Senaryoları](#5-avantajlar-ve-kullanım-senaryoları)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

## 1. Giriş

Üretken Yapay Zeka'nın, özellikle Büyük Dil Modelleri (LLM'ler) alanındaki hızlı gelişimi, milyarlarca parametreye sahip, çok çeşitli karmaşık görevleri yerine getirebilen modelleri ortaya çıkarmıştır. GPT-3, Llama ve Falcon gibi bu modeller dikkate değer genelleme yetenekleri sergilese de, onları belirli aşağı akış görevlerine veya tescilli veri kümelerine uyarlamak genellikle **ince ayar** adı verilen bir süreç gerektirir. Geleneksel olarak, ince ayar, modelin tüm parametrelerini veya önemli bir kısmını güncellemeyi içerir; bu da her göreve özel varyant için hesaplama kaynakları (GPU belleği, eğitim süresi) ve depolama açısından aşırı derecede maliyetli olabilir. Bu zorluk, çok sayıda uzmanlaşmış modele ihtiyaç duyulan endüstriyel uygulamalarda özellikle keskin bir hal alır.

**Hugging Face PEFT (Parametre Verimli İnce Ayar) kütüphanesi**, bu ikilemi ele alan kritik bir yenilik olarak ortaya çıkmaktadır. Önceden eğitilmiş LLM'leri, tüm modeli yeniden eğitme veya ince ayarlı modellerin tam kopyalarını depolama ihtiyacı olmadan yeni görevlere uyarlamak için hafif ve verimli bir yaklaşım sunar. Parametrelerin küçük bir alt kümesini seçici olarak güncelleyerek veya minimum sayıda yeni, eğitilebilir parametre tanıtarak, PEFT metodolojileri, tam ince ayara benzer bir performans elde edebilirken kaynak tüketimini büyük ölçüde azaltır. Bu belge, Hugging Face PEFT kütüphanesine derinlemesine bir bakış sunarak, temel ilkelerini, desteklenen metodolojilerini, mimari tasarımını ve pratik uygulamalarını inceleyerek, LLM adaptasyonunu demokratikleştirme ve ölçeklendirmedeki rolüne dair kapsamlı bir anlayış sağlamaktadır.

## 2. Büyük Dil Modellerinde İnce Ayarın Zorlukları

Büyük önceden eğitilmiş modellerde ince ayar yapmak, yaygın ve uygun maliyetli dağıtımlarını engelleyen birkaç önemli engel sunar:

*   **Hesaplama Maliyeti:** Milyarlarca parametreyi güncellemek, esas olarak yüksek miktarda VRAM'e sahip üst düzey GPU'lar olmak üzere önemli hesaplama gücü gerektirir. Tek bir tam ince ayar çalıştırması, pahalı donanımlarda günler veya haftalar sürebilir, bu da tekrarlayan deneyleri ve hızlı dağıtımı zorlaştırır.
*   **Bellek Ayak İzi:** Tüm bir LLM'yi, optimize edici durumlarını ve gradyanlarını ince ayar için GPU belleğine yüklemek, genellikle tüketici sınıfı veya hatta birçok profesyonel GPU'nun kapasitesini aşar. Bu, gradyan birikimi veya ZeRO (Zero Redundancy Optimizer) gibi teknikleri gerektirir, ancak bunlar sorunu yalnızca bir dereceye kadar hafifletir.
*   **Depolama Gereksinimleri:** Bir LLM'nin her ince ayarlı varyantı, tipik olarak model ağırlıklarının tam bir kopyasını depolamayı gerektirir; bu da model başına yüzlerce gigabayt olabilir. Düzinelerce veya yüzlerce özel model dağıtan bir kuruluş için bu, depolama maliyetlerinde ve yönetim karmaşıklığında bir patlamaya yol açar.
*   **Felaket Unutkanlığı:** Bir modeli yeni, belirli bir veri kümesinde ince ayar yaparken, modelin yeni görev için optimize etme lehine daha önce edindiği genel bilgiyi ve yetenekleri kaybetmesi olan **felaket unutkanlığı** riski vardır. Parametre verimli yöntemler, çekirdek önceden eğitilmiş ağırlıkları koruyarak bunu bazen hafifletebilir.
*   **Çevresel Etki:** Büyük modellerin eğitimi ve ince ayarı ile ilişkili enerji tüketimi, karbon ayak izlerine önemli ölçüde katkıda bulunarak çevresel kaygıları artırır.

Bu zorluklar, daha verimli adaptasyon stratejileri için zorunluluğu vurgulayarak, PEFT metodolojilerinin geliştirilmesi ve benimsenmesinin yolunu açmaktadır.

## 3. Parametre Verimli İnce Ayar (PEFT) Metodolojilerini Anlamak

PEFT kütüphanesi, çeşitli parametre verimli ince ayar tekniklerini birleştirerek, bunların uygulanması için birleşik bir API sunar. Bu yöntemler öncelikle, mevcut modele birkaç yeni eğitilebilir parametre eklemeye veya modelin davranışını etkileyen küçük, eğitilebilir adaptör modülleri oluşturmaya odaklanır.

### 3.1. LoRA (Düşük Ranklı Adaptasyon)

**LoRA** tartışmasız en yaygın olarak benimsenen ve etkili PEFT tekniğidir. Temel fikri, önceden eğitilmiş model ağırlıklarını dondurmak ve Transformer mimarisinin her katmanına eğitilebilir rank-ayrışım matrisleri enjekte etmektir. Özellikle, $W_0 \in \mathbb{R}^{d \times k}$ önceden eğitilmiş bir ağırlık matrisi için LoRA, iki düşük ranklı matris, $A \in \mathbb{R}^{d \times r}$ ve $B \in \mathbb{R}^{r \times k}$ tanıtır, burada $r \ll \min(d, k)$. İnce ayar sırasındaki ağırlık güncellemeleri daha sonra $W_0 + \Delta W$ olarak temsil edilir, burada $\Delta W = BA$. Sadece $A$ ve $B$ matrisleri eğitilir, bu da eğitilebilir parametre sayısını önemli ölçüde azaltır.

Bu yaklaşım çeşitli avantajlar sunar:
*   **Azaltılmış Eğitilebilir Parametreler:** Tam ince ayara kıyasla eğitilebilir parametre sayısını genellikle kat kat azaltır.
*   **Çıkarım Gecikmesi Yok:** Çıkarım sırasında, uyarlanmış ağırlıklar ($W_0 + BA$) orijinal $W_0$'a geri birleştirilebilir, bu da ek çıkarım gecikmesi yaratmaz.
*   **Bellek Verimliliği:** Eğitim sırasında GPU bellek gereksinimlerini önemli ölçüde düşürür.

### 3.2. Prefix-Tuning (Ön Ek Ayarı)

**Prefix-Tuning** (Li & Liang, 2021), büyük bir dil modelinin giriş dizisine küçük, sürekli, göreve özel bir "ön ek" eklemeyi içerir. Bu ön ekler, Transformer'ın dikkat katmanlarındaki anahtarlara ve değerlere önden eklenen eğitilebilir vektörlerdir. Önceden eğitilmiş modelin orijinal parametreleri donuk kalır. Bu ön ek vektörleri optimize edilerek, modelin temel bilgisi değiştirilmeden modelin davranışı belirli görevlere yönlendirilebilir. Ön ek, modelin iç temsillerini etkileyen 'yumuşak bir istem' görevi görür.

### 3.3. P-Tuning/P-Tuning v2

**P-Tuning** (Liu et al., 2021), istem mühendisliği kavramını, ayrık tokenlar yerine sürekli istem gömülerini optimize ederek genişletir. İstemleri manuel olarak oluşturmak yerine, P-tuning, bu sürekli istem gömülerini üretmek için küçük bir sinir ağı (örneğin, bir MLP) kullanır ve bunlar daha sonra önceden eğitilmiş modelin girdisine beslenir. Bu, modelin davranışı üzerinde daha etkileyici ve ince taneli kontrol sağlar.

**P-Tuning v2** (Liu et al., 2022), bu durumu Transformer'ın sadece giriş katmanına değil, *tüm* katmanlarına istem tokenlarını uygulayarak daha da iyileştirir. Diğer iyileştirmelerle birleşen bu çok katmanlı uygulama, P-Tuning v2'yi daha kararlı ve performanslı hale getirir, genellikle tam ince ayarla rekabetçi sonuçlar elde ederken yine de parametre verimliliğini korur. Çok katmanlı adaptasyonda prefix-tuning'e benzer ancak bir istem kodlayıcı aracılığıyla üretilen sürekli gömülerle çalışır.

### 3.4. Prompt-Tuning (İstem Ayarı)

**Prompt-Tuning** (Lester et al., 2021), sadece göreve özel "yumuşak istemlerin" (sürekli gömüler) öğrenildiği ve giriş gömülerine önden eklendiği daha basit bir varyanttır. P-Tuning'den farklı olarak, tipik olarak bir istem kodlayıcı ağı içermez; istem gömüleri doğrudan optimize edilir. Safça giriş katmanında çalışan en basit PEFT biçimi olarak kabul edilir.

### 3.5. QLoRA

**QLoRA** (Dettmers et al., 2023), LoRA'yı **niceleme** ile birleştiren gelişmiş bir tekniktir. Tam 16-bit ince ayar performansını korurken, 4-bit nicelenmiş temel modellerle bir LLM'nin ince ayarını yapmaya olanak tanır. QLoRA birkaç yenilik sunar:
*   **4-bit NormalFloat (NF4):** Normal dağılmış veriler için teorik olarak optimal yeni bir veri türü.
*   **Çift Niceleme:** Niceleme sabitlerinin kendilerini niceleyerek daha da fazla bellek tasarrufu.
*   **Sayfalı Optimize Ediciler:** Gradyan ani artışları sırasında bellek yetersizliği hatalarını önlemek için NVIDIA'nın birleşik belleğini kullanma.

QLoRA kullanarak, on milyarlarca parametreye sahip modeller tek bir GPU'da (örneğin, tek bir 80GB A100'de Llama 2 70B) ince ayar yapılabilir, bu da büyük ölçekli LLM adaptasyonunu çok daha geniş bir kitleye ulaştırır.

## 4. Hugging Face PEFT Kütüphanesi

Hugging Face PEFT kütüphanesi, popüler `transformers` kütüphanesi üzerine inşa edilmiştir ve parametre verimli ince ayar yöntemlerini uygulamak için sorunsuz ve sezgisel bir deneyim sunar. Tasarımı modülerliğe, kullanım kolaylığına ve entegrasyona odaklanmıştır.

### 4.1. Mimari ve Tasarım İlkeleri

Kütüphanenin mimarisi, farklı PEFT yöntemlerinin karmaşıklıklarını birleşik bir arayüze soyutlamak üzere tasarlanmıştır. Temel tasarım ilkeleri şunları içerir:
*   **Birleşik API:** Tek bir sınıf ve fonksiyon kümesi (`PeftModel`, `PeftConfig`, `get_peft_model`), kullanıcıların önemli kod parçalarını yeniden yazmaya gerek kalmadan çeşitli PEFT tekniklerini denemelerine olanak tanır.
*   **Modülerlik:** Her PEFT yöntemi (LoRA, Prefix-Tuning vb.), ayrı bir yapılandırma sınıfı ve ilgili bir adaptör modülü olarak uygulanır, bu da yeni yöntemler eklemeyi veya mevcut olanları genişletmeyi kolaylaştırır.
*   **Transformers ile Entegrasyon:** PEFT modelleri, `transformers.Trainer` ile sorunsuz bir şekilde entegre edilebilir, bu da bu teknikleri mevcut eğitim boru hatlarına uygulamayı kolaylaştırır. `PeftModel`, bir `transformers` modelini sararak normal bir `torch.nn.Module` gibi işlev görmesini sağlar.
*   **Kaynak Verimliliği:** Temel amaç, ince ayar sırasında bellek ayak izini ve hesaplama yükünü en aza indirmektir.

### 4.2. Temel Sınıflar ve Fonksiyonlar

*   `**PeftConfig**`: Bu, tüm PEFT yöntemine özgü yapılandırma nesnelerinin (örneğin, `LoraConfig`, `PrefixTuningConfig`) temel sınıfıdır. Belirli bir PEFT yöntemiyle ilgili parametreleri tanımlar; örneğin LoRA için `r`, `lora_alpha`, `target_modules` veya Prefix-Tuning için `num_virtual_tokens`.
*   `**PeftModel**`: Bu sınıf, bir `transformers` modelini sarar ve sağlanan `PeftConfig`'e göre PEFT katmanlarını/modüllerini enjekte eder. `torch.nn.Module`'den miras alır ve normal bir PyTorch modeli gibi davranır, ancak varsayılan olarak yalnızca PEFT parametreleri eğitilebilir.
*   `**get_peft_model(model, peft_config)**`: Bu kritik fonksiyon, bir `transformers` modelini ve bir `PeftConfig` nesnesini alır, ardından bir `PeftModel` örneği döndürür. PEFT modüllerinin temel modele enjekte edilmesini yönetir.
*   `**get_peft_model_state_dict(model)**`: Yalnızca eğitilebilir PEFT parametrelerini içeren durum sözlüğünü çıkarır.
*   `**set_peft_model_state_dict(model, peft_model_state_dict)**`: PEFT parametrelerini bir `PeftModel`'e yükler.

### 4.3. İş Akışı

Hugging Face PEFT kütüphanesini kullanmak için tipik iş akışı birkaç basit adımı içerir:

1.  **Önceden eğitilmiş bir temel modeli yükleyin:** İstenen LLM'yi yüklemek için `transformers`'tan `AutoModelForCausalLM` veya `AutoModelForSequenceClassification` kullanın.
2.  **Bir `PeftConfig` tanımlayın:** Seçtiğiniz PEFT yöntemi için bir yapılandırma nesnesi (örneğin, `LoraConfig`) örnekleyin ve parametrelerini belirtin.
3.  **Bir `PeftModel` oluşturun:** Temel modeli ve `PeftConfig`'i `get_peft_model()`'e ileterek PEFT ile sarılmış bir model elde edin.
4.  **Verileri ve Trainer'ı hazırlayın:** Veri kümenizi, belirtecinizi ve `transformers.TrainingArguments`'ı her zamanki gibi ayarlayın.
5.  **Modeli eğitin:** `PeftModel`'i `transformers.Trainer`'a iletin ve eğitimi başlatın. Yalnızca PEFT parametreleri güncellenecektir.
6.  **PEFT adaptörlerini kaydetme/yükleme:** Yalnızca PEFT adaptör ağırlıklarını kaydetmek için `model.save_pretrained(path)` kullanın. Bunlar daha sonra `PeftModel.from_pretrained(base_model, path)` kullanılarak temel bir modele geri yüklenebilir.
7.  **Adaptörleri birleştirme (isteğe bağlı):** LoRA gibi yöntemler için, adaptör ağırlıkları, PEFT kütüphanesi bağımlılığı olmadan dağıtım için `model.merge_and_unload()` kullanılarak temel modelin ağırlıklarına birleştirilebilir.

## 5. Avantajlar ve Kullanım Senaryoları

Hugging Face PEFT kütüphanesi, çok sayıda avantaj sunar ve büyük modellerle çalışma için yeni olanaklar açar:

*   **Önemli Kaynak Azaltma:** Birincil fayda, GPU bellek tüketiminde ve eğitim süresinde önemli ölçüde azalma, bu da ince ayarı daha az güçlü donanımlarda erişilebilir kılar ve geliştirme döngülerini hızlandırır.
*   **Azaltılmış Depolama Ayak İzi:** Her görev için tüm model yerine yalnızca küçük adaptör ağırlıklarının kaydedilmesi gerekir, bu da önemli depolama tasarrufları sağlar. Tek bir temel model, çok sayıda göreve özel adaptörü destekleyebilir.
*   **Felaket Unutkanlığını Azaltma:** Orijinal LLM ağırlıklarını donuk tutarak, PEFT yöntemleri genellikle temel modelde kodlanmış genel bilgiyi korumaya yardımcı olur ve felaket unutkanlığı riskini azaltır.
*   **Esneklik ve Deney:** Birleşik API, belirli bir görev için optimal stratejiyi bulmak üzere farklı PEFT yöntemlerini ve bunların hiperparametrelerini denemeyi kolaylaştırır.
*   **Geliştirilmiş Ölçeklenebilirlik:** Kuruluşlar, tek bir temel LLM'den yüzlerce alana özgü modeli ince ayar yapabilir, bu da ölçeklenebilir özelleştirmeye olanak tanır.
*   **Geniş Uygulanabilirlik:** PEFT teknikleri, metin sınıflandırma, duygu analizi, adlandırılmış varlık tanıma, özetleme ve diyalog oluşturma ve yaratıcı yazma gibi üretken görevler dahil olmak üzere geniş bir NLP görevi yelpazesinde etkilidir.

Tipik kullanım senaryoları şunları içerir:
*   **Alan Adaptasyonu:** Küçük, alana özgü bir veri kümesi kullanarak belirli bir endüstri (örneğin, hukuk, tıp, finans) için bir LLM'nin ince ayarı.
*   **Görev Uzmanlaşması:** Genel amaçlı bir LLM'yi, metinden belirli varlıkları çıkarmak veya niş bir bilgi tabanı hakkında soruları yanıtlamak gibi çok özel bir görevi yerine getirecek şekilde uyarlamak.
*   **Kişiselleştirme:** Bireysel kullanıcılar veya gruplar için üretken modellerin kişiselleştirilmiş versiyonlarını oluşturma.
*   **Uç Cihaz Dağıtımı:** Hala kaynak yoğun olsa da, PEFT ayarlı modeller, tamamen ince ayarlı modellere göre daha küçük ve sınırlı kaynaklara sahip cihazlarda dağıtıma potansiyel olarak daha uygun olabilir.

## 6. Kod Örneği

Aşağıdaki kısa örnek, Hugging Face PEFT kütüphanesini kullanarak önceden eğitilmiş bir dil modelinin ince ayarı için LoRA'yı nasıl uygulayacağınızı göstermektedir.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Önceden eğitilmiş bir temel model ve tokenlayıcı yükleyin
model_name = "facebook/opt-125m" # Gösterim için küçük bir model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Dolgu tokenı ayarlanmamışsa ayarlayın (GPT benzeri modeller için yaygın)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA yapılandırmasını tanımlayın
lora_config = LoraConfig(
    r=8,  # LoRA dikkat boyutu
    lora_alpha=16,  # LoRA ölçeklendirme için alfa parametresi
    target_modules=["q_proj", "v_proj"], # Hedef sorgu ve değer projeksiyon katmanları
    lora_dropout=0.05, # LoRA katmanları için dropout olasılığı
    bias="none", # Eğitilecek bias türü, "none" bias yok demek
    task_type=TaskType.CAUSAL_LM # Model için görev türü
)

# 3. PEFT modelini alın
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Gerçek bir senaryo için, şimdi bir veri kümesi hazırlar ve Trainer'a iletirdiniz.
# Örnek: Gösterim için sahte veri
# from datasets import Dataset
# raw_texts = ["Bu bir test cümlesidir.", "İnce ayar için başka bir cümle."]
# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, max_length=128)
# dummy_dataset = Dataset.from_dict({"text": raw_texts}).map(tokenize_function, batched=True)
#
# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-4,
#     per_device_train_batch_size=2,
#     num_train_epochs=1,
#     save_steps=100,
#     logging_steps=10,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dummy_dataset,
#     tokenizer=tokenizer,
# )
#
# trainer.train()
#
# # Yalnızca PEFT adaptör ağırlıklarını kaydedin
# model.save_pretrained("./lora_adapter")

(Kod örneği bölümünün sonu)
```

## 7. Sonuç

Hugging Face PEFT kütüphanesi, Üretken Yapay Zeka alanında, özellikle büyük dil modellerinin ince ayarı ile ilişkili zorlu sorunları ele alarak önemli bir ilerlemeyi temsil etmektedir. LoRA, Prefix-Tuning ve QLoRA gibi çeşitli parametre verimli ince ayar metodolojilerinin sağlam uygulamalarını sağlayarak, kütüphane araştırmacılara ve uygulayıcılara, milyarlarca parametreye sahip modelleri, önemli ölçüde azaltılmış hesaplama kaynakları, bellek ayak izi ve depolama gereksinimleriyle belirli görevlere ve alanlara uyarlamalarına olanak tanır.

`transformers` ekosistemi ile sorunsuz entegrasyonu, tanıdık ve verimli bir iş akışı sağlayarak, gelişmiş LLM özelleştirmesine erişimi demokratikleştirmektedir. Tam ince ayar ile karşılaştırılabilir performansı, kat kat daha az eğitilebilir parametre ile elde etme yeteneği, LLM dağıtımının manzarasını temelden değiştirerek onu daha sürdürülebilir, ölçeklenebilir ve erişilebilir hale getirmektedir. LLM'ler boyut ve karmaşıklık açısından büyümeye devam ettikçe, PEFT gibi kütüphanelerin rolü giderek kritik hale gelecek ve daha geniş bir topluluğun üretken yapay zekanın tam potansiyelini çeşitli ve etkili uygulamalar için kullanmasına olanak sağlayacaktır.

