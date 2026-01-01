# Full Fine-Tuning vs. PEFT: Trade-offs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Full Fine-Tuning](#2-full-fine-tuning)
- [3. Parameter-Efficient Fine-Tuning (PEFT)](#3-parameter-efficient-fine-tuning-peft)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
Large Language Models (LLMs) have demonstrated unprecedented capabilities across a wide range of natural language processing tasks. However, adapting these massive pre-trained models to specific downstream tasks or datasets typically requires **fine-tuning**. Two primary paradigms for this adaptation have emerged: **Full Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)**. This document will comprehensively explore the trade-offs between these two approaches, examining their mechanisms, advantages, and disadvantages to guide practitioners in making informed decisions regarding model adaptation strategies in generative AI.

## 2. Full Fine-Tuning
Full fine-tuning involves updating all or a significant majority of the **parameters** of a pre-trained model. This method directly extends the pre-training phase by continuing to train the entire model on a new, task-specific dataset.

*   **Mechanism:** During full fine-tuning, the entire pre-trained model, including its **embedding layers**, **transformer blocks**, and **head layers**, is loaded. All trainable weights are then adjusted using a task-specific dataset and a chosen optimization algorithm (e.g., AdamW). This process aims to specialize the model for the new task by optimizing a new loss function tailored to that specific objective (e.g., classification, generation, summarization). The fundamental assumption is that altering every parameter allows for maximum adaptability and the deepest integration of task-specific knowledge.

*   **Advantages:**
    *   **Maximum Performance Potential:** By allowing all parameters to adapt, full fine-tuning theoretically offers the highest potential for performance on the target task, as the model can fully specialize. This often translates to superior accuracy, fluency, and nuanced understanding for highly complex or domain-specific tasks.
    *   **Comprehensive Adaptability:** It enables the model to capture intricate, task-specific nuances and biases present in the new dataset, leading to superior generalization on data similar to the fine-tuning set. This is particularly beneficial when the target task significantly diverges from the model's pre-training distribution.
    *   **No Architectural Changes:** This approach leverages the existing model architecture directly, requiring no additional module integration beyond potentially a new classification or regression head. This simplicity in setup can be appealing for some practitioners.

*   **Disadvantages:**
    *   **High Computational Cost:** Full fine-tuning demands substantial computational resources. This includes high-end Graphics Processing Units (GPUs) with very large memory capacities (e.g., 80GB for a 7B parameter model, and significantly more for larger models like those with 70B parameters or more). This translates to longer training times, considerable energy consumption, and often requires access to specialized cloud infrastructure.
    *   **Large Storage Footprint:** Saving checkpoints during fine-tuning, or the final fine-tuned model, requires storing the entire model's parameters. For large models, this can mean hundreds of gigabytes (e.g., Llama 2 70B is over 130GB), making storage, version control, and distribution challenging.
    *   **Risk of Catastrophic Forgetting:** Over-tuning on a small, specific dataset can lead to the model "forgetting" general knowledge, linguistic capabilities, or world facts acquired during the extensive pre-training phase. This phenomenon, known as **catastrophic forgetting**, can reduce the model's performance on broader tasks or out-of-distribution data.
    *   **Prone to Overfitting:** With limited task-specific data, a model with hundreds of billions of parameters can easily **overfit**, memorizing the training examples rather than learning generalizable patterns. This leads to poor generalization performance on unseen examples.
    *   **Deployment Challenges:** The large size of fully fine-tuned models can make deployment in resource-constrained environments (e.g., edge devices, mobile applications) challenging, often requiring powerful inference servers.

## 3. Parameter-Efficient Fine-Tuning (PEFT)
PEFT methods represent a family of techniques designed to mitigate the drawbacks of full fine-tuning by updating only a small fraction of the model's parameters or by introducing a small number of new, trainable parameters. The core idea is to freeze most of the original pre-trained weights and adapt the model using significantly fewer computational and storage resources.

*   **Mechanism:** PEFT encompasses various sophisticated techniques, each with its unique approach to efficiency:
    *   **LoRA (Low-Rank Adaptation):** This method inserts small, trainable **rank-decomposition matrices** into the transformer layers of the pre-trained model. During fine-tuning, only the parameters of these small matrices are updated, while the vast majority of the pre-trained weights remain frozen. This dramatically reduces the number of trainable parameters.
    *   **QLoRA (Quantized LoRA):** An extension of LoRA that further optimizes memory usage by quantizing the pre-trained model to 4-bit (or other low-bit precision) and then applying LoRA adapters. This technique allows fine-tuning even larger models on commodity hardware by significantly reducing the memory footprint of the frozen base model.
    *   **Adapter-tuning:** This approach adds small, bottleneck-like modules, known as **adapters**, between layers of the pre-trained model. Only the parameters of these lightweight adapter modules are trained, while the foundational LLM weights are kept static.
    *   **Prefix-tuning / Prompt-tuning:** Instead of modifying the model's internal weights, these methods learn a small sequence of continuous vectors (a "prefix" or "soft prompt") that are prepended to the input embeddings. These learned vectors then guide the frozen LLM towards the desired task without altering its core knowledge.

*   **Advantages:**
    *   **Significantly Reduced Computational Cost:** PEFT methods drastically cut down on GPU memory requirements and training time. This makes fine-tuning accessible with consumer-grade GPUs or less powerful cloud instances, democratizing access to LLM adaptation.
    *   **Lower Storage Footprint:** Only the small adapter weights need to be stored, often just a few megabytes or gigabytes, rather than hundreds. This enables easy sharing, versioning, and swapping of multiple task-specific adapters on a single base model.
    *   **Mitigation of Catastrophic Forgetting:** By keeping most of the pre-trained weights frozen, PEFT methods inherently preserve the general knowledge and capabilities of the base model, significantly reducing the risk of catastrophic forgetting.
    *   **Faster Experimentation:** The reduced resource requirements and faster training cycles allow for quicker iteration and experimentation with different hyper-parameters, datasets, or PEFT configurations.
    *   **Multi-task Adaptation:** Multiple task-specific adapters can be trained on a single base model. This allows a single large model to be adapted for various tasks without the need to store multiple full fine-tuned copies, simplifying management and deployment.

*   **Disadvantages:**
    *   **Potentially Lower Peak Performance:** While PEFT often achieves performance comparable to full fine-tuning, it might not always reach the absolute peak performance, especially for highly complex tasks that genuinely require deep architectural changes or extensive modification of the base model's knowledge.
    *   **Requires Method Selection and Hyperparameter Tuning:** Choosing the optimal PEFT method (e.g., LoRA, Adapter, Prefix-tuning) and tuning its specific hyperparameters (e.g., LoRA rank `r`, alpha, target modules) can be a non-trivial task. The best configuration can vary significantly depending on the model, task, and dataset.
    *   **Limited Adaptability for Specific Cases:** In some rare cases, the constraints imposed by freezing most parameters might prevent the model from learning extremely specialized patterns or making fundamental changes to its internal representations that are crucial for a particular, highly unique task.

## 4. Code Example
```python
# Conceptual Python code to illustrate loading a base model and then
# applying a PEFT adapter vs. a full model load for fine-tuning.

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import torch # For checking memory usage conceptually

# --- Full Fine-Tuning Scenario (Conceptual Load) ---
print("--- Full Fine-Tuning Scenario ---")
# In a real scenario, this would involve loading a full fine-tuned model
# or training the entire base model.
try:
    # Simulating loading a hypothetical 'full-fine-tuned-model'
    full_model_name = "mistralai/Mistral-7B-Instruct-v0.2" # Example base model
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    # Loading the full model requires significant VRAM
    model_full = AutoModelForCausalLM.from_pretrained(full_model_name, torch_dtype=torch.bfloat16)
    print(f"Loaded base model '{full_model_name}' for conceptual full fine-tuning.")
    # A fully fine-tuned model would be stored and loaded as a complete entity.
    # Its memory footprint would be substantial (e.g., ~14GB for Mistral-7B in bfloat16).
    print(f"Number of parameters in base model: {model_full.num_parameters():,}")
    print(f"Conceptual memory usage for full model (bfloat16): {model_full.num_parameters() * 2 / (1024**3):.2f} GB")
    del model_full # Free up memory for the next illustration
    torch.cuda.empty_cache()
except Exception as e:
    print(f"Could not load full model directly (this is conceptual for VRAM demonstration): {e}")

# --- PEFT (LoRA) Scenario ---
print("\n--- PEFT (LoRA) Scenario ---")
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# For PEFT, we still load the base model, but most of its weights will be frozen.
# QLoRA or 8-bit loading would further reduce this inference time memory.
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8, # LoRA rank, a smaller r means fewer trainable parameters
    lora_alpha=16, # Scaling factor for LoRA updates
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Modules to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model - this wraps the base model with LoRA adapters
peft_model = get_peft_model(base_model, lora_config)

print(f"Loaded base model '{base_model_name}' and applied LoRA adapters.")
# peft_model.print_trainable_parameters() returns a string and also shows total
print(f"Number of trainable parameters in PEFT model:")
peft_model.print_trainable_parameters()
# The actual memory footprint of the PEFT model during training/inference
# is base_model_size (frozen) + adapter_size (trainable).
# Only adapter_size contributes to the training gradient calculation.

# Illustrative example of saving/loading PEFT adapters:
# peft_model.save_pretrained("./my_lora_adapter")
# To load:
# base_model_for_inference = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
# loaded_peft_model = PeftModel.from_pretrained(base_model_for_inference, "./my_lora_adapter")
# print("PEFT adapter loaded successfully for inference.")

(End of code example section)
```

## 5. Conclusion
The choice between full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) depends critically on a careful evaluation of available computational resources, desired performance objectives, and the specific nature of the task and dataset.

**Full fine-tuning** remains the gold standard for achieving the absolute highest performance, particularly when ample computational resources (high-end GPUs, significant memory) and large, diverse task-specific datasets are available. It allows for the most profound specialization and the deepest integration of new knowledge. However, its significant demands on hardware, storage, and the inherent risks of catastrophic forgetting and overfitting necessitate careful management and substantial investment.

**PEFT methods**, conversely, offer an increasingly attractive and pragmatic alternative for resource-constrained environments or scenarios requiring rapid iteration, multi-task adaptation, and robust knowledge preservation. Techniques like LoRA and QLoRA have democratized access to fine-tuning large models, enabling practitioners to achieve competitive performance with a fraction of the cost, complexity, and memory footprint. While PEFT might occasionally fall short of the absolute peak performance of full fine-tuning in highly specialized scenarios, the compelling trade-off in terms of efficiency, scalability, and reduced risk often makes it the more pragmatic and sustainable choice for a wide array of generative AI applications. As LLMs continue to grow exponentially in size, PEFT is poised to become the dominant and indispensable paradigm for effectively adapting these powerful models to diverse real-world use cases.

---
<br>

<a name="türkçe-içerik"></a>
## Tam İnce Ayar ve PEFT: Takaslar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giris)
- [2. Tam İnce Ayar](#2-tam-ince-ayar)
- [3. Parametre Verimli İnce Ayar (PEFT)](#3-parametre-verimli-ince-ayar-peft)
- [4. Kod Örneği](#4-kod-ornegi)
- [5. Sonuç](#5-sonuc)

## 1. Giriş
Büyük Dil Modelleri (BDM'ler), geniş bir doğal dil işleme görevi yelpazesinde eşi benzeri görülmemiş yetenekler sergilemiştir. Ancak, bu devasa önceden eğitilmiş modelleri belirli alt görevlere veya veri kümelerine uyarlamak genellikle **ince ayar** gerektirir. Bu uyarlama için iki temel yaklaşım ortaya çıkmıştır: **Tam İnce Ayar** ve **Parametre Verimli İnce Ayar (PEFT)**. Bu belge, üretken yapay zekada model uyarlama stratejileri konusunda uygulayıcılara bilinçli kararlar vermelerinde rehberlik etmek amacıyla, bu iki yaklaşım arasındaki takasları, mekanizmalarını, avantajlarını ve dezavantajlarını kapsamlı bir şekilde inceleyecektir.

## 2. Tam İnce Ayar
Tam ince ayar, önceden eğitilmiş bir modelin tüm **parametrelerini** veya önemli bir çoğunluğunu güncelleme işlemini içerir. Bu yöntem, ön eğitim aşamasını, tüm modeli yeni, göreve özgü bir veri kümesi üzerinde eğitmeye devam ederek doğrudan genişletir.

*   **Mekanizma:** Tam ince ayar sırasında, **gömme katmanları**, **transformer blokları** ve **çıkış katmanları** dahil olmak üzere tüm önceden eğitilmiş model yüklenir. Tüm eğitilebilir ağırlıklar, göreve özgü bir veri kümesi ve seçilen bir optimizasyon algoritması (örn. AdamW) kullanılarak ayarlanır. Bu süreç, yeni görevin hedefine (örn. sınıflandırma, üretim, özetleme) göre uyarlanmış yeni bir kayıp fonksiyonunu optimize ederek modeli belirli görev için uzmanlaştırmayı amaçlar. Temel varsayım, her parametreyi değiştirmenin maksimum uyarlanabilirlik ve göreve özgü bilginin en derin entegrasyonuna izin verdiğidir.

*   **Avantajları:**
    *   **Maksimum Performans Potansiyeli:** Tüm parametrelerin adapte olmasına izin vererek, tam ince ayar teorik olarak hedef görevde en yüksek performans potansiyelini sunar, çünkü model tamamen uzmanlaşabilir. Bu genellikle yüksek düzeyde karmaşık veya alana özgü görevler için üstün doğruluk, akıcılık ve incelikli anlayış anlamına gelir.
    *   **Kapsamlı Uyarlanabilirlik:** Modelin yeni veri kümesinde bulunan karmaşık, göreve özgü nüansları ve önyargıları yakalamasını sağlar, bu da ince ayar kümesine benzer veriler üzerinde üstün genelleme yeteneği kazandırır. Bu, özellikle hedef görevin modelin ön eğitim dağılımından önemli ölçüde farklılaştığı durumlarda faydalıdır.
    *   **Mimari Değişiklik Yok:** Bu yaklaşım, potansiyel olarak yeni bir sınıflandırma veya regresyon başlığı dışında ek modül entegrasyonu gerektirmeyen mevcut model mimarisini doğrudan kullanır. Kurulumdaki bu basitlik, bazı uygulayıcılar için cazip olabilir.

*   **Dezavantajları:**
    *   **Yüksek Hesaplama Maliyeti:** Tam ince ayar, büyük bellek kapasitesine sahip yüksek performanslı Grafik İşlem Birimleri (GPU'lar) (örn. 7B parametreli bir model için 80GB, 70B veya daha büyük parametreli modeller için çok daha fazlası) dahil olmak üzere önemli hesaplama kaynakları gerektirir. Bu, daha uzun eğitim süreleri, önemli enerji tüketimi ve genellikle özel bulut altyapısına erişim anlamına gelir.
    *   **Büyük Depolama Alanı İhtiyacı:** İnce ayar sırasında kontrol noktalarını veya son ince ayarlı modeli kaydetmek, modelin tüm parametrelerini depolamayı gerektirir. Büyük modeller için bu, yüzlerce gigabayt anlamına gelebilir (örn. Llama 2 70B 130GB'tan fazladır), bu da depolama, sürüm kontrolü ve dağıtımı zorlaştırır.
    *   **Felaket Unutma Riski:** Küçük, belirli bir veri kümesi üzerinde aşırı ince ayar yapmak, modelin kapsamlı ön eğitim aşamasında edindiği genel bilgileri, dilsel yetenekleri veya dünya gerçeklerini "unutmasına" yol açabilir. **Felaket unutma** olarak bilinen bu fenomen, modelin daha geniş görevlerde veya dağılım dışı verilerde performansını düşürebilir.
    *   **Aşırı Uyumlanmaya Eğilimli:** Sınırlı göreve özgü veriyle, yüz milyarlarca parametreye sahip bir model kolayca **aşırı uyumlanabilir**, genelleyici desenler öğrenmek yerine eğitim örneklerini ezberleyebilir. Bu, görülmeyen örneklerde kötü genelleme performansına yol açar.
    *   **Dağıtım Zorlukları:** Tamamen ince ayarlı modellerin büyük boyutu, kaynak kısıtlı ortamlarda (örn. uç cihazlar, mobil uygulamalar) dağıtımı zorlaştırabilir ve genellikle güçlü çıkarım sunucuları gerektirir.

## 3. Parametre Verimli İnce Ayar (PEFT)
PEFT yöntemleri, modelin parametrelerinin yalnızca küçük bir kısmını güncelleyerek veya az sayıda yeni, eğitilebilir parametre ekleyerek tam ince ayarın dezavantajlarını azaltmayı amaçlayan bir teknik ailesini temsil eder. Temel fikir, orijinal önceden eğitilmiş ağırlıkların çoğunu dondurmak ve modeli önemli ölçüde daha az hesaplama ve depolama kaynağı kullanarak uyarlamaktır.

*   **Mekanizma:** PEFT, her birinin kendine özgü bir verimlilik yaklaşımı olan çeşitli sofistike teknikleri kapsar:
    *   **LoRA (Low-Rank Adaptation):** Bu yöntem, önceden eğitilmiş modelin transformer katmanlarına küçük, eğitilebilir **düşük ranklı ayrıştırma matrisleri** ekler. İnce ayar sırasında sadece bu küçük matrislerin parametreleri güncellenirken, önceden eğitilmiş ağırlıkların büyük çoğunluğu donmuş kalır. Bu, eğitilebilir parametre sayısını önemli ölçüde azaltır.
    *   **QLoRA (Quantized LoRA):** LoRA'nın bir uzantısı olup, önceden eğitilmiş modeli 4-bit'e (veya başka düşük bit hassasiyetine) niceler ve ardından LoRA adaptörlerini uygular, böylece bellek kullanımını daha da optimize eder. Bu teknik, donmuş temel modelin bellek ayak izini önemli ölçüde azaltarak daha büyük modellerin bile tüketici sınıfı donanımlarda ince ayarlanmasına olanak tanır.
    *   **Adaptör Ayarı (Adapter-tuning):** Bu yaklaşım, önceden eğitilmiş modelin katmanları arasına **adaptörler** olarak bilinen küçük, dar boğaz benzeri modüller ekler. Temel BDM ağırlıkları statik tutulurken, sadece bu hafif adaptör modüllerinin parametreleri eğitilir.
    *   **Ön Ek Ayarı (Prefix-tuning / Prompt-tuning):** Modelin dahili ağırlıklarını değiştirmek yerine, bu yöntemler giriş gömmelerine önceden eklenen küçük bir dizi sürekli vektör (bir "ön ek" veya "yumuşak istem") öğrenir. Bu öğrenilmiş vektörler, donmuş BDM'yi çekirdek bilgisini değiştirmeden istenen göreve yönlendirir.

*   **Avantajları:**
    *   **Önemli Ölçüde Azaltılmış Hesaplama Maliyeti:** PEFT yöntemleri, GPU bellek gereksinimlerini ve eğitim süresini büyük ölçüde azaltır. Bu, tüketici sınıfı GPU'larla veya daha az güçlü bulut örnekleriyle ince ayarı erişilebilir kılar ve BDM uyarlamasına erişimi demokratikleştirir.
    *   **Daha Düşük Depolama Alanı İhtiyacı:** Yalnızca küçük adaptör ağırlıklarının depolanması gerekir, bu da genellikle yüzlerce gigabayt yerine sadece birkaç megabayt veya gigabayttır. Bu, tek bir temel model üzerinde birden fazla göreve özgü adaptörün kolayca paylaşılmasını, sürüm kontrolünü ve değiştirilmesini sağlar.
    *   **Felaket Unutmanın Azaltılması:** Önceden eğitilmiş ağırlıkların çoğunu donmuş tutarak, PEFT yöntemleri temel modelin genel bilgisini ve yeteneklerini doğal olarak korur, felaket unutma riskini önemli ölçüde azaltır.
    *   **Daha Hızlı Deneyler:** Azaltılmış kaynak gereksinimleri ve daha hızlı eğitim döngüleri, farklı hiperparametreler, veri kümeleri veya PEFT yapılandırmaları ile daha hızlı yineleme ve deneme yapılmasına olanak tanır.
    *   **Çoklu Görev Uyarlaması:** Tek bir temel model üzerinde birden fazla göreve özgü adaptör eğitilebilir. Bu, tek bir büyük modelin, birden fazla tam ince ayarlı kopyayı depolamaya gerek kalmadan çeşitli görevler için uyarlanmasına olanak tanır, bu da yönetimi ve dağıtımı basitleştirir.

*   **Dezavantajları:**
    *   **Potansiyel Olarak Daha Düşük Tepe Performansı:** PEFT genellikle tam ince ayara benzer performans elde etse de, özellikle derin mimari değişiklikler veya temel modelin bilgisinin kapsamlı şekilde değiştirilmesi gereken yüksek düzeyde karmaşık görevler için mutlak tepe performansına her zaman ulaşamayabilir.
    *   **Yöntem Seçimi ve Hiperparametre Ayarı Gerektirir:** Optimal PEFT yöntemini (örn. LoRA, Adaptör, Ön Ek Ayarı) seçmek ve spesifik hiperparametrelerini (örn. LoRA rank `r`, alfa, hedef modüller) ayarlamak önemsiz bir görev olmayabilir. En iyi yapılandırma, modele, göreve ve veri kümesine bağlı olarak önemli ölçüde değişebilir.
    *   **Belirli Durumlar İçin Sınırlı Uyarlanabilirlik:** Bazı nadir durumlarda, çoğu parametreyi dondurarak uygulanan kısıtlamalar, modelin belirli, oldukça benzersiz bir görev için kritik olan aşırı özel desenleri öğrenmesini veya dahili temsillerinde temel değişiklikler yapmasını engelleyebilir.

## 4. Kod Örneği
```python
# Bir temel modeli yüklemeyi ve ardından bir PEFT adaptörü uygulamayı
# tam model yüklemesine karşı ince ayar için karşılaştıran kavramsal Python kodu.

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import torch # Bellek kullanımını kavramsal olarak kontrol etmek için

# --- Tam İnce Ayar Senaryosu (Kavramsal Yükleme) ---
print("--- Tam İnce Ayar Senaryosu ---")
# Gerçek bir senaryoda, bu, tam ince ayarlı bir modeli yüklemeyi
# veya tüm temel modeli eğitmeyi içerir.
try:
    # Hipotetik 'tam-ince-ayarli-model'i yüklemeyi simüle etme
    full_model_name = "mistralai/Mistral-7B-Instruct-v0.2" # Örnek temel model
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    # Tam modelin yüklenmesi önemli VRAM gerektirir
    model_full = AutoModelForCausalLM.from_pretrained(full_model_name, torch_dtype=torch.bfloat16)
    print(f"Kavramsal tam ince ayar için '{full_model_name}' temel modeli yüklendi.")
    # Tamamen ince ayarlı bir model, eksiksiz bir varlık olarak depolanır ve yüklenir.
    # Bellek ayak izi önemli olacaktır (örn. bfloat16'da Mistral-7B için ~14GB).
    print(f"Temel modeldeki parametre sayısı: {model_full.num_parameters():,}")
    print(f"Tam model için kavramsal bellek kullanımı (bfloat16): {model_full.num_parameters() * 2 / (1024**3):.2f} GB")
    del model_full # Bir sonraki örnekleme için belleği boşaltın
    torch.cuda.empty_cache()
except Exception as e:
    print(f"Tam model doğrudan yüklenemedi (bu, VRAM gösterimi için kavramsaldır): {e}")

# --- PEFT (LoRA) Senaryosu ---
print("\n--- PEFT (LoRA) Senaryosu ---")
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# PEFT için hala temel modeli yüklüyoruz, ancak ağırlıklarının çoğu donmuş olacak.
# QLoRA veya 8-bit yükleme, bu çıkarım zamanı belleğini daha da azaltacaktır.
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)

# LoRA yapılandırmasını tanımla
lora_config = LoraConfig(
    r=8, # LoRA rankı, daha küçük bir r daha az eğitilebilir parametre demektir
    lora_alpha=16, # LoRA güncellemeleri için ölçeklendirme faktörü
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # LoRA'nın uygulanacağı modüller
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# PEFT modelini al - bu, temel modeli LoRA adaptörleriyle sarar
peft_model = get_peft_model(base_model, lora_config)

print(f"'{base_model_name}' temel modeli yüklendi ve LoRA adaptörleri uygulandı.")
# peft_model.print_trainable_parameters() bir dize döndürür ve ayrıca toplamı gösterir
print(f"PEFT modelindeki eğitilebilir parametre sayısı:")
peft_model.print_trainable_parameters()
# PEFT modelinin eğitim/çıkarım sırasındaki gerçek bellek ayak izi
# temel_model_boyutu (donmuş) + adaptör_boyutu (eğitilebilir) şeklindedir.
# Yalnızca adaptör_boyutu, eğitim gradyan hesaplamasına katkıda bulunur.

# PEFT adaptörlerini kaydetme/yükleme örneklemesi:
# peft_model.save_pretrained("./my_lora_adapter")
# Yüklemek için:
# base_model_for_inference = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
# loaded_peft_model = PeftModel.from_pretrained(base_model_for_inference, "./my_lora_adapter")
# print("PEFT adaptörü çıkarım için başarıyla yüklendi.")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
Tam ince ayar ile Parametre Verimli İnce Ayar (PEFT) arasındaki seçim, mevcut hesaplama kaynaklarının, hedeflenen performans amaçlarının ve görevin ve veri kümesinin özel doğasının dikkatli bir şekilde değerlendirilmesine kritik derecede bağlıdır.

**Tam ince ayar**, özellikle bol hesaplama kaynağı (yüksek performanslı GPU'lar, önemli bellek) ve geniş, çeşitli göreve özgü veri kümeleri mevcut olduğunda, mutlak en yüksek performansı elde etmek için altın standart olmaya devam etmektedir. En derin uzmanlaşmaya ve yeni bilginin en yoğun entegrasyonuna olanak tanır. Ancak, donanım, depolama üzerindeki önemli talepleri ve felaket unutma ile aşırı uyumlanma gibi doğal riskleri, dikkatli yönetimi ve önemli yatırımı zorunlu kılar.

**PEFT yöntemleri** ise, kaynak kısıtlı ortamlar veya hızlı yineleme, çoklu görev uyarlaması ve sağlam bilgi koruması gerektiren senaryolar için giderek daha çekici ve pragmatik bir alternatif sunmaktadır. LoRA ve QLoRA gibi teknikler, büyük modelleri ince ayar yapmaya erişimi demokratikleştirerek uygulayıcıların maliyetin, karmaşıklığın ve bellek ayak izinin küçük bir kısmıyla rekabetçi performans elde etmelerini sağlamıştır. PEFT, oldukça özel senaryolarda tam ince ayarın mutlak tepe performansının gerisinde kalabilse de, verimlilik, ölçeklenebilirlik ve azaltılmış risk açısından sunduğu cazip takas, onu geniş bir üretken yapay zeka uygulaması yelpazesi için daha pragmatik ve sürdürülebilir bir seçenek haline getirmektedir. BDM'ler boyut olarak katlanarak büyümeye devam ettikçe, PEFT bu güçlü modelleri çeşitli gerçek dünya kullanım durumlarına etkili bir şekilde uyarlamak için baskın ve vazgeçilmez bir paradigma olmaya adaydır.




