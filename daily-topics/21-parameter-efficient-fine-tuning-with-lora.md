# Parameter-Efficient Fine-Tuning with LoRA

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenge of Fine-Tuning Large Models](#2-the-challenge-of-fine-tuning-large-models)
- [3. LoRA: Low-Rank Adaptation](#3-lora-low-rank-adaptation)
  - [3.1. Core Principle](#31-core-principle)
  - [3.2. Mathematical Formulation](#32-mathematical-formulation)
  - [3.3. Advantages of LoRA](#33-advantages-of-lora)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<br>

### 1. Introduction
The advent of **large pre-trained models (LPMs)**, particularly in natural language processing (NLP) and computer vision, has revolutionized artificial intelligence. These models, often comprising billions of parameters, demonstrate remarkable capabilities across a wide range of tasks. However, adapting these colossal models to specific downstream tasks typically involves **fine-tuning**, a process where the entire model or a significant portion of its parameters are updated using task-specific data. This full fine-tuning approach presents substantial computational, memory, and storage challenges, particularly for resource-constrained environments or when adapting a single model to numerous distinct tasks.

**Parameter-Efficient Fine-Tuning (PEFT)** techniques have emerged as a crucial paradigm to address these limitations. PEFT methods aim to achieve comparable performance to full fine-tuning while only updating a small fraction of the model's parameters. Among these, **Low-Rank Adaptation (LoRA)** stands out as a highly effective and widely adopted approach. LoRA significantly reduces the number of trainable parameters by injecting small, low-rank matrices into the existing layers of the pre-trained model, thereby making fine-tuning more accessible, faster, and less resource-intensive. This document delves into the principles, mechanisms, and benefits of LoRA as a leading PEFT strategy.

### 2. The Challenge of Fine-Tuning Large Models
Fine-tuning large pre-trained models, such as GPT-3, LLaMA, or Stable Diffusion, involves updating the weights of potentially hundreds of billions of parameters. This process poses several significant hurdles:

*   **Computational Cost:** Training a large model requires immense computational power, typically involving multiple high-end GPUs for extended periods. Full fine-tuning necessitates recalculating gradients for every parameter, leading to prohibitive energy consumption and time.
*   **Memory Footprint:** Storing the model weights, optimizer states, and activations for backpropagation consumes vast amounts of GPU memory. For models with hundreds of billions of parameters, even loading the model can exhaust available memory on standard hardware, let alone the memory required for training.
*   **Storage Requirements:** Each fine-tuned version of a large model, tailored for a specific task, requires storing a complete copy of the model weights. If an organization needs to adapt a base model to dozens or hundreds of tasks, this quickly leads to terabytes or even petabytes of storage, making deployment and management cumbersome.
*   **Slow Experimentation:** The high cost and time associated with full fine-tuning hinder rapid experimentation and iteration, which are vital for research and development in AI. Researchers often need to test multiple hypotheses, datasets, and configurations, a process severely bottlenecked by slow fine-tuning cycles.
*   **Catastrophic Forgetting:** While full fine-tuning can achieve high performance on specific tasks, there is a risk of **catastrophic forgetting**, where the model loses some of its general capabilities learned during pre-training when overly specialized on new data.

These challenges underscore the necessity for more efficient adaptation strategies, making PEFT methods like LoRA indispensable in the era of foundation models.

### 3. LoRA: Low-Rank Adaptation
LoRA is a parameter-efficient fine-tuning technique that addresses the challenges of adapting large pre-trained models by introducing a low-rank decomposition of the weight update matrices. Instead of fine-tuning all parameters, LoRA freezes the original pre-trained weights and injects trainable low-rank decomposition matrices into the transformer architecture's attention mechanism (specifically, the query and value projection matrices).

#### 3.1. Core Principle
The fundamental idea behind LoRA is that the update to a pre-trained weight matrix `W_0` during fine-tuning, denoted as `ΔW`, often has a **low intrinsic rank**. This means that `ΔW` can be effectively approximated by the product of two much smaller matrices.

Specifically, for any dense layer represented by a weight matrix `W_0 ∈ R^(d_out × d_in)`, LoRA posits that the update `ΔW` can be approximated as `BA`, where `B ∈ R^(d_out × r)` and `A ∈ R^(r × d_in)`. Here, `r` is the **LoRA rank**, a hyperparameter that is typically much smaller than `min(d_in, d_out)`. This implies that `ΔW` is a low-rank matrix.

During fine-tuning, the original weight matrix `W_0` remains frozen. Only the newly introduced matrices `A` and `B` are trained. The output of a layer then becomes `h = W_0 x + BA x`, where `x` is the input. A scaling factor `alpha/r` is often applied to `BA x` to normalize the impact of the low-rank updates.

#### 3.2. Mathematical Formulation
Consider a pre-trained layer's weight matrix `W_0`. When adapting this layer to a new task, LoRA modifies the output calculation. For an input `x`, the original calculation would be `h = W_0 x`. With LoRA, this becomes:

`h = W_0 x + ΔW x`

where `ΔW` is the update matrix. LoRA approximates `ΔW` using a low-rank decomposition:

`ΔW = B A`

Here, `A` is an `r x d_in` matrix, and `B` is a `d_out x r` matrix. The rank `r` is typically a small integer (e.g., 4, 8, 16). The total number of parameters introduced by LoRA for this layer is `d_in * r + d_out * r`, which is significantly less than `d_in * d_out` (the parameters in `W_0`).

During training, `W_0` is kept frozen, and only the matrices `A` and `B` are trainable. This drastically reduces the number of parameters requiring gradient updates. A common practice is to scale the `BA` output by a factor `alpha/r`, where `alpha` is another hyperparameter (often set to `r` or `2r`), to counteract the effect of `r` on the magnitude of the `ΔW` update:

`h = W_0 x + (alpha/r) B A x`

The matrix `A` is typically initialized with random Gaussian values, and `B` is initialized with zeros, ensuring that the initial `ΔW` is zero and the fine-tuning starts exactly from the pre-trained model's capabilities.

#### 3.3. Advantages of LoRA
LoRA offers several compelling advantages over full fine-tuning and other PEFT methods:

*   **Drastically Reduced Trainable Parameters:** LoRA reduces the number of trainable parameters by orders of magnitude (e.g., from billions to millions or even thousands). This significantly cuts down computational costs and memory usage during training.
*   **Faster Training and Lower Memory Footprint:** With fewer parameters to update, gradient computation and optimization are much faster. The reduced memory footprint allows for fine-tuning on less powerful hardware, or with larger batch sizes on existing hardware.
*   **No Additional Inference Latency:** After fine-tuning, the learned `BA` matrices can be merged directly into the original `W_0` matrix by calculating `W_new = W_0 + BA`. This means that during inference, the model operates with `W_new` and incurs **no additional computational cost or latency** compared to the original fine-tuned model. This is a significant advantage over methods that require running extra layers or modules during inference.
*   **Storage Efficiency:** Instead of storing a full copy of the fine-tuned model for each task, only the small `A` and `B` matrices (LoRA adapters) need to be stored. This saves immense storage space, especially when adapting a base model to many different downstream tasks.
*   **Preservation of Pre-trained Knowledge:** By freezing `W_0`, LoRA helps preserve the valuable knowledge encoded in the large pre-trained model, reducing the risk of catastrophic forgetting.
*   **Flexibility and Modularity:** LoRA adapters can be easily swapped in and out, allowing for dynamic task switching or combining multiple adapters for multi-task learning.

### 4. Code Example
This example demonstrates how to set up a model for parameter-efficient fine-tuning using LoRA with the Hugging Face `peft` library. It illustrates the configuration and the reduction in trainable parameters.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load a pre-trained model and tokenizer
# Using a small, illustrative model (e.g., distilgpt2) for demonstration purposes.
# In real-world scenarios, one would use much larger models like Llama-2, Mistral, etc.
model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Original model's total parameters: {model.num_parameters():,}")

# 2. Define LoRA configuration
# 'r' (rank) defines the dimension of the low-rank matrices (A and B).
# 'lora_alpha' is a scaling factor for the low-rank updates.
# 'target_modules' specifies which layers to apply LoRA to. For causal LMs,
# 'c_attn' (combined attention) or specific query/key/value projections are common.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # Specify the task type for appropriate PEFT wrapping
    inference_mode=False,        # Set to True for inference after training (merges weights)
    r=8,                         # LoRA attention dimension (rank)
    lora_alpha=16,               # Alpha parameter for LoRA scaling (often 2*r)
    lora_dropout=0.1,            # Dropout probability for LoRA layers
    target_modules=["c_attn"],   # Example: Apply LoRA to attention projection matrices
    bias="none",                 # Bias can be 'none', 'all', or 'lora_only'
)

# 3. Get the PEFT model
# This function freezes the original model weights and adds trainable LoRA adapters.
lora_model = get_peft_model(model, lora_config)

# 4. Print trainable parameters to demonstrate efficiency
# Observe that only a small fraction of the total parameters are now trainable.
print("\nTrainable parameters with LoRA:")
lora_model.print_trainable_parameters()

# 5. Example of model usage (conceptual)
# The LoRA model can now be fine-tuned like a regular model using standard training loops.
# Only the LoRA adapters (A and B matrices) will be updated during backpropagation.
# For actual fine-tuning, one would proceed with data loading, optimizer setup,
# and a standard PyTorch training loop or the Hugging Face Trainer.
prompt = "Parameter-Efficient Fine-Tuning with LoRA is a revolutionary technique because it"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text using the LoRA-enabled model
# (Note: Without actual fine-tuning, the generated text will reflect the base model's knowledge)
outputs = lora_model.generate(**inputs, max_new_tokens=20, num_return_sequences=1)
print("\nGenerated Text with LoRA model (pre-fine-tuning):")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

(End of code example section)
```

### 5. Conclusion
LoRA has emerged as a transformative technique in the landscape of Generative AI, offering an elegant and highly effective solution to the challenges of fine-tuning large pre-trained models. By intelligently leveraging the low-rank nature of weight updates, LoRA drastically reduces the number of trainable parameters, leading to faster training times, lower memory consumption, and significantly reduced storage requirements for task-specific adaptations. Its ability to achieve performance comparable to full fine-tuning, coupled with its zero-latency inference capabilities, makes it an invaluable tool for researchers and practitioners alike.

The success of LoRA underscores the importance of parameter-efficient methods in democratizing access to and accelerating the deployment of cutting-edge AI models. As foundation models continue to grow in size and complexity, techniques like LoRA will remain essential for enabling their widespread adoption, fostering innovation, and pushing the boundaries of what is possible in artificial intelligence. Its simplicity, efficiency, and strong empirical performance cement LoRA's position as a cornerstone of modern fine-tuning strategies for large language models and other generative architectures.

---
<br>

<a name="türkçe-içerik"></a>
## LoRA ile Parametre-Verimli İnce Ayar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Büyük Modelleri İnce Ayarlamanın Zorlukları](#2-büyük-modelleri-ince-ayarlamanın-zorlukları)
- [3. LoRA: Düşük Rank Adaptasyonu](#3-lora-düşük-rank-adaptasyonu)
  - [3.1. Temel İlke](#31-temel-ilke)
  - [3.2. Matematiksel Formülasyon](#32-matematiksel-formülasyon)
  - [3.3. LoRA'nın Avantajları](#33-loranın-avantajları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<br>

### 1. Giriş
Özellikle doğal dil işleme (NLP) ve bilgisayar görüşü alanlarında **büyük önceden eğitilmiş modellerin (BÖEM'ler)** ortaya çıkışı, yapay zekayı devrim niteliğinde değiştirdi. Genellikle milyarlarca parametre içeren bu modeller, çok çeşitli görevlerde dikkat çekici yetenekler sergilemektedir. Ancak, bu devasa modelleri belirli aşağı akış görevlerine uyarlamak genellikle **ince ayar** (fine-tuning) içerir; bu, tüm modelin veya parametrelerinin önemli bir kısmının göreve özel veriler kullanılarak güncellendiği bir süreçtir. Bu tam ince ayar yaklaşımı, özellikle kaynak kısıtlı ortamlar için veya tek bir modeli çok sayıda farklı göreve uyarlarken önemli hesaplama, bellek ve depolama zorlukları yaratır.

Bu sınırlamaları ele almak için **Parametre-Verimli İnce Ayar (PEFT)** teknikleri önemli bir paradigma olarak ortaya çıkmıştır. PEFT yöntemleri, tüm ince ayara benzer performans elde etmeyi hedeflerken, modelin parametrelerinin yalnızca küçük bir kısmını günceller. Bunlar arasında **Düşük Rank Adaptasyonu (LoRA)**, son derece etkili ve yaygın olarak benimsenen bir yaklaşım olarak öne çıkmaktadır. LoRA, önceden eğitilmiş modelin mevcut katmanlarına küçük, düşük ranklı matrisler enjekte ederek eğitilebilir parametre sayısını önemli ölçüde azaltır, böylece ince ayarı daha erişilebilir, daha hızlı ve daha az kaynak yoğun hale getirir. Bu belge, önde gelen bir PEFT stratejisi olarak LoRA'nın prensiplerini, mekanizmalarını ve faydalarını derinlemesine incelemektedir.

### 2. Büyük Modelleri İnce Ayarlamanın Zorlukları
GPT-3, LLaMA veya Stable Diffusion gibi büyük önceden eğitilmiş modelleri ince ayarlamak, potansiyel olarak yüz milyarlarca parametrenin ağırlıklarını güncellemeyi içerir. Bu süreç, birkaç önemli engeli beraberinde getirir:

*   **Hesaplama Maliyeti:** Büyük bir modeli eğitmek, genellikle uzun süreler boyunca birden fazla yüksek performanslı GPU gerektiren muazzam bir hesaplama gücü gerektirir. Tam ince ayar, her parametre için gradyanların yeniden hesaplanmasını gerektirir, bu da aşırı enerji tüketimine ve zamana yol açar.
*   **Bellek Ayak İzi:** Model ağırlıklarını, optimize edici durumlarını ve geri yayılım için aktivasyonları saklamak, büyük miktarda GPU belleği tüketir. Yüz milyarlarca parametreye sahip modeller için, modeli yüklemek bile standart donanımlardaki mevcut belleği tüketebilir, eğitim için gereken bellek bir yana.
*   **Depolama Gereksinimleri:** Belirli bir görev için özelleştirilmiş büyük bir modelin her ince ayarlı sürümü, model ağırlıklarının tam bir kopyasını depolamayı gerektirir. Bir kuruluşun temel bir modeli düzinelerce veya yüzlerce göreve uyarlaması gerektiğinde, bu hızla terabaytlarca hatta petabaytlarca depolama alanına yol açar ve dağıtım ile yönetimi hantal hale getirir.
*   **Yavaş Deney Yapma:** Tam ince ayarla ilişkili yüksek maliyet ve zaman, yapay zeka araştırması ve geliştirmesi için hayati önem taşıyan hızlı deney yapmayı ve iterasyonu engeller. Araştırmacıların genellikle birden fazla hipotezi, veri setini ve yapılandırmayı test etmesi gerekir; bu süreç, yavaş ince ayar döngüleri tarafından ciddi şekilde tıkanır.
*   **Felaket Unutma:** Tam ince ayar belirli görevlerde yüksek performans elde edebilse de, modelin yeni verilere aşırı özelleştirildiğinde ön eğitim sırasında öğrendiği genel yeteneklerinin bir kısmını kaybettiği **felaket unutma** riski vardır.

Bu zorluklar, daha verimli adaptasyon stratejilerine olan ihtiyacı vurgulamakta ve LoRA gibi PEFT yöntemlerini temel modeller çağında vazgeçilmez kılmaktadır.

### 3. LoRA: Düşük Rank Adaptasyonu
LoRA, ağırlık güncelleme matrislerinin düşük ranklı bir ayrıştırmasını tanıtarak, büyük önceden eğitilmiş modelleri uyarlamanın zorluklarını ele alan parametre-verimli bir ince ayar tekniğidir. LoRA, tüm parametreleri ince ayarlamak yerine, orijinal önceden eğitilmiş ağırlıkları dondurur ve eğitilebilir düşük ranklı ayrıştırma matrislerini dönüştürücü mimarisinin dikkat mekanizmasına (özellikle sorgu ve değer projeksiyon matrisleri) enjekte eder.

#### 3.1. Temel İlke
LoRA'nın temel fikri, ince ayar sırasında önceden eğitilmiş bir ağırlık matrisi `W_0`'a yapılan güncellemenin, `ΔW` olarak gösterilen, genellikle **düşük bir içsel ranka** sahip olmasıdır. Bu, `ΔW`'nin iki çok daha küçük matrisin çarpımıyla etkili bir şekilde yaklaştırabileceği anlamına gelir.

Özellikle, `W_0 ∈ R^(d_out × d_in)` ile temsil edilen herhangi bir yoğun katman için LoRA, `ΔW`'nin `BA` olarak yaklaştırılabileceğini öne sürer; burada `B ∈ R^(d_out × r)` ve `A ∈ R^(r × d_in)`. Burada, `r`, genellikle `min(d_in, d_out)` değerinden çok daha küçük olan bir hiperparametredir ve **LoRA rankı** olarak adlandırılır. Bu, `ΔW`'nin düşük ranklı bir matris olduğu anlamına gelir.

İnce ayar sırasında, orijinal ağırlık matrisi `W_0` dondurulur. Yalnızca yeni tanıtılan `A` ve `B` matrisleri eğitilir. Bir katmanın çıktısı daha sonra `h = W_0 x + BA x` olur, burada `x` girdidir. Düşük ranklı güncellemelerin etkisini normalleştirmek için `BA x`'e genellikle `alpha/r` ölçeklendirme faktörü uygulanır.

#### 3.2. Matematiksel Formülasyon
Önceden eğitilmiş bir katmanın ağırlık matrisi `W_0`'ı ele alalım. Bu katmanı yeni bir göreve uyarlarken, LoRA çıktı hesaplamasını değiştirir. `x` girdisi için orijinal hesaplama `h = W_0 x` olacaktır. LoRA ile bu şöyle olur:

`h = W_0 x + ΔW x`

burada `ΔW` güncelleme matrisidir. LoRA, `ΔW`'yi düşük ranklı bir ayrıştırma kullanarak yaklaştırır:

`ΔW = B A`

Burada, `A`, `r x d_in` boyutunda bir matris ve `B`, `d_out x r` boyutunda bir matristir. `r` rankı tipik olarak küçük bir tam sayıdır (örneğin, 4, 8, 16). Bu katman için LoRA tarafından tanıtılan toplam parametre sayısı `d_in * r + d_out * r`'dir; bu, `d_in * d_out` ( `W_0`'daki parametreler) değerinden önemli ölçüde daha azdır.

Eğitim sırasında, `W_0` dondurulur ve yalnızca `A` ve `B` matrisleri eğitilebilir durumdadır. Bu, gradyan güncellemeleri gerektiren parametre sayısını önemli ölçüde azaltır. Ortak bir uygulama, `BA` çıktısını `alpha/r` faktörüyle ölçeklendirmektir; burada `alpha` başka bir hiperparametredir (genellikle `r` veya `2r` olarak ayarlanır), `r`'nin `ΔW` güncellemesinin büyüklüğü üzerindeki etkisini dengelemek için:

`h = W_0 x + (alpha/r) B A x`

`A` matrisi tipik olarak rastgele Gauss değerleriyle başlatılır ve `B` sıfırlarla başlatılır, bu da başlangıçtaki `ΔW`'nin sıfır olmasını ve ince ayarın tam olarak önceden eğitilmiş modelin yeteneklerinden başlamasını sağlar.

#### 3.3. LoRA'nın Avantajları
LoRA, tam ince ayara ve diğer PEFT yöntemlerine göre birkaç zorlayıcı avantaj sunar:

*   **Drasik Olarak Azaltılmış Eğitilebilir Parametreler:** LoRA, eğitilebilir parametre sayısını katlarca (örneğin, milyarlardan milyonlara veya hatta binlere) azaltır. Bu, eğitim sırasında hesaplama maliyetlerini ve bellek kullanımını önemli ölçüde düşürür.
*   **Daha Hızlı Eğitim ve Daha Düşük Bellek Ayak İzi:** Güncellenecek daha az parametreyle, gradyan hesaplaması ve optimizasyon çok daha hızlıdır. Azaltılmış bellek ayak izi, daha az güçlü donanımlarda veya mevcut donanımlarda daha büyük toplu iş boyutlarıyla ince ayar yapılmasına olanak tanır.
*   **Ek Çıkarım Gecikmesi Yok:** İnce ayardan sonra, öğrenilen `BA` matrisleri, `W_new = W_0 + BA` hesaplanarak doğrudan orijinal `W_0` matrisine birleştirilebilir. Bu, çıkarım sırasında modelin `W_new` ile çalıştığı ve orijinal ince ayarlı modele kıyasla **ek hesaplama maliyeti veya gecikme** yaşamadığı anlamına gelir. Bu, çıkarım sırasında ekstra katmanlar veya modüller çalıştırmayı gerektiren yöntemlere göre önemli bir avantajdır.
*   **Depolama Verimliliği:** Her görev için ince ayarlı modelin tam bir kopyasını depolamak yerine, yalnızca küçük `A` ve `B` matrislerinin (LoRA adaptörleri) depolanması gerekir. Bu, özellikle temel bir modeli birçok farklı aşağı akış görevine uyarlarken büyük miktarda depolama alanı tasarrufu sağlar.
*   **Önceden Eğitilmiş Bilginin Korunması:** `W_0`'ı dondurarak, LoRA, büyük önceden eğitilmiş modelde kodlanmış değerli bilginin korunmasına yardımcı olur ve felaket unutma riskini azaltır.
*   **Esneklik ve Modülerlik:** LoRA adaptörleri kolayca değiştirilebilir, bu da dinamik görev geçişine veya çoklu görev öğrenimi için birden fazla adaptörün birleştirilmesine olanak tanır.

### 4. Kod Örneği
Bu örnek, Hugging Face `peft` kütüphanesi ile LoRA kullanarak parametre-verimli ince ayar için bir modelin nasıl kurulacağını gösterir. Yapılandırmayı ve eğitilebilir parametrelerdeki azalmayı açıklar.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Önceden eğitilmiş bir model ve tokenizer yükleyin
# Gösterim amacıyla küçük, açıklayıcı bir model (örneğin, distilgpt2) kullanılıyor.
# Gerçek dünya senaryolarında, Llama-2, Mistral gibi çok daha büyük modeller kullanılacaktır.
model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Orijinal modelin toplam parametreleri: {model.num_parameters():,}")

# 2. LoRA yapılandırmasını tanımlayın
# 'r' (rank), düşük ranklı matrislerin (A ve B) boyutunu tanımlar.
# 'lora_alpha', düşük ranklı güncellemeler için bir ölçeklendirme faktörüdür.
# 'target_modules', LoRA'nın hangi katmanlara uygulanacağını belirtir. Nedensel LM'ler için,
# 'c_attn' (birleşik dikkat) veya belirli sorgu/anahtar/değer projeksiyonları yaygındır.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # Uygun PEFT sarmalaması için görev türünü belirtin
    inference_mode=False,        # Eğitim sonrası çıkarım için True olarak ayarlayın (ağırlıkları birleştirir)
    r=8,                         # LoRA dikkat boyutu (rank)
    lora_alpha=16,               # LoRA ölçeklendirme için Alfa parametresi (genellikle 2*r)
    lora_dropout=0.1,            # LoRA katmanları için dropout olasılığı
    target_modules=["c_attn"],   # Örnek: Dikkat projeksiyon matrislerine LoRA uygulayın
    bias="none",                 # Bias 'none', 'all' veya 'lora_only' olabilir
)

# 3. PEFT modelini alın
# Bu fonksiyon, orijinal model ağırlıklarını dondurur ve eğitilebilir LoRA adaptörleri ekler.
lora_model = get_peft_model(model, lora_config)

# 4. Verimliliği göstermek için eğitilebilir parametreleri yazdırın
# Toplam parametrelerin yalnızca küçük bir kısmının artık eğitilebilir olduğunu gözlemleyin.
print("\nLoRA ile eğitilebilir parametreler:")
lora_model.print_trainable_parameters()

# 5. Model kullanımına örnek (kavramsal)
# LoRA modeli, standart eğitim döngüleri kullanılarak artık normal bir model gibi ince ayarlanabilir.
# Geri yayılım sırasında yalnızca LoRA adaptörleri (A ve B matrisleri) güncellenecektir.
# Gerçek ince ayar için veri yükleme, optimizer kurulumu ve
# standart bir PyTorch eğitim döngüsü veya Hugging Face Trainer ile devam edilecektir.
prompt = "LoRA ile Parametre-Verimli İnce Ayar devrim niteliğinde bir tekniktir çünkü"
inputs = tokenizer(prompt, return_tensors="pt")

# LoRA etkin modelini kullanarak metin oluşturma
# (Not: Gerçek bir ince ayar olmadan, oluşturulan metin temel modelin bilgisini yansıtacaktır)
outputs = lora_model.generate(**inputs, max_new_tokens=20, num_return_sequences=1)
print("\nLoRA modeliyle oluşturulan metin (ince ayar öncesi):")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
LoRA, Üretken Yapay Zeka alanında dönüştürücü bir teknik olarak ortaya çıkmış, büyük önceden eğitilmiş modelleri ince ayarlamanın zorluklarına zarif ve son derece etkili bir çözüm sunmuştur. Ağırlık güncellemelerinin düşük ranklı doğasını akıllıca kullanarak, LoRA eğitilebilir parametre sayısını önemli ölçüde azaltır, bu da daha hızlı eğitim sürelerine, daha düşük bellek tüketimine ve göreve özel adaptasyonlar için önemli ölçüde azaltılmış depolama gereksinimlerine yol açar. Tam ince ayara benzer performans elde etme yeteneği, sıfır gecikmeli çıkarım yetenekleriyle birleştiğinde, onu hem araştırmacılar hem de uygulayıcılar için paha biçilmez bir araç haline getirir.

LoRA'nın başarısı, en son yapay zeka modellerine erişimi demokratikleştirme ve dağıtımını hızlandırmada parametre-verimli yöntemlerin önemini vurgulamaktadır. Temel modeller boyut ve karmaşıklık açısından büyümeye devam ettikçe, LoRA gibi teknikler, yaygın olarak benimsenmelerini sağlamak, yeniliği teşvik etmek ve yapay zekada mümkün olanın sınırlarını zorlamak için temel olmaya devam edecektir. Basitliği, verimliliği ve güçlü ampirik performansı, LoRA'nın büyük dil modelleri ve diğer üretken mimariler için modern ince ayar stratejilerinin temel taşı konumunu sağlamlaştırmaktadır.
