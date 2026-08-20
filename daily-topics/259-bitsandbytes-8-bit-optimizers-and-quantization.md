# BitsAndBytes: 8-bit Optimizers and Quantization

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Quantization with BitsAndBytes](#2-understanding-quantization-with-bitsandbytes)
  - [2.1. Quantization Fundamentals](#21-quantization-fundamentals)
  - [2.2. BitsAndBytes Quantization Techniques](#22-bitsandbytes-quantization-techniques)
- [3. 8-bit Optimizers](#3-8-bit-optimizers)
  - [3.1. The Challenge of Optimizer States](#31-the-challenge-of-optimizer-states)
  - [3.2. How BitsAndBytes Addresses This](#32-how-bitsandbytes-addresses-this)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

### 1. Introduction
<a name="1-introduction"></a>
The rapid advancement in **Generative AI** has led to the development of increasingly large and complex neural network models, particularly **Large Language Models (LLMs)**. While these models demonstrate unprecedented capabilities, their sheer size poses significant challenges in terms of computational resources, memory consumption, and deployment costs. Training and running models with billions of parameters often require specialized hardware and substantial financial investment, thereby limiting accessibility.

**BitsAndBytes** emerges as a crucial library designed to mitigate these challenges by enabling efficient execution of large models on more constrained hardware. It achieves this primarily through **quantization** and the implementation of **8-bit optimizers**. This document delves into the technical underpinnings of BitsAndBytes, exploring how its innovative approaches to reducing numerical precision facilitate the democratization of state-of-the-art AI models.

### 2. Understanding Quantization with BitsAndBytes
<a name="2-understanding-quantization-with-bitsandbytes"></a>

#### 2.1. Quantization Fundamentals
**Quantization** in the context of neural networks refers to the process of reducing the number of bits required to represent a numerical value, typically weights and activations. Most deep learning models are trained using 32-bit floating-point numbers (**FP32**), offering high precision. However, this precision comes at the cost of significant memory footprint and computational intensity.

By reducing the bit-width (e.g., from 32-bit to 8-bit integers or even 4-bit integers), several benefits are realized:
*   **Reduced Memory Usage**: Less memory is required to store model parameters and intermediate activations.
*   **Faster Computation**: Operations on lower-precision data types can often be executed more quickly by specialized hardware.
*   **Lower Power Consumption**: Reduced memory access and computation can lead to lower energy requirements.

The primary challenge in quantization is to achieve these benefits while minimizing the degradation of model performance. Naive quantization can lead to significant accuracy loss.

#### 2.2. BitsAndBytes Quantization Techniques
<a name="22-bitsandbytes-quantization-techniques"></a>
BitsAndBytes implements sophisticated quantization schemes that allow models to run with significantly reduced memory without substantial performance compromise. Key techniques include:

*   **8-bit Quantization**: BitsAndBytes pioneered stable 8-bit matrix multiplication, which is foundational for efficient 8-bit inference and training. This involves mapping the range of FP32 values to an 8-bit integer range.
*   **Dynamic Quantization**: Unlike static quantization where the scaling factor is determined once for the entire tensor, dynamic quantization computes the scaling factor on the fly for activations. This is particularly useful for activations which might have a wide dynamic range across different inputs.
*   **4-bit NormalFloat (NF4)**: This is a novel data type introduced by BitsAndBytes, specifically designed for normal distributions, which are common for weights in pre-trained neural networks. NF4 is an empirically optimal fixed-point data type that is theoretically information-optimal for normally distributed data. It quantizes 4-bit data to 2^4-1 = 15 values (excluding zero) by using a quantile-based quantization scheme, preserving more information than standard uniform 4-bit quantization.
*   **QLoRA (Quantized Low-Rank Adaptation)**: QLoRA leverages BitsAndBytes' 4-bit quantization to quantize a pre-trained language model to 4-bits and then attaches small, trainable adapter layers (LoRA modules). During fine-tuning, only the LoRA adapters are trained, while the vast majority of the 4-bit pre-trained model parameters remain frozen. This significantly reduces memory requirements for fine-tuning large models while retaining much of their original performance.

### 3. 8-bit Optimizers
<a name="3-8-bit-optimizers"></a>

#### 3.1. The Challenge of Optimizer States
Beyond model parameters themselves, a significant portion of GPU memory during training is consumed by the **optimizer states**. Optimizers like **Adam** and **AdamW**, widely used in deep learning, maintain internal state variables (e.g., first and second moments of gradients) for each parameter of the model. For a typical FP32 model, an Adam optimizer requires 12 bytes per parameter (4 bytes for the parameter itself, 4 bytes for the first moment, and 4 bytes for the second moment). For a billion-parameter model, this translates to 12 GB of memory just for the optimizer state, which can quickly exceed the capacity of consumer-grade GPUs.

#### 3.2. How BitsAndBytes Addresses This
<a name="32-how-bitsandbytes-addresses-this"></a>
BitsAndBytes introduces **8-bit optimizers** which drastically reduce the memory footprint of these optimizer states without compromising training stability or convergence. The core idea is to quantize the optimizer states (first and second moments) to 8-bit floating-point numbers instead of 32-bit.

*   **Dynamic Quantization for States**: The optimizer states are dynamically quantized to 8-bit. This means the scaling factors for quantization are computed dynamically for each parameter group and each training step, allowing the optimizer to adapt to varying ranges of momentum values.
*   **Mixed-Precision Gradients**: While optimizer states are 8-bit, gradients are still computed in FP32 or FP16 (if using automatic mixed precision), providing the necessary precision for accurate weight updates. The 8-bit states are then used to update the full-precision weights.
*   **Specific Implementations**: BitsAndBytes provides 8-bit versions of popular optimizers such as `optim.Adam8bit`, `optim.AdamW8bit`, and `optim.Lion8bit`. These can often be used as drop-in replacements for their 32-bit counterparts in existing training pipelines.

By reducing the optimizer state from 8 bytes per parameter to 2 bytes (for two 8-bit states), BitsAndBytes can reduce the total memory footprint for optimizer states by approximately 75%, making it feasible to fine-tune massive models on much more accessible hardware. This enables researchers and developers to iterate faster and experiment with larger models without requiring state-of-the-art GPU clusters.

### 4. Code Example
This example demonstrates loading a pre-trained model and then initializing an 8-bit optimizer from BitsAndBytes.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import optim

# 1. Load a pre-trained model (e.g., a small language model)
# For larger models, you would typically load them with load_in_8bit=True or load_in_4bit=True
model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Prepare dummy input and target for demonstration
inputs = tokenizer("Hello, my name is", return_tensors="pt")
labels = tokenizer("Hello, my name is John.", return_tensors="pt")['input_ids']

# 3. Initialize an 8-bit optimizer (e.g., AdamW8bit)
# This optimizer will manage the model's parameters in a memory-efficient 8-bit format
# Note: Ensure all model parameters require gradients for the optimizer to include them
optimizer = optim.AdamW8bit(model.parameters(), lr=5e-5)

# 4. Perform a dummy training step
# In a real scenario, this would be inside a training loop
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"Loss after one step with AdamW8bit: {loss.item()}")
print(f"Optimizer type: {type(optimizer)}")

# (Optional) Example of loading a quantized model (e.g., for QLoRA)
# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# model_quantized = AutoModelForCausalLM.from_pretrained(
#     "facebook/opt-125m", 
#     quantization_config=bnb_config
# )
# print(f"Quantized model dtype: {model_quantized.dtype}")

(End of code example section)
```

### 5. Conclusion
<a name="5-conclusion"></a>
BitsAndBytes represents a pivotal advancement in making cutting-edge Generative AI models more accessible and resource-efficient. Through its innovative **8-bit and 4-bit quantization techniques**, including the **NF4** data type and the **QLoRA** approach, it dramatically reduces the memory footprint of large language models during both inference and fine-tuning. Furthermore, its **8-bit optimizers** effectively tackle the substantial memory overhead typically associated with optimizer states, enabling the training of colossal models on consumer-grade GPUs. The library's ability to maintain high performance with significantly reduced computational demands is democratizing access to powerful AI models, fostering broader research, development, and application across diverse hardware environments. BitsAndBytes is not merely an optimization library; it is a catalyst for innovation in an era dominated by ever-growing AI models.

---
<br>

<a name="türkçe-içerik"></a>
## BitsAndBytes: 8-bit İyileştiriciler ve Nicemleme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. BitsAndBytes ile Nicemlemeyi Anlamak](#2-bitsandbytes-ile-nicemlemeyi-anlamak)
  - [2.1. Nicemleme Temelleri](#21-nicemleme-temelleri)
  - [2.2. BitsAndBytes Nicemleme Teknikleri](#22-bitsandbytes-nicemleme-teknikleri)
- [3. 8-bit İyileştiriciler](#3-8-bit-iyileştiriciler)
  - [3.1. İyileştirici Durumlarının Zorluğu](#31-iyileştirici-durumlarının-zorluğu)
  - [3.2. BitsAndBytes Bunu Nasıl Ele Alır](#32-bitsandbytes-bunu-nasıl-ele-alır)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

### 1. Giriş
<a name="1-giriş"></a>
**Üretken Yapay Zeka (Generative AI)** alanındaki hızlı ilerlemeler, özellikle **Büyük Dil Modelleri (LLM'ler)** olmak üzere giderek daha büyük ve karmaşık sinir ağı modellerinin geliştirilmesine yol açmıştır. Bu modeller eşi benzeri görülmemiş yetenekler sergilerken, büyük boyutları hesaplama kaynakları, bellek tüketimi ve dağıtım maliyetleri açısından önemli zorluklar ortaya koymaktadır. Milyarlarca parametreye sahip modelleri eğitmek ve çalıştırmak genellikle özel donanım ve önemli mali yatırım gerektirir, bu da erişilebilirliği sınırlar.

**BitsAndBytes**, daha kısıtlı donanımlarda büyük modellerin verimli bir şekilde yürütülmesini sağlayarak bu zorlukları azaltmak için tasarlanmış önemli bir kütüphane olarak ortaya çıkmaktadır. Bunu, temel olarak **nicemleme (quantization)** ve **8-bit iyileştiricilerin** uygulanması yoluyla başarır. Bu belge, BitsAndBytes'in teknik temellerini inceleyerek, sayısal hassasiyeti azaltmaya yönelik yenilikçi yaklaşımlarının son teknoloji yapay zeka modellerinin demokratikleşmesini nasıl kolaylaştırdığını keşfedecektir.

### 2. BitsAndBytes ile Nicemlemeyi Anlamak
<a name="2-bitsandbytes-ile-nicemlemeyi-anlamak"></a>

#### 2.1. Nicemleme Temelleri
Sinir ağları bağlamında **nicemleme**, tipik olarak ağırlıklar ve aktivasyonlar olmak üzere sayısal bir değeri temsil etmek için gereken bit sayısını azaltma sürecini ifade eder. Çoğu derin öğrenme modeli, yüksek hassasiyet sunan 32-bit kayan nokta sayıları (**FP32**) kullanılarak eğitilir. Ancak bu hassasiyet, önemli bellek alanı ve hesaplama yoğunluğu maliyetine sahiptir.

Bit genişliğini (örneğin, 32-bit'ten 8-bit tam sayılara veya hatta 4-bit tam sayılara) azaltarak çeşitli faydalar elde edilir:
*   **Azaltılmış Bellek Kullanımı**: Model parametrelerini ve ara aktivasyonları depolamak için daha az bellek gerekir.
*   **Daha Hızlı Hesaplama**: Daha düşük hassasiyetli veri türlerindeki işlemler, özel donanımlar tarafından genellikle daha hızlı yürütülebilir.
*   **Daha Düşük Güç Tüketimi**: Azaltılmış bellek erişimi ve hesaplama, daha düşük enerji gereksinimlerine yol açabilir.

Nicemlemedeki temel zorluk, model performansının bozulmasını en aza indirirken bu faydaları elde etmektir. Saf nicemleme, önemli doğruluk kaybına yol açabilir.

#### 2.2. BitsAndBytes Nicemleme Teknikleri
<a name="22-bitsandbytes-nicemleme-teknikleri"></a>
BitsAndBytes, modellerin önemli bir performans ödünü vermeden önemli ölçüde azaltılmış bellekle çalışmasını sağlayan gelişmiş nicemleme şemalarını uygular. Temel teknikler şunları içerir:

*   **8-bit Nicemleme**: BitsAndBytes, verimli 8-bit çıkarım ve eğitim için temel olan kararlı 8-bit matris çarpımına öncülük etmiştir. Bu, FP32 değerlerinin aralığını 8-bit bir tam sayı aralığına eşlemeyi içerir.
*   **Dinamik Nicemleme**: Ölçekleme faktörünün tüm tensör için bir kez belirlendiği statik nicemlemenin aksine, dinamik nicemleme, aktivasyonlar için ölçekleme faktörünü anında hesaplar. Bu, farklı girişler arasında geniş bir dinamik aralığa sahip olabilecek aktivasyonlar için özellikle kullanışlıdır.
*   **4-bit NormalFloat (NF4)**: Bu, önceden eğitilmiş sinir ağlarındaki ağırlıklar için yaygın olan normal dağılımlar için özel olarak tasarlanmış, BitsAndBytes tarafından tanıtılan yeni bir veri türüdür. NF4, normal dağılmış veriler için teorik olarak bilgi açısından optimal olan ampirik olarak optimal bir sabit nokta veri türüdür. Quantile tabanlı bir nicemleme şeması kullanarak 4-bit veriyi 2^4-1 = 15 değere (sıfır hariç) nicemleyerek, standart tekdüze 4-bit nicemlemeden daha fazla bilgi korur.
*   **QLoRA (Nicemlenmiş Düşük Dereceli Adaptasyon)**: QLoRA, önceden eğitilmiş bir dil modelini 4-bit'e nicemlemek ve ardından küçük, eğitilebilir adaptör katmanları (LoRA modülleri) eklemek için BitsAndBytes'in 4-bit nicemlemesini kullanır. İnce ayar sırasında, sadece LoRA adaptörleri eğitilirken, 4-bit önceden eğitilmiş model parametrelerinin büyük çoğunluğu dondurulmuş kalır. Bu, büyük modelleri ince ayarlamak için bellek gereksinimlerini önemli ölçüde azaltırken, orijinal performanslarının çoğunu korur.

### 3. 8-bit İyileştiriciler
<a name="3-8-bit-iyileştiriciler"></a>

#### 3.1. İyileştirici Durumlarının Zorluğu
Model parametrelerinin kendisinin ötesinde, eğitim sırasında GPU belleğinin önemli bir kısmı **iyileştirici durumları** tarafından tüketilir. Derin öğrenmede yaygın olarak kullanılan **Adam** ve **AdamW** gibi iyileştiriciler, modelin her parametresi için dahili durum değişkenleri (örneğin, gradyanların birinci ve ikinci momentleri) tutar. Tipik bir FP32 modeli için, bir Adam iyileştiricisi parametre başına 12 bayt (parametrenin kendisi için 4 bayt, birinci moment için 4 bayt ve ikinci moment için 4 bayt) gerektirir. Milyar parametreli bir model için bu, sadece iyileştirici durumu için 12 GB belleğe dönüşür ve bu da tüketici sınıfı GPU'ların kapasitesini hızla aşabilir.

#### 3.2. BitsAndBytes Bunu Nasıl Ele Alır
<a name="32-bitsandbytes-bunu-nasıl-ele-alır"></a>
BitsAndBytes, eğitim kararlılığını veya yakınsamasını tehlikeye atmadan bu iyileştirici durumlarının bellek ayak izini önemli ölçüde azaltan **8-bit iyileştiriciler** sunar. Temel fikir, iyileştirici durumlarını (birinci ve ikinci momentleri) 32-bit yerine 8-bit kayan nokta sayılarına nicemlemektir.

*   **Durumlar için Dinamik Nicemleme**: İyileştirici durumları dinamik olarak 8-bit'e nicemlenir. Bu, nicemleme için ölçekleme faktörlerinin her parametre grubu ve her eğitim adımı için dinamik olarak hesaplandığı anlamına gelir, bu da iyileştiricinin değişen momentum değerleri aralıklarına uyum sağlamasına olanak tanır.
*   **Karma Hassasiyetli Gradyanlar**: İyileştirici durumları 8-bit olsa da, gradyanlar hala FP32 veya FP16'da (otomatik karma hassasiyet kullanılıyorsa) hesaplanır ve doğru ağırlık güncellemeleri için gerekli hassasiyeti sağlar. Tam hassasiyetli ağırlıkları güncellemek için 8-bit durumlar kullanılır.
*   **Spesifik Uygulamalar**: BitsAndBytes, `optim.Adam8bit`, `optim.AdamW8bit` ve `optim.Lion8bit` gibi popüler iyileştiricilerin 8-bit sürümlerini sağlar. Bunlar, mevcut eğitim pipeline'larındaki 32-bit karşılıkları için genellikle doğrudan birer ikame olarak kullanılabilir.

İyileştirici durumunu parametre başına 8 bayttan 2 bayta (iki adet 8-bit durum için) düşürerek, BitsAndBytes iyileştirici durumları için toplam bellek ayak izini yaklaşık %75 oranında azaltabilir, bu da büyük modellerin çok daha erişilebilir donanımlarda ince ayar yapılmasını mümkün kılar. Bu, araştırmacıların ve geliştiricilerin daha hızlı yineleme yapmalarına ve son teknoloji GPU kümeleri gerektirmeden daha büyük modellerle deneme yapmalarına olanak tanır.

### 4. Kod Örneği
Bu örnek, önceden eğitilmiş bir modelin yüklenmesini ve ardından BitsAndBytes'ten bir 8-bit iyileştiricinin başlatılmasını göstermektedir.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import optim

# 1. Önceden eğitilmiş bir model yükle (örn. küçük bir dil modeli)
# Daha büyük modeller için, genellikle load_in_8bit=True veya load_in_4bit=True ile yüklenirler
model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Gösterim için sahte giriş ve hedef hazırla
inputs = tokenizer("Merhaba, benim adım", return_tensors="pt")
labels = tokenizer("Merhaba, benim adım Can.", return_tensors="pt")['input_ids']

# 3. Bir 8-bit iyileştirici başlat (örn. AdamW8bit)
# Bu iyileştirici, modelin parametrelerini bellek açısından verimli 8-bit formatında yönetecek
# Not: İyileştiricinin tüm model parametrelerini içermesi için gradyan gerektirmesi gerekir
optimizer = optim.AdamW8bit(model.parameters(), lr=5e-5)

# 4. Sahte bir eğitim adımı gerçekleştir
# Gerçek bir senaryoda, bu bir eğitim döngüsünün içinde olacaktır
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"AdamW8bit ile bir adımdan sonraki kayıp: {loss.item()}")
print(f"İyileştirici tipi: {type(optimizer)}")

# (Opsiyonel) Nicemlenmiş bir model yükleme örneği (örn. QLoRA için)
# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# model_quantized = AutoModelForCausalLM.from_pretrained(
#     "facebook/opt-125m", 
#     quantization_config=bnb_config
# )
# print(f"Nicemlenmiş model dtype: {model_quantized.dtype}")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
<a name="5-sonuç"></a>
BitsAndBytes, son teknoloji Üretken Yapay Zeka modellerini daha erişilebilir ve kaynak açısından daha verimli hale getirmede önemli bir ilerlemeyi temsil etmektedir. **NF4** veri türü ve **QLoRA** yaklaşımı da dahil olmak üzere yenilikçi **8-bit ve 4-bit nicemleme teknikleri** aracılığıyla, çıkarım ve ince ayar sırasında büyük dil modellerinin bellek ayak izini önemli ölçüde azaltır. Ayrıca, **8-bit iyileştiricileri**, genellikle iyileştirici durumlarıyla ilişkili önemli bellek yükünü etkin bir şekilde ele alarak, tüketici sınıfı GPU'larda devasa modellerin eğitilmesini mümkün kılar. Kütüphanenin, önemli ölçüde azaltılmış hesaplama talepleriyle yüksek performansı sürdürme yeteneği, güçlü yapay zeka modellerine erişimi demokratikleştirerek çeşitli donanım ortamlarında daha geniş araştırma, geliştirme ve uygulamayı teşvik etmektedir. BitsAndBytes sadece bir optimizasyon kütüphanesi değil; sürekli büyüyen yapay zeka modellerinin egemen olduğu bir çağda inovasyon için bir katalizördür.
