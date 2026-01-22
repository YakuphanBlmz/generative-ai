# BitsAndBytes: 8-bit Optimizers and Quantization

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Quantization in Deep Learning](#2-understanding-quantization-in-deep-learning)
- [3. The BitsAndBytes Library and 8-bit Optimizers](#3-the-bitsandbytes-library-and-8-bit-optimizers)
    - [3.1. Core Concepts and Benefits](#31-core-concepts-and-benefits)
    - [3.2. Technical Mechanisms](#32-technical-mechanisms)
- [4. Practical Implementation with BitsAndBytes](#4-practical-implementation-with-bitsandbytes)
- [5. Conclusion](#5-conclusion)

---

<a name="1-introduction"></a>
## 1. Introduction

The rapid advancement of deep learning, particularly in the domain of large language models (LLMs) and complex neural architectures, has led to a significant increase in model size and computational demands. These models often require vast amounts of memory for both their parameters and the optimizer states during training, making them inaccessible to researchers and practitioners with limited hardware resources. Training models with billions of parameters typically necessitates high-end accelerators, posing a substantial barrier to entry. **Quantization**, a technique that reduces the precision of numerical representations, has emerged as a critical strategy to mitigate these challenges. By representing parameters and activations with fewer bits, quantization can drastically reduce memory footprint, improve computational efficiency, and accelerate both training and inference processes.

The **BitsAndBytes** library, a lightweight wrapper around custom CUDA functions, has revolutionized the accessibility of large models by providing robust and efficient implementations of 8-bit quantization and optimization. This document will delve into the principles of 8-bit optimizers and quantization, exploring how BitsAndBytes leverages these techniques to enable the training and deployment of large-scale deep learning models on more modest hardware configurations. We will examine the underlying mechanisms, practical benefits, and provide a illustrative code example for its application.

<a name="2-understanding-quantization-in-deep-learning"></a>
## 2. Understanding Quantization in Deep Learning

**Quantization** is a process of mapping continuous values or values from a large set to a finite, smaller set of values. In the context of deep learning, it primarily involves reducing the bit-width of the numerical representation of model weights, activations, and optimizer states. Most deep learning models are trained using 32-bit floating-point numbers (**FP32**), which offer high precision but consume significant memory. Quantization aims to replace these FP32 representations with lower-precision formats, such as 16-bit floating-point (**FP16** or **BF16**) or 8-bit integers (**INT8**).

The primary motivations for quantization are:
*   **Memory Reduction:** Lower bit-width numbers require less storage. For instance, converting from FP32 to INT8 can reduce memory usage by 75% for the quantized elements. This allows larger models to fit into GPU memory, or smaller models to be run with larger batch sizes.
*   **Computational Efficiency:** Operations on lower-precision data often execute faster on modern hardware, especially specialized hardware accelerators designed for INT8 computations. This can lead to faster training and inference times.
*   **Energy Efficiency:** Reduced memory access and computation can also lead to lower power consumption, which is particularly relevant for edge devices and large-scale data centers.

While various quantization strategies exist, including **post-training quantization (PTQ)** where models are quantized after training, and **quantization-aware training (QAT)** where quantization is simulated during training, BitsAndBytes focuses on a specific application: quantizing optimizer states and, more recently, model weights themselves for efficient inference (e.g., LLM.int8() and QLoRA). The challenge with aggressive quantization, especially down to 8-bit, is to maintain model accuracy. Errors introduced by reduced precision can accumulate and degrade performance. BitsAndBytes tackles this by employing advanced techniques like **dynamic quantization** and **quantile-based scaling** to preserve numerical stability.

<a name="3-the-bitsandbytes-library-and-8-bit-optimizers"></a>
## 3. The BitsAndBytes Library and 8-bit Optimizers

The **BitsAndBytes** library has emerged as a cornerstone for democratizing large-scale deep learning, particularly for training models that previously demanded prohibitive hardware. It achieves this primarily through its innovative **8-bit optimizers** and efficient low-precision matrix multiplication routines.

<a name="31-core-concepts-and-benefits"></a>
### 3.1. Core Concepts and Benefits

At its heart, BitsAndBytes provides highly optimized custom CUDA kernels that enable operations on low-precision data types with minimal overhead. Key functionalities include:

*   **8-bit Optimizers:** The library offers 8-bit variants of popular optimizers like Adam, AdamW, and SGD. During training, optimizer states (e.g., moments for Adam) can consume a significant portion of GPU memory, often 2 to 4 times the size of the model parameters themselves (e.g., Adam stores two states per parameter, each in FP32). By quantizing these optimizer states to 8-bit, BitsAndBytes can reduce their memory footprint by up to 75%. This memory saving is crucial for fitting larger models or larger batch sizes onto a given GPU.
*   **Dynamic Quantization:** Unlike static quantization, where scaling factors are fixed, BitsAndBytes employs a **dynamic quantization** approach for optimizer states. This means the quantization parameters (e.g., min/max values for scaling) are recomputed on-the-fly for each tensor at each optimization step. This adaptive approach helps maintain numerical stability and precision, especially when gradient distributions change significantly during training.
*   **Quantile Quantization:** To handle the potentially wide range of values in gradients and optimizer states robustly, BitsAndBytes utilizes **quantile quantization**. Instead of simple min-max scaling, which can be sensitive to outliers, quantile quantization maps values based on their distribution, effectively ignoring extreme outliers and providing a more stable and accurate mapping to the 8-bit range.
*   **Efficient Mixed-Precision Training:** While optimizer states are 8-bit, BitsAndBytes often performs gradient updates and critical calculations in a higher precision (e.g., FP32 or BF16) to ensure accuracy, leveraging the benefits of **mixed-precision training**. This hybrid approach balances memory efficiency with numerical stability.
*   **4-bit and 8-bit Matrix Multiplication (GEMM):** Beyond optimizers, BitsAndBytes also provides highly optimized kernels for low-precision matrix multiplication. This is fundamental for its more recent applications like **LLM.int8()** quantization for inference and **QLoRA**, which allows for training massive language models (e.g., 65B parameters) on a single consumer GPU by quantizing model weights to 4-bit or 8-bit.

The overarching benefit of BitsAndBytes is the **democratization of large model training and inference**. It allows researchers and developers to experiment with and deploy models that were previously out of reach due to hardware constraints, significantly broadening access to state-of-the-art AI.

<a name="32-technical-mechanisms"></a>
### 3.2. Technical Mechanisms

The core mechanism behind BitsAndBytes' 8-bit optimizers involves storing the optimizer states (e.g., `exp_avg`, `exp_avg_sq` for Adam) in an 8-bit integer format. When these states are needed for a gradient update, they are **dequantized** back to a higher precision (e.g., FP32), used in the computation, and then **re-quantized** back to 8-bit for storage. This dequantize-compute-quantize cycle is meticulously optimized within custom CUDA kernels to minimize latency.

For the actual quantization, BitsAndBytes employs a **block-wise dynamic quantization** approach. Instead of quantizing the entire tensor with a single scaling factor, tensors are often divided into smaller blocks (e.g., 256 or 512 elements). Each block then gets its own dynamic scaling factor derived from the statistics within that block. This local scaling significantly improves the precision of the 8-bit representation, as it can adapt to local variations in the data distribution more effectively than a global scale. The use of **quantile-based scaling** further refines this by making the scaling factors robust to outliers, preventing a few extreme values from distorting the entire quantization range.

For example, an Adam optimizer's states for a parameter `p` are typically `(p.grad.exp_avg, p.grad.exp_avg_sq)`. In 8-bit Adam, these `exp_avg` and `exp_avg_sq` tensors are stored as 8-bit integers (`int8`). Along with these, small float tensors containing the scaling factors for each block are also stored. When `p` is updated, the `int8` states are loaded, dequantized using their respective scaling factors, used to compute the update, and then the new `exp_avg` and `exp_avg_sq` values are quantized back to `int8` with updated scaling factors for the next step. This continuous dynamic adjustment ensures high fidelity while leveraging memory savings.

<a name="4-practical-implementation-with-bitsandbytes"></a>
## 4. Practical Implementation with BitsAndBytes

Integrating BitsAndBytes into an existing PyTorch training pipeline is remarkably straightforward, often requiring only a single line change to replace a standard optimizer with its 8-bit counterpart. The library is designed for ease of use, making these powerful optimization techniques accessible.

Below is a simple Python code snippet demonstrating how to use an 8-bit AdamW optimizer from the `bitsandbytes` library.

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# 1. Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# 2. Instantiate the model
model = SimpleModel()
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Define some dummy data for demonstration
dummy_input = torch.randn(64, 100).to(device)
dummy_target = torch.randint(0, 10, (64,)).to(device)

# 4. Initialize the 8-bit optimizer from bitsandbytes
# Replace torch.optim.AdamW with bnb.optim.AdamW8bit
# The `optim_bits` parameter is often automatically handled by bnb,
# but explicitly setting to 8 ensures 8-bit optimizer states.
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)

# 5. Define a loss function
criterion = nn.CrossEntropyLoss()

# 6. Perform a dummy training step
optimizer.zero_grad() # Clear previous gradients
output = model(dummy_input) # Forward pass
loss = criterion(output, dummy_target) # Compute loss
loss.backward() # Backward pass to compute gradients
optimizer.step() # Update model parameters using the 8-bit optimizer

print(f"Loss after one step: {loss.item()}")
print("Model parameters updated using bnb.optim.AdamW8bit.")

# To check if the optimizer states are indeed 8-bit (this is internal to bnb,
# but conceptually, the large states are compressed)
# Actual state data types are managed by bnb's internal CUDA kernels.
# The user-facing API remains the same.

(End of code example section)
```

In this example, `bnb.optim.AdamW8bit` is used directly in place of `torch.optim.AdamW`. BitsAndBytes automatically handles the creation of the 8-bit quantized optimizer states and their dynamic dequantization/re-quantization during the `optimizer.step()` call. This seamless integration allows developers to leverage the memory benefits of 8-bit optimization without significant changes to their existing training code.

<a name="5-conclusion"></a>
## 5. Conclusion

The BitsAndBytes library, through its sophisticated implementation of 8-bit optimizers and quantization techniques, has significantly lowered the barrier to entry for training and deploying large-scale deep learning models. By intelligently reducing the memory footprint of optimizer states and leveraging highly optimized low-precision matrix multiplication kernels, it enables practitioners to work with models that would otherwise be out of reach on conventional hardware. The principles of dynamic and quantile-based quantization are central to maintaining numerical stability and model accuracy despite the reduced precision. The ease of integration into existing PyTorch workflows makes BitsAndBytes an invaluable tool in the modern deep learning ecosystem. As models continue to grow in complexity and size, innovations like BitsAndBytes will remain crucial for fostering accessibility, accelerating research, and pushing the boundaries of what's achievable in artificial intelligence.

---
<br>

<a name="türkçe-içerik"></a>
## BitsAndBytes: 8-bit Optimizatörler ve Kuantizasyon

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Derin Öğrenmede Kuantizasyonu Anlamak](#2-derin-öğrenmede-kuantizasyonu-anlamak)
- [3. BitsAndBytes Kütüphanesi ve 8-bit Optimizatörler](#3-bitsandbytes-kütüphanesi-ve-8-bit-optimizatörler)
    - [3.1. Temel Kavramlar ve Faydaları](#31-temel-kavramlar-ve-faydalari)
    - [3.2. Teknik Mekanizmalar](#32-teknik-mekanizmalar)
- [4. BitsAndBytes ile Pratik Uygulama](#4-pratik-uygulama-bitsandbytes)
- [5. Sonuç](#5-sonuç)

---

<a name="1-giriş"></a>
## 1. Giriş

Derin öğrenmedeki hızlı ilerlemeler, özellikle büyük dil modelleri (LLM'ler) ve karmaşık sinir ağı mimarileri alanında, model boyutlarında ve hesaplama gereksinimlerinde önemli artışlara yol açmıştır. Bu modeller genellikle hem parametreleri hem de eğitim sırasında optimizatör durumları için muazzam miktarda bellek gerektirir, bu da sınırlı donanım kaynaklarına sahip araştırmacılar ve uygulayıcılar için erişilemez hale gelir. Milyarlarca parametreye sahip modellerin eğitimi genellikle üst düzey hızlandırıcılara ihtiyaç duyar ve bu da önemli bir giriş engeli oluşturur. Sayısal temsillerin hassasiyetini azaltan bir teknik olan **kuantizasyon**, bu zorlukları hafifletmek için kritik bir strateji olarak ortaya çıkmıştır. Parametreleri ve aktivasyonları daha az bitle temsil ederek, kuantizasyon bellek ayak izini önemli ölçüde azaltabilir, hesaplama verimliliğini artırabilir ve hem eğitim hem de çıkarım süreçlerini hızlandırabilir.

Özel CUDA fonksiyonları etrafında hafif bir sarmalayıcı olan **BitsAndBytes** kütüphanesi, 8-bit kuantizasyon ve optimizasyonun sağlam ve verimli uygulamalarını sağlayarak büyük modellere erişimi devrim niteliğinde değiştirmiştir. Bu belge, 8-bit optimizatörlerin ve kuantizasyonun prensiplerini inceleyecek, BitsAndBytes'ın bu teknikleri kullanarak büyük ölçekli derin öğrenme modellerinin daha mütevazı donanım konfigürasyonlarında eğitilmesini ve dağıtılmasını nasıl sağladığını araştıracaktır. Temel mekanizmaları, pratik faydaları inceleyecek ve uygulamasını gösteren açıklayıcı bir kod örneği sunacağız.

<a name="2-derin-öğrenmede-kuantizasyonu-anlamak"></a>
## 2. Derin Öğrenmede Kuantizasyonu Anlamak

**Kuantizasyon**, sürekli değerleri veya büyük bir kümedeki değerleri sonlu, daha küçük bir değer kümesine eşleme işlemidir. Derin öğrenme bağlamında, öncelikle model ağırlıklarının, aktivasyonlarının ve optimizatör durumlarının sayısal temsilinin bit genişliğini azaltmayı içerir. Çoğu derin öğrenme modeli, yüksek hassasiyet sunan ancak önemli bellek tüketen 32-bit kayan nokta sayıları (**FP32**) kullanılarak eğitilir. Kuantizasyon, bu FP32 temsillerini 16-bit kayan nokta (**FP16** veya **BF16**) veya 8-bit tam sayılar (**INT8**) gibi daha düşük hassasiyetli formatlarla değiştirmeyi amaçlar.

Kuantizasyonun başlıca motivasyonları şunlardır:
*   **Bellek Azaltma:** Daha düşük bit genişliğine sahip sayılar daha az depolama gerektirir. Örneğin, FP32'den INT8'e dönüştürmek, kuantize edilmiş elemanlar için bellek kullanımını %75 oranında azaltabilir. Bu, daha büyük modellerin GPU belleğine sığmasına veya daha küçük modellerin daha büyük toplu iş boyutlarıyla çalıştırılmasına olanak tanır.
*   **Hesaplama Verimliliği:** Daha düşük hassasiyetli veriler üzerindeki işlemler, modern donanımlarda, özellikle INT8 hesaplamaları için tasarlanmış özel donanım hızlandırıcılarda genellikle daha hızlı yürütülür. Bu, daha hızlı eğitim ve çıkarım sürelerine yol açabilir.
*   **Enerji Verimliliği:** Azaltılmış bellek erişimi ve hesaplama, daha düşük güç tüketimine de yol açabilir, bu da özellikle uç cihazlar ve büyük ölçekli veri merkezleri için önemlidir.

Model eğitildikten sonra modellerin kuantize edildiği **eğitim sonrası kuantizasyon (PTQ)** ve eğitim sırasında kuantizasyonun simüle edildiği **kuantizasyon-farkındalıklı eğitim (QAT)** gibi çeşitli kuantizasyon stratejileri bulunmakla birlikte, BitsAndBytes belirli bir uygulamaya odaklanmıştır: optimizatör durumlarını ve daha yakın zamanda verimli çıkarım için model ağırlıklarını kuantize etmek (örneğin, LLM.int8() ve QLoRA). Agresif kuantizasyonun, özellikle 8-bit'e kadar, zorluğu model doğruluğunu korumaktır. Azaltılmış hassasiyetin neden olduğu hatalar birikebilir ve performansı düşürebilir. BitsAndBytes, sayısal kararlılığı korumak için **dinamik kuantizasyon** ve **kantil tabanlı ölçekleme** gibi gelişmiş teknikler kullanarak bu sorunu ele alır.

<a name="3-bitsandbytes-kütüphanesi-ve-8-bit-optimizatörler"></a>
## 3. BitsAndBytes Kütüphanesi ve 8-bit Optimizatörler

**BitsAndBytes** kütüphanesi, özellikle daha önce pahalı donanım gerektiren modellerin eğitimi için, büyük ölçekli derin öğrenmeyi demokratikleştirmek için bir köşe taşı olmuştur. Bunu, yenilikçi **8-bit optimizatörleri** ve verimli düşük hassasiyetli matris çarpım rutinleri aracılığıyla başarmaktadır.

<a name="31-temel-kavramlar-ve-faydalari"></a>
### 3.1. Temel Kavramlar ve Faydaları

BitsAndBytes, özünde, düşük hassasiyetli veri türleri üzerinde minimum ek yük ile işlem yapmaya olanak tanıyan yüksek optimize edilmiş özel CUDA çekirdekleri sağlar. Temel işlevsellikler şunları içerir:

*   **8-bit Optimizatörler:** Kütüphane, Adam, AdamW ve SGD gibi popüler optimizatörlerin 8-bit varyantlarını sunar. Eğitim sırasında, optimizatör durumları (örneğin, Adam için momentler), GPU belleğinin önemli bir bölümünü tüketebilir, genellikle model parametrelerinin kendisinin 2 ila 4 katı boyutunda (örneğin, Adam, her parametre için iki durum saklar, her biri FP32 olarak). Bu optimizatör durumlarını 8-bit'e kuantize ederek, BitsAndBytes bellek ayak izlerini %75'e kadar azaltabilir. Bu bellek tasarrufu, daha büyük modelleri veya daha büyük toplu iş boyutlarını belirli bir GPU'ya sığdırmak için çok önemlidir.
*   **Dinamik Kuantizasyon:** Ölçekleme faktörlerinin sabit olduğu statik kuantizasyonun aksine, BitsAndBytes optimizatör durumları için **dinamik kuantizasyon** yaklaşımını kullanır. Bu, kuantizasyon parametrelerinin (örneğin, ölçekleme için min/max değerleri) her optimizasyon adımında her tensör için anında yeniden hesaplandığı anlamına gelir. Bu adaptif yaklaşım, özellikle gradyan dağılımları eğitim sırasında önemli ölçüde değiştiğinde, sayısal kararlılığı ve hassasiyeti korumaya yardımcı olur.
*   **Kantil Kuantizasyon:** Gradyanlardaki ve optimizatör durumlarındaki potansiyel olarak geniş değer aralığını sağlam bir şekilde ele almak için BitsAndBytes **kantil kuantizasyonu** kullanır. Aykırı değerlere duyarlı olabilen basit min-max ölçeklemenin aksine, kantil kuantizasyon değerleri dağılımlarına göre eşleştirir, aşırı aykırı değerleri etkili bir şekilde göz ardı eder ve 8-bit aralığına daha kararlı ve doğru bir eşleme sağlar.
*   **Verimli Karma Hassasiyet Eğitimi:** Optimizatör durumları 8-bit olsa da, BitsAndBytes doğruluk sağlamak için genellikle daha yüksek hassasiyetli (örneğin, FP32 veya BF16) gradyan güncellemeleri ve kritik hesaplamalar yapar, **karma hassasiyetli eğitimin** faydalarını kullanır. Bu hibrit yaklaşım, bellek verimliliği ile sayısal kararlılığı dengeler.
*   **4-bit ve 8-bit Matris Çarpımı (GEMM):** Optimizatörlerin ötesinde, BitsAndBytes düşük hassasiyetli matris çarpımı için de oldukça optimize edilmiş çekirdekler sağlar. Bu, çıkarım için **LLM.int8()** kuantizasyonu ve **QLoRA** gibi daha yeni uygulamaları için temeldir; bu, model ağırlıklarını 4-bit veya 8-bit'e kuantize ederek tek bir tüketici GPU'sunda devasa dil modellerinin (örneğin, 65B parametre) eğitilmesine olanak tanır.

BitsAndBytes'ın en büyük faydası, **büyük model eğitimi ve çıkarımının demokratikleşmesidir**. Araştırmacıların ve geliştiricilerin, daha önce donanım kısıtlamaları nedeniyle erişilemeyen modelleri denemelerine ve dağıtmalarına olanak tanıyarak, son teknoloji AI'ye erişimi önemli ölçüde genişletir.

<a name="32-teknik-mekanizmalar"></a>
### 3.2. Teknik Mekanizmalar

BitsAndBytes'ın 8-bit optimizatörlerinin ardındaki temel mekanizma, optimizatör durumlarını (örneğin, Adam için `exp_avg`, `exp_avg_sq`) 8-bit tam sayı formatında depolamayı içerir. Bu durumlar bir gradyan güncellemesi için gerektiğinde, daha yüksek bir hassasiyete (örneğin, FP32) geri **dekuantize** edilir, hesaplamada kullanılır ve ardından depolama için tekrar 8-bit'e **re-kuantize** edilir. Bu dekuantize-hesapla-kuantize döngüsü, gecikmeyi en aza indirmek için özel CUDA çekirdekleri içinde titizlikle optimize edilmiştir.

Gerçek kuantizasyon için BitsAndBytes, **blok tabanlı dinamik kuantizasyon** yaklaşımını kullanır. Tüm tensörü tek bir ölçekleme faktörü ile kuantize etmek yerine, tensörler genellikle daha küçük bloklara (örneğin, 256 veya 512 eleman) bölünür. Her blok daha sonra o bloktaki istatistiklerden türetilen kendi dinamik ölçekleme faktörünü alır. Bu yerel ölçekleme, veri dağılımındaki yerel varyasyonlara global bir ölçekten daha etkili bir şekilde uyum sağlayabildiği için 8-bit temsilinin hassasiyetini önemli ölçüde artırır. **Kantil tabanlı ölçekleme** kullanımı, ölçekleme faktörlerini aykırı değerlere karşı daha sağlam hale getirerek, birkaç aşırı değerin tüm kuantizasyon aralığını bozmasını önleyerek bunu daha da iyileştirir.

Örneğin, bir Adam optimizatörünün `p` parametresi için durumları tipik olarak `(p.grad.exp_avg, p.grad.exp_avg_sq)`'dir. 8-bit Adam'da, bu `exp_avg` ve `exp_avg_sq` tensörleri 8-bit tam sayılar (`int8`) olarak depolanır. Bunlarla birlikte, her blok için ölçekleme faktörlerini içeren küçük kayan nokta tensörleri de depolanır. `p` güncellendiğinde, `int8` durumları yüklenir, ilgili ölçekleme faktörleri kullanılarak dekuantize edilir, güncellemeyi hesaplamak için kullanılır ve ardından yeni `exp_avg` ve `exp_avg_sq` değerleri bir sonraki adım için güncellenmiş ölçekleme faktörleri ile tekrar `int8`'e kuantize edilir. Bu sürekli dinamik ayarlama, bellek tasarruflarından yararlanırken yüksek doğruluk sağlar.

<a name="4-pratik-uygulama-bitsandbytes"></a>
## 4. BitsAndBytes ile Pratik Uygulama

BitsAndBytes'ı mevcut bir PyTorch eğitim ardışık düzenine entegre etmek oldukça basittir, genellikle standart bir optimizatörü 8-bit karşılığıyla değiştirmek için yalnızca tek bir satır değişikliği gerektirir. Kütüphane, bu güçlü optimizasyon tekniklerini erişilebilir kılmak için kolay kullanım için tasarlanmıştır.

Aşağıda, `bitsandbytes` kütüphanesinden 8-bit AdamW optimizatörünün nasıl kullanılacağını gösteren basit bir Python kod parçacığı bulunmaktadır.

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# 1. Basit bir sinir ağı tanımlayın
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# 2. Modeli örneklendirin
model = SimpleModel()
# Eğer varsa modeli GPU'ya taşıyın
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Gösterim için bazı sahte veriler tanımlayın
dummy_input = torch.randn(64, 100).to(device)
dummy_target = torch.randint(0, 10, (64,)).to(device)

# 4. bitsandbytes'tan 8-bit optimizatörü başlatın
# torch.optim.AdamW'yi bnb.optim.AdamW8bit ile değiştirin
# `optim_bits` parametresi genellikle bnb tarafından otomatik olarak işlenir,
# ancak açıkça 8 olarak ayarlamak 8-bit optimizatör durumlarını garanti eder.
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)

# 5. Bir kayıp fonksiyonu tanımlayın
criterion = nn.CrossEntropyLoss()

# 6. Sahte bir eğitim adımı gerçekleştirin
optimizer.zero_grad() # Önceki gradyanları temizleyin
output = model(dummy_input) # İleri yayılım (forward pass)
loss = criterion(output, dummy_target) # Kaybı hesaplayın
loss.backward() # Gradyanları hesaplamak için geri yayılım (backward pass)
optimizer.step() # 8-bit optimizatörü kullanarak model parametrelerini güncelleyin

print(f"Bir adımdan sonra kayıp: {loss.item()}")
print("Model parametreleri bnb.optim.AdamW8bit kullanılarak güncellendi.")

# Optimizatör durumlarının gerçekten 8-bit olup olmadığını kontrol etmek için (bu bnb'nin dahili bir özelliğidir,
# ancak kavramsal olarak büyük durumlar sıkıştırılır)
# Gerçek durum veri türleri bnb'nin dahili CUDA çekirdekleri tarafından yönetilir.
# Kullanıcıya yönelik API aynı kalır.

(Kod örneği bölümünün sonu)
```

Bu örnekte, `bnb.optim.AdamW8bit`, `torch.optim.AdamW` yerine doğrudan kullanılmıştır. BitsAndBytes, 8-bit kuantize edilmiş optimizatör durumlarının oluşturulmasını ve `optimizer.step()` çağrısı sırasında bunların dinamik dekuantizasyonunu/yeniden kuantizasyonunu otomatik olarak yönetir. Bu sorunsuz entegrasyon, geliştiricilerin mevcut eğitim kodlarında önemli değişiklikler yapmadan 8-bit optimizasyonunun bellek faydalarından yararlanmalarına olanak tanır.

<a name="5-sonuç"></a>
## 5. Sonuç

BitsAndBytes kütüphanesi, 8-bit optimizatörlerinin ve kuantizasyon tekniklerinin sofistike uygulaması aracılığıyla, büyük ölçekli derin öğrenme modellerinin eğitilmesi ve dağıtılması için giriş engelini önemli ölçüde düşürmüştür. Optimizatör durumlarının bellek ayak izini akıllıca azaltarak ve yüksek optimize edilmiş düşük hassasiyetli matris çarpım çekirdeklerinden yararlanarak, uygulayıcıların geleneksel donanımlarda aksi takdirde erişilemez olacak modellerle çalışmasına olanak tanır. Dinamik ve kantil tabanlı kuantizasyon prensipleri, azaltılmış hassasiyete rağmen sayısal kararlılığı ve model doğruluğunu korumak için merkezi öneme sahiptir. Mevcut PyTorch iş akışlarına kolay entegrasyon, BitsAndBytes'ı modern derin öğrenme ekosisteminde paha biçilmez bir araç haline getirmektedir. Modeller karmaşıklık ve boyut olarak büyümeye devam ettikçe, BitsAndBytes gibi yenilikler, erişilebilirliği teşvik etmek, araştırmayı hızlandırmak ve yapay zekada başarılabilir olanın sınırlarını zorlamak için kritik olmaya devam edecektir.