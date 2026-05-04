# Model Quantization: INT8, FP4, and NF4

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Fundamentals of Model Quantization](#2-fundamentals-of-model-quantization)
  - [2.1. Why Quantize?](#2.1-why-quantize)
  - [2.2. Core Concepts](#2.2-core-concepts)
  - [2.3. Quantization Schemes](#2.3-quantization-schemes)
- [3. Quantization Techniques: INT8, FP4, and NF4](#3-quantization-techniques-int8-fp4-and-nf4)
  - [3.1. INT8 Quantization](#3.1-int8-quantization)
  - [3.2. FP4 Quantization](#3.2-fp4-quantization)
  - [3.3. NF4 Quantization (NormalFloat 4-bit)](#3.3-nf4-quantization-normalfloat-4-bit)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The escalating complexity and parameter count of modern deep learning models, particularly **Large Language Models (LLMs)** and **Generative AI** architectures, have led to unprecedented computational and memory demands. Deploying these models, especially in resource-constrained environments or for high-throughput inference, presents significant challenges. **Model quantization** emerges as a critical technique to mitigate these issues by reducing the precision of the numerical representations of model parameters (weights) and activations. Instead of using high-precision floating-point numbers (e.g., FP32), quantization maps these values to lower-precision formats (e.g., integers or lower-bit floats), thereby reducing memory footprint, accelerating computation, and improving energy efficiency.

This document provides a comprehensive overview of model quantization, delving into its fundamental principles and exploring three prominent low-bit precision formats: **INT8**, **FP4**, and **NF4**. Each format offers distinct trade-offs between model accuracy, inference speed, and memory reduction, making them suitable for different deployment scenarios and hardware capabilities. Understanding these techniques is paramount for optimizing the deployment and scalability of state-of-the-art AI models.

<a name="2-fundamentals-of-model-quantization"></a>
## 2. Fundamentals of Model Quantization

<a name="2.1-why-quantize"></a>
### 2.1. Why Quantize?
The primary motivations for model quantization are rooted in the practical challenges of deploying large neural networks:
*   **Reduced Memory Footprint:** Lower precision numbers require less memory storage. For example, moving from 32-bit floats (FP32) to 8-bit integers (INT8) can reduce memory usage by 4x, allowing larger models to fit into available GPU or CPU memory, or enabling the deployment of models on edge devices with limited resources. This is particularly crucial for LLMs that have hundreds of billions of parameters.
*   **Faster Inference:** Quantized models can often achieve faster inference speeds. Processors, especially specialized hardware like **Tensor Processing Units (TPUs)** or **Neural Processing Units (NPUs)**, are often optimized for integer arithmetic, leading to significant throughput improvements. Even general-purpose GPUs can see speedups due to reduced memory bandwidth requirements and more efficient cache utilization.
*   **Lower Power Consumption:** Reduced computation and memory access translate directly into lower power consumption, making quantized models ideal for mobile devices, IoT applications, and green AI initiatives.
*   **Easier Deployment:** Smaller model files are easier to distribute and deploy, reducing network bandwidth requirements during model updates and deployment.

<a name="2.2-core-concepts"></a>
### 2.2. Core Concepts
Quantization involves mapping a range of floating-point values to a smaller range of integer or lower-bit floating-point values. Key concepts include:
*   **Quantization Scale ($S$) and Zero-Point ($Z$):** For linear quantization, a floating-point value $r$ is typically quantized to an integer $q$ using the formula: $q = \text{round}(r/S + Z)$. Conversely, dequantization restores the floating-point value: $r = (q - Z) * S$. The scale $S$ determines the resolution of the quantized range, and the zero-point $Z$ aligns the floating-point zero to an integer value, which is crucial for representing negative numbers symmetrically or accommodating asymmetric distributions.
*   **Calibration:** This process involves determining the appropriate scale and zero-point for each tensor (weights and activations). Calibration typically involves running a small representative dataset through the unquantized model to observe the statistical distribution (min/max values, histograms) of activations and weights. This information is then used to set the quantization parameters.
*   **Dynamic vs. Static Quantization:**
    *   **Static Quantization (Post-Training Static Quantization - PTQS):** The scales and zero-points for both weights and activations are determined during calibration and fixed throughout inference. This provides maximum speedup but requires careful calibration.
    *   **Dynamic Quantization (Post-Training Dynamic Quantization - PTQD):** Weights are quantized offline, but activations are quantized at runtime based on their observed min/max values. This offers flexibility but introduces overhead for dynamic range computation.
*   **Quantization-Aware Training (QAT):** Instead of quantizing after training, QAT simulates the effects of quantization during the training process. This allows the model to learn to compensate for the precision loss, often leading to higher accuracy compared to PTQ, albeit requiring retraining.

<a name="2.3-quantization-schemes"></a>
### 2.3. Quantization Schemes
*   **Symmetric vs. Asymmetric:**
    *   **Symmetric Quantization:** The floating-point range is centered around zero, and the integer range is also symmetric (e.g., -127 to 127 for INT8). The zero-point is usually 0. This is simpler but might not be optimal for activations with highly skewed distributions.
    *   **Asymmetric Quantization:** The floating-point range can be arbitrary, and the integer range covers all non-negative or negative values (e.g., 0 to 255 for unsigned INT8). A non-zero zero-point is used to map the floating-point zero precisely. This is often better for activations.
*   **Per-Tensor vs. Per-Channel:**
    *   **Per-Tensor Quantization:** A single scale and zero-point are used for the entire tensor. Simpler, but less precise for tensors with widely varying value distributions across channels.
    *   **Per-Channel Quantization:** A unique scale and zero-point are computed for each channel of a tensor (e.g., for each output channel of a convolutional layer's weights). This offers higher precision with a slight increase in overhead.

<a name="3-quantization-techniques-int8-fp4-and-nf4"></a>
## 3. Quantization Techniques: INT8, FP4, and NF4

<a name="3.1-int8-quantization"></a>
### 3.1. INT8 Quantization
**INT8 quantization** is one of the most widely adopted and mature quantization techniques. It maps 32-bit floating-point values to 8-bit integers, typically ranging from -128 to 127 (signed INT8) or 0 to 255 (unsigned INT8).
*   **Mechanism:** It primarily employs linear quantization, where a fixed scaling factor and zero-point are determined during calibration (either static or dynamic) to map the floating-point range to the 8-bit integer range. Matrix multiplications, a core operation in neural networks, can then be performed using efficient integer arithmetic, often resulting in significant speedups on hardware optimized for INT8 operations.
*   **Advantages:**
    *   **Mature Ecosystem:** Widely supported across various deep learning frameworks (TensorFlow, PyTorch, ONNX Runtime) and hardware platforms (NVIDIA GPUs, Intel CPUs, mobile NPUs).
    *   **Significant Performance Gains:** Offers up to 4x memory reduction and substantial speedups for inference, often with minimal accuracy degradation if properly calibrated (especially with QAT).
    *   **Hardware Acceleration:** Many modern AI accelerators include dedicated INT8 units, making it highly efficient.
*   **Limitations:**
    *   **Accuracy Trade-off:** While often robust, some models, particularly very large and complex ones like LLMs, can suffer noticeable accuracy drops with INT8 quantization, especially for activations with extreme outliers.
    *   **Calibration Sensitivity:** Requires careful calibration to determine optimal quantization parameters.
*   **Applications:** INT8 is a standard for deploying models in production for computer vision (e.g., image classification, object detection) and traditional NLP tasks where model sizes are manageable and accuracy preservation is critical. It's often the first choice for achieving inference acceleration.

<a name="3.2-fp4-quantization"></a>
### 3.2. FP4 Quantization
**FP4 quantization** represents a paradigm shift towards even lower precision, driven by the immense memory requirements of **Large Language Models (LLMs)**. While INT8 provides a 4x reduction, 4-bit formats aim for an 8x reduction in memory compared to FP32. FP4 refers to a 4-bit floating-point format, which, unlike INT8, retains some characteristics of floating-point numbers like an exponent and mantissa, albeit severely truncated.
*   **Mechanism:** The exact representation of FP4 can vary. A common scheme used in `bitsandbytes` (a library for optimizing LLMs) is a **1-bit sign, 2-bit exponent, and 1-bit mantissa** (E2M1), or variations like E2M0 with a denormal bit. This limited precision means that the representable values are sparse and non-uniformly distributed, which can be beneficial for capturing dynamic ranges more effectively than fixed-point integers at very low bit-widths.
*   **Advantages:**
    *   **Extreme Memory Reduction:** Achieves an 8x reduction in memory footprint compared to FP32, making it possible to load and fine-tune much larger LLMs (e.g., 65B models on consumer GPUs) with techniques like **QLoRA**.
    *   **Performance for LLMs:** While arithmetic operations might be more complex than INT8 on general-purpose hardware, the significant memory bandwidth savings often translate to overall speedups for LLM inference where memory access is a major bottleneck.
*   **Limitations:**
    *   **Accuracy Challenges:** The severe precision reduction can lead to greater accuracy degradation than INT8. Careful selection of quantization method (e.g., block-wise quantization) and robust techniques are needed to mitigate this.
    *   **Hardware Support:** Native hardware support for FP4 arithmetic is less common than for INT8, often requiring custom kernels or emulation, which can impact performance.
*   **Applications:** Primarily developed and applied in the context of LLMs, especially for efficient training (e.g., QLoRA for fine-tuning) and inference, where the ability to fit models into memory is the paramount concern.

<a name="3.3-nf4-quantization-normalfloat-4-bit"></a>
### 3.3. NF4 Quantization (NormalFloat 4-bit)
**NF4 (NormalFloat 4-bit)** is a novel 4-bit floating-point quantization data type introduced by Dettmers et al. in their QLoRA paper (2023). It is specifically designed to be "information-theoretically optimal" for data that follows a **normal distribution**, which is often the case for pre-trained neural network weights.
*   **Mechanism:** Unlike standard FP4 formats, NF4 defines a set of 2^4 = 16 specific numbers that are chosen to minimize the **quantization error** for a standard normal distribution (mean 0, variance 1). These values are non-uniform and are derived from quantiles of a standard normal distribution. For actual model weights, which have different means and variances, the weights are first normalized to a standard normal distribution (or a range close to it) before being mapped to the NF4 values, and then de-normalized after dequantization. This block-wise, empirically derived quantization scheme allows NF4 to preserve more information than generic FP4 or INT4 formats.
*   **Advantages:**
    *   **Optimality for Normal Distribution:** By exploiting the statistical properties of network weights, NF4 achieves higher accuracy preservation than other 4-bit schemes for LLMs.
    *   **Superior Accuracy for LLMs:** When combined with QLoRA, NF4 allows for fine-tuning LLMs with minimal performance degradation compared to FP32, while still achieving an 8x memory reduction for the base model weights.
    *   **Memory Efficiency:** Provides the same 8x memory reduction as FP4, enabling the deployment and fine-tuning of massive LLMs on commodity hardware.
*   **Limitations:**
    *   **Computational Overhead:** The normalization and denormalization steps, along with the non-uniform mapping, can introduce some computational overhead compared to simpler linear INT8 schemes. Custom kernels are essential for performance.
    *   **Less General-Purpose:** While highly effective for LLM weights, its optimality is tied to the assumption of normal distribution, making it potentially less ideal for other data types or activation functions without specific adaptations.
*   **Applications:** NF4, particularly through the `bitsandbytes` library and the QLoRA technique, has become a cornerstone for efficiently training and deploying very large language models. It enables researchers and practitioners to work with models that would otherwise be out of reach due due to memory constraints, democratizing access to powerful LLMs.

<a name="4-code-example"></a>
## 4. Code Example
This conceptual Python snippet illustrates a basic symmetric linear INT8 quantization process for a given tensor. It does not use any specific library but demonstrates the core idea of scaling and rounding.

```python
import torch

def quantize_int8_symmetric(tensor_fp32):
    """
    Conceptual symmetric INT8 quantization.
    Maps FP32 tensor values to INT8 range [-127, 127].
    
    Args:
        tensor_fp32 (torch.Tensor): The input 32-bit floating-point tensor.

    Returns:
        torch.Tensor: The quantized 8-bit integer tensor.
        float: The scale factor used for quantization.
    """
    # Determine the absolute maximum value in the tensor
    abs_max = torch.max(torch.abs(tensor_fp32))

    # Calculate the scale factor
    # We map [-abs_max, abs_max] to [-127, 127]
    scale = abs_max / 127.0

    # Quantize the tensor
    # Divide by scale, round to nearest integer, and clamp to INT8 range
    quantized_tensor = torch.round(tensor_fp32 / scale)
    quantized_tensor = torch.clamp(quantized_tensor, -127, 127)
    
    # Convert to integer type
    quantized_tensor = quantized_tensor.to(torch.int8)

    return quantized_tensor, scale

def dequantize_int8_symmetric(tensor_int8, scale):
    """
    Conceptual symmetric INT8 dequantization.
    Maps INT8 tensor values back to approximate FP32.

    Args:
        tensor_int8 (torch.Tensor): The quantized 8-bit integer tensor.
        scale (float): The scale factor used during quantization.

    Returns:
        torch.Tensor: The dequantized 32-bit floating-point tensor.
    """
    # Dequantize by multiplying with the scale
    dequantized_tensor = tensor_int8.to(torch.float32) * scale
    return dequantized_tensor

# Example Usage:
# Create a sample FP32 tensor
sample_tensor_fp32 = torch.tensor([-1.5, 0.0, 0.75, 2.5, -0.1], dtype=torch.float32)
print("Original FP32 Tensor:", sample_tensor_fp32)

# Quantize
quantized_tensor_int8, quantization_scale = quantize_int8_symmetric(sample_tensor_fp32)
print("Quantized INT8 Tensor:", quantized_tensor_int8)
print("Quantization Scale:", quantization_scale)

# Dequantize
dequantized_tensor_fp32 = dequantize_int8_symmetric(quantized_tensor_int8, quantization_scale)
print("Dequantized FP32 Tensor:", dequantized_tensor_fp32)

# Show approximation error
print("Approximation Error (Original - Dequantized):", sample_tensor_fp32 - dequantized_tensor_fp32)

(End of code example section)
```
<a name="5-conclusion"></a>
## 5. Conclusion
Model quantization is an indispensable technique in the era of large-scale AI, offering a pragmatic solution to the computational and memory burdens posed by increasingly complex neural networks. By strategically reducing numerical precision, it enables the deployment of powerful models on a wider range of hardware, from edge devices to enterprise-grade servers, significantly improving inference speed and energy efficiency.

**INT8 quantization** remains a robust and widely supported standard, providing substantial gains with generally acceptable accuracy for a broad spectrum of models. Its maturity and extensive hardware support make it a go-to for production deployments. As models continue to grow, particularly LLMs, the need for even lower precision has propelled the adoption of 4-bit formats. **FP4 quantization** offers radical memory savings, albeit with greater challenges in preserving accuracy and requiring specialized hardware or software optimizations. **NF4 quantization**, building upon information theory, provides a sophisticated approach to 4-bit precision, specifically tailored for the statistical properties of neural network weights. Its integration with techniques like QLoRA has been transformative for democratizing access to large language model fine-tuning and inference.

The choice between INT8, FP4, and NF4 depends heavily on the specific application, available hardware, and the acceptable trade-off between memory footprint, inference speed, and model accuracy. As research in generative AI continues to push the boundaries of model scale, advancements in quantization techniques will remain critical for translating theoretical breakthroughs into practical, efficient, and accessible AI solutions.

---
<br>

<a name="türkçe-içerik"></a>
## Model Kuantizasyonu: INT8, FP4 ve NF4

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Model Kuantizasyonunun Temelleri](#2-model-kuantizasyonunun-temelleri)
  - [2.1. Neden Kuantizasyon?](#2.1-neden-kuantizasyon)
  - [2.2. Temel Kavramlar](#2.2-temel-kavramlar)
  - [2.3. Kuantizasyon Şemaları](#2.3-kuantizasyon-şemaları)
- [3. Kuantizasyon Teknikleri: INT8, FP4 ve NF4](#3-kuantizasyon-teknikleri-int8-fp4-ve-nf4)
  - [3.1. INT8 Kuantizasyonu](#3.1-int8-kuantizasyonu)
  - [3.2. FP4 Kuantizasyonu](#3.2-fp4-kuantizasyonu)
  - [3.3. NF4 Kuantizasyonu (NormalFloat 4-bit)](#3.3-nf4-kuantizasyonu-normalfloat-4-bit)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Modern derin öğrenme modellerinin, özellikle de **Büyük Dil Modelleri (LLM'ler)** ve **Üretken Yapay Zeka (Generative AI)** mimarilerinin artan karmaşıklığı ve parametre sayısı, benzeri görülmemiş hesaplama ve bellek taleplerine yol açmıştır. Bu modelleri, özellikle kaynak kısıtlı ortamlarda veya yüksek verimli çıkarım için dağıtmak önemli zorluklar sunmaktadır. **Model kuantizasyonu**, model parametrelerinin (ağırlıklar) ve aktivasyonlarının sayısal gösterimlerinin hassasiyetini azaltarak bu sorunları hafifletmek için kritik bir teknik olarak ortaya çıkmaktadır. Yüksek hassasiyetli kayan noktalı sayılar (örn. FP32) kullanmak yerine, kuantizasyon bu değerleri daha düşük hassasiyetli formatlara (örn. tamsayılar veya daha düşük bitli kayan noktalar) dönüştürerek bellek ayak izini azaltır, hesaplamayı hızlandırır ve enerji verimliliğini artırır.

Bu belge, model kuantizasyonuna kapsamlı bir genel bakış sunmakta, temel prensiplerini incelemekte ve üç öne çıkan düşük bit hassasiyetli formatı araştırmaktadır: **INT8**, **FP4** ve **NF4**. Her bir format, model doğruluğu, çıkarım hızı ve bellek azaltma arasında farklı ödünleşimler sunarak, farklı dağıtım senaryoları ve donanım yetenekleri için uygun hale gelmektedir. Bu teknikleri anlamak, en son yapay zeka modellerinin dağıtımını ve ölçeklenebilirliğini optimize etmek için büyük önem taşımaktadır.

<a name="2-model-kuantizasyonunun-temelleri"></a>
## 2. Model Kuantizasyonunun Temelleri

<a name="2.1-neden-kuantizasyon"></a>
### 2.1. Neden Kuantizasyon?
Model kuantizasyonunun temel motivasyonları, büyük sinir ağlarını dağıtmanın pratik zorluklarına dayanmaktadır:
*   **Azaltılmış Bellek Ayak İzi:** Düşük hassasiyetli sayılar daha az bellek depolama alanı gerektirir. Örneğin, 32-bit kayan noktalardan (FP32) 8-bit tamsayılara (INT8) geçmek, bellek kullanımını 4 kat azaltarak, daha büyük modellerin mevcut GPU veya CPU belleğine sığmasına ya da sınırlı kaynaklara sahip kenar cihazlarda modellerin dağıtılmasına olanak tanır. Bu durum, yüz milyarlarca parametreye sahip LLM'ler için özellikle kritik öneme sahiptir.
*   **Daha Hızlı Çıkarım:** Kuantize edilmiş modeller genellikle daha hızlı çıkarım hızları elde edebilir. İşlemciler, özellikle **Tensor İşlem Birimleri (TPU'lar)** veya **Nöral İşlem Birimleri (NPU'lar)** gibi özel donanımlar, tamsayı aritmetiği için optimize edilmiştir ve önemli verim artışlarına yol açar. Genel amaçlı GPU'lar bile azalan bellek bant genişliği gereksinimleri ve daha verimli önbellek kullanımı nedeniyle hızlanmalar görebilir.
*   **Daha Düşük Güç Tüketimi:** Azaltılmış hesaplama ve bellek erişimi, doğrudan daha düşük güç tüketimine dönüşür, bu da kuantize edilmiş modelleri mobil cihazlar, IoT uygulamaları ve yeşil yapay zeka girişimleri için ideal kılar.
*   **Daha Kolay Dağıtım:** Daha küçük model dosyaları, dağıtımı ve güncellemeleri daha kolay hale getirir, model güncellemeleri ve dağıtım sırasında ağ bant genişliği gereksinimlerini azaltır.

<a name="2.2-temel-kavramlar"></a>
### 2.2. Temel Kavramlar
Kuantizasyon, bir dizi kayan nokta değerini daha küçük bir tamsayı veya daha düşük bitli kayan nokta değerleri aralığına eşlemeyi içerir. Temel kavramlar şunları içerir:
*   **Kuantizasyon Ölçeği ($S$) ve Sıfır Noktası ($Z$):** Doğrusal kuantizasyon için, bir kayan nokta değeri $r$, tipik olarak $q = \text{round}(r/S + Z)$ formülü kullanılarak bir tamsayı $q$'ye kuantize edilir. Tersine, dekuantizasyon, kayan nokta değerini geri yükler: $r = (q - Z) * S$. Ölçek $S$, kuantize edilmiş aralığın çözünürlüğünü belirlerken, sıfır noktası $Z$, kayan nokta sıfırını bir tamsayı değerine hizalar; bu, negatif sayıları simetrik olarak temsil etmek veya asimetrik dağılımları karşılamak için kritik öneme sahiptir.
*   **Kalibrasyon:** Bu süreç, her bir tensör (ağırlıklar ve aktivasyonlar) için uygun ölçek ve sıfır noktasını belirlemeyi içerir. Kalibrasyon tipik olarak, aktivasyonların ve ağırlıkların istatistiksel dağılımını (min/maks değerleri, histogramlar) gözlemlemek için kuantize edilmemiş model üzerinden küçük bir temsilci veri kümesini çalıştırmayı içerir. Bu bilgiler daha sonra kuantizasyon parametrelerini ayarlamak için kullanılır.
*   **Dinamik ve Statik Kuantizasyon:**
    *   **Statik Kuantizasyon (Eğitim Sonrası Statik Kuantizasyon - PTQS):** Hem ağırlıklar hem de aktivasyonlar için ölçekler ve sıfır noktaları kalibrasyon sırasında belirlenir ve çıkarım boyunca sabit kalır. Bu, maksimum hızlanma sağlar ancak dikkatli kalibrasyon gerektirir.
    *   **Dinamik Kuantizasyon (Eğitim Sonrası Dinamik Kuantizasyon - PTQD):** Ağırlıklar çevrimdışı kuantize edilir, ancak aktivasyonlar, gözlemlenen min/maks değerlerine göre çalışma zamanında kuantize edilir. Bu, esneklik sunar ancak dinamik aralık hesaplaması için ek yük getirir.
*   **Kuantizasyon Bilinçli Eğitim (QAT):** Eğitimden sonra kuantizasyon yapmak yerine, QAT, eğitim süreci boyunca kuantizasyonun etkilerini simüle eder. Bu, modelin hassasiyet kaybını telafi etmeyi öğrenmesini sağlar ve genellikle PTQ'ya kıyasla daha yüksek doğruluk sağlar, ancak yeniden eğitim gerektirir.

<a name="2.3-kuantizasyon-şemaları"></a>
### 2.3. Kuantizasyon Şemaları
*   **Simetrik ve Asimetrik:**
    *   **Simetrik Kuantizasyon:** Kayan nokta aralığı sıfır etrafında ortalanır ve tamsayı aralığı da simetriktir (örn. INT8 için -127 ila 127). Sıfır noktası genellikle 0'dır. Bu daha basittir ancak oldukça çarpık dağılımlara sahip aktivasyonlar için en uygun olmayabilir.
    *   **Asimetrik Kuantizasyon:** Kayan nokta aralığı keyfi olabilir ve tamsayı aralığı tüm negatif olmayan veya negatif değerleri kapsar (örn. işaretsiz INT8 için 0 ila 255). Kayan nokta sıfırını hassas bir şekilde eşleştirmek için sıfırdan farklı bir sıfır noktası kullanılır. Bu genellikle aktivasyonlar için daha iyidir.
*   **Tensör Başına ve Kanal Başına:**
    *   **Tensör Başına Kuantizasyon:** Tüm tensör için tek bir ölçek ve sıfır noktası kullanılır. Daha basittir, ancak kanallar arasında büyük farklılık gösteren değer dağılımlarına sahip tensörler için daha az hassastır.
    *   **Kanal Başına Kuantizasyon:** Bir tensörün her bir kanalı için (örn. bir evrişim katmanının ağırlıklarının her bir çıkış kanalı için) benzersiz bir ölçek ve sıfır noktası hesaplanır. Bu, biraz artan ek yük ile daha yüksek hassasiyet sunar.

<a name="3-kuantizasyon-teknikleri-int8-fp4-ve-nf4"></a>
## 3. Kuantizasyon Teknikleri: INT8, FP4 ve NF4

<a name="3.1-int8-kuantizasyonu"></a>
### 3.1. INT8 Kuantizasyonu
**INT8 kuantizasyonu**, en yaygın olarak benimsenen ve olgun kuantizasyon tekniklerinden biridir. 32-bit kayan nokta değerlerini, tipik olarak -128 ila 127 (işaretli INT8) veya 0 ila 255 (işaretsiz INT8) arasında değişen 8-bit tamsayılara eşler.
*   **Mekanizma:** Esas olarak doğrusal kuantizasyon kullanır; burada, kayan nokta aralığını 8-bit tamsayı aralığına eşlemek için kalibrasyon sırasında (statik veya dinamik olarak) sabit bir ölçeklendirme faktörü ve sıfır noktası belirlenir. Sinir ağlarının temel bir işlemi olan matris çarpımları daha sonra verimli tamsayı aritmetiği kullanılarak gerçekleştirilebilir ve genellikle INT8 işlemleri için optimize edilmiş donanımlarda önemli hızlanmalara yol açar.
*   **Avantajlar:**
    *   **Olgun Ekosistem:** Çeşitli derin öğrenme çerçeveleri (TensorFlow, PyTorch, ONNX Runtime) ve donanım platformları (NVIDIA GPU'lar, Intel CPU'lar, mobil NPU'lar) genelinde yaygın olarak desteklenir.
    *   **Önemli Performans Artışları:** %75'e varan bellek azaltma ve uygun şekilde kalibre edildiğinde (özellikle QAT ile) minimum doğruluk kaybıyla çıkarım için önemli hızlanmalar sunar.
    *   **Donanım Hızlandırma:** Birçok modern yapay zeka hızlandırıcısı, INT8 işlemleri için özel birimler içerir ve bu da onu oldukça verimli kılar.
*   **Sınırlamalar:**
    *   **Doğruluk Ödünleşimi:** Genellikle sağlam olsa da, bazı modeller, özellikle LLM'ler gibi çok büyük ve karmaşık olanlar, INT8 kuantizasyonu ile fark edilebilir doğruluk düşüşleri yaşayabilir, özellikle aşırı aykırı değerlere sahip aktivasyonlar için.
    *   **Kalibrasyon Hassasiyeti:** Optimal kuantizasyon parametrelerini belirlemek için dikkatli kalibrasyon gerektirir.
*   **Uygulamalar:** INT8, bilgisayar görüşü (örn. görüntü sınıflandırma, nesne tespiti) ve model boyutlarının yönetilebilir olduğu ve doğruluk korumanın kritik olduğu geleneksel NLP görevleri için üretimde modelleri dağıtmak için bir standarttır. Genellikle çıkarım hızlandırması elde etmek için ilk tercihtir.

<a name="3.2-fp4-kuantizasyonu"></a>
### 3.2. FP4 Kuantizasyonu
**FP4 kuantizasyonu**, **Büyük Dil Modellerinin (LLM'ler)** muazzam bellek gereksinimleri tarafından yönlendirilen, daha da düşük hassasiyete doğru bir paradigma değişimi temsil eder. INT8 4 kat azalma sağlarken, 4-bit formatlar FP32'ye kıyasla bellekte 8 kat azalma hedeflemektedir. FP4, 4-bit kayan nokta formatına atıfta bulunur; bu format, INT8'in aksine, üstel ve mantis gibi kayan nokta sayılarının bazı özelliklerini korur, ancak ciddi şekilde kısaltılmıştır.
*   **Mekanizma:** FP4'ün kesin gösterimi değişebilir. `bitsandbytes` (LLM'leri optimize etmek için bir kütüphane) içinde kullanılan yaygın bir şema, **1-bit işaret, 2-bit üstel ve 1-bit mantis** (E2M1) veya bir denormal bit ile E2M0 gibi varyasyonlardır. Bu sınırlı hassasiyet, temsil edilebilir değerlerin seyrek ve düzensiz dağılmış olduğu anlamına gelir; bu, çok düşük bit genişliklerinde sabit noktalı tamsayılara göre dinamik aralıkları daha etkili bir şekilde yakalamak için faydalı olabilir.
*   **Avantajlar:**
    *   **Aşırı Bellek Azaltma:** FP32'ye kıyasla bellekte 8 kat azalma sağlayarak, **QLoRA** gibi tekniklerle çok daha büyük LLM'leri (örn. tüketici GPU'larında 65B modeller) yüklemeyi ve ince ayar yapmayı mümkün kılar.
    *   **LLM'ler İçin Performans:** Genel amaçlı donanımlarda aritmetik işlemler INT8'den daha karmaşık olsa da, önemli bellek bant genişliği tasarrufları, bellek erişiminin büyük bir darboğaz olduğu LLM çıkarımı için genellikle genel hızlanmalara dönüşür.
*   **Sınırlamalar:**
    *   **Doğruluk Zorlukları:** Ciddi hassasiyet azaltma, INT8'den daha fazla doğruluk düşüşüne yol açabilir. Bunu hafifletmek için dikkatli kuantizasyon yöntemi seçimi (örn. blok tabanlı kuantizasyon) ve sağlam teknikler gereklidir.
    *   **Donanım Desteği:** FP4 aritmetiği için yerel donanım desteği INT8'e göre daha az yaygındır, genellikle özel çekirdekler veya öykünme gerektirir, bu da performansı etkileyebilir.
*   **Uygulamalar:** Esas olarak LLM'ler bağlamında geliştirilmiş ve uygulanmıştır, özellikle verimli eğitim (örn. ince ayar için QLoRA) ve çıkarım için, modelleri belleğe sığdırma yeteneğinin en önemli endişe olduğu yerlerde.

<a name="3.3-nf4-kuantizasyonu-normalfloat-4-bit"></a>
### 3.3. NF4 Kuantizasyonu (NormalFloat 4-bit)
**NF4 (NormalFloat 4-bit)**, Dettmers ve diğerleri tarafından QLoRA makalelerinde (2023) tanıtılan yeni bir 4-bit kayan nokta kuantizasyon veri türüdür. Özellikle **normal dağılımı** takip eden veriler için "bilgi teorik olarak optimal" olacak şekilde tasarlanmıştır; bu, önceden eğitilmiş sinir ağı ağırlıkları için sıklıkla böyledir.
*   **Mekanizma:** Standart FP4 formatlarından farklı olarak, NF4, standart bir normal dağılım (ortalama 0, varyans 1) için **kuantizasyon hatasını** en aza indirmek üzere seçilmiş 2^4 = 16 belirli sayı kümesini tanımlar. Bu değerler tek tip değildir ve standart bir normal dağılımın niceliklerinden türetilmiştir. Farklı ortalamalara ve varyanslara sahip gerçek model ağırlıkları için, ağırlıklar önce NF4 değerlerine eşlenmeden önce standart bir normal dağılıma (veya buna yakın bir aralığa) normalize edilir ve ardından dekuantizasyondan sonra denormalize edilir. Bu blok tabanlı, ampirik olarak türetilmiş kuantizasyon şeması, NF4'ün genel FP4 veya INT4 formatlarına göre daha fazla bilgi korumasını sağlar.
*   **Avantajlar:**
    *   **Normal Dağılım İçin Optimalite:** Ağ ağırlıklarının istatistiksel özelliklerini kullanarak, NF4, LLM'ler için diğer 4-bit şemalarından daha yüksek doğruluk koruması sağlar.
    *   **LLM'ler İçin Üstün Doğruluk:** QLoRA ile birleştirildiğinde, NF4, taban model ağırlıkları için hala 8 kat bellek azaltma sağlarken, FP32'ye kıyasla minimum performans düşüşü ile LLM'leri ince ayar yapmayı mümkün kılar.
    *   **Bellek Verimliliği:** FP4 ile aynı 8 kat bellek azaltmayı sağlayarak, büyük LLM'lerin ticari donanımlarda dağıtılmasını ve ince ayarını mümkün kılar.
*   **Sınırlamalar:**
    *   **Hesaplama Ek Yükü:** Normalizasyon ve denormalizasyon adımları, tek tip olmayan eşleme ile birlikte, daha basit doğrusal INT8 şemalarına göre bazı hesaplama ek yükleri getirebilir. Performans için özel çekirdekler esastır.
    *   **Daha Az Genel Amaçlı:** LLM ağırlıkları için oldukça etkili olsa da, optimalitesi normal dağılım varsayımına bağlıdır, bu da onu belirli adaptasyonlar olmadan diğer veri türleri veya aktivasyon fonksiyonları için potansiyel olarak daha az ideal hale getirir.
*   **Uygulamalar:** NF4, özellikle `bitsandbytes` kütüphanesi ve QLoRA tekniği aracılığıyla, çok büyük dil modellerinin verimli bir şekilde eğitilmesi ve dağıtılması için bir köşe taşı haline gelmiştir. Araştırmacıların ve uygulayıcıların bellek kısıtlamaları nedeniyle ulaşılamayacak modellerle çalışmasına olanak tanıyarak, güçlü LLM'lere erişimi demokratikleştirir.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği
Bu kavramsal Python kodu, belirli bir tensör için temel bir simetrik doğrusal INT8 kuantizasyon sürecini göstermektedir. Herhangi bir özel kütüphane kullanmaz, ancak ölçekleme ve yuvarlama temel fikrini sergiler.

```python
import torch

def quantize_int8_symmetric(tensor_fp32):
    """
    Kavramsal simetrik INT8 kuantizasyonu.
    FP32 tensör değerlerini [-127, 127] INT8 aralığına eşler.
    
    Argümanlar:
        tensor_fp32 (torch.Tensor): Giriş 32-bit kayan noktalı tensör.

    Dönüş:
        torch.Tensor: Kuantize edilmiş 8-bit tamsayı tensör.
        float: Kuantizasyon için kullanılan ölçek faktörü.
    """
    # Tensördeki mutlak maksimum değeri belirle
    abs_max = torch.max(torch.abs(tensor_fp32))

    # Ölçek faktörünü hesapla
    # [-abs_max, abs_max] aralığını [-127, 127] aralığına eşliyoruz
    scale = abs_max / 127.0

    # Tensörü kuantize et
    # Ölçekle böl, en yakın tamsayıya yuvarla ve INT8 aralığına sıkıştır
    quantized_tensor = torch.round(tensor_fp32 / scale)
    quantized_tensor = torch.clamp(quantized_tensor, -127, 127)
    
    # Tamsayı tipine dönüştür
    quantized_tensor = quantized_tensor.to(torch.int8)

    return quantized_tensor, scale

def dequantize_int8_symmetric(tensor_int8, scale):
    """
    Kavramsal simetrik INT8 dekuantizasyonu.
    INT8 tensör değerlerini yaklaşık FP32'ye geri eşler.

    Argümanlar:
        tensor_int8 (torch.Tensor): Kuantize edilmiş 8-bit tamsayı tensör.
        scale (float): Kuantizasyon sırasında kullanılan ölçek faktörü.

    Dönüş:
        torch.Tensor: Dekuantize edilmiş 32-bit kayan noktalı tensör.
    """
    # Ölçekle çarparak dekuantize et
    dequantized_tensor = tensor_int8.to(torch.float32) * scale
    return dequantized_tensor

# Örnek Kullanım:
# Bir örnek FP32 tensörü oluştur
sample_tensor_fp32 = torch.tensor([-1.5, 0.0, 0.75, 2.5, -0.1], dtype=torch.float32)
print("Orijinal FP32 Tensör:", sample_tensor_fp32)

# Kuantize et
quantized_tensor_int8, quantization_scale = quantize_int8_symmetric(sample_tensor_fp32)
print("Kuantize Edilmiş INT8 Tensör:", quantized_tensor_int8)
print("Kuantizasyon Ölçeği:", quantization_scale)

# Dekuantize et
dequantized_tensor_fp32 = dequantize_int8_symmetric(quantized_tensor_int8, quantization_scale)
print("Dekuantize Edilmiş FP32 Tensör:", dequantized_tensor_fp32)

# Yaklaşım hatasını göster
print("Yaklaşım Hatası (Orijinal - Dekuantize Edilmiş):", sample_tensor_fp32 - dequantized_tensor_fp32)

(Kod örneği bölümünün sonu)
```
<a name="5-sonuç"></a>
## 5. Sonuç
Model kuantizasyonu, giderek karmaşıklaşan sinir ağlarının yol açtığı hesaplama ve bellek yüklerine pratik bir çözüm sunarak büyük ölçekli yapay zeka çağında vazgeçilmez bir teknik haline gelmiştir. Sayısal hassasiyeti stratejik olarak azaltarak, güçlü modellerin kenar cihazlardan kurumsal düzeydeki sunuculara kadar daha geniş bir donanım yelpazesinde dağıtımını sağlar, çıkarım hızını ve enerji verimliliğini önemli ölçüde artırır.

**INT8 kuantizasyonu**, geniş bir model yelpazesi için genellikle kabul edilebilir doğrulukla önemli kazançlar sağlayan sağlam ve yaygın olarak desteklenen bir standart olmaya devam etmektedir. Olgunluğu ve kapsamlı donanım desteği, onu üretim dağıtımları için başvurulan bir seçenek haline getirmektedir. Modellerin, özellikle LLM'lerin büyümesi devam ettikçe, daha düşük hassasiyete olan ihtiyaç, 4-bit formatların benimsenmesini hızlandırmıştır. **FP4 kuantizasyonu**, doğruluk korumada daha büyük zorluklar olsa da ve özel donanım veya yazılım optimizasyonları gerektirse de radikal bellek tasarrufları sunar. Bilgi teorisine dayanan **NF4 kuantizasyonu**, sinir ağı ağırlıklarının istatistiksel özelliklerine özel olarak uyarlanmış, 4-bit hassasiyet için sofistike bir yaklaşım sunar. QLoRA gibi tekniklerle entegrasyonu, büyük dil modeli ince ayarına ve çıkarımına erişimi demokratikleştirerek dönüştürücü olmuştur.

INT8, FP4 ve NF4 arasındaki seçim, büyük ölçüde belirli uygulamaya, mevcut donanıma ve bellek ayak izi, çıkarım hızı ve model doğruluğu arasındaki kabul edilebilir ödünleşime bağlıdır. Üretken yapay zeka araştırmaları model ölçeğinin sınırlarını zorlamaya devam ederken, kuantizasyon tekniklerindeki gelişmeler, teorik atılımları pratik, verimli ve erişilebilir yapay zeka çözümlerine dönüştürmek için kritik olmaya devam edecektir.

