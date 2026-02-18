# MobileNet: Efficient Convolutional Neural Networks

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Depthwise Separable Convolutions: The Core Innovation](#2-depthwise-separable-convolutions-the-core-innovation)
  - [2.1. Standard Convolution Review](#21-standard-convolution-review)
  - [2.2. Depthwise Convolution](#22-depthwise-convolution)
  - [2.3. Pointwise Convolution](#23-pointwise-convolution)
  - [2.4. Computational Efficiency](#24-computational-efficiency)
- [3. MobileNet Architecture and Hyperparameters](#3-mobilenet-architecture-and-hyperparameters)
  - [3.1. Overall Architecture](#31-overall-architecture)
  - [3.2. Width Multiplier ($\alpha$)](#32-width-multiplier-alpha)
  - [3.3. Resolution Multiplier ($\rho$)](#33-resolution-multiplier-rho)
- [4. Code Example](#4-code-example)
- [5. Applications and Impact](#5-applications-and-impact)
- [6. Conclusion](#6-conclusion)

<br>

## 1. Introduction
Convolutional Neural Networks (CNNs) have revolutionized computer vision, achieving state-of-the-art performance in tasks like image classification, object detection, and semantic segmentation. However, traditional deep CNNs are often computationally intensive and possess a large number of parameters, making their deployment on resource-constrained devices (e.g., mobile phones, embedded systems, IoT devices) challenging due to limitations in processing power, memory, and battery life.

**MobileNet**, introduced by Google in 2017, addresses this critical challenge by proposing a class of efficient models designed specifically for mobile and embedded vision applications. The core innovation behind MobileNet is the introduction of **depthwise separable convolutions**, which significantly reduce computational cost and model size while maintaining competitive accuracy. This document will delve into the foundational concepts of MobileNet, explore its architecture, and discuss its impact on the field of efficient deep learning.

## 2. Depthwise Separable Convolutions: The Core Innovation
The primary building block of MobileNet is the **depthwise separable convolution**, a factorized convolution that decomposes a standard convolution into two separate, simpler operations: a depthwise convolution and a pointwise convolution. This factorization drastically reduces the number of computations and parameters.

### 2.1. Standard Convolution Review
A standard convolution operation simultaneously filters the input channels and combines them to produce a new set of output channels. For an input feature map $F$ of size $D_F \times D_F \times M$ (width $\times$ height $\times$ channels) and a convolutional kernel $K$ of size $D_K \times D_K \times M \times N$ (width $\times$ height $\times$ input channels $\times$ output channels), the standard convolution produces an output feature map $G$ of size $D_G \times D_G \times N$. The computational cost is approximately $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$.

### 2.2. Depthwise Convolution
The first part of a depthwise separable convolution is the **depthwise convolution**. This operation applies a single filter to each input channel independently. If there are $M$ input channels, $M$ different $D_K \times D_K \times 1$ filters are used, each processing one channel. This produces an intermediate feature map of size $D_G \times D_G \times M$. Importantly, this step only filters the input channels; it does not combine them. The computational cost is $D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$.

### 2.3. Pointwise Convolution
The second part is the **pointwise convolution**, which is a $1 \times 1$ convolution. This operation combines the outputs of the depthwise convolution across channels. It uses $N$ filters, each of size $1 \times 1 \times M$, to create a linearly combined output for each of the $N$ output channels. This step effectively learns the linear combination of the depthwise outputs across channels. The computational cost is $1 \cdot 1 \cdot M \cdot N \cdot D_F \cdot D_F$.

### 2.4. Computational Efficiency
By decomposing the standard convolution, the total computational cost of a depthwise separable convolution becomes the sum of the depthwise and pointwise convolution costs:
$(D_K \cdot D_K \cdot M \cdot D_F \cdot D_F) + (M \cdot N \cdot D_F \cdot D_F)$

The ratio of the computational cost of depthwise separable convolution to standard convolution is:
$\frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \frac{1}{N} + \frac{1}{D_K^2}$

This shows a significant reduction in computation, typically by a factor of 8 to 9 for common $3 \times 3$ filters. This reduction scales with the square of the kernel size and inversely with the number of output channels. This efficiency gain is the cornerstone of MobileNet's performance on mobile devices.

## 3. MobileNet Architecture and Hyperparameters
The MobileNet architecture is built upon a sequence of depthwise separable convolution blocks. It also introduces two global hyperparameters that allow model builders to trade off between latency, size, and accuracy for their specific application requirements.

### 3.1. Overall Architecture
The MobileNet model typically starts with a standard $3 \times 3$ convolution, followed by 13 depthwise separable convolution layers. Each depthwise separable layer consists of a depthwise convolution followed by a pointwise convolution. Both convolutional parts are typically followed by a **Batch Normalization** layer and a **ReLU** activation function, with the exception of the final fully connected layer which feeds into a softmax for classification. Downsampling is handled by stride-2 convolutions in either the depthwise or pointwise layers. After the convolutional layers, a global average pooling layer reduces spatial dimensions, followed by a fully connected layer (without ReLU) and a softmax layer for classification.

### 3.2. Width Multiplier ($\alpha$)
The **width multiplier**, denoted as $\alpha$ (alpha), allows for reducing the number of channels (feature maps) in each layer of the network. For a given layer, the number of input channels $M$ becomes $\alpha M$, and the number of output channels $N$ becomes $\alpha N$. The width multiplier effectively creates thinner models.
-   Values: $\alpha \in (0, 1]$, commonly 1.0, 0.75, 0.5, 0.25.
-   Impact: Reduces computational cost and the number of parameters quadratically (by $\alpha^2$). This fine-grained control enables tailoring model size and speed to specific latency requirements.

### 3.3. Resolution Multiplier ($\rho$)
The **resolution multiplier**, denoted as $\rho$ (rho), allows for reducing the input image resolution. This multiplier is applied to the input image and subsequently to all internal representation layers.
-   Values: $\rho \in (0, 1]$, commonly input resolutions like 224, 192, 160, 128 (corresponding to $\rho$ values relative to 224).
-   Impact: Reduces computational cost quadratically (by $\rho^2$). For instance, reducing the input resolution from $224 \times 224$ to $128 \times 128$ significantly reduces computations.

By combining the width multiplier and resolution multiplier, developers can efficiently explore a large space of models to find the optimal balance for their application.

## 4. Code Example
Here's a simplified Keras example demonstrating how to implement a single depthwise separable convolution block (DepthwiseConv2D + Conv2D) for a MobileNet-like structure.

```python
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, Input
from tensorflow.keras.models import Model

def depthwise_separable_block(inputs, filters, stride=1):
    """
    Creates a depthwise separable convolution block.
    
    Args:
        inputs: Input tensor.
        filters: Number of output filters (for pointwise convolution).
        stride: Stride for the depthwise convolution.
    
    Returns:
        Output tensor after the depthwise separable block.
    """
    # Depthwise Convolution
    # Apply one filter per input channel
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise Convolution (1x1 convolution)
    # Combine the outputs of the depthwise convolution
    x = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

# Example usage:
input_shape = (128, 128, 3) # Example input: 128x128 RGB image
inputs = Input(shape=input_shape)

# Apply a depthwise separable block with 64 output filters
outputs = depthwise_separable_block(inputs, filters=64, stride=1)

# Create a model
model = Model(inputs=inputs, outputs=outputs)
model.summary()

(End of code example section)
```

## 5. Applications and Impact
MobileNet's efficiency makes it suitable for a wide array of applications where computational resources are limited:
*   **Mobile and Embedded Devices**: Enabling on-device AI for smartphones, drones, robots, and other IoT devices.
*   **Real-time Object Detection**: Serving as a lightweight backbone for object detection frameworks like SSD (Single Shot MultiBox Detector) or YOLO (You Only Look Once), allowing for real-time inference on less powerful hardware.
*   **Augmented Reality (AR) and Virtual Reality (VR)**: Powering real-time scene understanding and object tracking in AR/VR applications.
*   **Autonomous Systems**: Contributing to perception systems in self-driving cars or industrial robots where low latency is crucial.

MobileNet has significantly influenced the research and development of efficient neural networks, paving the way for subsequent models like MobileNetV2 and MobileNetV3, which further optimize the architecture with innovations like inverted residuals and neural architecture search.

## 6. Conclusion
MobileNet represents a landmark achievement in the field of efficient deep learning. By introducing **depthwise separable convolutions** and providing **width** and **resolution multipliers**, it offers a powerful framework for constructing lightweight, high-performing convolutional neural networks suitable for deployment on resource-constrained devices. Its impact extends beyond simply enabling on-device AI; it has fostered a new era of research focused on balancing model complexity with practical deployment constraints, demonstrating that high accuracy does not always require massive computational resources. MobileNet remains a foundational model for anyone working on edge AI and real-time computer vision applications.

---
<br>

<a name="türkçe-içerik"></a>
## MobileNet: Verimli Evrişimsel Sinir Ağları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Derinlik Odaklı Ayrılabilir Evrişimler: Temel Yenilik](#2-derinlik-odaklı-ayrılabilir-evrişimler-temel-yenilik)
  - [2.1. Standart Evrişim İncelemesi](#21-standart-evrişim-incelemesi)
  - [2.2. Derinlik Odaklı Evrişim (Depthwise Convolution)](#22-derinlik-odaklı-evrişim-depthwise-convolution)
  - [2.3. Noktasal Evrişim (Pointwise Convolution)](#23-noktasal-evrişim-pointwise-convolution)
  - [2.4. Hesaplama Verimliliği](#24-hesaplama-verimliliği)
- [3. MobileNet Mimarisi ve Hiperparametreler](#3-mobilenet-mimarisi-ve-hiperparametreler)
  - [3.1. Genel Mimari](#31-genel-mimari)
  - [3.2. Genişlik Çarpanı ($\alpha$)](#32-genişlik-çarpanı-alpha)
  - [3.3. Çözünürlük Çarpanı ($\rho$)](#33-çözünürlük-çarpanı-rho)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Uygulamalar ve Etkisi](#5-uygulamalar-ve-etkisi)
- [6. Sonuç](#6-sonuç)

<br>

## 1. Giriş
Evrişimsel Sinir Ağları (CNN'ler) bilgisayar görüşünde devrim yaratarak görüntü sınıflandırma, nesne tespiti ve anlamsal segmentasyon gibi görevlerde son teknoloji performans elde etmiştir. Ancak, geleneksel derin CNN'ler genellikle yoğun hesaplamalıdır ve çok sayıda parametreye sahiptir, bu da işlem gücü, bellek ve pil ömrü kısıtlamaları nedeniyle kaynak kısıtlı cihazlara (örn. cep telefonları, gömülü sistemler, IoT cihazları) dağıtılmalarını zorlaştırmaktadır.

Google tarafından 2017'de tanıtılan **MobileNet**, özellikle mobil ve gömülü görüş uygulamaları için tasarlanmış verimli modeller sınıfı önererek bu kritik zorluğun üstesinden gelir. MobileNet'in temel yeniliği, hesaplama maliyetini ve model boyutunu önemli ölçüde azaltırken rekabetçi doğruluğu koruyan **derinlik odaklı ayrılabilir evrişimlerin** (depthwise separable convolutions) tanıtılmasıdır. Bu belge, MobileNet'in temel kavramlarını inceleyecek, mimarisini keşfedecek ve verimli derin öğrenme alanındaki etkisini tartışacaktır.

## 2. Derinlik Odaklı Ayrılabilir Evrişimler: Temel Yenilik
MobileNet'in birincil yapı taşı, standart bir evrişimi iki ayrı, daha basit işleme ayıran faktörlü bir evrişim olan **derinlik odaklı ayrılabilir evrişimdir**: bir derinlik odaklı evrişim ve bir noktasal evrişim. Bu faktörizasyon, hesaplama sayısını ve parametreleri çarpıcı biçimde azaltır.

### 2.1. Standart Evrişim İncelemesi
Standart bir evrişim işlemi, girdi kanallarını eş zamanlı olarak filtreler ve yeni bir çıktı kanalları kümesi oluşturmak için bunları birleştirir. $D_F \times D_F \times M$ (genişlik $\times$ yükseklik $\times$ kanal) boyutunda bir girdi özellik haritası $F$ ve $D_K \times D_K \times M \times N$ (genişlik $\times$ yükseklik $\times$ girdi kanalı $\times$ çıktı kanalı) boyutunda bir evrişim çekirdeği $K$ için, standart evrişim $D_G \times D_G \times N$ boyutunda bir çıktı özellik haritası $G$ üretir. Hesaplama maliyeti yaklaşık olarak $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$'dir.

### 2.2. Derinlik Odaklı Evrişim (Depthwise Convolution)
Derinlik odaklı ayrılabilir evrişimin ilk kısmı **derinlik odaklı evrişimdir**. Bu işlem, her bir girdi kanalına bağımsız olarak tek bir filtre uygular. $M$ tane girdi kanalı varsa, her biri bir kanalı işleyen $M$ farklı $D_K \times D_K \times 1$ filtre kullanılır. Bu, $D_G \times D_G \times M$ boyutunda ara bir özellik haritası üretir. Önemlisi, bu adım yalnızca girdi kanallarını filtreler; onları birleştirmez. Hesaplama maliyeti $D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$'dir.

### 2.3. Noktasal Evrişim (Pointwise Convolution)
İkinci kısım, $1 \times 1$ bir evrişim olan **noktasal evrişimdir**. Bu işlem, derinlik odaklı evrişimin çıktılarını kanallar boyunca birleştirir. $N$ adet çıktı kanalının her biri için doğrusal olarak birleştirilmiş bir çıktı oluşturmak üzere her biri $1 \times 1 \times M$ boyutunda olan $N$ filtre kullanır. Bu adım, derinlik odaklı çıktıların kanallar arası doğrusal kombinasyonunu etkili bir şekilde öğrenir. Hesaplama maliyeti $1 \cdot 1 \cdot M \cdot N \cdot D_F \cdot D_F$'dir.

### 2.4. Hesaplama Verimliliği
Standart evrişimi ayrıştırarak, derinlik odaklı ayrılabilir evrişimin toplam hesaplama maliyeti, derinlik odaklı ve noktasal evrişim maliyetlerinin toplamı olur:
$(D_K \cdot D_K \cdot M \cdot D_F \cdot D_F) + (M \cdot N \cdot D_F \cdot D_F)$

Derinlik odaklı ayrılabilir evrişimin hesaplama maliyetinin standart evrişime oranı:
$\frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \frac{1}{N} + \frac{1}{D_K^2}$

Bu, tipik $3 \times 3$ filtreler için 8 ila 9 katlık önemli bir hesaplama azalması gösterir. Bu azalma, çekirdek boyutunun karesiyle ölçeklenir ve çıktı kanalı sayısıyla ters orantılıdır. Bu verimlilik artışı, MobileNet'in mobil cihazlardaki performansının temelini oluşturur.

## 3. MobileNet Mimarisi ve Hiperparametreler
MobileNet mimarisi, bir dizi derinlik odaklı ayrılabilir evrişim bloğu üzerine inşa edilmiştir. Ayrıca, model oluşturucuların belirli uygulama gereksinimleri için gecikme, boyut ve doğruluk arasında denge kurmalarına olanak tanıyan iki global hiperparametre sunar.

### 3.1. Genel Mimari
MobileNet modeli genellikle standart bir $3 \times 3$ evrişim ile başlar, ardından 13 derinlik odaklı ayrılabilir evrişim katmanı gelir. Her bir derinlik odaklı ayrılabilir katman, bir derinlik odaklı evrişim ve ardından bir noktasal evrişimden oluşur. Her iki evrişimsel kısım da genellikle bir **Batch Normalization** katmanı ve bir **ReLU** aktivasyon fonksiyonu ile takip edilir, son tam bağlı katman hariç, bu katman sınıflandırma için bir softmax'e beslenir. Örnekleme azaltma (downsampling), derinlik odaklı veya noktasal katmanlardaki stride-2 evrişimlerle ele alınır. Evrişimsel katmanlardan sonra, global ortalama havuzlama katmanı uzamsal boyutları azaltır, ardından tam bağlı bir katman (ReLU olmadan) ve sınıflandırma için bir softmax katmanı gelir.

### 3.2. Genişlik Çarpanı ($\alpha$)
**Genişlik çarpanı**, $\alpha$ (alfa) ile gösterilir, ağın her katmanındaki kanal (özellik haritası) sayısını azaltmaya olanak tanır. Belirli bir katman için, girdi kanalı sayısı $M$, $\alpha M$ olur ve çıktı kanalı sayısı $N$, $\alpha N$ olur. Genişlik çarpanı, etkili bir şekilde daha ince modeller oluşturur.
-   Değerler: $\alpha \in (0, 1]$, yaygın olarak 1.0, 0.75, 0.5, 0.25.
-   Etkisi: Hesaplama maliyetini ve parametre sayısını karesel olarak ($\alpha^2$ ile) azaltır. Bu ince ayar kontrolü, model boyutunu ve hızını belirli gecikme gereksinimlerine göre uyarlamayı sağlar.

### 3.3. Çözünürlük Çarpanı ($\rho$)
**Çözünürlük çarpanı**, $\rho$ (rho) ile gösterilir, girdi görüntüsü çözünürlüğünü azaltmaya olanak tanır. Bu çarpan, girdi görüntüsüne ve ardından tüm dahili temsil katmanlarına uygulanır.
-   Değerler: $\rho \in (0, 1]$, yaygın olarak 224, 192, 160, 128 gibi girdi çözünürlükleri (224'e göre $\rho$ değerlerine karşılık gelir).
-   Etkisi: Hesaplama maliyetini karesel olarak ($\rho^2$ ile) azaltır. Örneğin, girdi çözünürlüğünü $224 \times 224$'ten $128 \times 128$'e düşürmek, hesaplamaları önemli ölçüde azaltır.

Genişlik çarpanı ve çözünürlük çarpanını birleştirerek, geliştiriciler uygulamaları için en uygun dengeyi bulmak üzere geniş bir model alanını verimli bir şekilde keşfedebilirler.

## 4. Kod Örneği
MobileNet benzeri bir yapı için tek bir derinlik odaklı ayrılabilir evrişim bloğunu (DepthwiseConv2D + Conv2D) nasıl uygulayacağınızı gösteren basitleştirilmiş bir Keras örneği aşağıdadır.

```python
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, Input
from tensorflow.keras.models import Model

def depthwise_separable_block(inputs, filters, stride=1):
    """
    Derinlik odaklı ayrılabilir bir evrişim bloğu oluşturur.
    
    Argümanlar:
        inputs: Girdi tensörü.
        filters: Çıktı filtre sayısı (noktasal evrişim için).
        stride: Derinlik odaklı evrişim için adım (stride).
    
    Dönüş:
        Derinlik odaklı ayrılabilir bloktan sonraki çıktı tensörü.
    """
    # Derinlik Odaklı Evrişim
    # Her girdi kanalı için tek bir filtre uygular
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Noktasal Evrişim (1x1 evrişim)
    # Derinlik odaklı evrişimin çıktılarını birleştirir
    x = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

# Örnek kullanım:
input_shape = (128, 128, 3) # Örnek girdi: 128x128 RGB görüntü
inputs = Input(shape=input_shape)

# 64 çıktı filtresi ile derinlik odaklı ayrılabilir bir blok uygula
outputs = depthwise_separable_block(inputs, filters=64, stride=1)

# Bir model oluştur
model = Model(inputs=inputs, outputs=outputs)
model.summary()

(Kod örneği bölümünün sonu)
```

## 5. Uygulamalar ve Etkisi
MobileNet'in verimliliği, hesaplama kaynaklarının kısıtlı olduğu geniş bir uygulama yelpazesi için uygun hale getirir:
*   **Mobil ve Gömülü Cihazlar**: Akıllı telefonlar, dronlar, robotlar ve diğer IoT cihazları için cihaz içi yapay zekayı etkinleştirme.
*   **Gerçek Zamanlı Nesne Tespiti**: SSD (Single Shot MultiBox Detector) veya YOLO (You Only Look Once) gibi nesne tespit çerçeveleri için hafif bir omurga görevi görerek daha az güçlü donanımlarda gerçek zamanlı çıkarım yapılmasına olanak tanır.
*   **Artırılmış Gerçeklik (AR) ve Sanal Gerçeklik (VR)**: AR/VR uygulamalarında gerçek zamanlı sahne anlama ve nesne takibini güçlendirme.
*   **Otonom Sistemler**: Gecikmenin kritik olduğu sürücüsüz araçlarda veya endüstriyel robotlarda algı sistemlerine katkıda bulunma.

MobileNet, verimli sinir ağlarının araştırma ve geliştirilmesini önemli ölçüde etkilemiş, ters kalıntılar (inverted residuals) ve nöral mimari arama gibi yeniliklerle mimariyi daha da optimize eden MobileNetV2 ve MobileNetV3 gibi sonraki modellere zemin hazırlamıştır.

## 6. Sonuç
MobileNet, verimli derin öğrenme alanında dönüm noktası niteliğinde bir başarıyı temsil etmektedir. **Derinlik odaklı ayrılabilir evrişimleri** tanıtarak ve **genişlik** ile **çözünürlük çarpanları** sağlayarak, kaynak kısıtlı cihazlara dağıtım için uygun, hafif, yüksek performanslı evrişimsel sinir ağları oluşturmak için güçlü bir çerçeve sunar. Etkisi sadece cihaz içi yapay zekayı etkinleştirmenin ötesine geçer; model karmaşıklığını pratik dağıtım kısıtlamalarıyla dengelemeye odaklanan yeni bir araştırma dönemini teşvik etmiş ve yüksek doğruluğun her zaman devasa hesaplama kaynakları gerektirmediğini göstermiştir. MobileNet, kenar yapay zeka ve gerçek zamanlı bilgisayar görüşü uygulamaları üzerinde çalışan herkes için temel bir model olmaya devam etmektedir.


