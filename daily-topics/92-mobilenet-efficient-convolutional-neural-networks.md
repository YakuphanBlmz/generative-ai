# MobileNet: Efficient Convolutional Neural Networks

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Core Innovation: Depthwise Separable Convolutions](#2-the-core-innovation-depthwise-separable-convolutions)
  - [2.1. Standard Convolution](#21-standard-convolution)
  - [2.2. Depthwise Convolution](#22-depthwise-convolution)
  - [2.3. Pointwise Convolution](#23-pointwise-convolution)
  - [2.4. Advantages of Depthwise Separable Convolutions](#24-advantages-of-depthwise-separable-convolutions)
- [3. MobileNet Architectures and Hyperparameters](#3-mobilenet-architectures-and-hyperparameters)
  - [3.1. MobileNetV1](#31-mobilenetv1)
  - [3.2. MobileNetV2](#32-mobilenetv2)
  - [3.3. MobileNetV3](#33-mobilenetv3)
  - [3.4. Hyperparameters for Efficiency Trade-offs](#34-hyperparameters-for-efficiency-trade-offs)
- [4. Code Example](#4-code-example)
- [5. Applications and Impact](#5-applications-and-impact)
- [6. Conclusion](#6-conclusion)

<br>

<a name="1-introduction"></a>
## 1. Introduction

In the rapidly evolving landscape of Artificial Intelligence, especially within the domain of computer vision, **Convolutional Neural Networks (CNNs)** have demonstrated unparalleled success in tasks such as image classification, object detection, and semantic segmentation. However, traditional CNN architectures, such as VGG or ResNet, often come with a substantial computational cost and a large number of parameters, making their deployment on resource-constrained devices like mobile phones, embedded systems, or edge devices challenging. This limitation spurred research into developing more **efficient convolutional neural networks**.

**MobileNet**, first introduced by Google in 2017, represents a significant breakthrough in this area. It is a class of efficient models designed specifically for mobile and embedded vision applications where computational power, memory, and energy consumption are critical factors. The fundamental innovation underpinning MobileNet's efficiency is the introduction of **depthwise separable convolutions**, which drastically reduce the number of parameters and computations while maintaining competitive accuracy. This document delves into the architectural details of MobileNet, its core components, various iterations, and its profound impact on deploying deep learning models in real-world, constrained environments.

<a name="2-the-core-innovation-depthwise-separable-convolutions"></a>
## 2. The Core Innovation: Depthwise Separable Convolutions

The cornerstone of MobileNet's efficiency lies in its strategic replacement of standard convolutional layers with **depthwise separable convolutions**. This factorization effectively breaks down a standard convolution into two separate, more efficient operations: a **depthwise convolution** and a **pointwise convolution**.

<a name="21-standard-convolution"></a>
### 2.1. Standard Convolution

To appreciate the efficiency gains, it is essential to first understand the mechanics and computational cost of a **standard convolution**. In a standard 2D convolution, a single filter is applied across all input channels simultaneously to produce one output channel. If an input feature map has dimensions $D_F \times D_F \times M$ (height, width, channels) and we apply $N$ filters of size $D_K \times D_K \times M$, the output feature map will have dimensions $D_G \times D_G \times N$. The computational cost (number of multiplications and additions) for a standard convolution is approximately $D_K \cdot D_K \cdot M \cdot N \cdot D_G \cdot D_G$. This implies that the cost grows quadratically with filter size and linearly with both input and output channels.

<a name="22-depthwise-convolution"></a>
### 2.2. Depthwise Convolution

The first part of a depthwise separable convolution is the **depthwise convolution**. Instead of applying a single filter across all input channels, a depthwise convolution applies a *single filter per input channel*. This means if the input feature map has $M$ channels, $M$ different filters (each of size $D_K \times D_K \times 1$) are used. Each filter processes its corresponding input channel independently, producing $M$ filtered output channels.
The computational cost for the depthwise convolution is approximately $D_K \cdot D_K \cdot M \cdot D_G \cdot D_G$. Notice that the dependency on the number of output channels $N$ from the standard convolution is absent here. This significantly reduces computation, as $N$ is often large.

<a name="23-pointwise-convolution"></a>
### 2.3. Pointwise Convolution

The depthwise convolution step effectively filters each input channel, but it does not combine information across different channels. This is where the **pointwise convolution** comes into play. A pointwise convolution is a standard 1x1 convolution (i.e., a filter size of $1 \times 1$). It is applied to the output of the depthwise convolution. Specifically, if the depthwise convolution produced $M$ feature maps, the pointwise convolution uses $N$ filters, each of size $1 \times 1 \times M$. Each of these $N$ filters combines the information from all $M$ depthwise output channels to produce a single output channel. Consequently, $N$ such filters produce an output feature map with $N$ channels.
The computational cost for the pointwise convolution is approximately $1 \cdot 1 \cdot M \cdot N \cdot D_G \cdot D_G$.

<a name="24-advantages-of-depthwise-separable-convolutions"></a>
### 2.4. Advantages of Depthwise Separable Convolutions

By factorizing a standard convolution into a depthwise convolution and a pointwise convolution, MobileNet achieves substantial reductions in both computational complexity and the number of parameters. The total cost of a depthwise separable convolution is approximately $(D_K \cdot D_K \cdot M \cdot D_G \cdot D_G) + (M \cdot N \cdot D_G \cdot D_G)$.
Comparing this to the standard convolution cost ($D_K \cdot D_K \cdot M \cdot N \cdot D_G \cdot D_G$), the reduction factor is approximately $\frac{1}{N} + \frac{1}{D_K^2}$. For typical filter sizes (e.g., $3 \times 3$) and large numbers of output channels, this factor can be a reduction of 8 to 9 times. This efficiency gain is crucial for real-time applications on constrained devices.

<a name="3-mobilenet-architectures-and-hyperparameters"></a>
## 3. MobileNet Architectures and Hyperparameters

MobileNet is not a single model but a family of architectures, with several iterations building upon the core concept of depthwise separable convolutions to further enhance efficiency and performance.

<a name="31-mobilenetv1"></a>
### 3.1. MobileNetV1

The original MobileNet architecture, introduced in 2017, established the fundamental block as a depthwise separable convolution followed by **Batch Normalization** and a **ReLU** activation function. It primarily consisted of 13 such layers. MobileNetV1 demonstrated that these compact models could achieve comparable accuracy to larger, more complex networks on various vision tasks, but with significantly fewer parameters and computations.

<a name="32-mobilenetv2"></a>
### 3.2. MobileNetV2

Released in 2018, MobileNetV2 introduced two key architectural innovations to improve performance:
*   **Inverted Residual Blocks:** Unlike standard residual blocks that compress spatial dimensions and then expand, inverted residual blocks first expand the number of channels using a $1 \times 1$ convolution, then apply a depthwise convolution, and finally project back to a lower number of channels using another $1 \times 1$ convolution. This creates a "bottleneck" structure for efficient feature learning.
*   **Linear Bottlenecks:** After the final $1 \times 1$ projection layer in an inverted residual block, MobileNetV2 removes the ReLU activation function. This is based on the insight that high-dimensional embeddings can capture all necessary information, but low-dimensional embeddings (the bottleneck) can lose information if non-linearities are applied. Using linear activations in the bottleneck helps prevent information loss.

These modifications allowed MobileNetV2 to achieve higher accuracy than V1 while maintaining or even reducing computational costs.

<a name="33-mobilenetv3"></a>
### 3.3. MobileNetV3

Introduced in 2019, MobileNetV3 pushed the boundaries of efficiency and accuracy even further by combining several advanced techniques:
*   **Neural Architecture Search (NAS):** MobileNetV3 leveraged automated search techniques, specifically **MnasNet**, to discover optimal network architectures for specific latency constraints.
*   **Squeeze-and-Excitation (SE) Blocks:** These lightweight attention mechanisms dynamically recalibrate channel-wise feature responses, improving feature representation.
*   **Novel Activation Functions:** MobileNetV3 introduced **h-swish** (hard swish) and **h-sigmoid** as efficient alternatives to Swish and Sigmoid, which are computationally expensive.
*   **Redesigned Initial and Final Layers:** The initial $3 \times 3$ convolutional layer and the final classification layers were redesigned to reduce latency while preserving high-dimensional features.

MobileNetV3 offers two main variants: MobileNetV3-Large for high-resource cases and MobileNetV3-Small for low-resource cases, providing a flexible spectrum of performance-efficiency trade-offs.

<a name="34-hyperparameters-for-efficiency-trade-offs"></a>
### 3.4. Hyperparameters for Efficiency Trade-offs

Beyond the architectural choices, MobileNet models offer two simple global hyperparameters that allow developers to fine-tune the balance between latency and accuracy:
*   **Width Multiplier ($\alpha$):** This parameter (typically between 0.25 and 1.0) scales the number of channels in each layer uniformly. An $\alpha$ of 0.5, for example, halves the number of input and output channels for every layer, significantly reducing computations and parameters, but also potentially reducing accuracy.
*   **Resolution Multiplier ($\rho$):** This parameter (typically between 0.75 and 1.0) scales the input image resolution. A smaller input resolution reduces the computational cost of all layers, offering another knob for efficiency adjustments.

By combining these hyperparameters, MobileNet allows for a family of models that can be tailored to specific application requirements and hardware constraints.

<a name="4-code-example"></a>
## 4. Code Example

Below is a short Python code snippet illustrating how to implement a depthwise separable convolution layer using TensorFlow/Keras. This function combines `DepthwiseConv2D` and `Conv2D` (for pointwise convolution) to create a single efficient block.

```python
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU

def depthwise_separable_conv_block(inputs, filters, strides=(1, 1), kernel_size=(3, 3)):
    """
    Creates a depthwise separable convolution block.

    Args:
        inputs: Input tensor to the block.
        filters: Number of output filters (for the pointwise convolution).
        strides: Stride for the depthwise convolution.
        kernel_size: Kernel size for the depthwise convolution.

    Returns:
        Output tensor from the depthwise separable convolution block.
    """
    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise Convolution (1x1 convolution)
    x = Conv2D(filters=filters,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Example usage:
# Create a dummy input tensor
input_shape = (1, 64, 64, 3) # Batch, Height, Width, Channels
dummy_input = tf.random.normal(input_shape)

# Create a depthwise separable convolution block with 64 output filters
output_filters = 64
output = depthwise_separable_conv_block(dummy_input, output_filters, strides=(2, 2))

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

(End of code example section)
```

<a name="5-applications-and-impact"></a>
## 5. Applications and Impact

MobileNet architectures have profoundly impacted the accessibility and deployment of deep learning models across a multitude of applications, particularly in resource-constrained environments:

*   **Mobile and Embedded Devices:** MobileNet models are the backbone for on-device AI capabilities in smartphones, smart cameras, drones, and other IoT devices. This enables features like real-time object detection, facial recognition, and augmented reality without relying heavily on cloud-based processing, improving privacy, latency, and robustness.
*   **Real-time Object Detection:** MobileNet has been integrated into efficient object detection frameworks like **SSD-MobileNet** and **YOLO-Lite**, enabling real-time detection on edge devices for applications in autonomous driving, robotics, and surveillance.
*   **Computer Vision Libraries:** Its influence extends to popular computer vision libraries (e.g., OpenCV's DNN module) and frameworks, making efficient pre-trained models readily available for various tasks.
*   **Healthcare:** Deployment of diagnostic AI models directly on medical imaging equipment or portable devices.
*   **Robotics:** Enabling robots to perceive and interact with their environment in real-time, even with limited onboard computing power.

The ability of MobileNet to deliver high performance with minimal computational overhead has democratized deep learning, making advanced AI capabilities available to a broader range of hardware and applications.

<a name="6-conclusion"></a>
## 6. Conclusion

MobileNet represents a monumental step forward in the quest for efficient deep learning. By meticulously redesigning the fundamental building blocks of CNNs through **depthwise separable convolutions**, and iteratively refining architectures with innovations like inverted residual blocks, linear bottlenecks, and Neural Architecture Search, MobileNet has consistently delivered models that strike an optimal balance between accuracy and computational cost. The introduction of global hyperparameters further empowers developers to fine-tune models for specific hardware and latency requirements.

The impact of MobileNet extends far beyond theoretical advancements; it has practically enabled the widespread adoption of sophisticated computer vision tasks on edge devices, fostering innovation in areas from mobile augmented reality to autonomous systems. As the demand for pervasive AI continues to grow, MobileNet, and the principles it champions, will remain crucial for bridging the gap between cutting-edge research and real-world, resource-constrained deployments, solidifying its legacy as a pivotal development in Generative AI and beyond.

---
<br>

<a name="türkçe-içerik"></a>
## MobileNet: Verimli Evrişimsel Sinir Ağları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Yenilik: Derinlemesine Ayrılabilir Evrişimler](#2-temel-yenilik-derinlemesine-ayrılabilir-evrişimler)
  - [2.1. Standart Evrişim](#21-standart-evrişim)
  - [2.2. Derinlemesine Evrişim](#22-derinlemesine-evrişim)
  - [2.3. Noktasal Evrişim](#23-noktasal-evrişim)
  - [2.4. Derinlemesine Ayrılabilir Evrişimlerin Avantajları](#24-derinlemesine-ayrılabilir-evrişimlerin-avantajları)
- [3. MobileNet Mimarileri ve Hiperparametreleri](#3-mobilenet-mimarileri-ve-hiperparametreleri)
  - [3.1. MobileNetV1](#31-mobilenetv1)
  - [3.2. MobileNetV2](#32-mobilenetv2)
  - [3.3. MobileNetV3](#33-mobilenetv3)
  - [3.4. Verimlilik Dengesi için Hiperparametreler](#34-verimlilik-dengesi-için-hiperparametreler)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Uygulamalar ve Etki](#5-uygulamalar-ve-etki)
- [6. Sonuç](#6-sonuç)

<br>

<a name="1-giriş"></a>
## 1. Giriş

Yapay Zeka'nın, özellikle de bilgisayar görüşü alanının hızla gelişen dünyasında, **Evrişimsel Sinir Ağları (CNN'ler)**, görüntü sınıflandırma, nesne tespiti ve anlamsal segmentasyon gibi görevlerde eşsiz başarılar sergilemiştir. Ancak, VGG veya ResNet gibi geleneksel CNN mimarileri genellikle önemli bir hesaplama maliyeti ve çok sayıda parametre ile birlikte gelir, bu da onları cep telefonları, gömülü sistemler veya kenar cihazlar gibi kaynak kısıtlı cihazlara dağıtmayı zorlaştırır. Bu sınırlama, daha **verimli evrişimsel sinir ağları** geliştirmeye yönelik araştırmaları tetiklemiştir.

**MobileNet**, ilk olarak 2017'de Google tarafından tanıtılan, bu alanda önemli bir atılımı temsil etmektedir. Hesaplama gücü, bellek ve enerji tüketiminin kritik faktörler olduğu mobil ve gömülü görüş uygulamaları için özel olarak tasarlanmış verimli modeller sınıfıdır. MobileNet'in verimliliğinin temel yeniliği, parametre ve hesaplama sayısını önemli ölçüde azaltırken rekabetçi doğruluğu sürdüren **derinlemesine ayrılabilir evrişimlerin** kullanılmasıdır. Bu belge, MobileNet'in mimari detaylarını, temel bileşenlerini, çeşitli iterasyonlarını ve derin öğrenme modellerini gerçek dünya, kısıtlı ortamlarda dağıtma üzerindeki derin etkisini ele almaktadır.

<a name="2-temel-yenilik-derinlemesine-ayrılabilir-evrişimler"></a>
## 2. Temel Yenilik: Derinlemesine Ayrılabilir Evrişimler

MobileNet'in verimliliğinin temel taşı, standart evrişim katmanlarının **derinlemesine ayrılabilir evrişimlerle** stratejik olarak değiştirilmesidir. Bu faktörizasyon, standart bir evrişimi etkili bir şekilde iki ayrı, daha verimli işleme böler: bir **derinlemesine evrişim** ve bir **noktasal evrişim**.

<a name="21-standart-evrişim"></a>
### 2.1. Standart Evrişim

Verimlilik kazanımlarını takdir etmek için, öncelikle bir **standart evrişimin** mekaniğini ve hesaplama maliyetini anlamak önemlidir. Standart bir 2D evrişimde, tek bir filtre tüm giriş kanallarına eşzamanlı olarak uygulanarak tek bir çıkış kanalı üretilir. Eğer bir giriş özellik haritası $D_F \times D_F \times M$ (yükseklik, genişlik, kanal) boyutlarına sahipse ve biz $N$ adet $D_K \times D_K \times M$ boyutunda filtre uygularsak, çıkış özellik haritası $D_G \times D_G \times N$ boyutlarında olacaktır. Standart bir evrişimin hesaplama maliyeti (çarpma ve toplama sayısı) yaklaşık olarak $D_K \cdot D_K \cdot M \cdot N \cdot D_G \cdot D_G$'dir. Bu, maliyetin filtre boyutuyla karesel olarak ve hem giriş hem de çıkış kanallarıyla doğrusal olarak arttığı anlamına gelir.

<a name="22-derinlemesine-evrişim"></a>
### 2.2. Derinlemesine Evrişim

Derinlemesine ayrılabilir bir evrişimin ilk kısmı **derinlemesine evrişimdir**. Tüm giriş kanallarına tek bir filtre uygulamak yerine, derinlemesine evrişim *her giriş kanalına tek bir filtre* uygular. Bu, eğer giriş özellik haritası $M$ kanala sahipse, $M$ farklı filtrenin (her biri $D_K \times D_K \times 1$ boyutunda) kullanıldığı anlamına gelir. Her filtre, kendi karşılık gelen giriş kanalını bağımsız olarak işleyerek $M$ adet filtrelenmiş çıkış kanalı üretir.
Derinlemesine evrişimin hesaplama maliyeti yaklaşık olarak $D_K \cdot D_K \cdot M \cdot D_G \cdot D_G$'dir. Standart evrişimden gelen $N$ çıkış kanalı sayısına olan bağımlılığın burada bulunmadığına dikkat edin. Bu, $N$ genellikle büyük olduğu için hesaplamayı önemli ölçüde azaltır.

<a name="23-noktasal-evrişim"></a>
### 2.3. Noktasal Evrişim

Derinlemesine evrişim adımı her giriş kanalını etkili bir şekilde filtreler, ancak farklı kanallar arasındaki bilgiyi birleştirmez. İşte burada **noktasal evrişim** devreye girer. Noktasal evrişim, standart bir 1x1 evrişimdir (yani, $1 \times 1$ boyutunda bir filtre). Derinlemesine evrişimin çıktısına uygulanır. Özellikle, derinlemesine evrişim $M$ özellik haritası ürettiyse, noktasal evrişim, her biri $1 \times 1 \times M$ boyutunda $N$ filtre kullanır. Bu $N$ filtreden her biri, tüm $M$ derinlemesine çıkış kanalından gelen bilgiyi birleştirerek tek bir çıkış kanalı üretir. Sonuç olarak, bu tür $N$ filtre, $N$ kanallı bir çıkış özellik haritası üretir.
Noktasal evrişimin hesaplama maliyeti yaklaşık olarak $1 \cdot 1 \cdot M \cdot N \cdot D_G \cdot D_G$'dir.

<a name="24-derinlemesine-ayrılabilir-evrişimlerin-avantajları"></a>
### 2.4. Derinlemesine Ayrılabilir Evrişimlerin Avantajları

Bir standart evrişimi derinlemesine evrişime ve noktasal evrişime ayırarak, MobileNet hem hesaplama karmaşıklığında hem de parametre sayısında önemli azalmalar elde eder. Derinlemesine ayrılabilir bir evrişimin toplam maliyeti yaklaşık olarak $(D_K \cdot D_K \cdot M \cdot D_G \cdot D_G) + (M \cdot N \cdot D_G \cdot D_G)$'dir.
Bunu standart evrişim maliyeti ($D_K \cdot D_K \cdot M \cdot N \cdot D_G \cdot D_G$) ile karşılaştırdığımızda, azaltma faktörü yaklaşık olarak $\frac{1}{N} + \frac{1}{D_K^2}$'dir. Tipik filtre boyutları (örn. $3 \times 3$) ve çok sayıda çıkış kanalı için, bu faktör 8 ila 9 katlık bir azalma olabilir. Bu verimlilik kazanımı, kısıtlı cihazlarda gerçek zamanlı uygulamalar için çok önemlidir.

<a name="3-mobilenet-mimarileri-ve-hiperparametreleri"></a>
## 3. MobileNet Mimarileri ve Hiperparametreleri

MobileNet tek bir model değil, verimliliği ve performansı daha da artırmak için derinlemesine ayrılabilir evrişimlerin temel konsepti üzerine inşa edilmiş, çeşitli iterasyonlara sahip bir mimariler ailesidir.

<a name="31-mobilenetv1"></a>
### 3.1. MobileNetV1

2017'de tanıtılan orijinal MobileNet mimarisi, temel bloğu **Batch Normalization** ve bir **ReLU** aktivasyon fonksiyonunu takiben bir derinlemesine ayrılabilir evrişim olarak belirlemiştir. Esas olarak 13 adet bu tür katmandan oluşuyordu. MobileNetV1, bu kompakt modellerin çeşitli görüş görevlerinde daha büyük, daha karmaşık ağlara benzer doğruluk elde edebileceğini, ancak önemli ölçüde daha az parametre ve hesaplama ile çalıştığını göstermiştir.

<a name="32-mobilenetv2"></a>
### 3.2. MobileNetV2

2018'de yayınlanan MobileNetV2, performansı artırmak için iki önemli mimari yenilik tanıttı:
*   **Tersine Çevrilmiş Artık Bloklar (Inverted Residual Blocks):** Standart artık blokların uzamsal boyutları sıkıştırıp sonra genişletmesinin aksine, tersine çevrilmiş artık bloklar önce 1x1 evrişim kullanarak kanal sayısını genişletir, ardından derinlemesine evrişim uygular ve son olarak başka bir 1x1 evrişim kullanarak daha düşük sayıda kanala geri yansıtır. Bu, verimli özellik öğrenimi için bir "darboğaz" yapısı oluşturur.
*   **Doğrusal Darboğazlar (Linear Bottlenecks):** Tersine çevrilmiş artık bloktaki son 1x1 yansıtma katmanından sonra, MobileNetV2 ReLU aktivasyon fonksiyonunu kaldırır. Bu, yüksek boyutlu gömülerin gerekli tüm bilgiyi yakalayabileceği, ancak doğrusal olmayanlar uygulanırsa düşük boyutlu gömülerin (darboğaz) bilgi kaybedebileceği anlayışına dayanır. Darboğazda doğrusal aktivasyonlar kullanmak, bilgi kaybını önlemeye yardımcı olur.

Bu değişiklikler, MobileNetV2'nin V1'den daha yüksek doğruluk elde etmesini sağlarken, hesaplama maliyetlerini korumasını veya hatta azaltmasını sağlamıştır.

<a name="33-mobilenetv3"></a>
### 3.3. MobileNetV3

2019'da tanıtılan MobileNetV3, çeşitli gelişmiş teknikleri birleştirerek verimlilik ve doğruluk sınırlarını daha da zorladı:
*   **Nöral Mimari Arama (NAS):** MobileNetV3, belirli gecikme kısıtlamaları için optimal ağ mimarilerini keşfetmek üzere otomatik arama tekniklerinden, özellikle **MnasNet**'ten yararlandı.
*   **Squeeze-and-Excitation (SE) Blokları:** Bu hafif dikkat mekanizmaları, kanal bazında özellik yanıtlarını dinamik olarak yeniden kalibre ederek özellik temsilini iyileştirir.
*   **Yeni Aktivasyon Fonksiyonları:** MobileNetV3, hesaplama açısından pahalı olan Swish ve Sigmoid'e verimli alternatifler olarak **h-swish** (hard swish) ve **h-sigmoid**'i tanıttı.
*   **Yeniden Tasarlanmış Başlangıç ve Son Katmanlar:** Başlangıçtaki 3x3 evrişim katmanı ve son sınıflandırma katmanları, yüksek boyutlu özellikleri korurken gecikmeyi azaltmak için yeniden tasarlandı.

MobileNetV3, yüksek kaynak gerektiren durumlar için MobileNetV3-Large ve düşük kaynak gerektiren durumlar için MobileNetV3-Small olmak üzere iki ana varyant sunarak esnek bir performans-verimlilik dengesi spektrumu sağlar.

<a name="34-verimlilik-dengesi-için-hiperparametreler"></a>
### 3.4. Verimlilik Dengesi için Hiperparametreler

Mimari seçimlerin ötesinde, MobileNet modelleri, geliştiricilerin gecikme ve doğruluk arasındaki dengeyi ince ayar yapmasına olanak tanıyan iki basit global hiperparametre sunar:
*   **Genişlik Çarpanı ($\alpha$):** Bu parametre (genellikle 0.25 ile 1.0 arasındadır) her katmandaki kanal sayısını tek tip olarak ölçekler. Örneğin, 0.5'lik bir $\alpha$, her katman için giriş ve çıkış kanalı sayısını yarıya indirerek hesaplamaları ve parametreleri önemli ölçüde azaltır, ancak potansiyel olarak doğruluğu da düşürür.
*   **Çözünürlük Çarpanı ($\rho$):** Bu parametre (genellikle 0.75 ile 1.0 arasındadır) giriş görüntüsünün çözünürlüğünü ölçekler. Daha küçük bir giriş çözünürlüğü, tüm katmanların hesaplama maliyetini azaltarak verimlilik ayarlamaları için başka bir düğme sunar.

Bu hiperparametreleri birleştirerek, MobileNet, belirli uygulama gereksinimlerine ve donanım kısıtlamalarına göre uyarlanabilen bir model ailesi sunar.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Aşağıda, TensorFlow/Keras kullanarak derinlemesine ayrılabilir bir evrişim katmanının nasıl uygulanacağını gösteren kısa bir Python kod parçacığı bulunmaktadır. Bu fonksiyon, `DepthwiseConv2D` ve `Conv2D`'yi (noktasal evrişim için) birleştirerek tek bir verimli blok oluşturur.

```python
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU

def derinlemesine_ayrılabilir_evrişim_bloğu(girişler, filtreler, adımlar=(1, 1), çekirdek_boyutu=(3, 3)):
    """
    Derinlemesine ayrılabilir bir evrişim bloğu oluşturur.

    Argümanlar:
        girişler: Bloğa giriş tensörü.
        filtreler: Çıkış filtrelerinin sayısı (noktasal evrişim için).
        adımlar: Derinlemesine evrişim için adım boyutu.
        çekirdek_boyutu: Derinlemesine evrişim için çekirdek boyutu.

    Döndürür:
        Derinlemesine ayrılabilir evrişim bloğundan çıkan çıkış tensörü.
    """
    # Derinlemesine Evrişim
    x = DepthwiseConv2D(kernel_size=çekirdek_boyutu,
                        strides=adımlar,
                        padding='same',
                        use_bias=False)(girişler)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Noktasal Evrişim (1x1 evrişim)
    x = Conv2D(filters=filtreler,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Örnek kullanım:
# Bir hayali giriş tensörü oluşturun
giriş_boyutu = (1, 64, 64, 3) # Parti boyutu, Yükseklik, Genişlik, Kanallar
hayali_giriş = tf.random.normal(giriş_boyutu)

# 64 çıkış filtreli bir derinlemesine ayrılabilir evrişim bloğu oluşturun
çıkış_filtreleri = 64
çıkış = derinlemesine_ayrılabilir_evrişim_bloğu(hayali_giriş, çıkış_filtreleri, adımlar=(2, 2))

print(f"Giriş boyutu: {hayali_giriş.shape}")
print(f"Çıkış boyutu: {çıkış.shape}")

(Kod örneği bölümünün sonu)
```

<a name="5-uygulamalar-ve-etki"></a>
## 5. Uygulamalar ve Etki

MobileNet mimarileri, özellikle kaynak kısıtlı ortamlarda, çok çeşitli uygulamalarda derin öğrenme modellerinin erişilebilirliğini ve dağıtımını derinden etkilemiştir:

*   **Mobil ve Gömülü Cihazlar:** MobileNet modelleri, akıllı telefonlar, akıllı kameralar, dronlar ve diğer IoT cihazlarındaki cihaz içi AI yeteneklerinin omurgasını oluşturur. Bu, bulut tabanlı işlemeye büyük ölçüde güvenmeden gerçek zamanlı nesne tespiti, yüz tanıma ve artırılmış gerçeklik gibi özellikleri etkinleştirerek gizliliği, gecikmeyi ve sağlamlığı artırır.
*   **Gerçek Zamanlı Nesne Tespiti:** MobileNet, **SSD-MobileNet** ve **YOLO-Lite** gibi verimli nesne tespit çerçevelerine entegre edilerek, otonom sürüş, robotik ve gözetim uygulamaları için kenar cihazlarda gerçek zamanlı tespiti mümkün kılmıştır.
*   **Bilgisayar Görüşü Kütüphaneleri:** Etkisi, popüler bilgisayar görüşü kütüphanelerine (örn. OpenCV'nin DNN modülü) ve çerçevelere yayılarak, verimli önceden eğitilmiş modelleri çeşitli görevler için kolayca erişilebilir hale getirmiştir.
*   **Sağlık Hizmetleri:** Teşhis AI modellerinin doğrudan tıbbi görüntüleme ekipmanlarına veya taşınabilir cihazlara dağıtılması.
*   **Robotik:** Sınırlı yerleşik işlem gücüne sahip olsa bile robotların çevrelerini gerçek zamanlı olarak algılamasını ve etkileşime girmesini sağlamak.

MobileNet'in minimum hesaplama yüküyle yüksek performans sunma yeteneği, derin öğrenmeyi demokratikleştirerek gelişmiş AI yeteneklerini daha geniş bir donanım ve uygulama yelpazesine sunmuştur.

<a name="6-sonuç"></a>
## 6. Sonuç

MobileNet, verimli derin öğrenme arayışında anıtsal bir adımı temsil etmektedir. CNN'lerin temel yapı taşlarını **derinlemesine ayrılabilir evrişimler** aracılığıyla titizlikle yeniden tasarlayarak ve tersine çevrilmiş artık bloklar, doğrusal darboğazlar ve Nöral Mimari Arama gibi yeniliklerle mimarileri yinelemeli olarak geliştirerek, MobileNet tutarlı bir şekilde doğruluk ve hesaplama maliyeti arasında optimal bir denge kuran modeller sunmuştur. Global hiperparametrelerin tanıtılması, geliştiricilere modelleri belirli donanım ve gecikme gereksinimleri için ince ayar yapma yeteneği daha da vermektedir.

MobileNet'in etkisi teorik gelişmelerin çok ötesine uzanır; sofistike bilgisayar görüşü görevlerinin kenar cihazlarda yaygın olarak benimsenmesini pratik olarak mümkün kılmış, mobil artırılmış gerçeklikten otonom sistemlere kadar birçok alanda yeniliği teşvik etmiştir. Yaygın AI'ya olan talep artmaya devam ettikçe, MobileNet ve savunduğu ilkeler, çığır açan araştırma ile gerçek dünya, kaynak kısıtlı dağıtımlar arasındaki boşluğu doldurmada kritik olmaya devam edecek, Generatif AI ve ötesinde önemli bir gelişme olarak mirasını pekiştirecektir.

