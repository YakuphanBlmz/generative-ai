# EfficientNet: Rethinking Model Scaling

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Limitations of Traditional Model Scaling](#2-limitations-of-traditional-model-scaling)
- [3. The EfficientNet Approach: Compound Scaling](#3-the-efficientnet-approach-compound-scaling)
  - [3.1. Baseline Network and Neural Architecture Search (NAS)](#31-baseline-network-and-neural-architecture-search-nas)
  - [3.2. The MBConv Block with Squeeze-and-Excitation](#32-the-mbconv-block-with-squeeze-and-excitation)
  - [3.3. The Compound Scaling Principle and Formula](#33-the-compound-scaling-principle-and-formula)
  - [3.4. Advantages and Impact](#34-advantages-and-impact)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The field of deep learning, particularly in computer vision, has witnessed significant advancements driven by increasingly larger and more complex neural network architectures. Historically, improving model performance often involved scaling up existing baseline models by increasing their **depth** (more layers), **width** (more channels per layer), or **resolution** (larger input image size). However, these conventional scaling methods typically scale only one dimension arbitrarily, often leading to sub-optimal performance and inefficient resource utilization. The seminal paper "EfficientNet: Rethinking Model Scaling" by Tan and Le (2019) introduced a novel and highly effective method for uniformly scaling all three dimensions—depth, width, and resolution—using a simple yet powerful **compound scaling** method. This approach systematically balances the scaling process across all dimensions, resulting in a family of models that achieve state-of-the-art accuracy with significantly fewer parameters and FLOPs compared to previous models. EfficientNet represents a pivotal shift in how researchers and practitioners approach the design and scaling of convolutional neural networks, emphasizing efficiency without sacrificing performance.

## 2. Limitations of Traditional Model Scaling
Prior to EfficientNet, a common strategy for enhancing a convolutional neural network's performance was to increase its capacity by making it "bigger." This typically involved one of three independent scaling dimensions:

*   **Depth Scaling:** Adding more layers to the network. Deeper networks can capture richer and more complex features. However, excessively deep networks can suffer from vanishing/exploding gradients and increased training difficulty, often leading to diminishing accuracy gains after a certain point. Residual connections (e.g., ResNet) and skip connections helped mitigate these issues, but optimal depth still requires careful consideration.
*   **Width Scaling:** Increasing the number of channels (filters) in each layer. Wider networks can capture more fine-grained features and are generally easier to train than deeper networks. However, arbitrary width scaling can lead to redundancy in filters and rapidly increase computational cost and memory footprint without proportional accuracy improvements, especially for highly redundant features.
*   **Resolution Scaling:** Feeding higher-resolution input images to the network. Higher resolution allows the network to perceive more detailed patterns. While beneficial, this comes at a substantial computational cost, as the operations (especially convolutions) scale quadratically with resolution. Moreover, a network designed for low resolution might not be optimally configured to leverage extremely high-resolution inputs effectively.

The primary limitation of these traditional methods is their **one-dimensional approach**. Scaling depth, width, or resolution in isolation often leads to an imbalance in the network's capacity and computational needs across different dimensions. For instance, a very deep network with low input resolution might struggle to extract meaningful features from sparse information, while a very wide network processing low-resolution images might have redundant feature detectors that are not fully utilized. The authors of EfficientNet observed that the optimal balance between these dimensions changes as the model scales, suggesting a need for a more coordinated scaling strategy.

## 3. The EfficientNet Approach: Compound Scaling
EfficientNet's core innovation lies in its **compound scaling** method, which systematically scales network **depth**, **width**, and **resolution** in a balanced manner. Instead of arbitrary scaling, EfficientNet uses a fixed set of scaling coefficients to uniformly scale all three dimensions based on a compound coefficient `phi`.

### 3.1. Baseline Network and Neural Architecture Search (NAS)
The foundation of the EfficientNet family is a baseline network, **EfficientNet-B0**, derived through **Neural Architecture Search (NAS)**. Specifically, the authors employed the NAS method called AutoML MnasNet, which optimizes for both accuracy and FLOPS. This search process identified an optimal mobile-sized baseline network.
EfficientNet-B0 is characterized by its use of **MBConv (Mobile Inverted Bottleneck Convolution)** blocks, which are the building blocks of MobileNetV2 and MnasNet. These blocks are known for their efficiency and effectiveness, employing depthwise separable convolutions to reduce computational cost.

### 3.2. The MBConv Block with Squeeze-and-Excitation
A key component of EfficientNet's efficiency stems from its **MBConv blocks**. These blocks, originating from MobileNetV2, feature:
*   **Depthwise Separable Convolutions:** Decomposing a standard convolution into a depthwise convolution (applying a single filter per input channel) and a pointwise convolution (a 1x1 convolution mixing channels). This significantly reduces computation.
*   **Inverted Residual Structure:** Unlike traditional residual blocks (e.g., ResNet) that use wide-narrow-wide structures, MBConv blocks use a narrow-wide-narrow structure. They first expand channels, apply depthwise convolution, and then project back to fewer channels.
*   **Squeeze-and-Excitation (SE) Networks:** Each MBConv block incorporates a **Squeeze-and-Excitation** module. This mechanism adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels. It first "squeezes" global spatial information into a channel descriptor and then "excites" each channel by learning channel-wise weights. This allows the network to focus on more informative features.

### 3.3. The Compound Scaling Principle and Formula
The authors observed that scaling dimensions are not independent; larger resolutions benefit from deeper and wider networks. To address this, they proposed a **compound scaling method** that uses a compound coefficient `phi` to uniformly scale all three dimensions:

$$
\text{depth: } d = \alpha^{\phi} \\
\text{width: } w = \beta^{\phi} \\
\text{resolution: } r = \gamma^{\phi} \\
\text{subject to: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \\
\alpha \ge 1, \beta \ge 1, \gamma \ge 1
$$

Here:
*   $\phi$ (phi) is a user-specified compound coefficient that controls the overall scaling of the model. Larger $\phi$ values result in larger models (e.g., B1, B2, ..., B7).
*   $\alpha$, $\beta$, $\gamma$ are coefficients that determine how much to scale the network's depth, width, and resolution, respectively, for a given $\phi$. These coefficients are found through a small grid search on the baseline model (EfficientNet-B0) under a fixed resource constraint (e.g., $2^2$).
    *   For EfficientNet-B0, the authors found $\alpha \approx 1.2$, $\beta \approx 1.1$, $\gamma \approx 1.15$.
*   The constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ ensures that for every increase in $\phi$ by 1, the total computational cost (FLOPS) approximately doubles. The exponents 2 for $\beta$ and $\gamma$ reflect that network width and resolution typically increase FLOPS quadratically.

This principled scaling approach allows for a systematic and efficient way to scale up the baseline EfficientNet-B0 to a family of models (EfficientNet-B1 to B7), each offering a different trade-off between accuracy and computational cost.

### 3.4. Advantages and Impact
The compound scaling strategy of EfficientNet offers several significant advantages:
*   **Superior Efficiency:** EfficientNet models achieve state-of-the-art accuracy on ImageNet with significantly fewer parameters and FLOPs compared to previous architectures. For example, EfficientNet-B7 achieves higher accuracy than previous large models like GPipe-trained NASNet-A, while being orders of magnitude smaller.
*   **Balanced Scaling:** By simultaneously and uniformly scaling all dimensions, EfficientNet avoids the diminishing returns often observed when scaling dimensions independently, leading to a more optimal use of computational resources.
*   **Generalizability:** The scaling coefficients derived on a smaller baseline (EfficientNet-B0) transfer effectively to larger models and different tasks, demonstrating the robustness of the compound scaling principle.
*   **Reduced Overfitting:** The efficiency of EfficientNet models means they can achieve high accuracy with fewer parameters, potentially reducing the risk of overfitting compared to heavily over-parameterized models.

EfficientNet has profoundly influenced the design of subsequent convolutional architectures, becoming a benchmark for efficiency and accuracy. Its principles have been applied and extended in various domains, underscoring the importance of systematic model scaling.

## 4. Code Example
Here is a conceptual Python code snippet illustrating a simplified version of an MBConv block with Squeeze-and-Excitation (SE). This example focuses on the structure rather than a full, runnable implementation.

```python
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        # Squeeze: Global Average Pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # Excitation: Two fully connected layers
        # First layer reduces channels, second expands back
        hidden_channels = max(1, int(in_channels * se_ratio))
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.SiLU(), # Swish activation, used in EfficientNet
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.excitation(self.squeeze(x))
        return x * se_weight # Channel-wise multiplication

class MBConvBlock(nn.Module):
    """
    Simplified Mobile Inverted Bottleneck Convolution block
    with Squeeze-and-Excitation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        hidden_channels = in_channels * expand_ratio
        
        # Pointwise Expansion (if expand_ratio > 1)
        self.expand_conv = nn.Identity()
        if expand_ratio > 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            )

        # Depthwise Convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, 
                      padding=(kernel_size-1)//2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(hidden_channels, se_ratio)

        # Pointwise Projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection (if input and output channels match and stride is 1)
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        
        if self.use_res_connect:
            x += identity # Add residual connection
        return x

# Example usage (conceptual):
# Define an input tensor
input_tensor = torch.randn(1, 32, 28, 28) # Batch, Channels, Height, Width

# Create an MBConv block
# in_channels=32, out_channels=16, kernel_size=3, stride=1, expand_ratio=6
mbconv_block = MBConvBlock(32, 16, 3, 1, 6) 

# Pass input through the block
output_tensor = mbconv_block(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")

(End of code example section)
```

## 5. Conclusion
EfficientNet represents a fundamental shift in the paradigm of neural network scaling, moving from arbitrary, single-dimensional adjustments to a principled, multi-dimensional **compound scaling** approach. By systematically balancing the network's **depth**, **width**, and **resolution** using a compound coefficient derived from **Neural Architecture Search (NAS)** on a baseline model (EfficientNet-B0), the authors successfully created a family of models that achieve unparalleled efficiency. The integration of **MBConv blocks** with **Squeeze-and-Excitation** modules further enhances their capacity to learn rich, context-aware features with minimal computational overhead. The profound impact of EfficientNet is evident in its ability to deliver state-of-the-art accuracy with significantly fewer parameters and FLOPs, thereby reducing computational resource demands and enabling faster training and inference. This work has not only established new benchmarks for model performance but has also inspired a wave of research into efficient network design, underscoring the critical importance of a holistic and systematic approach to scaling deep learning models. EfficientNet continues to be a foundational architecture in various applications, validating its enduring relevance in the rapidly evolving landscape of generative AI and computer vision.

---
<br>

<a name="türkçe-içerik"></a>
## EfficientNet: Model Ölçeklendirmesini Yeniden Düşünmek

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Geleneksel Model Ölçeklendirme Sınırlamaları](#2-geleneksel-model-ölçeklendirme-sınırlamaları)
- [3. EfficientNet Yaklaşımı: Bileşik Ölçeklendirme](#3-efficientnet-yaklaşımı-bileşik-ölçeklendirme)
  - [3.1. Temel Ağ ve Nöral Mimari Arama (NAS)](#31-temel-ağ-ve-nöral-mimari-arama-nas)
  - [3.2. Sıkma ve Uyarma (Squeeze-and-Excitation) Özellikli MBConv Bloğu](#32-sıkma-ve-uyarma-squeeze-and-excitation-özellikli-mbconv-bloğu)
  - [3.3. Bileşik Ölçeklendirme Prensibi ve Formülü](#33-bileşik-ölçeklendirme-prensibi-ve-formülü)
  - [3.4. Avantajları ve Etkisi](#34-avantajları-ve-etkisi)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
Derin öğrenme alanında, özellikle bilgisayar görüşünde, giderek büyüyen ve daha karmaşık sinir ağı mimarileri sayesinde önemli ilerlemeler kaydedilmiştir. Tarihsel olarak, model performansını artırmak genellikle mevcut temel modelleri **derinlik** (daha fazla katman), **genişlik** (katman başına daha fazla kanal) veya **çözünürlük** (daha büyük giriş görüntü boyutu) artırarak ölçeklendirmeyi içeriyordu. Ancak, bu geleneksel ölçeklendirme yöntemleri genellikle tek bir boyutu keyfi olarak ölçeklendirir, bu da çoğu zaman alt optimal performans ve verimsiz kaynak kullanımına yol açar. Tan ve Le (2019) tarafından yazılan "EfficientNet: Rethinking Model Scaling" adlı çığır açan makale, üç boyutun da (derinlik, genişlik ve çözünürlük) dengeli bir şekilde ölçeklendirilmesi için basit ama güçlü bir **bileşik ölçeklendirme** yöntemi sunmuştur. Bu yaklaşım, tüm boyutlardaki ölçeklendirme sürecini sistematik olarak dengeleyerek, önceki modellere kıyasla önemli ölçüde daha az parametre ve FLOP ile son teknoloji doğruluk elde eden bir model ailesi ortaya çıkarmıştır. EfficientNet, evrişimsel sinir ağlarının tasarımı ve ölçeklendirilmesine yönelik yaklaşımlarda önemli bir değişimi temsil etmekte olup, performanstan ödün vermeden verimliliği vurgulamaktadır.

## 2. Geleneksel Model Ölçeklendirme Sınırlamaları
EfficientNet'ten önce, bir evrişimsel sinir ağının performansını artırmak için yaygın bir strateji, kapasitesini "büyüterek" artırmaktı. Bu genellikle üç bağımsız ölçeklendirme boyutundan birini içeriyordu:

*   **Derinlik Ölçeklendirme:** Ağa daha fazla katman eklemek. Daha derin ağlar, daha zengin ve karmaşık özellikleri yakalayabilir. Ancak, aşırı derin ağlar, kaybolan/patlayan gradyanlar ve artan eğitim zorluğu yaşayabilir, bu da genellikle belirli bir noktadan sonra doğruluk kazanımlarında düşüşe yol açar. Artık bağlantılar (örneğin, ResNet) ve atlama bağlantıları bu sorunları hafifletmeye yardımcı oldu, ancak optimal derinlik hala dikkatli bir değerlendirme gerektirir.
*   **Genişlik Ölçeklendirme:** Her katmandaki kanal (filtre) sayısını artırmak. Daha geniş ağlar, daha ince ayrıntılı özellikleri yakalayabilir ve genellikle daha derin ağlardan daha kolay eğitilir. Ancak, keyfi genişlik ölçeklendirmesi, filtrelerde gereksizliğe yol açabilir ve özellikle yüksek derecede gereksiz özellikler için orantılı doğruluk iyileştirmeleri olmaksızın hesaplama maliyetini ve bellek ayak izini hızla artırabilir.
*   **Çözünürlük Ölçeklendirme:** Ağa daha yüksek çözünürlüklü giriş görüntüleri beslemek. Daha yüksek çözünürlük, ağın daha ayrıntılı desenleri algılamasına olanak tanır. Bu faydalı olsa da, özellikle evrişimler gibi işlemlerin çözünürlükle karesel olarak ölçeklenmesi nedeniyle önemli bir hesaplama maliyetiyle gelir. Dahası, düşük çözünürlük için tasarlanmış bir ağ, aşırı yüksek çözünürlüklü girişleri etkili bir şekilde kullanmak için optimal olarak yapılandırılmamış olabilir.

Bu geleneksel yöntemlerin temel sınırlaması, **tek boyutlu yaklaşımlarıdır**. Derinlik, genişlik veya çözünürlüğü izole bir şekilde ölçeklendirmek, genellikle ağın kapasitesinde ve farklı boyutlardaki hesaplama ihtiyaçlarında bir dengesizliğe yol açar. Örneğin, düşük giriş çözünürlüğüne sahip çok derin bir ağ, seyrek bilgilerden anlamlı özellikler çıkarmakta zorlanabilirken, düşük çözünürlüklü görüntüleri işleyen çok geniş bir ağ, tam olarak kullanılmayan gereksiz özellik dedektörlerine sahip olabilir. EfficientNet'in yazarları, bu boyutlar arasındaki optimal dengenin model ölçeklendikçe değiştiğini gözlemleyerek, daha koordineli bir ölçeklendirme stratejisine ihtiyaç duyulduğunu öne sürdüler.

## 3. EfficientNet Yaklaşımı: Bileşik Ölçeklendirme
EfficientNet'in temel yeniliği, ağ **derinliğini**, **genişliğini** ve **çözünürlüğünü** dengeli bir şekilde sistematik olarak ölçeklendiren **bileşik ölçeklendirme** yönteminde yatmaktadır. EfficientNet, keyfi ölçeklendirme yerine, bir bileşik katsayı `phi`'ye dayalı olarak her üç boyutu da tek tip olarak ölçeklendirmek için sabit bir dizi ölçeklendirme katsayısı kullanır.

### 3.1. Temel Ağ ve Nöral Mimari Arama (NAS)
EfficientNet ailesinin temeli, **Nöral Mimari Arama (NAS)** yoluyla türetilen bir temel ağ olan **EfficientNet-B0**'dır. Özellikle, yazarlar hem doğruluk hem de FLOP'lar için optimize eden AutoML MnasNet adlı NAS yöntemini kullanmışlardır. Bu arama süreci, mobil boyutlu optimal bir temel ağı belirlemiştir.
EfficientNet-B0, MobileNetV2 ve MnasNet'in yapı taşları olan **MBConv (Mobile Inverted Bottleneck Convolution)** bloklarının kullanımıyla karakterize edilir. Bu bloklar, hesaplama maliyetini azaltmak için derinlemesine ayrılabilir evrişimler kullanmalarıyla verimlilikleri ve etkinlikleriyle bilinirler.

### 3.2. Sıkma ve Uyarma (Squeeze-and-Excitation) Özellikli MBConv Bloğu
EfficientNet'in verimliliğinin temel bir bileşeni, **MBConv bloklarından** kaynaklanmaktadır. MobileNetV2'den gelen bu bloklar şunları içerir:
*   **Derinlemesine Ayrılabilir Evrişimler:** Standart bir evrişimi derinlemesine evrişime (giriş kanal başına tek bir filtre uygulama) ve nokta bazlı evrişime (kanalları karıştıran bir 1x1 evrişim) ayırma. Bu, hesaplamayı önemli ölçüde azaltır.
*   **Ters Artık Yapı:** Geleneksel artık bloklardan (örneğin, ResNet) farklı olarak, geniş-dar-geniş yapılar yerine, MBConv blokları dar-geniş-dar bir yapı kullanır. İlk olarak kanalları genişletir, derinlemesine evrişim uygular ve ardından daha az kanala geri yansıtır.
*   **Sıkma ve Uyarma (SE) Ağları:** Her MBConv bloğu, bir **Sıkma ve Uyarma** modülü içerir. Bu mekanizma, kanallar arası bağımlılıkları açıkça modelleyerek kanal bazlı özellik yanıtlarını adaptif olarak yeniden kalibre eder. İlk olarak küresel uzamsal bilgiyi bir kanal tanımlayıcısına "sıkıştırır" ve ardından kanal bazlı ağırlıkları öğrenerek her kanalı "uyarır". Bu, ağın daha bilgilendirici özelliklere odaklanmasını sağlar.

### 3.3. Bileşik Ölçeklendirme Prensibi ve Formülü
Yazarlar, ölçeklendirme boyutlarının bağımsız olmadığını; daha büyük çözünürlüklerin daha derin ve daha geniş ağlardan fayda sağladığını gözlemlemişlerdir. Bunu ele almak için, üç boyutu da (derinlik, genişlik ve çözünürlük) bir bileşik katsayı $\phi$ kullanarak tek tip olarak ölçeklendiren bir **bileşik ölçeklendirme yöntemi** önermişlerdir:

$$
\text{derinlik: } d = \alpha^{\phi} \\
\text{genişlik: } w = \beta^{\phi} \\
\text{çözünürlük: } r = \gamma^{\phi} \\
\text{kısıt: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \\
\alpha \ge 1, \beta \ge 1, \gamma \ge 1
$$

Burada:
*   $\phi$ (phi), modelin genel ölçeklendirmesini kontrol eden kullanıcı tanımlı bir bileşik katsayıdır. Daha büyük $\phi$ değerleri daha büyük modellere yol açar (örneğin, B1, B2, ..., B7).
*   $\alpha$, $\beta$, $\gamma$, verilen bir $\phi$ için ağın derinliğini, genişliğini ve çözünürlüğünü ne kadar ölçekleyeceğini belirleyen katsayılardır. Bu katsayılar, sabit bir kaynak kısıtlaması altında (örneğin, $2^2$) temel model (EfficientNet-B0) üzerinde küçük bir ızgara aramasıyla bulunur.
    *   EfficientNet-B0 için yazarlar $\alpha \approx 1.2$, $\beta \approx 1.1$, $\gamma \approx 1.15$ bulmuşlardır.
*   $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ kısıtı, $\phi$'de her 1 birimlik artış için toplam hesaplama maliyetinin (FLOP'lar) yaklaşık olarak iki katına çıkmasını sağlar. $\beta$ ve $\gamma$ için 2 üsleri, ağ genişliği ve çözünürlüğünün genellikle FLOP'ları karesel olarak artırdığını yansıtır.

Bu ilkeli ölçeklendirme yaklaşımı, temel EfficientNet-B0'ı bir model ailesine (EfficientNet-B1'den B7'ye) sistematik ve verimli bir şekilde ölçeklendirmeye olanak tanır ve her biri doğruluk ile hesaplama maliyeti arasında farklı bir denge sunar.

### 3.4. Avantajları ve Etkisi
EfficientNet'in bileşik ölçeklendirme stratejisi, birkaç önemli avantaj sunar:
*   **Üstün Verimlilik:** EfficientNet modelleri, önceki mimarilere kıyasla önemli ölçüde daha az parametre ve FLOP ile ImageNet'te son teknoloji doğruluk elde eder. Örneğin, EfficientNet-B7, GPipe ile eğitilmiş NASNet-A gibi önceki büyük modellerden daha yüksek doğruluk elde ederken, büyüklük açısından kat kat daha küçüktür.
*   **Dengeli Ölçeklendirme:** Tüm boyutları eşzamanlı ve tek tip olarak ölçeklendirerek, EfficientNet, boyutları bağımsız olarak ölçeklendirirken sıklıkla gözlemlenen azalan getirileri önler, bu da hesaplama kaynaklarının daha optimal kullanımına yol açar.
*   **Genellenebilirlik:** Daha küçük bir temelden (EfficientNet-B0) türetilen ölçeklendirme katsayıları, daha büyük modellere ve farklı görevlere etkili bir şekilde aktarılır, bu da bileşik ölçeklendirme prensibinin sağlamlığını gösterir.
*   **Azaltılmış Aşırı Uyum:** EfficientNet modellerinin verimliliği, yüksek doğruluk elde etmek için daha az parametre kullanmaları anlamına gelir, bu da aşırı parametrelendirilmiş modellere kıyasla aşırı uyum riskini potansiyel olarak azaltır.

EfficientNet, sonraki evrişim mimarilerinin tasarımını derinden etkilemiş, verimlilik ve doğruluk için bir ölçüt haline gelmiştir. İlkeleri çeşitli alanlarda uygulanmış ve genişletilmiştir, bu da sistematik model ölçeklendirmenin önemini vurgulamaktadır.

## 4. Kod Örneği
Burada, Sıkma ve Uyarma (SE) içeren basitleştirilmiş bir MBConv bloğunun kavramsal bir Python kod örneği bulunmaktadır. Bu örnek, tam ve çalıştırılabilir bir uygulamadan ziyade yapıya odaklanmaktadır.

```python
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """
    Sıkma ve Uyarma (Squeeze-and-Excitation) bloğu.
    """
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        # Sıkma: Global Ortalama Havuzlama (Global Average Pooling)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # Uyarma: İki tam bağlantılı katman
        # İlk katman kanal sayısını azaltır, ikincisi geri genişletir
        hidden_channels = max(1, int(in_channels * se_ratio))
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.SiLU(), # EfficientNet'te kullanılan Swish aktivasyonu
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.excitation(self.squeeze(x))
        return x * se_weight # Kanal bazında çarpma

class MBConvBlock(nn.Module):
    """
    Sıkma ve Uyarma özellikli Basitleştirilmiş Mobil Tersine Evrişim (MBConv) bloğu.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        hidden_channels = in_channels * expand_ratio
        
        # Nokta Bazlı Genişletme (expand_ratio > 1 ise)
        self.expand_conv = nn.Identity()
        if expand_ratio > 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            )

        # Derinlemesine Evrişim
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, 
                      padding=(kernel_size-1)//2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )

        # Sıkma ve Uyarma
        self.se = SqueezeExcitation(hidden_channels, se_ratio)

        # Nokta Bazlı Projeksiyon
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Artık bağlantı (giriş ve çıkış kanalları eşleşiyor ve adım 1 ise)
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        
        if self.use_res_connect:
            x += identity # Artık bağlantıyı ekle
        return x

# Örnek kullanım (kavramsal):
# Bir giriş tensörü tanımlayın
input_tensor = torch.randn(1, 32, 28, 28) # Parti, Kanallar, Yükseklik, Genişlik

# Bir MBConv bloğu oluşturun
# giriş_kanalları=32, çıkış_kanalları=16, çekirdek_boyutu=3, adım=1, genişletme_oranı=6
mbconv_block = MBConvBlock(32, 16, 3, 1, 6) 

# Girişi bloktan geçirin
output_tensor = mbconv_block(input_tensor)

print(f"Giriş şekli: {input_tensor.shape}")
print(f"Çıkış şekli: {output_tensor.shape}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
EfficientNet, sinir ağı ölçeklendirme paradigmasında temel bir değişimi temsil etmektedir; keyfi, tek boyutlu ayarlamalardan, ilkeli, çok boyutlu **bileşik ölçeklendirme** yaklaşımına geçiş yapmıştır. Bir temel model (EfficientNet-B0) üzerinde **Nöral Mimari Arama (NAS)**'dan türetilen bir bileşik katsayı kullanarak ağın **derinliğini**, **genişliğini** ve **çözünürlüğünü** sistematik olarak dengeleyerek, yazarlar eşsiz verimlilik sağlayan bir model ailesi yaratmayı başarmışlardır. **Sıkma ve Uyarma** modülleri ile **MBConv bloklarının** entegrasyonu, zengin, bağlama duyarlı özellikleri minimum hesaplama yüküyle öğrenme kapasitelerini daha da artırmaktadır. EfficientNet'in derin etkisi, önemli ölçüde daha az parametre ve FLOP ile son teknoloji doğruluk sunma yeteneğinde açıkça görülmektedir, bu da hesaplama kaynak taleplerini azaltmakta ve daha hızlı eğitim ve çıkarım sağlamaktadır. Bu çalışma, yalnızca model performansı için yeni kıyaslamalar oluşturmakla kalmamış, aynı zamanda verimli ağ tasarımı üzerine bir araştırma dalgasına da ilham vererek, derin öğrenme modellerini ölçeklendirmeye yönelik bütünsel ve sistematik bir yaklaşımın kritik önemini vurgulamıştır. EfficientNet, üretken yapay zeka ve bilgisayar görüşünün hızla gelişen ortamında kalıcı geçerliliğini doğrulayarak çeşitli uygulamalarda temel bir mimari olmaya devam etmektedir.




