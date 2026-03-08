# EfficientNet: Rethinking Model Scaling

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background and Motivation](#2-background-and-motivation)
- [3. EfficientNet Architecture and Compound Scaling](#3-efficientnet-architecture-and-compound-scaling)
  - [3.1 The Base Network (EfficientNet-B0)](#31-the-base-network-efficientnet-b0)
  - [3.2 Compound Scaling Principle](#32-compound-scaling-principle)
  - [3.3 Scaling Coefficients](#33-scaling-coefficients)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction <a name="1-introduction"></a>
The relentless pursuit of higher accuracy in deep learning models, particularly in computer vision, has traditionally been met with an increasing demand for computational resources. Convolutional Neural Networks (CNNs) have shown remarkable success in various tasks, but their development often involves a painstaking process of designing novel architectures or scaling existing ones. While architecture search has garnered significant attention, **model scaling**, the process of expanding a baseline network to achieve better performance, has historically been more ad-hoc. Researchers typically scale network **depth** (number of layers), **width** (number of channels), or **resolution** (input image size) individually, often leading to suboptimal efficiency and accuracy.

**EfficientNet: Rethinking Model Scaling**, a seminal paper by Tan and Le from Google Brain in 2019, revolutionized this paradigm by proposing a principled and systematic approach to model scaling. The core idea behind EfficientNet is to uniformly scale all three dimensions—depth, width, and resolution—using a **compound coefficient**. This method ensures that the balance between these dimensions is maintained, leading to significantly improved accuracy with fewer parameters and FLOPs compared to arbitrarily scaled models. EfficientNet models achieved state-of-the-art accuracy on ImageNet while being up to 10x more efficient than previous models, setting a new standard for efficient deep learning.

## 2. Background and Motivation <a name="2-background-and-motivation"></a>
Prior to EfficientNet, scaling neural networks often involved heuristics or manual tuning. Common practices included:
*   **Depth Scaling:** Adding more layers to a network (e.g., from ResNet-18 to ResNet-50). Deeper networks can capture richer and more complex features but are prone to vanishing gradients and increased training time.
*   **Width Scaling:** Increasing the number of channels in each layer (e.g., Wide ResNet). Wider networks tend to capture finer-grained features and are easier to train but can also lead to redundancy and higher memory consumption.
*   **Resolution Scaling:** Feeding higher-resolution images to the network (e.g., from 224x224 to 331x331). Higher resolutions can help detect smaller objects and finer details, but drastically increase computational cost for early layers.

The challenge with these individual scaling strategies is that they often ignore the interdependencies between network dimensions. For instance, if a network's depth is increased significantly without proportionally increasing its width or resolution, the model might struggle to process the enriched features effectively, or higher-resolution inputs might not fully benefit from a shallow network. The authors observed that scaling any single dimension improves accuracy, but the gains diminish for larger models, and larger models often require fine-tuning or specialized training regimes. This observation motivated the hypothesis that there exists an optimal balance between these scaling dimensions, and that scaling them together, rather than separately, could yield superior results. The goal was to find a **principled scaling strategy** that maximizes model efficiency and accuracy simultaneously.

## 3. EfficientNet Architecture and Compound Scaling <a name="3-efficientnet-architecture-and-compound-scaling"></a>
EfficientNet's success stems from two key components: a strong **baseline network** called EfficientNet-B0, and a novel **compound scaling method** that uniformly scales depth, width, and resolution.

### 3.1 The Base Network (EfficientNet-B0) <a name="31-the-base-network-efficientnet-b0"></a>
The starting point for EfficientNet is a highly optimized base model, **EfficientNet-B0**. This base model was designed using a **Neural Architecture Search (NAS)** technique, specifically the AutoML MNAS framework, which aims to optimize both accuracy and FLOPs. EfficientNet-B0 primarily utilizes **MBConv (Mobile Inverted Bottleneck Convolution)** blocks, similar to those found in MobileNetV2 and MobileNetV3. These blocks are characterized by:
*   **Depthwise Separable Convolutions:** Reducing computational cost by separating spatial and channel-wise convolutions.
*   **Inverted Residuals:** Using a bottleneck structure where the input and output are narrow, but the intermediate expansion layer is wide.
*   **Squeeze-and-Excitation (SE) Networks:** Dynamically recalibrating channel-wise feature responses, further improving representational power without significant computational overhead.

This highly efficient base architecture provides an excellent foundation for scaling, ensuring that subsequent scaled models inherit its inherent efficiency.

### 3.2 Compound Scaling Principle <a name="32-compound-scaling-principle"></a>
The most significant contribution of EfficientNet is its **compound scaling method**. Instead of arbitrarily scaling depth ($d$), width ($w$), or resolution ($r$) independently, EfficientNet proposes to scale them simultaneously using a compound coefficient $\phi$. This can be expressed by the following set of equations:

Depth: $d = \alpha^\phi$
Width: $w = \beta^\phi$
Resolution: $r = \gamma^\phi$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ and $\alpha \ge 1, \beta \ge 1, \gamma \ge 1$. Here, $\alpha, \beta, \gamma$ are constants determined by a small grid search on the base model, and $\phi$ is a user-specified coefficient that controls the overall resource scaling. The constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ ensures that for every increase in $\phi$, the total FLOPs roughly double, which is a common practice for scaling up models.

The intuition behind compound scaling is that a larger input image resolution requires a deeper network to capture more pixels and a wider network to capture more fine-grained patterns. Conversely, a deeper and wider network benefits from higher resolution inputs to process more features. By scaling all three dimensions together, EfficientNet achieves a better balance and synergy, leading to more efficient performance gains.

### 3.3 Scaling Coefficients <a name="33-scaling-coefficients"></a>
The optimal values for $\alpha, \beta, \gamma$ are found through a two-step process:
1.  **Step 1:** Fix $\phi = 1$ and perform a small grid search for $\alpha, \beta, \gamma$ on the small base model (EfficientNet-B0) under the constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$. For EfficientNet-B0, the authors found $\alpha \approx 1.2$, $\beta \approx 1.1$, and $\gamma \approx 1.15$ to be good initial values.
2.  **Step 2:** Fix $\alpha, \beta, \gamma$ as constants and scale the baseline network with different $\phi$ values to obtain EfficientNet-B1 to B7. For example, for EfficientNet-B1, $\phi$ would be 1, for B2, $\phi$ would be 2, and so on, with actual $\phi$ values calculated to reach specific target resources.

This systematic approach allows for the creation of a family of models, EfficientNet-B0 to EfficientNet-B7, which progressively increase in size and accuracy while maintaining optimal efficiency. Each model (e.g., EfficientNet-B3) corresponds to a different value of $\phi$, inheriting the same optimal scaling ratios $\alpha, \beta, \gamma$ derived from the base model.

## 4. Code Example <a name="4-code-example"></a>
This example demonstrates how to load a pre-trained EfficientNet model using the `timm` (PyTorch Image Models) library, which is commonly used for vision models in PyTorch. It shows the ease of accessing different scaled versions of the EfficientNet family.

```python
import timm
import torch

# Define a list of EfficientNet models to demonstrate
efficientnet_models = [
    'efficientnet_b0',
    'efficientnet_b3',
    'efficientnet_b7'
]

print("--- Demonstrating EfficientNet Models ---")

for model_name in efficientnet_models:
    print(f"\nLoading model: {model_name}")
    # Load a pre-trained EfficientNet model
    # pretrained=True downloads weights trained on ImageNet
    try:
        model = timm.create_model(model_name, pretrained=True)
        model.eval() # Set model to evaluation mode

        # Print some basic information about the loaded model
        print(f"  Model default input size: {model.default_cfg['input_size']}")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

        # Create a dummy input tensor matching the expected input size
        # Expected input_size is usually (C, H, W)
        _, H, W = model.default_cfg['input_size']
        dummy_input = torch.randn(1, 3, H, W) # Batch size 1, 3 channels, HxW resolution

        # Perform a forward pass
        with torch.no_grad(): # Disable gradient calculation for inference
            output = model(dummy_input)
        print(f"  Output shape for dummy input: {output.shape}")

    except Exception as e:
        print(f"  Error loading {model_name}: {e}")

print("\n--- End of demonstration ---")


(End of code example section)
```

## 5. Conclusion <a name="5-conclusion"></a>
EfficientNet marked a significant advancement in the design and scaling of convolutional neural networks. By introducing a principled **compound scaling method** that uniformly scales network depth, width, and resolution, it addressed the limitations of arbitrary scaling strategies. The family of EfficientNet models, ranging from B0 to B7, demonstrated superior accuracy-to-FLOPs ratios, achieving state-of-the-art performance on ImageNet with significantly fewer parameters and computational costs.

The key takeaways from EfficientNet are:
*   The importance of a strong, efficient **baseline network** (EfficientNet-B0) as the foundation for scaling.
*   The discovery that scaling network dimensions (depth, width, resolution) in a **balanced and compound manner** is crucial for optimal efficiency and accuracy.
*   The power of **Neural Architecture Search (NAS)** in finding highly efficient building blocks and base architectures.

EfficientNet has profoundly influenced subsequent research in efficient model design, inspiring numerous follow-up works focusing on even more efficient architectures, better scaling techniques, and applications across various domains, from object detection to semantic segmentation. Its methodologies continue to be a cornerstone for developing high-performance, resource-efficient deep learning models, making advanced AI more accessible and sustainable.

---
<br>

<a name="türkçe-içerik"></a>
## EfficientNet: Model Ölçeklendirmeyi Yeniden Düşünmek

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan ve Motivasyon](#2-arka-plan-ve-motivasyon)
- [3. EfficientNet Mimarisi ve Bileşik Ölçeklendirme](#3-efficientnet-mimarisi-ve-bileşik-ölçeklendirme)
  - [3.1 Temel Ağ (EfficientNet-B0)](#31-temel-ağ-efficientnet-b0)
  - [3.2 Bileşik Ölçeklendirme Prensibi](#32-bileşik-ölçeklendirme-prensibi)
  - [3.3 Ölçeklendirme Katsayıları](#33-ölçeklendirme-katsayıları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş <a name="1-giriş"></a>
Derin öğrenme modellerinde, özellikle bilgisayar görüşünde, daha yüksek doğruluk arayışı, geleneksel olarak artan hesaplama kaynakları talebiyle karşılanmıştır. Evrişimsel Sinir Ağları (CNN'ler) çeşitli görevlerde kayda değer başarı göstermiş olsa da, geliştirilmeleri genellikle yeni mimarilerin tasarlanması veya mevcut olanların ölçeklendirilmesi gibi zahmetli bir süreç içerir. Mimari arama önemli ilgi görmüş olsa da, bir taban ağını daha iyi performans elde etmek için genişletme süreci olan **model ölçeklendirme**, tarihsel olarak daha geçici bir yaklaşımla yapılmıştır. Araştırmacılar genellikle ağ **derinliğini** (katman sayısı), **genişliğini** (kanal sayısı) veya **çözünürlüğünü** (giriş görüntü boyutu) ayrı ayrı ölçeklendirmiş, bu da genellikle optimal olmayan verimlilik ve doğrulukla sonuçlanmıştır.

Google Brain'den Tan ve Le'nin 2019 tarihli ufuk açıcı makalesi **EfficientNet: Model Ölçeklendirmeyi Yeniden Düşünmek**, model ölçeklendirmeye prensipli ve sistematik bir yaklaşım önererek bu paradigmayı devrim niteliğinde değiştirdi. EfficientNet'in temel fikri, üç boyutu da (derinlik, genişlik ve çözünürlük) **bileşik bir katsayı** kullanarak tekdüze bir şekilde ölçeklendirmektir. Bu yöntem, bu boyutlar arasındaki dengenin korunmasını sağlayarak, keyfi olarak ölçeklendirilmiş modellere kıyasla önemli ölçüde daha az parametre ve FLOP ile daha yüksek doğruluk elde edilmesini sağlar. EfficientNet modelleri, ImageNet'te en gelişmiş doğruluk düzeyine ulaşırken, önceki modellere göre 10 kata kadar daha verimliydi ve verimli derin öğrenme için yeni bir standart belirledi.

## 2. Arka Plan ve Motivasyon <a name="2-arka-plan-ve-motivasyon"></a>
EfficientNet'ten önce, sinir ağlarını ölçeklendirmek genellikle sezgisel yöntemler veya manuel ayarlamalar içeriyordu. Yaygın uygulamalar şunları içeriyordu:
*   **Derinlik Ölçeklendirme:** Bir ağa daha fazla katman ekleme (örn. ResNet-18'den ResNet-50'ye). Daha derin ağlar daha zengin ve karmaşık özellikleri yakalayabilir ancak gradyanların kaybolmasına ve eğitim süresinin artmasına yatkındır.
*   **Genişlik Ölçeklendirme:** Her katmandaki kanal sayısını artırma (örn. Wide ResNet). Daha geniş ağlar daha ince taneli özellikleri yakalama eğilimindedir ve eğitilmesi daha kolaydır, ancak aynı zamanda fazlalığa ve daha yüksek bellek tüketimine yol açabilir.
*   **Çözünürlük Ölçeklendirme:** Ağa daha yüksek çözünürlüklü görüntüler besleme (örn. 224x224'ten 331x331'e). Daha yüksek çözünürlükler daha küçük nesnelerin ve daha ince ayrıntıların tespitine yardımcı olabilir, ancak erken katmanlar için hesaplama maliyetini önemli ölçüde artırır.

Bu bireysel ölçeklendirme stratejilerinin zorluğu, ağ boyutları arasındaki karşılıklı bağımlılıkları genellikle göz ardı etmeleridir. Örneğin, bir ağın derinliği önemli ölçüde artırılırken genişliği veya çözünürlüğü orantılı olarak artırılmazsa, model zenginleştirilmiş özellikleri etkili bir şekilde işlemekte zorlanabilir veya daha yüksek çözünürlüklü girdiler sığ bir ağdan tam olarak yararlanamayabilir. Yazarlar, herhangi bir tek boyutun ölçeklendirilmesinin doğruluğu artırdığını, ancak daha büyük modeller için kazanımların azaldığını ve daha büyük modellerin genellikle ince ayar veya özel eğitim rejimleri gerektirdiğini gözlemlemişlerdir. Bu gözlem, bu ölçeklendirme boyutları arasında optimal bir denge olduğu ve bunları ayrı ayrı değil de birlikte ölçeklendirmenin üstün sonuçlar verebileceği hipotezini motive etti. Amaç, model verimliliğini ve doğruluğunu eşzamanlı olarak maksimize eden **prensipli bir ölçeklendirme stratejisi** bulmaktı.

## 3. EfficientNet Mimarisi ve Bileşik Ölçeklendirme <a name="3-efficientnet-mimarisi-ve-bileşik-ölçeklendirme"></a>
EfficientNet'in başarısı iki anahtar bileşenden kaynaklanmaktadır: EfficientNet-B0 adlı güçlü bir **taban ağı** ve derinliği, genişliği ve çözünürlüğü tekdüze bir şekilde ölçeklendiren yeni bir **bileşik ölçeklendirme yöntemi**.

### 3.1 Temel Ağ (EfficientNet-B0) <a name="31-temel-ağ-efficientnet-b0"></a>
EfficientNet için başlangıç noktası, yüksek düzeyde optimize edilmiş bir temel model olan **EfficientNet-B0**'dır. Bu temel model, hem doğruluğu hem de FLOP'ları optimize etmeyi amaçlayan AutoML MNAS çerçevesi gibi bir **Nöral Mimari Arama (NAS)** tekniği kullanılarak tasarlanmıştır. EfficientNet-B0, MobileNetV2 ve MobileNetV3'te bulunanlara benzer şekilde, esas olarak **MBConv (Mobile Ters Darboğaz Evrişimi)** bloklarını kullanır. Bu bloklar şunlarla karakterize edilir:
*   **Derinlemesine Ayrılabilir Evrişimler:** Uzamsal ve kanal bazlı evrişimleri ayırarak hesaplama maliyetini düşürme.
*   **Ters Artıklar:** Giriş ve çıkışın dar olduğu, ancak ara genişletme katmanının geniş olduğu bir darboğaz yapısı kullanma.
*   **Sıkma-ve-Uyarma (SE) Ağları:** Kanal bazlı özellik tepkilerini dinamik olarak yeniden kalibre ederek, önemli bir hesaplama yükü olmadan temsil gücünü daha da artırma.

Bu yüksek verimli temel mimari, ölçeklendirme için mükemmel bir temel sağlar ve sonraki ölçeklendirilmiş modellerin doğal verimliliğini devralmasını garanti eder.

### 3.2 Bileşik Ölçeklendirme Prensibi <a name="32-bileşik-ölçeklendirme-prensibi"></a>
EfficientNet'in en önemli katkısı, **bileşik ölçeklendirme yöntemidir**. Derinlik ($d$), genişlik ($w$) veya çözünürlüğü ($r$) keyfi olarak ayrı ayrı ölçeklendirmek yerine, EfficientNet, bunları bir bileşik katsayı $\phi$ kullanarak eşzamanlı olarak ölçeklendirmeyi önerir. Bu, aşağıdaki denklem kümesiyle ifade edilebilir:

Derinlik: $d = \alpha^\phi$
Genişlik: $w = \beta^\phi$
Çözünürlük: $r = \gamma^\phi$

burada $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ ve $\alpha \ge 1, \beta \ge 1, \gamma \ge 1$ kısıtlamaları altındadır. Burada $\alpha, \beta, \gamma$, temel model üzerinde küçük bir ızgara aramasıyla belirlenen sabitlerdir ve $\phi$, genel kaynak ölçeklendirmesini kontrol eden kullanıcı tarafından belirlenen bir katsayıdır. $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ kısıtı, $\phi$'deki her artış için toplam FLOP'ların kabaca ikiye katlanmasını sağlar, bu da modelleri büyütmek için yaygın bir uygulamadır.

Bileşik ölçeklendirmenin ardındaki sezgi, daha büyük bir giriş görüntü çözünürlüğünün daha fazla pikseli yakalamak için daha derin bir ağa ve daha ince taneli desenleri yakalamak için daha geniş bir ağa ihtiyaç duymasıdır. Tersine, daha derin ve daha geniş bir ağ, daha fazla özelliği işlemek için daha yüksek çözünürlüklü girişlerden faydalanır. Her üç boyutu birlikte ölçeklendirerek, EfficientNet daha iyi bir denge ve sinerji elde eder, bu da daha verimli performans kazanımlarına yol açar.

### 3.3 Ölçeklendirme Katsayıları <a name="33-ölçeklendirme-katsayıları)</a>
$\alpha, \beta, \gamma$ için optimal değerler iki aşamalı bir süreçle bulunur:
1.  **Adım 1:** $\phi = 1$ olarak sabitlenir ve küçük temel model (EfficientNet-B0) üzerinde $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ kısıtı altında $\alpha, \beta, \gamma$ için küçük bir ızgara araması yapılır. EfficientNet-B0 için, yazarlar $\alpha \approx 1.2$, $\beta \approx 1.1$ ve $\gamma \approx 1.15$ değerlerini iyi başlangıç değerleri olarak bulmuşlardır.
2.  **Adım 2:** $\alpha, \beta, \gamma$ sabit olarak tutulur ve temel ağ farklı $\phi$ değerleriyle ölçeklendirilerek EfficientNet-B1'den B7'ye kadar modeller elde edilir. Örneğin, EfficientNet-B1 için $\phi$ 1, B2 için $\phi$ 2 olacaktır ve bu böyle devam eder; gerçek $\phi$ değerleri belirli hedef kaynaklara ulaşacak şekilde hesaplanır.

Bu sistematik yaklaşım, optimal verimliliği korurken boyut ve doğruluğu kademeli olarak artıran EfficientNet-B0'dan EfficientNet-B7'ye kadar bir model ailesinin oluşturulmasını sağlar. Her model (örn. EfficientNet-B3), temel modelden türetilen aynı optimal ölçeklendirme oranlarını $\alpha, \beta, \gamma$ miras alarak farklı bir $\phi$ değerine karşılık gelir.

## 4. Kod Örneği <a name="4-kod-örneği"></a>
Bu örnek, PyTorch'ta bilgisayar görüşü modelleri için yaygın olarak kullanılan `timm` (PyTorch Image Models) kütüphanesini kullanarak önceden eğitilmiş bir EfficientNet modelinin nasıl yükleneceğini göstermektedir. EfficientNet ailesinin farklı ölçeklendirilmiş versiyonlarına kolayca erişimi sergiler.

```python
import timm
import torch

# Gösterim için EfficientNet modellerinin bir listesini tanımlayın
efficientnet_modelleri = [
    'efficientnet_b0',
    'efficientnet_b3',
    'efficientnet_b7'
]

print("--- EfficientNet Modellerinin Gösterimi ---")

for model_adı in efficientnet_modelleri:
    print(f"\nModel yükleniyor: {model_adı}")
    # Önceden eğitilmiş bir EfficientNet modeli yükleyin
    # pretrained=True, ImageNet üzerinde eğitilmiş ağırlıkları indirir
    try:
        model = timm.create_model(model_adı, pretrained=True)
        model.eval() # Modeli değerlendirme moduna ayarlayın

        # Yüklenen model hakkında bazı temel bilgileri yazdırın
        print(f"  Modelin varsayılan giriş boyutu: {model.default_cfg['input_size']}")
        print(f"  Parametre sayısı: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

        # Beklenen giriş boyutuna uyan sahte bir giriş tensörü oluşturun
        # Beklenen input_size genellikle (C, H, W) şeklindedir
        _, H, W = model.default_cfg['input_size']
        sahte_giriş = torch.randn(1, 3, H, W) # Parti boyutu 1, 3 kanal, HxW çözünürlük

        # Bir ileri geçiş (forward pass) gerçekleştirin
        with torch.no_grad(): # Çıkarım için gradyan hesaplamayı devre dışı bırakın
            çıktı = model(sahte_giriş)
        print(f"  Sahte giriş için çıkış şekli: {çıktı.shape}")

    except Exception as e:
        print(f"  {model_adı} yüklenirken hata oluştu: {e}")

print("\n--- Gösterimin sonu ---")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç <a name="5-sonuç"></a>
EfficientNet, evrişimsel sinir ağlarının tasarımında ve ölçeklendirilmesinde önemli bir ilerleme kaydetti. Ağ derinliğini, genişliğini ve çözünürlüğünü tekdüze bir şekilde ölçeklendiren prensipli bir **bileşik ölçeklendirme yöntemi** sunarak, keyfi ölçeklendirme stratejilerinin sınırlamalarını giderdi. B0'dan B7'ye kadar uzanan EfficientNet model ailesi, üstün doğruluk-FLOPs oranları sergileyerek, çok daha az parametre ve hesaplama maliyetiyle ImageNet'te en gelişmiş performansı elde etti.

EfficientNet'ten çıkarılacak temel dersler şunlardır:
*   Ölçeklendirme için temel olarak güçlü, verimli bir **taban ağının** (EfficientNet-B0) önemi.
*   Ağ boyutlarını (derinlik, genişlik, çözünürlük) **dengeli ve bileşik bir şekilde** ölçeklendirmenin optimal verimlilik ve doğruluk için kritik olduğunun keşfedilmesi.
*   Yüksek verimli yapı taşlarını ve temel mimarileri bulmada **Nöral Mimari Arama (NAS)**'ın gücü.

EfficientNet, nesne algılamadan anlamsal segmentasyona kadar çeşitli alanlarda daha verimli mimarilere, daha iyi ölçeklendirme tekniklerine ve uygulamalara odaklanan sayısız sonraki çalışmaya ilham vererek, verimli model tasarımında sonraki araştırmaları derinden etkiledi. Metodolojileri, yüksek performanslı, kaynak verimli derin öğrenme modelleri geliştirmek için bir köşe taşı olmaya devam etmekte, gelişmiş yapay zekayı daha erişilebilir ve sürdürülebilir kılmaktadır.





