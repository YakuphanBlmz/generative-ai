# The Evolution of ResNet and Skip Connections

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenge of Deep Neural Networks](#2-the-challenge-of-deep-neural-networks)
- [3. The Vanishing/Exploding Gradient and Degradation Problems](#3-the-vanishingexploding-gradient-and-degradation-problems)
- [4. ResNet and the Innovation of Skip Connections](#4-resnet-and-the-innovation-of-skip-connections)
  - [4.1. Residual Learning](#41-residual-learning)
  - [4.2. The Residual Block](#42-the-residual-block)
  - [4.3. Mitigation of Training Difficulties](#43-mitigation-of-training-difficulties)
- [5. Impact and Legacy of ResNet](#5-impact-and-legacy-of-resnet)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The pursuit of deeper neural networks has been a driving force in the advancement of Artificial Intelligence, particularly in areas like computer vision. Theoretically, deeper networks possess a greater capacity to learn complex features and representations, leading to improved performance on challenging tasks. However, this theoretical advantage was historically hampered by practical difficulties in training very deep architectures. Prior to 2015, increasing the number of layers in a convolutional neural network (CNN) often led to performance saturation and then rapid degradation, a phenomenon counter-intuitive to the belief that deeper models should at least perform as well as their shallower counterparts. The introduction of **Residual Networks** (ResNet) by He et al. in 2015 marked a monumental shift in deep learning paradigm, effectively addressing these challenges through the ingenious concept of **skip connections**, also known as **residual connections**. This document explores the historical context, the technical problems ResNet solved, its architectural innovations, and its profound impact on the field.

<a name="2-the-challenge-of-deep-neural-networks"></a>
## 2. The Challenge of Deep Neural Networks
Before ResNet, the primary obstacles to training extremely deep neural networks were primarily twofold: the **vanishing/exploding gradient problem** and the **degradation problem**. The desire to create deeper models stemmed from the empirical success of architectures like AlexNet, VGG, and Inception, which demonstrated that more layers often correlated with higher accuracy on benchmark datasets. However, beyond a certain depth, simply stacking more layers did not lead to better performance; instead, it often resulted in higher training error, indicating that the networks were failing to learn effectively.

The general approach to designing neural networks involves layers of interconnected neurons, where each layer learns increasingly abstract representations of the input data. In conventional deep feedforward networks, the output of one layer serves as the input to the next, forming a sequential chain. This sequential dependency, while fundamental, becomes problematic as the chain lengthens.

<a name="3-the-vanishingexploding-gradient-and-degradation-problems"></a>
## 3. The Vanishing/Exploding Gradient and Degradation Problems
The **vanishing/exploding gradient problem** is a fundamental challenge in training deep neural networks, especially recurrent neural networks, but also applicable to very deep feedforward networks. During backpropagation, gradients are computed by applying the chain rule, which involves multiplying gradients across layers.
*   **Vanishing Gradients:** If the gradients of the activation functions (e.g., sigmoid or tanh, which squash outputs into a small range) are small, repeatedly multiplying these small values across many layers causes the gradient to shrink exponentially as it propagates backward towards the initial layers. This makes the updates to the weights of early layers very small, effectively preventing them from learning meaningful features.
*   **Exploding Gradients:** Conversely, if gradients are large, repeated multiplication can cause them to grow exponentially, leading to extremely large weight updates that destabilize the network and prevent convergence.

While techniques like **Batch Normalization** and appropriate weight initialization (e.g., Xavier/He initialization) helped mitigate vanishing/exploding gradients to some extent, they did not fully resolve the issue for ultra-deep networks.

More subtly, a distinct problem known as the **degradation problem** emerged. It was observed that when conventional deep networks were deepened by adding more layers, their training error often *increased*, rather than decreased, even when overfitting was not an issue (i.e., performance on the training set degraded). This suggested that the deeper models were inherently harder to optimize. Crucially, it was hypothesized that the network was struggling to learn an **identity mapping** (i.e., a function that simply passes its input through unchanged). If adding layers was truly beneficial, a deeper model should at least be able to learn the identity mapping for the additional layers, thus performing no worse than its shallower counterpart. The fact that it performed worse implied that even learning this simple identity mapping was difficult for standard convolutional layers.

<a name="4-resnet-and-the-innovation-of-skip-connections"></a>
## 4. ResNet and the Innovation of Skip Connections
The **Residual Network** (ResNet) architecture, introduced by Kaiming He and colleagues, provided an elegant solution to the degradation problem and significantly alleviated the vanishing gradient issue, enabling the training of networks with hundreds or even thousands of layers. The core innovation of ResNet lies in its use of **skip connections**.

<a name="41-residual-learning"></a>
### 4.1. Residual Learning
Instead of expecting convolutional layers to directly learn the desired mapping, $H(x)$, ResNet proposes that these layers learn a **residual mapping**, $F(x)$, where $F(x) = H(x) - x$. Consequently, the original mapping is recast as $H(x) = F(x) + x$. This formulation is critical because it is often easier to optimize the residual mapping $F(x)$ to zero than to approximate an identity mapping $H(x) = x$ with a stack of non-linear layers. If the optimal function for a block of layers is an identity mapping, the network only needs to learn to push the weights towards zero for the residual function $F(x)$, which is generally simpler than forcing a stack of non-linear layers to perfectly output the identity.

<a name="42-the-residual-block"></a>
### 4.2. The Residual Block
The fundamental building block of a ResNet is the **residual block**. In a typical residual block, the output of a stack of convolutional layers (the "main path" or "residual path") is added to the original input (the "shortcut connection" or "skip connection"). Mathematically, if $x$ is the input to the block, and $F(x)$ represents the transformations applied by the convolutional layers within the block, the output $y$ of the block is:

$y = F(x) + x$

The skip connection effectively bypasses one or more layers, directly passing the input $x$ to the output of the block, where it is summed with the output of the main path. A non-linear activation function (e.g., ReLU) is then typically applied after this summation. If the dimensions of $x$ and $F(x)$ do not match (e.g., due to stride or change in the number of filters), a linear projection (e.g., a 1x1 convolution) can be applied to the shortcut connection to match the dimensions, $y = F(x) + W_s x$.

<a name="43-mitigation-of-training-difficulties"></a>
### 4.3. Mitigation of Training Difficulties
*   **Alleviating Vanishing Gradients:** Skip connections provide alternative, direct pathways for the gradient to flow during backpropagation. Instead of relying solely on multiplicative gradients through many layers, the additive nature of the skip connection allows gradients to propagate directly to earlier layers, mitigating the vanishing gradient problem. This creates a kind of "information superhighway" for gradients.
*   **Solving the Degradation Problem:** By framing the learning task as an identity mapping plus a residual, the network finds it much easier to learn. If the optimal mapping for a block is indeed identity, the network simply learns $F(x) \approx 0$. This is significantly easier than forcing a stack of non-linear layers to precisely approximate $H(x) = x$. This mechanism ensures that adding more layers will not degrade performance, as the worst-case scenario is that the additional layers learn an identity mapping and contribute nothing detrimental.

<a name="5-impact-and-legacy-of-resnet"></a>
## 5. Impact and Legacy of ResNet
ResNet's introduction was a groundbreaking moment in deep learning. The original paper achieved state-of-the-art results on ImageNet classification (winning ILSVRC 2015) and object detection, demonstrating that significantly deeper networks could indeed be trained effectively. The architectural elegance and empirical success of ResNet quickly led to its widespread adoption across various computer vision tasks, including semantic segmentation, object detection, and image generation.

Its influence extends beyond its direct applications. The concept of skip connections became a fundamental component in many subsequent successful architectures. For example, **DenseNet** extended this idea by connecting each layer to every other subsequent layer in a feed-forward fashion, promoting feature reuse. Architectures like **U-Net** in medical image segmentation also heavily leverage skip connections (though often concatenated rather than summed) to combine low-level and high-level features. The residual learning paradigm also inspired **Highway Networks**, which predated ResNet and used "transform gates" and "carry gates" to control information flow, conceptually similar to how skip connections offer a pathway.

ResNet not only enabled the creation of ultra-deep networks but also changed how researchers approached network design, emphasizing the importance of gradient flow and ease of optimization. It remains a foundational architecture in the deep learning toolkit, routinely used as a backbone for more complex models and in transfer learning scenarios.

<a name="6-code-example"></a>
## 6. Code Example
Below is a conceptual Python snippet demonstrating a basic **Residual Block** using PyTorch. It illustrates how the input `x` is passed through a main path of convolutional layers and then added back to the original `x` via a skip connection before the final activation.

```python
import torch
import torch.nn as nn

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializes a basic residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolutional layer (affects spatial dimensions).
        """
        super(BasicResidualBlock, self).__init__()
        # First convolutional layer and batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer and batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path (skip connection) to handle dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, use a 1x1 convolution for the shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Store original input for the skip connection
        identity = x

        # Main path processing
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut connection to the main path's output
        # This is the core of the residual connection: F(x) + x
        out += self.shortcut(identity)
        out = self.relu(out) # Apply ReLU after summation
        return out

# Example usage (uncomment to run):
# input_tensor = torch.randn(1, 64, 32, 32) # Batch size 1, 64 channels, 32x32 image
# print(f"Input shape: {input_tensor.shape}")
#
# # Create a block that changes channels from 64 to 128 and halves spatial dimensions
# block = BasicResidualBlock(64, 128, stride=2)
# output_tensor = block(input_tensor)
# print(f"Output shape after residual block: {output_tensor.shape}") # Expected: torch.Size([1, 128, 16, 16])

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion
The introduction of ResNet and the concept of skip connections fundamentally transformed the landscape of deep learning. By ingeniously reformulating the learning objective to a residual function and providing direct gradient pathways, ResNet effectively overcame the long-standing challenges of training ultra-deep neural networks, namely the vanishing gradient and degradation problems. This innovation not only enabled the development of significantly deeper and more powerful models, leading to unprecedented performance in computer vision, but also inspired a wave of subsequent architectural designs that leveraged similar principles. ResNet's enduring legacy highlights the power of architectural innovation in unlocking the full potential of deep learning.

---
<br>

<a name="türkçe-içerik"></a>
## ResNet ve Atlama Bağlantılarının Evrimi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Derin Sinir Ağlarının Zorlukları](#2-derin-sinir-ağlarının-zorlukları)
- [3. Sönümlenen/Patlayan Gradyan ve Dejenarasyon Problemleri](#3-sönümlenenpatlayan-gradyan-ve-dejenarasyon-problemleri)
- [4. ResNet ve Atlama Bağlantılarının Yeniliği](#4-resnet-ve-atlama-bağlantılarının-yeniliği)
  - [4.1. Artık Öğrenme](#41-artık-öğrenme)
  - [4.2. Artık Blok](#42-artık-blok)
  - [4.3. Eğitim Zorluklarının Azaltılması](#43-eğitim-zorluklarının-azaltılması)
- [5. ResNet'in Etkisi ve Mirası](#5-resnetin-etkisi-ve-mirası)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Derin sinir ağlarının derinleştirilmesi arayışı, Yapay Zeka, özellikle de bilgisayar görüşü gibi alanlarda ilerlemenin itici gücü olmuştur. Teorik olarak, daha derin ağlar, karmaşık özellikleri ve temsilleri öğrenmek için daha büyük bir kapasiteye sahiptir, bu da zorlu görevlerde performansın artmasına yol açar. Ancak, bu teorik avantaj, çok derin mimarilerin eğitilmesindeki pratik zorluklar nedeniyle tarihsel olarak engellenmiştir. 2015'ten önce, bir evrişimli sinir ağındaki (CNN) katman sayısını artırmak, genellikle performans doygunluğuna ve ardından hızlı bir bozulmaya yol açıyordu; bu durum, daha derin modellerin en azından daha sığ benzerleri kadar iyi performans göstermesi gerektiği inancına aykırıydı. He ve ark. tarafından 2015'te **Residual Ağlar** (ResNet) tanıtılması, derin öğrenme paradigmasında anıtsal bir değişimi işaret etti ve **atlama bağlantıları** (skip connections) olarak da bilinen **artık bağlantılar** (residual connections) kavramı aracılığıyla bu zorlukları etkili bir şekilde ele aldı. Bu belge, tarihsel bağlamı, ResNet'in çözdüğü teknik sorunları, mimari yeniliklerini ve alan üzerindeki derin etkisini incelemektedir.

<a name="2-derin-sinir-ağlarının-zorlukları"></a>
## 2. Derin Sinir Ağlarının Zorlukları
ResNet'ten önce, aşırı derin sinir ağlarının eğitimindeki temel engeller başlıca iki taneydi: **sönümlenen/patlayan gradyan problemi** ve **dejenarasyon problemi**. Daha derin modeller oluşturma isteği, AlexNet, VGG ve Inception gibi mimarilerin ampirik başarısından kaynaklanıyordu; bu mimariler, daha fazla katmanın genellikle kıyaslama veri kümelerinde daha yüksek doğrulukla ilişkili olduğunu göstermişti. Ancak, belirli bir derinliğin ötesinde, sadece daha fazla katman eklemek daha iyi performansa yol açmadı; bunun yerine, genellikle daha yüksek eğitim hatasına neden oldu, bu da ağların etkili bir şekilde öğrenemediğini gösteriyordu.

Sinir ağlarını tasarlamanın genel yaklaşımı, birbirine bağlı nöron katmanlarını içerir; her katman, girdi verilerinin giderek daha soyut temsillerini öğrenir. Geleneksel derin ileri beslemeli ağlarda, bir katmanın çıktısı bir sonrakinin girdisi olarak hizmet eder ve sıralı bir zincir oluşturur. Bu sıralı bağımlılık, temel olmasına rağmen, zincir uzadıkça sorunlu hale gelir.

<a name="3-sönümlenenpatlayan-gradyan-ve-dejenarasyon-problemleri"></a>
## 3. Sönümlenen/Patlayan Gradyan ve Dejenarasyon Problemleri
**Sönümlenen/patlayan gradyan problemi**, derin sinir ağlarını, özellikle yinelemeli sinir ağlarını eğitirken karşılaşılan temel bir zorluktur, ancak çok derin ileri beslemeli ağlar için de geçerlidir. Geri yayılım (backpropagation) sırasında, gradyanlar zincir kuralı uygulanarak hesaplanır, bu da katmanlar arasında gradyanların çarpılmasını içerir.
*   **Sönümlenen Gradyanlar:** Aktivasyon fonksiyonlarının (örneğin, çıktıları küçük bir aralığa sıkıştıran sigmoid veya tanh) gradyanları küçükse, bu küçük değerlerin birçok katman boyunca tekrar tekrar çarpılması, gradyanın ilk katmanlara doğru geriye doğru yayılırken üstel olarak küçülmesine neden olur. Bu durum, erken katmanların ağırlıklarına yapılan güncellemeleri çok küçük hale getirir ve bunların anlamlı özellikler öğrenmesini engeller.
*   **Patlayan Gradyanlar:** Tersine, gradyanlar büyükse, tekrarlanan çarpma, gradyanların üstel olarak büyümesine neden olabilir, bu da ağın dengesini bozan ve yakınsamayı engelleyen aşırı büyük ağırlık güncellemelerine yol açar.

**Toplu Normalizasyon** (Batch Normalization) ve uygun ağırlık başlatma (örneğin, Xavier/He başlatma) gibi teknikler sönümlenen/patlayan gradyanları bir dereceye kadar azaltmaya yardımcı olsa da, ultra derin ağlar için sorunu tam olarak çözemediler.

Daha incelikli bir şekilde, **dejenarasyon problemi** olarak bilinen farklı bir sorun ortaya çıktı. Geleneksel derin ağlar daha fazla katman eklenerek derinleştirildiğinde, aşırı uyum (overfitting) bir sorun olmasa bile (yani, eğitim setindeki performans bozulduğunda bile), eğitim hatalarının azamak yerine *arttığı* gözlemlendi. Bu durum, daha derin modellerin optimize edilmesinin doğası gereği daha zor olduğunu gösteriyordu. En önemlisi, ağın bir **kimlik eşlemesi** (identity mapping) (yani, girdisini değiştirmeden geçiren bir fonksiyon) öğrenmekte zorlandığı varsayıldı. Katman eklemek gerçekten faydalı olsaydı, daha derin bir model, ek katmanlar için kimlik eşlemesini en azından öğrenebilmeli ve böylece daha sığ benzerinden daha kötü performans göstermemelidir. Daha kötü performans göstermesi, bu basit kimlik eşlemesini öğrenmenin bile standart evrişimsel katmanlar için zor olduğunu ima ediyordu.

<a name="4-resnet-ve-atlama-bağlantılarının-yeniliği"></a>
## 4. ResNet ve Atlama Bağlantılarının Yeniliği
Kaiming He ve ekibi tarafından tanıtılan **Residual Ağ** (ResNet) mimarisi, dejenarasyon problemine zarif bir çözüm sundu ve sönümlenen gradyan sorununu önemli ölçüde hafifleterek yüzlerce, hatta binlerce katmana sahip ağların eğitilmesini mümkün kıldı. ResNet'in temel yeniliği, **atlama bağlantılarını** (skip connections) kullanmasında yatmaktadır.

<a name="41-artık-öğrenme"></a>
### 4.1. Artık Öğrenme
Evrişimsel katmanlardan istenen eşlemeyi, $H(x)$, doğrudan öğrenmelerini beklemek yerine, ResNet bu katmanların bir **artık eşleme** (residual mapping), $F(x)$ öğrenmesini önerir; burada $F(x) = H(x) - x$. Sonuç olarak, orijinal eşleme $H(x) = F(x) + x$ olarak yeniden ifade edilir. Bu formülasyon kritiktir çünkü artık eşleme $F(x)$'i sıfıra optimize etmek, doğrusal olmayan katman yığını ile bir kimlik eşlemesini $H(x) = x$ yaklaşık olarak elde etmekten genellikle daha kolaydır. Eğer bir blok için optimal fonksiyon bir kimlik eşlemesi ise, ağın sadece artık fonksiyon $F(x)$ için ağırlıkları sıfıra doğru itmeyi öğrenmesi gerekir, ki bu, doğrusal olmayan katman yığınını mükemmel bir şekilde kimlik çıktısı vermeye zorlamaktan genellikle daha basittir.

<a name="42-artık-blok"></a>
### 4.2. Artık Blok
ResNet'in temel yapı taşı **artık bloktur**. Tipik bir artık blokta, bir evrişimsel katman yığınının ("ana yol" veya "artık yol") çıktısı, orijinal girdiye ("kısayol bağlantısı" veya "atlama bağlantısı") eklenir. Matematiksel olarak, eğer $x$ bloğa giren girdi ise ve $F(x)$ blok içindeki evrişimsel katmanlar tarafından uygulanan dönüşümleri temsil ediyorsa, bloğun çıktısı $y$ şudur:

$y = F(x) + x$

Atlama bağlantısı, bir veya daha fazla katmanı etkili bir şekilde atlayarak, girdi $x$'i doğrudan bloğun çıktısına iletir ve burada ana yolun çıktısı ile toplanır. Bu toplamadan sonra tipik olarak doğrusal olmayan bir aktivasyon fonksiyonu (örneğin, ReLU) uygulanır. Eğer $x$ ve $F(x)$'in boyutları eşleşmiyorsa (örneğin, adım (stride) veya filtre sayısındaki değişiklik nedeniyle), boyutları eşleştirmek için kısayol bağlantısına doğrusal bir projeksiyon (örneğin, 1x1 evrişim) uygulanabilir, $y = F(x) + W_s x$.

<a name="43-eğitim-zorluklarının-azaltılması"></a>
### 4.3. Eğitim Zorluklarının Azaltılması
*   **Sönümlenen Gradyanların Azaltılması:** Atlama bağlantıları, geri yayılım sırasında gradyanın akması için alternatif, doğrudan yollar sağlar. Yalnızca birçok katman üzerinden çarpımsal gradyanlara güvenmek yerine, atlama bağlantısının toplamsal yapısı, gradyanların doğrudan önceki katmanlara yayılmasına izin vererek sönümlenen gradyan problemini hafifletir. Bu, gradyanlar için bir tür "bilgi otobanı" yaratır.
*   **Dejenarasyon Probleminin Çözülmesi:** Öğrenme görevini bir kimlik eşlemesi artı bir artık olarak çerçeveleyerek, ağın öğrenmesi çok daha kolaylaşır. Eğer bir blok için optimal eşleme gerçekten kimlik ise, ağ basitçe $F(x) \approx 0$ öğrenir. Bu, doğrusal olmayan katman yığınını $H(x) = x$'i hassas bir şekilde yaklaştırmaya zorlamaktan önemli ölçüde daha kolaydır. Bu mekanizma, daha fazla katman eklemenin performansı düşürmeyeceğini garanti eder, çünkü en kötü senaryo, ek katmanların bir kimlik eşlemesi öğrenmesi ve zararlı hiçbir şey katmamasıdır.

<a name="5-resnetin-etkisi-ve-mirası"></a>
## 5. ResNet'in Etkisi ve Mirası
ResNet'in tanıtımı, derin öğrenmede çığır açan bir andı. Orijinal makale, ImageNet sınıflandırmasında (ILSVRC 2015'i kazanarak) ve nesne tespitinde en güncel sonuçlara ulaşarak, önemli ölçüde daha derin ağların gerçekten etkili bir şekilde eğitilebileceğini gösterdi. ResNet'in mimari zarafeti ve ampirik başarısı, anlamsal segmentasyon, nesne tespiti ve görüntü üretimi dahil olmak üzere çeşitli bilgisayar görüşü görevlerinde hızla yaygınlaşmasına yol açtı.

Etkisi, doğrudan uygulamalarının ötesine geçmektedir. Atlama bağlantıları kavramı, daha sonraki birçok başarılı mimarinin temel bir bileşeni haline geldi. Örneğin, **DenseNet**, her katmanı ileri beslemeli bir şekilde diğer tüm sonraki katmanlara bağlayarak bu fikri genişletti ve özelliklerin yeniden kullanımını teşvik etti. Tıbbi görüntü segmentasyonundaki **U-Net** gibi mimariler de, düşük seviyeli ve yüksek seviyeli özellikleri birleştirmek için atlama bağlantılarını (genellikle toplamak yerine birleştirerek) yoğun bir şekilde kullanır. Artık öğrenme paradigması, ResNet'ten önce ortaya çıkan ve bilgi akışını kontrol etmek için "dönüşüm geçitleri" ve "taşıma geçitleri" kullanan, atlama bağlantılarının bir yol sunmasına kavramsal olarak benzer **Highway Networks**'e de ilham verdi.

ResNet, sadece ultra derin ağların oluşturulmasını sağlamakla kalmadı, aynı zamanda araştırmacıların ağ tasarımına yaklaşımını da değiştirdi; gradyan akışının ve optimizasyon kolaylığının önemini vurguladı. Derin öğrenme araç setinde temel bir mimari olmaya devam etmekte, daha karmaşık modeller için bir omurga olarak ve transfer öğrenme senaryolarında düzenli olarak kullanılmaktadır.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği
Aşağıda, PyTorch kullanarak temel bir **Artık Blok**'u gösteren kavramsal bir Python kodu bulunmaktadır. Bu, `x` girdisinin evrişimsel katmanlardan oluşan ana bir yoldan nasıl geçtiğini ve ardından son aktivasyondan önce bir atlama bağlantısı aracılığıyla orijinal `x`'e nasıl geri eklendiğini göstermektedir.

```python
import torch
import torch.nn as nn

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Temel bir artık bloğu başlatır.

        Args:
            in_channels (int): Girdi kanal sayısı.
            out_channels (int): Çıktı kanal sayısı.
            stride (int): İlk evrişim katmanı için adım (uzamsal boyutları etkiler).
        """
        super(BasicResidualBlock, self).__init__()
        # İlk evrişim katmanı ve toplu normalizasyon
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # İkinci evrişim katmanı ve toplu normalizasyon
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Boyut uyumsuzluğunu gidermek için kısayol yolu (atlama bağlantısı)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Boyutlar değişirse, kısayol için 1x1 evrişim kullan
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Atlama bağlantısı için orijinal girdiyi sakla
        identity = x

        # Ana yol işleme
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Kısayol bağlantısını ana yolun çıktısına ekle
        # Bu, artık bağlantının çekirdeğidir: F(x) + x
        out += self.shortcut(identity)
        out = self.relu(out) # Toplamadan sonra ReLU uygula
        return out

# Kullanım örneği (çalıştırmak için yorum satırlarını kaldırın):
# input_tensor = torch.randn(1, 64, 32, 32) # Batch boyutu 1, 64 kanal, 32x32 görüntü
# print(f"Girdi şekli: {input_tensor.shape}")
#
# # Kanalları 64'ten 128'e değiştiren ve uzamsal boyutları yarıya indiren bir blok oluştur
# block = BasicResidualBlock(64, 128, stride=2)
# output_tensor = block(input_tensor)
# print(f"Artık bloktan sonraki çıktı şekli: {output_tensor.shape}") # Beklenen: torch.Size([1, 128, 16, 16])

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç
ResNet'in ve atlama bağlantıları kavramının tanıtılması, derin öğrenme alanını temelden dönüştürdü. Öğrenme hedefini ustaca bir artık fonksiyona dönüştürerek ve doğrudan gradyan yolları sağlayarak, ResNet, ultra derin sinir ağlarını eğitmenin uzun süredir devam eden zorluklarını, yani sönümlenen gradyan ve dejenarasyon problemlerini etkili bir şekilde aştı. Bu yenilik, sadece bilgisayar görüşünde benzeri görülmemiş performansa yol açan önemli ölçüde daha derin ve daha güçlü modellerin geliştirilmesini sağlamakla kalmadı, aynı zamanda benzer prensipleri kullanan sonraki mimari tasarımlar dalgasına da ilham verdi. ResNet'in kalıcı mirası, derin öğrenmenin tüm potansiyelini ortaya çıkarmada mimari yeniliğin gücünü vurgulamaktadır.

