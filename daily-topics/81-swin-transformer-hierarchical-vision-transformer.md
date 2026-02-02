# Swin Transformer: Hierarchical Vision Transformer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: Vision Transformers and Their Limitations](#2-background-vision-transformers-and-their-limitations)
- [3. Swin Transformer Architecture](#3-swin-transformer-architecture)
    - [3.1. Hierarchical Feature Representation](#31-hierarchical-feature-representation)
    - [3.2. Shifted Windows Mechanism (SW-MSA)](#32-shifted-windows-mechanism-sw-msa)
    - [3.3. Relative Position Bias](#33-relative-position-bias)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

---

### 1. Introduction
The **Transformer** architecture, originally developed for natural language processing (NLP), revolutionized sequence modeling and achieved remarkable success across various tasks. Its inherent ability to capture long-range dependencies through the **self-attention mechanism** made it a powerful tool. Inspired by this success, researchers extended Transformers to computer vision tasks, leading to the development of **Vision Transformers (ViT)**. While ViTs demonstrated competitive performance, they inherited certain limitations, particularly when scaled to high-resolution images or applied to dense prediction tasks like object detection and semantic segmentation.

The **Swin Transformer** (Swin standing for Shifted Windows) was introduced to address these limitations by incorporating two key innovations: **hierarchical feature representation** and a **shifted window-based self-attention mechanism**. Unlike standard ViTs that operate on a fixed-scale feature map, Swin Transformer constructs a multi-stage, hierarchical architecture akin to conventional convolutional neural networks (CNNs), generating feature maps at different resolutions. This enables it to handle varied scales and makes it suitable for dense prediction. Furthermore, by confining self-attention computations to non-overlapping local windows and introducing a shifted window partitioning scheme, Swin Transformer achieves linear computational complexity with respect to image size, a significant improvement over the quadratic complexity of global self-attention in standard ViTs. This paper delves into the architectural details and the underlying principles that make Swin Transformer a highly efficient and effective backbone for various vision tasks.

### 2. Background: Vision Transformers and Their Limitations
Traditional Convolutional Neural Networks (CNNs) have long been the dominant paradigm in computer vision, leveraging inductive biases such as local connectivity and translational equivariance to achieve high performance. However, the advent of the Transformer architecture in NLP, particularly with the success of models like BERT and GPT, spurred interest in its application to vision.

The **Vision Transformer (ViT)** pioneered this shift by treating images as sequences of patches. An image is first divided into a grid of fixed-size, non-overlapping patches. Each patch is then flattened, linearly projected into a vector, and positional embeddings are added. These patch embeddings form the input sequence to a standard Transformer encoder, which consists of multiple layers of multi-head self-attention (MSA) and feed-forward networks (FFN).

While ViTs demonstrated impressive performance on image classification tasks, especially with large datasets, they faced several inherent limitations:
1.  **Quadratic Computational Complexity:** The core challenge lies in the **global self-attention mechanism**. For an input sequence of length *N* (number of patches), the self-attention computation scales quadratically, *O(N^2)*. For high-resolution images, *N* can be very large (e.g., a 224x224 image with 16x16 patches yields *N=196*; for a 1024x1024 image, *N=4096*). This quadratic scaling makes ViTs computationally expensive and memory-intensive for high-resolution inputs, limiting their applicability to dense prediction tasks that require fine-grained feature maps.
2.  **Lack of Hierarchical Feature Representation:** Standard ViTs produce a single-resolution feature map at the output. This contrasts with CNNs, which build hierarchical feature representations (e.g., through pooling layers), progressively reducing spatial resolution while increasing channel depth. Hierarchical features are crucial for tasks like object detection and semantic segmentation, where information at multiple scales is necessary to detect objects of varying sizes or segment complex regions.
3.  **Limited Inductive Biases:** Unlike CNNs, ViTs lack inherent inductive biases such like locality and translation equivariance. While this makes them highly flexible, it also means they require vast amounts of data to learn these fundamental visual patterns, making them less sample-efficient than CNNs on smaller datasets.

These limitations motivated the development of Swin Transformer, aiming to retain the benefits of self-attention while overcoming the practical hurdles for broader adoption in computer vision.

### 3. Swin Transformer Architecture
The Swin Transformer addresses the limitations of standard Vision Transformers by introducing a novel hierarchical architecture and a more efficient self-attention mechanism. Its design principles aim to bridge the gap between the flexibility of Transformers and the efficiency and multi-scale capabilities of CNNs.

#### 3.1. Hierarchical Feature Representation
A core innovation of Swin Transformer is its ability to construct **hierarchical feature maps**, similar to feature pyramids in CNNs. This is achieved through a multi-stage design, where each stage involves patch merging and Swin Transformer blocks:

*   **Patch Partitioning:** Initially, an input RGB image (e.g., 224x224x3) is partitioned into non-overlapping patches. Unlike ViT's typical 16x16 patches, Swin Transformer usually starts with smaller patches (e.g., 4x4) to retain more fine-grained information. A linear embedding layer projects these raw-valued patches into a higher-dimensional feature space. For instance, a 4x4 patch results in 16 pixels, each with 3 color channels, yielding a 48-dimensional vector, which is then projected to a specified channel dimension *C*. This forms the first stage (Stage 1) with a feature map size of `H/4 x W/4 x C`.
*   **Swin Transformer Blocks:** Multiple Swin Transformer blocks are applied within each stage to learn representations.
*   **Patch Merging (Downsampling):** To create a hierarchical representation, Swin Transformer performs a "patch merging" operation between stages. In a patch merging layer, adjacent `2x2` patches are concatenated. For example, four `C`-dimensional features from four `1x1` spatial locations (forming a `2x2` region) are concatenated, resulting in a `4C`-dimensional feature. A linear layer is then applied to project this `4C`-dimensional feature down to `2C` dimensions. This process effectively halves the spatial resolution (`H/2` and `W/2`) and doubles the feature dimension (`2C`), creating a new, coarser-grained feature map. This downsampling mechanism is analogous to pooling layers in CNNs.
*   **Subsequent Stages:** These patch merging and Swin Transformer block applications are repeated for subsequent stages (Stage 2, 3, 4), progressively reducing the spatial resolution and increasing the channel dimension, thereby building a feature pyramid. This hierarchical structure allows the Swin Transformer to capture both fine-grained details at early stages and high-level semantic information at deeper stages, making it highly suitable for dense prediction tasks.

#### 3.2. Shifted Windows Mechanism (SW-MSA)
The most critical innovation for achieving linear computational complexity and enabling cross-window connections is the **shifted window-based multi-head self-attention (SW-MSA)** mechanism.

*   **Window-based Multi-head Self-Attention (W-MSA):** To reduce the quadratic complexity of global self-attention, Swin Transformer first partitions the feature map into non-overlapping local windows of a fixed size (e.g., 7x7 pixels). Self-attention is then computed independently within each of these local windows. If a feature map of size `H x W` is divided into windows of size `M x M`, there will be `(H/M) x (W/M)` windows. The computational complexity becomes *O(HWC + (H/M)(W/M)M^2C)*, which simplifies to *O(HWC + HW * M^2C / M^2)* or *O(HW * C * M^2)* (if `M` is much smaller than `H,W`). This represents a significant reduction from *O((HW)^2 * C)* complexity of global attention, achieving linear complexity with respect to the number of patches `HW`. However, this window-based attention limits the receptive field of each window and prevents information flow between different windows.

*   **Shifted Window-based Multi-head Self-Attention (SW-MSA):** To enable cross-window connections while maintaining computational efficiency, Swin Transformer introduces a shifted window partitioning approach. This is applied in alternating Swin Transformer blocks:
    *   In the first Swin Transformer block of a module, standard window partitioning (W-MSA) is used.
    *   In the *next* Swin Transformer block, the window partition is **shifted** by `(M/2, M/2)` pixels (where `M` is the window size) from the regular partitioning. This shift allows attention computations to cross the boundaries of the previous windows, thereby introducing connections between adjacent windows that were previously isolated.
    *   To efficiently compute self-attention with shifted windows, especially for irregularly sized windows at the boundaries, a **cyclic shift** and a **masking mechanism** are employed. The feature map is cyclically shifted to the top-left, effectively creating new windows that may span across the original window boundaries. A carefully designed attention mask is then applied to restrict attention computation only to elements within the valid, original windows, ensuring that attention is not computed across artificially connected sub-windows. This clever trick allows the shifted window attention to be computed efficiently in parallel within batch processing.

By alternating between regular and shifted window partitioning, the Swin Transformer allows each feature patch to interact with patches in adjacent windows, gradually expanding the effective receptive field and enabling global information propagation without resorting to computationally expensive global self-attention.

#### 3.3. Relative Position Bias
Another crucial component contributing to the Swin Transformer's performance is the inclusion of **relative position bias** in the self-attention computation. In standard Transformer self-attention, the similarity between query and key is computed as `Q * K^T`. Swin Transformer modifies this by adding a learnable relative position bias `B`:

`Attention(Q, K, V) = SoftMax( (Q K^T / sqrt(d)) + B ) V`

Where `B` is a learnable bias term that depends on the relative coordinates of the patches within a window. Specifically, for any two patches in a window, their relative position can be represented by a `(delta_x, delta_y)` pair. A small matrix `B` is learned for all possible relative positions within a window (e.g., for a 7x7 window, there are `(2M-1)x(2M-1)` possible relative positions).

Adding relative position bias helps the model incorporate positional information, which is particularly beneficial for vision tasks where spatial relationships are critical. Unlike absolute positional embeddings used in ViT, relative positional embeddings are more robust to varying input sizes and better capture local spatial interactions, contributing to the Swin Transformer's overall accuracy.

### 4. Code Example

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Swin Transformer's initial patch embedding and linear projection.
    Converts an image into a sequence of flattened patches and embeds them.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        # Calculate image dimensions and number of patches
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) e.g., (1, 3, 224, 224)
        x = self.proj(x) # (B, embed_dim, H/patch_size, W/patch_size) e.g., (1, 96, 56, 56)
        x = x.flatten(2) # (B, embed_dim, num_patches) e.g., (1, 96, 3136)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim) e.g., (1, 3136, 96)
        return x

class PatchMerging(nn.Module):
    """
    Swin Transformer's patch merging layer for hierarchical feature reduction.
    Downsamples spatial resolution by 2x and doubles feature dimension.
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution # (H, W)
        self.dim = dim # input dimension
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) # 2x2 adjacent patches -> 4*dim -> 2*dim
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        # x: (B, H*W, dim) e.g., (1, 56*56, 96)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C) # (B, H, W, C) e.g., (1, 56, 56, 96)

        # Divide into 4 sub-regions and concatenate
        x0 = x[:, 0::2, 0::2, :]  # Top-left (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # Top-right (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)  # Concat (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # Flatten (B, (H/2)*(W/2), 4*C)

        x = self.norm(x)
        x = self.reduction(x) # Linear projection (B, (H/2)*(W/2), 2*C)

        return x

# Example Usage:
if __name__ == '__main__':
    # Initial Patch Embedding
    patch_embed = PatchEmbed(img_size=224, patch_size=4, in_chans=3, embed_dim=96)
    dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224 image
    embedded_patches = patch_embed(dummy_input)
    print(f"Embedded patches shape: {embedded_patches.shape}") # Expected: (1, 3136, 96) -> (B, num_patches, embed_dim)

    # First Patch Merging operation
    # Input resolution (H, W) = (224/4, 224/4) = (56, 56)
    patch_merging = PatchMerging(input_resolution=(56, 56), dim=96)
    merged_patches = patch_merging(embedded_patches)
    print(f"Merged patches shape: {merged_patches.shape}") # Expected: (1, 784, 192) -> (B, num_patches/4, 2*embed_dim)

(End of code example section)
```

### 5. Conclusion
The Swin Transformer represents a pivotal advancement in the field of computer vision, effectively bridging the gap between the global modeling capabilities of Transformers and the inductive biases and efficiency of Convolutional Neural Networks. By introducing a **hierarchical feature representation** and the innovative **shifted window-based self-attention mechanism**, Swin Transformer overcomes the quadratic computational complexity inherent in standard Vision Transformers, making it scalable to high-resolution images and suitable for dense prediction tasks like object detection and semantic segmentation.

Its ability to build a feature pyramid allows it to capture multi-scale information crucial for complex visual understanding, while the shifted windows ensure efficient information exchange across local regions without incurring prohibitive computational costs. Furthermore, the inclusion of **relative position bias** enhances the model's spatial awareness and learning capacity.

As a result of these architectural innovations, Swin Transformer has achieved state-of-the-art performance across a wide array of vision benchmarks, establishing itself as a robust and efficient backbone for future computer vision research and applications. Its success underscores the importance of carefully designed inductive biases when adapting powerful general-purpose architectures to specific domains.

### 6. References
1.  Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Dai, J. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012-10022.
2.  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*.
3.  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in neural information processing systems*, 30.

---
<br>

<a name="türkçe-içerik"></a>
## Swin Transformer: Hiyerarşik Vizyon Trafosu

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: Vizyon Trafoları ve Sınırlamaları](#2-arka-plan-vizyon-trafoları-ve-sınırlamaları)
- [3. Swin Transformer Mimarisi](#3-swin-transformer-mimarisi)
    - [3.1. Hiyerarşik Özellik Temsili](#31-hiyerarşik-özellik-temsili)
    - [3.2. Kaydırılmış Pencere Mekanizması (SW-MSA)](#32-kaydırılmış-pencere-mekanizması-sw-msa)
    - [3.3. Göreceli Konum Sapması](#33-göreceli-konum-sapması)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Referanslar](#6-referanslar)

---

### 1. Giriş
Başlangıçta doğal dil işleme (NLP) için geliştirilen **Transformer** mimarisi, dizi modellemede devrim yarattı ve çeşitli görevlerde dikkat çekici başarılar elde etti. **Öz-dikkat mekanizması** aracılığıyla uzun menzilli bağımlılıkları yakalama konusundaki doğal yeteneği, onu güçlü bir araç haline getirdi. Bu başarıdan ilham alan araştırmacılar, Transformer'ları bilgisayar görü görevlerine uyarlayarak **Vizyon Trafoları (ViT)** gelişimine öncülük ettiler. ViT'ler rekabetçi performans göstermelerine rağmen, özellikle yüksek çözünürlüklü görüntülerle çalışırken veya nesne algılama ve anlamsal segmentasyon gibi yoğun tahmin görevlerine uygulandığında belirli sınırlamaları miras aldılar.

**Swin Transformer** (Swin, Kaydırılmış Pencereler anlamına gelir) bu sınırlamaları iki temel yenilikle ele almak üzere tanıtıldı: **hiyerarşik özellik temsili** ve **kaydırılmış pencere tabanlı öz-dikkat mekanizması**. Sabit ölçekli bir özellik haritası üzerinde çalışan standart ViT'lerin aksine, Swin Transformer, geleneksel evrişimli sinir ağlarına (CNN'ler) benzer çok aşamalı, hiyerarşik bir mimari oluşturur ve farklı çözünürlüklerde özellik haritaları üretir. Bu, çeşitli ölçekleri işlemesini sağlar ve yoğun tahmin görevleri için uygun hale getirir. Ayrıca, öz-dikkat hesaplamalarını çakışmayan yerel pencerelerle sınırlayarak ve kaydırılmış bir pencere bölümleme şeması sunarak, Swin Transformer görüntü boyutuyla doğrusal hesaplama karmaşıklığı elde eder; bu, standart ViT'lerdeki küresel öz-dikkat mekanizmasının karesel karmaşıklığına göre önemli bir iyileştirmedir. Bu makale, Swin Transformer'ı çeşitli vizyon görevleri için son derece verimli ve etkili bir omurga haline getiren mimari detayları ve temel prensipleri inceleyecektir.

### 2. Arka Plan: Vizyon Trafoları ve Sınırlamaları
Geleneksel Evrişimli Sinir Ağları (CNN'ler), yerel bağlantı ve çevirme değişmezliği gibi indüktif önyargılardan yararlanarak bilgisayar görüşünde uzun süredir baskın bir paradigma olmuştur. Ancak, NLP'deki Transformer mimarisinin, özellikle BERT ve GPT gibi modellerin başarısıyla birlikte ortaya çıkışı, bilgisayar görüsüne uygulanmasına olan ilgiyi artırdı.

**Vizyon Trafosu (ViT)**, görüntüleri yamalar dizisi olarak ele alarak bu değişimin öncüsü oldu. Bir görüntü önce sabit boyutlu, çakışmayan yamalar ızgarasına bölünür. Her yama daha sonra düzleştirilir, doğrusal olarak bir vektöre dönüştürülür ve konum gömüleri eklenir. Bu yama gömüleri, çok başlı öz-dikkat (MSA) ve ileri beslemeli ağlardan (FFN) oluşan standart bir Transformer kodlayıcısına giriş dizisini oluşturur.

ViT'ler, özellikle büyük veri kümeleriyle görüntü sınıflandırma görevlerinde etkileyici performans göstermelerine rağmen, birkaç doğal sınırlamayla karşılaştılar:
1.  **Karesel Hesaplama Karmaşıklığı:** Temel zorluk, **küresel öz-dikkat mekanizmasında** yatmaktadır. *N* uzunluğundaki (yama sayısı) bir giriş dizisi için, öz-dikkat hesaplaması karesel olarak ölçeklenir, *O(N^2)*. Yüksek çözünürlüklü görüntüler için *N* çok büyük olabilir (örneğin, 16x16 yamalara sahip 224x224 bir görüntü için *N=196*; 1024x1024 bir görüntü için *N=4096*). Bu karesel ölçekleme, ViT'leri yüksek çözünürlüklü girdiler için hesaplama açısından pahalı ve bellek yoğun hale getirir, bu da ince taneli özellik haritaları gerektiren yoğun tahmin görevlerine uygulanabilirliklerini sınırlar.
2.  **Hiyerarşik Özellik Temsili Eksikliği:** Standart ViT'ler çıktıda tek çözünürlüklü bir özellik haritası üretir. Bu durum, uzaysal çözünürlüğü kademeli olarak azaltırken kanal derinliğini artıran (örneğin, havuzlama katmanları aracılığıyla) hiyerarşik özellik temsilleri oluşturan CNN'lerden farklıdır. Hiyerarşik özellikler, farklı boyutlardaki nesneleri algılamak veya karmaşık bölgeleri bölümlere ayırmak için birden çok ölçekte bilgiye ihtiyaç duyulan nesne algılama ve anlamsal segmentasyon gibi görevler için çok önemlidir.
3.  **Sınırlı İndüktif Önyargılar:** CNN'lerin aksine, ViT'ler yerellik ve çevirme değişmezliği gibi doğal indüktif önyargılardan yoksundur. Bu durum onları son derece esnek hale getirse de, aynı zamanda bu temel görsel modelleri öğrenmek için büyük miktarda veri gerektirdikleri anlamına gelir ve bu da onları daha küçük veri kümelerinde CNN'lerden daha az örnek verimli yapar.

Bu sınırlamalar, öz-dikkat mekanizmasının faydalarını korurken bilgisayar görüsünde daha geniş bir benimseme için pratik engelleri aşmayı amaçlayan Swin Transformer'ın geliştirilmesini motive etti.

### 3. Swin Transformer Mimarisi
Swin Transformer, standart Vizyon Trafolarının sınırlamalarını yeni bir hiyerarşik mimari ve daha verimli bir öz-dikkat mekanizması sunarak ele almaktadır. Tasarım prensipleri, Transformer'ların esnekliği ile CNN'lerin verimliliği ve çok ölçekli yetenekleri arasındaki boşluğu kapatmayı amaçlamaktadır.

#### 3.1. Hiyerarşik Özellik Temsili
Swin Transformer'ın temel yeniliklerinden biri, CNN'lerdeki özellik piramitlerine benzer şekilde, **hiyerarşik özellik haritaları** oluşturma yeteneğidir. Bu, her aşamanın yama birleştirme ve Swin Transformer bloklarını içeren çok aşamalı bir tasarımla elde edilir:

*   **Yama Bölümleme:** Başlangıçta, bir giriş RGB görüntüsü (örneğin, 224x224x3) çakışmayan yamalara bölünür. ViT'nin tipik 16x16 yamalarının aksine, Swin Transformer genellikle daha ince taneli bilgiyi korumak için daha küçük yamalarla (örneğin, 4x4) başlar. Bir doğrusal gömme katmanı, bu ham değerli yamaları daha yüksek boyutlu bir özellik uzayına yansıtır. Örneğin, 4x4 bir yama, her biri 3 renk kanalına sahip 16 pikselden oluşur ve 48 boyutlu bir vektör verir, bu daha sonra belirtilen bir kanal boyutu *C*'ye yansıtılır. Bu, `H/4 x W/4 x C` boyutunda bir özellik haritasıyla ilk aşamayı (Aşama 1) oluşturur.
*   **Swin Transformer Blokları:** Her aşamada temsilleri öğrenmek için birden çok Swin Transformer bloğu uygulanır.
*   **Yama Birleştirme (Boyut Azaltma):** Hiyerarşik bir temsil oluşturmak için Swin Transformer, aşamalar arasında bir "yama birleştirme" işlemi gerçekleştirir. Bir yama birleştirme katmanında, bitişik `2x2` yamalar birleştirilir. Örneğin, dört `1x1` uzaysal konumdan (bir `2x2` bölge oluşturan) dört `C` boyutlu özellik birleştirilir ve `4C` boyutlu bir özellik elde edilir. Daha sonra bu `4C` boyutlu özelliği `2C` boyutuna düşürmek için doğrusal bir katman uygulanır. Bu işlem, uzamsal çözünürlüğü etkili bir şekilde yarıya (H/2 ve W/2) düşürür ve özellik boyutunu (2C) iki katına çıkararak yeni, daha kaba taneli bir özellik haritası oluşturur. Bu boyut azaltma mekanizması, CNN'lerdeki havuzlama katmanlarına benzerdir.
*   **Sonraki Aşamalar:** Bu yama birleştirme ve Swin Transformer bloğu uygulamaları sonraki aşamalar (Aşama 2, 3, 4) için tekrarlanır, uzamsal çözünürlüğü aşamalı olarak azaltır ve kanal boyutunu artırır, böylece bir özellik piramidi oluşturur. Bu hiyerarşik yapı, Swin Transformer'ın erken aşamalarda ince taneli detayları ve daha derin aşamalarda üst düzey anlamsal bilgiyi yakalamasına olanak tanır, bu da onu yoğun tahmin görevleri için oldukça uygun hale getirir.

#### 3.2. Kaydırılmış Pencere Mekanizması (SW-MSA)
Doğrusal hesaplama karmaşıklığı elde etmek ve pencereler arası bağlantıları sağlamak için en kritik yenilik, **kaydırılmış pencere tabanlı çok başlı öz-dikkat (SW-MSA)** mekanizmasıdır.

*   **Pencere Tabanlı Çok Başlı Öz-Dikkat (W-MSA):** Küresel öz-dikkat mekanizmasının karesel karmaşıklığını azaltmak için Swin Transformer önce özellik haritasını sabit boyutlu (örneğin, 7x7 piksel) çakışmayan yerel pencerelere böler. Öz-dikkat daha sonra bu yerel pencerelerin her birinde bağımsız olarak hesaplanır. `H x W` boyutunda bir özellik haritası `M x M` boyutunda pencerelere bölünürse, `(H/M) x (W/M)` pencere olacaktır. Hesaplama karmaşıklığı *O(HWC + (H/M)(W/M)M^2C)* haline gelir, bu da *O(HWC + HW * M^2C / M^2)* veya *O(HW * C * M^2)* (eğer `M`, `H,W`'den çok daha küçükse) olarak basitleşir. Bu, küresel dikkatin *O((HW)^2 * C)* karmaşıklığına göre önemli bir azalmayı temsil eder ve `HW` yama sayısına göre doğrusal karmaşıklık elde eder. Ancak, bu pencere tabanlı dikkat, her pencerenin alıcı alanını sınırlar ve farklı pencereler arasındaki bilgi akışını engeller.

*   **Kaydırılmış Pencere Tabanlı Çok Başlı Öz-Dikkat (SW-MSA):** Hesaplama verimliliğini korurken pencereler arası bağlantıları sağlamak için Swin Transformer, kaydırılmış bir pencere bölümleme yaklaşımı sunar. Bu, dönüşümlü Swin Transformer bloklarında uygulanır:
    *   Bir modülün ilk Swin Transformer bloğunda, standart pencere bölümleme (W-MSA) kullanılır.
    *   *Sonraki* Swin Transformer bloğunda, pencere bölümlemesi normal bölümlemeden `(M/2, M/2)` piksel (burada `M` pencere boyutudur) kadar **kaydırılır**. Bu kaydırma, dikkat hesaplamalarının önceki pencerelerin sınırlarını aşmasını sağlar, böylece daha önce izole edilmiş bitişik pencereler arasında bağlantılar kurulur.
    *   Kaydırılmış pencerelerle öz-dikkat hesaplamasını verimli bir şekilde yapmak için, özellikle sınırlardaki düzensiz boyutlu pencereler için, **döngüsel bir kaydırma** ve bir **maskeleme mekanizması** kullanılır. Özellik haritası döngüsel olarak sol üste kaydırılır ve bu da orijinal pencere sınırlarını aşabilecek yeni pencereler oluşturur. Dikkatli bir şekilde tasarlanmış bir dikkat maskesi daha sonra dikkat hesaplamasını yalnızca geçerli, orijinal pencerelerdeki elemanlarla sınırlandırmak için uygulanır ve dikkat hesaplamasının yapay olarak bağlanmış alt pencereler arasında yapılmamasını sağlar. Bu zekice numara, kaydırılmış pencere dikkatini toplu işleme içinde verimli bir şekilde paralel olarak hesaplamaya olanak tanır.

Düzenli ve kaydırılmış pencere bölümlemeleri arasında geçiş yaparak, Swin Transformer, her özellik yamasının bitişik pencerelerdeki yamalarla etkileşime girmesine olanak tanır, böylece etkili alıcı alanı kademeli olarak genişletir ve hesaplama açısından pahalı küresel öz-dikkat mekanizmasına başvurmadan küresel bilgi yayılımını sağlar.

#### 3.3. Göreceli Konum Sapması
Swin Transformer'ın performansına katkıda bulunan bir diğer önemli bileşen, öz-dikkat hesaplamasına **göreceli konum sapması**nın dahil edilmesidir. Standart Transformer öz-dikkat mekanizmasında, sorgu ve anahtar arasındaki benzerlik `Q * K^T` olarak hesaplanır. Swin Transformer bunu öğrenilebilir bir göreceli konum sapması `B` ekleyerek değiştirir:

`Dikkat(Q, K, V) = SoftMax( (Q K^T / sqrt(d)) + B ) V`

Burada `B`, pencere içindeki yamaların göreceli koordinatlarına bağlı olan, öğrenilebilir bir sapma terimidir. Spesifik olarak, bir penceredeki herhangi iki yama için göreceli konumları bir `(delta_x, delta_y)` çifti ile temsil edilebilir. Bir pencere içindeki tüm olası göreceli konumlar için küçük bir `B` matrisi öğrenilir (örneğin, 7x7 bir pencere için `(2M-1)x(2M-1)` olası göreceli konum vardır).

Göreceli konum sapması eklemek, modelin konumsal bilgiyi dahil etmesine yardımcı olur; bu, uzaysal ilişkilerin kritik olduğu görsel görevler için özellikle faydalıdır. ViT'de kullanılan mutlak konumsal gömülerin aksine, göreceli konumsal gömüler, değişen giriş boyutlarına karşı daha sağlamdır ve yerel uzaysal etkileşimleri daha iyi yakalar, Swin Transformer'ın genel doğruluğuna katkıda bulunur.

### 4. Kod Örneği

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Swin Transformer'ın başlangıçtaki yama gömme ve doğrusal projeksiyon katmanı.
    Bir görüntüyü düzleştirilmiş yama dizisine dönüştürür ve bunları gömer.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        # Görüntü boyutlarını ve yama sayısını hesapla
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        # Yama boyutu kadar çekirdek boyutuna ve adıma sahip bir evrişim katmanı
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) örn., (1, 3, 224, 224)
        x = self.proj(x) # (B, embed_dim, H/patch_size, W/patch_size) örn., (1, 96, 56, 56)
        x = x.flatten(2) # (B, embed_dim, num_patches) örn., (1, 96, 3136)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim) örn., (1, 3136, 96)
        return x

class PatchMerging(nn.Module):
    """
    Swin Transformer'ın hiyerarşik özellik azaltımı için yama birleştirme katmanı.
    Uzaysal çözünürlüğü 2 kat azaltır ve özellik boyutunu iki katına çıkarır.
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution # (H, W)
        self.dim = dim # giriş boyutu
        # 2x2 bitişik yamalar birleşir -> 4*dim -> 2*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        # x: (B, H*W, dim) örn., (1, 56*56, 96)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "giriş özelliğinin boyutu yanlış"
        assert H % 2 == 0 and W % 2 == 0, f"x boyutu ({H}*{W}) çift değil."

        x = x.view(B, H, W, C) # (B, H, W, C) örn., (1, 56, 56, 96)

        # 4 alt bölgeye böl ve birleştir
        x0 = x[:, 0::2, 0::2, :]  # Sol üst (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # Sol alt (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # Sağ üst (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # Sağ alt (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)  # Birleştir (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # Düzleştir (B, (H/2)*(W/2), 4*C)

        x = self.norm(x)
        x = self.reduction(x) # Doğrusal projeksiyon (B, (H/2)*(W/2), 2*C)

        return x

# Örnek Kullanım:
if __name__ == '__main__':
    # Başlangıç Yama Gömme
    patch_embed = PatchEmbed(img_size=224, patch_size=4, in_chans=3, embed_dim=96)
    dummy_input = torch.randn(1, 3, 224, 224) # Batch boyutu 1, 3 kanal, 224x224 görüntü
    embedded_patches = patch_embed(dummy_input)
    print(f"Gömülü yamaların şekli: {embedded_patches.shape}") # Beklenen: (1, 3136, 96) -> (B, num_patches, embed_dim)

    # İlk Yama Birleştirme işlemi
    # Giriş çözünürlüğü (Y, G) = (224/4, 224/4) = (56, 56)
    patch_merging = PatchMerging(input_resolution=(56, 56), dim=96)
    merged_patches = patch_merging(embedded_patches)
    print(f"Birleştirilmiş yamaların şekli: {merged_patches.shape}") # Beklenen: (1, 784, 192) -> (B, num_patches/4, 2*embed_dim)

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
Swin Transformer, bilgisayar görü alanında dönüm noktası niteliğinde bir ilerlemeyi temsil etmekte olup, Transformer'ların küresel modelleme yetenekleri ile Evrişimli Sinir Ağlarının indüktif önyargıları ve verimliliği arasındaki boşluğu etkili bir şekilde kapatmaktadır. **Hiyerarşik özellik temsili** ve yenilikçi **kaydırılmış pencere tabanlı öz-dikkat mekanizması**nı tanıtarak, Swin Transformer, standart Vizyon Trafolarının doğasında bulunan karesel hesaplama karmaşıklığını aşar, böylece yüksek çözünürlüklü görüntülere ölçeklenebilir hale gelir ve nesne algılama ve anlamsal segmentasyon gibi yoğun tahmin görevleri için uygun olur.

Özellik piramidi oluşturma yeteneği, karmaşık görsel anlama için kritik olan çok ölçekli bilgiyi yakalamasına olanak tanırken, kaydırılmış pencereler, fahiş hesaplama maliyetleri olmadan yerel bölgeler arasında verimli bilgi alışverişini sağlar. Ayrıca, **göreceli konum sapması**nın dahil edilmesi, modelin uzaysal farkındalığını ve öğrenme kapasitesini artırır.

Bu mimari yeniliklerin bir sonucu olarak, Swin Transformer, geniş bir görsel kıyaslama yelpazesinde son teknoloji performansa ulaşmış ve gelecekteki bilgisayar görü araştırmaları ve uygulamaları için sağlam ve verimli bir omurga olarak kendini kanıtlamıştır. Başarısı, güçlü genel amaçlı mimarileri belirli alanlara uyarlarken dikkatlice tasarlanmış indüktif önyargıların önemini vurgulamaktadır.

### 6. Referanslar
1.  Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Dai, J. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012-10022.
2.  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*.
3.  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All Need. *Advances in neural information processing systems*, 30.







