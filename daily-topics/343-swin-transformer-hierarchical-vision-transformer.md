# Swin Transformer: Hierarchical Vision Transformer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: Limitations of Standard Vision Transformers (ViT)](#2-background-limitations-of-standard-vision-transformers-vit)
- [3. Swin Transformer Architecture](#3-swin-transformer-architecture)
    - [3.1. Patch Partition and Linear Embedding](#31-patch-partition-and-linear-embedding)
    - [3.2. Swin Transformer Blocks](#32-swin-transformer-blocks)
        - [3.2.1. Window-based Multi-head Self-Attention (W-MSA)](#321-window-based-multi-head-self-attention-w-msa)
        - [3.2.2. Shifted Window Multi-head Self-Attention (SW-MSA)](#322-shifted-window-multi-head-self-attention-sw-msa)
    - [3.3. Patch Merging](#33-patch-merging)
    - [3.4. Relative Position Bias](#34-relative-position-bias)
- [4. Key Advantages](#4-key-advantages)
- [5. Applications](#5-applications)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<br>

### 1. Introduction
The **Swin Transformer**, introduced in 2021 by Liu et al., marks a significant advancement in the field of computer vision by extending the success of Transformers from natural language processing (NLP) to image processing. While original Vision Transformers (ViT) demonstrated the potential of self-attention mechanisms for image understanding, they faced considerable challenges, particularly with **high-resolution images** and dense prediction tasks due to their quadratic computational complexity with respect to image size. The Swin Transformer ("Swin" standing for Shifted Windows) addresses these limitations by introducing a **hierarchical architecture** and a novel **shifted window mechanism**. This design allows it to build **hierarchical feature maps** similar to convolutional neural networks (CNNs) and compute self-attention only within local windows, thereby achieving linear computational complexity. Consequently, the Swin Transformer has emerged as a powerful and efficient **general-purpose backbone** for a wide array of vision tasks, including image classification, object detection, and semantic segmentation, often outperforming its CNN and ViT predecessors. Its ability to generate multi-scale representations makes it highly adaptable and effective across various scales of visual information.

### 2. Background: Limitations of Standard Vision Transformers (ViT)
Before the advent of the Swin Transformer, the original **Vision Transformer (ViT)** pioneered the application of standard Transformer architecture directly to images. ViT achieved impressive results on image classification benchmarks by treating images as sequences of fixed-size patches and applying a standard Transformer encoder. However, it suffered from several inherent limitations that hindered its broader adoption, especially for tasks requiring fine-grained spatial understanding:

*   **Quadratic Computational Complexity:** The most significant drawback of ViT is its global self-attention mechanism. Each patch attends to every other patch in the image. For an input image of `H x W` pixels and patches of size `P x P`, the number of patches is `(H*W)/(P*P)`. The computational complexity of self-attention is quadratic with respect to the number of patches, meaning `O((H*W/P^2)^2)`. This becomes prohibitively expensive for high-resolution images, limiting its practical applicability.
*   **Lack of Inductive Biases:** Unlike CNNs, which inherently leverage inductive biases such as locality and translation equivariance through their convolutional filters, standard Transformers are global in nature. This lack of inherent inductive biases means ViT models typically require exceptionally large datasets to learn robust representations, making them data-hungry compared to CNNs.
*   **Fixed-size Tokens:** ViT processes images into a flat sequence of fixed-size tokens, losing the explicit spatial relationships and multi-scale information that are crucial for dense prediction tasks like object detection and semantic segmentation. The absence of a **hierarchical feature representation** prevents it from effectively capturing features at different scales, which is essential for understanding objects of varying sizes or segmenting regions with diverse patterns.
*   **Difficulty with Dense Prediction Tasks:** Due to the fixed-size tokenization and global attention leading to high computational costs, adapting ViT to dense prediction tasks, which require outputting predictions for every pixel or region, proved challenging and inefficient.

These limitations underscored the need for a more efficient and adaptable Transformer architecture for computer vision, paving the way for innovations like the Swin Transformer.

### 3. Swin Transformer Architecture
The Swin Transformer's core innovation lies in its ability to construct a **hierarchical feature map** and compute self-attention with **linear computational complexity** relative to image size. This is achieved through two main mechanisms: **shifted window-based self-attention** and **patch merging layers**. The overall architecture is analogous to a typical CNN, featuring multiple stages that progressively reduce resolution and increase feature dimension.

#### 3.1. Patch Partition and Linear Embedding
The process begins by dividing an input RGB image into non-overlapping patches. For example, a `224x224x3` image might be divided into `4x4` pixel patches, resulting in `(224/4) * (224/4) = 56 * 56 = 3136` patches. Each patch initially has raw pixel values across its channels (e.g., `4x4x3 = 48` dimensions). A **linear embedding layer** then projects these raw feature dimensions to an arbitrary dimension, typically 96, to form the initial set of **token embeddings**. This stage is conceptually similar to the stem of a CNN, where initial features are extracted.

#### 3.2. Swin Transformer Blocks
The backbone of the Swin Transformer is built upon stacked **Swin Transformer Blocks**. Each block consists of two main self-attention modules, followed by a **Layer Normalization (LN)**, a **Multi-Layer Perceptron (MLP)** with **GELU activation**, and residual connections. The crucial aspect is the alternating use of **Window-based Multi-head Self-Attention (W-MSA)** and **Shifted Window Multi-head Self-Attention (SW-MSA)**.

##### 3.2.1. Window-based Multi-head Self-Attention (W-MSA)
Unlike the global self-attention in standard ViTs, W-MSA computes self-attention within local, non-overlapping windows. The input feature map is partitioned into `M x M` non-overlapping windows (e.g., `M=7`). Self-attention is then computed independently within each window. If an image has `H x W` patches, and each window is `M x M` patches, the total number of patches is `H*W`. The number of windows is `(H/M) * (W/M)`. The complexity reduces from `O((H*W)^2)` for global attention to `O(M^2 * H*W)` for window-based attention, assuming `M` is fixed. This significantly reduces computational cost but limits information exchange *between* windows.

##### 3.2.2. Shifted Window Multi-head Self-Attention (SW-MSA)
To overcome the limitation of W-MSA, where information is not exchanged between different windows, the Swin Transformer introduces **Shifted Window Multi-head Self-Attention (SW-MSA)** in consecutive blocks. In this mechanism, the windows are shifted by `(M/2, M/2)` pixels relative to the partition in the previous block. This shift creates new window configurations that cross the boundaries of the previous windows, thereby enabling connections and information flow between adjacent windows from the preceding layer. To maintain the number of windows and allow for efficient batch processing, a **cyclic shifting** approach is employed, where features shifted outside the boundary are wrapped around. This effectively provides **cross-window connectivity** while still adhering to the local self-attention computation, maintaining linear complexity.

#### 3.3. Patch Merging
To create a **hierarchical representation** and reduce the sequence length for deeper layers (mimicking pooling layers in CNNs), the Swin Transformer employs a **patch merging layer** between stages. After a certain number of Swin Transformer Blocks, a patch merging layer is applied. This layer concatenates the features of `2x2` neighboring patches, effectively reducing the number of tokens by a factor of 4. A linear layer is then applied to project the concatenated `4C` dimensional features to `2C` dimensions (where `C` is the current feature dimension), thus increasing the feature channel dimension while reducing spatial resolution. This hierarchical design allows the Swin Transformer to process features at different scales, making it suitable for tasks that require dense predictions.

#### 3.4. Relative Position Bias
To incorporate positional information, Swin Transformers do not use fixed positional embeddings like original Transformers. Instead, they enhance the self-attention mechanism by adding a **relative position bias** to the similarity scores during attention calculation. This bias term is learned and accounts for the relative distances between patches within a window. It helps the model better understand the spatial relationships between different parts of the image and significantly improves performance, especially for visual tasks where relative spatial cues are crucial.

### 4. Key Advantages
The design choices of the Swin Transformer confer several distinct advantages over previous Transformer architectures and even many CNNs:

*   **Hierarchical Feature Representation:** The patch merging layers enable the model to generate feature maps at various scales, akin to feature pyramids in CNNs. This **multi-scale capability** is essential for dense prediction tasks like object detection and semantic segmentation, where objects of different sizes need to be recognized.
*   **Linear Computational Complexity:** By computing self-attention within local windows, the Swin Transformer's computational complexity scales **linearly** with the image size, `O(H*W)`, rather than quadratically `O((H*W)^2)`. This makes it significantly more efficient and applicable to high-resolution images.
*   **Inter-Window Connectivity:** The **shifted window mechanism** cleverly addresses the limitation of local window attention by allowing information to flow between adjacent windows across layers, effectively providing a global receptive field without incurring the high computational cost of global attention.
*   **Flexibility and Generalizability:** The hierarchical structure makes Swin Transformer a versatile **general-purpose backbone** that can be readily adopted for various computer vision tasks. Its ability to extract multi-scale features provides a robust foundation for building advanced vision models.
*   **State-of-the-Art Performance:** Swin Transformers have consistently achieved **state-of-the-art results** on major computer vision benchmarks across image classification, object detection (e.g., COCO), and semantic segmentation (e.g., ADE20K), demonstrating their superior performance and efficiency.

### 5. Applications
The Swin Transformer's robust and efficient architecture has led to its widespread adoption across a multitude of computer vision applications:

*   **Image Classification:** As a strong backbone, it provides highly accurate feature representations for categorizing images into predefined classes. Its hierarchical features help capture both fine-grained and global patterns.
*   **Object Detection:** Swin Transformers are widely used as feature extractors in modern object detection frameworks (e.g., Faster R-CNN, Mask R-CNN, Cascade R-CNN, DETR variants). Their multi-scale feature maps are particularly beneficial for detecting objects of varying sizes within an image, outperforming many CNN-based backbones.
*   **Semantic Segmentation:** For tasks requiring pixel-level understanding, such as semantic segmentation (assigning a class label to each pixel), Swin Transformers excel. The hierarchical representation allows the model to capture context at different scales, crucial for accurately segmenting intricate regions.
*   **Instance Segmentation:** Similar to object detection, Swin Transformers provide powerful features for instance segmentation, which involves detecting and segmenting each individual object instance in an image.
*   **Video Understanding:** Extensions of Swin Transformer, such as Video Swin Transformer, have been developed for spatio-temporal modeling in video analysis tasks like action recognition and video object detection.
*   **Medical Imaging:** Its ability to handle high-resolution images and extract hierarchical features makes it a promising candidate for various medical imaging applications, including disease detection, segmentation of anatomical structures, and diagnosis assistance.

### 6. Code Example
Here is a simplified Python code snippet illustrating the **Patch Embedding** layer, which is the initial step in a Swin Transformer, converting an input image into a sequence of embedded patches.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Swin Transformer's Patch Embedding Layer.
    Converts an input image into a sequence of embedded patches.
    It performs patch partitioning and linear projection.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        # Calculate number of patches across height and width
        num_patches_h = img_size[0] // patch_size[0]
        num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = num_patches_h * num_patches_w

        # Store patch and image sizes for validation
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Convolutional layer to extract patches and project them to embed_dim
        # Kernel size and stride are equal to patch_size for non-overlapping patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Layer normalization after projection for stable training
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) e.g., (Batch, Channels, Height, Width)
        B, C, H, W = x.shape
        # Assert input image size matches the expected size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model's expected size ({self.img_size[0]}*{self.img_size[1]})."

        # Apply convolution for patch extraction and linear projection
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # Flatten the spatial dimensions to get a sequence of patches
        # (B, embed_dim, num_patches_h, num_patches_w) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2) # Permute to (Batch, num_patches, embed_dim)
        # Apply layer normalization to the embeddings
        x = self.norm(x)
        return x

# Example Usage:
# Assuming an input image of 224x224 pixels with 3 channels
# img_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224 image
# patch_size_val = (4, 4) # Each patch is 4x4 pixels
# embed_dim_val = 96 # Projection dimension for each patch embedding

# Initialize the PatchEmbedding layer
# patch_embed_layer = PatchEmbedding(img_size=(224, 224), patch_size=patch_size_val, in_channels=3, embed_dim=embed_dim_val)

# Pass the image through the patch embedding layer
# output_patches = patch_embed_layer(img_input)

# Expected output shape: (Batch_size, Number_of_patches, Embedding_dimension)
# Number of patches = (224/4) * (224/4) = 56 * 56 = 3136
# print(f"Output shape after Patch Embedding: {output_patches.shape}") # Expected: torch.Size([1, 3136, 96])

(End of code example section)
```

### 7. Conclusion
The Swin Transformer represents a pivotal development in computer vision, effectively bridging the gap between the global modeling capabilities of Transformers and the inductive biases (like locality and hierarchy) traditionally found in CNNs. By introducing a **hierarchical architecture** and the ingenious **shifted window mechanism**, Swin Transformers overcome the computational and architectural limitations of earlier Vision Transformers, achieving **linear computational complexity** and generating **multi-scale feature representations**. This design enables them to serve as a highly efficient and powerful **general-purpose backbone** for a wide range of visual tasks, from high-level image classification to intricate dense prediction tasks like object detection and semantic segmentation. Its superior performance and adaptability have solidified its position as a foundational model in modern computer vision research and applications, paving the way for further innovations in efficient and effective visual understanding.

---
<br>

<a name="türkçe-içerik"></a>
## Swin Transformer: Hiyerarşik Vizyon Transformeri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: Standart Vizyon Transformer'larının (ViT) Sınırlamaları](#2-arka-plan-standart-vizyon-transformerlarinin-vit-sinirlamalari)
- [3. Swin Transformer Mimarisi](#3-swin-transformer-mimarisi)
    - [3.1. Yama Bölütleme ve Doğrusal Gömme](#31-yama-bolutleme-ve-dogrusal-gomme)
    - [3.2. Swin Transformer Blokları](#32-swin-transformer-bloklari)
        - [3.2.1. Pencere Tabanlı Çok Kafalı Öz-Dikkat (W-MSA)](#321-pencere-tabanli-cok-kafali-oz-dikkat-w-msa)
        - [3.2.2. Kaydırılmış Pencere Çok Kafalı Öz-Dikkat (SW-MSA)](#322-kaydirilmis-pencere-cok-kafali-oz-dikkat-sw-msa)
    - [3.3. Yama Birleştirme](#33-yama-birlestirme)
    - [3.4. Göreceli Konum Önyargısı](#34-goreceli-konum-onyargisi)
- [4. Temel Avantajlar](#4-temel-avantajlar)
- [5. Uygulamalar](#5-uygulamalar)
- [6. Kod Örneği](#6-kod-ornegi)
- [7. Sonuç](#7-sonuc)

<br>

### 1. Giriş
2021 yılında Liu ve arkadaşları tarafından tanıtılan **Swin Transformer**, Doğal Dil İşleme'den (NLP) görüntü işlemeye Transformer'ların başarısını genişleterek bilgisayar görüşü alanında önemli bir ilerlemeyi temsil etmektedir. Orijinal Vizyon Transformer'ları (ViT) görüntü anlama için öz-dikkat mekanizmalarının potansiyelini göstermiş olsa da, özellikle **yüksek çözünürlüklü görüntüler** ve yoğun tahmin görevleri için görüntü boyutuna göre karesel hesaplama karmaşıklığı nedeniyle önemli zorluklarla karşılaştılar. Swin Transformer ("Swin" Shifted Windows'un kısaltmasıdır) bu sınırlamaları **hiyerarşik bir mimari** ve yeni bir **kaydırılmış pencere mekanizması** sunarak ele alır. Bu tasarım, Evrişimsel Sinir Ağlarına (CNN'ler) benzer şekilde **hiyerarşik özellik haritaları** oluşturmasına ve öz-dikkati yalnızca yerel pencereler içinde hesaplamasına olanak tanıyarak doğrusal hesaplama karmaşıklığı elde eder. Sonuç olarak, Swin Transformer, görüntü sınıflandırma, nesne tespiti ve anlamsal bölütleme dahil olmak üzere çok çeşitli vizyon görevleri için güçlü ve verimli bir **genel amaçlı omurga** olarak ortaya çıkmış ve çoğu zaman CNN ve ViT öncüllerini geride bırakmıştır. Çok ölçekli temsiller oluşturma yeteneği, görsel bilginin çeşitli ölçeklerinde son derece uyarlanabilir ve etkili olmasını sağlar.

### 2. Arka Plan: Standart Vizyon Transformer'larının (ViT) Sınırlamaları
Swin Transformer'ın ortaya çıkışından önce, orijinal **Vizyon Transformer'ı (ViT)**, standart Transformer mimarisinin doğrudan görüntülere uygulanmasına öncülük etti. ViT, görüntüleri sabit boyutlu yamaların dizileri olarak ele alarak ve standart bir Transformer kodlayıcı uygulayarak görüntü sınıflandırma kıyaslama testlerinde etkileyici sonuçlar elde etti. Ancak, özellikle ince taneli uzamsal anlayış gerektiren görevler için daha geniş çaplı benimsenmesini engelleyen bazı doğal sınırlamalara sahipti:

*   **Karesel Hesaplama Karmaşıklığı:** ViT'nin en önemli dezavantajı, küresel öz-dikkat mekanizmasıdır. Her yama, görüntüdeki diğer her yamaya dikkat eder. `H x W` piksellik bir giriş görüntüsü ve `P x P` boyutundaki yamalar için, yama sayısı `(H*W)/(P*P)`'dir. Öz-dikkatin hesaplama karmaşıklığı, yama sayısına göre kareseldir, yani `O((H*W/P^2)^2)`'dir. Bu, yüksek çözünürlüklü görüntüler için aşırı derecede pahalı hale gelir ve pratik uygulanabilirliğini sınırlar.
*   **Endüktif Önyargı Eksikliği:** Evrişimsel filtreleri aracılığıyla yerellik ve öteleme eşdeğerliği gibi endüktif önyargıları doğal olarak kullanan CNN'lerin aksine, standart Transformer'lar doğaları gereği küreseldir. İçsel endüktif önyargıların bu eksikliği, ViT modellerinin sağlam temsiller öğrenmek için genellikle son derece büyük veri kümelerine ihtiyaç duyduğu anlamına gelir, bu da onları CNN'lere kıyasla veri-aç yapar.
*   **Sabit Boyutlu Token'lar:** ViT, görüntüleri sabit boyutlu token'ların düz bir dizisine dönüştürür, bu da nesne tespiti ve anlamsal bölütleme gibi yoğun tahmin görevleri için kritik olan açık uzamsal ilişkileri ve çok ölçekli bilgileri kaybeder. **Hiyerarşik bir özellik temsilinin** olmaması, farklı boyutlardaki nesneleri anlamak veya çeşitli desenlere sahip bölgeleri bölütlemek için gerekli olan farklı ölçeklerdeki özellikleri etkin bir şekilde yakalamasını engeller.
*   **Yoğun Tahmin Görevlerinde Zorluk:** Sabit boyutlu token'laştırma ve yüksek hesaplama maliyetine yol açan küresel dikkat nedeniyle, ViT'yi her piksel veya bölge için tahminler çıkarmayı gerektiren yoğun tahmin görevlerine uyarlamak zorlu ve verimsiz olmuştur.

Bu sınırlamalar, bilgisayar görüşü için daha verimli ve uyarlanabilir bir Transformer mimarisine duyulan ihtiyacın altını çizdi ve Swin Transformer gibi yeniliklerin önünü açtı.

### 3. Swin Transformer Mimarisi
Swin Transformer'ın temel yeniliği, **hiyerarşik bir özellik haritası** oluşturma ve görüntü boyutuna göre **doğrusal hesaplama karmaşıklığına** sahip öz-dikkat hesaplama yeteneğinde yatmaktadır. Bu, iki ana mekanizma aracılığıyla elde edilir: **kaydırılmış pencere tabanlı öz-dikkat** ve **yama birleştirme katmanları**. Genel mimari, çözünürlüğü kademeli olarak azaltan ve özellik boyutunu artıran birden çok aşamaya sahip tipik bir CNN'e benzer.

#### 3.1. Yama Bölütleme ve Doğrusal Gömme
Süreç, bir giriş RGB görüntüsünü çakışmayan yamalara bölmekle başlar. Örneğin, `224x224x3` boyutundaki bir görüntü `4x4` piksellik yamalara bölünebilir, bu da `(224/4) * (224/4) = 56 * 56 = 3136` yama ile sonuçlanır. Her yama başlangıçta kanalları boyunca ham piksel değerlerine sahiptir (örn. `4x4x3 = 48` boyut). Bir **doğrusal gömme katmanı** daha sonra bu ham özellik boyutlarını keyfi bir boyuta, genellikle 96'ya yansıtarak ilk **token gömmelerini** oluşturur. Bu aşama, bir CNN'in başlangıç kısmına (stem) kavramsal olarak benzer, burada ilk özellikler çıkarılır.

#### 3.2. Swin Transformer Blokları
Swin Transformer'ın omurgası, yığılmış **Swin Transformer Blokları** üzerine inşa edilmiştir. Her blok, iki ana öz-dikkat modülünden, ardından bir **Katman Normalizasyonu (LN)**, bir **Çok Katmanlı Algılayıcı (MLP)** ile **GELU aktivasyonu** ve artalan bağlantılarından oluşur. Önemli nokta, **Pencere Tabanlı Çok Kafalı Öz-Dikkat (W-MSA)** ve **Kaydırılmış Pencere Çok Kafalı Öz-Dikkat (SW-MSA)**'nın dönüşümlü olarak kullanılmasıdır.

##### 3.2.1. Pencere Tabanlı Çok Kafalı Öz-Dikkat (W-MSA)
Standart ViT'lerdeki küresel öz-dikkatten farklı olarak, W-MSA öz-dikkati yerel, çakışmayan pencereler içinde hesaplar. Giriş özellik haritası `M x M` çakışmayan pencerelere bölünür (örn. `M=7`). Öz-dikkat daha sonra her pencere içinde bağımsız olarak hesaplanır. Bir görüntü `H x W` yamaya sahipse ve her pencere `M x M` yama ise, toplam yama sayısı `H*W` olur. Pencere sayısı `(H/M) * (W/M)`'dir. Karmaşıklık, küresel dikkat için `O((H*W)^2)`'den, `M`'nin sabit olduğu varsayılarak pencere tabanlı dikkat için `O(M^2 * H*W)`'ye düşer. Bu, hesaplama maliyetini önemli ölçüde azaltır ancak pencereler *arasındaki* bilgi alışverişini sınırlar.

##### 3.2.2. Kaydırılmış Pencere Çok Kafalı Öz-Dikkat (SW-MSA)
W-MSA'nın farklı pencereler arasında bilgi alışverişi yapılmaması sınırlamasının üstesinden gelmek için, Swin Transformer ardışık bloklarda **Kaydırılmış Pencere Çok Kafalı Öz-Dikkat (SW-MSA)**'yı tanıtır. Bu mekanizmada, pencereler önceki bloktaki bölümlemeye göre `(M/2, M/2)` piksel kaydırılır. Bu kaydırma, önceki pencerelerin sınırlarını aşan yeni pencere yapılandırmaları oluşturur ve böylece önceki katmandaki bitişik pencereler arasında bağlantı ve bilgi akışı sağlar. Pencere sayısını korumak ve verimli toplu işlemeye izin vermek için, sınırlar dışına kaydırılan özelliklerin etrafına sarıldığı **döngüsel kaydırma** yaklaşımı kullanılır. Bu, yerel öz-dikkat hesaplamasına bağlı kalarak ve doğrusal karmaşıklığı koruyarak etkili bir şekilde **pencereler arası bağlantı** sağlar.

#### 3.3. Yama Birleştirme
**Hiyerarşik bir temsil** oluşturmak ve daha derin katmanlar için dizi uzunluğunu azaltmak (CNN'lerdeki havuzlama katmanlarını taklit ederek) için Swin Transformer, aşamalar arasında bir **yama birleştirme katmanı** kullanır. Belirli sayıda Swin Transformer Bloğundan sonra bir yama birleştirme katmanı uygulanır. Bu katman, `2x2` komşu yamaların özelliklerini birleştirerek token sayısını 4 faktörüne düşürür. Daha sonra, birleştirilmiş `4C` boyutundaki özelliklerini `2C` boyutuna yansıtmak için (burada `C` mevcut özellik boyutudur) doğrusal bir katman uygulanır, böylece uzamsal çözünürlük azalırken özellik kanal boyutu artırılır. Bu hiyerarşik tasarım, Swin Transformer'ın farklı ölçeklerdeki özellikleri işlemesine olanak tanır ve yoğun tahmin gerektiren görevler için uygun hale getirir.

#### 3.4. Göreceli Konum Önyargısı
Konumsal bilgiyi dahil etmek için, Swin Transformer'lar orijinal Transformer'lar gibi sabit konumsal gömmeler kullanmazlar. Bunun yerine, dikkat hesaplaması sırasında benzerlik skorlarına bir **göreceli konum önyargısı** ekleyerek öz-dikkat mekanizmasını geliştirirler. Bu önyargı terimi öğrenilir ve bir pencere içindeki yamalar arasındaki göreceli mesafeleri hesaba katar. Özellikle göreceli uzamsal ipuçlarının çok önemli olduğu görsel görevler için modelin görüntünün farklı bölümleri arasındaki uzamsal ilişkileri daha iyi anlamasına yardımcı olur ve performansı önemli ölçüde artırır.

### 4. Temel Avantajlar
Swin Transformer'ın tasarım seçimleri, önceki Transformer mimarilerine ve hatta birçok CNN'e göre çeşitli belirgin avantajlar sağlar:

*   **Hiyerarşik Özellik Temsili:** Yama birleştirme katmanları, modelin CNN'lerdeki özellik piramitlerine benzer şekilde çeşitli ölçeklerde özellik haritaları oluşturmasına olanak tanır. Bu **çok ölçekli yetenek**, farklı boyutlardaki nesnelerin tanınması gereken nesne tespiti ve anlamsal bölütleme gibi yoğun tahmin görevleri için çok önemlidir.
*   **Doğrusal Hesaplama Karmaşıklığı:** Öz-dikkati yerel pencereler içinde hesaplayarak, Swin Transformer'ın hesaplama karmaşıklığı görüntü boyutuyla **doğrusal** olarak `O(H*W)` ölçeklenir, karesel `O((H*W)^2)` olarak değil. Bu, onu önemli ölçüde daha verimli ve yüksek çözünürlüklü görüntüler için uygulanabilir kılar.
*   **Pencereler Arası Bağlantı:** **Kaydırılmış pencere mekanizması**, yerel pencere dikkat mekanizmasının sınırlamasını, bilgi akışının katmanlar arasında bitişik pencereler arasında akmasına izin vererek akıllıca giderir, böylece küresel dikkatin yüksek hesaplama maliyetine katlanmadan etkili bir küresel algı alanı sağlar.
*   **Esneklik ve Genellenebilirlik:** Hiyerarşik yapı, Swin Transformer'ı çeşitli bilgisayar görüşü görevleri için kolayca benimsenebilen çok yönlü bir **genel amaçlı omurga** haline getirir. Çok ölçekli özellikler çıkarma yeteneği, gelişmiş vizyon modelleri oluşturmak için sağlam bir temel sağlar.
*   **En Güncel Performans:** Swin Transformer'lar, görüntü sınıflandırma, nesne tespiti (örn. COCO) ve anlamsal bölütleme (örn. ADE20K) dahil olmak üzere büyük bilgisayar görüşü kıyaslama testlerinde sürekli olarak **en güncel sonuçları** elde etmiş, üstün performanslarını ve verimliliklerini kanıtlamışlardır.

### 5. Uygulamalar
Swin Transformer'ın sağlam ve verimli mimarisi, çok sayıda bilgisayar görüşü uygulamasında yaygın olarak benimsenmesine yol açmıştır:

*   **Görüntü Sınıflandırma:** Güçlü bir omurga olarak, görüntüleri önceden tanımlanmış sınıflara ayırmak için son derece doğru özellik temsilleri sağlar. Hiyerarşik özellikleri, hem ince taneli hem de küresel desenleri yakalamaya yardımcı olur.
*   **Nesne Tespiti:** Swin Transformer'lar, modern nesne tespiti çerçevelerinde (örn. Faster R-CNN, Mask R-CNN, Cascade R-CNN, DETR varyantları) özellik çıkarıcı olarak yaygın olarak kullanılmaktadır. Çok ölçekli özellik haritaları, özellikle bir görüntüdeki farklı boyutlardaki nesneleri tespit etmek için faydalıdır ve birçok CNN tabanlı omurgayı geride bırakır.
*   **Anlamsal Bölütleme:** Her piksele bir sınıf etiketi atama gibi piksel düzeyinde anlama gerektiren görevler (anlamsal bölütleme) için Swin Transformer'lar üstündür. Hiyerarşik temsil, modelin farklı ölçeklerde bağlam yakalamasına olanak tanır, bu da karmaşık bölgeleri doğru bir şekilde bölütlemek için kritik öneme sahiptir.
*   **Örnek Bölütleme:** Nesne tespitine benzer şekilde, Swin Transformer'lar, bir görüntüdeki her bir nesne örneğini tespit etmeyi ve bölütlemeyi içeren örnek bölütleme için güçlü özellikler sağlar.
*   **Video Anlama:** Swin Transformer'ın Video Swin Transformer gibi uzantıları, eylem tanıma ve video nesne tespiti gibi video analizi görevlerinde uzamsal-zamansal modelleme için geliştirilmiştir.
*   **Tıbbi Görüntüleme:** Yüksek çözünürlüklü görüntüleri işleme ve hiyerarşik özellikler çıkarma yeteneği, hastalık tespiti, anatomik yapıların bölütlenmesi ve teşhis yardımı dahil olmak üzere çeşitli tıbbi görüntüleme uygulamaları için umut vadeden bir aday yapar.

### 6. Kod Örneği
İşte bir Swin Transformer'ın ilk adımı olan **Yama Gömme (Patch Embedding)** katmanını gösteren basitleştirilmiş bir Python kod parçacığı; bu katman, bir giriş görüntüsünü gömülü yamalar dizisine dönüştürür.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Swin Transformer'ın Yama Gömme Katmanı.
    Giriş görüntüsünü gömülü yamalar dizisine dönüştürür.
    Yama bölütleme ve doğrusal yansıtma işlemlerini gerçekleştirir.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        # Yükseklik ve genişlik boyunca yama sayısını hesapla
        num_patches_h = img_size[0] // patch_size[0]
        num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = num_patches_h * num_patches_w

        # Doğrulama için yama ve görüntü boyutlarını sakla
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Yamaları çıkarmak ve embed_dim boyutuna yansıtmak için evrişim katmanı
        # Çekirdek boyutu ve adım, çakışmayan yamalar için patch_size'a eşittir
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Yansıtmadan sonra kararlı eğitim için katman normalizasyonu
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) örn., (Parti, Kanal, Yükseklik, Genişlik)
        B, C, H, W = x.shape
        # Giriş görüntü boyutunun beklenen boyutla eşleştiğini doğrula
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Giriş görüntü boyutu ({H}*{W}), modelin beklenen boyutu ({self.img_size[0]}*{self.img_size[1]}) ile eşleşmiyor."

        # Yama çıkarma ve doğrusal yansıtma için evrişim uygulayın
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # Mekansal boyutları düzleştirerek bir yama dizisi elde edin
        # (B, embed_dim, num_patches_h, num_patches_w) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2) # (Parti, num_patches, embed_dim) şekline dönüştür
        # Gömmelere katman normalizasyonu uygulayın
        x = self.norm(x)
        return x

# Örnek Kullanım:
# 3 kanallı 224x224 piksel boyutunda bir giriş görüntüsü varsayalım
# img_input = torch.randn(1, 3, 224, 224) # Parti boyutu 1, 3 kanal, 224x224 görüntü
# patch_size_val = (4, 4) # Her yama 4x4 pikseldir
# embed_dim_val = 96 # Her yama gömme için yansıtma boyutu

# PatchEmbedding katmanını başlat
# patch_embed_layer = PatchEmbedding(img_size=(224, 224), patch_size=patch_size_val, in_channels=3, embed_dim=embed_dim_val)

# Görüntüyü yama gömme katmanından geçir
# output_patches = patch_embed_layer(img_input)

# Beklenen çıkış şekli: (Parti_boyutu, Yama_sayısı, Gömme_boyutu)
# Yama sayısı = (224/4) * (224/4) = 56 * 56 = 3136
# print(f"Yama Gömme sonrası çıkış şekli: {output_patches.shape}") # Beklenen: torch.Size([1, 3136, 96])

(Kod örneği bölümünün sonu)
```

### 7. Sonuç
Swin Transformer, Transformer'ların küresel modelleme yetenekleri ile geleneksel olarak CNN'lerde bulunan endüktif önyargılar (yerellik ve hiyerarşi gibi) arasındaki boşluğu etkili bir şekilde kapatarak bilgisayar görüşünde önemli bir gelişmeyi temsil etmektedir. **Hiyerarşik bir mimari** ve dahice **kaydırılmış pencere mekanizması** sunarak, Swin Transformer'lar önceki Vizyon Transformer'larının hesaplama ve mimari sınırlamalarını aşarak **doğrusal hesaplama karmaşıklığı** elde eder ve **çok ölçekli özellik temsilleri** üretir. Bu tasarım, onları yüksek seviyeli görüntü sınıflandırmasından nesne tespiti ve anlamsal bölütleme gibi karmaşık yoğun tahmin görevlerine kadar geniş bir görsel görev yelpazesi için son derece verimli ve güçlü bir **genel amaçlı omurga** olarak hizmet etmelerini sağlar. Üstün performansı ve uyarlanabilirliği, modern bilgisayar görüşü araştırma ve uygulamalarında temel bir model olarak konumunu sağlamlaştırmış, verimli ve etkili görsel anlama alanında daha fazla yeniliğin önünü açmıştır.