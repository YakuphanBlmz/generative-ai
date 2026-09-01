# Vision Transformers (ViT): An Image is Worth 16x16 Words

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Genesis of ViT: Beyond CNNs](#2-the-genesis-of-vit-beyond-cnns)
- [3. Architecture of Vision Transformers](#3-architecture-of-vision-transformers)
  - [3.1. Patch Embedding](#31-patch-embedding)
  - [3.2. Positional Encoding](#32-positional-encoding)
  - [3.3. Transformer Encoder](#33-transformer-encoder)
  - [3.4. MLP Head](#34-mlp-head)
- [4. Training and Performance](#4-training-and-performance)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)
- [7. Future Directions and Limitations](#7-future-directions-and-limitations)

<a name="1-introduction"></a>
### 1. Introduction
The realm of computer vision has witnessed a profound transformation with the advent of **deep learning** models, predominantly **Convolutional Neural Networks (CNNs)**. For nearly a decade, CNNs reigned supreme, demonstrating unparalleled performance in tasks such as image classification, object detection, and semantic segmentation. Their hierarchical structure, local receptive fields, and translation invariance were considered inherently suitable for processing visual data. However, a parallel revolution was unfolding in **Natural Language Processing (NLP)** with the introduction of the **Transformer** architecture by Vaswani et al. in 2017. Transformers, initially designed for sequence-to-sequence tasks, leveraged the concept of **self-attention** to model long-range dependencies efficiently, moving away from recurrent or convolutional layers.

The groundbreaking paper "An Image Is Worth 16x16 Words: Transformers for Image Recognition At Scale" by Dosovitskiy et al. (2020) challenged the established paradigm by demonstrating that a pure Transformer applied directly to sequences of image patches could achieve state-of-the-art results in image classification tasks. This marked the birth of **Vision Transformers (ViT)**, effectively bridging the gap between NLP and computer vision and opening new avenues for research. The core insight of ViT is to treat an image not as a 2D grid of pixels, but as a sequence of flattened 2D patches, analogous to words in a sentence, which can then be fed into a standard Transformer encoder. This document delves into the architectural nuances, operational principles, advantages, and implications of Vision Transformers, exploring how they have reshaped our understanding of visual representation learning.

<a name="2-the-genesis-of-vit-beyond-cnns"></a>
### 2. The Genesis of ViT: Beyond CNNs
Before ViT, the dominant paradigm in computer vision was undeniably centered around **Convolutional Neural Networks (CNNs)**. Architectures like AlexNet, VGG, ResNet, and InceptionNet pushed the boundaries of accuracy on large-scale datasets such as ImageNet. CNNs excel due to their inherent inductive biases: **locality** (processing small neighborhoods of pixels) and **translation equivariance** (detecting features regardless of their position). These biases made CNNs highly effective and data-efficient for image processing, as they didn't need to learn these properties from scratch.

However, the Transformer architecture, with its **self-attention mechanism**, offered a fundamentally different approach. In NLP, self-attention allowed models to weigh the importance of different parts of the input sequence when processing a specific element, capturing global dependencies effectively. This global view stood in contrast to the local focus of convolutions. The success of Transformers in NLP tasks, particularly with large pre-training datasets, prompted researchers to investigate their applicability to other domains.

The initial attempts to adapt Transformers for vision often involved integrating attention mechanisms into CNN architectures or using self-attention on local windows, rather than processing the entire image globally. ViT's seminal contribution was its audacious simplicity: it demonstrated that a *pure* Transformer, without any convolutional layers, could perform competitively. This radical departure suggested that the inductive biases of CNNs, while beneficial, might not be strictly necessary, especially when abundant data is available for the model to learn these representations itself. The crucial conceptual leap was reinterpreting image data into a sequence format amenable to Transformer processing, thereby challenging the long-held assumption that convolutions were indispensable for visual understanding. The motivation was to leverage the Transformer's ability to model global relationships and its excellent scalability with increasing data and model size, a trait well-demonstrated in large NLP models.

<a name="3-architecture-of-vision-transformers"></a>
### 3. Architecture of Vision Transformers
The core idea behind ViT is to process an image as a sequence of fixed-size patches, similar to how a Transformer processes a sequence of words. This conversion allows the direct application of the standard Transformer encoder. The architecture can be broken down into several key components: **Patch Embedding**, **Positional Encoding**, the **Transformer Encoder** itself, and a final **MLP Head** for classification.

#### 3.1. Patch Embedding
The first step in a ViT pipeline is to convert the input image into a sequence of flat, linear embeddings. Given an input image `H x W x C` (Height x Width x Channels), the image is divided into a grid of non-overlapping square patches of a fixed size, say `P x P`. Each patch will therefore have dimensions `P x P x C`.
For example, if an image is `224x224x3` and the patch size is `16x16`, then the image will be divided into `(224/16) x (224/16) = 14 x 14 = 196` patches.
Each of these `P x P x C` patches is then flattened into a 1D vector of size `P*P*C`. These flattened vectors are then projected into a higher-dimensional embedding space (e.g., `D` dimensions, where `D` is the latent vector size of the Transformer) using a linear projection layer. This process transforms each patch into a **patch embedding**.
In addition to these patch embeddings, a special learnable **"[CLS]" token** (similar to the one used in BERT) is prepended to the sequence of patch embeddings. This CLS token serves as the global representation of the image, and its corresponding output at the Transformer encoder's final layer is used for downstream classification.

<a name="32-positional-encoding"></a>
#### 3.2. Positional Encoding
Transformers are inherently permutation-invariant, meaning they do not inherently understand the relative or absolute positions of tokens in a sequence. However, spatial information is crucial for images. To reintroduce this spatial context, ViT employs **learnable 1D positional embeddings**. These embeddings are simply added to the patch embeddings *before* they are fed into the Transformer encoder. Each patch embedding is augmented with a positional embedding that corresponds to its original position in the image grid. The CLS token also receives its own dedicated positional embedding. These embeddings are learned during the training process, allowing the model to infer the spatial relationships between patches.

<a name="33-transformer-encoder"></a>
#### 3.3. Transformer Encoder
The core of the ViT model is the standard Transformer encoder, identical to the one used in the original Transformer architecture for NLP. It consists of a stack of identical layers, each containing two primary sub-layers:
1.  **Multi-Head Self-Attention (MHSA):** This mechanism allows the model to weigh the importance of different patches (tokens) when processing each individual patch. For each patch, the MHSA mechanism computes a weighted sum of all other patch embeddings, where the weights are derived from their pairwise similarity. This enables the model to capture global contextual information across the entire image.
2.  **Multilayer Perceptron (MLP):** A simple feed-forward network applied independently to each position (patch embedding) after the MHSA. This MLP typically consists of two linear layers with a GELU activation function in between.
Each of these sub-layers is followed by a **Layer Normalization** step, and a **residual connection** is applied around each of them, facilitating stable training of deep networks. The output of the Transformer encoder is a sequence of contextualized embeddings, one for each input patch and the CLS token.

<a name="34-mlp-head"></a>
#### 3.4. MLP Head
Finally, for image classification tasks, the output embedding corresponding to the special **CLS token** from the last layer of the Transformer encoder is taken. This single vector is considered the aggregate representation of the entire image. It is then passed through a simple **Multilayer Perceptron (MLP) head**, which typically consists of one or two linear layers and a non-linear activation function, followed by a final linear layer that projects the representation into the desired number of output classes. A Softmax function is then applied to these logits to obtain class probabilities.

<a name="4-training-and-performance"></a>
### 4. Training and Performance
Vision Transformers typically require substantial amounts of data for effective training, a characteristic inherited from their NLP counterparts. Unlike CNNs, which benefit from strong inductive biases (locality, translation equivariance), ViTs are more "data-hungry." They learn these spatial hierarchies and relationships directly from the data through the self-attention mechanism.
The original ViT paper demonstrated that training on large-scale datasets, such as **JFT-300M** (a private dataset with 300 million images), was crucial for ViT to outperform state-of-the-art CNNs. When trained on smaller datasets like ImageNet, ViTs often performed worse than equivalent CNNs without additional regularization or specific training strategies. This highlights a key trade-off: **ViTs sacrifice some inductive bias for increased model capacity and flexibility**, allowing them to potentially learn more powerful and generalized representations given sufficient data.

To mitigate the data requirement, several strategies have emerged:
1.  **Transfer Learning:** Pre-training ViTs on very large datasets (e.g., JFT, ImageNet-21k) and then fine-tuning them on smaller, task-specific datasets has proven to be highly effective. This approach leverages the knowledge learned from extensive general data.
2.  **Knowledge Distillation:** Using a pre-trained CNN as a teacher model to guide the training of a ViT student model can help inject CNN's inductive biases and improve performance on smaller datasets.
3.  **Data Augmentation:** Employing aggressive data augmentation techniques (e.g., RandAugment, Mixup, Cutmix) during training helps to increase the effective size and diversity of the training data.

When adequately trained, ViTs have shown impressive performance, matching or even surpassing the accuracy of CNNs on various benchmarks, particularly in large-scale image classification. Their ability to model global relationships across an image, combined with their scalability, makes them a compelling alternative to traditional convolutional approaches. Furthermore, the **attention maps** generated by the self-attention mechanism can often be visualized, offering insights into which parts of the image the model focuses on for its predictions, providing a level of interpretability.

<a name="5-code-example"></a>
### 5. Code Example
This Python snippet illustrates the basic idea of how an image is conceptually divided into patches and flattened, suitable for Vision Transformers. It uses PyTorch for tensors.

```python
import torch
import torch.nn as nn

def img_to_patches(image, patch_size):
    """
    Converts an image tensor into a sequence of flattened patches.

    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        patch_size (int): Size of the square patch (P).

    Returns:
        torch.Tensor: Tensor of flattened patches, shape (B, N, P*P*C),
                      where N is the number of patches.
    """
    B, C, H, W = image.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by the patch size.")

    # Reshape image into (B, C, H/P, P, W/P, P)
    # Then permute to (B, H/P, W/P, P, P, C)
    # Then flatten (P, P, C) into (P*P*C)
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # (B, H/P, W/P, C, P, P) -> (B, H/P, W/P, P, P, C)
    
    # Flatten patches to (B, N, P*P*C)
    patches = patches.view(B, -1, patch_size * patch_size * C)
    return patches

# Example usage:
batch_size = 1
channels = 3
height = 224
width = 224
patch_size = 16

# Create a dummy image tensor
dummy_image = torch.randn(batch_size, channels, height, width)
print(f"Original image shape: {dummy_image.shape}")

# Convert image to patches
image_patches = img_to_patches(dummy_image, patch_size)
print(f"Shape of flattened patches: {image_patches.shape}")

# Expected output for image_patches.shape: (1, (224/16)*(224/16), 16*16*3) = (1, 196, 768)

(End of code example section)
```

<a name="6-conclusion"></a>
### 6. Conclusion
The introduction of Vision Transformers (ViT) represents a pivotal moment in the history of computer vision, challenging the long-standing dominance of Convolutional Neural Networks (CNNs). By successfully adapting the Transformer architecture from Natural Language Processing (NLP) to image tasks, ViT demonstrated that the inductive biases inherent in convolutions are not strictly indispensable, particularly when equipped with sufficient training data. The core innovation of ViT lies in its ability to treat images as sequences of flattened patches, allowing the powerful self-attention mechanism to capture global dependencies across the entire image.

While ViTs initially required massive datasets for superior performance, subsequent research and improvements in training strategies (such as transfer learning, distillation, and aggressive data augmentation) have made them more accessible and competitive across various scales. Their remarkable scalability with model size and data, combined with their interpretability through attention maps, positions ViTs as a formidable and versatile architecture for a wide array of visual tasks. The "An Image is Worth 16x16 Words" paradigm has not only unlocked new levels of performance but has also fostered a deeper conceptual unification between the fields of computer vision and natural language processing, promising exciting future developments.

<a name="7-future-directions-and-limitations"></a>
### 7. Future Directions and Limitations
Despite their impressive capabilities, Vision Transformers present several avenues for further research and also carry inherent limitations.

**Future Directions:**
1.  **Hybrid Architectures:** Combining the strengths of CNNs (local processing, inductive biases) and Transformers (global context, scalability) offers a promising direction. Models like Swin Transformers and CoAtNet have already explored this by introducing hierarchical structures or local attention windows, achieving improved efficiency and performance.
2.  **Efficiency and Deployment:** The computational cost of global self-attention scales quadratically with the number of patches, which can be prohibitive for high-resolution images or real-time applications. Research into efficient attention mechanisms (e.g., sparse attention, linear attention, attention with learnable queries) and optimized model architectures (e.g., MobileViT) is crucial for wider deployment.
3.  **Domain Adaptation and Few-Shot Learning:** Enhancing ViTs' performance in scenarios with limited data or for specific domain adaptation tasks remains an active area. Techniques from meta-learning or self-supervised learning could play a significant role.
4.  **Generative Models:** Extending ViTs to generative tasks, such as image generation, super-resolution, and image-to-image translation, leveraging their global contextual understanding. Diffusion models with Transformer backbones are already showing great promise.
5.  **Multimodal Learning:** Integrating ViTs with other modalities (e.g., text, audio) for truly multimodal understanding. Architectures like CLIP and DALL-E have already demonstrated the power of vision-language Transformers.

**Limitations:**
1.  **Data Hunger:** As discussed, pure ViTs require very large datasets for pre-training to achieve competitive performance, making them challenging to train from scratch on smaller, custom datasets without extensive data augmentation or transfer learning.
2.  **Computational Cost:** The quadratic complexity of self-attention with respect to the input sequence length (number of patches) can be a bottleneck, especially for very high-resolution images where the number of patches becomes substantial. This often necessitates downsampling or using smaller patch sizes, which might lose fine-grained details.
3.  **Inductive Bias:** The lack of strong inductive biases like locality and translation equivariance means ViTs must learn these properties from data. While this offers flexibility, it also implies a less efficient learning process compared to CNNs on smaller datasets.
4.  **Interpretability beyond Attention Maps:** While attention maps provide some insight, fully understanding the complex interplay of attention weights and learned feature representations within deep Transformer layers remains a challenge.

Despite these limitations, the Vision Transformer paradigm has irrevocably altered the landscape of computer vision research. Its success underscores the power of large-scale pre-training and the versatility of the self-attention mechanism, paving the way for more unified and powerful AI models across different data modalities.

---
<br>

<a name="türkçe-içerik"></a>
## Vision Transformers (ViT): Bir Resim 16x16 Kelime Değerindedir

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. ViT'in Doğuşu: CNN'lerin Ötesi](#2-vitin-doğuşu-cnnlerin-ötesi)
- [3. Vision Transformer Mimarisi](#3-vision-transformer-mimarisi)
  - [3.1. Yama Gömme (Patch Embedding)](#31-yama-gömme-patch-embedding)
  - [3.2. Konumsal Kodlama (Positional Encoding)](#32-konumsal-kodlama-positional-encoding)
  - [3.3. Transformer Kodlayıcı (Transformer Encoder)](#33-transformer-kodlayıcı-transformer-encoder)
  - [3.4. MLP Başlığı (MLP Head)](#34-mlp-başlığı-mlp-head)
- [4. Eğitim ve Performans](#4-eğitim-ve-performans)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)
- [7. Gelecek Yönelimler ve Sınırlamalar](#7-gelecek-yönelimler-ve-sınırlamalar)

<a name="1-giriş"></a>
### 1. Giriş
Bilgisayar görüşü alanı, ağırlıklı olarak **Evrişimsel Sinir Ağları (CNN'ler)** olmak üzere **derin öğrenme** modellerinin ortaya çıkmasıyla köklü bir dönüşüme tanık olmuştur. Yaklaşık on yıldır CNN'ler, görüntü sınıflandırma, nesne tespiti ve anlamsal segmentasyon gibi görevlerde eşsiz performans sergileyerek egemenliğini sürdürmüştür. Hiyerarşik yapıları, yerel alıcı alanları ve öteleme değişmezlikleri, görsel verileri işlemek için doğal olarak uygun kabul edilmiştir. Ancak, 2017'de Vaswani ve arkadaşları tarafından **Transformer** mimarisinin tanıtılmasıyla **Doğal Dil İşleme (NLP)** alanında paralel bir devrim yaşanmıştır. Başlangıçta diziden diziye görevler için tasarlanan Transformer'lar, yinelemeli veya evrişimsel katmanlardan uzaklaşarak, uzun menzilli bağımlılıkları etkili bir şekilde modellemek için **self-attention (öz-dikkat)** kavramını kullanmıştır.

Dosovitskiy ve arkadaşları (2020) tarafından yayımlanan "An Image Is Worth 16x16 Words: Transformers for Image Recognition At Scale" başlıklı çığır açan makale, doğrudan görüntü yamaları dizilerine uygulanan saf bir Transformer'ın görüntü sınıflandırma görevlerinde son teknoloji sonuçlar elde edebileceğini göstererek yerleşik paradigmayı sorgulamıştır. Bu, **Vision Transformers (ViT)**'ın doğuşuna işaret etmiş, NLP ve bilgisayar görüşü arasındaki boşluğu etkili bir şekilde kapatmış ve yeni araştırma yollarını açmıştır. ViT'in temel içgörüsü, bir görüntüyü 2D piksel ızgarası olarak değil, bir cümledeki kelimelere benzer şekilde düzleştirilmiş 2D yamaların bir dizisi olarak ele almak ve daha sonra standart bir Transformer kodlayıcıya beslemektir. Bu belge, Vision Transformer'ların mimari nüanslarını, operasyonel prensiplerini, avantajlarını ve etkilerini inceleyerek görsel temsil öğrenme anlayışımızı nasıl yeniden şekillendirdiklerini araştırmaktadır.

<a name="2-vitin-doğuşu-cnnlerin-ötesi"></a>
### 2. ViT'in Doğuşu: CNN'lerin Ötesi
ViT'den önce bilgisayar görüşü alanındaki baskın paradigma, tartışmasız bir şekilde **Evrişimsel Sinir Ağları (CNN'ler)** etrafında yoğunlaşmıştı. AlexNet, VGG, ResNet ve InceptionNet gibi mimariler, ImageNet gibi büyük ölçekli veri kümelerinde doğruluk sınırlarını zorlamıştı. CNN'ler, doğal indüktif ön yargıları sayesinde başarılı oldular: **yerellik** (pikselin küçük bölgelerini işleme) ve **öteleme eşvaryansı** (konumlarından bağımsız olarak özellikleri algılama). Bu ön yargılar, CNN'leri görüntü işleme için oldukça etkili ve veri açısından verimli hale getirdi, çünkü bu özellikleri sıfırdan öğrenmelerine gerek kalmadı.

Ancak, **self-attention (öz-dikkat) mekanizması** ile Transformer mimarisi, temelden farklı bir yaklaşım sundu. NLP'de self-attention, modellerin belirli bir öğeyi işlerken girdi dizisinin farklı kısımlarının önemini tartmasına izin vererek küresel bağımlılıkları etkili bir şekilde yakaladı. Bu küresel bakış açısı, evrişimlerin yerel odağının aksineydi. Transformer'ların NLP görevlerindeki başarısı, özellikle büyük ön eğitim veri kümeleriyle, araştırmacıları diğer alanlara uygulanabilirliğini araştırmaya sevk etti.

Transformer'ları vizyon için uyarlamaya yönelik ilk girişimler genellikle dikkat mekanizmalarını CNN mimarilerine entegre etmeyi veya tüm görüntüyü küresel olarak işlemek yerine yerel pencerelerde self-attention kullanmayı içeriyordu. ViT'in çığır açan katkısı, cüretkar basitliğiydi: hiçbir evrişimsel katman içermeyen *saf* bir Transformer'ın rekabetçi bir şekilde performans gösterebileceğini gösterdi. Bu radikal ayrım, CNN'lerin indüktif ön yargılarının, faydalı olsalar da, özellikle bol miktarda veri mevcut olduğunda modelin bu temsilleri kendi başına öğrenmesi için kesinlikle gerekli olmayabileceğini öne sürdü. Temel kavramsal sıçrama, görüntü verilerini Transformer işlemeye uygun bir dizi biçimine yeniden yorumlamaktı, böylece evrişimlerin görsel anlama için vazgeçilmez olduğu şeklindeki uzun süredir devam eden varsayımı sorgulandı. Amaç, Transformer'ın küresel ilişkileri modelleme yeteneğini ve büyük NLP modellerinde iyi gösterilen, artan veri ve model boyutuyla mükemmel ölçeklenebilirliğini kullanmaktı.

<a name="3-vision-transformer-mimarisi"></a>
### 3. Vision Transformer Mimarisi
ViT'in arkasındaki temel fikir, bir görüntüyü, bir Transformer'ın bir kelime dizisini işlediği gibi, sabit boyutlu yamaların bir dizisi olarak işlemektir. Bu dönüşüm, standart Transformer kodlayıcının doğrudan uygulanmasına izin verir. Mimari, birkaç ana bileşene ayrılabilir: **Yama Gömme (Patch Embedding)**, **Konumsal Kodlama (Positional Encoding)**, **Transformer Kodlayıcı**'nın kendisi ve sınıflandırma için son bir **MLP Başlığı (MLP Head)**.

#### 3.1. Yama Gömme (Patch Embedding)
ViT hattındaki ilk adım, girdi görüntüsünü düz, doğrusal gömmelerin bir dizisine dönüştürmektir. `H x W x C` (Yükseklik x Genişlik x Kanal) boyutunda bir girdi görüntüsü verildiğinde, görüntü, sabit bir `P x P` boyutundaki, birbiriyle çakışmayan kare yamaların bir ızgarasına bölünür. Her yama bu nedenle `P x P x C` boyutlarına sahip olacaktır.
Örneğin, bir görüntü `224x224x3` boyutunda ise ve yama boyutu `16x16` ise, görüntü `(224/16) x (224/16) = 14 x 14 = 196` yamaya bölünecektir.
Bu `P x P x C` yamaların her biri daha sonra `P*P*C` boyutunda 1D bir vektöre düzleştirilir. Bu düzleştirilmiş vektörler, daha sonra bir doğrusal projeksiyon katmanı kullanılarak daha yüksek boyutlu bir gömme alanına (örneğin, Transformer'ın gizli vektör boyutu olan `D` boyutları) yansıtılır. Bu işlem, her yamayı bir **yama gömmesine (patch embedding)** dönüştürür.
Bu yama gömmelerine ek olarak, (BERT'te kullanılanlara benzer) özel, öğrenilebilir bir **"[CLS]" belirteci**, yama gömmeleri dizisinin başına eklenir. Bu CLS belirteci, görüntünün küresel temsilcisi olarak hizmet eder ve Transformer kodlayıcının son katmanındaki karşılık gelen çıktısı, aşağı akış sınıflandırması için kullanılır.

<a name="32-konumsal-kodlama-positional-encoding"></a>
#### 3.2. Konumsal Kodlama (Positional Encoding)
Transformer'lar doğal olarak permütasyon-değişmezdir, yani bir dizideki belirteçlerin göreceli veya mutlak konumlarını doğal olarak anlamazlar. Ancak, uzamsal bilgi görüntüler için çok önemlidir. Bu uzamsal bağlamı yeniden tanıtmak için ViT, **öğrenilebilir 1D konumsal gömmeler** kullanır. Bu gömmeler, Transformer kodlayıcıya beslenmeden *önce* yama gömmelerine basitçe eklenir. Her yama gömmesi, görüntü ızgarasındaki orijinal konumuna karşılık gelen bir konumsal gömme ile artırılır. CLS belirteci de kendi özel konumsal gömmesini alır. Bu gömmeler, eğitim süreci sırasında öğrenilir ve modelin yamalar arasındaki uzamsal ilişkileri çıkarmasına olanak tanır.

<a name="33-transformer-kodlayıcı-transformer-encoder"></a>
#### 3.3. Transformer Kodlayıcı (Transformer Encoder)
ViT modelinin çekirdeği, NLP için orijinal Transformer mimarisinde kullanılan standart Transformer kodlayıcıdır. Her biri iki ana alt katman içeren bir dizi özdeş katmandan oluşur:
1.  **Çok Başlı Öz-Dikkat (Multi-Head Self-Attention - MHSA):** Bu mekanizma, modelin her bir yamayı işlerken farklı yamaların (belirteçlerin) önemini tartmasına olanak tanır. Her yama için MHSA mekanizması, diğer tüm yama gömmelerinin ağırlıklı bir toplamını hesaplar; burada ağırlıklar, bunların ikili benzerliklerinden türetilir. Bu, modelin tüm görüntüdeki küresel bağlamsal bilgileri yakalamasını sağlar.
2.  **Çok Katmanlı Algılayıcı (Multilayer Perceptron - MLP):** MHSA'dan sonra her konuma (yama gömmesi) bağımsız olarak uygulanan basit bir ileri beslemeli ağ. Bu MLP tipik olarak, aralarında bir GELU aktivasyon fonksiyonu bulunan iki doğrusal katmandan oluşur.
Bu alt katmanların her biri, bir **Katman Normalizasyonu** adımıyla takip edilir ve derin ağların kararlı eğitimini kolaylaştıran bir **artık bağlantı** her birinin etrafında uygulanır. Transformer kodlayıcının çıktısı, her girdi yaması ve CLS belirteci için bir tane olmak üzere bağlamsallaştırılmış gömmelerin bir dizisidir.

<a name="34-mlp-başlığı-mlp-head"></a>
#### 3.4. MLP Başlığı (MLP Head)
Son olarak, görüntü sınıflandırma görevleri için, Transformer kodlayıcının son katmanından çıkan özel **CLS belirtecine** karşılık gelen çıktı gömme alınır. Bu tek vektör, tüm görüntünün toplu temsili olarak kabul edilir. Daha sonra, tipik olarak bir veya iki doğrusal katmandan ve doğrusal olmayan bir aktivasyon fonksiyonundan oluşan, ardından gösterimi istenen çıktı sınıfı sayısına yansıtan son bir doğrusal katmandan oluşan basit bir **Çok Katmanlı Algılayıcı (MLP) başlığı** aracılığıyla geçirilir. Sınıf olasılıklarını elde etmek için bu lojitlere bir Softmax fonksiyonu uygulanır.

<a name="4-eğitim-ve-performans"></a>
### 4. Eğitim ve Performans
Vision Transformer'lar, NLP'deki benzerlerinden miras aldıkları bir özellik olarak, etkili eğitim için genellikle önemli miktarda veri gerektirir. Yerellik, öteleme eşvaryansı gibi güçlü indüktif ön yargılardan yararlanan CNN'lerin aksine, ViT'ler daha "veri açıdır". Bu uzamsal hiyerarşileri ve ilişkileri, self-attention mekanizması aracılığıyla doğrudan verilerden öğrenirler.
Orijinal ViT makalesi, **JFT-300M** (300 milyon görüntülü özel bir veri kümesi) gibi büyük ölçekli veri kümelerinde eğitimin, ViT'in son teknoloji CNN'lerden daha iyi performans göstermesi için çok önemli olduğunu göstermiştir. ImageNet gibi daha küçük veri kümelerinde eğitildiğinde, ViT'ler ek düzenlileştirme veya özel eğitim stratejileri olmadan genellikle eşdeğer CNN'lerden daha kötü performans göstermiştir. Bu, önemli bir dengeyi vurgular: **ViT'ler, artan model kapasitesi ve esnekliği için bazı indüktif ön yargılardan ödün verir**, bu da yeterli veri verildiğinde potansiyel olarak daha güçlü ve genelleştirilmiş temsiller öğrenmelerine olanak tanır.

Veri gereksinimini azaltmak için çeşitli stratejiler ortaya çıkmıştır:
1.  **Transfer Öğrenimi:** ViT'leri çok büyük veri kümelerinde (örneğin, JFT, ImageNet-21k) önceden eğitmek ve daha sonra bunları daha küçük, göreve özel veri kümelerinde ince ayar yapmak son derece etkili olmuştur. Bu yaklaşım, kapsamlı genel verilerden öğrenilen bilgiyi kullanır.
2.  **Bilgi Damıtma:** Önceden eğitilmiş bir CNN'i bir ViT öğrenci modelinin eğitimini yönlendirmek için bir öğretmen modeli olarak kullanmak, CNN'in indüktif ön yargılarını enjekte etmeye ve daha küçük veri kümelerinde performansı artırmaya yardımcı olabilir.
3.  **Veri Artırma:** Eğitim sırasında agresif veri artırma teknikleri (örneğin, RandAugment, Mixup, Cutmix) kullanmak, eğitim verilerinin etkili boyutunu ve çeşitliliğini artırmaya yardımcı olur.

Yeterince eğitildiğinde, ViT'ler, özellikle büyük ölçekli görüntü sınıflandırmasında, çeşitli kıyaslama testlerinde CNN'lerin doğruluğuna eşit veya hatta aşan etkileyici bir performans göstermiştir. Bir görüntü boyunca küresel ilişkileri modelleme yetenekleri, ölçeklenebilirlikleriyle birleştiğinde, onları geleneksel evrişimsel yaklaşımlara cazip bir alternatif haline getirir. Ayrıca, self-attention mekanizması tarafından üretilen **dikkat haritaları** genellikle görselleştirilebilir, bu da modelin tahminleri için görüntünün hangi bölümlerine odaklandığına dair içgörüler sunarak bir yorumlanabilirlik düzeyi sağlar.

<a name="5-kod-örneği"></a>
### 5. Kod Örneği
Bu Python kodu, bir görüntünün kavramsal olarak nasıl yamalara bölündüğünü ve düzleştirildiğini, Vision Transformer'lar için uygun hale getirildiğini gösteren temel fikri açıklamaktadır. Tensorlar için PyTorch kullanır.

```python
import torch
import torch.nn as nn

def img_to_patches(image, patch_size):
    """
    Bir görüntü tensörünü düzleştirilmiş yamaların bir dizisine dönüştürür.

    Argümanlar:
        image (torch.Tensor): (B, C, H, W) şeklindeki girdi görüntü tensörü.
        patch_size (int): Kare yamanın boyutu (P).

    Dönüş:
        torch.Tensor: Düzleştirilmiş yamaların tensörü, şekil (B, N, P*P*C),
                      burada N yama sayısıdır.
    """
    B, C, H, W = image.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("Görüntü boyutları yama boyutuna bölünebilir olmalıdır.")

    # Görüntüyü (B, C, H/P, P, W/P, P) şeklinde yeniden şekillendir
    # Sonra (B, H/P, W/P, P, P, C) şeklinde permütasyon yap
    # Sonra (P, P, C)'yi (P*P*C)'ye düzleştir
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # (B, H/P, W/P, C, P, P) -> (B, H/P, W/P, P, P, C)
    
    # Yamaları (B, N, P*P*C) şekline düzleştir
    patches = patches.view(B, -1, patch_size * patch_size * C)
    return patches

# Örnek kullanım:
batch_size = 1
channels = 3
height = 224
width = 224
patch_size = 16

# Bir örnek görüntü tensörü oluştur
dummy_image = torch.randn(batch_size, channels, height, width)
print(f"Orijinal görüntü şekli: {dummy_image.shape}")

# Görüntüyü yamalara dönüştür
image_patches = img_to_patches(dummy_image, patch_size)
print(f"Düzleştirilmiş yamaların şekli: {image_patches.shape}")

# image_patches.shape için beklenen çıktı: (1, (224/16)*(224/16), 16*16*3) = (1, 196, 768)

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
### 6. Sonuç
Vision Transformer'ların (ViT) tanıtımı, bilgisayar görüşü tarihinde bir dönüm noktasını temsil ederek Evrişimsel Sinir Ağları'nın (CNN'ler) uzun süredir devam eden egemenliğini sorgulamıştır. Doğal Dil İşleme'den (NLP) alınan Transformer mimarisini görüntü görevlerine başarıyla uyarlayan ViT, evrişimlerdeki doğal indüktif ön yargıların, özellikle yeterli eğitim verisiyle donatıldığında, kesinlikle vazgeçilmez olmadığını göstermiştir. ViT'nin temel yeniliği, görüntüleri düzleştirilmiş yamalar dizisi olarak ele alma yeteneğinde yatmaktadır, bu da güçlü self-attention mekanizmasının tüm görüntüdeki küresel bağımlılıkları yakalamasına olanak tanır.

ViT'ler başlangıçta üstün performans için büyük veri kümeleri gerektirirken, sonraki araştırmalar ve eğitim stratejilerindeki iyileştirmeler (transfer öğrenimi, damıtma ve agresif veri artırma gibi) onları çeşitli ölçeklerde daha erişilebilir ve rekabetçi hale getirmiştir. Model boyutu ve verilerle olağanüstü ölçeklenebilirlikleri, dikkat haritaları aracılığıyla yorumlanabilirlikleriyle birleştiğinde, ViT'leri çok çeşitli görsel görevler için zorlu ve çok yönlü bir mimari olarak konumlandırmaktadır. "Bir Resim 16x16 Kelime Değerindedir" paradigması, yalnızca yeni performans seviyelerinin kilidini açmakla kalmamış, aynı zamanda bilgisayar görüşü ve doğal dil işleme alanları arasında daha derin bir kavramsal birleşmeyi teşvik ederek heyecan verici gelecekteki gelişmeleri de vaat etmektedir.

<a name="7-gelecek-yönelimler-ve-sınırlamalar"></a>
### 7. Gelecek Yönelimler ve Sınırlamalar
Etkileyici yeteneklerine rağmen, Vision Transformer'lar daha fazla araştırma için birkaç yol sunmakta ve ayrıca doğuştan gelen sınırlamalar taşımaktadır.

**Gelecek Yönelimler:**
1.  **Hibrit Mimariler:** CNN'lerin (yerel işleme, indüktif ön yargılar) ve Transformer'ların (küresel bağlam, ölçeklenebilirlik) güçlü yönlerini birleştirmek umut vadeden bir yöndür. Swin Transformer'lar ve CoAtNet gibi modeller, hiyerarşik yapılar veya yerel dikkat pencereleri sunarak bunu zaten araştırmış, daha iyi verimlilik ve performans elde etmişlerdir.
2.  **Verimlilik ve Dağıtım:** Küresel self-attention'ın hesaplama maliyeti, yama sayısı ile kare orantılı olarak artar ve bu, yüksek çözünürlüklü görüntüler veya gerçek zamanlı uygulamalar için aşırı pahalı olabilir. Verimli dikkat mekanizmaları (örneğin, seyrek dikkat, doğrusal dikkat, öğrenilebilir sorgularla dikkat) ve optimize edilmiş model mimarileri (örneğin, MobileViT) üzerine araştırmalar, daha geniş dağıtım için çok önemlidir.
3.  **Alan Adaptasyonu ve Az Verili Öğrenme (Few-Shot Learning):** Sınırlı veri senaryolarında veya belirli alan adaptasyon görevleri için ViT'lerin performansını artırmak aktif bir alandır. Meta öğrenme veya self-supervised öğrenme teknikleri önemli bir rol oynayabilir.
4.  **Üretici Modeller:** ViT'leri, küresel bağlamsal anlayışlarından yararlanarak görüntü üretimi, süper çözünürlük ve görüntüden görüntüye çeviri gibi üretici görevlere genişletmek. Transformer omurgalı difüzyon modelleri şimdiden büyük umut vaat ediyor.
5.  **Çok Modlu Öğrenme:** Gerçekten çok modlu anlama için ViT'leri diğer modalitelerle (örneğin, metin, ses) entegre etmek. CLIP ve DALL-E gibi mimariler, görme-dil Transformer'larının gücünü zaten göstermiştir.

**Sınırlamalar:**
1.  **Veri Açlığı:** Bahsedildiği gibi, saf ViT'ler, rekabetçi performans elde etmek için ön eğitim için çok büyük veri kümeleri gerektirir, bu da onları kapsamlı veri artırma veya transfer öğrenimi olmadan daha küçük, özel veri kümelerinde sıfırdan eğitmeyi zorlaştırır.
2.  **Hesaplama Maliyeti:** Self-attention'ın girdi dizisi uzunluğuna (yama sayısı) göre karesel karmaşıklığı, özellikle yama sayısının önemli hale geldiği çok yüksek çözünürlüklü görüntüler için bir darboğaz olabilir. Bu genellikle örneklemeyi düşürmeyi veya daha küçük yama boyutları kullanmayı gerektirir, bu da ince taneli ayrıntıların kaybolmasına neden olabilir.
3.  **İndüktif Ön Yargı:** Yerellik ve öteleme eşvaryansı gibi güçlü indüktif ön yargıların olmaması, ViT'lerin bu özellikleri verilerden öğrenmesi gerektiği anlamına gelir. Bu esneklik sunsa da, daha küçük veri kümelerinde CNN'lere kıyasla daha az verimli bir öğrenme süreci anlamına gelir.
4.  **Dikkat Haritalarının Ötesinde Yorumlanabilirlik:** Dikkat haritaları bazı içgörüler sağlasa da, derin Transformer katmanlarındaki dikkat ağırlıklarının ve öğrenilen özellik temsillerinin karmaşık etkileşimini tam olarak anlamak bir zorluk olmaya devam etmektedir.

Bu sınırlamalara rağmen, Vision Transformer paradigması bilgisayar görüşü araştırmalarının çehresini geri dönülmez bir şekilde değiştirmiştir. Başarısı, büyük ölçekli ön eğitimin gücünü ve self-attention mekanizmasının çok yönlülüğünü vurgulayarak, farklı veri modaliteleri arasında daha birleşik ve güçlü AI modellerinin yolunu açmaktadır.










