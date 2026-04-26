# MAE: Masked Autoencoders are Scalable Vision Learners

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: Self-Supervised Learning and Vision Transformers](#2-background-self-supervised-learning-and-vision-transformers)
- [3. MAE Architecture and Mechanism](#3-mae-architecture-and-mechanism)
  - [3.1. Patching and Masking](#31-patching-and-masking)
  - [3.2. Encoder](#32-encoder)
  - [3.3. Decoder](#33-decoder)
  - [3.4. Asymmetric Design and Loss Function](#34-asymmetric-design-and-loss-function)
- [4. Training and Performance Insights](#4-training-and-performance-insights)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction

The remarkable success of **self-supervised learning (SSL)** in Natural Language Processing (NLP), particularly with models like BERT, has demonstrated the immense potential of pre-training large models on vast amounts of unlabeled data. These models learn rich, transferable representations by performing pretext tasks, such as masked token prediction, on the input data itself. In computer vision, while SSL has made significant strides, it has historically lagged behind NLP in terms of scalability and the simplicity of pre-training objectives for transformer-based architectures.

**Masked Autoencoders (MAE)**, introduced by He et al. (2022), represent a pivotal advancement in bringing the scalability and effectiveness of masked modeling to computer vision. MAE adapts the core idea of masking a portion of the input and reconstructing the missing parts to image data, specifically designed to leverage the power of **Vision Transformers (ViTs)**. Unlike previous SSL methods in vision that often rely on complex contrastive learning frameworks or auxiliary networks, MAE offers a refreshingly simple and elegant approach: it randomly masks a high proportion of image patches and then reconstructs the missing pixels. This simplicity, combined with an asymmetric encoder-decoder architecture, allows MAE to efficiently pre-train large-scale ViT models, achieving state-of-the-art performance across various downstream vision tasks with significantly reduced computational cost. The paradigm shift brought by MAE opens new avenues for learning robust visual representations from unlabeled image datasets at an unprecedented scale.

### 2. Background: Self-Supervised Learning and Vision Transformers

To fully appreciate the innovations introduced by MAE, it is essential to understand the landscape of self-supervised learning in computer vision and the advent of Vision Transformers.

**Self-Supervised Learning (SSL) in Vision:** Before MAE, dominant SSL paradigms in vision included:
*   **Contrastive Learning:** Methods like SimCLR, MoCo, and BYOL learn representations by maximizing agreement between different augmentations of the same image (positive pairs) and minimizing agreement with other images (negative pairs, explicitly or implicitly). While highly effective, these methods often involve complex augmentation strategies, large batch sizes, or momentum encoders, adding to their complexity and computational demands.
*   **Non-Contrastive Methods:** Approaches like BYOL and DINO moved away from explicit negative pairs, often relying on stop-gradient operations or student-teacher networks to prevent collapse.
*   **Masked Image Modeling (MIM):** Pre-dating MAE, early attempts at MIM in vision often focused on predicting discrete visual tokens (e.g., using a VQ-VAE to discretize image patches into tokens) or relied on smaller masking ratios. These methods often struggled with the high redundancy of natural images, where local patches contain much less semantic information than individual words in text.

**Vision Transformers (ViTs):** The seminal work on ViTs demonstrated that transformer architectures, originally designed for NLP, could achieve competitive or superior performance to Convolutional Neural Networks (CNNs) on image classification tasks when trained on large datasets. ViTs operate by splitting an image into a grid of non-overlapping patches, linearly embedding these patches, adding positional information, and then feeding the resulting sequence of vectors into a standard transformer encoder. The primary challenge for ViTs, especially without massive supervised pre-training (like on JFT-300M), was their data hunger, making them ideal candidates for SSL techniques that could leverage abundant unlabeled data.

The redundancy inherent in natural images – where a masked patch can often be trivially predicted from its immediate neighbors – posed a significant hurdle for applying masked language modeling directly to vision. MAE addresses this by introducing a high masking ratio and an asymmetric encoder-decoder design.

### 3. MAE Architecture and Mechanism

MAE's core strength lies in its elegantly simple yet highly effective architecture, which is specifically tailored to the characteristics of image data and Vision Transformers. It comprises three main components: a **masking strategy**, an **encoder**, and a **decoder**.

#### 3.1. Patching and Masking

The initial step in MAE, similar to a standard ViT, involves dividing an input image into a sequence of non-overlapping **patches**. For a typical image, this might result in a grid of 16x16 pixel patches.

Crucially, MAE then applies a **high masking ratio**, typically around 75%. This means that 75% of the image patches are randomly selected and removed. This aggressive masking serves two primary purposes:
1.  **Reduces Redundancy:** By masking a large portion, the model is forced to reconstruct information from a sparse set of visible patches, preventing trivial solutions where masked patches can be inferred from nearby, highly correlated pixels. This encourages the model to learn a more global understanding of the image content.
2.  **Increased Efficiency:** The encoder only processes the remaining 25% of visible (unmasked) patches, significantly reducing the computational cost during the encoding phase.

The masked patches are simply discarded; their original pixel values are what the decoder will eventually attempt to reconstruct.

#### 3.2. Encoder

The MAE **encoder** is a standard Vision Transformer. Its critical characteristic is that it only operates on the **visible (unmasked) patches**.
*   Each visible patch is linearly embedded into a high-dimensional vector.
*   Positional embeddings are added to these patch embeddings to retain spatial information.
*   The resulting sequence of visible patch embeddings is then fed into a series of transformer blocks (multi-head self-attention and MLP layers).

Because the encoder only processes a small fraction of the total patches, it becomes highly efficient, especially during the pre-training phase with large models. This is a significant departure from approaches that process all patches, even if some are masked, or from contrastive methods that typically process full images.

#### 3.3. Decoder

The MAE **decoder** is a much shallower and simpler transformer compared to the encoder. Its role is to reconstruct the original pixel values of the **masked patches**.
*   The decoder takes as input the latent representations (output embeddings) of the visible patches from the encoder.
*   It also receives a set of **mask tokens**, which are learnable embeddings representing the *missing* patches. Positional embeddings are added to these mask tokens to indicate their original locations in the image.
*   The encoder output embeddings and the mask tokens (with their respective positional embeddings) are concatenated to form the complete sequence, which is then fed into the decoder transformer blocks.

The decoder then processes this combined sequence to predict the pixel values for each masked patch. Its primary objective is to take the abstract understanding from the visible patches and contextualize it to fill in the gaps. The decoder's simplicity is intentional; it contributes minimal computational overhead, further enhancing MAE's overall efficiency.

#### 3.4. Asymmetric Design and Loss Function

The **asymmetric design** – a deep encoder processing only visible patches and a shallow decoder reconstructing all masked patches – is a cornerstone of MAE's efficiency and effectiveness.
*   The encoder learns a robust, abstract representation from incomplete visual information.
*   The decoder acts as a lightweight task-specific head for reconstruction during pre-training.

The **loss function** used for MAE pre-training is straightforward: **Mean Squared Error (MSE)** between the reconstructed pixel values of the masked patches and their original, ground-truth pixel values. This direct pixel-level reconstruction objective, applied only to the masked patches, forces the model to learn meaningful visual semantics rather than memorizing surface-level textures. The simplicity of this objective is a key differentiator from more complex contrastive losses.

### 4. Training and Performance Insights

The pre-training phase of MAE is designed to be highly scalable and efficient:
1.  **Pre-training:** A ViT model is pre-trained as an MAE on a large unlabeled image dataset (e.g., ImageNet-1K). The encoder learns to extract features from sparsely sampled image patches, and the decoder learns to reconstruct the masked regions based on these features and positional information. The high masking ratio is crucial here, as it pushes the model to learn higher-level semantic understanding rather than merely interpolating local pixel values.
2.  **Fine-tuning:** After pre-training, the decoder is discarded. The pre-trained encoder (the ViT backbone) is then combined with a simple classification head and fine-tuned on a labeled dataset for a specific downstream task (e.g., image classification on ImageNet, object detection on COCO, semantic segmentation on ADE20K).

**Key Performance Insights:**
*   **Superior Scalability:** MAE can effectively pre-train very large ViT models (e.g., ViT-Huge, ViT-G) with remarkable efficiency due to the encoder processing only 25% of patches. This significantly reduces computation time and memory usage compared to full-image processing methods.
*   **High Performance:** MAE-pre-trained models achieve state-of-the-art results on various benchmarks, often surpassing supervised pre-training or other SSL methods, especially when scaled up. For instance, ViT-Large trained with MAE achieves 87.8% top-1 accuracy on ImageNet-1K with only 300 epochs of pre-training.
*   **Robustness to Masking:** The high masking ratio (75%) proved to be surprisingly effective, leading to better representations than lower masking ratios. This suggests that extensive missing information forces the model to capture more abstract and global visual features.
*   **Simplicity and Universality:** MAE's simple pixel reconstruction task and MSE loss are general enough to work across different vision tasks without requiring task-specific modifications during pre-training.

### 5. Code Example

This conceptual Python code snippet illustrates the basic idea of patching an image and then randomly masking a high percentage of those patches, which is a core part of the MAE pre-training process.

```python
import numpy as np

def create_image_patches(image, patch_size):
    """
    Divides an image into non-overlapping patches.
    Args:
        image (np.array): Input image (H, W, C).
        patch_size (int): Size of the square patch.
    Returns:
        np.array: Array of patches (N_patches, patch_size, patch_size, C).
    """
    h, w, c = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, \
        "Image dimensions must be divisible by patch_size."

    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size, :]
            patches.append(patch)
    return np.array(patches)

def apply_random_masking(patches, mask_ratio):
    """
    Applies random masking to a given set of image patches.
    Args:
        patches (np.array): Array of patches (N_patches, patch_size, patch_size, C).
        mask_ratio (float): Proportion of patches to mask (e.g., 0.75 for 75%).
    Returns:
        tuple: (visible_patches, masked_indices, unmasked_indices, full_indices)
    """
    num_patches = patches.shape[0]
    num_masked = int(num_patches * mask_ratio)

    # Generate random permutation of indices
    full_indices = np.arange(num_patches)
    np.random.shuffle(full_indices)

    # Select indices for masked and unmasked patches
    masked_indices = full_indices[:num_masked]
    unmasked_indices = full_indices[num_masked:]

    # Get visible patches (those that are NOT masked)
    visible_patches = patches[unmasked_indices]

    # For MAE, masked patches are replaced by a special [MASK] token in the decoder input
    # Here, we just return the indices and visible patches.
    return visible_patches, masked_indices, unmasked_indices, full_indices

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy image (e.g., a grayscale image for simplicity)
    dummy_image = np.random.rand(224, 224, 3) # A 224x224 RGB image
    patch_size = 16
    mask_ratio = 0.75

    print(f"Original image shape: {dummy_image.shape}")

    # 1. Create patches
    image_patches = create_image_patches(dummy_image, patch_size)
    print(f"Number of patches: {image_patches.shape[0]}, patch shape: {image_patches.shape[1:]}")

    # 2. Apply masking
    visible_patches, masked_indices, unmasked_indices, _ = \
        apply_random_masking(image_patches, mask_ratio)

    print(f"Number of visible patches: {len(visible_patches)}")
    print(f"Number of masked patches: {len(masked_indices)}")
    print(f"Masked ratio applied: {len(masked_indices) / image_patches.shape[0]:.2f}")

    # visible_patches would be fed to the MAE encoder
    # masked_indices would be used to place mask tokens in the decoder input
    # image_patches[masked_indices] would be the ground truth for reconstruction loss

(End of code example section)
```

### 6. Conclusion

MAE: Masked Autoencoders are Scalable Vision Learners has significantly advanced the field of self-supervised learning for computer vision. By translating the highly successful masked modeling paradigm from NLP to vision, MAE provides an elegant, efficient, and powerful method for pre-training large Vision Transformer models. Its core innovations – a high masking ratio, an asymmetric encoder-decoder architecture, and a simple pixel reconstruction objective – enable it to learn rich, transferable visual representations from vast amounts of unlabeled data with unprecedented scalability. MAE's performance across various downstream tasks validates its effectiveness and establishes a new benchmark for self-supervised pre-training in vision. The conceptual simplicity and empirical success of MAE pave the way for future research into even larger and more capable general-purpose vision models, potentially reducing the reliance on massive labeled datasets and democratizing access to high-performance visual AI.

---
<br>

<a name="türkçe-içerik"></a>
## MAE: Maskelenmiş Otomatik Kodlayıcılar Ölçeklenebilir Görsel Öğrenicilerdir

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: Kendi Kendine Denetimli Öğrenme ve Görsel Transformatörler](#2-arka-plan-kendi-kendine-denetimli-öğrenme-ve-görsel-transformatörler)
- [3. MAE Mimarisi ve Mekanizması](#3-mae-mimarisi-ve-mekanizması)
  - [3.1. Yama Oluşturma ve Maskeleme](#31-yama-oluşturma-ve-maskeleme)
  - [3.2. Kodlayıcı (Encoder)](#32-kodlayıcı-encoder)
  - [3.3. Kod Çözücü (Decoder)](#33-kod-çözücü-decoder)
  - [3.4. Asimetrik Tasarım ve Kayıp Fonksiyonu](#34-asimetrik-tasarım-ve-kayıp-fonksiyonu)
- [4. Eğitim ve Performans Analizi](#4-eğitim-ve-performans-analizi)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

### 1. Giriş

Doğal Dil İşleme (NLP) alanında, özellikle BERT gibi modellerle, **kendi kendine denetimli öğrenmenin (SSL)** olağanüstü başarısı, büyük etiketlenmemiş veri kümeleri üzerinde büyük modelleri önceden eğitmenin muazzam potansiyelini göstermiştir. Bu modeller, giriş verisinin kendisi üzerinde maskelenmiş belirteç tahmini gibi kurgusal görevleri yerine getirerek zengin, aktarılabilir temsiller öğrenirler. Bilgisayar görüşünde SSL önemli ilerlemeler kaydetmiş olsa da, ölçeklenebilirlik ve transformatör tabanlı mimariler için ön eğitim hedeflerinin basitliği açısından NLP'nin gerisinde kalmıştır.

He ve diğerleri (2022) tarafından tanıtılan **Maskelenmiş Otomatik Kodlayıcılar (MAE)**, maskeleme modellemesinin ölçeklenebilirliğini ve etkinliğini bilgisayar görüşüne taşıyan çok önemli bir ilerlemeyi temsil etmektedir. MAE, girişin bir kısmını maskeleme ve eksik parçaları yeniden yapılandırma ana fikrini görüntü verilerine uyarlar ve özellikle **Görsel Transformatörlerin (ViT'ler)** gücünden yararlanmak için tasarlanmıştır. Görüşteki karmaşık karşılaştırmalı öğrenme çerçevelerine veya yardımcı ağlara dayanan önceki SSL yöntemlerinin aksine, MAE ferahlatıcı derecede basit ve zarif bir yaklaşım sunar: görüntü yamalarının büyük bir oranını rastgele maskeler ve ardından eksik pikselleri yeniden yapılandırır. Bu basitlik, asimetrik bir kodlayıcı-kod çözücü mimarisiyle birleştiğinde, MAE'nin büyük ölçekli ViT modellerini verimli bir şekilde önceden eğitmesine olanak tanır ve önemli ölçüde azaltılmış hesaplama maliyetiyle çeşitli aşağı akış görüş görevlerinde en son teknoloji performansına ulaşır. MAE'nin getirdiği paradigma değişimi, etiketlenmemiş görüntü veri kümelerinden benzeri görülmemiş ölçekte sağlam görsel temsiller öğrenmek için yeni yollar açmaktadır.

### 2. Arka Plan: Kendi Kendine Denetimli Öğrenme ve Görsel Transformatörler

MAE tarafından sunulan yenilikleri tam olarak anlamak için, bilgisayar görüşünde kendi kendine denetimli öğrenme ve Görsel Transformatörlerin ortaya çıkışı bağlamını kavramak çok önemlidir.

**Görüşte Kendi Kendine Denetimli Öğrenme (SSL):** MAE'den önce, görüşte baskın SSL paradigmaları şunları içeriyordu:
*   **Karşılaştırmalı Öğrenme:** SimCLR, MoCo ve BYOL gibi yöntemler, aynı görüntünün farklı büyütmeleri (pozitif çiftler) arasındaki uyumu en üst düzeye çıkararak ve diğer görüntülerle (negatif çiftler, açıkça veya dolaylı olarak) uyumu en aza indirerek temsiller öğrenir. Son derece etkili olsalar da, bu yöntemler genellikle karmaşık büyütme stratejileri, büyük toplu iş boyutları veya momentum kodlayıcıları içerir ve bu da karmaşıklıklarını ve hesaplama gereksinimlerini artırır.
*   **Karşılaştırmalı Olmayan Yöntemler:** BYOL ve DINO gibi yaklaşımlar, açık negatif çiftlerden uzaklaşarak, genellikle çöküşü önlemek için durdurma gradyanı işlemleri veya öğrenci-öğretmen ağlarına güveniyordu.
*   **Maskelenmiş Görüntü Modelleme (MIM):** MAE'den önce, görüşteki MIM'ye yönelik ilk girişimler genellikle ayrık görsel belirteçleri tahmin etmeye (örneğin, görüntü yamalarını belirteçlere ayrıştırmak için bir VQ-VAE kullanarak) veya daha küçük maskeleme oranlarına dayanıyordu. Bu yöntemler, doğal görüntülerdeki yüksek yedeklilikle mücadele ediyordu; burada yerel yamalar, metindeki tek tek kelimelerden çok daha az anlamsal bilgi içerir.

**Görsel Transformatörler (ViT'ler):** ViT'ler üzerine yapılan çığır açan çalışma, başlangıçta NLP için tasarlanmış transformatör mimarilerinin, büyük veri kümeleri üzerinde eğitildiğinde görüntü sınıflandırma görevlerinde Evrişimsel Sinir Ağlarına (CNN'ler) kıyasla rekabetçi veya üstün performans gösterebildiğini göstermiştir. ViT'ler, bir görüntüyü çakışmayan yamalar ızgarasına ayırarak, bu yamaları doğrusal olarak gömerek, konumsal bilgi ekleyerek ve ardından ortaya çıkan vektör dizisini standart bir transformatör kodlayıcıya besleyerek çalışır. ViT'ler için birincil zorluk, özellikle büyük denetimli ön eğitim (JFT-300M'deki gibi) olmadan, veri açlığıydı ve bu da onları bol miktarda etiketlenmemiş veriden yararlanabilen SSL teknikleri için ideal adaylar haline getiriyordu.

Doğal görüntülerde doğal olan fazlalık – maskelenmiş bir yamanın genellikle en yakın komşularından kolayca tahmin edilebileceği – maskelenmiş dil modellemesini doğrudan görüşe uygulamak için önemli bir engel oluşturuyordu. MAE, yüksek bir maskeleme oranı ve asimetrik bir kodlayıcı-kod çözücü tasarımı sunarak bu sorunu ele alır.

### 3. MAE Mimarisi ve Mekanizması

MAE'nin temel gücü, görüntü verilerinin ve Görsel Transformatörlerin özelliklerine özel olarak uyarlanmış, zarifçe basit ancak son derece etkili mimarisinde yatmaktadır. Üç ana bileşenden oluşur: bir **maskeleme stratejisi**, bir **kodlayıcı** ve bir **kod çözücü**.

#### 3.1. Yama Oluşturma ve Maskeleme

MAE'deki ilk adım, standart bir ViT'ye benzer şekilde, bir giriş görüntüsünü bir dizi çakışmayan **yamaya** bölmeyi içerir. Tipik bir görüntü için bu, 16x16 piksellik yamalardan oluşan bir ızgara ile sonuçlanabilir.

Önemli olarak, MAE daha sonra tipik olarak %75 civarında **yüksek bir maskeleme oranı** uygular. Bu, görüntü yamalarının %75'inin rastgele seçilip çıkarılması anlamına gelir. Bu agresif maskeleme iki temel amaca hizmet eder:
1.  **Gereksizliği Azaltır:** Büyük bir bölümü maskeleyerek, modelin seyrek bir görünür yama kümesinden bilgi yeniden yapılandırması zorlanır ve maskelenmiş yamaların yakın, yüksek oranda ilişkili piksellerden çıkarılabileceği önemsiz çözümler önlenir. Bu, modeli görüntü içeriğinin daha küresel bir anlayışını öğrenmeye teşvik eder.
2.  **Verimliliği Artırır:** Kodlayıcı yalnızca kalan %25'lik görünür (maskelenmemiş) yamaları işler ve kodlama aşamasında hesaplama maliyetini önemli ölçüde azaltır.

Maskelenmiş yamalar basitçe atılır; orijinal piksel değerleri, kod çözücünün sonunda yeniden yapılandırmaya çalışacağı şeylerdir.

#### 3.2. Kodlayıcı (Encoder)

MAE **kodlayıcı**, standart bir Görsel Transformatördür. Kritik özelliği, yalnızca **görünür (maskelenmemiş) yamalar** üzerinde çalışmasıdır.
*   Her görünür yama, doğrusal olarak yüksek boyutlu bir vektöre gömülür.
*   Konumsal bilgilerini korumak için bu yama gömülmelerine konumsal gömülmeler eklenir.
*   Ortaya çıkan görünür yama gömülmeleri dizisi daha sonra bir dizi transformatör bloğuna (çok kafalı kendi kendine dikkat ve MLP katmanları) beslenir.

Kodlayıcı, toplam yamaların yalnızca küçük bir kısmını işlediği için, özellikle büyük modellerle ön eğitim aşamasında son derece verimli hale gelir. Bu, bazıları maskelenmiş olsa bile tüm yamaları işleyen yaklaşımlardan veya tipik olarak tam görüntüleri işleyen karşılaştırmalı yöntemlerden önemli bir sapmadır.

#### 3.3. Kod Çözücü (Decoder)

MAE **kod çözücü**, kodlayıcıya kıyasla çok daha sığ ve basit bir transformatördür. Rolü, **maskelenmiş yamaların** orijinal piksel değerlerini yeniden yapılandırmaktır.
*   Kod çözücü, kodlayıcıdan gelen görünür yamaların gizli temsillerini (çıktı gömülmeleri) girdi olarak alır.
*   Ayrıca, *eksik* yamaları temsil eden öğrenilebilir gömülmeler olan bir dizi **maske belirteci** alır. Bu maske belirteçlerine, görüntüdeki orijinal konumlarını belirtmek için konumsal gömülmeler eklenir.
*   Kodlayıcı çıktı gömülmeleri ve maske belirteçleri (kendi konumsal gömülmeleriyle birlikte) birleştirilerek, daha sonra kod çözücü transformatör bloklarına beslenen tam diziyi oluşturur.

Kod çözücü daha sonra bu birleşik diziyi işleyerek her maskelenmiş yama için piksel değerlerini tahmin eder. Birincil amacı, görünür yamalardan gelen soyut anlayışı alıp boşlukları doldurmak için bağlamlandırmaktır. Kod çözücünün basitliği kasıtlıdır; minimum hesaplama yükü katkısında bulunur ve MAE'nin genel verimliliğini daha da artırır.

#### 3.4. Asimetrik Tasarım ve Kayıp Fonksiyonu

Asimetrik tasarım – yalnızca görünür yamaları işleyen derin bir kodlayıcı ve tüm maskelenmiş yamaları yeniden yapılandıran sığ bir kod çözücü – MAE'nin verimliliğinin ve etkinliğinin temel taşıdır.
*   Kodlayıcı, eksik görsel bilgilerden sağlam, soyut bir temsil öğrenir.
*   Kod çözücü, ön eğitim sırasında yeniden yapılandırma için hafif bir göreve özgü başlık görevi görür.

MAE ön eğitimi için kullanılan **kayıp fonksiyonu** basittir: maskelenmiş yamaların yeniden yapılandırılmış piksel değerleri ile orijinal, gerçek piksel değerleri arasındaki **Ortalama Kare Hatası (MSE)**. Yalnızca maskelenmiş yamalara uygulanan bu doğrudan piksel düzeyinde yeniden yapılandırma hedefi, modeli yüzey düzeyindeki dokuları ezberlemek yerine anlamlı görsel anlambilim öğrenmeye zorlar. Bu hedefin basitliği, daha karmaşık karşılaştırmalı kayıplardan önemli bir ayırt edicidir.

### 4. Eğitim ve Performans Analizi

MAE'nin ön eğitim aşaması, son derece ölçeklenebilir ve verimli olacak şekilde tasarlanmıştır:
1.  **Ön Eğitim:** Bir ViT modeli, büyük bir etiketlenmemiş görüntü veri kümesi (örn., ImageNet-1K) üzerinde bir MAE olarak önceden eğitilir. Kodlayıcı, seyrek örneklenmiş görüntü yamalarından özellikler çıkarmayı öğrenir ve kod çözücü, bu özelliklere ve konumsal bilgilere dayanarak maskelenmiş bölgeleri yeniden yapılandırmayı öğrenir. Yüksek maskeleme oranı burada çok önemlidir, çünkü modeli yalnızca yerel piksel değerlerini enterpolasyon yapmak yerine daha üst düzey anlamsal anlayış öğrenmeye iter.
2.  **İnce Ayar:** Ön eğitimden sonra, kod çözücü atılır. Önceden eğitilmiş kodlayıcı (ViT omurgası) daha sonra basit bir sınıflandırma başlığıyla birleştirilir ve belirli bir aşağı akış görevi için (örn., ImageNet üzerinde görüntü sınıflandırma, COCO üzerinde nesne algılama, ADE20K üzerinde anlamsal bölümleme) etiketli bir veri kümesi üzerinde ince ayarlanır.

**Temel Performans Analizleri:**
*   **Üstün Ölçeklenebilirlik:** MAE, kodlayıcının yamaların yalnızca %25'ini işlemesi nedeniyle çok büyük ViT modellerini (örn., ViT-Huge, ViT-G) olağanüstü verimlilikle önceden eğitebilir. Bu, tam görüntü işleme yöntemlerine kıyasla hesaplama süresini ve bellek kullanımını önemli ölçüde azaltır.
*   **Yüksek Performans:** MAE ile önceden eğitilmiş modeller, çeşitli karşılaştırmalarda en son teknoloji sonuçlar elde eder, genellikle denetimli ön eğitimi veya diğer SSL yöntemlerini geride bırakır, özellikle ölçeklendirildiğinde. Örneğin, MAE ile eğitilen ViT-Large, yalnızca 300 ön eğitim dönemi ile ImageNet-1K'de %87.8 en yüksek-1 doğruluğuna ulaşır.
*   **Maskelemeye Karşı Dayanıklılık:** Yüksek maskeleme oranı (%75) şaşırtıcı derecede etkili oldu ve daha düşük maskeleme oranlarından daha iyi temsiller sağladı. Bu, kapsamlı eksik bilginin modeli daha soyut ve küresel görsel özellikleri yakalamaya zorladığını göstermektedir.
*   **Basitlik ve Evrensellik:** MAE'nin basit piksel yeniden yapılandırma görevi ve MSE kaybı, ön eğitim sırasında göreve özgü değişiklikler gerektirmeden farklı görüş görevlerinde çalışacak kadar geneldir.

### 5. Kod Örneği

Bu kavramsal Python kod parçacığı, bir görüntüyü yamalara ayırma ve ardından bu yamaların yüksek bir yüzdesini rastgele maskeleme temel fikrini göstermektedir; bu, MAE ön eğitim sürecinin temel bir parçasıdır.

```python
import numpy as np

def create_image_patches(image, patch_size):
    """
    Bir görüntüyü çakışmayan yamalara böler.
    Argümanlar:
        image (np.array): Giriş görüntüsü (Yükseklik, Genişlik, Kanal).
        patch_size (int): Kare yamanın boyutu.
    Dönüş:
        np.array: Yamalar dizisi (Yama_Sayısı, yama_boyutu, yama_boyutu, Kanal).
    """
    h, w, c = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, \
        "Görüntü boyutları yama_boyutuna bölünebilir olmalıdır."

    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size, :]
            patches.append(patch)
    return np.array(patches)

def apply_random_masking(patches, mask_ratio):
    """
    Belirli bir görüntü yamaları kümesine rastgele maskeleme uygular.
    Argümanlar:
        patches (np.array): Yamalar dizisi (Yama_Sayısı, yama_boyutu, yama_boyutu, Kanal).
        mask_ratio (float): Maskelenecek yamaların oranı (örn. %75 için 0.75).
    Dönüş:
        tuple: (görünür_yamalar, maskeli_indisler, maskesiz_indisler, tüm_indisler)
    """
    num_patches = patches.shape[0]
    num_masked = int(num_patches * mask_ratio)

    # İndislerin rastgele permütasyonunu oluştur
    full_indices = np.arange(num_patches)
    np.random.shuffle(full_indices)

    # Maskelenmiş ve maskesiz yamalar için indisleri seç
    masked_indices = full_indices[:num_masked]
    unmasked_indices = full_indices[num_masked:]

    # Görünür yamaları al (maskelenmemiş olanlar)
    visible_patches = patches[unmasked_indices]

    # MAE için, maskelenmiş yamalar kod çözücü girişinde özel bir [MASK] belirteci ile değiştirilir.
    # Burada, sadece indisleri ve görünür yamaları döndürüyoruz.
    return visible_patches, masked_indices, unmasked_indices, full_indices

# --- Örnek Kullanım ---
if __name__ == "__main__":
    # Sahte bir görüntü oluştur (örn. basitlik için gri tonlamalı bir görüntü)
    dummy_image = np.random.rand(224, 224, 3) # 224x224 RGB görüntü
    patch_size = 16
    mask_ratio = 0.75

    print(f"Orijinal görüntü boyutu: {dummy_image.shape}")

    # 1. Yamaları oluştur
    image_patches = create_image_patches(dummy_image, patch_size)
    print(f"Yama sayısı: {image_patches.shape[0]}, yama boyutu: {image_patches.shape[1:]}")

    # 2. Maskelemeyi uygula
    visible_patches, masked_indices, unmasked_indices, _ = \
        apply_random_masking(image_patches, mask_ratio)

    print(f"Görünür yama sayısı: {len(visible_patches)}")
    print(f"Maskelenmiş yama sayısı: {len(masked_indices)}")
    print(f"Uygulanan maskeleme oranı: {len(masked_indices) / image_patches.shape[0]:.2f}")

    # visible_patches, MAE kodlayıcısına beslenir
    # masked_indices, kod çözücü girişine maske belirteçleri yerleştirmek için kullanılır
    # image_patches[masked_indices], yeniden yapılandırma kaybı için gerçek değer olur

(Kod örneği bölümünün sonu)
```

### 6. Sonuç

MAE: Maskelenmiş Otomatik Kodlayıcılar Ölçeklenebilir Görsel Öğrenicilerdir, bilgisayar görüşü için kendi kendine denetimli öğrenme alanını önemli ölçüde ileriye taşımıştır. NLP'den gelen son derece başarılı maskelenmiş modelleme paradigmasını görüşe çevirerek, MAE, büyük Görsel Transformatör modellerini önceden eğitmek için zarif, verimli ve güçlü bir yöntem sunar. Temel yenilikleri – yüksek maskeleme oranı, asimetrik bir kodlayıcı-kod çözücü mimarisi ve basit bir piksel yeniden yapılandırma hedefi – benzeri görülmemiş ölçeklenebilirlik ile çok miktarda etiketlenmemiş veriden zengin, aktarılabilir görsel temsiller öğrenmesini sağlar. MAE'nin çeşitli aşağı akış görevlerindeki performansı, etkinliğini doğrular ve görüşte kendi kendine denetimli ön eğitim için yeni bir referans noktası oluşturur. MAE'nin kavramsal basitliği ve ampirik başarısı, daha büyük ve daha yetenekli genel amaçlı görüş modellerine yönelik gelecekteki araştırmaların önünü açar, potansiyel olarak büyük etiketli veri kümelerine olan bağımlılığı azaltır ve yüksek performanslı görsel yapay zekaya erişimi demokratikleştirir.
