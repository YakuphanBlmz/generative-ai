# Singular Value Decomposition (SVD) in AI

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Mathematical Foundations of SVD](#2-mathematical-foundations-of-svd)
- [3. Applications of SVD in AI](#3-applications-of-svd-in-ai)
    - [3.1. Dimensionality Reduction and Principal Component Analysis (PCA)](#31-dimensionality-reduction-and-principal-component-analysis-pca)
    - [3.2. Noise Reduction and Data Compression](#32-noise-reduction-and-data-compression)
    - [3.3. Recommender Systems](#33-recommender-systems)
    - [3.4. Natural Language Processing (NLP)](#34-natural-language-processing-nlp)
    - [3.5. Image Processing and Computer Vision](#35-image-processing-and-computer-vision)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
**Singular Value Decomposition (SVD)** is a powerful and widely utilized matrix factorization technique that has found extensive application across various fields, particularly within the realm of **Artificial Intelligence (AI)** and **Machine Learning (ML)**. At its core, SVD decomposes a rectangular matrix into three simpler matrices, revealing the underlying structure of the data and providing insights into its most significant components. This decomposition is not merely a mathematical curiosity; it serves as a fundamental tool for tasks such as **dimensionality reduction**, **noise reduction**, **data compression**, and the extraction of **latent features** from complex datasets. Its robustness and generality make it invaluable for handling high-dimensional data, which is ubiquitous in modern AI applications, from image and text processing to recommender systems and bioinformatics. Understanding SVD is crucial for anyone delving into the theoretical underpinnings and practical implementations of many advanced AI algorithms.

<a name="2-mathematical-foundations-of-svd"></a>
## 2. Mathematical Foundations of SVD
SVD is a factorization of a real or complex matrix. For a given $m \times n$ matrix $A$, the SVD is given by the formula:

$A = U \Sigma V^T$

Where:
*   $U$ is an $m \times m$ **orthogonal matrix** whose columns are the **left singular vectors** of $A$. These vectors form an orthonormal basis for the column space of $A$.
*   $\Sigma$ (Sigma) is an $m \times n$ **diagonal matrix** whose diagonal entries $\sigma_i$ are the **singular values** of $A$. These singular values are real, non-negative, and typically ordered in decreasing magnitude ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$, where $k = \min(m, n)$). The singular values quantify the "importance" or "strength" of each corresponding singular vector pair.
*   $V^T$ (V transpose) is an $n \times n$ **orthogonal matrix** whose rows are the **right singular vectors** of $A$. These vectors form an orthonormal basis for the row space of $A$. ($V$ itself is an $n \times n$ orthogonal matrix whose columns are the right singular vectors).

The singular values $\sigma_i$ are the square roots of the eigenvalues of $A^T A$ (and $A A^T$). The columns of $U$ are the eigenvectors of $A A^T$, and the columns of $V$ are the eigenvectors of $A^T A$. Unlike eigenvalue decomposition, SVD is applicable to any $m \times n$ matrix, regardless of whether it is square or symmetric, making it far more versatile for real-world data matrices.

The significance of SVD lies in its ability to decompose a matrix into a set of **orthogonal components** weighted by their singular values. By retaining only the largest singular values and their corresponding singular vectors (i.e., a **low-rank approximation**), one can approximate the original matrix while discarding less significant information, which is central to many of its applications in AI.

<a name="3-applications-of-svd-in-ai"></a>
## 3. Applications of SVD in AI
SVD's ability to decompose complex data matrices into their fundamental components makes it a cornerstone technique in numerous AI applications.

<a name="31-dimensionality-reduction-and-principal-component-analysis-pca"></a>
### 3.1. Dimensionality Reduction and Principal Component Analysis (PCA)
One of the most prominent applications of SVD is in **dimensionality reduction**, where it forms the mathematical basis for **Principal Component Analysis (PCA)**. In AI, datasets often contain a high number of features, many of which may be redundant or noisy. PCA uses SVD to transform the data into a new coordinate system, where the data is projected onto orthogonal axes (principal components) that capture the maximum variance. By selecting only the principal components corresponding to the largest singular values, the dimensionality of the dataset can be significantly reduced while retaining most of the important information. This not only reduces computational load but also helps in mitigating the **curse of dimensionality** and improving model generalization.

<a name="32-noise-reduction-and-data-compression"></a>
### 3.2. Noise Reduction and Data Compression
SVD is highly effective for **noise reduction** and **data compression**. In many real-world datasets, especially those derived from sensors or measurements, noise is inherent. By performing an SVD and then reconstructing the matrix using only a subset of the largest singular values and their corresponding vectors, one can effectively filter out the noise components, which are often associated with smaller singular values. This **low-rank approximation** also leads to significant **data compression**. For instance, in image processing, an image matrix can be approximated with fewer components, drastically reducing its storage size with minimal perceptual loss.

<a name="33-recommender-systems"></a>
### 3.3. Recommender Systems
SVD plays a crucial role in **recommender systems**, particularly in techniques like **collaborative filtering**. In systems where users rate items (e.g., movies, products), the user-item interaction matrix is often sparse and high-dimensional. SVD can decompose this matrix into latent factors representing underlying preferences or characteristics of users and items. By reducing the dimensionality, SVD helps to identify hidden patterns and relationships, enabling the system to predict how a user might rate an unrated item. This approach is fundamental to **matrix factorization** models that power many modern recommendation engines.

<a name="34-natural-language-processing-nlp)"></a>
### 3.4. Natural Language Processing (NLP)
In **Natural Language Processing (NLP)**, SVD is instrumental in techniques like **Latent Semantic Analysis (LSA)**. LSA applies SVD to a term-document matrix (or term-context matrix) to discover **latent semantic relationships** between words and documents. By reducing the dimensionality of this matrix, SVD helps to overcome issues like synonymy (different words having the same meaning) and polysemy (one word having multiple meanings) by grouping related terms and documents in a lower-dimensional semantic space. This allows for more robust information retrieval, document clustering, and topic modeling.

<a name="35-image-processing-and-computer-vision"></a>
### 3.5. Image Processing and Computer Vision
Beyond general data compression, SVD has specific applications in **image processing** and **computer vision**. It can be used for **image compression**, as mentioned earlier, by reconstructing an image with fewer singular values. Additionally, SVD can be used for tasks like image denoising, watermarking, and even facial recognition (related to **Eigenfaces**, which uses PCA, itself based on SVD). The decomposition helps in extracting dominant features and patterns from image data.

<a name="4-code-example"></a>
## 4. Code Example
Here's a short Python example demonstrating SVD using NumPy on a sample matrix.

```python
import numpy as np

# Create a sample matrix A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print("Original Matrix A:")
print(A)

# Perform Singular Value Decomposition
U, s, Vt = np.linalg.svd(A)

# s is a 1D array of singular values, convert to a diagonal matrix for reconstruction
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)

print("\nU (Left Singular Vectors):")
print(U)
print("\ns (Singular Values - 1D array):")
print(s)
print("\nSigma (Diagonal Matrix of Singular Values):")
print(Sigma)
print("\nVt (Right Singular Vectors Transpose):")
print(Vt)

# Verify the decomposition by reconstructing A
# Note: For reconstruction with full matrices, use U @ Sigma @ Vt
# If we used the 'reduced' SVD from numpy, we would need to handle Sigma's shape
A_reconstructed = U @ Sigma @ Vt

print("\nReconstructed Matrix A (U @ Sigma @ Vt):")
print(A_reconstructed)

# Example of low-rank approximation (e.g., using only the first 2 singular values)
k = 2 # Number of singular values to keep
U_k = U[:, :k]
Sigma_k = np.diag(s[:k])
Vt_k = Vt[:k, :]

A_low_rank = U_k @ Sigma_k @ Vt_k

print(f"\nLow-rank approximation of A (k={k}):")
print(A_low_rank)

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion
**Singular Value Decomposition (SVD)** stands as a cornerstone linear algebra technique with profound implications and widespread applications in **Artificial Intelligence**. Its ability to elegantly decompose any matrix into orthogonal components and singular values provides a fundamental mechanism for uncovering the intrinsic structure of data. From drastically reducing the **dimensionality** of complex datasets and filtering out **noise**, to enabling sophisticated **recommender systems** and extracting **latent semantic meanings** in natural language, SVD's versatility is unmatched. As AI continues to grapple with increasingly larger and more intricate datasets, the principles and practical applications of SVD will remain indispensable for efficient data processing, insightful feature extraction, and the development of robust, scalable AI models. Its enduring relevance underscores its foundational importance in the evolving landscape of intelligent systems.

---
<br>

<a name="türkçe-içerik"></a>
## Yapay Zekada Tekil Değer Ayrışımı (SVD)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. SVD'nin Matematiksel Temelleri](#2-svdnin-matematiksel-temelleri)
- [3. SVD'nin Yapay Zekadaki Uygulamaları](#3-svdnin-yapay-zekadaki-uygulamaları)
    - [3.1. Boyut İndirgeme ve Temel Bileşen Analizi (PCA)](#31-boyut-indirgeme-ve-temel-bileşen-analizi-pca)
    - [3.2. Gürültü Azaltma ve Veri Sıkıştırma](#32-gürültü-azaltma-ve-veri-sıkıştırma)
    - [3.3. Tavsiye Sistemleri](#33-tavsiye-sistemleri)
    - [3.4. Doğal Dil İşleme (NLP)](#34-doğal-dil-işleme-nlp)
    - [3.5. Görüntü İşleme ve Bilgisayar Görüşü](#35-görüntü-i̇şleme-ve-bilgisayar-görüşü)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
**Tekil Değer Ayrışımı (SVD)**, çeşitli alanlarda, özellikle de **Yapay Zeka (YZ)** ve **Makine Öğrenimi (MÖ)** dünyasında yaygın olarak kullanılan güçlü bir matris çarpanlara ayırma tekniğidir. Özünde, SVD dikdörtgen bir matrisi üç daha basit matrise ayırarak verinin temel yapısını ortaya çıkarır ve en önemli bileşenleri hakkında bilgi sağlar. Bu ayrışım sadece matematiksel bir merak değil; **boyut indirgeme**, **gürültü azaltma**, **veri sıkıştırma** ve karmaşık veri kümelerinden **gizli özelliklerin** çıkarılması gibi görevler için temel bir araç olarak hizmet eder. Sağlamlığı ve genelliği, görüntü ve metin işlemeden tavsiye sistemlerine ve biyoinformatiğe kadar modern yapay zeka uygulamalarında yaygın olan yüksek boyutlu verileri işlemek için onu paha biçilmez kılar. SVD'yi anlamak, birçok gelişmiş yapay zeka algoritmasının teorik temelini ve pratik uygulamalarını inceleyen herkes için çok önemlidir.

<a name="2-svdnin-matematiksel-temelleri"></a>
## 2. SVD'nin Matematiksel Temelleri
SVD, bir reel veya karmaşık matrisin çarpanlara ayrışmasıdır. Verilen bir $m \times n$ boyutlu $A$ matrisi için SVD formülü şu şekildedir:

$A = U \Sigma V^T$

Burada:
*   $U$, sütunları $A$'nın **sol tekil vektörleri** olan $m \times m$ boyutlu bir **ortogonal matristir**. Bu vektörler, $A$'nın sütun uzayı için bir ortonormal baz oluşturur.
*   $\Sigma$ (Sigma), köşegen girişleri $\sigma_i$ olarak $A$'nın **tekil değerleri** olan $m \times n$ boyutlu bir **köşegen matristir**. Bu tekil değerler reel, negatif olmayan ve tipik olarak azalan büyüklük sırasına göre düzenlenir ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$, burada $k = \min(m, n)$). Tekil değerler, her karşılık gelen tekil vektör çiftinin "önemini" veya "gücünü" nicelendirir.
*   $V^T$ (V devriği), satırları $A$'nın **sağ tekil vektörleri** olan $n \times n$ boyutlu bir **ortogonal matristir**. Bu vektörler, $A$'nın satır uzayı için bir ortonormal baz oluşturur. ($V$ kendisi, sütunları sağ tekil vektörler olan $n \times n$ boyutlu ortogonal bir matristir).

Tekil değerler $\sigma_i$, $A^T A$ (ve $A A^T$) matrislerinin özdeğerlerinin karekökleridir. $U$'nun sütunları $A A^T$'nin özvektörleri, $V$'nin sütunları ise $A^T A$'nın özvektörleridir. Özdeğer ayrışımının aksine, SVD herhangi bir $m \times n$ boyutlu matrise, kare veya simetrik olmasına bakılmaksızın uygulanabilir, bu da onu gerçek dünya veri matrisleri için çok daha çok yönlü hale getirir.

SVD'nin önemi, bir matrisi tekil değerleri tarafından ağırlıklandırılmış bir **ortogonal bileşenler** kümesine ayrıştırma yeteneğinde yatar. Yalnızca en büyük tekil değerleri ve bunlara karşılık gelen tekil vektörleri (yani, bir **düşük-rank yaklaşımı**) koruyarak, daha az önemli bilgiyi atarken orijinal matrisi yaklaşık olarak hesaplamak mümkündür, ki bu, yapay zekadaki birçok uygulamasının merkezindedir.

<a name="3-svdnin-yapay-zekadaki-uygulamaları"></a>
## 3. SVD'nin Yapay Zekadaki Uygulamaları
SVD'nin karmaşık veri matrislerini temel bileşenlerine ayrıştırma yeteneği, sayısız yapay zeka uygulamasında temel bir teknik olmasını sağlar.

<a name="31-boyut-indirgeme-ve-temel-bileşen-analizi-pca"></a>
### 3.1. Boyut İndirgeme ve Temel Bileşen Analizi (PCA)
SVD'nin en öne çıkan uygulamalarından biri, **Temel Bileşen Analizi (PCA)**'nın matematiksel temelini oluşturan **boyut indirgeme**dir. Yapay zekada, veri kümeleri genellikle çok sayıda özellik içerir ve bunların çoğu gereksiz veya gürültülü olabilir. PCA, SVD'yi kullanarak veriyi yeni bir koordinat sistemine dönüştürür; burada veri, maksimum varyansı yakalayan ortogonal eksenlere (temel bileşenler) yansıtılır. En büyük tekil değerlere karşılık gelen temel bileşenler seçilerek, verinin boyutu önemli ölçüde azaltılırken önemli bilgilerin çoğu korunur. Bu sadece hesaplama yükünü azaltmakla kalmaz, aynı zamanda **boyutluluk lanetinin** etkilerini hafifletmeye ve modelin genelleştirme yeteneğini geliştirmeye yardımcı olur.

<a name="32-gürültü-azaltma-ve-veri-sıkıştırma"></a>
### 3.2. Gürültü Azaltma ve Veri Sıkıştırma
SVD, **gürültü azaltma** ve **veri sıkıştırma** için oldukça etkilidir. Sensörlerden veya ölçümlerden türetilenler gibi birçok gerçek dünya veri kümesinde gürültü doğaldır. Bir SVD gerçekleştirip ardından matrisi yalnızca en büyük tekil değerlerin ve bunlara karşılık gelen vektörlerin bir alt kümesini kullanarak yeniden yapılandırarak, genellikle daha küçük tekil değerlerle ilişkili olan gürültü bileşenleri etkili bir şekilde filtrelenebilir. Bu **düşük-rank yaklaşımı** aynı zamanda önemli **veri sıkıştırma** sağlar. Örneğin, görüntü işlemede, bir görüntü matrisi daha az bileşenle yaklaştırılabilir, bu da depolama boyutunu minimal algısal kayıpla büyük ölçüde azaltır.

<a name="33-tavsiye-sistemleri"></a>
### 3.3. Tavsiye Sistemleri
SVD, **tavsiye sistemlerinde**, özellikle **işbirlikçi filtreleme** gibi tekniklerde çok önemli bir rol oynar. Kullanıcıların öğeleri (örneğin, filmler, ürünler) derecelendirdiği sistemlerde, kullanıcı-öğe etkileşim matrisi genellikle seyrektir ve yüksek boyutludur. SVD, bu matrisi, kullanıcıların ve öğelerin temel tercihlerini veya özelliklerini temsil eden **gizli faktörlere** ayrıştırabilir. Boyutluluğu azaltarak, SVD gizli desenleri ve ilişkileri belirlemeye yardımcı olur, bu da sistemin bir kullanıcının derecelendirilmemiş bir öğeyi nasıl değerlendireceğini tahmin etmesini sağlar. Bu yaklaşım, birçok modern tavsiye motorunu destekleyen **matris çarpanlara ayırma** modellerinin temelini oluşturur.

<a name="34-doğal-dil-i̇şleme-nlp)"></a>
### 3.4. Doğal Dil İşleme (NLP)
**Doğal Dil İşleme (NLP)**'de SVD, **Gizli Anlamsal Analiz (LSA)** gibi tekniklerde önemli bir rol oynar. LSA, kelimeler ve belgeler arasındaki **gizli anlamsal ilişkileri** keşfetmek için bir terim-belge matrisine (veya terim-bağlam matrisine) SVD uygular. Bu matrisin boyutluluğunu azaltarak, SVD eşanlamlılık (farklı kelimelerin aynı anlama sahip olması) ve çokanlamlılık (bir kelimenin birden fazla anlama sahip olması) gibi sorunların üstesinden gelmeye yardımcı olur, ilgili terimleri ve belgeleri daha düşük boyutlu bir anlamsal uzayda gruplandırır. Bu, daha sağlam bilgi alma, belge kümeleme ve konu modelleme sağlar.

<a name="35-görüntü-i̇şleme-ve-bilgisayar-görüşü"></a>
### 3.5. Görüntü İşleme ve Bilgisayar Görüşü
Genel veri sıkıştırmanın ötesinde, SVD'nin **görüntü işleme** ve **bilgisayar görüşü** alanında özel uygulamaları vardır. Daha önce belirtildiği gibi, daha az tekil değerle bir görüntüyü yeniden yapılandırarak **görüntü sıkıştırma** için kullanılabilir. Ek olarak, SVD görüntü denoising (gürültü giderme), filigranlama ve hatta yüz tanıma (PCA ile ilişkili olan **Eigenfaces** ile ilgili) gibi görevler için kullanılabilir. Ayrışım, görüntü verilerinden baskın özellikleri ve desenleri çıkarmaya yardımcı olur.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği
Burada, NumPy kullanarak örnek bir matris üzerinde SVD'yi gösteren kısa bir Python örneği bulunmaktadır.

```python
import numpy as np

# Örnek bir A matrisi oluşturun
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print("Orijinal Matris A:")
print(A)

# Tekil Değer Ayrışımını gerçekleştirin
U, s, Vt = np.linalg.svd(A)

# s tekil değerlerin 1D dizisidir, yeniden yapılandırma için köşegen matrise dönüştürün
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)

print("\nU (Sol Tekil Vektörler):")
print(U)
print("\ns (Tekil Değerler - 1D dizi):")
print(s)
print("\nSigma (Tekil Değerlerin Köşegen Matrisi):")
print(Sigma)
print("\nVt (Sağ Tekil Vektörlerin Transpozu):")
print(Vt)

# Ayrışımı A'yı yeniden yapılandırarak doğrulayın
# Not: Tam matrislerle yeniden yapılandırma için U @ Sigma @ Vt kullanın
# Eğer numpy'dan 'indirgenmiş' SVD kullanılsaydı, Sigma'nın şeklini ele almamız gerekirdi.
A_yeniden_yapilandirilmis = U @ Sigma @ Vt

print("\nYeniden Yapılandırılmış Matris A (U @ Sigma @ Vt):")
print(A_yeniden_yapilandirilmis)

# Düşük-rank yaklaşımı örneği (örneğin, yalnızca ilk 2 tekil değeri kullanarak)
k = 2 # Saklanacak tekil değer sayısı
U_k = U[:, :k]
Sigma_k = np.diag(s[:k])
Vt_k = Vt[:k, :]

A_düşük_rank = U_k @ Sigma_k @ Vt_k

print(f"\nA'nın düşük-rank yaklaşımı (k={k}):")
print(A_düşük_rank)

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç
**Tekil Değer Ayrışımı (SVD)**, **Yapay Zeka**'da derin etkileri ve yaygın uygulamaları olan temel bir lineer cebir tekniğidir. Herhangi bir matrisi ortogonal bileşenlere ve tekil değerlere zarifçe ayrıştırma yeteneği, verinin içsel yapısını ortaya çıkarmak için temel bir mekanizma sağlar. Karmaşık veri kümelerinin **boyutluluğunu** önemli ölçüde azaltmaktan ve **gürültüyü** filtrelemekten, gelişmiş **tavsiye sistemlerini** etkinleştirmeye ve doğal dildeki **gizli anlamsal anlamları** çıkarmaya kadar, SVD'nin çok yönlülüğü benzersizdir. Yapay zeka giderek daha büyük ve daha karmaşık veri kümeleriyle başa çıkmaya devam ettikçe, SVD'nin prensipleri ve pratik uygulamaları, verimli veri işleme, içgörülü özellik çıkarımı ve sağlam, ölçeklenebilir yapay zeka modellerinin geliştirilmesi için vazgeçilmez kalacaktır. Kalıcı alaka düzeyi, gelişen akıllı sistemler manzarasındaki temel önemini vurgulamaktadır.