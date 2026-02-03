# Singular Value Decomposition (SVD) in AI

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Theoretical Foundations of Singular Value Decomposition](#2-theoretical-foundations-of-singular-value-decomposition)
  - [2.1. Mathematical Formulation](#21-mathematical-formulation)
  - [2.2. Interpretation of Components](#22-interpretation-of-components)
  - [2.3. Connection to Eigenvalue Decomposition](#23-connection-to-eigenvalue-decomposition)
- [3. Applications of SVD in AI](#3-applications-of-svd-in-ai)
  - [3.1. Dimensionality Reduction (PCA)](#31-dimensionality-reduction-pca)
  - [3.1. Recommender Systems](#32-recommender-systems)
  - [3.3. Natural Language Processing (LSA)](#33-natural-language-processing-lsa)
  - [3.4. Image Compression and Denoising](#34-image-compression-and-denoising)
  - [3.5. Noise Reduction and Data Compression](#35-noise-reduction-and-data-compression)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
**Singular Value Decomposition (SVD)** is a fundamental matrix factorization technique with widespread applications across various scientific and engineering disciplines, including the rapidly evolving field of **Artificial Intelligence (AI)**. At its core, SVD decomposes a given matrix into three simpler matrices, revealing the underlying structure and essential characteristics of the data it represents. This decomposition is particularly powerful because it can be applied to any arbitrary matrix (rectangular or square), unlike **eigenvalue decomposition** which is restricted to square matrices.

In the context of AI, SVD serves as a versatile tool for tasks such as **dimensionality reduction**, **data compression**, **noise reduction**, and the extraction of **latent features** from complex datasets. Its ability to effectively capture the most significant variance in data while discarding less important information makes it invaluable for preprocessing steps, improving model performance, and enhancing the interpretability of high-dimensional data. From **natural language processing** to **recommender systems** and **computer vision**, SVD underpins several classical and contemporary AI algorithms, offering both theoretical elegance and practical utility. This document will delve into the theoretical underpinnings of SVD and explore its diverse applications within the realm of AI.

## 2. Theoretical Foundations of Singular Value Decomposition
### 2.1. Mathematical Formulation
SVD states that any real or complex matrix $A$ of dimensions $m \times n$ can be decomposed into the product of three other matrices:

$A = U \Sigma V^T$

Where:
*   $U$: An $m \times m$ **orthogonal matrix** whose columns are the **left singular vectors** of $A$. These vectors form an orthonormal basis for the column space of $A$.
*   $\Sigma$: An $m \times n$ **diagonal matrix** containing the **singular values** of $A$ on its diagonal, ordered from largest to smallest. The off-diagonal elements are zero. The singular values ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$) quantify the "strength" or "importance" of each corresponding singular vector pair in representing the original data. $r$ is the rank of matrix $A$.
*   $V^T$: The **transpose** of an $n \times n$ **orthogonal matrix** $V$, whose columns are the **right singular vectors** of $A$. These vectors form an orthonormal basis for the row space of $A$. ($V^T$ is often referred to as $V$ transpose or $V$ dagger).

The orthogonality property means that $U^T U = I$ and $V^T V = I$, where $I$ is the identity matrix. This ensures that the transformations preserve lengths and angles, making the singular vectors excellent bases for representing the data.

### 2.2. Interpretation of Components
The singular values in $\Sigma$ directly indicate how much variance each corresponding singular vector captures. Larger singular values correspond to more significant dimensions or features in the data. By retaining only the largest singular values and their corresponding singular vectors (i.e., using a truncated SVD), we can approximate the original matrix with a lower-rank matrix, effectively achieving **dimensionality reduction** or **data compression**. The left singular vectors ($U$) often represent features of the rows (e.g., document topics in text analysis), while the right singular vectors ($V$) represent features of the columns (e.g., word importance).

### 2.3. Connection to Eigenvalue Decomposition
While SVD is more general, it is intimately related to **eigenvalue decomposition**. For a symmetric positive semi-definite matrix $X$, its eigenvalue decomposition is $X = P \Lambda P^T$. The connection to SVD comes from considering $A A^T$ and $A^T A$:
*   The columns of $U$ (left singular vectors) are the **eigenvectors** of $A A^T$.
*   The columns of $V$ (right singular vectors) are the **eigenvectors** of $A^T A$.
*   The non-zero singular values of $A$ are the square roots of the non-zero **eigenvalues** of both $A A^T$ and $A^T A$.

This relationship highlights that SVD essentially finds the principal axes of the data's variance, similar to **Principal Component Analysis (PCA)**, which is often implemented using SVD.

## 3. Applications of SVD in AI

SVD's ability to decompose a matrix into its fundamental components makes it a powerful tool for understanding, manipulating, and compressing data in AI systems.

### 3.1. Dimensionality Reduction (PCA)
One of the most prominent applications of SVD is in **dimensionality reduction**. SVD provides an elegant and computationally stable way to perform **Principal Component Analysis (PCA)**. In PCA, we seek to project high-dimensional data onto a lower-dimensional subspace while preserving as much variance as possible. When SVD is applied to the data matrix (or its covariance matrix), the right singular vectors ($V$) directly correspond to the principal components, and the singular values ($\Sigma$) indicate the amount of variance explained by each component. By keeping only the top `k` singular values and their corresponding vectors, we can reduce the dimensionality of the data from $n$ to $k$, which can significantly speed up subsequent machine learning algorithms and mitigate the **curse of dimensionality**.

### 3.2. Recommender Systems
SVD is a cornerstone of many **recommender systems**, particularly those based on **collaborative filtering**. In such systems, user-item interaction data (e.g., ratings) is typically represented as a large, sparse matrix. SVD can decompose this matrix into latent factors representing abstract features of users and items. For example, if a matrix $R$ contains user ratings for movies, SVD decomposes $R$ into $U \Sigma V^T$. The columns of $U$ can be interpreted as user profiles in a latent feature space, and the columns of $V$ as movie profiles in the same space. By multiplying a user's latent profile with a movie's latent profile, we can predict a rating for movies the user hasn't seen, thereby generating recommendations. This approach is often called **latent factor models** or **matrix factorization**.

### 3.3. Natural Language Processing (LSA)
In **Natural Language Processing (NLP)**, SVD is central to **Latent Semantic Analysis (LSA)**. LSA is a technique used to analyze relationships between a set of documents and the terms they contain by producing a set of "concepts" or "topics" related to the documents and terms. Typically, a **term-document matrix** is constructed, where rows represent terms and columns represent documents (or vice versa), and entries indicate term frequency (TF-IDF). Applying SVD to this matrix reveals latent semantic relationships. The singular vectors capture underlying "topics" or "themes" in the corpus, allowing for **topic modeling**, improved **information retrieval**, and handling of **synonymy** and **polysemy** issues by mapping terms and documents to a lower-dimensional semantic space.

### 3.4. Image Compression and Denoising
SVD finds practical application in **image processing**, particularly for **image compression** and **denoising**. An image can be represented as a matrix of pixel values. By performing SVD on this matrix, and then reconstructing the image using only a subset of the largest singular values and their corresponding vectors, significant data compression can be achieved. Lower-rank approximations essentially retain the most visually important information while discarding finer details. This can drastically reduce storage space. Similarly, by filtering out small singular values, which often correspond to noise, SVD can effectively denoise images without losing much of the original signal quality.

### 3.5. Noise Reduction and Data Compression
Beyond specific domains, SVD is generally valuable for **noise reduction** and **data compression**. Small singular values in the $\Sigma$ matrix often correspond to noise or less significant variations in the data. By setting these small singular values to zero or simply discarding them when reconstructing the matrix, SVD acts as a powerful low-pass filter, effectively separating signal from noise. This principle extends to general data compression: by representing the original data with fewer singular values and vectors, we create a compact, lower-rank approximation that retains most of the essential information, leading to significant memory savings and faster processing for large datasets.

## 4. Code Example
This Python example demonstrates how SVD can be used for dimensionality reduction and data approximation using `numpy`. We'll create a simple matrix, perform SVD, and then reconstruct it using a reduced number of singular values.

```python
import numpy as np

# Create a sample matrix A (e.g., representing some data)
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

print("Original Matrix A:\n", A)
print("-" * 30)

# Perform Singular Value Decomposition
U, s, Vt = np.linalg.svd(A)

# s contains the singular values.
# U is the left singular vectors.
# Vt is the transpose of the right singular vectors.

print("Left Singular Vectors (U):\n", U)
print("-" * 30)
print("Singular Values (s):\n", s)
print("-" * 30)
print("Right Singular Vectors (Vt):\n", Vt)
print("-" * 30)

# Reconstruct the original matrix using all singular values
# Need to convert s to a diagonal matrix with the correct shape
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s) # Place singular values on diagonal

A_reconstructed_full = U @ Sigma @ Vt
print("Reconstructed Matrix (Full Rank):\n", A_reconstructed_full)
print("-" * 30)

# Perform dimensionality reduction by keeping only the top k singular values
k = 2 # Let's keep the top 2 singular values

# Truncate U, s, and Vt
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

# Reconstruct the matrix with reduced rank
Sigma_k = np.zeros((k, k))
Sigma_k[:k, :k] = np.diag(s_k)

A_reconstructed_reduced = U_k @ Sigma_k @ Vt_k
print(f"Reconstructed Matrix (Rank {k}):\n", A_reconstructed_reduced)
print("-" * 30)

# Compare the original with the reduced rank approximation
print("Difference between Original and Reduced Rank Approximation:\n", A - A_reconstructed_reduced)
print("-" * 30)

(End of code example section)
```

## 5. Conclusion
Singular Value Decomposition stands as a cornerstone in the mathematical toolkit for **data analysis** and **machine learning**. Its ability to robustly decompose any matrix into orthogonal components and a set of ordered singular values provides an unparalleled method for uncovering the inherent structure of data. In AI, SVD's versatility extends from fundamental tasks like **dimensionality reduction** (via PCA) and **noise filtering** to sophisticated applications in **recommender systems**, **natural language processing (LSA)**, and **computer vision**.

While the advent of deep learning has introduced new paradigms for feature extraction and pattern recognition, SVD remains an invaluable technique for its interpretability, computational efficiency on appropriately sized datasets, and its foundational role in understanding complex data matrices. It provides a clear, mathematically sound approach to understanding variance, identifying latent factors, and achieving data parsimony, thereby continuing to be a relevant and powerful tool in the ever-expanding landscape of Artificial Intelligence.

---
<br>

<a name="türkçe-içerik"></a>
## Yapay Zekada Tekil Değer Ayrışımı (SVD)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Tekil Değer Ayrışımının Teorik Temelleri](#2-tekil-değer-ayrışımının-teorik-temelleri)
  - [2.1. Matematiksel Formülasyon](#21-matematiksel-formülasyon)
  - [2.2. Bileşenlerin Yorumlanması](#22-bileşenlerin-yorumlanması)
  - [2.3. Özdeğer Ayrışımı ile Bağlantısı](#23-özdeğer-ayrışımı-ile-bağlantısı)
- [3. SVD'nin Yapay Zekadaki Uygulamaları](#3-svdnin-yapay-zekadaki-uygulamaları)
  - [3.1. Boyut Azaltma (PCA)](#31-boyut-azaltma-pca)
  - [3.1. Tavsiye Sistemleri](#32-tavsiye-sistemleri)
  - [3.3. Doğal Dil İşleme (LSA)](#33-doğal-dil-işleme-lsa)
  - [3.4. Görüntü Sıkıştırma ve Gürültü Azaltma](#34-görüntü-sıkıştırma-ve-gürültü-azaltma)
  - [3.5. Gürültü Azaltma ve Veri Sıkıştırma](#35-gürültü-azaltma-ve-veri-sıkıştırma)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Tekil Değer Ayrışımı (SVD)**, hızla gelişen **Yapay Zeka (YZ)** alanı da dahil olmak üzere çeşitli bilimsel ve mühendislik disiplinlerinde geniş uygulamalara sahip temel bir matris çarpanlara ayırma tekniğidir. Özünde SVD, verilen bir matrisi üç daha basit matrise ayırarak, temsil ettiği verinin altında yatan yapıyı ve temel özelliklerini ortaya çıkarır. Bu ayrıştırma, **özdeğer ayrışımının** kare matrislerle sınırlı olmasının aksine, herhangi bir rastgele matrise (dikdörtgen veya kare) uygulanabildiği için özellikle güçlüdür.

YZ bağlamında SVD, **boyut azaltma**, **veri sıkıştırma**, **gürültü azaltma** ve karmaşık veri kümelerinden **gizli özelliklerin** çıkarılması gibi görevler için çok yönlü bir araç olarak hizmet eder. Verideki en önemli varyansı etkin bir şekilde yakalama ve daha az önemli bilgiyi atma yeteneği, ön işleme adımları, model performansını iyileştirme ve yüksek boyutlu verinin yorumlanabilirliğini artırma açısından paha biçilmez kılar. **Doğal dil işlemeden** **tavsiye sistemlerine** ve **bilgisayar görüsüne** kadar SVD, hem teorik zarafet hem de pratik fayda sunarak birçok klasik ve çağdaş YZ algoritmasının temelini oluşturur. Bu belge, SVD'nin teorik temellerini inceleyecek ve YZ alanındaki çeşitli uygulamalarını keşfedecektir.

## 2. Tekil Değer Ayrışımının Teorik Temelleri
### 2.1. Matematiksel Formülasyon
SVD, $m \times n$ boyutlarında herhangi bir reel veya karmaşık $A$ matrisinin, üç başka matrisin çarpımı olarak ayrıştırılabileceğini belirtir:

$A = U \Sigma V^T$

Burada:
*   $U$: Sütunları $A$'nın **sol tekil vektörleri** olan bir $m \times m$ **ortogonal matristir**. Bu vektörler, $A$'nın sütun uzayı için bir ortonormal taban oluşturur.
*   $\Sigma$: Köşegeninde $A$'nın **tekil değerlerini** barındıran, en büyüğünden en küçüğüne doğru sıralanmış bir $m \times n$ **köşegen matristir**. Köşegen dışı elemanlar sıfırdır. Tekil değerler ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$), her bir karşılık gelen tekil vektör çiftinin orijinal veriyi temsil etmedeki "gücünü" veya "önemini" nicel olarak belirler. $r$, $A$ matrisinin rankıdır.
*   $V^T$: Sütunları $A$'nın **sağ tekil vektörleri** olan bir $n \times n$ **ortogonal matris** $V$'nin **transpozesidir**. Bu vektörler, $A$'nın satır uzayı için bir ortonormal taban oluşturur. ($V^T$ genellikle $V$ transpoze veya $V$ hançer olarak anılır).

Ortogonallik özelliği, $U^T U = I$ ve $V^T V = I$ olduğu anlamına gelir; burada $I$ birim matristir. Bu, dönüşümlerin uzunlukları ve açıları korumasını sağlar, bu da tekil vektörleri veriyi temsil etmek için mükemmel tabanlar yapar.

### 2.2. Bileşenlerin Yorumlanması
$\Sigma$'daki tekil değerler, her bir karşılık gelen tekil vektörün ne kadar varyans yakaladığını doğrudan gösterir. Daha büyük tekil değerler, verideki daha önemli boyutlara veya özelliklere karşılık gelir. Yalnızca en büyük tekil değerleri ve bunlara karşılık gelen tekil vektörleri (yani, kesilmiş bir SVD kullanarak) tutarak, orijinal matrisi daha düşük ranklı bir matrisle yaklaştırabiliriz, böylece etkili bir şekilde **boyut azaltma** veya **veri sıkıştırma** elde ederiz. Sol tekil vektörler ($U$) genellikle satırların özelliklerini (örneğin, metin analizinde belge konuları) temsil ederken, sağ tekil vektörler ($V$) sütunların özelliklerini (örneğin, kelime önemi) temsil eder.

### 2.3. Özdeğer Ayrışımı ile Bağlantısı
SVD daha genel olsa da, **özdeğer ayrışımı** ile yakından ilişkilidir. Simetrik pozitif yarı-kesin bir $X$ matrisi için özdeğer ayrışımı $X = P \Lambda P^T$'dir. SVD ile bağlantı, $A A^T$ ve $A^T A$'yı dikkate almaktan gelir:
*   $U$'nun sütunları (sol tekil vektörler), $A A^T$'nin **özvektörleridir**.
*   $V$'nin sütunları (sağ tekil vektörler), $A^T A$'nın **özvektörleridir**.
*   $A$'nın sıfır olmayan tekil değerleri, hem $A A^T$'nin hem de $A^T A$'nın sıfır olmayan **özdeğerlerinin** karekökleridir.

Bu ilişki, SVD'nin aslında **Temel Bileşen Analizi (PCA)**'ne benzer şekilde, verinin varyansının temel eksenlerini bulduğunu gösterir; PCA genellikle SVD kullanılarak uygulanır.

## 3. SVD'nin Yapay Zekadaki Uygulamaları

SVD'nin bir matrisi temel bileşenlerine ayrıştırma yeteneği, YZ sistemlerinde veriyi anlama, manipüle etme ve sıkıştırma için güçlü bir araç olmasını sağlar.

### 3.1. Boyut Azaltma (PCA)
SVD'nin en öne çıkan uygulamalarından biri **boyut azaltmadır**. SVD, **Temel Bileşen Analizi (PCA)**'nı gerçekleştirmek için zarif ve hesaplama açısından kararlı bir yol sağlar. PCA'da, yüksek boyutlu verileri mümkün olduğunca fazla varyans koruyarak daha düşük boyutlu bir alt uzaya yansıtmayı amaçlarız. SVD, veri matrisine (veya onun kovaryans matrisine) uygulandığında, sağ tekil vektörler ($V$) doğrudan temel bileşenlere karşılık gelir ve tekil değerler ($\Sigma$) her bileşen tarafından açıklanan varyans miktarını gösterir. Yalnızca en üst `k` tekil değeri ve bunlara karşılık gelen vektörleri tutarak, verinin boyutunu $n$'den $k$'ye düşürebiliriz; bu, sonraki makine öğrenimi algoritmalarını önemli ölçüde hızlandırabilir ve **boyutluluk lanetini** hafifletebilir.

### 3.2. Tavsiye Sistemleri
SVD, birçok **tavsiye sisteminin** temel taşıdır, özellikle de **işbirlikçi filtrelemeye** dayalı olanların. Bu tür sistemlerde, kullanıcı-öğe etkileşim verileri (örneğin, derecelendirmeler) genellikle büyük, seyrek bir matris olarak temsil edilir. SVD, bu matrisi kullanıcıların ve öğelerin soyut özelliklerini temsil eden gizli faktörlere ayrıştırabilir. Örneğin, bir $R$ matrisi kullanıcıların filmlere verdiği derecelendirmeleri içeriyorsa, SVD, $R$'yi $U \Sigma V^T$ olarak ayrıştırır. $U$'nun sütunları, gizli bir özellik uzayındaki kullanıcı profilleri olarak, $V$'nin sütunları ise aynı uzaydaki film profilleri olarak yorumlanabilir. Bir kullanıcının gizli profilini bir filmin gizli profiliyle çarparak, kullanıcının daha önce izlemediği filmler için bir derecelendirme tahmin edebilir ve böylece öneriler oluşturabiliriz. Bu yaklaşıma genellikle **gizli faktör modelleri** veya **matris çarpanlara ayırma** denir.

### 3.3. Doğal Dil İşleme (LSA)
**Doğal Dil İşleme (NLP)**'de SVD, **Gizli Anlamsal Analiz (LSA)** için merkezidir. LSA, bir dizi belge ile içerdikleri terimler arasındaki ilişkileri analiz etmek için kullanılan, belgelerle ve terimlerle ilgili bir dizi "kavram" veya "konu" üreten bir tekniktir. Tipik olarak, satırların terimleri ve sütunların belgeleri (veya tam tersi) temsil ettiği ve girişlerin terim sıklığını (TF-IDF) gösterdiği bir **terim-belge matrisi** oluşturulur. Bu matrise SVD uygulamak, gizli anlamsal ilişkileri ortaya çıkarır. Tekil vektörler, bir külliyattaki temel "konuları" veya "temaları" yakalar, bu da **konu modellemesi**, geliştirilmiş **bilgi erişimi** ve terimlerle belgelerin daha düşük boyutlu bir anlamsal uzaya eşlenmesi yoluyla **eş anlamlılık** ve **çok anlamlılık** sorunlarının ele alınmasını sağlar.

### 3.4. Görüntü Sıkıştırma ve Gürültü Azaltma
SVD, **görüntü işleme**, özellikle **görüntü sıkıştırma** ve **gürültü azaltma** konularında pratik uygulama bulur. Bir görüntü, piksel değerlerinin bir matrisi olarak temsil edilebilir. Bu matris üzerinde SVD gerçekleştirilerek ve ardından görüntüyü yalnızca en büyük tekil değerlerin bir alt kümesi ve bunlara karşılık gelen vektörler kullanılarak yeniden yapılandırarak, önemli veri sıkıştırma elde edilebilir. Düşük ranklı yaklaşımlar, görsel olarak en önemli bilgiyi korurken daha ince ayrıntıları atar. Bu, depolama alanını büyük ölçüde azaltabilir. Benzer şekilde, genellikle gürültüye karşılık gelen küçük tekil değerleri filtreleyerek, SVD, orijinal sinyal kalitesinden fazla ödün vermeden görüntüleri etkili bir şekilde gürültüden arındırabilir.

### 3.5. Gürültü Azaltma ve Veri Sıkıştırma
Belirli alanların ötesinde, SVD genellikle **gürültü azaltma** ve **veri sıkıştırma** için değerlidir. $\Sigma$ matrisindeki küçük tekil değerler genellikle verideki gürültüye veya daha az önemli varyasyonlara karşılık gelir. Bu küçük tekil değerleri sıfırlayarak veya matrisi yeniden yapılandırırken basitçe atarak, SVD güçlü bir alçak geçiren filtre görevi görür ve sinyali gürültüden etkili bir şekilde ayırır. Bu ilke genel veri sıkıştırmaya da uzanır: orijinal veriyi daha az tekil değer ve vektörle temsil ederek, temel bilgilerin çoğunu koruyan kompakt, daha düşük ranklı bir yaklaşım oluştururuz, bu da büyük veri kümeleri için önemli bellek tasarrufu ve daha hızlı işlem sağlar.

## 4. Kod Örneği
Bu Python örneği, SVD'nin `numpy` kullanarak boyut azaltma ve veri yaklaştırma için nasıl kullanılabileceğini göstermektedir. Basit bir matris oluşturacak, SVD uygulayacak ve ardından azaltılmış sayıda tekil değer kullanarak yeniden yapılandıracağız.

```python
import numpy as np

# Örnek bir A matrisi oluşturun (örn. bazı verileri temsil ediyor)
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

print("Orijinal Matris A:\n", A)
print("-" * 30)

# Tekil Değer Ayrışımını Gerçekleştirin
U, s, Vt = np.linalg.svd(A)

# s tekil değerleri içerir.
# U sol tekil vektörlerdir.
# Vt sağ tekil vektörlerin transpozesidir.

print("Sol Tekil Vektörler (U):\n", U)
print("-" * 30)
print("Tekil Değerler (s):\n", s)
print("-" * 30)
print("Sağ Tekil Vektörler (Vt):\n", Vt)
print("-" * 30)

# Orijinal matrisi tüm tekil değerleri kullanarak yeniden oluşturun
# s'yi doğru şekle sahip bir köşegen matrise dönüştürmek gerekiyor
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s) # Tekil değerleri köşegene yerleştirin

A_tam_yeniden_yapılandırılmış = U @ Sigma @ Vt
print("Yeniden Yapılandırılmış Matris (Tam Rank):\n", A_tam_yeniden_yapılandırılmış)
print("-" * 30)

# En üst k tekil değeri tutarak boyut azaltma gerçekleştirin
k = 2 # En üst 2 tekil değeri tutalım

# U, s ve Vt'yi kısaltın
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

# Matrisi azaltılmış rank ile yeniden oluşturun
Sigma_k = np.zeros((k, k))
Sigma_k[:k, :k] = np.diag(s_k)

A_azaltılmış_yeniden_yapılandırılmış = U_k @ Sigma_k @ Vt_k
print(f"Yeniden Yapılandırılmış Matris (Rank {k}):\n", A_azaltılmış_yeniden_yapılandırılmış)
print("-" * 30)

# Orijinal ile azaltılmış rank yaklaşımını karşılaştırın
print("Orijinal ile Azaltılmış Rank Yaklaşımı Arasındaki Fark:\n", A - A_azaltılmış_yeniden_yapılandırılmış)
print("-" * 30)

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
Tekil Değer Ayrışımı, **veri analizi** ve **makine öğrenimi** için matematiksel araç setinin bir köşe taşıdır. Herhangi bir matrisi ortogonal bileşenlere ve sıralı tekil değerlere ayrıştırma yeteneği, verinin içsel yapısını ortaya çıkarmak için eşsiz bir yöntem sunar. YZ'de, SVD'nin çok yönlülüğü, **boyut azaltma** (PCA aracılığıyla) ve **gürültü filtreleme** gibi temel görevlerden **tavsiye sistemleri**, **doğal dil işleme (LSA)** ve **bilgisayar görüsü** gibi karmaşık uygulamalara kadar uzanır.

Derin öğrenmenin ortaya çıkışı, özellik çıkarımı ve örüntü tanıma için yeni paradigmalar sunsa da, SVD, yorumlanabilirliği, uygun boyutlardaki veri kümeleri üzerindeki hesaplama verimliliği ve karmaşık veri matrislerini anlamadaki temel rolü nedeniyle paha biçilmez bir teknik olmaya devam etmektedir. Varyansı anlamak, gizli faktörleri belirlemek ve veri cimriliğini sağlamak için açık, matematiksel olarak sağlam bir yaklaşım sunarak, Yapay Zeka'nın sürekli genişleyen manzarasında ilgili ve güçlü bir araç olmaya devam etmektedir.