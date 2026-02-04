# K-Means Clustering Algorithm

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Algorithm Methodology](#2-algorithm-methodology)
  - [2.1. Initialization](#21-initialization)
  - [2.2. Assignment Step](#22-assignment-step)
  - [2.3. Update Step](#23-update-step)
  - [2.4. Convergence](#24-convergence)
  - [2.5. Choosing the Optimal K](#25-choosing-the-optimal-k)
- [3. Applications](#3-applications)
- [4. Advantages and Disadvantages](#4-advantages-and-disadvantages)
  - [4.1. Advantages](#41-advantages)
  - [4.2. Disadvantages](#42-disadvantages)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

The **K-Means Clustering Algorithm** is a fundamental and widely-used unsupervised machine learning algorithm designed to partition `n` observations into `k` clusters, where each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype of the cluster. As an **unsupervised learning** technique, K-Means does not require labeled data for training, making it suitable for exploring the inherent structure within datasets. Its primary goal is to minimize the **intra-cluster variance**, which is the sum of squared distances between data points and their respective cluster centroids. This algorithm is particularly effective for problems where the underlying data distribution is expected to form distinct, isotropic (spherically shaped) clusters. Developed by Stuart Lloyd in 1957 as a technique for pulse-code modulation, and later popularized by MacQueen in 1967, K-Means remains a cornerstone in data analysis, pattern recognition, and various fields requiring data segmentation. Its simplicity, efficiency, and scalability make it a popular choice for large datasets, despite some known limitations regarding cluster shape and sensitivity to initial conditions.

<a name="2-algorithm-methodology"></a>
## 2. Algorithm Methodology

The K-Means algorithm operates iteratively to refine cluster assignments and centroid positions. The process can be broken down into distinct steps:

<a name="21-initialization"></a>
### 2.1. Initialization

The first step involves randomly selecting `k` data points from the dataset to serve as the initial **centroids** for the `k` clusters. Alternatively, more sophisticated methods like K-Means++ can be used to choose initial centroids that are well-separated, which often leads to faster convergence and better quality clusters by reducing the likelihood of suboptimal local optima. The choice of `k`, the number of clusters, is a crucial parameter that must be determined beforehand.

<a name="22-assignment-step"></a>
### 2.2. Assignment Step

In this step, each data point in the dataset is assigned to the nearest centroid. The "nearest" typically refers to the smallest **Euclidean distance** (or squared Euclidean distance) between the data point and each of the `k` centroids. For a data point `x` and a centroid `c`, the Euclidean distance `d(x, c)` is calculated as `sqrt(sum((x_i - c_i)^2))`. This step effectively partitions the dataset into `k` Voronoi cells, where each cell corresponds to a cluster defined by its centroid.

<a name="23-update-step"></a>
### 2.3. Update Step

After all data points have been assigned to clusters, the centroids of the clusters are recalculated. For each cluster, the new centroid is computed as the **mean** of all data points assigned to that cluster. Mathematically, if `C_j` is the set of data points assigned to cluster `j`, and `|C_j|` is the number of points in `C_j`, the new centroid `c_j'` is `(1 / |C_j|) * sum(x for x in C_j)`. This recalculation moves the centroids to the center of their respective clusters, minimizing the sum of squared distances within each cluster.

<a name="24-convergence"></a>
### 2.4. Convergence

The assignment and update steps are repeated iteratively until a **convergence criterion** is met. Common convergence criteria include:
*   Centroid positions no longer change significantly between iterations.
*   Data point assignments to clusters no longer change.
*   A maximum number of iterations has been reached.
*   A minimum decrease in the total sum of squared distances (inertia) is observed.
Once convergence is achieved, the algorithm terminates, and the final cluster assignments and centroids are outputted.

<a name="25-choosing-the-optimal-k"></a>
### 2.5. Choosing the Optimal K

Determining the optimal number of clusters, `k`, is often a challenge. A commonly used heuristic is the **Elbow Method**. This method involves running K-Means for a range of `k` values (e.g., from 1 to 10) and plotting the **within-cluster sum of squares (WCSS)**, also known as inertia, against `k`. WCSS measures the sum of squared distances between each point and its centroid within a cluster, summed across all clusters. As `k` increases, WCSS generally decreases. The "elbow" point on the plot, where the rate of decrease significantly slows down, is often chosen as the optimal `k`, as it represents a good balance between minimizing WCSS and not having too many clusters. Other methods include the silhouette score and gap statistic.

<a name="3-applications"></a>
## 3. Applications

K-Means clustering is a versatile algorithm with a wide array of applications across various domains:

*   **Customer Segmentation:** Businesses use K-Means to group customers based on purchasing behavior, demographics, or website interactions. This helps in targeted marketing campaigns, personalized recommendations, and understanding customer needs.
*   **Image Compression/Segmentation:** In image processing, K-Means can reduce the number of distinct colors in an image (quantization) or segment an image into distinct regions based on pixel intensity or color, which is useful for object recognition or medical image analysis.
*   **Document Clustering:** Grouping similar documents or articles based on their content, useful for organizing large text corpuses, topic modeling, or information retrieval systems.
*   **Anomaly Detection:** Identifying unusual patterns or outliers in datasets. By clustering normal data points, instances that fall far from any cluster centroid can be flagged as anomalies, applicable in fraud detection or network intrusion detection.
*   **Genomic Sequence Analysis:** Clustering genes with similar expression patterns to understand biological functions or pathways.
*   **Geospatial Analysis:** Grouping locations with similar characteristics, such as crime hotspots, or optimal placement of facilities.

<a name="4-advantages-and-disadvantages"></a>
## 4. Advantages and Disadvantages

Like any algorithm, K-Means has its strengths and weaknesses, which dictate its suitability for particular tasks.

<a name="41-advantages"></a>
### 4.1. Advantages

*   **Simplicity and Interpretability:** The algorithm is conceptually straightforward and easy to implement and understand, making its results relatively interpretable.
*   **Computational Efficiency:** K-Means is computationally efficient and scales well to large datasets, especially compared to hierarchical clustering methods, due to its linear time complexity in the number of data points.
*   **Speed:** It typically converges quickly, especially with good initialization strategies like K-Means++.
*   **Effectiveness for Spherical Clusters:** It performs exceptionally well when clusters are globular and clearly separable.

<a name="42-disadvantages"></a>
### 4.2. Disadvantages

*   **Sensitivity to Initial Centroids:** The final clustering result can be highly dependent on the initial placement of centroids, potentially leading to different results with different random initializations. Multiple runs with different initializations are often recommended.
*   **Requires Pre-specified `k`:** The number of clusters `k` must be determined beforehand, which is often difficult without prior knowledge or the use of heuristic methods like the Elbow Method.
*   **Difficulty with Non-Globular Clusters:** K-Means struggles with clusters that are not spherical or have varying densities, shapes, or sizes. It tends to create spherical clusters of similar size, even when the underlying data suggests otherwise.
*   **Sensitivity to Outliers:** Outliers can significantly influence centroid positions, potentially distorting the clusters. Preprocessing steps like outlier detection and removal can mitigate this.
*   **Assumes Equal Variance:** It implicitly assumes that clusters have similar variances, which might not always be the case in real-world data.

<a name="5-code-example"></a>
## 5. Code Example

This Python example demonstrates how to apply the K-Means algorithm using the `scikit-learn` library to cluster synthetic 2D data.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate synthetic data for clustering
# We create 3 distinct blobs (clusters) for demonstration
X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=0.60)

# 2. Initialize and fit the K-Means model
# We set n_clusters to 3, as we know the true number of clusters
# random_state ensures reproducibility
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# 3. Get cluster labels and centroids
labels = kmeans.labels_        # Cluster label for each data point
centroids = kmeans.cluster_centers_ # Coordinates of the cluster centroids

# 4. Visualize the clustering results
plt.figure(figsize=(8, 6))
# Plot data points, colored by their assigned cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.7)
# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroids')
plt.title('K-Means Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Print the final centroids
print("Final Centroids:\n", centroids)
print("Number of iterations to converge:", kmeans.n_iter_)

(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion

The K-Means clustering algorithm stands as a powerful and accessible tool for exploring unlabeled datasets and uncovering hidden group structures. Its iterative approach of centroid refinement and data point assignment makes it efficient for various tasks, from customer segmentation to image analysis. While its simplicity is a significant advantage, it is crucial to be aware of its limitations, particularly concerning the assumption of spherical clusters, sensitivity to initial centroid placement, and the need to pre-specify the number of clusters, `k`. Advanced initialization techniques (like K-Means++), careful preprocessing, and methods like the Elbow Method for `k` selection can help mitigate some of these challenges. Despite these considerations, K-Means remains a foundational algorithm in machine learning, often serving as a baseline for more complex clustering problems and continuing to provide valuable insights across scientific and industrial applications.

---
<br>

<a name="türkçe-içerik"></a>
## K-Ortalama Kümeleme Algoritması

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Algoritma Metodolojisi](#2-algoritma-metodolojisi)
  - [2.1. Başlatma](#21-başlatma)
  - [2.2. Atama Adımı](#22-atama-adımı)
  - [2.3. Güncelleme Adımı](#23-güncelleme-adımı)
  - [2.4. Yakınsama](#24-yakınsama)
  - [2.5. Optimal K Değerini Seçme](#25-optimal-k-değerini-seçme)
- [3. Uygulamalar](#3-uygulamalar)
- [4. Avantajlar ve Dezavantajlar](#4-avantajlar-ve-dezavantajlar)
  - [4.1. Avantajlar](#41-avantajlar)
  - [4.2. Dezavantajlar](#42-dezavantajlar)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

**K-Ortalama Kümeleme Algoritması**, `n` gözlemi `k` kümeye ayırmak için tasarlanmış temel ve yaygın olarak kullanılan bir **denetimsiz makine öğrenimi** algoritmasıdır; her gözlem, kendi kümesinin bir prototipi olarak hizmet veren en yakın ortalama (merkez) ile kümeye aittir. Bir **denetimsiz öğrenme** tekniği olarak, K-Ortalama, eğitim için etiketli veri gerektirmez, bu da veri kümeleri içindeki doğal yapıyı keşfetmek için uygun hale getirir. Temel amacı, veri noktaları ile ilgili küme merkezleri arasındaki kareli mesafelerin toplamı olan **küme içi varyansı** en aza indirmektir. Bu algoritma, altında yatan veri dağılımının belirgin, izotropik (küresel şekilli) kümeler oluşturmasının beklendiği problemler için özellikle etkilidir. Stuart Lloyd tarafından 1957'de darbe kodu modülasyonu için bir teknik olarak geliştirilen ve daha sonra 1967'de MacQueen tarafından popülerleştirilen K-Ortalama, veri analizi, örüntü tanıma ve veri segmentasyonu gerektiren çeşitli alanlarda bir köşe taşı olmaya devam etmektedir. Basitliği, verimliliği ve ölçeklenebilirliği, küme şekli ve başlangıç koşullarına duyarlılığı ile ilgili bilinen bazı sınırlamalarına rağmen, büyük veri kümeleri için popüler bir seçimdir.

<a name="2-algoritma-metodolojisi"></a>
## 2. Algoritma Metodolojisi

K-Ortalama algoritması, küme atamalarını ve merkez konumlarını iyileştirmek için tekrarlamalı olarak çalışır. Süreç, farklı adımlara ayrılabilir:

<a name="21-başlatma"></a>
### 2.1. Başlatma

İlk adım, veri kümesinden `k` adet veri noktasının `k` küme için başlangıç **merkezleri** olarak rastgele seçilmesini içerir. Alternatif olarak, K-Means++ gibi daha sofistike yöntemler, genellikle daha hızlı yakınsamaya ve daha kaliteli kümelere yol açan, suboptimal yerel optimallik olasılığını azaltan iyi ayrılmış başlangıç merkezleri seçmek için kullanılabilir. Küme sayısı `k`'nin seçimi, önceden belirlenmesi gereken kritik bir parametredir.

<a name="22-atama-adımı"></a>
### 2.2. Atama Adımı

Bu adımda, veri kümesindeki her veri noktası en yakın merkeze atanır. "En yakın" genellikle veri noktası ile `k` merkezden her biri arasındaki en küçük **Öklid mesafesini** (veya kareli Öklid mesafesini) ifade eder. Bir `x` veri noktası ve bir `c` merkezi için, Öklid mesafesi `d(x, c)`, `sqrt(sum((x_i - c_i)^2))` olarak hesaplanır. Bu adım, veri kümesini `k` adet Voronoi hücresine etkili bir şekilde böler; her hücre, merkezi tarafından tanımlanan bir kümeye karşılık gelir.

<a name="23-güncelleme-adımı"></a>
### 2.3. Güncelleme Adımı

Tüm veri noktaları kümelere atandıktan sonra, kümelerin merkezleri yeniden hesaplanır. Her küme için, yeni merkez, o kümeye atanan tüm veri noktalarının **ortalaması** olarak hesaplanır. Matematiksel olarak, eğer `C_j`, `j` kümesine atanan veri noktaları kümesi ise ve `|C_j|`, `C_j`'deki nokta sayısı ise, yeni merkez `c_j'`, `(1 / |C_j|) * sum(x for x in C_j)`'dir. Bu yeniden hesaplama, merkezleri ilgili kümelerinin merkezine taşır ve her küme içindeki kareli mesafelerin toplamını en aza indirir.

<a name="24-yakınsama"></a>
### 2.4. Yakınsama

Atama ve güncelleme adımları, bir **yakınsama kriteri** karşılanana kadar tekrarlamalı olarak yürütülür. Yaygın yakınsama kriterleri şunları içerir:
*   Merkez konumları yinelemeler arasında önemli ölçüde değişmez.
*   Veri noktalarının kümelere atanması değişmez.
*   Maksimum yineleme sayısına ulaşılmıştır.
*   Toplam kareli mesafelerin (atalet) toplamında minimum bir azalma gözlemlenir.
Yakınsama sağlandığında, algoritma sonlanır ve nihai küme atamaları ile merkezler çıktı olarak verilir.

<a name="25-optimal-k-değerini-seçme"></a>
### 2.5. Optimal K Değerini Seçme

Optimal küme sayısı `k`'yı belirlemek genellikle bir zorluktur. Yaygın olarak kullanılan bir sezgisel yöntem **Dirsek Yöntemi**'dir. Bu yöntem, bir dizi `k` değeri için (örn. 1'den 10'a kadar) K-Ortalama'yı çalıştırmayı ve **küme içi kareler toplamını (WCSS)**, yani ataleti, `k`'ye karşı grafiğe dökmeyi içerir. WCSS, her küme içindeki her nokta ile merkezleri arasındaki kareli mesafelerin toplamını ölçer ve tüm kümeler için toplanır. `k` arttıkça, WCSS genellikle azalır. Grafikteki "dirsek" noktası, azalma hızının önemli ölçüde yavaşladığı yer, genellikle optimal `k` olarak seçilir, çünkü WCSS'yi en aza indirme ile çok fazla kümeye sahip olmama arasında iyi bir dengeyi temsil eder. Diğer yöntemler arasında siluet skoru ve boşluk istatistiği bulunur.

<a name="3-uygulamalar"></a>
## 3. Uygulamalar

K-Ortalama kümeleme, çeşitli alanlarda geniş bir uygulama yelpazesine sahip çok yönlü bir algoritmadır:

*   **Müşteri Segmentasyonu:** İşletmeler, satın alma davranışı, demografi veya web sitesi etkileşimlerine göre müşterileri gruplandırmak için K-Ortalama'yı kullanır. Bu, hedeflenmiş pazarlama kampanyalarına, kişiselleştirilmiş önerilere ve müşteri ihtiyaçlarını anlamaya yardımcı olur.
*   **Görüntü Sıkıştırma/Segmentasyonu:** Görüntü işlemede, K-Ortalama bir görüntüdeki farklı renk sayısını azaltabilir (niceleme) veya piksel yoğunluğuna veya rengine göre bir görüntüyü farklı bölgelere ayırabilir; bu, nesne tanıma veya tıbbi görüntü analizi için kullanışlıdır.
*   **Belge Kümeleme:** İçeriklerine göre benzer belgeleri veya makaleleri gruplandırmak, büyük metin koleksiyonlarını düzenlemek, konu modellemesi veya bilgi erişim sistemleri için kullanışlıdır.
*   **Anomali Tespiti:** Veri kümelerindeki alışılmadık desenleri veya aykırı değerleri belirleme. Normal veri noktalarını kümeleyerek, herhangi bir küme merkezinden uzak düşen örnekler anomali olarak işaretlenebilir; bu, dolandırıcılık tespiti veya ağ saldırı tespiti gibi alanlarda uygulanabilir.
*   **Genomik Dizi Analizi:** Biyolojik işlevleri veya yolları anlamak için benzer ifade desenlerine sahip genleri kümeleme.
*   **Coğrafi Uzaysal Analiz:** Suç noktaları veya tesislerin optimal yerleşimi gibi benzer özelliklere sahip konumları gruplandırma.

<a name="4-avantajlar-ve-dezavantajlar"></a>
## 4. Avantajlar ve Dezavantajlar

Her algoritma gibi, K-Ortalama'nın da belirli görevler için uygunluğunu belirleyen güçlü ve zayıf yönleri vardır.

<a name="41-avantajlar"></a>
### 4.1. Avantajlar

*   **Basitlik ve Yorumlanabilirlik:** Algoritma kavramsal olarak basittir ve uygulaması ve anlaşılması kolaydır, bu da sonuçlarının nispeten yorumlanabilir olmasını sağlar.
*   **Hesaplama Verimliliği:** K-Ortalama, hesaplama açısından verimlidir ve veri noktalarının sayısına göre doğrusal zaman karmaşıklığı nedeniyle, özellikle hiyerarşik kümeleme yöntemlerine kıyasla büyük veri kümelerine iyi ölçeklenir.
*   **Hız:** Özellikle K-Means++ gibi iyi başlatma stratejileriyle tipik olarak hızlı bir şekilde yakınsar.
*   **Küresel Kümeler İçin Etkinlik:** Kümeler küresel olduğunda ve açıkça ayrılabilir olduğunda olağanüstü performans gösterir.

<a name="42-dezavantajlar"></a>
### 4.2. Dezavantajlar

*   **Başlangıç Merkezlerine Duyarlılık:** Nihai kümeleme sonucu, merkezlerin başlangıçtaki yerleşimine büyük ölçüde bağlı olabilir ve bu da farklı rastgele başlatmalarla farklı sonuçlara yol açabilir. Çoğu zaman farklı başlatmalarla birden fazla çalıştırma önerilir.
*   **Önceden Belirlenmiş `k` Gereksinimi:** Küme sayısı `k` önceden belirlenmelidir, bu da genellikle ön bilgi olmadan veya Dirsek Yöntemi gibi sezgisel yöntemler kullanılmadan zordur.
*   **Küresel Olmayan Kümelerle Zorluk:** K-Ortalama, küresel olmayan veya farklı yoğunluklara, şekillere veya boyutlara sahip kümelerle zorlanır. Altında yatan veri aksini önermesine rağmen, benzer büyüklükte küresel kümeler oluşturma eğilimindedir.
*   **Aykırı Değerlere Duyarlılık:** Aykırı değerler merkez konumlarını önemli ölçüde etkileyebilir ve potansiyel olarak kümeleri bozabilir. Aykırı değer tespiti ve çıkarılması gibi ön işleme adımları bunu hafifletebilir.
*   **Eşit Varyans Varsayımı:** Kümelerin benzer varyanslara sahip olduğunu ima eder, bu gerçek dünya verilerinde her zaman böyle olmayabilir.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği

Bu Python örneği, sentetik 2D verilerini kümelemek için `scikit-learn` kütüphanesini kullanarak K-Ortalama algoritmasının nasıl uygulanacağını göstermektedir.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Kümeleme için sentetik veri oluşturma
# Gösterim amacıyla 3 farklı küme oluşturuyoruz
X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=0.60)

# 2. K-Ortalama modelini başlatma ve eğitme
# n_clusters'ı 3 olarak ayarlıyoruz, çünkü gerçek küme sayısını biliyoruz
# random_state, tekrarlanabilirliği sağlar
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# 3. Küme etiketlerini ve merkezleri alma
labels = kmeans.labels_        # Her veri noktası için küme etiketi
centroids = kmeans.cluster_centers_ # Küme merkezlerinin koordinatları

# 4. Kümeleme sonuçlarını görselleştirme
plt.figure(figsize=(8, 6))
# Veri noktalarını, atandıkları kümeye göre renklendirerek çiz
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.7)
# Merkezleri çiz
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Merkezler')
plt.title('K-Ortalama Kümeleme Sonucu')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.grid(True)
plt.show()

# Nihai merkezleri yazdır
print("Nihai Merkezler:\n", centroids)
print("Yakınsama için gereken iterasyon sayısı:", kmeans.n_iter_)

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç

K-Ortalama kümeleme algoritması, etiketlenmemiş veri kümelerini keşfetmek ve gizli grup yapılarını ortaya çıkarmak için güçlü ve erişilebilir bir araç olarak durmaktadır. Merkezlerin iyileştirilmesi ve veri noktası ataması konusundaki tekrarlamalı yaklaşımı, müşteri segmentasyonundan görüntü analizine kadar çeşitli görevler için verimli olmasını sağlar. Basitliği önemli bir avantaj olsa da, küresel küme varsayımı, başlangıç merkezlerinin yerleşimine duyarlılık ve `k` küme sayısını önceden belirtme ihtiyacı gibi sınırlamalarının farkında olmak kritik öneme sahiptir. Gelişmiş başlatma teknikleri (K-Means++ gibi), dikkatli ön işleme ve `k` seçimi için Dirsek Yöntemi gibi yaklaşımlar bu zorlukların bazılarını hafifletmeye yardımcı olabilir. Bu hususlara rağmen, K-Ortalama makine öğreniminde temel bir algoritma olmaya devam etmekte, genellikle daha karmaşık kümeleme problemleri için bir temel görevi görmekte ve bilimsel ve endüstriyel uygulamalarda değerli içgörüler sunmaya devam etmektedir.
