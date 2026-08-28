# DBSCAN: Density-Based Spatial Clustering

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts](#2-core-concepts)
- [3. Algorithm Steps](#3-algorithm-steps)
- [4. Advantages and Disadvantages](#4-advantages-and-disadvantages)
- [5. Applications](#5-applications)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is a non-parametric, density-based clustering algorithm introduced by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu in 1996. Unlike partitioning methods such as K-means, which require the number of clusters to be specified in advance and assume spherical cluster shapes, DBSCAN can discover clusters of arbitrary shapes and identify **outliers** or **noise points**. Its fundamental principle revolves around the notion of "density reachability" and "density connectivity" to group together closely packed data points, marking as outliers those points that lie alone in low-density regions. This makes DBSCAN particularly robust for datasets with varying densities and complex geometric structures, where traditional centroid-based or model-based clustering methods might fail to accurately delineate clusters.

<a name="2-core-concepts"></a>
## 2. Core Concepts

Understanding DBSCAN requires familiarity with several key definitions:

*   **ε (epsilon)**: Often denoted as `eps`, this parameter defines the maximum radius of the neighborhood around a data point `p`. It dictates how close points must be to each other to be considered part of the same neighborhood. All points within a distance `ε` from `p` are considered neighbors.

*   **MinPts**: This parameter specifies the minimum number of data points required to form a dense region (a cluster). A point `p` is considered a **core point** if at least `MinPts` points (including `p` itself) are within its `ε`-neighborhood.

*   **Core Point**: A data point `p` is a **core point** if its `ε`-neighborhood contains at least `MinPts` points. These points are central to forming clusters.

*   **Border Point**: A data point `q` is a **border point** if it is not a core point but is within the `ε`-neighborhood of a core point. Border points are on the edge of a cluster but do not have enough neighbors to be core points themselves.

*   **Noise Point (Outlier)**: A data point `n` is a **noise point** (or **outlier**) if it is neither a core point nor a border point. These points are considered to be in sparse regions and do not belong to any cluster.

*   **Directly Density-Reachable**: A point `p` is **directly density-reachable** from a core point `q` if `p` is within the `ε`-neighborhood of `q`. This concept forms the basis of cluster expansion.

*   **Density-Reachable**: A point `p` is **density-reachable** from a point `q` with respect to `ε` and `MinPts` if there is a chain of directly density-reachable points `p_1, ..., p_n` such that `p_1 = q`, `p_n = p`, and each `p_i+1` is directly density-reachable from `p_i`, and all `p_i` in the chain (except possibly `p_n`) are core points. This defines a single cluster.

*   **Density-Connected**: Two points `p` and `q` are **density-connected** with respect to `ε` and `MinPts` if there is a core point `o` such that both `p` and `q` are density-reachable from `o`. This relationship allows clusters to merge and form complex shapes, ensuring that all points within a cluster are interconnected through dense regions.

<a name="3-algorithm-steps"></a>
## 3. Algorithm Steps

The DBSCAN algorithm proceeds as follows:

1.  **Initialize**: Start with an arbitrary unvisited data point `p` in the dataset.
2.  **Find Neighbors**: Retrieve all points within the `ε`-neighborhood of `p`.
3.  **Check Core Point Status**:
    *   If the `ε`-neighborhood of `p` contains fewer than `MinPts` points, `p` is marked as **noise** (for now).
    *   If the `ε`-neighborhood of `p` contains `MinPts` or more points, `p` is classified as a **core point**. A new cluster is initiated for `p`, and all points in its `ε`-neighborhood are added to a list (or queue) for expansion.
4.  **Expand Cluster**: For each point `q` in the expansion list:
    *   Mark `q` as visited and assign it to the current cluster.
    *   Find all unvisited points `N_ε(q)` within the `ε`-neighborhood of `q`.
    *   If `q` is a core point (i.e., `|N_ε(q)| >= MinPts`), then all points in `N_ε(q)` that are not yet classified as noise or part of another cluster are added to the expansion list. This step ensures that the cluster grows as far as possible along density-reachable paths.
5.  **Repeat**: Continue expanding the current cluster until no more points can be added (the expansion list is empty).
6.  **Iterate**: Select another unvisited data point and repeat the process from step 1 until all points in the dataset have been visited and assigned a cluster or labeled as noise. Points initially marked as noise might be reclassified as border points if they are later found to be within the `ε`-neighborhood of a core point belonging to an existing cluster.

<a name="4-advantages-and-disadvantages"></a>
## 4. Advantages and Disadvantages

**Advantages**:
*   **Arbitrary Shape Clusters**: DBSCAN can discover clusters of arbitrary shapes, unlike K-means which is restricted to convex shapes.
*   **Noise Handling**: It effectively identifies and separates noise points (outliers) from actual clusters, which is crucial in many real-world applications.
*   **No Pre-defined Number of Clusters**: The algorithm does not require the user to specify the number of clusters (`k`) in advance. The number of clusters is determined by the data's density structure.
*   **Robust to Order of Points**: The results are mostly independent of the order of points in the dataset, except for border points that can be assigned to different clusters if multiple core points can reach them.

**Disadvantages**:
*   **Parameter Sensitivity**: The performance of DBSCAN is highly sensitive to the choice of the `ε` and `MinPts` parameters. Incorrect selection can lead to poor clustering results (e.g., merging separate clusters or splitting a single cluster).
*   **Varying Density**: It struggles with datasets where clusters have significantly varying densities. A single pair of `ε` and `MinPts` values may not be suitable for identifying clusters in regions of very different densities.
*   **High Dimensionality**: In high-dimensional spaces, the concept of density becomes less meaningful due to the "curse of dimensionality," making it challenging to choose appropriate `ε` values.
*   **Boundary Points Ambiguity**: Border points might be arbitrarily assigned to one of two clusters if they are reachable from core points of different clusters.

<a name="5-applications"></a>
## 5. Applications

DBSCAN's ability to find arbitrarily shaped clusters and handle noise makes it suitable for a wide range of applications:

*   **Spatial Data Mining**: Identifying clusters in geographical data, such as pinpointing areas with high densities of certain phenomena (e.g., crime hotspots, disease outbreaks).
*   **Anomaly Detection**: Detecting unusual patterns or outliers in datasets, for example, in fraud detection (identifying unusual transaction patterns) or network intrusion detection.
*   **Image Processing**: Segmenting images by grouping pixels with similar properties (e.g., color, texture) into regions.
*   **Customer Segmentation**: Identifying distinct groups of customers based on their purchasing behavior or demographics, where customer groups might not have spherical distributions.
*   **Bioinformatics**: Analyzing protein structures or gene expression data to find clusters of related biological entities.
*   **Traffic Management**: Identifying traffic congestion patterns or accident-prone areas based on sensor data.

<a name="6-code-example"></a>
## 6. Code Example

This Python example demonstrates how to use `DBSCAN` from scikit-learn to cluster a synthetic dataset and visualize the results.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset (e.g., two interleaving half-circles)
# This dataset is challenging for K-means but suitable for DBSCAN.
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Scale the data to ensure all features contribute equally to distance calculations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN clustering
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Visualize the results
plt.figure(figsize=(10, 6))
# Plot core points and border points
plt.scatter(X_scaled[clusters != -1, 0], X_scaled[clusters != -1, 1],
            c=clusters[clusters != -1], cmap='viridis', s=50, label='Clusters')
# Plot noise points
plt.scatter(X_scaled[clusters == -1, 0], X_scaled[clusters == -1, 1],
            c='gray', marker='x', s=100, label='Noise')

plt.title('DBSCAN Clustering of make_moons Dataset')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion

DBSCAN stands as a powerful and versatile clustering algorithm, particularly adept at uncovering clusters of arbitrary shapes and handling noisy datasets without requiring prior knowledge of the number of clusters. Its reliance on density-based concepts—epsilon-neighborhoods and minimum points—provides a robust framework for identifying natural groupings in spatial data. While its performance is sensitive to parameter selection, especially `ε` and `MinPts`, and it may struggle with varying cluster densities, its strengths in outlier detection and flexibility in cluster geometry make it an indispensable tool in various fields, from spatial data mining to anomaly detection and bioinformatics. Understanding its core principles and judiciously applying its parameters allows for effective extraction of meaningful patterns from complex datasets.

---
<br>

<a name="türkçe-içerik"></a>
## DBSCAN: Yoğunluk Tabanlı Mekansal Kümeleme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar](#2-temel-kavramlar)
- [3. Algoritma Adımları](#3-algoritma-adımları)
- [4. Avantajlar ve Dezavantajlar](#4-avantajlar-ve-dezavantajlar)
- [5. Uygulama Alanları](#5-uygulama-alanları)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, 1996 yılında Martin Ester, Hans-Peter Kriegel, Jörg Sander ve Xiaowei Xu tarafından tanıtılan, parametrik olmayan, yoğunluk tabanlı bir kümeleme algoritmasıdır. Küme sayısının önceden belirtilmesini gerektiren ve küresel küme şekilleri varsayan K-ortalamalar gibi bölümleme yöntemlerinin aksine, DBSCAN, rastgele şekilli kümeleri keşfedebilir ve **aykırı değerleri** veya **gürültü noktalarını** tanımlayabilir. Temel prensibi, yakın paketlenmiş veri noktalarını gruplamak için "yoğunluk erişilebilirliği" ve "yoğunluk bağlantısı" kavramları etrafında dönerken, düşük yoğunluklu bölgelerde tek başına bulunan noktaları aykırı değer olarak işaretler. Bu özelliği, DBSCAN'ı değişen yoğunluklara ve karmaşık geometrik yapılara sahip veri kümeleri için özellikle sağlam kılar; bu tür durumlarda geleneksel merkez tabanlı veya model tabanlı kümeleme yöntemleri kümeleri doğru bir şekilde ayırt edemeyebilir.

<a name="2-temel-kavramlar"></a>
## 2. Temel Kavramlar

DBSCAN'ı anlamak için çeşitli temel tanımlara aşina olmak gerekir:

*   **ε (epsilon)**: Genellikle `eps` olarak adlandırılan bu parametre, bir veri noktası `p` etrafındaki komşuluğun maksimum yarıçapını tanımlar. Noktaların aynı komşuluğun parçası olarak kabul edilmesi için birbirine ne kadar yakın olması gerektiğini belirler. `p` noktasından `ε` mesafesindeki tüm noktalar komşu olarak kabul edilir.

*   **MinPts**: Bu parametre, yoğun bir bölge (bir küme) oluşturmak için gereken minimum veri noktası sayısını belirtir. Bir `p` noktası, `ε`-komşuluğu içinde en az `MinPts` nokta (kendi dahil) içeriyorsa, bir **çekirdek nokta** olarak kabul edilir.

*   **Çekirdek Nokta**: Bir veri noktası `p`, `ε`-komşuluğu içinde en az `MinPts` nokta içeriyorsa bir **çekirdek nokta**dır. Bu noktalar küme oluşumu için merkezidir.

*   **Sınır Noktası**: Bir veri noktası `q`, çekirdek nokta değilse, ancak bir çekirdek noktanın `ε`-komşuluğu içinde ise bir **sınır noktası**dır. Sınır noktaları bir kümenin kenarındadır ancak kendileri çekirdek nokta olacak kadar komşuya sahip değildir.

*   **Gürültü Noktası (Aykırı Değer)**: Bir veri noktası `n`, ne çekirdek nokta ne de sınır nokta ise bir **gürültü noktası**dır (veya **aykırı değer**). Bu noktalar seyrek bölgelerde bulunur ve herhangi bir kümeye ait değildir.

*   **Doğrudan Yoğunluk-Erişilebilir**: Bir `p` noktası, bir çekirdek nokta `q`'dan **doğrudan yoğunluk-erişilebilir**dir eğer `p`, `q`'nun `ε`-komşuluğu içindeyse. Bu kavram küme genişlemesinin temelini oluşturur.

*   **Yoğunluk-Erişilebilir**: Bir `p` noktası, `ε` ve `MinPts`'e göre bir `q` noktasından **yoğunluk-erişilebilir**dir eğer `p_1, ..., p_n` doğrudan yoğunluk-erişilebilir noktalar zinciri varsa; öyle ki `p_1 = q`, `p_n = p`, ve her `p_i+1`, `p_i`'den doğrudan yoğunluk-erişilebilir ve zincirdeki tüm `p_i`'ler (muhtemelen `p_n` hariç) çekirdek noktalardır. Bu bir tek kümeyi tanımlar.

*   **Yoğunluk-Bağlantılı**: İki nokta `p` ve `q`, `ε` ve `MinPts`'e göre **yoğunluk-bağlantılı**dır eğer hem `p` hem de `q`'nun yoğunluk-erişilebilir olduğu bir çekirdek nokta `o` varsa. Bu ilişki, kümelerin birleşmesine ve karmaşık şekiller oluşturmasına izin verir, böylece bir küme içindeki tüm noktaların yoğun bölgeler aracılığıyla birbirine bağlı olmasını sağlar.

<a name="3-algoritma-adımları"></a>
## 3. Algoritma Adımları

DBSCAN algoritması aşağıdaki adımları izler:

1.  **Başlat**: Veri kümesindeki rastgele ziyaret edilmemiş bir veri noktası `p` ile başlayın.
2.  **Komşuları Bul**: `p` noktasının `ε`-komşuluğu içindeki tüm noktaları bulun.
3.  **Çekirdek Nokta Durumunu Kontrol Et**:
    *   Eğer `p`'nin `ε`-komşuluğu `MinPts`'ten az nokta içeriyorsa, `p` (şimdilik) **gürültü** olarak işaretlenir.
    *   Eğer `p`'nin `ε`-komşuluğu `MinPts` veya daha fazla nokta içeriyorsa, `p` bir **çekirdek nokta** olarak sınıflandırılır. `p` için yeni bir küme başlatılır ve `ε`-komşuluğundaki tüm noktalar genişletme için bir listeye (veya kuyruğa) eklenir.
4.  **Kümeyi Genişlet**: Genişletme listesindeki her `q` noktası için:
    *   `q`'yu ziyaret edildi olarak işaretleyin ve mevcut kümeye atayın.
    *   `q`'nun `ε`-komşuluğu içindeki ziyaret edilmemiş tüm `N_ε(q)` noktalarını bulun.
    *   Eğer `q` bir çekirdek nokta ise (yani, `|N_ε(q)| >= MinPts`), o zaman henüz gürültü olarak sınıflandırılmamış veya başka bir kümenin parçası olmayan `N_ε(q)`'daki tüm noktalar genişletme listesine eklenir. Bu adım, kümenin yoğunluk-erişilebilir yollar boyunca mümkün olduğunca büyümesini sağlar.
5.  **Tekrarla**: Mevcut kümeyi, başka nokta eklenemeyene kadar (genişletme listesi boşalana kadar) genişletmeye devam edin.
6.  **İterasyon**: Başka bir ziyaret edilmemiş veri noktası seçin ve veri kümesindeki tüm noktalar ziyaret edilip bir kümeye atanana veya gürültü olarak etiketlenene kadar süreci adım 1'den itibaren tekrarlayın. Başlangıçta gürültü olarak işaretlenen noktalar, daha sonra mevcut bir kümeye ait bir çekirdek noktanın `ε`-komşuluğunda bulunurlarsa sınır noktaları olarak yeniden sınıflandırılabilir.

<a name="4-avantajlar-ve-dezavantajlar"></a>
## 4. Avantajlar ve Dezavantajlar

**Avantajlar**:
*   **Rastgele Şekilli Kümeler**: DBSCAN, K-ortalamalar gibi dışbükey şekillerle sınırlı algoritmaların aksine, rastgele şekilli kümeleri keşfedebilir.
*   **Gürültü İşleme**: Gürültü noktalarını (aykırı değerleri) gerçek kümelerden etkili bir şekilde tanımlar ve ayırır, bu da birçok gerçek dünya uygulamasında kritik öneme sahiptir.
*   **Önceden Tanımlı Küme Sayısı Yok**: Algoritma, kullanıcının önceden `k` küme sayısını belirtmesini gerektirmez. Küme sayısı, verinin yoğunluk yapısı tarafından belirlenir.
*   **Nokta Sırasına Karşı Dayanıklı**: Sonuçlar, veri kümesindeki noktaların sırasından çoğunlukla bağımsızdır, ancak birden fazla çekirdek noktanın onlara ulaşabildiği sınır noktaları için farklı kümelere atanabilirler.

**Dezavantajlar**:
*   **Parametre Hassasiyeti**: DBSCAN'ın performansı, `ε` ve `MinPts` parametrelerinin seçimine oldukça duyarlıdır. Yanlış seçim, kötü kümeleme sonuçlarına yol açabilir (örn. ayrı kümeleri birleştirme veya tek bir kümeyi bölme).
*   **Değişen Yoğunluk**: Kümelerin yoğunluklarının önemli ölçüde değiştiği veri kümeleriyle başa çıkmakta zorlanır. Tek bir `ε` ve `MinPts` çifti, çok farklı yoğunluktaki bölgelerdeki kümeleri tanımlamak için uygun olmayabilir.
*   **Yüksek Boyutluluk**: Yüksek boyutlu uzaylarda, "boyutsallık laneti" nedeniyle yoğunluk kavramı daha az anlamlı hale gelir, bu da uygun `ε` değerlerini seçmeyi zorlaştırır.
*   **Sınır Noktası Belirsizliği**: Sınır noktaları, farklı kümelere ait çekirdek noktalardan erişilebilirlerse, keyfi olarak iki kümeden birine atanabilir.

<a name="5-uygulama-alanları"></a>
## 5. Uygulama Alanları

DBSCAN'ın rastgele şekilli kümeleri bulma ve gürültüyü işleme yeteneği, geniş bir uygulama yelpazesi için uygun olmasını sağlar:

*   **Mekansal Veri Madenciliği**: Coğrafi verilerdeki kümeleri tanımlama, örneğin belirli olayların yüksek yoğunlukta olduğu alanları belirleme (örn. suç sıcak noktaları, hastalık salgınları).
*   **Anomali Tespiti**: Veri kümelerindeki olağandışı desenleri veya aykırı değerleri tespit etme, örneğin dolandırıcılık tespitinde (olağandışı işlem desenlerini tanımlama) veya ağa sızma tespitinde.
*   **Görüntü İşleme**: Benzer özelliklere (örn. renk, doku) sahip pikselleri bölgelere gruplayarak görüntüleri bölümlere ayırma.
*   **Müşteri Segmentasyonu**: Satın alma davranışları veya demografik özelliklerine göre farklı müşteri gruplarını tanımlama, burada müşteri grupları küresel dağılımlara sahip olmayabilir.
*   **Biyoinformatik**: İlgili biyolojik varlıkların kümelerini bulmak için protein yapılarını veya gen ifade verilerini analiz etme.
*   **Trafik Yönetimi**: Sensör verilerine dayanarak trafik sıkışıklığı modellerini veya kaza riskli alanları belirleme.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği

Bu Python örneği, sentetik bir veri kümesini kümelemek ve sonuçları görselleştirmek için scikit-learn'den `DBSCAN`'ın nasıl kullanılacağını gösterir.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

# Sentetik bir veri kümesi oluşturun (örn. iki iç içe geçmiş yarım daire)
# Bu veri kümesi K-ortalamalar için zorlayıcıdır ancak DBSCAN için uygundur.
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Mesafe hesaplamalarına tüm özelliklerin eşit katkıda bulunmasını sağlamak için veriyi ölçeklendirin
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN kümelemesi uygulayın
# eps: Bir örneğin diğerinin komşuluğunda kabul edilmesi için iki örnek arasındaki maksimum mesafe.
# min_samples: Bir noktanın çekirdek nokta olarak kabul edilmesi için komşuluktaki örnek sayısı (veya toplam ağırlık).
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Sonuçları görselleştirin
plt.figure(figsize=(10, 6))
# Çekirdek noktaları ve sınır noktalarını çizin
plt.scatter(X_scaled[clusters != -1, 0], X_scaled[clusters != -1, 1],
            c=clusters[clusters != -1], cmap='viridis', s=50, label='Kümeler')
# Gürültü noktalarını çizin
plt.scatter(X_scaled[clusters == -1, 0], X_scaled[clusters == -1, 1],
            c='gray', marker='x', s=100, label='Gürültü')

plt.title('make_moons Veri Kümesinin DBSCAN Kümelenmesi')
plt.xlabel('Özellik 1 (Ölçeklenmiş)')
plt.ylabel('Özellik 2 (Ölçeklenmiş)')
plt.legend()
plt.grid(True)
plt.show()

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç

DBSCAN, özellikle rastgele şekilli kümeleri ortaya çıkarmada ve gürültülü veri kümelerini küme sayısı hakkında önceden bilgi gerektirmeden işlemede usta olan güçlü ve çok yönlü bir kümeleme algoritmasıdır. Yoğunluk tabanlı kavramlara — epsilon-komşulukları ve minimum noktalar — dayanması, mekansal verilerdeki doğal gruplamaları tanımlamak için sağlam bir çerçeve sağlar. Performansı, özellikle `ε` ve `MinPts` gibi parametre seçimine duyarlı olsa da ve değişen küme yoğunluklarıyla mücadele edebilse de, aykırı değer tespitindeki güçlü yönleri ve küme geometrisindeki esnekliği, onu mekansal veri madenciliğinden anomali tespitine ve biyoinformatiğe kadar çeşitli alanlarda vazgeçilmez bir araç haline getirir. Temel prensiplerini anlamak ve parametrelerini dikkatli bir şekilde uygulamak, karmaşık veri kümelerinden anlamlı modelleri etkili bir şekilde çıkarmayı sağlar.




