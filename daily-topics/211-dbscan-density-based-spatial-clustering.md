# DBSCAN: Density-Based Spatial Clustering

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Principles](#2-core-concepts-and-principles)
  - [2.1. Epsilon (ε) Neighborhood](#21-epsilon-ε-neighborhood)
  - [2.2. MinPts (Minimum Points)](#22-minpts-minimum-points)
  - [2.3. Types of Points](#23-types-of-points)
    - [2.3.1. Core Point](#231-core-point)
    - [2.3.2. Border Point](#232-border-point)
    - [2.3.3. Noise Point](#233-noise-point)
  - [2.4. Density Reachability and Connectivity](#24-density-reachability-and-connectivity)
    - [2.4.1. Directly Density-Reachable](#241-directly-density-reachable)
    - [2.4.2. Density-Reachable](#242-density-reachable)
    - [2.4.3. Density-Connected](#243-density-connected)
- [3. Algorithm Workflow](#3-algorithm-workflow)
- [4. Advantages and Disadvantages](#4-advantages-and-disadvantages)
  - [4.1. Advantages](#41-advantages)
  - [4.2. Disadvantages](#42-disadvantages)
- [5. Parameter Selection](#5-parameter-selection)
- [6. Applications](#6-applications)
- [7. Code Example](#7-code-example)
- [8. Conclusion](#8-conclusion)

## 1. Introduction
Clustering is a fundamental unsupervised machine learning task that involves grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups (clusters). Among the various clustering algorithms, **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** stands out as a powerful method capable of discovering clusters of arbitrary shapes and identifying outliers (noise) in spatial databases. Unlike partition-based algorithms like K-Means, which require the number of clusters to be specified beforehand and struggle with non-convex shapes, DBSCAN determines the clusters based on the density distribution of data points. This characteristic makes it particularly robust and versatile for a wide range of real-world applications where cluster shapes are unknown and noise is prevalent.

## 2. Core Concepts and Principles
DBSCAN operates on the principle of density, defining clusters as regions of high density separated by regions of lower density. To achieve this, it relies on two crucial parameters and several definitions related to point types and their relationships.

### 2.1. Epsilon (ε) Neighborhood
The **ε-neighborhood** (also referred to as epsilon neighborhood or radius) of a point `p` is defined as the set of all points within a maximum distance `ε` from `p`. This distance is typically measured using Euclidean distance, but other metrics can also be employed. Mathematically, for a point `p` and a distance function `dist`, the ε-neighborhood of `p` is `Nε(p) = {q ∈ D | dist(p, q) ≤ ε}`, where `D` is the dataset.

### 2.2. MinPts (Minimum Points)
**MinPts** is the minimum number of points required to form a dense region. A point `p` is considered dense if its ε-neighborhood contains at least `MinPts` points (including `p` itself). This parameter effectively sets a threshold for what constitutes a "dense" area.

### 2.3. Types of Points
Based on the ε-neighborhood and MinPts, DBSCAN categorizes each data point into one of three types:

#### 2.3.1. Core Point
A point `p` is a **core point** if its ε-neighborhood contains at least `MinPts` points. These points are at the "heart" of dense regions and are crucial for forming clusters.

#### 2.3.2. Border Point
A point `q` is a **border point** if it is not a core point itself, but it falls within the ε-neighborhood of a core point. Border points are on the edge of a cluster and are less densely packed than core points.

#### 2.3.3. Noise Point
A point `n` is a **noise point** (or an outlier) if it is neither a core point nor a border point. These points are isolated in low-density regions and do not belong to any cluster.

### 2.4. Density Reachability and Connectivity
DBSCAN defines relationships between points based on their density.

#### 2.4.1. Directly Density-Reachable
A point `q` is **directly density-reachable** from a core point `p` if `q` is within the ε-neighborhood of `p` (`q ∈ Nε(p)`) and `p` is a core point. This establishes a direct link between points in a dense region.

#### 2.4.2. Density-Reachable
A point `q` is **density-reachable** from a point `p` with respect to `ε` and `MinPts` if there is a chain of points `p1, ..., pn`, where `p1 = p` and `pn = q`, such that `pi+1` is directly density-reachable from `pi`. All points in a cluster must be density-reachable from each other.

#### 2.4.3. Density-Connected
Two points `p` and `q` are **density-connected** with respect to `ε` and `MinPts` if there is a core point `o` such that both `p` and `q` are density-reachable from `o`. This concept allows for connecting points that might not be directly density-reachable from each other but are part of the same dense region through an intermediary core point. A cluster in DBSCAN consists of all points that are density-connected to each other.

## 3. Algorithm Workflow
The DBSCAN algorithm typically proceeds as follows:

1.  **Initialize:** Start with an arbitrary unvisited data point `p`.
2.  **Find Neighborhood:** Retrieve the ε-neighborhood of `p`, `Nε(p)`.
3.  **Check Density:**
    *   If `|Nε(p)| < MinPts`, then `p` is marked as noise (initially, it might be reclassified later as a border point).
    *   If `|Nε(p)| ≥ MinPts`, then `p` is a core point, and a new cluster is initiated. All points in `Nε(p)` are added to a list called `seeds`.
4.  **Expand Cluster:** For each point `q` in `seeds`:
    *   If `q` has not been visited, mark it as visited.
    *   Retrieve `Nε(q)`.
    *   If `|Nε(q)| ≥ MinPts`, then `q` is also a core point, and all points in `Nε(q)` that are not yet part of any cluster are added to `seeds`. This process effectively expands the cluster.
    *   If `q` is not a core point but is directly reachable from a core point in the current cluster expansion, it is considered a border point and assigned to the current cluster.
5.  **Repeat:** Continue this process of expanding the cluster until no more points can be added (i.e., `seeds` is empty).
6.  **Next Unvisited Point:** Select another unvisited point from the dataset and repeat the entire process from step 1, forming a new cluster or identifying new noise points.
7.  **Termination:** The algorithm terminates when all points in the dataset have been visited.

## 4. Advantages and Disadvantages

### 4.1. Advantages
*   **Discovery of Arbitrary Shaped Clusters:** DBSCAN can identify clusters of complex, non-linear shapes, unlike K-Means which is restricted to convex shapes.
*   **Robust to Noise:** It inherently handles outliers by designating them as noise points, preventing them from distorting cluster centroids or boundaries.
*   **No Prior Knowledge of K:** It does not require the user to specify the number of clusters (`k`) beforehand, which is a significant advantage when the underlying data structure is unknown.
*   **Flexibility:** Can be used with any distance metric.

### 4.2. Disadvantages
*   **Parameter Sensitivity:** The performance of DBSCAN is highly dependent on the choice of `ε` and `MinPts`. Incorrect parameter values can lead to poor clustering results.
*   **Varying Densities:** It struggles with datasets where clusters have significantly varying densities. A single `ε` and `MinPts` pair might be suitable for dense clusters but too restrictive for sparser ones, or vice versa.
*   **High-Dimensional Data:** In high-dimensional spaces, the concept of density can become less meaningful (due to the "curse of dimensionality"), making it difficult to choose appropriate `ε` values.
*   **Border Point Ambiguity:** Points on the border of two clusters may be assigned to either cluster, depending on the order of processing.

## 5. Parameter Selection
Selecting appropriate values for `ε` and `MinPts` is crucial for DBSCAN's performance.
*   **MinPts:** A common heuristic is to set `MinPts` to `2 * dimensionality` of the dataset. For 2D data, `MinPts` is often set to 4. A larger `MinPts` makes the algorithm more robust to noise but might merge sparse clusters.
*   **ε (epsilon):** The value of `ε` is more challenging to determine. A popular approach involves plotting the k-distance graph (or reachability plot). For each point, calculate the distance to its k-th nearest neighbor (where `k = MinPts`). Sort these distances in ascending order and plot them. A "knee" or "elbow" in the graph often indicates a good value for `ε`, representing a significant increase in distance to the k-th neighbor, suggesting a boundary between dense and sparse regions.

## 6. Applications
DBSCAN's ability to handle arbitrary shapes and noise makes it suitable for numerous applications:
*   **Geospatial Data Analysis:** Identifying regions of interest, urban planning, earthquake epicenter detection.
*   **Anomaly Detection:** Detecting unusual patterns in credit card fraud, network intrusion detection, or manufacturing defects.
*   **Medical Imaging:** Identifying distinct anatomical structures or disease patterns in image data.
*   **Customer Segmentation:** Grouping customers based on purchasing behavior without assuming spherical clusters.
*   **Traffic Pattern Analysis:** Discovering traffic congestion areas or typical travel routes.

## 7. Code Example
Here's a simple Python example using `sklearn.cluster.DBSCAN` to cluster a synthetic dataset.

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate sample data with non-linear shapes
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Initialize and apply DBSCAN
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X)

# Visualize the results
plt.figure(figsize=(8, 6))
# Plot points assigned to clusters
plt.scatter(X[clusters != -1, 0], X[clusters != -1, 1], c=clusters[clusters != -1], cmap='viridis', label='Clusters')
# Plot noise points
plt.scatter(X[clusters == -1, 0], X[clusters == -1, 1], c='gray', marker='x', s=100, label='Noise')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

(End of code example section)
```

## 8. Conclusion
DBSCAN is a powerful and intuitive density-based clustering algorithm that excels at identifying clusters of arbitrary shapes and separating noise. Its strength lies in its ability to automatically determine the number of clusters and handle outliers effectively, making it a valuable tool in exploratory data analysis and various domain-specific applications. While its performance is sensitive to parameter selection and it may struggle with varying cluster densities, careful parameter tuning, often guided by domain knowledge and techniques like the k-distance graph, can unlock its full potential. DBSCAN remains a cornerstone in the field of unsupervised learning, offering a distinct advantage over distance-based methods for datasets with complex spatial structures.
---
<br>

<a name="türkçe-içerik"></a>
## DBSCAN: Yoğunluk Tabanlı Mekansal Kümeleme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve İlkeler](#2-temel-kavramlar-ve-ilkeler)
  - [2.1. Epsilon (ε) Komşuluğu](#21-epsilon-ε-komşuluğu)
  - [2.2. MinPts (Minimum Noktalar)](#22-minpts-minimum-noktalar)
  - [2.3. Nokta Türleri](#23-nokta-türleri)
    - [2.3.1. Çekirdek Nokta](#231-çekirdek-nokta)
    - [2.3.2. Sınır Noktası](#232-sınır-noktası)
    - [2.3.3. Gürültü Noktası](#233-gürültü-noktası)
  - [2.4. Yoğunluk Erişilebilirliği ve Bağlantısı](#24-yoğunluk-erişilebilirliği-ve-bağlantısı)
    - [2.4.1. Doğrudan Yoğunluk Erişilebilirliği](#241-doğrudan-yoğunluk-erişilebilirliği)
    - [2.4.2. Yoğunluk Erişilebilirliği](#242-yoğunluk-erişilebilirliği)
    - [2.4.3. Yoğunluk Bağlantısı](#243-yoğunluk-bağlantısı)
- [3. Algoritma İş Akışı](#3-algoritma-iş-akışı)
- [4. Avantajları ve Dezavantajları](#4-avantajları-ve-dezavantajları)
  - [4.1. Avantajları](#41-avantajları)
  - [4.2. Dezavantajları](#42-dezavantajları)
- [5. Parametre Seçimi](#5-parametre-seçimi)
- [6. Uygulama Alanları](#6-uygulama-alanları)
- [7. Kod Örneği](#7-kod-örneği)
- [8. Sonuç](#8-sonuç)

## 1. Giriş
Kümeleme, bir dizi nesneyi, aynı gruptaki (küme olarak adlandırılan) nesnelerin birbirlerine, diğer gruplardakilere göre daha benzer olacak şekilde gruplama işlemini içeren temel bir denetimsiz makine öğrenimi görevidir. Çeşitli kümeleme algoritmaları arasında, **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, karmaşık şekillere sahip kümeleri keşfedebilen ve mekansal veri tabanlarındaki aykırı değerleri (gürültüleri) tanımlayabilen güçlü bir yöntem olarak öne çıkmaktadır. K-Means gibi küme sayısının önceden belirtilmesini gerektiren ve dışbükey olmayan şekillerle başa çıkmakta zorlanan bölümleme tabanlı algoritmaların aksine, DBSCAN, veri noktalarının yoğunluk dağılımına dayanarak kümeleri belirler. Bu özelliği, küme şekillerinin bilinmediği ve gürültünün yaygın olduğu geniş bir gerçek dünya uygulaması yelpazesi için onu özellikle sağlam ve çok yönlü kılar.

## 2. Temel Kavramlar ve İlkeler
DBSCAN, yoğunluk prensibine göre çalışır ve kümeleri, daha düşük yoğunluklu bölgelerle ayrılmış yüksek yoğunluklu bölgeler olarak tanımlar. Bunu başarmak için, iki kritik parametreye ve nokta türleri ile ilişkilerine dair çeşitli tanımlara dayanır.

### 2.1. Epsilon (ε) Komşuluğu
Bir `p` noktasının **ε-komşuluğu** (epsilon komşuluğu veya yarıçapı olarak da anılır), `p` noktasından maksimum `ε` mesafesindeki tüm noktaların kümesi olarak tanımlanır. Bu mesafe genellikle Öklid mesafesi kullanılarak ölçülür, ancak başka metrikler de kullanılabilir. Matematiksel olarak, bir `p` noktası ve bir `dist` mesafe fonksiyonu için, `p` noktasının ε-komşuluğu `Nε(p) = {q ∈ D | dist(p, q) ≤ ε}` olarak ifade edilir; burada `D` veri kümesidir.

### 2.2. MinPts (Minimum Noktalar)
**MinPts**, yoğun bir bölge oluşturmak için gereken minimum nokta sayısıdır. Bir `p` noktası, ε-komşuluğu en az `MinPts` kadar nokta içeriyorsa (kendi `p` noktası da dahil olmak üzere) yoğun kabul edilir. Bu parametre, "yoğun" bir alanın ne olduğunu belirleyen bir eşik değeri ayarlar.

### 2.3. Nokta Türleri
ε-komşuluğu ve MinPts'ye dayanarak, DBSCAN her veri noktasını üç türden birine ayırır:

#### 2.3.1. Çekirdek Nokta
Bir `p` noktası, ε-komşuluğu en az `MinPts` kadar nokta içeriyorsa bir **çekirdek nokta**dır. Bu noktalar yoğun bölgelerin "kalbindedir" ve kümelerin oluşumu için kritik öneme sahiptir.

#### 2.3.2. Sınır Noktası
Bir `q` noktası, kendisi çekirdek nokta olmamasına rağmen, bir çekirdek noktanın ε-komşuluğuna düşüyorsa bir **sınır noktası**dır. Sınır noktaları bir kümenin kenarındadır ve çekirdek noktalara göre daha az yoğun bir şekilde paketlenmiştir.

#### 2.3.3. Gürültü Noktası
Bir `n` noktası, ne çekirdek nokta ne de sınır noktası ise bir **gürültü noktası**dır (veya aykırı değerdir). Bu noktalar düşük yoğunluklu bölgelerde izole edilmiş durumdadır ve herhangi bir kümeye ait değildir.

### 2.4. Yoğunluk Erişilebilirliği ve Bağlantısı
DBSCAN, noktalar arasındaki ilişkileri yoğunluklarına göre tanımlar.

#### 2.4.1. Doğrudan Yoğunluk Erişilebilirliği
Bir `q` noktası, bir çekirdek nokta `p`'den **doğrudan yoğunluk erişilebilir**dir, eğer `q`, `p`'nin ε-komşuluğu içinde (`q ∈ Nε(p)`) ise ve `p` bir çekirdek noktaysa. Bu, yoğun bir bölgedeki noktalar arasında doğrudan bir bağlantı kurar.

#### 2.4.2. Yoğunluk Erişilebilirliği
Bir `q` noktası, `ε` ve `MinPts`'ye göre bir `p` noktasından **yoğunluk erişilebilir**dir, eğer `p1 = p` ve `pn = q` olacak şekilde `p1, ..., pn` noktalarından oluşan bir zincir varsa ve `pi+1`, `pi`'den doğrudan yoğunluk erişilebilir ise. Bir kümedeki tüm noktalar birbirlerinden yoğunluk erişilebilir olmalıdır.

#### 2.4.3. Yoğunluk Bağlantısı
İki `p` ve `q` noktası, `ε` ve `MinPts`'ye göre **yoğunluk bağlantılı**dır, eğer hem `p` hem de `q`'nun kendisinden yoğunluk erişilebilir olduğu bir çekirdek nokta `o` varsa. Bu kavram, birbirlerinden doğrudan yoğunluk erişilebilir olmasalar bile, ara bir çekirdek nokta aracılığıyla aynı yoğun bölgenin parçası olan noktaları bağlamaya olanak tanır. DBSCAN'de bir küme, birbirine yoğunluk bağlantılı tüm noktalardan oluşur.

## 3. Algoritma İş Akışı
DBSCAN algoritması genellikle şu şekilde ilerler:

1.  **Başlangıç:** Herhangi bir ziyaret edilmemiş veri noktası `p` ile başlanır.
2.  **Komşuluk Bul:** `p` noktasının ε-komşuluğu olan `Nε(p)` alınır.
3.  **Yoğunluk Kontrolü:**
    *   Eğer `|Nε(p)| < MinPts` ise, `p` gürültü olarak işaretlenir (başlangıçta, daha sonra bir sınır noktası olarak yeniden sınıflandırılabilir).
    *   Eğer `|Nε(p)| ≥ MinPts` ise, `p` bir çekirdek noktadır ve yeni bir küme başlatılır. `Nε(p)`'deki tüm noktalar `seeds` adlı bir listeye eklenir.
4.  **Kümeyi Genişlet:** `seeds` içindeki her `q` noktası için:
    *   Eğer `q` ziyaret edilmemişse, ziyaret edildi olarak işaretlenir.
    *   `Nε(q)` alınır.
    *   Eğer `|Nε(q)| ≥ MinPts` ise, `q` da bir çekirdek noktadır ve `Nε(q)`'daki henüz hiçbir kümeye dahil olmayan tüm noktalar `seeds`'e eklenir. Bu süreç, kümeyi etkili bir şekilde genişletir.
    *   Eğer `q` bir çekirdek nokta değilse ancak mevcut küme genişlemesindeki bir çekirdek noktadan doğrudan erişilebilirse, bir sınır noktası olarak kabul edilir ve mevcut kümeye atanır.
5.  **Tekrarla:** Küme genişletme sürecine, daha fazla nokta eklenemeyene kadar (yani `seeds` boş olana kadar) devam edilir.
6.  **Bir Sonraki Ziyaret Edilmemiş Nokta:** Veri kümesinden başka bir ziyaret edilmemiş nokta seçilir ve 1. adımdan itibaren tüm süreç tekrarlanır, yeni bir küme oluşturulur veya yeni gürültü noktaları tanımlanır.
7.  **Sonlandırma:** Veri kümesindeki tüm noktalar ziyaret edildiğinde algoritma sonlanır.

## 4. Avantajları ve Dezavantajları

### 4.1. Avantajları
*   **Keyfi Şekilli Kümeleri Keşfetme:** DBSCAN, K-Means'in dışbükey şekillerle sınırlı olmasının aksine, karmaşık, doğrusal olmayan şekillere sahip kümeleri tanımlayabilir.
*   **Gürültüye Karşı Sağlamlık:** Aykırı değerleri doğal olarak gürültü noktaları olarak belirleyerek, bunların küme merkezlerini veya sınırlarını bozmasını engeller.
*   **K Bilgisine Önceden Gerek Yok:** Kullanıcının küme sayısını (`k`) önceden belirtmesini gerektirmez; bu, temel veri yapısının bilinmediği durumlarda önemli bir avantajdır.
*   **Esneklik:** Herhangi bir mesafe metriği ile kullanılabilir.

### 4.2. Dezavantajları
*   **Parametre Hassasiyeti:** DBSCAN'in performansı `ε` ve `MinPts` seçimlerine oldukça bağımlıdır. Yanlış parametre değerleri kötü kümeleme sonuçlarına yol açabilir.
*   **Değişen Yoğunluklar:** Kümelerin yoğunlukları önemli ölçüde değişen veri kümeleriyle başa çıkmakta zorlanır. Tek bir `ε` ve `MinPts` çifti, yoğun kümeler için uygun olabilirken, daha seyrek kümeler için çok kısıtlayıcı olabilir veya tam tersi.
*   **Yüksek Boyutlu Veriler:** Yüksek boyutlu uzaylarda, yoğunluk kavramı daha az anlamlı hale gelebilir ("boyutluluk laneti" nedeniyle), bu da uygun `ε` değerlerini seçmeyi zorlaştırır.
*   **Sınır Noktası Belirsizliği:** İki kümenin sınırındaki noktalar, işlem sırasına bağlı olarak her iki kümeye de atanabilir.

## 5. Parametre Seçimi
DBSCAN'in performansı için `ε` ve `MinPts` için uygun değerleri seçmek kritik öneme sahiptir.
*   **MinPts:** Yaygın bir sezgisel kural, `MinPts`'yi veri kümesinin `2 * boyutluluğu` olarak ayarlamaktır. 2D veriler için `MinPts` genellikle 4 olarak ayarlanır. Daha büyük bir `MinPts` algoritmayı gürültüye karşı daha sağlam hale getirir ancak seyrek kümeleri birleştirebilir.
*   **ε (epsilon):** `ε` değerini belirlemek daha zordur. Popüler bir yaklaşım, k-uzaklık grafiğini (veya erişilebilirlik grafiğini) çizmeyi içerir. Her nokta için, k-inci en yakın komşusuna olan mesafeyi (`k = MinPts` olduğunda) hesaplayın. Bu mesafeleri artan sırada sıralayın ve grafiğini çizin. Grafikteki bir "diz" veya "dirsek", `ε` için iyi bir değeri gösterir; bu, k-inci komşuya olan mesafede önemli bir artışı temsil eder ve yoğun ve seyrek bölgeler arasında bir sınırı düşündürür.

## 6. Uygulama Alanları
DBSCAN'in keyfi şekilleri ve gürültüyü işleme yeteneği onu çok sayıda uygulama için uygun kılar:
*   **Coğrafi Veri Analizi:** İlgi alanlarını belirleme, şehir planlama, deprem merkez üssü tespiti.
*   **Anomali Tespiti:** Kredi kartı dolandırıcılığı, ağ saldırısı tespiti veya üretim kusurlarında olağandışı modelleri tespit etme.
*   **Tıbbi Görüntüleme:** Görüntü verilerindeki farklı anatomik yapıları veya hastalık modellerini tanımlama.
*   **Müşteri Segmentasyonu:** Müşterileri, küresel kümeler varsaymadan, satın alma davranışlarına göre gruplandırma.
*   **Trafik Modeli Analizi:** Trafik sıkışıklığı alanlarını veya tipik seyahat rotalarını keşfetme.

## 7. Kod Örneği
İşte sentetik bir veri kümesini kümelemek için `sklearn.cluster.DBSCAN` kullanan basit bir Python örneği.

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Doğrusal olmayan şekillere sahip örnek veri üret
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# DBSCAN'i başlat ve uygula
# eps: Bir noktanın diğerinin komşuluğunda sayılması için iki örnek arasındaki maksimum mesafe.
# min_samples: Bir noktanın çekirdek nokta olarak kabul edilmesi için komşuluktaki örnek sayısı (veya toplam ağırlık).
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X)

# Sonuçları görselleştir
plt.figure(figsize=(8, 6))
# Kümelere atanan noktaları çiz
plt.scatter(X[clusters != -1, 0], X[clusters != -1, 1], c=clusters[clusters != -1], cmap='viridis', label='Kümeler')
# Gürültü noktalarını çiz
plt.scatter(X[clusters == -1, 0], X[clusters == -1, 1], c='gray', marker='x', s=100, label='Gürültü')
plt.title('DBSCAN Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.grid(True)
plt.show()

(Kod örneği bölümünün sonu)
```

## 8. Sonuç
DBSCAN, keyfi şekilli kümeleri tanımlama ve gürültüyü ayırmada başarılı olan güçlü ve sezgisel bir yoğunluk tabanlı kümeleme algoritmasıdır. Gücü, küme sayısını otomatik olarak belirleme ve aykırı değerleri etkili bir şekilde işleme yeteneğinde yatar, bu da onu keşifsel veri analizi ve çeşitli alanlara özgü uygulamalarda değerli bir araç haline getirir. Parametre seçimine karşı performansı hassas olsa ve değişen küme yoğunluklarıyla zorlanabilse de, alan bilgisi ve k-uzaklık grafiği gibi tekniklerle yönlendirilen dikkatli parametre ayarlaması, tam potansiyelini ortaya çıkarabilir. DBSCAN, karmaşık mekansal yapılara sahip veri kümeleri için mesafe tabanlı yöntemlere göre belirgin bir avantaj sunarak, denetimsiz öğrenme alanında bir köşe taşı olmaya devam etmektedir.

