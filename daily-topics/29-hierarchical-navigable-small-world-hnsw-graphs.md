# Hierarchical Navigable Small World (HNSW) Graphs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Principles](#2-core-concepts-and-principles)
    - [2.1. Navigable Small World (NSW) Graphs](#21-navigable-small-world-nsw-graphs)
    - [2.2. Hierarchical Structure](#22-hierarchical-structure)
    - [2.3. Graph Construction](#23-graph-construction)
    - [2.4. Search Algorithm](#24-search-algorithm)
- [3. Advantages and Disadvantages](#3-advantages-and-disadvantages)
    - [3.1. Advantages](#31-advantages)
    - [3.2. Disadvantages](#32-disadvantages)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction

In the rapidly evolving landscape of **Generative AI**, the ability to efficiently process, store, and retrieve information from massive datasets of high-dimensional vectors is paramount. Tasks such as semantic search, recommendation systems, anomaly detection, and the underlying mechanisms of **Large Language Models (LLMs)** and **image generation models** heavily rely on finding data points that are "similar" to a given query. This notion of similarity is often quantified by distance metrics in high-dimensional vector spaces. The challenge lies in performing **Nearest Neighbor (NN) search** or **K-Nearest Neighbor (KNN) search** in these spaces, which becomes computationally prohibitive for large datasets when exact solutions are required.

**Approximate Nearest Neighbor (ANN) search** algorithms emerged as a crucial solution to this scalability problem, sacrificing a minuscule amount of accuracy for orders of magnitude improvement in speed. Among the most prominent and performant ANN algorithms is the **Hierarchical Navigable Small World (HNSW) graph**. HNSW is a graph-based indexing structure that organizes high-dimensional data points in a way that facilitates extremely fast and accurate approximate nearest neighbor searches. Its robust design makes it a cornerstone technology for modern **vector databases** and many real-world AI applications requiring efficient similarity search.

This document will delve into the foundational principles of HNSW graphs, exploring their construction, search mechanisms, inherent advantages, and limitations.

## 2. Core Concepts and Principles

HNSW graphs build upon the concept of Navigable Small World graphs by introducing a hierarchical structure, allowing for efficient multi-scale search.

### 2.1. Navigable Small World (NSW) Graphs

The predecessor to HNSW, **Navigable Small World (NSW) graphs**, are inspired by the "small-world phenomenon" – the idea that any two people in the world can be connected through a short chain of acquaintances. In an NSW graph, data points are represented as nodes, and connections (edges) between nodes represent some measure of proximity or similarity. The key characteristic of NSW graphs is that they possess both short-range and long-range connections. Short-range connections facilitate local searches, while long-range connections enable "jumps" across the graph, significantly reducing the path length between distant nodes.

During a search in an NSW graph, starting from an arbitrary entry point, the algorithm greedily moves to the neighbor closest to the query point. This process continues until a local optimum is reached, meaning no direct neighbor is closer to the query than the current node. While effective, NSW graphs can suffer from local optima issues, potentially leading to suboptimal search paths and reduced recall.

### 2.2. Hierarchical Structure

HNSW addresses the limitations of flat NSW graphs by introducing a **hierarchical structure**. This multi-layer design allows for a coarse-to-fine search strategy, enabling rapid traversal across the graph before refining the search at lower layers.

-   **Layers:** An HNSW graph consists of multiple layers, indexed from 0 upwards.
    -   **Top Layers (High Layers):** These layers are sparse, containing only a subset of the data points. Connections in these layers are generally long-range, allowing for quick "jumps" across large distances in the vector space. They facilitate rapid identification of the general vicinity of the query.
    -   **Bottom Layer (Base Layer, Layer 0):** This layer contains *all* data points and has dense connections (both short-range and long-range). It is responsible for the fine-grained search and accurate retrieval of nearest neighbors once the search has been narrowed down by the upper layers.
-   **Probabilistic Layer Assignment:** Each data point is probabilistically assigned a maximum layer (`L_max`) during insertion. Points assigned to higher layers also exist in all layers below `L_max` down to layer 0. This probabilistic approach ensures a varied distribution of points across layers, contributing to the graph's navigability. The probability distribution is typically exponential, meaning fewer points are assigned to higher layers, creating the desired sparse structure.

### 2.3. Graph Construction

Building an HNSW graph is an iterative process where data points (vectors) are inserted one by one. For each new data point `p`:

1.  **Determine Maximum Layer (`L_new`):** A random maximum layer `L_new` is assigned to `p` based on a predefined probability distribution.
2.  **Find Entry Point (`ep`):** The algorithm identifies an entry point `ep` (usually the last inserted node or a random one) in the highest existing layer of the graph.
3.  **Greedy Traversal (Top-Down):** Starting from `ep` in the top layer, the algorithm performs a **greedy search** downwards. In each layer `l` from the highest to `L_new + 1`, it finds the neighbor closest to `p` among the current candidates. This closest neighbor then becomes the entry point for the next lower layer `l-1`. This process quickly narrows down the search space to the approximate vicinity of `p`.
4.  **Connect in `L_new` to Layer 0:** Once the search reaches layer `L_new`, a more detailed search is performed. For each layer `l` from `min(L_new, max_layer)` down to 0:
    *   The algorithm identifies `efConstruction` (a parameter) nearest neighbors to `p` among the existing nodes in layer `l` by performing a beam search starting from the entry point found in the previous layer.
    *   From these `efConstruction` candidates, `p` is connected to its `M` closest neighbors (another parameter).
    *   To maintain graph connectivity and prevent nodes from having an excessive number of connections, **neighbor pruning** is applied. If adding `p` as a neighbor to an existing node `q` would exceed `q`'s maximum allowed connections (`M_max`), the furthest neighbor of `q` might be removed, or `p` might not be added if it's too far.
-   **`M` Parameter:** Defines the maximum number of outgoing connections for each node in a layer. A higher `M` leads to better recall but increased memory usage and construction time.
-   **`efConstruction` Parameter:** Controls the size of the dynamic candidate list during graph construction. A larger `efConstruction` improves the accuracy of neighbor selection during construction, leading to a higher quality index and better search recall, but at the cost of slower build times.

### 2.4. Search Algorithm

Searching for the `k` nearest neighbors of a query vector `q` in an HNSW graph follows a similar multi-layer, greedy approach:

1.  **Entry Point Selection:** The search begins at a predefined entry point (`ep`), typically a random node or the node closest to the query `q` found in the highest layer during a previous search.
2.  **Greedy Traversal (Top-Down):** Starting from the highest layer where `ep` exists, the algorithm performs a greedy search. In each layer `l` from the highest down to layer 1:
    *   It identifies the neighbor closest to `q` among the current candidates.
    *   This closest neighbor becomes the new `ep` for the next lower layer `l-1`.
    *   This process effectively navigates through the sparse top layers, quickly guiding the search towards the general region of `q`'s nearest neighbors.
3.  **Beam Search in Layer 0:** Once the search reaches layer 0, a more thorough **beam search** is performed.
    *   A dynamic candidate list of size `efSearch` (a parameter) is maintained, typically using a priority queue (min-heap) to keep track of the `efSearch` closest nodes found so far.
    *   Starting from the `ep` found in layer 1, the algorithm explores neighbors in layer 0, adding them to the candidate list and expanding from the closest ones until `efSearch` candidates have been examined or no closer nodes can be found.
    *   The `k` closest nodes from the final `efSearch` candidate list are returned as the approximate nearest neighbors.
-   **`efSearch` Parameter:** Controls the size of the dynamic candidate list during the search phase. A larger `efSearch` improves search accuracy (recall) but increases search time. It allows for a flexible trade-off between speed and accuracy at query time.

## 3. Advantages and Disadvantages

HNSW graphs offer a compelling balance of performance characteristics, making them highly suitable for a wide range of applications.

### 3.1. Advantages

-   **High Performance and Speed:** HNSW provides state-of-the-art query speeds, often achieving logarithmic time complexity with respect to the dataset size. This makes it incredibly efficient for real-time applications requiring fast similarity searches.
-   **Excellent Recall (Accuracy):** By carefully tuning parameters like `M`, `efConstruction`, and `efSearch`, HNSW can achieve very high recall rates, often exceeding 95-99% for `k=10` or `k=100`, even on large datasets. This high accuracy-to-speed ratio is a major strength.
-   **Scalability:** HNSW scales well to large datasets containing millions or even billions of high-dimensional vectors, making it suitable for modern Generative AI applications that frequently deal with vast embedding spaces.
-   **Flexible Trade-offs:** The parameters `efConstruction` and `efSearch` provide fine-grained control over the speed-accuracy trade-off during both graph construction and query time. This flexibility allows users to optimize the index for their specific needs.
-   **Robustness:** It performs well across various data distributions and dimensionalities, making it a versatile choice for different types of vector data.

### 3.2. Disadvantages

-   **Memory Footprint:** HNSW graphs can be memory-intensive. Each node stores its connections for each layer, in addition to the original high-dimensional vector data. For very large datasets with high dimensionality, memory consumption can be a significant concern.
-   **Construction Time:** While search is fast, building the HNSW index, especially for large datasets with high `efConstruction` values, can be computationally expensive and time-consuming. This is often a one-time cost, but it needs to be factored into deployment.
-   **Parameter Sensitivity:** Optimal performance heavily depends on the correct tuning of parameters (`M`, `efConstruction`, `efSearch`). Suboptimal parameter choices can lead to reduced recall or slower search times. Finding the best parameters often requires experimentation.
-   **No Exact Guarantee:** As an ANN algorithm, HNSW does not guarantee finding the *absolute* nearest neighbors. It provides an approximation, which is typically sufficient for most applications but might not be acceptable for scenarios demanding absolute precision.
-   **Disk I/O and Updates:** While `hnswlib` supports saving/loading indexes, frequent updates or deletions of individual vectors can be complex and less efficient than batch operations, as it involves modifying the graph structure.

## 4. Code Example

Here's a short, illustrative Python code snippet demonstrating how to use the `hnswlib` library to create, populate, and query an HNSW index.

```python
import hnswlib
import numpy as np

# 1. Define parameters
dimensions = 128  # Dimension of the vectors
num_elements = 10000  # Number of data points
M = 16  # Max number of outgoing connections in the graph
ef_construction = 200  # Size of the candidate list during construction
ef_search = 50  # Size of the candidate list during search

# 2. Generate some random data (e.g., embeddings)
data = np.float32(np.random.rand(num_elements, dimensions))
labels = np.arange(num_elements) # Unique identifier for each vector

# 3. Initialize HNSW index
# 'l2' for Euclidean distance, 'ip' for inner product, 'cosine' for cosine similarity
p = hnswlib.Index(space='l2', dim=dimensions)

# 4. Initialize the index with parameters
p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

# Optional: Set number of threads for index building
p.set_num_threads(4)

# 5. Add data points to the index
print("Adding data to the HNSW index...")
p.add_items(data, labels)
print("Data added successfully.")

# 6. Perform a nearest neighbor search for a query vector
query_vector = np.float32(np.random.rand(1, dimensions))
k = 5  # Number of nearest neighbors to retrieve

print(f"\nSearching for {k} nearest neighbors...")
p.set_ef(ef_search) # Set ef parameter for search
labels, distances = p.knn_query(query_vector, k=k)

print(f"Query vector: {query_vector[0][:5]}...")
print(f"Found {k} nearest neighbors:")
for i in range(k):
    print(f"  - Label: {labels[0][i]}, Distance: {distances[0][i]:.4f}")

# Optional: Save and load the index
# index_path = "hnsw_index.bin"
# p.save_index(index_path)
# print(f"\nIndex saved to {index_path}")

# p_loaded = hnswlib.Index(space='l2', dim=dimensions)
# p_loaded.load_index(index_path)
# p_loaded.set_ef(ef_search)
# labels_loaded, distances_loaded = p_loaded.knn_query(query_vector, k=k)
# print(f"Loaded index search results (first label): {labels_loaded[0][0]}")

(End of code example section)
```

## 5. Conclusion

Hierarchical Navigable Small World (HNSW) graphs stand as a testament to ingenious algorithm design, providing a highly efficient and accurate solution for Approximate Nearest Neighbor search in high-dimensional spaces. By leveraging a multi-layer graph structure that combines the principles of small-world networks with a coarse-to-fine search strategy, HNSW effectively mitigates the computational burden of similarity search on vast datasets.

Its exceptional performance in terms of both speed and recall has made it an indispensable component in a wide array of modern AI systems. From powering the **vector databases** that underpin **semantic search** and **recommendation engines** to facilitating the real-time retrieval of contextual information for **large language models** and ensuring the quality of outputs in **generative image and video models**, HNSW graphs enable these complex systems to operate at scale. While challenges such as memory consumption and optimal parameter tuning exist, the advantages offered by HNSW far outweigh its drawbacks for the majority of applications in the Generative AI domain. As the dimensionality and volume of data continue to grow, HNSW will undoubtedly remain a critical technology, driving the capabilities and efficiency of future AI innovations.

---
<br>

<a name="türkçe-içerik"></a>
## Hiyerarşik Gezinilebilir Küçük Dünya (HNSW) Grafikleri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Prensipler](#2-temel-kavramlar-ve-prensipler)
    - [2.1. Gezinilebilir Küçük Dünya (NSW) Grafikleri](#21-gezinilebilir-küçük-dünya-nsw-grafikleri)
    - [2.2. Hiyerarşik Yapı](#22-hiyerarşik-yapı)
    - [2.3. Grafik Oluşturma](#23-grafik-oluşturma)
    - [2.4. Arama Algoritması](#24-arama-algoritması)
- [3. Avantajlar ve Dezavantajlar](#3-avantajlar-ve-dezavantajlar)
    - [3.1. Avantajlar](#31-avantajlar)
    - [3.2. Dezavantajlar](#32-dezavantajlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş

**Üretken Yapay Zeka (Generative AI)**'nın hızla gelişen ortamında, yüksek boyutlu vektörlerden oluşan devasa veri kümelerinden bilgiyi verimli bir şekilde işleme, depolama ve alma yeteneği büyük önem taşımaktadır. Anlamsal arama, öneri sistemleri, anomali tespiti ve **Büyük Dil Modelleri (LLM'ler)** ile **görüntü üretim modellerinin** temel mekanizmaları gibi görevler, belirli bir sorguya "benzer" veri noktalarını bulmaya büyük ölçüde bağlıdır. Bu benzerlik kavramı, genellikle yüksek boyutlu vektör uzaylarında mesafe metrikleri ile nicelendirilir. Zorluk, tam çözümler gerektiğinde büyük veri kümeleri için hesaplama açısından çok pahalı hale gelen bu uzaylarda **En Yakın Komşu (NN) araması** veya **K-En Yakın Komşu (KNN) araması** yapmaktır.

**Yaklaşık En Yakın Komşu (ANN) arama** algoritmaları, ölçeklenebilirlik sorununa kritik bir çözüm olarak ortaya çıkmış, hassasiyetten çok küçük bir miktar feragat ederek hızda kat kat iyileşmeler sağlamıştır. En önde gelen ve performanslı ANN algoritmalarından biri **Hiyerarşik Gezinilebilir Küçük Dünya (HNSW) grafiğidir**. HNSW, yüksek boyutlu veri noktalarını, son derece hızlı ve doğru yaklaşık en yakın komşu aramalarını kolaylaştıracak şekilde düzenleyen, grafik tabanlı bir indeksleme yapısıdır. Sağlam tasarımı, modern **vektör veritabanları** ve verimli benzerlik araması gerektiren birçok gerçek dünya yapay zeka uygulaması için temel bir teknoloji olmasını sağlamıştır.

Bu belge, HNSW grafiklerinin temel prensiplerini inceleyecek, yapısını, arama mekanizmalarını, içsel avantajlarını ve sınırlamalarını keşfedecektir.

## 2. Temel Kavramlar ve Prensipler

HNSW grafikleri, Hiyerarşik bir yapı getirerek ve çok ölçekli aramaya izin vererek Gezinilebilir Küçük Dünya grafikleri kavramı üzerine inşa edilmiştir.

### 2.1. Gezinilebilir Küçük Dünya (NSW) Grafikleri

HNSW'nin öncüsü olan **Gezinilebilir Küçük Dünya (NSW) grafikleri**, "küçük dünya fenomeni"nden esinlenmiştir – dünyadaki herhangi iki kişinin kısa bir tanıdıklar zinciri aracılığıyla bağlanabileceği fikri. Bir NSW grafiğinde, veri noktaları düğümler olarak temsil edilir ve düğümler arasındaki bağlantılar (kenarlar) bir yakınlık veya benzerlik ölçüsünü temsil eder. NSW grafiklerinin temel özelliği, hem kısa menzilli hem de uzun menzilli bağlantılara sahip olmalarıdır. Kısa menzilli bağlantılar yerel aramaları kolaylaştırırken, uzun menzilli bağlantılar grafik boyunca "sıçramalara" olanak tanıyarak uzak düğümler arasındaki yol uzunluğunu önemli ölçüde azaltır.

Bir NSW grafiğinde arama sırasında, rastgele bir başlangıç noktasından başlayarak, algoritma sorgu noktasına en yakın komşuya açgözlü bir şekilde hareket eder. Bu süreç, yerel bir optimuma ulaşılana kadar devam eder; yani, hiçbir doğrudan komşu, mevcut düğümden sorguya daha yakın değildir. Etkili olsa da, NSW grafikleri yerel optimum sorunlarından muzdarip olabilir, bu da potansiyel olarak suboptimal arama yollarına ve düşük geri çağırmaya yol açabilir.

### 2.2. Hiyerarşik Yapı

HNSW, düz NSW grafiklerinin sınırlamalarını **hiyerarşik bir yapı** getirerek ele alır. Bu çok katmanlı tasarım, kaba-ince arama stratejisine olanak tanır ve alt katmanlarda aramayı daraltmadan önce grafik boyunca hızlı geçişe imkan verir.

-   **Katmanlar:** Bir HNSW grafiği, 0'dan yukarı doğru indekslenmiş birden çok katmandan oluşur.
    -   **Üst Katmanlar (Yüksek Katmanlar):** Bu katmanlar seyrektir ve yalnızca veri noktalarının bir alt kümesini içerir. Bu katmanlardaki bağlantılar genellikle uzun menzillidir, bu da vektör uzayında büyük mesafeler boyunca hızlı "sıçramalara" olanak tanır. Sorgunun genel yakınlığını hızlı bir şekilde belirlemeyi kolaylaştırırlar.
    -   **Alt Katman (Temel Katman, Katman 0):** Bu katman *tüm* veri noktalarını içerir ve yoğun bağlantılara (hem kısa menzilli hem de uzun menzilli) sahiptir. Arama üst katmanlar tarafından daraltıldıktan sonra, en yakın komşuların ince taneli aranmasından ve doğru bir şekilde alınmasından sorumludur.
-   **Olasılıksal Katman Ataması:** Her veri noktasına, ekleme sırasında olasılıksal olarak maksimum bir katman (`L_max`) atanır. Daha yüksek katmanlara atanan noktalar, `L_max` altındaki tüm katmanlarda da, katman 0'a kadar var olur. Bu olasılıksal yaklaşım, noktaların katmanlar arasında çeşitli bir dağılımını sağlayarak grafiğin gezinilebilirliğine katkıda bulunur. Olasılık dağılımı genellikle üsteldir, yani daha az nokta daha yüksek katmanlara atanır, bu da istenen seyrek yapıyı oluşturur.

### 2.3. Grafik Oluşturma

Bir HNSW grafiği oluşturmak, veri noktalarının (vektörlerin) tek tek eklenmesiyle tekrarlayan bir süreçtir. Her yeni `p` veri noktası için:

1.  **Maksimum Katmanı Belirleme (`L_new`):** `p`'ye, önceden tanımlanmış bir olasılık dağılımına dayalı olarak rastgele bir maksimum katman (`L_new`) atanır.
2.  **Giriş Noktası Bulma (`ep`):** Algoritma, grafiğin en yüksek mevcut katmanında bir giriş noktası (`ep`) belirler (genellikle en son eklenen düğüm veya rastgele bir düğüm).
3.  **Açgözlü Geçiş (Yukarıdan Aşağıya):** En yüksek katmandaki `ep`'den başlayarak, algoritma aşağı doğru **açgözlü bir arama** yapar. En yüksekten `L_new + 1`'e kadar her `l` katmanında, mevcut adaylar arasında `p`'ye en yakın komşuyu bulur. Bu en yakın komşu daha sonra bir sonraki alt `l-1` katmanı için giriş noktası olur. Bu süreç, arama alanını `p`'nin yaklaşık yakınlığına hızla daraltır.
4.  **`L_new`'den Katman 0'a Bağlama:** Arama `L_new` katmanına ulaştığında, daha ayrıntılı bir arama yapılır. `min(L_new, max_layer)`'den 0'a kadar her `l` katmanı için:
    *   Algoritma, önceki katmanda bulunan giriş noktasından başlayarak bir ışın araması yaparak, `l` katmanındaki mevcut düğümler arasında `p`'ye `efConstruction` (bir parametre) en yakın komşuyu belirler.
    *   Bu `efConstruction` adaylarından, `p`, `M` (başka bir parametre) en yakın komşusuna bağlanır.
    *   Grafik bağlantısını sürdürmek ve düğümlerin aşırı sayıda bağlantıya sahip olmasını önlemek için **komşu budama** uygulanır. Eğer `p`'yi mevcut bir `q` düğümüne komşu olarak eklemek, `q`'nun izin verilen maksimum bağlantı sayısını (`M_max`) aşarsa, `q`'nun en uzak komşusu kaldırılabilir veya `p` çok uzaksa eklenmeyebilir.
-   **`M` Parametresi:** Her düğümün bir katmanda maksimum giden bağlantı sayısını tanımlar. Daha yüksek bir `M`, daha iyi geri çağırmaya yol açar ancak bellek kullanımını ve oluşturma süresini artırır.
-   **`efConstruction` Parametresi:** Grafik oluşturma sırasında dinamik aday listesinin boyutunu kontrol eder. Daha büyük bir `efConstruction`, oluşturma sırasında komşu seçiminin doğruluğunu artırarak daha yüksek kaliteli bir indeks ve daha iyi arama geri çağırma sağlar, ancak daha yavaş oluşturma süreleri pahasına.

### 2.4. Arama Algoritması

Bir sorgu vektörü `q`'nun `k` en yakın komşusunu bir HNSW grafiğinde aramak, benzer bir çok katmanlı, açgözlü yaklaşımı izler:

1.  **Giriş Noktası Seçimi:** Arama, önceden tanımlanmış bir giriş noktasında (`ep`) başlar, bu genellikle rastgele bir düğüm veya önceki bir arama sırasında en yüksek katmanda bulunan `q` sorgusuna en yakın düğümdür.
2.  **Açgözlü Geçiş (Yukarıdan Aşağıya):** `ep`'nin bulunduğu en yüksek katmandan başlayarak, algoritma açgözlü bir arama yapar. En yüksekten katman 1'e kadar her `l` katmanında:
    *   Mevcut adaylar arasında `q`'ya en yakın komşuyu belirler.
    *   Bu en yakın komşu, bir sonraki alt `l-1` katmanı için yeni `ep` olur.
    *   Bu süreç, seyrek üst katmanlar boyunca etkili bir şekilde gezinerek, aramayı `q`'nun en yakın komşularının genel bölgesine hızla yönlendirir.
3.  **Katman 0'da Işın Araması:** Arama katman 0'a ulaştığında, daha kapsamlı bir **ışın araması** yapılır.
    *   `efSearch` (bir parametre) boyutunda dinamik bir aday listesi tutulur, genellikle bir öncelik kuyruğu (min-heap) kullanılarak şimdiye kadar bulunan `efSearch` en yakın düğüm takip edilir.
    *   Katman 1'de bulunan `ep`'den başlayarak, algoritma katman 0'daki komşuları keşfeder, bunları aday listesine ekler ve `efSearch` aday incelenene kadar veya daha yakın düğümler bulunamayana kadar en yakınlardan genişler.
    *   Nihai `efSearch` aday listesinden `k` en yakın düğüm, yaklaşık en yakın komşular olarak döndürülür.
-   **`efSearch` Parametresi:** Arama aşamasında dinamik aday listesinin boyutunu kontrol eder. Daha büyük bir `efSearch`, arama doğruluğunu (geri çağırmayı) artırır ancak arama süresini artırır. Sorgu sırasında hız ve doğruluk arasında esnek bir dengeye olanak tanır.

## 3. Avantajlar ve Dezavantajlar

HNSW grafikleri, geniş bir uygulama yelpazesi için oldukça uygun hale getiren ikna edici bir performans özellikleri dengesi sunar.

### 3.1. Avantajlar

-   **Yüksek Performans ve Hız:** HNSW, veri kümesi boyutuna göre genellikle logaritmik zaman karmaşıklığına ulaşarak son teknoloji sorgu hızları sunar. Bu, hızlı benzerlik aramaları gerektiren gerçek zamanlı uygulamalar için inanılmaz derecede verimli olmasını sağlar.
-   **Mükemmel Geri Çağırma (Doğruluk):** `M`, `efConstruction` ve `efSearch` gibi parametreleri dikkatlice ayarlayarak, HNSW, `k=10` veya `k=100` için bile büyük veri kümelerinde genellikle %95-99'u aşan çok yüksek geri çağırma oranları elde edebilir. Bu yüksek doğruluk/hız oranı önemli bir güçlü yönüdür.
-   **Ölçeklenebilirlik:** HNSW, milyonlarca hatta milyarlarca yüksek boyutlu vektör içeren büyük veri kümelerine iyi ölçeklenir, bu da onu geniş gömme alanlarıyla sık sık uğraşan modern Üretken Yapay Zeka uygulamaları için uygun hale getirir.
-   **Esnek Takaslar:** `efConstruction` ve `efSearch` parametreleri, hem grafik oluşturma hem de sorgu zamanı boyunca hız-doğruluk takası üzerinde hassas kontrol sağlar. Bu esneklik, kullanıcıların indeksi özel ihtiyaçlarına göre optimize etmelerine olanak tanır.
-   **Sağlamlık:** Çeşitli veri dağılımları ve boyutlarında iyi performans gösterir, bu da onu farklı vektör veri türleri için çok yönlü bir seçim haline getirir.

### 3.2. Dezavantajlar

-   **Bellek Ayak İzi:** HNSW grafikleri bellek yoğun olabilir. Her düğüm, orijinal yüksek boyutlu vektör verilerine ek olarak her katman için bağlantılarını saklar. Yüksek boyutlu çok büyük veri kümeleri için bellek tüketimi önemli bir endişe kaynağı olabilir.
-   **Oluşturma Süresi:** Arama hızlı olsa da, HNSW indeksini oluşturmak, özellikle yüksek `efConstruction` değerlerine sahip büyük veri kümeleri için hesaplama açısından pahalı ve zaman alıcı olabilir. Bu genellikle tek seferlik bir maliyettir, ancak dağıtıma dahil edilmesi gerekir.
-   **Parametre Duyarlılığı:** Optimum performans, parametrelerin (`M`, `efConstruction`, `efSearch`) doğru ayarlanmasına büyük ölçüde bağlıdır. Suboptimal parametre seçimleri, geri çağırmanın azalmasına veya arama sürelerinin yavaşlamasına yol açabilir. En iyi parametreleri bulmak genellikle deneme gerektirir.
-   **Kesin Garanti Yok:** Bir ANN algoritması olarak, HNSW *mutlak* en yakın komşuları bulmayı garanti etmez. Bir yaklaşım sağlar, bu çoğu uygulama için yeterlidir ancak mutlak hassasiyet gerektiren senaryolar için kabul edilemeyebilir.
-   **Disk G/Ç ve Güncellemeler:** `hnswlib` indeksleri kaydetmeyi/yüklemeyi desteklese de, tek tek vektörlerin sık sık güncellenmesi veya silinmesi, grafik yapısını değiştirmeyi gerektirdiği için toplu işlemlerden daha karmaşık ve daha az verimli olabilir.

## 4. Kod Örneği

İşte `hnswlib` kütüphanesini kullanarak bir HNSW indeksini oluşturmayı, doldurmayı ve sorgulamayı gösteren kısa, açıklayıcı bir Python kod parçası.

```python
import hnswlib
import numpy as np

# 1. Parametreleri tanımla
dimensions = 128  # Vektörlerin boyutu
num_elements = 10000  # Veri noktası sayısı
M = 16  # Grafikteki maksimum giden bağlantı sayısı
ef_construction = 200  # Oluşturma sırasında aday listesinin boyutu
ef_search = 50  # Arama sırasında aday listesinin boyutu

# 2. Rastgele veri (örneğin gömme vektörleri) oluştur
data = np.float32(np.random.rand(num_elements, dimensions))
labels = np.arange(num_elements) # Her vektör için benzersiz tanımlayıcı

# 3. HNSW indeksini başlat
# 'l2' Öklid mesafesi için, 'ip' iç çarpım için, 'cosine' kosinüs benzerliği için
p = hnswlib.Index(space='l2', dim=dimensions)

# 4. İndeksi parametrelerle başlat
p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

# İsteğe bağlı: İndeks oluşturma için iş parçacığı sayısını ayarla
p.set_num_threads(4)

# 5. İndekse veri noktaları ekle
print("Veriler HNSW indeksine ekleniyor...")
p.add_items(data, labels)
print("Veriler başarıyla eklendi.")

# 6. Bir sorgu vektörü için en yakın komşu araması yap
query_vector = np.float32(np.random.rand(1, dimensions))
k = 5  # Alınacak en yakın komşu sayısı

print(f"\n{k} en yakın komşu aranıyor...")
p.set_ef(ef_search) # Arama için ef parametresini ayarla
labels, distances = p.knn_query(query_vector, k=k)

print(f"Sorgu vektörü: {query_vector[0][:5]}...")
print(f"Bulunan {k} en yakın komşu:")
for i in range(k):
    print(f"  - Etiket: {labels[0][i]}, Mesafe: {distances[0][i]:.4f}")

# İsteğe bağlı: İndeksi kaydet ve yükle
# index_path = "hnsw_index.bin"
# p.save_index(index_path)
# print(f"\nİndeks {index_path} konumuna kaydedildi.")

# p_loaded = hnswlib.Index(space='l2', dim=dimensions)
# p_loaded.load_index(index_path)
# p_loaded.set_ef(ef_search)
# labels_loaded, distances_loaded = p_loaded.knn_query(query_vector, k=k)
# print(f"Yüklenen indeks arama sonuçları (ilk etiket): {labels_loaded[0][0]}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

Hiyerarşik Gezinilebilir Küçük Dünya (HNSW) grafikleri, yüksek boyutlu uzaylarda Yaklaşık En Yakın Komşu araması için son derece verimli ve doğru bir çözüm sunan dahiyane algoritma tasarımının bir kanıtıdır. Küçük dünya ağları ilkelerini kaba-ince arama stratejisiyle birleştiren çok katmanlı bir grafik yapısı kullanarak, HNSW, büyük veri kümelerinde benzerlik aramasının hesaplama yükünü etkili bir şekilde azaltır.

Hem hız hem de geri çağırma açısından olağanüstü performansı, onu çok çeşitli modern yapay zeka sistemlerinde vazgeçilmez bir bileşen haline getirmiştir. **Anlamsal arama** ve **öneri motorlarının** temelini oluşturan **vektör veritabanlarını** güçlendirmekten, **büyük dil modelleri** için bağlamsal bilginin gerçek zamanlı olarak alınmasını kolaylaştırmaya ve **üretken görüntü ve video modellerinde** çıktıların kalitesini sağlamaya kadar, HNSW grafikleri bu karmaşık sistemlerin ölçekli çalışmasına olanak tanır. Bellek tüketimi ve optimal parametre ayarlaması gibi zorluklar mevcut olsa da, HNSW'nin sunduğu avantajlar, Üretken Yapay Zeka alanındaki çoğu uygulama için dezavantajlarından çok daha fazladır. Veri boyutu ve hacmi artmaya devam ettikçe, HNSW şüphesiz gelecekteki yapay zeka yeniliklerinin yeteneklerini ve verimliliğini yönlendiren kritik bir teknoloji olmaya devam edecektir.