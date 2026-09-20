# Metadata Filtering in Vector Databases

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Vector Databases and Metadata](#2-understanding-vector-databases-and-metadata)
- [3. Mechanisms and Benefits of Metadata Filtering](#3-mechanisms-and-benefits-of-metadata-filtering)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
In the rapidly evolving landscape of Generative AI, **vector databases** have emerged as a foundational technology for managing and querying high-dimensional data, especially **vector embeddings** generated from various data types like text, images, and audio. These embeddings capture the semantic meaning of data, enabling powerful **similarity search** capabilities essential for applications such as semantic search, recommendation systems, and retrieval-augmented generation (RAG). However, similarity search alone often falls short when users require results that are not only semantically similar but also adhere to specific criteria or attributes. This is where **metadata filtering** becomes indispensable.

**Metadata filtering** refers to the process of applying additional structured or semi-structured conditions (based on associated metadata) to the results of a vector similarity search. While vector search excels at finding items that are "like" a query, metadata filtering allows users to specify "which kind" of similar items they are interested in. For instance, in a product search scenario, a user might want to find "similar shoes" (vector search) that are also "red" and "size 10" (metadata filtering). This document delves into the intricacies of metadata filtering in vector databases, exploring its mechanisms, benefits, and practical implications for building robust and intelligent AI applications.

## 2. Understanding Vector Databases and Metadata
**Vector databases** are specialized data stores optimized for storing, indexing, and querying **vector embeddings**. Unlike traditional relational databases that operate on structured tables, vector databases focus on the numerical representation of data points in a high-dimensional space. The core operation in these databases is the **similarity search**, typically performed using algorithms like **Approximate Nearest Neighbor (ANN)**, which efficiently find vectors closest to a given query vector. This proximity in vector space corresponds to semantic similarity between the original data items.

Alongside the high-dimensional vectors, most real-world data points come with associated descriptive information, known as **metadata**. Metadata can be any additional attributes that describe the vector's original content or context. Examples include:
*   For text embeddings: author, publication date, category, tags, language.
*   For image embeddings: resolution, color palette, object tags, photographer.
*   For product embeddings: price, brand, color, size, availability, user ratings.

This metadata is typically stored in a structured or semi-structured format (e.g., JSON, key-value pairs) alongside the vector embedding in the vector database. While the vector itself captures semantic meaning, the metadata provides factual, categorical, or contextual information that complements the semantic understanding. The synergy between vector similarity and metadata attributes is crucial for achieving highly relevant and precise search results in complex real-world applications.

## 3. Mechanisms and Benefits of Metadata Filtering
**Metadata filtering** mechanisms in vector databases generally fall into three categories: **pre-filtering**, **post-filtering**, and **hybrid filtering**. Each approach offers different trade-offs in terms of performance, **precision**, and **recall**.

### 3.1. Pre-filtering
In **pre-filtering**, the metadata conditions are applied *before* the vector similarity search is performed. The database first filters the entire collection of vectors based on the specified metadata criteria, creating a much smaller subset of vectors. The similarity search is then conducted only within this pre-filtered subset.

*   **Mechanism:**
    1.  Parse metadata query (e.g., `category = 'electronics' AND price < 500`).
    2.  Identify all vectors whose associated metadata matches the query.
    3.  Perform vector similarity search *only* on the identified subset of vectors.
*   **Benefits:** Highly accurate results because all retrieved vectors will strictly satisfy both semantic similarity and metadata conditions. Can significantly improve search performance if the metadata filter is highly selective, drastically reducing the search space for the ANN algorithm.
*   **Drawbacks:** If the metadata filter is too restrictive, it might exclude relevant documents that would otherwise be semantically close but just outside the strict metadata criteria. Can be slower if the metadata filtering itself is complex or operates on a non-indexed attribute over a very large dataset.

### 3.2. Post-filtering
**Post-filtering** applies metadata conditions *after* the vector similarity search has been completed. The database first retrieves a large number of top-K semantically similar vectors, and then the metadata filter is applied to this initial result set.

*   **Mechanism:**
    1.  Perform a vector similarity search across the entire collection to retrieve the top-N (where N >> K, K being the final desired number) semantically most similar vectors.
    2.  Apply metadata filtering to this top-N result set.
    3.  Return the top-K results from the filtered set.
*   **Benefits:** Ensures that the core semantic similarity search is not constrained by metadata, potentially capturing a broader range of semantically relevant items. Simpler to implement in some architectures as it separates the vector search and metadata filtering concerns.
*   **Drawbacks:** Can be inefficient if many results need to be retrieved initially and then discarded. The final result set might have fewer than K items if the metadata filter is very restrictive, or it might miss relevant items that were semantically similar but just outside the initial top-N retrieved.

### 3.3. Hybrid Filtering
**Hybrid filtering** combines aspects of both pre- and post-filtering, or involves more sophisticated integrated indexing techniques. Some advanced vector databases build unified indices that allow for efficient filtering on both vector and metadata dimensions simultaneously.

*   **Mechanism:** This often involves techniques like multi-modal indexing, where metadata attributes are integrated directly into the vector index structures, or by intelligently optimizing query plans to decide whether to filter first or search first based on query selectivity and index types. For example, a metadata filter might prune segments of an ANN index.
*   **Benefits:** Offers the best of both worlds, aiming for high **precision** and **recall** while maintaining good performance. It can dynamically adapt to query characteristics.
*   **Drawbacks:** More complex to implement and manage, requiring sophisticated indexing and query optimization capabilities within the vector database.

### Benefits of Metadata Filtering:
*   **Enhanced Relevance:** Provides more precise and contextually appropriate results by combining semantic understanding with specific factual criteria.
*   **Improved User Experience:** Allows users to refine searches intuitively, leading to higher satisfaction.
*   **Reduced Noise:** Eliminates irrelevant items from the search results that might be semantically similar but fail to meet other crucial criteria.
*   **Granular Control:** Enables fine-grained control over the search space, which is critical for compliance, personalization, and access control scenarios.
*   **Performance Optimization:** In pre-filtering scenarios, metadata filtering can significantly reduce the number of vectors considered for similarity search, leading to faster query times.
*   **Use Cases:** Critical for e-commerce product search, content recommendation, legal document discovery, code search, and many other enterprise AI applications.

## 4. Code Example
This Python example illustrates a conceptual vector database client integrating metadata filtering. It simulates adding vectors with metadata and performing a search.

```python
import numpy as np

class ConceptualVectorDB:
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.ids = []
        self.next_id = 0

    def add_vector(self, vector, meta={}):
        """Adds a vector and its associated metadata to the database."""
        self.vectors.append(np.array(vector))
        self.metadata.append(meta)
        self.ids.append(self.next_id)
        self.next_id += 1
        return self.next_id - 1

    def _calculate_similarity(self, vec1, vec2):
        """Conceptual similarity function (e.g., dot product for simplicity)."""
        return np.dot(vec1, vec2)

    def search(self, query_vector, top_k=5, filter_conditions=None):
        """
        Performs a vector similarity search with optional metadata filtering.
        This example uses post-filtering for simplicity.
        """
        if not self.vectors:
            return []

        # 1. Perform initial similarity search on all vectors
        similarities = []
        for i, vec in enumerate(self.vectors):
            similarity = self._calculate_similarity(query_vector, vec)
            similarities.append((similarity, self.ids[i], i)) # (similarity, original_id, internal_index)

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 2. Apply metadata filtering to the top-N (here, all initially sorted)
        filtered_results = []
        for sim, original_id, internal_idx in similarities:
            item_metadata = self.metadata[internal_idx]
            
            # Check if item passes all filter conditions
            passes_filter = True
            if filter_conditions:
                for key, expected_value in filter_conditions.items():
                    if item_metadata.get(key) != expected_value:
                        passes_filter = False
                        break
            
            if passes_filter:
                filtered_results.append({
                    "id": original_id,
                    "similarity": sim,
                    "metadata": item_metadata
                })
                # Break if we have enough filtered results
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results

# Initialize the database
db = ConceptualVectorDB()

# Add some vectors with metadata
db.add_vector([0.1, 0.9, 0.2], {"category": "electronics", "brand": "TechCo", "price": 1200})
db.add_vector([0.8, 0.2, 0.1], {"category": "clothing", "brand": "Fashionova", "color": "blue"})
db.add_vector([0.15, 0.85, 0.25], {"category": "electronics", "brand": "GadgetCorp", "price": 800})
db.add_vector([0.7, 0.3, 0.15], {"category": "clothing", "brand": "Fashionova", "color": "red"})
db.add_vector([0.05, 0.95, 0.1], {"category": "electronics", "brand": "TechCo", "price": 1500})

# Define a query vector (e.g., for "high-tech gadgets")
query = [0.1, 0.9, 0.15]

print("--- Search without metadata filter ---")
results_no_filter = db.search(query, top_k=3)
for r in results_no_filter:
    print(f"ID: {r['id']}, Similarity: {r['similarity']:.4f}, Metadata: {r['metadata']}")

print("\n--- Search with metadata filter (category: electronics) ---")
results_with_filter = db.search(query, top_k=3, filter_conditions={"category": "electronics"})
for r in results_with_filter:
    print(f"ID: {r['id']}, Similarity: {r['similarity']:.4f}, Metadata: {r['metadata']}")

print("\n--- Search with metadata filter (brand: TechCo) ---")
results_brand_filter = db.search(query, top_k=3, filter_conditions={"brand": "TechCo"})
for r in results_brand_filter:
    print(f"ID: {r['id']}, Similarity: {r['similarity']:.4f}, Metadata: {r['metadata']}")

(End of code example section)
```
## 5. Conclusion
**Metadata filtering** is a critical capability that transforms raw **vector similarity search** into a powerful, precise, and highly relevant information retrieval system. By allowing users to combine the nuanced semantic understanding provided by **vector embeddings** with concrete, structured attributes, vector databases can address the complexities of real-world queries more effectively. Whether through **pre-filtering**, **post-filtering**, or advanced **hybrid filtering** techniques, the integration of metadata significantly enhances the utility of vector databases across a multitude of Generative AI applications, from sophisticated recommendation engines to precise knowledge retrieval systems. As AI systems become increasingly sophisticated and demand more contextual awareness, the role of robust metadata management and filtering in vector databases will only continue to grow in importance, becoming an indispensable component of modern data infrastructure.
---
<br>

<a name="türkçe-içerik"></a>
## Vektör Veritabanlarında Meta Veri Filtreleme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Vektör Veritabanları ve Meta Veriyi Anlamak](#2-vektör-veritabanları-ve-meta-veriyi-anlamak)
- [3. Meta Veri Filtrelemenin Mekanizmaları ve Faydaları](#3-meta-veri-filtrelemenin-mekanizmaları-ve-faydalari)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
Üretken Yapay Zeka'nın (Generative AI) hızla gelişen ortamında, **vektör veritabanları**, metin, görsel ve ses gibi çeşitli veri türlerinden üretilen **vektör gömmeleri** (vector embeddings) gibi yüksek boyutlu verileri yönetmek ve sorgulamak için temel bir teknoloji olarak ortaya çıkmıştır. Bu gömmeler, verinin anlamsal anlamını yakalayarak, anlamsal arama, tavsiye sistemleri ve bilgi geri çağırma ile artırılmış üretim (RAG - Retrieval-Augmented Generation) gibi uygulamalar için gerekli güçlü **benzerlik araması** yeteneklerini etkinleştirir. Ancak, kullanıcıların sadece anlamsal olarak benzer değil, aynı zamanda belirli kriterlere veya özniteliklere de uyan sonuçlar istediği durumlarda benzerlik araması tek başına yetersiz kalır. İşte bu noktada **meta veri filtreleme** vazgeçilmez hale gelir.

**Meta veri filtreleme**, bir vektör benzerlik aramasının sonuçlarına ek yapılandırılmış veya yarı yapılandırılmış koşullar (ilişkili meta verilere dayalı) uygulama sürecini ifade eder. Vektör araması, bir sorguya "benzer" öğeleri bulmada başarılıyken, meta veri filtreleme, kullanıcıların "ne tür" benzer öğelerle ilgilendiklerini belirtmelerine olanak tanır. Örneğin, bir ürün arama senaryosunda, bir kullanıcı "benzer ayakkabılar" (vektör araması) arayabilirken, aynı zamanda "kırmızı" ve "42 numara" (meta veri filtreleme) olanları da isteyebilir. Bu belge, vektör veritabanlarındaki meta veri filtrelemesinin inceliklerini, mekanizmalarını, faydalarını ve sağlam ve akıllı yapay zeka uygulamaları oluşturmak için pratik çıkarımlarını derinlemesine incelemektedir.

## 2. Vektör Veritabanları ve Meta Veriyi Anlamak
**Vektör veritabanları**, **vektör gömmelerini** depolamak, indekslemek ve sorgulamak için optimize edilmiş özel veri depolama sistemleridir. Yapılandırılmış tablolar üzerinde çalışan geleneksel ilişkisel veritabanlarının aksine, vektör veritabanları verilerin yüksek boyutlu bir uzaydaki sayısal temsiline odaklanır. Bu veritabanlarındaki temel işlem, genellikle **Yaklaşık En Yakın Komşu (ANN)** gibi algoritmalar kullanılarak gerçekleştirilen **benzerlik aramasıdır**, bu algoritmalar verilen bir sorgu vektörüne en yakın vektörleri verimli bir şekilde bulur. Vektör uzayındaki bu yakınlık, orijinal veri öğeleri arasındaki anlamsal benzerliğe karşılık gelir.

Yüksek boyutlu vektörlerin yanı sıra, çoğu gerçek dünya veri noktası, **meta veri** olarak bilinen ilişkili açıklayıcı bilgilerle birlikte gelir. Meta veri, vektörün orijinal içeriğini veya bağlamını tanımlayan herhangi bir ek öznitelik olabilir. Örnekler şunları içerir:
*   Metin gömmeleri için: yazar, yayın tarihi, kategori, etiketler, dil.
*   Görsel gömmeleri için: çözünürlük, renk paleti, nesne etiketleri, fotoğrafçı.
*   Ürün gömmeleri için: fiyat, marka, renk, boyut, stok durumu, kullanıcı puanları.

Bu meta veri, genellikle vektör gömme ile birlikte vektör veritabanında yapılandırılmış veya yarı yapılandırılmış bir formatta (örneğin, JSON, anahtar-değer çiftleri) depolanır. Vektörün kendisi anlamsal anlamı yakalarken, meta veri, anlamsal anlayışı tamamlayan olgusal, kategorik veya bağlamsal bilgi sağlar. Vektör benzerliği ile meta veri öznitelikleri arasındaki sinerji, karmaşık gerçek dünya uygulamalarında yüksek düzeyde alakalı ve kesin arama sonuçları elde etmek için çok önemlidir.

## 3. Meta Veri Filtrelemenin Mekanizmaları ve Faydaları
Vektör veritabanlarındaki **meta veri filtreleme** mekanizmaları genellikle üç kategoriye ayrılır: **ön-filtreleme**, **sonra-filtreleme** ve **hibrit filtreleme**. Her yaklaşım, performans, **kesinlik** ve **geri çağırma** açısından farklı ödünleşimler sunar.

### 3.1. Ön-filtreleme
**Ön-filtreleme**de, meta veri koşulları vektör benzerlik araması yapılmadan *önce* uygulanır. Veritabanı önce tüm vektör koleksiyonunu belirtilen meta veri kriterlerine göre filtreler ve çok daha küçük bir vektör alt kümesi oluşturur. Benzerlik araması daha sonra yalnızca bu ön-filtrelenmiş alt küme içinde yapılır.

*   **Mekanizma:**
    1.  Meta veri sorgusunu ayrıştırın (örneğin, `kategori = 'elektronik' VE fiyat < 500`).
    2.  İlişkili meta verisi sorguyla eşleşen tüm vektörleri tanımlayın.
    3.  Yalnızca tanımlanan vektör alt kümesi üzerinde vektör benzerlik araması yapın.
*   **Faydaları:** Sonuçların yüksek doğruluğu, çünkü geri çağrılan tüm vektörler hem anlamsal benzerlik hem de meta veri koşullarını kesinlikle karşılayacaktır. Meta veri filtresi çok seçiciyse, ANN algoritması için arama alanını drastik bir şekilde azaltarak arama performansını önemli ölçüde artırabilir.
*   **Dezavantajları:** Meta veri filtresi çok kısıtlayıcıysa, anlamsal olarak yakın olabilecek ancak katı meta veri kriterlerinin hemen dışında kalan ilgili belgeleri hariç tutabilir. Meta veri filtrelemesinin kendisi karmaşıksa veya çok büyük bir veri kümesi üzerinde dizinlenmemiş bir öznitelik üzerinde çalışıyorsa daha yavaş olabilir.

### 3.2. Sonra-filtreleme
**Sonra-filtreleme**, meta veri koşullarını vektör benzerlik araması tamamlandıktan *sonra* uygular. Veritabanı önce çok sayıda en iyi K anlamsal olarak benzer vektörü getirir ve ardından bu başlangıçtaki sonuç kümesine meta veri filtresi uygulanır.

*   **Mekanizma:**
    1.  İlk N (N >> K, K nihai istenen sayı olmak üzere) anlamsal olarak en benzer vektörü almak için tüm koleksiyon üzerinde bir vektör benzerlik araması yapın.
    2.  Bu ilk N sonuç kümesine meta veri filtrelemesi uygulayın.
    3.  Filtrelenmiş kümeden en iyi K sonuçları döndürün.
*   **Faydaları:** Ana anlamsal benzerlik aramasının meta veri tarafından kısıtlanmamasını sağlar, potansiyel olarak daha geniş bir anlamsal olarak ilgili öğe yelpazesini yakalar. Bazı mimarilerde uygulanması daha basittir, çünkü vektör araması ve meta veri filtreleme konularını ayırır.
*   **Dezavantajları:** Başlangıçta birçok sonucun getirilmesi ve ardından atılması gerekiyorsa verimsiz olabilir. Meta veri filtresi çok kısıtlayıcıysa nihai sonuç kümesi K'den daha az öğe içerebilir veya başlangıçtaki en iyi N'nin hemen dışında olan ilgili öğeleri kaçırabilir.

### 3.3. Hibrit Filtreleme
**Hibrit filtreleme**, hem ön- hem de sonra-filtrelemenin yönlerini birleştirir veya daha sofistike entegre indeksleme tekniklerini içerir. Bazı gelişmiş vektör veritabanları, hem vektör hem de meta veri boyutlarında eş zamanlı olarak verimli filtrelemeye izin veren birleşik dizinler oluşturur.

*   **Mekanizma:** Bu genellikle, meta veri özniteliklerinin doğrudan vektör dizin yapılarına entegre edildiği çok modlu indeksleme gibi teknikleri veya sorgu seçiciliğine ve dizin türlerine göre önce filtreleme mi yoksa önce arama mı yapılacağına karar vermek için sorgu planlarını akıllıca optimize etmeyi içerir. Örneğin, bir meta veri filtresi, bir ANN dizininin segmentlerini budayabilir.
*   **Faydaları:** Hem yüksek **kesinlik** hem de **geri çağırma** hedefleyerek iyi performans sağlarken her iki dünyanın da en iyisini sunar. Sorgu özelliklerine dinamik olarak uyum sağlayabilir.
*   **Dezavantajları:** Vektör veritabanı içinde sofistike indeksleme ve sorgu optimizasyon yetenekleri gerektirdiğinden uygulanması ve yönetimi daha karmaşıktır.

### Meta Veri Filtrelemenin Faydaları:
*   **Gelişmiş Alaka Düzeyi:** Anlamsal anlayışı belirli olgusal kriterlerle birleştirerek daha kesin ve bağlamsal olarak uygun sonuçlar sağlar.
*   **Geliştirilmiş Kullanıcı Deneyimi:** Kullanıcıların aramaları sezgisel olarak iyileştirmesine olanak tanıyarak daha yüksek memnuniyet sağlar.
*   **Gürültüyü Azaltma:** Anlamsal olarak benzer olabilen ancak diğer önemli kriterleri karşılayamayan alakasız öğeleri arama sonuçlarından çıkarır.
*   **Ayrıntılı Kontrol:** Uyumluluk, kişiselleştirme ve erişim kontrolü senaryoları için kritik olan arama alanı üzerinde ince taneli kontrol sağlar.
*   **Performans Optimizasyonu:** Ön-filtreleme senaryolarında, meta veri filtreleme, benzerlik araması için dikkate alınan vektör sayısını önemli ölçüde azaltabilir, bu da daha hızlı sorgu sürelerine yol açar.
*   **Kullanım Durumları:** E-ticaret ürün araması, içerik tavsiyesi, yasal belge keşfi, kod araması ve diğer birçok kurumsal yapay zeka uygulaması için kritik öneme sahiptir.

## 4. Kod Örneği
Bu Python örneği, meta veri filtrelemeyi entegre eden kavramsal bir vektör veritabanı istemcisini göstermektedir. Meta verili vektör eklemeyi ve arama yapmayı simüle eder.

```python
import numpy as np

class ConceptualVectorDB:
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.ids = []
        self.next_id = 0

    def add_vector(self, vector, meta={}):
        """Veritabanına bir vektör ve ilişkili meta verilerini ekler."""
        self.vectors.append(np.array(vector))
        self.metadata.append(meta)
        self.ids.append(self.next_id)
        self.next_id += 1
        return self.next_id - 1

    def _calculate_similarity(self, vec1, vec2):
        """Kavramsal benzerlik fonksiyonu (örn. basitlik için nokta çarpımı)."""
        return np.dot(vec1, vec2)

    def search(self, query_vector, top_k=5, filter_conditions=None):
        """
        İsteğe bağlı meta veri filtreleme ile bir vektör benzerlik araması yapar.
        Bu örnek, basitlik için sonra-filtreleme kullanır.
        """
        if not self.vectors:
            return []

        # 1. Tüm vektörler üzerinde başlangıç benzerlik araması yapın
        similarities = []
        for i, vec in enumerate(self.vectors):
            similarity = self._calculate_similarity(query_vector, vec)
            similarities.append((similarity, self.ids[i], i)) # (benzerlik, orijinal_id, dahili_indeks)

        # Benzerliğe göre azalan sırada sıralayın
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 2. En iyi N'ye (burada, başlangıçta sıralanmış tümüne) meta veri filtrelemesi uygulayın
        filtered_results = []
        for sim, original_id, internal_idx in similarities:
            item_metadata = self.metadata[internal_idx]
            
            # Öğenin tüm filtre koşullarını geçip geçmediğini kontrol edin
            passes_filter = True
            if filter_conditions:
                for key, expected_value in filter_conditions.items():
                    if item_metadata.get(key) != expected_value:
                        passes_filter = False
                        break
            
            if passes_filter:
                filtered_results.append({
                    "id": original_id,
                    "similarity": sim,
                    "metadata": item_metadata
                })
                # Yeterli sayıda filtrelenmiş sonuç varsa döngüyü sonlandırın
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results

# Veritabanını başlatın
db = ConceptualVectorDB()

# Meta verili bazı vektörler ekleyin
db.add_vector([0.1, 0.9, 0.2], {"category": "electronics", "brand": "TechCo", "price": 1200})
db.add_vector([0.8, 0.2, 0.1], {"category": "clothing", "brand": "Fashionova", "color": "blue"})
db.add_vector([0.15, 0.85, 0.25], {"category": "electronics", "brand": "GadgetCorp", "price": 800})
db.add_vector([0.7, 0.3, 0.15], {"category": "clothing", "brand": "Fashionova", "color": "red"})
db.add_vector([0.05, 0.95, 0.1], {"category": "electronics", "brand": "TechCo", "price": 1500})

# Bir sorgu vektörü tanımlayın (örn. "yüksek teknolojili gadget'lar" için)
query = [0.1, 0.9, 0.15]

print("--- Meta veri filtresi olmadan arama ---")
results_no_filter = db.search(query, top_k=3)
for r in results_no_filter:
    print(f"ID: {r['id']}, Benzerlik: {r['similarity']:.4f}, Meta Veri: {r['metadata']}")

print("\n--- Meta veri filtresiyle arama (kategori: elektronik) ---")
results_with_filter = db.search(query, top_k=3, filter_conditions={"category": "electronics"})
for r in results_with_filter:
    print(f"ID: {r['id']}, Benzerlik: {r['similarity']:.4f}, Meta Veri: {r['metadata']}")

print("\n--- Meta veri filtresiyle arama (marka: TechCo) ---")
results_brand_filter = db.search(query, top_k=3, filter_conditions={"brand": "TechCo"})
for r in results_brand_filter:
    print(f"ID: {r['id']}, Benzerlik: {r['similarity']:.4f}, Meta Veri: {r['metadata']}")

(Kod örneği bölümünün sonu)
```
## 5. Sonuç
**Meta veri filtreleme**, ham **vektör benzerlik aramasını** güçlü, kesin ve son derece alakalı bir bilgi geri çağırma sistemine dönüştüren kritik bir yetenektir. Kullanıcıların **vektör gömmeleri** tarafından sağlanan incelikli anlamsal anlayışı somut, yapılandırılmış özniteliklerle birleştirmesine olanak tanıyarak, vektör veritabanları gerçek dünya sorgularının karmaşıklıklarını daha etkili bir şekilde ele alabilir. İster **ön-filtreleme**, **sonra-filtreleme** veya gelişmiş **hibrit filtreleme** teknikleri aracılığıyla olsun, meta verinin entegrasyonu, sofistike tavsiye motorlarından hassas bilgi geri çağırma sistemlerine kadar çok sayıda Üretken Yapay Zeka uygulaması genelinde vektör veritabanlarının faydasını önemli ölçüde artırır. Yapay zeka sistemleri giderek daha sofistike hale geldikçe ve daha fazla bağlamsal farkındalık talep ettikçe, vektör veritabanlarında sağlam meta veri yönetimi ve filtrelemesinin rolü yalnızca önemini artırmaya devam edecek ve modern veri altyapısının vazgeçilmez bir bileşeni haline gelecektir.
