# Reformer: The Efficient Transformer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: Limitations of Standard Transformers](#2-background-limitations-of-standard-transformers)
- [3. Reformer's Core Innovations](#3-reformers-core-innovations)
  - [3.1. Locality-Sensitive Hashing (LSH) Attention](#31-locality-sensitive-hashing-lsh-attention)
  - [3.2. Reversible Layers for Memory Efficiency](#32-reversible-layers-for-memory-efficiency)
  - [3.3. Chunking for Feed-Forward Networks](#33-chunking-for-feed-forward-networks)
  - [3.4. Axial Positional Encodings](#34-axial-positional-encodings)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

---

## 1. Introduction

The **Transformer** architecture, introduced in 2017 by Vaswani et al., revolutionized the field of Natural Language Processing (NLP) and subsequently impacted various other domains, including computer vision and reinforcement learning. Its reliance on the **self-attention mechanism** allowed it to model long-range dependencies effectively, leading to state-of-the-art results across numerous tasks. However, the remarkable success of Transformers comes with a significant computational cost, particularly concerning memory and time complexity for long input sequences. The primary bottleneck lies in the self-attention mechanism, which computes pairwise interactions between all tokens in a sequence, resulting in quadratic memory and time complexity with respect to the sequence length. This inherent limitation restricts the practical applicability of Transformers to sequences of moderate length, posing a significant challenge for tasks involving very long documents, high-resolution images, or extensive genomic data.

The **Reformer** model, proposed by Kitaev et al. in 2020, addresses these critical efficiency challenges by introducing a series of ingenious architectural modifications designed to drastically reduce the memory footprint and computational cost of Transformers while preserving their performance capabilities. Reformer makes it feasible to process sequences orders of magnitude longer than what was previously practical, enabling new applications and pushing the boundaries of what is achievable with attention-based models. This document will delve into the core innovations of the Reformer, explaining how it achieves remarkable efficiency gains through **Locality-Sensitive Hashing (LSH) Attention**, **reversible layers**, **chunking**, and **axial positional encodings**.

## 2. Background: Limitations of Standard Transformers

To appreciate the innovations of Reformer, it is crucial to understand the limitations of the standard Transformer architecture. A typical Transformer consists of an encoder and a decoder stack, each comprising multiple identical layers. Each layer contains a **multi-head self-attention mechanism** and a **position-wise feed-forward network**, followed by residual connections and layer normalization.

The most significant computational and memory bottleneck arises from the **self-attention mechanism**. For an input sequence of length $L$ and a hidden dimension $D_k$, the attention mechanism computes query (Q), key (K), and value (V) matrices. The attention weights are then calculated as $softmax(QK^T / \sqrt{D_k})V$.
*   **Quadratic Complexity of Attention:** The computation of $QK^T$ involves multiplying an $L \times D_k$ matrix by a $D_k \times L$ matrix, resulting in an $L \times L$ attention matrix. This operation has a time complexity of $O(L^2 D_k)$ and requires storing an $L \times L$ matrix, leading to a memory complexity of $O(L^2)$. For very long sequences, this quadratic scaling becomes prohibitive. For instance, a sequence of length 65,536 would require an $O(4 \times 10^9)$ attention matrix, consuming terabytes of memory.
*   **Memory for Activations in Backpropagation:** During training, standard Transformer layers store intermediate activations for all layers to compute gradients via backpropagation. If there are $N$ layers, this amounts to $O(N \cdot L \cdot D_{model})$ memory, where $D_{model}$ is the model's hidden dimension. This memory consumption further limits the maximum sequence length that can be processed on available hardware, even if the attention mechanism itself could handle it.
*   **Positional Encodings:** Standard absolute positional encodings typically have a fixed maximum length, or their interpolation performance degrades beyond trained lengths, making them less suitable for extremely long and variable sequence lengths.

These limitations collectively hinder the application of Transformers to tasks such as modeling very long documents, high-resolution images, or genomic sequences, which inherently possess extensive contextual information. Reformer directly tackles these issues through its novel design.

## 3. Reformer's Core Innovations

Reformer introduces four primary innovations to mitigate the memory and computational burden of standard Transformers: Locality-Sensitive Hashing (LSH) Attention, reversible layers, chunking for feed-forward networks, and axial positional encodings.

### 3.1. Locality-Sensitive Hashing (LSH) Attention

The most radical departure from the standard Transformer is the replacement of the dot-product self-attention with **Locality-Sensitive Hashing (LSH) Attention**. Instead of computing attention scores between every query and every key, which yields the $O(L^2)$ bottleneck, LSH Attention aims to only compute attention over a subset of keys that are "close" to a given query in embedding space.

The core idea is based on **Locality-Sensitive Hashing (LSH)**, a technique used to group data points that are similar to each other. In the context of attention, LSH groups queries that are likely to attend to the same keys.
1.  **Hashing:** Each query vector $Q$ and key vector $K$ is projected into a lower-dimensional space and then assigned to a "bucket" based on a hash function. Queries and keys that are "close" in the original embedding space are more likely to fall into the same hash bucket. Reformer uses random rotation hashing, where vectors are projected onto a random hyperplane and then assigned to buckets based on the sign of their projection. To improve robustness and reduce collision errors, multiple hash functions (e.g., `num_hashes` = 8) are typically used, and the queries are sorted by their hash bucket assignments.
2.  **Attention within Buckets:** After hashing, queries are sorted according to their assigned buckets. Attention is then computed only between queries and keys within the *same* hash bucket. This dramatically reduces the number of pairwise comparisons.
3.  **Chunking and Caching:** To further optimize, the sorted queries are processed in chunks. For each chunk, the attention computation is performed, but critically, keys from the *previous* chunk that fall into the *same* bucket are also included. This ensures that a query can still attend to relevant keys even if they are not strictly within its immediate chunk, provided they share the same hash bucket.

This approach reduces the attention complexity from $O(L^2)$ to $O(L \log L)$, a significant improvement for long sequences. While LSH Attention is an approximation of full attention, the empirical results show that it performs comparably to standard attention for many tasks, especially when using multiple hash rounds.

### 3.2. Reversible Layers for Memory Efficiency

Standard Transformers consume a significant amount of memory during training due to storing activations from each layer for backpropagation. Reformer mitigates this by adopting **reversible layers**, inspired by the **RevNet** architecture. In a reversible residual network, the outputs of a layer can be used to perfectly reconstruct the inputs of that layer. This eliminates the need to store intermediate activations for all layers. Instead, only the activations of the final layer need to be stored; all preceding activations can be recomputed on the fly during the backward pass.

Reversible layers achieve this by splitting the input activation $X$ into two parts, $X_1$ and $X_2$. A typical Transformer block involves two main sub-layers: an attention sub-layer $F$ and a feed-forward sub-layer $G$. In Reformer's reversible layers, the update rule is structured as follows:
$Y_1 = X_1 + F(X_2)$
$Y_2 = X_2 + G(Y_1)$

Crucially, given $(Y_1, Y_2)$, one can reconstruct $(X_1, X_2)$ using:
$X_2 = Y_2 - G(Y_1)$
$X_1 = Y_1 - F(X_2)$

This reversible design means that the memory cost for storing activations during training becomes $O(1)$ with respect to the number of layers, instead of $O(N)$ for $N$ layers in a standard Transformer. This is a profound memory saving, allowing for deeper models or, more importantly, much longer sequences to be trained within the same memory budget.

### 3.3. Chunking for Feed-Forward Networks

Even with LSH Attention and reversible layers, the **feed-forward networks (FFNs)** can still consume substantial memory, especially when processing very long sequences. A standard FFN typically involves a matrix multiplication with a large weight matrix, followed by an activation function, and another matrix multiplication. While the time complexity of FFNs is generally $O(L \cdot D_{model}^2)$, the intermediate activation layer can have a very large dimension (e.g., $4 \cdot D_{model}$), requiring significant memory to store during the forward pass, especially for the gradients during the backward pass.

Reformer addresses this by applying **chunking** to the feed-forward network computations. Instead of processing the entire sequence through the FFN at once, the input sequence is divided into smaller chunks. Each chunk is then processed independently through the FFN. This significantly reduces the peak memory requirement, as only the activations for a single chunk need to be held in memory at any given time. This approach reduces the memory footprint for FFNs from $O(L \cdot D_{model})$ to $O(\text{chunk\_size} \cdot D_{model})$, making it possible to handle extremely long sequences without running out of memory during FFN computations.

### 3.4. Axial Positional Encodings

For scenarios involving extremely long sequences, such as high-resolution images flattened into a single sequence or very long textual documents, the standard absolute positional encodings might become inefficient or lose effectiveness. Conventional positional encodings are typically applied as a single vector addition across the sequence dimension. When the sequence length becomes astronomically large, these encodings may struggle to represent fine-grained positions or scale effectively.

Reformer proposes **axial positional encodings** for such cases. Instead of representing the position as a single vector, it represents it as a sum of two (or more) vectors, each corresponding to a different "axis" or dimension. For example, a 2D image flattened into a 1D sequence could have its position $i$ mapped to coordinates $(x, y)$, where $i = x \cdot \text{width} + y$. Axial positional encodings would then add a separate embedding for $x$ and another for $y$.
This factorizes the positional encoding, allowing it to cover much larger effective sequence lengths without requiring an embedding table of equally large dimensions. It's particularly useful for inputs that naturally have multiple spatial or temporal dimensions that are flattened into a single sequence, such as images, where positions can be thought of as `(height_pos, width_pos)`. This reduces the size of the embedding matrix required and potentially improves generalization to unseen, longer sequence lengths by composition.

## 4. Code Example

The following Python snippet illustrates a simplified conceptual view of how a **Locality-Sensitive Hashing (LSH)** function might group queries into buckets, a core component of LSH Attention. It does not implement the full attention mechanism but shows the hashing step.

```python
import torch
import torch.nn.functional as F

def lsh_hash_simple(vectors, num_hashes, hash_size):
    """
    A simplified conceptual LSH hashing function.
    In a real Reformer, this would involve random rotations.
    This example uses a simple projection for demonstration.

    Args:
        vectors (torch.Tensor): Input vectors (queries/keys), shape (batch_size, seq_len, dim)
        num_hashes (int): Number of independent hash functions.
        hash_size (int): The number of buckets per hash function (e.g., power of 2).

    Returns:
        torch.Tensor: Hash bucket assignments, shape (batch_size, seq_len, num_hashes)
    """
    batch_size, seq_len, dim = vectors.shape

    # For simplicity, we'll use a fixed set of random projections.
    # In a real LSH, these would be generated dynamically or learned.
    # Shape: (num_hashes, dim, hash_size) -- conceptual, as actual hashing is often simpler.
    # Let's use random fixed directions for a simpler illustration of bucketing.
    # Each hash function projects the vector onto a random vector and quantizes the result.
    
    # Generate num_hashes random vectors for projection (simulates random planes)
    # Using a fixed seed for reproducibility in example
    torch.manual_seed(0) 
    random_projections = torch.randn(num_hashes, dim)
    
    # Normalize projections to unit vectors
    random_projections = F.normalize(random_projections, p=2, dim=-1)

    # Compute dot products for each hash function
    # (batch_size, seq_len, dim) @ (num_hashes, dim) -> (batch_size, seq_len, num_hashes)
    projections = torch.einsum('bsd,hd->bsh', vectors, random_projections)

    # Assign to buckets based on the projection result.
    # Here, we simply take the argmax or a scaled/quantized version.
    # For a simple bucket assignment, let's use the sign or map to an integer range.
    # A common LSH trick for angular distance is to use random planes and check which side the vector falls on.
    # Let's quantize the projection value into 'hash_size' buckets.
    
    # Scale and shift to make values positive and within a range for bucketing
    # This is a very simplified bucketing. Real LSH functions are more robust.
    scaled_projections = (projections - projections.min()) / (projections.max() - projections.min() + 1e-9)
    bucket_assignments = (scaled_projections * (hash_size - 1)).long()

    return bucket_assignments

# Example usage:
batch_size = 2
seq_len = 10
dim = 64
num_hashes = 2
hash_size = 4 # e.g., 4 buckets per hash function

# Create some dummy query vectors
queries = torch.randn(batch_size, seq_len, dim)

# Compute LSH buckets
buckets = lsh_hash_simple(queries, num_hashes, hash_size)

print("Query vectors shape:", queries.shape)
print("Computed LSH buckets shape:", buckets.shape)
print("Example buckets for batch 0, hash 0:\n", buckets[0, :, 0])
print("Example buckets for batch 0, hash 1:\n", buckets[0, :, 1])
print("\nNote: In a real LSH Attention, queries would be sorted by these buckets,")
print("and attention would only be computed within or across adjacent buckets.")

(End of code example section)
```

## 5. Conclusion

The Reformer model stands as a monumental achievement in the quest for efficient Transformer architectures. By cleverly addressing the two most significant bottlenecks—the quadratic memory and time complexity of self-attention and the linear memory scaling for activations during training—Reformer has unlocked the potential of attention-based models for processing unprecedentedly long sequences.

Its primary innovations, **Locality-Sensitive Hashing (LSH) Attention**, replace the full attention matrix computation with an approximation that scales sub-quadratically ($O(L \log L)$). The adoption of **reversible layers** drastically reduces memory consumption during backpropagation from $O(N \cdot L \cdot D_{model})$ to $O(L \cdot D_{model})$ (or $O(1)$ with respect to number of layers), making deeper models and longer sequences feasible. Furthermore, **chunking for feed-forward networks** and **axial positional encodings** offer additional memory savings and improved handling of extremely long, multi-dimensional inputs.

Reformer's contributions extend beyond merely making Transformers faster and more memory-efficient; they represent a significant step towards developing truly scalable and universally applicable attention mechanisms. This efficiency allows researchers and practitioners to tackle problems with much richer and longer contextual dependencies, opening new avenues for research in long-document understanding, high-resolution image generation, and complex biological sequence analysis. The principles introduced by Reformer continue to inspire further research into efficient Transformer variants, solidifying its place as a foundational work in the field of large-scale deep learning models.

## 6. References

*   Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The Efficient Transformer. *International Conference on Learning Representations (ICLR)*.
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.
*   Gomez, A. N., Ren, M., Urtasun, R., & Hinton, G. (2017). The Reversible Residual Network: Backpropagation Without Storing Activations. *Advances in Neural Information Processing Systems (NeurIPS)*.

---
<br>

<a name="türkçe-içerik"></a>
## Reformer: Verimli Transformer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: Standart Transformer'ların Sınırlamaları](#2-arka-plan-standart-transformers-sınırlamaları)
- [3. Reformer'ın Temel Yenilikleri](#3-reformers-temel-yenilikleri)
  - [3.1. Konum Duyarlı Hashing (LSH) Dikkat Mekanizması](#31-konum-duyarlı-hashing-lsh-dikkat-mekanizması)
  - [3.2. Bellek Verimliliği için Tersine Çevrilebilir Katmanlar](#32-bellek-verimliliği-için-tersine-çevrilebilir-katmanlar)
  - [3.3. İleri Beslemeli Ağlar için Parçalama (Chunking)](#33-ileri-beslemeli-ağlar-için-parçalama-chunking)
  - [3.4. Eksenel Konumsal Kodlamalar](#34-eksenel-konumsal-kodlamalar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Referanslar](#6-referanslar)

---

## 1. Giriş

Vaswani ve arkadaşları tarafından 2017'de tanıtılan **Transformer** mimarisi, Doğal Dil İşleme (NLP) alanında devrim yaratmış ve sonrasında bilgisayar görüsü ve pekiştirmeli öğrenme gibi diğer birçok alanı etkilemiştir. **Kendi kendine dikkat mekanizmasına** dayanması, uzun menzilli bağımlılıkları etkili bir şekilde modellemesine olanak tanıyarak sayısız görevde son teknoloji sonuçlara yol açmıştır. Ancak, Transformer'ların olağanüstü başarısı, özellikle uzun girdi dizileri için bellek ve zaman karmaşıklığı açısından önemli bir hesaplama maliyetiyle birlikte gelir. Birincil darboğaz, bir dizideki tüm jetonlar arasındaki ikili etkileşimleri hesaplayan ve dizi uzunluğuna göre kuadratik bellek ve zaman karmaşıklığına neden olan kendi kendine dikkat mekanizmasında yatmaktadır. Bu doğal sınırlama, Transformer'ların yalnızca orta uzunluktaki dizilere pratik uygulanabilirliğini kısıtlamakta, çok uzun belgeler, yüksek çözünürlüklü görüntüler veya kapsamlı genomik veriler içeren görevler için önemli bir zorluk teşkil etmektedir.

Kitaev ve arkadaşları tarafından 2020'de önerilen **Reformer** modeli, Transformer'ların performans yeteneklerini korurken bellek ayak izini ve hesaplama maliyetini önemli ölçüde azaltmak için tasarlanmış bir dizi ustaca mimari değişikliği tanıtarak bu kritik verimlilik zorluklarını ele almaktadır. Reformer, daha önce mümkün olandan kat kat daha uzun dizileri işlemenin önünü açarak yeni uygulamalar sağlıyor ve dikkat tabanlı modellerle nelerin başarılabileceğinin sınırlarını zorluyor. Bu belge, **Konum Duyarlı Hashing (LSH) Dikkat Mekanizması**, **tersine çevrilebilir katmanlar**, **parçalama (chunking)** ve **eksenel konumsal kodlamalar** aracılığıyla Reformer'ın nasıl olağanüstü verimlilik kazanımları elde ettiğini açıklayarak temel yeniliklerini inceleyecektir.

## 2. Arka Plan: Standart Transformer'ların Sınırlamaları

Reformer'ın yeniliklerini takdir etmek için standart Transformer mimarisinin sınırlamalarını anlamak çok önemlidir. Tipik bir Transformer, her biri birden çok özdeş katmandan oluşan bir kodlayıcı ve bir kod çözücü yığınından oluşur. Her katman bir **çok kafalı kendi kendine dikkat mekanizması** ve bir **konuma bağlı ileri beslemeli ağ** içerir, ardından kalıntı bağlantılar ve katman normalleştirmesi gelir.

En önemli hesaplama ve bellek darboğazı **kendi kendine dikkat mekanizmasından** kaynaklanır. $L$ uzunluğundaki bir girdi dizisi ve $D_k$ gizli boyutu için dikkat mekanizması, sorgu (Q), anahtar (K) ve değer (V) matrislerini hesaplar. Dikkat ağırlıkları daha sonra $softmax(QK^T / \sqrt{D_k})V$ olarak hesaplanır.
*   **Dikkat Mekanizmasının Kuadratik Karmaşıklığı:** $QK^T$ hesaplaması, $L \times D_k$ boyutunda bir matrisin $D_k \times L$ boyutunda bir matrisle çarpılmasını içerir ve bu da $L \times L$ boyutunda bir dikkat matrisiyle sonuçlanır. Bu işlemin zaman karmaşıklığı $O(L^2 D_k)$'dir ve $L \times L$ boyutunda bir matrisin depolanmasını gerektirerek $O(L^2)$ bellek karmaşıklığına yol açar. Çok uzun diziler için bu kuadratik ölçeklendirme engelleyici hale gelir. Örneğin, 65.536 uzunluğundaki bir dizi, $O(4 \times 10^9)$ boyutunda bir dikkat matrisi gerektirir ve terabaytlarca bellek tüketir.
*   **Geriye Yayılımda Aktifleştirmeler için Bellek:** Eğitim sırasında, standart Transformer katmanları, geriye yayılım yoluyla gradyanları hesaplamak için tüm katmanların ara aktifleştirmelerini depolar. $N$ katman varsa, bu $O(N \cdot L \cdot D_{model})$ bellek anlamına gelir; burada $D_{model}$ modelin gizli boyutudur. Bu bellek tüketimi, dikkat mekanizması bunu kaldırabilse bile, mevcut donanımda işlenebilecek maksimum dizi uzunluğunu daha da sınırlar.
*   **Konumsal Kodlamalar:** Standart mutlak konumsal kodlamalar genellikle sabit bir maksimum uzunluğa sahiptir veya interpolasyon performansları eğitimli uzunlukların ötesinde bozulur, bu da onları son derece uzun ve değişken dizi uzunlukları için daha az uygun hale getirir.

Bu sınırlamalar, çok uzun belgelerin modellenmesi, yüksek çözünürlüklü görüntüler veya kapsamlı bağlamsal bilgiye sahip genomik diziler gibi görevlere Transformer'ların uygulanmasını topluca engeller. Reformer, bu sorunları doğrudan yeni tasarımıyla ele alır.

## 3. Reformer'ın Temel Yenilikleri

Reformer, standart Transformer'ların bellek ve hesaplama yükünü azaltmak için dört temel yenilik sunar: Konum Duyarlı Hashing (LSH) Dikkat Mekanizması, tersine çevrilebilir katmanlar, ileri beslemeli ağlar için parçalama ve eksenel konumsal kodlamalar.

### 3.1. Konum Duyarlı Hashing (LSH) Dikkat Mekanizması

Standart Transformer'dan en radikal ayrılış, nokta çarpım kendi kendine dikkat mekanizmasının **Konum Duyarlı Hashing (LSH) Dikkat Mekanizması** ile değiştirilmesidir. $O(L^2)$ darboğazına yol açan her sorgu ile her anahtar arasındaki dikkat skorlarını hesaplamak yerine, LSH Dikkat Mekanizması, belirli bir sorguya "yakın" olan anahtarların yalnızca bir alt kümesi üzerinde dikkat hesaplamayı hedefler.

Temel fikir, birbirine benzer veri noktalarını gruplamak için kullanılan bir teknik olan **Konum Duyarlı Hashing (LSH)**'e dayanır. Dikkat bağlamında, LSH, aynı anahtarlara dikkat etmesi muhtemel sorguları gruplar.
1.  **Hashing:** Her sorgu vektörü $Q$ ve anahtar vektörü $K$, daha düşük boyutlu bir uzaya yansıtılır ve ardından bir hash fonksiyonuna göre bir "kovaya" atanır. Orijinal gömme uzayında "yakın" olan sorgular ve anahtarların aynı hash kovasına düşme olasılığı daha yüksektir. Reformer, vektörlerin rastgele bir hiperdüzleme yansıtıldığı ve ardından yansıtmanın işaretine göre kovalarına atandığı rastgele rotasyon hashing'ini kullanır. Sağlamlığı artırmak ve çarpışma hatalarını azaltmak için genellikle birden çok hash fonksiyonu (örn., `num_hashes` = 8) kullanılır ve sorgular hash kova atamalarına göre sıralanır.
2.  **Kovalar İçindeki Dikkat:** Hashing'den sonra, sorgular atanmış kovalarına göre sıralanır. Dikkat daha sonra yalnızca *aynı* hash kovasındaki sorgular ve anahtarlar arasında hesaplanır. Bu, ikili karşılaştırma sayısını önemli ölçüde azaltır.
3.  **Parçalama ve Önbellekleme:** Daha fazla optimizasyon için sıralanmış sorgular parçalar halinde işlenir. Her parça için dikkat hesaplaması yapılır, ancak kritik olarak, *aynı* kovaya düşen *önceki* parçadan anahtarlar da dahil edilir. Bu, bir sorgunun, hemen yakınındaki parçada olmasa bile, aynı hash kovasını paylaşması koşuluyla ilgili anahtarlara hala dikkat edebilmesini sağlar.

Bu yaklaşım, dikkat karmaşıklığını $O(L^2)$'den $O(L \log L)$'e düşürür; bu, uzun diziler için önemli bir iyileşmedir. LSH Dikkat Mekanizması tam dikkatin bir yaklaştırması olsa da, ampirik sonuçlar, özellikle birden fazla hash turu kullanıldığında, birçok görev için standart dikkat mekanizmasıyla karşılaştırılabilir performans gösterdiğini göstermektedir.

### 3.2. Bellek Verimliliği için Tersine Çevrilebilir Katmanlar

Standart Transformer'lar, geriye yayılım için her katmandan gelen aktifleştirmeleri depolaması nedeniyle eğitim sırasında önemli miktarda bellek tüketir. Reformer, **RevNet** mimarisinden esinlenerek **tersine çevrilebilir katmanları** benimseyerek bunu hafifletir. Tersine çevrilebilir bir kalıntı ağında, bir katmanın çıktıları o katmanın girdilerini mükemmel bir şekilde yeniden yapılandırmak için kullanılabilir. Bu, tüm katmanlar için ara aktifleştirmeleri depolama ihtiyacını ortadan kaldırır. Bunun yerine, yalnızca son katmanın aktifleştirmelerinin depolanması gerekir; önceki tüm aktifleştirmeler, geriye doğru geçiş sırasında anında yeniden hesaplanabilir.

Tersine çevrilebilir katmanlar, girdi aktifleştirmesi $X$'i iki kısma, $X_1$ ve $X_2$'ye ayırarak bunu başarır. Tipik bir Transformer bloğu iki ana alt katmanı içerir: bir dikkat alt katmanı $F$ ve bir ileri beslemeli alt katman $G$. Reformer'ın tersine çevrilebilir katmanlarında, güncelleme kuralı şu şekilde yapılandırılır:
$Y_1 = X_1 + F(X_2)$
$Y_2 = X_2 + G(Y_1)$

Önemli olarak, $(Y_1, Y_2)$ verildiğinde, $(X_1, X_2)$ şu kullanılarak yeniden yapılandırılabilir:
$X_2 = Y_2 - G(Y_1)$
$X_1 = Y_1 - F(X_2)$

Bu tersine çevrilebilir tasarım, eğitim sırasında aktifleştirmeleri depolamanın bellek maliyetinin, standart bir Transformer'da $N$ katman için $O(N)$ yerine, katman sayısına göre $O(1)$ olacağı anlamına gelir. Bu, derinlemesine bellek tasarrufu sağlar ve aynı bellek bütçesi içinde daha derin modellerin veya daha da önemlisi çok daha uzun dizilerin eğitilmesine olanak tanır.

### 3.3. İleri Beslemeli Ağlar için Parçalama (Chunking)

LSH Dikkat Mekanizması ve tersine çevrilebilir katmanlarla bile, **ileri beslemeli ağlar (FFN'ler)**, özellikle çok uzun diziler işlenirken hala önemli bellek tüketebilir. Standart bir FFN tipik olarak büyük bir ağırlık matrisi ile matris çarpımını, ardından bir aktivasyon fonksiyonunu ve başka bir matris çarpımını içerir. FFN'lerin zaman karmaşıklığı genellikle $O(L \cdot D_{model}^2)$ olsa da, ara aktivasyon katmanı çok büyük bir boyuta sahip olabilir (örn., $4 \cdot D_{model}$), özellikle geriye doğru geçiş sırasında gradyanlar için ileri geçiş sırasında depolamak için önemli bellek gerektirebilir.

Reformer bunu, ileri beslemeli ağ hesaplamalarına **parçalama (chunking)** uygulayarak ele alır. Tüm diziyi bir seferde FFN'den geçirmek yerine, girdi dizisi daha küçük parçalara bölünür. Her parça daha sonra FFN aracılığıyla bağımsız olarak işlenir. Bu, herhangi bir zamanda yalnızca tek bir parçanın aktifleştirmelerinin bellekte tutulması gerektiğinden, en yüksek bellek gereksinimini önemli ölçüde azaltır. Bu yaklaşım, FFN'ler için bellek ayak izini $O(L \cdot D_{model})$'den $O(\text{parça\_boyutu} \cdot D_{model})$'e düşürerek, FFN hesaplamaları sırasında belleğin tükenmeden son derece uzun dizilerin işlenmesini mümkün kılar.

### 3.4. Eksenel Konumsal Kodlamalar

Yüksek çözünürlüklü görüntülerin tek bir diziye düzleştirilmesi veya çok uzun metin belgeleri gibi son derece uzun dizileri içeren senaryolar için, standart mutlak konumsal kodlamalar verimsiz hale gelebilir veya etkinliğini kaybedebilir. Geleneksel konumsal kodlamalar genellikle dizi boyutu boyunca tek bir vektör eklemesi olarak uygulanır. Dizi uzunluğu astronomik derecede büyüdüğünde, bu kodlamalar ince taneli konumları temsil etmekte veya etkili bir şekilde ölçeklenmekte zorlanabilir.

Reformer, bu tür durumlar için **eksenel konumsal kodlamalar** önerir. Konumu tek bir vektör olarak temsil etmek yerine, her biri farklı bir "eksen" veya boyuta karşılık gelen iki (veya daha fazla) vektörün toplamı olarak temsil eder. Örneğin, 1D bir diziye düzleştirilmiş 2D bir görüntü, $i$ konumunu $(x, y)$ koordinatlarına eşleyebilir; burada $i = x \cdot \text{genişlik} + y$. Eksenel konumsal kodlamalar daha sonra $x$ için ayrı bir gömme ve $y$ için başka bir gömme ekleyecektir.
Bu, konumsal kodlamayı faktörize eder ve eşit derecede büyük boyutlarda bir gömme tablosu gerektirmeden çok daha büyük etkili dizi uzunluklarını kapsamasını sağlar. Özellikle, konumların `(yükseklik_konumu, genişlik_konumu)` olarak düşünülebileceği, doğal olarak birden çok uzamsal veya zamansal boyuta sahip girdiler için (örneğin, görüntüler) kullanışlıdır. Bu, gereken gömme matrisinin boyutunu azaltır ve bileşim yoluyla görülmemiş, daha uzun dizi uzunluklarına genelleme yeteneğini potansiyel olarak artırır.

## 4. Kod Örneği

Aşağıdaki Python kodu parçacığı, LSH Dikkat Mekanizması'nın temel bir bileşeni olan **Konum Duyarlı Hashing (LSH)** fonksiyonunun sorguları kovalara nasıl gruplandırabileceğine dair basitleştirilmiş kavramsal bir görünüm sunar. Tam dikkat mekanizmasını uygulamaz, ancak hash adımını gösterir.

```python
import torch
import torch.nn.functional as F

def lsh_hash_simple(vectors, num_hashes, hash_size):
    """
    Basitleştirilmiş kavramsal bir LSH hash fonksiyonu.
    Gerçek bir Reformer'da bu, rastgele rotasyonları içerecektir.
    Bu örnek, gösterim amacıyla basit bir projeksiyon kullanır.

    Argümanlar:
        vectors (torch.Tensor): Girdi vektörleri (sorgular/anahtarlar), şekil (batch_size, seq_len, dim)
        num_hashes (int): Bağımsız hash fonksiyonlarının sayısı.
        hash_size (int): Her hash fonksiyonu başına kova sayısı (örn., 2'nin kuvveti).

    Döndürür:
        torch.Tensor: Hash kova atamaları, şekil (batch_size, seq_len, num_hashes)
    """
    batch_size, seq_len, dim = vectors.shape

    # Basitlik için, sabit bir rastgele projeksiyon kümesi kullanacağız.
    # Gerçek bir LSH'de bunlar dinamik olarak üretilir veya öğrenilirdi.
    # Şekil: (num_hashes, dim, hash_size) -- kavramsal, çünkü gerçek hash daha basittir.
    # Kovalama işleminin daha basit bir örneği için rastgele sabit yönler kullanalım.
    # Her hash fonksiyonu vektörü rastgele bir vektöre yansıtır ve sonucu niceleştirir.
    
    # Projeksiyon için num_hashes adet rastgele vektör üretin (rastgele düzlemleri simüle eder)
    # Örnekte tekrarlanabilirlik için sabit bir seed kullanılıyor
    torch.manual_seed(0) 
    random_projections = torch.randn(num_hashes, dim)
    
    # Projeksiyonları birim vektörlere normalize edin
    random_projections = F.normalize(random_projections, p=2, dim=-1)

    # Her hash fonksiyonu için nokta çarpımlarını hesaplayın
    # (batch_size, seq_len, dim) @ (num_hashes, dim) -> (batch_size, seq_len, num_hashes)
    projections = torch.einsum('bsd,hd->bsh', vectors, random_projections)

    # Projeksiyon sonucuna göre kovalara atama yapın.
    # Burada, basitçe argmax veya ölçeklendirilmiş/niceleştirilmiş bir sürüm alıyoruz.
    # Basit bir kova ataması için, işareti kullanabilir veya bir tamsayı aralığına eşleyebiliriz.
    # Açısal uzaklık için yaygın bir LSH hilesi, rastgele düzlemler kullanmak ve vektörün hangi tarafa düştüğünü kontrol etmektir.
    # Projeksiyon değerini 'hash_size' kovasına niceleştirelim.
    
    # Değerleri pozitif ve kovalama için bir aralıkta yapmak için ölçeklendirin ve kaydırın
    # Bu çok basitleştirilmiş bir kovalama yöntemidir. Gerçek LSH fonksiyonları daha sağlamdır.
    scaled_projections = (projections - projections.min()) / (projections.max() - projections.min() + 1e-9)
    bucket_assignments = (scaled_projections * (hash_size - 1)).long()

    return bucket_assignments

# Örnek kullanım:
batch_size = 2
seq_len = 10
dim = 64
num_hashes = 2
hash_size = 4 # örn., her hash fonksiyonu başına 4 kova

# Bazı yapay sorgu vektörleri oluşturun
queries = torch.randn(batch_size, seq_len, dim)

# LSH kovalarını hesaplayın
buckets = lsh_hash_simple(queries, num_hashes, hash_size)

print("Sorgu vektörlerinin şekli:", queries.shape)
print("Hesaplanan LSH kovalarının şekli:", buckets.shape)
print("Batch 0, Hash 0 için örnek kovalar:\n", buckets[0, :, 0])
print("Batch 0, Hash 1 için örnek kovalar:\n", buckets[0, :, 1])
print("\nNot: Gerçek bir LSH Dikkat Mekanizmasında, sorgular bu kovalara göre sıralanır,")
print("ve dikkat yalnızca aynı veya bitişik kovalar içinde veya arasında hesaplanırdı.")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

Reformer modeli, verimli Transformer mimarileri arayışında anıtsal bir başarı olarak durmaktadır. Kendi kendine dikkatin kuadratik bellek ve zaman karmaşıklığı ile eğitim sırasında aktifleştirmeler için doğrusal bellek ölçeklendirmesi gibi en önemli iki darboğazı akıllıca ele alarak, Reformer, dikkat tabanlı modellerin benzeri görülmemiş uzunluktaki dizileri işlemesi potansiyelini açığa çıkarmıştır.

Başlıca yenilikleri olan **Konum Duyarlı Hashing (LSH) Dikkat Mekanizması**, tam dikkat matrisi hesaplamasını sub-kuadratik ($O(L \log L)$) ölçeklenen bir yaklaşımla değiştirir. **Tersine çevrilebilir katmanların** benimsenmesi, geriye yayılım sırasında bellek tüketimini $O(N \cdot L \cdot D_{model})$'den $O(L \cdot D_{model})$'e (veya katman sayısına göre $O(1)$'e) düşürerek, daha derin modelleri ve daha uzun dizileri uygulanabilir hale getirir. Ayrıca, **ileri beslemeli ağlar için parçalama** ve **eksenel konumsal kodlamalar** ek bellek tasarrufu ve son derece uzun, çok boyutlu girdilerin daha iyi ele alınmasını sağlar.

Reformer'ın katkıları, yalnızca Transformer'ları daha hızlı ve daha bellek verimli hale getirmenin ötesine geçmektedir; bunlar, gerçekten ölçeklenebilir ve evrensel olarak uygulanabilir dikkat mekanizmaları geliştirme yolunda önemli bir adımı temsil etmektedir. Bu verimlilik, araştırmacıların ve uygulayıcıların çok daha zengin ve daha uzun bağlamsal bağımlılıkları olan sorunlarla başa çıkmalarına olanak tanıyarak, uzun belge anlama, yüksek çözünürlüklü görüntü üretimi ve karmaşık biyolojik dizi analizi alanlarında yeni araştırma yolları açmaktadır. Reformer tarafından tanıtılan ilkeler, verimli Transformer varyantları üzerine daha fazla araştırmaya ilham vermeye devam ederek, büyük ölçekli derin öğrenme modelleri alanında temel bir eser olarak yerini sağlamlaştırmaktadır.

## 6. Referanslar

*   Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The Efficient Transformer. *International Conference on Learning Representations (ICLR)*.
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.
*   Gomez, A. N., Ren, M., Urtasun, R., & Hinton, G. (2017). The Reversible Residual Network: Backpropagation Without Storing Activations. *Advances in Neural Information Processing Systems (NeurIPS)*.
