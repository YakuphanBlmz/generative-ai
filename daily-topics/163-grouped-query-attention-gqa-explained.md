# Grouped-Query Attention (GQA) Explained

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: From MHA to MQA](#2-background-from-mha-to-mqa)
  - [2.1 Multi-Head Attention (MHA)](#21-multi-head-attention-mha)
  - [2.2 Multi-Query Attention (MQA)](#22-multi-query-attention-mqa)
- [3. Grouped-Query Attention (GQA) Mechanism](#3-grouped-query-attention-gqa-mechanism)
  - [3.1 The GQA Principle](#31-the-gqa-principle)
  - [3.2 Operational Mechanics](#32-operational-mechanics)
  - [3.3 Trade-offs and Benefits](#33-trade-offs-and-benefits)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction

The **Transformer** architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized sequential data processing, particularly in **Natural Language Processing (NLP)**. A core component of this architecture is the **attention mechanism**, which allows the model to weigh the importance of different parts of the input sequence when processing a specific element. As **Large Language Models (LLMs)** continue to scale in size and complexity, the computational and memory demands of the attention mechanism, especially during the **inference** phase, become a significant bottleneck. **Grouped-Query Attention (GQA)** emerges as an innovative solution designed to mitigate these challenges by offering a strategic compromise between the computational intensity of traditional **Multi-Head Attention (MHA)** and the efficiency but potential quality trade-offs of **Multi-Query Attention (MQA)**. This document provides a comprehensive explanation of GQA, detailing its underlying principles, operational mechanics, and its pivotal role in enabling more efficient and scalable LLMs.

## 2. Background: From MHA to MQA

To understand GQA, it is essential to first grasp the evolution of attention mechanisms in Transformers, particularly MHA and its more memory-efficient successor, MQA.

### 2.1 Multi-Head Attention (MHA)

In **Multi-Head Attention (MHA)**, the input queries (Q), keys (K), and values (V) are first linearly transformed into multiple distinct "heads." Each head then independently computes scaled dot-product attention. For a given input sequence, if there are `N` attention heads, MHA generates `N` independent sets of query, key, and value matrices (`Q_i`, `K_i`, `V_i` for `i=1...N`). Each head thus learns a different set of attention weights, allowing the model to capture diverse relationships and semantic meanings across different representation subspaces. The outputs from all heads are then concatenated and linearly projected back to the original dimension. While highly effective at capturing rich contextual information, MHA can be computationally expensive and memory-intensive, especially for long sequences and models with many heads, as each head requires its own distinct key and value projections.

### 2.2 Multi-Query Attention (MQA)

**Multi-Query Attention (MQA)** was proposed to address the inference-time latency and memory footprint issues of MHA, particularly concerning the **Key-Value (KV) cache**. During auto-regressive decoding in LLMs, previously computed keys and values are stored in a cache to avoid recomputation, significantly speeding up subsequent token generation. In MHA, each attention head maintains its own independent KV cache, which grows linearly with the sequence length and the number of heads. MQA drastically reduces this by having all attention heads share a *single* set of key and value projections. This means instead of `N` sets of `K` and `V` matrices, there is only one `K` and one `V` matrix, which are then used by all query heads. This significantly shrinks the KV cache size, leading to substantial reductions in memory usage and improvements in inference speed. However, this shared KV mechanism can sometimes lead to a slight degradation in model quality or convergence speed compared to MHA, as the capacity for diverse attention patterns might be reduced.

## 3. Grouped-Query Attention (GQA) Mechanism

**Grouped-Query Attention (GQA)** represents a clever compromise between the high quality of MHA and the high efficiency of MQA. It aims to achieve MQA's efficiency benefits without fully sacrificing MHA's representational capacity.

### 3.1 The GQA Principle

The core idea behind GQA is to group the query heads. Instead of having each query head compute its own unique keys and values (MHA) or having all query heads share a single set of keys and values (MQA), GQA assigns a *group* of query heads to share a common set of key and value projections. Specifically, if there are `N` query heads and `G` key/value groups, then each group of `N/G` query heads shares one distinct key and value projection. When `G=1`, GQA reduces to MQA. When `G=N` (where `N` is the number of query heads), GQA becomes equivalent to MHA. By choosing an intermediate value for `G` (i.e., `1 < G < N`), GQA allows for a tunable trade-off between MHA's expressiveness and MQA's efficiency.

### 3.2 Operational Mechanics

Let's formalize the operational mechanics:

1.  **Query Projections:** The input `Q` is projected into `N` distinct query heads, similar to MHA. Each query head `Q_i` (where `i` ranges from `1` to `N`) maintains its independent parameters.
2.  **Key and Value Projections:** Instead of `N` separate key/value projections (MHA) or `1` shared key/value projection (MQA), GQA employs `G` distinct key and value projections. Let these be `K_j'` and `V_j'` for `j` from `1` to `G`.
3.  **Grouping:** The `N` query heads are divided into `G` groups. Each group `g` consists of `N/G` query heads. All query heads within a particular group `g` share the same `K_g'` and `V_g'` projections.
4.  **Attention Computation:** For each query head `Q_i`, the attention is computed using its own `Q_i` and the shared `K_j'` and `V_j'` corresponding to its group `j`.
    The attention for a query head `Q_i` belonging to group `j` is calculated as:
    `Attention(Q_i, K_j', V_j') = Softmax((Q_i K_j'^T) / sqrt(d_k)) V_j'`
5.  **Concatenation and Output:** The outputs from all `N` attention heads are concatenated and linearly projected to produce the final output, identical to MHA.

This setup significantly reduces the size of the KV cache compared to MHA by a factor of `N/G` and improves parallelism during inference compared to MQA, while retaining more diverse attention patterns than MQA.

### 3.3 Trade-offs and Benefits

**Benefits of GQA:**

*   **Improved Inference Speed and Latency:** By reducing the number of KV projections from `N` to `G`, GQA drastically shrinks the KV cache size, leading to faster memory access and reduced memory bandwidth requirements during inference, particularly for long sequences. This translates to lower latency in auto-regressive decoding.
*   **Reduced Memory Footprint:** The smaller KV cache directly results in lower memory consumption, allowing for the deployment of larger models or longer context windows on the same hardware.
*   **Near-MHA Quality:** GQA maintains a higher degree of expressiveness than MQA because multiple distinct key and value sets are still employed (albeit shared by groups of query heads). This often leads to performance very close to MHA in terms of perplexity or other quality metrics, avoiding the quality degradation sometimes observed with MQA.
*   **Tunable Efficiency-Quality Balance:** The number of groups `G` is a hyperparameter that can be tuned. This allows practitioners to explicitly balance the trade-off between computational efficiency (smaller `G` closer to MQA) and model quality (larger `G` closer to MHA) based on specific application requirements.

**Considerations:**

*   **Hyperparameter Tuning:** Selecting the optimal `G` value requires some experimentation, as it is dataset and model architecture dependent.
*   **Slight Overhead vs. MQA:** While vastly more efficient than MHA, GQA does introduce a minor overhead compared to pure MQA (when `G=1`) due to having more than one shared KV projection, though this is usually negligible given the performance gains.

## 4. Code Example

A conceptual Python example illustrating the core idea of GQA, where `num_query_heads` are grouped to share `num_kv_groups` for keys and values.

```python
import torch
import torch.nn as nn
import math

class GQAAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_groups, head_dim):
        super().__init__()
        assert num_query_heads % num_kv_groups == 0, "num_query_heads must be divisible by num_kv_groups"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_groups # Number of KV groups/heads
        self.head_dim = head_dim
        self.scaling = self.head_dim ** -0.5

        # Linear layers for Query, Key, Value projections
        # Queries are independent per head (num_query_heads * head_dim)
        # Keys and Values are grouped (num_kv_groups * head_dim)
        self.q_proj = nn.Linear(embed_dim, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, embed_dim, bias=False)

    def forward(self, query, key, value, attention_mask=None):
        batch_size, seq_len, _ = query.size()

        # Project Q, K, V
        # Q: (B, S, num_query_heads * head_dim) -> (B, S, num_query_heads, head_dim)
        # K, V: (B, S, num_kv_groups * head_dim) -> (B, S, num_kv_groups, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Replicate K, V for query heads within each group
        # This is where GQA's core idea is implemented:
        # Each num_query_heads // num_kv_heads query heads share the same K, V.
        # k_expanded: (B, S, num_query_heads, head_dim)
        # v_expanded: (B, S, num_query_heads, head_dim)
        k_expanded = k.unsqueeze(2).expand(-1, -1, self.num_query_heads // self.num_kv_heads, -1, -1).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        v_expanded = v.unsqueeze(2).expand(-1, -1, self.num_query_heads // self.num_kv_heads, -1, -1).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)

        # Transpose for attention: (B, num_heads, S, head_dim)
        q = q.transpose(1, 2)
        k_expanded = k_expanded.transpose(1, 2)
        v_expanded = v_expanded.transpose(1, 2)

        # Scaled Dot-Product Attention
        # (B, num_heads, S, head_dim) @ (B, num_heads, head_dim, S) -> (B, num_heads, S, S)
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # (B, num_heads, S, S) @ (B, num_heads, S, head_dim) -> (B, num_heads, S, head_dim)
        attn_output = torch.matmul(attn_weights, v_expanded)

        # Concatenate heads and project back to embed_dim
        # (B, num_heads, S, head_dim) -> (B, S, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_query_heads * self.head_dim)
        output = self.out_proj(attn_output)

        return output

# Example usage:
# embed_dim = 768 # Dimensionality of the model's embeddings
# num_query_heads = 12 # Total number of query attention heads
# num_kv_groups = 4 # Number of groups for K, V (MQA if 1, MHA if num_query_heads)
# head_dim = embed_dim // num_query_heads # Dimension of each attention head

# gqa_layer = GQAAttention(embed_dim, num_query_heads, num_kv_groups, head_dim)
# input_tensor = torch.randn(1, 10, embed_dim) # Batch size 1, sequence length 10, embed_dim

# output_tensor = gqa_layer(input_tensor, input_tensor, input_tensor)
# print(output_tensor.shape) # Expected: (1, 10, embed_dim)

(End of code example section)
```

## 5. Conclusion

Grouped-Query Attention (GQA) represents a crucial advancement in the design of efficient Transformer architectures, particularly vital for the development and deployment of **Large Language Models (LLMs)**. By striking a thoughtful balance between the high representational power of **Multi-Head Attention (MHA)** and the superior inference efficiency of **Multi-Query Attention (MQA)**, GQA enables significant reductions in **Key-Value (KV) cache** memory footprint and improvements in **inference latency** without a substantial drop in model quality. Its configurable grouping parameter offers developers the flexibility to optimize for specific performance-quality trade-offs, making it an indispensable technique for pushing the boundaries of what is achievable with scaled-up generative AI models. As LLMs continue to grow, mechanisms like GQA will remain central to addressing their increasing computational demands, paving the way for more powerful and accessible AI systems.

---
<br>

<a name="türkçe-içerik"></a>
## Gruplandırılmış Sorgu Dikkat Mekanizması (GQA) Açıklaması

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: MHA'dan MQA'ya](#2-arka-plan-mhadan-mqaya)
  - [2.1 Çoklu Başlı Dikkat Mekanizması (MHA)](#21-çoklu-başlı-dikkat-mekanizması-mha)
  - [2.2 Çoklu Sorgu Dikkat Mekanizması (MQA)](#22-çoklu-sorgu-dikkat-mekanizması-mqa)
- [3. Gruplandırılmış Sorgu Dikkat Mekanizması (GQA)](#3-gruplandırılmış-sorgu-dikkat-mekanizması-gqa)
  - [3.1 GQA Prensibi](#31-gqa-prensibi)
  - [3.2 Operasyonel Mekanik](#32-operasyonel-mekanik)
  - [3.3 Değiş Tokuşlar ve Faydalar](#33-değiş-tokuşlar-ve-faydalar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş

"Attention Is All You Need" (Vaswani ve diğerleri, 2017) makalesinde tanıtılan **Transformer** mimarisi, özellikle **Doğal Dil İşleme (NLP)** alanında sıralı veri işleme konusunda devrim yarattı. Bu mimarinin temel bir bileşeni, modelin bir elemanı işlerken giriş dizisinin farklı kısımlarının önemini tartmasına olanak tanıyan **dikkat mekanizmasıdır**. **Büyük Dil Modelleri (BDM'ler)** boyut ve karmaşıklık açısından büyümeye devam ettikçe, dikkat mekanizmasının özellikle **çıkarım (inference)** aşamasındaki hesaplama ve bellek gereksinimleri önemli bir darboğaz haline gelmektedir. **Gruplandırılmış Sorgu Dikkat Mekanizması (GQA)**, geleneksel **Çoklu Başlı Dikkat Mekanizması (MHA)**'nın hesaplama yoğunluğu ile **Çoklu Sorgu Dikkat Mekanizması (MQA)**'nın verimliliği arasındaki stratejik bir uzlaşma sunarak bu zorlukları hafifletmek için tasarlanmış yenilikçi bir çözüm olarak ortaya çıkmıştır. Bu belge, GQA'nın altında yatan prensipleri, operasyonel mekaniklerini ve daha verimli ve ölçeklenebilir BDM'ler sağlamadaki önemli rolünü detaylandıran kapsamlı bir açıklama sunmaktadır.

## 2. Arka Plan: MHA'dan MQA'ya

GQA'yı anlamak için, Transformer'lardaki dikkat mekanizmalarının evrimini, özellikle MHA'yı ve onun daha bellek verimli halefi MQA'yı kavramak önemlidir.

### 2.1 Çoklu Başlı Dikkat Mekanizması (MHA)

**Çoklu Başlı Dikkat Mekanizması (MHA)**'nda, girdi sorguları (Q), anahtarlar (K) ve değerler (V) ilk olarak birden çok farklı "başa" doğrusal olarak dönüştürülür. Her başlık daha sonra bağımsız olarak ölçeklenmiş nokta-çarpım dikkatini hesaplar. Belirli bir girdi dizisi için, `N` dikkat başlığı varsa, MHA `N` bağımsız sorgu, anahtar ve değer matrisleri (`i=1...N` için `Q_i`, `K_i`, `V_i`) kümesi oluşturur. Her başlık böylece farklı dikkat ağırlıkları öğrenir ve modelin farklı temsil alt uzaylarında çeşitli ilişkileri ve anlamsal anlamları yakalamasına olanak tanır. Tüm başlıkların çıktıları daha sonra birleştirilir ve orijinal boyuta doğrusal olarak geri yansıtılır. Zengin bağlamsal bilgiyi yakalamada oldukça etkili olmasına rağmen, MHA, özellikle uzun diziler ve çok sayıda başlığa sahip modeller için hesaplama açısından pahalı ve bellek açısından yoğun olabilir, çünkü her başlık kendi ayrı anahtar ve değer projeksiyonlarına ihtiyaç duyar.

### 2.2 Çoklu Sorgu Dikkat Mekanizması (MQA)

**Çoklu Sorgu Dikkat Mekanizması (MQA)**, MHA'nın çıkarım zamanı gecikmesi ve bellek ayak izi sorunlarını, özellikle **Anahtar-Değer (KV) önbelleği** ile ilgili olanları ele almak için önerilmiştir. BDM'lerde otomatik gerileyen kod çözme sırasında, önceden hesaplanan anahtarlar ve değerler, yeniden hesaplamayı önlemek için bir önbellekte saklanır, bu da sonraki jeton üretimini önemli ölçüde hızlandırır. MHA'da, her dikkat başlığı kendi bağımsız KV önbelleğini tutar, bu da dizi uzunluğu ve başlık sayısıyla doğrusal olarak büyür. MQA, tüm dikkat başlıklarının *tek bir* anahtar ve değer projeksiyonu kümesini paylaşmasını sağlayarak bunu önemli ölçüde azaltır. Bu, `N` set `K` ve `V` matrisi yerine, sadece bir `K` ve bir `V` matrisi olduğu ve bunların tüm sorgu başlıkları tarafından kullanıldığı anlamına gelir. Bu, KV önbellek boyutunu önemli ölçüde küçülterek bellek kullanımında büyük azalmalar ve çıkarım hızında iyileşmeler sağlar. Ancak, bu paylaşılan KV mekanizması, çeşitli dikkat modelleri kapasitesi azalabileceğinden, MHA'ya kıyasla model kalitesinde veya yakınsama hızında bazen hafif bir düşüşe yol açabilir.

## 3. Gruplandırılmış Sorgu Dikkat Mekanizması (GQA)

**Gruplandırılmış Sorgu Dikkat Mekanizması (GQA)**, MHA'nın yüksek kalitesi ile MQA'nın yüksek verimliliği arasında akıllıca bir uzlaşmayı temsil eder. MHA'nın temsil kapasitesinden tamamen ödün vermeden MQA'nın verimlilik faydalarını elde etmeyi amaçlar.

### 3.1 GQA Prensibi

GQA'nın temel fikri, sorgu başlıklarını gruplandırmaktır. Her sorgu başlığının kendi benzersiz anahtarlarını ve değerlerini (MHA) hesaplaması veya tüm sorgu başlıklarının tek bir anahtar ve değer kümesini (MQA) paylaşması yerine, GQA, bir grup sorgu başlığını ortak bir anahtar ve değer projeksiyonları kümesini paylaşmak üzere atar. Özellikle, `N` sorgu başlığı ve `G` anahtar/değer grubu varsa, her `N/G` sorgu başlığı grubu bir ayrı anahtar ve değer projeksiyonunu paylaşır. `G=1` olduğunda, GQA, MQA'ya indirgenir. `G=N` (burada `N` sorgu başlığı sayısıdır) olduğunda, GQA, MHA'ya eşdeğer hale gelir. `G` için ara bir değer (yani, `1 < G < N`) seçerek, GQA, MHA'nın ifade gücü ile MQA'nın verimliliği arasında ayarlanabilir bir değiş tokuşa olanak tanır.

### 3.2 Operasyonel Mekanik

Operasyonel mekaniği resmileştirelim:

1.  **Sorgu Projeksiyonları:** Girdi `Q`, MHA'ya benzer şekilde `N` farklı sorgu başlığına yansıtılır. Her sorgu başlığı `Q_i` (burada `i`, `1`'den `N`'ye kadar değişir) bağımsız parametrelerini korur.
2.  **Anahtar ve Değer Projeksiyonları:** `N` ayrı anahtar/değer projeksiyonu (MHA) veya `1` paylaşılan anahtar/değer projeksiyonu (MQA) yerine, GQA, `G` ayrı anahtar ve değer projeksiyonu kullanır. Bunlar, `j`, `1`'den `G`'ye kadar olmak üzere `K_j'` ve `V_j'` olsun.
3.  **Gruplandırma:** `N` sorgu başlığı `G` gruba ayrılır. Her `g` grubu `N/G` sorgu başlığından oluşur. Belirli bir `g` grubu içindeki tüm sorgu başlıkları aynı `K_g'` ve `V_g'` projeksiyonlarını paylaşır.
4.  **Dikkat Hesaplaması:** Her sorgu başlığı `Q_i` için dikkat, kendi `Q_i`'si ve grubuna `j` karşılık gelen paylaşılan `K_j'` ve `V_j'` kullanılarak hesaplanır.
    `j` grubuna ait bir sorgu başlığı `Q_i` için dikkat şöyle hesaplanır:
    `Dikkat(Q_i, K_j', V_j') = Softmax((Q_i K_j'^T) / sqrt(d_k)) V_j'`
5.  **Birleştirme ve Çıktı:** Tüm `N` dikkat başlığından gelen çıktılar birleştirilir ve nihai çıktıyı üretmek için doğrusal olarak yansıtılır, bu MHA ile aynıdır.

Bu kurulum, KV önbelleğinin boyutunu MHA'ya kıyasla `N/G` faktörü kadar önemli ölçüde azaltır ve MQA'ya kıyasla çıkarım sırasında paralelliği artırırken, MQA'dan daha çeşitli dikkat modellerini korur.

### 3.3 Değiş Tokuşlar ve Faydalar

**GQA'nın Faydaları:**

*   **Geliştirilmiş Çıkarım Hızı ve Gecikme:** KV projeksiyonlarının sayısını `N`'den `G`'ye düşürerek, GQA KV önbellek boyutunu büyük ölçüde küçültür, bu da özellikle uzun diziler için çıkarım sırasında daha hızlı bellek erişimi ve azaltılmış bellek bant genişliği gereksinimleri anlamına gelir. Bu, otomatik gerileyen kod çözmede daha düşük gecikmeye dönüşür.
*   **Azaltılmış Bellek Ayak İzi:** Daha küçük KV önbelleği, doğrudan daha düşük bellek tüketimine yol açar, bu da aynı donanım üzerinde daha büyük modellerin veya daha uzun bağlam pencerelerinin dağıtılmasına olanak tanır.
*   **MHA'ya Yakın Kalite:** GQA, birden çok ayrı anahtar ve değer seti hala kullanıldığından (ancak sorgu başlığı grupları tarafından paylaşılsa da) MQA'dan daha yüksek bir ifade derecesi sürdürür. Bu, genellikle kafa karışıklığı veya diğer kalite metrikleri açısından MHA'ya çok yakın bir performansla sonuçlanır ve MQA ile bazen gözlemlenen kalite düşüşünü önler.
*   **Ayarlanabilir Verimlilik-Kalite Dengesi:** `G` grup sayısı, ayarlanabilen bir hiperparametredir. Bu, uygulayıcıların belirli uygulama gereksinimlerine göre hesaplama verimliliği (MQA'ya daha yakın daha küçük `G`) ve model kalitesi (MHA'ya daha yakın daha büyük `G`) arasındaki değiş tokuşu açıkça dengelemesine olanak tanır.

**Dikkat Edilmesi Gerekenler:**

*   **Hiperparametre Ayarlaması:** Optimum `G` değerini seçmek, veri kümesi ve model mimarisine bağlı olduğundan bazı deneyler gerektirir.
*   **MQA'ya Kıyasla Hafif Ek Yük:** MHA'dan çok daha verimli olmasına rağmen, GQA, birden fazla paylaşılan KV projeksiyonuna sahip olduğu için saf MQA'ya (`G=1` olduğunda) kıyasla küçük bir ek yük getirir, ancak bu, performans kazanımları göz önüne alındığında genellikle ihmal edilebilir düzeydedir.

## 4. Kod Örneği

`num_query_heads` sorgu başlıklarının anahtarlar ve değerler için `num_kv_groups` kullanarak gruplandırılmasının temel fikrini gösteren kavramsal bir Python örneği.

```python
import torch
import torch.nn as nn
import math

class GQAAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_groups, head_dim):
        super().__init__()
        # num_query_heads, num_kv_groups'a tam bölünmelidir
        assert num_query_heads % num_kv_groups == 0, "num_query_heads, num_kv_groups'a bölünebilir olmalıdır"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_groups # KV grupları/başlıkları sayısı
        self.head_dim = head_dim
        self.scaling = self.head_dim ** -0.5

        # Sorgu, Anahtar, Değer projeksiyonları için doğrusal katmanlar
        # Sorgular her başlık için bağımsızdır (num_query_heads * head_dim)
        # Anahtarlar ve Değerler gruplandırılmıştır (num_kv_groups * head_dim)
        self.q_proj = nn.Linear(embed_dim, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_groups * head_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, embed_dim, bias=False)

    def forward(self, query, key, value, attention_mask=None):
        batch_size, seq_len, _ = query.size()

        # Q, K, V'yi yansıt
        # Q: (B, S, num_query_heads * head_dim) -> (B, S, num_query_heads, head_dim)
        # K, V: (B, S, num_kv_groups * head_dim) -> (B, S, num_kv_groups, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Her grup içindeki sorgu başlıkları için K, V'yi çoğalt
        # GQA'nın ana fikrinin uygulandığı yer burasıdır:
        # Her num_query_heads // num_kv_heads sorgu başlığı aynı K, V'yi paylaşır.
        # k_expanded: (B, S, num_query_heads, head_dim)
        # v_expanded: (B, S, num_query_heads, head_dim)
        # unsqueeze(2) ile num_kv_heads boyutu arasına yeni bir boyut ekleyerek,
        # expand ile bu boyutu num_query_heads // num_kv_heads kadar genişleterek,
        # reshape ile nihai (B, S, num_query_heads, head_dim) yapısına ulaşıyoruz.
        k_expanded = k.unsqueeze(2).expand(-1, -1, self.num_query_heads // self.num_kv_heads, -1, -1).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        v_expanded = v.unsqueeze(2).expand(-1, -1, self.num_query_heads // self.num_kv_heads, -1, -1).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)


        # Dikkat için transpoze et: (B, num_heads, S, head_dim)
        q = q.transpose(1, 2)
        k_expanded = k_expanded.transpose(1, 2)
        v_expanded = v_expanded.transpose(1, 2)

        # Ölçeklenmiş Nokta-Çarpım Dikkat Mekanizması
        # (B, num_heads, S, head_dim) @ (B, num_heads, head_dim, S) -> (B, num_heads, S, S)
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            # Dikkat maskesi varsa, 0 olan yerleri -sonsuz ile doldur
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax uygula
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # (B, num_heads, S, S) @ (B, num_heads, S, head_dim) -> (B, num_heads, S, head_dim)
        attn_output = torch.matmul(attn_weights, v_expanded)

        # Başlıkları birleştir ve embed_dim'e geri yansıt
        # (B, num_heads, S, head_dim) -> (B, S, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_query_heads * self.head_dim)
        output = self.out_proj(attn_output)

        return output

# Örnek kullanım:
# embed_dim = 768 # Modelin gömme boyutu
# num_query_heads = 12 # Toplam sorgu dikkat başlığı sayısı
# num_kv_groups = 4 # K, V için grup sayısı (1 ise MQA, num_query_heads ise MHA)
# head_dim = embed_dim // num_query_heads # Her dikkat başlığının boyutu

# gqa_katmani = GQAAttention(embed_dim, num_query_heads, num_kv_groups, head_dim)
# girdi_tensör = torch.randn(1, 10, embed_dim) # Parti boyutu 1, dizi uzunluğu 10, embed_dim

# cikti_tensör = gqa_katmani(girdi_tensör, girdi_tensör, girdi_tensör)
# print(cikti_tensör.shape) # Beklenen: (1, 10, embed_dim)

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

Gruplandırılmış Sorgu Dikkat Mekanizması (GQA), verimli Transformer mimarilerinin tasarımında kritik bir ilerlemeyi temsil eder ve özellikle **Büyük Dil Modelleri (BDM'ler)**'in geliştirilmesi ve dağıtımı için hayati öneme sahiptir. **Çoklu Başlı Dikkat Mekanizması (MHA)**'nın yüksek temsil gücü ile **Çoklu Sorgu Dikkat Mekanizması (MQA)**'nın üstün çıkarım verimliliği arasında düşünceli bir denge kurarak, GQA, **Anahtar-Değer (KV) önbellek** bellek ayak izinde önemli azalmalar ve model kalitesinde önemli bir düşüş olmadan **çıkarım gecikmesinde** iyileşmeler sağlar. Yapılandırılabilir gruplama parametresi, geliştiricilere belirli performans-kalite değiş tokuşları için optimizasyon yapma esnekliği sunar, bu da onu ölçeklendirilmiş üretken yapay zeka modelleriyle başarabileceklerimizin sınırlarını zorlamak için vazgeçilmez bir teknik haline getirir. BDM'ler büyümeye devam ettikçe, GQA gibi mekanizmalar artan hesaplama taleplerini karşılamada merkezi olmaya devam edecek ve daha güçlü ve erişilebilir yapay zeka sistemlerinin yolunu açacaktır.


