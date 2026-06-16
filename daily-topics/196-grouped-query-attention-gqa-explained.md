# Grouped-Query Attention (GQA) Explained

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: From Multi-Head to Multi-Query Attention](#2-background-from-multi-head-to-multi-query-attention)
  - [2.1. Multi-Head Attention (MHA)](#21-multi-head-attention-mha)
  - [2.2. Multi-Query Attention (MQA)](#22-multi-query-attention-mqa)
- [3. Grouped-Query Attention (GQA) Mechanism](#3-grouped-query-attention-gqa-mechanism)
  - [3.1. Core Concept](#31-core-concept)
  - [3.2. Practical Implications](#32-practical-implications)
- [4. Advantages of GQA](#4-advantages-of-gqa)
  - [4.1. Enhanced Inference Efficiency](#41-enhanced-inference-efficiency)
  - [4.2. Balanced Performance](#42-balanced-performance)
  - [4.3. Scalability and Context Length](#43-scalability-and-context-length)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The transformer architecture, primarily driven by the **attention mechanism**, has revolutionized the field of Natural Language Processing (NLP) and is fundamental to the success of modern Large Language Models (LLMs). While **Multi-Head Attention (MHA)** has been a cornerstone for capturing diverse relational information, its computational and memory footprint, especially during inference, poses significant challenges for deploying increasingly larger models. **Grouped-Query Attention (GQA)** emerges as an innovative optimization technique designed to enhance the efficiency of transformer models, particularly during the auto-regressive decoding phase, without substantially compromising model quality. This document provides a comprehensive explanation of GQA, detailing its mechanism, contrasting it with prior attention variants, and highlighting its critical advantages for the development and deployment of efficient LLMs.

## 2. Background: From Multi-Head to Multi-Query Attention
To fully appreciate the innovation behind GQA, it is essential to first understand the evolution of attention mechanisms that precede it: Multi-Head Attention (MHA) and Multi-Query Attention (MQA).

### 2.1. Multi-Head Attention (MHA)
**Multi-Head Attention (MHA)** is the standard attention mechanism introduced in the seminal "Attention Is All You Need" paper. In MHA, the input queries (Q), keys (K), and values (V) are linearly projected into multiple distinct "heads." Each head then independently performs the scaled dot-product attention function:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
where $d_k$ is the dimension of the keys. The outputs from all attention heads are then concatenated and linearly projected back to the original dimension. This parallelism allows the model to attend to different parts of the input sequence simultaneously and from different representation subspaces, enriching its understanding.
However, during auto-regressive decoding (generating one token at a time), the Key and Value states for all previously generated tokens (the **KV cache**) need to be stored. For models with a large number of attention heads and long context windows, this KV cache can become prohibitively large, consuming significant memory bandwidth and leading to slower inference. If a model has $N_h$ attention heads, it maintains $N_h$ sets of distinct Key and Value projections.

### 2.2. Multi-Query Attention (MQA)
To address the memory and bandwidth bottleneck of MHA, **Multi-Query Attention (MQA)** was proposed. In MQA, instead of having separate Key and Value projections for each attention head, all attention heads share a *single* set of Key and Value projections. This means that while the Query matrix ($Q$) is still divided into $N_h$ independent query heads, there is only one Key matrix ($K$) and one Value matrix ($V$) that are used by all query heads.
$$ \text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right)V $$
where $Q_i$ is the query for the $i$-th head, and $K, V$ are the shared key and value matrices.
The primary advantage of MQA is a drastic reduction in the size of the KV cache. Instead of storing $N_h$ sets of Keys and Values, only one set is stored. This significantly improves inference speed, especially for applications sensitive to memory bandwidth. The trade-off, however, can be a slight degradation in model quality compared to MHA, as sharing K/V projections across all heads might limit the model's capacity to learn diverse attention patterns.

## 3. Grouped-Query Attention (GQA) Mechanism
**Grouped-Query Attention (GQA)** represents an intelligent compromise between the high quality of MHA and the high efficiency of MQA. It aims to achieve MQA-like inference speeds while retaining MHA-like model quality by introducing a concept of "groups" for Key and Value heads.

### 3.1. Core Concept
In GQA, the $N_q$ query heads are divided into $G$ groups. Each group then shares a *single* set of Key and Value projections. This means that instead of $N_q$ distinct Key/Value heads (as in MHA) or 1 shared Key/Value head (as in MQA), GQA uses $N_{kv}$ Key/Value heads, where $N_{kv}$ is a divisor of $N_q$ and $N_{kv} > 1$. Specifically, $N_{kv} = N_q / G$.
If we denote the total number of query heads as $N_q$, and the number of key/value heads as $N_{kv}$:
*   **MHA:** $N_q = N_{kv}$ (each query head has its own K/V head).
*   **MQA:** $N_{kv} = 1$ (all query heads share one K/V head).
*   **GQA:** $1 < N_{kv} < N_q$ (query heads are grouped, and each group shares one K/V head).

For instance, if a model has 8 query heads:
*   In MHA, there would be 8 Key heads and 8 Value heads.
*   In MQA, there would be 1 Key head and 1 Value head.
*   In GQA, one might choose 2 Key/Value heads. This means the 8 query heads are divided into 2 groups of 4 queries each, where each group shares one K/V head. Or 4 Key/Value heads, dividing the 8 query heads into 4 groups of 2 queries each.

The process involves:
1.  Projecting the input into $N_q$ query matrices, $N_{kv}$ key matrices, and $N_{kv}$ value matrices.
2.  For each of the $G$ groups, the queries belonging to that group attend to the shared key and value matrices assigned to that group.
3.  The outputs from all query heads are concatenated and linearly projected.

### 3.2. Practical Implications
The primary benefit of GQA lies in its ability to significantly reduce the size of the KV cache compared to MHA. The memory required for the KV cache becomes proportional to $N_{kv}$ rather than $N_q$. By choosing an appropriate $N_{kv}$ (e.g., $N_{kv} = N_q / 2$ or $N_q / 4$), GQA can drastically cut down memory usage and memory bandwidth requirements during inference, leading to faster decoding. The flexibility to choose $N_{kv}$ allows developers to fine-tune the balance between efficiency and quality.

## 4. Advantages of GQA
GQA offers several compelling advantages that make it a crucial optimization for modern LLMs:

### 4.1. Enhanced Inference Efficiency
The most direct benefit of GQA is its impact on inference speed. By reducing the number of Key and Value heads, GQA drastically shrinks the size of the KV cache. This reduction directly translates to:
*   **Lower memory footprint:** Models can run on devices with less available VRAM.
*   **Reduced memory bandwidth:** Less data needs to be fetched from memory, speeding up computation, especially for long sequence lengths.
*   **Faster decoding:** Auto-regressive generation, where each new token requires accessing the KV cache, becomes significantly faster. This is particularly important for interactive applications.

### 4.2. Balanced Performance
Unlike MQA, which can sometimes lead to a noticeable drop in model quality due to aggressive sharing of K/V projections, GQA provides a more balanced approach. By allowing for multiple Key/Value groups ($1 < N_{kv} < N_q$), GQA retains more of MHA's capacity to learn diverse representations. This means GQA models can often achieve near MHA-level quality while still enjoying significant MQA-level efficiency gains. It bridges the gap effectively, offering a sweet spot for practical LLM deployment.

### 4.3. Scalability and Context Length
The reduced memory usage per token also allows GQA models to handle longer context windows. As LLMs are increasingly being used for tasks requiring extensive context (e.g., summarization of long documents, coding assistance), the ability to support larger KV caches without excessive memory overhead is vital. GQA facilitates the deployment of larger models with expanded context capabilities on existing hardware infrastructure.

## 5. Code Example
Here’s a conceptual Python-like pseudocode snippet illustrating the difference in Key/Value head counts for MHA, MQA, and GQA. This example focuses on the *number* of K/V heads, not the full attention calculation.

```python
class AttentionMechanism:
    def __init__(self, num_query_heads: int, num_key_value_heads: int):
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads

        if num_query_heads % num_key_value_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_key_value_heads for GQA.")

        self.num_groups = num_query_heads // num_key_value_heads
        print(f"Initializing attention with {num_query_heads} query heads and {num_key_value_heads} KV heads.")
        print(f"This means there are {self.num_groups} groups, each sharing 1 KV head.")

# Example 1: Multi-Head Attention (MHA)
# Each query head has its own K/V head.
mha = AttentionMechanism(num_query_heads=8, num_key_value_heads=8)
# Output: Initializing attention with 8 query heads and 8 KV heads.
#         This means there are 1 groups, each sharing 1 KV head. (Each head is its own group)

print("-" * 30)

# Example 2: Multi-Query Attention (MQA)
# All query heads share a single K/V head.
mqa = AttentionMechanism(num_query_heads=8, num_key_value_heads=1)
# Output: Initializing attention with 8 query heads and 1 KV heads.
#         This means there are 8 groups, each sharing 1 KV head.

print("-" * 30)

# Example 3: Grouped-Query Attention (GQA)
# Query heads are divided into groups, each group shares a K/V head.
gqa_example_1 = AttentionMechanism(num_query_heads=8, num_key_value_heads=2)
# Output: Initializing attention with 8 query heads and 2 KV heads.
#         This means there are 4 groups, each sharing 1 KV head.

print("-" * 30)

gqa_example_2 = AttentionMechanism(num_query_heads=8, num_key_value_heads=4)
# Output: Initializing attention with 8 query heads and 4 KV heads.
#         This means there are 2 groups, each sharing 1 KV head.

(End of code example section)
```

## 6. Conclusion
Grouped-Query Attention (GQA) stands as a significant advancement in optimizing the transformer architecture for large language models. By striking a strategic balance between the model quality of Multi-Head Attention and the inference efficiency of Multi-Query Attention, GQA enables faster, more memory-efficient auto-regressive decoding. This innovation is crucial for the practical deployment of increasingly complex LLMs, allowing them to operate effectively with larger context windows on more accessible hardware, thereby democratizing access to powerful generative AI capabilities. As LLMs continue to grow in scale and application, techniques like GQA will remain indispensable for pushing the boundaries of what is computationally feasible and economically viable in the realm of artificial intelligence.

---
<br>

<a name="türkçe-içerik"></a>
## Gruplandırılmış Sorgu Dikkat Mekanizması (GQA) Açıklandı

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: Çoklu Başlıklı Dikkatten Çoklu Sorgulu Dikkate](#2-arka-plan-çoklu-başlıklı-dikkatten-çoklu-sorgulu-dikkate)
  - [2.1. Çoklu Başlıklı Dikkat (MHA)](#21-çoklu-başlıklı-dikkat-mha)
  - [2.2. Çoklu Sorgulu Dikkat (MQA)](#22-çoklu-sorgulu-dikkat-mqa)
- [3. Gruplandırılmış Sorgu Dikkat (GQA) Mekanizması](#3-gruplandırılmış-sorgu-dikkat-gqa-mekanizması)
  - [3.1. Temel Konsept](#31-temel-konsept)
  - [3.2. Pratik Çıkarımlar](#32-pratik-çıkarımlar)
- [4. GQA'nın Avantajları](#4-gqanın-avantajları)
  - [4.1. Gelişmiş Çıkarım Verimliliği](#41-gelişmiş-çıkarım-verimliliği)
  - [4.2. Dengeli Performans](#42-dengeli-performans)
  - [4.3. Ölçeklenebilirlik ve Bağlam Uzunluğu](#43-ölçeklenebilirlik-ve-bağlam-uzunluğu)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
Öncelikle **dikkat mekanizması** tarafından yönlendirilen transformer mimarisi, Doğal Dil İşleme (NLP) alanında devrim yaratmış ve modern Büyük Dil Modellerinin (LLM'ler) başarısının temelini oluşturmuştur. **Çoklu Başlıklı Dikkat (MHA)**, çeşitli ilişkisel bilgileri yakalamak için bir köşe taşı olmasına rağmen, özellikle çıkarım (inference) sırasında ortaya çıkan hesaplama ve bellek ayak izi, giderek büyüyen modellerin dağıtımı için önemli zorluklar yaratmaktadır. **Gruplandırılmış Sorgu Dikkat (GQA)**, model kalitesinden önemli ölçüde ödün vermeden transformer modellerinin verimliliğini, özellikle otomatik gerilemeli kod çözme aşamasında artırmak için tasarlanmış yenilikçi bir optimizasyon tekniği olarak ortaya çıkmıştır. Bu belge, GQA'nın mekanizmasını detaylandırarak, önceki dikkat varyantlarıyla karşılaştırarak ve verimli LLM'lerin geliştirilmesi ve dağıtımı için kritik avantajlarını vurgulayarak kapsamlı bir açıklama sunmaktadır.

## 2. Arka Plan: Çoklu Başlıklı Dikkatten Çoklu Sorgulu Dikkate
GQA'nın ardındaki yeniliği tam olarak takdir etmek için, onu önceleyen dikkat mekanizmalarının evrimini anlamak çok önemlidir: Çoklu Başlıklı Dikkat (MHA) ve Çoklu Sorgulu Dikkat (MQA).

### 2.1. Çoklu Başlıklı Dikkat (MHA)
**Çoklu Başlıklı Dikkat (MHA)**, "Attention Is All You Need" adlı çığır açan makalede tanıtılan standart dikkat mekanizmasıdır. MHA'da, girdi sorguları (Q), anahtarlar (K) ve değerler (V) birden çok farklı "başlığa" doğrusal olarak yansıtılır. Her başlık daha sonra ölçekli nokta-çarpım dikkat fonksiyonunu bağımsız olarak gerçekleştirir:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
burada $d_k$ anahtarların boyutudur. Tüm dikkat başlıklarından gelen çıktılar daha sonra birleştirilir ve doğrusal olarak orijinal boyuta geri yansıtılır. Bu paralellik, modelin girdi dizisinin farklı bölümlerine aynı anda ve farklı temsil altuzaylarından dikkat etmesine olanak tanıyarak anlayışını zenginleştirir.
Ancak, otomatik gerilemeli kod çözme sırasında (bir seferde bir jeton üretme), daha önce üretilen tüm jetonların Anahtar ve Değer durumlarının ( **KV önbelleği** olarak adlandırılır) saklanması gerekir. Çok sayıda dikkat başlığına ve uzun bağlam pencerelerine sahip modeller için bu KV önbelleği aşırı derecede büyük hale gelebilir, önemli bellek bant genişliği tüketebilir ve daha yavaş çıkarıma yol açabilir. Bir modelin $N_h$ dikkat başlığı varsa, $N_h$ ayrı Anahtar ve Değer projeksiyonu kümesini korur.

### 2.2. Çoklu Sorgulu Dikkat (MQA)
MHA'nın bellek ve bant genişliği darboğazını gidermek için **Çoklu Sorgulu Dikkat (MQA)** önerilmiştir. MQA'da, her dikkat başlığı için ayrı Anahtar ve Değer projeksiyonlarına sahip olmak yerine, tüm dikkat başlıkları *tek bir* Anahtar ve Değer projeksiyonu kümesini paylaşır. Bu, Sorgu matrisi ($Q$) hala $N_h$ bağımsız sorgu başlığına bölünürken, tüm sorgu başlıkları tarafından kullanılan sadece bir Anahtar matrisi ($K$) ve bir Değer matrisi ($V$) olduğu anlamına gelir.
$$ \text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right)V $$
burada $Q_i$, $i$-inci başlık için sorgu, $K$ ve $V$ ise paylaşılan anahtar ve değer matrisleridir.
MQA'nın temel avantajı, KV önbelleğinin boyutunda radikal bir azalmadır. $N_h$ Anahtar ve Değer kümesi depolamak yerine, sadece bir küme depolanır. Bu, özellikle bellek bant genişliğine duyarlı uygulamalar için çıkarım hızını önemli ölçüde artırır. Ancak, bu durumun bedeli, K/V projeksiyonlarını tüm başlıklar arasında paylaşmanın modelin farklı dikkat desenlerini öğrenme kapasitesini sınırlayabilmesi nedeniyle MHA'ya kıyasla model kalitesinde hafif bir bozulma olabilir.

## 3. Gruplandırılmış Sorgu Dikkat (GQA) Mekanizması
**Gruplandırılmış Sorgu Dikkat (GQA)**, MHA'nın yüksek kalitesi ile MQA'nın yüksek verimliliği arasında akıllı bir uzlaşmayı temsil eder. Anahtar ve Değer başlıkları için "gruplar" kavramını tanıtarak, MHA benzeri model kalitesini korurken MQA benzeri çıkarım hızları elde etmeyi amaçlar.

### 3.1. Temel Konsept
GQA'da, $N_q$ sorgu başlığı $G$ gruba ayrılır. Her grup daha sonra *tek bir* Anahtar ve Değer projeksiyonu kümesini paylaşır. Bu, $N_q$ ayrı Anahtar/Değer başlığı (MHA'da olduğu gibi) veya 1 paylaşılan Anahtar/Değer başlığı (MQA'da olduğu gibi) yerine, GQA'nın $N_{kv}$ Anahtar/Değer başlığı kullandığı anlamına gelir; burada $N_{kv}$, $N_q$'nin bir bölenidir ve $N_{kv} > 1$'dir. Özellikle, $N_{kv} = N_q / G$.
Toplam sorgu başlığı sayısını $N_q$ ve anahtar/değer başlığı sayısını $N_{kv}$ olarak gösterirsek:
*   **MHA:** $N_q = N_{kv}$ (her sorgu başlığının kendi K/V başlığı vardır).
*   **MQA:** $N_{kv} = 1$ (tüm sorgu başlıkları tek bir K/V başlığını paylaşır).
*   **GQA:** $1 < N_{kv} < N_q$ (sorgu başlıkları gruplandırılır ve her grup tek bir K/V başlığını paylaşır).

Örneğin, bir modelin 8 sorgu başlığı varsa:
*   MHA'da 8 Anahtar başlığı ve 8 Değer başlığı olurdu.
*   MQA'da 1 Anahtar başlığı ve 1 Değer başlığı olurdu.
*   GQA'da, 2 Anahtar/Değer başlığı seçilebilir. Bu, 8 sorgu başlığının her biri 4 sorgudan oluşan 2 gruba ayrıldığı ve her grubun bir K/V başlığını paylaştığı anlamına gelir. Veya 4 Anahtar/Değer başlığı seçilirse, 8 sorgu başlığı her biri 2 sorgudan oluşan 4 gruba ayrılır.

Süreç şunları içerir:
1.  Girdiyi $N_q$ sorgu matrisine, $N_{kv}$ anahtar matrisine ve $N_{kv}$ değer matrisine yansıtmak.
2.  $G$ grubun her biri için, o gruba ait sorgular, o gruba atanmış paylaşılan anahtar ve değer matrislerine dikkat eder.
3.  Tüm sorgu başlıklarından gelen çıktılar birleştirilir ve doğrusal olarak yansıtılır.

### 3.2. Pratik Çıkarımlar
GQA'nın temel faydası, KV önbelleğinin boyutunu MHA'ya kıyasla önemli ölçüde azaltma yeteneğidir. KV önbelleği için gereken bellek, $N_q$ yerine $N_{kv}$ ile orantılı hale gelir. Uygun bir $N_{kv}$ seçerek (örn. $N_{kv} = N_q / 2$ veya $N_q / 4$), GQA, çıkarım sırasında bellek kullanımını ve bellek bant genişliği gereksinimlerini önemli ölçüde azaltabilir ve bu da daha hızlı kod çözmeye yol açar. $N_{kv}$'yi seçme esnekliği, geliştiricilerin verimlilik ve kalite arasındaki dengeyi ince ayar yapmasına olanak tanır.

## 4. GQA'nın Avantajları
GQA, modern LLM'ler için onu kritik bir optimizasyon haline getiren çeşitli çekici avantajlar sunar:

### 4.1. Gelişmiş Çıkarım Verimliliği
GQA'nın en doğrudan faydası, çıkarım hızı üzerindeki etkisidir. Anahtar ve Değer başlıklarının sayısını azaltarak, GQA, KV önbelleğinin boyutunu radikal bir şekilde küçültür. Bu azalma doğrudan şunlara dönüşür:
*   **Daha düşük bellek ayak izi:** Modeller daha az kullanılabilir VRAM'e sahip cihazlarda çalışabilir.
*   **Azaltılmış bellek bant genişliği:** Bellekten daha az veri çekilmesi gerekir, bu da özellikle uzun dizi uzunlukları için hesaplamayı hızlandırır.
*   **Daha hızlı kod çözme:** Her yeni jetonun KV önbelleğine erişmesini gerektiren otomatik gerilemeli üretim, önemli ölçüde hızlanır. Bu, özellikle etkileşimli uygulamalar için önemlidir.

### 4.2. Dengeli Performans
K/V projeksiyonlarının agresif bir şekilde paylaşılması nedeniyle bazen model kalitesinde belirgin bir düşüşe yol açabilen MQA'nın aksine, GQA daha dengeli bir yaklaşım sunar. Birden çok Anahtar/Değer grubuna izin vererek ($1 < N_{kv} < N_q$), GQA, MHA'nın çeşitli temsilleri öğrenme kapasitesinin çoğunu korur. Bu, GQA modellerinin genellikle MHA seviyesine yakın kaliteyi korurken önemli MQA seviyesinde verimlilik kazanımları elde edebileceği anlamına gelir. LLM'lerin pratik dağıtımı için etkili bir tatlı nokta sunarak bu boşluğu etkili bir şekilde kapatır.

### 4.3. Ölçeklenebilirlik ve Bağlam Uzunluğu
Jeton başına azaltılmış bellek kullanımı, GQA modellerinin daha uzun bağlam pencerelerini işlemesine de olanak tanır. LLM'ler, giderek artan bir şekilde kapsamlı bağlam gerektiren görevler için (örneğin, uzun belgelerin özetlenmesi, kod yardımı) kullanılmakta olduğundan, aşırı bellek yükü olmadan daha büyük KV önbelleklerini destekleme yeteneği hayati önem taşımaktadır. GQA, mevcut donanım altyapısında genişletilmiş bağlam yeteneklerine sahip daha büyük modellerin dağıtımını kolaylaştırır.

## 5. Kod Örneği
İşte MHA, MQA ve GQA için Anahtar/Değer başlık sayılarındaki farkı gösteren kavramsal, Python benzeri bir sözde kod parçacığı. Bu örnek, tam dikkat hesaplamasına değil, K/V başlıklarının *sayısına* odaklanmaktadır.

```python
class DikkatMekanizması:
    def __init__(self, sorgu_başlık_sayısı: int, anahtar_değer_başlık_sayısı: int):
        self.sorgu_başlık_sayısı = sorgu_başlık_sayısı
        self.anahtar_değer_başlık_sayısı = anahtar_değer_başlık_sayısı

        if sorgu_başlık_sayısı % anahtar_değer_başlık_sayısı != 0:
            raise ValueError("GQA için sorgu_başlık_sayısı, anahtar_değer_başlık_sayısı'na bölünebilir olmalıdır.")

        self.grup_sayısı = sorgu_başlık_sayısı // anahtar_değer_başlık_sayısı
        print(f"{sorgu_başlık_sayısı} sorgu başlığı ve {anahtar_değer_başlık_sayısı} KV başlığı ile dikkat mekanizması başlatılıyor.")
        print(f"Bu, her biri 1 KV başlığı paylaşan {self.grup_sayısı} grup olduğu anlamına gelir.")

# Örnek 1: Çoklu Başlıklı Dikkat (MHA)
# Her sorgu başlığının kendi K/V başlığı vardır.
mha = DikkatMekanizması(sorgu_başlık_sayısı=8, anahtar_değer_başlık_sayısı=8)
# Çıktı: 8 sorgu başlığı ve 8 KV başlığı ile dikkat mekanizması başlatılıyor.
#         Bu, her biri 1 KV başlığı paylaşan 1 grup olduğu anlamına gelir. (Her başlık kendi grubudur)

print("-" * 30)

# Örnek 2: Çoklu Sorgulu Dikkat (MQA)
# Tüm sorgu başlıkları tek bir K/V başlığını paylaşır.
mqa = DikkatMekanizması(sorgu_başlık_sayısı=8, anahtar_değer_başlık_sayısı=1)
# Çıktı: 8 sorgu başlığı ve 1 KV başlığı ile dikkat mekanizması başlatılıyor.
#         Bu, her biri 1 KV başlığı paylaşan 8 grup olduğu anlamına gelir.

print("-" * 30)

# Örnek 3: Gruplandırılmış Sorgu Dikkat (GQA)
# Sorgu başlıkları gruplara ayrılır, her grup bir K/V başlığını paylaşır.
gqa_örnek_1 = DikkatMekanizması(sorgu_başlık_sayısı=8, anahtar_değer_başlık_sayısı=2)
# Çıktı: 8 sorgu başlığı ve 2 KV başlığı ile dikkat mekanizması başlatılıyor.
#         Bu, her biri 1 KV başlığı paylaşan 4 grup olduğu anlamına gelir.

print("-" * 30)

gqa_örnek_2 = DikkatMekanizması(sorgu_başlık_sayısı=8, anahtar_değer_başlık_sayısı=4)
# Çıktı: 8 sorgu başlığı ve 4 KV başlığı ile dikkat mekanizması başlatılıyor.
#         Bu, her biri 1 KV başlığı paylaşan 2 grup olduğu anlamına gelir.

(Kod örneği bölümünün sonu)
```

## 6. Sonuç
Gruplandırılmış Sorgu Dikkat (GQA), büyük dil modelleri için transformer mimarisini optimize etmede önemli bir ilerlemeyi temsil etmektedir. Çoklu Başlıklı Dikkat'in model kalitesi ile Çoklu Sorgulu Dikkat'in çıkarım verimliliği arasında stratejik bir denge kurarak, GQA daha hızlı, daha bellek verimli otomatik gerilemeli kod çözme olanağı sağlar. Bu yenilik, giderek karmaşıklaşan LLM'lerin pratik dağıtımı için kritik öneme sahiptir ve daha erişilebilir donanımlarda daha büyük bağlam pencereleriyle etkili bir şekilde çalışmalarına olanak tanıyarak güçlü üretken yapay zeka yeteneklerine erişimi demokratikleştirmektedir. LLM'ler ölçek ve uygulama açısından büyümeye devam ettikçe, GQA gibi teknikler, yapay zeka alanında hesaplama açısından mümkün ve ekonomik olarak uygulanabilir olanın sınırlarını zorlamak için vazgeçilmez olmaya devam edecektir.
