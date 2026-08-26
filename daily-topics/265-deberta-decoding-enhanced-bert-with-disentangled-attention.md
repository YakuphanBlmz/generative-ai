# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: BERT and Self-Attention](#2-background-bert-and-self-attention)
    - [2.1. BERT's Architecture and Pre-training Tasks](#21-berts-architecture-and-pre-training-tasks)
    - [2.2. Limitations of BERT's Positional Embeddings](#22-limitations-of-berts-positional-embeddings)
- [3. DeBERTa's Core Innovations](#3-debertas-core-innovations)
    - [3.1. Disentangled Attention Mechanism](#31-disentangled-attention-mechanism)
    - [3.2. Enhanced Mask Decoder (EMD)](#32-enhanced-mask-decoder-emd)
    - [3.3. Absolute Positional Embeddings](#33-absolute-positional-embeddings)
- [4. Performance and Empirical Results](#4-performance-and-empirical-results)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction
The advent of **pre-trained language models** has revolutionized Natural Language Processing (NLP), with **BERT (Bidirectional Encoder Representations from Transformers)** standing as a foundational breakthrough. BERT effectively captures contextual representations through its **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)** objectives. However, subsequent research identified avenues for improvement, particularly concerning how contextual information, especially positional relationships, is processed within the **Transformer architecture**.

**DeBERTa (Decoding-enhanced BERT with Disentangled Attention)**, proposed by Microsoft in 2021, represents a significant advancement in this lineage. It introduces two novel mechanisms: **disentangled attention** and the **enhanced mask decoder (EMD)**. These innovations aim to more effectively encode the interactions between tokens and their positions, leading to superior performance across a wide range of NLP tasks. DeBERTa has consistently achieved state-of-the-art results on benchmarks such as GLUE, SuperGLUE, and SQuAD, demonstrating its capability to learn more robust and fine-grained language representations compared to its predecessors like BERT and RoBERTa. This document will delve into the architectural innovations that define DeBERTa and contribute to its exceptional performance.

### 2. Background: BERT and Self-Attention

To appreciate DeBERTa's contributions, it is essential to first understand the context provided by BERT and the fundamental **self-attention mechanism**.

#### 2.1. BERT's Architecture and Pre-training Tasks
**BERT** is a multi-layer **Transformer encoder** designed to learn deep bidirectional representations from unlabeled text. Its primary innovation lies in its pre-training strategy, which includes:
*   **Masked Language Modeling (MLM):** Randomly masking some tokens in the input and training the model to predict the original vocabulary id of the masked word based only on its context. This forces the model to learn bidirectional contextual relationships.
*   **Next Sentence Prediction (NSP):** Predicting whether two sentences are consecutive in the original document. This helps the model understand sentence relationships, which is crucial for downstream tasks like question answering and natural language inference.

The core of BERT, like all Transformer models, is the **self-attention mechanism**. This mechanism allows each token in a sequence to weigh the importance of all other tokens in the sequence when computing its own representation. For a given token, its representation is a weighted sum of all token embeddings, where weights are determined by the similarity between the query vector of the current token and the key vectors of all other tokens. This is formally expressed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Here, $Q$ (Query), $K$ (Key), and $V$ (Value) are linear transformations of the input token embeddings.

#### 2.2. Limitations of BERT's Positional Embeddings
In BERT, the input to the Transformer layers is a sum of three embeddings: **token embedding**, **segment embedding**, and **absolute positional embedding**. The absolute positional embedding provides information about the fixed position of each token in the sequence. While effective, this approach has a limitation in how it integrates positional information with token content. Specifically, BERT's self-attention computes attention scores by summing the content embedding and absolute positional embedding of each token before computing the query and key vectors. This fusion means that the attention score between two tokens inherently mixes their content and absolute positions, making it difficult for the model to separately learn the influence of content similarity and relative positional proximity. For instance, the attention score between two tokens might be high either because their content is very similar, or because they are close to each other, or both. This entanglement can hinder the model's ability to discern more nuanced contextual relationships.

### 3. DeBERTa's Core Innovations

DeBERTa introduces two principal innovations to address the limitations of prior Transformer models, particularly in how they handle contextual and positional information. These are the **disentangled attention mechanism** and the **enhanced mask decoder (EMD)**.

#### 3.1. Disentangled Attention Mechanism
The most significant architectural change in DeBERTa is its **disentangled attention mechanism**. Unlike BERT, which sums content and absolute positional embeddings before computing attention, DeBERTa treats content and relative position as separate pieces of information. This means that instead of a single query and key vector per token, DeBERTa generates two sets of representations for each token: one for its content and one for its relative position.

For each token $i$ at position $p_i$ and token $j$ at position $p_j$, the attention weight $A_{ij}$ from token $i$ to token $j$ is computed based on four components:
1.  **Content-to-content attention:** The similarity between the content query of token $i$ and the content key of token $j$. This captures how much token $i$ focuses on the content of token $j$.
2.  **Content-to-position attention:** The similarity between the content query of token $i$ and the relative position key of token $j$ (relative to $i$). This captures how much token $i$ focuses on the relative position of token $j$.
3.  **Position-to-content attention:** The similarity between the relative position query of token $i$ (relative to $j$) and the content key of token $j$. This captures how much the relative position of token $i$ influences its focus on the content of token $j$.
4.  **Position-to-position attention:** (Optional, and often omitted in practice for efficiency) The similarity between the relative position query of token $i$ and the relative position key of token $j$.

The attention score between token $i$ and token $j$ is the sum of these components. For example, a simplified formulation for content-to-content and content-to-position could be:

$$
A_{i,j} = \mathbf{q}_i^c \mathbf{k}_j^c + \mathbf{q}_i^c \mathbf{r}_{j|i}
$$

where $\mathbf{q}_i^c$ and $\mathbf{k}_j^c$ are content query and key vectors, and $\mathbf{r}_{j|i}$ is the embedding for the relative position of token $j$ with respect to token $i$. DeBERTa's full formulation is more complex, involving relative position embeddings for keys and queries. This disentangled approach allows the model to explicitly learn and differentiate the influence of content and relative position, leading to richer and more accurate contextual representations.

#### 3.2. Enhanced Mask Decoder (EMD)
BERT's Masked Language Modeling (MLM) objective aims to predict masked tokens. However, the standard MLM approach, where a prediction head directly operates on the final hidden states, might not fully leverage the absolute positional information that can be crucial for predicting specific tokens, especially for short sequences or when predicting semantically distinct words. For example, in the sentence "The capital of France is [MASK]", knowing that "[MASK]" is at a certain absolute position can narrow down the prediction choices significantly.

The **Enhanced Mask Decoder (EMD)** in DeBERTa addresses this by introducing absolute positional embeddings *after* the Transformer layers, specifically for the decoding phase. While relative positional embeddings are used within the Transformer layers via disentangled attention, EMD reintroduces absolute position information during the final prediction stage. This is achieved by adding the absolute positional embeddings to the final hidden states *before* the Softmax layer responsible for predicting the masked tokens. This mechanism ensures that the model can leverage both local (relative position through disentangled attention) and global (absolute position through EMD) positional information effectively for token prediction.

#### 3.3. Absolute Positional Embeddings
In DeBERTa, absolute positional embeddings are not directly added to the input embeddings like in BERT. Instead, they are *deferred* and integrated only during the **Enhanced Mask Decoder (EMD)** phase. This separation is strategic:
*   **Within Transformer Layers:** Disentangled attention relies solely on **relative positional embeddings**, allowing the model to focus on the proximity and ordering of tokens within a local context. This avoids confounding content and absolute position in early attention computations.
*   **During Decoding (EMD):** **Absolute positional embeddings** are added to the contextualized representations *after* all Transformer layers have processed the sequence. This provides the final prediction head with crucial global positional context, which can be particularly beneficial for disambiguation and precise token prediction, especially in tasks where the specific position of a word (e.g., first word of a sentence, last word of a paragraph) carries significant meaning.

This dual approach combines the strengths of relative positional information for contextual understanding within the Transformer blocks and absolute positional information for fine-grained prediction.

### 4. Performance and Empirical Results
DeBERTa has demonstrated remarkable performance improvements over previous state-of-the-art models, including BERT, RoBERTa, and ELECTRA. It achieved significant gains across a broad spectrum of NLP benchmarks:
*   **GLUE and SuperGLUE:** DeBERTa surpassed previous models on these challenging general language understanding benchmarks, often achieving scores that even outperformed human baselines on tasks like RTE (Recognizing Textual Entailment) and WSC (Winograd Schema Challenge).
*   **SQuAD (Stanford Question Answering Dataset):** It showed superior performance in question answering tasks, indicating its enhanced capability to understand complex relationships between questions and passages.
*   **Ranking on Leaderboards:** DeBERTa models, particularly larger variants like DeBERTa-v3-large, have frequently topped public leaderboards for various NLP tasks, solidifying its position as one of the most powerful pre-trained language models to date.

These consistent improvements validate the effectiveness of its core innovations: the **disentangled attention mechanism** provides a more nuanced understanding of content-position interactions, while the **enhanced mask decoder** ensures that absolute positional information is leveraged optimally during the critical decoding phase. The model's ability to decouple these aspects of positional information contributes directly to its superior understanding of context and language.

### 5. Code Example

To illustrate how to use a DeBERTa model, here's a short Python snippet using the Hugging Face Transformers library to load a pre-trained DeBERTa model and its tokenizer for sequence classification.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the pre-trained DeBERTa model name
model_name = "microsoft/deberta-v3-base"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example input text
text = "DeBERTa is a powerful language model with disentangled attention."

# Tokenize the input text
# `return_tensors="pt"` ensures PyTorch tensors are returned
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Pass the inputs to the model
with torch.no_grad(): # Disable gradient calculations for inference
    outputs = model(**inputs)

# The `outputs` object contains logits (raw predictions)
# For classification, we often apply softmax to get probabilities
probabilities = torch.softmax(outputs.logits, dim=1)

print(f"Model: {model_name}")
print(f"Input text: '{text}'")
print(f"Logits: {outputs.logits}")
print(f"Probabilities: {probabilities}")

# Note: For actual classification, you would train this model on a labeled dataset.
# The output here will be based on the pre-trained model's default "random" head.

(End of code example section)
```

### 6. Conclusion
**DeBERTa** marks a significant evolutionary step in the landscape of **pre-trained Transformer-based language models**. By meticulously refining how positional information is processed and integrated, it overcomes key limitations present in earlier models like BERT. The **disentangled attention mechanism**, which separates content and relative position in attention calculations, allows for a more granular and accurate understanding of token interactions. Coupled with the **Enhanced Mask Decoder (EMD)**, which reintroduces crucial absolute positional embeddings during the final prediction stage, DeBERTa achieves a superior synthesis of local and global contextual cues. Its consistent top performance across a diverse set of challenging NLP benchmarks underscores the efficacy of these innovations, positioning DeBERTa as a robust and highly capable model for a wide array of natural language understanding tasks. The principles introduced by DeBERTa continue to influence the development of more sophisticated and efficient language models.
---
<br>

<a name="türkçe-içerik"></a>
## DeBERTa: Ayrıştırılmış Dikkat Mekanizması ile Geliştirilmiş Çözümleme Yeteneğine Sahip BERT

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: BERT ve Öz-Dikkat](#2-arka-plan-bert-ve-öz-dikkat)
    - [2.1. BERT'in Mimarisi ve Ön Eğitim Görevleri](#21-bertin-mimarisi-ve-ön-eğitim-görevleri)
    - [2.2. BERT'in Konumsal Gömülülerin Sınırlamaları](#22-bertin-konumsal-gömülülerin-sınırlamaları)
- [3. DeBERTa'nın Temel Yenilikleri](#3-debertanın-temel-yenilikleri)
    - [3.1. Ayrıştırılmış Dikkat Mekanizması](#31-ayrıştırılmış-dikkat-mekanizması)
    - [3.2. Geliştirilmiş Maske Çözücü (EMD)](#32-geliştirilmiş-maske-çözücü-emd)
    - [3.3. Mutlak Konumsal Gömülüler](#33-mutlak-konumsal-gömülüler)
- [4. Performans ve Deneysel Sonuçlar](#4-performans-ve-deneysel-sonuçlar)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

### 1. Giriş
**Önceden eğitilmiş dil modellerinin** ortaya çıkışı, Doğal Dil İşleme (NLP) alanında devrim yaratmıştır ve **BERT (Bidirectional Encoder Representations from Transformers)** bu alanda temel bir atılım olarak kabul edilmektedir. BERT, **Maskelenmiş Dil Modellemesi (MLM)** ve **Sonraki Cümle Tahmini (NSP)** hedefleri aracılığıyla bağlamsal temsilleri etkili bir şekilde yakalamıştır. Ancak, sonraki araştırmalar, özellikle konumsal ilişkilerin **Transformer mimarisi** içinde nasıl işlendiği konusunda iyileştirme yolları olduğunu ortaya koymuştur.

Microsoft tarafından 2021 yılında önerilen **DeBERTa (Decoding-enhanced BERT with Disentangled Attention)**, bu soyda önemli bir ilerlemeyi temsil etmektedir. İki yeni mekanizma sunar: **ayrıştırılmış dikkat (disentangled attention)** ve **geliştirilmiş maske çözücü (enhanced mask decoder - EMD)**. Bu yenilikler, jetonlar (tokenler) ve onların konumları arasındaki etkileşimleri daha etkili bir şekilde kodlamayı amaçlayarak, geniş bir NLP görev yelpazesinde üstün performans sağlar. DeBERTa, GLUE, SuperGLUE ve SQuAD gibi karşılaştırmalı testlerde sürekli olarak son teknoloji sonuçlar elde ederek, BERT ve RoBERTa gibi öncüllerine kıyasla daha sağlam ve ayrıntılı dil temsilleri öğrenme yeteneğini göstermiştir. Bu belge, DeBERTa'yı tanımlayan ve olağanüstü performansına katkıda bulunan mimari yenilikleri inceleyecektir.

### 2. Arka Plan: BERT ve Öz-Dikkat

DeBERTa'nın katkılarını takdir etmek için, BERT ve temel **öz-dikkat mekanizması** tarafından sağlanan bağlamı anlamak önemlidir.

#### 2.1. BERT'in Mimarisi ve Ön Eğitim Görevleri
**BERT**, etiketlenmemiş metinden derin çift yönlü temsiller öğrenmek için tasarlanmış çok katmanlı bir **Transformer kodlayıcısıdır**. Temel yeniliği, şunları içeren ön eğitim stratejisinde yatmaktadır:
*   **Maskelenmiş Dil Modellemesi (MLM):** Girişteki bazı jetonları rastgele maskeleyerek, modelin maskelenmiş kelimenin orijinal kelime dağarcığı kimliğini yalnızca bağlamına göre tahmin etmesi için eğitilmesi. Bu, modeli çift yönlü bağlamsal ilişkileri öğrenmeye zorlar.
*   **Sonraki Cümle Tahmini (NSP):** İki cümlenin orijinal belgede ardışık olup olmadığını tahmin etme. Bu, modelin soru yanıtlama ve doğal dil çıkarımı gibi sonraki görevler için kritik olan cümle ilişkilerini anlamasına yardımcı olur.

BERT'in kalbi, tüm Transformer modelleri gibi, **öz-dikkat mekanizmasıdır**. Bu mekanizma, bir dizideki her jetonun kendi temsilini hesaplarken dizideki diğer tüm jetonların önemini tartmasına olanak tanır. Belirli bir jeton için, temsili, mevcut jetonun sorgu vektörü ile diğer tüm jetonların anahtar vektörleri arasındaki benzerliğe göre belirlenen ağırlıkların, tüm jeton gömülülerin ağırlıklı toplamıdır. Bu, resmi olarak şu şekilde ifade edilir:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Burada, $Q$ (Sorgu), $K$ (Anahtar) ve $V$ (Değer) girdi jeton gömülülerinin doğrusal dönüşümleridir.

#### 2.2. BERT'in Konumsal Gömülülerin Sınırlamaları
BERT'te Transformer katmanlarına girdi, üç gömülü (embedding) toplamıdır: **jeton gömülü**, **segment gömülü** ve **mutlak konumsal gömülü**. Mutlak konumsal gömülü, her jetonun dizideki sabit konumu hakkında bilgi sağlar. Etkili olmakla birlikte, bu yaklaşımın konumsal bilgiyi jeton içeriğiyle bütünleştirme biçiminde bir sınırlaması vardır. Özellikle, BERT'in öz-dikkat mekanizması, sorgu ve anahtar vektörlerini hesaplamadan önce her jetonun içerik gömülüsünü ve mutlak konumsal gömülüsünü toplayarak dikkat skorlarını hesaplar. Bu birleşim, iki jeton arasındaki dikkat skorunun doğal olarak içeriği ve mutlak konumlarını karıştırması anlamına gelir; bu da modelin içerik benzerliği ve göreceli konumsal yakınlığın etkisini ayrı ayrı öğrenmesini zorlaştırır. Örneğin, iki jeton arasındaki dikkat skoru, içerikleri çok benzer olduğu için veya birbirlerine yakın oldukları için veya her ikisi birden yüksek olabilir. Bu karışıklık, modelin daha incelikli bağlamsal ilişkileri ayırt etme yeteneğini engelleyebilir.

### 3. DeBERTa'nın Temel Yenilikleri

DeBERTa, önceki Transformer modellerinin sınırlamalarını, özellikle bağlamsal ve konumsal bilgileri işleme biçimlerini ele almak için iki temel yenilik sunar. Bunlar, **ayrıştırılmış dikkat mekanizması** ve **geliştirilmiş maske çözücü (EMD)**'dir.

#### 3.1. Ayrıştırılmış Dikkat Mekanizması
DeBERTa'daki en önemli mimari değişiklik, **ayrıştırılmış dikkat mekanizmasıdır**. Dikkat hesaplamadan önce içerik ve mutlak konumsal gömülüleri toplayan BERT'in aksine, DeBERTa içerik ve göreceli konumu ayrı bilgi parçaları olarak ele alır. Bu, her jeton için tek bir sorgu ve anahtar vektör yerine, DeBERTa her jeton için iki takım temsil oluşturur: biri içeriği için, diğeri ise göreceli konumu için.

$p_i$ konumundaki $i$ jetonu ile $p_j$ konumundaki $j$ jetonu için, $i$ jetonundan $j$ jetonuna olan dikkat ağırlığı $A_{ij}$, dört bileşene dayanarak hesaplanır:
1.  **İçerikten içeriğe dikkat:** $i$ jetonunun içerik sorgusu ile $j$ jetonunun içerik anahtarı arasındaki benzerlik. Bu, $i$ jetonunun $j$ jetonunun içeriğine ne kadar odaklandığını yakalar.
2.  **İçerikten konuma dikkat:** $i$ jetonunun içerik sorgusu ile $j$ jetonunun göreceli konum anahtarı (i'ye göre) arasındaki benzerlik. Bu, $i$ jetonunun $j$ jetonunun göreceli konumuna ne kadar odaklandığını yakalar.
3.  **Konumdan içeriğe dikkat:** $i$ jetonunun göreceli konum sorgusu ($j$'ye göre) ile $j$ jetonunun içerik anahtarı arasındaki benzerlik. Bu, $i$ jetonunun göreceli konumunun, $j$ jetonunun içeriğine odaklanmasını ne kadar etkilediğini yakalar.
4.  **Konumdan konuma dikkat:** (İsteğe bağlı ve verimlilik için pratikte genellikle atlanır) $i$ jetonunun göreceli konum sorgusu ile $j$ jetonunun göreceli konum anahtarı arasındaki benzerlik.

$i$ jetonu ile $j$ jetonu arasındaki dikkat skoru, bu bileşenlerin toplamıdır. Örneğin, içerikten içeriğe ve içerikten konuma yönelik basitleştirilmiş bir formülasyon şöyle olabilir:

$$
A_{i,j} = \mathbf{q}_i^c \mathbf{k}_j^c + \mathbf{q}_i^c \mathbf{r}_{j|i}
$$

Burada $\mathbf{q}_i^c$ ve $\mathbf{k}_j^c$ içerik sorgu ve anahtar vektörleridir ve $\mathbf{r}_{j|i}$ ise $j$ jetonunun $i$ jetonuna göreceli konumuna ilişkin gömülüdür. DeBERTa'nın tam formülasyonu, anahtarlar ve sorgular için göreceli konum gömülülerini içeren daha karmaşıktır. Bu ayrıştırılmış yaklaşım, modelin içerik ve göreceli konumun etkisini açıkça öğrenmesine ve farklılaştırmasına olanak tanıyarak daha zengin ve daha doğru bağlamsal temsiller elde edilmesini sağlar.

#### 3.2. Geliştirilmiş Maske Çözücü (EMD)
BERT'in Maskelenmiş Dil Modellemesi (MLM) hedefi, maskelenmiş jetonları tahmin etmeyi amaçlar. Ancak, standart MLM yaklaşımı, son gizli durumlar üzerinde doğrudan bir tahmin kafasının çalıştığı durumlarda, özellikle kısa dizilerde veya anlamsal olarak farklı kelimeleri tahmin ederken kritik olabilecek mutlak konumsal bilgiyi tam olarak kullanamayabilir. Örneğin, "Fransa'nın başkenti [MASK]'dir" cümlesinde, "[MASK]"'in belirli bir mutlak konumda olduğunu bilmek, tahmin seçeneklerini önemli ölçüde daraltabilir.

DeBERTa'daki **Geliştirilmiş Maske Çözücü (EMD)**, Transformer katmanlarından *sonra*, özellikle çözme aşaması için mutlak konumsal gömülüleri tanıtarak bu sorunu ele alır. Transformer katmanları içinde ayrıştırılmış dikkat yoluyla göreceli konumsal gömülüler kullanılırken, EMD son tahmin aşamasında mutlak konum bilgisini yeniden devreye sokar. Bu, maskelenmiş jetonları tahmin etmekten sorumlu Softmax katmanından *önce* son gizli durumlara mutlak konumsal gömülülerin eklenmesiyle başarılır. Bu mekanizma, modelin jeton tahmini için hem yerel (ayrıştırılmış dikkat yoluyla göreceli konum) hem de küresel (EMD yoluyla mutlak konum) konumsal bilgiyi etkili bir şekilde kullanabilmesini sağlar.

#### 3.3. Mutlak Konumsal Gömülüler
DeBERTa'da, mutlak konumsal gömülüler BERT'teki gibi doğrudan girdi gömülülerine eklenmez. Bunun yerine, **Geliştirilmiş Maske Çözücü (EMD)** aşamasına kadar *ertelenir* ve yalnızca bu aşamada entegre edilir. Bu ayrım stratejiktir:
*   **Transformer Katmanları İçinde:** Ayrıştırılmış dikkat, yalnızca **göreceli konumsal gömülülere** dayanır ve modelin yerel bir bağlam içinde jetonların yakınlığına ve sıralamasına odaklanmasına olanak tanır. Bu, erken dikkat hesaplamalarında içerik ve mutlak konumun karışmasını önler.
*   **Çözümleme Sırasında (EMD):** **Mutlak konumsal gömülüler**, tüm Transformer katmanları diziyi işledikten *sonra* bağlamsallaştırılmış gösterimlere eklenir. Bu, son tahmin başlığına kritik küresel konumsal bağlam sağlar; bu, özellikle bir kelimenin belirli konumunun (örneğin, bir cümlenin ilk kelimesi, bir paragrafın son kelimesi) önemli bir anlam taşıdığı görevlerde, belirsizliği giderme ve kesin jeton tahmini için özellikle faydalı olabilir.

Bu ikili yaklaşım, Transformer blokları içindeki bağlamsal anlama için göreceli konumsal bilginin güçlü yönlerini ve ayrıntılı tahmin için mutlak konumsal bilginin güçlü yönlerini birleştirir.

### 4. Performans ve Deneysel Sonuçlar
DeBERTa, BERT, RoBERTa ve ELECTRA dahil olmak üzere önceki en son teknoloji modeller üzerinde dikkat çekici performans iyileştirmeleri göstermiştir. Geniş bir NLP karşılaştırmalı test yelpazesinde önemli kazanımlar elde etmiştir:
*   **GLUE ve SuperGLUE:** DeBERTa, bu zorlu genel dil anlama karşılaştırmalı testlerinde önceki modelleri geride bırakmış, genellikle RTE (Metinsel Çıkarımı Tanıma) ve WSC (Winograd Şema Zorluğu) gibi görevlerde insan taban çizgilerini bile aşan skorlar elde etmiştir.
*   **SQuAD (Stanford Soru Cevaplama Veri Kümesi):** Soru yanıtlama görevlerinde üstün performans göstererek, sorular ve pasajlar arasındaki karmaşık ilişkileri anlama yeteneğinin arttığını kanıtlamıştır.
*   **Lider Tablolarındaki Sıralama:** DeBERTa modelleri, özellikle DeBERTa-v3-large gibi daha büyük varyantlar, çeşitli NLP görevleri için kamuya açık lider tablolarında sıkça zirveye çıkarak, bugüne kadarki en güçlü önceden eğitilmiş dil modellerinden biri olarak konumunu sağlamlaştırmıştır.

Bu tutarlı iyileştirmeler, temel yeniliklerinin etkinliğini doğrulamaktadır: **ayrıştırılmış dikkat mekanizması** içerik-konum etkileşimlerinin daha incelikli bir şekilde anlaşılmasını sağlarken, **geliştirilmiş maske çözücü** ise kritik çözme aşamasında mutlak konumsal bilginin en iyi şekilde kullanılmasını sağlar. Modelin konumsal bilginin bu yönlerini ayırma yeteneği, bağlamı ve dili üstün bir şekilde anlamasına doğrudan katkıda bulunur.

### 5. Kod Örneği

Bir DeBERTa modelinin nasıl kullanılacağını göstermek için, Hugging Face Transformers kütüphanesini kullanarak önceden eğitilmiş bir DeBERTa modelini ve jetonlayıcısını dizi sınıflandırması için yükleyen kısa bir Python kodu aşağıdadır.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Önceden eğitilmiş DeBERTa model adını tanımla
model_name = "microsoft/deberta-v3-base"

# Jetonlayıcıyı (tokenizer) yükle
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dizi sınıflandırması için modeli yükle
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Örnek girdi metni
text = "DeBERTa, ayrıştırılmış dikkat mekanizması ile güçlü bir dil modelidir."

# Girdi metnini jetonlarına ayır (tokenize et)
# `return_tensors="pt"` PyTorch tensörlerinin döndürülmesini sağlar
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Girdileri modele ilet
with torch.no_grad(): # Çıkarım için gradyan hesaplamalarını devre dışı bırak
    outputs = model(**inputs)

# `outputs` nesnesi logitleri (ham tahminler) içerir
# Sınıflandırma için, olasılıkları elde etmek üzere genellikle softmax uygulanır
probabilities = torch.softmax(outputs.logits, dim=1)

print(f"Model: {model_name}")
print(f"Girdi metni: '{text}'")
print(f"Logitler: {outputs.logits}")
print(f"Olasılıklar: {probabilities}")

# Not: Gerçek sınıflandırma için, bu modeli etiketli bir veri kümesi üzerinde eğitmeniz gerekir.
# Buradaki çıktı, önceden eğitilmiş modelin varsayılan "rastgele" başlığına dayalı olacaktır.

(Kod örneği bölümünün sonu)
```

### 6. Sonuç
**DeBERTa**, **önceden eğitilmiş Transformer tabanlı dil modelleri** ortamında önemli bir evrimsel adımı işaret etmektedir. Konumsal bilginin işlenme ve entegre edilme biçimini titizlikle iyileştirerek, BERT gibi önceki modellerde bulunan temel sınırlamaların üstesinden gelmiştir. Dikkat hesaplamalarında içerik ve göreceli konumu ayıran **ayrıştırılmış dikkat mekanizması**, jeton etkileşimlerinin daha ayrıntılı ve doğru bir şekilde anlaşılmasını sağlar. Kritik son tahmin aşamasında önemli mutlak konumsal gömülüleri yeniden tanıtan **Geliştirilmiş Maske Çözücü (EMD)** ile birleştiğinde, DeBERTa yerel ve küresel bağlamsal ipuçlarının üstün bir sentezini elde eder. Çeşitli zorlu NLP karşılaştırmalı testlerinde gösterdiği tutarlı üstün performans, bu yeniliklerin etkinliğini vurgulayarak, DeBERTa'yı çok çeşitli doğal dil anlama görevleri için sağlam ve yüksek yetenekli bir model olarak konumlandırmaktadır. DeBERTa tarafından tanıtılan ilkeler, daha sofistike ve verimli dil modellerinin geliştirilmesini etkilemeye devam etmektedir.
