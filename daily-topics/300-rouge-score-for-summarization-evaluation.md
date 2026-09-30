# ROUGE Score for Summarization Evaluation

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and ROUGE Variants](#2-core-concepts-and-rouge-variants)
  - [2.1. ROUGE-N (N-gram Overlap)](#21-rouge-n-n-gram-overlap)
  - [2.2. ROUGE-L (Longest Common Subsequence)](#22-rouge-l-longest-common-subsequence)
  - [2.3. ROUGE-W (Weighted Longest Common Subsequence)](#23-rouge-w-weighted-longest-common-subsequence)
  - [2.4. ROUGE-S (Skip-Bigram Overlap)](#24-rouge-s-skip-bigram-overlap)
- [3. Calculation Methodology](#3-calculation-methodology)
- [4. Advantages and Disadvantages of ROUGE](#4-advantages-and-disadvantages-of-rouge)
  - [4.1. Advantages](#41-advantages)
  - [4.2. Disadvantages](#42-disadvantages)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction
The evaluation of automatically generated text summaries is a critical, yet challenging, task in Natural Language Processing (NLP). Unlike other generative tasks, summarization requires assessing both the fidelity of information extracted from source text and the fluency and coherence of the condensed output. Manual evaluation by human judges is often considered the gold standard but is prohibitively expensive, time-consuming, and can suffer from subjectivity issues at scale. Consequently, automatic evaluation metrics have become indispensable for rapid prototyping, iteration, and benchmarking of summarization models. Among these, **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** stands out as the most widely adopted and influential metric suite for assessing summary quality. Developed by Chin-Yew Lin in 2004, ROUGE quantifies the similarity between an automatically produced summary (the "system summary") and one or more human-written summaries (the "reference summaries") by counting overlapping units such as n-grams, word sequences, or skip-bigrams. Its prevalence stems from its relative simplicity, interpretability, and demonstrated correlation with human judgments in many summarization tasks. This document provides a comprehensive overview of the ROUGE score, detailing its core concepts, variants, calculation methodology, practical implications, and limitations.

### 2. Core Concepts and ROUGE Variants
ROUGE operates on the principle of comparing the units present in a system summary against those in reference summaries. The fundamental idea is that a good summary will share many linguistic units with human-authored summaries of the same source text. ROUGE measures **recall**, which indicates how much of the reference summary is covered by the system summary. While typically reported as F1-scores (a harmonic mean of precision and recall), the "Recall-Oriented" aspect emphasizes its primary design goal.

ROUGE is not a single metric but a suite of metrics, each designed to capture different aspects of overlap:

#### 2.1. ROUGE-N (N-gram Overlap)
ROUGE-N measures the overlap of n-grams between the system summary and the reference summaries. An **n-gram** is a contiguous sequence of *n* items (words) from a given sample of text.
-   **ROUGE-1**: Measures the overlap of **unigrams** (single words). It captures the presence of individual salient words.
-   **ROUGE-2**: Measures the overlap of **bigrams** (sequences of two words). It captures the fluency of two-word phrases and some basic syntactic structures.
-   **ROUGE-3**, **ROUGE-4**, etc.: Measure the overlap of trigrams, four-grams, and so on. Higher N values capture longer common sequences, indicating better sentence-level structure and informativeness.

The formula for ROUGE-N is:
$$ \text{ROUGE-N Recall} = \frac{\sum_{S \in \{\text{Reference Summaries}\}} \sum_{\text{n-gram} \in S} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{S \in \{\text{Reference Summaries}\}} \sum_{\text{n-gram} \in S} \text{Count}(\text{n-gram})} $$
Where $\text{Count}_{\text{match}}(\text{n-gram})$ is the maximum number of n-grams co-occurring in the system summary and a reference summary, and $\text{Count}(\text{n-gram})$ is the number of n-grams in the reference summary.

#### 2.2. ROUGE-L (Longest Common Subsequence)
ROUGE-L is based on the **Longest Common Subsequence (LCS)** between the system summary and the reference summaries. An LCS is a sequence that appears in both summaries in the same relative order, but not necessarily contiguously. This metric rewards summaries that preserve sentence-level structure and word order, even if some words are skipped. It is considered more flexible than ROUGE-N because it does not require contiguous matches. ROUGE-L is particularly useful for evaluating summaries where rephrasing or reordering of clauses might occur, as long as the core information sequence is maintained.

The recall for ROUGE-L is calculated as:
$$ \text{ROUGE-L Recall} = \frac{\text{Length}(\text{LCS}(\text{System Summary, Reference Summary}))}{\text{Length}(\text{Reference Summary})} $$

#### 2.3. ROUGE-W (Weighted Longest Common Subsequence)
ROUGE-W is an extension of ROUGE-L that addresses a limitation where ROUGE-L gives equal weight to all common subsequences regardless of their length. ROUGE-W assigns higher weights to **longer common subsequences** more than to several shorter ones. This makes it more sensitive to the contiguity of matches, giving more credit to system summaries that generate longer, unbroken segments of text found in the references. The weighting function typically involves squaring the length of common subsequences.

#### 2.4. ROUGE-S (Skip-Bigram Overlap)
ROUGE-S measures the overlap of **skip-bigrams**. A skip-bigram is any pair of words in sentence order, allowing for arbitrary gaps (skips) between them. For instance, in the sentence "the cat sat on the mat", "the...mat" is a skip-bigram. ROUGE-S is a more lenient metric than ROUGE-N for higher N values, as it does not require contiguity. It is useful for capturing lexical and semantic similarity even when there are minor structural differences or interpolations of words. ROUGE-S often comes with a maximum skip distance constraint (e.g., ROUGE-S* with a skip limit).

### 3. Calculation Methodology
For each ROUGE variant, the score is typically calculated as a combination of **precision**, **recall**, and **F1-score**.
Let:
-   $X$ be the system summary.
-   $Y$ be a reference summary.
-   `Overlap(X, Y)` be the count of overlapping units (n-grams, LCS length, skip-bigrams) between $X$ and $Y$.
-   `Length(X)` be the total number of units in $X$.
-   `Length(Y)` be the total number of units in $Y$.

The scores are typically computed as follows:
-   **Precision (P)**: How much of the system summary is also present in the reference summary?
    $$ P = \frac{\text{Overlap}(X, Y)}{\text{Length}(X)} $$
-   **Recall (R)**: How much of the reference summary is also present in the system summary?
    $$ R = \frac{\text{Overlap}(X, Y)}{\text{Length}(Y)} $$
-   **F1-Score**: The harmonic mean of precision and recall. This is often the most reported score as it balances both aspects.
    $$ \text{F1} = \frac{2 \times P \times R}{P + R} $$

When multiple reference summaries are available (which is highly recommended for robustness), the ROUGE score is typically computed by taking the maximum recall (or F1) achieved against any single reference summary, or by averaging the scores across all references. Standard implementations usually calculate recall for all system-reference pairs and then combine these for a final score.

Preprocessing steps, such as lowercasing, stemming, removing stop words, and tokenization, can significantly affect ROUGE scores. The standard practice is to lowercase and remove punctuation before n-gram counting. Stemming and stop word removal are less common in modern ROUGE implementations as they can sometimes obscure important semantic distinctions or phrase structures.

### 4. Advantages and Disadvantages of ROUGE

#### 4.1. Advantages
1.  **Automatic and Efficient**: ROUGE allows for rapid and automatic evaluation of summarization systems, enabling quick iteration and experimentation without human intervention.
2.  **Widely Adopted**: It is the de-facto standard metric in summarization research, making it easy to compare results across different studies and benchmarks.
3.  **Correlation with Human Judgments**: Numerous studies have shown that ROUGE scores, particularly ROUGE-1 and ROUGE-2 F1, and ROUGE-L F1, correlate reasonably well with human judgments of summary quality and content overlap.
4.  **Interpretability**: The core idea of n-gram or LCS overlap is intuitively understandable, making the scores relatively easy to interpret.
5.  **Granular Analysis**: Different ROUGE variants (ROUGE-N, ROUGE-L, ROUGE-S) provide different perspectives on summary quality, from word choice to sentence structure and overall information flow.

#### 4.2. Disadvantages
1.  **Lack of Semantic Understanding**: ROUGE is a purely lexical overlap metric. It cannot understand synonyms, paraphrases, or semantically similar sentences that use different words. For example, "large car" and "big automobile" might convey the same meaning but receive a low ROUGE score if only one is in the reference.
2.  **Reference Dependency**: ROUGE scores are highly dependent on the quality and diversity of reference summaries. If the reference summaries are poor or too few, the ROUGE score may not accurately reflect the system's true quality. Creating multiple high-quality reference summaries is costly.
3.  **Sensitivity to Paraphrasing**: A system summary that effectively paraphrases information from the source text but does not use the exact phrasing of the reference will be penalized, even if it is a high-quality summary.
4.  **Limited Scope**: ROUGE primarily assesses content overlap and recall. It does not directly evaluate other crucial aspects of summary quality, such as:
    *   **Fluency**: Grammatical correctness and naturalness of language.
    *   **Coherence**: Logical flow and organizational structure of ideas.
    *   **Factuality/Faithfulness**: Whether the summary contains information that is factually consistent with the source document. A summary might score high on ROUGE but contain hallucinations if it matches reference words that are also hallucinations.
    *   **Novelty/Information Coverage**: Whether the summary covers the most important information adequately without being redundant.
5.  **Difficulty with Abstractive Summarization**: While useful for extractive summaries, ROUGE faces greater challenges with abstractive summaries, which often rephrase and generate new sentences, reducing direct lexical overlap with references. Newer metrics like BERTScore or faithfulness metrics are often combined with ROUGE for abstractive summarization evaluation.

### 5. Code Example
Here's a simple Python code snippet demonstrating how to calculate ROUGE scores using the `rouge-score` library, which is a common and robust implementation.

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference_summary: str, system_summary: str):
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    between a system summary and a reference summary.

    Args:
        reference_summary (str): The human-written reference summary.
        system_summary (str): The machine-generated system summary.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, system_summary)

    # Extracting F1 scores for each ROUGE type
    rouge1_f1 = scores['rouge1'].fmeasure
    rouge2_f1 = scores['rouge2'].fmeasure
    rougeL_f1 = scores['rougeL'].fmeasure

    return {
        "rouge-1_f1": rouge1_f1,
        "rouge-2_f1": rouge2_f1,
        "rouge-L_f1": rougeL_f1,
    }

# Example usage:
reference = "The quick brown fox jumps over the lazy dog."
system = "A quick brown fox jumps over a sleepy dog."

rouge_results = calculate_rouge(reference, system)
print(f"ROUGE Scores: {rouge_results}")

# Another example with less overlap
reference_2 = "Artificial intelligence is transforming many industries."
system_2 = "AI is changing several sectors."

rouge_results_2 = calculate_rouge(reference_2, system_2)
print(f"ROUGE Scores (less overlap): {rouge_results_2}")

(End of code example section)
```

### 6. Conclusion
ROUGE remains an indispensable tool for the automatic evaluation of summarization systems. Its various metrics provide a robust framework for quantifying lexical and structural overlap between system-generated and human-written summaries, offering valuable insights into content coverage and basic linguistic patterns. While widely adopted due to its efficiency and correlation with human judgments, it is crucial to acknowledge ROUGE's inherent limitations, particularly its inability to capture semantic nuances, logical coherence, and factual consistency. Researchers and practitioners often complement ROUGE with other metrics, such as **BERTScore** for semantic similarity or human evaluation for holistic quality, especially in the context of highly abstractive models. Understanding ROUGE's strengths and weaknesses is key to its effective application and to driving further advancements in the field of automatic text summarization.

---
<br>

<a name="türkçe-içerik"></a>
## ROUGE Skoru ile Özetleme Değerlendirmesi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. ROUGE Temel Kavramları ve Varyantları](#2-rouge-temel-kavramları-ve-varyantları)
  - [2.1. ROUGE-N (N-gram Çakışması)](#21-rouge-n-n-gram-çakışması)
  - [2.2. ROUGE-L (En Uzun Ortak Alt Dizi)](#22-rouge-l-en-uzun-ortak-alt-dizi)
  - [2.3. ROUGE-W (Ağırlıklı En Uzun Ortak Alt Dizi)](#23-rouge-w-ağırlıklı-en-uzun-ortak-alt-dizi)
  - [2.4. ROUGE-S (Atlamalı İkili N-gram Çakışması)](#24-rouge-s-atlamalı-ikili-n-gram-çakışması)
- [3. Hesaplama Metodolojisi](#3-hesaplama-metodolojisi)
- [4. ROUGE'un Avantajları ve Dezavantajları](#4-rouge'un-avantajları-ve-dezavantajları)
  - [4.1. Avantajları](#41-avantajları)
  - [4.2. Dezavantajları](#42-dezavantajları)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

### 1. Giriş
Otomatik olarak oluşturulan metin özetlerinin değerlendirilmesi, Doğal Dil İşleme (NLP) alanında kritik ancak zorlu bir görevdir. Diğer üretken görevlerin aksine, özetleme, hem kaynak metinden çıkarılan bilginin doğruluğunu hem de yoğunlaştırılmış çıktının akıcılığını ve tutarlılığını değerlendirmeyi gerektirir. İnsan hakemler tarafından yapılan manuel değerlendirme genellikle altın standart olarak kabul edilse de, çok pahalı, zaman alıcıdır ve ölçekte öznellik sorunlarına yol açabilir. Sonuç olarak, otomatik değerlendirme metrikleri, özetleme modellerinin hızlı prototiplemesi, yinelemesi ve karşılaştırması için vazgeçilmez hale gelmiştir. Bunlar arasında, **ROUGE (Recall-Oriented Understudy for Gisting Evaluation - Özetleme Değerlendirmesi için Geri Çağırma Odaklı Yardımcı İnceleme)**, özet kalitesini değerlendirmek için en yaygın kabul gören ve etkili metrik paketi olarak öne çıkmaktadır. Chin-Yew Lin tarafından 2004 yılında geliştirilen ROUGE, otomatik olarak üretilmiş bir özet (sistem özeti) ile bir veya daha fazla insan tarafından yazılmış özet (referans özetleri) arasındaki benzerliği, n-gramlar, kelime dizileri veya atlamalı ikili n-gramlar gibi çakışan birimleri sayarak ölçer. Yaygınlığı, nispeten basitliğinden, yorumlanabilirliğinden ve birçok özetleme görevinde insan yargılarıyla gösterdiği korelasyondan kaynaklanmaktadır. Bu belge, ROUGE skoruna ilişkin kapsamlı bir genel bakış sunmakta; temel kavramlarını, varyantlarını, hesaplama metodolojisini, pratik çıkarımlarını ve sınırlamalarını detaylandırmaktadır.

### 2. ROUGE Temel Kavramları ve Varyantları
ROUGE, bir sistem özetinde bulunan birimleri referans özetlerindeki birimlerle karşılaştırma ilkesine göre çalışır. Temel fikir, iyi bir özetin, aynı kaynak metnin insan tarafından yazılmış özetleriyle birçok dilsel birimi paylaşmasıdır. ROUGE, referans özetinin ne kadarının sistem özeti tarafından kapsandığını gösteren **geri çağırmayı (recall)** ölçer. Genellikle F1-skorları (hassasiyet ve geri çağırmanın harmonik ortalaması) olarak rapor edilse de, "Geri Çağırma Odaklı" yönü, birincil tasarım hedefini vurgular.

ROUGE tek bir metrik değil, her biri çakışmanın farklı yönlerini yakalamak için tasarlanmış bir metrik paketidir:

#### 2.1. ROUGE-N (N-gram Çakışması)
ROUGE-N, sistem özeti ile referans özetleri arasındaki n-gramların çakışmasını ölçer. Bir **n-gram**, verilen bir metin örneğinden *n* adet öğenin (kelimenin) bitişik bir dizisidir.
-   **ROUGE-1**: **Tekli n-gramların (unigramlar)** (tek kelimeler) çakışmasını ölçer. Bireysel önemli kelimelerin varlığını yakalar.
-   **ROUGE-2**: **İkili n-gramların (bigramlar)** (iki kelimelik diziler) çakışmasını ölçer. İki kelimelik ifadelerin akıcılığını ve bazı temel sözdizimsel yapıları yakalar.
-   **ROUGE-3**, **ROUGE-4**, vb.: Üçlü n-gramların, dörtlü n-gramların vb. çakışmasını ölçer. Daha yüksek N değerleri, daha uzun ortak dizileri yakalayarak daha iyi cümle seviyesi yapı ve bilgilendiricilik anlamına gelir.

ROUGE-N formülü şöyledir:
$$ \text{ROUGE-N Geri Çağırma} = \frac{\sum_{S \in \{\text{Referans Özetleri}\}} \sum_{\text{n-gram} \in S} \text{Eşleşme Sayısı}(\text{n-gram})}{\sum_{S \in \{\text{Referans Özetleri}\}} \sum_{\text{n-gram} \in S} \text{Sayı}(\text{n-gram})} $$
Burada $\text{Eşleşme Sayısı}(\text{n-gram})$, sistem özeti ve bir referans özeti arasında birlikte en fazla bulunan n-gram sayısıdır ve $\text{Sayı}(\text{n-gram})$, referans özetindeki n-gram sayısıdır.

#### 2.2. ROUGE-L (En Uzun Ortak Alt Dizi)
ROUGE-L, sistem özeti ile referans özetleri arasındaki **En Uzun Ortak Alt Dizi (LCS)** temel alır. Bir LCS, her iki özette de aynı göreceli sırayla, ancak mutlaka bitişik olmayan bir dizidir. Bu metrik, bazı kelimeler atlanmış olsa bile cümle seviyesindeki yapıyı ve kelime sırasını koruyan özetleri ödüllendirir. Bitişik eşleşmeler gerektirmediği için ROUGE-N'den daha esnek kabul edilir. ROUGE-L, temel bilgi dizisi korunsa bile cümlelerin yeniden ifade edildiği veya yeniden sıralandığı özetleri değerlendirmek için özellikle kullanışlıdır.

ROUGE-L için geri çağırma şu şekilde hesaplanır:
$$ \text{ROUGE-L Geri Çağırma} = \frac{\text{Uzunluk}(\text{LCS}(\text{Sistem Özeti, Referans Özeti}))}{\text{Uzunluk}(\text{Referans Özeti})} $$

#### 2.3. ROUGE-W (Ağırlıklı En Uzun Ortak Alt Dizi)
ROUGE-W, ROUGE-L'nin bir sınırlamasını ele alan bir uzantısıdır; ROUGE-L, uzunluklarına bakılmaksızın tüm ortak alt dizilere eşit ağırlık verir. ROUGE-W, **daha uzun ortak alt dizilere** birkaç kısa olana göre daha yüksek ağırlıklar atar. Bu, eşleşmelerin sürekliliğine daha duyarlı olmasını sağlar ve referanslarda bulunan daha uzun, kesintisiz metin segmentleri oluşturan sistem özetlerine daha fazla puan verir. Ağırlıklandırma fonksiyonu tipik olarak ortak alt dizilerin uzunluğunun karesini içerir.

#### 2.4. ROUGE-S (Atlamalı İkili N-gram Çakışması)
ROUGE-S, **atlamalı ikili n-gramların (skip-bigram)** çakışmasını ölçer. Bir atlamalı ikili n-gram, bir cümlede, aralarında keyfi boşluklara (atlamalara) izin veren herhangi bir kelime çiftidir. Örneğin, "kedinin üzerinde oturan fare" cümlesinde, "kedinin...fare" bir atlamalı ikili n-gramdır. ROUGE-S, daha yüksek N değerleri için ROUGE-N'den daha hoşgörülü bir metriktir, çünkü bitişiklik gerektirmez. Küçük yapısal farklılıklar veya kelime aralarına eklemeler olduğunda bile sözcüksel ve anlamsal benzerliği yakalamak için kullanışlıdır. ROUGE-S genellikle maksimum atlama mesafesi kısıtlamasıyla birlikte gelir (örn. atlama sınırı olan ROUGE-S*).

### 3. Hesaplama Metodolojisi
Her ROUGE varyantı için, skor tipik olarak **hassasiyet (precision)**, **geri çağırma (recall)** ve **F1-skorunun** bir kombinasyonu olarak hesaplanır.
Şunlar olsun:
-   $X$: Sistem özeti.
-   $Y$: Bir referans özeti.
-   `Çakışma(X, Y)`: $X$ ve $Y$ arasındaki çakışan birimlerin (n-gramlar, LCS uzunluğu, atlamalı ikili n-gramlar) sayısı.
-   `Uzunluk(X)`: $X$'teki toplam birim sayısı.
-   `Uzunluk(Y)`: $Y$'deki toplam birim sayısı.

Skorlar tipik olarak aşağıdaki gibi hesaplanır:
-   **Hassasiyet (P)**: Sistem özetinin ne kadarı referans özetinde de mevcuttur?
    $$ P = \frac{\text{Çakışma}(X, Y)}{\text{Uzunluk}(X)} $$
-   **Geri Çağırma (R)**: Referans özetinin ne kadarı sistem özetinde de mevcuttur?
    $$ R = \frac{\text{Çakışma}(X, Y)}{\text{Uzunluk}(Y)} $$
-   **F1-Skoru**: Hassasiyet ve geri çağırmanın harmonik ortalaması. Her iki yönü de dengelediği için genellikle en çok rapor edilen skordur.
    $$ \text{F1} = \frac{2 \times P \times R}{P + R} $$

Birden fazla referans özeti mevcut olduğunda (sağlamlık için şiddetle tavsiye edilir), ROUGE skoru tipik olarak herhangi bir referans özete karşı elde edilen maksimum geri çağırma (veya F1) alınarak veya tüm referanslardaki skorların ortalaması alınarak hesaplanır. Standart uygulamalar genellikle tüm sistem-referans çiftleri için geri çağırmayı hesaplar ve ardından bunları nihai bir skor için birleştirir.

Küçük harfe dönüştürme, kök bulma, durak kelimeleri kaldırma ve belirteçlere ayırma gibi ön işleme adımları ROUGE skorlarını önemli ölçüde etkileyebilir. Standart uygulama, n-gram sayımından önce küçük harfe dönüştürme ve noktalama işaretlerini kaldırmadır. Kök bulma ve durak kelimeleri kaldırma, önemli anlamsal ayrımları veya cümle yapılarını bazen gizleyebileceği için modern ROUGE uygulamalarında daha az yaygındır.

### 4. ROUGE'un Avantajları ve Dezavantajları

#### 4.1. Avantajları
1.  **Otomatik ve Verimli**: ROUGE, özetleme sistemlerinin hızlı ve otomatik değerlendirilmesine olanak tanır, böylece insan müdahalesi olmadan hızlı yineleme ve deney yapmayı sağlar.
2.  **Yaygın Olarak Kabul Edilmiş**: Özetleme araştırmalarında fiili standart bir metrik olup, farklı çalışmalar ve karşılaştırmalar arasında sonuçları karşılaştırmayı kolaylaştırır.
3.  **İnsan Yargılarıyla Korelasyon**: Çok sayıda çalışma, ROUGE skorlarının, özellikle ROUGE-1 ve ROUGE-2 F1 ile ROUGE-L F1'in, özet kalitesi ve içerik çakışmasına ilişkin insan yargılarıyla makul ölçüde iyi korelasyon gösterdiğini kanıtlamıştır.
4.  **Yorumlanabilirlik**: N-gram veya LCS çakışmasının temel fikri sezgisel olarak anlaşılabilir, bu da skorların nispeten kolay yorumlanmasını sağlar.
5.  **Ayrıntılı Analiz**: Farklı ROUGE varyantları (ROUGE-N, ROUGE-L, ROUGE-S), kelime seçiminden cümle yapısına ve genel bilgi akışına kadar özet kalitesi hakkında farklı bakış açıları sunar.

#### 4.2. Dezavantajları
1.  **Anlamsal Anlama Eksikliği**: ROUGE tamamen sözcüksel bir çakışma metriğidir. Eşanlamlıları, yeniden ifadeleri veya farklı kelimeler kullanan anlamsal olarak benzer cümleleri anlayamaz. Örneğin, "büyük araba" ve "geniş otomobil" aynı anlama gelebilir ancak referanslardan yalnızca biri kullanılıyorsa düşük bir ROUGE skoru alabilir.
2.  **Referans Bağımlılığı**: ROUGE skorları, referans özetlerinin kalitesine ve çeşitliliğine büyük ölçüde bağlıdır. Referans özetleri kötü veya çok azsa, ROUGE skoru sistemin gerçek kalitesini doğru bir şekilde yansıtmayabilir. Birden fazla yüksek kaliteli referans özeti oluşturmak maliyetlidir.
3.  **Yeniden İfadeye Duyarlılık**: Kaynak metindeki bilgiyi etkili bir şekilde yeniden ifade eden ancak referansın tam ifadesini kullanmayan bir sistem özeti, yüksek kaliteli bir özet olsa bile cezalandırılacaktır.
4.  **Sınırlı Kapsam**: ROUGE öncelikle içerik çakışmasını ve geri çağırmayı değerlendirir. Özet kalitesinin diğer önemli yönlerini doğrudan değerlendirmez, örneğin:
    *   **Akıcılık**: Dilin dilbilgisel doğruluğu ve doğallığı.
    *   **Tutarlılık**: Fikirlerin mantıksal akışı ve organizasyonel yapısı.
    *   **Gerçekçilik/Sadakat**: Özetin, kaynak belgeyle olgusal olarak tutarlı bilgiler içerip içermediği. Bir özet, referans kelimelerle eşleşiyorsa yüksek ROUGE skoru alabilir ancak halüsinasyonlar içeriyorsa bunlar da halüsinasyon olabilir.
    *   **Yenilik/Bilgi Kapsamı**: Özetin en önemli bilgileri gereksiz yere tekrar etmeden yeterince kapsayıp kapsamadığı.
5.  **Soyutlayıcı Özetleme ile Zorluk**: Çıkarımsal özetler için yararlı olsa da, ROUGE, genellikle cümleleri yeniden ifade eden ve yeni cümleler üreten soyutlayıcı özetlerle daha büyük zorluklar yaşar, bu da referanslarla doğrudan sözcüksel çakışmayı azaltır. BERTScore veya sadakat metrikleri gibi daha yeni metrikler, soyutlayıcı özetleme değerlendirmesi için ROUGE ile birlikte kullanılır.

### 5. Kod Örneği
İşte yaygın ve sağlam bir uygulama olan `rouge-score` kütüphanesini kullanarak ROUGE skorlarının nasıl hesaplanacağını gösteren basit bir Python kod parçacığı.

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference_summary: str, system_summary: str):
    """
    Bir sistem özeti ile referans özet arasında ROUGE-1, ROUGE-2 ve ROUGE-L F1 skorlarını hesaplar.

    Argümanlar:
        reference_summary (str): İnsan tarafından yazılmış referans özet.
        system_summary (str): Makine tarafından oluşturulmuş sistem özeti.

    Döndürür:
        dict: ROUGE-1, ROUGE-2 ve ROUGE-L F1 skorlarını içeren bir sözlük.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, system_summary)

    # Her ROUGE tipi için F1 skorlarını çıkarma
    rouge1_f1 = scores['rouge1'].fmeasure
    rouge2_f1 = scores['rouge2'].fmeasure
    rougeL_f1 = scores['rougeL'].fmeasure

    return {
        "rouge-1_f1": rouge1_f1,
        "rouge-2_f1": rouge2_f1,
        "rouge-L_f1": rougeL_f1,
    }

# Örnek kullanım:
referans = "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar."
sistem = "Çevik kahverengi bir tilki uykulu bir köpeğin üzerinden atlıyor."

rouge_sonuçları = calculate_rouge(referans, sistem)
print(f"ROUGE Skorları: {rouge_sonuçları}")

# Daha az çakışma olan başka bir örnek
referans_2 = "Yapay zeka birçok endüstriyi dönüştürüyor."
sistem_2 = "YZ birçok sektörü değiştiriyor."

rouge_sonuçları_2 = calculate_rouge(referans_2, sistem_2)
print(f"ROUGE Skorları (daha az çakışma): {rouge_sonuçları_2}")

(Kod örneği bölümünün sonu)
```

### 6. Sonuç
ROUGE, özetleme sistemlerinin otomatik değerlendirmesi için vazgeçilmez bir araç olmaya devam etmektedir. Çeşitli metrikleri, sistem tarafından üretilen ve insan tarafından yazılan özetler arasındaki sözcüksel ve yapısal çakışmayı nicelleştirmek için sağlam bir çerçeve sunarak, içerik kapsamı ve temel dilsel kalıplar hakkında değerli bilgiler sağlar. Verimliliği ve insan yargılarıyla korelasyonu nedeniyle yaygın olarak benimsenmiş olsa da, ROUGE'un anlamsal nüansları, mantıksal tutarlılığı ve olgusal tutarlılığı yakalayamaması gibi doğal sınırlamalarını kabul etmek çok önemlidir. Araştırmacılar ve uygulayıcılar, özellikle yüksek derecede soyutlayıcı modeller bağlamında, anlamsal benzerlik için **BERTScore** veya bütünsel kalite için insan değerlendirmesi gibi diğer metriklerle ROUGE'u sıklıkla tamamlarlar. ROUGE'un güçlü ve zayıf yönlerini anlamak, onu etkili bir şekilde uygulamak ve otomatik metin özetleme alanında daha fazla ilerleme sağlamak için anahtardır.

