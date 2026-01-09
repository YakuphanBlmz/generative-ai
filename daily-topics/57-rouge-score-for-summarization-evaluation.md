# ROUGE Score for Summarization Evaluation

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Types of ROUGE Scores](#2-types-of-rouge-scores)
  - [2.1. ROUGE-N](#21-rouge-n)
  - [2.2. ROUGE-L](#22-rouge-l)
  - [2.3. ROUGE-S (Skip-Bigram)](#23-rouge-s-skip-bigram)
- [3. Advantages and Disadvantages](#3-advantages-and-disadvantages)
  - [3.1. Advantages](#31-advantages)
  - [3.2. Disadvantages](#32-disadvantages)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

In the rapidly evolving field of Generative AI, automatic text summarization stands as a critical task, aiming to distill the most important information from a source text into a concise and coherent summary. Evaluating the quality of these generated summaries is paramount for research, development, and practical applications. While human evaluation remains the gold standard, it is often resource-intensive, time-consuming, and subjective. Consequently, **automatic evaluation metrics** have become indispensable tools. Among these, the **ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score** is arguably the most widely adopted and influential metric for assessing the quality of automatically generated summaries by comparing them to human-produced reference summaries.

Developed by Chin-Yew Lin in 2004, ROUGE is not a single metric but a suite of metrics that quantify the overlap between an automatically generated candidate summary and a set of human-authored reference summaries. Its core principle is based on measuring the number of overlapping units (such as n-grams, word sequences, or word pairs) between the candidate and reference texts. The higher the overlap, the better the candidate summary is deemed to be, reflecting its ability to capture key information present in the human-written gold standards. ROUGE scores are typically reported as **precision**, **recall**, and **F-measure** (or F1-score), providing different perspectives on how well the candidate summary aligns with the reference.

<a name="2-types-of-rouge-scores"></a>
## 2. Types of ROUGE Scores

ROUGE offers several variants, each focusing on different linguistic units and aspects of text similarity. The most commonly used types are ROUGE-N, ROUGE-L, and ROUGE-S.

<a name="21-rouge-n"></a>
### 2.1. ROUGE-N

**ROUGE-N** measures the overlap of n-grams between the candidate and reference summaries. An **n-gram** is a contiguous sequence of *n* items (words) from a given sample of text.
The score is calculated as follows:

*   **ROUGE-N Recall:** Measures how many n-grams in the reference summary appear in the candidate summary.
    $ \text{Recall} = \frac{\sum_{S \in \{\text{Reference Summaries}\}} \sum_{\text{n-gram} \in S} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{S \in \{\text{Reference Summaries}\}} \sum_{\text{n-gram} \in S} \text{Count}(\text{n-gram})} $
    Where $\text{Count}_{\text{match}}(\text{n-gram})$ is the maximum number of n-grams co-occurring in a candidate summary and a reference summary, and $\text{Count}(\text{n-gram})$ is the number of n-grams in the reference summary.

*   **ROUGE-N Precision:** Measures how many n-grams in the candidate summary also appear in the reference summary.
    $ \text{Precision} = \frac{\sum_{S \in \{\text{Reference Summaries}\}} \sum_{\text{n-gram} \in S} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{\text{n-gram} \in \text{Candidate Summary}} \text{Count}(\text{n-gram})} $
    Where $\text{Count}(\text{n-gram})$ is the number of n-grams in the candidate summary.

*   **ROUGE-N F-measure (F1-score):** The harmonic mean of precision and recall, offering a balanced view.
    $ \text{F-measure} = \frac{(1 + \beta^2) \times \text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}} $
    Typically, $\beta = 1$ for an F1-score, giving equal weight to precision and recall.

The most common instances are:
*   **ROUGE-1:** Measures the overlap of unigrams (single words). It indicates how much the candidate summary captures individual important words from the reference.
*   **ROUGE-2:** Measures the overlap of bigrams (sequences of two words). It reflects the fluency and grammatical structure by checking for common two-word phrases.

<a name="22-rouge-l"></a>
### 2.2. ROUGE-L

**ROUGE-L** (Longest Common Subsequence) is based on the **longest common subsequence (LCS)** between the candidate and reference summaries. Unlike ROUGE-N, LCS does not require the n-grams to be consecutive. It aims to capture the longest sequence of words that appear in both texts in the same order, even if other words intervene. This metric is better at reflecting sentence-level structure and overall content flow without strict adjacency requirements, making it more flexible for evaluating summaries that might rephrase content.

The calculation for ROUGE-L also involves precision, recall, and F-measure, adapted for LCS:
*   **ROUGE-L Recall:** $ \text{Recall}_{\text{LCS}} = \frac{\text{Length}(\text{LCS}(C, R))}{\text{Length}(R)} $
*   **ROUGE-L Precision:** $ \text{Precision}_{\text{LCS}} = \frac{\text{Length}(\text{LCS}(C, R))}{\text{Length}(C)} $
*   **ROUGE-L F-measure:** $ \text{F-measure}_{\text{LCS}} = \frac{(1 + \beta^2) \times \text{Precision}_{\text{LCS}} \times \text{Recall}_{\text{LCS}}}{\beta^2 \times \text{Precision}_{\text{LCS}} + \text{Recall}_{\text{LCS}}} $
Where $C$ is the candidate summary, $R$ is the reference summary, and $\text{Length}(\text{LCS}(C, R))$ is the length of the longest common subsequence of words between $C$ and $R$.

<a name="23-rouge-s-skip-bigram"></a>
### 2.3. ROUGE-S (Skip-Bigram)

**ROUGE-S** is based on **skip-bigram co-occurrence**, meaning it measures the overlap of word pairs that are not necessarily consecutive but are within a certain maximum distance (or "skip" limit). For example, "cat sat mat" would have skip-bigrams like "cat mat" (with one skip). This metric is useful for capturing flexible word order and semantic similarity that might be missed by strict n-gram matching, though it is less frequently reported than ROUGE-N and ROUGE-L.

<a name="3-advantages-and-disadvantages"></a>
## 3. Advantages and Disadvantages

Like any evaluation metric, ROUGE scores come with their own set of strengths and weaknesses.

<a name="31-advantages"></a>
### 3.1. Advantages

*   **Objectivity and Automation:** ROUGE provides an **objective and fully automatic** way to evaluate summaries, eliminating human bias and enabling large-scale, rapid assessment of summarization models. This is crucial for iterative development and hyperparameter tuning.
*   **Widely Adopted and Benchmarked:** ROUGE has become the **de facto standard metric** in the NLP community for summarization tasks. Its widespread use allows for easy comparison of different models and research findings across various datasets.
*   **Reproducibility:** Given the same candidate and reference summaries, ROUGE scores are **fully reproducible**, ensuring consistency in evaluation across different experiments.
*   **Multiple Granularities:** The various ROUGE types (N, L, S) offer different perspectives on summary quality, from word-level overlap to structural similarity, allowing for a multifaceted analysis.

<a name="32-disadvantages"></a>
### 3.2. Disadvantages

*   **Lack of Semantic Understanding:** ROUGE is primarily a **lexical overlap metric**. It struggles to account for synonyms, paraphrases, or semantically similar phrases that use different words. For example, "buy a car" and "purchase an automobile" convey the same meaning but would have low ROUGE scores. This is a significant limitation, especially for **abstractive summarization** models that often rephrase content.
*   **Dependency on Reference Summaries:** The quality of ROUGE scores is highly dependent on the **quality and diversity of human-authored reference summaries**. A single, poorly written, or biased reference can significantly skew results. Using multiple reference summaries helps mitigate this, but generating them is costly.
*   **Surface-Level Metrics:** ROUGE does not directly assess crucial aspects of summary quality such as **fluency, coherence, factual consistency, or grammatical correctness**. A summary might achieve high ROUGE scores by simply extracting sentences from the source (extractive summarization) but still be incoherent.
*   **Potential for "Gaming" the Metric:** Models can sometimes be optimized to "game" ROUGE scores by prioritizing keyword overlap, potentially leading to summaries that perform well on ROUGE but are not necessarily high-quality in a human judgment.
*   **Contextual Blindness:** ROUGE treats words as independent units (in ROUGE-N) or sequences, without deep understanding of the context or importance of specific terms within the broader document.

<a name="4-code-example"></a>
## 4. Code Example

To illustrate the practical application of ROUGE score calculation, we can use the `rouge_score` library in Python. This library provides a straightforward interface for computing ROUGE-N, ROUGE-L, and other variants.

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores between a candidate
    and a reference text.

    Args:
        reference (str): The gold standard summary.
        candidate (str): The summary generated by a model.

    Returns:
        dict: A dictionary containing ROUGE F-measure scores for ROUGE-1, ROUGE-2, and ROUGE-L.
    """
    # Initialize the ROUGE scorer with the desired metrics.
    # use_stemmer=True applies stemming, which can improve recall for related words.
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(reference, candidate)

    # Extract F-measure for each ROUGE type
    results = {
        'rouge1_fmeasure': scores['rouge1'].fmeasure,
        'rouge2_fmeasure': scores['rouge2'].fmeasure,
        'rougeL_fmeasure': scores['rougeL'].fmeasure,
    }
    return results

# Example usage 1: High overlap
reference_summary_1 = "The quick brown fox jumps over the lazy dog."
candidate_summary_1 = "A quick brown fox jumps over the lazy dog."

rouge_scores_1 = calculate_rouge(reference_summary_1, candidate_summary_1)
print(f"Example 1 - Reference: '{reference_summary_1}'")
print(f"            Candidate: '{candidate_summary_1}'")
print(f"            ROUGE Scores: {rouge_scores_1}")
print("-" * 30)

# Example usage 2: Moderate overlap, some rephrasing
reference_summary_2 = "Artificial intelligence is a rapidly evolving field with many applications."
candidate_summary_2 = "AI is a fast-growing area of study, showing diverse applications."

rouge_scores_2 = calculate_rouge(reference_summary_2, candidate_summary_2)
print(f"Example 2 - Reference: '{reference_summary_2}'")
print(f"            Candidate: '{candidate_summary_2}'")
print(f"            ROUGE Scores: {rouge_scores_2}")
print("-" * 30)

# Example usage 3: Low overlap, different phrasing
reference_summary_3 = "The government decided to implement new economic policies."
candidate_summary_3 = "New fiscal measures were introduced by the administration."

rouge_scores_3 = calculate_rouge(reference_summary_3, candidate_summary_3)
print(f"Example 3 - Reference: '{reference_summary_3}'")
print(f"            Candidate: '{candidate_summary_3}'")
print(f"            ROUGE Scores: {rouge_scores_3}")

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

The ROUGE score remains a cornerstone in the automatic evaluation of summarization systems, providing an efficient and standardized method for quantifying the lexical and structural overlap between generated and human-written summaries. Its variants—ROUGE-N, ROUGE-L, and ROUGE-S—offer different perspectives on similarity, making it a versatile tool for initial assessments and large-scale benchmarking.

However, it is crucial to recognize ROUGE's inherent limitations, particularly its inability to capture semantic nuances, factual consistency, coherence, and fluency. As generative AI models, especially abstractive summarizers, become more sophisticated and capable of producing highly novel and rephrased content, the shortcomings of purely lexical overlap metrics become more pronounced. Therefore, while ROUGE continues to serve as a valuable indicator, it should ideally be complemented by other advanced metrics (such as **BERTScore**, **MoverScore**, or **BLEURT** which leverage contextual embeddings) and, most importantly, thorough human evaluation to achieve a truly holistic and robust assessment of summarization quality. The pursuit of more comprehensive and human-aligned evaluation metrics is an ongoing and vital area of research in natural language processing.

---
<br>

<a name="türkçe-içerik"></a>
## Özet Değerlendirme için ROUGE Skoru

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. ROUGE Skor Türleri](#2-rouge-skor-türleri)
  - [2.1. ROUGE-N](#21-rouge-n)
  - [2.2. ROUGE-L](#22-rouge-l)
  - [2.3. ROUGE-S (Atlamalı İkili Kelime)](#23-rouge-s-atlamalı-ikili-kelime)
- [3. Avantajlar ve Dezavantajlar](#3-avantajlar-ve-dezavantajlar)
  - [3.1. Avantajlar](#31-avantajlar)
  - [3.2. Dezavantajlar](#32-dezavantajlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Üretken Yapay Zeka (Generative AI) alanındaki hızlı gelişmelerde, otomatik metin özetleme kritik bir görev olarak öne çıkmaktadır ve kaynak metinden en önemli bilgileri özlü ve tutarlı bir özete damıtmayı amaçlamaktadır. Üretilen bu özetlerin kalitesini değerlendirmek, araştırma, geliştirme ve pratik uygulamalar için büyük önem taşımaktadır. İnsan değerlendirmesi hala altın standart olsa da, genellikle kaynak yoğun, zaman alıcı ve özneldir. Sonuç olarak, **otomatik değerlendirme metrikleri** vazgeçilmez araçlar haline gelmiştir. Bunlar arasında, otomatik olarak oluşturulan özetlerin insan tarafından üretilen referans özetlerle karşılaştırılması yoluyla kalitesini değerlendirmek için tartışmasız en yaygın kabul gören ve etkili metrik, **ROUGE (Recall-Oriented Understudy for Gisting Evaluation - Özetleme Değerlendirmesi için Geri Çağırmaya Yönelik İnceleme) skoru**dur.

Chin-Yew Lin tarafından 2004 yılında geliştirilen ROUGE, tek bir metrik değil, otomatik olarak oluşturulmuş bir aday özet ile bir dizi insan tarafından yazılmış referans özet arasındaki çakışmayı nicel olarak belirleyen bir metrikler paketidir. Temel prensibi, aday ve referans metinler arasındaki çakışan birimlerin (n-gramlar, kelime dizileri veya kelime çiftleri gibi) sayısını ölçmeye dayanır. Çakışma ne kadar yüksek olursa, aday özetin o kadar iyi olduğu kabul edilir ve bu da insan tarafından yazılan altın standartlardaki anahtar bilgileri yakalama yeteneğini yansıtır. ROUGE skorları genellikle **kesinlik (precision)**, **geri çağırma (recall)** ve **F-ölçüsü** (veya F1-skoru) olarak rapor edilir ve aday özetin referansla ne kadar iyi örtüştüğüne dair farklı bakış açıları sunar.

<a name="2-rouge-skor-türleri"></a>
## 2. ROUGE Skor Türleri

ROUGE, her biri metin benzerliğinin farklı dilsel birimlerine ve yönlerine odaklanan çeşitli varyantlar sunar. En yaygın kullanılan türler ROUGE-N, ROUGE-L ve ROUGE-S'dir.

<a name="21-rouge-n"></a>
### 2.1. ROUGE-N

**ROUGE-N**, aday ve referans özetler arasındaki n-gram çakışmasını ölçer. Bir **n-gram**, verilen bir metin örneğindeki *n* öğeden (kelime) oluşan bitişik bir dizidir.
Skor aşağıdaki gibi hesaplanır:

*   **ROUGE-N Geri Çağırma (Recall):** Referans özetteki kaç n-gram'ın aday özette yer aldığını ölçer.
    $ \text{Geri Çağırma} = \frac{\sum_{S \in \{\text{Referans Özetler}\}} \sum_{\text{n-gram} \in S} \text{Eşleşme Sayısı}(\text{n-gram})}{\sum_{S \in \{\text{Referans Özetler}\}} \sum_{\text{n-gram} \in S} \text{Sayım}(\text{n-gram})} $
    Burada $\text{Eşleşme Sayısı}(\text{n-gram})$, bir aday özet ile bir referans özette birlikte bulunan n-gramların maksimum sayısıdır ve $\text{Sayım}(\text{n-gram})$ referans özetteki n-gramların sayısıdır.

*   **ROUGE-N Kesinlik (Precision):** Aday özetteki kaç n-gram'ın referans özette de yer aldığını ölçer.
    $ \text{Kesinlik} = \frac{\sum_{S \in \{\text{Referans Özetler}\}} \sum_{\text{n-gram} \in S} \text{Eşleşme Sayısı}(\text{n-gram})}{\sum_{\text{n-gram} \in \text{Aday Özet}} \text{Sayım}(\text{n-gram})} $
    Burada $\text{Sayım}(\text{n-gram})$, aday özetteki n-gramların sayısıdır.

*   **ROUGE-N F-ölçüsü (F1-skoru):** Kesinlik ve geri çağırmanın harmonik ortalamasıdır ve dengeli bir görünüm sunar.
    $ \text{F-ölçüsü} = \frac{(1 + \beta^2) \times \text{Kesinlik} \times \text{Geri Çağırma}}{\beta^2 \times \text{Kesinlik} + \text{Geri Çağırma}} $
    Tipik olarak, F1-skoru için $\beta = 1$ olup, kesinlik ve geri çağırmaya eşit ağırlık verir.

En yaygın örnekler şunlardır:
*   **ROUGE-1:** Tekli kelimelerin (unigramlar) çakışmasını ölçer. Aday özetin referanstan bireysel önemli kelimeleri ne kadar yakaladığını gösterir.
*   **ROUGE-2:** İkili kelimelerin (bigramlar) çakışmasını ölçer. Ortak iki kelimeli ifadeleri kontrol ederek akıcılığı ve dilbilgisel yapıyı yansıtır.

<a name="22-rouge-l"></a>
### 2.2. ROUGE-L

**ROUGE-L** (Longest Common Subsequence - En Uzun Ortak Alt Dizi), aday ve referans özetler arasındaki **en uzun ortak alt diziye (LCS)** dayanır. ROUGE-N'den farklı olarak, LCS n-gramların ardışık olmasını gerektirmez. Her iki metinde de aynı sırada, ancak araya başka kelimeler girmiş olsa bile, görünen en uzun kelime dizisini yakalamayı amaçlar. Bu metrik, katı bitişiklik gereksinimleri olmaksızın cümle düzeyindeki yapıyı ve genel içerik akışını daha iyi yansıtır, bu da içeriği yeniden ifade edebilecek özetleri değerlendirmek için daha esnek olmasını sağlar.

ROUGE-L'nin hesaplaması da LCS'ye uyarlanmış kesinlik, geri çağırma ve F-ölçüsünü içerir:
*   **ROUGE-L Geri Çağırma:** $ \text{Geri Çağırma}_{\text{LCS}} = \frac{\text{Uzunluk}(\text{LCS}(A, R))}{\text{Uzunluk}(R)} $
*   **ROUGE-L Kesinlik:** $ \text{Kesinlik}_{\text{LCS}} = \frac{\text{Uzunluk}(\text{LCS}(A, R))}{\text{Uzunluk}(A)} $
*   **ROUGE-L F-ölçüsü:** $ \text{F-ölçüsü}_{\text{LCS}} = \frac{(1 + \beta^2) \times \text{Kesinlik}_{\text{LCS}} \times \text{Geri Çağırma}_{\text{LCS}}}{\beta^2 \times \text{Kesinlik}_{\text{LCS}} + \text{Geri Çağırma}_{\text{LCS}}} $
Burada $A$ aday özet, $R$ referans özet ve $\text{Uzunluk}(\text{LCS}(A, R))$, $A$ ve $R$ arasındaki kelimelerin en uzun ortak alt dizisinin uzunluğudur.

<a name="23-rouge-s-atlamalı-ikili-kelime"></a>
### 2.3. ROUGE-S (Atlamalı İkili Kelime)

**ROUGE-S**, **atlamalı ikili kelime (skip-bigram) birlikteliğine** dayanır; yani, ardışık olması gerekmeyen ancak belirli bir maksimum mesafe (veya "atlama" sınırı) içinde olan kelime çiftlerinin çakışmasını ölçer. Örneğin, "kedi hasır oturdu" ifadesinde "kedi oturdu" (bir atlamayla) gibi atlamalı ikili kelimeler bulunur. Bu metrik, katı n-gram eşleştirmesiyle kaçırılabilecek esnek kelime sıralamasını ve anlamsal benzerliği yakalamak için kullanışlıdır, ancak ROUGE-N ve ROUGE-L'den daha az rapor edilmektedir.

<a name="3-avantajlar-ve-dezavantajlar"></a>
## 3. Avantajlar ve Dezavantajlar

Her değerlendirme metriği gibi, ROUGE skorlarının da kendi güçlü ve zayıf yönleri vardır.

<a name="31-avantajlar"></a>
### 3.1. Avantajlar

*   **Nesnellik ve Otomasyon:** ROUGE, özetleri değerlendirmek için **nesnel ve tamamen otomatik** bir yol sunar, insan önyargısını ortadan kaldırır ve özetleme modellerinin büyük ölçekli, hızlı değerlendirilmesini sağlar. Bu, tekrarlı geliştirme ve hiperparametre ayarlaması için çok önemlidir.
*   **Yaygın Kabul ve Kıyaslama:** ROUGE, NLP topluluğunda özetleme görevleri için **fiili standart metrik** haline gelmiştir. Yaygın kullanımı, farklı modellerin ve araştırma bulgularının çeşitli veri kümeleri arasında kolayca karşılaştırılmasını sağlar.
*   **Tekrarlanabilirlik:** Aynı aday ve referans özetler verildiğinde, ROUGE skorları **tamamen tekrarlanabilir**dir, bu da farklı deneylerde değerlendirme tutarlılığını sağlar.
*   **Çoklu Granülerlik:** Çeşitli ROUGE türleri (N, L, S), kelime düzeyindeki çakışmadan yapısal benzerliğe kadar özet kalitesine farklı perspektifler sunarak çok yönlü bir analiz yapılmasına olanak tanır.

<a name="32-dezavantajlar"></a>
### 3.2. Dezavantajlar

*   **Anlamsal Anlayış Eksikliği:** ROUGE öncelikle **sözcüksel bir çakışma metriğidir**. Eş anlamlıları, eşanlamlı ifadeleri veya farklı kelimeler kullanan anlamsal olarak benzer ifadeleri açıklamakta zorlanır. Örneğin, "araba satın almak" ve "bir otomobil tedarik etmek" aynı anlamı taşır ancak düşük ROUGE skorlarına sahip olur. Bu, özellikle içeriği yeniden ifade eden **soyutlayıcı özetleme** modelleri için önemli bir sınırlamadır.
*   **Referans Özetlere Bağımlılık:** ROUGE skorlarının kalitesi, **insan tarafından yazılmış referans özetlerinin kalitesine ve çeşitliliğine** büyük ölçüde bağlıdır. Tek, kötü yazılmış veya yanlı bir referans, sonuçları önemli ölçüde çarpıtabilir. Birden fazla referans özeti kullanmak bunu azaltmaya yardımcı olur, ancak bunları oluşturmak maliyetlidir.
*   **Yüzey Seviyesi Metrikler:** ROUGE, özet kalitesinin **akıcılık, tutarlılık, olgusal tutarlılık veya dilbilgisel doğruluk** gibi önemli yönlerini doğrudan değerlendirmez. Bir özet, yalnızca kaynaktan cümleler çıkararak (çıkarımcı özetleme) yüksek ROUGE skorları elde edebilir, ancak yine de tutarsız olabilir.
*   **Metriği "Kandırma" Potansiyeli:** Modeller bazen anahtar kelime çakışmasına öncelik vererek ROUGE skorlarını "kandırmak" için optimize edilebilir, bu da ROUGE'da iyi performans gösteren ancak insan yargısında mutlaka yüksek kaliteli olmayan özetlere yol açabilir.
*   **Bağlamsal Körlük:** ROUGE, kelimeleri (ROUGE-N'de) veya dizileri, daha geniş belge içindeki bağlamı veya belirli terimlerin önemini derinlemesine anlamadan bağımsız birimler olarak ele alır.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

ROUGE skoru hesaplamasının pratik uygulamasını göstermek için Python'daki `rouge_score` kütüphanesini kullanabiliriz. Bu kütüphane, ROUGE-N, ROUGE-L ve diğer varyantları hesaplamak için basit bir arayüz sağlar.

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    """
    Bir aday özet ile bir referans metin arasındaki ROUGE-1, ROUGE-2
    ve ROUGE-L skorlarını hesaplar.

    Args:
        reference (str): Altın standart özet.
        candidate (str): Bir model tarafından oluşturulan özet.

    Returns:
        dict: ROUGE-1, ROUGE-2 ve ROUGE-L için ROUGE F-ölçüsü skorlarını içeren bir sözlük.
    """
    # İstenen metriklerle ROUGE skorlayıcısını başlatın.
    # use_stemmer=True kök ayıklama uygular, bu da ilgili kelimeler için geri çağırmayı artırabilir.
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Skorları hesapla
    scores = scorer.score(reference, candidate)

    # Her ROUGE türü için F-ölçüsünü çıkar
    results = {
        'rouge1_fmeasure': scores['rouge1'].fmeasure,
        'rouge2_fmeasure': scores['rouge2'].fmeasure,
        'rougeL_fmeasure': scores['rougeL'].fmeasure,
    }
    return results

# Örnek kullanım 1: Yüksek çakışma
referans_özet_1 = "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar."
aday_özet_1 = "Çabuk kahverengi bir tilki tembel köpeğin üzerinden atlar."

rouge_skorları_1 = calculate_rouge(referans_özet_1, aday_özet_1)
print(f"Örnek 1 - Referans: '{referans_özet_1}'")
print(f"            Aday: '{aday_özet_1}'")
print(f"            ROUGE Skorları: {rouge_skorları_1}")
print("-" * 30)

# Örnek kullanım 2: Orta çakışma, bazı yeniden ifade etmeler
referans_özet_2 = "Yapay zeka, birçok uygulamaya sahip, hızla gelişen bir alandır."
aday_özet_2 = "YZ, çeşitli uygulamalar gösteren, hızlı büyüyen bir çalışma alanıdır."

rouge_skorları_2 = calculate_rouge(referans_özet_2, aday_özet_2)
print(f"Örnek 2 - Referans: '{referans_özet_2}'")
print(f"            Aday: '{aday_özet_2}'")
print(f"            ROUGE Skorları: {rouge_skorları_2}")
print("-" * 30)

# Örnek kullanım 3: Düşük çakışma, farklı ifade
referans_özet_3 = "Hükümet, yeni ekonomi politikaları uygulamaya karar verdi."
aday_özet_3 = "Yönetim tarafından yeni mali önlemler getirildi."

rouge_skorları_3 = calculate_rouge(referans_özet_3, aday_özet_3)
print(f"Örnek 3 - Referans: '{referans_özet_3}'")
print(f"            Aday: '{aday_özet_3}'")
print(f"            ROUGE Skorları: {rouge_skorları_3}")

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

ROUGE skoru, üretilen ve insan tarafından yazılan özetler arasındaki sözcüksel ve yapısal çakışmayı nicel olarak belirlemek için verimli ve standartlaştırılmış bir yöntem sağlayarak, özetleme sistemlerinin otomatik değerlendirmesinde temel bir köşe taşı olmaya devam etmektedir. ROUGE-N, ROUGE-L ve ROUGE-S gibi varyantları, benzerliğe farklı perspektifler sunarak, ilk değerlendirmeler ve büyük ölçekli kıyaslamalar için çok yönlü bir araç haline getirir.

Ancak, ROUGE'un doğasında var olan sınırlamalarını, özellikle anlamsal nüansları, olgusal tutarlılığı, tutarlılığı ve akıcılığı yakalayamamasını kabul etmek çok önemlidir. Üretken yapay zeka modelleri, özellikle soyutlayıcı özetleyiciler, giderek daha sofistike hale geldikçe ve oldukça yeni ve yeniden ifade edilmiş içerik üretme yeteneğine sahip oldukça, tamamen sözcüksel çakışma metriklerinin eksiklikleri daha belirgin hale gelmektedir. Bu nedenle, ROUGE değerli bir gösterge olarak hizmet etmeye devam etse de, özetleme kalitesinin gerçekten bütünsel ve sağlam bir değerlendirmesini elde etmek için ideal olarak diğer gelişmiş metrikler (bağlamsal gömüleri kullanan **BERTScore**, **MoverScore** veya **BLEURT** gibi) ve en önemlisi kapsamlı insan değerlendirmesi ile tamamlanmalıdır. Daha kapsamlı ve insan odaklı değerlendirme metrikleri arayışı, doğal dil işlemede devam eden ve hayati bir araştırma alanıdır.