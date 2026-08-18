# Machine Translation: Statistical vs. Neural Methods

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Statistical Machine Translation (SMT)](#2-statistical-machine-translation-smt)
  - [2.1. Core Principles](#21-core-principles)
  - [2.2. Advantages and Disadvantages](#22-advantages-and-disadvantages)
- [3. Neural Machine Translation (NMT)](#3-neural-machine-translation-nmt)
  - [3.1. Core Principles](#31-core-principles)
  - [3.2. Advantages and Disadvantages](#32-advantages-and-disadvantages)
- [4. Comparative Analysis and Evolution](#4-comparative-analysis-and-evolution)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
Machine Translation (MT) stands as a foundational yet continually evolving field within Natural Language Processing (NLP), dedicated to the automatic conversion of text or speech from one natural language into another. Its utility spans various domains, from facilitating global communication and commerce to enabling access to information across linguistic barriers. Historically, the pursuit of automatic translation has seen several paradigms, each driven by advancements in computational linguistics and artificial intelligence. This document explores two dominant approaches: **Statistical Machine Translation (SMT)** and **Neural Machine Translation (NMT)**, analyzing their underlying methodologies, operational principles, comparative strengths, weaknesses, and the transformative shift witnessed in the field with the advent of deep learning.

## 2. Statistical Machine Translation (SMT)
**Statistical Machine Translation (SMT)** emerged as the predominant paradigm in MT during the 1990s, fundamentally shifting from rule-based systems to approaches that leveraged large parallel corpora to learn translation patterns. Instead of relying on hand-crafted linguistic rules, SMT employs statistical models to estimate the most probable translation of a given text.

### 2.1. Core Principles
SMT operates on the principle of finding the target sentence *T* that maximizes the posterior probability *P(T|S)*, where *S* is the source sentence. This is typically achieved using Bayes' Theorem: *P(T|S) = P(S|T) * P(T) / P(S)*. Since *P(S)* is constant for a given source sentence, the problem reduces to finding *argmax<sub>T</sub> P(S|T) * P(T)*. This formulation introduces two critical components:
1.  **Translation Model (P(S|T))**: This model quantifies the probability that a source sentence *S* would be generated from a target sentence *T*. It is responsible for translating individual words or **phrases** and aligning them between languages. Early SMT systems were **word-based**, focusing on translating words independently. However, subsequent advancements led to **phrase-based SMT (PBSMT)**, which translates contiguous sequences of words (phrases), significantly improving fluency and handling of reordering. Phrase tables, containing pairs of source and target phrases with their translation probabilities, are central to PBSMT.
2.  **Language Model (P(T))**: This model assesses the fluency and grammatical correctness of the target sentence *T* in the target language, independently of the source sentence. It assigns a probability to a sequence of words, typically based on **n-gram** probabilities (the likelihood of a word appearing given the preceding *n-1* words). A higher probability from the language model indicates a more natural-sounding translation.

The process of finding the optimal translation *T* that maximizes this product is known as **decoding**, a computationally intensive search problem. SMT systems typically integrate additional features, such as word bonus, distortion models (to account for word reordering), and lexical weighting, combined linearly with weights optimized through techniques like Minimum Error Rate Training (MERT).

### 2.2. Advantages and Disadvantages
**Advantages of SMT:**
*   **Data-driven:** SMT systems learn directly from data, making them adaptable to new language pairs and domains without extensive rule engineering.
*   **Robustness:** They can handle grammatical variations and irregularities more effectively than purely rule-based systems.
*   **Explainability:** The models, particularly phrase tables, offered some degree of interpretability regarding how translations were derived.

**Disadvantages of SMT:**
*   **Feature engineering:** Required careful design of features and statistical models.
*   **Data sparsity:** Performance heavily relied on the availability of vast parallel corpora, and rare words or phrases posed significant challenges.
*   **Local optimization:** SMT models often optimized components independently, leading to potential suboptimal global translations.
*   **Limited context:** N-gram language models capture only short-range dependencies, limiting their ability to understand and generate long-range coherent text.
*   **Phrase boundary issues:** Segmentation into phrases could be ambiguous and lead to suboptimal choices.

## 3. Neural Machine Translation (NMT)
**Neural Machine Translation (NMT)** represents a paradigm shift from SMT, utilizing artificial neural networks to learn an end-to-end mapping from source to target text. Introduced in the mid-2010s, NMT rapidly surpassed SMT in performance, becoming the dominant approach in modern MT systems.

### 3.1. Core Principles
NMT models are typically based on an **encoder-decoder architecture**.
1.  **Encoder**: The encoder processes the source sentence, transforming it into a continuous-space representation, often called a **context vector** or **thought vector**. This vector is intended to encapsulate the semantic meaning of the entire source sentence. Early NMT systems used Recurrent Neural Networks (RNNs) such as Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRU) to process sequences word by word, capturing dependencies.
2.  **Decoder**: The decoder then takes this context vector and generates the target sentence word by word. It's also typically an RNN that, at each step, outputs a word based on the context vector and the words generated so far.

A crucial innovation that significantly boosted NMT performance and addressed the "bottleneck" issue of encoding an entire sentence into a single fixed-size vector was the **attention mechanism**. Attention allows the decoder to "look back" at different parts of the source sentence with varying degrees of focus while generating each target word. This enables the model to handle long sentences more effectively and establish more direct dependencies between source and target words.

The most profound architectural innovation in NMT is the **Transformer** model, introduced in 2017. Transformers entirely eschewed recurrent layers, relying instead on **self-attention mechanisms** to process input sequences in parallel. This parallelism significantly improved training speed and allowed models to capture long-range dependencies more efficiently than RNNs. The Transformer architecture, with its multi-head attention and feed-forward networks, has become the de-facto standard for NMT and many other NLP tasks.

### 3.2. Advantages and Disadvantages
**Advantages of NMT:**
*   **End-to-end learning:** NMT models learn directly from raw text, eliminating the need for separate, hand-engineered components (e.g., phrase tables, language models, alignment models). This simplifies the pipeline and allows for global optimization.
*   **Fluency and accuracy:** NMT systems often produce significantly more fluent and natural-sounding translations, largely due to their ability to model long-range dependencies and complex linguistic nuances.
*   **Contextual understanding:** Through distributed representations (**embeddings**) and attention mechanisms, NMT can capture the context of words more effectively.
*   **Generalization:** They can generalize better to unseen data and handle variations more gracefully.
*   **Handling reordering:** NMT intrinsically handles complex word reordering without explicit distortion models.

**Disadvantages of NMT:**
*   **Computational cost:** Training NMT models, especially large Transformer-based ones, requires substantial computational resources (GPUs/TPUs) and large datasets.
*   **Data hungry:** While more robust, NMT still performs best with very large parallel corpora. Performance can degrade significantly on low-resource language pairs.
*   **Lack of interpretability:** NMT models are often considered "black boxes," making it challenging to understand *why* a particular translation was produced or to debug errors systematically.
*   **Hallucinations:** NMT models can sometimes generate plausible-looking but factually incorrect or unfaithful translations (hallucinations), especially when encountering out-of-domain input.
*   **Sensitivity to domain shift:** Performance can degrade when translating texts from domains different from the training data.

## 4. Comparative Analysis and Evolution
The transition from SMT to NMT represents a monumental shift in machine translation, akin to the leap from statistical methods to deep learning in other AI domains. The core differences lie in their fundamental approaches:
*   **Architecture:** SMT relied on discrete, modular components (translation model, language model, decoder), each optimized separately. NMT employs a single, monolithic neural network that learns all aspects of translation end-to-end.
*   **Representation:** SMT used symbolic representations (words, phrases) and statistical counts. NMT uses continuous, dense **vector representations (embeddings)**, allowing for richer semantic understanding.
*   **Context Handling:** SMT's context was limited by n-gram windows and phrase boundaries. NMT, especially with attention and Transformers, can capture long-range dependencies across entire sentences.
*   **Fluency vs. Adequacy:** While SMT often struggled with fluency, NMT excels at generating natural-sounding text. SMT, however, sometimes maintained better **adequacy** (faithfulness to the source) in challenging cases due to its more explicit alignment mechanisms. This is less true with modern NMT which achieves superior adequacy.
*   **Explainability:** SMT offered some degree of interpretability through phrase tables. NMT is notoriously difficult to interpret, though research into attention visualization and model probing is ongoing.
*   **Performance:** NMT has consistently demonstrated superior translation quality across most language pairs, particularly in terms of fluency and handling of complex sentence structures.

The evolution from SMT to NMT was driven by the availability of large datasets, increased computational power, and theoretical breakthroughs in neural network architectures. This shift highlights the power of end-to-end learning in tasks involving complex pattern recognition and generation within natural language.

## 5. Code Example
This Python snippet illustrates a highly simplified conceptual representation of word embeddings and how a "lookup" might work, fundamental to NMT. In a real NMT system, embeddings are learned and updated during training, and multiple layers process these embeddings.

```python
import numpy as np

# A very simplified vocabulary and corresponding embedding matrix
# In NMT, embeddings are typically high-dimensional (e.g., 512, 1024)
# and learned from data.
vocab = {"<PAD>": 0, "<UNK>": 1, "merhaba": 2, "hello": 3, "dünya": 4, "world": 5}
embedding_dim = 4
embedding_matrix = np.array([
    [0.0, 0.0, 0.0, 0.0],  # <PAD>
    [0.1, 0.1, 0.1, 0.1],  # <UNK>
    [0.5, 0.2, 0.3, 0.8],  # merhaba
    [0.6, 0.1, 0.4, 0.7],  # hello
    [0.9, 0.4, 0.1, 0.2],  # dünya
    [0.8, 0.3, 0.2, 0.3]   # world
])

def get_word_embedding(word, vocab_map, emb_matrix):
    """
    Retrieves the embedding vector for a given word.
    Handles unknown words by returning the <UNK> embedding.
    """
    idx = vocab_map.get(word, vocab_map["<UNK>"])
    return emb_matrix[idx]

# Example usage:
word1 = "merhaba"
word2 = "yapayzeka" # Unknown word
word3 = "world"

emb1 = get_word_embedding(word1, vocab, embedding_matrix)
emb2 = get_word_embedding(word2, vocab, embedding_matrix)
emb3 = get_word_embedding(word3, vocab, embedding_matrix)

print(f"Embedding for '{word1}': {emb1}")
print(f"Embedding for '{word2}' (UNK): {emb2}")
print(f"Embedding for '{word3}': {emb3}")

# In a real NMT model, these embeddings would then be fed into
# encoder layers (e.g., Transformer blocks) for further processing.

(End of code example section)
```
## 6. Conclusion
The journey of machine translation from rule-based systems to SMT and ultimately to NMT exemplifies the rapid advancements in artificial intelligence and natural language processing. While SMT provided a robust and data-driven framework for decades, NMT, powered by deep learning and particularly the Transformer architecture, has ushered in an era of unprecedented translation quality, fluency, and efficiency. The shift highlights the profound impact of end-to-end learning and neural representations in capturing the intricate complexities of human language. Despite its challenges, such as the need for vast data and interpretability issues, NMT continues to evolve, with ongoing research focusing on low-resource translation, multilingual models, and better control over translation outputs, pushing the boundaries of what automated language conversion can achieve.

---
<br>

<a name="türkçe-içerik"></a>
## Makine Çevirisi: İstatistiksel ve Nöral Yöntemler

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. İstatistiksel Makine Çevirisi (İMÇ)](#2-İstatistiksel-makine-Çevirisi-İmÇ)
  - [2.1. Temel Prensipler](#21-temel-prensipler)
  - [2.2. Avantajlar ve Dezavantajlar](#22-avantajlar-ve-dezavantajlar)
- [3. Nöral Makine Çevirisi (NMÇ)](#3-nöral-makine-Çevirisi-nmÇ)
  - [3.1. Temel Prensipler](#31-temel-prensipler)
  - [3.2. Avantajlar ve Dezavantajlar](#32-avantajlar-ve-dezavantajlar)
- [4. Karşılaştırmalı Analiz ve Evrim](#4-karşılaştırmalı-analiz-ve-evrim)
- [5. Kod Örneği](#5-kod-Örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
Makine Çevirisi (MÇ), Doğal Dil İşleme (NLP) alanında temel ve sürekli gelişen bir alan olup, metin veya konuşmanın bir doğal dilden diğerine otomatik olarak çevrilmesine odaklanmıştır. Küresel iletişim ve ticareti kolaylaştırmaktan dil engelleri arasında bilgiye erişimi sağlamaya kadar çeşitli alanlarda faydalıdır. Tarihsel olarak, otomatik çeviri arayışı, hesaplamalı dilbilim ve yapay zekadaki gelişmelerle yönlendirilen çeşitli paradigmalar görmüştür. Bu belge, iki baskın yaklaşımı incelemektedir: **İstatistiksel Makine Çevirisi (İMÇ)** ve **Nöral Makine Çevirisi (NMÇ)**. Her birinin altında yatan metodolojileri, çalışma prensiplerini, karşılaştırmalı güçlü ve zayıf yönlerini ve derin öğrenmenin ortaya çıkışıyla birlikte alanda yaşanan dönüştürücü değişimi analiz etmektedir.

## 2. İstatistiksel Makine Çevirisi (İMÇ)
**İstatistiksel Makine Çevirisi (İMÇ)**, 1990'larda MÇ'de baskın paradigma olarak ortaya çıkmış, kural tabanlı sistemlerden büyük paralel korpusları kullanarak çeviri kalıplarını öğrenen yaklaşımlara temel bir geçiş yapmıştır. El yapımı dilbilimsel kurallara güvenmek yerine, İMÇ, belirli bir metnin en olası çevirisini tahmin etmek için istatistiksel modeller kullanır.

### 2.1. Temel Prensipler
İMÇ, kaynak cümle *S* için hedef cümle *T*'nin olasılığını (*P(T|S)*) maksimize etme prensibiyle çalışır. Bu genellikle Bayes Teoremi kullanılarak elde edilir: *P(T|S) = P(S|T) * P(T) / P(S)*. *P(S)*, belirli bir kaynak cümle için sabit olduğundan, sorun *argmax<sub>T</sub> P(S|T) * P(T)* değerini bulmaya indirgenir. Bu formülasyon iki kritik bileşen sunar:
1.  **Çeviri Modeli (P(S|T))**: Bu model, bir kaynak cümle *S*'nin bir hedef cümle *T*'den türetilme olasılığını nicelendirir. Tek tek kelimeleri veya **ifadeleri** çevirmekten ve bunları diller arasında hizalamaktan sorumludur. İlk İMÇ sistemleri, kelimeleri bağımsız olarak çevirmeye odaklanan **kelime tabanlı** idi. Ancak, daha sonraki gelişmeler, bitişik kelime dizilerini (**ifade tabanlı İMÇ (İTİMÇ)**) çeviren ve akıcılığı ve yeniden sıralamayı ele almayı önemli ölçüde geliştiren sistemlere yol açtı. Kaynak ve hedef ifade çiftlerini ve çeviri olasılıklarını içeren ifade tabloları, İTİMÇ'nin merkezindedir.
2.  **Dil Modeli (P(T))**: Bu model, kaynak cümleden bağımsız olarak hedef dildeki hedef cümle *T*'nin akıcılığını ve dilbilgisel doğruluğunu değerlendirir. Bir kelime dizisine, genellikle **n-gram** olasılıklarına (bir kelimenin önceki *n-1* kelime verildiğinde ortaya çıkma olasılığı) dayalı bir olasılık atar. Dil modelinden gelen daha yüksek bir olasılık, daha doğal sesli bir çeviriye işaret eder.

Bu ürünü maksimize eden optimal çeviri *T*'yi bulma süreci, hesaplama açısından yoğun bir arama problemi olan **kod çözme (decoding)** olarak bilinir. İMÇ sistemleri tipik olarak kelime bonusu, bozulma modelleri (kelime yeniden sıralamasını hesaba katmak için) ve sözcüksel ağırlıklandırma gibi ek özellikleri, Minimum Hata Oranı Eğitimi (MERT) gibi tekniklerle optimize edilmiş ağırlıklarla doğrusal olarak birleştirir.

### 2.2. Avantajlar ve Dezavantajlar
**İMÇ'nin Avantajları:**
*   **Veriye dayalı:** İMÇ sistemleri doğrudan veriden öğrenir, bu da onları kapsamlı kural mühendisliğine gerek kalmadan yeni dil çiftlerine ve alanlara uyarlanabilir kılar.
*   **Sağlamlık:** Dilbilgisel varyasyonları ve düzensizlikleri tamamen kural tabanlı sistemlerden daha etkili bir şekilde ele alabilirler.
*   **Açıklanabilirlik:** Modeller, özellikle ifade tabloları, çevirilerin nasıl türetildiğine dair bir dereceye kadar yorumlanabilirlik sunuyordu.

**İMÇ'nin Dezavantajları:**
*   **Özellik mühendisliği:** Özelliklerin ve istatistiksel modellerin dikkatli bir şekilde tasarlanmasını gerektiriyordu.
*   **Veri seyrekliği:** Performans, büyük paralel korpusların mevcudiyetine büyük ölçüde bağlıydı ve nadir kelimeler veya ifadeler önemli zorluklar yaratıyordu.
*   **Yerel optimizasyon:** İMÇ modelleri genellikle bileşenleri bağımsız olarak optimize etti, bu da potansiyel olarak suboptimal küresel çevirilere yol açtı.
*   **Sınırlı bağlam:** N-gram dil modelleri yalnızca kısa menzilli bağımlılıkları yakalar, bu da uzun menzilli tutarlı metni anlama ve üretme yeteneklerini sınırlar.
*   **İfade sınırı sorunları:** İfadelere bölme belirsiz olabilir ve suboptimal seçimlere yol açabilir.

## 3. Nöral Makine Çevirisi (NMÇ)
**Nöral Makine Çevirisi (NMÇ)**, İMÇ'den bir paradigma değişimi olup, yapay sinir ağlarını kullanarak kaynak metinden hedef metne uçtan uca bir eşleme öğrenir. 2010'ların ortalarında tanıtılan NMÇ, performansta İMÇ'yi hızla geride bırakarak modern MÇ sistemlerinde baskın yaklaşım haline geldi.

### 3.1. Temel Prensipler
NMÇ modelleri tipik olarak bir **kodlayıcı-kod çözücü (encoder-decoder)** mimarisine dayanır.
1.  **Kodlayıcı (Encoder)**: Kodlayıcı, kaynak cümleyi işleyerek onu bir **bağlam vektörü** veya **düşünce vektörü** olarak adlandırılan sürekli uzay gösterimine dönüştürür. Bu vektör, tüm kaynak cümlenin anlamsal anlamını kapsamak üzere tasarlanmıştır. İlk NMÇ sistemleri, bağımlılıkları yakalayarak dizileri kelime kelime işlemek için Uzun Kısa Süreli Bellek (LSTM) veya Kapılı Tekrarlayan Birimler (GRU) gibi Tekrarlayan Sinir Ağları (RNN'ler) kullanmıştır.
2.  **Kod Çözücü (Decoder)**: Kod çözücü daha sonra bu bağlam vektörünü alır ve hedef cümleyi kelime kelime üretir. O da tipik olarak, her adımda bağlam vektörüne ve şimdiye kadar üretilen kelimelere dayanarak bir kelime çıkaran bir RNN'dir.

NMÇ performansını önemli ölçüde artıran ve tüm bir cümleyi tek bir sabit boyutlu vektöre kodlama "darboğaz" sorununu ele alan çok önemli bir yenilik, **dikkat mekanizmasıydı**. Dikkat, kod çözücünün her hedef kelimeyi üretirken kaynak cümlenin farklı bölümlerine değişen derecelerde odaklanarak "geri bakmasına" olanak tanır. Bu, modelin uzun cümleleri daha etkili bir şekilde ele almasını ve kaynak ile hedef kelimeler arasında daha doğrudan bağımlılıklar kurmasını sağlar.

NMÇ'deki en derin mimari yenilik, 2017'de tanıtılan **Transformer** modelidir. Transformer'lar, tekrarlayan katmanlardan tamamen vazgeçmiş, bunun yerine giriş dizilerini paralel olarak işlemek için **self-attention mekanizmalarına** güvenmiştir. Bu paralellik, eğitim hızını önemli ölçüde artırdı ve modellerin uzun menzilli bağımlılıkları RNN'lerden daha verimli bir şekilde yakalamasına olanak sağladı. Çoklu başlı dikkat ve ileri beslemeli ağları ile Transformer mimarisi, NMÇ ve diğer birçok NLP görevi için fiili standart haline geldi.

### 3.1. Avantajlar ve Dezavantajlar
**NMÇ'nin Avantajları:**
*   **Uçtan uca öğrenme:** NMÇ modelleri doğrudan ham metinden öğrenir, ayrı, el yapımı bileşenlere (örneğin, ifade tabloları, dil modelleri, hizalama modelleri) olan ihtiyacı ortadan kaldırır. Bu, pipeline'ı basitleştirir ve küresel optimizasyona izin verir.
*   **Akıcılık ve doğruluk:** NMÇ sistemleri, uzun menzilli bağımlılıkları ve karmaşık dilsel nüansları modelleme yetenekleri sayesinde genellikle önemli ölçüde daha akıcı ve doğal sesli çeviriler üretir.
*   **Bağlamsal anlama:** Dağıtılmış gösterimler (**gömülü temsiller** veya **embeddingler**) ve dikkat mekanizmaları aracılığıyla NMÇ, kelimelerin bağlamını daha etkili bir şekilde yakalayabilir.
*   **Genelleme:** Görülmeyen verilere daha iyi genelleme yapabilir ve varyasyonları daha zarif bir şekilde ele alabilirler.
*   **Yeniden sıralamayı ele alma:** NMÇ, karmaşık kelime yeniden sıralamasını açık bozulma modelleri olmadan içsel olarak ele alır.

**NMÇ'nin Dezavantajları:**
*   **Hesaplama maliyeti:** NMÇ modellerini, özellikle büyük Transformer tabanlı olanları eğitmek, önemli hesaplama kaynakları (GPU/TPU) ve büyük veri kümeleri gerektirir.
*   **Veri açlığı:** Daha sağlam olmasına rağmen, NMÇ hala çok büyük paralel korpuslarla en iyi performansı gösterir. Düşük kaynaklı dil çiftlerinde performans önemli ölçüde düşebilir.
*   **Yorumlanabilirlik eksikliği:** NMÇ modelleri genellikle "kara kutu" olarak kabul edilir, bu da belirli bir çevirinin *neden* üretildiğini anlamayı veya hataları sistematik olarak ayıklamayı zorlaştırır.
*   **Halüsinasyonlar:** NMÇ modelleri bazen, özellikle alan dışı girdilerle karşılaştığında, olası görünen ancak aslında yanlış veya aslına sadık olmayan çeviriler (halüsinasyonlar) üretebilir.
*   **Alan kaymasına duyarlılık:** Eğitim verilerinden farklı alanlardaki metinleri çevirirken performans düşebilir.

## 4. Karşılaştırmalı Analiz ve Evrim
İMÇ'den NMÇ'ye geçiş, makine çevirisinde anıtsal bir değişimi temsil eder, diğer yapay zeka alanlarındaki istatistiksel yöntemlerden derin öğrenmeye sıçramaya benzer. Temel farklılıklar, temel yaklaşımlarında yatmaktadır:
*   **Mimari:** İMÇ, her biri ayrı ayrı optimize edilmiş ayrı, modüler bileşenlere (çeviri modeli, dil modeli, kod çözücü) güveniyordu. NMÇ, çevirinin tüm yönlerini uçtan uca öğrenen tek, monolitik bir sinir ağı kullanır.
*   **Temsil:** İMÇ, sembolik temsiller (kelimeler, ifadeler) ve istatistiksel sayımlar kullanıyordu. NMÇ, daha zengin anlamsal anlamaya olanak tanıyan sürekli, yoğun **vektör temsilleri (gömülü temsiller)** kullanır.
*   **Bağlam İşleme:** İMÇ'nin bağlamı n-gram pencereleri ve ifade sınırlarıyla sınırlıydı. NMÇ, özellikle dikkat ve Transformer'lar ile tüm cümleler boyunca uzun menzilli bağımlılıkları yakalayabilir.
*   **Akıcılık ve Yeterlilik:** İMÇ genellikle akıcılıkla mücadele ederken, NMÇ doğal sesli metin üretmede üstündür. Ancak İMÇ, daha açık hizalama mekanizmaları nedeniyle zorlu durumlarda bazen daha iyi **yeterliliği** (kaynağa sadakat) koruyordu. Modern NMÇ'de bu durum, üstün yeterlilik elde edilmesiyle daha az doğrudur.
*   **Yorumlanabilirlik:** İMÇ, ifade tabloları aracılığıyla bir dereceye kadar yorumlanabilirlik sunuyordu. NMÇ'yi yorumlamak zor olsa da, dikkat görselleştirmesi ve model incelemesi üzerine araştırmalar devam etmektedir.
*   **Performans:** NMÇ, çoğu dil çiftinde, özellikle akıcılık ve karmaşık cümle yapılarını ele alma açısından, sürekli olarak üstün çeviri kalitesi göstermiştir.

İMÇ'den NMÇ'ye evrim, büyük veri kümelerinin mevcudiyeti, artan hesaplama gücü ve sinir ağı mimarilerindeki teorik atılımlarla yönlendirilmiştir. Bu değişim, doğal dil içindeki karmaşık kalıp tanıma ve üretim içeren görevlerde uçtan uca öğrenmenin derin etkisini vurgulamaktadır.

## 5. Kod Örneği
Bu Python kodu, kelime gömülü temsillerinin oldukça basitleştirilmiş kavramsal bir temsilini ve NMÇ için temel olan bir "arama" işleminin nasıl çalışabileceğini göstermektedir. Gerçek bir NMÇ sisteminde, gömülü temsiller eğitim sırasında öğrenilir ve güncellenir ve birden çok katman bu gömülü temsilleri işler.

```python
import numpy as np

# Çok basitleştirilmiş bir kelime dağarcığı ve karşılık gelen gömülü temsil matrisi
# NMÇ'de, gömülü temsiller tipik olarak yüksek boyutludur (örneğin, 512, 1024)
# ve veriden öğrenilir.
vocab = {"<PAD>": 0, "<UNK>": 1, "merhaba": 2, "hello": 3, "dünya": 4, "world": 5}
embedding_dim = 4
embedding_matrix = np.array([
    [0.0, 0.0, 0.0, 0.0],  # <PAD> (Doldurma)
    [0.1, 0.1, 0.1, 0.1],  # <UNK> (Bilinmeyen kelime)
    [0.5, 0.2, 0.3, 0.8],  # merhaba
    [0.6, 0.1, 0.4, 0.7],  # hello
    [0.9, 0.4, 0.1, 0.2],  # dünya
    [0.8, 0.3, 0.2, 0.3]   # world
])

def get_word_embedding(word, vocab_map, emb_matrix):
    """
    Belirli bir kelimenin gömülü temsil vektörünü alır.
    Bilinmeyen kelimeler için <UNK> gömülü temsilini döndürür.
    """
    idx = vocab_map.get(word, vocab_map["<UNK>"])
    return emb_matrix[idx]

# Örnek kullanım:
word1 = "merhaba"
word2 = "yapayzeka" # Bilinmeyen kelime
word3 = "world"

emb1 = get_word_embedding(word1, vocab, embedding_matrix)
emb2 = get_word_embedding(word2, vocab, embedding_matrix)
emb3 = get_word_embedding(word3, vocab, embedding_matrix)

print(f"'{word1}' için gömülü temsil: {emb1}")
print(f"'{word2}' (UNK) için gömülü temsil: {emb2}")
print(f"'{word3}' için gömülü temsil: {emb3}")

# Gerçek bir NMÇ modelinde, bu gömülü temsiller daha sonra
# daha fazla işleme için kodlayıcı katmanlarına (örneğin, Transformer bloklarına) beslenir.

(Kod örneği bölümünün sonu)
```
## 6. Sonuç
Makine çevirisinin kural tabanlı sistemlerden İMÇ'ye ve nihayetinde NMÇ'ye uzanan yolculuğu, yapay zeka ve doğal dil işlemedeki hızlı ilerlemeleri örneklendirmektedir. İMÇ, onlarca yıl boyunca sağlam ve veriye dayalı bir çerçeve sunarken, derin öğrenme ve özellikle Transformer mimarisi tarafından desteklenen NMÇ, eşi benzeri görülmemiş çeviri kalitesi, akıcılığı ve verimliliği çağına girmiştir. Bu değişim, insan dilinin karmaşık inceliklerini yakalamada uçtan uca öğrenmenin ve nöral temsillerin derin etkisini vurgulamaktadır. Büyük veri ihtiyacı ve yorumlanabilirlik sorunları gibi zorluklarına rağmen, NMÇ gelişmeye devam etmekte, düşük kaynaklı çeviri, çok dilli modeller ve çeviri çıktıları üzerinde daha iyi kontrol konularına odaklanan araştırmalarla otomatik dil dönüşümünün sınırlarını zorlamaktadır.




