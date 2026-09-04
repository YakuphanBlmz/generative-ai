# Word2Vec: CBOW vs. Skip-Gram Architectures

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Word Embeddings and Contextual Semantics](#2-word-embeddings-and-contextual-semantics)
- [3. CBOW (Continuous Bag-of-Words) Architecture](#3-cbow-continuous-bag-of-words-architecture)
    - [3.1. Mechanism and Structure](#31-mechanism-and-structure)
    - [3.2. Advantages and Disadvantages](#32-advantages-and-disadvantages)
- [4. Skip-Gram Architecture](#4-skip-gram-architecture)
    - [4.1. Mechanism and Structure](#41-mechanism-and-structure)
    - [4.2. Advantages and Disadvantages](#42-advantages-and-disadvantages)
- [5. Comparative Analysis and Practical Considerations](#5-comparative-analysis-and-practical-considerations)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

### 1. Introduction
The advent of **Word2Vec** by Mikolov et al. in 2013 marked a significant paradigm shift in Natural Language Processing (NLP), moving from traditional sparse representations of words (like one-hot encoding) to dense, continuous vector representations known as **word embeddings**. These embeddings are designed to capture semantic and syntactic relationships between words, where words with similar meanings or contexts are positioned closer in the vector space. Word2Vec comprises two primary architectures: **Continuous Bag-of-Words (CBOW)** and **Skip-Gram**. While both aim to generate high-quality word embeddings, they achieve this through inverse predictive tasks, leading to distinct performance characteristics and suitability for different applications. This document provides an in-depth exploration of these two architectures, detailing their underlying mechanisms, comparative advantages, and practical implications.

### 2. Word Embeddings and Contextual Semantics
At the core of Word2Vec lies the distributional hypothesis, which posits that words appearing in similar contexts tend to have similar meanings. **Word embeddings** are low-dimensional vector representations that numerically encode this contextual information. Unlike one-hot encoding, which treats each word as an independent entity and results in high-dimensional, sparse vectors, word embeddings are dense and capture intricate semantic relationships. For instance, in a well-trained embedding space, the vector for "king" minus "man" plus "woman" might approximate the vector for "queen," demonstrating the ability of these embeddings to capture analogies and semantic nuances. The success of Word2Vec stems from its ability to learn these embeddings efficiently from large corpora using shallow neural networks.

### 3. CBOW (Continuous Bag-of-Words) Architecture
The CBOW model is designed to predict a **target word** based on its surrounding **context words**. This architecture operates under the assumption that the meaning of a word can be inferred from the words that frequently appear around it.

#### 3.1. Mechanism and Structure
In the CBOW model, the input layer receives multiple one-hot encoded vectors representing the context words within a fixed-size window around the target word. These input vectors are then projected onto a shared hidden layer (the projection layer), which has a dimensionality equal to the desired size of the word embeddings. This projection involves multiplying each one-hot vector by a weight matrix (W_in), where each row corresponds to the embedding of a word in the vocabulary.

Crucially, the CBOW architecture averages the vectors of the context words in the hidden layer before passing them to the output layer. This averaging operation simplifies the model and aggregates the contextual information. The output layer is typically a softmax layer, which predicts the probability distribution over the entire vocabulary for the target word. The objective function is to maximize the probability of the actual target word given the context words. Training involves adjusting the weight matrices (W_in and W_out) through backpropagation to minimize the prediction error. The final word embeddings are usually extracted from the W_in matrix.

#### 3.2. Advantages and Disadvantages
**Advantages:**
*   **Computational Efficiency:** CBOW tends to be faster to train than Skip-Gram, especially with large datasets. This is largely due to the averaging of context vectors, which simplifies the hidden layer calculations.
*   **Good for Frequent Words:** It performs well for common words, as their contexts are generally well-represented in the corpus.

**Disadvantages:**
*   **Less Effective for Rare Words:** Since it averages the context, CBOW might struggle to represent rare words accurately, as their specific contextual nuances can be diluted or overpowered by more common words in the context window.
*   **Potentially Smoothed Semantics:** The averaging of context vectors can lead to a "smoothing" effect, potentially losing some fine-grained semantic distinctions present in the individual context words.

### 4. Skip-Gram Architecture
In contrast to CBOW, the Skip-Gram model is designed to predict the **surrounding context words** given a **target word**. This inverse task often leads to different strengths in embedding quality.

#### 4.1. Mechanism and Structure
The Skip-Gram model takes a single one-hot encoded vector of the target word as input. This input is then multiplied by a weight matrix (W_in) to project it onto the hidden layer, which again represents the word embedding for the target word. From this single hidden layer vector, the model attempts to predict multiple context words within a defined window.

The output layer for Skip-Gram is more complex than CBOW's. Instead of predicting one target word, it effectively has multiple output heads (or one output head applied multiple times), each predicting one context word. Each prediction involves another weight matrix (W_out) and a softmax function, calculating the probability of each word in the vocabulary being a context word for the given target word. The objective is to maximize the sum of log probabilities of the actual context words given the input word. Similar to CBOW, backpropagation is used to update the weight matrices, and the embeddings are typically derived from W_in.

#### 4.2. Advantages and Disadvantages
**Advantages:**
*   **Excellent for Rare Words:** Skip-Gram often produces better embeddings for rare words. Because it predicts context from a single target word, it can better leverage the limited occurrences of rare words and capture their specific semantic contexts.
*   **Captures More Semantic Nuances:** By predicting multiple context words independently, Skip-Gram can capture more fine-grained semantic and syntactic relationships, especially when the target word has a rich and varied context.
*   **Better for Analogies:** Often performs superior in tasks involving word analogies due to its ability to capture subtle semantic regularities.

**Disadvantages:**
*   **Slower to Train:** Skip-Gram is generally slower to train than CBOW, especially on very large datasets, because it makes multiple predictions (one for each context word) for every input word, increasing the computational burden on the output layer.
*   **Computationally Intensive Output Layer:** The output layer requires calculating probabilities for all words in the vocabulary for each context word, which can be computationally expensive. Techniques like Negative Sampling or Hierarchical Softmax are often employed to mitigate this.

### 5. Comparative Analysis and Practical Considerations
The choice between CBOW and Skip-Gram often depends on the specific task, dataset characteristics, and available computational resources.

| Feature                 | CBOW (Continuous Bag-of-Words)                                | Skip-Gram                                                         |
| :---------------------- | :------------------------------------------------------------ | :---------------------------------------------------------------- |
| **Input**               | Context words                                                 | Target word                                                       |
| **Output**              | Target word                                                   | Context words                                                     |
| **Training Speed**      | Faster, especially with large datasets                        | Slower, due to multiple context predictions                       |
| **Rare Words**          | Less effective; context averaging can dilute information      | More effective; better at capturing nuances of infrequent words    |
| **Semantic Quality**    | Good for frequent words, can be "smoother"                    | Excellent for rare words, captures more fine-grained semantics      |
| **Analogy Tasks**       | Moderate performance                                          | Superior performance                                              |
| **Computational Load**  | Lower, especially on the output layer                         | Higher, particularly for output layer without optimizations         |

In practical scenarios:
*   **CBOW** is often preferred when dealing with very large datasets and when the primary goal is to obtain reasonably good embeddings quickly, especially if the vocabulary primarily consists of frequent words. It's a good choice for applications where computational efficiency is paramount.
*   **Skip-Gram** is typically recommended when higher quality embeddings are critical, particularly for tasks involving semantic analogies or when the dataset contains many rare words whose precise semantic capture is important. It tends to perform better in capturing a broader range of semantic relationships, albeit at a higher computational cost.

Both architectures benefit significantly from optimizations like **Negative Sampling** and **Hierarchical Softmax**, which reduce the computational burden of the softmax layer, making training more feasible for large vocabularies.

### 6. Code Example
This conceptual Python snippet illustrates how Word2Vec (specifically Skip-Gram, by default) can be used with the `gensim` library.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Ensure you have NLTK data downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Sample text corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "I love to eat apples and bananas.",
    "Dogs and cats are common pets.",
    "The fox is a clever animal."
]

# Tokenize sentences into words
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train a Skip-Gram Word2Vec model
# vector_size: dimensionality of the word vectors
# window: maximum distance between the current and predicted word within a sentence
# min_count: ignores all words with total frequency lower than this
# sg: 1 for Skip-Gram, 0 for CBOW
model_skipgram = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=10
)

# Train a CBOW Word2Vec model
model_cbow = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,
    epochs=10
)

# Get vector for a word (e.g., "fox")
fox_vector_sg = model_skipgram.wv['fox']
fox_vector_cbow = model_cbow.wv['fox']

print("Fox vector (Skip-Gram):", fox_vector_sg[:5]) # print first 5 elements
print("Fox vector (CBOW):", fox_vector_cbow[:5]) # print first 5 elements

# Find most similar words
print("\nWords most similar to 'fox' (Skip-Gram):", model_skipgram.wv.most_similar('fox', topn=3))
print("Words most similar to 'fox' (CBOW):", model_cbow.wv.most_similar('fox', topn=3))

(End of code example section)
```

### 7. Conclusion
Word2Vec's CBOW and Skip-Gram architectures represent foundational contributions to the field of word embeddings, enabling machines to understand and process human language with unprecedented efficacy. While CBOW efficiently predicts a target word from its context, making it suitable for larger datasets and frequent words, Skip-Gram excels at capturing nuanced semantics and handling rare words by predicting context from a target word. The choice between these two powerful models hinges on a careful evaluation of dataset characteristics, computational constraints, and the specific requirements of the downstream NLP task. Their enduring impact underscores the importance of contextual understanding in developing robust language models.

---
<br>

<a name="türkçe-içerik"></a>
## Word2Vec: CBOW ve Skip-Gram Mimarileri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Kelime Gömülmeleri ve Bağlamsal Anlam Bilimi](#2-kelime-gömülmeleri-ve-bağlamsal-anlam-bilimi)
- [3. CBOW (Sürekli Kelime Torbası) Mimarisi](#3-cbow-sürekli-kelime-torbası-mimarisi)
    - [3.1. Mekanizma ve Yapı](#31-mekanizma-ve-yapı)
    - [3.2. Avantajlar ve Dezavantajlar](#32-avantajlar-ve-dezavantajlar)
- [4. Skip-Gram Mimarisi](#4-skip-gram-mimarisi)
    - [4.1. Mekanizma ve Yapı](#41-mekanizma-ve-yapı)
    - [4.2. Avantajlar ve Dezavantajlar](#42-avantajlar-ve-dezavantajlar)
- [5. Karşılaştırmalı Analiz ve Pratik Değerlendirmeler](#5-karşılaştırmalı-analiz-ve-pratik-değerlendirmeler)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

### 1. Giriş
Mikolov ve arkadaşları tarafından 2013 yılında tanıtılan **Word2Vec**, Doğal Dil İşleme (NLP) alanında önemli bir paradigma değişimine işaret etti. Bu değişim, kelimelerin geleneksel seyrek gösterimlerinden (örneğin, tek-sıcak kodlama) yoğun, sürekli vektör gösterimlerine, yani **kelime gömülmelerine** (word embeddings) geçişi sağladı. Bu gömülmeler, kelimeler arasındaki anlamsal ve sentaktik ilişkileri yakalamak üzere tasarlanmıştır; benzer anlamlara veya bağlamlara sahip kelimeler, vektör uzayında birbirine daha yakın konumlanır. Word2Vec iki temel mimariden oluşur: **Sürekli Kelime Torbası (CBOW)** ve **Skip-Gram**. Her ikisi de yüksek kaliteli kelime gömülmeleri üretmeyi hedeflerken, bunu ters tahmin görevleri aracılığıyla gerçekleştirirler, bu da farklı performans özellikleri ve farklı uygulamalar için uygunluklarına yol açar. Bu belge, bu iki mimarinin altında yatan mekanizmaları, karşılaştırmalı avantajlarını ve pratik çıkarımlarını detaylandıran kapsamlı bir inceleme sunmaktadır.

### 2. Kelime Gömülmeleri ve Bağlamsal Anlam Bilimi
Word2Vec'in temelinde, benzer bağlamlarda ortaya çıkan kelimelerin benzer anlamlara sahip olma eğiliminde olduğunu öne süren dağılımsal hipotez yatar. **Kelime gömülmeleri**, bu bağlamsal bilgiyi sayısal olarak kodlayan düşük boyutlu vektör gösterimleridir. Her kelimeyi bağımsız bir varlık olarak ele alan ve yüksek boyutlu, seyrek vektörler üreten tek-sıcak kodlamanın aksine, kelime gömülmeleri yoğundur ve karmaşık anlamsal ilişkileri yakalar. Örneğin, iyi eğitilmiş bir gömülme uzayında, "kral" eksi "erkek" artı "kadın" vektörü, "kraliçe" vektörüne yaklaşabilir, bu da bu gömülmelerin analojileri ve anlamsal incelikleri yakalama yeteneğini gösterir. Word2Vec'in başarısı, bu gömülmeleri büyük metin veri kümelerinden (corpus) sığ sinir ağları kullanarak verimli bir şekilde öğrenebilmesinden kaynaklanmaktadır.

### 3. CBOW (Sürekli Kelime Torbası) Mimarisi
CBOW modeli, bir **hedef kelimeyi** çevresindeki **bağlam kelimelerine** dayanarak tahmin etmek üzere tasarlanmıştır. Bu mimari, bir kelimenin anlamının, etrafında sıkça görünen kelimelerden çıkarılabileceği varsayımıyla çalışır.

#### 3.1. Mekanizma ve Yapı
CBOW modelinde, giriş katmanı, hedef kelimenin etrafındaki sabit boyutlu bir pencere içindeki bağlam kelimelerini temsil eden birden çok tek-sıcak kodlanmış vektör alır. Bu giriş vektörleri daha sonra, istenen kelime gömülmelerinin boyutuna eşit bir boyuta sahip olan paylaşımlı bir gizli katmana (projeksiyon katmanı) yansıtılır. Bu projeksiyon, her bir tek-sıcak vektörün bir ağırlık matrisiyle (W_in) çarpılmasını içerir; burada her satır, kelime dağarcığındaki bir kelimenin gömülmesine karşılık gelir.

Önemli olarak, CBOW mimarisi, gizli katmandaki bağlam kelimelerinin vektörlerini çıktı katmanına geçirmeden önce ortalamasını alır. Bu ortalama işlemi, modeli basitleştirir ve bağlamsal bilgiyi toplar. Çıktı katmanı tipik olarak bir softmax katmanıdır ve hedef kelime için tüm kelime dağarcığı üzerindeki olasılık dağılımını tahmin eder. Amaç fonksiyonu, bağlam kelimeleri verildiğinde gerçek hedef kelimenin olasılığını maksimize etmektir. Eğitim, tahmin hatasını minimize etmek için geri yayılım (backpropagation) aracılığıyla ağırlık matrislerini (W_in ve W_out) ayarlamayı içerir. Nihai kelime gömülmeleri genellikle W_in matrisinden çıkarılır.

#### 3.2. Avantajlar ve Dezavantajlar
**Avantajları:**
*   **Hesaplama Verimliliği:** CBOW, özellikle büyük veri kümeleriyle çalışırken Skip-Gram'dan daha hızlı eğitilme eğilimindedir. Bu, büyük ölçüde bağlam vektörlerinin ortalamasının alınmasından kaynaklanır ve bu, gizli katman hesaplamalarını basitleştirir.
*   **Sık Kelimeler İçin İyi:** Yaygın kelimeler için iyi performans gösterir, çünkü bağlamları genellikle metin içinde iyi temsil edilir.

**Dezavantajları:**
*   **Nadir Kelimeler İçin Daha Az Etkili:** Bağlamı ortalaması nedeniyle, CBOW nadir kelimeleri doğru bir şekilde temsil etmekte zorlanabilir, çünkü özel bağlamsal incelikleri, bağlam penceresindeki daha yaygın kelimeler tarafından seyreltilebilir veya bastırılabilir.
*   **Potansiyel Olarak Yumuşatılmış Anlamlar:** Bağlam vektörlerinin ortalaması, bireysel bağlam kelimelerinde bulunan bazı ince anlamsal ayrımların potansiyel olarak kaybolmasına yol açan bir "yumuşatma" etkisine neden olabilir.

### 4. Skip-Gram Mimarisi
CBOW'un aksine, Skip-Gram modeli, bir **hedef kelime** verildiğinde **çevresindeki bağlam kelimelerini** tahmin etmek üzere tasarlanmıştır. Bu ters görev genellikle gömülme kalitesinde farklı güçlü yönlere yol açar.

#### 4.1. Mekanizma ve Yapı
Skip-Gram modeli, hedef kelimenin tek-sıcak kodlanmış vektörünü girdi olarak alır. Bu girdi daha sonra, hedef kelime için kelime gömülmesini temsil eden gizli katmana yansıtmak üzere bir ağırlık matrisi (W_in) ile çarpılır. Bu tek gizli katman vektöründen, model tanımlanmış bir pencere içindeki birden çok bağlam kelimesini tahmin etmeye çalışır.

Skip-Gram için çıktı katmanı CBOW'dan daha karmaşıktır. Bir hedef kelimeyi tahmin etmek yerine, her biri bir bağlam kelimesini tahmin eden birden çok çıktı kafasına (veya bir çıktı kafasının birden çok kez uygulanmasına) sahiptir. Her tahmin, başka bir ağırlık matrisi (W_out) ve bir softmax fonksiyonu içerir, bu da verilen hedef kelime için kelime dağarcığındaki her kelimenin bir bağlam kelimesi olma olasılığını hesaplar. Amaç, giriş kelimesi verildiğinde gerçek bağlam kelimelerinin log olasılıklarının toplamını maksimize etmektir. CBOW'a benzer şekilde, geri yayılım ağırlık matrislerini güncellemek için kullanılır ve gömülmeler tipik olarak W_in'den türetilir.

#### 4.2. Avantajlar ve Dezavantajlar
**Avantajları:**
*   **Nadir Kelimeler İçin Mükemmel:** Skip-Gram genellikle nadir kelimeler için daha iyi gömülmeler üretir. Bağlamı tek bir hedef kelimeden tahmin ettiği için, nadir kelimelerin sınırlı oluşumlarından daha iyi yararlanabilir ve bunların özel anlamsal bağlamlarını yakalayabilir.
*   **Daha Fazla Anlamsal Nüansı Yakalar:** Birden çok bağlam kelimesini bağımsız olarak tahmin ederek, Skip-Gram, özellikle hedef kelimenin zengin ve çeşitli bir bağlamı olduğunda, daha ince anlamsal ve sentaktik ilişkileri yakalayabilir.
*   **Analojiler İçin Daha İyi:** Anlamsal düzenlilikleri yakalama yeteneği nedeniyle kelime analojileri içeren görevlerde genellikle üstün performans gösterir.

**Dezavantajları:**
*   **Daha Yavaş Eğitim:** Skip-Gram, özellikle çok büyük veri kümelerinde, CBOW'dan daha yavaş eğitilir, çünkü her giriş kelimesi için birden çok tahmin (her bağlam kelimesi için bir tane) yapar ve çıktı katmanındaki hesaplama yükünü artırır.
*   **Hesaplama Yoğun Çıktı Katmanı:** Çıktı katmanı, her bağlam kelimesi için kelime dağarcığındaki tüm kelimeler için olasılıkların hesaplanmasını gerektirir, bu da hesaplama açısından pahalı olabilir. Bunu hafifletmek için genellikle Negatif Örnekleme (Negative Sampling) veya Hiyerarşik Softmax (Hierarchical Softmax) gibi teknikler kullanılır.

### 5. Karşılaştırmalı Analiz ve Pratik Değerlendirmeler
CBOW ve Skip-Gram arasındaki seçim genellikle belirli göreve, veri kümesi özelliklerine ve mevcut hesaplama kaynaklarına bağlıdır.

| Özellik                 | CBOW (Sürekli Kelime Torbası)                                  | Skip-Gram                                                           |
| :---------------------- | :------------------------------------------------------------- | :------------------------------------------------------------------ |
| **Girdi**               | Bağlam kelimeleri                                              | Hedef kelime                                                        |
| **Çıktı**               | Hedef kelime                                                   | Bağlam kelimeleri                                                   |
| **Eğitim Hızı**         | Daha hızlı, özellikle büyük veri kümeleriyle                   | Daha yavaş, çoklu bağlam tahminleri nedeniyle                       |
| **Nadir Kelimeler**     | Daha az etkili; bağlam ortalaması bilgiyi seyreltebilir         | Daha etkili; seyrek kelimelerin inceliklerini yakalamada daha iyi   |
| **Anlamsal Kalite**     | Sık kelimeler için iyi, daha "pürüzsüz" olabilir               | Nadir kelimeler için mükemmel, daha ince anlamsal bilgileri yakalar |
| **Analoji Görevleri**   | Orta performans                                                | Üstün performans                                                    |
| **Hesaplama Yükü**      | Daha düşük, özellikle çıktı katmanında                          | Daha yüksek, özellikle optimizasyonlar olmadan çıktı katmanı için    |

Pratik senaryolarda:
*   **CBOW**, çok büyük veri kümeleriyle uğraşırken ve temel amaç, özellikle kelime dağarcığı çoğunlukla sık kelimelerden oluşuyorsa, makul derecede iyi gömülmeleri hızlı bir şekilde elde etmek olduğunda sıklıkla tercih edilir. Hesaplama verimliliğinin öncelikli olduğu uygulamalar için iyi bir seçimdir.
*   **Skip-Gram**, daha yüksek kaliteli gömülmelerin kritik olduğu durumlarda, özellikle anlamsal analojiler içeren görevler veya veri setinde hassas anlamsal yakalamanın önemli olduğu birçok nadir kelime bulunduğunda genellikle önerilir. Daha yüksek bir hesaplama maliyetiyle birlikte, daha geniş bir anlamsal ilişki yelpazesini yakalamada daha iyi performans gösterme eğilimindedir.

Her iki mimari de **Negatif Örnekleme (Negative Sampling)** ve **Hiyerarşik Softmax (Hierarchical Softmax)** gibi optimizasyonlardan önemli ölçüde faydalanır; bu optimizasyonlar softmax katmanının hesaplama yükünü azaltarak, büyük kelime dağarcıkları için eğitimi daha uygulanabilir hale getirir.

### 6. Kod Örneği
Bu kavramsal Python kodu, `gensim` kütüphanesiyle Word2Vec'in (varsayılan olarak Skip-Gram) nasıl kullanılabileceğini göstermektedir.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# NLTK verilerinin indirildiğinden emin olun
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Örnek metin veri kümesi
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "I love to eat apples and bananas.",
    "Dogs and cats are common pets.",
    "The fox is a clever animal."
]

# Cümleleri kelimelere ayırma (tokenization)
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Bir Skip-Gram Word2Vec modeli eğitme
# vector_size: kelime vektörlerinin boyutu
# window: cümle içinde mevcut ve tahmin edilen kelime arasındaki maksimum mesafe
# min_count: toplam frekansı bundan daha düşük olan tüm kelimeleri göz ardı eder
# sg: Skip-Gram için 1, CBOW için 0
model_skipgram = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=10
)

# Bir CBOW Word2Vec modeli eğitme
model_cbow = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,
    epochs=10
)

# Bir kelimenin vektörünü alma (örn. "fox")
fox_vector_sg = model_skipgram.wv['fox']
fox_vector_cbow = model_cbow.wv['fox']

print("Fox vektörü (Skip-Gram):", fox_vector_sg[:5]) # ilk 5 elemanı yazdır
print("Fox vektörü (CBOW):", fox_vector_cbow[:5]) # ilk 5 elemanı yazdır

# En benzer kelimeleri bulma
print("\n'fox' kelimesine en benzer kelimeler (Skip-Gram):", model_skipgram.wv.most_similar('fox', topn=3))
print("'fox' kelimesine en benzer kelimeler (CBOW):", model_cbow.wv.most_similar('fox', topn=3))

(Kod örneği bölümünün sonu)
```

### 7. Sonuç
Word2Vec'in CBOW ve Skip-Gram mimarileri, kelime gömülmeleri alanına temel katkılar sağlayarak makinelerin insan dilini benzeri görülmemiş bir etkinlikle anlamasını ve işlemesini mümkün kılmıştır. CBOW, bağlamından bir hedef kelimeyi verimli bir şekilde tahmin ederek büyük veri kümeleri ve sık kelimeler için uygunken, Skip-Gram, bir hedef kelimeden bağlamı tahmin ederek nüanslı anlamları yakalamada ve nadir kelimeleri işlemekte üstündür. Bu iki güçlü model arasındaki seçim, veri kümesi özelliklerinin, hesaplama kısıtlamalarının ve aşağı akış NLP görevinin özel gereksinimlerinin dikkatli bir şekilde değerlendirilmesine bağlıdır. Kalıcı etkileri, sağlam dil modelleri geliştirmede bağlamsal anlayışın önemini vurgulamaktadır.

