# The Role of Embeddings in Natural Language Processing

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. What are Embeddings?](#2-what-are-embeddings)
  - [2.1. From Discrete to Dense Representations](#21-from-discrete-to-dense-representations)
  - [2.2. The Semantic Space](#22-the-semantic-space)
- [3. Types of Embeddings](#3-types-of-embeddings)
  - [3.1. Count-Based Embeddings](#31-count-based-embeddings)
  - [3.2. Predictive Embeddings](#32-predictive-embeddings)
    - [3.2.1. Word2Vec (CBOW & Skip-gram)](#321-word2vec-cbow--skip-gram)
    - [3.2.2. GloVe (Global Vectors for Word Representation)](#322-glove-global-vectors-for-word-representation)
  - [3.3. Contextual Embeddings](#33-contextual-embeddings)
    - [3.3.1. ELMo (Embeddings from Language Models)](#331-elmo-embeddings-from-language-models)
    - [3.3.2. BERT (Bidirectional Encoder Representations from Transformers)](#332-bert-bidirectional-encoder-representations-from-transformers)
- [4. Applications of Embeddings in NLP](#4-applications-of-embeddings-in-nlp)
  - [4.1. Semantic Search and Information Retrieval](#41-semantic-search-and-information-retrieval)
  - [4.2. Text Classification and Sentiment Analysis](#42-text-classification-and-sentiment-analysis)
  - [4.3. Machine Translation and Cross-Lingual Tasks](#43-machine-translation-and-cross-lingual-tasks)
  - [4.4. Named Entity Recognition (NER)](#44-named-entity-recognition-ner)
  - [4.5. Question Answering](#45-question-answering)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

Natural Language Processing (NLP) stands at the forefront of artificial intelligence research, striving to enable computers to understand, interpret, and generate human language in a meaningful way. A fundamental challenge in NLP has always been how to represent words, phrases, and entire documents in a format that machines can effectively process. Traditional methods, such as **one-hot encoding**, often suffered from the curse of dimensionality and failed to capture the semantic relationships between words. The advent of **embeddings** has revolutionized this aspect, providing dense, continuous vector representations that encapsulate rich semantic and syntactic information.

This document delves into the crucial role of embeddings in modern NLP systems. We will explore their conceptual foundation, trace their evolution from simple count-based models to sophisticated contextualized representations, and examine their profound impact on various NLP tasks, from semantic search to machine translation. Understanding embeddings is paramount for anyone seeking to comprehend the capabilities and advancements within the field of Generative AI and NLP.

<a name="2-what-are-embeddings"></a>
## 2. What are Embeddings?

At its core, an **embedding** is a low-dimensional, dense vector representation of discrete objects, such as words, phrases, or even entire documents, in a continuous vector space. Unlike traditional discrete representations, where each word is treated as an independent entity (e.g., in one-hot encoding), embeddings aim to map similar objects closer to each other in this vector space.

<a name="21-from-discrete-to-dense-representations"></a>
### 2.1. From Discrete to Dense Representations

Historically, words were often represented using **sparse** methods. For instance, **one-hot encoding** assigns a unique binary vector to each word in a vocabulary, where only one element is '1' and all others are '0'. While simple, this approach has several drawbacks:
1.  **High Dimensionality:** For a vocabulary of V words, each vector has V dimensions, leading to extremely large vectors for substantial vocabularies.
2.  **Lack of Semantic Meaning:** All words are equidistant from each other, meaning there is no inherent information about their relationships or similarities. "King" and "Queen" are as different as "King" and "Banana" in this representation.
3.  **Sparsity:** Most elements in the vector are zero, which is computationally inefficient.

Embeddings overcome these limitations by mapping words to fixed-size, real-valued vectors, typically with dimensions ranging from 50 to 300. These are **dense** representations, meaning most elements in the vector are non-zero.

<a name="22-the-semantic-space"></a>
### 2.2. The Semantic Space

The true power of embeddings lies in their ability to capture semantic and syntactic properties. Words that are semantically or syntactically similar tend to be located closer together in the vector space. For example, in a well-trained embedding space, the vector for "king" might be close to "queen," "prince," and "royal." Furthermore, fascinating **vector arithmetic** properties emerge, such as `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`. This suggests that embeddings learn analogies and relationships, encoding complex linguistic patterns directly into numerical representations. This continuous space allows for nuanced comparisons and operations that are impossible with discrete representations.

<a name="3-types-of-embeddings"></a>
## 3. Types of Embeddings

The evolution of embeddings has seen several significant paradigms, each building upon its predecessors to capture more sophisticated linguistic information.

<a name="31-count-based-embeddings"></a>
### 3.1. Count-Based Embeddings

Early attempts to create dense representations involved statistical analysis of word co-occurrences.
*   **Term Frequency-Inverse Document Frequency (TF-IDF):** While not strictly an embedding, TF-IDF represents words based on their frequency in a document relative to their frequency across all documents. It gives higher weight to words that are important in a specific document but rare overall.
*   **Co-occurrence Matrix:** This method builds a matrix where rows and columns represent words, and cell values indicate how often words appear together within a certain context window. Techniques like Singular Value Decomposition (SVD) can then reduce the dimensionality of this matrix to obtain dense vectors. While better than one-hot, these methods still struggle with sparsity and capturing deeper semantic relationships.

<a name="32-predictive-embeddings"></a>
### 3.2. Predictive Embeddings

A major breakthrough came with neural network-based approaches that learn word representations by predicting context words. These models learn embeddings by optimizing an objective function that tries to predict surrounding words given a target word, or vice-versa.

<a name="321-word2vec-cbow--skip-gram"></a>
#### 3.2.1. Word2Vec (CBOW & Skip-gram)

Introduced by Mikolov et al. in 2013, **Word2Vec** is a highly influential framework for learning word embeddings. It offers two main architectures:
*   **Continuous Bag-of-Words (CBOW):** Predicts the current word based on its surrounding context words.
*   **Skip-gram:** Predicts surrounding context words given the current word. This architecture often performs better on smaller datasets and captures subtle semantic relationships more effectively.

Word2Vec models learn embeddings by training a shallow neural network on a large corpus of text. The resulting word vectors effectively capture semantic similarities and analogies, marking a significant leap in NLP capabilities.

<a name="322-glove-global-vectors-for-word-representation"></a>
#### 3.2.2. GloVe (Global Vectors for Word Representation)

Developed by Stanford researchers, **GloVe** (Global Vectors for Word Representation) combines the advantages of both count-based and predictive methods. It trains on the global word-word co-occurrence statistics from a corpus and uses an explicit model to learn word vectors such that their dot product equals the logarithm of their co-occurrence probability. GloVe often produces word vectors with comparable or superior quality to Word2Vec, especially for capturing analogical relationships.

<a name="33-contextual-embeddings"></a>
### 3.3. Contextual Embeddings

While Word2Vec and GloVe provided excellent fixed word embeddings, a critical limitation remained: each word had a single, context-independent representation. This meant that words like "bank" (river bank vs. financial bank) would have the same vector regardless of their usage. **Contextual embeddings** address this by generating word representations that vary depending on the word's specific context in a sentence.

<a name="331-elmo-embeddings-from-language-models"></a>
#### 3.3.1. ELMo (Embeddings from Language Models)

Introduced by Allen AI, **ELMo** was one of the first widely adopted contextual embedding models. It uses a deep bidirectional LSTM (Long Short-Term Memory) network trained on a large text corpus to predict the next word in a sequence (forward LSTM) and the previous word in a sequence (backward LSTM). The final ELMo embedding for a word is a weighted sum of the internal states of these LSTMs, dynamically adjusting the representation based on the surrounding words.

<a name="332-bert-bidirectional-encoder-representations-from-transformers)"></a>
#### 3.3.2. BERT (Bidirectional Encoder Representations from Transformers)

Developed by Google, **BERT** represents a pivotal moment in NLP. Unlike ELMo, which uses LSTMs, BERT employs the **Transformer architecture** and is pre-trained using two novel unsupervised tasks:
1.  **Masked Language Model (MLM):** Randomly masks some tokens from the input and trains the model to predict the original vocabulary id of the masked word based on its context. This allows for truly *bidirectional* learning of context.
2.  **Next Sentence Prediction (NSP):** The model predicts whether two sentences are consecutive in the original text.

BERT generates highly expressive contextual embeddings that have become the backbone of many state-of-the-art NLP models, often achieved through **fine-tuning** on specific downstream tasks. Subsequent models like RoBERTa, XLNet, and GPT-3 have built upon or refined these contextual embedding principles.

<a name="4-applications-of-embeddings-in-nlp"></a>
## 4. Applications of Embeddings in NLP

The adoption of embeddings has dramatically improved the performance of virtually every NLP task. Their ability to capture semantic meaning and relationships has made them indispensable.

<a name="41-semantic-search-and-information-retrieval"></a>
### 4.1. Semantic Search and Information Retrieval

Traditional search engines often rely on keyword matching. With embeddings, search queries and documents can be converted into vectors, and similarity can be measured using **cosine similarity** or Euclidean distance. This enables **semantic search**, where results are returned not just because they contain keywords, but because their underlying meaning is relevant to the query, even if different words are used. This significantly enhances the relevance of search results.

<a name="42-text-classification-and-sentiment-analysis"></a>
### 4.2. Text Classification and Sentiment Analysis

For tasks like classifying documents into categories (e.g., news topics, spam detection) or determining the sentiment of a text (positive, negative, neutral), embeddings provide a powerful input representation. Instead of sparse bag-of-words features, a document can be represented by the average of its word embeddings or through more sophisticated methods that use attention mechanisms over embeddings, leading to higher accuracy and better generalization.

<a name="43-machine-translation-and-cross-lingual-tasks"></a>
### 4.3. Machine Translation and Cross-Lingual Tasks

Embeddings, particularly **multilingual embeddings** that map words from different languages into a shared vector space, have been crucial for advancing machine translation. By aligning embedding spaces across languages, models can understand the semantic equivalence of words and phrases, even across linguistic barriers. This has enabled more fluid and context-aware translations.

<a name="44-named-entity-recognition-ner)"></a>
### 4.4. Named Entity Recognition (NER)

NER involves identifying and classifying named entities (e.g., person names, organizations, locations) in text. Embeddings provide crucial features for sequence labeling models (like LSTMs or Transformers) used in NER. The contextual information encoded in modern embeddings allows models to differentiate between "Washington" as a person and "Washington" as a city, based on the surrounding words.

<a name="45-question-answering"></a>
### 4.5. Question Answering

Systems that answer questions based on a given text benefit immensely from embeddings. By converting both the question and relevant passages into vector representations, models can identify passages that are semantically similar to the question and then pinpoint the exact answer within those passages. The nuanced understanding provided by contextual embeddings allows for highly accurate and relevant answers.

<a name="5-code-example"></a>
## 5. Code Example

This short Python snippet demonstrates a conceptual word embedding lookup using a mock dictionary. In a real-world scenario, `word_to_vector` would be derived from a pre-trained model.

```python
import numpy as np

# A mock dictionary mapping words to their embedding vectors.
# In practice, these vectors are learned from vast amounts of text data by models
# like Word2Vec, GloVe, or derived from large language models (e.g., BERT, GPT).
word_to_vector = {
    "king": np.array([0.6, 0.2, 0.9]),
    "queen": np.array([0.5, 0.3, 0.8]),
    "man": np.array([0.7, 0.1, 0.85]),
    "woman": np.array([0.4, 0.4, 0.75]),
    "royal": np.array([0.55, 0.25, 0.88])
}

def get_word_embedding(word: str) -> np.ndarray:
    """
    Retrieves the embedding vector for a given word.
    Returns a zero vector if the word is not found in our mock dictionary.
    """
    return word_to_vector.get(word.lower(), np.zeros(3))

# Example usage: Get the embedding for 'king'
word = "king"
embedding = get_word_embedding(word)
print(f"The embedding for '{word}' is: {embedding}")

# Example for an unseen word ('castle')
unseen_word = "castle"
unseen_embedding = get_word_embedding(unseen_word)
print(f"The embedding for '{unseen_word}' (unseen) is: {unseen_embedding}")

(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion

Embeddings have fundamentally transformed the field of Natural Language Processing, moving it from sparse, context-agnostic representations to dense, semantically rich vector spaces. From the pioneering efforts of Word2Vec and GloVe to the advanced contextual models like ELMo and BERT, each iteration has pushed the boundaries of what machines can understand about human language. They provide a powerful numerical foundation upon which complex NLP models can build, enabling machines to grasp nuances, identify relationships, and process language with unprecedented accuracy and contextual awareness. As Generative AI continues to evolve, embeddings will remain a cornerstone technology, indispensable for bridging the gap between human linguistic complexity and machine computational efficiency. The future of NLP is undeniably intertwined with the ongoing advancements in embedding techniques, promising even more sophisticated and human-like language understanding.

---
<br>

<a name="türkçe-içerik"></a>
## Doğal Dil İşlemede Gömülü Temsillerin Rolü

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Gömülü Temsiller Nedir?](#2-gömülü-temsiller-nedir)
  - [2.1. Ayrık Temsillerden Yoğun Temsillere](#21-ayrık-temsillerden-yoğun-temsillere)
  - [2.2. Semantik Uzay](#22-semantik-uzay)
- [3. Gömülü Temsil Türleri](#3-gömülü-temsil-türleri)
  - [3.1. Sayıma Dayalı Gömülü Temsiller](#31-sayıma-dayalı-gömülü-temsiller)
  - [3.2. Tahmine Dayalı Gömülü Temsiller](#32-tahmine-dayalı-gömülü-temsiller)
    - [3.2.1. Word2Vec (CBOW & Skip-gram)](#321-word2vec-cbow--skip-gram)
    - [3.2.2. GloVe (Global Vectors for Word Representation)](#322-glove-global-vectors-for-word-representation)
  - [3.3. Bağlamsal Gömülü Temsiller](#33-bağlamsal-gömülü-temsiller)
    - [3.3.1. ELMo (Embeddings from Language Models)](#331-elmo-embeddings-from-language-models)
    - [3.3.2. BERT (Bidirectional Encoder Representations from Transformers)](#332-bert-bidirectional-encoder-representations-from-transformers)
- [4. Gömülü Temsillerin NLP'deki Uygulamaları](#4-gömülü-temsillerin-nlpdeki-uygulamaları)
  - [4.1. Semantik Arama ve Bilgi Erişimi](#41-semantik-arama-ve-bilgi-erişimi)
  - [4.2. Metin Sınıflandırma ve Duygu Analizi](#42-metin-sınıflandırma-ve-duygu-analizi)
  - [4.3. Makine Çevirisi ve Diller Arası Görevler](#43-makine-çevirisi-ve-diller-arası-görevler)
  - [4.4. Adlandırılmış Varlık Tanıma (NER)](#44-adlandırılmış-varlık-tanıma-ner)
  - [4.5. Soru Cevaplama](#45-soru-cevaplama)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Doğal Dil İşleme (NLP), yapay zeka araştırmalarının ön saflarında yer almakta olup, bilgisayarların insan dilini anlamasını, yorumlamasını ve anlamlı bir şekilde üretmesini sağlamayı amaçlar. NLP'deki temel zorluklardan biri her zaman kelimeleri, ifadeleri ve tüm belgeleri makinelerin etkili bir şekilde işleyebileceği bir biçimde nasıl temsil edeceğimiz olmuştur. **Tekil kodlama (one-hot encoding)** gibi geleneksel yöntemler genellikle boyutluluk lanetinden muzdarip olmuş ve kelimeler arasındaki semantik ilişkileri yakalamada başarısız olmuştur. **Gömülü temsillerin (embeddings)** ortaya çıkışı bu alanı devrim niteliğinde değiştirmiş, zengin semantik ve sentaktik bilgiyi kapsayan yoğun, sürekli vektör temsilleri sağlamıştır.

Bu belge, modern NLP sistemlerinde gömülü temsillerin kritik rolünü incelemektedir. Kavramsal temellerini keşfedecek, basit sayıma dayalı modellerden sofistike bağlamsal temsillerin evrimini izleyecek ve semantik aramadan makine çevirisine kadar çeşitli NLP görevleri üzerindeki derin etkilerini inceleyeceğiz. Gömülü temsilleri anlamak, Üretken Yapay Zeka ve NLP alanındaki yetenekleri ve ilerlemeleri kavramak isteyen herkes için hayati öneme sahiptir.

<a name="2-gömülü-temsiller-nedir"></a>
## 2. Gömülü Temsiller Nedir?

Temelde, bir **gömülü temsil (embedding)**, kelimeler, ifadeler veya hatta tüm belgeler gibi ayrık nesnelerin sürekli bir vektör uzayında düşük boyutlu, yoğun bir vektörle temsilidir. Her kelimenin bağımsız bir varlık olarak ele alındığı geleneksel ayrık temsillerden (örneğin, tekil kodlama) farklı olarak, gömülü temsiller benzer nesneleri bu vektör uzayında birbirine daha yakın eşlemeyi amaçlar.

<a name="21-ayrık-temsillerden-yoğun-temsillere"></a>
### 2.1. Ayrık Temsillerden Yoğun Temsillere

Tarihsel olarak, kelimeler genellikle **seyrek (sparse)** yöntemler kullanılarak temsil edilmiştir. Örneğin, **tekil kodlama (one-hot encoding)**, bir sözlükteki her kelimeye benzersiz bir ikili vektör atar; burada yalnızca bir eleman '1' ve diğerleri '0'dır. Basit olmakla birlikte, bu yaklaşımın çeşitli dezavantajları vardır:
1.  **Yüksek Boyutluluk:** V kelimeden oluşan bir sözlük için, her vektörün V boyutu vardır, bu da büyük sözlükler için son derece büyük vektörlere yol açar.
2.  **Semantik Anlam Eksikliği:** Tüm kelimeler birbirinden eşit uzaklıktadır, bu da aralarındaki ilişkiler veya benzerlikler hakkında doğal bir bilgi olmadığı anlamına gelir. Bu temsilde "Kral" ve "Kraliçe", "Kral" ve "Muz" kadar farklıdır.
3.  **Seyreklik:** Vektördeki çoğu eleman sıfırdır, bu da hesaplama açısından verimsizdir.

Gömülü temsiller, kelimeleri genellikle 50 ila 300 boyutlarında sabit boyutlu, gerçek değerli vektörlere eşleyerek bu sınırlamaları aşar. Bunlar **yoğun (dense)** temsillerdir, yani vektördeki çoğu eleman sıfırdan farklıdır.

<a name="22-semantik-uzay"></a>
### 2.2. Semantik Uzay

Gömülü temsillerin gerçek gücü, semantik ve sentaktik özellikleri yakalama yeteneklerinde yatmaktadır. Semantik veya sentaktik olarak benzer kelimeler, vektör uzayında birbirine daha yakın konumlanma eğilimindedir. Örneğin, iyi eğitilmiş bir gömülü temsil uzayında, "kral" kelimesinin vektörü "kraliçe", "prens" ve "kraliyet" kelimelerine yakın olabilir. Ayrıca, `vektör("kral") - vektör("erkek") + vektör("kadın") ≈ vektör("kraliçe")` gibi büyüleyici **vektör aritmetiği** özellikleri ortaya çıkar. Bu, gömülü temsillerin analojileri ve ilişkileri öğrendiğini, karmaşık dilbilimsel kalıpları doğrudan sayısal gösterimlere kodladığını göstermektedir. Bu sürekli uzay, ayrık temsillerle imkansız olan nüanslı karşılaştırmalara ve işlemlere olanak tanır.

<a name="3-gömülü-temsil-türleri"></a>
## 3. Gömülü Temsil Türleri

Gömülü temsillerin evrimi, her biri öncekileri üzerine inşa ederek daha sofistike dilbilimsel bilgileri yakalayan birkaç önemli paradigma görmüştür.

<a name="31-sayıma-dayalı-gömülü-temsiller"></a>
### 3.1. Sayıma Dayalı Gömülü Temsiller

Yoğun temsiller oluşturmaya yönelik ilk girişimler, kelime eş-oluşumlarının istatistiksel analizini içeriyordu.
*   **Terim Sıklığı-Ters Belge Sıklığı (TF-IDF):** Kesinlikle bir gömülü temsil olmasa da, TF-IDF kelimeleri bir belgedeki sıklıklarına göre, tüm belgelerdeki sıklıklarına kıyasla temsil eder. Belirli bir belgede önemli olan ancak genel olarak nadir olan kelimelere daha yüksek ağırlık verir.
*   **Eş-Oluşum Matrisi:** Bu yöntem, satır ve sütunların kelimeleri temsil ettiği ve hücre değerlerinin kelimelerin belirli bir bağlam penceresi içinde ne sıklıkta birlikte göründüğünü gösteren bir matris oluşturur. Tekil Değer Ayrışımı (SVD) gibi teknikler, yoğun vektörler elde etmek için bu matrisin boyutluluğunu azaltabilir. Tekil kodlamadan daha iyi olmakla birlikte, bu yöntemler hala seyreklik ve daha derin semantik ilişkileri yakalamada zorlanmaktadır.

<a name="32-tahmine-dayalı-gömülü-temsiller"></a>
### 3.2. Tahmine Dayalı Gömülü Temsiller

Sinir ağı tabanlı yaklaşımlarla büyük bir atılım geldi; bu yaklaşımlar, bağlam kelimelerini tahmin ederek kelime temsillerini öğrenir. Bu modeller, hedef kelime verildiğinde çevresindeki kelimeleri tahmin etmeye veya tersini yapmaya çalışan bir amaç fonksiyonunu optimize ederek gömülü temsilleri öğrenir.

<a name="321-word2vec-cbow--skip-gram"></a>
#### 3.2.1. Word2Vec (CBOW & Skip-gram)

Mikolov ve arkadaşları tarafından 2013 yılında tanıtılan **Word2Vec**, kelime gömülü temsillerini öğrenmek için oldukça etkili bir çerçevedir. İki ana mimari sunar:
*   **Continuous Bag-of-Words (CBOW):** Mevcut kelimeyi çevresindeki bağlam kelimelerine göre tahmin eder.
*   **Skip-gram:** Mevcut kelime verildiğinde çevresindeki bağlam kelimelerini tahmin eder. Bu mimari genellikle daha küçük veri kümelerinde daha iyi performans gösterir ve ince semantik ilişkileri daha etkili bir şekilde yakalar.

Word2Vec modelleri, büyük bir metin kümesi üzerinde sığ bir sinir ağı eğiterek gömülü temsilleri öğrenir. Ortaya çıkan kelime vektörleri, semantik benzerlikleri ve analojileri etkili bir şekilde yakalar ve NLP yeteneklerinde önemli bir sıçrama yapar.

<a name="322-glove-global-vectors-for-word-representation"></a>
#### 3.2.2. GloVe (Global Vectors for Word Representation)

Stanford araştırmacıları tarafından geliştirilen **GloVe** (Global Vectors for Word Representation), hem sayıma dayalı hem de tahmine dayalı yöntemlerin avantajlarını birleştirir. Bir metin kümesindeki küresel kelime-kelime eş-oluşum istatistikleri üzerinde eğitilir ve noktasal çarpımları eş-oluşum olasılıklarının logaritmasına eşit olacak şekilde kelime vektörlerini öğrenmek için açık bir model kullanır. GloVe, özellikle analojik ilişkileri yakalamak için Word2Vec'e benzer veya ondan üstün kalitede kelime vektörleri üretir.

<a name="33-bağlamsal-gömülü-temsiller"></a>
### 3.3. Bağlamsal Gömülü Temsiller

Word2Vec ve GloVe mükemmel sabit kelime gömülü temsilleri sağlarken, kritik bir sınırlama devam etti: her kelimenin tek, bağlamdan bağımsız bir temsili vardı. Bu, "banka" (nehir kıyısı veya finans kurumu) gibi kelimelerin kullanımlarına bakılmaksızın aynı vektöre sahip olacağı anlamına geliyordu. **Bağlamsal gömülü temsiller (contextual embeddings)**, kelimenin bir cümledeki belirli bağlamına bağlı olarak değişen kelime temsilleri oluşturarak bu sorunu çözer.

<a name="331-elmo-embeddings-from-language-models"></a>
#### 3.3.1. ELMo (Embeddings from Language Models)

Allen AI tarafından tanıtılan **ELMo**, yaygın olarak benimsenen ilk bağlamsal gömülü temsil modellerinden biriydi. Bir dizi içindeki bir sonraki kelimeyi (ileri LSTM) ve bir dizi içindeki önceki kelimeyi (geri LSTM) tahmin etmek için büyük bir metin kümesi üzerinde eğitilmiş derin çift yönlü bir LSTM (Long Short-Term Memory) ağı kullanır. Bir kelime için nihai ELMo gömülü temsil, bu LSTM'lerin dahili durumlarının ağırlıklı bir toplamıdır ve çevresindeki kelimelere göre temsili dinamik olarak ayarlar.

<a name="332-bert-bidirectional-encoder-representations-from-transformers)"></a>
#### 3.3.2. BERT (Bidirectional Encoder Representations from Transformers)

Google tarafından geliştirilen **BERT**, NLP'de önemli bir anı temsil etmektedir. LSTMLer kullanan ELMo'dan farklı olarak, BERT **Transformer mimarisini** kullanır ve iki yeni denetimsiz görev kullanılarak önceden eğitilir:
1.  **Maskeli Dil Modeli (Masked Language Model - MLM):** Girişten bazı belirteçleri rastgele maskeler ve modeli, bağlamına dayanarak maskelenen kelimenin orijinal kelime dağarcığı kimliğini tahmin etmek için eğitir. Bu, bağlamın gerçek *çift yönlü* öğrenilmesine olanak tanır.
2.  **Sonraki Cümle Tahmini (Next Sentence Prediction - NSP):** Model, iki cümlenin orijinal metinde art arda gelip gelmediğini tahmin eder.

BERT, birçok son teknoloji NLP modelinin temelini oluşturan, genellikle belirli aşağı akış görevlerinde **ince ayar (fine-tuning)** yoluyla elde edilen son derece etkileyici bağlamsal gömülü temsiller üretir. RoBERTa, XLNet ve GPT-3 gibi sonraki modeller bu bağlamsal gömülü temsil ilkelerini temel almıştır veya geliştirmiştir.

<a name="4-gömülü-temsillerin-nlpdeki-uygulamaları"></a>
## 4. Gömülü Temsillerin NLP'deki Uygulamaları

Gömülü temsillerin benimsenmesi, hemen hemen her NLP görevinin performansını önemli ölçüde artırmıştır. Semantik anlam ve ilişkileri yakalama yetenekleri onları vazgeçilmez kılmıştır.

<a name="41-semantik-arama-ve-bilgi-erişimi"></a>
### 4.1. Semantik Arama ve Bilgi Erişimi

Geleneksel arama motorları genellikle anahtar kelime eşleştirmeye dayanır. Gömülü temsillerle, arama sorguları ve belgeler vektörlere dönüştürülebilir ve benzerlik **kosinüs benzerliği** veya Öklid uzaklığı kullanılarak ölçülebilir. Bu, anahtar kelimeler içerdiği için değil, temel anlamları sorguyla ilgili olduğu için, farklı kelimeler kullanılsa bile sonuçların döndürüldüğü **semantik arama** sağlar. Bu, arama sonuçlarının alaka düzeyini önemli ölçüde artırır.

<a name="42-metin-sınıflandırma-ve-duygu-analizi"></a>
### 4.2. Metin Sınıflandırma ve Duygu Analizi

Belgeleri kategorilere ayırma (örneğin, haber başlıkları, spam tespiti) veya bir metnin duyarlılığını (olumlu, olumsuz, nötr) belirleme gibi görevler için gömülü temsiller güçlü bir girdi temsili sağlar. Seyrek kelime çantası özellikler yerine, bir belge kelime gömülü temsillerinin ortalaması veya gömülü temsiller üzerinde dikkat mekanizmaları kullanan daha sofistike yöntemlerle temsil edilebilir, bu da daha yüksek doğruluk ve daha iyi genelleme sağlar.

<a name="43-makine-çevirisi-ve-diller-arası-görevler"></a>
### 4.3. Makine Çevirisi ve Diller Arası Görevler

Gömülü temsiller, özellikle farklı dillerdeki kelimeleri ortak bir vektör uzayına eşleyen **çok dilli gömülü temsiller**, makine çevirisini ilerletmek için çok önemli olmuştur. Diller arası gömülü temsil uzaylarını hizalayarak, modeller kelime ve ifadelerin anlamsal eşdeğerliğini, dilsel engeller arasında bile anlayabilir. Bu, daha akıcı ve bağlama duyarlı çeviriler sağlamıştır.

<a name="44-adlandırılmış-varlık-tanıma-ner)"></a>
### 4.4. Adlandırılmış Varlık Tanıma (NER)

NER, metindeki adlandırılmış varlıkları (örneğin, kişi adları, kuruluşlar, yerler) tanımlamayı ve sınıflandırmayı içerir. Gömülü temsiller, NER'de kullanılan dizi etiketleme modelleri (LSTM'ler veya Transformatörler gibi) için önemli özellikler sağlar. Modern gömülü temsillerde kodlanan bağlamsal bilgi, modellerin "Washington"ı bir kişi olarak ve "Washington"ı bir şehir olarak, çevresindeki kelimelere göre ayırt etmesine olanak tanır.

<a name="45-soru-cevaplama"></a>
### 4.5. Soru Cevaplama

Verilen bir metne dayanarak soruları yanıtlayan sistemler, gömülü temsillerden büyük ölçüde faydalanır. Hem soruyu hem de ilgili pasajları vektör temsillerine dönüştürerek, modeller soruyla anlamsal olarak benzer pasajları tanımlayabilir ve daha sonra bu pasajlar içindeki kesin cevabı bulabilir. Bağlamsal gömülü temsillerin sağladığı nüanslı anlayış, son derece doğru ve alakalı cevaplar sağlar.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği

Bu kısa Python kodu, sahte bir sözlük kullanarak kavramsal bir kelime gömülü temsil aramasını gösterir. Gerçek dünya senaryosunda, `word_to_vector` önceden eğitilmiş bir modelden türetilir.

```python
import numpy as np

# Kelimeleri gömülü temsil vektörleriyle eşleştiren sahte bir sözlük.
# Pratikte bu vektörler, Word2Vec, GloVe gibi modeller tarafından
# veya büyük dil modellerinden (örn. BERT, GPT) türetilerek büyük metin veri kümelerinden öğrenilir.
word_to_vector = {
    "kral": np.array([0.6, 0.2, 0.9]),
    "kraliçe": np.array([0.5, 0.3, 0.8]),
    "adam": np.array([0.7, 0.1, 0.85]),
    "kadın": np.array([0.4, 0.4, 0.75]),
    "kraliyet": np.array([0.55, 0.25, 0.88])
}

def get_word_embedding(word: str) -> np.ndarray:
    """
    Belirli bir kelime için gömülü temsil vektörünü alır.
    Kelime sahte sözlüğümüzde bulunamazsa sıfır vektörü döndürür.
    """
    return word_to_vector.get(word.lower(), np.zeros(3))

# Kullanım örneği: 'kral' kelimesinin gömülü temsilini al
word = "kral"
embedding = get_word_embedding(word)
print(f"'{word}' kelimesinin gömülü temsili: {embedding}")

# Görülmeyen bir kelime için örnek ('kale')
unseen_word = "kale"
unseen_embedding = get_word_embedding(unseen_word)
print(f"'{unseen_word}' (görülmeyen) kelimesinin gömülü temsili: {unseen_embedding}")

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç

Gömülü temsiller, Doğal Dil İşleme alanını kökten değiştirmiş, seyrek, bağlamdan bağımsız temsillerden yoğun, anlamsal açıdan zengin vektör uzaylarına taşımıştır. Word2Vec ve GloVe'nin öncü çalışmalarından ELMo ve BERT gibi gelişmiş bağlamsal modellere kadar, her iterasyon makinelerin insan dili hakkında anlayabileceği sınırları zorlamıştır. Karmaşık NLP modellerinin üzerine inşa edilebileceği güçlü bir sayısal temel sağlarlar, makinelerin nüansları kavramasına, ilişkileri tanımlamasına ve dili benzeri görülmemiş doğruluk ve bağlamsal farkındalıkla işlemesine olanak tanırlar. Üretken Yapay Zeka gelişmeye devam ettikçe, gömülü temsiller köşe taşı bir teknoloji olarak kalacak, insan dilbilimsel karmaşıklığı ile makine hesaplama verimliliği arasındaki boşluğu doldurmak için vazgeçilmez olacaktır. NLP'nin geleceği, şüphesiz, gömülü temsil tekniklerindeki devam eden ilerlemelerle iç içedir ve daha da sofistike ve insan benzeri dil anlayışı vaat etmektedir.