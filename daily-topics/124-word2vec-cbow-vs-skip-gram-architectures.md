# Word2Vec: CBOW vs. Skip-Gram Architectures

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Word Embeddings and Word2Vec](#2-word-embeddings-and-word2vec)
  - [2.1. The Need for Word Embeddings](#21-the-need-for-word-embeddings)
  - [2.2. Introduction to Word2Vec](#22-introduction-to-word2vec)
- [3. Word2Vec Architectures](#3-word2vec-architectures)
  - [3.1. Continuous Bag-of-Words (CBOW)](#31-continuous-bag-of-words-cbow)
    - [3.1.1. Architecture and Mechanism](#311-architecture-and-mechanism)
    - [3.1.2. Strengths and Weaknesses](#312-strengths-and-weaknesses)
  - [3.2. Skip-Gram](#32-skip-gram)
    - [3.2.1. Architecture and Mechanism](#321-architecture-and-mechanism)
    - [3.2.2. Strengths and Weaknesses](#322-strengths-and-weaknesses)
  - [3.3. Key Differences and Use Cases](#33-key-differences-and-use-cases)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

<a name="1-introduction"></a>
### 1. Introduction
The realm of **Generative Artificial Intelligence** has witnessed profound advancements in recent years, particularly in the domain of **Natural Language Processing (NLP)**. A foundational component enabling sophisticated language understanding and generation is the concept of **word embeddings**. These are dense, low-dimensional vector representations of words that capture semantic and syntactic relationships, allowing machine learning models to process textual data more effectively than traditional sparse representations.

Among the pioneering and most influential techniques for learning word embeddings is **Word2Vec**, introduced by Mikolov et al. in 2013. Word2Vec is not a single algorithm but a framework encompassing two distinct neural network architectures: **Continuous Bag-of-Words (CBOW)** and **Skip-Gram**. While both aim to generate high-quality word vectors based on the distributional hypothesis (words appearing in similar contexts tend to have similar meanings), their underlying mechanisms and predictive tasks differ significantly.

This comprehensive document delves into the intricacies of Word2Vec, providing an in-depth comparative analysis of its CBOW and Skip-Gram architectures. We will explore their operational principles, architectural designs, specific strengths, inherent weaknesses, and optimal use cases, supported by an illustrative code example, to offer a complete understanding of these cornerstone techniques in modern NLP.

<a name="2-word-embeddings-and-word2vec"></a>
### 2. Word Embeddings and Word2Vec

<a name="21-the-need-for-word-embeddings"></a>
#### 2.1. The Need for Word Embeddings
Prior to the advent of sophisticated word embedding techniques, words in NLP tasks were often represented using methods like **one-hot encoding**. In this approach, each word in a vocabulary is assigned a unique index, and represented by a binary vector where only the dimension corresponding to that word is 1, and all others are 0. While simple, one-hot encoding suffers from several critical limitations:
*   **High Dimensionality:** For large vocabularies, these vectors become extremely long and sparse, leading to the **curse of dimensionality**.
*   **Lack of Semantic Information:** One-hot vectors treat all words as equidistant and orthogonal, failing to capture any semantic or syntactic relationships between them. For instance, "king" and "queen" would be as distinct as "king" and "table."
*   **Computational Inefficiency:** Processing such sparse, high-dimensional vectors is computationally expensive.

**Word embeddings** address these challenges by mapping words to dense, continuous vector representations in a lower-dimensional space. The core idea is that words with similar meanings will have similar vector representations (i.e., be close to each other in the vector space). This allows models to generalize better, understand nuances of language, and perform tasks like sentiment analysis, machine translation, and text classification with greater accuracy.

<a name="22-introduction-to-Word2Vec"></a>
#### 2.2. Introduction to Word2Vec
Word2Vec, developed by a team at Google led by Tomas Mikolov, revolutionized the field by offering an efficient method for learning these high-quality **distributed representations** (embeddings). Unlike earlier neural language models, Word2Vec is not designed to predict the next word in a sequence but rather to learn word associations. Its fundamental principle is the **distributional hypothesis**, which posits that words that appear in similar contexts often share similar meanings.

Word2Vec models are trained on large text corpora, where they learn to associate words based on their co-occurrence patterns. The output is a set of **word vectors** (or word embeddings) where each word in the vocabulary is represented by a vector of real numbers. The position of a word in this vector space is learned from its context, such that words with similar contexts are positioned close together. Word2Vec leverages a shallow neural network architecture, making it computationally efficient while producing powerful semantic representations. As mentioned, it consists of two primary architectures: CBOW and Skip-Gram, which we will now explore in detail.

<a name="3-word2vec-architectures"></a>
### 3. Word2Vec Architectures
The two main architectures within the Word2Vec framework—Continuous Bag-of-Words (CBOW) and Skip-Gram—employ different strategies to learn word embeddings, primarily by reversing their predictive tasks.

<a name="31-continuous-bag-of-words-cbow"></a>
#### 3.1. Continuous Bag-of-Words (CBOW)

<a name="311-architecture-and-mechanism"></a>
##### 3.1.1. Architecture and Mechanism
The **Continuous Bag-of-Words (CBOW)** model aims to predict a target word given its surrounding context words. Its architecture typically consists of three layers: an input layer, a projection layer, and an output layer.

1.  **Input Layer:** This layer receives the one-hot encoded vectors of the context words. For a given target word, the context typically includes words appearing within a fixed-size window both before and after the target word. For example, if the sentence is "The quick brown fox jumps over the lazy dog" and "fox" is the target word with a window size of 2, the context words would be "quick", "brown", "jumps", and "over".
2.  **Projection Layer:** Instead of concatenating the input vectors, CBOW averages the vectors of the context words in this layer. This averaging operation effectively creates a single, continuous vector representing the 'bag' of context words. This is where the "Continuous Bag-of-Words" name originates – it treats the context words as an unordered bag, continuous because of the vector averaging.
3.  **Output Layer:** This layer uses a **softmax function** to output the probability distribution of the target word over the entire vocabulary. The goal during training is to maximize the probability of the actual target word given the context.

In essence, CBOW asks: "Given these surrounding words, what is the most likely word that fits in the middle?"

<a name="312-strengths-and-weaknesses"></a>
##### 3.1.2. Strengths and Weaknesses
**Strengths of CBOW:**
*   **Training Speed:** CBOW is generally faster to train than Skip-Gram, especially for large datasets. This is because it averages the context word vectors in the projection layer, performing fewer updates per training step compared to Skip-Gram which updates for each context word.
*   **Accuracy for Frequent Words:** It performs well in capturing the representations of frequently occurring words because it leverages multiple context words to predict a single target word.

**Weaknesses of CBOW:**
*   **Less Effective for Infrequent Words:** CBOW struggles to learn good representations for rare or less frequent words. Since rare words appear less often as target words, the model has fewer opportunities to update their specific embeddings based on diverse contexts.
*   **Less Detail for Semantic Nuances:** The averaging of context vectors can sometimes lead to a loss of information, making it less effective at capturing the subtle semantic nuances that might be present in a wider range of context-target word pairs.

<a name="32-skip-gram"></a>
#### 3.2. Skip-Gram

<a name="321-architecture-and-mechanism"></a>
##### 3.2.1. Architecture and Mechanism
The **Skip-Gram** model operates in the reverse manner to CBOW: it predicts the surrounding context words given a target word. Like CBOW, its architecture also typically involves three layers.

1.  **Input Layer:** This layer receives the one-hot encoded vector of a single target word. For instance, if "fox" is the target word, its one-hot vector is the input.
2.  **Projection Layer:** This layer acts as a lookup table for the input target word. The one-hot vector is multiplied by a weight matrix (which essentially stores the word embeddings) to directly project the target word into its dense vector representation. This vector is then used to predict its context words.
3.  **Output Layer:** This layer uses a **softmax function** to predict the probability distribution of multiple context words within a specified window around the target word. For each context word in the window, the model computes a probability. The training objective is to maximize the probabilities of the actual context words given the target word.

In essence, Skip-Gram asks: "Given this word, what are the most likely words that would appear around it in a sentence?"

<a name="322-strengths-and-weaknesses"></a>
##### 3.2.2. Strengths and Weaknesses
**Strengths of Skip-Gram:**
*   **Better for Infrequent Words:** Skip-Gram is highly effective at learning good representations for rare words or phrases. Since it predicts multiple context words for each target word, a rare target word gets more opportunities to update its vector through various context predictions.
*   **Captures Semantic Nuances:** It generally performs better at capturing finer semantic relationships and nuances, especially when the dataset is smaller or contains a rich vocabulary with many infrequent terms.
*   **Effective with Smaller Datasets:** Often shows superior performance with smaller training datasets compared to CBOW.

**Weaknesses of Skip-Gram:**
*   **Slower Training Speed:** Skip-Gram is computationally more intensive and slower to train than CBOW. For each target word, it needs to perform updates for each predicted context word in the window, which can be numerous.
*   **Higher Computational Cost:** Due to the multiple predictions and updates, its computational cost can be higher, especially with large window sizes and expansive vocabularies.

<a name="33-key-differences-and-use-cases"></a>
#### 3.3. Key Differences and Use Cases
The fundamental difference between CBOW and Skip-Gram lies in their predictive tasks:

*   **CBOW:** Predicts the target word from its context words. (Context -> Target)
*   **Skip-Gram:** Predicts the context words from a target word. (Target -> Context)

This distinction leads to differing performance characteristics and preferred use cases:

| Feature           | CBOW                                              | Skip-Gram                                            |
| :---------------- | :------------------------------------------------ | :--------------------------------------------------- |
| **Predictive Task** | Predicts middle word from context                 | Predicts context words from middle word              |
| **Training Speed**| Faster (especially on large datasets)             | Slower                                               |
| **Accuracy**      | Good for frequent words                           | Good for infrequent words/phrases; higher overall accuracy for rich semantics |
| **Data Size**     | Better for very large datasets                    | Better for smaller datasets (can learn from few examples) |
| **Computational Cost** | Lower per training step                       | Higher per training step                             |

**Choosing between CBOW and Skip-Gram:**
*   **CBOW** is generally preferred when the dataset is very large, and the primary goal is to obtain word embeddings quickly, focusing on common words. It's often chosen for tasks where speed is critical and the vocabulary does not contain a significant proportion of rare words whose precise embeddings are paramount.
*   **Skip-Gram** is typically recommended when higher accuracy is desired, especially for understanding semantic relationships of rare words or when working with smaller datasets. It shines in tasks that require a more nuanced understanding of individual word meanings, such as complex semantic analogies or disambiguation.

The choice ultimately depends on the specific requirements of the NLP task, the characteristics of the training corpus (size, vocabulary richness, frequency distribution), and available computational resources.

<a name="4-code-example"></a>
### 4. Code Example
This Python code snippet demonstrates how to initialize and conceptually train both CBOW and Skip-Gram models using the `gensim` library. It illustrates the `sg` parameter which differentiates between the two architectures and showcases how to access word vectors and find similar words.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK punkt tokenizer is available for word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Sample text data (a small corpus for demonstration)
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast animal, the fox, is known for its agility.",
    "Dogs are often loyal companions.",
    "Cats are also popular pets."
]

# Tokenize sentences into words. Word2Vec expects a list of lists of words.
# Convert to lowercase for consistent vocabulary.
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

print("Tokenized sentences for training:")
for s in tokenized_sentences:
    print(s)

# --- Training CBOW model ---
print("\n--- Training Continuous Bag-of-Words (CBOW) model ---")
# sg=0 specifies the CBOW architecture (default)
# vector_size: Dimensionality of the word vectors (e.g., 100 dimensions)
# window: Maximum distance between the current and predicted word within a sentence
# min_count: Ignores all words with total frequency lower than this. Set to 1 for this small corpus.
# workers: Number of worker threads to use for training (= faster training)
cbow_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100, # 100-dimensional word vectors
    window=5,      # context window of 5 words on each side
    min_count=1,   # include all words, even rare ones in this small corpus
    sg=0,          # CBOW architecture
    workers=4      # use 4 threads for training
)
print("CBOW model training complete.")

# Access word vectors and find similar words for the CBOW model
print(f"Vector for 'fox' (CBOW, first 5 elements): {cbow_model.wv['fox'][:5]}...")
print("\nWords similar to 'fox' (CBOW model):")
try:
    for word, score in cbow_model.wv.most_similar('fox', topn=3):
        print(f"  {word}: {score:.4f}")
except KeyError:
    print(" 'fox' not in vocabulary. Adjust min_count if using a different corpus.")


# --- Training Skip-Gram model ---
print("\n--- Training Skip-Gram model ---")
# sg=1 specifies the Skip-Gram architecture
skipgram_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100, # 100-dimensional word vectors
    window=5,      # context window of 5 words on each side
    min_count=1,   # include all words
    sg=1,          # Skip-Gram architecture
    workers=4
)
print("Skip-Gram model training complete.")

# Access word vectors and find similar words for the Skip-Gram model
print(f"Vector for 'fox' (Skip-Gram, first 5 elements): {skipgram_model.wv['fox'][:5]}...")
print("\nWords similar to 'fox' (Skip-Gram model):")
try:
    for word, score in skipgram_model.wv.most_similar('fox', topn=3):
        print(f"  {word}: {score:.4f}")
except KeyError:
    print(" 'fox' not in vocabulary. Adjust min_count if using a different corpus.")

# Note: The similarity results on a tiny corpus like this will not be semantically meaningful
# but the example demonstrates the API usage for both architectures.

(End of code example section)
```

<a name="5-conclusion"></a>
### 5. Conclusion
Word2Vec stands as a landmark innovation in Natural Language Processing, fundamentally transforming how machines understand and process human language by representing words as dense, meaningful vectors. The framework's two primary architectures, **Continuous Bag-of-Words (CBOW)** and **Skip-Gram**, offer distinct yet powerful approaches to learning these **word embeddings**, each with its own set of advantages and disadvantages.

CBOW, by predicting a target word from its surrounding context, excels in computational efficiency and performance on frequent words, making it a suitable choice for large datasets where speed is a priority. Conversely, Skip-Gram, by predicting context words from a target word, demonstrates superior capability in capturing the nuances of less frequent words and often yields higher overall accuracy in semantic representation, particularly valuable for smaller corpora or tasks requiring detailed semantic understanding.

The choice between CBOW and Skip-Gram is not absolute but rather a strategic decision guided by the characteristics of the training data, the specific requirements of the NLP task, and available computational resources. Both architectures have laid crucial groundwork for subsequent advancements in word representation learning, influencing models like GloVe and FastText, and remaining integral components in the vast landscape of modern Generative AI and deep learning-based language models. Understanding their mechanisms and trade-offs is essential for anyone delving into the complexities of machine language comprehension.

<a name="6-references"></a>
### 6. References
*   Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient estimation of word representations in vector space*. arXiv preprint arXiv:1301.3781.
*   Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed representations of words and phrases and their compositionality*. Advances in neural information processing systems, 26.

---
<br>

<a name="türkçe-içerik"></a>
## Word2Vec: CBOW ve Skip-Gram Mimarileri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Kelime Gömüleri ve Word2Vec](#2-kelime-gömüleri-ve-word2vec)
  - [2.1. Kelime Gömülerine Duyulan İhtiyaç](#21-kelime-gömülerine-duyulan-ihtiyaç)
  - [2.2. Word2Vec'e Giriş](#22-word2vec'e-giriş)
- [3. Word2Vec Mimarileri](#3-word2vec-mimarileri)
  - [3.1. Sürekli Kelime Torbası (CBOW)](#31-sürekli-kelime-torbası-cbow)
    - [3.1.1. Mimari ve Mekanizma](#311-mimari-ve-mekanizma)
    - [3.1.2. Güçlü ve Zayıf Yönleri](#312-güçlü-ve-zayıf-yönleri)
  - [3.2. Skip-Gram](#32-skip-gram)
    - [3.2.1. Mimari ve Mekanizma](#321-mimari-ve-mekanizma)
    - [3.2.2. Güçlü ve Zayıf Yönleri](#322-güçlü-ve-zayıf-yönleri)
  - [3.3. Temel Farklılıklar ve Kullanım Senaryoları](#33-temel-farklılıklar-ve-kullanım-senaryoları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Kaynaklar](#6-kaynaklar)

<a name="1-giriş"></a>
### 1. Giriş
**Üretken Yapay Zeka** alanı, özellikle **Doğal Dil İşleme (NLP)** alanında son yıllarda önemli gelişmeler kaydetmiştir. Gelişmiş dil anlama ve üretme yeteneklerinin temel bir bileşeni, **kelime gömüleri (word embeddings)** kavramıdır. Bunlar, kelimelerin anlamsal ve sözdizimsel ilişkilerini yakalayan yoğun, düşük boyutlu vektör temsilleridir ve makine öğrenimi modellerinin metinsel verileri geleneksel seyrek temsillerden daha etkili bir şekilde işlemesini sağlar.

Kelime gömülerini öğrenmek için öncü ve en etkili tekniklerden biri, Mikolov ve arkadaşları tarafından 2013 yılında tanıtılan **Word2Vec**'tir. Word2Vec tek bir algoritma değil, iki farklı yapay sinir ağı mimarisini içeren bir çerçevedir: **Sürekli Kelime Torbası (CBOW)** ve **Skip-Gram**. Her ikisi de dağılımsal hipoteze (benzer bağlamlarda görünen kelimeler benzer anlamlara sahip olma eğilimindedir) dayalı yüksek kaliteli kelime vektörleri oluşturmayı hedeflerken, temel mekanizmaları ve tahmin görevleri önemli ölçüde farklılık gösterir.

Bu kapsamlı belge, Word2Vec'in inceliklerine derinlemesine inerek, CBOW ve Skip-Gram mimarilerinin karşılaştırmalı bir analizini sunmaktadır. İşleyiş prensiplerini, mimari tasarımlarını, belirli güçlü ve zayıf yönlerini ve optimize edilmiş kullanım senaryolarını, açıklayıcı bir kod örneğiyle destekleyerek, modern NLP'deki bu temel tekniklerin tam bir anlayışını sunacağız.

<a name="2-kelime-gömüleri-ve-word2vec"></a>
### 2. Kelime Gömüleri ve Word2Vec

<a name="21-kelime-gömülerine-duyulan-ihtiyaç"></a>
#### 2.1. Kelime Gömülerine Duyulan İhtiyaç
Gelişmiş kelime gömü tekniklerinin ortaya çıkmasından önce, NLP görevlerindeki kelimeler genellikle **tek-sıcak kodlama (one-hot encoding)** gibi yöntemler kullanılarak temsil edilirdi. Bu yaklaşımda, bir kelime dağarcığındaki her kelimeye benzersiz bir indeks atanır ve o kelimeye karşılık gelen boyutun 1, diğer tüm boyutların ise 0 olduğu ikili bir vektörle temsil edilir. Basit olmasına rağmen, tek-sıcak kodlama bazı kritik sınırlamalara sahiptir:
*   **Yüksek Boyutsallık:** Büyük kelime dağarcıkları için bu vektörler aşırı uzun ve seyrek hale gelir, bu da **boyutsallık lanetine** yol açar.
*   **Anlamsal Bilgi Eksikliği:** Tek-sıcak vektörler, tüm kelimeleri eşit uzaklıkta ve ortogonal olarak ele alır, aralarındaki anlamsal veya sözdizimsel ilişkileri yakalayamaz. Örneğin, "kral" ve "kraliçe", "kral" ve "masa" kadar farklı olarak işlem görür.
*   **Hesaplama Verimsizliği:** Bu tür seyrek, yüksek boyutlu vektörlerin işlenmesi hesaplama açısından maliyetlidir.

**Kelime gömüleri**, kelimeleri daha düşük boyutlu bir uzayda yoğun, sürekli vektör temsillerine eşleyerek bu zorlukları giderir. Temel fikir, benzer anlamlara sahip kelimelerin benzer vektör temsillerine sahip olmasıdır (yani, vektör uzayında birbirine yakın olması). Bu, modellerin daha iyi genelleme yapmasına, dilin nüanslarını anlamasına ve duygu analizi, makine çevirisi ve metin sınıflandırması gibi görevleri daha yüksek doğrulukla gerçekleştirmesine olanak tanır.

<a name="22-word2vec'e-giriş"></a>
#### 2.2. Word2Vec'e Giriş
Tomas Mikolov liderliğindeki Google'daki bir ekip tarafından geliştirilen Word2Vec, bu yüksek kaliteli **dağıtık temsilleri (embeddings)** öğrenmek için verimli bir yöntem sunarak alanı devrim niteliğinde değiştirdi. Önceki sinirsel dil modellerinin aksine, Word2Vec bir dizideki bir sonraki kelimeyi tahmin etmek için değil, kelime ilişkilerini öğrenmek için tasarlanmıştır. Temel prensibi, benzer bağlamlarda görünen kelimelerin genellikle benzer anlamları paylaştığını öne süren **dağılımsal hipotezdir**.

Word2Vec modelleri, büyük metin korpusları üzerinde eğitilir ve burada kelimeleri birlikte ortaya çıkma kalıplarına göre ilişkilendirmeyi öğrenirler. Çıktı, kelime dağarcığındaki her kelimenin gerçek sayılardan oluşan bir vektörle temsil edildiği bir dizi **kelime vektörü** (veya kelime gömüsü)'dür. Bir kelimenin bu vektör uzayındaki konumu, bağlamından öğrenilir, öyle ki benzer bağlamlara sahip kelimeler birbirine yakın konumlandırılır. Word2Vec, sığ bir yapay sinir ağı mimarisinden yararlanarak, güçlü anlamsal temsiller üretirken hesaplama açısından verimli olmasını sağlar. Belirtildiği gibi, iki ana mimariden oluşur: CBOW ve Skip-Gram, şimdi bunları ayrıntılı olarak inceleyeceğiz.

<a name="3-word2vec-mimarileri"></a>
### 3. Word2Vec Mimarileri
Word2Vec çerçevesindeki iki ana mimari – Sürekli Kelime Torbası (CBOW) ve Skip-Gram – kelime gömülerini öğrenmek için farklı stratejiler kullanır ve esas olarak tahmin görevlerini tersine çevirir.

<a name="31-sürekli-kelime-torbası-cbow"></a>
#### 3.1. Sürekli Kelime Torbası (CBOW)

<a name="311-mimari-ve-mekanizma"></a>
##### 3.1.1. Mimari ve Mekanizma
**Sürekli Kelime Torbası (CBOW)** modeli, çevreleyen bağlam kelimeleri göz önüne alındığında bir hedef kelimeyi tahmin etmeyi amaçlar. Mimarisi tipik olarak üç katmandan oluşur: bir girdi katmanı, bir projeksiyon katmanı ve bir çıktı katmanı.

1.  **Girdi Katmanı:** Bu katman, bağlam kelimelerinin tek-sıcak kodlanmış vektörlerini alır. Belirli bir hedef kelime için bağlam, tipik olarak hedef kelimeden hem önce hem de sonra sabit boyutlu bir pencere içinde görünen kelimeleri içerir. Örneğin, cümle "The quick brown fox jumps over the lazy dog" ise ve "fox" hedef kelime, pencere boyutu 2 ise, bağlam kelimeleri "quick", "brown", "jumps" ve "over" olacaktır.
2.  **Projeksiyon Katmanı:** Girdi vektörlerini birleştirmek yerine, CBOW bu katmanda bağlam kelimelerinin vektörlerini ortalar. Bu ortalama işlemi, bağlam kelimelerinin 'torbasını' temsil eden tek, sürekli bir vektör oluşturur. "Sürekli Kelime Torbası" adı buradan gelir – bağlam kelimelerini sıralanmamış bir torba olarak ele alır ve vektör ortalaması nedeniyle süreklidir.
3.  **Çıktı Katmanı:** Bu katman, tüm kelime dağarcığı üzerindeki hedef kelimenin olasılık dağılımını çıktı olarak vermek için bir **softmax fonksiyonu** kullanır. Eğitim sırasında amaç, bağlam verildiğinde gerçek hedef kelimenin olasılığını en üst düzeye çıkarmaktır.

Esasen CBOW şunu sorar: "Bu çevreleyen kelimeler göz önüne alındığında, ortasına en çok uyan kelime nedir?"

<a name="312-güçlü-ve-zayıf-yönleri"></a>
##### 3.1.2. Güçlü ve Zayıf Yönleri
**CBOW'un Güçlü Yönleri:**
*   **Eğitim Hızı:** CBOW, özellikle büyük veri kümeleri için Skip-Gram'dan daha hızlı eğitilir. Bunun nedeni, projeksiyon katmanında bağlam kelime vektörlerini ortalaması ve her eğitim adımında Skip-Gram'a göre daha az güncelleme yapmasıdır.
*   **Sık Kullanılan Kelimelerde Doğruluk:** Tek bir hedef kelimeyi tahmin etmek için birden fazla bağlam kelimesinden yararlandığı için sık kullanılan kelimelerin temsillerini yakalamada iyi performans gösterir.

**CBOW'un Zayıf Yönleri:**
*   **Seyrek Kelimelerde Daha Az Etkili:** CBOW, nadir veya daha az sıklıkta kullanılan kelimeler için iyi temsiller öğrenmekte zorlanır. Nadir kelimeler hedef kelime olarak daha az ortaya çıktığı için, modelin farklı bağlamlara dayalı olarak belirli gömülerini güncelleme fırsatları daha azdır.
*   **Anlamsal Nüanslar İçin Daha Az Detay:** Bağlam vektörlerinin ortalaması bazen bilgi kaybına yol açabilir, bu da daha geniş bir bağlam-hedef kelime çiftlerinde mevcut olabilecek ince anlamsal nüansları yakalamada daha az etkili olmasına neden olur.

<a name="32-skip-gram"></a>
#### 3.2. Skip-Gram

<a name="321-mimari-ve-mekanizma"></a>
##### 3.2.1. Mimari ve Mekanizma
**Skip-Gram** modeli, CBOW'un tersi şekilde çalışır: bir hedef kelime verildiğinde çevreleyen bağlam kelimelerini tahmin eder. CBOW gibi, mimarisi de tipik olarak üç katman içerir.

1.  **Girdi Katmanı:** Bu katman, tek bir hedef kelimenin tek-sıcak kodlanmış vektörünü alır. Örneğin, "fox" hedef kelime ise, onun tek-sıcak vektörü girdidir.
2.  **Projeksiyon Katmanı:** Bu katman, girdi hedef kelimesi için bir arama tablosu görevi görür. Tek-sıcak vektör, ağırlık matrisi (temel olarak kelime gömülerini depolayan) ile çarpılarak hedef kelimeyi doğrudan yoğun vektör temsiline dönüştürür. Bu vektör daha sonra bağlam kelimelerini tahmin etmek için kullanılır.
3.  **Çıktı Katmanı:** Bu katman, hedef kelime etrafındaki belirli bir pencere içindeki birden çok bağlam kelimesinin olasılık dağılımını tahmin etmek için bir **softmax fonksiyonu** kullanır. Penceredeki her bağlam kelimesi için model bir olasılık hesaplar. Eğitim amacı, hedef kelime verildiğinde gerçek bağlam kelimelerinin olasılıklarını en üst düzeye çıkarmaktır.

Esasen Skip-Gram şunu sorar: "Bu kelime verildiğinde, bir cümlede etrafında en çok hangi kelimelerin görünmesi muhtemeldir?"

<a name="322-güçlü-ve-zayıf-yönleri"></a>
##### 3.2.2. Güçlü ve Zayıf Yönleri
**Skip-Gram'ın Güçlü Yönleri:**
*   **Seyrek Kelimeler İçin Daha İyi:** Skip-Gram, nadir kelimeler veya kelime öbekleri için iyi temsiller öğrenmede oldukça etkilidir. Her hedef kelime için birden fazla bağlam kelimesini tahmin ettiğinden, nadir bir hedef kelime, çeşitli bağlam tahminleri aracılığıyla vektörünü güncellemek için daha fazla fırsat bulur.
*   **Anlamsal Nüansları Yakalar:** Genellikle daha ince anlamsal ilişkileri ve nüansları yakalamada daha iyi performans gösterir, özellikle veri kümesi daha küçükse veya birçok nadir terim içeren zengin bir kelime dağarcığına sahipse.
*   **Daha Küçük Veri Kümelerinde Etkili:** Genellikle CBOW'a kıyasla daha küçük eğitim veri kümeleriyle üstün performans gösterir.

**Skip-Gram'ın Zayıf Yönleri:**
*   **Daha Yavaş Eğitim Hızı:** Skip-Gram, CBOW'dan daha fazla hesaplama yoğundur ve daha yavaş eğitilir. Her hedef kelime için, penceredeki her tahmin edilen bağlam kelimesi için güncellemeler yapması gerekir ki bu da çok sayıda olabilir.
*   **Daha Yüksek Hesaplama Maliyeti:** Birden çok tahmin ve güncelleme nedeniyle, özellikle büyük pencere boyutları ve geniş kelime dağarcıkları ile hesaplama maliyeti daha yüksek olabilir.

<a name="33-temel-farklılıklar-ve-kullanım-senaryoları"></a>
#### 3.3. Temel Farklılıklar ve Kullanım Senaryoları
CBOW ve Skip-Gram arasındaki temel fark, tahmin görevlerinde yatmaktadır:

*   **CBOW:** Bağlam kelimelerinden hedef kelimeyi tahmin eder. (Bağlam -> Hedef)
*   **Skip-Gram:** Hedef kelimeden bağlam kelimelerini tahmin eder. (Hedef -> Bağlam)

Bu ayrım, farklı performans özelliklerine ve tercih edilen kullanım senaryolarına yol açar:

| Özellik           | CBOW                                              | Skip-Gram                                            |
| :---------------- | :------------------------------------------------ | :--------------------------------------------------- |
| **Tahmin Görevi** | Bağlamdan orta kelimeyi tahmin eder               | Orta kelimeden bağlam kelimelerini tahmin eder       |
| **Eğitim Hızı**   | Daha hızlı (özellikle büyük veri kümelerinde)     | Daha yavaş                                           |
| **Doğruluk**      | Sık kullanılan kelimelerde iyi                    | Seyrek kelimeler/kelime öbeklerinde iyi; zengin anlamsallar için genel olarak daha yüksek doğruluk |
| **Veri Boyutu**   | Çok büyük veri kümeleri için daha iyi             | Daha küçük veri kümeleri için daha iyi (az örnekten öğrenebilir) |
| **Hesaplama Maliyeti** | Her eğitim adımında daha düşük                | Her eğitim adımında daha yüksek                      |

**CBOW ve Skip-Gram Arasında Seçim Yapmak:**
*   **CBOW** genellikle veri kümesi çok büyük olduğunda ve temel amaç, yaygın kelimelere odaklanarak kelime gömülerini hızlı bir şekilde elde etmek olduğunda tercih edilir. Hızın kritik olduğu ve kelime dağarcığının kesin gömülerinin çok önemli olduğu nadir kelimelerin önemli bir oranını içermediği görevler için sıklıkla seçilir.
*   **Skip-Gram** genellikle daha yüksek doğruluk istendiğinde, özellikle nadir kelimelerin anlamsal ilişkilerini anlamak için veya daha küçük veri kümeleriyle çalışırken önerilir. Karmaşık anlamsal analojiler veya anlam belirsizliği gibi bireysel kelime anlamlarının daha incelikli bir şekilde anlaşılmasını gerektiren görevlerde öne çıkar.

Seçim nihayetinde NLP görevinin özel gereksinimlerine, eğitim korpusunun özelliklerine (boyut, kelime dağarcığı zenginliği, frekans dağılımı) ve mevcut hesaplama kaynaklarına bağlıdır.

<a name="4-kod-örneği"></a>
### 4. Kod Örneği
Bu Python kod parçacığı, `gensim` kütüphanesini kullanarak hem CBOW hem de Skip-Gram modellerinin nasıl başlatılacağını ve kavramsal olarak eğitileceğini göstermektedir. İki mimariyi ayıran `sg` parametresini açıklar ve kelime vektörlerine nasıl erişileceğini ve benzer kelimelerin nasıl bulunacağını gösterir.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# word_tokenize için NLTK punkt tokenizatörünün mevcut olduğundan emin olun
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Örnek metin verisi (demonstrasyon için küçük bir korpus)
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast animal, the fox, is known for its agility.",
    "Dogs are often loyal companions.",
    "Cats are also popular pets."
]

# Cümleleri kelimelere ayırma. Word2Vec, kelime listelerinin listesini bekler.
# Tutarlı bir kelime dağarcığı için küçük harfe dönüştürün.
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

print("Eğitim için token'lara ayrılmış cümleler:")
for s in tokenized_sentences:
    print(s)

# --- CBOW model eğitimi ---
print("\n--- Sürekli Kelime Torbası (CBOW) model eğitimi ---")
# sg=0, CBOW mimarisini belirtir (varsayılan)
# vector_size: Kelime vektörlerinin boyutsallığı (örn. 100 boyut)
# window: Bir cümle içinde mevcut ve tahmin edilen kelime arasındaki maksimum mesafe
# min_count: Toplam frekansı bundan daha düşük olan tüm kelimeleri yoksayar. Bu küçük korpus için 1 olarak ayarlandı.
# workers: Model eğitimi için kullanılacak işçi iş parçacığı sayısı (= daha hızlı eğitim)
cbow_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100, # 100 boyutlu kelime vektörleri
    window=5,      # her iki tarafta 5 kelimelik bağlam penceresi
    min_count=1,   # bu küçük korpusta nadir olanlar da dahil tüm kelimeleri dahil et
    sg=0,          # CBOW mimarisi
    workers=4      # eğitim için 4 iş parçacığı kullan
)
print("CBOW model eğitimi tamamlandı.")

# CBOW modeli için kelime vektörlerine erişin ve benzer kelimeleri bulun
print(f"'fox' kelimesinin vektörü (CBOW, ilk 5 eleman): {cbow_model.wv['fox'][:5]}...")
print("\n'fox' kelimesine benzer kelimeler (CBOW modeli):")
try:
    for word, score in cbow_model.wv.most_similar('fox', topn=3):
        print(f"  {word}: {score:.4f}")
except KeyError:
    print(" 'fox' kelime dağarcığında yok. Farklı bir korpus kullanıyorsanız min_count değerini ayarlayın.")


# --- Skip-Gram model eğitimi ---
print("\n--- Skip-Gram model eğitimi ---")
# sg=1, Skip-Gram mimarisini belirtir
skipgram_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100, # 100 boyutlu kelime vektörleri
    window=5,      # her iki tarafta 5 kelimelik bağlam penceresi
    min_count=1,   # tüm kelimeleri dahil et
    sg=1,          # Skip-Gram mimarisi
    workers=4
)
print("Skip-Gram model eğitimi tamamlandı.")

# Skip-Gram modeli için kelime vektörlerine erişin ve benzer kelimeleri bulun
print(f"'fox' kelimesinin vektörü (Skip-Gram, ilk 5 eleman): {skipgram_model.wv['fox'][:5]}...")
print("\n'fox' kelimesine benzer kelimeler (Skip-Gram modeli):")
try:
    for word, score in skipgram_model.wv.most_similar('fox', topn=3):
        print(f"  {word}: {score:.4f}")
except KeyError:
    print(" 'fox' kelime dağarcığında yok. Farklı bir korpus kullanıyorsanız min_count değerini ayarlayın.")

# Not: Bunun gibi küçük bir korpusta benzerlik sonuçları anlamsal olarak anlamlı olmayacaktır,
# ancak örnek her iki mimarinin API kullanımını göstermektedir.

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
### 5. Sonuç
Word2Vec, kelimeleri yoğun, anlamlı vektörler olarak temsil ederek makinelerin insan dilini anlama ve işleme şeklini temelden dönüştüren Doğal Dil İşleme alanında bir dönüm noktası yenilik olarak durmaktadır. Çerçevenin iki ana mimarisi olan **Sürekli Kelime Torbası (CBOW)** ve **Skip-Gram**, her biri kendi avantaj ve dezavantajlarına sahip, bu **kelime gömülerini** öğrenmek için farklı ancak güçlü yaklaşımlar sunar.

CBOW, hedef bir kelimeyi çevreleyen bağlamından tahmin ederek, hesaplama verimliliği ve sık kullanılan kelimelerdeki performansıyla öne çıkar, bu da hızı öncelikli olan büyük veri kümeleri için uygun bir seçim olmasını sağlar. Tersine, Skip-Gram, hedef bir kelimeden bağlam kelimelerini tahmin ederek, daha az sıklıkta kullanılan kelimelerin nüanslarını yakalamada üstün yetenek gösterir ve genellikle anlamsal temsilde daha yüksek genel doğruluk sağlar; bu, özellikle daha küçük korpuslar veya ayrıntılı anlamsal anlayış gerektiren görevler için değerlidir.

CBOW ve Skip-Gram arasındaki seçim mutlak değil, eğitim verilerinin özelliklerine, NLP görevinin özel gereksinimlerine ve mevcut hesaplama kaynaklarına göre yönlendirilen stratejik bir karardır. Her iki mimari de kelime temsili öğrenimindeki sonraki gelişmeler için kritik bir temel atmış, GloVe ve FastText gibi modelleri etkilemiş ve modern Üretken Yapay Zeka ve derin öğrenme tabanlı dil modellerinin geniş manzarasında ayrılmaz bileşenler olmaya devam etmektedir. Mekanizmalarını ve ödünleşimlerini anlamak, makine dilini anlamanın karmaşıklıklarına dalan herkes için temeldir.

<a name="6-kaynaklar"></a>
### 6. Kaynaklar
*   Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient estimation of word representations in vector space*. arXiv preprint arXiv:1301.3781.
*   Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed representations of words and phrases and their compositionality*. Advances in neural information processing systems, 26.





