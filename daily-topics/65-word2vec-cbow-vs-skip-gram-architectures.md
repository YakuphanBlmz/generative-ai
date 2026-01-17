# Word2Vec: CBOW vs. Skip-Gram Architectures

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Word Embeddings and Word2Vec](#2-word-embeddings-and-word2vec)
- [3. CBOW Architecture](#3-cbow-architecture)
    - [3.1. Mechanism](#31-mechanism)
    - [3.2. Advantages and Disadvantages](#32-advantages-and-disadvantages)
- [4. Skip-Gram Architecture](#4-skip-gram-architecture)
    - [4.1. Mechanism](#41-mechanism)
    - [4.2. Advantages and Disadvantages](#42-advantages-and-disadvantages)
- [5. CBOW vs. Skip-Gram: A Comparative Analysis](#5-cbow-vs-skip-gram-a-comparative-analysis)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

## 1. Introduction

The ability of machines to understand and process human language has been a long-standing goal in Artificial Intelligence. A foundational step in achieving this is the effective representation of words in a way that captures their semantic and syntactic relationships. Traditional methods, such as one-hot encoding, treat each word as an independent entity, failing to convey any meaning beyond simple identity. This limitation led to the development of **word embeddings**, which are dense vector representations of words in a continuous vector space. These vectors are designed such that words with similar meanings are positioned closely together in the embedding space.

Among the most influential algorithms for learning word embeddings is **Word2Vec**, introduced by Mikolov et al. in 2013. Word2Vec offers an efficient computational framework for learning high-quality word representations from large corpora. It comprises two distinct model architectures: the **Continuous Bag-of-Words (CBOW)** model and the **Skip-Gram** model. While both aim to learn effective word embeddings, they achieve this through different predictive tasks and, consequently, exhibit varying strengths and weaknesses. This document will provide a comprehensive examination of both CBOW and Skip-Gram architectures, detailing their underlying mechanisms, discussing their respective advantages and disadvantages, and offering a comparative analysis to elucidate their suitable applications.

## 2. Word Embeddings and Word2Vec

**Word embeddings** are low-dimensional, dense vector representations of words, where each dimension captures a latent semantic or syntactic feature of the word. Unlike sparse representations like one-hot vectors, which scale poorly with vocabulary size and inherently lack semantic information, word embeddings allow for mathematical operations (e.g., vector addition and subtraction) to reveal interesting semantic relationships, such as "king - man + woman = queen".

**Word2Vec** is a particularly effective and computationally efficient framework for learning these embeddings. At its core, Word2Vec relies on the distributional hypothesis, which states that words that appear in similar contexts tend to have similar meanings. Both CBOW and Skip-Gram models leverage this hypothesis by training a shallow neural network to predict words based on their contexts, or vice versa, thereby learning the word vectors as a side product of this prediction task. The final learned weights of the hidden layer in these networks serve as the word embeddings. This unsupervised learning approach enables Word2Vec to process vast amounts of text data without requiring labeled datasets, making it highly adaptable to diverse linguistic tasks.

## 3. CBOW Architecture

The **Continuous Bag-of-Words (CBOW)** architecture aims to predict a target word based on its surrounding context words. The "bag-of-words" aspect refers to the fact that the order of words in the context window does not influence the prediction; only their presence matters.

### 3.1. Mechanism

In the CBOW model, the input layer receives multiple one-hot encoded context words within a fixed-size window around the target word. These one-hot vectors are then projected into a shared continuous embedding space through a single hidden layer, which essentially involves looking up the word embeddings for each context word. The vectors for all context words are then averaged or summed to create a single context vector. This averaged vector is then fed into an output layer, typically a softmax layer, which predicts the probability distribution over the entire vocabulary for the target word. The objective function is to maximize the probability of the actual target word given the context. During training, the weights of the input-to-hidden layer (which are the word embeddings) and the hidden-to-output layer are adjusted using backpropagation and an optimization algorithm like stochastic gradient descent.

The training process involves:
1.  **Input Layer:** One-hot vectors of context words.
2.  **Projection Layer:** Each one-hot vector is multiplied by an input weight matrix (embedding matrix) `W`, yielding the word vector for each context word.
3.  **Averaging Layer:** The word vectors of all context words are averaged to form a single context vector `h`.
4.  **Output Layer:** The context vector `h` is multiplied by an output weight matrix `W'`, and the result is passed through a softmax function to produce the probability distribution of the target word.
5.  **Loss Calculation & Backpropagation:** The model adjusts `W` and `W'` to minimize the loss (e.g., negative log-likelihood) between the predicted and actual target word.

### 3.2. Advantages and Disadvantages

**Advantages of CBOW:**
*   **Faster Training:** CBOW tends to train faster than Skip-Gram, especially for large datasets, because it predicts a single target word from multiple context words, effectively processing fewer prediction tasks per training sample.
*   **Good for Frequent Words:** It generally performs well with frequent words, as it averages the context vectors, which helps in regularizing the embeddings of common words by smoothing out noise.
*   **Better for Syntactic Tasks:** Some research suggests CBOW might be slightly better at capturing syntactic regularities due to its focus on predicting a word from its surrounding words.

**Disadvantages of CBOW:**
*   **Less Effective for Rare Words:** CBOW struggles to learn good representations for rare words. Since it averages context representations, unique contextual information for infrequent words might be diluted, leading to less distinct embeddings.
*   **Limited Semantic Capture for Specificity:** While good for frequent words and syntactic tasks, it might not capture the nuanced semantic relationships as effectively as Skip-Gram, especially for highly specific or polysemous words, as it collapses all context into a single representation.

## 4. Skip-Gram Architecture

In contrast to CBOW, the **Skip-Gram** architecture takes a target word as input and aims to predict its surrounding context words. This approach effectively reverses the prediction task of CBOW.

### 4.1. Mechanism

The Skip-Gram model takes a single one-hot encoded target word as input. This input word is then projected into a continuous embedding space via an input weight matrix `W`, similar to the CBOW model, yielding its word vector. This word vector is then used to predict multiple context words within a predefined window. Specifically, the model generates a separate prediction for each context word in the window. The output layer, typically using softmax, then outputs a probability distribution for each context word. The objective is to maximize the sum of the log probabilities of all context words given the central word.

The training process involves:
1.  **Input Layer:** One-hot vector of the target word.
2.  **Projection Layer:** The one-hot vector is multiplied by an input weight matrix `W` to obtain the word vector for the target word. This vector is the hidden layer activation.
3.  **Output Layer:** The word vector is then multiplied by an output weight matrix `W'`, and the result is passed through a softmax function to predict the probability distribution for *each* context word. This is repeated for every context word in the window.
4.  **Loss Calculation & Backpropagation:** The model adjusts `W` and `W'` to minimize the combined loss (sum of negative log-likelihoods) for all predicted context words.

To mitigate the computational cost of the softmax function over the entire vocabulary, Word2Vec often employs optimization techniques such as **Negative Sampling** or **Hierarchical Softmax**. Negative Sampling, in particular, transforms the multiclass classification problem into a set of binary classification problems, significantly speeding up training by only updating a small subset of weights for each training sample.

### 4.2. Advantages and Disadvantages

**Advantages of Skip-Gram:**
*   **Excellent for Rare Words:** Skip-Gram is highly effective at learning representations for rare words and phrases. Because it predicts context words from a single target word, it can capture more nuanced semantic information, even for words that appear infrequently.
*   **Better for Semantic Regularities:** It is generally superior at capturing semantic regularities, as evidenced by its ability to discover complex relationships (e.g., "Paris is to France as Berlin is to Germany").
*   **Handles Specificity Better:** By predicting each context word individually, Skip-Gram can better handle polysemous words and capture more specific semantic relationships.

**Disadvantages of Skip-Gram:**
*   **Slower Training:** Skip-Gram typically trains slower than CBOW, especially with large datasets and large context windows, because it makes multiple predictions (one for each context word) for every target word.
*   **Computational Cost:** Without optimization techniques like Negative Sampling, the computational cost of the output layer (calculating softmax over the entire vocabulary) can be substantial.

## 5. CBOW vs. Skip-Gram: A Comparative Analysis

The choice between CBOW and Skip-Gram largely depends on the specific application and the characteristics of the training data. Both models contribute significantly to the field of natural language processing by providing effective word embeddings, yet their architectural differences lead to distinct performance profiles.

| Feature               | CBOW (Continuous Bag-of-Words)                                   | Skip-Gram                                                         |
| :-------------------- | :--------------------------------------------------------------- | :---------------------------------------------------------------- |
| **Input**             | Context words                                                    | Target word                                                       |
| **Output**            | Target word                                                      | Context words                                                     |
| **Prediction Task**   | Predict current word from context                                | Predict context words from current word                           |
| **Training Speed**    | Faster, especially on large datasets                             | Slower, as it makes multiple predictions per training instance    |
| **Performance w/ Rare Words** | Less effective; context averaging dilutes information        | More effective; captures nuanced semantics for rare words        |
| **Semantic Regularities** | Good for syntactic tasks, less so for nuanced semantics         | Excellent for capturing semantic relationships and analogies      |
| **Data Size Preference** | Larger datasets where speed is critical                          | Smaller datasets where high-quality embeddings for all words are needed |
| **Computational Complexity** | Input averaging makes it simpler per sample                   | Multiple output predictions per sample increase complexity         |

In essence, if the corpus is very large and contains many frequent words, and the primary goal is to achieve fast training with reasonable quality embeddings, **CBOW** might be preferred. It performs well by leveraging the frequent co-occurrence patterns. However, if the corpus contains a significant number of rare words or if capturing fine-grained semantic relationships and analogies is crucial, **Skip-Gram** is generally the superior choice, despite its slower training speed. The additional computational cost often pays off in the quality of the learned embeddings, particularly for less frequent terms. Both architectures, when combined with optimization techniques like Negative Sampling, offer robust solutions for generating powerful word representations.

## 6. Code Example

The following Python code snippet illustrates a conceptual preprocessing step for Word2Vec, specifically demonstrating how context-target pairs can be generated from a sentence for a given window size. This is a foundational step before feeding data into either CBOW or Skip-Gram models.

```python
import collections

def generate_word2vec_pairs(text, window_size):
    """
    Generates (context, target) pairs for CBOW or (target, context) pairs for Skip-Gram.
    This example focuses on creating the raw data structure.

    Args:
        text (list): A list of words (tokenized sentence).
        window_size (int): The size of the context window on each side of the target word.

    Returns:
        list: A list of tuples, where each tuple is (input_word(s), output_word(s)).
    """
    data = []
    text_length = len(text)

    for i, target_word in enumerate(text):
        # Determine the start and end indices for the context window
        start_index = max(0, i - window_size)
        end_index = min(text_length, i + window_size + 1)
        
        # Extract context words, excluding the target word itself
        context_words = [text[j] for j in range(start_index, end_index) if j != i]
        
        if context_words: # Only add if there are context words
            # For CBOW, input is context, output is target
            # For Skip-Gram, input is target, output is each context word (multiple pairs)
            
            # This example structure is more for conceptual data generation:
            # CBOW conceptual pair: (context_words_list, target_word)
            data.append((context_words, target_word)) # Representing one CBOW-like sample

            # Skip-Gram conceptual pairs: (target_word, context_word_1), (target_word, context_word_2), ...
            # For brevity, we'll only show the CBOW-like structure explicitly in the output.
            # In a real Skip-Gram implementation, you'd iterate through context_words
            # and create (target_word, single_context_word) pairs.
            
    return data

# Example usage:
sentence = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
window = 2
generated_pairs = generate_word2vec_pairs(sentence, window)

print(f"Original sentence: {sentence}\n")
print(f"Generated (context, target) conceptual pairs (window={window}):")
for pair in generated_pairs:
    print(f"  Context: {pair[0]}, Target: {pair[1]}")


(End of code example section)
```
## 7. Conclusion

Word2Vec has revolutionized the field of natural language processing by providing a highly effective and efficient method for learning dense, continuous word embeddings. The two primary architectures within the Word2Vec framework, CBOW and Skip-Gram, approach the task of learning word representations from different angles. CBOW excels at predicting a target word from its surrounding context, making it generally faster to train and particularly effective for frequent words in large corpora. Its averaging of context vectors can, however, dilute the specificity for rare words. Conversely, Skip-Gram predicts context words from a given target word, demonstrating superior performance in capturing nuanced semantic relationships and producing high-quality embeddings for rare words, albeit at the cost of slower training times.

The selection between CBOW and Skip-Gram is a pragmatic decision, contingent upon factors such as corpus size, computational resources, and the desired quality of embeddings for frequent versus infrequent terms. Both models, especially when augmented with optimizations like Negative Sampling, have proven indispensable tools in modern NLP, underpinning advances in machine translation, sentiment analysis, information retrieval, and various other language understanding tasks. Their enduring legacy lies in demonstrating the power of simple neural architectures to uncover complex linguistic patterns from raw text data, paving the way for more sophisticated deep learning models in the domain of generative AI.
---
<br>

<a name="türkçe-içerik"></a>
## Word2Vec: CBOW ve Skip-Gram Mimarileri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Kelime Gömülmeleri ve Word2Vec](#2-kelime-gömülmeleri-ve-word2vec)
- [3. CBOW Mimarisi](#3-cbow-mimarisi)
    - [3.1. Mekanizma](#31-mekanizma)
    - [3.2. Avantajları ve Dezavantajları](#32-avantajları-ve-dezavantajları)
- [4. Skip-Gram Mimarisi](#4-skip-gram-mimarisi)
    - [4.1. Mekanizma](#41-mekanizma)
    - [4.2. Avantajları ve Dezavantajları](#42-avantajları-ve-dezavantajları)
- [5. CBOW ve Skip-Gram: Karşılaştırmalı Bir Analiz](#5-cbow-ve-skip-gram-karşılaştırmalı-bir-analiz)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

## 1. Giriş

Makinelerin insan dilini anlama ve işleme yeteneği, Yapay Zeka alanında uzun süredir devam eden bir hedeftir. Bunu başarmanın temel adımlarından biri, kelimelerin anlamsal ve sözdizimsel ilişkilerini yakalayacak şekilde etkili bir temsilidir. Geleneksel yöntemler, tek-hot kodlama gibi, her kelimeyi basit bir kimliğin ötesinde herhangi bir anlam ifade etmeyen bağımsız bir varlık olarak ele alır. Bu sınırlama, kelimelerin sürekli bir vektör uzayında yoğun vektör temsillerini ifade eden **kelime gömülmelerinin (word embeddings)** geliştirilmesine yol açmıştır. Bu vektörler, benzer anlamlara sahip kelimelerin gömme uzayında birbirine yakın konumlandırılmasını sağlayacak şekilde tasarlanmıştır.

Kelime gömülmelerini öğrenmek için en etkili algoritmalardan biri, Mikolov ve arkadaşları tarafından 2013 yılında tanıtılan **Word2Vec**'tir. Word2Vec, büyük metin koleksiyonlarından (korpuslardan) yüksek kaliteli kelime temsillerini öğrenmek için etkili bir hesaplama çerçevesi sunar. İki farklı model mimarisi içerir: **Sürekli Kelime Çantası (Continuous Bag-of-Words - CBOW)** modeli ve **Skip-Gram** modeli. Her ikisi de etkili kelime gömülmeleri öğrenmeyi amaçlarken, bunu farklı tahmin görevleri aracılığıyla gerçekleştirirler ve dolayısıyla farklı güçlü ve zayıf yönler sergilerler. Bu belge, hem CBOW hem de Skip-Gram mimarilerini kapsamlı bir şekilde inceleyecek, temel mekanizmalarını detaylandıracak, ilgili avantaj ve dezavantajlarını tartışacak ve uygun uygulamalarını aydınlatmak için karşılaştırmalı bir analiz sunacaktır.

## 2. Kelime Gömülmeleri ve Word2Vec

**Kelime gömülmeleri**, kelimelerin düşük boyutlu, yoğun vektör temsilleridir; burada her boyut, kelimenin gizli bir anlamsal veya sözdizimsel özelliğini yakalar. Kelime dağarcığı boyutuyla kötü ölçeklenen ve doğası gereği anlamsal bilgiden yoksun olan tek-hot vektörler gibi seyrek temsillerin aksine, kelime gömülmeleri, "kral - erkek + kadın = kraliçe" gibi ilginç anlamsal ilişkileri ortaya çıkarmak için matematiksel işlemlere (örn. vektör toplama ve çıkarma) izin verir.

**Word2Vec**, özellikle bu gömülmeleri öğrenmek için etkili ve hesaplama açısından verimli bir çerçevedir. Özünde, Word2Vec, benzer bağlamlarda görünen kelimelerin benzer anlamlara sahip olduğu şeklindeki dağılımsal hipoteze dayanır. Hem CBOW hem de Skip-Gram modelleri, kelimeleri bağlamlarına göre veya tersi şekilde tahmin etmek için sığ bir sinir ağı eğiterek bu hipotezi kullanır ve böylece kelime vektörlerini bu tahmin görevinin bir yan ürünü olarak öğrenir. Bu ağlardaki gizli katmanın nihai öğrenilen ağırlıkları, kelime gömülmeleri olarak hizmet eder. Bu denetimsiz öğrenme yaklaşımı, Word2Vec'in etiketli veri kümelerine ihtiyaç duymadan büyük miktarda metin verisini işlemesine olanak tanıyarak, çeşitli dilbilimsel görevlere yüksek ölçüde uyarlanabilir olmasını sağlar.

## 3. CBOW Mimarisi

**Sürekli Kelime Çantası (CBOW)** mimarisi, hedef bir kelimeyi çevresindeki bağlam kelimelerine dayanarak tahmin etmeyi amaçlar. "Kelime çantası" yönü, bağlam penceresindeki kelimelerin sırasının tahmini etkilemediği; yalnızca varlıklarının önemli olduğu gerçeğini ifade eder.

### 3.1. Mekanizma

CBOW modelinde, girdi katmanı, hedef kelimenin etrafındaki sabit boyutlu bir pencere içindeki birden çok tek-hot kodlanmış bağlam kelimesini alır. Bu tek-hot vektörler daha sonra tek bir gizli katman aracılığıyla paylaşılan sürekli bir gömme uzayına yansıtılır, bu da esasen her bağlam kelimesi için kelime gömülmelerini aramayı içerir. Tüm bağlam kelimeleri için vektörler daha sonra tek bir bağlam vektörü oluşturmak üzere ortalama alınır veya toplanır. Bu ortalama alınmış vektör daha sonra, tipik olarak bir softmax katmanı olan bir çıktı katmanına beslenir ve bu katman, hedef kelime için tüm kelime dağarcığı üzerindeki olasılık dağılımını tahmin eder. Amaç fonksiyonu, bağlam verildiğinde gerçek hedef kelimenin olasılığını maksimize etmektir. Eğitim sırasında, girdi-gizli katman (kelime gömülmeleri olan) ve gizli-çıktı katmanının ağırlıkları, geri yayılım ve stokastik gradyan inişi gibi bir optimizasyon algoritması kullanılarak ayarlanır.

Eğitim süreci şunları içerir:
1.  **Girdi Katmanı:** Bağlam kelimelerinin tek-hot vektörleri.
2.  **Yansıtma Katmanı:** Her tek-hot vektör, bir girdi ağırlık matrisi (gömme matrisi) `W` ile çarpılarak her bağlam kelimesi için kelime vektörünü verir.
3.  **Ortalama Alma Katmanı:** Tüm bağlam kelimelerinin kelime vektörleri ortalama alınarak tek bir bağlam vektörü `h` oluşturulur.
4.  **Çıktı Katmanı:** Bağlam vektörü `h`, bir çıktı ağırlık matrisi `W'` ile çarpılır ve sonuç, hedef kelimenin olasılık dağılımını üretmek için bir softmax fonksiyonundan geçirilir.
5.  **Kayıp Hesaplama ve Geri Yayılım:** Model, tahmin edilen ve gerçek hedef kelime arasındaki kaybı (örn. negatif log-olabilirlik) minimize etmek için `W` ve `W'` ağırlıklarını ayarlar.

### 3.2. Avantajları ve Dezavantajları

**CBOW'un Avantajları:**
*   **Daha Hızlı Eğitim:** CBOW, özellikle büyük veri kümeleri için Skip-Gram'dan daha hızlı eğitim eğilimindedir, çünkü birden çok bağlam kelimesinden tek bir hedef kelimeyi tahmin eder ve böylece her eğitim örneği başına daha az tahmin görevi işler.
*   **Sık Kullanılan Kelimeler İçin İyi:** Bağlam vektörlerini ortalaması nedeniyle sık kullanılan kelimelerle genellikle iyi performans gösterir, bu da yaygın kelimelerin gömülmelerini gürültüyü gidererek düzenlemeye yardımcı olur.
*   **Sözdizimsel Görevler İçin Daha İyi:** Bazı araştırmalar, CBOW'un çevresindeki kelimelerden bir kelimeyi tahmin etmeye odaklanması nedeniyle sözdizimsel düzenlilikleri yakalamada biraz daha iyi olabileceğini öne sürmektedir.

**CBOW'un Dezavantajları:**
*   **Nadir Kelimeler İçin Daha Az Etkili:** CBOW, nadir kelimeler için iyi temsiller öğrenmekte zorlanır. Bağlam temsillerini ortalaması nedeniyle, seyrek kelimeler için benzersiz bağlamsal bilgiler seyreltilebilir ve bu da daha az belirgin gömülmelere yol açabilir.
*   **Spesifiklik İçin Sınırlı Anlamsal Yakalama:** Sık kullanılan kelimeler ve sözdizimsel görevler için iyi olsa da, özellikle yüksek oranda spesifik veya çok anlamlı kelimeler için, tüm bağlamı tek bir temsile indirgediği için anlamsal ilişkileri Skip-Gram kadar etkili bir şekilde yakalayamayabilir.

## 4. Skip-Gram Mimarisi

CBOW'un aksine, **Skip-Gram** mimarisi girdi olarak bir hedef kelimeyi alır ve çevresindeki bağlam kelimelerini tahmin etmeyi amaçlar. Bu yaklaşım, CBOW'un tahmin görevini etkili bir şekilde tersine çevirir.

### 4.1. Mekanizma

Skip-Gram modeli, girdi olarak tek bir tek-hot kodlanmış hedef kelimeyi alır. Bu girdi kelimesi daha sonra, CBOW modeline benzer şekilde, bir girdi ağırlık matrisi `W` aracılığıyla sürekli bir gömme uzayına yansıtılır ve kelime vektörünü verir. Bu kelime vektörü daha sonra önceden tanımlanmış bir pencere içindeki birden çok bağlam kelimesini tahmin etmek için kullanılır. Özellikle, model penceredeki her bağlam kelimesi için ayrı bir tahmin üretir. Çıktı katmanı, tipik olarak softmax kullanarak, her bağlam kelimesi için bir olasılık dağılımı çıkarır. Amaç, merkezi kelime verildiğinde tüm bağlam kelimelerinin log olasılıklarının toplamını maksimize etmektir.

Eğitim süreci şunları içerir:
1.  **Girdi Katmanı:** Hedef kelimenin tek-hot vektörü.
2.  **Yansıtma Katmanı:** Tek-hot vektör, hedef kelime için kelime vektörünü elde etmek üzere bir girdi ağırlık matrisi `W` ile çarpılır. Bu vektör, gizli katman aktivasyonudur.
3.  **Çıktı Katmanı:** Kelime vektörü daha sonra bir çıktı ağırlık matrisi `W'` ile çarpılır ve sonuç, *her* bağlam kelimesi için olasılık dağılımını tahmin etmek üzere bir softmax fonksiyonundan geçirilir. Bu, penceredeki her bağlam kelimesi için tekrarlanır.
4.  **Kayıp Hesaplama ve Geri Yayılım:** Model, tahmin edilen tüm bağlam kelimeleri için birleşik kaybı (negatif log-olabilirliklerin toplamı) minimize etmek üzere `W` ve `W'` ağırlıklarını ayarlar.

Tüm kelime dağarcığı üzerindeki softmax fonksiyonunun hesaplama maliyetini azaltmak için, Word2Vec genellikle **Negatif Örnekleme (Negative Sampling)** veya **Hiyerarşik Softmax (Hierarchical Softmax)** gibi optimizasyon tekniklerini kullanır. Özellikle Negatif Örnekleme, çok sınıflı sınıflandırma problemini bir dizi ikili sınıflandırma problemine dönüştürerek, her eğitim örneği için yalnızca küçük bir ağırlık alt kümesini güncelleyerek eğitimi önemli ölçüde hızlandırır.

### 4.2. Avantajları ve Dezavantajları

**Skip-Gram'ın Avantajları:**
*   **Nadir Kelimeler İçin Mükemmel:** Skip-Gram, nadir kelimeler ve ifadeler için temsilleri öğrenmede oldukça etkilidir. Tek bir hedef kelimeden bağlam kelimelerini tahmin ettiği için, seyrek görünen kelimeler için bile daha incelikli anlamsal bilgileri yakalayabilir.
*   **Anlamsal Düzenlilikler İçin Daha İyi:** Genellikle, karmaşık ilişkileri (örn. "Paris Fransa'ya, Berlin Almanya'ya") keşfetme yeteneği ile kanıtlandığı gibi, anlamsal düzenlilikleri yakalamada daha üstündür.
*   **Spesifikliği Daha İyi Yönetir:** Her bağlam kelimesini ayrı ayrı tahmin ederek, Skip-Gram çok anlamlı kelimeleri daha iyi işleyebilir ve daha spesifik anlamsal ilişkileri yakalayabilir.

**Skip-Gram'ın Dezavantajları:**
*   **Daha Yavaş Eğitim:** Skip-Gram, özellikle büyük veri kümeleri ve geniş bağlam pencereleri ile, her hedef kelime için birden çok tahmin (her bağlam kelimesi için bir tane) yaptığı için CBOW'dan daha yavaş eğitim eğilimindedir.
*   **Hesaplama Maliyeti:** Negatif Örnekleme gibi optimizasyon teknikleri olmadan, çıktı katmanının hesaplama maliyeti (tüm kelime dağarcığı üzerinde softmax hesaplama) önemli olabilir.

## 5. CBOW ve Skip-Gram: Karşılaştırmalı Bir Analiz

CBOW ve Skip-Gram arasındaki seçim büyük ölçüde belirli uygulamaya ve eğitim verilerinin özelliklerine bağlıdır. Her iki model de etkili kelime gömülmeleri sağlayarak doğal dil işleme alanına önemli katkılarda bulunsa da, mimari farklılıkları farklı performans profillerine yol açar.

| Özellik                | CBOW (Sürekli Kelime Çantası)                                    | Skip-Gram                                                        |
| :--------------------- | :--------------------------------------------------------------- | :--------------------------------------------------------------- |
| **Girdi**              | Bağlam kelimeleri                                                | Hedef kelime                                                     |
| **Çıktı**              | Hedef kelime                                                     | Bağlam kelimeleri                                                |
| **Tahmin Görevi**      | Bağlamdan mevcut kelimeyi tahmin etme                           | Mevcut kelimeden bağlam kelimelerini tahmin etme                |
| **Eğitim Hızı**        | Daha hızlı, özellikle büyük veri kümelerinde                     | Daha yavaş, her eğitim örneği için birden çok tahmin yapar       |
| **Nadir Kelimelerle Performans** | Daha az etkili; bağlam ortalaması bilgiyi seyreltir          | Daha etkili; nadir kelimeler için incelikli anlamları yakalar    |
| **Anlamsal Düzenlilikler** | Sözdizimsel görevler için iyi, incelikli anlamlar için daha az  | Anlamsal ilişkileri ve analojileri yakalamada mükemmel           |
| **Veri Boyutu Tercihi** | Hızın kritik olduğu daha büyük veri kümeleri                     | Tüm kelimeler için yüksek kaliteli gömülmelerin gerektiği daha küçük veri kümeleri |
| **Hesaplama Karmaşıklığı** | Girdi ortalaması her örnek başına daha basittir               | Her örnek başına birden çok çıktı tahmini karmaşıklığı artırır   |

Özünde, eğer korpus çok büyükse ve birçok sık kullanılan kelime içeriyorsa ve birincil amaç makul kaliteli gömülmelerle hızlı eğitim sağlamaksa, **CBOW** tercih edilebilir. Sık eşdizimlilik kalıplarını kullanarak iyi performans gösterir. Ancak, eğer korpus önemli sayıda nadir kelime içeriyorsa veya ince taneli anlamsal ilişkileri ve analojileri yakalamak kritikse, daha yavaş eğitim hızına rağmen **Skip-Gram** genellikle üstün bir seçimdir. Ek hesaplama maliyeti, özellikle daha az sık kullanılan terimler için öğrenilen gömülmelerin kalitesinde genellikle kendini gösterir. Her iki mimari de, Negatif Örnekleme gibi optimizasyon teknikleriyle birleştirildiğinde, güçlü kelime temsilleri oluşturmak için sağlam çözümler sunar.

## 6. Kod Örneği

Aşağıdaki Python kod parçacığı, Word2Vec için kavramsal bir ön işleme adımını göstermektedir; belirli bir pencere boyutu için bir cümleden bağlam-hedef çiftlerinin nasıl oluşturulabileceğini özellikle sergiler. Bu, verileri CBOW veya Skip-Gram modellerine beslemeden önceki temel bir adımdır.

```python
import collections

def generate_word2vec_pairs(text, window_size):
    """
    CBOW için (bağlam, hedef) veya Skip-Gram için (hedef, bağlam) çiftlerini oluşturur.
    Bu örnek, ham veri yapısının oluşturulmasına odaklanmaktadır.

    Args:
        text (list): Kelimelerin listesi (tokenleştirilmiş cümle).
        window_size (int): Hedef kelimenin her iki tarafındaki bağlam penceresinin boyutu.

    Returns:
        list: Her bir demetin (tuple) (girdi_kelime(ler)i, çıktı_kelime(ler)i) olduğu bir demet listesi.
    """
    data = []
    text_length = len(text)

    for i, target_word in enumerate(text):
        # Bağlam penceresi için başlangıç ve bitiş indekslerini belirle
        start_index = max(0, i - window_size)
        end_index = min(text_length, i + window_size + 1)
        
        # Hedef kelimenin kendisi hariç bağlam kelimelerini çıkar
        context_words = [text[j] for j in range(start_index, end_index) if j != i]
        
        if context_words: # Sadece bağlam kelimeleri varsa ekle
            # CBOW için, girdi bağlamdır, çıktı hedeftir
            # Skip-Gram için, girdi hedeftir, çıktı her bir bağlam kelimesidir (birden çok çift)
            
            # Bu örnek yapı, daha çok kavramsal veri üretimi içindir:
            # CBOW kavramsal çifti: (bağlam_kelimeleri_listesi, hedef_kelime)
            data.append((context_words, target_word)) # Bir CBOW benzeri örneği temsil eder

            # Skip-Gram kavramsal çiftleri: (hedef_kelime, bağlam_kelimesi_1), (hedef_kelime, bağlam_kelimesi_2), ...
            # Kısalık adına, çıktıda sadece CBOW benzeri yapıyı açıkça göstereceğiz.
            # Gerçek bir Skip-Gram uygulamasında, bağlam_kelimeleri üzerinden döngü yapar
            # ve (hedef_kelime, tek_bağlam_kelimesi) çiftleri oluştururdunuz.
            
    return data

# Örnek kullanım:
sentence = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
window = 2
generated_pairs = generate_word2vec_pairs(sentence, window)

print(f"Orijinal cümle: {sentence}\n")
print(f"Oluşturulan (bağlam, hedef) kavramsal çiftler (pencere={window}):")
for pair in generated_pairs:
    print(f"  Bağlam: {pair[0]}, Hedef: {pair[1]}")


(Kod örneği bölümünün sonu)
```
## 7. Sonuç

Word2Vec, yoğun, sürekli kelime gömülmelerini öğrenmek için oldukça etkili ve verimli bir yöntem sağlayarak doğal dil işleme alanında devrim yaratmıştır. Word2Vec çerçevesindeki iki ana mimari olan CBOW ve Skip-Gram, kelime temsillerini öğrenme görevine farklı açılardan yaklaşır. CBOW, çevresindeki bağlamdan bir hedef kelimeyi tahmin etmede üstündür, bu da onu genellikle daha hızlı eğitilebilir kılar ve büyük korpuslardaki sık kullanılan kelimeler için özellikle etkilidir. Ancak, bağlam vektörlerinin ortalaması, nadir kelimeler için spesifikliği seyreltme eğilimindedir. Tersine, Skip-Gram, verilen bir hedef kelimeden bağlam kelimelerini tahmin eder, incelikli anlamsal ilişkileri yakalamada üstün performans gösterir ve nadir kelimeler için yüksek kaliteli gömülmeler üretir, ancak daha yavaş eğitim süreleri pahasına.

CBOW ve Skip-Gram arasındaki seçim, korpus boyutu, hesaplama kaynakları ve sık kullanılan ile seyrek terimler için istenen gömülmelerin kalitesi gibi faktörlere bağlı pratik bir karardır. Her iki model de, özellikle Negatif Örnekleme gibi optimizasyonlarla desteklendiğinde, modern NLP'de vazgeçilmez araçlar olduğunu kanıtlamış, makine çevirisi, duygu analizi, bilgi erişimi ve çeşitli diğer dil anlama görevlerinde ilerlemelerin temelini oluşturmuştur. Onların kalıcı mirası, ham metin verilerinden karmaşık dilbilimsel kalıpları ortaya çıkarmak için basit sinir mimarilerinin gücünü göstererek, üretken yapay zeka alanında daha gelişmiş derin öğrenme modellerinin önünü açmasında yatmaktadır.

