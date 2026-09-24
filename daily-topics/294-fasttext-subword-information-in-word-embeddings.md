# FastText: Subword Information in Word Embeddings

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The FastText Model Architecture](#2-the-fasttext-model-architecture)
- [3. Advantages and Disadvantages](#3-advantages-and-disadvantages)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
**Word embeddings** have revolutionized Natural Language Processing (NLP) by providing dense vector representations of words, capturing semantic and syntactic relationships. Models like **Word2Vec** (Skip-gram and CBOW) and **GloVe** have been widely successful in tasks such as sentiment analysis, machine translation, and information retrieval. However, these models treat each word as an atomic unit. If a word is not present in the training vocabulary (an **Out-Of-Vocabulary (OOV)** word), they cannot generate a vector for it. This limitation is particularly problematic for languages with rich morphology, where words can take many different forms, or for handling typos and neologisms.

**FastText**, introduced by Facebook AI Research (Bojanowski et al., 2017), addresses this fundamental challenge by incorporating **subword information** into its word embedding generation process. Instead of learning embeddings for entire words, FastText learns embeddings for character n-grams, or "subwords," and represents each word as the sum of its constituent character n-gram vectors. This approach allows FastText to generate robust word representations even for rare or OOV words, as their representations can be inferred from the embeddings of their known subcomponents. This innovation significantly enhances the utility of word embeddings in a broader range of real-world applications and languages.

## 2. The FastText Model Architecture
The core idea behind FastText is to decompose words into their constituent **character n-grams**. For example, if we consider the word "eating" and character n-grams of length 3 to 6, it would be represented by vectors for `<eat>`, `<ati>`, `<tin>`, `<ing>`, `eat`, `atin`, `ting`, `eati`, `atin`, `ting`, `eatin`, `ating`, `eating`, and also the special boundary tags `<` and `>`. The angled brackets `<` and `>` are added to distinguish between a prefix/suffix and a full word. For instance, the n-gram `<app>` would represent the word "app", while `app` (without brackets) could be a subword of "apple" or "application".

FastText adapts the **CBOW (Continuous Bag-Of-Words)** or **Skip-gram** architectures from Word2Vec. In the Skip-gram model, instead of predicting context words from a target word, FastText predicts context words from the sum of the character n-gram vectors of the target word. Similarly, in the CBOW model, it predicts a target word (represented as a sum of its n-gram vectors) from the sum of its context words.

More formally, for a word *w*, its vector representation *v_w* is computed as the sum of the vector representations of its constituent character n-grams. That is, if *G_w* is the set of character n-grams for word *w*, then *v_w = Σ_{g ∈ G_w} x_g*, where *x_g* is the vector representation for n-gram *g*. During training, the model learns these character n-gram embeddings. When an OOV word is encountered, its representation can still be formed by summing the vectors of its constituent n-grams, provided those n-grams were seen during training. This mechanism gives FastText its unique ability to handle unknown words effectively.

The model typically uses a softmax function for classification (predicting context words or target words), but to improve computational efficiency, especially with large vocabularies, it often employs **hierarchical softmax** or **negative sampling**, similar to Word2Vec. These optimizations reduce the computational cost of training by avoiding a full softmax computation over the entire vocabulary at each step.

## 3. Advantages and Disadvantages
FastText's approach of leveraging subword information provides several significant advantages, along with a few minor drawbacks.

### Advantages:
*   **Handling Out-Of-Vocabulary (OOV) Words:** This is FastText's most prominent advantage. Since a word's embedding is the sum of its character n-gram embeddings, FastText can generate reasonable representations for words it has never encountered during training, as long as some of their constituent n-grams are known. This is particularly useful for rare words, proper nouns, and words with minor spelling variations.
*   **Morphologically Rich Languages:** Languages like Turkish, Finnish, German, or Hungarian, which rely heavily on affixes and compounding, benefit immensely from FastText. A single root word can generate hundreds or thousands of inflected forms. FastText can capture semantic similarities between these forms even if specific inflections were rare in the training data, as they share common subword units.
*   **Robustness to Typos and Misspellings:** Minor spelling errors often preserve many character n-grams. FastText can thus derive a vector for a misspelled word that is closer to the correctly spelled word than a model that treats them as entirely distinct tokens.
*   **Efficiency:** Despite its sophisticated subword handling, FastText is computationally efficient. It can train embeddings on large datasets quickly, often faster than other models like Word2Vec while achieving competitive or superior performance on many tasks, especially for morphologically complex languages.
*   **Word Similarity for Related Words:** FastText can better capture the similarity between morphologically related words (e.g., "run," "running," "runner") because they share common subword components.

### Disadvantages:
*   **Larger Model Size:** The vocabulary of character n-grams is typically much larger than the vocabulary of unique words, leading to larger model files compared to atomistic word embedding models.
*   **Increased Training Time (Marginal):** While generally efficient, the overhead of processing character n-grams for every word can sometimes lead to slightly longer training times compared to a highly optimized Word2Vec implementation, though this is often negligible for practical purposes.
*   **Loss of Nuance for Entire Words:** In some cases, representing a word purely as a sum of its n-grams might slightly dilute the specific nuances of the full word, especially if the word has a strong idiosyncratic meaning that isn't fully captured by its subcomponents. However, for most NLP tasks, the benefits outweigh this potential drawback.

## 4. Code Example
This example demonstrates how to train a FastText model using the `gensim` library in Python and obtain word vectors.

```python
from gensim.models import FastText
from gensim.utils import simple_preprocess

# Sample corpus (list of sentences)
corpus = [
    "FastText is a powerful word embedding model.",
    "It handles out-of-vocabulary words effectively.",
    "Morphologically rich languages benefit significantly from FastText.",
    "Word embeddings are crucial for many NLP tasks.",
    "This model uses subword information."
]

# Preprocess the corpus: tokenize and lowercase
# gensim.utils.simple_preprocess splits a string into a list of lowercase tokens
processed_corpus = [simple_preprocess(sentence) for sentence in corpus]

# Train the FastText model
# min_n and max_n define the range of character n-grams to consider
# vector_size is the dimensionality of the word vectors
# window is the maximum distance between the current and predicted word within a sentence
# min_count ignores all words with total frequency lower than this
# workers specify the number of worker threads to use for training
model = FastText(
    sentences=processed_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,
    max_n=6,
    sg=1,  # 1 for Skip-gram, 0 for CBOW
    epochs=10,
    workers=4
)

# Get the vector for a known word
known_word_vector = model.wv['powerful']
print(f"Vector for 'powerful' (first 5 dimensions): {known_word_vector[:5]}")
print(f"Vector length for 'powerful': {len(known_word_vector)}")

# Get the vector for an OOV word (e.g., a misspelled word or neologism)
# FastText can still generate a vector even if 'poweful' was not in the original corpus
oov_word_vector = model.wv['poweful'] # 'powerful' misspelled
print(f"Vector for 'poweful' (first 5 dimensions): {oov_word_vector[:5]}")
print(f"Vector length for 'poweful': {len(oov_word_vector)}")

# Find most similar words to a known word
similar_words = model.wv.most_similar('languages', topn=3)
print(f"Words most similar to 'languages': {similar_words}")

# Find most similar words to an OOV word (its representation is inferred from subwords)
similar_oov_words = model.wv.most_similar('morpholog', topn=2) # Partial word/subword
print(f"Words most similar to 'morpholog': {similar_oov_words}")

(End of code example section)
```

## 5. Conclusion
FastText stands as a significant advancement in the field of word embeddings, particularly in its elegant solution to the **Out-Of-Vocabulary (OOV)** word problem. By intelligently decomposing words into **character n-grams** and learning embeddings for these subword units, FastText extends the utility and robustness of word representations beyond what atomistic models like Word2Vec or GloVe can offer. Its capability to generate meaningful vectors for rare, unknown, or even misspelled words makes it exceptionally valuable for languages with rich morphology and for applications where comprehensive vocabulary coverage is challenging. While it incurs a slight overhead in model size and potentially training time, these are often outweighed by its enhanced performance in handling linguistic nuances and achieving greater generalization. FastText continues to be a cornerstone in modern NLP pipelines, demonstrating the power of incorporating fine-grained subword information for more robust and universally applicable language models.

---
<br>

<a name="türkçe-içerik"></a>
## FastText: Kelime Gömülmelerinde Alt Kelime Bilgisi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. FastText Model Mimarisi](#2-fasttext-model-mimarisi)
- [3. Avantajlar ve Dezavantajlar](#3-avantajlar-ve-dezavantajlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Kelime gömülmeleri (word embeddings)**, kelimelerin anlamsal ve sentaktik ilişkilerini yakalayan yoğun vektör temsilleri sağlayarak Doğal Dil İşleme (NLP) alanında devrim yaratmıştır. **Word2Vec** (Skip-gram ve CBOW) ve **GloVe** gibi modeller duygu analizi, makine çevirisi ve bilgi erişimi gibi görevlerde büyük başarılar elde etmiştir. Ancak, bu modeller her kelimeyi atomik bir birim olarak ele alır. Bir kelime eğitim sözlüğünde mevcut değilse (**Sözlük Dışı (Out-Of-Vocabulary - OOV)** kelime), bu kelime için bir vektör üretemezler. Bu sınırlama, özellikle zengin morfolojiye sahip diller için, kelimelerin birçok farklı biçim alabildiği veya yazım hataları ve neolojizmlerin (yeni türetilmiş kelimeler) ele alınması gerektiği durumlarda sorunludur.

FastText, Facebook AI Research tarafından sunulan (Bojanowski ve diğerleri, 2017) bu temel zorluğun üstesinden, kelime gömülmesi oluşturma sürecine **alt kelime bilgisini** dahil ederek gelmiştir. FastText, tüm kelimeler için gömülme öğrenmek yerine, karakter n-gram'ları veya "alt kelimeler" için gömülmeler öğrenir ve her kelimeyi, onu oluşturan karakter n-gram vektörlerinin toplamı olarak temsil eder. Bu yaklaşım, FastText'in nadir veya OOV kelimeler için bile sağlam kelime temsilleri oluşturmasına olanak tanır, çünkü temsilleri bilinen alt bileşenlerinin gömülmelerinden çıkarılabilir. Bu yenilik, kelime gömülmelerinin daha geniş bir gerçek dünya uygulaması ve dil yelpazesinde kullanışlılığını önemli ölçüde artırmaktadır.

## 2. FastText Model Mimarisi
FastText'in arkasındaki temel fikir, kelimeleri onları oluşturan **karakter n-gram'larına** ayırmaktır. Örneğin, "yemek" kelimesini ve 3 ila 6 uzunluğundaki karakter n-gram'larını ele alırsak, kelime `<yem>`, `<eme>`, `<mek>`, `yem`, `emek`, `mek`, `yeme`, `emek`, `mek`, `yemek`, `emek` ve ayrıca özel sınır etiketleri `<` ve `>` için vektörlerle temsil edilecektir. Açılı parantezler `<` ve `>` bir önek/sonek ile tam bir kelimeyi ayırt etmek için eklenir. Örneğin, `<elma>` n-gram'ı "elma" kelimesini temsil ederken, `elma` (parantezsiz) "elmacılık" veya "elmacı" gibi kelimelerin bir alt kelimesi olabilir.

FastText, Word2Vec'ten **CBOW (Continuous Bag-Of-Words)** veya **Skip-gram** mimarilerini uyarlar. Skip-gram modelinde, hedef kelimeden bağlam kelimelerini tahmin etmek yerine, FastText hedef kelimenin karakter n-gram vektörlerinin toplamından bağlam kelimelerini tahmin eder. Benzer şekilde, CBOW modelinde, bağlam kelimelerinin toplamından bir hedef kelimeyi (n-gram vektörlerinin toplamı olarak temsil edilen) tahmin eder.

Daha resmi olarak, bir *w* kelimesi için, onun vektör temsili *v_w*, onu oluşturan karakter n-gram'larının vektör temsillerinin toplamı olarak hesaplanır. Yani, eğer *G_w*, *w* kelimesi için karakter n-gram'ları kümesi ise, *v_w = Σ_{g ∈ G_w} x_g* olur, burada *x_g*, *g* n-gram'ı için vektör temsilidir. Eğitim sırasında model bu karakter n-gram gömülmelerini öğrenir. Bir OOV kelimeyle karşılaşıldığında, bu kelimenin temsili, n-gram'ları eğitim sırasında görülmüşse, onu oluşturan n-gram'ların vektörlerini toplayarak hala oluşturulabilir. Bu mekanizma, FastText'e bilinmeyen kelimeleri etkili bir şekilde ele alma konusundaki benzersiz yeteneğini verir.

Model tipik olarak sınıflandırma için (bağlam kelimelerini veya hedef kelimeleri tahmin etmek için) bir softmax fonksiyonu kullanır, ancak hesaplama verimliliğini artırmak için, özellikle büyük sözlüklerde, Word2Vec'e benzer şekilde **hiyerarşik softmax** veya **negatif örnekleme** kullanır. Bu optimizasyonlar, her adımda tüm sözlük üzerinde tam bir softmax hesaplamasından kaçınarak eğitimin hesaplama maliyetini azaltır.

## 3. Avantajlar ve Dezavantajlar
FastText'in alt kelime bilgisini kullanma yaklaşımı, birkaç önemli avantajın yanı sıra bazı küçük dezavantajlar da sunar.

### Avantajlar:
*   **Sözlük Dışı (OOV) Kelimelerle Başa Çıkma:** Bu, FastText'in en belirgin avantajıdır. Bir kelimenin gömülmesi, karakter n-gram gömülmelerinin toplamı olduğundan, FastText, eğitim sırasında hiç karşılaşmadığı kelimeler için bile, bazı oluşturan n-gramları biliniyorsa, makul temsiller üretebilir. Bu, nadir kelimeler, özel isimler ve küçük yazım farklılıkları olan kelimeler için özellikle kullanışlıdır.
*   **Morfolojik Açıdan Zengin Diller:** Türkçe, Fince, Almanca veya Macarca gibi ekler ve birleştirmelere yoğun şekilde dayanan diller, FastText'ten muazzam faydalar sağlar. Tek bir kök kelime yüzlerce veya binlerce çekimli form üretebilir. FastText, bu formlar arasındaki anlamsal benzerlikleri, belirli çekimler eğitim verisinde nadir olsa bile, ortak alt kelime birimlerini paylaştıkları için yakalayabilir.
*   **Yazım Hatalarına ve Yanlış Yazımlara Karşı Dayanıklılık:** Küçük yazım hataları genellikle birçok karakter n-gram'ını korur. FastText bu nedenle yanlış yazılmış bir kelime için, onları tamamen farklı belirteçler olarak ele alan bir modele göre doğru yazılmış kelimeye daha yakın bir vektör türetebilir.
*   **Verimlilik:** Gelişmiş alt kelime işlemine rağmen, FastText hesaplama açısından verimlidir. Genellikle Word2Vec gibi diğer modellerden daha hızlı bir şekilde büyük veri kümeleri üzerinde gömülmeler eğitebilirken, birçok görevde, özellikle morfolojik olarak karmaşık diller için rekabetçi veya üstün performans elde eder.
*   **İlgili Kelimeler İçin Kelime Benzerliği:** FastText, morfolojik olarak ilişkili kelimeler arasındaki benzerliği (örn. "koşmak", "koşuyor", "koşucu") daha iyi yakalayabilir, çünkü ortak alt kelime bileşenlerini paylaşırlar.

### Dezavantajlar:
*   **Daha Büyük Model Boyutu:** Karakter n-gramlarının sözlüğü, tipik olarak benzersiz kelimelerin sözlüğünden çok daha büyüktür, bu da atomik kelime gömme modellerine kıyasla daha büyük model dosyalarına yol açar.
*   **Artan Eğitim Süresi (Marjinal):** Genel olarak verimli olsa da, her kelime için karakter n-gramlarını işlemenin ek yükü, bazen yüksek düzeyde optimize edilmiş bir Word2Vec uygulamasına göre biraz daha uzun eğitim sürelerine yol açabilir, ancak bu pratik amaçlar için genellikle ihmal edilebilir düzeydedir.
*   **Tüm Kelimeler İçin Nüans Kaybı:** Bazı durumlarda, bir kelimeyi yalnızca n-gramlarının toplamı olarak temsil etmek, kelimenin alt bileşenleri tarafından tam olarak yakalanamayan güçlü, kendine özgü bir anlama sahipse, tüm kelimenin belirli nüanslarını biraz seyreltebilir. Ancak, çoğu NLP görevi için faydalar bu potansiyel dezavantajdan daha ağır basmaktadır.

## 4. Kod Örneği
Bu örnek, Python'da `gensim` kütüphanesini kullanarak bir FastText modelinin nasıl eğitileceğini ve kelime vektörlerinin nasıl alınacağını gösterir.

```python
from gensim.models import FastText
from gensim.utils import simple_preprocess

# Örnek metin kümesi (cümle listesi)
corpus = [
    "FastText güçlü bir kelime gömme modelidir.",
    "Sözlük dışı kelimeleri etkili bir şekilde ele alır.",
    "Morfolojik açıdan zengin diller FastText'ten önemli ölçüde faydalanır.",
    "Kelime gömülmeleri birçok NLP görevi için çok önemlidir.",
    "Bu model alt kelime bilgisini kullanır."
]

# Metin kümesini ön işleme: belirteçlere ayırma ve küçük harfe dönüştürme
# gensim.utils.simple_preprocess, bir dizeyi küçük harfli belirteçlerin listesine böler
processed_corpus = [simple_preprocess(sentence) for sentence in corpus]

# FastText modelini eğitme
# min_n ve max_n, dikkate alınacak karakter n-gramlarının aralığını tanımlar
# vector_size, kelime vektörlerinin boyutluluğudur
# window, bir cümledeki mevcut ve tahmin edilen kelime arasındaki maksimum mesafedir
# min_count, toplam frekansı bundan daha düşük olan tüm kelimeleri göz ardı eder
# workers, eğitim için kullanılacak işçi iş parçacığı sayısını belirtir
model = FastText(
    sentences=processed_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,
    max_n=6,
    sg=1,  # Skip-gram için 1, CBOW için 0
    epochs=10,
    workers=4
)

# Bilinen bir kelimenin vektörünü alma
known_word_vector = model.wv['güçlü']
print(f"'güçlü' kelimesinin vektörü (ilk 5 boyut): {known_word_vector[:5]}")
print(f"'güçlü' kelimesinin vektör uzunluğu: {len(known_word_vector)}")

# Bir OOV kelimesinin vektörünü alma (örn. yanlış yazılmış bir kelime veya neolojizm)
# FastText, 'güçül' orijinal metin kümesinde olmasa bile bir vektör oluşturabilir
oov_word_vector = model.wv['güçül'] # 'güçlü' yanlış yazılmış
print(f"'güçül' kelimesinin vektörü (ilk 5 boyut): {oov_word_vector[:5]}")
print(f"'güçül' kelimesinin vektör uzunluğu: {len(oov_word_vector)}")

# Bilinen bir kelimeye en benzer kelimeleri bulma
similar_words = model.wv.most_similar('diller', topn=3)
print(f"'diller' kelimesine en benzer kelimeler: {similar_words}")

# Bir OOV kelimeye en benzer kelimeleri bulma (temsili alt kelimelerden çıkarılır)
similar_oov_words = model.wv.most_similar('morfolo', topn=2) # Kısmi kelime/alt kelime
print(f"'morfolo' kelimesine en benzer kelimeler: {similar_oov_words}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
FastText, kelime gömülmeleri alanında önemli bir ilerleme olarak öne çıkmaktadır, özellikle **Sözlük Dışı (OOV)** kelime sorununa sunduğu zarif çözümle. Kelimeleri akıllıca **karakter n-gram'larına** ayırarak ve bu alt kelime birimleri için gömülmeler öğrenerek, FastText, kelime temsillerinin kullanışlılığını ve sağlamlığını Word2Vec veya GloVe gibi atomik modellerin sunabileceğinin ötesine taşır. Nadir, bilinmeyen veya hatta yanlış yazılmış kelimeler için anlamlı vektörler oluşturma yeteneği, onu zengin morfolojiye sahip diller ve kapsamlı kelime dağarcığı kapsamının zorlu olduğu uygulamalar için son derece değerli kılar. Model boyutu ve potansiyel olarak eğitim süresi açısından hafif bir ek yük getirse de, bunlar genellikle dilsel incelikleri ele almada ve daha fazla genelleme elde etmede sağladığı gelişmiş performansla dengelenir. FastText, daha sağlam ve evrensel olarak uygulanabilir dil modelleri için ince taneli alt kelime bilgisini dahil etmenin gücünü göstererek modern NLP süreçlerinde bir köşe taşı olmaya devam etmektedir.





