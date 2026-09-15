# FastText: Subword Information in Word Embeddings

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Architecture of FastText](#2-the-architecture-of-fasttext)
  - [2.1. Subword Units (N-grams)](#21-subword-units-n-grams)
  - [2.2. Averaging Subword Embeddings](#22-averaging-subword-embeddings)
- [3. Advantages and Use Cases](#3-advantages-and-use-cases)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
Word embeddings have revolutionized Natural Language Processing (NLP) by enabling the representation of words as dense, low-dimensional vectors that capture semantic and syntactic relationships. Models like **Word2Vec** (Mikolov et al., 2013) and **GloVe** (Pennington et al., 2014) effectively generate these embeddings by learning from the co-occurrence statistics of words in large text corpora. While highly successful, these traditional methods treat each word as an indivisible atomic unit. This atomic view presents several limitations:

1.  **Out-of-Vocabulary (OOV) Words:** If a word is not present in the training corpus vocabulary, it cannot be assigned a meaningful embedding. This is a significant challenge for handling rare words, neologisms, or misspelled words.
2.  **Morphological Richness:** Languages with complex morphology (e.g., Turkish, Finnish) have many word forms derived from a single root. Treating "run," "running," and "runs" as entirely distinct tokens ignores their shared semantic core and the morphological relationships between them. This leads to inefficient use of data, as the model must learn separate embeddings for each inflection.
3.  **Representing Rare Words:** Words that appear infrequently in the corpus may not have enough contextual information to learn robust embeddings, even if they are in the vocabulary.

**FastText**, introduced by Bojanowski et al. (2017) at Facebook AI Research (FAIR), addresses these limitations by incorporating **subword information** into its word embedding generation process. Unlike Word2Vec or GloVe, FastText does not learn embeddings for full words directly. Instead, it represents each word as a **bag of character n-grams**. The embedding for a word is then derived by summing or averaging the embeddings of its constituent character n-grams. This fundamental architectural shift allows FastText to effectively handle OOV words, leverage morphological information, and produce high-quality embeddings even for rare words. Beyond word embeddings, FastText is also notable for its efficient text classification capabilities, but this document will focus specifically on its word embedding methodology.

## 2. The Architecture of FastText
FastText's core innovation lies in its treatment of words as compositions of smaller, constituent parts: **character n-grams**. This approach fundamentally differs from traditional word embedding models that treat words as atomic entities. The architecture builds upon the efficiency of Word2Vec's Skip-gram and CBOW models but extends them to operate on subword units.

In essence, FastText represents a word `w` as a collection of its character n-grams, plus the word itself as a special n-gram. For example, the word "apple" with `min_n=3` and `max_n=6` might be represented by `<ap`, `app`, `ppl`, `ple>`, `<appl`, `apple>`, `<apple>` (where `<` and `>` are boundary markers for the word itself). Each of these n-grams is assigned a unique vector representation. The final word vector for "apple" is then the sum of all these n-gram vectors.

### 2.1. Subword Units (N-grams)
The "subword units" in FastText are typically **character n-grams**. A character n-gram is a contiguous sequence of `n` characters within a word. For instance, given the word "eating" and `n=3` (trigrams), the character n-grams would include:
*   "eat"
*   "ati"
*   "tin"
*   "ing"

To differentiate n-grams that are word fragments from the full word itself, FastText often adds special boundary characters, `<` and `>`, to the beginning and end of a word. For "eating" with boundaries, this becomes `<eating>`. The n-grams might then include:
*   `<ea`
*   `eat`
*   `ati`
*   `tin`
*   `ing`
*   `ng>`
*   Additionally, the full word `eating` itself is considered a special n-gram.

The range of `n` (minimum `min_n` and maximum `max_n` length of n-grams) is a crucial hyperparameter. Typical values are `min_n=3` and `max_n=6`. By varying `n`, FastText can capture different granularities of subword information. Shorter n-grams capture morphemes or prefixes/suffixes, while longer n-grams cover more of the word's structure.

Each unique character n-gram observed in the training corpus is assigned its own unique vector. These vectors are learned during the training process.

### 2.2. Averaging Subword Embeddings
Once the character n-grams for a word are identified, FastText generates the embedding for the entire word by **summing or averaging the vectors of its constituent character n-grams**.

Let `w` be a word, and `G_w` be the set of its character n-grams. Each `g` in `G_w` has an associated vector `v_g`. The word embedding `V_w` for `w` is then calculated as:

`V_w = (1 / |G_w|) * Σ(g ∈ G_w) v_g` (if averaging)
or
`V_w = Σ(g ∈ G_w) v_g` (if summing)

This simple aggregation mechanism is powerful. Consider the word "unbelievable." It would be decomposed into n-grams like "un-", "-believ-", "-able." The vector for "unbelievable" would thus be a composite of the vectors representing these meaningful subword units.

This approach offers several key advantages:
*   **Handling OOV Words:** If a word has never been seen during training, FastText can still construct an embedding for it by summing the vectors of its known character n-grams. Even if the full word is OOV, its subcomponents might be known, allowing for a reasonable approximation of its meaning.
*   **Morphological Awareness:** Common prefixes (e.g., "un-", "re-"), suffixes (e.g., "-ing", "-ed", "-tion"), and roots contribute their vector representations across many words. This allows FastText to implicitly learn and leverage morphological regularities. For example, "run," "running," and "runs" will share the embeddings of their common n-grams related to "run," leading to semantically closer vectors.
*   **Efficient Representation of Rare Words:** Even if a full word is rare, its constituent n-grams are likely to appear in many other words, providing sufficient data to learn robust n-gram embeddings. The rare word's embedding then benefits from the well-trained vectors of its subcomponents.

The training objective for FastText's word embeddings is similar to that of Word2Vec's Skip-gram model. It aims to predict context words given a target word, but crucially, the target word's representation is formed by summing its n-gram vectors.

## 3. Advantages and Use Cases
FastText's unique approach to incorporating subword information yields several significant advantages that extend its applicability and performance compared to traditional word embedding models:

1.  **Effective Handling of Out-of-Vocabulary (OOV) Words:** This is arguably FastText's most celebrated feature. Since a word's vector is an aggregate of its character n-grams, FastText can generate a meaningful vector for any word, even if it was not present in the training vocabulary, as long as its constituent n-grams have been observed. This is particularly beneficial for:
    *   **Rare words and neologisms:** New words or words with low frequency in the training corpus can still get decent representations.
    *   **Misspellings and typos:** Small variations in spelling might still allow the model to construct a vector from mostly correct n-grams.
    *   **Infrequent proper nouns:** Names of people, places, or organizations that might not be in the training data can still be embedded.

2.  **Robustness to Morphological Variations:** For morphologically rich languages (e.g., Turkish, Arabic, Finnish, German), a single root word can have hundreds of inflected forms. FastText leverages the shared character n-grams (morphemes) across these forms, allowing the model to:
    *   **Capture semantic similarities:** "run," "running," and "runs" will have similar vectors because they share the n-grams associated with the root "run."
    *   **Learn more efficiently:** The model doesn't need to learn entirely separate vectors for each inflected form, pooling statistical strength across morphologically related words. This reduces the data requirements for learning high-quality embeddings.

3.  **Improved Embeddings for Rare Words:** Even when a word is in the vocabulary but appears very infrequently, traditional models struggle to learn robust embeddings due to limited contextual evidence. FastText mitigates this by allowing rare words to "borrow" statistical strength from their constituent n-grams, which are likely to appear in many other common words. This leads to more reliable and semantically richer representations for less frequent terms.

4.  **Language Agnostic:** The character n-gram approach is not tied to any specific language's grammatical rules or script, making FastText highly adaptable to various languages, including those with complex writing systems or agglutinative structures.

**Typical Use Cases:**

*   **Information Retrieval and Search:** Improved embeddings for rare or OOV query terms can lead to more relevant search results.
*   **Machine Translation:** Better handling of morphological variations and rare words in both source and target languages.
*   **Named Entity Recognition (NER):** Recognizing entities that might be new or infrequent in training data.
*   **Sentiment Analysis and Text Classification:** Even if a review contains a novel or less common adjective, FastText can still provide a meaningful embedding, contributing to more accurate classification.
*   **Cross-Lingual NLP:** Potentially aligning embeddings across languages by leveraging shared subword patterns or common roots.

While FastText requires more memory to store all the n-gram vectors compared to just word vectors, its advantages in handling OOV words and morphological richness often outweigh this drawback, especially in practical applications dealing with diverse and dynamic vocabularies.

## 4. Code Example
This example demonstrates how to train a basic FastText word embedding model and retrieve word vectors using the `fasttext` Python library. We'll use a simple list of sentences for illustration.

```python
import fasttext
import os

# Create a dummy text file for training
# FastText typically expects a plain text file, one sentence per line
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("FastText is an excellent tool for word embeddings.\n")
    f.write("It handles out-of-vocabulary words very well.\n")
    f.write("Subword information is key to its performance.\n")
    f.write("This approach is great for morphologically rich languages.\n")

# Train a FastText model
# Parameters:
# input: training file path
# model: type of model (skipgram or cbow)
# dim: size of word vectors
# min_n: min length of char ngrams
# max_n: max length of char ngrams
# epoch: number of epochs
# lr: learning rate
model = fasttext.train_words(
    input="data.txt",
    model="skipgram",
    dim=100,
    min_n=3,
    max_n=6,
    epoch=10,
    lr=0.05
)

# Get the vector for a known word
known_word_vector = model.get_word_vector("embeddings")
print(f"Vector for 'embeddings' (first 5 values): {known_word_vector[:5]}")

# Get the vector for an OOV word (e.g., 'subword_embedding', not in training data)
# FastText can still generate a vector based on its n-grams.
oov_word_vector = model.get_word_vector("subword_embedding")
print(f"Vector for 'subword_embedding' (first 5 values): {oov_word_vector[:5]}")

# Find the 5 nearest neighbors for a word
nearest_neighbors = model.get_nearest_neighbors("embeddings", k=5)
print(f"Nearest neighbors for 'embeddings': {nearest_neighbors}")

# Clean up the dummy file
os.remove("data.txt")

(End of code example section)
```

## 5. Conclusion
FastText represents a significant advancement in the field of word embeddings, particularly by bridging the gap between traditional word-level models and character-level approaches. Its ingenious incorporation of **subword information**, specifically through character n-grams, allows it to effectively address critical limitations such as the handling of **out-of-vocabulary (OOV) words**, leveraging **morphological richness**, and generating robust representations for **rare words**.

By viewing words as compositions of smaller, shared units, FastText not only improves the quality and coverage of word embeddings but also offers enhanced adaptability across diverse languages, regardless of their morphological complexity. While requiring slightly more memory due to the storage of n-gram vectors, the benefits in terms of model robustness and semantic understanding, especially in real-world applications with evolving and less curated vocabularies, are substantial. FastText has solidified its position as an indispensable tool in the modern NLP toolkit, offering a pragmatic and powerful solution for generating high-quality word representations.

---
<br>

<a name="türkçe-içerik"></a>
## FastText: Kelime Gömülmelerinde Alt Kelime Bilgisi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. FastText Mimarisi](#2-fasttext-mimarisi)
  - [2.1. Alt Kelime Birimleri (N-gramlar)](#21-alt-kelime-birimleri-n-gramlar)
  - [2.2. Alt Kelime Gömülmelerinin Ortalaması](#22-alt-kelime-gömülmelerinin-ortalaması)
- [3. Avantajlar ve Kullanım Alanları](#3-avantajlar-ve-kullanım-alanları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
Kelime gömülmeleri (word embeddings), kelimeleri anlamsal ve sentaktik ilişkileri yakalayan yoğun, düşük boyutlu vektörler olarak temsil ederek Doğal Dil İşleme (NLP) alanında devrim yaratmıştır. **Word2Vec** (Mikolov ve diğerleri, 2013) ve **GloVe** (Pennington ve diğerleri, 2014) gibi modeller, büyük metin korpuslarındaki kelimelerin birlikte geçme istatistiklerinden öğrenerek bu gömülmeleri etkin bir şekilde üretir. Son derece başarılı olmalarına rağmen, bu geleneksel yöntemler her kelimeyi bölünmez, atomik bir birim olarak ele alır. Bu atomik bakış açısı birkaç sınırlama sunar:

1.  **Sözlük Dışı (Out-of-Vocabulary - OOV) Kelimeler:** Bir kelime eğitim korpusu sözlüğünde mevcut değilse, ona anlamlı bir gömülme atanamaz. Bu durum, nadir kelimeleri, neolojizmleri (yeni türetilmiş kelimeler) veya yazım hatalı kelimeleri ele almada önemli bir zorluk teşkil eder.
2.  **Morfolojik Zenginlik:** Karmaşık morfolojiye sahip dillerde (örn. Türkçe, Fince) tek bir kökten türeyen birçok kelime biçimi bulunur. "Koş", "koşuyor" ve "koştu" kelimelerini tamamen ayrı belirteçler (token) olarak ele almak, onların paylaştığı anlamsal çekirdeği ve aralarındaki morfolojik ilişkileri göz ardı eder. Bu durum, modelin her çekim için ayrı gömülmeler öğrenmesi gerektiğinden, verinin verimsiz kullanılmasına yol açar.
3.  **Nadir Kelimelerin Temsili:** Korpusta seyrek görülen kelimeler, sözlükte olsalar bile, sağlam gömülmeler öğrenmek için yeterli bağlamsal bilgiye sahip olmayabilirler.

Facebook AI Research (FAIR) ekibinden Bojanowski ve diğerleri (2017) tarafından tanıtılan **FastText**, bu sınırlamaları **alt kelime bilgisini** kelime gömülme üretme sürecine dahil ederek ele alır. Word2Vec veya GloVe'nin aksine, FastText tam kelimeler için doğrudan gömülmeler öğrenmez. Bunun yerine, her kelimeyi bir **karakter n-gramları torbası** olarak temsil eder. Bir kelimenin gömülme vektörü, daha sonra onu oluşturan karakter n-gramlarının gömülmelerinin toplanması veya ortalaması alınmasıyla türetilir. Bu temel mimari değişiklik, FastText'in OOV kelimeleri etkin bir şekilde işlemesine, morfolojik bilgiyi kullanmasına ve nadir kelimeler için bile yüksek kaliteli gömülmeler üretmesine olanak tanır. Kelime gömülmelerinin ötesinde, FastText etkili metin sınıflandırma yetenekleriyle de dikkat çekse de, bu belge özellikle kelime gömülme metodolojisine odaklanacaktır.

## 2. FastText Mimarisi
FastText'in temel yeniliği, kelimeleri daha küçük, oluşturan parçaların bileşimleri olarak ele almasında yatmaktadır: **karakter n-gramları**. Bu yaklaşım, kelimeleri atomik varlıklar olarak ele alan geleneksel kelime gömülme modellerinden temelden farklıdır. Mimari, Word2Vec'in Skip-gram ve CBOW modellerinin verimliliğine dayanır, ancak bunları alt kelime birimleri üzerinde çalışacak şekilde genişletir.

Özünde, FastText bir `w` kelimesini, onun karakter n-gramlarından ve özel bir n-gram olarak kelimenin kendisinden oluşan bir koleksiyon olarak temsil eder. Örneğin, `min_n=3` ve `max_n=6` değerlerine sahip "elma" kelimesi, `<el`, `elm`, `lma`, `ma>`, `<elma>`, `<elma>` (burada `<` ve `>` kelimenin kendisi için sınır belirteçleridir) ile temsil edilebilir. Bu n-gramların her birine benzersiz bir vektör gösterimi atanır. "Elma" kelimesi için nihai kelime vektörü, tüm bu n-gram vektörlerinin toplamıdır.

### 2.1. Alt Kelime Birimleri (N-gramlar)
FastText'teki "alt kelime birimleri" genellikle **karakter n-gramlarıdır**. Bir karakter n-gramı, bir kelime içindeki ardışık `n` karakter dizisidir. Örneğin, "yemek" kelimesi ve `n=3` (üçlü n-gramlar) verildiğinde, karakter n-gramları şunları içerecektir:
*   "yem"
*   "eme"
*   "mek"

Kelime parçacığı olan n-gramları, kelimenin kendisinden ayırmak için FastText, kelimenin başına ve sonuna genellikle özel sınır karakterleri, `<` ve `>`, ekler. "Yemek" kelimesi sınırlar ile `<yemek>` olur. Bu durumda n-gramlar şunları içerebilir:
*   `<ye`
*   `yem`
*   `eme`
*   `mek`
*   `ek>`
*   Ek olarak, tam kelime `yemek`'in kendisi özel bir n-gram olarak kabul edilir.

`n` aralığı (n-gramların minimum `min_n` ve maksimum `max_n` uzunluğu) kritik bir hiperparametredir. Tipik değerler `min_n=3` ve `max_n=6`'dır. `n` değerini değiştirerek, FastText farklı alt kelime bilgisi ayrıntılarını yakalayabilir. Daha kısa n-gramlar morfemleri veya ön ekleri/son ekleri yakalarken, daha uzun n-gramlar kelimenin yapısının daha fazlasını kapsar.

Eğitim korpusunda gözlemlenen her benzersiz karakter n-gramına kendi benzersiz vektörü atanır. Bu vektörler eğitim süreci boyunca öğrenilir.

### 2.2. Alt Kelime Gömülmelerinin Ortalaması
Bir kelime için karakter n-gramları belirlendikten sonra, FastText, tüm kelime için gömülmeyi, onu oluşturan karakter n-gramlarının vektörlerinin **toplanması veya ortalamasının alınmasıyla** üretir.

`w` bir kelime ve `G_w` onun karakter n-gramlarının kümesi olsun. `G_w` içindeki her `g`'nin ilişkili bir `v_g` vektörü vardır. `w` için kelime gömülme vektörü `V_w` daha sonra şu şekilde hesaplanır:

`V_w = (1 / |G_w|) * Σ(g ∈ G_w) v_g` (ortalama alınırsa)
veya
`V_w = Σ(g ∈ G_w) v_g` (toplanırsa)

Bu basit toplama mekanizması güçlüdür. "İnanılmaz" kelimesini düşünün. Bu kelime "inan-", "-ılmaz" gibi n-gramlara ayrılacaktır. "İnanılmaz" kelimesinin vektörü bu anlamlı alt kelime birimlerini temsil eden vektörlerin bir bileşimi olacaktır.

Bu yaklaşım birkaç temel avantaj sunar:
*   **OOV Kelimelerin İşlenmesi:** Bir kelime eğitim sırasında hiç görülmemiş olsa bile, FastText, bilinen karakter n-gramlarının vektörlerini toplayarak onun için bir gömülme oluşturabilir. Tam kelime OOV olsa bile, alt bileşenleri biliniyor olabilir, bu da anlamının makul bir şekilde tahmin edilmesini sağlar.
*   **Morfolojik Farkındalık:** Yaygın ön ekler (örn. "ön-", "yeniden-"), son ekler (örn. "-iyor", "-di", "-lik") ve kökler birçok kelimede kendi vektör temsillerini katkıda bulunurlar. Bu, FastText'in morfolojik düzenlilikleri dolaylı olarak öğrenmesine ve kullanmasına olanak tanır. Örneğin, "koş", "koşuyor" ve "koştu" kelimeleri, "koş" ile ilgili ortak n-gramların gömülmelerini paylaşacak ve bu da anlamsal olarak daha yakın vektörlere yol açacaktır.
*   **Nadir Kelimelerin Verimli Temsili:** Tam bir kelime nadir olsa bile, onu oluşturan n-gramlar büyük olasılıkla diğer birçok kelimede görünerek sağlam n-gram gömülmeleri öğrenmek için yeterli veri sağlar. Nadir kelimenin gömülmesi, daha sonra alt bileşenlerinin iyi eğitilmiş vektörlerinden faydalanır.

FastText'in kelime gömülmeleri için eğitim hedefi, Word2Vec'in Skip-gram modelininkine benzerdir. Bir hedef kelime verildiğinde bağlam kelimelerini tahmin etmeyi amaçlar, ancak kritik olarak, hedef kelimenin temsili n-gram vektörlerinin toplanmasıyla oluşturulur.

## 3. Avantajlar ve Kullanım Alanları
FastText'in alt kelime bilgisini dahil etme konusundaki benzersiz yaklaşımı, geleneksel kelime gömülme modellerine kıyasla uygulanabilirliğini ve performansını artıran birkaç önemli avantaj sağlar:

1.  **Sözlük Dışı (OOV) Kelimelerin Etkin İşlenmesi:** Bu, tartışmasız FastText'in en çok övülen özelliğidir. Bir kelimenin vektörü karakter n-gramlarının bir toplamı olduğu için, FastText, onu oluşturan n-gramları gözlemlendiği sürece, eğitim sözlüğünde bulunmayan herhangi bir kelime için anlamlı bir vektör üretebilir. Bu durum özellikle şu alanlarda faydalıdır:
    *   **Nadir kelimeler ve neolojizmler:** Yeni kelimeler veya eğitim korpusunda düşük sıklıkta bulunan kelimeler bile iyi temsiller elde edebilir.
    *   **Yazım hataları ve yanlış yazımlar:** Yazımdaki küçük farklılıklar, modelin çoğunlukla doğru n-gramlardan bir vektör oluşturmasına yine de olanak tanıyabilir.
    *   **Sık olmayan özel isimler:** Eğitim verisinde bulunmayabilecek kişi, yer veya kuruluş adları yine de gömülebilir.

2.  **Morfolojik Varyasyonlara Karşı Sağlamlık:** Morfolojik olarak zengin dillerde (örn. Türkçe, Arapça, Fince, Almanca), tek bir kök kelimenin yüzlerce çekimli biçimi olabilir. FastText, bu biçimler arasındaki paylaşılan karakter n-gramlarını (morfemler) kullanarak modelin şunları yapmasına olanak tanır:
    *   **Anlamsal benzerlikleri yakalama:** "Koş", "koşuyor" ve "koştu" kelimeleri, "koş" köküyle ilişkili n-gramları paylaştıkları için benzer vektörlere sahip olacaktır.
    *   **Daha verimli öğrenme:** Modelin her çekimli biçim için tamamen ayrı vektörler öğrenmesine gerek kalmaz, morfolojik olarak ilgili kelimeler arasında istatistiksel gücü birleştirir. Bu, yüksek kaliteli gömülmeler öğrenmek için veri gereksinimlerini azaltır.

3.  **Nadir Kelimeler için Geliştirilmiş Gömülmeler:** Bir kelime sözlükte olsa bile çok seyrek göründüğünde, geleneksel modeller sınırlı bağlamsal kanıt nedeniyle sağlam gömülmeler öğrenmekte zorlanır. FastText, nadir kelimelerin, birçok diğer yaygın kelimede görünme olasılığı yüksek olan kendi bileşen n-gramlarından istatistiksel güç "ödünç almasına" izin vererek bunu hafifletir. Bu, daha az sıklıkta kullanılan terimler için daha güvenilir ve anlamsal olarak daha zengin temsiller sağlar.

4.  **Dil Bağımsızlığı:** Karakter n-gram yaklaşımı, belirli bir dilin dilbilgisi kurallarına veya yazısına bağlı değildir, bu da FastText'i karmaşık yazı sistemlerine veya eklemeli yapılara sahip diller de dahil olmak üzere çeşitli dillere son derece uyarlanabilir hale getirir.

**Tipik Kullanım Alanları:**

*   **Bilgi Erişimi ve Arama:** Nadir veya OOV sorgu terimleri için geliştirilmiş gömülmeler, daha alakalı arama sonuçlarına yol açabilir.
*   **Makine Çevirisi:** Hem kaynak hem de hedef dillerdeki morfolojik varyasyonların ve nadir kelimelerin daha iyi ele alınması.
*   **Adlandırılmış Varlık Tanıma (NER):** Eğitim verilerinde yeni veya seyrek olabilecek varlıkları tanıma.
*   **Duygu Analizi ve Metin Sınıflandırma:** Bir inceleme yeni veya daha az yaygın bir sıfat içerse bile, FastText yine de anlamlı bir gömülme sağlayarak daha doğru sınıflandırmaya katkıda bulunabilir.
*   **Diller Arası NLP:** Paylaşılan alt kelime kalıplarını veya ortak kökleri kullanarak diller arasında gömülmeleri potansiyel olarak hizalama.

FastText, yalnızca kelime vektörlerine kıyasla tüm n-gram vektörlerini depolamak için daha fazla bellek gerektirse de, OOV kelimeleri ve morfolojik zenginliği ele almadaki avantajları genellikle bu dezavantajı ağır basar, özellikle çeşitli ve dinamik sözlüklerle uğraşan pratik uygulamalarda.

## 4. Kod Örneği
Bu örnek, `fasttext` Python kütüphanesini kullanarak temel bir FastText kelime gömülme modelinin nasıl eğitileceğini ve kelime vektörlerinin nasıl alınacağını göstermektedir. Örneklemek için basit bir cümle listesi kullanacağız.

```python
import fasttext
import os

# Eğitim için bir metin dosyası oluşturma
# FastText genellikle düz metin dosyası bekler, her satır bir cümle olmalı
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("FastText kelime gömülmeleri için mükemmel bir araçtır.\n")
    f.write("Sözlük dışı kelimeleri çok iyi işler.\n")
    f.write("Alt kelime bilgisi performansı için anahtardır.\n")
    f.write("Bu yaklaşım morfolojik açıdan zengin diller için harikadır.\n")

# Bir FastText modeli eğitme
# Parametreler:
# input: eğitim dosyası yolu
# model: model tipi (skipgram veya cbow)
# dim: kelime vektörlerinin boyutu
# min_n: karakter n-gramlarının minimum uzunluğu
# max_n: karakter n-gramlarının maksimum uzunluğu
# epoch: epoch sayısı
# lr: öğrenme oranı
model = fasttext.train_words(
    input="data.txt",
    model="skipgram",
    dim=100,
    min_n=3,
    max_n=6,
    epoch=10,
    lr=0.05
)

# Bilinen bir kelime için vektör al
known_word_vector = model.get_word_vector("gömülmeleri")
print(f"'gömülmeleri' kelimesi için vektör (ilk 5 değer): {known_word_vector[:5]}")

# OOV bir kelime için vektör al (örn. 'alt_kelime_gömülmesi', eğitim verisinde yok)
# FastText yine de n-gramlarına dayanarak bir vektör oluşturabilir.
oov_word_vector = model.get_word_vector("alt_kelime_gömülmesi")
print(f"'alt_kelime_gömülmesi' kelimesi için vektör (ilk 5 değer): {oov_word_vector[:5]}")

# Bir kelime için en yakın 5 komşuyu bul
nearest_neighbors = model.get_nearest_neighbors("gömülmeleri", k=5)
print(f"'gömülmeleri' kelimesi için en yakın komşular: {nearest_neighbors}")

# Geçici dosyayı temizle
os.remove("data.txt")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
FastText, özellikle geleneksel kelime düzeyindeki modeller ile karakter düzeyindeki yaklaşımlar arasındaki boşluğu doldurarak kelime gömülmeleri alanında önemli bir ilerlemeyi temsil etmektedir. **Alt kelime bilgisini**, özellikle karakter n-gramları aracılığıyla ustaca dahil etmesi, **sözlük dışı (OOV) kelimelerin** ele alınması, **morfolojik zenginliğin** kullanılması ve **nadir kelimeler** için sağlam temsiller üretilmesi gibi kritik sınırlamaları etkin bir şekilde gidermesine olanak tanır.

Kelimeyi, daha küçük, paylaşılan birimlerin bileşimleri olarak görmesiyle FastText, kelime gömülmelerinin kalitesini ve kapsamını iyileştirmekle kalmaz, aynı zamanda morfolojik karmaşıklıklarından bağımsız olarak çeşitli dillere gelişmiş uyarlanabilirlik sunar. N-gram vektörlerinin depolanması nedeniyle biraz daha fazla bellek gerektirmesine rağmen, özellikle gelişen ve daha az düzenlenmiş sözlüklere sahip gerçek dünya uygulamalarında model sağlamlığı ve anlamsal anlama açısından faydaları önemlidir. FastText, yüksek kaliteli kelime temsilleri üretmek için pragmatik ve güçlü bir çözüm sunarak modern NLP araç setinde vazgeçilmez bir araç olarak konumunu sağlamlaştırmıştır.
