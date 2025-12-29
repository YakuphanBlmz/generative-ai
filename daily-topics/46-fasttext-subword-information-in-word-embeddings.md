# FastText: Subword Information in Word Embeddings

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background: The Evolution of Word Embeddings](#2-background-the-evolution-of-word-embeddings)
- [3. FastText Architecture: Harnessing Subword Units](#3-fasttext-architecture-harnessing-subword-units)
- [4. The Power of Character N-grams](#4-the-power-of-character-n-grams)
- [5. Advantages of FastText](#5-advantages-of-fasttext)
- [6. Limitations of FastText](#6-limitations-of-fasttext)
- [7. Applications of FastText](#7-applications-of-fasttext)
- [8. Code Example](#8-code-example)
- [9. Conclusion](#9-conclusion)

## 1. Introduction
In the realm of Natural Language Processing (NLP), **word embeddings** have revolutionized how machines understand and process human language by representing words as dense vectors in a continuous vector space. These representations capture semantic and syntactic relationships, enabling downstream NLP tasks to perform with remarkable accuracy. While models like Word2Vec and GloVe effectively embed whole words, they face inherent limitations when dealing with **out-of-vocabulary (OOV) words** and languages with rich morphology.

**FastText**, an extension of Word2Vec developed by Facebook AI Research (FAIR), addresses these challenges by incorporating **subword information** into its embedding generation process. Instead of treating each word as an atomic unit, FastText decomposes words into their constituent **character n-grams** (subwords). This innovative approach allows FastText to generate robust word representations for rare words, infer meanings for OOV words, and handle morphologically complex languages more effectively, thereby significantly enhancing the versatility and performance of word embeddings in diverse linguistic contexts. This document provides a comprehensive overview of FastText, exploring its architecture, advantages, limitations, and practical applications.

## 2. Background: The Evolution of Word Embeddings
Before the advent of deep learning, traditional NLP approaches relied heavily on **one-hot encodings** or sparse count-based representations like **Term Frequency-Inverse Document Frequency (TF-IDF)**. While simple, these methods suffered from the curse of dimensionality, lacked the ability to capture semantic relationships, and struggled with synonymy and polysemy. The groundbreaking work on **distributional semantics** posited that words appearing in similar contexts tend to have similar meanings, laying the foundation for modern word embeddings.

The true breakthrough came with **Word2Vec**, introduced by Mikolov et al. in 2013. Word2Vec comprises two neural network architectures: **Continuous Bag-of-Words (CBOW)** and **Skip-gram**. CBOW predicts a word given its context, while Skip-gram predicts the context given a word. Both models learn dense, low-dimensional vector representations where semantic relationships are encoded by vector arithmetic (e.g., King - Man + Woman ≈ Queen). Following Word2Vec, **GloVe (Global Vectors for Word Representation)** emerged, combining the global matrix factorization of Latent Semantic Analysis (LSA) with the local context window methods of Word2Vec. GloVe constructs a word-word co-occurrence matrix and then factorizes it to obtain word vectors that capture both local and global statistical information.

Despite their success, Word2Vec and GloVe share a fundamental limitation: they learn embeddings for entire words. This poses problems for:
*   **Out-of-Vocabulary (OOV) words:** Any word not seen during training cannot be assigned an embedding, leading to a loss of information.
*   **Morphologically rich languages:** Languages like Turkish, Finnish, or German have extensive inflectional and derivational morphology, where a single root word can generate hundreds of forms (e.g., "run," "running," "runs," "runner"). Treating each form as a distinct word drastically increases vocabulary size and makes it difficult to learn meaningful representations for less frequent forms.
*   **Rare words:** Infrequent words often lack sufficient context for stable embedding learning, resulting in poorer quality representations.

FastText was specifically designed to mitigate these issues by looking beyond whole-word units, introducing the concept of subword information to enrich word representations.

## 3. FastText Architecture: Harnessing Subword Units
FastText's core innovation lies in its treatment of words as bags of **character n-grams**, rather than atomic units. This paradigm shift allows the model to leverage morphological regularities and handle OOV words gracefully. While its underlying neural network architecture is similar to Word2Vec's CBOW model, the key distinction lies in its input representation.

In a standard CBOW model, the input layer receives one-hot encodings of context words, and the output layer predicts the target word. In FastText, for each word in the vocabulary, its embedding is not learned directly. Instead, FastText trains embeddings for the **character n-grams** that constitute the word, along with a special full-word token.

Let's illustrate with an example: the word "apple". If we consider character n-grams of length 3 (trigrams), "apple" would be broken down into:
*   `<ap`
*   `app`
*   `ppl`
*   `ple`
*   `le>`
*   Additionally, the special token `<apple>` representing the full word itself is also included. (Angle brackets ` < > ` are added to distinguish n-grams from full words and capture prefixes/suffixes).

During training, FastText learns vector representations for each of these character n-grams. When generating the embedding for a word like "apple", FastText simply **sums up the vector representations of all its constituent character n-grams and its full-word token**. This sum constitutes the final word embedding.

The training objective remains similar to CBOW: given a context (a sum of its word embeddings), predict the target word. However, the prediction step is optimized using **Hierarchical Softmax** or **Negative Sampling**, similar to Word2Vec, to manage the large output vocabulary. FastText can also be used for text classification, where it aggregates word embeddings (derived from n-grams) to form sentence embeddings, which are then passed to a linear classifier. This makes it a powerful and efficient baseline for text classification tasks.

The choice of `n-gram` length is crucial. Typically, `min_n` (minimum n-gram length) and `max_n` (maximum n-gram length) parameters are configured, e.g., `min_n=3, max_n=6`, meaning n-grams from trigrams to hexagrams are considered. This range allows the model to capture both short, common morphemes and longer subword patterns.

## 4. The Power of Character N-grams
The use of character n-grams is the cornerstone of FastText's effectiveness, particularly in overcoming the limitations of previous word embedding models. This mechanism offers several significant advantages:

1.  **Handling Out-of-Vocabulary (OOV) Words:** When FastText encounters a word that was not present in its training vocabulary, it can still construct a meaningful embedding. It does this by decomposing the OOV word into its character n-grams. If the model has learned embeddings for these n-grams during training (which is highly probable, as n-grams are much more frequent and reusable than full words), it can sum their vectors to create an embedding for the OOV word. This "compositionality" provides a robust way to infer the meaning of unseen words based on their structural components. For example, if "unbelievable" was OOV, but the model saw "un-", "believe", "-able", it could compose an embedding.

2.  **Robustness to Morphologically Rich Languages:** Languages with complex inflectional and derivational morphology benefit immensely. Instead of treating "run," "running," "runs," "runner," "rerun" as completely distinct entities, FastText recognizes their shared root ("run") and common suffixes/prefixes. By learning embeddings for `run`, `ing`, `s`, `er`, `re-`, the model can efficiently represent all these variations. This leads to:
    *   **Reduced vocabulary size:** The number of unique character n-grams is far smaller than the number of unique word forms in morphologically complex languages.
    *   **Better representations for rare forms:** Even if a specific word form (e.g., "unbeknownst") is rare, its constituent n-grams ("un", "beknown", "st") are likely common, leading to a more stable and accurate embedding.
    *   **Capturing grammatical nuances:** The embeddings for related words will naturally be closer in the vector space due to shared n-grams, reflecting their morphological and semantic connections.

3.  **Capturing Subword Semantic Information:** Character n-grams can encode subtle semantic information related to prefixes, suffixes, and root words. For instance, words sharing the prefix "anti-" (e.g., "antivirus", "antisocial") will have similar n-gram components and thus closer embeddings, reflecting their common opposition-related meaning. This allows FastText to capture nuances that whole-word models might miss.

4.  **Improved Embeddings for Rare Words:** Similar to OOV words, rare words benefit from shared n-gram components. If a word appears infrequently, its context might be too sparse to learn a reliable whole-word embedding. However, its character n-grams are likely to be more frequent across the corpus within other words, allowing for more stable learning of their representations, which then aggregate to form a better rare word embedding.

In essence, character n-grams allow FastText to generalize beyond specific word forms, creating a more flexible and powerful word embedding model capable of handling the inherent complexities and irregularities of natural language.

## 5. Advantages of FastText
FastText offers several compelling advantages over traditional word embedding models like Word2Vec and GloVe, making it a highly valuable tool in many NLP applications:

1.  **Effective Handling of Out-of-Vocabulary (OOV) Words:** This is arguably its most significant advantage. By constructing embeddings for OOV words from their constituent character n-grams, FastText can generate reasonable representations even for words it has never encountered during training. This is crucial for real-world applications where new words, typos, or domain-specific terminology frequently appear.

2.  **Robustness in Morphologically Rich Languages:** FastText excels in languages with extensive inflectional and derivational morphology (e.g., Turkish, Arabic, German, Finnish). It implicitly captures morphological patterns by learning embeddings for common prefixes, suffixes, and root forms. This means that words like "run," "running," and "runner" will have inherently similar embeddings due to their shared "run" n-grams, reflecting their semantic relatedness more accurately than whole-word models.

3.  **Improved Embeddings for Rare Words:** Words that appear infrequently in the training corpus often receive poor quality embeddings in Word2Vec/GloVe due to insufficient contextual information. FastText mitigates this by allowing rare words to share statistical strength with other words that contain similar character n-grams. The n-gram embeddings are typically learned from a larger pool of occurrences across different words, leading to more stable and meaningful representations for rare words.

4.  **Capturing Subword Semantic Information:** Beyond just handling morphology, FastText can capture semantic nuances conveyed by prefixes and suffixes. For example, words with the prefix "un-" (e.g., "unhappy", "untrue") share a negation component, which FastText can implicitly learn and embed, leading to semantically closer vectors for such words.

5.  **Efficiency for Text Classification:** FastText is not only a word embedding model but also a highly efficient and effective baseline for **text classification**. It can be trained to perform classification tasks by taking the average of word embeddings (derived from n-grams) for a given text and feeding this into a linear classifier. This approach is surprisingly competitive with more complex deep learning models for many text classification benchmarks, offering faster training and inference times.

6.  **Reduced Vocabulary Size (for character n-grams):** While the total number of unique character n-grams might be large, it is often more manageable than the full word vocabulary in highly inflectional languages. This can lead to more efficient memory usage in some contexts.

7.  **Open-Source and Well-Maintained:** Developed by Facebook AI Research, FastText is open-source, actively maintained, and comes with pre-trained models for 157 languages, making it readily accessible and deployable for a wide range of NLP tasks.

These advantages collectively make FastText a powerful and versatile tool, bridging the gap between traditional word embeddings and the complexities of natural language, especially in low-resource settings or for languages with rich morphological structures.

## 6. Limitations of FastText
Despite its numerous advantages, particularly in handling OOV words and morphologically rich languages, FastText is not without its limitations. Understanding these drawbacks is crucial for deciding when and where to deploy FastText effectively:

1.  **Increased Model Size and Memory Footprint:** The primary cost of FastText's subword approach is the significantly larger number of parameters. Instead of learning embeddings for only `V` words (where `V` is vocabulary size), FastText learns embeddings for all unique character n-grams, which can be orders of magnitude greater than `V`. This leads to much larger model files and higher memory consumption during training and inference, especially when using a wide range of `min_n` and `max_n` values.

2.  **Computationally More Intensive Training (in some aspects):** While FastText is often touted for its speed in text classification, the training process for generating word embeddings can be more computationally intensive than basic Word2Vec models. This is because for each word, its embedding is derived by summing multiple n-gram vectors. During optimization, updates need to be propagated back to all relevant n-gram vectors, leading to more computations per word in the training batch.

3.  **Less Effective for Non-Morphological Languages:** For languages with very little morphology (e.g., English to a lesser extent, or some analytic languages), the benefits of subword information might be less pronounced. While it still helps with OOV words and typos, the gains compared to a well-trained Word2Vec or GloVe model might not justify the increased model complexity and size.

4.  **Potential for Over-Segmentation or Under-Segmentation:** The choice of `min_n` and `max_n` parameters can be critical and dataset-dependent. If `min_n` is too small, it might create too many trivial or noisy n-grams. If `max_n` is too large, it might create n-grams that are almost as long as full words, diminishing the subword benefit and increasing parameter count unnecessarily. Finding the optimal range often requires empirical tuning.

5.  **Difficulty in Distinguishing Homographs:** Like many context-independent word embedding models, FastText still struggles with **homographs** (words spelled the same but with different meanings, e.g., "bank" as a financial institution vs. "bank" as a river bank). Since the embedding is a sum of its n-grams, and the n-grams are identical, the model will produce a single, averaged representation for such words, irrespective of their context. Contextualized embeddings (e.g., ELMo, BERT) are better suited for this challenge.

6.  **Less Interpretable Subword Embeddings:** While word embeddings are somewhat interpretable (e.g., analogies), the individual character n-gram embeddings are much harder to interpret directly. Their meaning is only truly realized when combined to form a full word embedding.

In summary, while FastText provides a powerful mechanism for leveraging subword information, its increased resource requirements and potential for diminishing returns in certain linguistic contexts necessitate careful consideration during model selection and deployment.

## 7. Applications of FastText
FastText's unique capabilities, particularly its ability to handle OOV words and morphologically rich languages, make it a versatile tool across a wide array of Natural Language Processing applications.

1.  **Text Classification:** This is one of FastText's primary applications and where it truly shines as a simple yet powerful baseline. FastText provides a fast and efficient algorithm for text classification by taking the average of the word embeddings (which are themselves compositions of character n-gram embeddings) in a document and feeding this into a linear classifier. This approach achieves competitive accuracy on many benchmarks, often outperforming more complex deep learning models while being orders of magnitude faster to train and predict. Examples include sentiment analysis, spam detection, topic categorization, and language identification.

2.  **Semantic Search and Information Retrieval:** By generating robust word embeddings, FastText enables more effective semantic search. Queries can be matched with documents not just on keyword exactness, but also on semantic similarity. For instance, if a user searches for "automobile," documents containing "car" or "vehicle" might also be retrieved, thanks to the closer vector representations. Its ability to handle OOV words means that queries with typos or less common terms can still yield relevant results.

3.  **Machine Translation (as a component):** While not a full machine translation system itself, FastText embeddings can serve as crucial input features for neural machine translation (NMT) models. By providing richer, subword-aware representations, especially for low-frequency words or in morphologically complex source/target languages, FastText can improve the overall quality of translation.

4.  **Named Entity Recognition (NER) and Part-of-Speech (POS) Tagging:** Embeddings from FastText can be used as features in sequence labeling models (e.g., CRF, Bi-LSTM-CRF) for tasks like NER and POS tagging. Its ability to handle OOV words is particularly beneficial for NER, where new entity names frequently appear. The subword information helps the model generalize better to unseen proper nouns or complex morphological variations.

5.  **Spell Checking and Correction:** FastText's inherent understanding of subword structure makes it highly valuable for identifying and correcting spelling errors. A misspelled word's embedding, even if it's OOV, can be calculated. By finding words in the vocabulary with the closest embeddings (and similar character n-grams), potential corrections can be suggested.

6.  **Language Modeling and Generation:** While less direct than its classification role, FastText's robust word embeddings can improve the quality of language models. Better word representations lead to more accurate probability distributions over sequences of words, which is fundamental for tasks like text generation or auto-completion.

7.  **Cross-Lingual Word Embeddings:** FastText's architecture lends itself to learning **cross-lingual word embeddings**. By training FastText on parallel corpora or using specific alignment techniques, it's possible to create embedding spaces where words with similar meanings in different languages are close to each other, even for morphologically diverse languages.

The adaptability, efficiency, and strong performance across these diverse applications underscore FastText's status as a fundamental and widely-used tool in the modern NLP toolkit.

## 8. Code Example
This Python code snippet demonstrates how to train a FastText model using the `gensim` library. We'll use a small, sample corpus.

```python
from gensim.models import FastText
from gensim.utils import simple_preprocess
import logging

# Configure logging for better visibility during training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Sample corpus (list of sentences)
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "FastText is a powerful tool for natural language processing.",
    "It handles out-of-vocabulary words and morphologically rich languages effectively.",
    "Subword information, specifically character n-grams, are key to its success.",
    "Running, runs, runner, rerun all share the root run and related n-grams.",
    "The cat sat on the mat."
]

# Preprocess the corpus: tokenize and lowercase each sentence
# simple_preprocess removes punctuation, converts to lowercase, and tokenizes
processed_corpus = [simple_preprocess(doc) for doc in corpus]

print("Processed Corpus:")
for sentence in processed_corpus:
    print(sentence)

# Train the FastText model
# vector_size: Dimensionality of the word vectors
# window: Maximum distance between the current and predicted word within a sentence
# min_count: Ignores all words with total frequency lower than this
# workers: Use these many worker threads to train the model
# sg: 1 for skip-gram, 0 for CBOW (FastText uses skip-gram by default for word embeddings)
# min_n: Minimum length of character n-grams
# max_n: Maximum length of character n-grams
# epochs: Number of training iterations
model = FastText(
    sentences=processed_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    min_n=3,
    max_n=6,
    epochs=10
)

print("\nFastText model training complete.")

# Example usage: Get word vectors
word_vector_fast = model.wv['fast']
word_vector_text = model.wv['text']
word_vector_running = model.wv['running']

print(f"\nVector for 'fast' (first 5 dimensions): {word_vector_fast[:5]}")
print(f"Vector for 'text' (first 5 dimensions): {word_vector_text[:5]}")
print(f"Vector for 'running' (first 5 dimensions): {word_vector_running[:5]}")

# Demonstrate OOV handling (e.g., a misspelled word or a rare word)
# Let's assume 'runnnning' is an OOV word or a typo
# FastText can still provide an embedding by combining n-grams
oov_word = "runnnning" # Typo
oov_vector = model.wv[oov_word]
print(f"\nVector for OOV word '{oov_word}' (first 5 dimensions): {oov_vector[:5]}")

# Find most similar words
print("\nWords most similar to 'running':")
print(model.wv.most_similar('running'))

print("\nWords most similar to 'cat':")
print(model.wv.most_similar('cat'))

print("\nWord similarity between 'fast' and 'text':")
print(model.wv.similarity('fast', 'text'))

print("\nWord similarity between 'cat' and 'dog':")
print(model.wv.similarity('cat', 'dog'))

(End of code example section)
```

## 9. Conclusion
FastText stands as a significant advancement in the field of word embeddings, building upon the foundational concepts of Word2Vec while ingeniously addressing its inherent limitations. By decomposing words into **character n-grams**, FastText successfully integrates **subword information** into its vector representations, enabling it to generate robust embeddings for **out-of-vocabulary (OOV) words** and provide superior representations for **morphologically rich languages**.

Its capacity to infer meaning from word structure, rather than treating words as atomic units, makes it exceptionally valuable in practical scenarios involving noisy text, rare terminology, or diverse linguistic datasets. Furthermore, its efficiency as a baseline for **text classification** tasks underscores its versatility and practical utility. While FastText introduces a larger model footprint and increased computational demands during training due to its reliance on a vast dictionary of character n-grams, these trade-offs are often justified by its enhanced performance in handling linguistic complexities.

In summary, FastText has cemented its position as an indispensable tool in the modern NLP toolkit, offering a powerful and pragmatic solution for deriving meaningful word representations that generalize effectively across a broad spectrum of languages and application domains. Its contribution lies in pushing the boundaries of word embedding models towards a more nuanced and resilient understanding of human language.

---
<br>

<a name="türkçe-içerik"></a>
## FastText: Kelime Gömülmelerinde Alt Kelime Bilgisi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan: Kelime Gömülmelerinin Evrimi](#2-arka-plan-kelime-gömülmelerinin-evrimi)
- [3. FastText Mimarisi: Alt Kelime Birimlerinden Yararlanma](#3-fasttext-mimarisi-alt-kelime-birimlerinden-yararlanma)
- [4. Karakter N-gramlarının Gücü](#4-karakter-n-gramlarının-gücü)
- [5. FastText'in Avantajları](#5-fasttextin-avantajları)
- [6. FastText'in Sınırlamaları](#6-fasttextin-sınırlamaları)
- [7. FastText'in Uygulamaları](#7-fasttextin-uygulamaları)
- [8. Kod Örneği](#8-kod-örneği)
- [9. Sonuç](#9-sonuç)

## 1. Giriş
Doğal Dil İşleme (NLP) alanında, **kelime gömülmeleri** (word embeddings) kelimeleri sürekli bir vektör uzayında yoğun vektörler olarak temsil ederek makinelerin insan dilini anlama ve işleme biçimini devrim niteliğinde değiştirdi. Bu temsiller, anlamsal ve sözdizimsel ilişkileri yakalayarak alt düzey NLP görevlerinin dikkat çekici bir doğrulukla performans göstermesini sağladı. Word2Vec ve GloVe gibi modeller tüm kelimeleri etkili bir şekilde gömse de, **kelime dağarcığı dışı (OOV) kelimeler** ve zengin morfolojiye sahip dillerle başa çıkmada doğal sınırlamalara sahiptir.

Facebook AI Research (FAIR) tarafından geliştirilen Word2Vec'in bir uzantısı olan **FastText**, kelime gömülme oluşturma sürecine **alt kelime bilgisini** dahil ederek bu zorlukların üstesinden gelir. Her kelimeyi atomik bir birim olarak ele almak yerine, FastText kelimeleri oluşturan **karakter n-gramlarına** (alt kelimeler) ayırır. Bu yenilikçi yaklaşım, FastText'in nadir kelimeler için sağlam kelime temsilleri oluşturmasına, OOV kelimeler için anlamlar çıkarabilmesine ve morfolojik olarak karmaşık dilleri daha etkili bir şekilde işlemesine olanak tanır, böylece kelime gömülmelerinin çok çeşitli dilsel bağlamlarda çok yönlülüğünü ve performansını önemli ölçüde artırır. Bu belge, FastText'in mimarisini, avantajlarını, sınırlamalarını ve pratik uygulamalarını keşfeden kapsamlı bir genel bakış sunmaktadır.

## 2. Arka Plan: Kelime Gömülmelerinin Evrimi
Derin öğrenmenin ortaya çıkışından önce, geleneksel NLP yaklaşımları büyük ölçüde **tek-sıcak kodlamalara** (one-hot encodings) veya **Terim Sıklığı-Ters Belge Sıklığı (TF-IDF)** gibi seyrek sayıma dayalı gösterimlere dayanıyordu. Basit olsalar da, bu yöntemler boyutluluğun laneti sorunundan muzdaripti, anlamsal ilişkileri yakalama yeteneğinden yoksundu ve eşanlamlılık ve çokanlamlılık ile mücadele ediyordu. **Dağılımsal anlambilim** üzerine çığır açan çalışmalar, benzer bağlamlarda ortaya çıkan kelimelerin benzer anlamlara sahip olma eğiliminde olduğunu varsayarak modern kelime gömülmelerinin temelini attı.

Gerçek atılım, Mikolov ve arkadaşları tarafından 2013'te tanıtılan **Word2Vec** ile geldi. Word2Vec iki sinir ağı mimarisinden oluşur: **Sürekli Kelime Torbası (CBOW)** ve **Skip-gram**. CBOW, bağlamı verilen bir kelimeyi tahmin ederken, Skip-gram bir kelime verildiğinde bağlamı tahmin eder. Her iki model de anlamsal ilişkilerin vektör aritmetiği ile kodlandığı yoğun, düşük boyutlu vektör temsilleri öğrenir (örn. Kral - Erkek + Kadın ≈ Kraliçe). Word2Vec'i takiben, Latent Semantik Analizi'nin (LSA) küresel matris çarpanlara ayırmasını Word2Vec'in yerel bağlam penceresi yöntemleriyle birleştiren **GloVe (Global Vectors for Word Representation)** ortaya çıktı. GloVe, bir kelime-kelime birlikte oluşum matrisi oluşturur ve ardından hem yerel hem de küresel istatistiksel bilgiyi yakalayan kelime vektörlerini elde etmek için bunu çarpanlara ayırır.

Başarılarına rağmen, Word2Vec ve GloVe temel bir sınırlamayı paylaşır: tüm kelimeler için gömülmeler öğrenirler. Bu durum, aşağıdaki sorunlara yol açar:
*   **Kelime Dağarcığı Dışı (OOV) Kelimeler:** Eğitim sırasında görülmeyen herhangi bir kelimeye gömülme atanamaz, bu da bilgi kaybına yol açar.
*   **Morfolojik Olarak Zengin Diller:** Türkçe, Fince veya Almanca gibi diller, tek bir kök kelimenin yüzlerce form üretebildiği (örn. "koşmak," "koşuyor," "koşar," "koşucu") kapsamlı çekimsel ve türetimsel morfolojiye sahiptir. Her formu ayrı bir kelime olarak ele almak, kelime dağarcığı boyutunu büyük ölçüde artırır ve daha az sıklıkta olan formlar için anlamlı temsiller öğrenmeyi zorlaştırır.
*   **Nadir Kelimeler:** Sık olmayan kelimeler, genellikle kararlı gömülme öğrenimi için yeterli bağlamdan yoksundur ve bu da daha düşük kaliteli temsillerle sonuçlanır.

FastText, kelime temsillerini zenginleştirmek için alt kelime bilgisini tanıtarak bu sorunları hafifletmek için özel olarak tasarlanmıştır.

## 3. FastText Mimarisi: Alt Kelime Birimlerinden Yararlanma
FastText'in temel yeniliği, kelimeleri atomik birimler yerine **karakter n-gramları** torbası olarak ele almasında yatmaktadır. Bu paradigma değişimi, modelin morfolojik düzenliliklerden yararlanmasını ve OOV kelimeleri sorunsuz bir şekilde işlemesini sağlar. Temel sinir ağı mimarisi Word2Vec'in CBOW modeline benzese de, temel fark giriş temsilindedir.

Standart bir CBOW modelinde, giriş katmanı bağlam kelimelerinin tek-sıcak kodlamalarını alır ve çıkış katmanı hedef kelimeyi tahmin eder. FastText'te, kelime dağarcığındaki her kelime için gömülmesi doğrudan öğrenilmez. Bunun yerine, FastText kelimeyi oluşturan **karakter n-gramları** ve özel bir tam kelime belirteci için gömülmeler eğitir.

Bir örnekle açıklayalım: "elma" kelimesi. Uzunluğu 3 olan karakter n-gramlarını (trigramlar) ele alırsak, "elma" şu şekilde ayrılır:
*   `<el`
*   `elm`
*   `lma`
*   `ma>`
*   Ek olarak, tam kelimenin kendisini temsil eden özel belirteç `<elma>` da dahil edilir. (N-gramları tam kelimelerden ayırmak ve önekleri/sonekleri yakalamak için açılı parantezler ` < > ` eklenir).

Eğitim sırasında FastText, bu karakter n-gramlarının her biri için vektör temsilleri öğrenir. "Elma" gibi bir kelime için gömülme oluşturulurken, FastText basitçe **onu oluşturan tüm karakter n-gramlarının ve tam kelime belirtecinin vektör temsillerini toplar**. Bu toplam, nihai kelime gömülmesini oluşturur.

Eğitim hedefi CBOW'a benzer kalır: bir bağlam (kelime gömülmelerinin toplamı) verildiğinde, hedef kelimeyi tahmin etmek. Ancak, tahmin adımı, Word2Vec'e benzer şekilde, büyük çıktı kelime dağarcığını yönetmek için **Hiyerarşik Softmax** veya **Negatif Örnekleme** kullanılarak optimize edilir. FastText, metin sınıflandırması için de kullanılabilir, burada kelime gömülmelerini (n-gramlardan türetilmiş) cümle gömülmeleri oluşturmak üzere toplar ve bunlar daha sonra doğrusal bir sınıflandırıcıya geçirilir. Bu, metin sınıflandırma görevleri için onu güçlü ve verimli bir temel yapar.

`n-gram` uzunluğunun seçimi kritiktir. Genellikle `min_n` (minimum n-gram uzunluğu) ve `max_n` (maksimum n-gram uzunluğu) parametreleri yapılandırılır, örneğin `min_n=3, max_n=6`, yani trigramlardan heksagramlara kadar n-gramlar dikkate alınır. Bu aralık, modelin hem kısa, yaygın morfemleri hem de daha uzun alt kelime kalıplarını yakalamasına olanak tanır.

## 4. Karakter N-gramlarının Gücü
Karakter n-gramlarının kullanımı, özellikle önceki kelime gömülme modellerinin sınırlamalarının üstesinden gelmede FastText'in etkinliğinin temel taşıdır. Bu mekanizma birkaç önemli avantaj sunar:

1.  **Kelime Dağarcığı Dışı (OOV) Kelimelerin İşlenmesi:** FastText, eğitim kelime dağarcığında bulunmayan bir kelimeyle karşılaştığında, yine de anlamlı bir gömülme oluşturabilir. Bunu, OOV kelimeyi karakter n-gramlarına ayırarak yapar. Model, bu n-gramlar için eğitim sırasında gömülmeler öğrendiyse (n-gramlar, tam kelimelerden çok daha sık ve yeniden kullanılabilir olduğundan bu oldukça olasıdır), vektörlerini toplayarak OOV kelime için bir gömülme oluşturabilir. Bu "bileşiklik", bilinmeyen kelimelerin anlamını yapısal bileşenlerine göre çıkarmanın sağlam bir yolunu sunar. Örneğin, "inanılmaz" OOV ise, ancak model "in-", "inan", "-ılmaz"ı gördüyse, bir gömülme oluşturabilir.

2.  **Morfolojik Olarak Zengin Dillere Karşı Sağlamlık:** Karmaşık çekimsel ve türetimsel morfolojiye sahip diller (örn. Türkçe, Arapça, Almanca, Fince) büyük ölçüde fayda sağlar. FastText, "koşmak," "koşuyor," "koşar," "koşucu," "yeniden koşmak" gibi kelimeleri tamamen farklı varlıklar olarak ele almak yerine, ortak köklerini ("koş") ve yaygın sonek/öneklerini tanır. `koş`, `uyor`, `ar`, `ucu`, `yeniden-` için gömülmeler öğrenerek, model tüm bu varyasyonları verimli bir şekilde temsil edebilir. Bu durum şunlara yol açar:
    *   **Azaltılmış kelime dağarcığı boyutu:** Morfolojik olarak karmaşık dillerde benzersiz karakter n-gramlarının sayısı, benzersiz kelime formlarının sayısından çok daha küçüktür.
    *   **Nadir formlar için daha iyi temsiller:** Belirli bir kelime formu (örn. "bilinmedik") nadir olsa bile, onu oluşturan n-gramlar ("bilin", "medik") korpus içinde diğer kelimelerde yaygın olabilir ve bu da daha kararlı ve doğru bir gömülmeye yol açar.
    *   **Dilbilgisel nüansları yakalama:** İlişkili kelimelerin gömülmeleri, paylaşılan n-gramlar nedeniyle vektör uzayında doğal olarak birbirine daha yakın olacak ve morfolojik ve anlamsal bağlantılarını yansıtacaktır.

3.  **Alt Kelime Anlamsal Bilgisini Yakalama:** Karakter n-gramları, önekler, sonekler ve kök kelimelerle ilgili ince anlamsal bilgileri kodlayabilir. Örneğin, "anti-" önekini paylaşan kelimeler (örn. "antivirüs", "antisosyal"), benzer n-gram bileşenlerine ve dolayısıyla daha yakın gömülmelere sahip olacak, bu da ortak karşıtlıkla ilgili anlamlarını yansıtacaktır. Bu, FastText'in tüm kelime modellerinin kaçırabileceği nüansları yakalamasına olanak tanır.

4.  **Nadir Kelimeler İçin Geliştirilmiş Gömülmeler:** OOV kelimelerine benzer şekilde, nadir kelimeler de paylaşılan n-gram bileşenlerinden faydalanır. Bir kelime seyrek olarak ortaya çıkarsa, bağlamı güvenilir bir tüm kelime gömülmesi öğrenmek için çok seyrek olabilir. Ancak, karakter n-gramları, korpus boyunca diğer kelimelerde daha sık görülebilir ve bu da temsillerinin daha kararlı bir şekilde öğrenilmesine olanak tanır, bu da daha iyi bir nadir kelime gömülmesi oluşturmak için birleşir.

Özünde, karakter n-gramları FastText'in belirli kelime formlarının ötesine genelleşmesini sağlayarak, doğal dilin doğal karmaşıklıklarını ve düzensizliklerini ele alabilen daha esnek ve güçlü bir kelime gömülme modeli yaratır.

## 5. FastText'in Avantajları
FastText, Word2Vec ve GloVe gibi geleneksel kelime gömülme modellerine göre birkaç çekici avantaj sunar ve bu da onu birçok NLP uygulamasında son derece değerli bir araç haline getirir:

1.  **Kelime Dağarcığı Dışı (OOV) Kelimelerin Etkili İşlenmesi:** Bu, tartışmasız en önemli avantajıdır. OOV kelimeler için gömülmeleri oluşturan karakter n-gramlarından oluşturarak, FastText eğitim sırasında hiç karşılaşmadığı kelimeler için bile makul temsiller üretebilir. Bu, yeni kelimelerin, yazım hatalarının veya alana özgü terminolojinin sıkça ortaya çıktığı gerçek dünya uygulamaları için çok önemlidir.

2.  **Morfolojik Olarak Zengin Dillerde Sağlamlık:** FastText, kapsamlı çekimsel ve türetimsel morfolojiye sahip dillerde (örn. Türkçe, Arapça, Almanca, Fince) öne çıkar. Ortak önekler, sonekler ve kök formları için gömülmeler öğrenerek morfolojik kalıpları örtük olarak yakalar. Bu, "koşmak," "koşuyor" ve "koşucu" gibi kelimelerin, paylaşılan "koş" n-gramları nedeniyle doğal olarak benzer gömülmelere sahip olacağı anlamına gelir ve anlamsal ilişkilerini tüm kelime modellerinden daha doğru bir şekilde yansıtır.

3.  **Nadir Kelimeler İçin Geliştirilmiş Gömülmeler:** Eğitim korpusunda seyrek olarak ortaya çıkan kelimeler, yetersiz bağlamsal bilgi nedeniyle Word2Vec/GloVe'de genellikle düşük kaliteli gömülmeler alır. FastText, nadir kelimelerin benzer karakter n-gramlarını içeren diğer kelimelerle istatistiksel gücü paylaşmasına izin vererek bunu hafifletir. N-gram gömülmeleri, genellikle farklı kelimelerdeki daha büyük bir oluşum havuzundan öğrenilir ve bu da nadir kelimeler için daha kararlı ve anlamlı temsillerle sonuçlanır.

4.  **Alt Kelime Anlamsal Bilgisini Yakalama:** Sadece morfolojiyi ele almanın ötesinde, FastText önekler ve sonekler tarafından iletilen anlamsal nüansları yakalayabilir. Örneğin, "un-" önekini içeren kelimeler (örn. "mutsuz", "doğru olmayan") bir olumsuzluk bileşenini paylaşır, FastText bunu örtük olarak öğrenebilir ve gömebilir, bu da bu tür kelimeler için anlamsal olarak daha yakın vektörlere yol açar.

5.  **Metin Sınıflandırması İçin Verimlilik:** FastText sadece bir kelime gömülme modeli değil, aynı zamanda **metin sınıflandırması** için son derece verimli ve etkili bir temeldir. Bir metindeki kelime gömülmelerinin (karakter n-gram gömülmelerinin bileşimleri olan) ortalamasını alarak ve bunu doğrusal bir sınıflandırıcıya besleyerek sınıflandırma görevlerini gerçekleştirmek üzere eğitilebilir. Bu yaklaşım, birçok metin sınıflandırma kıyaslamasında daha karmaşık derin öğrenme modelleriyle şaşırtıcı derecede rekabetçi bir doğruluk elde ederken, daha hızlı eğitim ve çıkarım süreleri sunar.

6.  **Azaltılmış Kelime Dağarcığı Boyutu (karakter n-gramları için):** Benzersiz karakter n-gramlarının toplam sayısı büyük olsa da, yüksek derecede çekimsel dillerde tam kelime dağarcığından genellikle daha yönetilebilirdir. Bu, bazı bağlamlarda daha verimli bellek kullanımına yol açabilir.

7.  **Açık Kaynak ve İyi Bakımlı:** Facebook AI Research tarafından geliştirilen FastText, açık kaynaklıdır, aktif olarak sürdürülmektedir ve 157 dil için önceden eğitilmiş modellerle birlikte gelir, bu da onu çok çeşitli NLP görevleri için kolayca erişilebilir ve dağıtılabilir kılar.

Bu avantajlar toplu olarak FastText'i güçlü ve çok yönlü bir araç haline getirir, geleneksel kelime gömülmeleri ile doğal dilin karmaşıklıkları arasındaki boşluğu doldurur, özellikle düşük kaynaklı ayarlarda veya zengin morfolojik yapılara sahip diller için.

## 6. FastText'in Sınırlamaları
FastText'in özellikle OOV kelimeleri ve morfolojik olarak zengin dilleri işleme konusunda birçok avantajına rağmen, sınırlamaları da mevcuttur. Bu dezavantajları anlamak, FastText'i ne zaman ve nerede etkili bir şekilde kullanacağınıza karar vermek için çok önemlidir:

1.  **Artan Model Boyutu ve Bellek Ayak İzi:** FastText'in alt kelime yaklaşımının temel maliyeti, önemli ölçüde daha fazla sayıda parametredir. Yalnızca `V` kelime (burada `V` kelime dağarcığı boyutudur) için gömülmeler öğrenmek yerine, FastText tüm benzersiz karakter n-gramları için gömülmeler öğrenir, bu da `V`'den kat kat daha fazla olabilir. Bu durum, özellikle geniş bir `min_n` ve `max_n` değer aralığı kullanıldığında, çok daha büyük model dosyalarına ve eğitim ve çıkarım sırasında daha yüksek bellek tüketimine yol açar.

2.  **Hesaplama Açısından Daha Yoğun Eğitim (bazı yönlerden):** FastText genellikle metin sınıflandırmasındaki hızıyla övülse de, kelime gömülmeleri oluşturma eğitim süreci temel Word2Vec modellerinden daha hesaplama açısından yoğun olabilir. Bunun nedeni, her kelime için gömülmesinin birden çok n-gram vektörünün toplanmasıyla elde edilmesidir. Optimizasyon sırasında, güncellemelerin ilgili tüm n-gram vektörlerine geri yayılması gerekir, bu da eğitim toplu işindeki her kelime için daha fazla hesaplamaya yol açar.

3.  **Morfolojik Olmayan Diller İçin Daha Az Etkili:** Çok az morfolojiye sahip diller için (örn. daha az ölçüde İngilizce veya bazı analitik diller), alt kelime bilgisinin faydaları daha az belirgin olabilir. OOV kelimeler ve yazım hataları için hala yardımcı olsa da, iyi eğitilmiş bir Word2Vec veya GloVe modeline kıyasla elde edilen kazanımlar, artan model karmaşıklığını ve boyutunu haklı çıkarmayabilir.

4.  **Aşırı Bölümlendirme veya Eksik Bölümlendirme Potansiyeli:** `min_n` ve `max_n` parametrelerinin seçimi kritik olabilir ve veri setine bağlıdır. Eğer `min_n` çok küçükse, çok fazla önemsiz veya gürültülü n-gram oluşturabilir. Eğer `max_n` çok büyükse, neredeyse tam kelimeler kadar uzun n-gramlar oluşturabilir, bu da alt kelime faydasını azaltır ve parametre sayısını gereksiz yere artırır. Optimal aralığı bulmak genellikle ampirik ayarlama gerektirir.

5.  **Eşanlamlı Kelimeleri Ayırt Etmede Zorluk:** Birçok bağlamdan bağımsız kelime gömülme modeli gibi, FastText de **eşanlamlı kelimeler** (aynı yazılan ancak farklı anlamlara sahip kelimeler, örn. "banka" finans kurumu olarak, "banka" nehir kıyısı olarak) ile hala zorlanır. Gömülme n-gramlarının bir toplamı olduğundan ve n-gramlar aynı olduğundan, model bu tür kelimeler için bağlamlarından bağımsız olarak tek, ortalama bir temsil üretecektir. Bağlamsal gömülmeler (örn. ELMo, BERT) bu zorluk için daha uygundur.

6.  **Daha Az Yorumlanabilir Alt Kelime Gömülmeleri:** Kelime gömülmeleri bir dereceye kadar yorumlanabilirken (örn. benzetmeler), bireysel karakter n-gram gömülmelerini doğrudan yorumlamak çok daha zordur. Anlamları ancak tam bir kelime gömülmesi oluşturmak için birleştirildiklerinde gerçekten gerçekleşir.

Özetle, FastText alt kelime bilgisinden yararlanmak için güçlü bir mekanizma sunarken, artan kaynak gereksinimleri ve belirli dilsel bağlamlarda azalan getiriler potansiyeli, model seçimi ve dağıtımı sırasında dikkatli bir değerlendirme gerektirir.

## 7. FastText'in Uygulamaları
FastText'in benzersiz yetenekleri, özellikle OOV kelimeleri ve morfolojik olarak zengin dilleri işleme yeteneği, onu çok çeşitli Doğal Dil İşleme uygulamalarında çok yönlü bir araç haline getirir.

1.  **Metin Sınıflandırması:** Bu, FastText'in birincil uygulamalarından biridir ve basit ama güçlü bir temel olarak gerçekten parladığı yerdir. FastText, bir belgedeki kelime gömülmelerinin (karakter n-gram gömülmelerinin bileşimleri olan) ortalamasını alarak ve bunu doğrusal bir sınıflandırıcıya besleyerek metin sınıflandırma görevleri için hızlı ve verimli bir algoritma sağlar. Bu yaklaşım, birçok kıyaslamada rekabetçi doğruluk elde ederken, genellikle daha karmaşık derin öğrenme modellerinden kat kat daha hızlı eğitim ve tahmin süreleri sunar. Örnekler arasında duygu analizi, spam tespiti, konu kategorizasyonu ve dil tanımlama bulunur.

2.  **Anlamsal Arama ve Bilgi Erişimi:** Sağlam kelime gömülmeleri oluşturarak FastText, daha etkili anlamsal aramayı mümkün kılar. Sorgular, sadece anahtar kelime kesinliğine göre değil, aynı zamanda anlamsal benzerliğe göre de belgelerle eşleştirilebilir. Örneğin, bir kullanıcı "otomobil" ararsa, "araba" veya "taşıt" içeren belgeler de daha yakın vektör temsilleri sayesinde alınabilir. OOV kelimeleri işleme yeteneği, yazım hatalı veya daha az yaygın terimlerle yapılan sorguların bile ilgili sonuçlar verebileceği anlamına gelir.

3.  **Makine Çevirisi (bir bileşen olarak):** Tam bir makine çevirisi sistemi olmasa da, FastText gömülmeleri nöral makine çevirisi (NMT) modelleri için kritik girdi özellikleri olarak hizmet edebilir. Özellikle düşük frekanslı kelimeler veya morfolojik olarak karmaşık kaynak/hedef diller için daha zengin, alt kelimeye duyarlı temsiller sağlayarak, FastText çevirinin genel kalitesini artırabilir.

4.  **Adlandırılmış Varlık Tanıma (NER) ve Sözcük Türü (POS) Etiketleme:** FastText'ten elde edilen gömülmeler, NER ve POS etiketleme gibi görevler için dizi etiketleme modellerinde (örn. CRF, Bi-LSTM-CRF) özellik olarak kullanılabilir. OOV kelimeleri işleme yeteneği, özellikle yeni varlık adlarının sıkça ortaya çıktığı NER için faydalıdır. Alt kelime bilgisi, modelin bilinmeyen özel isimlere veya karmaşık morfolojik varyasyonlara daha iyi genelleşmesine yardımcı olur.

5.  **Yazım Denetimi ve Düzeltme:** FastText'in alt kelime yapısını doğal olarak anlaması, yazım hatalarını tanımlama ve düzeltme konusunda onu son derece değerli kılar. Yanlış yazılmış bir kelimenin gömülmesi, OOV olsa bile hesaplanabilir. Kelime dağarcığındaki en yakın gömülmelere (ve benzer karakter n-gramlarına) sahip kelimeleri bularak potansiyel düzeltmeler önerilebilir.

6.  **Dil Modelleme ve Üretme:** Sınıflandırma rolünden daha az doğrudan olsa da, FastText'in sağlam kelime gömülmeleri dil modellerinin kalitesini artırabilir. Daha iyi kelime temsilleri, metin üretimi veya otomatik tamamlama gibi görevler için temel olan kelime dizileri üzerinde daha doğru olasılık dağılımlarına yol açar.

7.  **Çok Dilli Kelime Gömülmeleri:** FastText'in mimarisi, **çok dilli kelime gömülmeleri** öğrenmeye elverişlidir. FastText'i paralel korpuslarda eğiterek veya belirli hizalama teknikleri kullanarak, farklı dillerdeki benzer anlamlara sahip kelimelerin birbirine yakın olduğu gömülme uzayları oluşturmak mümkündür, hatta morfolojik olarak farklı diller için bile.

Bu çeşitli uygulamalardaki uyarlanabilirlik, verimlilik ve güçlü performans, FastText'in modern NLP araç setinde temel ve yaygın olarak kullanılan bir araç olarak konumunu pekiştirmektedir.

## 8. Kod Örneği
Bu Python kod parçacığı, `gensim` kütüphanesini kullanarak bir FastText modelinin nasıl eğitileceğini göstermektedir. Küçük, örnek bir korpus kullanacağız.

```python
from gensim.models import FastText
from gensim.utils import simple_preprocess
import logging

# Eğitim sırasında daha iyi görünürlük için loglamayı yapılandırın
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Örnek korpus (cümle listesi)
corpus = [
    "Hızlı kahverengi tilki, tembel köpeğin üzerinden atlar.",
    "FastText, doğal dil işleme için güçlü bir araçtır.",
    "Kelime dağarcığı dışı kelimeleri ve morfolojik olarak zengin dilleri etkili bir şekilde ele alır.",
    "Alt kelime bilgisi, özellikle karakter n-gramları, başarısının anahtarıdır.",
    "Koşmak, koşuyor, koşucu, yeniden koşmak hepsi ortak 'koş' kökünü ve ilgili n-gramları paylaşır.",
    "Kedi minderin üzerinde oturdu."
]

# Korpusu ön işleyin: her cümleyi tokenlara ayırın ve küçük harfe dönüştürün
# simple_preprocess noktalama işaretlerini kaldırır, küçük harfe dönüştürür ve tokenlara ayırır
processed_corpus = [simple_preprocess(doc) for doc in corpus]

print("Ön İşlenmiş Korpus:")
for sentence in processed_corpus:
    print(sentence)

# FastText modelini eğitin
# vector_size: Kelime vektörlerinin boyutluluğu
# window: Bir cümle içindeki mevcut kelime ile tahmin edilen kelime arasındaki maksimum mesafe
# min_count: Toplam frekansı bundan düşük olan tüm kelimeleri göz ardı eder
# workers: Modeli eğitmek için bu kadar işçi iş parçacığı kullanın
# sg: Skip-gram için 1, CBOW için 0 (FastText kelime gömülmeleri için varsayılan olarak skip-gram kullanır)
# min_n: Karakter n-gramlarının minimum uzunluğu
# max_n: Karakter n-gramlarının maksimum uzunluğu
# epochs: Eğitim iterasyonlarının sayısı
model = FastText(
    sentences=processed_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    min_n=3,
    max_n=6,
    epochs=10
)

print("\nFastText model eğitimi tamamlandı.")

# Örnek kullanım: Kelime vektörlerini alın
word_vector_hizli = model.wv['hızlı']
word_vector_metin = model.wv['metin']
word_vector_koşmak = model.wv['koşmak']

print(f"\n'hızlı' kelimesinin vektörü (ilk 5 boyut): {word_vector_hizli[:5]}")
print(f"'metin' kelimesinin vektörü (ilk 5 boyut): {word_vector_metin[:5]}")
print(f"'koşmak' kelimesinin vektörü (ilk 5 boyut): {word_vector_koşmak[:5]}")

# OOV işleme demonstrasyonu (örn. yanlış yazılmış bir kelime veya nadir bir kelime)
# 'koşşmak' kelimesinin bir OOV kelime veya yazım hatası olduğunu varsayalım
# FastText, n-gramları birleştirerek yine de bir gömülme sağlayabilir
oov_word = "koşşmak" # Yazım hatası
oov_vector = model.wv[oov_word]
print(f"\nOOV kelime '{oov_word}' vektörü (ilk 5 boyut): {oov_vector[:5]}")

# En benzer kelimeleri bulun
print("\n'koşmak' kelimesine en benzer kelimeler:")
print(model.wv.most_similar('koşmak'))

print("\n'kedi' kelimesine en benzer kelimeler:")
print(model.wv.most_similar('kedi'))

print("\n'hızlı' ve 'metin' kelimeleri arasındaki kelime benzerliği:")
print(model.wv.similarity('hızlı', 'metin'))

print("\n'kedi' ve 'köpek' kelimeleri arasındaki kelime benzerliği:")
print(model.wv.similarity('kedi', 'köpek'))

(Kod örneği bölümünün sonu)
```

## 9. Sonuç
FastText, kelime gömülmeleri alanında önemli bir ilerleme olarak durmakta, Word2Vec'in temel kavramları üzerine inşa ederken, doğal sınırlamalarını ustaca ele almaktadır. Kelimeleri **karakter n-gramlarına** ayırarak, FastText **alt kelime bilgisini** vektör temsillerine başarıyla entegre eder ve böylece **kelime dağarcığı dışı (OOV) kelimeler** için sağlam gömülmeler oluşturabilir ve **morfolojik olarak zengin diller** için üstün temsiller sağlayabilir.

Kelimeleri atomik birimler olarak ele almak yerine, kelime yapısından anlam çıkarabilme yeteneği, gürültülü metinler, nadir terminoloji veya çeşitli dilsel veri setleri içeren pratik senaryolarda onu son derece değerli kılar. Ayrıca, **metin sınıflandırma** görevleri için bir temel olarak verimliliği, çok yönlülüğünü ve pratik faydasını vurgular. FastText, geniş bir karakter n-gram sözlüğüne dayanması nedeniyle eğitim sırasında daha büyük bir model ayak izi ve artan hesaplama talepleri getirse de, bu ödünleşmeler genellikle dilsel karmaşıklıkları ele alma konusundaki gelişmiş performansı ile haklı çıkarılır.

Özetle, FastText, modern NLP araç setinde vazgeçilmez bir araç olarak konumunu sağlamlaştırmış, geniş bir dil ve uygulama alanı yelpazesinde etkili bir şekilde genelleşen anlamlı kelime temsilleri elde etmek için güçlü ve pragmatik bir çözüm sunmaktadır. Katkısı, kelime gömülme modellerinin sınırlarını, insan dilinin daha incelikli ve dirençli bir anlayışına doğru itmektir.




