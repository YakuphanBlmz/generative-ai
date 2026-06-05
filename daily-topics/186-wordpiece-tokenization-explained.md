# WordPiece Tokenization Explained

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Principles of WordPiece Tokenization](#2-principles-of-wordpiece-tokenization)
- [3. Training the WordPiece Model](#3-training-the-wordpiece-model)
- [4. Application and Advantages](#4-application-and-advantages)
- [5. Code Example](#5-code-example)
- [6. Limitations and Considerations](#6-limitations-and-considerations)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
In the realm of Natural Language Processing (NLP), **tokenization** is a fundamental process that involves breaking down raw text into smaller units called **tokens**. These tokens serve as the basic input for most NLP models. Traditional tokenization methods often involve splitting text by spaces or punctuation, yielding **word-level tokens**. While straightforward, this approach faces significant challenges, particularly with **Out-Of-Vocabulary (OOV)** words, rare words, and morphologically rich languages. For instance, a word like "unbelievable" might be seen as a single token, yet its constituent parts ("un-", "believe", "-able") carry distinct semantic contributions.

**WordPiece tokenization**, introduced by Google and prominently used in influential models like BERT (Bidirectional Encoder Representations from Transformers), addresses these limitations by employing **subword tokenization**. Instead of strictly word-level units, WordPiece segments words into a finite set of subword units, which can be full words, prefixes, suffixes, or root forms. This technique allows models to handle a vast vocabulary more efficiently, mitigate the OOV problem, and better capture the nuances of morphology and syntax by decomposing complex words into their meaningful components. This document provides a detailed exposition of WordPiece tokenization, its underlying principles, training methodology, practical applications, and inherent limitations.

<a name="2-principles-of-wordpiece-tokenization"></a>
## 2. Principles of WordPiece Tokenization
WordPiece tokenization operates on the principle of breaking down words into smaller, frequently occurring subword units. Unlike pure character-level tokenization, which loses semantic context, or pure word-level tokenization, which suffers from OOV issues, WordPiece strikes a balance.

The core idea is to create a vocabulary of subword units that can represent virtually any word in a given language. When a word needs to be tokenized, WordPiece attempts to segment it into the longest possible subword units from its vocabulary, greedily, from left to right. If a word cannot be perfectly segmented into known subwords, it might be represented using a combination of known subwords and potentially an unknown token if no subword matches can be found for a part of it.

Key characteristics and comparisons:
*   **Subword Units:** WordPiece's vocabulary consists of pieces that can be entire words (e.g., "apple"), word prefixes (e.g., "un"), suffixes (e.g., "##ing"), or word roots (e.g., "walk"). The "##" prefix is a common convention to denote that a subword is not the beginning of a word, but rather a continuation. For example, "tokenization" might be broken into "token", "##iza", and "##tion".
*   **Greedy Longest Match:** During inference (tokenization of new text), the algorithm prioritizes finding the longest possible subword from its learned vocabulary that matches a segment of the input word, moving from the beginning of the word to the end.
*   **Comparison with Byte Pair Encoding (BPE):** WordPiece is often compared to BPE, another popular subword tokenization algorithm. While both aim to create subword units, their merging criteria differ. BPE iteratively merges the most frequent pairs of characters or subwords. WordPiece, on the other hand, selects the merge that maximizes the likelihood of the training data when the original words are segmented into their subword units. This distinction often leads to slightly different vocabulary compositions. WordPiece tends to prefer longer, more semantically coherent units.
*   **Handling OOV Words:** A significant advantage of WordPiece is its ability to handle OOV words. Any word, no matter how rare or new, can be broken down into known subword units. For instance, if "unbelievable" is an OOV word, it can still be represented by known tokens like "un", "##believ", "##able", ensuring that the model has some meaningful representation for it. This significantly reduces the need for large, dynamic vocabularies and the problem of unknown tokens.

<a name="3-training-the-wordpiece-model"></a>
## 3. Training the WordPiece Model
The training of a WordPiece model is a crucial step that determines its effectiveness. The goal is to construct a fixed-size vocabulary of subword units from a large corpus of text. The process is distinct from how BPE operates, focusing on maximizing the probability of the training corpus given the chosen segmentation.

Here's a simplified overview of the training procedure:
1.  **Initial Vocabulary:** The process typically starts with an initial vocabulary consisting of all unique characters present in the training corpus. Each word in the corpus is initially represented as a sequence of these characters.
2.  **Iterative Merging:** The algorithm then iteratively expands the vocabulary by adding new subword units. In each iteration, it identifies the pair of subword units whose merge would result in the greatest increase in the likelihood of the training data. This "likelihood" is typically calculated based on the frequency of the merged unit, essentially finding the pair that, when combined, would most frequently appear as a meaningful unit within words.
    *   For example, if "t", "o", "k", "e", "n" are characters, and "token" appears frequently, the algorithm might find that merging "t" and "o" to form "to", then "to" and "k" to form "tok", and so on, improves the likelihood.
    *   The crucial difference from BPE is that WordPiece computes a score for each potential merge pair based on a specific likelihood function. This function aims to find subwords that are most "predictive" or "informative" in a statistical sense. Specifically, if two tokens `x` and `y` are candidates for merging, WordPiece evaluates `(frequency(xy) / (frequency(x) * frequency(y)))` and selects the merge that maximizes this ratio. This aims to find merges that are highly probable *given* their constituent parts, implying a strong statistical dependency.
3.  **Vocabulary Size:** This merging process continues until the desired vocabulary size (a hyperparameter, often ranging from 30,000 to 50,000 for large models) is reached. The final vocabulary contains a mix of single characters, common subwords, and frequent full words.
4.  **Handling Word Boundaries:** When a WordPiece model is trained, it learns how to segment words into subword units. It implicitly learns to differentiate between subwords that start a word and those that are continuations. This is often achieved by adding a special character (e.g., `_` or `##`) to the beginning of subword units that are not the start of a word. For instance, "tokenization" might be segmented as `['token', '##iza', '##tion']`. The leading `##` explicitly indicates that `iza` and `tion` are not standalone words but parts of a larger word.

The training ensures that the resulting subword vocabulary is optimized for the specific language and domain of the training corpus, providing efficient and semantically rich representations for downstream NLP tasks.

<a name="4-application-and-advantages"></a>
## 4. Application and Advantages
WordPiece tokenization has gained widespread adoption due to its significant advantages in modern NLP architectures, most notably in transformer-based models like BERT, ALBERT, and Electra. Its effectiveness stems from its ability to bridge the gap between character-level and word-level representations.

### Key Advantages:
1.  **Reduced Vocabulary Size:** By representing words as sequences of subword units, WordPiece drastically reduces the size of the required vocabulary compared to pure word-level tokenization. Instead of needing to store every possible word form, the model only needs a finite set of subwords. This leads to more efficient memory usage and faster processing.
2.  **Effective Handling of OOV Words:** As discussed, OOV words are naturally decomposed into known subword units. This means models are less likely to encounter truly "unknown" input, improving robustness and generalization capabilities, especially in domains with evolving terminology or morphologically rich languages.
3.  **Semantic Richness and Morphological Understanding:** Subword units often correspond to meaningful morphological components (prefixes, suffixes, roots). By breaking down words like "unbelievable" into "un", "##believ", "##able", the model can infer partial meaning from its components, even if the full word was not explicitly seen during training. This enhances the model's understanding of word structure and semantics.
4.  **Balance between Granularity and Context:** WordPiece offers a good balance. It's more granular than word-level tokenization, allowing for finer distinctions, but less granular than character-level tokenization, which often dilutes semantic information due to excessive fragmentation. This optimal granularity allows models to leverage both fine-grained and broader contextual information.
5.  **Improved Generalization:** Models trained with WordPiece can generalize better to unseen words or variations of words because they learn representations for common subword components. This is particularly beneficial in tasks like machine translation, text generation, and named entity recognition where new word forms are common.

These advantages collectively contribute to the superior performance of models that employ WordPiece tokenization, making it a cornerstone technique in contemporary NLP.

<a name="5-code-example"></a>
## 5. Code Example
This example demonstrates how to use a pre-trained WordPiece tokenizer, specifically `BertTokenizer`, from the Hugging Face `transformers` library to tokenize a sample sentence.

```python
from transformers import BertTokenizer

# Load a pre-trained BERT tokenizer.
# BERT uses WordPiece tokenization.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample text to tokenize.
text = "WordPiece tokenization handles out-of-vocabulary words effectively."

# Tokenize the text using the WordPiece tokenizer.
# The `do_lower_case=True` is inherent in 'bert-base-uncased'
# and `add_special_tokens=True` adds [CLS] and [SEP].
tokens = tokenizer.tokenize(text)

# Print the original text and the resulting tokens.
print("Original Text:", text)
print("WordPiece Tokens:", tokens)

# It's also common to convert tokens to their IDs for model input.
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs (without special tokens):", input_ids)

# To get the full model input with special tokens and attention mask,
# you would use `tokenizer.encode_plus` or `tokenizer(text, return_tensors='pt')`.
# For illustration, let's show how a full word is broken down.
word_to_segment = "tokenization"
segmented_word = tokenizer.tokenize(word_to_segment)
print(f"Segmentation of '{word_to_segment}':", segmented_word)

word_to_segment_oov = "unbelievable" # Common example for OOV handling
segmented_word_oov = tokenizer.tokenize(word_to_segment_oov)
print(f"Segmentation of '{word_to_segment_oov}' (OOV-like):", segmented_word_oov)

(End of code example section)
```

<a name="6-limitations-and-considerations"></a>
## 6. Limitations and Considerations
Despite its widespread adoption and significant advantages, WordPiece tokenization is not without its limitations and requires careful consideration in certain scenarios.

1.  **Fixed Vocabulary Size:** The vocabulary is fixed after training. While this helps with efficiency, it means that truly novel subword units (e.g., from new domains or languages not present in the training corpus) cannot be created on-the-fly. The model is constrained by the subwords it has learned.
2.  **Non-unique Segmentation:** WordPiece, especially with its greedy longest-match approach, can sometimes lead to different segmentations for the same word if different vocabularies are used or if a word could be formed by multiple valid subword combinations. While the training optimizes for a specific likelihood, during inference, the greedy approach doesn't guarantee the "most optimal" segmentation in all linguistic contexts, particularly for highly ambiguous words or agglutinative languages.
3.  **Lack of Explicit Morphological Information:** While subword units often align with morphological components, WordPiece does not explicitly encode or guarantee perfect morphological analysis. The segmentation is statistically driven, not linguistically rule-based. Thus, a model might learn to split "running" into "run" and "##ning" not because it understands the verb root and present participle suffix, but because that split maximizes the likelihood in the training data.
4.  **Token Length and Subword Cohesion:** The choice of vocabulary size is a hyperparameter. A very small vocabulary might lead to overly fragmented words (closer to character-level), potentially losing context. A very large vocabulary might dilute the benefits of subword tokenization, becoming closer to word-level. Finding the right balance is crucial.
5.  **Handling of Rare Characters:** If the initial character vocabulary or the training corpus does not include certain rare characters or symbols, they might be mapped to an `<unk>` (unknown) token or handled poorly, potentially losing information. This is less common with robust, large-scale pre-trained tokenizers but can be a factor with custom training.

Understanding these limitations is essential for effectively deploying and interpreting models that rely on WordPiece tokenization.

<a name="7-conclusion"></a>
## 7. Conclusion
WordPiece tokenization stands as a pivotal advancement in Natural Language Processing, offering an elegant solution to the inherent trade-offs between character-level and word-level tokenization. By strategically segmenting words into frequently occurring subword units, it effectively tackles the challenges of Out-Of-Vocabulary words, reduces vocabulary size, and provides richer semantic representations, particularly for morphologically complex languages. Its statistical approach to vocabulary construction, prioritizing subword combinations that maximize data likelihood, has made it an indispensable component in high-performance transformer models such as BERT.

While WordPiece offers substantial benefits in terms of efficiency, generalization, and robustness against unseen vocabulary, it is important to acknowledge its limitations, including its fixed vocabulary size and the potential for non-unique or statistically driven, rather than purely linguistically driven, segmentations. Nonetheless, its contribution to enabling larger, more effective language models that can process vast amounts of diverse text data cannot be overstated. WordPiece remains a cornerstone technique, empowering advanced NLP applications across a multitude of domains.

---
<br>

<a name="türkçe-içerik"></a>
## WordPiece Tokenizasyonu Açıklaması

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. WordPiece Tokenizasyonunun İlkeleri](#2-wordpiece-tokenizasyonunun-ilkeleri)
- [3. WordPiece Modelinin Eğitimi](#3-wordpiece-modelinin-eğitimi)
- [4. Uygulama ve Avantajlar](#4-uygulama-ve-avantajlar)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sınırlamalar ve Dikkat Edilmesi Gerekenler](#6-sınırlamalar-ve-dikkat-edilmesi-gerekenler)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Doğal Dil İşleme (NLP) alanında, **tokenizasyon** ham metni **token** adı verilen daha küçük birimlere ayırma sürecini içeren temel bir işlemdir. Bu tokenlar çoğu NLP modeli için temel girdi görevi görür. Geleneksel tokenizasyon yöntemleri genellikle metni boşluklara veya noktalama işaretlerine göre bölerek **kelime düzeyinde tokenlar** üretir. Bu yaklaşım basit olsa da, özellikle **Kelimeler Sözlük Dışı (Out-Of-Vocabulary - OOV)**, nadir kelimeler ve morfolojik olarak zengin dillerle ilgili önemli zorluklarla karşılaşır. Örneğin, "inanılmaz" gibi bir kelime tek bir token olarak görülebilirken, kurucu parçaları ("in-", "inan", "-ılmaz") farklı anlamsal katkılar taşır.

Google tarafından tanıtılan ve BERT (Transformatorlardan Çift Yönlü Kodlayıcı Temsilleri) gibi etkili modellerde yaygın olarak kullanılan **WordPiece tokenizasyonu**, bu sınırlamaları **alt kelime tokenizasyonu** kullanarak giderir. WordPiece, kelime düzeyindeki birimler yerine kelimeleri, tam kelimeler, ön ekler, son ekler veya kök biçimleri olabilen sonlu bir alt kelime birimleri kümesine böler. Bu teknik, modellerin geniş bir kelime dağarcığını daha verimli bir şekilde ele almasına, OOV sorununu azaltmasına ve karmaşık kelimeleri anlamlı bileşenlerine ayırarak morfoloji ve sözdizimi nüanslarını daha iyi yakalamasına olanak tanır. Bu belge, WordPiece tokenizasyonunun, temel ilkelerinin, eğitim metodolojisinin, pratik uygulamalarının ve doğal sınırlamalarının ayrıntılı bir açıklamasını sunmaktadır.

<a name="2-wordpiece-tokenizasyonunun-ilkeleri"></a>
## 2. WordPiece Tokenizasyonunun İlkeleri
WordPiece tokenizasyonu, kelimeleri daha küçük, sıkça geçen alt kelime birimlerine ayırma ilkesine göre çalışır. Anlamsal bağlamı kaybolan saf karakter düzeyinde tokenizasyonun veya OOV sorunları yaşayan saf kelime düzeyinde tokenizasyonun aksine, WordPiece bir denge kurar.

Temel fikir, belirli bir dildeki neredeyse her kelimeyi temsil edebilecek bir alt kelime birimleri sözlüğü oluşturmaktır. Bir kelimenin tokenize edilmesi gerektiğinde, WordPiece, kelimeyi soldan sağa doğru, açgözlü bir şekilde, sözlüğündeki mümkün olan en uzun alt kelime birimlerine ayırmaya çalışır. Eğer bir kelime bilinen alt kelimelere mükemmel bir şekilde ayrılamazsa, bilinen alt kelimelerin bir kombinasyonu ve eğer bir kısmı için eşleşme bulunamazsa potansiyel olarak bilinmeyen bir token kullanılarak temsil edilebilir.

Temel özellikler ve karşılaştırmalar:
*   **Alt Kelime Birimleri:** WordPiece'in kelime dağarcığı, tüm kelimeler (örn. "elma"), kelime ön ekleri (örn. "un"), son ekler (örn. "##ing") veya kelime kökleri (örn. "yürü") olabilen parçalardan oluşur. "##" ön eki, bir alt kelimenin bir kelimenin başlangıcı olmadığını, daha ziyade bir devamı olduğunu belirtmek için yaygın bir kuraldır. Örneğin, "tokenization" kelimesi "token", "##iza" ve "##tion" olarak bölünebilir.
*   **Açgözlü En Uzun Eşleşme:** Çıkarım sırasında (yeni metnin tokenizasyonu), algoritma, öğrenilmiş sözlüğünden, giriş kelimesinin bir bölümüyle eşleşen mümkün olan en uzun alt kelimeyi bulmaya öncelik verir ve kelimenin başından sonuna doğru ilerler.
*   **Bayt Çifti Kodlama (BPE) ile Karşılaştırma:** WordPiece, genellikle başka bir popüler alt kelime tokenizasyon algoritması olan BPE ile karşılaştırılır. Her ikisi de alt kelime birimleri oluşturmayı amaçlarken, birleştirme kriterleri farklıdır. BPE, en sık çift karakterleri veya alt kelimeleri yinelemeli olarak birleştirir. WordPiece ise, orijinal kelimelerin alt kelime birimlerine bölündüğünde eğitim verilerinin olasılığını en üst düzeye çıkaran birleştirmeyi seçer. Bu ayrım genellikle biraz farklı kelime dağarcığı bileşimlerine yol açar. WordPiece, daha uzun, anlamsal olarak daha tutarlı birimleri tercih etme eğilimindedir.
*   **OOV Kelimelerin İşlenmesi:** WordPiece'in önemli bir avantajı, OOV kelimeleri işleyebilme yeteneğidir. Ne kadar nadir veya yeni olursa olsun herhangi bir kelime, bilinen alt kelime birimlerine ayrılabilir. Örneğin, "inanılmaz" OOV bir kelimeyse, yine de "in", "##an", "##ılmaz" gibi bilinen tokenlarla temsil edilebilir, bu da modelin o kelime için anlamlı bir gösterime sahip olmasını sağlar. Bu, büyük, dinamik kelime dağarcıklarına olan ihtiyacı ve bilinmeyen token sorununu önemli ölçüde azaltır.

<a name="3-wordpiece-modelinin-eğitimi"></a>
## 3. WordPiece Modelinin Eğitimi
Bir WordPiece modelinin eğitimi, etkinliğini belirleyen kritik bir adımdır. Amaç, geniş bir metin kümesinden sabit boyutlu bir alt kelime birimleri sözlüğü oluşturmaktır. Süreç, BPE'nin nasıl çalıştığından farklıdır ve seçilen segmentasyon verildiğinde eğitim kümesinin olasılığını maksimize etmeye odaklanır.

Eğitim prosedürünün basitleştirilmiş bir genel bakışı:
1.  **Başlangıç Sözlüğü:** Süreç genellikle, eğitim kümesinde bulunan tüm benzersiz karakterlerden oluşan bir başlangıç sözlüğüyle başlar. Kümedeki her kelime başlangıçta bu karakter dizisi olarak temsil edilir.
2.  **Yinelemeli Birleştirme:** Algoritma daha sonra yeni alt kelime birimleri ekleyerek sözlüğü yinelemeli olarak genişletir. Her yinelemede, eğitim verilerinin olasılığında en büyük artışı sağlayacak alt kelime birimleri çiftini belirler. Bu "olasılık", birleştirilmiş birimin sıklığına göre hesaplanır ve temel olarak, birleştirildiğinde kelimeler içinde en sık anlamlı birim olarak görünecek çifti bulur.
    *   Örneğin, "t", "o", "k", "e", "n" karakterleri ise ve "token" sıkça görünüyorsa, algoritma "t" ve "o"yu "to" oluşturmak için, ardından "to" ve "k"yı "tok" oluşturmak için birleştirmenin olasılığı artırdığını bulabilir.
    *   BPE'den temel fark, WordPiece'in belirli bir olasılık fonksiyonuna dayalı olarak her potansiyel birleştirme çifti için bir puan hesaplamasıdır. Bu fonksiyon, istatistiksel anlamda en "tahmin edici" veya "bilgilendirici" alt kelimeleri bulmayı amaçlar. Özellikle, iki token `x` ve `y` birleştirme adayları ise, WordPiece `(sıklık(xy) / (sıklık(x) * sıklık(y)))` değerini değerlendirir ve bu oranı maksimize eden birleştirmeyi seçer. Bu, bileşen parçaları *verildiğinde* yüksek olasılıklı birleştirmeler bulmayı hedefler ve güçlü bir istatistiksel bağımlılık anlamına gelir.
3.  **Sözlük Boyutu:** Bu birleştirme süreci, istenen sözlük boyutuna (büyük modeller için genellikle 30.000 ila 50.000 arasında değişen bir hiperparametre) ulaşılana kadar devam eder. Nihai sözlük, tek karakterlerin, yaygın alt kelimelerin ve sık kullanılan tam kelimelerin bir karışımını içerir.
4.  **Kelime Sınırlarının İşlenmesi:** Bir WordPiece modeli eğitildiğinde, kelimeleri alt kelime birimlerine nasıl ayıracağını öğrenir. Bir kelimeyi başlatan alt kelimeler ile devam edenleri zımnen ayırt etmeyi öğrenir. Bu genellikle, bir kelimenin başlangıcı olmayan alt kelime birimlerinin başına özel bir karakter (örn. `_` veya `##`) eklenerek başarılır. Örneğin, "tokenization" `['token', '##iza', '##tion']` olarak bölünebilir. Baştaki `##` açıkça `iza` ve `tion`'ın bağımsız kelimeler değil, daha büyük bir kelimenin parçaları olduğunu gösterir.

Eğitim, ortaya çıkan alt kelime sözlüğünün eğitim kümesinin belirli dili ve alanı için optimize edilmesini sağlayarak, sonraki NLP görevleri için verimli ve anlamsal olarak zengin temsiller sunar.

<a name="4-uygulama-ve-avantajlar"></a>
## 4. Uygulama ve Avantajlar
WordPiece tokenizasyonu, modern NLP mimarilerinde, özellikle BERT, ALBERT ve Electra gibi transformatör tabanlı modellerde önemli avantajları nedeniyle yaygın olarak benimsenmiştir. Etkinliği, karakter düzeyindeki ve kelime düzeyindeki temsiller arasındaki boşluğu doldurabilme yeteneğinden kaynaklanmaktadır.

### Temel Avantajlar:
1.  **Azaltılmış Sözlük Boyutu:** Kelimeleri alt kelime birimleri dizileri olarak temsil ederek, WordPiece, saf kelime düzeyinde tokenizasyona kıyasla gereken sözlüğün boyutunu önemli ölçüde azaltır. Modelin her olası kelime formunu depolaması yerine, yalnızca sonlu bir alt kelime kümesi depolaması gerekir. Bu, daha verimli bellek kullanımına ve daha hızlı işlemeye yol açar.
2.  **OOV Kelimelerin Etkili İşlenmesi:** Tartışıldığı gibi, OOV kelimeler doğal olarak bilinen alt kelime birimlerine ayrılır. Bu, modellerin gerçekten "bilinmeyen" girdilerle karşılaşma olasılığının daha düşük olduğu anlamına gelir, özellikle gelişen terminolojiye sahip alanlarda veya morfolojik olarak zengin dillerde sağlamlığı ve genelleştirme yeteneklerini artırır.
3.  **Anlamsal Zenginlik ve Morfolojik Anlayış:** Alt kelime birimleri genellikle anlamlı morfolojik bileşenlere (ön ekler, son ekler, kökler) karşılık gelir. "İnanılmaz" gibi kelimeleri "in", "##an", "##ılmaz" olarak parçalayarak, model tam kelime eğitim sırasında açıkça görülmese bile bileşenlerinden kısmi anlam çıkarabilir. Bu, modelin kelime yapısı ve anlambilimi hakkındaki anlayışını geliştirir.
4.  **Taneciklik ve Bağlam Arasındaki Denge:** WordPiece iyi bir denge sunar. Kelime düzeyinde tokenizasyondan daha taneciklidir, daha ince ayrımlara izin verir, ancak aşırı parçalanma nedeniyle anlamsal bilgiyi genellikle seyrelten karakter düzeyinde tokenizasyondan daha az taneciklidir. Bu optimal taneciklik, modellerin hem ince taneli hem de daha geniş bağlamsal bilgileri kullanmasına olanak tanır.
5.  **Geliştirilmiş Genelleştirme:** WordPiece ile eğitilmiş modeller, ortak alt kelime bileşenleri için temsiller öğrendikleri için görünmeyen kelimelere veya kelime varyasyonlarına daha iyi genelleşebilir. Bu, yeni kelime biçimlerinin yaygın olduğu makine çevirisi, metin üretimi ve adlandırılmış varlık tanıma gibi görevlerde özellikle faydalıdır.

Bu avantajlar, WordPiece tokenizasyonu kullanan modellerin üstün performansına topluca katkıda bulunarak onu çağdaş NLP'de temel bir teknik haline getirmiştir.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği
Bu örnek, Hugging Face `transformers` kütüphanesinden önceden eğitilmiş bir WordPiece tokenizer'ı, özellikle `BertTokenizer`'ı kullanarak örnek bir cümleyi tokenize etmeyi göstermektedir.

```python
from transformers import BertTokenizer

# Önceden eğitilmiş bir BERT tokenizer yükle.
# BERT, WordPiece tokenizasyonu kullanır.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize edilecek örnek metin.
text = "WordPiece tokenization handles out-of-vocabulary words effectively."

# Metni WordPiece tokenizer kullanarak tokenize et.
# `do_lower_case=True`, 'bert-base-uncased' içinde doğal olarak bulunur
# ve `add_special_tokens=True` özel tokenlar olan [CLS] ve [SEP] ekler.
tokens = tokenizer.tokenize(text)

# Orijinal metni ve ortaya çıkan tokenları yazdır.
print("Orijinal Metin:", text)
print("WordPiece Tokenları:", tokens)

# Tokenları model girişi için ID'lerine dönüştürmek de yaygındır.
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token ID'leri (özel tokenlar hariç):", input_ids)

# Özel tokenlar ve dikkat maskesi ile tam model girdisini almak için,
# `tokenizer.encode_plus` veya `tokenizer(text, return_tensors='pt')` kullanmanız gerekir.
# Örnek olarak, tam bir kelimenin nasıl parçalandığını gösterelim.
word_to_segment = "tokenization"
segmented_word = tokenizer.tokenize(word_to_segment)
print(f"'{word_to_segment}' kelimesinin parçalanması:", segmented_word)

word_to_segment_oov = "unbelievable" # OOV işleme için yaygın örnek
segmented_word_oov = tokenizer.tokenize(word_to_segment_oov)
print(f"'{word_to_segment_oov}' kelimesinin parçalanması (OOV benzeri):", segmented_word_oov)

(Kod örneği bölümünün sonu)
```

<a name="6-sınırlamalar-ve-dikkat-edilmesi-gerekenler"></a>
## 6. Sınırlamalar ve Dikkat Edilmesi Gerekenler
WordPiece tokenizasyonu, yaygın olarak benimsenmesine ve önemli avantajlarına rağmen, sınırlamalara sahip değildir ve belirli senaryolarda dikkatli değerlendirme gerektirir.

1.  **Sabit Sözlük Boyutu:** Sözlük, eğitimden sonra sabittir. Bu, verimliliğe yardımcı olsa da, (eğitim kümesinde bulunmayan yeni alanlardan veya dillerden) gerçekten yeni alt kelime birimlerinin anında oluşturulamayacağı anlamına gelir. Model, öğrendiği alt kelimelerle sınırlıdır.
2.  **Benzersiz Olmayan Segmentasyon:** WordPiece, özellikle açgözlü en uzun eşleşme yaklaşımıyla, farklı sözlükler kullanıldığında veya bir kelime birden fazla geçerli alt kelime kombinasyonuyla oluşturulabiliyorsa, aynı kelime için bazen farklı segmentasyonlara yol açabilir. Eğitim belirli bir olasılığı optimize ederken, çıkarım sırasında açgözlü yaklaşım, özellikle oldukça belirsiz kelimeler veya eklemeli diller için tüm dilbilimsel bağlamlarda "en optimal" segmentasyonu garanti etmez.
3.  **Açık Morfolojik Bilgi Eksikliği:** Alt kelime birimleri genellikle morfolojik bileşenlerle hizalansa da, WordPiece açıkça mükemmel morfolojik analizi kodlamaz veya garanti etmez. Segmentasyon istatistiksel olarak yönlendirilir, dilbilimsel kural tabanlı değildir. Bu nedenle, bir model "running" kelimesini "run" ve "##ning" olarak ayırmayı, fiil kökünü ve şimdiki zaman ortaç ekini anladığı için değil, bu ayrımın eğitim verilerindeki olasılığı maksimize ettiği için öğrenmiş olabilir.
4.  **Token Uzunluğu ve Alt Kelime Tutarlılığı:** Sözlük boyutu seçimi bir hiperparametredir. Çok küçük bir sözlük, aşırı parçalanmış kelimelere (karakter düzeyine daha yakın) yol açabilir ve potansiyel olarak bağlamı kaybetmesine neden olabilir. Çok büyük bir sözlük, alt kelime tokenizasyonunun faydalarını seyreltebilir ve kelime düzeyine daha yakın hale gelebilir. Doğru dengeyi bulmak çok önemlidir.
5.  **Nadir Karakterlerin İşlenmesi:** Başlangıç karakter sözlüğü veya eğitim kümesi belirli nadir karakterleri veya sembolleri içermiyorsa, bunlar bir `<unk>` (bilinmeyen) token'a eşlenebilir veya kötü bir şekilde işlenebilir ve potansiyel olarak bilgi kaybedebilir. Bu durum, sağlam, büyük ölçekli önceden eğitilmiş tokenizasyonlayıcılar için daha az yaygındır ancak özel eğitimle bir faktör olabilir.

Bu sınırlamaları anlamak, WordPiece tokenizasyonuna dayanan modelleri etkili bir şekilde dağıtmak ve yorumlamak için gereklidir.

<a name="7-sonuç"></a>
## 7. Sonuç
WordPiece tokenizasyonu, karakter düzeyindeki ve kelime düzeyindeki tokenizasyon arasındaki doğal dengelere zarif bir çözüm sunarak Doğal Dil İşleme'de önemli bir ilerleme olarak durmaktadır. Kelimeleri stratejik olarak sıkça geçen alt kelime birimlerine ayırarak, Kelimeler Sözlük Dışı sorunlarını etkili bir şekilde çözer, sözlük boyutunu azaltır ve özellikle morfolojik olarak karmaşık diller için daha zengin anlamsal temsiller sağlar. Veri olasılığını maksimize eden alt kelime kombinasyonlarını önceliklendiren istatistiksel sözlük oluşturma yaklaşımı, BERT gibi yüksek performanslı transformatör modellerinde vazgeçilmez bir bileşen haline getirmiştir.

WordPiece, verimlilik, genelleştirme ve görünmeyen kelime dağarcığına karşı sağlamlık açısından önemli faydalar sunarken, sabit sözlük boyutu ve benzersiz olmayan veya tamamen dilbilimsel olarak yönlendirilen değil istatistiksel olarak yönlendirilen segmentasyon potansiyeli dahil olmak üzere sınırlamalarını kabul etmek önemlidir. Bununla birlikte, çok çeşitli metin verilerini işleyebilen daha büyük, daha etkili dil modellerini mümkün kılmadaki katkısı göz ardı edilemez. WordPiece, sayısız alanda gelişmiş NLP uygulamalarını güçlendiren temel bir teknik olmaya devam etmektedir.
