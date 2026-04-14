# SentencePiece: A Simple and Language Independent Subword Tokenizer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenge of Tokenization and the Rise of Subwords](#2-the-challenge-of-tokenization-and-the-rise-of-subwords)
- [3. SentencePiece: Core Concepts and Methodology](#3-sentencepiece-core-concepts-and-methodology)
  - [3.1. Language Independence](#31-language-independence)
  - [3.2. Training Algorithms: BPE and Unigram Language Model](#32-training-algorithms-bpe-and-unigram-language-model)
  - [3.3. Handling Whitespace and Unknown Tokens](#33-handling-whitespace-and-unknown-tokens)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

---

<a name="1-introduction"></a>
## 1. Introduction
In the realm of Natural Language Processing (NLP), **tokenization** is a foundational step, transforming raw text into a sequence of discrete units called tokens. These tokens serve as the input for nearly all subsequent NLP tasks, from machine translation to text generation and sentiment analysis. Historically, tokenization has been performed at either the character level or the word level. While character-level tokenization handles out-of-vocabulary (OOV) words robustly, it often generates very long sequences, losing semantic context. Word-level tokenization, conversely, provides more semantic meaning per token but struggles with OOV words, morphological variations, and the challenge of managing extremely large vocabularies in highly inflected languages.

The emergence of deep learning models, particularly **Transformers**, has highlighted the limitations of traditional tokenization methods, especially when dealing with vast, multilingual datasets. This led to the widespread adoption of **subword tokenization**, a hybrid approach that aims to strike a balance between character and word-level methods. Subword tokenization breaks down words into smaller, frequently occurring units, or subwords, allowing models to learn representations for morphologically rich words, reduce OOV rates, and manage vocabulary size efficiently.

**SentencePiece**, developed by Google, stands out as a highly effective and versatile subword tokenizer. Its core innovation lies in its **language independence** and the way it treats input text as a raw stream of characters, including whitespace. Unlike many tokenizers that rely on language-specific pre-tokenization rules (e.g., splitting by spaces, handling punctuation), SentencePiece directly learns subword units from raw text. This document will delve into the mechanics, advantages, and applications of SentencePiece, demonstrating its critical role in modern generative AI and NLP pipelines.

<a name="2-the-challenge-of-tokenization-and-the-rise-of-subwords"></a>
## 2. The Challenge of Tokenization and the Rise of Subwords
Traditional tokenization approaches face several significant hurdles in contemporary NLP:

*   **Out-of-Vocabulary (OOV) Words:** In word-level tokenization, any word not present in the pre-defined vocabulary becomes an OOV token, often represented as `[UNK]` (unknown). This leads to a loss of information and hinders model performance, particularly in tasks involving rare words, proper nouns, or domains not well-represented in the training data.
*   **Vocabulary Size:** To mitigate OOV issues, one might expand the vocabulary, but this quickly becomes computationally expensive. Large vocabularies demand more memory and increase the dimensionality of word embeddings, making models slower to train and infer.
*   **Morphological Richness:** Languages with complex morphology (e.g., Turkish, German, Finnish) can generate a vast number of word forms from a single root word (e.g., "koş-" -> "koştum", "koşuyor", "koşarak"). Word-level tokenization treats each as a distinct token, leading to OOV issues and preventing models from learning shared representations for related words.
*   **Consistent Tokenization across Languages:** Different languages have different rules for word boundaries, punctuation, and compounding. Applying universal pre-tokenization rules or relying on language-specific tools complicates multilingual model development.

**Subword tokenization** addresses these challenges by segmenting words into smaller, semantically meaningful units. For instance, "unbelievable" might be broken into "un", "believe", and "able". This approach offers several benefits:
*   **Reduced OOV Rates:** Even if "unbelievable" isn't in the vocabulary, its subwords might be, allowing the model to infer its meaning.
*   **Managed Vocabulary Size:** By learning frequent subword units, the total vocabulary size can be kept manageable while still covering a wide range of words.
*   **Handling Morphology:** Shared subwords like "un-" or "-ing" help models learn common prefixes and suffixes, enabling better generalization to unseen word forms.
*   **Cross-Lingual Consistency:** A universal subword tokenization strategy can be applied across multiple languages, simplifying multilingual model architectures.

The rise of powerful neural architectures, particularly **Transformer models**, which rely on fixed-size input sequences and embeddings, has made subword tokenization an indispensable component. Tokenizers like WordPiece (used by BERT), BPE (Byte Pair Encoding, used by GPT-2/3), and Unigram (used by ALBERT) have become standard, with SentencePiece offering a unified and flexible framework for implementing these algorithms.

<a name="3-sentencepiece-core-concepts-and-methodology"></a>
## 3. SentencePiece: Core Concepts and Methodology
SentencePiece distinguishes itself through its fundamental design principles, primarily its language independence and its approach to treating text as raw character sequences.

<a name="31-language-independence"></a>
### 3.1. Language Independence
A key differentiator of SentencePiece is its design to be **language-agnostic**. Traditional tokenizers often require pre-tokenization steps that involve language-specific rules for splitting words, normalizing text, or handling punctuation. These steps can be complex to implement for diverse languages and introduce inconsistencies when training multilingual models.

SentencePiece circumvents this by operating directly on the **raw byte sequence** of the input text. It considers the entire input as a stream of Unicode characters, including whitespace characters, which are typically treated as delimiters in other tokenization schemes. This "raw text" approach means SentencePiece doesn't need to know anything about word boundaries or specific linguistic properties of a language. Instead, it learns these patterns implicitly during the training phase from a given corpus. This universal approach simplifies data preprocessing across different languages, making it particularly suitable for large-scale multilingual NLP applications.

<a name="32-training-algorithms-bpe-and-unigram-language-model"></a>
### 3.2. Training Algorithms: BPE and Unigram Language Model
SentencePiece primarily implements two subword segmentation algorithms: **Byte Pair Encoding (BPE)** and the **Unigram Language Model**.

*   **Byte Pair Encoding (BPE):** Originally a data compression algorithm, BPE was adapted for subword tokenization. It starts by treating each character as an individual subword unit. Then, it iteratively merges the most frequent adjacent pair of units into a new, single unit. This process continues until a predefined vocabulary size is reached or no more merges are possible. For example, if "new" and "est" are frequent adjacent pairs, they might merge to "newest". BPE is deterministic, meaning it produces a single, fixed segmentation for any given input.

*   **Unigram Language Model:** Unlike BPE, the Unigram Language Model algorithm allows for multiple possible segmentations for a given input sequence. It views segmentation as a maximum likelihood estimation problem. It starts with a large initial vocabulary (e.g., all unique substrings of a certain length) and iteratively prunes the vocabulary. In each iteration, it calculates the loss of removing each subword unit. Units that minimally increase the loss (i.e., those that are redundant or less important for accurate segmentation) are removed until the desired vocabulary size is achieved. During inference, it uses the Viterbi algorithm to find the most probable segmentation based on the learned subword probabilities. A key advantage of the Unigram model is its ability to sample multiple segmentations, which can be beneficial for noise robustness and augmenting training data.

SentencePiece offers implementations for both, allowing users to choose the algorithm best suited for their specific task and dataset. Both algorithms are trained unsupervised, learning optimal subword units directly from the provided text corpus.

<a name="33-handling-whitespace-and-unknown-tokens"></a>
### 3.3. Handling Whitespace and Unknown Tokens
One of the most distinctive features of SentencePiece is its treatment of whitespace. Instead of splitting text by spaces as a preliminary step, SentencePiece considers the **space character** as a first-class citizen, just like any other character. When it segments text, it prepends a special underscore `_` character to denote the beginning of a word that was originally preceded by a space. For example, " Hello world." might be tokenized into `_Hello`, `_world`, `.`. This consistent encoding ensures that the original whitespace information is preserved, allowing for perfect reversibility (tokenized segments can be joined back to reconstruct the original string exactly, including spaces). This is crucial for tasks like machine translation where maintaining exact original formatting might be important.

For **unknown tokens** (tokens not present in the learned vocabulary), SentencePiece handles them by breaking them down into their constituent characters or smaller known subword units. If a character itself is not in the vocabulary, it might be represented by a special `UNK` token, but this is less common as SentencePiece typically learns all individual characters. This strategy significantly reduces the OOV problem inherent in word-level tokenizers, ensuring that all input text can be processed without losing information due to unknown words.

<a name="4-code-example"></a>
## 4. Code Example

The following Python code snippet demonstrates how to train a SentencePiece tokenizer from a simple text file and then use it to encode and decode a sample sentence.

```python
import sentencepiece as spm

# 1. Prepare a dummy text file for training
# In a real scenario, this would be a large text corpus.
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("Hello SentencePiece, this is a language independent tokenizer.\n")
    f.write("It handles complex words like 'multilingual' and 'tokenization' effectively.\n")
    f.write("Türkçe de destekler, örneğin 'merhaba dünya' ve 'dilbilgisel'.\n")
    f.write("SentencePiece is truly a simple and powerful tool.\n")

# 2. Train a SentencePiece model
# --input: path to the training corpus
# --model_prefix: prefix for the output model files (e.g., .model and .vocab)
# --vocab_size: desired vocabulary size
# --model_type: BPE or unigram (default is unigram)
# --character_coverage: amount of characters covered by the model (default 0.9995)
spm.SentencePieceTrainer.train(
    '--input=corpus.txt --model_prefix=my_spp_model --vocab_size=100 --model_type=bpe --character_coverage=1.0'
)

# 3. Load the trained model
sp = spm.SentencePieceProcessor()
sp.load("my_spp_model.model")

# 4. Encode a sentence
text_to_encode = "SentencePiece is truly a simple and powerful tool for NLP."
encoded_tokens = sp.encode_as_pieces(text_to_encode)
encoded_ids = sp.encode_as_ids(text_to_encode)

print(f"Original text: '{text_to_encode}'")
print(f"Encoded as pieces: {encoded_tokens}")
print(f"Encoded as IDs: {encoded_ids}")

# 5. Decode the IDs back to text
decoded_text = sp.decode_ids(encoded_ids)
print(f"Decoded text: '{decoded_text}'")

# Example with Turkish text
turkish_text = "Merhaba dünya, bu SentencePiece'in gücüdür."
encoded_turkish_pieces = sp.encode_as_pieces(turkish_text)
print(f"Turkish text: '{turkish_text}'")
print(f"Encoded Turkish as pieces: {encoded_turkish_pieces}")


(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion
SentencePiece has established itself as an indispensable tool in modern NLP and generative AI, particularly with the widespread adoption of Transformer-based models. Its fundamental design principles—treating text as raw character sequences and incorporating whitespace as a learnable token—provide a powerful and elegant solution to many of the traditional challenges of tokenization.

By offering a truly **language-independent** approach, SentencePiece simplifies the development and deployment of multilingual NLP systems, eliminating the need for complex, language-specific preprocessing pipelines. Its implementation of robust subword algorithms like **BPE** and the **Unigram Language Model** ensures efficient vocabulary management, significantly reduces **out-of-vocabulary (OOV)** issues, and enables models to better handle morphologically rich languages. The ability to perfectly reconstruct original text from its tokenized form further enhances its utility and reliability.

As generative AI models continue to grow in size and complexity, often trained on vast and diverse datasets encompassing multiple languages, the role of a consistent, efficient, and language-agnostic tokenizer like SentencePiece becomes ever more critical. It underpins the ability of these advanced models to process and generate human language with remarkable flexibility and accuracy, paving the way for more sophisticated and globally applicable AI applications.

---
<br>

<a name="türkçe-içerik"></a>
## SentencePiece: Basit ve Dil Bağımsız Bir Alt Kelime Tokenizerı

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Tokenizasyonun Zorlukları ve Alt Kelimelerin Yükselişi](#2-tokenizasyonun-zorlukları-ve-alt-kelimelerin-yükselişi)
- [3. SentencePiece: Temel Kavramlar ve Metodoloji](#3-sentencepiece-temel-kavramlar-ve-metodoloji)
  - [3.1. Dil Bağımsızlığı](#31-dil-bağımsızlığı)
  - [3.2. Eğitim Algoritmaları: BPE ve Unigram Dil Modeli](#32-eğitim-algoritmaları-bpe-ve-unigram-dil-modeli)
  - [3.3. Boşluk ve Bilinmeyen Tokenların Yönetimi](#33-boşluk-ve-bilinmeyen-tokenların-yönetimi)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

---

<a name="1-giriş"></a>
## 1. Giriş
Doğal Dil İşleme (NLP) alanında, **tokenizasyon** temel bir adımdır ve ham metni **token** adı verilen ayrı birimlerin bir dizisine dönüştürür. Bu tokenlar, makine çevirisinden metin üretimine ve duygu analizine kadar neredeyse tüm sonraki NLP görevleri için girdi görevi görür. Tarihsel olarak, tokenizasyon ya karakter seviyesinde ya da kelime seviyesinde gerçekleştirilmiştir. Karakter seviyesindeki tokenizasyon, kelime dışı (OOV) kelimeleri sağlam bir şekilde ele alırken, genellikle çok uzun diziler oluşturur ve anlamsal bağlamı kaybeder. Kelime seviyesindeki tokenizasyon ise token başına daha fazla anlamsal anlam sağlarken, OOV kelimeleri, morfolojik varyasyonları ve oldukça çekimli dillerde son derece büyük kelime dağarcıklarını yönetme zorluğuyla karşılaşır.

Derin öğrenme modellerinin, özellikle de **Transformer'ların** ortaya çıkışı, geleneksel tokenizasyon yöntemlerinin sınırlılıklarını, özellikle büyük, çok dilli veri kümeleriyle uğraşırken vurgulamıştır. Bu durum, karakter ve kelime seviyesi yöntemleri arasında bir denge kurmayı amaçlayan hibrit bir yaklaşım olan **alt kelime tokenizasyonunun** yaygın olarak benimsenmesine yol açmıştır. Alt kelime tokenizasyonu, kelimeleri daha küçük, sıkça geçen birimlere veya alt kelimelere ayırır; bu da modellerin morfolojik olarak zengin kelimeler için temsiller öğrenmesini, OOV oranlarını düşürmesini ve kelime dağarcığı boyutunu verimli bir şekilde yönetmesini sağlar.

Google tarafından geliştirilen **SentencePiece**, oldukça etkili ve çok yönlü bir alt kelime tokenizerı olarak öne çıkmaktadır. Temel yeniliği, **dil bağımsızlığı** ve giriş metnini boşluklar da dahil olmak üzere ham bir karakter akışı olarak ele almasıdır. Dil-spesifik ön-tokenizasyon kurallarına (örneğin, boşluklara göre ayırma, noktalama işaretlerini işleme) dayanan birçok tokenlayıcının aksine, SentencePiece alt kelime birimlerini doğrudan ham metinden öğrenir. Bu belge, modern üretken yapay zeka ve NLP süreçlerindeki kritik rolünü göstererek SentencePiece'in mekaniklerini, avantajlarını ve uygulamalarını derinlemesine inceleyecektir.

<a name="2-tokenizasyonun-zorlukları-ve-alt-kelimelerin-yükselişi"></a>
## 2. Tokenizasyonun Zorlukları ve Alt Kelimelerin Yükselişi
Geleneksel tokenizasyon yaklaşımları, günümüz NLP'sinde önemli zorluklarla karşılaşmaktadır:

*   **Kelime Dışı (OOV) Kelimeler:** Kelime seviyesindeki tokenizasyonda, önceden tanımlanmış kelime dağarcığında bulunmayan her kelime bir OOV tokenı haline gelir ve genellikle `[UNK]` (bilinmeyen) olarak temsil edilir. Bu durum bilgi kaybına yol açar ve özellikle nadir kelimeler, özel isimler veya eğitim verilerinde iyi temsil edilmeyen alanlarla ilgili görevlerde model performansını düşürür.
*   **Kelime Dağarcığı Boyutu:** OOV sorunlarını azaltmak için kelime dağarcığı genişletilebilir, ancak bu hızla hesaplama açısından pahalı hale gelir. Büyük kelime dağarcıkları daha fazla bellek gerektirir ve kelime gömme boyutunu artırarak modellerin eğitilmesini ve çıkarım yapmasını yavaşlatır.
*   **Morfolojik Zenginlik:** Karmaşık morfolojiye sahip diller (örneğin Türkçe, Almanca, Fince) tek bir kök kelimeden çok sayıda kelime formu (örneğin "koş-" -> "koştum", "koşuyor", "koşarak") üretebilir. Kelime seviyesindeki tokenizasyon, her birini ayrı bir token olarak ele alır, bu da OOV sorunlarına yol açar ve modellerin ilgili kelimeler için ortak temsiller öğrenmesini engeller.
*   **Diller Arası Tutarlı Tokenizasyon:** Farklı dillerin kelime sınırları, noktalama işaretleri ve birleşik kelimeler için farklı kuralları vardır. Evrensel ön-tokenizasyon kuralları uygulamak veya dile özgü araçlara güvenmek, çok dilli model geliştirmeyi karmaşıklaştırır.

**Alt kelime tokenizasyonu** bu zorlukları kelimeleri daha küçük, anlamsal olarak anlamlı birimlere ayırarak ele alır. Örneğin, "inanılmaz" kelimesi "in", "an", "ılmaz" gibi alt kelimelere ayrılabilir. Bu yaklaşım çeşitli faydalar sunar:
*   **Azaltılmış OOV Oranları:** "İnanılmaz" kelimesi kelime dağarcığında olmasa bile, alt kelimeleri olabilir ve modelin anlamını çıkarmasına olanak tanır.
*   **Yönetilebilir Kelime Dağarcığı Boyutu:** Sık kullanılan alt kelime birimlerini öğrenerek, toplam kelime dağarcığı boyutu yönetilebilir tutulurken geniş bir kelime yelpazesi kapsanabilir.
*   **Morfolojinin Yönetimi:** "un-" veya "-ing" gibi ortak alt kelimeler, modellerin yaygın önek ve sonekleri öğrenmesine yardımcı olarak, görülmeyen kelime formlarına daha iyi genelleme yapmayı sağlar.
*   **Diller Arası Tutarlılık:** Çok dilli model mimarilerini basitleştiren evrensel bir alt kelime tokenizasyon stratejisi birden fazla dilde uygulanabilir.

Sabit boyutlu giriş dizilerine ve gömülmelere dayanan güçlü sinirsel mimarilerin, özellikle **Transformer modellerinin** yükselişi, alt kelime tokenizasyonunu vazgeçilmez bir bileşen haline getirmiştir. WordPiece (BERT tarafından kullanılır), BPE (Byte Pair Encoding, GPT-2/3 tarafından kullanılır) ve Unigram (ALBERT tarafından kullanılır) gibi tokenlayıcılar standart hale gelmiş olup, SentencePiece bu algoritmaları uygulamak için birleşik ve esnek bir çerçeve sunmaktadır.

<a name="3-sentencepiece-temel-kavramlar-ve-metodoloji"></a>
## 3. SentencePiece: Temel Kavramlar ve Metodoloji
SentencePiece, öncelikle dil bağımsızlığı ve metni ham karakter dizileri olarak ele alma yaklaşımıyla kendini farklılaştırır.

<a name="31-dil-bağımsızlığı"></a>
### 3.1. Dil Bağımsızlığı
SentencePiece'in temel ayırt edici özelliklerinden biri, **dil-agnostik** olacak şekilde tasarlanmış olmasıdır. Geleneksel tokenlayıcılar genellikle kelimeleri ayırmak, metni normalleştirmek veya noktalama işaretlerini işlemek için dile özgü kurallar içeren ön-tokenizasyon adımları gerektirir. Bu adımlar farklı diller için uygulanması karmaşık olabilir ve çok dilli modeller eğitilirken tutarsızlıklar yaratabilir.

SentencePiece, bu durumu, giriş metninin **ham bayt dizisi** üzerinde doğrudan çalışarak aşar. Boşluk karakterleri de dahil olmak üzere tüm girişi bir Unicode karakter akışı olarak kabul eder; boşluk karakterleri genellikle diğer tokenizasyon şemalarında ayırıcı olarak ele alınır. Bu "ham metin" yaklaşımı, SentencePiece'in bir dilin kelime sınırları veya belirli dilbilimsel özellikleri hakkında herhangi bir bilgiye ihtiyaç duymadığı anlamına gelir. Bunun yerine, bu kalıpları, verilen bir derlemden eğitim aşamasında örtük olarak öğrenir. Bu evrensel yaklaşım, farklı dillerdeki veri ön işleme adımlarını basitleştirerek, özellikle büyük ölçekli çok dilli NLP uygulamaları için uygun hale getirir.

<a name="32-eğitim-algoritmaları-bpe-ve-unigram-dil-modeli"></a>
### 3.2. Eğitim Algoritmaları: BPE ve Unigram Dil Modeli
SentencePiece temel olarak iki alt kelime segmentasyon algoritması uygular: **Bayt Çifti Kodlaması (BPE)** ve **Unigram Dil Modeli**.

*   **Bayt Çifti Kodlaması (BPE):** Başlangıçta bir veri sıkıştırma algoritması olan BPE, alt kelime tokenizasyonu için adapte edilmiştir. Her karakteri ayrı bir alt kelime birimi olarak ele alarak başlar. Ardından, en sık yan yana gelen birim çiftini yeni, tek bir birim olarak yinelemeli olarak birleştirir. Bu işlem, önceden tanımlanmış bir kelime dağarcığı boyutuna ulaşılana veya daha fazla birleştirme mümkün olmayana kadar devam eder. Örneğin, "yeni" ve "lik" sık geçen komşu çiftlerse, "yenilik" olarak birleşebilirler. BPE deterministiktir, yani verilen herhangi bir girdi için tek, sabit bir segmentasyon üretir.

*   **Unigram Dil Modeli:** BPE'den farklı olarak, Unigram Dil Modeli algoritması, verilen bir girdi dizisi için birden fazla olası segmentasyona izin verir. Segmentasyonu bir maksimum olasılık tahmini problemi olarak görür. Büyük bir başlangıç kelime dağarcığı (örneğin, belirli bir uzunluktaki tüm benzersiz alt dizeler) ile başlar ve kelime dağarcığını yinelemeli olarak budar. Her iterasyonda, her alt kelime birimini kaldırmanın neden olduğu kaybı hesaplar. Kaybı minimum düzeyde artıran birimler (yani, doğru segmentasyon için gereksiz veya daha az önemli olanlar) istenen kelime dağarcığı boyutuna ulaşılana kadar kaldırılır. Çıkarım sırasında, öğrenilen alt kelime olasılıklarına dayanarak en olası segmentasyonu bulmak için Viterbi algoritmasını kullanır. Unigram modelinin önemli bir avantajı, gürültüye karşı sağlamlık ve eğitim verilerini artırma için faydalı olabilecek birden fazla segmentasyonu örnekleyebilmesidir.

SentencePiece her ikisi için de uygulamalar sunar, bu da kullanıcıların kendi görevleri ve veri setleri için en uygun algoritmayı seçmelerine olanak tanır. Her iki algoritma da denetimsiz olarak eğitilir ve sağlanan metin derleminden doğrudan optimal alt kelime birimlerini öğrenir.

<a name="33-boşluk-ve-bilinmeyen-tokenların-yönetimi"></a>
### 3.3. Boşluk ve Bilinmeyen Tokenların Yönetimi
SentencePiece'in en belirgin özelliklerinden biri, boşluk karakterini ele alma şeklidir. Metni ön işlem adımı olarak boşluklara göre ayırmak yerine, SentencePiece **boşluk karakterini** diğer karakterler gibi birinci sınıf bir vatandaş olarak kabul eder. Metni böldüğünde, başlangıçta bir boşlukla başlayan bir kelimenin önüne özel bir alt çizgi `_` karakteri ekler. Örneğin, "Merhaba dünya." metni `_Merhaba`, `_dünya`, `.` olarak tokenize edilebilir. Bu tutarlı kodlama, orijinal boşluk bilgisinin korunmasını sağlar, bu da mükemmel geri dönüştürülebilirlik (tokenlanmış segmentlerin, boşluklar da dahil olmak üzere orijinal diziyi tam olarak yeniden yapılandırmak için birleştirilmesi) sağlar. Bu, makine çevirisi gibi orijinal biçimlendirmenin korunmasının önemli olabileceği görevler için kritik öneme sahiptir.

**Bilinmeyen tokenlar** (öğrenilen kelime dağarcığında bulunmayan tokenlar) için SentencePiece, bunları oluşturan karakterlere veya daha küçük bilinen alt kelime birimlerine ayırarak ele alır. Bir karakterin kendisi kelime dağarcığında yoksa, özel bir `UNK` tokenı ile temsil edilebilir, ancak SentencePiece genellikle tüm bireysel karakterleri öğrendiği için bu daha az yaygındır. Bu strateji, kelime seviyesindeki tokenlayıcılarda doğal olan OOV sorununu önemli ölçüde azaltır ve bilinmeyen kelimeler nedeniyle bilgi kaybı olmadan tüm giriş metninin işlenmesini sağlar.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Aşağıdaki Python kod parçacığı, basit bir metin dosyasından bir SentencePiece tokenizerının nasıl eğitileceğini ve ardından örnek bir cümleyi kodlamak ve kod çözmek için nasıl kullanılacağını göstermektedir.

```python
import sentencepiece as spm

# 1. Eğitim için geçici bir metin dosyası hazırla
# Gerçek bir senaryoda, bu büyük bir metin derlemi olurdu.
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("Hello SentencePiece, this is a language independent tokenizer.\n")
    f.write("It handles complex words like 'multilingual' and 'tokenization' effectively.\n")
    f.write("Türkçe de destekler, örneğin 'merhaba dünya' ve 'dilbilgisel'.\n")
    f.write("SentencePiece is truly a simple and powerful tool.\n")

# 2. Bir SentencePiece modeli eğit
# --input: eğitim derleminin yolu
# --model_prefix: çıktı model dosyaları için önek (örn. .model ve .vocab)
# --vocab_size: istenen kelime dağarcığı boyutu
# --model_type: BPE veya unigram (varsayılan unigramdır)
# --character_coverage: modelin kapsadığı karakter miktarı (varsayılan 0.9995)
spm.SentencePieceTrainer.train(
    '--input=corpus.txt --model_prefix=my_spp_model --vocab_size=100 --model_type=bpe --character_coverage=1.0'
)

# 3. Eğitilmiş modeli yükle
sp = spm.SentencePieceProcessor()
sp.load("my_spp_model.model")

# 4. Bir cümleyi kodla
text_to_encode = "SentencePiece is truly a simple and powerful tool for NLP."
encoded_tokens = sp.encode_as_pieces(text_to_encode)
encoded_ids = sp.encode_as_ids(text_to_encode)

print(f"Orijinal metin: '{text_to_encode}'")
print(f"Parçalara ayrılmış hali: {encoded_tokens}")
print(f"ID'lere kodlanmış hali: {encoded_ids}")

# 5. ID'leri tekrar metne çevir
decoded_text = sp.decode_ids(encoded_ids)
print(f"Çözümlenmiş metin: '{decoded_text}'")

# Türkçe metin örneği
turkish_text = "Merhaba dünya, bu SentencePiece'in gücüdür."
encoded_turkish_pieces = sp.encode_as_pieces(turkish_text)
print(f"Türkçe metin: '{turkish_text}'")
print(f"Parçalara ayrılmış Türkçe metin: {encoded_turkish_pieces}")


(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç
SentencePiece, modern NLP ve üretken yapay zekada, özellikle Transformer tabanlı modellerin yaygınlaşmasıyla vazgeçilmez bir araç haline gelmiştir. Metni ham karakter dizileri olarak ele alma ve boşlukları öğrenilebilir bir token olarak dahil etme gibi temel tasarım prensipleri, tokenizasyonun geleneksel zorluklarının çoğuna güçlü ve zarif bir çözüm sunar.

Gerçekten **dil bağımsız** bir yaklaşım sunarak, SentencePiece, çok dilli NLP sistemlerinin geliştirilmesini ve dağıtımını basitleştirir, karmaşık, dile özgü ön işleme süreçlerine olan ihtiyacı ortadan kaldırır. **BPE** ve **Unigram Dil Modeli** gibi sağlam alt kelime algoritmalarını uygulaması, verimli kelime dağarcığı yönetimini sağlar, **kelime dışı (OOV)** sorunlarını önemli ölçüde azaltır ve modellerin morfolojik olarak zengin dilleri daha iyi işlemesini mümkün kılar. Orijinal metni tokenize edilmiş formundan mükemmel bir şekilde yeniden yapılandırabilme yeteneği, kullanışlılığını ve güvenilirliğini daha da artırır.

Üretken yapay zeka modelleri boyut ve karmaşıklık açısından büyümeye devam ettikçe, genellikle birden fazla dili kapsayan geniş ve çeşitli veri kümeleri üzerinde eğitildikçe, SentencePiece gibi tutarlı, verimli ve dil-agnostik bir tokenlayıcının rolü giderek daha kritik hale gelmektedir. Bu ileri modellerin insan dilini olağanüstü esneklik ve doğrulukla işlemesini ve üretmesini destekleyerek, daha gelişmiş ve küresel olarak uygulanabilir yapay zeka uygulamalarının önünü açmaktadır.




