# Text Classification with Convolutional Neural Networks

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Text Classification](#2-understanding-text-classification)
- [3. Convolutional Neural Networks (CNNs) for Text](#3-convolutional-neural-networks-cnns-for-text)
- [4. Advantages and Limitations](#4-advantages-and-limitations)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

Text classification is a foundational task in **Natural Language Processing (NLP)**, involving the assignment of predefined categories or labels to blocks of text. This process is crucial for a myriad of applications, including spam detection, sentiment analysis, topic labeling, and content moderation. While traditional machine learning techniques like Support Vector Machines (SVMs) and Naive Bayes classifiers have long been employed, the advent of deep learning has revolutionized the field, offering more sophisticated and accurate methods. Among these, **Convolutional Neural Networks (CNNs)**, originally popularized for their exceptional performance in image recognition, have demonstrated remarkable efficacy in processing sequential data like text.

In the context of **Generative AI**, text classification plays a vital role in evaluating, filtering, and understanding the output of large language models (LLMs). For instance, classifying generated text as "safe" or "unsafe," identifying its stylistic attributes, or verifying its coherence are all applications that benefit from robust text classification systems. This document will delve into the principles of text classification using CNNs, exploring their architecture, advantages, and practical implementation in the realm of modern AI.

<a name="2-understanding-text-classification"></a>
## 2. Understanding Text Classification

At its core, **text classification** is a supervised learning problem where an algorithm learns to map input text to one or more target categories. The input data typically consists of a collection of text documents, each pre-assigned a label. The goal is to train a model that can accurately predict these labels for new, unseen text.

Common types of text classification include:
*   **Binary Classification:** Assigning text to one of two categories (e.g., spam/not spam, positive/negative sentiment).
*   **Multi-class Classification:** Assigning text to one of several mutually exclusive categories (e.g., news articles classified as sports, politics, or entertainment).
*   **Multi-label Classification:** Assigning text to multiple categories simultaneously (e.g., a movie review classified as "action," "comedy," and "sci-fi").

Traditional approaches to text classification often involve a two-step process:
1.  **Feature Extraction:** Converting raw text into numerical feature vectors. Popular methods include **Bag-of-Words (BoW)**, **TF-IDF (Term Frequency-Inverse Document Frequency)**, and N-gram representations. These methods convert text into sparse, high-dimensional vectors, often losing semantic and contextual information.
2.  **Classification Algorithm:** Applying a classical machine learning algorithm (e.g., SVM, Naive Bayes, Logistic Regression) to the extracted features.

While effective for simpler tasks, traditional methods struggle with capturing complex semantic relationships, word order, and nuanced context, which are inherently present in human language. This is where deep learning models, particularly CNNs, offer a significant advantage by learning hierarchical features directly from the raw text representations.

<a name="3-convolutional-neural-networks-cnns-for-text"></a>
## 3. Convolutional Neural Networks (CNNs) for Text

CNNs gained prominence in computer vision due to their ability to automatically learn **hierarchical features** from image data, such as edges, textures, and object parts. While text data is one-dimensional and sequential, the core concept of convolution can be effectively adapted. Instead of processing pixels, CNNs for text operate on **word embeddings**.

The typical architecture of a CNN for text classification involves several key components:

1.  **Input Layer with Word Embeddings:**
    *   Raw text (sentences or documents) is first tokenized into individual words.
    *   Each word is then converted into a dense, low-dimensional **word embedding** vector (e.g., Word2Vec, GloVe, FastText). These embeddings capture semantic meaning, where words with similar meanings are represented by similar vectors in the embedding space. This transforms the input text into a sequence of vectors, forming a 2D matrix (sequence length x embedding dimension). This matrix can be thought of as a "one-channel image" for the CNN.

2.  **Convolutional Layers:**
    *   A convolutional layer applies multiple **filters** (or kernels) of varying sizes across the embedded input. Each filter is essentially a small matrix that slides over a fixed-size window of words (e.g., 2, 3, or 5 words) within the input sequence.
    *   For each window, the filter performs an element-wise multiplication with the words in the window and sums the results, producing a single scalar. This operation is repeated across the entire input sequence, generating a **feature map**.
    *   Different filters are designed to detect different patterns or **N-grams** (e.g., phrases, idioms, specific word combinations) within the text. For instance, a filter of size 3 might detect three-word phrases, while a filter of size 5 might capture five-word patterns. The output of a convolutional layer consists of multiple feature maps, each corresponding to a specific filter.

3.  **Activation Function:**
    *   After the convolution operation, a non-linear **activation function** (commonly ReLU - Rectified Linear Unit) is applied to the feature maps. This introduces non-linearity, allowing the model to learn more complex patterns.

4.  **Pooling Layers:**
    *   Pooling layers, most commonly **max-pooling**, are used to downsample the feature maps, reducing their dimensionality and making the model more robust to variations in feature position.
    *   Max-pooling selects the maximum value from a fixed-size window in each feature map. This operation captures the most important feature (the strongest activation) detected by a filter within a particular region, regardless of its exact position. This helps in identifying the most salient features across the entire text.

5.  **Flatten Layer and Fully Connected Layers:**
    *   The outputs from the pooling layers (often from multiple filters of different sizes concatenated) are then flattened into a single, long feature vector.
    *   This flattened vector is fed into one or more fully connected (dense) layers, which learn to combine the high-level features extracted by the convolutional layers.
    *   A final dense layer with a softmax activation function is used for multi-class classification (outputting probabilities for each class) or a sigmoid activation for binary classification.

The power of CNNs for text lies in their ability to automatically learn relevant local patterns (N-grams) and hierarchical features, without requiring manual feature engineering. By using multiple filters of varying sizes, CNNs can capture different granularities of linguistic patterns, making them highly effective for various text classification tasks.

<a name="4-advantages-and-limitations"></a>
## 4. Advantages and Limitations

**Advantages of CNNs for Text Classification:**

*   **Automatic Feature Extraction:** CNNs automatically learn meaningful features (like N-grams) directly from the word embeddings, eliminating the need for extensive manual feature engineering (e.g., TF-IDF).
*   **Local Feature Learning:** They are highly effective at detecting local patterns and relationships within fixed-size windows of text, such as phrases and sequences of words.
*   **Positional Invariance (within filter scope):** Through pooling, CNNs can identify key features regardless of their exact position within a sentence, contributing to robustness.
*   **Parallel Computation:** Convolutional operations are highly parallelizable, making CNNs efficient to train on GPUs.
*   **Parameter Sharing:** Filters are applied across the entire input sequence, meaning the same set of weights (parameters) is used repeatedly. This reduces the total number of parameters, mitigating overfitting and improving generalization.
*   **Effective with Pre-trained Embeddings:** When combined with powerful pre-trained word embeddings, CNNs can leverage vast amounts of semantic knowledge, leading to better performance, especially with limited labeled training data.

**Limitations of CNNs for Text Classification:**

*   **Limited Long-Range Dependency Capture:** While effective at capturing local patterns, traditional CNNs (especially those with small filter sizes) might struggle to capture very long-range dependencies across distant words in a sentence or document without very deep architectures or sophisticated pooling strategies. This is an area where models like Transformers often excel.
*   **Fixed Filter Sizes:** The requirement to pre-define filter sizes means that patterns shorter or longer than the chosen filter sizes might be less optimally captured. Using multiple filter sizes can mitigate this but adds complexity.
*   **Computational Cost:** For very long documents or sequences, processing through multiple convolutional layers can still be computationally intensive, although generally less so than some recurrent architectures for long sequences.
*   **Lack of Sequential Understanding (explicitly):** Unlike Recurrent Neural Networks (RNNs) or Transformers, CNNs do not inherently process text in a strict sequential order. They capture patterns by sliding windows, which is different from a step-by-step processing of tokens.

Despite these limitations, CNNs remain a powerful and efficient choice for a wide array of text classification tasks, particularly when local features and phrase-level patterns are important.

<a name="5-code-example"></a>
## 5. Code Example

Here's a simple Keras/TensorFlow example demonstrating a basic CNN architecture for text classification. This snippet illustrates the key layers: `Embedding`, `Conv1D`, `GlobalMaxPooling1D`, and `Dense`.

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. Dummy Data (Replace with your actual dataset)
sentences = [
    "This is a great movie, loved it!",
    "Terrible plot, waste of time.",
    "Excellent performance by the actors.",
    "Boring and predictable story.",
    "Highly recommend this film.",
    "Absolute disaster, avoid at all costs."
]
labels = np.array([1, 0, 1, 0, 1, 0]) # 1 for positive, 0 for negative

# 2. Tokenization and Padding
vocab_size = 1000
embedding_dim = 100
max_len = 20 # Maximum sequence length

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 3. Define the CNN Model
input_layer = Input(shape=(max_len,))

# Embedding Layer: Converts word indices to dense vectors
x = Embedding(vocab_size, embedding_dim, input_length=max_len)(input_layer)

# Convolutional Layer: Applies filters to detect local patterns
# We use multiple filter sizes (e.g., 3, 4, 5) and concatenate their outputs for better feature extraction.
conv_blocks = []
filter_sizes = [3, 4, 5]
for f_size in filter_sizes:
    conv = Conv1D(filters=128, kernel_size=f_size, activation='relu')(x)
    pool = GlobalMaxPooling1D()(conv) # Global Max Pooling to get the most important feature per filter
    conv_blocks.append(pool)

# Concatenate the outputs of different filter sizes
x = tf.keras.layers.concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

# Fully Connected Layers for classification
x = Dense(64, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x) # Sigmoid for binary classification

model = Model(inputs=input_layer, outputs=output_layer)

# 4. Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Model Summary
model.summary()

# 6. Dummy Training (for demonstration purposes, use proper train/test split)
# model.fit(padded_sequences, labels, epochs=10, batch_size=2)

print("\nModel setup complete. Use model.fit() with your actual data for training.")

(End of code example section)
```
<a name="6-conclusion"></a>
## 6. Conclusion

Convolutional Neural Networks have proven to be a highly effective and efficient architecture for a wide range of text classification tasks. By leveraging **word embeddings** and employing **filters** to detect local **N-gram patterns**, CNNs can automatically learn meaningful features from text data, overcoming many of the limitations of traditional bag-of-words approaches. Their ability to capture important phrases and local semantic relationships, combined with their computational efficiency due to parallel processing and parameter sharing, makes them a strong contender in the NLP toolkit.

In the rapidly evolving landscape of **Generative AI**, robust text classification models like those built with CNNs are indispensable. They can be utilized to ensure the safety and quality of generated content, perform stylistic analysis, or even categorize the intent behind user prompts. While more complex architectures like Transformers excel at long-range dependencies, CNNs offer a simpler, often faster, and still highly performant solution for many common text classification challenges, making them a valuable tool for both research and industrial applications.

---
<br>

<a name="türkçe-içerik"></a>
## Evrişimli Sinir Ağları ile Metin Sınıflandırması

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Metin Sınıflandırmasını Anlamak](#2-metin-sınıflandırmasını-anlamak)
- [3. Metin İçin Evrişimli Sinir Ağları (ESA'lar)](#3-metin-için-evrişimli-sinir-ağları-esalar)
- [4. Avantajlar ve Sınırlamalar](#4-avantajlar-ve-sınırlamalar)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Metin sınıflandırması, **Doğal Dil İşleme (DDI)** alanında temel bir görev olup, metin bloklarına önceden tanımlanmış kategoriler veya etiketler atamayı içerir. Bu süreç, spam tespiti, duygu analizi, konu etiketleme ve içerik denetimi gibi sayısız uygulama için hayati öneme sahiptir. Destek Vektör Makineleri (DVM'ler) ve Naive Bayes sınıflandırıcıları gibi geleneksel makine öğrenimi teknikleri uzun süredir kullanılsa da, derin öğrenmenin ortaya çıkışı bu alanı devrim niteliğinde değiştirmiş, daha sofistike ve doğru yöntemler sunmuştur. Bunlar arasında, başlangıçta görüntü tanımadaki olağanüstü performanslarıyla popülerlik kazanan **Evrişimli Sinir Ağları (ESA'lar)**, metin gibi sıralı verileri işlemede dikkate değer bir etkinlik göstermiştir.

**Üretken Yapay Zeka** bağlamında, metin sınıflandırması, büyük dil modellerinin (BHM'ler) çıktısını değerlendirmede, filtrelemede ve anlamada hayati bir rol oynar. Örneğin, üretilen metni "güvenli" veya "güvenli değil" olarak sınıflandırmak, stilistik özelliklerini belirlemek veya tutarlılığını doğrulamak, sağlam metin sınıflandırma sistemlerinden fayda sağlayan uygulamalardır. Bu belge, ESA'ları kullanarak metin sınıflandırmasının prensiplerini inceleyecek, mimarilerini, avantajlarını ve modern yapay zeka alanındaki pratik uygulamalarını keşfedecektir.

<a name="2-metin-sınıflandırmasını-anlamak"></a>
## 2. Metin Sınıflandırmasını Anlamak

Özünde, **metin sınıflandırması**, bir algoritmanın giriş metnini bir veya daha fazla hedef kategoriye eşlemeyi öğrendiği, denetimli bir öğrenme problemidir. Giriş verileri genellikle, her biri önceden bir etikete atanmış bir metin belgesi koleksiyonundan oluşur. Amaç, yeni, görülmeyen metinler için bu etiketleri doğru bir şekilde tahmin edebilen bir model eğitmektir.

Yaygın metin sınıflandırma türleri şunları içerir:
*   **İkili Sınıflandırma:** Metni iki kategoriden birine atama (örn. spam/spam değil, pozitif/negatif duygu).
*   **Çok Sınıflı Sınıflandırma:** Metni, birbirini dışlayan birkaç kategoriden birine atama (örn. spor, politika veya eğlence olarak sınıflandırılan haber makaleleri).
*   **Çok Etiketli Sınıflandırma:** Metni aynı anda birden çok kategoriye atama (örn. "aksiyon", "komedi" ve "bilim kurgu" olarak sınıflandırılan bir film incelemesi).

Metin sınıflandırmasına yönelik geleneksel yaklaşımlar genellikle iki adımlı bir süreç içerir:
1.  **Özellik Çıkarımı:** Ham metni sayısal özellik vektörlerine dönüştürme. Popüler yöntemler arasında **Kelime Torbası (BoW)**, **TF-IDF (Terim Sıklığı-Ters Belge Sıklığı)** ve N-gram temsilleri bulunur. Bu yöntemler, metni seyrek, yüksek boyutlu vektörlere dönüştürerek genellikle anlamsal ve bağlamsal bilgiyi kaybeder.
2.  **Sınıflandırma Algoritması:** Çıkarılan özelliklere klasik bir makine öğrenimi algoritması (örn. DVM, Naive Bayes, Lojistik Regresyon) uygulama.

Daha basit görevler için etkili olsalar da, geleneksel yöntemler, insan dilinde doğal olarak bulunan karmaşık anlamsal ilişkileri, kelime sırasını ve incelikli bağlamı yakalamakta zorlanır. Derin öğrenme modelleri, özellikle ESA'lar, ham metin temsillerinden hiyerarşik özellikleri doğrudan öğrenerek burada önemli bir avantaj sunar.

<a name="3-metin-için-evrişimli-sinir-ağları-esalar"></a>
## 3. Metin İçin Evrişimli Sinir Ağları (ESA'lar)

ESA'lar, görüntü verilerinden kenarlar, dokular ve nesne parçaları gibi **hiyerarşik özellikleri** otomatik olarak öğrenme yetenekleri nedeniyle bilgisayar görüşünde öne çıkmıştır. Metin verileri tek boyutlu ve sıralı olsa da, evrişim kavramı etkili bir şekilde uyarlanabilir. Pikselleri işlemek yerine, metin için ESA'lar **kelime gömmeleri** üzerinde çalışır.

Metin sınıflandırması için tipik bir ESA mimarisi birkaç temel bileşeni içerir:

1.  **Kelime Gömme Katmanı ile Giriş Katmanı:**
    *   Ham metin (cümleler veya belgeler) önce ayrı kelimelere belirteçlere ayrılır.
    *   Her kelime daha sonra yoğun, düşük boyutlu bir **kelime gömme** vektörüne dönüştürülür (örn. Word2Vec, GloVe, FastText). Bu gömmeler anlamsal anlamı yakalar, burada benzer anlama sahip kelimeler gömme uzayında benzer vektörlerle temsil edilir. Bu, giriş metnini bir dizi vektöre dönüştürerek 2D bir matris (sıra uzunluğu x gömme boyutu) oluşturur. Bu matris, ESA için "tek kanallı bir görüntü" olarak düşünülebilir.

2.  **Evrişim Katmanları:**
    *   Bir evrişim katmanı, gömülü giriş üzerinde değişen boyutlarda birden çok **filtre** (veya çekirdek) uygular. Her filtre, giriş dizisindeki sabit boyutlu bir kelime penceresi (örn. 2, 3 veya 5 kelime) üzerinde kayan küçük bir matristir.
    *   Her pencere için, filtre penceredeki kelimelerle eleman bazında çarpma yapar ve sonuçları toplayarak tek bir skaler üretir. Bu işlem, tüm giriş dizisi boyunca tekrarlanır ve bir **özellik haritası** oluşturur.
    *   Farklı filtreler, metin içindeki farklı kalıpları veya **N-gramları** (örn. cümleler, deyimler, belirli kelime kombinasyonları) tespit etmek için tasarlanmıştır. Örneğin, 3 boyutlu bir filtre üç kelimelik cümleleri tespit edebilirken, 5 boyutlu bir filtre beş kelimelik kalıpları yakalayabilir. Bir evrişim katmanının çıktısı, her biri belirli bir filtreye karşılık gelen birden çok özellik haritasından oluşur.

3.  **Aktivasyon Fonksiyonu:**
    *   Evrişim işleminden sonra, özellik haritalarına doğrusal olmayan bir **aktivasyon fonksiyonu** (genellikle ReLU - Düzeltilmiş Doğrusal Birim) uygulanır. Bu, doğrusal olmama özelliği ekleyerek modelin daha karmaşık kalıpları öğrenmesini sağlar.

4.  **Havuzlama Katmanları:**
    *   Havuzlama katmanları, en yaygın olarak **maks-havuzlama**, özellik haritalarının boyutunu küçültmek ve modeli özellik konumundaki varyasyonlara karşı daha sağlam hale getirmek için kullanılır.
    *   Maks-havuzlama, her özellik haritasındaki sabit boyutlu bir pencereden maksimum değeri seçer. Bu işlem, belirli bir bölgedeki bir filtre tarafından tespit edilen en önemli özelliği (en güçlü aktivasyonu), tam konumuna bakılmaksızın yakalar. Bu, tüm metin boyunca en belirgin özelliklerin belirlenmesine yardımcı olur.

5.  **Düzleştirme Katmanı ve Tam Bağlantılı Katmanlar:**
    *   Havuzlama katmanlarından gelen çıktılar (genellikle farklı boyutlardaki birden çok filtrenin birleştirilmesiyle) daha sonra tek, uzun bir özellik vektörüne düzleştirilir.
    *   Bu düzleştirilmiş vektör, evrişim katmanları tarafından çıkarılan yüksek seviyeli özellikleri birleştirmeyi öğrenen bir veya daha fazla tam bağlantılı (yoğun) katmana beslenir.
    *   Çok sınıflı sınıflandırma için (her sınıf için olasılıklar üreterek) veya ikili sınıflandırma için sigmoid aktivasyonlu son bir yoğun katman kullanılır.

ESA'ların metin için gücü, manuel özellik mühendisliğine gerek kalmadan ilgili yerel kalıpları (N-gramlar) ve hiyerarşik özellikleri otomatik olarak öğrenebilme yeteneklerinden kaynaklanır. Değişen boyutlarda birden çok filtre kullanarak, ESA'lar farklı dilsel kalıp tanelerini yakalayabilir, bu da onları çeşitli metin sınıflandırma görevleri için oldukça etkili kılar.

<a name="4-avantajlar-ve-sınırlamalar"></a>
## 4. Avantajlar ve Sınırlamalar

**Metin Sınıflandırması İçin ESA'ların Avantajları:**

*   **Otomatik Özellik Çıkarımı:** ESA'lar, kelime gömmelerinden doğrudan anlamlı özellikleri (N-gramlar gibi) otomatik olarak öğrenir, kapsamlı manuel özellik mühendisliğine (örn. TF-IDF) olan ihtiyacı ortadan kaldırır.
*   **Yerel Özellik Öğrenimi:** Metin içindeki sabit boyutlu pencerelerdeki yerel kalıpları ve ilişkileri, örneğin cümleleri ve kelime dizilerini tespit etmede oldukça etkilidirler.
*   **Konumsal Değişmezlik (filtre kapsamı içinde):** Havuzlama yoluyla, ESA'lar temel özellikleri bir cümledeki tam konumlarından bağımsız olarak tanımlayabilir ve sağlamlığa katkıda bulunur.
*   **Paralel Hesaplama:** Evrişim işlemleri yüksek derecede paralelleştirilebilir, bu da ESA'ları GPU'larda eğitmeyi verimli hale getirir.
*   **Parametre Paylaşımı:** Filtreler tüm giriş dizisi boyunca uygulanır, yani aynı ağırlık kümesi (parametreler) tekrar tekrar kullanılır. Bu, toplam parametre sayısını azaltır, aşırı uyumu hafifletir ve genellemeyi iyileştirir.
*   **Önceden Eğitilmiş Gömelerle Etkili:** Güçlü önceden eğitilmiş kelime gömmelerle birleştirildiğinde, ESA'lar özellikle sınırlı etiketli eğitim verisiyle daha iyi performans göstererek büyük miktarda anlamsal bilgiyi kullanabilir.

**Metin Sınıflandırması İçin ESA'ların Sınırlamaları:**

*   **Sınırlı Uzun Menzilli Bağımlılık Yakalama:** Yerel kalıpları yakalamada etkili olsalar da, geleneksel ESA'lar (özellikle küçük filtre boyutlarına sahip olanlar), çok derin mimariler veya sofistike havuzlama stratejileri olmadan bir cümledeki veya belgedeki uzak kelimeler arasındaki çok uzun menzilli bağımlılıkları yakalamakta zorlanabilirler. Bu, Transformer'lar gibi modellerin genellikle üstün olduğu bir alandır.
*   **Sabit Filtre Boyutları:** Filtre boyutlarını önceden tanımlama gerekliliği, seçilen filtre boyutlarından daha kısa veya daha uzun kalıpların daha az optimal şekilde yakalanabileceği anlamına gelir. Birden çok filtre boyutu kullanmak bunu hafifletebilir ancak karmaşıklık ekler.
*   **Hesaplama Maliyeti:** Çok uzun belgeler veya diziler için, birden çok evrişim katmanından geçmek hala hesaplama açısından yoğun olabilir, ancak genellikle uzun diziler için bazı tekrarlayan mimarilerden daha azdır.
*   **Sıralı Anlayış Eksikliği (açıkça):** Tekrarlayan Sinir Ağları (TSA'lar) veya Transformer'ların aksine, ESA'lar metni katı bir sıralı düzende işlemezler. Desenleri kayan pencerelerle yakalarlar, bu da belirteçlerin adım adım işlenmesinden farklıdır.

Bu sınırlamalara rağmen, ESA'lar, özellikle yerel özellikler ve öbek seviyesi kalıplar önemli olduğunda, çok çeşitli metin sınıflandırma görevleri için güçlü ve verimli bir seçenek olmaya devam etmektedir.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği

İşte metin sınıflandırması için temel bir ESA mimarisini gösteren basit bir Keras/TensorFlow örneği. Bu kod parçacığı, `Embedding`, `Conv1D`, `GlobalMaxPooling1D` ve `Dense` gibi temel katmanları göstermektedir.

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. Sahte Veri (Gerçek veri setinizle değiştirin)
sentences = [
    "Bu harika bir film, çok sevdim!",
    "Korkunç bir senaryo, zaman kaybı.",
    "Oyuncuların performansı mükemmeldi.",
    "Sıkıcı ve tahmin edilebilir bir hikaye.",
    "Bu filmi şiddetle tavsiye ediyorum.",
    "Tam bir felaket, ne pahasına olursa olsun kaçının."
]
labels = np.array([1, 0, 1, 0, 1, 0]) # 1 pozitif, 0 negatif için

# 2. Tokenizasyon ve Doldurma
vocab_size = 1000 # Kelime dağarcığı boyutu
embedding_dim = 100 # Gömme boyutu
max_len = 20 # Maksimum dizi uzunluğu

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 3. ESA Modelini Tanımlama
input_layer = Input(shape=(max_len,))

# Gömme Katmanı: Kelime indekslerini yoğun vektörlere dönüştürür
x = Embedding(vocab_size, embedding_dim, input_length=max_len)(input_layer)

# Evrişim Katmanı: Yerel kalıpları tespit etmek için filtreler uygular
# Daha iyi özellik çıkarımı için birden çok filtre boyutu (örn. 3, 4, 5) kullanırız ve çıktılarını birleştiririz.
conv_blocks = []
filter_sizes = [3, 4, 5]
for f_size in filter_sizes:
    conv = Conv1D(filters=128, kernel_size=f_size, activation='relu')(x)
    # Her filtre için en önemli özelliği elde etmek üzere Global Maks Havuzlama
    pool = GlobalMaxPooling1D()(conv) 
    conv_blocks.append(pool)

# Farklı filtre boyutlarının çıktılarını birleştirme
x = tf.keras.layers.concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

# Sınıflandırma için Tam Bağlantılı Katmanlar
x = Dense(64, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x) # İkili sınıflandırma için Sigmoid aktivasyon

model = Model(inputs=input_layer, outputs=output_layer)

# 4. Modeli Derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Model Özeti
model.summary()

# 6. Sahte Eğitim (gösterim amaçlı, gerçek train/test ayrımı kullanın)
# model.fit(padded_sequences, labels, epochs=10, batch_size=2)

print("\nModel kurulumu tamamlandı. Eğitim için gerçek verilerinizle model.fit() kullanın.")

(Kod örneği bölümünün sonu)
```
<a name="6-sonuç"></a>
## 6. Sonuç

Evrişimli Sinir Ağları, çok çeşitli metin sınıflandırma görevleri için oldukça etkili ve verimli bir mimari olduğunu kanıtlamıştır. **Kelime gömmelerini** kullanarak ve yerel **N-gram kalıplarını** tespit etmek için **filtreler** uygulayarak, ESA'lar metin verilerinden anlamlı özellikleri otomatik olarak öğrenebilir ve geleneksel kelime torbası yaklaşımlarının birçok sınırlamasını aşabilir. Önemli cümleleri ve yerel anlamsal ilişkileri yakalama yetenekleri, paralel işleme ve parametre paylaşımı sayesinde sağladıkları hesaplama verimliliğiyle birleştiğinde, onları DDI araç setinde güçlü bir aday haline getirmektedir.

Hızla gelişen **Üretken Yapay Zeka** ortamında, ESA'larla oluşturulanlar gibi sağlam metin sınıflandırma modelleri vazgeçilmezdir. Üretilen içeriğin güvenliğini ve kalitesini sağlamak, stilistik analiz yapmak veya hatta kullanıcı istemlerinin ardındaki amacı kategorize etmek için kullanılabilirler. Transformer'lar gibi daha karmaşık mimariler uzun menzilli bağımlılıklarda üstün olsa da, ESA'lar birçok yaygın metin sınıflandırma zorluğu için daha basit, genellikle daha hızlı ve hala yüksek performanslı bir çözüm sunarak hem araştırma hem de endüstriyel uygulamalar için değerli bir araçtır.


