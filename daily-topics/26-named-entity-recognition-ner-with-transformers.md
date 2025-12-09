# Named Entity Recognition (NER) with Transformers

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Named Entity Recognition (NER)](#2-understanding-named-entity-recognition-ner)
  - [2.1. Definition and Importance](#21-definition-and-importance)
  - [2.2. Traditional Approaches to NER](#22-traditional-approaches-to-ner)
- [3. The Role of Transformers in NER](#3-the-role-of-transformers-in-ner)
  - [3.1. Limitations of Previous Models](#31-limitations-of-previous-models)
  - [3.2. Transformer Architecture Overview](#32-transformer-architecture-overview)
  - [3.3. How Transformers Enhance NER](#33-how-transformers-enhance-ner)
  - [3.4. Popular Transformer Models for NER](#34-popular-transformer-models-for-ner)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
Named Entity Recognition (NER), a fundamental task in **Natural Language Processing (NLP)**, involves identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, medical codes, time expressions, and monetary values. This capability is crucial for a wide array of applications, including information extraction, question answering, text summarization, and machine translation. Historically, NER systems relied on rule-based methods, statistical models like Hidden Markov Models (HMMs) and Conditional Random Fields (CRFs), and earlier neural network architectures such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. While these approaches achieved significant success, they often struggled with capturing long-range dependencies, handling ambiguous contexts, and generalizing to new domains without extensive feature engineering.

The advent of the **Transformer architecture** in 2017 marked a paradigm shift in NLP. Transformers, particularly through models like **BERT (Bidirectional Encoder Representations from Transformers)**, **GPT (Generative Pre-trained Transformer)**, and their numerous successors, have revolutionized how machines understand and process human language. Their core mechanism, **self-attention**, allows them to weigh the importance of different words in a sequence when processing each word, effectively overcoming the limitations of sequential processing inherent in RNNs. This document delves into the application of Transformers to the NER task, exploring how these powerful models have significantly advanced the state-of-the-art, offering unparalleled accuracy and robustness. We will discuss the underlying principles of Transformers, their advantages in the context of NER, and provide a practical code example demonstrating their use.

<a name="2-understanding-named-entity-recognition-ner"></a>
## 2. Understanding Named Entity Recognition (NER)

<a name="21-definition-and-importance"></a>
### 2.1. Definition and Importance
**Named Entity Recognition (NER)** is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories. These categories typically include:
*   **Person:** Names of individuals (e.g., *Elon Musk*)
*   **Organization:** Names of companies, agencies, institutions (e.g., *Google, United Nations*)
*   **Location:** Geographical entities (e.g., *Paris, Mount Everest*)
*   **Date:** Absolute or relative dates (e.g., *January 1st, next week*)
*   **Time:** Time expressions (e.g., *10:30 AM*)
*   **Money:** Monetary values (e.g., *$500,000*)
*   **Percent:** Percentage expressions (e.g., *15%*)

The ability to accurately identify these entities unlocks a wealth of structured information from raw text. Its importance spans across various domains:
*   **Information Extraction:** Automatically populating databases with key facts from documents.
*   **Search and Recommendation Systems:** Enhancing search relevance by identifying entities in queries and documents.
*   **Customer Support:** Routing inquiries based on entities mentioned (e.g., product names, locations).
*   **Healthcare:** Extracting medical conditions, drug names, and patient information from clinical notes.
*   **Legal Tech:** Identifying parties, dates, and clauses in legal documents.
*   **News Analysis:** Tracking trends and events by recognizing key people, organizations, and locations in news articles.

<a name="22-traditional-approaches-to-ner"></a>
### 2.2. Traditional Approaches to NER
Before the deep learning era, NER systems primarily relied on:
*   **Rule-Based Systems:** These systems used hand-crafted rules, regular expressions, and dictionaries to identify entities. While precise for specific domains, they were brittle, required extensive manual effort, and struggled with scalability and generalization.
*   **Statistical Models:** Machine learning models like **Hidden Markov Models (HMMs)**, **Support Vector Machines (SVMs)**, and especially **Conditional Random Fields (CRFs)** became popular. CRFs, in particular, were effective because they could model the dependencies between neighboring labels and incorporate a rich set of features (e.g., word shape, capitalization, part-of-speech tags, gazetteers). However, these models still required meticulous feature engineering and struggled to capture complex, non-linear patterns or long-range contextual information without significant computational cost.
*   **Early Neural Networks:** With the rise of deep learning, architectures such as **Recurrent Neural Networks (RNNs)**, particularly **Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRUs)**, paired with **word embeddings** (e.g., Word2Vec, GloVe), offered a significant improvement. These models could automatically learn features from the data, alleviating the need for manual feature engineering. LSTMs with CRFs (Bi-LSTM-CRF) became the standard for sequence labeling tasks like NER, as they could model sequential dependencies and leverage the CRF layer for optimal label sequences. Despite their advancements, RNN-based models faced limitations in processing very long sequences due to vanishing/exploding gradients and their inherently sequential nature, which hindered parallelization and efficient capture of very long-range dependencies.

<a name="3-the-role-of-transformers-in-ner"></a>
## 3. The Role of Transformers in NER

<a name="31-limitations-of-previous-models"></a>
### 3.1. Limitations of Previous Models
While traditional statistical models and even early neural networks like LSTMs made significant strides in NER, they encountered inherent limitations:
*   **Limited Contextual Understanding:** RNNs process text sequentially, making it difficult to effectively capture dependencies between words that are far apart in a sentence. This limited their ability to resolve ambiguities or understand complex relationships.
*   **Lack of Parallelization:** The sequential nature of RNNs restricts parallel processing, leading to slow training times, especially with very large datasets or long sequences.
*   **Feature Engineering Dependency (for statistical models):** Statistical models heavily relied on meticulously engineered features, which were time-consuming to create and often domain-specific, hindering portability and scalability.
*   **Vanishing/Exploding Gradients:** RNNs were susceptible to vanishing or exploding gradients, making it challenging to learn long-term dependencies effectively, despite mechanisms like LSTMs.
*   **Ambiguity Resolution:** Resolving ambiguities where a word could be an entity or not, or belong to different entity types based on subtle contextual cues, remained a significant challenge.

<a name="32-transformer-architecture-overview"></a>
### 3.2. Transformer Architecture Overview
The **Transformer architecture**, introduced by Vaswani et al. in "Attention Is All You Need" (2017), revolutionized NLP by completely abandoning recurrence and convolutions in favor of a mechanism called **self-attention**.
Key components of a Transformer include:
*   **Self-Attention Mechanism:** At its core, self-attention allows the model to weigh the importance of all other words in the input sequence when encoding a particular word. This creates a highly contextualized representation for each word, as it considers global dependencies rather than just local or sequential ones. For each token, three vectors are computed: **Query (Q)**, **Key (K)**, and **Value (V)**. The attention scores are calculated by the dot product of Q with K, scaled, and passed through a softmax function to get attention weights. These weights are then multiplied by V to obtain the output.
*   **Multi-Head Attention:** Instead of performing a single attention function, multi-head attention linearly projects Q, K, and V multiple times with different, learned linear projections. This allows the model to jointly attend to information from different representation subspaces at different positions, enriching the contextual understanding.
*   **Positional Encoding:** Since Transformers lack recurrence, they do not inherently understand the order of words. **Positional encodings** are added to the input embeddings to inject information about the relative or absolute position of tokens in the sequence.
*   **Feed-Forward Networks:** After the attention layers, each position in the sequence passes through an identical, independently applied **position-wise feed-forward network**.
*   **Residual Connections and Layer Normalization:** Each sub-layer in the Transformer (self-attention, feed-forward) employs **residual connections** around it, followed by **layer normalization**. This helps in training deeper networks and preventing gradient issues.

<a name="33-how-transformers-enhance-ner"></a>
### 3.3. How Transformers Enhance NER
Transformers significantly boost NER performance due to several key advantages:
*   **Global Contextual Understanding:** The self-attention mechanism allows Transformers to capture dependencies between any two words in a sequence, regardless of their distance. This global perspective is crucial for understanding ambiguous entities or those dependent on far-off context. For example, in "Apple released a new phone today" vs. "Apple is a fruit", the model can discern the entity type of "Apple" based on the entire sentence.
*   **Parallelization:** The non-sequential nature of self-attention allows for massive parallelization during training, enabling the processing of much larger datasets and the development of deeper, more complex models.
*   **Pre-trained Language Models:** The most significant impact comes from **pre-trained Transformer-based language models**. These models (like BERT, RoBERTa, XLNet) are trained on vast amounts of text data (e.g., Wikipedia, BookCorpus) in a self-supervised manner (e.g., Masked Language Modeling, Next Sentence Prediction). This pre-training phase allows them to learn rich, generalized representations of language, including syntax, semantics, and world knowledge.
*   **Fine-tuning for NER:** For NER, a pre-trained Transformer model can be fine-tuned on a labeled NER dataset. This involves adding a simple classification head (e.g., a linear layer) on top of the Transformer's output layer for each token, mapping its context-rich representation to an entity tag (e.g., B-PER, I-PER, O). The fine-tuning process adapts the pre-trained knowledge to the specific NER task with relatively small amounts of labeled data, leading to state-of-the-art results.
*   **Handling Out-of-Vocabulary (OOV) Words:** Tokenization strategies like **WordPiece** or **Byte-Pair Encoding (BPE)**, used by Transformers, break down unknown words into subword units. This allows the model to infer meanings for OOV words by combining the representations of their known subword components, improving robustness.

<a name="34-popular-transformer-models-for-ner"></a>
### 3.4. Popular Transformer Models for NER
Many Transformer-based models have been adapted and fine-tuned for NER:
*   **BERT (Bidirectional Encoder Representations from Transformers):** One of the pioneering models, BERT's bidirectional context learning from masked language modeling made it incredibly effective for NER. Fine-tuning BERT for NER typically involves adding a linear layer on top of its token-level outputs.
*   **RoBERTa (A Robustly Optimized BERT Pretraining Approach):** An optimized version of BERT, trained with more data, larger batches, and longer training times, often yielding better performance than BERT.
*   **DistilBERT:** A distilled version of BERT, smaller and faster while retaining much of BERT's performance, making it suitable for deployment.
*   **XLNet:** Uses a permutation-based auto-regressive approach to learn bidirectional context, aiming to overcome BERT's mask token discrepancy during fine-tuning.
*   **Electra:** A more computationally efficient pre-training approach that trains a generator and a discriminator. The discriminator learns to identify "replaced tokens" generated by the generator, leading to faster pre-training and often better performance than BERT.
*   **Various Multilingual Models:** Models like **mBERT** (multilingual BERT) and **XLM-RoBERTa** are pre-trained on text from multiple languages, enabling cross-lingual NER or direct NER in resource-poor languages.

The fine-tuning paradigm with these pre-trained Transformers has become the dominant approach for achieving state-of-the-art results across various NER datasets and languages, significantly reducing the data and computational resources needed compared to training models from scratch.

<a name="4-code-example"></a>
## 4. Code Example

This Python code snippet demonstrates how to perform Named Entity Recognition using a pre-trained Transformer model from the Hugging Face `transformers` library. We use the `pipeline` abstraction for simplicity, which wraps tokenization, model inference, and post-processing.

```python
from transformers import pipeline

# 1. Load a pre-trained Named Entity Recognition (NER) model
# The "ner" pipeline automatically uses a suitable tokenizer and model.
# "dbmdz/bert-large-cased-finetuned-conll03-english" is a common choice for English NER.
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

# 2. Define the input text
text = "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico on April 4, 1975."

# 3. Perform NER on the text
results = ner_pipeline(text)

# 4. Print the identified entities
print("Named Entity Recognition Results:")
for entity in results:
    print(f"  Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}")

# Example with another sentence
text_2 = "Dr. Sarah Miller, a physician from London, attended the WHO conference."
results_2 = ner_pipeline(text_2)
print("\nNamed Entity Recognition Results (Sentence 2):")
for entity in results_2:
    print(f"  Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.4f}")

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion
Named Entity Recognition (NER) has undergone a profound transformation with the advent of the Transformer architecture. By leveraging the power of self-attention mechanisms and massive pre-training on extensive text corpora, Transformer-based models have significantly surpassed the performance of traditional statistical and earlier neural network approaches. Their ability to capture global contextual dependencies, process information in parallel, and generalize effectively through fine-tuning has made them the de facto standard for state-of-the-art NER systems.

The fine-tuning paradigm, where pre-trained models like BERT, RoBERTa, or XLM-RoBERTa are adapted to specific NER datasets, has not only boosted accuracy but also reduced the computational and data demands for developing high-performing NER solutions. This advancement has profound implications across various industries, enabling more efficient information extraction, enhancing search functionalities, and driving innovation in fields from healthcare to legal tech. As research in generative AI and Transformer architectures continues to evolve, we can anticipate even more sophisticated and robust NER capabilities, pushing the boundaries of what machines can understand from human language. The ongoing development of multimodal Transformers and more efficient architectures promises a future where named entity recognition is not only highly accurate but also seamlessly integrated into complex reasoning and generation tasks.

---
<br>

<a name="türkçe-içerik"></a>
## Transformer Modelleri ile Adlandırılmış Varlık Tanıma (NER)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Adlandırılmış Varlık Tanıma (NER) Nedir?](#2-adlandırılmış-varlık-tanıma-ner-nedir)
  - [2.1. Tanım ve Önemi](#21-tanım-ve-önemi)
  - [2.2. NER'e Geleneksel Yaklaşımlar](#22-nere-geleneksel-yaklaşımlar)
- [3. NER'de Transformer Modellerinin Rolü](#3-nerde-transformer-modellerinin-rolü)
  - [3.1. Önceki Modellerin Sınırlamaları](#31-önceki-modellerin-sınırlamaları)
  - [3.2. Transformer Mimarisine Genel Bakış](#32-transformer-mimarisine-genel-bakış)
  - [3.3. Transformer Modelleri NER'i Nasıl Geliştirir?](#33-transformer-modelleri-neri-nasıl-geliştirir)
  - [3.4. NER için Popüler Transformer Modelleri](#34-ner-için-popüler-transformer-modelleri)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
**Doğal Dil İşleme (NLP)** alanında temel bir görev olan Adlandırılmış Varlık Tanıma (NER), metindeki adlandırılmış varlıkları kişi adları, kuruluşlar, konumlar, tıbbi kodlar, zaman ifadeleri ve parasal değerler gibi önceden tanımlanmış kategorilere ayırmayı ve sınıflandırmayı içerir. Bu yetenek, bilgi çıkarımı, soru yanıtlama, metin özetleme ve makine çevirisi gibi çok çeşitli uygulamalar için hayati öneme sahiptir. Tarihsel olarak, NER sistemleri kural tabanlı yöntemlere, Gizli Markov Modelleri (HMM'ler) ve Koşullu Rastgele Alanlar (CRF'ler) gibi istatistiksel modellere ve Tekrarlayan Sinir Ağları (RNN'ler) ile Uzun Kısa Süreli Bellek (LSTM) ağları gibi önceki sinir ağı mimarilerine dayanmaktaydı. Bu yaklaşımlar önemli başarılar elde etse de, uzun menzilli bağımlılıkları yakalama, belirsiz bağlamları ele alma ve kapsamlı özellik mühendisliği olmadan yeni alanlara genelleme konusunda zorluklar yaşamışlardır.

2017 yılında **Transformer mimarisinin** ortaya çıkışı, NLP'de bir paradigma değişikliğine işaret etti. Özellikle **BERT (Bidirectional Encoder Representations from Transformers)**, **GPT (Generative Pre-trained Transformer)** ve onların sayısız halefi gibi modeller aracılığıyla Transformer'lar, makinelerin insan dilini anlama ve işleme şeklini devrim niteliğinde değiştirdi. Temel mekanizmaları olan **öz-dikkat (self-attention)**, her bir kelimeyi işlerken bir dizideki diğer kelimelerin önemini tartmalarına olanak tanıyarak, RNN'lerdeki sıralı işlemenin doğal sınırlamalarını etkili bir şekilde aşar. Bu belge, Transformer'ların NER görevine uygulanmasını, bu güçlü modellerin son teknolojiyi nasıl önemli ölçüde geliştirdiğini ve benzersiz doğruluk ve sağlamlık sunduğunu incelemektedir. Transformer'ların temel prensiplerini, NER bağlamındaki avantajlarını tartışacak ve kullanımlarını gösteren pratik bir kod örneği sunacağız.

<a name="2-adlandırılmış-varlık-tanıma-ner-nedir"></a>
## 2. Adlandırılmış Varlık Tanıma (NER) Nedir?

<a name="21-tanım-ve-önemi"></a>
### 2.1. Tanım ve Önemi
**Adlandırılmış Varlık Tanıma (NER)**, yapılandırılmamış metinde geçen adlandırılmış varlıkları bulmayı ve önceden tanımlanmış kategorilere göre sınıflandırmayı amaçlayan bir bilgi çıkarımı alt görevidir. Bu kategoriler genellikle şunları içerir:
*   **Kişi:** Bireylerin adları (örn. *Elon Musk*)
*   **Kuruluş:** Şirketlerin, ajansların, kurumların adları (örn. *Google, Birleşmiş Milletler*)
*   **Konum:** Coğrafi varlıklar (örn. *Paris, Everest Dağı*)
*   **Tarih:** Mutlak veya göreceli tarihler (örn. *1 Ocak, gelecek hafta*)
*   **Zaman:** Zaman ifadeleri (örn. *Sabah 10:30*)
*   **Para:** Parasal değerler (örn. *500.000 $*)
*   **Yüzde:** Yüzde ifadeleri (örn. *%15*)

Bu varlıkları doğru bir şekilde tanımlama yeteneği, ham metinden zengin yapılandırılmış bilgilerin kilidini açar. Önemi çeşitli alanlara yayılır:
*   **Bilgi Çıkarımı:** Belgelerden anahtar bilgileri otomatik olarak veritabanlarına doldurma.
*   **Arama ve Öneri Sistemleri:** Sorgulardaki ve belgelerdeki varlıkları tanımlayarak arama alaka düzeyini artırma.
*   **Müşteri Desteği:** Bahsedilen varlıklara (örn. ürün adları, konumlar) göre sorguları yönlendirme.
*   **Sağlık Hizmetleri:** Klinik notlardan tıbbi durumları, ilaç adlarını ve hasta bilgilerini çıkarma.
*   **Hukuk Teknolojisi:** Hukuki belgelerdeki tarafları, tarihleri ve maddeleri belirleme.
*   **Haber Analizi:** Haber makalelerindeki anahtar kişileri, kuruluşları ve konumları tanıyarak trendleri ve olayları takip etme.

<a name="22-nere-geleneksel-yaklaşımlar"></a>
### 2.2. NER'e Geleneksel Yaklaşımlar
Derin öğrenme döneminden önce, NER sistemleri öncelikle şunlara dayanmaktaydı:
*   **Kural Tabanlı Sistemler:** Bu sistemler, varlıkları tanımlamak için el yapımı kurallar, düzenli ifadeler ve sözlükler kullanırdı. Belirli alanlar için hassas olsa da, kırılganlardı, yoğun manuel çaba gerektiriyorlardı ve ölçeklenebilirlik ve genelleştirilebilirlik konusunda zorluklar yaşıyorlardı.
*   **İstatistiksel Modeller:** **Gizli Markov Modelleri (HMM'ler)**, **Destek Vektör Makineleri (SVM'ler)** ve özellikle **Koşullu Rastgele Alanlar (CRF'ler)** gibi makine öğrenimi modelleri popüler hale geldi. Özellikle CRF'ler, komşu etiketler arasındaki bağımlılıkları modelleyebilmeleri ve zengin bir özellik kümesi (örn. kelime şekli, büyük/küçük harf kullanımı, sözcük türü etiketleri, gazete adları) içerebilmeleri nedeniyle etkiliydi. Ancak, bu modeller hala titiz özellik mühendisliği gerektiriyordu ve karmaşık, doğrusal olmayan kalıpları veya uzun menzilli bağlamsal bilgileri önemli bir hesaplama maliyeti olmadan yakalamakta zorlanıyorlardı.
*   **Erken Sinir Ağları:** Derin öğrenmenin yükselişiyle birlikte, **Tekrarlayan Sinir Ağları (RNN'ler)**, özellikle **Uzun Kısa Süreli Bellek (LSTM)** ağları ve **Kapılı Tekrarlayan Birimler (GRU'lar)** gibi mimariler, **kelime gömmeleri** (örn. Word2Vec, GloVe) ile birleşerek önemli bir gelişme sundu. Bu modeller, veri özelliklerini otomatik olarak öğrenebiliyor ve manuel özellik mühendisliği ihtiyacını ortadan kaldırıyordu. LSTM'ler ile CRF'ler (Bi-LSTM-CRF), sıralı bağımlılıkları modelleyebilmeleri ve optimal etiket dizileri için CRF katmanını kullanabilmeleri nedeniyle NER gibi dizi etiketleme görevleri için standart hale geldi. İlerlemelerine rağmen, RNN tabanlı modeller çok uzun dizileri işleme konusunda, kaybolan/patlayan gradyanlar ve doğal olarak sıralı yapıları nedeniyle sınırlamalarla karşılaştı, bu da paralelleştirmeyi ve çok uzun menzilli bağımlılıkların verimli bir şekilde yakalanmasını engelliyordu.

<a name="3-nerde-transformer-modellerinin-rolü"></a>
## 3. NER'de Transformer Modellerinin Rolü

<a name="31-önceki-modellerin-sınırlamaları"></a>
### 3.1. Önceki Modellerin Sınırlamaları
Geleneksel istatistiksel modeller ve hatta LSTM'ler gibi erken sinir ağları NER'de önemli ilerlemeler kaydetse de, içsel sınırlamalarla karşılaştılar:
*   **Sınırlı Bağlamsal Anlama:** RNN'ler metni sıralı olarak işler, bu da bir cümlede birbirinden uzak olan kelimeler arasındaki bağımlılıkları etkili bir şekilde yakalamayı zorlaştırır. Bu, belirsizlikleri giderme veya karmaşık ilişkileri anlama yeteneklerini sınırlar.
*   **Paralelleştirme Eksikliği:** RNN'lerin sıralı yapısı paralel işlemeyi kısıtlar ve özellikle çok büyük veri kümeleri veya uzun dizilerle yavaş eğitim sürelerine yol açar.
*   **Özellik Mühendisliği Bağımlılığı (istatistiksel modeller için):** İstatistiksel modeller, titizlikle tasarlanmış özelliklere büyük ölçüde güveniyordu; bu özellikler oluşturulması zaman alıcıydı ve genellikle alana özgüydü, bu da taşınabilirliği ve ölçeklenebilirliği engelliyordu.
*   **Kaybolan/Patlayan Gradyanlar:** RNN'ler, LSTM gibi mekanizmalara rağmen, kaybolan veya patlayan gradyanlara karşı hassastı, bu da uzun vadeli bağımlılıkları etkili bir şekilde öğrenmeyi zorlaştırıyordu.
*   **Belirsizliği Giderme:** Bir kelimenin varlık olup olmadığı veya ince bağlamsal ipuçlarına dayanarak farklı varlık türlerine ait olup olmadığı gibi belirsizlikleri gidermek önemli bir sorun olarak kaldı.

<a name="32-transformer-mimarisine-genel-bakış"></a>
### 3.2. Transformer Mimarisine Genel Bakış
Vaswani ve arkadaşları tarafından "Attention Is All You Need" (2017) adlı makalede tanıtılan **Transformer mimarisi**, tekrarlayan yapıları ve evrişimleri tamamen terk ederek **öz-dikkat (self-attention)** adı verilen bir mekanizma lehine NLP'de devrim yarattı.
Bir Transformer'ın temel bileşenleri şunları içerir:
*   **Öz-Dikkat Mekanizması:** Özünde, öz-dikkat, modelin belirli bir kelimeyi kodlarken girdi dizisindeki diğer tüm kelimelerin önemini tartmasına olanak tanır. Bu, her kelime için son derece bağlamsallaştırılmış bir temsil oluşturur, çünkü yalnızca yerel veya sıralı olanlar yerine küresel bağımlılıkları dikkate alır. Her jeton için üç vektör hesaplanır: **Sorgu (Q)**, **Anahtar (K)** ve **Değer (V)**. Dikkat skorları, Q'nun K ile nokta çarpımıyla hesaplanır, ölçeklenir ve dikkat ağırlıklarını elde etmek için bir softmax fonksiyonundan geçirilir. Bu ağırlıklar daha sonra V ile çarpılarak çıktı elde edilir.
*   **Çok Başlı Dikkat (Multi-Head Attention):** Tek bir dikkat fonksiyonu yerine, çok başlı dikkat Q, K ve V'yi farklı, öğrenilmiş doğrusal projeksiyonlarla birden çok kez doğrusal olarak yansıtır. Bu, modelin farklı konumdaki farklı temsil alt uzaylarından gelen bilgilere birlikte dikkat etmesini sağlayarak bağlamsal anlayışı zenginleştirir.
*   **Konumsal Kodlama (Positional Encoding):** Transformer'lar tekrarlayan bir yapıya sahip olmadıkları için kelimelerin sırasını doğal olarak anlamazlar. Dizideki jetonların göreceli veya mutlak konumu hakkında bilgi vermek için giriş gömmelerine **konumsal kodlamalar** eklenir.
*   **İleri Beslemeli Ağlar (Feed-Forward Networks):** Dikkat katmanlarından sonra, dizideki her konum, aynı, bağımsız olarak uygulanan **konum bazlı ileri beslemeli ağdan** geçer.
*   **Kalıntı Bağlantıları ve Katman Normalizasyonu:** Transformer'daki her alt katman (öz-dikkat, ileri beslemeli) etrafında **kalıntı bağlantıları** kullanır ve ardından **katman normalizasyonu** yapar. Bu, daha derin ağları eğitmeye ve gradyan sorunlarını önlemeye yardımcı olur.

<a name="33-transformer-modelleri-neri-nasıl-geliştirir"></a>
### 3.3. Transformer Modelleri NER'i Nasıl Geliştirir?
Transformer'lar, birkaç temel avantaj sayesinde NER performansını önemli ölçüde artırır:
*   **Küresel Bağlamsal Anlama:** Öz-dikkat mekanizması, Transformer'ların bir dizideki herhangi iki kelime arasındaki bağımlılıkları, aralarındaki mesafeye bakılmaksızın yakalamasına olanak tanır. Bu küresel bakış açısı, belirsiz varlıkları veya uzaktaki bağlama bağlı olanları anlamak için çok önemlidir. Örneğin, "Apple bugün yeni bir telefon çıkardı" cümlesi ile "Apple bir meyvedir" cümlesinde, model "Apple" kelimesinin varlık tipini tüm cümleye dayanarak ayırt edebilir.
*   **Paralelleştirme:** Öz-dikkat'in sıralı olmayan yapısı, eğitim sırasında büyük ölçekli paralelleştirmeye olanak tanır, bu da çok daha büyük veri kümelerinin işlenmesini ve daha derin, daha karmaşık modellerin geliştirilmesini mümkün kılar.
*   **Önceden Eğitilmiş Dil Modelleri:** En önemli etki, **önceden eğitilmiş Transformer tabanlı dil modellerinden** gelmektedir. Bu modeller (BERT, RoBERTa, XLNet gibi) geniş miktarda metin verisi (örn. Wikipedia, BookCorpus) üzerinde kendiliğinden denetimli bir şekilde (örn. Maskeli Dil Modelleme, Sonraki Cümle Tahmini) eğitilir. Bu ön eğitim aşaması, dilin sözdizimi, anlambilimi ve dünya bilgisi dahil olmak üzere zengin, genelleştirilmiş temsillerini öğrenmelerine olanak tanır.
*   **NER için İnce Ayar:** NER için, önceden eğitilmiş bir Transformer modeli etiketli bir NER veri kümesi üzerinde ince ayar yapılabilir. Bu, Transformer'ın her jeton için çıktı katmanının üzerine basit bir sınıflandırma başlığı (örn. doğrusal bir katman) eklemeyi içerir ve onun bağlam açısından zengin temsilini bir varlık etiketine (örn. B-PER, I-PER, O) eşler. İnce ayar süreci, önceden eğitilmiş bilgiyi nispeten az miktarda etiketli veriyle belirli NER görevine uyarlar ve son teknoloji sonuçlara yol açar.
*   **Kelimelik Dışı (OOV) Kelimeleri İşleme:** Transformer'lar tarafından kullanılan **WordPiece** veya **Byte-Pair Encoding (BPE)** gibi jetonlama stratejileri, bilinmeyen kelimeleri alt kelime birimlerine ayırır. Bu, modelin OOV kelimeler için bilinen alt kelime bileşenlerinin temsillerini birleştirerek anlamlar çıkarmasına olanak tanır ve sağlamlığı artırır.

<a name="34-ner-için-popüler-transformer-modelleri"></a>
### 3.4. NER için Popüler Transformer Modelleri
Birçok Transformer tabanlı model, NER için adapte edilmiş ve ince ayar yapılmıştır:
*   **BERT (Bidirectional Encoder Representations from Transformers):** Öncü modellerden biri olan BERT'in maskeli dil modellemesinden iki yönlü bağlam öğrenimi, onu NER için inanılmaz derecede etkili kılmıştır. BERT'i NER için ince ayar yapmak genellikle jeton düzeyindeki çıktıların üzerine doğrusal bir katman eklemeyi içerir.
*   **RoBERTa (A Robustly Optimized BERT Pretraining Approach):** BERT'in optimize edilmiş bir versiyonudur, daha fazla veri, daha büyük yığınlar ve daha uzun eğitim süreleri ile eğitilmiş olup, genellikle BERT'ten daha iyi performans gösterir.
*   **DistilBERT:** BERT'in damıtılmış bir versiyonudur, daha küçük ve daha hızlıdır, ancak BERT'in performansının çoğunu korur, bu da onu dağıtım için uygun hale getirir.
*   **XLNet:** İnce ayar sırasında BERT'in maske jetonu tutarsızlığını aşmayı amaçlayan, iki yönlü bağlamı öğrenmek için permütasyon tabanlı bir otomatik-regresif yaklaşım kullanır.
*   **Electra:** Bir jeneratör ve bir ayrıştırıcı eğiten daha hesaplama açısından verimli bir ön eğitim yaklaşımıdır. Ayrıştırıcı, jeneratör tarafından üretilen "değiştirilen jetonları" tanımlamayı öğrenir, bu da daha hızlı ön eğitime ve genellikle BERT'ten daha iyi performansa yol açar.
*   **Çeşitli Çok Dilli Modeller:** **mBERT** (çok dilli BERT) ve **XLM-RoBERTa** gibi modeller, birden çok dilden metin üzerinde önceden eğitilmiştir, bu da diller arası NER'i veya kaynak açısından fakir dillerde doğrudan NER'i mümkün kılar.

Bu önceden eğitilmiş Transformer'larla yapılan ince ayar paradigması, çeşitli NER veri kümelerinde ve dillerde en son teknoloji sonuçlara ulaşmak için baskın yaklaşım haline gelmiştir ve modelleri sıfırdan eğitmeye kıyasla gereken veri ve hesaplama kaynaklarını önemli ölçüde azaltmıştır.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Bu Python kod parçacığı, Hugging Face `transformers` kütüphanesinden önceden eğitilmiş bir Transformer modeli kullanarak Adlandırılmış Varlık Tanıma'yı nasıl gerçekleştireceğinizi gösterir. Basitlik için jetonlaştırma, model çıkarımı ve son işlemeyi kapsayan `pipeline` soyutlamasını kullanıyoruz.

```python
from transformers import pipeline

# 1. Önceden eğitilmiş bir Adlandırılmış Varlık Tanıma (NER) modelini yükle
# "ner" pipeline'ı otomatik olarak uygun bir jetonlayıcı ve model kullanır.
# "dbmdz/bert-large-cased-finetuned-conll03-english" İngilizce NER için yaygın bir seçimdir.
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

# 2. Giriş metnini tanımla
text = "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico on April 4, 1975."

# 3. Metin üzerinde NER uygula
results = ner_pipeline(text)

# 4. Tanımlanan varlıkları yazdır
print("Adlandırılmış Varlık Tanıma Sonuçları:")
for entity in results:
    print(f"  Varlık: {entity['word']}, Tip: {entity['entity_group']}, Skor: {entity['score']:.4f}")

# Başka bir cümle ile örnek
text_2 = "Dr. Sarah Miller, a physician from London, attended the WHO conference."
results_2 = ner_pipeline(text_2)
print("\nAdlandırılmış Varlık Tanıma Sonuçları (Cümle 2):")
for entity in results_2:
    print(f"  Varlık: {entity['word']}, Tip: {entity['entity_group']}, Skor: {entity['score']:.4f}")

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç
Adlandırılmış Varlık Tanıma (NER), Transformer mimarisinin ortaya çıkışıyla derin bir dönüşüm geçirdi. Öz-dikkat mekanizmalarının ve geniş metin korpusları üzerindeki büyük ölçekli ön eğitimin gücünden yararlanarak, Transformer tabanlı modeller geleneksel istatistiksel ve önceki sinir ağı yaklaşımlarının performansını önemli ölçüde geride bıraktı. Küresel bağlamsal bağımlılıkları yakalama, bilgiyi paralel olarak işleme ve ince ayar yoluyla etkili bir şekilde genelleme yetenekleri, onları en son teknoloji NER sistemleri için fiili standart haline getirdi.

BERT, RoBERTa veya XLM-RoBERTa gibi önceden eğitilmiş modellerin belirli NER veri kümelerine adapte edildiği ince ayar paradigması, doğruluğu artırmakla kalmamış, aynı zamanda yüksek performanslı NER çözümleri geliştirmek için gereken hesaplama ve veri taleplerini de azaltmıştır. Bu ilerleme, çeşitli endüstrilerde derin etkilere sahiptir; daha verimli bilgi çıkarımı, arama işlevlerinin geliştirilmesi ve sağlıktan hukuk teknolojisine kadar birçok alanda yenilikçiliği teşvik etmektedir. Üretken yapay zeka ve Transformer mimarileri üzerindeki araştırmalar geliştikçe, makinelerin insan dilinden anlayabildiği sınırları zorlayan daha da sofistike ve sağlam NER yetenekleri bekleyebiliriz. Çok modlu Transformer'ların ve daha verimli mimarilerin devam eden gelişimi, adlandırılmış varlık tanımanın sadece yüksek doğruluklu değil, aynı zamanda karmaşık akıl yürütme ve üretim görevlerine sorunsuz bir şekilde entegre olduğu bir gelecek vaat ediyor.
