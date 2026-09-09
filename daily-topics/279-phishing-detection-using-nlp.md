# Phishing Detection using NLP

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Phishing and NLP Fundamentals](#2-understanding-phishing-and-nlp-fundamentals)
    - [2.1. The Evolving Threat of Phishing](#21-the-evolving-threat-of-phishing)
    - [2.2. Natural Language Processing (NLP) Fundamentals](#22-natural-language-processing-nlp-fundamentals)
- [3. NLP Techniques for Phishing Detection](#3-nlp-techniques-for-phishing-detection)
    - [3.1. Text Preprocessing](#31-text-preprocessing)
    - [3.2. Feature Extraction](#32-feature-extraction)
    - [3.3. Machine Learning and Deep Learning Models](#33-machine-learning-and-deep-learning-models)
- [4. Code Example](#4-code-example)
- [5. Challenges and Future Directions](#5-challenges-and-future-directions)
- [6. Conclusion](#6-conclusion)

---

## 1. Introduction

Phishing remains one of the most pervasive and dangerous cyber threats, continually evolving in sophistication to deceive users into divulging sensitive information. These attacks often leverage social engineering tactics embedded within textual content, such as emails, instant messages, or website prompts, making **Natural Language Processing (NLP)** a critical domain for their detection and mitigation. NLP, a subfield of artificial intelligence, focuses on enabling computers to understand, interpret, and generate human language. Its application in cybersecurity, particularly for threat intelligence and anomaly detection, has seen significant advancements. This document explores the comprehensive application of NLP techniques for identifying and preventing phishing attacks, detailing the methodologies, relevant algorithms, and the underlying principles that make NLP an indispensable tool in the modern security landscape. We will delve into various stages, from text preprocessing to advanced machine learning and deep learning models, illustrating how linguistic patterns and semantic analysis can reveal malicious intent hidden within seemingly legitimate communications.

## 2. Understanding Phishing and NLP Fundamentals

### 2.1. The Evolving Threat of Phishing

Phishing attacks are a form of cybercrime where attackers masquerade as trustworthy entities in electronic communication to trick individuals into providing sensitive data, typically login credentials, financial information, or proprietary data. The methods employed are diverse and continually adapt:

*   **Email Phishing:** The most common form, sending deceptive emails that appear to originate from legitimate sources (banks, service providers, government agencies).
*   **Spear Phishing:** Highly targeted attacks customized for specific individuals or organizations, often requiring prior research into the target.
*   **Whaling:** A specific type of spear phishing targeting senior executives or high-profile individuals within an organization.
*   **Smishing (SMS Phishing):** Phishing attempts conducted via text messages, often containing malicious links or requests for callbacks to fraudulent numbers.
*   **Vishing (Voice Phishing):** Phishing conducted over telephone calls, where attackers impersonate trusted entities to extract information.

The common thread across these variants is the reliance on deceptive language and social engineering. Attackers exploit psychological vulnerabilities, creating a sense of urgency, fear, or curiosity to bypass rational thought. The textual characteristics—such as unusual grammar, suspicious URLs, generic greetings, and urgent calls to action—are key indicators that NLP can effectively analyze.

### 2.2. Natural Language Processing (NLP) Fundamentals

NLP provides a suite of techniques to analyze and derive meaning from human language. For phishing detection, several foundational NLP concepts are crucial:

*   **Tokenization:** The process of breaking down a text into smaller units, such as words or subwords (tokens). For example, "Phishing detection is vital." becomes ["Phishing", "detection", "is", "vital", "."].
*   **Stop-word Removal:** Eliminating common words (e.g., "the", "a", "is", "of") that carry little semantic meaning and might unnecessarily increase data dimensionality.
*   **Stemming and Lemmatization:**
    *   **Stemming:** Reducing words to their root form (e.g., "running", "runs", "ran" → "run"). It's a heuristic process, often chopping off suffixes.
    *   **Lemmatization:** Reducing words to their base or dictionary form (lemma), considering the word's context and part of speech (e.g., "better" → "good"). It is more linguistically sophisticated than stemming.
*   **Part-of-Speech (POS) Tagging:** Identifying the grammatical role of words (e.g., noun, verb, adjective), which can provide structural insights into suspicious sentences.
*   **Named Entity Recognition (NER):** Identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, monetary values, expressions of times, etc. This can help identify legitimate entities being impersonated or manipulated.

These foundational steps transform raw, unstructured text into a structured format suitable for computational analysis, laying the groundwork for more advanced detection mechanisms.

## 3. NLP Techniques for Phishing Detection

The application of NLP in phishing detection involves a sequence of steps, from initial data preparation to advanced model training and deployment.

### 3.1. Text Preprocessing

Before any meaningful analysis can occur, raw text data from emails, messages, or web pages must be meticulously cleaned and prepared. This stage is paramount as the quality of input directly impacts model performance. Key preprocessing steps include:

*   **Cleaning:** Removing irrelevant characters, HTML tags, JavaScript code, and special symbols that do not contribute to semantic meaning.
*   **Lowercasing:** Converting all text to lowercase to ensure consistency and prevent the model from treating "Phishing" and "phishing" as different words.
*   **Punctuation Removal:** While sometimes useful for specific linguistic analysis, often punctuation is removed to focus on word content.
*   **Handling URLs/IP Addresses:** URLs and IP addresses are critical indicators. They can be extracted as separate features or masked (e.g., replacing all URLs with a `<URL>` token) to simplify the text while retaining their presence. Often, **URL parsing** and **domain reputation checks** are performed as separate, powerful features.
*   **Handling Emojis/Special Characters:** Depending on the context (e.g., SMS phishing), emojis might be relevant indicators or noise. They are either removed or converted to textual descriptions.

### 3.2. Feature Extraction

Once text is preprocessed, it needs to be converted into numerical representations (features) that machine learning models can understand. This is a crucial step that translates linguistic patterns into mathematical vectors.

*   **Bag-of-Words (BoW):** This model represents text as an unordered collection of words, disregarding grammar and word order. It counts the frequency of each word in a document. The entire vocabulary forms a vector space, and each document is represented as a vector of word counts.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** An improvement over BoW, TF-IDF weighs word frequencies by how rare they are across the entire corpus. Words that appear frequently in a document but rarely in the corpus (i.e., unique to the document) receive higher scores, indicating their importance. This helps in identifying context-specific keywords.
*   **N-grams:** Instead of single words (unigrams), N-grams consider sequences of N words (e.g., bigrams for two words, trigrams for three). This captures some limited contextual information, such as "account verification" or "urgent action required."
*   **Word Embeddings:** More sophisticated techniques that represent words as dense vectors in a continuous vector space, where words with similar meanings are located closer together.
    *   **Word2Vec (Skip-gram, CBOW):** Learns word embeddings by predicting context from a target word or vice-versa.
    *   **GloVe (Global Vectors for Word Representation):** Combines global matrix factorization and local context window methods.
    *   **FastText:** Extends Word2Vec by representing words as sums of their character n-grams, allowing it to handle out-of-vocabulary words and morphologically rich languages.
*   **Contextual Embeddings (Deep Learning based):** These models generate word embeddings that are *context-dependent*, meaning the same word can have different embeddings based on the surrounding words in a sentence.
    *   **BERT (Bidirectional Encoder Representations from Transformers):** A powerful transformer-based model pre-trained on a large corpus, capable of understanding the bidirectional context of words. Fine-tuning BERT on phishing datasets can yield highly accurate feature representations.
    *   **RoBERTa, XLNet, ELECTRA, GPT:** Other transformer architectures offering similar or improved capabilities, often leveraging massive pre-training on diverse text corpora.
*   **Linguistic Features:** Beyond word content, specific linguistic characteristics can serve as features:
    *   **Sentiment Analysis:** Detecting the emotional tone (e.g., fear, urgency) in a message.
    *   **Readability Scores:** Assessing the complexity of the language (e.g., Flesch-Kincaid grade level).
    *   **Stylometric Features:** Analyzing writing style, such as average sentence length, punctuation frequency, character distribution, and unique word count, which can deviate significantly in phishing attempts.

### 3.3. Machine Learning and Deep Learning Models

With numerical features extracted, various classification models can be employed to distinguish legitimate communications from phishing attempts.

*   **Traditional Machine Learning Models:**
    *   **Naive Bayes:** A probabilistic classifier based on Bayes' theorem, assuming independence between features. It's often effective for text classification due to its simplicity and good performance with high-dimensional data.
    *   **Support Vector Machines (SVM):** A powerful algorithm that finds an optimal hyperplane to separate data points into different classes. SVMs are effective in high-dimensional spaces and with clear margins of separation.
    *   **Random Forests:** An ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
    *   **Logistic Regression:** A linear model for binary classification, suitable for providing probability estimates.
*   **Deep Learning Models:** These models are particularly effective with rich, high-dimensional features like word embeddings and can automatically learn intricate patterns.
    *   **Recurrent Neural Networks (RNNs):** Designed for sequential data, RNNs (and their variants like **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**) can process text word by word, retaining memory of previous words in the sequence. This is crucial for understanding context and dependency within a sentence.
    *   **Convolutional Neural Networks (CNNs):** While popular in image processing, CNNs are also effective for text classification. They apply filters over sequences of words (or word embeddings) to detect local patterns (e.g., n-grams) that indicate phishing.
    *   **Transformer Models:** Revolutionized NLP with their self-attention mechanisms, allowing them to weigh the importance of different words in a sentence irrespective of their distance. Models like **BERT** (as mentioned in feature extraction) can also be fine-tuned as classifiers directly, offering state-of-the-art performance by learning deep contextual representations.

The choice of model often depends on the dataset size, computational resources, and the desired balance between interpretability and accuracy. Hybrid approaches, combining traditional ML with deep learning features, are also common.

## 4. Code Example

This short Python snippet demonstrates a basic phishing detection workflow: text preprocessing and TF-IDF vectorization, followed by a simple Logistic Regression classifier.

```python
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample dataset: emails (text) and their labels (0: legitimate, 1: phishing)
data = [
    ("Your account has been suspended. Click here to verify.", 1),
    ("Dear customer, your order #12345 has been shipped.", 0),
    ("URGENT: Your bank account needs immediate attention. Login now.", 1),
    ("Meeting reminder for tomorrow at 10 AM in conference room B.", 0),
    ("We detected unusual activity on your account. Please update your details.", 1),
    ("Regarding the project proposal, please review the attached document.", 0),
    ("ATTENTION: Your password has expired. Click this link to reset it.", 1),
    ("Hello, just checking in on your availability for next week.", 0),
]

texts = [item[0] for item in data]
labels = [item[1] for item in data]

# 1. Text Preprocessing Function
def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text) # Replace URLs with <URL> token
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.strip() # Remove leading/trailing whitespace
    return text

# Apply preprocessing
preprocessed_texts = [preprocess_text(text) for text in texts]

# 2. Feature Extraction using TF-IDF
# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(preprocessed_texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 3. Model Training (Logistic Regression)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Example prediction
new_email_phishing = "Action required: Your account has been compromised. Log in immediately."
new_email_legit = "Hello, following up on our discussion regarding the project timeline."

# Preprocess new emails
preprocessed_phishing = preprocess_text(new_email_phishing)
preprocessed_legit = preprocess_text(new_email_legit)

# Vectorize new emails
vectorized_phishing = tfidf_vectorizer.transform([preprocessed_phishing])
vectorized_legit = tfidf_vectorizer.transform([preprocessed_legit])

# Predict
prediction_phishing = model.predict(vectorized_phishing)[0]
prediction_legit = model.predict(vectorized_legit)[0]

print(f"\n'{new_email_phishing}' -> Predicted: {'Phishing' if prediction_phishing == 1 else 'Legitimate'}")
print(f"'{new_email_legit}' -> Predicted: {'Phishing' if prediction_legit == 1 else 'Legitimate'}")

(End of code example section)
```

## 5. Challenges and Future Directions

Despite significant progress, NLP-based phishing detection faces several ongoing challenges:

*   **Evolving Tactics and Adversarial Attacks:** Phishers continuously adapt their language, using new synonyms, grammatical structures, and sophisticated social engineering narratives to bypass detection systems. Adversarial attacks can subtly alter malicious text to fool models without changing its human-readable intent.
*   **Concept Drift:** The characteristics of phishing emails can change over time, rendering older models less effective. Continuous monitoring and retraining are necessary to combat this **concept drift**.
*   **Multilingual Phishing:** Attacks are not limited to English. Detecting phishing in various languages requires models trained on diverse multilingual datasets and robust cross-lingual NLP techniques.
*   **Low-Resource Languages:** Many languages lack sufficient labeled data or pre-trained models, making detection challenging for them.
*   **Contextual Ambiguity:** Distinguishing genuine urgent requests from malicious ones can be difficult, especially when legitimate communications use similar language (e.g., password reset notifications).
*   **Explainability (XAI):** Understanding *why* a model flagged an email as phishing is crucial for security analysts. Many advanced deep learning models are black boxes, making explainable AI techniques vital for trust and effective incident response.

Future directions in NLP for phishing detection include:

*   **Advanced Semantic Understanding:** Leveraging more powerful contextual embedding models and graph neural networks to better understand the relationships between entities and concepts within a message, identifying subtle inconsistencies.
*   **Fusion with Other Modalities:** Integrating NLP with other detection methods, such as URL analysis (checking domain age, WHOIS records, reputation), image analysis (for embedded malicious images), and behavioral analysis (sender reputation, past communication patterns).
*   **Active Learning and Few-Shot Learning:** Strategies to effectively learn from limited labeled data, reducing the manual effort required for data annotation, particularly for new and emerging phishing variants.
*   **Reinforcement Learning for Adaptive Defense:** Developing systems that can learn from interactions and feedback, adapting their detection strategies in real-time to counter evolving threats.
*   **Ethical AI and Bias Mitigation:** Ensuring that models do not inadvertently perpetuate biases present in training data, leading to unfair or incorrect classifications.

## 6. Conclusion

Natural Language Processing stands as a cornerstone in the ongoing battle against phishing attacks. By transforming unstructured textual data into actionable insights, NLP techniques enable the identification of linguistic anomalies, semantic inconsistencies, and social engineering cues indicative of malicious intent. From fundamental text preprocessing and robust feature extraction methods like TF-IDF and advanced word embeddings, to sophisticated machine learning and deep learning classification models such as SVMs, LSTMs, and Transformers, the arsenal of NLP tools is constantly expanding. While challenges persist, particularly concerning the dynamic nature of phishing tactics and the need for explainability, continuous innovation in the field promises even more resilient and adaptable detection systems. The integration of NLP with other cybersecurity intelligence, coupled with advancements in areas like contextual understanding and adaptive learning, paves the way for a future where digital communications are significantly more secure against the persistent threat of phishing.

---
<br>

<a name="türkçe-içerik"></a>
## Doğal Dil İşleme Kullanarak Oltalama Tespiti

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Oltalama ve DİP Temellerini Anlamak](#2-oltalama-ve-dip-temellerini-anlamak)
    - [2.1. Oltalamanın Gelişen Tehdidi](#21-oltalamanın-gelişen-tehdidi)
    - [2.2. Doğal Dil İşleme (DİP) Temelleri](#22-doğal-dil-işleme-dip-temelleri)
- [3. Oltalama Tespiti için DİP Teknikleri](#3-oltalama-tespiti-için-dip-teknikleri)
    - [3.1. Metin Ön İşleme](#31-metin-ön-işleme)
    - [3.2. Öznitelik Çıkarımı](#32-öznitelik-çıkarımı)
    - [3.3. Makine Öğrenimi ve Derin Öğrenme Modelleri](#33-makine-öğrenimi-ve-derin-öğrenme-modelleri)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Zorluklar ve Gelecek Yönelimler](#5-zorluklar-ve-gelecek-yönelimler)
- [6. Sonuç](#6-sonuç)

---

## 1. Giriş

Oltalama (phishing), kullanıcıları hassas bilgilerini ifşa etmeleri için kandırmak amacıyla sürekli gelişen ve karmaşıklaşan en yaygın ve tehlikeli siber tehditlerden biri olmaya devam etmektedir. Bu saldırılar genellikle e-postalar, anlık mesajlar veya web sitesi uyarıları gibi metinsel içeriklere yerleştirilmiş sosyal mühendislik taktiklerini kullanır ve bu da **Doğal Dil İşleme (DİP)**'yi bunların tespiti ve azaltılması için kritik bir alan haline getirir. Yapay zekanın bir alt alanı olan DİP, bilgisayarların insan dilini anlamasını, yorumlamasını ve üretmesini sağlamaya odaklanır. Siber güvenlik, özellikle tehdit istihbaratı ve anomali tespiti alanındaki uygulamaları önemli ilerlemeler kaydetmiştir. Bu belge, oltalama saldırılarını tanımlamak ve önlemek için DİP tekniklerinin kapsamlı uygulamasını incelemekte, metodolojileri, ilgili algoritmaları ve DİP'i modern güvenlik ortamında vazgeçilmez bir araç haline getiren temel ilkeleri detaylandırmaktadır. Metin ön işlemeden gelişmiş makine öğrenimi ve derin öğrenme modellerine kadar çeşitli aşamalara inecek, dilsel kalıpların ve anlamsal analizin, görünüşte yasal iletişimlerin içinde gizlenmiş kötü niyetli amacı nasıl ortaya çıkarabileceğini göstereceğiz.

## 2. Oltalama ve DİP Temellerini Anlamak

### 2.1. Oltalamanın Gelişen Tehdidi

Oltalama saldırıları, saldırganların elektronik iletişimde güvenilir varlıklar gibi davranarak bireyleri genellikle oturum açma kimlik bilgileri, finansal bilgiler veya tescilli veriler gibi hassas verileri sağlamaları için kandırdığı bir siber suç biçimidir. Kullanılan yöntemler çeşitlidir ve sürekli uyum sağlar:

*   **E-posta Oltalaması:** En yaygın biçim olup, yasal kaynaklardan (bankalar, servis sağlayıcılar, devlet kurumları) geldiği izlenimi veren aldatıcı e-postalar göndermeyi içerir.
*   **Hedefli Oltalama (Spear Phishing):** Belirli kişilere veya kuruluşlara yönelik, genellikle hedef hakkında önceden araştırma gerektiren oldukça hedefli saldırılar.
*   **Balina Avı (Whaling):** Bir kuruluş içindeki üst düzey yöneticileri veya yüksek profilli kişileri hedef alan belirli bir hedefli oltalama türü.
*   **SMS Oltalaması (Smishing):** Metin mesajları aracılığıyla gerçekleştirilen oltalama girişimleri, genellikle kötü amaçlı bağlantılar veya dolandırıcı numaralara geri arama istekleri içerir.
*   **Sesli Oltalama (Vishing):** Telefon görüşmeleri üzerinden gerçekleştirilen oltalama, saldırganların bilgileri çıkarmak için güvenilir varlıkları taklit ettiği durumlar.

Bu varyantların ortak noktası, aldatıcı dil ve sosyal mühendisliğe dayanmalarıdır. Saldırganlar, rasyonel düşünmeyi atlamak için aciliyet, korku veya merak duygusu yaratarak psikolojik güvenlik açıklarını kullanırlar. Olağandışı dilbilgisi, şüpheli URL'ler, genel selamlamalar ve acil eylem çağrıları gibi metinsel özellikler, DİP'in etkili bir şekilde analiz edebileceği temel göstergelerdir.

### 2.2. Doğal Dil İşleme (DİP) Temelleri

DİP, insan dilinden anlam çıkarmak ve analiz etmek için bir dizi teknik sunar. Oltalama tespiti için birkaç temel DİP kavramı çok önemlidir:

*   **Belirteçlere Ayırma (Tokenization):** Bir metni kelimeler veya alt kelimeler (belirteçler) gibi daha küçük birimlere ayırma işlemidir. Örneğin, "Oltalama tespiti hayati öneme sahiptir." cümlesi ["Oltalama", "tespiti", "hayati", "öneme", "sahiptir", "."] haline gelir.
*   **Durma Kelimelerini Kaldırma (Stop-word Removal):** Çok az anlamsal anlam taşıyan ve veri boyutluluğunu gereksiz yere artırabilecek yaygın kelimelerin (örn. "bir", "ve", "ile") elenmesi.
*   **Kök Bulma (Stemming) ve Lemmatizasyon:**
    *   **Kök Bulma (Stemming):** Kelimeleri kök formlarına indirgeme (örn. "koştu", "koşuyor", "koşmak" → "koş"). Sezgisel bir süreçtir, genellikle son ekleri keser.
    *   **Lemmatizasyon:** Kelimeleri bağlamlarını ve sözdizimsel rollerini dikkate alarak temel veya sözlük formlarına (lemma) indirgeme (örn. "daha iyi" → "iyi"). Kök bulmadan daha dilbilimsel olarak sofistikedir.
*   **Sözcük Türü Etiketleme (Part-of-Speech (POS) Tagging):** Kelimelerin dilbilgisel rolünü (örn. isim, fiil, sıfat) belirleme, bu da şüpheli cümlelerin yapısal içgörülerini sağlayabilir.
*   **Adlandırılmış Varlık Tanıma (Named Entity Recognition (NER)):** Metindeki adlandırılmış varlıkları kişi adları, kuruluşlar, konumlar, parasal değerler, zaman ifadeleri vb. gibi önceden tanımlanmış kategorilere göre tanımlama ve sınıflandırma. Bu, taklit edilen veya manipüle edilen meşru varlıkları belirlemeye yardımcı olabilir.

Bu temel adımlar, ham, yapılandırılmamış metni, hesaplamalı analiz için uygun yapılandırılmış bir biçime dönüştürerek daha gelişmiş tespit mekanizmalarının temelini atar.

## 3. Oltalama Tespiti için DİP Teknikleri

DİP'in oltalama tespitinde uygulanması, ilk veri hazırlığından gelişmiş model eğitimi ve dağıtımına kadar bir dizi adımı içerir.

### 3.1. Metin Ön İşleme

Anlamlı bir analiz yapılmadan önce, e-postalardan, mesajlardan veya web sayfalarından gelen ham metin verileri titizlikle temizlenmeli ve hazırlanmalıdır. Bu aşama çok önemlidir, çünkü girdinin kalitesi doğrudan model performansını etkiler. Temel ön işleme adımları şunları içerir:

*   **Temizleme:** Anlamsal anlama katkıda bulunmayan alakasız karakterlerin, HTML etiketlerinin, JavaScript kodunun ve özel sembollerin kaldırılması.
*   **Küçük Harfe Çevirme:** Tutarlılık sağlamak ve modelin "Oltalama" ve "oltalama" kelimelerini farklı kelimeler olarak ele almasını önlemek için tüm metni küçük harfe dönüştürme.
*   **Noktalama İşaretlerini Kaldırma:** Bazen belirli dilsel analizler için yararlı olsa da, noktalama işaretleri genellikle kelime içeriğine odaklanmak için kaldırılır.
*   **URL/IP Adreslerini İşleme:** URL'ler ve IP adresleri kritik göstergelerdir. Ayrı özellikler olarak çıkarılabilir veya metni basitleştirirken varlıklarını korumak için maskelenebilir (örn. tüm URL'leri bir `<URL>` belirteciyle değiştirerek). Genellikle, ayrı ve güçlü özellikler olarak **URL ayrıştırma** ve **alan adı itibar kontrolü** yapılır.
*   **Emoji/Özel Karakterleri İşleme:** Bağlama bağlı olarak (örn. SMS oltalaması), emojiler ilgili göstergeler veya gürültü olabilir. Ya kaldırılır ya da metinsel açıklamalara dönüştürülür.

### 3.2. Öznitelik Çıkarımı

Metin ön işleme tabi tutulduktan sonra, makine öğrenimi modellerinin anlayabileceği sayısal gösterimlere (özelliklere) dönüştürülmesi gerekir. Bu, dilsel kalıpları matematiksel vektörlere çeviren kritik bir adımdır.

*   **Kelime Torbası (Bag-of-Words - BoW):** Bu model, metni gramer ve kelime sırasını göz ardı ederek kelimelerin sırasız bir koleksiyonu olarak temsil eder. Bir belgedeki her kelimenin sıklığını sayar. Tüm kelime dağarcığı bir vektör alanı oluşturur ve her belge bir kelime sayımı vektörü olarak temsil edilir.
*   **TF-IDF (Terim Sıklığı-Ters Belge Sıklığı):** BoW'a göre bir iyileştirme olan TF-IDF, kelime sıklıklarını tüm külliyatta ne kadar nadir olduklarına göre ağırlıklandırır. Bir belgede sıkça, ancak külliyatta nadiren görünen kelimeler (yani belgeye özgü olanlar) daha yüksek puanlar alır ve bu da onların önemini gösterir. Bu, bağlama özgü anahtar kelimelerin belirlenmesine yardımcı olur.
*   **N-gramlar:** Tek kelimeler (unigramlar) yerine, N-gramlar N kelimelik dizileri (örn. iki kelimelik bigramlar, üç kelimelik trigramlar) dikkate alır. Bu, "hesap doğrulama" veya "acil işlem gerekli" gibi bazı sınırlı bağlamsal bilgileri yakalar.
*   **Kelime Gömülüleri (Word Embeddings):** Kelimeleri sürekli bir vektör uzayında yoğun vektörler olarak temsil eden, benzer anlamlara sahip kelimelerin birbirine daha yakın konumlandığı daha sofistike teknikler.
    *   **Word2Vec (Skip-gram, CBOW):** Hedef kelimeden bağlamı veya tersini tahmin ederek kelime gömülülerini öğrenir.
    *   **GloVe (Global Vectors for Word Representation):** Global matris çarpanlara ayırma ve yerel bağlam penceresi yöntemlerini birleştirir.
    *   **FastText:** Kelimeleri karakter n-gramlarının toplamı olarak temsil ederek Word2Vec'i genişletir, böylece sözlük dışı kelimeleri ve morfolojik olarak zengin dilleri işleyebilir.
*   **Bağlamsal Gömülüler (Derin Öğrenme tabanlı):** Bu modeller, *bağlama bağlı* kelime gömülüleri oluşturur; yani aynı kelime, bir cümledeki çevreleyen kelimelere göre farklı gömülülere sahip olabilir.
    *   **BERT (Bidirectional Encoder Representations from Transformers):** Büyük bir külliyatta önceden eğitilmiş, kelimelerin çift yönlü bağlamını anlayabilen güçlü bir transformatör tabanlı model. Oltalama veri kümelerinde BERT'i ince ayar yapmak, son derece doğru özellik gösterimleri sağlayabilir.
    *   **RoBERTa, XLNet, ELECTRA, GPT:** Çeşitli metin külliyatlarında büyük ön eğitimden yararlanan benzer veya geliştirilmiş yetenekler sunan diğer transformatör mimarileri.
*   **Dilsel Özellikler:** Kelime içeriğinin ötesinde, belirli dilsel özellikler öznitelik olarak hizmet edebilir:
    *   **Duygu Analizi:** Bir mesajdaki duygusal tonu (örn. korku, aciliyet) tespit etme.
    *   **Okunabilirlik Puanları:** Dilin karmaşıklığını değerlendirme (örn. Flesch-Kincaid okuma kolaylığı seviyesi).
    *   **Stilometrik Özellikler:** Ortalama cümle uzunluğu, noktalama işareti sıklığı, karakter dağılımı ve benzersiz kelime sayısı gibi yazı stilini analiz etme, bu da oltalama girişimlerinde önemli ölçüde farklılık gösterebilir.

### 3.3. Makine Öğrenimi ve Derin Öğrenme Modelleri

Sayısal özellikler çıkarıldıktan sonra, meşru iletişimleri oltalama girişimlerinden ayırmak için çeşitli sınıflandırma modelleri kullanılabilir.

*   **Geleneksel Makine Öğrenimi Modelleri:**
    *   **Naive Bayes:** Özellikler arasında bağımsızlık varsayımına dayanan, Bayes teoremini temel alan olasılıksal bir sınıflandırıcıdır. Yüksek boyutlu verilerle basitliği ve iyi performansı nedeniyle metin sınıflandırması için genellikle etkilidir.
    *   **Destek Vektör Makineleri (SVM):** Veri noktalarını farklı sınıflara ayırmak için optimal bir hiper düzlem bulan güçlü bir algoritmadır. SVM'ler, yüksek boyutlu uzaylarda ve net ayırma marjlarıyla etkilidir.
    *   **Rastgele Ormanlar (Random Forests):** Eğitim sırasında çok sayıda karar ağacı oluşturan ve bireysel ağaçların sınıflarının modu (sınıflandırma) veya ortalama tahmini (regresyon) olan sınıfı çıktı veren bir topluluk öğrenme yöntemidir.
    *   **Lojistik Regresyon:** İkili sınıflandırma için doğrusal bir modeldir, olasılık tahminleri sağlamak için uygundur.
*   **Derin Öğrenme Modelleri:** Bu modeller, kelime gömülüleri gibi zengin, yüksek boyutlu özelliklerle özellikle etkilidir ve karmaşık kalıpları otomatik olarak öğrenebilir.
    *   **Tekrarlayan Sinir Ağları (RNN'ler):** Sıralı veriler için tasarlanmış RNN'ler (ve **Uzun Kısa Süreli Bellek (LSTM)** ve **Gated Recurrent Unit (GRU)** gibi varyantları), metni kelime kelime işleyebilir, sıradaki önceki kelimelerin hafızasını koruyabilir. Bu, bir cümle içindeki bağlamı ve bağımlılığı anlamak için çok önemlidir.
    *   **Evrişimsel Sinir Ağları (CNN'ler):** Görüntü işlemede popüler olmasına rağmen, CNN'ler metin sınıflandırması için de etkilidir. Oltalama olduğunu gösteren yerel kalıpları (örn. n-gramlar) tespit etmek için kelime dizileri (veya kelime gömülüleri) üzerinde filtreler uygularlar.
    *   **Transformatör Modelleri:** Kendine dikkat mekanizmalarıyla DİP'te devrim yarattılar, cümledeki farklı kelimelerin önemini mesafelerinden bağımsız olarak tartmalarına izin verdiler. **BERT** (özellik çıkarmada bahsedildiği gibi) gibi modeller de doğrudan sınıflandırıcı olarak ince ayarlanabilir ve derin bağlamsal gösterimler öğrenerek en son teknoloji performansı sunar.

Model seçimi genellikle veri kümesi boyutuna, hesaplama kaynaklarına ve yorumlanabilirlik ile doğruluk arasındaki istenen dengeye bağlıdır. Geleneksel ML'yi derin öğrenme özellikleriyle birleştiren hibrit yaklaşımlar da yaygındır.

## 4. Kod Örneği

Bu kısa Python kodu, temel bir oltalama tespiti iş akışını göstermektedir: metin ön işleme ve TF-IDF vektörleştirmesi, ardından basit bir Lojistik Regresyon sınıflandırıcısı.

```python
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Örnek veri kümesi: e-postalar (metin) ve etiketleri (0: meşru, 1: oltalama)
data = [
    ("Your account has been suspended. Click here to verify.", 1),
    ("Dear customer, your order #12345 has been shipped.", 0),
    ("URGENT: Your bank account needs immediate attention. Login now.", 1),
    ("Meeting reminder for tomorrow at 10 AM in conference room B.", 0),
    ("We detected unusual activity on your account. Please update your details.", 1),
    ("Regarding the project proposal, please review the attached document.", 0),
    ("ATTENTION: Your password has expired. Click this link to reset it.", 1),
    ("Hello, just checking in on your availability for next week.", 0),
]

texts = [item[0] for item in data]
labels = [item[1] for item in data]

# 1. Metin Ön İşleme Fonksiyonu
def preprocess_text(text):
    text = text.lower() # Küçük harfe çevirme
    text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text) # URL'leri <URL> belirteciyle değiştir
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Noktalama işaretlerini kaldır
    text = re.sub(r'\d+', '', text) # Sayıları kaldır
    text = text.strip() # Baştaki/sondaki boşlukları kaldır
    return text

# Ön işlemeyi uygula
preprocessed_texts = [preprocess_text(text) for text in texts]

# 2. TF-IDF kullanarak Öznitelik Çıkarımı
# TF-IDF Vektörleyiciyi başlat
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(preprocessed_texts)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 3. Model Eğitimi (Lojistik Regresyon)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 4. Model Değerlendirmesi
y_pred = model.predict(X_test)
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, zero_division=0))

# Örnek tahmin
new_email_phishing = "Action required: Your account has been compromised. Log in immediately."
new_email_legit = "Hello, following up on our discussion regarding the project timeline."

# Yeni e-postaları ön işle
preprocessed_phishing = preprocess_text(new_email_phishing)
preprocessed_legit = preprocess_text(new_email_legit)

# Yeni e-postaları vektörleştir
vectorized_phishing = tfidf_vectorizer.transform([preprocessed_phishing])
vectorized_legit = tfidf_vectorizer.transform([preprocessed_legit])

# Tahmin et
prediction_phishing = model.predict(vectorized_phishing)[0]
prediction_legit = model.predict(vectorized_legit)[0]

print(f"\n'{new_email_phishing}' -> Tahmin Edilen: {'Oltalama' if prediction_phishing == 1 else 'Meşru'}")
print(f"'{new_email_legit}' -> Tahmin Edilen: {'Oltalama' if prediction_legit == 1 else 'Meşru'}")

(Kod örneği bölümünün sonu)
```

## 5. Zorluklar ve Gelecek Yönelimler

Önemli ilerlemelere rağmen, DİP tabanlı oltalama tespiti bazı zorluklarla karşılaşmaktadır:

*   **Gelişen Taktikler ve Düşmanca Saldırılar:** Oltalamacılar, tespit sistemlerini atlatmak için dillerini, yeni eş anlamlıları, dilbilgisel yapıları ve sofistike sosyal mühendislik anlatılarını sürekli olarak uyarlamaktadır. Düşmanca saldırılar, kötü amaçlı metni insan tarafından okunabilir niyetini değiştirmeden modelleri kandırmak için ince bir şekilde değiştirebilir.
*   **Kavram Kayması (Concept Drift):** Oltalama e-postalarının özellikleri zamanla değişebilir, bu da eski modellerin etkinliğini azaltır. Bu **kavram kayması** ile mücadele etmek için sürekli izleme ve yeniden eğitim gereklidir.
*   **Çok Dilli Oltalama:** Saldırılar İngilizce ile sınırlı değildir. Çeşitli dillerdeki oltalama tespitini yapmak, çeşitli çok dilli veri kümeleri üzerinde eğitilmiş modellere ve sağlam diller arası DİP tekniklerine ihtiyaç duyar.
*   **Düşük Kaynaklı Diller:** Birçok dilde yeterli etiketli veri veya önceden eğitilmiş model bulunmadığından, bu diller için tespit zorlaşmaktadır.
*   **Bağlamsal Belirsizlik:** Meşru acil istekleri kötü amaçlı olanlardan ayırmak zor olabilir, özellikle meşru iletişimler benzer bir dil kullandığında (örn. şifre sıfırlama bildirimleri).
*   **Açıklanabilirlik (XAI):** Bir modelin neden bir e-postayı oltalama olarak işaretlediğini anlamak, güvenlik analistleri için çok önemlidir. Birçok gelişmiş derin öğrenme modeli bir kara kutudur, bu da güven ve etkili olay müdahalesi için açıklanabilir yapay zeka tekniklerini hayati hale getirir.

Oltalama tespiti için DİP'teki gelecek yönelimler şunları içerir:

*   **Gelişmiş Anlamsal Anlama:** Daha güçlü bağlamsal gömülü modelleri ve grafik sinir ağlarını kullanarak bir mesajdaki varlıklar ve kavramlar arasındaki ilişkileri daha iyi anlamak, ince tutarsızlıkları belirlemek.
*   **Diğer Modalitelerle Füzyon:** DİP'i URL analizi (alan adı yaşı, WHOIS kayıtları, itibar kontrolü), görüntü analizi (gömülü kötü amaçlı görüntüler için) ve davranışsal analiz (gönderen itibarı, geçmiş iletişim modelleri) gibi diğer tespit yöntemleriyle entegre etmek.
*   **Aktif Öğrenme ve Az Örnekli Öğrenme (Few-Shot Learning):** Sınırlı etiketli verilerden etkili bir şekilde öğrenme stratejileri, özellikle yeni ve ortaya çıkan oltalama varyantları için veri açıklama için gereken manuel çabayı azaltma.
*   **Uyarlanabilir Savunma için Pekiştirmeli Öğrenme:** Etkileşimlerden ve geri bildirimlerden öğrenebilen, gelişen tehditlere karşı tespit stratejilerini gerçek zamanlı olarak uyarlayan sistemler geliştirme.
*   **Etik Yapay Zeka ve Önyargı Azaltma:** Modellerin, eğitim verilerinde mevcut olan önyargıları istemeden sürdürmediğinden emin olmak, haksız veya yanlış sınıflandırmalara yol açmamak.

## 6. Sonuç

Doğal Dil İşleme, oltalama saldırılarına karşı devam eden mücadelede bir köşe taşıdır. Yapılandırılmamış metin verilerini eyleme geçirilebilir içgörülere dönüştürerek, DİP teknikleri kötü niyetli amacı gösteren dilsel anormalliklerin, anlamsal tutarsızlıkların ve sosyal mühendislik ipuçlarının belirlenmesini sağlar. Temel metin ön işleme ve TF-IDF ve gelişmiş kelime gömülüleri gibi sağlam özellik çıkarma yöntemlerinden, SVM'ler, LSTM'ler ve Transformatörler gibi sofistike makine öğrenimi ve derin öğrenme sınıflandırma modellerine kadar, DİP araçları sürekli genişlemektedir. Özellikle oltalama taktiklerinin dinamik doğası ve açıklanabilirlik ihtiyacı gibi zorluklar devam etse de, bu alandaki sürekli yenilik, daha dayanıklı ve uyarlanabilir tespit sistemleri vaat etmektedir. DİP'in diğer siber güvenlik istihbaratıyla entegrasyonu, bağlamsal anlama ve uyarlanabilir öğrenme gibi alanlardaki gelişmelerle birleştiğinde, dijital iletişimlerin sürekli oltalama tehdidine karşı önemli ölçüde daha güvenli olduğu bir geleceğin önünü açmaktadır.


