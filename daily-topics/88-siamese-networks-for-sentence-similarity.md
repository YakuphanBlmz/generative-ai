# Siamese Networks for Sentence Similarity

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Siamese Network Architecture for Sentence Similarity](#2-siamese-network-architecture-for-sentence-similarity)
- [3. Training Methodologies and Loss Functions](#3-training-methodologies-and-loss-functions)
- [4. Applications of Siamese Networks in Sentence Similarity](#4-applications-of-siamese-networks-in-sentence-similarity)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The ability to accurately determine the **semantic similarity** between two pieces of text, particularly sentences, is a fundamental and critical task in Natural Language Processing (NLP). This capability underpins a vast array of applications, from information retrieval and question answering to paraphrase detection and document clustering. Traditional approaches often relied on lexical overlap or statistical models like TF-IDF, which frequently fail to capture the nuanced meaning and contextual relationships inherent in human language.

**Siamese Networks** have emerged as a powerful paradigm for learning robust and meaningful representations of text, making them exceptionally well-suited for tasks involving similarity measurement. Unlike traditional classification networks that map inputs to distinct categories, Siamese Networks are designed to learn a similarity function that can gauge how related two inputs are. For sentence similarity, this involves mapping sentences into a high-dimensional embedding space where semantically similar sentences are positioned closely together, while dissimilar sentences are pushed apart. This document delves into the architecture, training methodologies, and diverse applications of Siamese Networks specifically tailored for discerning sentence similarity.

## 2. Siamese Network Architecture for Sentence Similarity
A Siamese Network, named after the conjoined twins due to its symmetrical structure, consists of two or more identical **subnetworks** (or "towers") that share the exact same weights and architecture. For sentence similarity, the typical setup involves two such identical subnetworks, each processing one sentence from a pair.

The core principle is as follows:
1.  **Input:** The network receives a pair of sentences, `Sentence A` and `Sentence B`.
2.  **Identical Subnetworks:** Each sentence is fed into one of the identical subnetworks. These subnetworks are often powerful neural architectures capable of encoding sequential data, such as Recurrent Neural Networks (RNNs – e.g., LSTMs, GRUs) or, more commonly in modern NLP, **Transformer-based models** (e.g., BERT, RoBERTa, XLNet, or specialized variants like Sentence-BERT). The critical aspect is that both subnetworks share all their parameters, ensuring that they learn the same transformation function.
3.  **Embedding Generation:** Each subnetwork processes its input sentence and produces a fixed-size **vector embedding** (also known as a sentence embedding or feature vector) that encapsulates the semantic meaning of that sentence. Let these be `u` for Sentence A and `v` for Sentence B.
4.  **Similarity Measurement:** The final layer of a Siamese Network typically does not perform a classification. Instead, the generated embeddings `u` and `v` are passed to a **distance metric** or similarity function. Common choices include:
    *   **Cosine Similarity:** Measures the cosine of the angle between two vectors. A value closer to 1 indicates higher similarity, 0 indicates orthogonality, and -1 indicates complete dissimilarity. This is particularly popular for high-dimensional embeddings.
    *   **Euclidean Distance:** Measures the straight-line distance between two points in Euclidean space. Smaller distances indicate higher similarity.
    *   **Manhattan Distance (L1 Distance):** Sum of absolute differences of their Cartesian coordinates.
    *   **Learned Similarity Function:** Sometimes, a small feed-forward neural network is placed on top of the absolute difference or concatenation of `u` and `v` to learn a more complex similarity mapping.

The shared weights are crucial because they force the network to learn an embedding space where the distance between embeddings directly corresponds to the semantic similarity of the original sentences. If the weights were not shared, each subnetwork might learn different representations for the same concepts, making direct comparison meaningless.

Modern implementations often leverage pre-trained Transformer models as the backbone for the subnetworks. These models, pre-trained on vast text corpora, already possess a strong understanding of language. Fine-tuning them within a Siamese framework allows them to specialize in producing embeddings optimized for similarity tasks, significantly enhancing performance compared to training from scratch.

## 3. Training Methodologies and Loss Functions
Training Siamese Networks for sentence similarity involves providing pairs or triplets of sentences and optimizing the network's weights such that similar sentences yield close embeddings, and dissimilar sentences yield distant embeddings. The choice of **loss function** is paramount in achieving this objective.

### 3.1. Contrastive Loss
**Contrastive Loss** is one of the foundational loss functions for Siamese Networks. It operates on pairs of inputs (sentence A, sentence B) along with a label indicating whether the pair is similar (positive pair, `y=1`) or dissimilar (negative pair, `y=0`).

The goal is to:
*   Minimize the distance between embeddings of positive pairs.
*   Maximize the distance between embeddings of negative pairs, up to a certain **margin**.

The contrastive loss function can be formulated as:
$L(u, v, y) = y \cdot D(u, v)^2 + (1 - y) \cdot \max(0, \text{margin} - D(u, v))^2$

Where:
*   `u` and `v` are the embeddings of Sentence A and Sentence B, respectively.
*   `y` is the label (1 for similar, 0 for dissimilar).
*   `D(u, v)` is the distance metric (e.g., Euclidean distance) between `u` and `v`.
*   `margin` is a hyperparameter that defines how far apart dissimilar pairs should be pushed. If `D(u, v)` for a negative pair is greater than the `margin`, no penalty is incurred, as the objective is already met.

### 3.2. Triplet Loss
**Triplet Loss** extends the idea of contrastive loss by operating on triplets of sentences: an **anchor (A)**, a **positive (P)** example (semantically similar to A), and a **negative (N)** example (semantically dissimilar to A).

The objective is to ensure that the anchor embedding is closer to the positive embedding than it is to the negative embedding by at least a specified margin.
Specifically, for embeddings `e_A`, `e_P`, and `e_N`:
$L(e_A, e_P, e_N) = \max(0, D(e_A, e_P) - D(e_A, e_N) + \text{margin})$

Where:
*   `D` is the distance metric (e.g., Euclidean distance).
*   `margin` is a hyperparameter ensuring a minimum separation between `(A, P)` and `(A, N)` distances.

Triplet loss often leads to better-separated embedding spaces but requires careful **triplet mining** strategies (e.g., hard negative mining) during training to select effective triplets that contribute meaningfully to the learning process, avoiding trivial triplets that are already well-separated.

### 3.3. Multiple Negatives Ranking Loss (MNRL)
Also known as **Softmax Loss for Semantic Textual Similarity (STS)** or NLLLoss, this approach is often used with cosine similarity and operates on a batch of sentences. For each sentence `a_i` in a batch, it's considered an anchor. Another sentence `b_i` in the batch is its positive pair. All other `b_j` (where `j != i`) in the batch are treated as negative examples for `a_i`.

The loss aims to maximize the similarity between `a_i` and `b_i` while minimizing similarity with all other `b_j` within the batch. This can be framed as a classification task where for each `a_i`, the model needs to identify `b_i` as its correct match among all `b`s in the batch.

The loss function is often defined using cross-entropy, treating the similarity scores between `a_i` and all `b`s as logits for a softmax probability distribution.
$L = -\sum_{i=1}^{N} \log \frac{e^{\text{sim}(a_i, b_i)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(a_i, b_j)/\tau}}$
Where `sim` is typically cosine similarity, `N` is the batch size, and `$\tau$` (tau) is a temperature parameter that scales the similarity scores.

This loss is highly effective for tasks like information retrieval and often yields state-of-the-art results when combined with pre-trained Transformer models.

### 3.4. Dataset Requirements
Training these models requires appropriately labeled datasets. For contrastive loss, pairs of sentences labeled as similar or dissimilar are needed. For triplet loss, triplets (anchor, positive, negative) are required. Publicly available datasets such as **SNLI (Stanford Natural Language Inference)**, **MultiNLI**, and **STS Benchmark** are frequently used for training and evaluating sentence similarity models, often adapted to create suitable pairs or triplets.

## 4. Applications of Siamese Networks in Sentence Similarity
The ability of Siamese Networks to learn meaningful sentence embeddings and quantify their similarity has led to their widespread adoption across various NLP applications.

### 4.1. Information Retrieval and Semantic Search
Perhaps one of the most prominent applications, Siamese Networks enhance information retrieval by moving beyond keyword matching. When a user queries (e.g., "how to fix a leaky faucet"), the system encodes the query into an embedding. It then compares this query embedding to a database of pre-computed embeddings for documents or passages. Documents with high semantic similarity to the query are retrieved, even if they don't share exact keywords. This enables more intelligent and context-aware search results.

### 4.2. Question Answering (QA)
In QA systems, particularly those relying on retrieval-based methods, Siamese Networks can be used to find the most relevant passage or document that contains the answer to a given question. The question is embedded, and its similarity to all potential answer candidates (e.g., sentences from a knowledge base) is calculated. The candidate with the highest similarity is then presented or used for further processing.

### 4.3. Duplicate Detection and Paraphrase Detection
Siamese Networks are highly effective at identifying duplicate content, such as duplicate bug reports, forum posts, or news articles, even if they are worded differently. For paraphrase detection, the network determines if two sentences express the same meaning, regardless of their surface-level textual differences. This is crucial for tasks like detecting plagiarism or summarizing redundant information.

### 4.4. Clustering and Topic Modeling
By generating dense vector representations for sentences, Siamese Networks enable effective clustering. Sentences with similar meanings will naturally group together in the embedding space. This can be used for automatic topic modeling, organizing large text collections, or identifying conversational threads in customer support logs.

### 4.5. Recommendation Systems
In content-based recommendation systems, Siamese Networks can recommend items (e.g., articles, products, movies) whose descriptions are semantically similar to items a user has previously engaged with or explicitly liked. By embedding both user preferences and item descriptions, the system can find optimal matches.

### 4.6. Natural Language Inference (NLI)
While NLI tasks (entailment, contradiction, neutral) are often framed as classification, the underlying mechanism of comparing two sentences (premise and hypothesis) for their relationship aligns well with the capabilities of Siamese-like architectures. Models trained for sentence similarity can provide strong base embeddings for NLI.

## 5. Code Example
This example demonstrates how to use the `sentence-transformers` library, which internally utilizes Siamese Networks built on pre-trained Transformer models, to generate sentence embeddings and compute their cosine similarity.

```python
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load a pre-trained Sentence Transformer model.
# This model is a type of Siamese Network where the encoder is a Transformer.
# 'all-MiniLM-L6-v2' is a lightweight yet powerful model.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define a list of sentences for embedding and similarity comparison
sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "The dog barked at the mailman.",
    "What is the weather like today?",
    "How is the current climate outside?"
]

print("Encoding sentences into embeddings...")
# 3. Encode the sentences into dense vector embeddings
# convert_to_tensor=True ensures the output is a PyTorch tensor
embeddings = model.encode(sentences, convert_to_tensor=True)

print(f"Embeddings shape: {embeddings.shape}") # Should be (num_sentences, embedding_dim)

# 4. Compute cosine similarity between all pairs of embeddings
# util.cos_sim returns a matrix where (i, j) is the similarity between sentence[i] and sentence[j]
cosine_scores = util.cos_sim(embeddings, embeddings)

print("\nPairwise Cosine Similarity Scores:")
# Print the similarity scores in a readable format
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)): # Only print unique pairs
        print(f"Sentence {i+1} ('{sentences[i]}') vs. Sentence {j+1} ('{sentences[j]}'):")
        print(f"  Similarity: {cosine_scores[i][j].item():.4f}\n")

# Expected output:
# - (Sentence 1, Sentence 2) should have high similarity (paraphrases)
# - (Sentence 4, Sentence 5) should have high similarity (similar meaning)
# - Other pairs should have lower similarity

(End of code example section)
```

## 6. Conclusion
Siamese Networks have revolutionized the field of sentence similarity by providing an elegant and effective framework for learning robust semantic representations. By employing identical subnetworks with shared weights, these models are capable of mapping sentences into a meaningful embedding space where proximity directly correlates with semantic relatedness. The adoption of advanced loss functions like Contrastive, Triplet, and Multiple Negatives Ranking Loss, coupled with the power of pre-trained Transformer models as subnetworks, has led to significant advancements in the accuracy and efficiency of similarity detection. From enhancing semantic search and question answering to enabling precise duplicate detection and intelligent content clustering, Siamese Networks are indispensable tools in the modern NLP toolkit, continually pushing the boundaries of how machines understand and process human language. Their versatility and capacity to generalize across various domains underscore their enduring importance in the evolving landscape of Generative AI and natural language understanding.

---
<br>

<a name="türkçe-içerik"></a>
## Cümle Benzerliği için Siyam Ağları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Cümle Benzerliği için Siyam Ağı Mimarisi](#2-cümle-benzerliği-için-siyam-ağı-mimarisi)
- [3. Eğitim Metodolojileri ve Kayıp Fonksiyonları](#3-eğitim-metodolojileri-ve-kayıp-fonksiyonları)
- [4. Cümle Benzerliğinde Siyam Ağlarının Uygulamaları](#4-cümle-benzerliğinde-siyam-ağlarının-uygulamaları)
- [5. Kod Örneği](#5-kod-Örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
İki metin parçası, özellikle de cümleler arasındaki **anlamsal benzerliği** doğru bir şekilde belirleme yeteneği, Doğal Dil İşleme (NLP) alanında temel ve kritik bir görevdir. Bu yetenek, bilgi erişiminden soru cevaplamaya, eşanlamlı cümle tespitinden belge kümelemeye kadar çok çeşitli uygulamaların temelini oluşturur. Geleneksel yaklaşımlar genellikle sözcüksel örtüşmeye veya TF-IDF gibi istatistiksel modellere dayanıyordu; ancak bunlar, insan dilinin doğasında bulunan incelikli anlamları ve bağlamsal ilişkileri yakalamada yetersiz kalıyordu.

**Siyam Ağları** (Siamese Networks), metinlerin sağlam ve anlamlı gösterimlerini öğrenmek için güçlü bir paradigma olarak ortaya çıkmış ve benzerlik ölçümü içeren görevler için son derece uygun hale gelmiştir. Girdileri farklı kategorilere eşleyen geleneksel sınıflandırma ağlarının aksine, Siyam Ağları, iki girdinin ne kadar ilişkili olduğunu ölçebilen bir benzerlik fonksiyonu öğrenmek üzere tasarlanmıştır. Cümle benzerliği için bu, cümleleri, anlamsal olarak benzer cümlelerin birbirine yakın konumlandırıldığı, benzer olmayan cümlelerin ise birbirinden uzaklaştırıldığı yüksek boyutlu bir gömme (embedding) uzayına eşlemeyi içerir. Bu belge, cümle benzerliğini ayırt etmek için özel olarak tasarlanmış Siyam Ağlarının mimarisini, eğitim metodolojilerini ve çeşitli uygulamalarını detaylandırmaktadır.

## 2. Cümle Benzerliği için Siyam Ağı Mimarisi
Simetrik yapısından dolayı bitişik ikizlere atıfta bulunularak adlandırılan bir Siyam Ağı, tamamen aynı ağırlıkları ve mimariyi paylaşan iki veya daha fazla özdeş **alt ağdan** (veya "kule"den) oluşur. Cümle benzerliği için tipik kurulum, her biri bir çiftin bir cümlesini işleyen iki özdeş alt ağı içerir.

Temel prensip şu şekildedir:
1.  **Girdi:** Ağ, bir çift cümle (`Cümle A` ve `Cümle B`) alır.
2.  **Özdeş Alt Ağlar:** Her cümle, özdeş alt ağlardan birine beslenir. Bu alt ağlar genellikle sıralı verileri kodlayabilen güçlü nöral mimarilerdir; örneğin Tekrarlayan Sinir Ağları (RNN'ler – örn. LSTM'ler, GRU'lar) veya modern NLP'de daha yaygın olarak **Transformer tabanlı modeller** (örn. BERT, RoBERTa, XLNet veya Sentence-BERT gibi özel varyantlar). Kritik nokta, her iki alt ağın da tüm parametrelerini paylaşmasıdır; bu, aynı dönüşüm fonksiyonunu öğrenmelerini sağlar.
3.  **Gömme Üretimi:** Her alt ağ, girdi cümlesini işler ve o cümlenin anlamsal anlamını kapsayan sabit boyutlu bir **vektör gömme** (cümle gömme veya özellik vektörü olarak da bilinir) üretir. Cümle A için bu `u`, Cümle B için ise `v` olsun.
4.  **Benzerlik Ölçümü:** Siyam Ağının son katmanı genellikle bir sınıflandırma gerçekleştirmez. Bunun yerine, üretilen gömmeler `u` ve `v`, bir **uzaklık metriği** veya benzerlik fonksiyonuna iletilir. Yaygın seçenekler şunlardır:
    *   **Kosinüs Benzerliği:** İki vektör arasındaki açının kosinüsünü ölçer. 1'e yakın bir değer daha yüksek benzerliği, 0 dikeyselliği ve -1 tam tersi yönde farklılığı gösterir. Özellikle yüksek boyutlu gömmeler için popülerdir.
    *   **Öklid Uzaklığı:** Öklid uzayında iki nokta arasındaki düz çizgi mesafesini ölçer. Daha küçük mesafeler daha yüksek benzerliği gösterir.
    *   **Manhattan Uzaklığı (L1 Uzaklığı):** Kartezyen koordinatlarının mutlak farklarının toplamı.
    *   **Öğrenilen Benzerlik Fonksiyonu:** Bazen, daha karmaşık bir benzerlik eşlemesi öğrenmek için `u` ve `v`'nin mutlak farkının veya birleştirmesinin (concatenation) üzerine küçük bir ileri beslemeli sinir ağı yerleştirilir.

Paylaşılan ağırlıklar çok önemlidir, çünkü bunlar ağı, gömmeler arasındaki uzaklığın orijinal cümlelerin anlamsal benzerliğiyle doğrudan ilişkili olduğu bir gömme uzayı öğrenmeye zorlar. Ağırlıklar paylaşılmasaydı, her alt ağ aynı kavramlar için farklı gösterimler öğrenebilir ve bu da doğrudan karşılaştırmayı anlamsız kılabilirdi.

Modern uygulamalar, alt ağların omurgası olarak genellikle önceden eğitilmiş Transformer modellerini kullanır. Bu modeller, büyük metin korpusları üzerinde önceden eğitilerek dil hakkında güçlü bir anlayışa sahiptir. Onları bir Siyam çerçevesinde ince ayar yapmak, benzerlik görevleri için optimize edilmiş gömmeler üretmede uzmanlaşmalarını sağlayarak, sıfırdan eğitime kıyasla performansı önemli ölçüde artırır.

## 3. Eğitim Metodolojileri ve Kayıp Fonksiyonları
Cümle benzerliği için Siyam Ağlarını eğitmek, cümle çiftlerini veya üçlülerini sağlamayı ve ağın ağırlıklarını, benzer cümlelerin yakın gömmeler, benzer olmayan cümlelerin ise uzak gömmeler üretmesini sağlayacak şekilde optimize etmeyi içerir. **Kayıp fonksiyonunun** seçimi bu hedefe ulaşmada çok önemlidir.

### 3.1. Kontrastif Kayıp (Contrastive Loss)
**Kontrastif Kayıp**, Siyam Ağları için temel kayıp fonksiyonlarından biridir. Çift girdiler (cümle A, cümle B) ve çiftin benzer olup olmadığını (pozitif çift, `y=1`) veya farklı olup olmadığını (negatif çift, `y=0`) belirten bir etiket üzerinde çalışır.

Amaç şunları sağlamaktır:
*   Pozitif çiftlerin gömmeleri arasındaki mesafeyi en aza indirmek.
*   Negatif çiftlerin gömmeleri arasındaki mesafeyi belirli bir **marjına** kadar en üst düzeye çıkarmak.

Kontrastif kayıp fonksiyonu şu şekilde formüle edilebilir:
$L(u, v, y) = y \cdot D(u, v)^2 + (1 - y) \cdot \max(0, \text{marjin} - D(u, v))^2$

Burada:
*   `u` ve `v` sırasıyla Cümle A ve Cümle B'nin gömmeleridir.
*   `y` etikettir (1 benzer, 0 farklı için).
*   `D(u, v)`, `u` ve `v` arasındaki uzaklık metriğidir (örn. Öklid uzaklığı).
*   `marjin`, farklı çiftlerin ne kadar uzağa itilmesi gerektiğini tanımlayan bir hiperparametredir. Negatif bir çift için `D(u, v)` `marjinden` büyükse, amaç zaten karşılandığı için ceza uygulanmaz.

### 3.2. Üçlü Kayıp (Triplet Loss)
**Üçlü Kayıp**, kontrastif kaybın fikrini cümle üçlüleri üzerinde çalışarak genişletir: bir **çapa (Anchor - A)**, bir **pozitif (Positive - P)** örnek (A ile anlamsal olarak benzer) ve bir **negatif (Negative - N)** örnek (A ile anlamsal olarak farklı).

Amaç, çapa gömmenin, negatif gömmeden en az belirli bir marjin kadar pozitif gömmeye daha yakın olmasını sağlamaktır.
Özellikle, `e_A`, `e_P` ve `e_N` gömmeleri için:
$L(e_A, e_P, e_N) = \max(0, D(e_A, e_P) - D(e_A, e_N) + \text{marjin})$

Burada:
*   `D` uzaklık metriğidir (örn. Öklid uzaklığı).
*   `marjin`, `(A, P)` ve `(A, N)` uzaklıkları arasında minimum bir ayrım sağlayan bir hiperparametredir.

Üçlü kayıp genellikle daha iyi ayrılmış gömme uzaylarına yol açar ancak öğrenme sürecine anlamlı katkıda bulunan etkili üçlüleri seçmek ve zaten iyi ayrılmış önemsiz üçlülerden kaçınmak için eğitim sırasında dikkatli **üçlü madenciliği** stratejileri (örn. hard negative mining) gerektirir.

### 3.3. Çoklu Negatifler Sıralama Kaybı (Multiple Negatives Ranking Loss - MNRL)
**Anlamsal Metinsel Benzerlik (STS) için Softmax Kaybı** veya NLLLoss olarak da bilinen bu yaklaşım, genellikle kosinüs benzerliği ile kullanılır ve bir cümle grubuna (batch) uygulanır. Bir gruptaki her `a_i` cümlesi bir çapa olarak kabul edilir. Gruptaki başka bir `b_i` cümlesi, onun pozitif çiftidir. Gruptaki diğer tüm `b_j`'ler (`j != i`), `a_i` için negatif örnekler olarak ele alınır.

Kayıp, `a_i` ve `b_i` arasındaki benzerliği maksimize ederken, gruptaki diğer tüm `b_j`'ler ile olan benzerliği minimize etmeyi amaçlar. Bu, her `a_i` için modelin gruptaki tüm `b`'ler arasında `b_i`'yi doğru eşleşme olarak tanımlaması gereken bir sınıflandırma görevi olarak çerçevelenebilir.

Kayıp fonksiyonu genellikle çapraz entropi kullanılarak tanımlanır ve `a_i` ile tüm `b`'ler arasındaki benzerlik skorları bir softmax olasılık dağılımı için logitler olarak ele alınır.
$L = -\sum_{i=1}^{N} \log \frac{e^{\text{sim}(a_i, b_i)/\tau}}{\sum_{j=1}^{N} e^{\text{sim}(a_i, b_j)/\tau}}$
Burada `sim` genellikle kosinüs benzerliğidir, `N` batch boyutudur ve `$\tau$` (tau) benzerlik skorlarını ölçekleyen bir sıcaklık parametresidir.

Bu kayıp, bilgi erişimi gibi görevler için son derece etkilidir ve önceden eğitilmiş Transformer modelleriyle birleştirildiğinde genellikle en güncel sonuçları verir.

### 3.4. Veri Kümesi Gereksinimleri
Bu modelleri eğitmek için uygun şekilde etiketlenmiş veri kümeleri gerekir. Kontrastif kayıp için benzer veya farklı olarak etiketlenmiş cümle çiftleri gereklidir. Üçlü kayıp için üçlüler (çapa, pozitif, negatif) gereklidir. **SNLI (Stanford Doğal Dil Çıkarımı)**, **MultiNLI** ve **STS Benchmark** gibi genel kullanıma açık veri kümeleri, cümle benzerliği modellerini eğitmek ve değerlendirmek için sıkça kullanılır ve uygun çiftler veya üçlüler oluşturmak üzere genellikle adapte edilir.

## 4. Cümle Benzerliğinde Siyam Ağlarının Uygulamaları
Siyam Ağlarının anlamlı cümle gömmeleri öğrenme ve bunların benzerliğini nicelendirme yeteneği, çeşitli NLP uygulamalarında yaygın olarak benimsenmelerine yol açmıştır.

### 4.1. Bilgi Erişimi ve Anlamsal Arama
Belki de en önemli uygulamalardan biri olan Siyam Ağları, anahtar kelime eşleştirmesinin ötesine geçerek bilgi erişimini geliştirir. Bir kullanıcı sorgu yaptığında (örn. "sızdıran bir musluk nasıl tamir edilir"), sistem sorguyu bir gömmeye kodlar. Ardından bu sorgu gömmesini, belgeler veya pasajlar için önceden hesaplanmış gömmeler veritabanıyla karşılaştırır. Sorguyla yüksek anlamsal benzerliğe sahip belgeler, tam anahtar kelimeleri paylaşmasalar bile alınır. Bu, daha akıllı ve bağlama duyarlı arama sonuçları sağlar.

### 4.2. Soru Cevaplama (QA)
Soru Cevaplama sistemlerinde, özellikle erişim tabanlı yöntemlere dayananlarda, Siyam Ağları verilen bir sorunun cevabını içeren en alakalı pasajı veya belgeyi bulmak için kullanılabilir. Soru gömülür ve tüm potansiyel cevap adaylarıyla (örn. bir bilgi tabanından cümleler) benzerliği hesaplanır. En yüksek benzerliğe sahip aday daha sonra sunulur veya daha fazla işlem için kullanılır.

### 4.3. Tekrar Tespiti ve Eşanlamlı Cümle Tespiti
Siyam Ağları, farklı şekilde ifade edilmiş olsalar bile, yinelenen hata raporları, forum gönderileri veya haber makaleleri gibi yinelenen içeriği tespit etmede oldukça etkilidir. Eşanlamlı cümle tespiti için ağ, yüzey seviyesi metinsel farklılıklarına bakılmaksızın iki cümlenin aynı anlamı ifade edip etmediğini belirler. Bu, intihal tespiti veya gereksiz bilgileri özetleme gibi görevler için çok önemlidir.

### 4.4. Kümeleme ve Konu Modelleme
Cümleler için yoğun vektör temsilleri oluşturarak, Siyam Ağları etkili kümelemeyi mümkün kılar. Benzer anlama sahip cümleler, gömme uzayında doğal olarak bir araya gelir. Bu, otomatik konu modelleme, büyük metin koleksiyonlarını düzenleme veya müşteri destek kayıtlarındaki konuşma dizilerini belirleme için kullanılabilir.

### 4.5. Tavsiye Sistemleri
İçerik tabanlı tavsiye sistemlerinde, Siyam Ağları, açıklamaları bir kullanıcının daha önce etkileşimde bulunduğu veya açıkça beğendiği öğelere anlamsal olarak benzer öğeleri (örn. makaleler, ürünler, filmler) önerebilir. Hem kullanıcı tercihlerini hem de öğe açıklamalarını gömerek sistem, optimal eşleşmeleri bulabilir.

### 4.6. Doğal Dil Çıkarımı (NLI)
NLI görevleri (çıkarım, çelişki, nötr) genellikle sınıflandırma olarak çerçevelenirken, iki cümlenin (öncül ve hipotez) ilişkilerini karşılaştıran temel mekanizma, Siyam benzeri mimarilerin yetenekleriyle iyi uyum sağlar. Cümle benzerliği için eğitilmiş modeller, NLI için güçlü temel gömmeler sağlayabilir.

## 5. Kod Örneği
Bu örnek, önceden eğitilmiş Transformer modelleri üzerine kurulu Siyam Ağlarını dahili olarak kullanan `sentence-transformers` kütüphanesini kullanarak cümle gömmeleri oluşturmayı ve kosinüs benzerliğini hesaplamayı göstermektedir.

```python
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Önceden eğitilmiş bir Sentence Transformer modelini yükleyin.
# Bu model, kodlayıcısı bir Transformer olan bir Siyam Ağı türüdür.
# 'all-MiniLM-L6-v2' hafif ama güçlü bir modeldir.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Gömme ve benzerlik karşılaştırması için bir cümle listesi tanımlayın
sentences = [
    "Kedi, paspasın üzerinde oturdu.",
    "Bir kedi, halının üzerinde dinleniyordu.",
    "Köpek postacıya havladı.",
    "Bugün hava nasıl?",
    "Dışarıdaki mevcut iklim nasıl?"
]

print("Cümleler gömmelere kodlanıyor...")
# 3. Cümleleri yoğun vektör gömmelerine kodlayın
# convert_to_tensor=True çıktının bir PyTorch tensörü olmasını sağlar
embeddings = model.encode(sentences, convert_to_tensor=True)

print(f"Gömme boyutu: {embeddings.shape}") # (cümle_sayısı, gömme_boyutu) olmalı

# 4. Tüm gömme çiftleri arasındaki kosinüs benzerliğini hesaplayın
# util.cos_sim, (i, j) değerinin cümle[i] ve cümle[j] arasındaki benzerlik olduğu bir matris döndürür
cosine_scores = util.cos_sim(embeddings, embeddings)

print("\nÇiftler Arası Kosinüs Benzerlik Skorları:")
# Benzerlik skorlarını okunabilir bir formatta yazdırın
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)): # Sadece benzersiz çiftleri yazdır
        print(f"Cümle {i+1} ('{sentences[i]}') vs. Cümle {j+1} ('{sentences[j]}'):")
        print(f"  Benzerlik: {cosine_scores[i][j].item():.4f}\n")

# Beklenen çıktı:
# - (Cümle 1, Cümle 2) yüksek benzerliğe sahip olmalı (eşanlamlılar)
# - (Cümle 4, Cümle 5) yüksek benzerliğe sahip olmalı (benzer anlam)
# - Diğer çiftler daha düşük benzerliğe sahip olmalı

(Kod örneği bölümünün sonu)
```

## 6. Sonuç
Siyam Ağları, sağlam anlamsal temsilleri öğrenmek için zarif ve etkili bir çerçeve sağlayarak cümle benzerliği alanında devrim yaratmıştır. Paylaşılan ağırlıklara sahip özdeş alt ağlar kullanarak, bu modeller cümleleri, yakınlığın anlamsal ilişkiyle doğrudan ilişkili olduğu anlamlı bir gömme uzayına eşleyebilir. Kontrastif, Üçlü ve Çoklu Negatifler Sıralama Kaybı gibi gelişmiş kayıp fonksiyonlarının benimsenmesi, alt ağlar olarak önceden eğitilmiş Transformer modellerinin gücüyle birleştiğinde, benzerlik tespitinin doğruluğunda ve verimliliğinde önemli ilerlemelere yol açmıştır. Anlamsal aramayı ve soru cevaplamayı geliştirmekten hassas tekrar tespitini ve akıllı içerik kümelemeyi sağlamaya kadar, Siyam Ağları modern NLP araç setinde vazgeçilmez araçlardır ve Üretken Yapay Zeka ve doğal dil anlama alanının sürekli gelişen manzarasında insan dilinin makineler tarafından anlaşılma ve işlenme sınırlarını sürekli olarak zorlamaktadır. Uygulamalardaki çok yönlülükleri ve çeşitli alanlara genellenebilme kapasiteleri, onların değişen önemini vurgulamaktadır.