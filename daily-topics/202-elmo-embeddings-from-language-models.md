# ELMo: Embeddings from Language Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Problem with Static Word Embeddings](#2-the-problem-with-static-word-embeddings)
- [3. ELMo Architecture and Principles](#3-elmo-architecture-and-principles)
    - [3.1. Bidirectional Language Model (BiLM)](#31-bidirectional-language-model-bilm)
    - [3.2. Contextual Embeddings](#32-contextual-embeddings)
    - [3.3. Weighted Layer Combination](#33-weighted-layer-combination)
    - [3.4. Pre-training and Fine-tuning](#34-pre-training-and-fine-tuning)
- [4. Advantages and Impact](#4-advantages-and-impact)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)
- [7. References](#7-references)

<a name="1-introduction"></a>
## 1. Introduction
The field of Natural Language Processing (NLP) has undergone a profound transformation with the advent of sophisticated techniques for representing words and phrases numerically. Prior to the **transformer architecture** and models like BERT, **ELMo (Embeddings from Language Models)**, introduced by Peters et al. in 2018, marked a pivotal shift from static word embeddings to **contextualized word representations**. ELMo demonstrated that deep, pre-trained language models could effectively capture complex characteristics of word use, including **syntax** and **semantics**, across diverse linguistic contexts. This innovation significantly improved performance on a wide array of downstream NLP tasks, laying foundational groundwork for the subsequent wave of **transformer-based models**. ELMo's core contribution was its ability to generate embeddings that are a function of the entire input sentence, thereby resolving the long-standing ambiguity issue inherent in words with multiple meanings.

<a name="2-the-problem-with-static-word-embeddings"></a>
## 2. The Problem with Static Word Embeddings
Before ELMo, popular word embedding techniques such as **Word2Vec** (Mikolov et al., 2013) and **GloVe** (Pennington et al., 2014) produced a single, fixed vector representation for each word in a vocabulary. While these static embeddings represented a significant advancement over one-hot encoding by capturing semantic relationships (e.g., "king" - "man" + "woman" = "queen"), they suffered from a critical limitation: **polysemy**. Many words in natural language have multiple meanings depending on the surrounding context. For instance, the word "bank" can refer to a financial institution or the side of a river. A static embedding model would assign the exact same vector to "bank" in both "I went to the bank to deposit money" and "The boat docked at the river bank." This inability to differentiate between context-dependent meanings limited the expressive power of NLP models and often required complex workarounds or task-specific feature engineering. ELMo directly addressed this fundamental challenge by creating dynamic, context-sensitive embeddings.

<a name="3-elmo-architecture-and-principles"></a>
## 3. ELMo Architecture and Principles
ELMo's power stems from its innovative architecture, which leverages a deep **Bidirectional Language Model (BiLM)** to generate embeddings that are dynamically adjusted based on the input text.

<a name="31-bidirectional-language-model-bilm"></a>
### 3.1. Bidirectional Language Model (BiLM)
The core of ELMo is a **character-based convolutional neural network (CNN)** followed by a multi-layer, bidirectional Long Short-Term Memory (BiLSTM) network. Instead of processing words as atomic units, ELMo first processes characters, allowing it to handle out-of-vocabulary (OOV) words and morphological variations more effectively. The BiLM is composed of two independent LSTMs:
*   A **forward LSTM** that models the probability of a word given its preceding context ($P(w_k | w_1, ..., w_{k-1})$).
*   A **backward LSTM** that models the probability of a word given its succeeding context ($P(w_k | w_{k+1}, ..., w_n)$).

These two LSTMs are trained jointly to maximize the log likelihood of predicting the next word in the forward pass and the previous word in the backward pass. The internal states of these LSTMs capture rich contextual information about each word.

<a name="32-contextual Embeddings"></a>
### 3.2. Contextual Embeddings
Unlike static embeddings, ELMo's representations for a word $w_k$ are not fixed. Instead, they are a function of the entire sentence $S = (w_1, ..., w_n)$ in which $w_k$ appears. For each word, ELMo extracts a series of **contextualized representations** from the internal layers of its BiLM. Specifically, for each token, ELMo computes representations from:
1.  The **character-level convolutional layer**, capturing sub-word information.
2.  Each layer of the **forward LSTM**.
3.  Each layer of the **backward LSTM**.

These representations are distinct for each layer, capturing different levels of linguistic information. Lower layers tend to model **syntactic features** (e.g., part-of-speech, constituency), while higher layers tend to capture **semantic information** (e.g., word sense, discourse context).

<a name="33-weighted Layer Combination"></a>
### 3.3. Weighted Layer Combination
To derive a single, consolidated ELMo embedding for a word, the representations from different layers are combined. This combination is not a simple concatenation but a **weighted sum** of the layer-specific representations. Crucially, these weights are **learnable task-specific parameters**. This means that for different downstream NLP tasks (e.g., sentiment analysis, named entity recognition), the model can learn to emphasize different layers of the BiLM, effectively selecting the most relevant type of linguistic information for that specific task. A scalar mixing parameter $\gamma$ is also learned, allowing the model to scale the ELMo representation relative to the original task-specific features. This flexibility is a key strength of ELMo, enabling it to adapt to various NLP challenges with superior performance.

<a name="34-pre-training-and-fine-tuning"></a>
### 3.4. Pre-training and Fine-tuning
ELMo follows a **two-phase training strategy**:
1.  **Pre-training:** The deep BiLM is trained on a very large text corpus (e.g., 1B Word Benchmark) using the language modeling objective. This phase is computationally intensive but results in a highly generalizable model that has learned intricate patterns of language use.
2.  **Fine-tuning:** Once pre-trained, the ELMo model is integrated into a task-specific NLP architecture. The pre-trained weights of the BiLM layers remain fixed, but the task-specific weights (for the weighted sum) are trained along with the rest of the task model. This allows the ELMo embeddings to be finely tuned to the specific needs of the target task, extracting the most relevant contextual information.

<a name="4-advantages-and-impact"></a>
## 4. Advantages and Impact
ELMo introduced several significant advantages over previous word embedding methods:

*   **Contextualization:** The ability to generate context-dependent word representations was a game-changer, effectively resolving the issue of polysemy and improving the understanding of linguistic nuances.
*   **Deep Representations:** By leveraging multiple layers of a deep neural network, ELMo captures a rich hierarchy of linguistic features, from morphology and syntax in lower layers to semantics and discourse in higher layers.
*   **Transfer Learning:** ELMo was one of the early and highly successful demonstrations of **transfer learning** in NLP, where a model pre-trained on a massive unlabeled corpus could be effectively fine-tuned for various supervised downstream tasks with limited labeled data.
*   **Performance Boost:** ELMo significantly improved the state-of-the-art across a wide range of NLP benchmarks, including question answering, natural language inference, semantic role labeling, and named entity recognition. Its impact spurred further research into large-scale pre-trained language models.

ELMo's innovative approach paved the way for more complex and powerful contextual embedding models like BERT, GPT, and their successors. It solidified the paradigm of **pre-train and fine-tune** as a dominant force in modern NLP.

<a name="5-code-example"></a>
## 5. Code Example
This Python snippet demonstrates how to use the pre-trained ELMo model from the `allennlp` library to generate contextual embeddings for a sentence.

```python
import allennlp
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import os

# --- Configuration for ELMo model ---
# These are links to the pre-trained ELMo model's options and weights files.
# In a production environment, you would typically download these files once
# to avoid repeated downloads and potential network issues.
# For this example, we assume they are accessible.
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

# --- Initialize ELMo model ---
# num_output_representations=1 means we want one combined representation from the layers.
# dropout=0 for demonstration purposes, typically >0 during training.
print("Initializing ELMo model...")
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
print("ELMo model initialized.")

# --- Prepare input sentences ---
# ELMo takes a list of tokenized sentences.
# Example: "I went to the bank to deposit money."
# Example: "The boat docked at the river bank."
sentences = [
    ["I", "went", "to", "the", "bank", "to", "deposit", "money", "."],
    ["The", "boat", "docked", "at", "the", "river", "bank", "."]
]
print(f"\nInput sentences: {sentences}")

# --- Convert words to character IDs ---
# ELMo works with character-level inputs, so we need to convert word tokens
# into a tensor of character IDs.
character_ids = batch_to_ids(sentences)
print(f"Shape of character IDs tensor: {character_ids.shape}") # Expected: (batch_size, max_seq_len, max_word_len)

# --- Generate ELMo embeddings ---
# The forward pass through the ELMo model.
# The output `elmo_embeddings` is a dictionary.
# `elmo_embeddings['elmo_representations']` is a list of tensors,
# where each tensor corresponds to an output representation requested by num_output_representations.
# Each tensor has shape (batch_size, sequence_length, embedding_dim).
print("Generating ELMo embeddings...")
elmo_embeddings = elmo(character_ids)
print("ELMo embeddings generated.")

# --- Access and print embedding details ---
# Access the combined embeddings for the first sentence.
# [0] refers to the first output representation (since num_output_representations=1).
# [0] refers to the first sentence in the batch.
first_sentence_embeddings = elmo_embeddings['elmo_representations'][0][0] # Shape: (seq_len, embedding_dim)

print(f"\nELMo embeddings shape for the first sentence ('I went to the bank to deposit money.'): {first_sentence_embeddings.shape}")
print(f"Embedding dimension: {first_sentence_embeddings.shape[1]}") # ELMo typically has 1024 dimensions.

# Print partial embeddings for the word "bank" in the first sentence (index 4)
# and in the second sentence (index 6) to illustrate contextualization.
# detach().cpu().numpy() converts the tensor to a numpy array for easy printing.
bank_embedding_sentence1 = first_sentence_embeddings[4][:5].detach().cpu().numpy()
print(f"\nPartial ELMo embedding for 'bank' in sentence 1 (financial context): {bank_embedding_sentence1}")

# Access embeddings for the second sentence
second_sentence_embeddings = elmo_embeddings['elmo_representations'][0][1]
bank_embedding_sentence2 = second_sentence_embeddings[6][:5].detach().cpu().numpy()
print(f"Partial ELMo embedding for 'bank' in sentence 2 (river context): {bank_embedding_sentence2}")

# You would typically feed these embeddings into another neural network
# for a specific downstream NLP task (e.g., classification, sequence tagging).
print("\nNote: The full 1024-dimensional embeddings for 'bank' in the two sentences would be different,")
print("demonstrating ELMo's contextual awareness.")

# Clean up temporary cached files if any (optional, depends on allennlp version/config)
# for file_path in [options_file, weight_file]:
#     if file_path.startswith("http") and os.path.exists(os.path.basename(file_path)):
#         os.remove(os.path.basename(file_path))

(End of code example section)
```
<a name="6-conclusion"></a>
## 6. Conclusion
ELMo represented a landmark achievement in NLP, fundamentally changing how word embeddings were perceived and utilized. By moving from static to **contextualized representations**, ELMo effectively tackled the pervasive problem of polysemy, allowing models to grasp the nuanced meaning of words based on their surrounding text. Its architecture, employing a deep BiLM with a weighted layer combination, showcased the power of **deep learning** and **transfer learning** for language understanding. While newer transformer-based models have since pushed the boundaries even further, ELMo's contributions were instrumental in demonstrating the immense potential of large-scale pre-trained language models and paved the way for the modern era of highly performant NLP systems. It remains a critical piece of the historical and conceptual understanding of deep contextual embeddings.

<a name="7-references"></a>
## 7. References
*   Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). *Deep contextualized word representations*. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 2227-2237.
*   Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. Proceedings of International Conference on Learning Representations (ICLR), 2013.
*   Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532-1543.

---
<br>

<a name="türkçe-içerik"></a>
## ELMo: Dil Modellerinden Gömüler

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Statik Kelime Gömülerinin Sorunu](#2-statik-kelime-gömülerinin-sorunu)
- [3. ELMo Mimarisi ve İlkeleri](#3-elmo-mimarisi-ve-ilkeleri)
    - [3.1. Çift Yönlü Dil Modeli (BiLM)](#31-çift-yönlü-dil-modeli-bilm)
    - [3.2. Bağlamsal Gömüler](#32-bağlamsal-gömüler)
    - [3.3. Ağırlıklı Katman Kombinasyonu](#33-ağırlıklı-katman-kombinasyonu)
    - [3.4. Ön Eğitim ve İnce Ayar](#34-ön-eğitim-ve-ince-ayar)
- [4. Avantajları ve Etkisi](#4-avantajları-ve-etkisi)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)
- [7. Kaynaklar](#7-kaynaklar)

<a name="1-giriş"></a>
## 1. Giriş
Doğal Dil İşleme (NLP) alanı, kelimeleri ve ifadeleri sayısal olarak temsil etmek için gelişmiş tekniklerin ortaya çıkışıyla derin bir dönüşüm geçirdi. **Transformer mimarisi** ve BERT gibi modellerden önce, Peters ve arkadaşları tarafından 2018'de tanıtılan **ELMo (Embeddings from Language Models)**, statik kelime gömülerinden **bağlamsal kelime temsillerine** doğru önemli bir geçişe işaret etti. ELMo, derin, önceden eğitilmiş dil modellerinin, çeşitli dilbilimsel bağlamlarda **sözdizimi** ve **anlambilim** dahil olmak üzere kelime kullanımının karmaşık özelliklerini etkili bir şekilde yakalayabildiğini gösterdi. Bu yenilik, çok çeşitli aşağı akış NLP görevlerinde performansı önemli ölçüde artırdı ve sonraki **transformer tabanlı modeller** dalgası için temel bir zemin oluşturdu. ELMo'nun temel katkısı, tüm giriş cümlesinin bir fonksiyonu olan gömüler oluşturabilmesi ve böylece birden çok anlama sahip kelimelerde doğal olarak bulunan uzun süredir devam eden belirsizlik sorununu çözmesidir.

<a name="2-statik-kelime-gömülerinin-sorunu"></a>
## 2. Statik Kelime Gömülerinin Sorunu
ELMo'dan önce, **Word2Vec** (Mikolov ve arkadaşları, 2013) ve **GloVe** (Pennington ve arkadaşları, 2014) gibi popüler kelime gömme teknikleri, bir kelime dağarcığındaki her kelime için tek, sabit bir vektör temsili üretiyordu. Bu statik gömüler, tek-sıcak kodlamaya göre anlamsal ilişkileri (örneğin, "kral" - "adam" + "kadın" = "kraliçe") yakalayarak önemli bir ilerlemeyi temsil etse de, kritik bir sınırlamadan muzdaripti: **çok anlamlılık (polysemy)**. Doğal dildeki birçok kelime, çevredeki bağlama bağlı olarak birden çok anlama sahiptir. Örneğin, "banka" kelimesi bir finans kurumuna veya bir nehir kenarına atıfta bulunabilir. Statik bir gömme modeli, "Para yatırmak için bankaya gittim" ve "Tekne nehir bankasında demirledi" cümlelerinde "banka" kelimesine tamamen aynı vektörü atayacaktır. Bağlama bağlı anlamlar arasında ayrım yapma yeteneğinin bu eksikliği, NLP modellerinin ifade gücünü sınırlıyor ve genellikle karmaşık geçici çözümler veya göreve özel özellik mühendisliği gerektiriyordu. ELMo, dinamik, bağlama duyarlı gömüler oluşturarak bu temel sorunu doğrudan ele aldı.

<a name="3-elmo-mimarisi-ve-ilkeleri"></a>
## 3. ELMo Mimarisi ve İlkeleri
ELMo'nun gücü, giriş metnine göre dinamik olarak ayarlanan gömüler oluşturmak için derin bir **Çift Yönlü Dil Modeli (BiLM)** kullanan yenilikçi mimarisinden gelir.

<a name="31-çift-yönlü-dil-modeli-bilm"></a>
### 3.1. Çift Yönlü Dil Modeli (BiLM)
ELMo'nun çekirdeği, çok katmanlı, çift yönlü Uzun Kısa Süreli Bellek (BiLSTM) ağını takiben **karakter tabanlı bir evrişimli sinir ağı (CNN)**'dır. Kelimeleri atomik birimler olarak işlemek yerine, ELMo önce karakterleri işleyerek kelime dağarcığı dışı (OOV) kelimeleri ve morfolojik varyasyonları daha etkili bir şekilde ele almasını sağlar. BiLM, iki bağımsız LSTM'den oluşur:
*   Önceki bağlamı verilen bir kelimenin olasılığını modelleyen bir **ileri LSTM** ($P(w_k | w_1, ..., w_{k-1})$).
*   Sonraki bağlamı verilen bir kelimenin olasılığını modelleyen bir **geri LSTM** ($P(w_k | w_{k+1}, ..., w_n)$).

Bu iki LSTM, ileri geçişte bir sonraki kelimeyi ve geri geçişte bir önceki kelimeyi tahmin etmenin log olasılığını maksimize etmek için birlikte eğitilir. Bu LSTM'lerin iç durumları, her kelime hakkında zengin bağlamsal bilgileri yakalar.

<a name="32-bağlamsal-gömüler"></a>
### 3.2. Bağlamsal Gömüler
Statik gömülerin aksine, bir $w_k$ kelimesi için ELMo'nun temsilleri sabit değildir. Bunun yerine, $w_k$'nın göründüğü tüm $S = (w_1, ..., w_n)$ cümlesinin bir fonksiyonudur. Her kelime için ELMo, BiLM'sinin iç katmanlarından bir dizi **bağlamsal temsil** çıkarır. Özellikle, her token için ELMo şunlardan temsiller hesaplar:
1.  **Karakter seviyesi evrişimli katman**, alt kelime bilgisini yakalar.
2.  **İleri LSTM**'nin her katmanı.
3.  **Geri LSTM**'nin her katmanı.

Bu temsiller her katman için farklıdır ve farklı seviyelerde dilbilimsel bilgileri yakalar. Alt katmanlar **sözdizimsel özellikleri** (örneğin, sözcük türü, sözdizimsel bileşen) modelleme eğilimindeyken, daha yüksek katmanlar **anlamsal bilgileri** (örneğin, kelime anlamı, söylem bağlamı) yakalama eğilimindedir.

<a name="33-ağırlıklı-katman-kombinasyonu"></a>
### 3.3. Ağırlıklı Katman Kombinasyonu
Bir kelime için tek, birleştirilmiş bir ELMo gömüsü elde etmek için, farklı katmanlardan gelen temsiller birleştirilir. Bu kombinasyon basit bir birleştirme değil, katmana özgü temsillerin **ağırlıklı bir toplamıdır**. Kritik olarak, bu ağırlıklar **öğrenilebilir göreve özel parametrelerdir**. Bu, farklı aşağı akış NLP görevleri (örneğin, duygu analizi, adlandırılmış varlık tanıma) için, modelin BiLM'nin farklı katmanlarını vurgulamayı öğrenebileceği ve o belirli görev için en alakalı dilbilimsel bilgi türünü etkili bir şekilde seçebileceği anlamına gelir. Modelin, ELMo temsilini orijinal göreve özgü özelliklere göre ölçeklendirmesine izin veren bir skaler karıştırma parametresi $\gamma$ da öğrenilir. Bu esneklik, ELMo'nun çeşitli NLP zorluklarına üstün performansla uyum sağlamasını sağlayan temel bir gücüdür.

<a name="34-ön-eğitim-ve-ince-ayar"></a>
### 3.4. Ön Eğitim ve İnce Ayar
ELMo, **iki aşamalı bir eğitim stratejisi** izler:
1.  **Ön Eğitim:** Derin BiLM, dil modelleme amacı kullanılarak çok büyük bir metin kümesi (örneğin, 1 Milyar Kelime Karşılaştırması) üzerinde eğitilir. Bu aşama hesaplama açısından yoğun olsa da, dil kullanımının karmaşık kalıplarını öğrenmiş, yüksek düzeyde genellenebilir bir modelle sonuçlanır.
2.  **İnce Ayar:** Önceden eğitildikten sonra, ELMo modeli göreve özel bir NLP mimarisine entegre edilir. BiLM katmanlarının önceden eğitilmiş ağırlıkları sabit kalır, ancak göreve özel ağırlıklar (ağırlıklı toplam için) görev modelinin geri kalanıyla birlikte eğitilir. Bu, ELMo gömülerinin hedef görevin özel ihtiyaçlarına göre ince ayar yapılmasına ve en alakalı bağlamsal bilginin çıkarılmasına olanak tanır.

<a name="4-avantajları-ve-etkisi"></a>
## 4. Avantajları ve Etkisi
ELMo, önceki kelime gömme yöntemlerine göre birçok önemli avantaj sunmuştur:

*   **Bağlamsallaştırma:** Bağlama bağlı kelime temsilleri oluşturma yeteneği, oyunun kurallarını değiştirdi, çok anlamlılık sorununu etkili bir şekilde çözdü ve dilsel nüansların daha iyi anlaşılmasını sağladı.
*   **Derin Temsiller:** Derin bir sinir ağının birden çok katmanını kullanarak, ELMo, morfoloji ve sözdiziminden (alt katmanlarda) anlambilim ve söyleme (üst katmanlarda) kadar zengin bir dilbilimsel özellik hiyerarşisini yakalar.
*   **Transfer Öğrenimi:** ELMo, NLP'de **transfer öğreniminin** erken ve oldukça başarılı gösterimlerinden biriydi; büyük, etiketlenmemiş bir metin kümesi üzerinde önceden eğitilmiş bir model, sınırlı etiketli veri ile çeşitli denetimli aşağı akış görevleri için etkili bir şekilde ince ayar yapılabilirdi.
*   **Performans Artışı:** ELMo, soru yanıtlama, doğal dil çıkarımı, anlamsal rol etiketleme ve adlandırılmış varlık tanıma dahil olmak üzere çok çeşitli NLP karşılaştırmalarında son teknolojiyi önemli ölçüde geliştirdi. Etkisi, büyük ölçekli önceden eğitilmiş dil modelleri üzerine daha fazla araştırmayı teşvik etti.

ELMo'nun yenilikçi yaklaşımı, BERT, GPT ve ardılları gibi daha karmaşık ve güçlü bağlamsal gömme modellerinin önünü açtı. **Ön eğitim ve ince ayar** paradigmasını modern NLP'de baskın bir güç olarak sağlamlaştırdı.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği
Bu Python kodu parçacığı, bir cümle için bağlamsal gömüler oluşturmak üzere `allennlp` kütüphanesinden önceden eğitilmiş ELMo modelinin nasıl kullanılacağını gösterir.

```python
import allennlp
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import os

# --- ELMo modeli için yapılandırma ---
# Bunlar, önceden eğitilmiş ELMo modelinin seçenek ve ağırlık dosyalarına bağlantılardır.
# Bir üretim ortamında, tekrar eden indirmeleri ve olası ağ sorunlarını önlemek için
# bu dosyaları genellikle bir kez indirirsiniz.
# Bu örnek için, erişilebilir olduklarını varsayıyoruz.
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

# --- ELMo modelini başlat ---
# num_output_representations=1, katmanlardan bir birleşik temsil istediğimiz anlamına gelir.
# dropout=0 gösterim amaçlıdır, eğitim sırasında genellikle >0'dır.
print("ELMo modeli başlatılıyor...")
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
print("ELMo modeli başlatıldı.")

# --- Giriş cümlelerini hazırla ---
# ELMo, tokenlere ayrılmış cümlelerin bir listesini alır.
# Örnek: "Para yatırmak için bankaya gittim."
# Örnek: "Tekne nehir bankasında demirledi."
sentences = [
    ["I", "went", "to", "the", "bank", "to", "deposit", "money", "."],
    ["The", "boat", "docked", "at", "the", "river", "bank", "."]
]
print(f"\nGiriş cümleleri: {sentences}")

# --- Kelimeleri karakter ID'lerine dönüştür ---
# ELMo karakter seviyesi girdilerle çalıştığı için, kelime tokenlerini
# karakter ID'lerinden oluşan bir tensöre dönüştürmemiz gerekir.
character_ids = batch_to_ids(sentences)
print(f"Karakter ID'leri tensörünün şekli: {character_ids.shape}") # Beklenen: (batch_size, max_seq_len, max_word_len)

# --- ELMo gömülerini oluştur ---
# ELMo modeli üzerinden ileri geçiş.
# Çıktı `elmo_embeddings` bir sözlüktür.
# `elmo_embeddings['elmo_representations']` bir tensör listesidir,
# burada her tensör num_output_representations tarafından istenen bir çıktı temsiline karşılık gelir.
# Her tensörün şekli (batch_size, sequence_length, embedding_dim) şeklindedir.
print("ELMo gömüleri oluşturuluyor...")
elmo_embeddings = elmo(character_ids)
print("ELMo gömüleri oluşturuldu.")

# --- Gömü ayrıntılarını eriş ve yazdır ---
# İlk cümle için birleşik gömülere erişin.
# [0], ilk çıktı temsiline (num_output_representations=1 olduğu için) atıfta bulunur.
# [0], partideki ilk cümleye atıfta bulunur.
first_sentence_embeddings = elmo_embeddings['elmo_representations'][0][0] # Şekil: (seq_len, embedding_dim)

print(f"\nİlk cümle için ELMo gömüleri şekli ('I went to the bank to deposit money.'): {first_sentence_embeddings.shape}")
print(f"Gömü boyutu: {first_sentence_embeddings.shape[1]}") # ELMo genellikle 1024 boyuta sahiptir.

# Bağlamsallaştırmayı göstermek için ilk cümlede (indeks 4) ve
# ikinci cümlede (indeks 6) "bank" kelimesinin kısmi gömülerini yazdırın.
# detach().cpu().numpy() tensörü kolay yazdırma için bir numpy dizisine dönüştürür.
bank_embedding_sentence1 = first_sentence_embeddings[4][:5].detach().cpu().numpy()
print(f"\n1. cümledeki 'bank' kelimesinin kısmi ELMo gömüsü (finansal bağlam): {bank_embedding_sentence1}")

# İkinci cümle için gömülere erişim
second_sentence_embeddings = elmo_embeddings['elmo_representations'][0][1]
bank_embedding_sentence2 = second_sentence_embeddings[6][:5].detach().cpu().numpy()
print(f"2. cümledeki 'bank' kelimesinin kısmi ELMo gömüsü (nehir bağlamı): {bank_embedding_sentence2}")

# Bu gömüleri genellikle belirli bir aşağı akış NLP görevi için
# (örneğin, sınıflandırma, dizi etiketleme) başka bir sinir ağına beslersiniz.
print("\nNot: İki cümledeki 'bank' kelimesi için tam 1024 boyutlu gömüler farklı olacaktır,")
print("bu da ELMo'nun bağlamsal farkındalığını gösterir.")

# Geçici önbelleğe alınmış dosyaları temizle (isteğe bağlı, allennlp sürümüne/yapılandırmasına bağlıdır)
# for file_path in [options_file, weight_file]:
#     if file_path.startswith("http") and os.path.exists(os.path.basename(file_path)):
#         os.remove(os.path.basename(file_path))

(Kod örneği bölümünün sonu)
```
<a name="6-sonuç"></a>
## 6. Sonuç
ELMo, NLP'de bir dönüm noktası niteliğinde bir başarıyı temsil etti ve kelime gömülerinin algılanışını ve kullanımını temelden değiştirdi. Statik temsillerden **bağlamsallaştırılmış temsiller**e geçerek, ELMo çok anlamlılık sorununu etkili bir şekilde ele aldı ve modellerin kelimelerin nüanslı anlamlarını çevreleyen metne göre kavramasına olanak tanıdı. Ağırlıklı katman kombinasyonuna sahip derin bir BiLM kullanan mimarisi, dil anlayışı için **derin öğrenme** ve **transfer öğreniminin** gücünü sergiledi. Daha yeni transformer tabanlı modeller o zamandan beri sınırları daha da zorlamış olsa da, ELMo'nun katkıları, büyük ölçekli önceden eğitilmiş dil modellerinin muazzam potansiyelini göstermede etkili oldu ve yüksek performanslı NLP sistemlerinin modern çağına yol açtı. Derin bağlamsal gömülerin tarihsel ve kavramsal anlayışının kritik bir parçası olmaya devam etmektedir.

<a name="7-kaynaklar"></a>
## 7. Kaynaklar
*   Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). *Deep contextualized word representations*. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 2227-2237.
*   Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. Proceedings of International Conference on Learning Representations (ICLR), 2013.
*   Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532-1543.